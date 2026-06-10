import { test, expect } from '@playwright/test';

// Tests for trainer.js: audioToFloat32Spec, trainCreate/Step/Destroy/GetParams.
// Uses test_wasm.html (loads web_trainer.js + trainer.js, no onnxruntime or piper).

test.setTimeout(60_000);

test.beforeEach(async ({ page }) => {
    await page.goto('/test_wasm.html');
    await page.waitForFunction(() => window.webTrainerReady === true &&
        typeof audioToFloat32Spec === 'function', { timeout: 20_000 });
});

// ── audioToFloat32Spec ────────────────────────────────────────────────────────

test('audioToFloat32Spec returns null when WASM not ready', async ({ page }) => {
    const result = await page.evaluate(() => {
        const saved = window.webTrainerReady;
        window.webTrainerReady = false;
        const r = audioToFloat32Spec(new Float32Array(32000), 16000);
        window.webTrainerReady = saved;
        return r;
    });
    expect(result).toBeNull();
});

test('audioToFloat32Spec produces T >= 157 for 2 s of audio at 16 kHz', async ({ page }) => {
    const result = await page.evaluate(() => {
        const samples = new Float32Array(32000);
        for (let i = 0; i < samples.length; i++)
            samples[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / 16000);
        return audioToFloat32Spec(samples, 16000);
    });
    expect(result).not.toBeNull();
    expect(result.T).toBeGreaterThanOrEqual(157);
    expect(result.spectrogram.length).toBe(result.T * 40);
});

test('audioToFloat32Spec pads short audio to minFrames (157)', async ({ page }) => {
    // 0.5 s = 8000 samples → ~50 frames → should be padded to 157
    const result = await page.evaluate(() => {
        const samples = new Float32Array(8000);
        for (let i = 0; i < samples.length; i++)
            samples[i] = 0.2 * Math.sin(2 * Math.PI * 440 * i / 16000);
        return audioToFloat32Spec(samples, 16000);
    });
    expect(result).not.toBeNull();
    expect(result.T).toBe(157);
    expect(result.spectrogram.length).toBe(157 * 40);
});

test('audioToFloat32Spec resamples non-16k input', async ({ page }) => {
    // 2 s at 22050 Hz — should produce the same T as 2 s at 16 kHz (approximately)
    const result = await page.evaluate(() => {
        const n = 22050 * 2;
        const samples = new Float32Array(n);
        for (let i = 0; i < n; i++) samples[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / 22050);
        return audioToFloat32Spec(samples, 22050);
    });
    expect(result).not.toBeNull();
    expect(result.T).toBeGreaterThanOrEqual(157);
    expect(result.spectrogram.length).toBe(result.T * 40);
});

test('audioToFloat32Spec values are in expected float range [0, ~10]', async ({ page }) => {
    const stats = await page.evaluate(() => {
        const samples = new Float32Array(32000);
        for (let i = 0; i < samples.length; i++)
            samples[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / 16000);
        const r = audioToFloat32Spec(samples, 16000);
        let lo = Infinity, hi = -Infinity;
        for (const v of r.spectrogram) { if (v < lo) lo = v; if (v > hi) hi = v; }
        return { lo, hi };
    });
    expect(stats.lo).toBeGreaterThanOrEqual(0);
    expect(stats.hi).toBeLessThanOrEqual(12);   // uint8 max (255) × 1/25.6 ≈ 9.96
});

// ── trainCreate / trainDestroy ────────────────────────────────────────────────

test('trainCreate returns non-zero pointers', async ({ page }) => {
    const result = await page.evaluate(() => {
        const { net, adam } = trainCreate(1e-3);
        const ok = net !== 0 && adam !== 0;
        trainDestroy(net, adam);
        return ok;
    });
    expect(result).toBe(true);
});

test('trainDestroy does not crash on repeated calls', async ({ page }) => {
    await expect(page.evaluate(() => {
        for (let i = 0; i < 3; i++) {
            const { net, adam } = trainCreate(1e-3);
            trainDestroy(net, adam);
        }
        return 'ok';
    })).resolves.toBe('ok');
});

// ── trainGetParams ────────────────────────────────────────────────────────────

test('trainGetParams returns 24801 float32 values', async ({ page }) => {
    const n = await page.evaluate(() => {
        const { net, adam } = trainCreate(1e-3);
        const p = trainGetParams(net);
        trainDestroy(net, adam);
        return p.length;
    });
    expect(n).toBe(24801);
});

// ── trainStep ────────────────────────────────────────────────────────────────

test('trainStep returns a finite positive loss', async ({ page }) => {
    const loss = await page.evaluate(() => {
        const { net, adam } = trainCreate(1e-3);
        const T = 157 + 5;
        const spec = new Float32Array(T * 40).fill(0.5);
        const loss = trainStep(net, adam, spec, T, 1.0);
        trainDestroy(net, adam);
        return loss;
    });
    expect(isFinite(loss)).toBe(true);
    expect(loss).toBeGreaterThan(0);
});

test('trainStep loss decreases over 10 steps on a fixed positive spectrogram', async ({ page }) => {
    const losses = await page.evaluate(() => {
        const { net, adam } = trainCreate(1e-3);
        const T = 157 + 10;
        // Deterministic spectrogram
        const spec = new Float32Array(T * 40);
        let seed = 42;
        for (let i = 0; i < spec.length; i++) {
            seed = (seed * 1664525 + 1013904223) >>> 0;
            spec[i] = (seed >>> 0) / 4294967296 * 5;  // [0, 5] — typical feature range
        }
        const losses = [];
        for (let i = 0; i < 10; i++) losses.push(trainStep(net, adam, spec, T, 1.0));
        trainDestroy(net, adam);
        return losses;
    });
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
});

test('trainStep with audioToFloat32Spec output: loss is finite', async ({ page }) => {
    const loss = await page.evaluate(() => {
        const samples = new Float32Array(32000);
        for (let i = 0; i < samples.length; i++)
            samples[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / 16000);
        const feat = audioToFloat32Spec(samples, 16000);
        if (!feat) return null;
        const { net, adam } = trainCreate(1e-3);
        const loss = trainStep(net, adam, feat.spectrogram, feat.T, 1.0);
        trainDestroy(net, adam);
        return loss;
    });
    expect(loss).not.toBeNull();
    expect(isFinite(loss)).toBe(true);
    expect(loss).toBeGreaterThan(0);
});
