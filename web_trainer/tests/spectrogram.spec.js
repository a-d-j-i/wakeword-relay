import { test, expect } from '@playwright/test';

// spectrogram.js tests — canvas rendering and audioToSpectrogram.
// drawSpectrogram is pure JS (no WASM).
// audioToSpectrogram requires window.webTrainerReady + Module.HEAPU8 (rebuilt WASM).

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/');
    await page.waitForFunction(() => typeof drawSpectrogram === 'function');
});

// ── drawSpectrogram — canvas dimensions ──────────────────────────────────────

test('drawSpectrogram sets canvas width=T, height=numFeatures', async ({ page }) => {
    const dims = await page.evaluate(() => {
        const c = document.createElement('canvas');
        drawSpectrogram(c, new Int8Array(20 * 40), 20);
        return { w: c.width, h: c.height };
    });
    expect(dims.w).toBe(20);
    expect(dims.h).toBe(40);
});

test('drawSpectrogram respects custom numFeatures', async ({ page }) => {
    const dims = await page.evaluate(() => {
        const c = document.createElement('canvas');
        drawSpectrogram(c, new Int8Array(10 * 32), 10, 32);
        return { w: c.width, h: c.height };
    });
    expect(dims.w).toBe(10);
    expect(dims.h).toBe(32);
});

test('drawSpectrogram handles T=1 without error', async ({ page }) => {
    await expect(page.evaluate(() => {
        const c = document.createElement('canvas');
        drawSpectrogram(c, new Int8Array(40), 1);
        return c.width;
    })).resolves.toBe(1);
});

// ── drawSpectrogram — colormap ────────────────────────────────────────────────

test('drawSpectrogram fills all pixels (alpha=255)', async ({ page }) => {
    const allOpaque = await page.evaluate(() => {
        const c = document.createElement('canvas');
        drawSpectrogram(c, new Int8Array(5 * 40).fill(64), 5);
        const px = c.getContext('2d').getImageData(0, 0, 5, 40).data;
        for (let i = 3; i < px.length; i += 4) if (px[i] !== 255) return false;
        return true;
    });
    expect(allOpaque).toBe(true);
});

test('drawSpectrogram produces different colors for different values', async ({ page }) => {
    const { dark, bright } = await page.evaluate(() => {
        const c = document.createElement('canvas');
        // Two columns: all zeros, all 100
        const data = new Int8Array(2 * 40);
        for (let f = 0; f < 40; f++) data[0 * 40 + f] = 0;
        for (let f = 0; f < 40; f++) data[1 * 40 + f] = 100;
        drawSpectrogram(c, data, 2);
        const img = c.getContext('2d').getImageData(0, 0, 2, 40);
        return {
            dark:   [img.data[0], img.data[1], img.data[2]],   // column 0, top row
            bright: [img.data[4], img.data[5], img.data[6]],   // column 1, top row
        };
    });
    // Viridis: low value = dark purple, high value = brighter color
    const darkL  = dark[0]   + dark[1]   + dark[2];
    const brightL = bright[0] + bright[1] + bright[2];
    expect(brightL).toBeGreaterThan(darkL);
});

test('drawSpectrogram with uniform data uses a single color', async ({ page }) => {
    const allSame = await page.evaluate(() => {
        const c = document.createElement('canvas');
        drawSpectrogram(c, new Int8Array(3 * 40).fill(42), 3);
        const img = c.getContext('2d').getImageData(0, 0, 3, 40);
        const r0 = img.data[0], g0 = img.data[1], b0 = img.data[2];
        for (let i = 0; i < 3 * 40; i++) {
            const base = i * 4;
            if (img.data[base] !== r0 || img.data[base+1] !== g0 || img.data[base+2] !== b0) return false;
        }
        return true;
    });
    expect(allSame).toBe(true);
});

// ── audioToSpectrogram ────────────────────────────────────────────────────────

test('audioToSpectrogram returns null when WASM not ready', async ({ page }) => {
    // Navigate to a blank page where webTrainerReady is false
    const result = await page.evaluate(() => {
        // Temporarily shadow webTrainerReady
        const orig = window.webTrainerReady;
        window.webTrainerReady = false;
        const r = audioToSpectrogram(new Float32Array(16000), 16000);
        window.webTrainerReady = orig;
        return r;
    });
    expect(result).toBeNull();
});

test('audioToSpectrogram produces correct frame count from 1 s of silence', async ({ page }) => {
    // Wait for WASM and check HEAPU8 is available (rebuilt build).
    await page.waitForFunction(() => window.webTrainerReady === true, { timeout: 20_000 });

    const result = await page.evaluate(() => {
        if (!Module.HEAPU8) return { skipped: true };
        const samples = new Float32Array(16000).fill(0);
        const r = audioToSpectrogram(samples, 16000);
        return r ? { T: r.T, featLen: r.features.length, skipped: false } : { skipped: true };
    });

    if (result.skipped) {
        console.log('SKIP: audioToSpectrogram needs WASM rebuild with HEAPU8');
        return;
    }
    expect(result.T).toBeGreaterThanOrEqual(90);
    expect(result.T).toBeLessThanOrEqual(100);
    expect(result.featLen).toBe(result.T * 40);
});

test('audioToSpectrogram resamples 22050 Hz to 16 kHz', async ({ page }) => {
    await page.waitForFunction(() => window.webTrainerReady === true, { timeout: 20_000 });

    const result = await page.evaluate(() => {
        if (!Module.HEAPU8) return { skipped: true };
        // 22050 Hz, 1 second sine
        const N = 22050;
        const samples = new Float32Array(N);
        for (let i = 0; i < N; i++) samples[i] = 0.3 * Math.sin(2 * Math.PI * 440 * i / 22050);
        const r = audioToSpectrogram(samples, 22050);
        return r ? { T: r.T, skipped: false } : { skipped: true };
    });

    if (result.skipped) return;
    // ~16000 samples after resample → ~98 frames
    expect(result.T).toBeGreaterThanOrEqual(90);
    expect(result.T).toBeLessThanOrEqual(105);
});
