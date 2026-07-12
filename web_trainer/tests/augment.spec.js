import { test, expect } from '@playwright/test';

// augment.js is a synchronous pure-JS library loaded by index.html.
// Tests navigate to / (index.html) and call augment functions directly.

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/');
    // Wait until augment.js is loaded (augmentSample is defined).
    await page.waitForFunction(() => typeof augmentSample === 'function');
});

// ── loadWav ──────────────────────────────────────────────────────────────────

test('loadWav round-trips through encodeWav', async ({ page }) => {
    // encodeWav is defined in piper.js (also loaded by index.html)
    const result = await page.evaluate(() => {
        const original = new Float32Array([0.1, -0.2, 0.3, -0.4, 0.5]);
        const blob = encodeWav(original, 22050);
        return blob.arrayBuffer().then(ab => {
            const { samples, sampleRate } = loadWav(ab);
            return { sampleRate, values: Array.from(samples) };
        });
    });
    expect(result.sampleRate).toBe(22050);
    expect(result.values).toHaveLength(5);
    // 16-bit quantization loses ~0.003 precision
    [0.1, -0.2, 0.3, -0.4, 0.5].forEach((v, i) =>
        expect(result.values[i]).toBeCloseTo(v, 2));
});

// ── pitchShift ───────────────────────────────────────────────────────────────

test('pitchShift by 0 semitones is identity', async ({ page }) => {
    const result = await page.evaluate(() => {
        const signal = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
        return Array.from(pitchShift(signal, 0));
    });
    expect(result).toHaveLength(5);
    [0.1, 0.2, 0.3, 0.4, 0.5].forEach((v, i) =>
        expect(result[i]).toBeCloseTo(v, 5));
});

test('pitchShift up shortens the array', async ({ page }) => {
    const result = await page.evaluate(() => {
        const signal = new Float32Array(1000);
        signal.fill(0.5);
        return pitchShift(signal, 12).length;  // +1 octave = 2× rate → half length
    });
    expect(result).toBeCloseTo(500, -1);  // within ±10 samples
});

test('pitchShift down lengthens the array', async ({ page }) => {
    const result = await page.evaluate(() => {
        const signal = new Float32Array(1000);
        signal.fill(0.5);
        return pitchShift(signal, -12).length;  // -1 octave = 0.5× rate → double length
    });
    expect(result).toBeCloseTo(2000, -1);
});

// ── applyGain ────────────────────────────────────────────────────────────────

test('applyGain +20 dB multiplies amplitude by 10', async ({ page }) => {
    const result = await page.evaluate(() => {
        const signal = new Float32Array([0.1, 0.2, -0.3]);
        return Array.from(applyGain(signal, 20));
    });
    expect(result[0]).toBeCloseTo(1.0, 3);
    expect(result[1]).toBeCloseTo(2.0, 3);
    expect(result[2]).toBeCloseTo(-3.0, 3);
});

test('applyGain 0 dB is identity', async ({ page }) => {
    const result = await page.evaluate(() => {
        const signal = new Float32Array([0.1, -0.5, 0.3]);
        return Array.from(applyGain(signal, 0));
    });
    [0.1, -0.5, 0.3].forEach((v, i) => expect(result[i]).toBeCloseTo(v, 5));
});

// ── convolve ─────────────────────────────────────────────────────────────────

test('convolve with Dirac delta is identity', async ({ page }) => {
    const result = await page.evaluate(() => {
        const signal = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
        const ir = new Float32Array([1.0]);
        return Array.from(convolve(signal, ir));
    });
    expect(result).toHaveLength(5);
    [0.1, 0.2, 0.3, 0.4, 0.5].forEach((v, i) =>
        expect(result[i]).toBeCloseTo(v, 4));
});

test('convolve output length = signal + ir - 1', async ({ page }) => {
    const len = await page.evaluate(() => {
        const signal = new Float32Array(100).fill(1);
        const ir = new Float32Array(30).fill(0.1);
        return convolve(signal, ir).length;
    });
    expect(len).toBe(129);
});

// ── applyReverb ──────────────────────────────────────────────────────────────

test('applyReverb returns same length as input', async ({ page }) => {
    const len = await page.evaluate(() => {
        const signal = new Float32Array(1000).map((_, i) => Math.sin(i / 100));
        const ir = new Float32Array(200).fill(0).map((_, i) => Math.exp(-i / 50));
        return applyReverb(signal, ir).length;
    });
    expect(len).toBe(1000);
});

test('applyReverb output amplitude does not exceed 1', async ({ page }) => {
    const maxAbs = await page.evaluate(() => {
        // Worst-case: long loud IR + loud signal
        const signal = new Float32Array(500).fill(1.0);
        const ir = new Float32Array(200).fill(1.0);
        const out = applyReverb(signal, ir);
        let m = 0;
        for (const v of out) if (Math.abs(v) > m) m = Math.abs(v);
        return m;
    });
    expect(maxAbs).toBeLessThanOrEqual(1.0 + 1e-6);
});

// ── mixNoise ─────────────────────────────────────────────────────────────────

test('mixNoise with snr=0 dB produces equal signal and noise power', async ({ page }) => {
    // At SNR=0dB, RMS(signal) == RMS(noise_contribution)
    const ratio = await page.evaluate(() => {
        // Pure tone signal and pure tone noise at different frequencies
        const N = 4000;
        const signal = new Float32Array(N).map((_, i) => 0.3 * Math.sin(i * 0.1));
        const noise  = new Float32Array(N).map((_, i) => 0.3 * Math.sin(i * 0.3));

        const rms = s => Math.sqrt(s.reduce((a, v) => a + v*v, 0) / s.length);

        const out = mixNoise(signal, noise, 0);  // SNR = 0 dB
        // output ≈ signal + noise_scaled; roughly |out| ≈ sqrt(2) * rms(signal)
        return rms(out) / rms(signal);
    });
    // At SNR=0, both components have equal RMS, so combined ≈ sqrt(2)×input RMS
    expect(ratio).toBeGreaterThan(1.0);
    expect(ratio).toBeLessThan(3.0);
});

// ── augmentSample ─────────────────────────────────────────────────────────────

test('augmentSample returns Float32Array of plausible length', async ({ page }) => {
    const info = await page.evaluate(() => {
        const N = 8000;
        const signal = new Float32Array(N).map((_, i) => 0.2 * Math.sin(i / 50));
        const noise  = new Float32Array(N).map((_, i) => 0.05 * Math.sin(i / 20));
        const out = augmentSample(signal, 16000, {
            noises: [noise],
            pitchProb: 1.0,       // always pitch-shift
            eqProb: 1.0,           // always EQ
            noiseProb: 1.0,        // always add noise
            reverbProb: 0.0,       // skip reverb (needs real IR)
            gainRange: [-6, 0],
            semitoneRange: [-1, 1],
        });
        let maxAbs = 0;
        for (const v of out) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
        return { length: out.length, maxAbs };
    });
    // pitch shift ±1 semitone changes length by at most ~6%
    expect(info.length).toBeGreaterThan(7500);
    expect(info.length).toBeLessThan(8500);
    // output must not clip
    expect(info.maxAbs).toBeLessThanOrEqual(1.0 + 1e-5);
});

test('augmentSample with reverb does not clip', async ({ page }) => {
    const maxAbs = await page.evaluate(() => {
        const N = 2000;
        const signal = new Float32Array(N).map((_, i) => 0.5 * Math.sin(i / 30));
        const ir = new Float32Array(100).map((_, i) => Math.exp(-i / 20) * 0.8);
        const out = augmentSample(signal, 16000, {
            irs: [ir],
            reverbProb: 1.0,
            pitchProb: 0.0,
            eqProb: 0.0,
            noiseProb: 0.0,
            gainRange: [0, 0],
        });
        let m = 0;
        for (const v of out) if (Math.abs(v) > m) m = Math.abs(v);
        return m;
    });
    expect(maxAbs).toBeLessThanOrEqual(1.0 + 1e-5);
});
