import { test, expect } from '@playwright/test';

// MicroFrontend WASM tests (milestone 1).
// Requires docs/common/web_trainer.js and docs/common/web_trainer.wasm.
//
// Input buffers are passed via ccall 'array' type (placed on WASM stack)
// rather than via Module.HEAP16 etc., which are closure-private in the
// modularized build.

test.setTimeout(30_000);

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/test_wasm.html');
    await page.waitForFunction(() => window.webTrainerReady === true, { timeout: 20_000 });
});

test('frontend_num_features returns 40', async ({ page }) => {
    const n = await page.evaluate(() =>
        Module.ccall('frontend_num_features', 'number', [], []));
    expect(n).toBe(40);
});

test('frontend_create returns a non-null pointer', async ({ page }) => {
    const ptr = await page.evaluate(() =>
        Module.ccall('frontend_create', 'number', [], []));
    expect(ptr).not.toBe(0);
    await page.evaluate(p =>
        Module.ccall('frontend_destroy', null, ['number'], [p]), ptr);
});

test('frontend_process produces ~98 frames from 1 s of silence', async ({ page }) => {
    const nFrames = await page.evaluate(() => {
        const fe = Module.ccall('frontend_create', 'number', [], []);

        // 16000 int16 zeros = 1 s silence.  Pass via 'array' type (WASM stack).
        const pcmBytes = new Uint8Array(16000 * 2).fill(0);
        // Allocate output buffer on WASM heap (write-only for this test).
        const featPtr = Module._malloc(200 * 40);

        const frames = Module.ccall('frontend_process', 'number',
            ['number', 'array', 'number', 'number'],
            [fe, pcmBytes, 16000, featPtr]);

        Module._free(featPtr);
        Module.ccall('frontend_destroy', null, ['number'], [fe]);
        return frames;
    });
    // 16000 samples / 160-sample step ≈ 100 iterations; first ~2 fill the 30ms window.
    expect(nFrames).toBeGreaterThanOrEqual(90);
    expect(nFrames).toBeLessThanOrEqual(100);
});

test('frontend_process produces the same frame count for a 440 Hz sine', async ({ page }) => {
    const nFrames = await page.evaluate(() => {
        const fe = Module.ccall('frontend_create', 'number', [], []);

        // 440 Hz sine at amplitude ~0.3 × 32767, encoded as bytes of an Int16Array.
        const pcmInt16 = new Int16Array(16000);
        for (let i = 0; i < 16000; i++) {
            pcmInt16[i] = Math.round(9830 * Math.sin(2 * Math.PI * 440 * i / 16000));
        }
        const pcmBytes = new Uint8Array(pcmInt16.buffer);
        const featPtr  = Module._malloc(200 * 40);

        const frames = Module.ccall('frontend_process', 'number',
            ['number', 'array', 'number', 'number'],
            [fe, pcmBytes, 16000, featPtr]);

        Module._free(featPtr);
        Module.ccall('frontend_destroy', null, ['number'], [fe]);
        return frames;
    });
    expect(nFrames).toBeGreaterThanOrEqual(90);
    expect(nFrames).toBeLessThanOrEqual(100);
});

test('multiple frontend_create/destroy cycles do not crash', async ({ page }) => {
    await expect(page.evaluate(() => {
        for (let i = 0; i < 5; i++) {
            const fe = Module.ccall('frontend_create', 'number', [], []);
            if (!fe) throw new Error('frontend_create returned null');
            Module.ccall('frontend_destroy', null, ['number'], [fe]);
        }
        return 'ok';
    })).resolves.toBe('ok');
});
