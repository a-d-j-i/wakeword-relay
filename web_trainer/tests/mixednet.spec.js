import { test, expect } from '@playwright/test';

// MixedNet forward pass, backprop, and Adam optimizer tests (milestones 2-3).
// Requires www/web_trainer.js and www/web_trainer.wasm.
//
// Spectrogram inputs are passed via ccall 'array' type (WASM stack) rather
// than via Module.HEAPF32, which is closure-private in the modularized build.

test.setTimeout(60_000);

test.beforeEach(async ({ page }) => {
    await page.goto('/test_wasm.html');
    await page.waitForFunction(() => window.webTrainerReady === true, { timeout: 20_000 });
});

// ── Architecture sanity ───────────────────────────────────────────────────────

test('mixednet_min_frames returns 157 for pooled=1', async ({ page }) => {
    const n = await page.evaluate(() => {
        const m = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const f = Module.ccall('mixednet_min_frames', 'number', ['number'], [m]);
        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return f;
    });
    expect(n).toBe(157);
});

test('mixednet_min_frames returns 204 for pooled=0', async ({ page }) => {
    const n = await page.evaluate(() => {
        const m = Module.ccall('mixednet_create', 'number', ['number'], [0]);
        const f = Module.ccall('mixednet_min_frames', 'number', ['number'], [m]);
        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return f;
    });
    expect(n).toBe(204);
});

test('mixednet_create returns a non-null pointer', async ({ page }) => {
    const ptr = await page.evaluate(() =>
        Module.ccall('mixednet_create', 'number', ['number'], [1]));
    expect(ptr).not.toBe(0);
    await page.evaluate(p =>
        Module.ccall('mixednet_destroy', null, ['number'], [p]), ptr);
});

test('mixednet_num_params returns 24801 for pooled=1', async ({ page }) => {
    const n = await page.evaluate(() => {
        const m = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const params = Module.ccall('mixednet_num_params', 'number', ['number'], [m]);
        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return params;
    });
    expect(n).toBe(24801);
});

test('mixednet_num_params returns 25825 for pooled=0', async ({ page }) => {
    const n = await page.evaluate(() => {
        const m = Module.ccall('mixednet_create', 'number', ['number'], [0]);
        const params = Module.ccall('mixednet_num_params', 'number', ['number'], [m]);
        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return params;
    });
    expect(n).toBe(25825);
});

// ── Forward pass ──────────────────────────────────────────────────────────────

test('mixednet_forward returns probability in [0, 1]', async ({ page }) => {
    const prob = await page.evaluate(() => {
        const m = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const T = Module.ccall('mixednet_min_frames', 'number', ['number'], [m]) + 5;

        const specF32  = new Float32Array(T * 40).fill(0.1);
        const specBytes = new Uint8Array(specF32.buffer);

        const p = Module.ccall('mixednet_forward', 'number',
            ['number', 'array', 'number'],
            [m, specBytes, T]);

        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return p;
    });
    expect(prob).toBeGreaterThanOrEqual(0);
    expect(prob).toBeLessThanOrEqual(1);
});

test('mixednet_forward output differs for different random seeds', async ({ page }) => {
    const { p1, p2 } = await page.evaluate(() => {
        const m1 = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const m2 = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        Module.ccall('mixednet_init_random', null, ['number', 'number'], [m1, 1]);
        Module.ccall('mixednet_init_random', null, ['number', 'number'], [m2, 2]);

        const T = Module.ccall('mixednet_min_frames', 'number', ['number'], [m1]) + 5;
        const specF32  = new Float32Array(T * 40).fill(0.2);
        const specBytes = new Uint8Array(specF32.buffer);

        const p1 = Module.ccall('mixednet_forward', 'number', ['number', 'array', 'number'], [m1, specBytes, T]);
        const p2 = Module.ccall('mixednet_forward', 'number', ['number', 'array', 'number'], [m2, specBytes, T]);

        Module.ccall('mixednet_destroy', null, ['number'], [m1]);
        Module.ccall('mixednet_destroy', null, ['number'], [m2]);
        return { p1, p2 };
    });
    // Different seeds → different weights → different outputs
    expect(Math.abs(p1 - p2)).toBeGreaterThan(1e-4);
});

// ── Training ──────────────────────────────────────────────────────────────────

test('train_step returns a finite positive loss', async ({ page }) => {
    const loss = await page.evaluate(() => {
        const m   = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const opt = Module.ccall('adam_create', 'number', ['number', 'number'], [m, 1e-3]);
        const T   = Module.ccall('mixednet_min_frames', 'number', ['number'], [m]) + 5;

        const specF32  = new Float32Array(T * 40).fill(0.1);
        const specBytes = new Uint8Array(specF32.buffer);

        const loss = Module.ccall('train_step', 'number',
            ['number', 'number', 'array', 'number', 'number'],
            [m, opt, specBytes, T, 1.0]);

        Module.ccall('adam_destroy', null, ['number'], [opt]);
        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return loss;
    });
    expect(isFinite(loss)).toBe(true);
    expect(loss).toBeGreaterThan(0);
});

test('train_step loss decreases over 20 steps on a fixed spectrogram', async ({ page }) => {
    const losses = await page.evaluate(() => {
        const m   = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const opt = Module.ccall('adam_create', 'number', ['number', 'number'], [m, 1e-3]);
        const T   = Module.ccall('mixednet_min_frames', 'number', ['number'], [m]) + 10;
        const F   = 40;

        // Deterministic spectrogram via LCG (same seed = same data each step).
        const specF32 = new Float32Array(T * F);
        let seed = 123;
        for (let i = 0; i < T * F; i++) {
            seed = (seed * 1664525 + 1013904223) >>> 0;
            specF32[i] = (seed / 0x100000000) * 0.5;
        }
        const specBytes = new Uint8Array(specF32.buffer);

        const losses = [];
        for (let i = 0; i < 20; i++) {
            losses.push(Module.ccall('train_step', 'number',
                ['number', 'number', 'array', 'number', 'number'],
                [m, opt, specBytes, T, 1.0]));
        }

        Module.ccall('adam_destroy', null, ['number'], [opt]);
        Module.ccall('mixednet_destroy', null, ['number'], [m]);
        return losses;
    });

    expect(losses[0]).toBeGreaterThan(0);
    // 20 Adam steps on a fixed input must reduce loss
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
});

// ── Parameter round-trip ──────────────────────────────────────────────────────

test('get_params / set_params copies weights between two networks', async ({ page }) => {
    // After copying m1's weights into m2, both should produce identical forward output.
    const { p1, p2_before, p2_after, match } = await page.evaluate(() => {
        const m1 = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        const m2 = Module.ccall('mixednet_create', 'number', ['number'], [1]);
        Module.ccall('mixednet_init_random', null, ['number', 'number'], [m1, 42]);
        Module.ccall('mixednet_init_random', null, ['number', 'number'], [m2, 99]);

        const T = Module.ccall('mixednet_min_frames', 'number', ['number'], [m1]) + 5;
        const specF32  = new Float32Array(T * 40).fill(0.15);
        const specBytes = new Uint8Array(specF32.buffer);

        const p1        = Module.ccall('mixednet_forward', 'number', ['number', 'array', 'number'], [m1, specBytes, T]);
        const p2_before = Module.ccall('mixednet_forward', 'number', ['number', 'array', 'number'], [m2, specBytes, T]);

        // Copy m1 → m2
        const n      = Module.ccall('mixednet_num_params', 'number', ['number'], [m1]);
        const srcPtr = Module.ccall('mixednet_get_params', 'number', ['number'], [m1]);
        Module.ccall('mixednet_set_params', null, ['number', 'number', 'number'], [m2, srcPtr, n]);
        Module._free(srcPtr);

        const p2_after = Module.ccall('mixednet_forward', 'number', ['number', 'array', 'number'], [m2, specBytes, T]);

        Module.ccall('mixednet_destroy', null, ['number'], [m1]);
        Module.ccall('mixednet_destroy', null, ['number'], [m2]);

        return { p1, p2_before, p2_after, match: Math.abs(p1 - p2_after) < 1e-5 };
    });

    // Different seeds must produce different outputs
    expect(Math.abs(p1 - p2_before)).toBeGreaterThan(1e-4);
    // After weight copy, outputs must match
    expect(match).toBe(true);
});
