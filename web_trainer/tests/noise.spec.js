import { test, expect } from '@playwright/test';

// Build a minimal valid noise bundle in Node.js (test runner context).
function buildMockBundle({ numClips = 3, clipSamples = 64, sampleRate = 16000, fillValue = 1000 } = {}) {
    const headerSize = 15;   // 4+1+2+4+4
    const dataSize   = numClips * clipSamples * 2;
    const buf        = Buffer.alloc(headerSize + dataSize);

    buf.write('MWWB', 0, 'ascii');              // magic
    buf.writeUInt8(1, 4);                        // version
    buf.writeUInt16LE(sampleRate, 5);            // sample rate
    buf.writeUInt32LE(clipSamples, 7);           // samples per clip
    buf.writeUInt32LE(numClips, 11);             // num clips

    for (let i = 0; i < numClips * clipSamples; i++) {
        buf.writeInt16LE(fillValue, headerSize + i * 2);
    }
    return buf;
}

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/');
    await page.waitForFunction(() => typeof noiseParseBundle === 'function');
    // Clear any OPFS state left by previous tests.
    await page.evaluate(() => noiseClearOPFS());
});

// ── noiseParseBundle ──────────────────────────────────────────────────────────

test('noiseParseBundle reads header fields correctly', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 5, clipSamples: 64, sampleRate: 16000 });
    const result = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = noiseParseBundle(bytes.buffer);
        return { sampleRate: parsed.sampleRate, clipSamples: parsed.clipSamples, numClips: parsed.numClips };
    }, bundle.toString('base64'));

    expect(result.sampleRate).toBe(16000);
    expect(result.clipSamples).toBe(64);
    expect(result.numClips).toBe(5);
});

test('noiseParseBundle data length matches numClips × clipSamples', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 4, clipSamples: 32 });
    const len = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = noiseParseBundle(bytes.buffer);
        return parsed.data.length;
    }, bundle.toString('base64'));
    expect(len).toBe(4 * 32);
});

test('noiseParseBundle data values are correct int16', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 2, clipSamples: 8, fillValue: -500 });
    const vals = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = noiseParseBundle(bytes.buffer);
        return Array.from(parsed.data.slice(0, 4));
    }, bundle.toString('base64'));
    vals.forEach(v => expect(v).toBe(-500));
});

test('noiseParseBundle throws on bad magic', async ({ page }) => {
    const err = await page.evaluate(() => {
        const bad = new Uint8Array(64).fill(0);
        try { noiseParseBundle(bad.buffer); return null; }
        catch (e) { return e.message; }
    });
    expect(err).toMatch(/magic/i);
});

test('noiseParseBundle throws on truncated data', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 10, clipSamples: 100 });
    // Truncate to header only
    const truncated = bundle.slice(0, 15);
    const err = await page.evaluate((b64) => {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        try { noiseParseBundle(bytes.buffer); return null; }
        catch (e) { return e.message; }
    }, truncated.toString('base64'));
    expect(err).toMatch(/truncated/i);
});

// ── noiseGetSample ────────────────────────────────────────────────────────────

test('noiseGetSample returns Float32Array of correct length', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 5, clipSamples: 64 });
    const len = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = noiseParseBundle(bytes.buffer);
        const sample = noiseGetSample(parsed);
        return { len: sample.length, type: sample.constructor.name };
    }, bundle.toString('base64'));
    expect(len.len).toBe(64);
    expect(len.type).toBe('Float32Array');
});

test('noiseGetSample scales int16 by 1/32768', async ({ page }) => {
    // fillValue = 16384 → float32 = 16384 / 32768 = 0.5
    const bundle = buildMockBundle({ numClips: 2, clipSamples: 8, fillValue: 16384 });
    const first = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = noiseParseBundle(bytes.buffer);
        return noiseGetSample(parsed)[0];
    }, bundle.toString('base64'));
    expect(first).toBeCloseTo(0.5, 4);
});

test('noiseGetSample handles negative int16 values', async ({ page }) => {
    // fillValue = -16384 → float32 = -16384 / 32768 = -0.5
    const bundle = buildMockBundle({ numClips: 2, clipSamples: 8, fillValue: -16384 });
    const first = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = noiseParseBundle(bytes.buffer);
        return noiseGetSample(parsed)[0];
    }, bundle.toString('base64'));
    expect(first).toBeCloseTo(-0.5, 4);
});

test('noiseGetSample throws when no clips loaded', async ({ page }) => {
    const err = await page.evaluate(() => {
        try { noiseGetSample(null); return null; }
        catch (e) { return e.message; }
    });
    expect(err).toMatch(/no noise clips/i);
});

// ── OPFS round-trip ───────────────────────────────────────────────────────────

test('noiseHasBundle returns false when nothing stored', async ({ page }) => {
    const result = await page.evaluate(() => noiseHasBundle());
    expect(result).toBe(false);
});

test('noiseStoreToOPFS + noiseHasBundle + noiseLoadFromOPFS round-trip', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 3, clipSamples: 32, fillValue: 777 });
    const result = await page.evaluate(async (b64) => {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        await noiseStoreToOPFS(bytes.buffer);
        const has = await noiseHasBundle();
        if (!has) throw new Error('noiseHasBundle returned false after store');
        const buf    = await noiseLoadFromOPFS();
        const parsed = noiseParseBundle(buf);
        return { numClips: parsed.numClips, clipSamples: parsed.clipSamples, firstVal: parsed.data[0] };
    }, bundle.toString('base64'));
    expect(result.numClips).toBe(3);
    expect(result.clipSamples).toBe(32);
    expect(result.firstVal).toBe(777);
});

test('noiseClearOPFS removes the cached file', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 2, clipSamples: 16 });
    const result = await page.evaluate(async (b64) => {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        await noiseStoreToOPFS(bytes.buffer);
        await noiseClearOPFS();
        return noiseHasBundle();
    }, bundle.toString('base64'));
    expect(result).toBe(false);
});

test('noiseInit returns null when nothing cached', async ({ page }) => {
    const result = await page.evaluate(() => noiseInit());
    expect(result).toBeNull();
});

test('noiseInit returns parsed bundle when cached', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 4, clipSamples: 48 });
    const result = await page.evaluate(async (b64) => {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        await noiseStoreToOPFS(bytes.buffer);
        const parsed = await noiseInit();
        return parsed ? { numClips: parsed.numClips, clipSamples: parsed.clipSamples } : null;
    }, bundle.toString('base64'));
    expect(result).not.toBeNull();
    expect(result.numClips).toBe(4);
    expect(result.clipSamples).toBe(48);
});

// ── noiseDownload (mocked route) ──────────────────────────────────────────────

test('noiseDownload fetches, stores, and returns parsed bundle', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 5, clipSamples: 64, fillValue: 200 });

    await page.route('/noise_test.bin', route => route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        body: bundle,
    }));

    const result = await page.evaluate(async () => {
        const parsed = await noiseDownload('/noise_test.bin', null);
        const has    = await noiseHasBundle();
        return { numClips: parsed.numClips, clipSamples: parsed.clipSamples, cached: has };
    });

    expect(result.numClips).toBe(5);
    expect(result.clipSamples).toBe(64);
    expect(result.cached).toBe(true);
});

test('noiseDownload onProgress is called with byte counts', async ({ page }) => {
    const bundle = buildMockBundle({ numClips: 3, clipSamples: 32 });

    await page.route('/noise_progress.bin', route => route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        headers: { 'content-length': String(bundle.byteLength) },
        body: bundle,
    }));

    const called = await page.evaluate(async () => {
        let progressCalled = false;
        await noiseDownload('/noise_progress.bin', (loaded, total) => {
            if (total > 0 && loaded > 0) progressCalled = true;
        });
        return progressCalled;
    });

    expect(called).toBe(true);
});

test('noiseDownload throws on HTTP error', async ({ page }) => {
    await page.route('/noise_404.bin', route => route.fulfill({ status: 404 }));

    const err = await page.evaluate(async () => {
        try {
            await noiseDownload('/noise_404.bin', null);
            return null;
        } catch (e) {
            return e.message;
        }
    });

    expect(err).toMatch(/404/);
});
