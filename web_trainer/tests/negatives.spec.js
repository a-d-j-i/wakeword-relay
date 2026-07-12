import { test, expect } from '@playwright/test';

// Build a minimal valid negatives bundle in Node.js (test runner context).
// categories: [{ name: string, numSamples: number }]
// numFrames, numFeatures: dimensions per sample
function buildMockBundle(categories, numFrames = 10, numFeatures = 5) {
    const parts = [];

    // Magic
    parts.push(Buffer.from('MWWN'));
    // Version
    parts.push(Buffer.from([1]));
    // NumFrames (uint16 LE)
    const fH = Buffer.alloc(2); fH.writeUInt16LE(numFrames); parts.push(fH);
    // NumFeatures
    parts.push(Buffer.from([numFeatures]));
    // NumCategories
    parts.push(Buffer.from([categories.length]));

    for (const { name, numSamples, fillValue = 100 } of categories) {
        const nameBytes = Buffer.from(name, 'utf8');
        parts.push(Buffer.from([nameBytes.length]));
        parts.push(nameBytes);
        const nsH = Buffer.alloc(4); nsH.writeUInt32LE(numSamples); parts.push(nsH);
        // Data: numSamples × numFrames × numFeatures uint16 LE
        const dataLen = numSamples * numFrames * numFeatures * 2;
        const data = Buffer.alloc(dataLen);
        for (let i = 0; i < dataLen; i += 2) data.writeUInt16LE(fillValue, i);
        parts.push(data);
    }

    return Buffer.concat(parts);
}

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/');
    await page.waitForFunction(() => typeof negParseBundle === 'function');
});

// ── negParseBundle ────────────────────────────────────────────────────────────

test('negParseBundle reads header fields correctly', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 3 }], 10, 5);
    const result = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        return {
            numFrames:     parsed.numFrames,
            numFeatures:   parsed.numFeatures,
            categoryNames: Object.keys(parsed.categories),
        };
    }, bundle.toString('base64'));

    expect(result.numFrames).toBe(10);
    expect(result.numFeatures).toBe(5);
    expect(result.categoryNames).toEqual(['speech']);
});

test('negParseBundle reads numSamples per category', async ({ page }) => {
    const bundle = buildMockBundle([
        { name: 'speech',       numSamples: 7 },
        { name: 'no_speech',    numSamples: 4 },
        { name: 'dinner_party', numSamples: 2 },
    ], 8, 4);

    const counts = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        return Object.fromEntries(
            Object.entries(parsed.categories).map(([k, v]) => [k, v.numSamples])
        );
    }, bundle.toString('base64'));

    expect(counts.speech).toBe(7);
    expect(counts.no_speech).toBe(4);
    expect(counts.dinner_party).toBe(2);
});

test('negParseBundle data length matches numSamples × numFrames × numFeatures', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 5 }], 10, 5);
    const dataLen = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        return parsed.categories['speech'].data.length;
    }, bundle.toString('base64'));
    expect(dataLen).toBe(5 * 10 * 5);  // numSamples × numFrames × numFeatures
});

test('negParseBundle data values are correct uint16', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 2, fillValue: 256 }], 3, 2);
    const vals = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        return Array.from(parsed.categories['speech'].data.slice(0, 6));
    }, bundle.toString('base64'));
    expect(vals).toHaveLength(6);
    vals.forEach(v => expect(v).toBe(256));
});

test('negParseBundle throws on bad magic', async ({ page }) => {
    const err = await page.evaluate(() => {
        const bad = new Uint8Array(32).fill(0);
        try { negParseBundle(bad.buffer); return null; }
        catch (e) { return e.message; }
    });
    expect(err).toMatch(/magic/i);
});

test('negParseBundle handles two categories sequentially', async ({ page }) => {
    const bundle = buildMockBundle([
        { name: 'a', numSamples: 2, fillValue: 10 },
        { name: 'b', numSamples: 3, fillValue: 20 },
    ], 4, 3);

    const result = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        return {
            aFirst: parsed.categories['a'].data[0],
            bFirst: parsed.categories['b'].data[0],
        };
    }, bundle.toString('base64'));

    expect(result.aFirst).toBe(10);
    expect(result.bFirst).toBe(20);
});

// ── negGetSample ──────────────────────────────────────────────────────────────

test('negGetSample returns Float32Array of correct length', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 5, fillValue: 100 }], 10, 5);
    const len = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        const cats   = parsed.categories;
        const sample = negGetSample(cats, 'speech');
        return sample.length;
    }, bundle.toString('base64'));
    expect(len).toBe(10 * 5);  // numFrames × numFeatures
});

test('negGetSample scales uint16 by 0.0390625', async ({ page }) => {
    // fillValue=256 → float32 = 256 × 0.0390625 = 10.0
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 2, fillValue: 256 }], 5, 4);
    const first = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        const sample = negGetSample(parsed.categories, 'speech');
        return sample[0];
    }, bundle.toString('base64'));
    expect(first).toBeCloseTo(10.0, 4);
});

test('negGetSample throws on unknown category', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 2 }], 5, 4);
    const err = await page.evaluate((b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        try { negGetSample(parsed.categories, 'no_speech'); return null; }
        catch (e) { return e.message; }
    }, bundle.toString('base64'));
    expect(err).toMatch(/unknown category/i);
});

// ── IndexedDB round-trip ──────────────────────────────────────────────────────

test('negHas returns false on fresh DB', async ({ page }) => {
    const result = await page.evaluate(async () => {
        const db = await negOpenDB();
        await negClear(db);
        return negHas(db);
    });
    expect(result).toBe(false);
});

test('negStore + negHas + negInfo round-trip', async ({ page }) => {
    const bundle = buildMockBundle([
        { name: 'speech',    numSamples: 3 },
        { name: 'no_speech', numSamples: 2 },
    ], 8, 4);

    const info = await page.evaluate(async (b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        const db     = await negOpenDB();
        await negClear(db);
        await negStore(db, parsed);
        const has    = await negHas(db);
        if (!has) throw new Error('negHas returned false after store');
        return negInfo(db);
    }, bundle.toString('base64'));

    expect(info.speech.numSamples).toBe(3);
    expect(info.no_speech.numSamples).toBe(2);
});

test('negLoad returns categories with correct data after store', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 4, fillValue: 512 }], 6, 3);
    const firstVal = await page.evaluate(async (b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        const db     = await negOpenDB();
        await negClear(db);
        await negStore(db, parsed);
        const cats   = await negLoad(db);
        return cats['speech'].data[0];
    }, bundle.toString('base64'));
    expect(firstVal).toBe(512);
});

test('negClear empties the DB', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 2 }], 5, 4);
    const result = await page.evaluate(async (b64) => {
        const bytes  = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const parsed = negParseBundle(bytes.buffer);
        const db     = await negOpenDB();
        await negStore(db, parsed);
        await negClear(db);
        return negHas(db);
    }, bundle.toString('base64'));
    expect(result).toBe(false);
});

// ── negDownload (mocked route) ────────────────────────────────────────────────

test('negDownload fetches, parses, and caches bundle from URL', async ({ page }) => {
    const bundle = buildMockBundle([
        { name: 'speech',       numSamples: 5, fillValue: 200 },
        { name: 'dinner_party', numSamples: 3, fillValue: 100 },
    ], 10, 5);

    await page.route('/negatives_test.bin', route => route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        body: bundle,
    }));

    const result = await page.evaluate(async () => {
        const db   = await negOpenDB();
        await negClear(db);
        const cats = await negDownload('/negatives_test.bin', db, null);
        const info = await negInfo(db);
        return {
            hasSpeech:  'speech' in cats,
            hasDinner:  'dinner_party' in cats,
            speechCount: info.speech?.numSamples,
            dinnerCount: info.dinner_party?.numSamples,
        };
    });

    expect(result.hasSpeech).toBe(true);
    expect(result.hasDinner).toBe(true);
    expect(result.speechCount).toBe(5);
    expect(result.dinnerCount).toBe(3);
});

test('negDownload onProgress is called with byte counts', async ({ page }) => {
    const bundle = buildMockBundle([{ name: 'speech', numSamples: 2 }], 8, 4);

    await page.route('/negatives_progress.bin', route => route.fulfill({
        status: 200,
        contentType: 'application/octet-stream',
        headers: { 'content-length': String(bundle.byteLength) },
        body: bundle,
    }));

    const called = await page.evaluate(async () => {
        const db   = await negOpenDB();
        await negClear(db);
        let progressCalled = false;
        await negDownload('/negatives_progress.bin', db, (loaded, total) => {
            if (total > 0 && loaded > 0) progressCalled = true;
        });
        return progressCalled;
    });

    expect(called).toBe(true);
});

test('negDownload throws on HTTP error', async ({ page }) => {
    await page.route('/negatives_404.bin', route => route.fulfill({ status: 404 }));

    const err = await page.evaluate(async () => {
        const db = await negOpenDB();
        try {
            await negDownload('/negatives_404.bin', db, null);
            return null;
        } catch (e) {
            return e.message;
        }
    });

    expect(err).toMatch(/404/);
});
