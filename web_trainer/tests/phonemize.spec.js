import { test, expect } from '@playwright/test';

// Phonemizer WASM loads ~884KB of embedded data; allow extra time.
test.setTimeout(60_000);

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/');

    // Wait until espeak-ng WASM + data are fully loaded and initialised.
    await expect(page.locator('#load-status'))
        .toContainText('Phonemizer ready', { timeout: 30_000 });
});

test('page title', async ({ page }) => {
    await expect(page).toHaveTitle(/Wake Word Trainer/);
});

test('textToIpa — English hey lumus', async ({ page }) => {
    const ipa = await page.evaluate(() => textToIpa('hey lumus', 'en'));
    console.log('IPA (en):', ipa);

    // 'hey' → contains the /eɪ/ diphthong
    expect(ipa).toContain('eɪ');
    // 'lumus' → contains /uː/
    expect(ipa).toContain('uː');
    // All tokens should be non-empty strings
    const tokens = ipa.split(' ').filter(t => t.length > 0);
    expect(tokens.length).toBeGreaterThan(4);
});

test('textToIpa — English hey jarvis', async ({ page }) => {
    const ipa = await page.evaluate(() => textToIpa('hey jarvis', 'en'));
    console.log('IPA (en):', ipa);
    expect(ipa).toContain('eɪ');   // 'hey'
    expect(ipa).toContain('dʒ');   // 'jarvis'
});

test('textToIpa — Spanish chispa magica', async ({ page }) => {
    const ipa = await page.evaluate(() => textToIpa('chispa magica', 'es'));
    console.log('IPA (es):', ipa);
    expect(ipa).toContain('tʃ');   // 'ch' in Spanish
    expect(ipa).toContain('x');    // 'g' in Spanish before 'i' → /x/
});

test('textToIpa — Spanish buenas noches', async ({ page }) => {
    const ipa = await page.evaluate(() => textToIpa('buenas noches', 'es'));
    console.log('IPA (es):', ipa);
    expect(ipa).toContain('tʃ');   // 'ch' in 'noches'
    expect(ipa).toContain('n');
});

test('textToIpa — output is space-separated tokens', async ({ page }) => {
    const ipa = await page.evaluate(() => textToIpa('hello', 'en'));
    // Must not be empty
    expect(ipa.length).toBeGreaterThan(0);
    // Must not start or end with a space
    expect(ipa.trim()).toEqual(ipa);
    // Every character must be printable
    expect(ipa).not.toMatch(/[\x00-\x1f]/);
});

test('ipaToIds — maps known phonemes from a minimal phoneme_id_map', async ({ page }) => {
    const ids = await page.evaluate(() => {
        // Minimal phoneme_id_map matching espeak IPA output for 'hi'
        const map = { '^': [1], '$': [2], 'h': [10], 'aɪ': [11] };
        return Array.from(ipaToIds('h aɪ', map));
    });
    // Should be [1 (BOS), 10, 11, 2 (EOS)] — BigInt64Array serialises as BigInt
    expect(ids).toEqual([1n, 10n, 11n, 2n]);
});

test('ipaToIds — skips unknown phonemes gracefully', async ({ page }) => {
    const ids = await page.evaluate(() => {
        const map = { '^': [1], '$': [2], 'a': [5] };
        return Array.from(ipaToIds('a ʌ b', map));  // 'ʌ' and 'b' are not in map
    });
    // BOS + 'a' + EOS; 'ʌ' and 'b' silently dropped
    expect(ids).toEqual([1n, 5n, 2n]);
});
