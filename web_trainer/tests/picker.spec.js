import { test, expect } from '@playwright/test';

test.setTimeout(60_000);

test.beforeEach(async ({ page }) => {
    page.on('console', m => console.log('[page]', m.text()));
    await page.goto('/trainer/');
    await expect(page.locator('#load-status'))
        .toContainText('Phonemizer ready', { timeout: 30_000 });
});

test('single picker: both files at once auto-loads', async ({ page }) => {
    await page.setInputFiles('#file-voice', [
        { name: 'fake-voice.onnx', mimeType: 'application/octet-stream',
          buffer: Buffer.from([1, 2, 3, 4]) },
        { name: 'fake-voice.onnx.json', mimeType: 'application/json',
          buffer: Buffer.from(JSON.stringify({
              audio: { sample_rate: 22050 }, num_speakers: 1,
              espeak: { voice: 'es-419' }, phoneme_id_map: { _: [0], '^': [1], $: [2] },
          })) },
    ]);
    // auto-load fires; fake onnx is garbage but config parses → status reaches "Model ready"
    await expect(page.locator('#load-status'))
        .toContainText('Model ready', { timeout: 10_000 });
});

test('single picker: one at a time', async ({ page }) => {
    await page.setInputFiles('#file-voice', [
        { name: 'v.onnx', mimeType: 'application/octet-stream', buffer: Buffer.from([1]) },
    ]);
    await expect(page.locator('#load-status')).toContainText('still missing the config');
    await expect(page.locator('#load-status')).toContainText('v.onnx');

    await page.setInputFiles('#file-voice', [
        { name: 'v.onnx.json', mimeType: 'application/json',
          buffer: Buffer.from(JSON.stringify({
              audio: { sample_rate: 16000 }, num_speakers: 1,
              espeak: { voice: 'es' }, phoneme_id_map: { _: [0], '^': [1], $: [2] },
          })) },
    ]);
    await expect(page.locator('#load-status'))
        .toContainText('Model ready', { timeout: 10_000 });
});

test('es-419 loads with zero network fetches (embedded data)', async ({ page }) => {
    const fetched = [];
    page.on('request', r => {
        if (r.url().includes('espeak-ng-data')) fetched.push(r.url());
    });
    const ipa = await page.evaluate(async () => {
        await loadLanguage('es-419');
        return textToIpa('cielo con lluvia', 'es-419');
    });
    console.log('IPA:', ipa, '| espeak-ng-data fetches:', JSON.stringify(fetched));
    expect(ipa).toContain('s');       // seseo — not Castilian θ
    expect(ipa).not.toContain('θ');
    expect(fetched.length).toBe(0);   // everything came from the .data preload
});
