import { test, expect } from '@playwright/test';

// Tab UI structural tests — do not require WASM to be fully initialised.

test.beforeEach(async ({ page }) => {
    await page.goto('/trainer/');
    await page.waitForLoadState('domcontentloaded');
});

// ── Tab structure ─────────────────────────────────────────────────────────────

test('page title is Wake Word Trainer', async ({ page }) => {
    await expect(page).toHaveTitle('Wake Word Trainer');
});

test('main tab buttons are present', async ({ page }) => {
    await expect(page.locator('.tab-btn[data-tab="tts"]')).toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="augment"]')).toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="train"]')).toBeVisible();
});

test('TTS tab is active by default', async ({ page }) => {
    await expect(page.locator('.tab-btn[data-tab="tts"]')).toHaveClass(/active/);
    await expect(page.locator('#tab-tts')).toBeVisible();
});

test('non-active tab panels are hidden by default', async ({ page }) => {
    await expect(page.locator('#tab-augment')).not.toBeVisible();
    await expect(page.locator('#tab-train')).not.toBeVisible();
});

// ── Tab switching ─────────────────────────────────────────────────────────────

test('clicking Augment tab shows Augment panel and hides others', async ({ page }) => {
    await page.click('.tab-btn[data-tab="augment"]');
    await expect(page.locator('#tab-augment')).toBeVisible();
    await expect(page.locator('#tab-tts')).not.toBeVisible();
    await expect(page.locator('#tab-train')).not.toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="augment"]')).toHaveClass(/active/);
});

test('clicking Train tab shows Train panel', async ({ page }) => {
    await page.click('.tab-btn[data-tab="train"]');
    await expect(page.locator('#tab-train')).toBeVisible();
    await expect(page.locator('#tab-tts')).not.toBeVisible();
});

test('only one tab panel is visible at a time', async ({ page }) => {
    for (const tab of ['augment', 'train', 'tts']) {
        await page.click(`.tab-btn[data-tab="${tab}"]`);
        const visible = await page.locator('.tab-panel.active').count();
        expect(visible).toBe(1);
    }
});

test('active tab button gets active class, others lose it', async ({ page }) => {
    await page.click('.tab-btn[data-tab="train"]');
    expect(await page.locator('.tab-btn.active').count()).toBe(1);
    await expect(page.locator('.tab-btn.active')).toHaveAttribute('data-tab', 'train');

    await page.click('.tab-btn[data-tab="augment"]');
    await expect(page.locator('.tab-btn.active')).toHaveAttribute('data-tab', 'augment');
});

// ── Per-tab content ───────────────────────────────────────────────────────────

test('TTS tab contains model load controls and generate button', async ({ page }) => {
    await expect(page.locator('#file-voice')).toBeVisible();
    await expect(page.locator('#btn-generate')).toBeDisabled();
    await expect(page.locator('#phrase')).toHaveValue('hey lumus');
    await expect(page.locator('#lang')).toBeVisible();
});

test('Augment tab contains pitch slider and reverb dropdown', async ({ page }) => {
    await page.click('.tab-btn[data-tab="augment"]');
    await expect(page.locator('#aug-pitch')).toBeVisible();
    await expect(page.locator('#aug-reverb')).toBeVisible();
    await expect(page.locator('#aug-status')).toBeAttached();
});

test('Train tab shows milestone roadmap', async ({ page }) => {
    await page.click('.tab-btn[data-tab="train"]');
    await expect(page.locator('.milestone-list')).toBeVisible();
    const items = page.locator('.milestone-list li');
    await expect(items).toHaveCount(12);
});

// ── Augment controls ──────────────────────────────────────────────────────────

test('pitch slider updates displayed value', async ({ page }) => {
    await page.click('.tab-btn[data-tab="augment"]');
    await page.fill('#aug-pitch', '2');
    await page.dispatchEvent('#aug-pitch', 'input');
    await expect(page.locator('#aug-pitch-val')).toHaveText('2');
});

test('noise SNR slider updates displayed value', async ({ page }) => {
    await page.click('.tab-btn[data-tab="augment"]');
    await page.fill('#aug-noise-snr', '-5');
    await page.dispatchEvent('#aug-noise-snr', 'input');
    await expect(page.locator('#aug-noise-val')).toHaveText('-5');
});

test('augmentation runs live on control change after loading a WAV', async ({ page }) => {
    await page.waitForFunction(() => window.webTrainerReady === true, null, { timeout: 30_000 });

    // Minimal 0.5 s 16 kHz mono WAV (440 Hz sine)
    const n = 8000, data = Buffer.alloc(44 + n * 2);
    data.write('RIFF', 0); data.writeUInt32LE(36 + n * 2, 4); data.write('WAVE', 8);
    data.write('fmt ', 12); data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20);
    data.writeUInt16LE(1, 22); data.writeUInt32LE(16000, 24); data.writeUInt32LE(32000, 28);
    data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
    data.write('data', 36); data.writeUInt32LE(n * 2, 40);
    for (let i = 0; i < n; i++)
        data.writeInt16LE(Math.round(12000 * Math.sin(2 * Math.PI * 440 * i / 16000)), 44 + i * 2);

    await page.click('.tab-btn[data-tab="augment"]');
    await page.setInputFiles('#aug-file', {
        name: 'tone.wav', mimeType: 'audio/wav', buffer: data });

    // Loading triggers an immediate augment run
    await expect(page.locator('#aug-status')).toContainText('Done.', { timeout: 15_000 });

    // Moving a control re-runs augmentation without any button
    await page.fill('#aug-pitch', '2');
    await page.dispatchEvent('#aug-pitch', 'input');
    await expect(page.locator('#aug-status')).toContainText('Done.', { timeout: 15_000 });
});
