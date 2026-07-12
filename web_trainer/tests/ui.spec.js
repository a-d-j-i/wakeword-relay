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

test('all four tab buttons are present', async ({ page }) => {
    await expect(page.locator('.tab-btn[data-tab="frontend"]')).toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="tts"]')).toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="augment"]')).toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="train"]')).toBeVisible();
});

test('Frontend tab is active by default', async ({ page }) => {
    await expect(page.locator('.tab-btn[data-tab="frontend"]')).toHaveClass(/active/);
    await expect(page.locator('#tab-frontend')).toBeVisible();
});

test('non-active tab panels are hidden by default', async ({ page }) => {
    await expect(page.locator('#tab-tts')).not.toBeVisible();
    await expect(page.locator('#tab-augment')).not.toBeVisible();
    await expect(page.locator('#tab-train')).not.toBeVisible();
});

// ── Tab switching ─────────────────────────────────────────────────────────────

test('clicking TTS tab shows TTS panel and hides others', async ({ page }) => {
    await page.click('.tab-btn[data-tab="tts"]');
    await expect(page.locator('#tab-tts')).toBeVisible();
    await expect(page.locator('#tab-frontend')).not.toBeVisible();
    await expect(page.locator('#tab-augment')).not.toBeVisible();
    await expect(page.locator('#tab-train')).not.toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="tts"]')).toHaveClass(/active/);
});

test('clicking Augment tab shows Augment panel', async ({ page }) => {
    await page.click('.tab-btn[data-tab="augment"]');
    await expect(page.locator('#tab-augment')).toBeVisible();
    await expect(page.locator('#tab-frontend')).not.toBeVisible();
    await expect(page.locator('.tab-btn[data-tab="augment"]')).toHaveClass(/active/);
});

test('clicking Train tab shows Train panel', async ({ page }) => {
    await page.click('.tab-btn[data-tab="train"]');
    await expect(page.locator('#tab-train')).toBeVisible();
    await expect(page.locator('#tab-frontend')).not.toBeVisible();
});

test('only one tab panel is visible at a time', async ({ page }) => {
    for (const tab of ['tts', 'augment', 'train', 'frontend']) {
        await page.click(`.tab-btn[data-tab="${tab}"]`);
        const visible = await page.locator('.tab-panel.active').count();
        expect(visible).toBe(1);
    }
});

test('active tab button gets active class, others lose it', async ({ page }) => {
    await page.click('.tab-btn[data-tab="tts"]');
    expect(await page.locator('.tab-btn.active').count()).toBe(1);
    await expect(page.locator('.tab-btn.active')).toHaveAttribute('data-tab', 'tts');

    await page.click('.tab-btn[data-tab="augment"]');
    await expect(page.locator('.tab-btn.active')).toHaveAttribute('data-tab', 'augment');
});

// ── Per-tab content ───────────────────────────────────────────────────────────

test('Frontend tab contains a file input and status element', async ({ page }) => {
    await expect(page.locator('#fe-file')).toBeAttached();
    await expect(page.locator('#frontend-status')).toBeAttached();
});

test('TTS tab contains model load controls and generate button', async ({ page }) => {
    await page.click('.tab-btn[data-tab="tts"]');
    await expect(page.locator('#file-voice')).toBeVisible();
    await expect(page.locator('#btn-generate')).toBeDisabled();
    await expect(page.locator('#phrase')).toHaveValue('hey lumus');
    await expect(page.locator('#lang')).toBeVisible();
});

test('Augment tab contains pitch slider and reverb dropdown', async ({ page }) => {
    await page.click('.tab-btn[data-tab="augment"]');
    await expect(page.locator('#aug-pitch')).toBeVisible();
    await expect(page.locator('#aug-reverb')).toBeVisible();
    await expect(page.locator('#aug-run')).toBeDisabled();
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
