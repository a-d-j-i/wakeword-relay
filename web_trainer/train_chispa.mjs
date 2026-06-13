/**
 * Playwright script: train "chispa magica" end-to-end in the browser.
 * Run: node train_chispa.mjs
 *
 * Playwright v1.60: page.setDefaultTimeout() always wins over the per-call
 * { timeout: N } option.  We therefore call setDefaultTimeout before each
 * operation that needs a non-default timeout.
 */
import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const ONNX_FILE   = '/vms2/work_tmp/download/piper_voices/es_AR-daniela-high.onnx';
const JSON_FILE   = '/vms2/work_tmp/download/piper_voices/es_AR-daniela-high.onnx.json';
const NEG_FILE    = path.join(__dirname, 'www', 'negatives.mwwn');
const BASE_URL    = 'http://localhost:8765';
const PHRASE      = 'chispa magica';
const POS_SAMPLES = 500;
const STEPS       = 10000;
const OUTPUT_DIR  = __dirname;

// ~2 s/sample on 4 workers → 250 s for 500 samples; give 3× headroom
const PREPARE_TIMEOUT = 750_000;
// 10 000 steps at ~50 ms each → 500 s; give 2× headroom
const TRAIN_TIMEOUT   = 1_000_000;

function ts() { return new Date().toISOString().slice(11, 19); }
function log(msg) { console.log(`[${ts()}] ${msg}`); }

// Wrapper: set timeout, wait, restore quick default
async function waitFor(page, fn, timeout) {
    page.setDefaultTimeout(timeout);
    try { return await fn(); }
    finally { page.setDefaultTimeout(30_000); }
}

(async () => {
    const browser = await chromium.launch({
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    const ctx = await browser.newContext({ acceptDownloads: true });
    const page = await ctx.newPage();
    page.setDefaultTimeout(30_000);

    page.on('console', msg => {
        const t = msg.text();
        if (t.includes('favicon') || t.includes('ORT')) return;
        console.log(`  [browser] ${t}`);
    });
    page.on('pageerror', err => console.error(`  [page-error] ${err}`));

    // ── Load page + phonemizer ────────────────────────────────────────────────
    log('Opening app…');
    await page.goto(BASE_URL);
    log('Waiting for phonemizer WASM…');
    await waitFor(page, () => page.waitForFunction(
        () => document.getElementById('load-status-text')?.textContent?.includes('Phonemizer ready')
    ), 60_000);
    log('Phonemizer ready.');

    // ── TTS Tab: load Piper model ─────────────────────────────────────────────
    await page.click('[data-tab="tts"]');
    log('Loading Piper model files…');
    await page.setInputFiles('#file-model',  ONNX_FILE);
    await page.setInputFiles('#file-config', JSON_FILE);
    await page.waitForFunction(() => !document.getElementById('btn-load').disabled);
    await page.click('#btn-load');

    log('Waiting for model to load…');
    await waitFor(page, () => page.waitForFunction(
        () => document.getElementById('load-status-text')?.textContent?.includes('✓ Model ready')
    ), 180_000);
    log('Model: ' + await page.$eval('#load-status-text', el => el.textContent));

    // ── Train Tab: load negatives ─────────────────────────────────────────────
    await page.click('[data-tab="train"]');
    log('Loading negatives via file input…');
    await page.setInputFiles('#neg-file', NEG_FILE);

    log('Waiting for negatives to be ready in memory…');
    await waitFor(page, () => page.waitForFunction(
        () => document.getElementById('prereq-neg')?.textContent?.includes('✓ Negatives cached')
    ), 120_000);
    log('Negatives: ' + await page.$eval('#neg-status', el => el.textContent));

    // ── Configure training parameters ─────────────────────────────────────────
    await page.fill('#train-phrase', PHRASE);
    await page.selectOption('#train-lang', 'es');
    await page.fill('#train-pos-samples', String(POS_SAMPLES));
    await page.fill('#train-steps', String(STEPS));

    // ── Prepare TTS samples ───────────────────────────────────────────────────
    log(`Preparing ${POS_SAMPLES} TTS samples…`);
    await page.waitForFunction(() => !document.getElementById('train-prepare-btn').disabled);
    await page.click('#train-prepare-btn');

    await waitFor(page, () => page.waitForFunction(() => {
        const t = document.getElementById('train-status')?.textContent ?? '';
        return t.includes('samples generated') || t.includes('Prepare failed');
    }), PREPARE_TIMEOUT);

    const prepStatus = await page.$eval('#train-status', el => el.textContent);
    if (prepStatus.includes('failed')) throw new Error('Prepare failed: ' + prepStatus);
    log('Prepare done: ' + prepStatus);

    // ── Start training ────────────────────────────────────────────────────────
    log(`Starting training: ${STEPS} steps…`);
    await page.waitForFunction(() => !document.getElementById('train-start-btn').disabled);
    await page.click('#train-start-btn');

    const trainStart = Date.now();
    const deadline   = Date.now() + TRAIN_TIMEOUT;
    while (Date.now() < deadline) {
        await page.waitForTimeout(30_000);
        const status  = await page.$eval('#train-status', el => el.textContent).catch(() => '');
        const elapsed = ((Date.now() - trainStart) / 60_000).toFixed(1);
        log(`  [${elapsed} min] ${status}`);
        if (status.includes('Training complete') || status.includes('Stopped')) break;
        const startEnabled = await page.$eval('#train-start-btn', el => !el.disabled).catch(() => false);
        if (startEnabled) { log('Start button re-enabled — training finished.'); break; }
    }
    log('Training finished.');

    // ── Export TFLite ─────────────────────────────────────────────────────────
    log('Exporting TFLite…');
    await page.waitForFunction(() => !document.getElementById('train-tflite-btn').disabled);

    const [download] = await Promise.all([
        ctx.waitForEvent('download'),
        page.click('#train-tflite-btn'),
    ]);

    await waitFor(page, () => page.waitForFunction(
        () => document.getElementById('train-status')?.textContent?.includes('TFLite exported')
    ), 120_000);
    log('Export: ' + await page.$eval('#train-status', el => el.textContent));

    const tflitePath = path.join(OUTPUT_DIR, 'chispa_magica.tflite');
    await download.saveAs(tflitePath);
    const size = fs.statSync(tflitePath).size;
    log(`TFLite saved → ${tflitePath} (${(size / 1024).toFixed(1)} KB)`);

    await browser.close();
    log('Done.');
})();
