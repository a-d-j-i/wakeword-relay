'use strict';

// ── Tab switching ─────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ── Module initialisation ─────────────────────────────────────────────────────

// Phonemizer (espeak-ng WASM) — used by TTS tab.
initPhonemizer()
    .then(() => {
        document.getElementById('load-status-text').textContent = 'Phonemizer ready. Select model files above.';
        document.getElementById('load-status').className = 'status ok';
    })
    .catch(e => {
        document.getElementById('load-status-text').textContent = 'Phonemizer init failed: ' + e.message;
        document.getElementById('load-status').className = 'status err';
    });

// ── Training prereqs display ──────────────────────────────────────────────────

const prereqTts   = document.getElementById('prereq-tts');
const prereqNeg   = document.getElementById('prereq-neg');
const prereqNoise = document.getElementById('prereq-noise');
const trainPrepareBtn = document.getElementById('train-prepare-btn');

function updatePrereqs() {
    const hasTts   = modelBlobUrl !== null;
    const hasNeg   = negCats   !== null;
    const hasNoise = noiseParsed !== null;

    prereqTts.textContent = hasTts ? '✓ TTS loaded' : '✗ TTS: load model on TTS tab';
    prereqTts.className   = 'badge ' + (hasTts ? 'prereq-ok' : 'prereq-warn');

    prereqNeg.textContent = hasNeg ? '✓ Negatives cached' : '✗ Negatives: load on Train tab';
    prereqNeg.className   = 'badge ' + (hasNeg ? 'prereq-ok' : 'prereq-warn');

    prereqNoise.textContent = hasNoise ? '✓ Noise cached' : '○ Noise: optional';
    prereqNoise.className   = 'badge' + (hasNoise ? ' prereq-ok' : '');

    trainPrepareBtn.disabled = !(hasTts && hasNeg);
}

// web_trainer WASM — used by Frontend and Augment tabs for spectrograms.
createWebTrainer().then(m => {
    window.Module = m;
    window.webTrainerReady = true;
    document.getElementById('frontend-status').textContent = 'Ready. Load a WAV file to see its spectrogram.';
    document.getElementById('frontend-status').className = 'status ok';
    document.getElementById('augment-wasm-status').textContent = '';
}).catch(e => {
    document.getElementById('frontend-status').textContent = 'WASM init failed: ' + e.message;
    document.getElementById('frontend-status').className = 'status err';
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function showSpectrogram(canvas, wrap, samples, sampleRate) {
    const result = audioToSpectrogram(samples, sampleRate);
    if (!result) { wrap.style.display = 'none'; return; }
    drawSpectrogram(canvas, result.features, result.T);
    canvas.title = `${result.T} frames · ${(result.T * 10 / 1000).toFixed(2)} s · 40 features`;
    wrap.style.display = 'block';
}

// ── Frontend tab ──────────────────────────────────────────────────────────────

const feFileInput   = document.getElementById('fe-file');
const feCanvas      = document.getElementById('fe-canvas');
const feCanvasWrap  = document.getElementById('fe-canvas-wrap');
const feAudio       = document.getElementById('fe-audio');
const feStats       = document.getElementById('fe-stats');

feFileInput.addEventListener('change', async () => {
    const file = feFileInput.files[0];
    if (!file) return;
    feStats.textContent = 'Processing…';
    try {
        const buf  = await file.arrayBuffer();
        feAudio.src = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
        feAudio.style.display = '';
        const wav  = loadWav(buf);
        showSpectrogram(feCanvas, feCanvasWrap, wav.samples, wav.sampleRate);
        const result = audioToSpectrogram(wav.samples, wav.sampleRate);
        feStats.textContent = result
            ? `${result.T} frames · ${(wav.samples.length / wav.sampleRate).toFixed(2)} s · ${wav.sampleRate} Hz source`
            : 'WASM not ready — rebuild required for spectrogram';
    } catch (e) {
        feStats.textContent = 'Error: ' + e.message;
        console.error(e);
    }
});

// ── TTS tab ───────────────────────────────────────────────────────────────────

let modelBlobUrl = null;   // Blob URL of the loaded .onnx; passed to synthesis workers
let voiceConfig  = null;

const fileModel   = document.getElementById('file-model');
const fileConfig  = document.getElementById('file-config');
const btnLoad      = document.getElementById('btn-load');
const loadStatus   = document.getElementById('load-status');
const loadSpinner  = document.getElementById('load-spinner');
const loadStatusTx = document.getElementById('load-status-text');
function loadSetStatus(text, cls) {
    loadStatusTx.textContent = text;
    loadStatus.className = 'status' + (cls ? ' ' + cls : '');
}
function loadSetWorking(on) { loadSpinner.style.display = on ? '' : 'none'; }
const phraseInput = document.getElementById('phrase');
const langSelect  = document.getElementById('lang');
const numSamples  = document.getElementById('num-samples');
const btnGenerate = document.getElementById('btn-generate');
const ipaOut      = document.getElementById('ipa-out');
const genProgress = document.getElementById('gen-progress');
const genStatus   = document.getElementById('gen-status');
const genSpinner  = document.getElementById('gen-spinner');
const genStatusTx = document.getElementById('gen-status-text');

function genSetStatus(text, cls) {
    genStatusTx.textContent = text;
    genStatus.className = 'status' + (cls ? ' ' + cls : '');
}
function genSetWorking(on) { genSpinner.style.display = on ? '' : 'none'; }
const audioList   = document.getElementById('audio-list');

function checkFiles() {
    btnLoad.disabled = !(fileModel.files.length && fileConfig.files.length);
}
fileModel.addEventListener('change', checkFiles);
fileConfig.addEventListener('change', checkFiles);

btnLoad.addEventListener('click', async () => {
    btnLoad.disabled = true;
    loadSetWorking(true);
    loadSetStatus('Loading…');
    await new Promise(r => requestAnimationFrame(r));
    try {
        const [modelBuf, configText] = await Promise.all([
            fileModel.files[0].arrayBuffer(),
            fileConfig.files[0].text(),
        ]);
        voiceConfig = JSON.parse(configText);
        if (modelBlobUrl) URL.revokeObjectURL(modelBlobUrl);
        modelBlobUrl = URL.createObjectURL(new Blob([modelBuf]));
        loadSetStatus('✓ Model ready — ' + voiceConfig.audio.sample_rate + ' Hz, ' +
            (voiceConfig.num_speakers ?? 1) + ' speaker(s)', 'ok');
        btnGenerate.disabled = false;
        updatePrereqs();
    } catch (e) {
        loadSetStatus('Error: ' + e.message, 'err');
        btnLoad.disabled = false;
    } finally {
        loadSetWorking(false);
    }
});

btnGenerate.addEventListener('click', async () => {
    if (!modelBlobUrl || !voiceConfig) return;
    btnGenerate.disabled = true;
    audioList.innerHTML  = '';
    ipaOut.textContent   = '';
    genSetStatus('');
    genSetWorking(true);
    genProgress.value = 0;
    genProgress.max   = 1;
    genProgress.style.display = 'block';
    await new Promise(r => requestAnimationFrame(r));
    genProgress.value = 0;
    genProgress.max   = 1;

    const text  = phraseInput.value.trim();
    const lang  = langSelect.value;
    const count = Math.max(1, parseInt(numSamples.value) || 10);

    try {
        const espeakVoice = voiceConfig?.espeak?.voice ?? lang;
        ipaOut.textContent = 'IPA: ' + textToIpa(text, espeakVoice);

        const samples = await generateSamples(modelBlobUrl, voiceConfig, text, lang, count,
            (done, total) => {
                genProgress.value = done;
                genProgress.max   = total;
                genSetStatus(done + ' / ' + total);
            });

        audioList.innerHTML = '';
        samples.forEach((s, i) => {
            const url  = URL.createObjectURL(s.wav);
            const item = document.createElement('div');
            item.className = 'audio-item';
            item.innerHTML =
                `<span class="sample-num">${i + 1}</span>` +
                `<audio controls src="${url}"></audio>` +
                `<span class="sample-params">noise=${s.noiseScale.toFixed(2)} ` +
                `speed=${s.lengthScale.toFixed(2)} pitch=${s.noiseW.toFixed(2)}</span>` +
                `<a href="${url}" download="sample_${i + 1}.wav">↓</a>`;

            // Spectrogram: decode WAV and run frontend
            const canvas = document.createElement('canvas');
            canvas.className = 'spectrogram';
            item.appendChild(canvas);
            s.wav.arrayBuffer().then(ab => {
                const wav = loadWav(ab);
                const res = audioToSpectrogram(wav.samples, wav.sampleRate);
                if (res) drawSpectrogram(canvas, res.features, res.T);
                else canvas.style.display = 'none';
            });

            audioList.appendChild(item);
        });

        genSetStatus(count + ' samples generated.', 'ok');
    } catch (e) {
        genSetStatus('Error: ' + e.message, 'err');
        console.error(e);
    } finally {
        genSetWorking(false);
        genProgress.style.display = 'none';
        btnGenerate.disabled = false;
    }
});

// ── Augment tab ───────────────────────────────────────────────────────────────

let loadedIRs       = null;
let augSourceSamples = null;
let augSourceRate    = null;

const augFileInput   = document.getElementById('aug-file');
const augRunBtn      = document.getElementById('aug-run');
const augStatus      = document.getElementById('aug-status');
const augBeforeAudio = document.getElementById('aug-before-audio');
const augAfterAudio  = document.getElementById('aug-after-audio');
const augBeforeCanvas = document.getElementById('aug-before-canvas');
const augAfterCanvas  = document.getElementById('aug-after-canvas');
const augBeforeWrap  = document.getElementById('aug-before-wrap');
const augAfterWrap   = document.getElementById('aug-after-wrap');
const pitchSlider    = document.getElementById('aug-pitch');
const pitchVal       = document.getElementById('aug-pitch-val');
const eqCheck        = document.getElementById('aug-eq');
const reverbSelect   = document.getElementById('aug-reverb');
const noiseSlider    = document.getElementById('aug-noise-snr');
const noiseVal       = document.getElementById('aug-noise-val');
const noiseCheck     = document.getElementById('aug-noise-en');

pitchSlider.addEventListener('input', () => { pitchVal.textContent = pitchSlider.value; });
noiseSlider.addEventListener('input', () => { noiseVal.textContent = noiseSlider.value; });

// Lazy-load IRs on first augment run.
async function ensureIRs() {
    if (loadedIRs) return loadedIRs;
    loadedIRs = await fetchImpulseResponses('');
    return loadedIRs;
}

augFileInput.addEventListener('change', async () => {
    const file = augFileInput.files[0];
    if (!file) return;
    augStatus.textContent = 'Loading…';
    try {
        const buf = await file.arrayBuffer();
        const wav = loadWav(buf);
        augSourceSamples = wav.samples;
        augSourceRate    = wav.sampleRate;

        const url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
        augBeforeAudio.src = url;
        showSpectrogram(augBeforeCanvas, augBeforeWrap, wav.samples, wav.sampleRate);
        augRunBtn.disabled = false;
        augStatus.textContent = `Loaded ${wav.samples.length} samples @ ${wav.sampleRate} Hz`;
        augStatus.className = 'status ok';
    } catch (e) {
        augStatus.textContent = 'Error: ' + e.message;
        augStatus.className = 'status err';
    }
});

// ── Train tab — negatives ─────────────────────────────────────────────────────

let negDB  = null;
let negCats = null;   // in-memory categories after load

const negStatus   = document.getElementById('neg-status');
const negBadge    = document.getElementById('neg-badge');
const negFile     = document.getElementById('neg-file');
const negUrlInput = document.getElementById('neg-url');
const negDlBtn    = document.getElementById('neg-dl-btn');
const negProgress = document.getElementById('neg-progress');
const negClearBtn = document.getElementById('neg-clear-btn');

document.getElementById('neg-bundled-link').addEventListener('click', e => {
    e.preventDefault();
    negUrlInput.value = 'negatives.mwwn';
    negDlBtn.click();
});

document.getElementById('noise-bundled-link').addEventListener('click', e => {
    e.preventDefault();
    document.getElementById('noise-url').value = 'noise.mwwb';
    document.getElementById('noise-dl-btn').click();
});

function negSetStatus(text, cls) {
    negStatus.textContent = text;
    negStatus.className   = 'status' + (cls ? ' ' + cls : '');
}

function negUpdateCached(info) {
    const parts = Object.entries(info)
        .map(([k, v]) => `${k}: ${v.numSamples.toLocaleString()}`)
        .join(' · ');
    negSetStatus(parts ? `Cached — ${parts}` : 'Cache empty.', 'ok');
    negBadge.textContent = 'cached';
    negClearBtn.style.display = '';
}

async function negInit() {
    try {
        negDB = await negOpenDB();
        if (await negHas(negDB)) {
            const info = await negInfo(negDB);
            negUpdateCached(info);
            negCats = await negLoad(negDB);
        } else {
            negSetStatus('Not cached. Load a negatives.bin bundle or paste a download URL.');
            negBadge.textContent = 'not cached';
        }
    } catch (e) {
        negSetStatus('IndexedDB unavailable: ' + e.message, 'err');
    }
    updatePrereqs();
}

negFile.addEventListener('change', async () => {
    const file = negFile.files[0];
    if (!file) return;
    negSetStatus('Loading…');
    try {
        const buf    = await file.arrayBuffer();
        const parsed = negParseBundle(buf);
        await negStore(negDB, parsed);
        negCats = await negLoad(negDB);
        const info = await negInfo(negDB);
        negUpdateCached(info);
        updatePrereqs();
    } catch (e) {
        negSetStatus('Error: ' + e.message, 'err');
    }
    negFile.value = '';
});

negDlBtn.addEventListener('click', async () => {
    const url = negUrlInput.value.trim();
    if (!url) { negSetStatus('Paste a bundle URL first.', 'err'); return; }
    negDlBtn.disabled    = true;
    negProgress.style.display = '';
    negProgress.value    = 0;
    negProgress.max      = 1;
    negSetStatus('Downloading…');
    try {
        negCats = await negDownload(url, negDB, (loaded, total) => {
            if (total) { negProgress.value = loaded; negProgress.max = total; }
            negSetStatus(`Downloading… ${(loaded / 1024 / 1024).toFixed(1)} MB`);
        });
        const info = await negInfo(negDB);
        negUpdateCached(info);
        updatePrereqs();
    } catch (e) {
        negSetStatus('Download failed: ' + e.message, 'err');
    } finally {
        negProgress.style.display = 'none';
        negDlBtn.disabled = false;
    }
});

negClearBtn.addEventListener('click', async () => {
    if (!negDB) return;
    await negClear(negDB);
    negCats = null;
    negSetStatus('Cache cleared. Load a bundle to continue.');
    negBadge.textContent = 'not cached';
    negClearBtn.style.display = 'none';
});

negInit();

// ── Train tab — background noise ──────────────────────────────────────────────

let noiseParsed = null;

const noiseStatus   = document.getElementById('noise-status');
const noiseBadge    = document.getElementById('noise-badge');
const noiseFile     = document.getElementById('noise-file');
const noiseUrlInput = document.getElementById('noise-url');
const noiseDlBtn    = document.getElementById('noise-dl-btn');
const noiseProgress = document.getElementById('noise-progress');
const noiseClearBtn = document.getElementById('noise-clear-btn');

function noiseSetStatus(text, cls) {
    noiseStatus.textContent = text;
    noiseStatus.className   = 'status' + (cls ? ' ' + cls : '');
}

function noiseUpdateCached(parsed) {
    const secs = (parsed.clipSamples / parsed.sampleRate).toFixed(1);
    noiseSetStatus(`Cached — ${parsed.numClips} clips × ${secs} s @ ${parsed.sampleRate} Hz`, 'ok');
    noiseBadge.textContent = 'cached';
    noiseClearBtn.style.display = '';
}

async function noiseSetup() {
    try {
        noiseParsed = await noiseInit();
        if (noiseParsed) {
            noiseUpdateCached(noiseParsed);
        } else {
            noiseSetStatus('Not cached. Load a noise.bin bundle or paste a download URL.');
            noiseBadge.textContent = 'not cached';
        }
    } catch (e) {
        noiseSetStatus('OPFS unavailable: ' + e.message, 'err');
    }
    updatePrereqs();
}

noiseFile.addEventListener('change', async () => {
    const file = noiseFile.files[0];
    if (!file) return;
    noiseSetStatus('Loading…');
    try {
        const buf = await file.arrayBuffer();
        noiseParsed = noiseParseBundle(buf);
        await noiseStoreToOPFS(buf);
        noiseUpdateCached(noiseParsed);
        updatePrereqs();
    } catch (e) {
        noiseSetStatus('Error: ' + e.message, 'err');
    }
    noiseFile.value = '';
});

noiseDlBtn.addEventListener('click', async () => {
    const url = noiseUrlInput.value.trim();
    if (!url) { noiseSetStatus('Paste a bundle URL first.', 'err'); return; }
    noiseDlBtn.disabled          = true;
    noiseProgress.style.display  = '';
    noiseProgress.value          = 0;
    noiseProgress.max            = 1;
    noiseSetStatus('Downloading…');
    try {
        noiseParsed = await noiseDownload(url, (loaded, total) => {
            if (total) { noiseProgress.value = loaded; noiseProgress.max = total; }
            noiseSetStatus(`Downloading… ${(loaded / 1024 / 1024).toFixed(1)} MB`);
        });
        noiseUpdateCached(noiseParsed);
        updatePrereqs();
    } catch (e) {
        noiseSetStatus('Download failed: ' + e.message, 'err');
    } finally {
        noiseProgress.style.display = 'none';
        noiseDlBtn.disabled = false;
    }
});

noiseClearBtn.addEventListener('click', async () => {
    await noiseClearOPFS();
    noiseParsed = null;
    noiseSetStatus('Cache cleared. Load a bundle to continue.');
    noiseBadge.textContent = 'not cached';
    noiseClearBtn.style.display = 'none';
    updatePrereqs();
});

noiseSetup();

// ── Augment tab ───────────────────────────────────────────────────────────────

augRunBtn.addEventListener('click', async () => {
    if (!augSourceSamples) return;
    augRunBtn.disabled = true;
    augStatus.textContent = 'Augmenting…';

    try {
        const irs = reverbSelect.value !== 'none' ? await ensureIRs() : [];
        const selectedIR = reverbSelect.value === 'all'
            ? irs
            : reverbSelect.value !== 'none'
                ? [irs[parseInt(reverbSelect.value)] || irs[0]]
                : [];

        // Resample to 16kHz for augmentation
        const s16k = audioToSpectrogram !== null
            ? (() => {
                // Use the same resampler as spectrogram.js (private _resampleTo16k is not exposed;
                // just call pitchShift with 0 semitones after manually resampling)
                const r = augSourceRate / 16000;
                const n = Math.round(augSourceSamples.length / r);
                const out = new Float32Array(n);
                for (let i = 0; i < n; i++) {
                    const pos = i * r;
                    const i0  = Math.floor(pos);
                    const i1  = Math.min(i0 + 1, augSourceSamples.length - 1);
                    out[i] = augSourceSamples[i0] * (1 - (pos - i0)) + augSourceSamples[i1] * (pos - i0);
                }
                return out;
            })()
            : augSourceSamples;

        const semitones = parseFloat(pitchSlider.value);

        const noiseClips = (noiseCheck.checked && noiseParsed)
            ? [noiseGetSample(noiseParsed)]
            : [];

        const augmented = augmentSample(s16k, 16000, {
            irs: selectedIR,
            noises: noiseClips,
            pitchProb:    Math.abs(semitones) > 0.01 ? 1.0 : 0.0,
            eqProb:       eqCheck.checked ? 1.0 : 0.0,
            reverbProb:   selectedIR.length > 0 ? 1.0 : 0.0,
            noiseProb:    noiseCheck.checked ? 1.0 : 0.0,
            gainRange:    [0, 0],
            semitoneRange: [semitones, semitones],
            snrRange:     [parseFloat(noiseSlider.value), parseFloat(noiseSlider.value)],
        });

        const wavBlob = encodeWav(augmented, 16000);
        augAfterAudio.src = URL.createObjectURL(wavBlob);
        showSpectrogram(augAfterCanvas, augAfterWrap, augmented, 16000);

        augStatus.textContent = `Done. ${augmented.length} samples @ 16kHz`;
        augStatus.className = 'status ok';
    } catch (e) {
        augStatus.textContent = 'Error: ' + e.message;
        augStatus.className = 'status err';
        console.error(e);
    } finally {
        augRunBtn.disabled = false;
    }
});

// ── Train tab — training loop ─────────────────────────────────────────────────

let _trainNet       = null;   // WASM MixedNet pointer
let _trainAdam      = null;   // WASM Adam pointer
let _trainMinFrames = 157;    // minFrames for current net (pooled=1→157, pooled=0→204)
let _posWavs        = [];     // decoded positive samples: { samples: Float32Array, sampleRate }[]
let _lossHistory    = [];     // per-step smoothed loss values (for chart)
let _abortFlag      = false;

const trainStartBtn   = document.getElementById('train-start-btn');
const trainStopBtn    = document.getElementById('train-stop-btn');
const trainExportBtn  = document.getElementById('train-export-btn');
const trainTfliteBtn  = document.getElementById('train-tflite-btn');
const trainStatus     = document.getElementById('train-status');
const trainProgress   = document.getElementById('train-progress');
const trainLossWrap   = document.getElementById('train-loss-wrap');
const trainLossCanvas = document.getElementById('train-loss-canvas');
const trainPhraseIn   = document.getElementById('train-phrase');
const trainLangSel    = document.getElementById('train-lang');
const trainPosSamples = document.getElementById('train-pos-samples');
const trainStepsIn    = document.getElementById('train-steps');
const trainNegRatio   = document.getElementById('train-neg-ratio');
const trainLrIn       = document.getElementById('train-lr');
const trainArchSel    = document.getElementById('train-arch');

// ── Loss chart ────────────────────────────────────────────────────────────────

function _smoothLoss(raw, win = 20) {
    const out = [];
    let sum = 0;
    const buf = [];
    for (const v of raw) {
        buf.push(v); sum += v;
        if (buf.length > win) sum -= buf.shift();
        out.push(sum / buf.length);
    }
    return out;
}

function drawLossChart(canvas, losses) {
    const W = canvas.width  = canvas.offsetWidth || 600;
    const H = canvas.height = 120;
    if (losses.length < 2) return;

    const ctx = canvas.getContext('2d');
    const pad = { l: 46, r: 10, t: 8, b: 22 };
    const iW  = W - pad.l - pad.r;
    const iH  = H - pad.t - pad.b;

    let lo = Infinity, hi = -Infinity;
    for (const v of losses) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi <= lo) { lo -= 0.05; hi += 0.05; }
    const span = hi - lo;
    lo -= span * 0.05; hi += span * 0.05;

    const xOf = i => pad.l + (i / (losses.length - 1)) * iW;
    const yOf = v => pad.t + (1 - (v - lo) / (hi - lo)) * iH;

    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
        const y = pad.t + g * iH / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
        ctx.fillStyle = '#aaa'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
        ctx.fillText((hi - g * (hi - lo) / 4).toFixed(3), pad.l - 3, y + 3);
    }

    ctx.beginPath(); ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 1.5;
    ctx.moveTo(xOf(0), yOf(losses[0]));
    for (let i = 1; i < losses.length; i++) ctx.lineTo(xOf(i), yOf(losses[i]));
    ctx.stroke();

    ctx.fillStyle = '#888'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(`step (${losses.length})`, pad.l + iW / 2, H - 4);
}

// ── Prepare phase ─────────────────────────────────────────────────────────────

trainPrepareBtn.addEventListener('click', async () => {
    if (!modelBlobUrl || !voiceConfig || !negCats) {
        trainStatus.textContent = 'Prerequisites not met — check status badges above.';
        trainStatus.className = 'status err';
        return;
    }

    trainPrepareBtn.disabled = true;
    trainStartBtn.disabled   = true;
    trainProgress.style.display = '';
    trainProgress.value = 0;
    const count = Math.max(5, parseInt(trainPosSamples.value) || 50);
    trainProgress.max = count;
    trainStatus.textContent = 'Generating TTS samples…';
    trainStatus.className = 'status';

    try {
        await ensureIRs();   // lazy-load impulse responses for augmentation

        const raw = await generateSamples(
            modelBlobUrl, voiceConfig,
            trainPhraseIn.value.trim() || 'hey lumus',
            trainLangSel.value,
            count,
            (done, total) => {
                trainProgress.value = done;
                trainProgress.max   = total;
                trainStatus.textContent = `Generating TTS… ${done} / ${total}`;
            }
        );

        _posWavs = [];
        for (const s of raw) {
            const buf = await s.wav.arrayBuffer();
            _posWavs.push(loadWav(buf));
        }

        trainStatus.textContent = `Ready — ${_posWavs.length} samples generated. Click "Start Training".`;
        trainStatus.className = 'status ok';
        trainStartBtn.disabled = false;
    } catch (e) {
        trainStatus.textContent = 'Prepare failed: ' + e.message;
        trainStatus.className = 'status err';
        console.error(e);
    } finally {
        trainProgress.style.display = 'none';
        trainPrepareBtn.disabled = false;
    }
});

// ── Training loop ─────────────────────────────────────────────────────────────

const _NEG_CATS = ['speech', 'no_speech', 'dinner_party'];

function _resample(samples, srcRate) {
    if (srcRate === 16000) return samples;
    const r = srcRate / 16000;
    const n = Math.round(samples.length / r);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const pos = i * r;
        const i0  = Math.floor(pos);
        const i1  = Math.min(i0 + 1, samples.length - 1);
        out[i] = samples[i0] * (1 - (pos - i0)) + samples[i1] * (pos - i0);
    }
    return out;
}

trainStartBtn.addEventListener('click', async () => {
    if (_posWavs.length === 0 || !negCats) return;

    const numSteps  = Math.max(1, parseInt(trainStepsIn.value)  || 1000);
    const negPerPos = Math.max(1, parseInt(trainNegRatio.value) || 5);
    const lr        = parseFloat(trainLrIn.value) || 1e-3;
    const pooled    = trainArchSel ? parseInt(trainArchSel.value) : 1;

    // Reset state
    if (_trainNet) trainDestroy(_trainNet, _trainAdam);
    _lossHistory = [];
    _abortFlag   = false;

    const { net, adam, minFrames } = trainCreate(lr, pooled);
    _trainNet = net; _trainAdam = adam; _trainMinFrames = minFrames;

    trainStartBtn.disabled  = true;
    trainPrepareBtn.disabled = true;
    trainStopBtn.disabled   = false;
    trainExportBtn.disabled = false;
    trainTfliteBtn.disabled = false;
    trainProgress.style.display = '';
    trainProgress.value = 0;
    trainProgress.max   = numSteps;
    trainLossWrap.style.display = '';

    let _skipCount = 0;
    for (let step = 0; step < numSteps && !_abortFlag; step++) {
        try {
        // Positive step
        const wav  = _posWavs[Math.floor(Math.random() * _posWavs.length)];
        const s16k = _resample(wav.samples, wav.sampleRate);
        if (s16k.length > 80000) { console.warn(`skip overlong sample: ${s16k.length} samples`); continue; }
        const aug  = augmentSample(s16k, 16000, {
            irs:        loadedIRs || [],
            noises:     noiseParsed ? [noiseGetSample(noiseParsed)] : [],
            pitchProb:  0.5,
            eqProb:     0.5,
            reverbProb: loadedIRs && loadedIRs.length > 0 ? 0.5 : 0.0,
            noiseProb:  noiseParsed ? 0.75 : 0.0,
            gainRange:  [-45, 0],
        });
        window._dbgAugLen = aug.length;
        const feat = audioToFloat32Spec(aug, 16000, _trainMinFrames);
        if (!feat) continue;

        const posLoss = trainStep(_trainNet, _trainAdam, feat.spectrogram, feat.T, 1.0);

        // Negative steps
        let negLossSum = 0;
        for (let j = 0; j < negPerPos; j++) {
            const cat     = _NEG_CATS[Math.floor(Math.random() * _NEG_CATS.length)];
            let   spec    = negGetSample(negCats, cat);  // Float32Array(160×40), already scaled
            let   negT    = 160;
            // pooled=0 needs exactly minFrames=204; zero-pad if needed
            if (_trainMinFrames > negT) {
                const padded = new Float32Array(_trainMinFrames * 40);
                padded.set(spec);
                spec = padded;
                negT = _trainMinFrames;
            }
            negLossSum += trainStep(_trainNet, _trainAdam, spec, negT, 0.0);
        }

        _lossHistory.push((posLoss + negLossSum / negPerPos) / 2);
        } catch (e) {
            _skipCount++;
            console.error(`Training step ${step} skipped (${e.message}). T=${window._dbgT}, aug.len=${window._dbgAugLen}`);
            continue;
        }

        if (step % 10 === 0) {
            trainProgress.value = step + 1;
            const loss = _lossHistory[_lossHistory.length - 1];
            trainStatus.textContent = `Step ${step + 1} / ${numSteps} — loss ${loss.toFixed(4)}`;
            trainStatus.className = 'status';
            drawLossChart(trainLossCanvas, _smoothLoss(_lossHistory));
            await new Promise(r => setTimeout(r, 0));
        }
    }

    // Final chart
    if (_lossHistory.length > 0) drawLossChart(trainLossCanvas, _smoothLoss(_lossHistory));
    trainProgress.style.display = 'none';
    trainStatus.textContent = _abortFlag
        ? `Stopped at step ${_lossHistory.length} — loss ${_lossHistory[_lossHistory.length - 1]?.toFixed(4) ?? '?'}`
        : `Training complete — ${_lossHistory.length} steps, final loss ${_lossHistory[_lossHistory.length - 1]?.toFixed(4) ?? '?'}`;
    trainStatus.className = 'status ok';
    trainStartBtn.disabled  = false;
    trainPrepareBtn.disabled = false;
    trainStopBtn.disabled   = true;
});

trainStopBtn.addEventListener('click', () => { _abortFlag = true; });

// ── Export weights ────────────────────────────────────────────────────────────

trainExportBtn.addEventListener('click', () => {
    if (!_trainNet) return;

    const params = trainGetParams(_trainNet);

    // Bundle format: magic "MWWW" + uint32 num_params + float32[] weights
    const buf = new ArrayBuffer(8 + params.byteLength);
    const dv  = new DataView(buf);
    dv.setUint8(0, 0x4D); dv.setUint8(1, 0x57);   // "MW"
    dv.setUint8(2, 0x57); dv.setUint8(3, 0x57);   // "WW"
    dv.setUint32(4, params.length, true);
    new Float32Array(buf, 8).set(params);

    const blob = new Blob([buf], { type: 'application/octet-stream' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'wakeword_weights.bin'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
});

// ── Export TFLite ─────────────────────────────────────────────────────────────

trainTfliteBtn.addEventListener('click', async () => {
    if (!_trainNet) return;

    trainTfliteBtn.disabled = true;
    trainStatus.textContent = 'Exporting TFLite… (calibrating INT8 quantization)';
    trainStatus.className = 'status';

    try {
        const resp = await fetch('tflite_template.bin');
        if (!resp.ok) throw new Error(`Template fetch failed: ${resp.status}`);
        const tmplBytes = new Uint8Array(await resp.arrayBuffer());

        // Run in next microtask so the status text renders first
        await new Promise(r => setTimeout(r, 0));

        const tflite = trainExportTFLite(_trainNet, tmplBytes, 500);
        if (!tflite) throw new Error('Export returned null (check console)');

        const blob = new Blob([tflite], { type: 'application/octet-stream' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = 'wakeword.tflite'; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 60000);

        trainStatus.textContent = `TFLite exported (${(tflite.byteLength / 1024).toFixed(1)} KB)`;
        trainStatus.className = 'status ok';
    } catch (e) {
        trainStatus.textContent = 'TFLite export failed: ' + e.message;
        trainStatus.className = 'status err';
    } finally {
        trainTfliteBtn.disabled = false;
    }
});
