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

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

// Phonemizer (espeak-ng WASM) — used by TTS tab.
initPhonemizer()
    .then(() => {
        document.getElementById('load-status').textContent = 'Phonemizer ready. Select model files above.';
        document.getElementById('load-status').className = 'status ok';
    })
    .catch(e => {
        document.getElementById('load-status').textContent = 'Phonemizer init failed: ' + e.message;
        document.getElementById('load-status').className = 'status err';
    });

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
const feStats       = document.getElementById('fe-stats');

feFileInput.addEventListener('change', async () => {
    const file = feFileInput.files[0];
    if (!file) return;
    feStats.textContent = 'Processing…';
    try {
        const buf  = await file.arrayBuffer();
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

let ortSession  = null;
let voiceConfig = null;

const fileModel   = document.getElementById('file-model');
const fileConfig  = document.getElementById('file-config');
const btnLoad     = document.getElementById('btn-load');
const loadStatus  = document.getElementById('load-status');
const phraseInput = document.getElementById('phrase');
const langSelect  = document.getElementById('lang');
const numSamples  = document.getElementById('num-samples');
const btnGenerate = document.getElementById('btn-generate');
const ipaOut      = document.getElementById('ipa-out');
const genProgress = document.getElementById('gen-progress');
const genStatus   = document.getElementById('gen-status');
const audioList   = document.getElementById('audio-list');

function checkFiles() {
    btnLoad.disabled = !(fileModel.files.length && fileConfig.files.length);
}
fileModel.addEventListener('change', checkFiles);
fileConfig.addEventListener('change', checkFiles);

btnLoad.addEventListener('click', async () => {
    btnLoad.disabled = true;
    loadStatus.textContent = 'Loading…';
    loadStatus.className = 'status';
    try {
        const [modelBuf, configText] = await Promise.all([
            fileModel.files[0].arrayBuffer(),
            fileConfig.files[0].text(),
        ]);
        voiceConfig = JSON.parse(configText);
        ortSession  = await ort.InferenceSession.create(modelBuf, { executionProviders: ['wasm'] });
        loadStatus.textContent = '✓ Model loaded — ' + voiceConfig.audio.sample_rate + ' Hz, ' +
            (voiceConfig.num_speakers ?? 1) + ' speaker(s)';
        loadStatus.className = 'status ok';
        btnGenerate.disabled = false;
    } catch (e) {
        loadStatus.textContent = 'Error: ' + e.message;
        loadStatus.className = 'status err';
        btnLoad.disabled = false;
    }
});

btnGenerate.addEventListener('click', async () => {
    if (!ortSession || !voiceConfig) return;
    btnGenerate.disabled = true;
    audioList.innerHTML  = '';
    ipaOut.textContent   = '';
    genStatus.textContent = '';
    genProgress.style.display = 'block';
    genProgress.value = 0;
    genProgress.max   = 1;

    const text  = phraseInput.value.trim();
    const lang  = langSelect.value;
    const count = Math.max(1, parseInt(numSamples.value) || 10);

    try {
        ipaOut.textContent = 'IPA: ' + textToIpa(text, lang);

        const samples = await generateSamples(ortSession, voiceConfig, text, lang, count,
            (done, total) => {
                genProgress.value = done;
                genProgress.max   = total;
                genStatus.textContent = done + ' / ' + total;
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

        genStatus.textContent = count + ' samples generated.';
        genStatus.className = 'status ok';
    } catch (e) {
        genStatus.textContent = 'Error: ' + e.message;
        genStatus.className = 'status err';
        console.error(e);
    } finally {
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
