'use strict';

// ── Secure-context warning ────────────────────────────────────────────────────
if (!window.isSecureContext) {
    document.getElementById('ssl-warning').style.display = 'block';
}

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

const fileVoice   = document.getElementById('file-voice');
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

// The .onnx and .onnx.json can be picked together (multi-select) or one at a
// time — each selection fills its slot and keeps whatever the other one holds.
let pickedModel  = null;
let pickedConfig = null;

fileVoice.addEventListener('change', () => {
    for (const f of fileVoice.files) {
        if (f.name.endsWith('.onnx'))      pickedModel  = f;
        else if (f.name.endsWith('.json')) pickedConfig = f;
    }
    // Reset so re-selecting the same file always fires 'change' again.
    fileVoice.value = '';
    if (!pickedModel)
        loadSetStatus('✓ ' + pickedConfig.name + ' — still missing the model: ' +
            'click the picker again and select the .onnx file', 'warn');
    else if (!pickedConfig)
        loadSetStatus('✓ ' + pickedModel.name + ' — still missing the config: ' +
            'click the picker again and select the .onnx.json file', 'warn');
    else
        // Defer past the change event so the file dialog can tear down
        // before the (heavy) model read/caching starts.
        setTimeout(loadVoiceModel, 0);
});

async function loadVoiceModel() {
    loadSetWorking(true);
    loadSetStatus('Loading…');
    await new Promise(r => requestAnimationFrame(r));
    try {
        const [modelBuf, configText] = await Promise.all([
            pickedModel.arrayBuffer(),
            pickedConfig.text(),
        ]);
        voiceConfig = JSON.parse(configText);
        if (modelBlobUrl) URL.revokeObjectURL(modelBlobUrl);
        modelBlobUrl = URL.createObjectURL(new Blob([modelBuf]));
        loadSetStatus('✓ Model ready — ' + voiceConfig.audio.sample_rate + ' Hz, ' +
            (voiceConfig.num_speakers ?? 1) + ' speaker(s)', 'ok');
        btnGenerate.disabled = false;
        updatePrereqs();
        updatePreviewModelStatus();
        _applyModelToUI(voiceConfig);
        piperCacheSave(modelBuf, configText).catch(e => console.warn('piper cache save:', e));
    } catch (e) {
        loadSetStatus('Error: ' + e.message, 'err');
    } finally {
        loadSetWorking(false);
    }
}

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

// ── Piper model OPFS cache ────────────────────────────────────────────────────

const ttsCachedRow     = document.getElementById('tts-cached-row');
const ttsCacheStatus   = document.getElementById('tts-cache-status');
const ttsCacheClearBtn = document.getElementById('tts-cache-clear-btn');

function _applyModelToUI(cfg) {
    const label = `✓ Cached: ${cfg.audio.sample_rate} Hz, ${cfg.num_speakers ?? 1} speaker(s)`;
    ttsCacheStatus.textContent = label;
    ttsCachedRow.style.display = '';
}

ttsCacheClearBtn.addEventListener('click', async () => {
    await piperCacheClear();
    ttsCachedRow.style.display = 'none';
});

// Auto-load model from OPFS on startup.
(async () => {
    try {
        if (!(await piperCacheHas())) return;
        const { onnxBuf, configText } = await piperCacheLoad();
        voiceConfig  = JSON.parse(configText);
        modelBlobUrl = URL.createObjectURL(new Blob([onnxBuf]));
        loadSetStatus('✓ Model loaded from cache — ' + voiceConfig.audio.sample_rate + ' Hz', 'ok');
        btnGenerate.disabled = false;
        updatePrereqs();
        updatePreviewModelStatus();
        _applyModelToUI(voiceConfig);
    } catch (e) {
        console.warn('piper cache load:', e);
    }
})();

// ── In-browser negative generation ───────────────────────────────────────────

const negGenBtn      = document.getElementById('neg-gen-btn');
const negGenCount    = document.getElementById('neg-gen-count');
const negGenLang     = document.getElementById('neg-gen-lang');
const negGenProgress = document.getElementById('neg-gen-progress');
const negGenStatus   = document.getElementById('neg-gen-status');

// Diverse phrases that are NOT the wake word, covering varied phoneme patterns.
const _NEG_PHRASES_EN = [
    'what time is it', 'how are you doing today', "what's the weather like",
    'can you help me with something', 'turn on the lights', 'turn off the television',
    'set a timer for five minutes', 'play some music please', 'increase the volume',
    'good morning everyone', "I'll be back in a minute", 'thank you very much',
    'the weather looks nice today', "let's have dinner at seven", "I'm running a bit late",
    'one two three four five six seven eight nine ten',
    'the meeting is on Thursday at three o clock',
    'did you remember to charge your phone', 'we should go for a walk later',
    'I think it is going to rain tomorrow', 'we need to buy more coffee',
    'did you watch the game last night', 'the kids want pizza for dinner',
    'hello how are you', 'goodbye see you later', 'yes of course', 'no thank you',
    'okay sounds good', 'sure why not', 'never mind forget it', 'just a second',
    'I was just thinking about what to have for lunch',
    'the traffic was terrible this morning',
    'the project deadline has been moved to next Friday',
    'can someone please pass the salt',
    'I have a doctor appointment tomorrow morning',
];
const _NEG_PHRASES_ES = [
    'buenos días cómo estás', 'buenas tardes qué tal', 'buenas noches hasta mañana',
    'muchas gracias de nada', 'por favor necesito ayuda', 'qué hora es ahora mismo',
    'cuántos años tienes', 'dónde está el baño por favor', 'está bien perfecto',
    'hasta luego nos vemos mañana', 'uno dos tres cuatro cinco seis siete',
    'el tiempo está muy bonito hoy', 'vamos a cenar a las ocho',
    'puedes ayudarme con algo', 'enciende las luces de la sala',
    'apaga la televisión por favor', 'pon música suave de fondo',
    'necesito un vaso de agua', 'qué hay de comer hoy',
    'tengo una reunión a las tres de la tarde',
    'el tráfico estaba terrible esta mañana',
    'no te olvides de tomar tu medicina',
    'cuánto tiempo tardará esto', 'puedes repetir eso por favor',
    'me parece muy buena idea', 'qué película vemos esta noche',
];

function _negGenSetStatus(text, cls) {
    negGenStatus.textContent = text;
    negGenStatus.className   = 'status' + (cls ? ' ' + cls : '');
}

// Pad or crop a Uint16Array spectrogram to exactly targetT × numFeats frames.
function _padSpec(features, T, targetT = 160, numFeats = 40) {
    const out   = new Uint16Array(targetT * numFeats);
    const copyT = Math.min(T, targetT);
    out.set(features.subarray(0, copyT * numFeats));
    return out;
}

// Generate a silence+noise clip at 16 kHz and return its Uint16 spectrogram.
function _silenceSpec(durationMs, noiseAmp = 0.005) {
    const n       = Math.round(16000 * durationMs / 1000);
    const samples = new Float32Array(n);
    for (let i = 0; i < n; i++)
        samples[i] = (Math.random() * 2 - 1) * noiseAmp;
    const res = audioToSpectrogram(samples, 16000);
    return res ? _padSpec(res.features, res.T) : new Uint16Array(160 * 40);
}

negGenBtn.addEventListener('click', async () => {
    if (!modelBlobUrl || !voiceConfig) {
        _negGenSetStatus('Load a Piper model in the TTS tab first.', 'err');
        return;
    }
    if (!window.webTrainerReady) {
        _negGenSetStatus('WASM not ready yet — try again in a moment.', 'err');
        return;
    }

    negGenBtn.disabled = true;
    negGenProgress.style.display = '';
    _negGenSetStatus('Generating…');

    if (!negDB) negDB = await negOpenDB();

    const N      = Math.max(20, parseInt(negGenCount.value) || 150);
    const lang   = negGenLang.value;
    const phrases = lang === 'es' ? _NEG_PHRASES_ES : _NEG_PHRASES_EN;

    // How many TTS samples we need (speech + dinner_party share the same synthesised audio).
    const ttsNeeded    = N * 2;  // half for speech, half repurposed for dinner_party
    const perPhrase    = Math.max(1, Math.ceil(ttsNeeded / phrases.length));
    const totalSteps   = phrases.length + N;  // TTS phrases + no_speech clips
    let   stepsDone    = 0;

    function _tick(label) {
        stepsDone++;
        negGenProgress.max   = totalSteps;
        negGenProgress.value = stepsDone;
        _negGenSetStatus(label);
    }

    try {
        // ── 1. Synthesise speech audio ────────────────────────────────────────
        const allSpeechSpecs    = [];  // Uint16Array[160*40]
        const allDinnerSpecs    = [];  // speech + noise overlay

        for (let pi = 0; pi < phrases.length; pi++) {
            const phrase  = phrases[pi];
            const samples = await generateSamples(
                modelBlobUrl, voiceConfig, phrase, lang, perPhrase, null);

            for (const s of samples) {
                const buf  = await s.wav.arrayBuffer();
                const wav  = loadWav(buf);
                // Resample to 16 kHz
                const r    = wav.sampleRate / 16000;
                const n16  = Math.round(wav.samples.length / r);
                const s16k = new Float32Array(n16);
                for (let i = 0; i < n16; i++) {
                    const pos = i * r;
                    const i0  = Math.floor(pos);
                    const i1  = Math.min(i0 + 1, wav.samples.length - 1);
                    s16k[i]   = wav.samples[i0] * (1 - (pos - i0)) + wav.samples[i1] * (pos - i0);
                }

                // speech: clean spectrogram
                const res = audioToSpectrogram(s16k, 16000);
                if (res) allSpeechSpecs.push(_padSpec(res.features, res.T));

                // dinner_party: same audio + white noise overlay (SNR ~5 dB)
                const noisy = s16k.slice();
                const amp   = 0.03 + Math.random() * 0.07;
                for (let i = 0; i < noisy.length; i++)
                    noisy[i] += (Math.random() * 2 - 1) * amp;
                const res2 = audioToSpectrogram(noisy, 16000);
                if (res2) allDinnerSpecs.push(_padSpec(res2.features, res2.T));
            }

            _tick(`Synthesising phrase ${pi + 1} / ${phrases.length}…`);
        }

        // ── 2. Silence clips for no_speech ────────────────────────────────────
        const allSilenceSpecs = [];
        for (let i = 0; i < N; i++) {
            const ms  = 500 + Math.random() * 2000;  // 0.5 – 2.5 s
            const amp = 0.001 + Math.random() * 0.015;
            allSilenceSpecs.push(_silenceSpec(ms, amp));
            if (i % 20 === 0) _tick(`Generating silence ${i + 1} / ${N}…`);
        }

        // ── 3. Trim to N samples each ─────────────────────────────────────────
        const speechData  = _packSpecs(allSpeechSpecs.slice(0, N));
        const dinnerData  = _packSpecs(allDinnerSpecs.slice(0, N));
        const silenceData = _packSpecs(allSilenceSpecs.slice(0, N));

        // ── 4. Store in IndexedDB ─────────────────────────────────────────────
        _negGenSetStatus('Saving to cache…');
        await negStore(negDB, {
            categories: {
                speech:       { numSamples: speechData.n,  numFrames: 160, numFeatures: 40, data: speechData.buf },
                dinner_party: { numSamples: dinnerData.n,  numFrames: 160, numFeatures: 40, data: dinnerData.buf },
                no_speech:    { numSamples: silenceData.n, numFrames: 160, numFeatures: 40, data: silenceData.buf },
            },
        });
        negCats = await negLoad(negDB);
        negSetStatus(`Generated: ${speechData.n} speech · ${dinnerData.n} dinner_party · ${silenceData.n} no_speech`, 'ok');
        negBadge.textContent = 'cached';
        negClearBtn.style.display = '';
        updatePrereqs();
        _negGenSetStatus(`Done — ${speechData.n + dinnerData.n + silenceData.n} spectrograms stored.`, 'ok');
    } catch (err) {
        _negGenSetStatus('Error: ' + err.message, 'err');
        console.error(err);
    } finally {
        negGenBtn.disabled           = false;
        negGenProgress.style.display = 'none';
    }
});

// Pack an array of Uint16Array[160*40] into one flat Uint16Array.
function _packSpecs(specs) {
    const n   = specs.length;
    const buf = new Uint16Array(n * 160 * 40);
    for (let i = 0; i < n; i++) buf.set(specs[i], i * 160 * 40);
    return { n, buf };
}

// ── Augment tab ───────────────────────────────────────────────────────────────

augRunBtn.addEventListener('click', async () => {
    if (!augSourceSamples) return;
    augRunBtn.disabled = true;
    augStatus.textContent = 'Augmenting…';

    try {
        const irs = reverbSelect.value !== 'none' ? await ensureIRs() : [];
        const selectedIR = irs.length === 0 ? []
            : reverbSelect.value === 'all'
                ? irs
                : reverbSelect.value !== 'none'
                    ? [irs[parseInt(reverbSelect.value)] ?? irs[0]]
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

let _posWavs       = [];     // decoded positive samples: { samples: Float32Array, sampleRate }[]
let _recordedWavs  = [];     // mic-recorded positive samples added via Record tab
let _lossHistory   = [];     // per-step loss values accumulated from worker progress messages
let _trainedParams = null;   // Float32Array of model weights, received from worker after training
let _trainWorker   = null;   // Web Worker running the training loop

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

    if (_trainWorker) { _trainWorker.terminate(); _trainWorker = null; }
    _trainedParams = null;

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

trainStartBtn.addEventListener('click', () => {
    const _allPosWavs = [..._posWavs, ..._recordedWavs];
    if (_allPosWavs.length === 0 || !negCats) return;

    const numSteps  = Math.max(1, parseInt(trainStepsIn.value)  || 1000);
    const negPerPos = Math.max(1, parseInt(trainNegRatio.value) || 5);
    const lr        = parseFloat(trainLrIn.value) || 1e-3;
    const pooled    = trainArchSel ? parseInt(trainArchSel.value) : 1;

    // Terminate any prior worker before starting a new run.
    if (_trainWorker) { _trainWorker.terminate(); _trainWorker = null; }
    _lossHistory   = [];
    _trainedParams = null;

    trainStartBtn.disabled   = true;
    trainPrepareBtn.disabled = true;
    trainStopBtn.disabled    = false;
    trainExportBtn.disabled  = true;
    trainTfliteBtn.disabled  = true;
    trainProgress.style.display = '';
    trainProgress.value = 0;
    trainProgress.max   = numSteps;
    trainLossWrap.style.display = '';

    _trainWorker = new Worker('js/train_worker.js');

    _trainWorker.onmessage = (e) => {
        const d = e.data;

        if (d.type === 'progress') {
            _lossHistory.push(...d.losses);
            trainProgress.value = d.step;
            const latest = _lossHistory[_lossHistory.length - 1];
            trainStatus.textContent = `Step ${d.step} / ${d.numSteps} — loss ${latest.toFixed(4)}`;
            trainStatus.className = 'status';
            drawLossChart(trainLossCanvas, _smoothLoss(_lossHistory));
            return;
        }

        if (d.type === 'done') {
            _trainedParams = d.params;
            if (_lossHistory.length > 0) drawLossChart(trainLossCanvas, _smoothLoss(_lossHistory));
            trainProgress.style.display = 'none';
            const finalLoss = _lossHistory[_lossHistory.length - 1]?.toFixed(4) ?? '?';
            trainStatus.textContent = d.stopped
                ? `Stopped at step ${_lossHistory.length} — loss ${finalLoss}`
                : `Training complete — ${_lossHistory.length} steps, final loss ${finalLoss}`;
            trainStatus.className    = 'status ok';
            trainStartBtn.disabled   = false;
            trainPrepareBtn.disabled = false;
            trainStopBtn.disabled    = true;
            trainExportBtn.disabled  = false;
            trainTfliteBtn.disabled  = false;
        }
    };

    _trainWorker.onerror = (e) => {
        trainStatus.textContent  = 'Worker error: ' + e.message;
        trainStatus.className    = 'status err';
        trainStartBtn.disabled   = false;
        trainPrepareBtn.disabled = false;
        trainStopBtn.disabled    = true;
    };

    _trainWorker.postMessage({
        type:        'train',
        posWavs:     _allPosWavs,
        negCats,
        noiseParsed,
        irs:         loadedIRs || [],
        numSteps,
        negPerPos,
        lr,
        pooled,
    });
});

trainStopBtn.addEventListener('click', () => { _trainWorker?.postMessage({ type: 'stop' }); });

// ── Export weights ────────────────────────────────────────────────────────────

trainExportBtn.addEventListener('click', () => {
    if (!_trainedParams) return;

    const params = _trainedParams;
    const buf    = new ArrayBuffer(8 + params.byteLength);
    const dv     = new DataView(buf);
    dv.setUint8(0, 0x4D); dv.setUint8(1, 0x57);
    dv.setUint8(2, 0x57); dv.setUint8(3, 0x57);
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
    if (!_trainWorker || !_trainedParams) return;

    trainTfliteBtn.disabled = true;
    trainStatus.textContent = 'Exporting TFLite… (calibrating INT8 quantization)';
    trainStatus.className   = 'status';

    try {
        const resp = await fetch('tflite_template.bin');
        if (!resp.ok) throw new Error(`Template fetch failed: ${resp.status}`);
        const tmplBytes = new Uint8Array(await resp.arrayBuffer());

        const tfliteBytes = await new Promise((resolve, reject) => {
            const h = (ev) => {
                if (ev.data.type === 'tflite') {
                    _trainWorker.removeEventListener('message', h);
                    resolve(ev.data.bytes);
                } else if (ev.data.type === 'tflite_error') {
                    _trainWorker.removeEventListener('message', h);
                    reject(new Error(ev.data.message));
                }
            };
            _trainWorker.addEventListener('message', h);
            _trainWorker.postMessage(
                { type: 'export_tflite', templateBytes: tmplBytes, calibN: 500 },
                [tmplBytes.buffer]
            );
        });

        const blob = new Blob([tfliteBytes], { type: 'application/octet-stream' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = 'wakeword.tflite'; a.click();
        setTimeout(() => URL.revokeObjectURL(url), 60000);

        try { await tfliteCacheSave(tfliteBytes); } catch(e) { console.warn('tflite cache save:', e); }

        trainStatus.textContent = `TFLite exported (${(tfliteBytes.byteLength / 1024).toFixed(1)} KB) — available on Test tab`;
        trainStatus.className   = 'status ok';
    } catch (e) {
        trainStatus.textContent = 'TFLite export failed: ' + e.message;
        trainStatus.className   = 'status err';
    } finally {
        trainTfliteBtn.disabled = false;
    }
});

// ── Preview tab ───────────────────────────────────────────────────────────────

const prevPhrase     = document.getElementById('prev-phrase');
const prevLang       = document.getElementById('prev-lang');
const prevCount      = document.getElementById('prev-count');
const prevGenBtn     = document.getElementById('prev-gen-btn');
const prevStatus     = document.getElementById('prev-status');
const prevProgress   = document.getElementById('prev-progress');
const prevResults    = document.getElementById('prev-results');
const prevModelStat  = document.getElementById('prev-model-status');
const prevPitchProb  = document.getElementById('prev-pitch-prob');
const prevPitchSt    = document.getElementById('prev-pitch-st');
const prevReverbProb = document.getElementById('prev-reverb-prob');
const prevNoiseProb  = document.getElementById('prev-noise-prob');
const prevSnrMin     = document.getElementById('prev-snr-min');
const prevEq         = document.getElementById('prev-eq');

[
    [prevPitchProb,  document.getElementById('prev-pitch-prob-val')],
    [prevPitchSt,    document.getElementById('prev-pitch-st-val')],
    [prevReverbProb, document.getElementById('prev-reverb-prob-val')],
    [prevNoiseProb,  document.getElementById('prev-noise-prob-val')],
    [prevSnrMin,     document.getElementById('prev-snr-min-val')],
].forEach(([el, out]) => el.addEventListener('input', () => { out.textContent = el.value; }));

function updatePreviewModelStatus() {
    if (modelBlobUrl) {
        const sr = voiceConfig?.audio?.sample_rate ?? '?';
        prevModelStat.textContent = `✓ Model ready — ${sr} Hz`;
        prevModelStat.className   = 'status ok';
        prevGenBtn.disabled       = false;
    } else {
        prevModelStat.textContent = 'No model loaded — go to the TTS tab first.';
        prevModelStat.className   = 'status';
        prevGenBtn.disabled       = true;
    }
}

prevGenBtn.addEventListener('click', async () => {
    if (!modelBlobUrl || !voiceConfig) return;

    prevGenBtn.disabled      = true;
    prevResults.innerHTML    = '';
    prevStatus.textContent   = '';
    prevStatus.className     = 'status';
    prevProgress.style.display = '';
    prevProgress.value = 0;

    const phrase     = prevPhrase.value.trim() || 'hey lumus';
    const lang       = prevLang.value;
    const count      = Math.max(1, parseInt(prevCount.value) || 8);
    const pitchProb  = parseFloat(prevPitchProb.value);
    const pitchSt    = parseFloat(prevPitchSt.value);
    const reverbProb = parseFloat(prevReverbProb.value);
    const noiseProb  = parseFloat(prevNoiseProb.value);
    const snrMin     = parseFloat(prevSnrMin.value);
    const eqProb     = prevEq.checked ? 0.5 : 0.0;

    try {
        let irs = loadedIRs || [];
        if (reverbProb > 0 && irs.length === 0) irs = await ensureIRs();

        prevProgress.max = count;
        prevStatus.textContent = 'Generating TTS…';

        const samples = await generateSamples(
            modelBlobUrl, voiceConfig, phrase, lang, count,
            (done, total) => {
                prevProgress.value = done;
                prevStatus.textContent = `Generating TTS… ${done} / ${total}`;
            }
        );

        prevStatus.textContent = 'Augmenting…';

        for (let i = 0; i < samples.length; i++) {
            const s   = samples[i];
            const buf = await s.wav.arrayBuffer();
            const wav = loadWav(buf);

            // Resample to 16 kHz
            const ratio = wav.sampleRate / 16000;
            const n16   = Math.round(wav.samples.length / ratio);
            const s16k  = new Float32Array(n16);
            for (let j = 0; j < n16; j++) {
                const pos = j * ratio;
                const i0  = Math.floor(pos);
                const i1  = Math.min(i0 + 1, wav.samples.length - 1);
                s16k[j] = wav.samples[i0] * (1 - (pos - i0)) + wav.samples[i1] * (pos - i0);
            }

            const aug = augmentSample(s16k, 16000, {
                irs,
                noises:        noiseParsed ? [noiseGetSample(noiseParsed)] : [],
                pitchProb,
                eqProb,
                reverbProb:    irs.length > 0 ? reverbProb : 0.0,
                noiseProb:     noiseParsed ? noiseProb : 0.0,
                gainRange:     [-45, 0],
                semitoneRange: [-pitchSt, pitchSt],
                snrRange:      [snrMin, 10],
            });

            const url  = URL.createObjectURL(encodeWav(aug, 16000));
            const card = document.createElement('div');
            card.className = 'prev-card';
            card.innerHTML =
                `<div class="prev-card-top">` +
                `<span class="sample-num">${i + 1}</span>` +
                `<audio controls src="${url}" style="height:28px;flex:1"></audio>` +
                `<span class="sample-params">` +
                    `noise=${s.noiseScale.toFixed(2)} ` +
                    `speed=${s.lengthScale.toFixed(2)} ` +
                    `pitch=${s.noiseW.toFixed(2)}` +
                `</span>` +
                `<a href="${url}" download="preview_${i + 1}.wav">↓</a>` +
                `</div>`;

            const res = audioToSpectrogram(aug, 16000);
            if (res) {
                const canvas = document.createElement('canvas');
                canvas.className = 'spectrogram';
                const wrap = document.createElement('div');
                wrap.className = 'spec-wrap';
                wrap.appendChild(canvas);
                card.appendChild(wrap);
                drawSpectrogram(canvas, res.features, res.T);
            }
            prevResults.appendChild(card);
        }

        prevStatus.textContent = `${samples.length} samples generated and augmented.`;
        prevStatus.className   = 'status ok';
    } catch (err) {
        prevStatus.textContent = 'Error: ' + err.message;
        prevStatus.className   = 'status err';
        console.error(err);
    } finally {
        prevGenBtn.disabled        = false;
        prevProgress.style.display = 'none';
    }
});

// ── Record tab ────────────────────────────────────────────────────────────────

const recStartBtn   = document.getElementById('rec-start-btn');
const recStopBtn    = document.getElementById('rec-stop-btn');
const recClearBtn   = document.getElementById('rec-clear-btn');
const recTimer      = document.getElementById('rec-timer');
const recPreview    = document.getElementById('rec-preview');
const recAudio      = document.getElementById('rec-audio');
const recSpecWrap   = document.getElementById('rec-spec-wrap');
const recCanvas     = document.getElementById('rec-canvas');
const recAddBtn     = document.getElementById('rec-add-btn');
const recAddStatus  = document.getElementById('rec-add-status');
const recPoolBadge  = document.getElementById('rec-pool-badge');
const recPoolStatus = document.getElementById('rec-pool-status');
const recPoolList   = document.getElementById('rec-pool-list');

let _mediaRecorder  = null;
let _recChunks      = [];
let _recTimerHandle = null;
let _recStartTime   = 0;
let _pendingWav     = null;   // { samples: Float32Array, sampleRate: 16000 } awaiting "Add"

function _recUpdateTimer() {
    const secs = ((Date.now() - _recStartTime) / 1000).toFixed(1);
    recTimer.textContent = secs + ' s';
}

function _recUpdatePool() {
    const n = _recordedWavs.length;
    recPoolBadge.textContent  = String(n);
    recPoolStatus.textContent = n === 0
        ? 'No recordings yet. Recordings are merged with TTS-generated samples at training time.'
        : `${n} recording${n > 1 ? 's' : ''} ready — will be merged with TTS samples at training start.`;
    recPoolStatus.className   = n > 0 ? 'status ok' : 'status';
    if (negCats) trainStartBtn.disabled = false;
}

recStartBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        _recChunks     = [];
        _mediaRecorder = new MediaRecorder(stream);

        _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _recChunks.push(e.data); };

        _mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            clearInterval(_recTimerHandle);
            recTimer.textContent = '';

            try {
                const blob = new Blob(_recChunks, { type: _mediaRecorder.mimeType });
                const arrBuf = await blob.arrayBuffer();
                const audioCtx = new AudioContext();
                const decoded  = await audioCtx.decodeAudioData(arrBuf);
                await audioCtx.close();

                const srcRate = decoded.sampleRate;
                const src     = decoded.getChannelData(0);   // mono; mix stereo if needed
                if (decoded.numberOfChannels > 1) {
                    const ch1 = decoded.getChannelData(1);
                    for (let i = 0; i < src.length; i++) src[i] = (src[i] + ch1[i]) * 0.5;
                }

                // Resample to 16 kHz
                const ratio = srcRate / 16000;
                const n16   = Math.round(src.length / ratio);
                const s16k  = new Float32Array(n16);
                for (let i = 0; i < n16; i++) {
                    const pos = i * ratio;
                    const i0  = Math.floor(pos);
                    const i1  = Math.min(i0 + 1, src.length - 1);
                    s16k[i] = src[i0] * (1 - (pos - i0)) + src[i1] * (pos - i0);
                }

                _pendingWav = { samples: s16k, sampleRate: 16000 };

                recAudio.src = URL.createObjectURL(encodeWav(s16k, 16000));
                recPreview.style.display = '';
                recAddStatus.textContent = '';

                const res = audioToSpectrogram(s16k, 16000);
                if (res) {
                    recSpecWrap.style.display = 'block';
                    drawSpectrogram(recCanvas, res.features, res.T);
                } else {
                    recSpecWrap.style.display = 'none';
                }
            } catch (err) {
                recAddStatus.textContent = 'Decode error: ' + err.message;
                recAddStatus.className   = 'status err';
                console.error(err);
            }
        };

        _mediaRecorder.start(100);   // collect data every 100 ms
        _recStartTime   = Date.now();
        _recTimerHandle = setInterval(_recUpdateTimer, 100);

        recStartBtn.disabled = true;
        recStopBtn.disabled  = false;
        recClearBtn.style.display = 'none';
        recPreview.style.display  = 'none';
    } catch (err) {
        recAddStatus.textContent = 'Microphone error: ' + err.message;
        recAddStatus.className   = 'status err';
    }
});

recStopBtn.addEventListener('click', () => {
    if (_mediaRecorder && _mediaRecorder.state !== 'inactive') _mediaRecorder.stop();
    recStartBtn.disabled      = false;
    recStopBtn.disabled       = true;
    recClearBtn.style.display = '';
});

recClearBtn.addEventListener('click', () => {
    _pendingWav               = null;
    recPreview.style.display  = 'none';
    recClearBtn.style.display = 'none';
    recAddStatus.textContent  = '';
});

recAddBtn.addEventListener('click', () => {
    if (!_pendingWav) return;

    const idx = _recordedWavs.length + 1;
    _recordedWavs.push(_pendingWav);
    _pendingWav = null;

    recAddStatus.textContent = `Added as recording #${idx}.`;
    recAddStatus.className   = 'status ok';
    recPreview.style.display = 'none';
    recClearBtn.style.display = 'none';

    // Add item to pool list
    const dur  = (_recordedWavs[idx - 1].samples.length / 16000).toFixed(2);
    const url  = recAudio.src;
    const item = document.createElement('div');
    item.className = 'rec-item';
    item.dataset.idx = String(idx - 1);

    const miniCanvas = document.createElement('canvas');
    miniCanvas.className = 'spectrogram';

    const rmBtn = document.createElement('button');
    rmBtn.textContent = '✕';
    rmBtn.className   = 'secondary';
    rmBtn.style.cssText = 'padding:0.2rem 0.5rem;font-size:0.8rem';
    rmBtn.addEventListener('click', () => {
        const i = parseInt(item.dataset.idx);
        _recordedWavs.splice(i, 1);
        // Update indices for remaining items
        [...recPoolList.querySelectorAll('.rec-item')].forEach((el, j) => {
            el.dataset.idx = String(j);
        });
        item.remove();
        _recUpdatePool();
    });

    const numSpan = document.createElement('span');
    numSpan.className = 'sample-num';
    numSpan.textContent = String(idx);

    const audio = document.createElement('audio');
    audio.controls = true;
    audio.src = url;
    audio.style.cssText = 'height:28px;flex:1';

    const durSpan = document.createElement('span');
    durSpan.className = 'sample-params';
    durSpan.textContent = `${dur} s`;

    item.appendChild(numSpan);
    item.appendChild(audio);
    item.appendChild(miniCanvas);
    item.appendChild(durSpan);
    item.appendChild(rmBtn);
    recPoolList.appendChild(item);

    // Draw mini spectrogram
    const res = audioToSpectrogram(_recordedWavs[idx - 1].samples, 16000);
    if (res) drawSpectrogram(miniCanvas, res.features, res.T);
    else miniCanvas.style.display = 'none';

    _recUpdatePool();
});
