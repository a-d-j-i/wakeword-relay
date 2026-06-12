// piper.js — Piper TTS synthesis pipeline for the wake-word web trainer.
//
// Dependencies (must be loaded before this file):
//   piper_phonemize.js  — our espeak-ng WASM module (built from vendor/espeak-ng)
//   onnxruntime-web     — Microsoft ONNX Runtime Web (runs the Piper VITS model)

'use strict';

// ── Phonemizer ───────────────────────────────────────────────────────────────

let _phonemizeModule = null;

async function initPhonemizer() {
    _phonemizeModule = await createPiperPhonemize();
    const rc = _phonemizeModule.ccall('phonemize_init', 'number', [], []);
    if (rc !== 0) throw new Error('espeak-ng init failed (rc=' + rc + ')');
}

// Returns a space-separated IPA string for the given text and espeak language code.
// e.g. textToIpa('hey lumus', 'en') → 'h eɪ l uː m ə s'
function textToIpa(text, lang) {
    if (!_phonemizeModule) throw new Error('Phonemizer not initialised');
    let ptr = _phonemizeModule.ccall('phonemize', 'number', ['string', 'string'], [text, lang]);
    if (ptr === 0 && lang.includes('-')) {
        const base = lang.split('-')[0];
        ptr = _phonemizeModule.ccall('phonemize', 'number', ['string', 'string'], [text, base]);
    }
    if (ptr === 0) throw new Error('phonemize() returned null for "' + text + '"');
    const ipa = _phonemizeModule.UTF8ToString(ptr);
    _phonemizeModule.ccall('phonemize_free', null, ['number'], [ptr]);
    return ipa;
}

// ── Phoneme → ID mapping ─────────────────────────────────────────────────────

// Maps space-separated IPA string to a BigInt64Array of phoneme IDs using
// the phoneme_id_map from the Piper voice config JSON.
function ipaToIds(ipa, phonemeIdMap) {
    const bos = phonemeIdMap['^']?.[0] ?? 1;
    const eos = phonemeIdMap['$']?.[0] ?? 2;
    const pad = phonemeIdMap['_']?.[0] ?? 0;

    const tokens = ipa.split(' ').filter(t => t.length > 0);
    const ids = [bos, pad];  // BOS + PAD

    for (const tok of tokens) {
        const mapped = phonemeIdMap[tok];
        if (mapped) {
            for (const id of mapped) ids.push(id);
            ids.push(pad);
        } else {
            // Multi-char tokens like "tʃ" or "ˈi": each codepoint is its own phoneme
            for (const ch of tok) {
                const chMapped = phonemeIdMap[ch];
                if (chMapped) { for (const id of chMapped) ids.push(id); ids.push(pad); }
            }
        }
    }

    ids.push(eos);
    return new BigInt64Array(ids.map(BigInt));
}

// ── WAV encoding ─────────────────────────────────────────────────────────────

// Converts a Float32Array (audio samples in [-1, 1]) to a WAV Blob.
function encodeWav(samples, sampleRate) {
    const numSamples = samples.length;
    const buf = new ArrayBuffer(44 + numSamples * 2);
    const v = new DataView(buf);
    const write = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };

    write(0,  'RIFF');
    v.setUint32( 4, 36 + numSamples * 2, true);
    write(8,  'WAVE');
    write(12, 'fmt ');
    v.setUint32(16, 16,            true);   // PCM fmt chunk
    v.setUint16(20, 1,             true);   // PCM
    v.setUint16(22, 1,             true);   // mono
    v.setUint32(24, sampleRate,    true);
    v.setUint32(28, sampleRate * 2,true);   // byte rate
    v.setUint16(32, 2,             true);   // block align
    v.setUint16(34, 16,            true);   // bits per sample
    write(36, 'data');
    v.setUint32(40, numSamples * 2, true);

    let off = 44;
    for (let i = 0; i < numSamples; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        off += 2;
    }
    return new Blob([buf], { type: 'audio/wav' });
}

// ── VITS model inference ─────────────────────────────────────────────────────

// voiceConfig: parsed JSON from the .onnx.json file
// text: phrase to synthesise
// lang: espeak language code ('en', 'es', …)
// noiseScale, lengthScale, noiseW: Piper synthesis parameters
//   noiseScale  — phoneme variability (default 0.667)
//   lengthScale — speaking speed  (default 1.0, higher = slower)
//   noiseW      — prosody/pitch variability (default 0.8)
async function synthesise(session, voiceConfig, text, lang, noiseScale, lengthScale, noiseW) {
    const sampleRate   = voiceConfig.audio.sample_rate;
    const phonemeIdMap = voiceConfig.phoneme_id_map;

    const espeakVoice = voiceConfig.espeak?.voice ?? lang;
    const ipa        = textToIpa(text, espeakVoice);
    const phonemeIds = ipaToIds(ipa, phonemeIdMap);

    const feeds = {
        input: new ort.Tensor('int64',   phonemeIds,                                [1, phonemeIds.length]),
        input_lengths: new ort.Tensor('int64', new BigInt64Array([BigInt(phonemeIds.length)]), [1]),
        scales: new ort.Tensor('float32', new Float32Array([noiseScale, lengthScale, noiseW]), [3]),
    };

    // Handle optional speaker-id (multi-speaker models)
    if (voiceConfig.num_speakers > 1) {
        feeds.sid = new ort.Tensor('int64', new BigInt64Array([0n]), [1]);
    }

    const results   = await session.run(feeds);
    const outputKey = Object.keys(results)[0];
    const audio     = results[outputKey].data;   // Float32Array

    return { wav: encodeWav(audio, sampleRate), sampleRate, ipa };
}

// ── Public API ───────────────────────────────────────────────────────────────

// Generate `count` diverse WAV samples for `text` using a worker pool.
// modelUrl: Blob URL (or any fetchable URL) of the .onnx model file.
// Returns an array of { wav: Blob, sampleRate, ipa, noiseScale, lengthScale, noiseW }
// in sample-index order.
async function generateSamples(modelUrl, voiceConfig, text, lang, count, onProgress) {
    const sampleRate  = voiceConfig.audio.sample_rate;
    const numSpeakers = voiceConfig.num_speakers ?? 1;
    const espeakVoice = voiceConfig.espeak?.voice ?? lang;

    // Compute phoneme IDs once on the main thread (espeak-ng is already loaded here).
    const ipa        = textToIpa(text, espeakVoice);
    const phonemeIds = ipaToIds(ipa, voiceConfig.phoneme_id_map);

    // Pool size: up to 4 workers (keeps memory reasonable; each loads a full ORT session).
    const poolSize = Math.min(count, navigator.hardwareConcurrency || 4, 4);
    const workers  = Array.from({ length: poolSize }, () => new Worker('js/piper_worker.js'));

    // Init all workers in parallel; each fetches the model from the Blob URL independently.
    await Promise.all(workers.map(w => new Promise((resolve, reject) => {
        const h = ({ data }) => {
            if (data.type === 'ready') { w.removeEventListener('message', h); resolve(); }
            if (data.type === 'error') { w.removeEventListener('message', h); reject(new Error(data.message)); }
        };
        w.addEventListener('message', h);
        w.postMessage({ type: 'init', modelUrl, sampleRate, numSpeakers });
    })));

    // Map + reduce: dispatch items to idle workers, collect results in order.
    const results = new Array(count);
    let nextId = 0, doneCount = 0;

    return new Promise((resolve, reject) => {
        function dispatch(worker) {
            if (nextId >= count) return;
            const id          = nextId++;
            const noiseScale  = 0.3  + Math.random() * 0.7;   // [0.3, 1.0]
            const lengthScale = 0.85 + Math.random() * 0.3;   // [0.85, 1.15]
            const noiseW      = 0.2  + Math.random() * 1.3;   // [0.2, 1.5]
            const idsBuf      = phonemeIds.buffer.slice(0);    // copy for transfer

            const h = ({ data }) => {
                if (data.id !== id) return;
                worker.removeEventListener('message', h);
                if (data.type === 'error') {
                    workers.forEach(w => w.terminate());
                    reject(new Error('sample ' + id + ': ' + data.message));
                    return;
                }
                results[id] = {
                    wav:         new Blob([data.wavBuf], { type: 'audio/wav' }),
                    sampleRate:  data.sampleRate,
                    ipa,
                    noiseScale:  data.noiseScale,
                    lengthScale: data.lengthScale,
                    noiseW:      data.noiseW,
                };
                doneCount++;
                if (onProgress) onProgress(doneCount, count);
                if (doneCount === count) { workers.forEach(w => w.terminate()); resolve(results); }
                else dispatch(worker);
            };
            worker.addEventListener('message', h);
            worker.postMessage(
                { type: 'synthesise', id, phonemeIdsBuf: idsBuf, noiseScale, lengthScale, noiseW },
                [idsBuf]
            );
        }
        workers.forEach(w => dispatch(w));
    });
}
