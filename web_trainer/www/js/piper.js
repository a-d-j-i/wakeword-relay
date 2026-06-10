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
    const ptr = _phonemizeModule.ccall('phonemize', 'number', ['string', 'string'], [text, lang]);
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

    const tokens = ipa.split(' ').filter(t => t.length > 0);
    const ids = [bos];

    for (const tok of tokens) {
        const mapped = phonemeIdMap[tok];
        if (mapped) {
            for (const id of mapped) ids.push(id);
        }
        // unknown phonemes are silently skipped
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

    const ipa        = textToIpa(text, lang);
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

// Generate `count` diverse WAV samples for `text`.
// Returns an array of { wav: Blob, sampleRate, ipa, noiseScale, lengthScale, noiseW }.
async function generateSamples(session, voiceConfig, text, lang, count, onProgress) {
    const results = [];
    for (let i = 0; i < count; i++) {
        // Vary noise_scale and noise_w for diversity; keep length_scale near 1.
        const noiseScale  = 0.3  + Math.random() * 0.7;   // [0.3, 1.0]
        const lengthScale = 0.85 + Math.random() * 0.3;   // [0.85, 1.15]
        const noiseW      = 0.2  + Math.random() * 1.3;   // [0.2, 1.5]

        const result = await synthesise(session, voiceConfig, text, lang,
                                        noiseScale, lengthScale, noiseW);
        result.noiseScale  = noiseScale;
        result.lengthScale = lengthScale;
        result.noiseW      = noiseW;
        results.push(result);

        if (onProgress) onProgress(i + 1, count);
    }
    return results;
}
