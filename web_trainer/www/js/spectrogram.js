// spectrogram.js — Canvas spectrogram widget for the wake-word web trainer.
//
// Dependencies: web_trainer.js (WASM) must be loaded and window.webTrainerReady
// must be true before calling audioToSpectrogram().

'use strict';

// ── Viridis colormap ─────────────────────────────────────────────────────────

const _VIRIDIS = (() => {
    const pts = [
        [0.00,  68,  1,  84],
        [0.13,  71, 40, 122],
        [0.25,  59, 82, 139],
        [0.38,  44,113, 142],
        [0.50,  33,145, 140],
        [0.63,  53,183, 121],
        [0.75,  94,201,  98],
        [0.88, 177,220,  25],
        [1.00, 253,231,  37],
    ];
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
        const t = i / 255;
        let j = 0;
        while (j < pts.length - 2 && pts[j + 1][0] <= t) j++;
        const [t0, r0, g0, b0] = pts[j];
        const [t1, r1, g1, b1] = pts[j + 1];
        const f = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
        lut[i * 3]     = Math.round(r0 + (r1 - r0) * f);
        lut[i * 3 + 1] = Math.round(g0 + (g1 - g0) * f);
        lut[i * 3 + 2] = Math.round(b0 + (b1 - b0) * f);
    }
    return lut;
})();

// ── Canvas rendering ─────────────────────────────────────────────────────────

// Draw a spectrogram heatmap on canvas.
// data: Int8Array of T * numFeatures values, row-major [frame][feature].
// T: number of time frames.
// numFeatures: features per frame (default 40).
// Low-frequency features are drawn at the bottom of the canvas.
function drawSpectrogram(canvas, data, T, numFeatures = 40) {
    canvas.width  = Math.max(T, 1);
    canvas.height = numFeatures;

    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(T, numFeatures);

    // Auto-scale to actual value range
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < T * numFeatures; i++) {
        if (data[i] < lo) lo = data[i];
        if (data[i] > hi) hi = data[i];
    }
    const span = hi > lo ? hi - lo : 1;

    for (let t = 0; t < T; t++) {
        for (let f = 0; f < numFeatures; f++) {
            const norm  = Math.max(0, Math.min(255, Math.round((data[t * numFeatures + f] - lo) / span * 255)));
            const yFlip = numFeatures - 1 - f;   // low freq at bottom
            const px    = (yFlip * T + t) * 4;
            img.data[px]     = _VIRIDIS[norm * 3];
            img.data[px + 1] = _VIRIDIS[norm * 3 + 1];
            img.data[px + 2] = _VIRIDIS[norm * 3 + 2];
            img.data[px + 3] = 255;
        }
    }
    ctx.putImageData(img, 0, 0);
}

// ── Audio → spectrogram via WASM MicroFrontend ───────────────────────────────

// Resample a Float32Array from srcRate to 16000 Hz via linear interpolation.
function _resampleTo16k(samples, srcRate) {
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

// Process a Float32Array of audio through the WASM MicroFrontend.
// Returns { features: Int8Array (T * 40), T } or null if WASM is not ready.
// Requires web_trainer.js to be loaded with HEAPU8 in EXPORTED_RUNTIME_METHODS.
function audioToSpectrogram(samples, sampleRate) {
    if (!window.webTrainerReady || !window.Module) return null;
    if (!Module.HEAPU8) {
        console.warn('spectrogram: HEAPU8 not exported — rebuild with HEAPU8 in EXPORTED_RUNTIME_METHODS');
        return null;
    }

    const s16k = _resampleTo16k(samples, sampleRate);

    const int16 = new Int16Array(s16k.length);
    for (let i = 0; i < s16k.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(s16k[i] * 32767)));
    }

    const fe       = Module.ccall('frontend_create', 'number', [], []);
    const maxFrames = Math.ceil(int16.length / 160) + 10;
    const featPtr  = Module._malloc(maxFrames * 40);
    const pcmBytes = new Uint8Array(int16.buffer);

    const T = Module.ccall('frontend_process', 'number',
        ['number', 'array', 'number', 'number'],
        [fe, pcmBytes, int16.length, featPtr]);

    Module.ccall('frontend_destroy', null, ['number'], [fe]);

    // Copy features out of WASM heap before freeing
    const features = new Int8Array(T * 40);
    for (let i = 0; i < T * 40; i++) features[i] = Module.HEAPU8[featPtr + i];

    Module._free(featPtr);
    return { features, T };
}
