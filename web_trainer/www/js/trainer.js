'use strict';

// Training loop helpers: WASM wrappers for the MicroFrontend → spectrogram
// conversion and the MixedNet + Adam training step.
//
// Requires web_trainer.js to be loaded and window.webTrainerReady to be true
// before any function is called.
//
// Spectrogram scale: uint8 MicroFrontend output × (1/25.6) → float32.
// This matches the negative dataset bundle scale (negatives.js NEG_SCALE).

const _FEAT_SCALE = 1.0 / 25.6;   // uint8 [0-255] → float32 [0 - ~10]

// Resample Float32Array from srcRate to 16 kHz via linear interpolation.
function _resampleTo16k(samples, srcRate) {
    if (srcRate === 16000) return samples;
    const r   = srcRate / 16000;
    const n   = Math.round(samples.length / r);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const pos = i * r;
        const i0  = Math.floor(pos);
        const i1  = Math.min(i0 + 1, samples.length - 1);
        out[i] = samples[i0] * (1 - (pos - i0)) + samples[i1] * (pos - i0);
    }
    return out;
}

// Process Float32Array audio through the WASM MicroFrontend.
// Resamples to 16 kHz internally. Pads to minFrames (157) if the clip is short.
// Returns { spectrogram: Float32Array (T×40), T } or null if WASM not ready.
function audioToFloat32Spec(samples, sampleRate = 16000) {
    if (!window.webTrainerReady || !window.Module) return null;
    if (!Module.HEAPU8 || !Module.HEAPF32) {
        console.warn('trainer: HEAPU8/HEAPF32 not exported — rebuild WASM');
        return null;
    }

    const s16k  = _resampleTo16k(samples, sampleRate);
    const int16 = new Int16Array(s16k.length);
    for (let i = 0; i < s16k.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(s16k[i] * 32767)));
    }

    const fe        = Module.ccall('frontend_create', 'number', [], []);
    const maxFrames = Math.ceil(int16.length / 160) + 10;
    const featPtr   = Module._malloc(maxFrames * 40);
    const pcmBytes  = new Uint8Array(int16.buffer);

    const T = Module.ccall('frontend_process', 'number',
        ['number', 'array', 'number', 'number'],
        [fe, pcmBytes, int16.length, featPtr]);

    Module.ccall('frontend_destroy', null, ['number'], [fe]);

    const minFrames = Module.ccall('mixednet_min_frames', 'number', [], []);
    const actualT   = Math.max(T, minFrames);

    const spectrogram = new Float32Array(actualT * 40);
    for (let i = 0; i < T * 40; i++) spectrogram[i] = Module.HEAPU8[featPtr + i] * _FEAT_SCALE;
    // Frames beyond T are zero-padded (Float32Array initialises to 0).

    Module._free(featPtr);
    return { spectrogram, T: actualT };
}

// Allocate a new MixedNet and Adam on the WASM heap.
// Returns { net, adam } — both are numeric WASM pointers.
function trainCreate(lr = 1e-3) {
    const net  = Module.ccall('mixednet_create', 'number', [], []);
    const adam = Module.ccall('adam_create', 'number', ['number', 'number'], [net, lr]);
    return { net, adam };
}

// Free WASM objects created by trainCreate.
function trainDestroy(net, adam) {
    if (adam) Module.ccall('adam_destroy',    null, ['number'], [adam]);
    if (net)  Module.ccall('mixednet_destroy', null, ['number'], [net]);
}

// Run one forward+backward+Adam step.
// spectrogram: Float32Array of T×40 values (output of audioToFloat32Spec or negGetSample).
// label: 1.0 (positive) or 0.0 (negative).
// Returns BCE loss (float).
function trainStep(net, adam, spectrogram, T, label) {
    const bytes  = T * 40 * 4;
    const ptr    = Module._malloc(bytes);
    Module.HEAPF32.set(spectrogram.subarray(0, T * 40), ptr >> 2);
    const loss = Module.ccall('train_step', 'number',
        ['number', 'number', 'number', 'number', 'number'],
        [net, adam, ptr, T, label]);
    Module._free(ptr);
    return loss;
}

// Return the current flat float32 weights as a copy.
// 24 801 values for the production MixedNet config.
function trainGetParams(net) {
    const n   = Module.ccall('mixednet_num_params', 'number', ['number'], [net]);
    const ptr = Module.ccall('mixednet_get_params', 'number', ['number'], [net]);
    const out = Module.HEAPF32.slice(ptr >> 2, (ptr >> 2) + n);
    Module._free(ptr);
    return out;
}

window.audioToFloat32Spec = audioToFloat32Spec;
window.trainCreate        = trainCreate;
window.trainDestroy       = trainDestroy;
window.trainStep          = trainStep;
window.trainGetParams     = trainGetParams;
