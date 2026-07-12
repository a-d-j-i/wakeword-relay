'use strict';

// Training loop helpers: WASM wrappers for the MicroFrontend → spectrogram
// conversion and the MixedNet + Adam training step.
//
// Requires web_trainer.js to be loaded and window.webTrainerReady to be true
// before any function is called.
//
// Spectrogram scale: uint16 MicroFrontend output × (1/25.6) → float32.
// This matches the negative dataset bundle scale (negatives.js NEG_SCALE) and
// microWakeWord (inference.py / data.py multiply the uint16 frontend output by
// 0.0390625). The frontend output routinely exceeds 255 for speech (range ~0..670),
// so it must be read as uint16 — reading it as bytes wraps every value > 255.

const _FEAT_SCALE = 1.0 / 25.6;   // uint16 [0-~670] → float32 [0 - ~26]

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
// Resamples to 16 kHz internally. Pads to minFrames if the clip is short.
// minFrames: use trainState.minFrames (157 for pooled, 204 for full-window).
// Returns { spectrogram: Float32Array (T×40), T } or null if WASM not ready.
function audioToFloat32Spec(samples, sampleRate = 16000, minFrames = 157) {
    if (!window.webTrainerReady || !window.Module) return null;
    if (!Module.HEAPU16 || !Module.HEAPF32) {
        console.warn('trainer: HEAPU16/HEAPF32 not exported — rebuild WASM');
        return null;
    }

    const s16k  = _resampleTo16k(samples, sampleRate);
    const int16 = new Int16Array(s16k.length);
    for (let i = 0; i < s16k.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(s16k[i] * 32767)));
    }

    const fe        = Module.ccall('frontend_create', 'number', [], []);
    const maxFrames = Math.max(Math.ceil(int16.length / 160) + 10, minFrames + 1);
    const featPtr = Module._malloc(maxFrames * 40 * 2);  // uint16: 2 bytes per feature

    // Heap-allocate PCM — ccall 'array' type uses stackAlloc, which overflows for large audio.
    const pcmPtr = Module._malloc(int16.byteLength);
    Module.HEAPU8.set(new Uint8Array(int16.buffer), pcmPtr);
    const T = Module.ccall('frontend_process', 'number',
        ['number', 'number', 'number', 'number'],
        [fe, pcmPtr, int16.length, featPtr]);
    Module._free(pcmPtr);

    Module.ccall('frontend_destroy', null, ['number'], [fe]);

    window._dbgT = T;  // expose for error logging in training loop
    const actualT     = Math.max(T, minFrames);
    const spectrogram = new Float32Array(actualT * 40);
    const base = featPtr >> 1;  // HEAPU16 is indexed in 16-bit words
    for (let i = 0; i < T * 40; i++) spectrogram[i] = Module.HEAPU16[base + i] * _FEAT_SCALE;
    // Frames beyond T are zero-padded (Float32Array initialises to 0).

    Module._free(featPtr);
    return { spectrogram, T: actualT };
}

// Allocate a new MixedNet and Adam on the WASM heap.
// pooled: 1 = global-avg-pool + Dense(64→1) [24801 params, minFrames=157]
//         0 = flatten(17×64) + Dense(1088→1) [25825 params, minFrames=204]
// Returns { net, adam, minFrames, numParams }.
function trainCreate(lr = 1e-3, pooled = 0) {
    const net       = Module.ccall('mixednet_create', 'number', ['number'], [pooled]);
    const adam      = Module.ccall('adam_create',    'number', ['number', 'number'], [net, lr]);
    const minFrames = Module.ccall('mixednet_min_frames', 'number', ['number'], [net]);
    const numParams = Module.ccall('mixednet_num_params', 'number', ['number'], [net]);
    return { net, adam, minFrames, numParams };
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

// Export trained MixedNet weights as a streaming INT8 TFLite file.
// templateBytes: Uint8Array of a pooled=0 TFLite template (topology only, weights replaced)
// calibN: number of synthetic calibration samples (500 recommended)
// Returns Uint8Array of the output TFLite, or null on failure.
function trainExportTFLite(net, templateBytes, calibN = 500) {
    if (!net || !templateBytes) return null;

    // Copy template into WASM heap
    const tmplPtr = Module._malloc(templateBytes.byteLength);
    Module.HEAPU8.set(templateBytes, tmplPtr);

    // Allocate output size slot (int32)
    const outSzPtr = Module._malloc(4);
    Module.HEAP32[outSzPtr >> 2] = 0;

    const outPtr = Module.ccall('mixednet_export_tflite', 'number',
        ['number', 'number', 'number', 'number', 'number'],
        [net, tmplPtr, templateBytes.byteLength, calibN, outSzPtr]);

    Module._free(tmplPtr);

    if (!outPtr) { Module._free(outSzPtr); return null; }

    const outSz = Module.HEAP32[outSzPtr >> 2];
    Module._free(outSzPtr);

    const result = Module.HEAPU8.slice(outPtr, outPtr + outSz);
    Module._free(outPtr);
    return result;
}

window.audioToFloat32Spec = audioToFloat32Spec;
window.trainCreate        = trainCreate;
window.trainDestroy       = trainDestroy;
window.trainStep          = trainStep;
window.trainGetParams     = trainGetParams;
window.trainExportTFLite  = trainExportTFLite;
