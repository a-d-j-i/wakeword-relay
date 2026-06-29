'use strict';

// Web Worker: runs the MixedNet training loop off the main thread.
//
// Protocol (main → worker):
//   { type:'train', posWavs, negCats, noiseParsed, irs, numSteps, negPerPos, lr, pooled }
//   { type:'stop' }
//   { type:'export_tflite', templateBytes:Uint8Array, calibN:number }
//
// Protocol (worker → main):
//   { type:'progress', step, numSteps, losses:number[] }  — batch of losses every 10 steps
//   { type:'done', stopped:bool, params:Float32Array }     — params transferred
//   { type:'tflite', bytes:Uint8Array }                   — bytes transferred
//   { type:'tflite_error', message:string }

importScripts('../web_trainer.js', 'augment.js');

const _NEG_SCALE         = 0.0390625;
const _NOISE_INT16_SCALE = 1.0 / 32768.0;
const _FEAT_SCALE        = 1.0 / 25.6;
const _NEG_CATS          = ['speech', 'no_speech', 'dinner_party'];

let _M         = null;   // WASM Module instance
let _net       = null;   // MixedNet WASM pointer (kept alive for TFLite export after training)
let _adam      = null;   // Adam WASM pointer
let _abortFlag = false;

// ── Inline sampling helpers (pure, no DOM dependencies) ──────────────────────

function _resample16k(samples, srcRate) {
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

function _negGet(cats, cat) {
    const c      = cats[cat];
    if (!c) throw new Error(`Unknown category: ${cat}`);
    const stride = c.numFrames * c.numFeatures;
    const idx    = Math.floor(Math.random() * c.numSamples);
    const slice  = c.data.subarray(idx * stride, (idx + 1) * stride);
    const out    = new Float32Array(stride);
    for (let i = 0; i < stride; i++) out[i] = slice[i] * _NEG_SCALE;
    return out;
}

function _noiseGet(parsed) {
    const i   = Math.floor(Math.random() * parsed.numClips);
    const off = i * parsed.clipSamples;
    const src = parsed.data.subarray(off, off + parsed.clipSamples);
    const out = new Float32Array(parsed.clipSamples);
    for (let j = 0; j < parsed.clipSamples; j++) out[j] = src[j] * _NOISE_INT16_SCALE;
    return out;
}

// ── WASM wrappers (adapted from trainer.js, using local _M) ──────────────────

function _toSpec(samples, sampleRate, minFrames) {
    const s16k  = _resample16k(samples, sampleRate);
    const int16 = new Int16Array(s16k.length);
    for (let i = 0; i < s16k.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(s16k[i] * 32767)));
    }
    const fe      = _M.ccall('frontend_create', 'number', [], []);
    const maxFr   = Math.max(Math.ceil(int16.length / 160) + 10, minFrames + 1);
    const featPtr = _M._malloc(maxFr * 40 * 2);
    const pcmPtr  = _M._malloc(int16.byteLength);
    _M.HEAPU8.set(new Uint8Array(int16.buffer), pcmPtr);
    const T = _M.ccall('frontend_process', 'number',
        ['number', 'number', 'number', 'number'],
        [fe, pcmPtr, int16.length, featPtr]);
    _M._free(pcmPtr);
    _M.ccall('frontend_destroy', null, ['number'], [fe]);
    const actualT = Math.max(T, minFrames);
    const spec    = new Float32Array(actualT * 40);
    const base    = featPtr >> 1;
    for (let i = 0; i < T * 40; i++) spec[i] = _M.HEAPU16[base + i] * _FEAT_SCALE;
    _M._free(featPtr);
    return { spectrogram: spec, T: actualT };
}

function _trainStep(net, adam, spec, T, label) {
    const ptr  = _M._malloc(T * 40 * 4);
    _M.HEAPF32.set(spec.subarray(0, T * 40), ptr >> 2);
    const loss = _M.ccall('train_step', 'number',
        ['number', 'number', 'number', 'number', 'number'],
        [net, adam, ptr, T, label]);
    _M._free(ptr);
    return loss;
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (e) => {
    const msg = e.data;

    if (msg.type === 'stop') {
        _abortFlag = true;
        return;
    }

    if (msg.type === 'train') {
        _abortFlag = false;
        const { posWavs, negCats, noiseParsed, irs, numSteps, negPerPos, lr, pooled } = msg;

        // Free any prior net/adam from a previous run on this worker.
        if (_adam) { _M.ccall('adam_destroy',     null, ['number'], [_adam]); _adam = null; }
        if (_net)  { _M.ccall('mixednet_destroy', null, ['number'], [_net]);  _net  = null; }

        if (!_M) _M = await createWebTrainer();

        _net  = _M.ccall('mixednet_create', 'number', ['number'], [pooled]);
        _adam = _M.ccall('adam_create',     'number', ['number', 'number'], [_net, lr]);
        const minFrames = _M.ccall('mixednet_min_frames', 'number', ['number'], [_net]);

        const batch = [];

        for (let step = 0; step < numSteps && !_abortFlag; step++) {
            try {
                const wav  = posWavs[Math.floor(Math.random() * posWavs.length)];
                const s16k = _resample16k(wav.samples, wav.sampleRate);
                if (s16k.length > 80000) continue;

                const aug = augmentSample(s16k, 16000, {
                    irs:        irs || [],
                    noises:     noiseParsed ? [_noiseGet(noiseParsed)] : [],
                    pitchProb:  0.5,
                    eqProb:     0.5,
                    reverbProb: irs && irs.length > 0 ? 0.5 : 0.0,
                    noiseProb:  noiseParsed ? 0.75 : 0.0,
                    gainRange:  [-45, 0],
                });

                const feat = _toSpec(aug, 16000, minFrames);
                specAugment(feat.spectrogram, feat.T);
                const posLoss = _trainStep(_net, _adam, feat.spectrogram, feat.T, 1.0);

                let negSum = 0;
                for (let j = 0; j < negPerPos; j++) {
                    const cat = _NEG_CATS[Math.floor(Math.random() * _NEG_CATS.length)];
                    let spec  = _negGet(negCats, cat);
                    let negT  = 160;
                    if (minFrames > negT) {
                        const padded = new Float32Array(minFrames * 40);
                        padded.set(spec);
                        spec = padded;
                        negT = minFrames;
                    }
                    specAugment(spec, negT);
                    negSum += _trainStep(_net, _adam, spec, negT, 0.0);
                }

                batch.push((posLoss + negSum / negPerPos) / 2);
            } catch (err) {
                console.error(`[train_worker] step ${step} skipped: ${err.message}`);
            }

            if (step % 10 === 0) {
                if (batch.length > 0) {
                    self.postMessage({ type: 'progress', step: step + 1, numSteps, losses: batch.slice() });
                    batch.length = 0;
                }
                // Yield so a queued 'stop' message can be processed.
                await new Promise(r => setTimeout(r, 0));
            }
        }

        // Flush any remaining losses from the final partial batch.
        if (batch.length > 0) {
            self.postMessage({ type: 'progress', step: numSteps, numSteps, losses: batch.slice() });
        }

        // Extract weights and transfer them to the main thread.
        const n      = _M.ccall('mixednet_num_params', 'number', ['number'], [_net]);
        const ptr    = _M.ccall('mixednet_get_params',  'number', ['number'], [_net]);
        const params = _M.HEAPF32.slice(ptr >> 2, (ptr >> 2) + n);
        _M._free(ptr);

        // _net and _adam stay alive so export_tflite can use them.
        self.postMessage({ type: 'done', stopped: _abortFlag, params }, [params.buffer]);
        return;
    }

    if (msg.type === 'export_tflite') {
        if (!_net || !_M) {
            self.postMessage({ type: 'tflite_error', message: 'No trained model available' });
            return;
        }
        try {
            const { templateBytes, calibN } = msg;
            const tmplPtr  = _M._malloc(templateBytes.byteLength);
            _M.HEAPU8.set(templateBytes, tmplPtr);
            const outSzPtr = _M._malloc(4);
            _M.HEAP32[outSzPtr >> 2] = 0;
            const outPtr   = _M.ccall('mixednet_export_tflite', 'number',
                ['number', 'number', 'number', 'number', 'number'],
                [_net, tmplPtr, templateBytes.byteLength, calibN, outSzPtr]);
            _M._free(tmplPtr);
            if (!outPtr) {
                _M._free(outSzPtr);
                throw new Error('mixednet_export_tflite returned null');
            }
            const outSz  = _M.HEAP32[outSzPtr >> 2];
            _M._free(outSzPtr);
            const result = _M.HEAPU8.slice(outPtr, outPtr + outSz);
            _M._free(outPtr);
            self.postMessage({ type: 'tflite', bytes: result }, [result.buffer]);
        } catch (err) {
            self.postMessage({ type: 'tflite_error', message: err.message });
        }
        return;
    }
};
