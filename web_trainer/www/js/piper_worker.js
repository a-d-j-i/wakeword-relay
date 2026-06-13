'use strict';

// Piper synthesis worker — runs VITS ONNX inference off the main thread.
//
// The main thread handles espeak-ng phonemisation (phoneme IDs are the same
// for every sample of a given phrase) and sends a pre-computed BigInt64Array
// buffer to each synthesise call, avoiding duplicate WASM loading.
//
// Protocol:
//   RECEIVE  { type:'init',      modelUrl, sampleRate, numSpeakers }
//   → POST   { type:'ready' }
//
//   RECEIVE  { type:'synthesise', id, phonemeIdsBuf:ArrayBuffer,
//              noiseScale, lengthScale, noiseW }
//   → POST   { type:'result',    id, wavBuf:ArrayBuffer (transferred),
//              sampleRate, noiseScale, lengthScale, noiseW }
//
//   On any error:
//   → POST   { type:'error',     id? (absent for init), message }

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.js');

let _session     = null;
let _sampleRate  = 22050;
let _numSpeakers = 1;

function _encodeWav(samples, sampleRate) {
    const n   = samples.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v   = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    str(0,  'RIFF');
    v.setUint32( 4, 36 + n * 2,    true);
    str(8,  'WAVE');
    str(12, 'fmt ');
    v.setUint32(16, 16,            true);
    v.setUint16(20, 1,             true);   // PCM
    v.setUint16(22, 1,             true);   // mono
    v.setUint32(24, sampleRate,    true);
    v.setUint32(28, sampleRate * 2,true);
    v.setUint16(32, 2,             true);
    v.setUint16(34, 16,            true);
    str(36, 'data');
    v.setUint32(40, n * 2,         true);
    let off = 44;
    for (let i = 0; i < n; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        off += 2;
    }
    return buf;   // transferable ArrayBuffer
}

self.addEventListener('message', async ({ data }) => {
    if (data.type === 'init') {
        try {
            ort.env.wasm.wasmPaths  = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
            ort.env.wasm.numThreads = 1;   // one thread per worker; pool gives parallelism
            const resp = await fetch(data.modelUrl);
            if (!resp.ok) throw new Error('model fetch HTTP ' + resp.status);
            _session     = await ort.InferenceSession.create(
                await resp.arrayBuffer(), { executionProviders: ['wasm'] });
            _sampleRate  = data.sampleRate;
            _numSpeakers = data.numSpeakers;
            self.postMessage({ type: 'ready' });
        } catch (e) {
            self.postMessage({ type: 'error', message: e.message });
        }
        return;
    }

    if (data.type === 'synthesise') {
        const { id, phonemeIdsBuf, noiseScale, lengthScale, noiseW } = data;
        try {
            const phonemeIds = new BigInt64Array(phonemeIdsBuf);
            const feeds = {
                input:         new ort.Tensor('int64', phonemeIds,
                                              [1, phonemeIds.length]),
                input_lengths: new ort.Tensor('int64',
                                              new BigInt64Array([BigInt(phonemeIds.length)]), [1]),
                scales:        new ort.Tensor('float32',
                                              new Float32Array([noiseScale, lengthScale, noiseW]), [3]),
            };
            if (_numSpeakers > 1) feeds.sid = new ort.Tensor('int64', new BigInt64Array([0n]), [1]);

            const out    = await _session.run(feeds);
            const audio  = out[Object.keys(out)[0]].data;
            const wavBuf = _encodeWav(audio, _sampleRate);

            self.postMessage(
                { type: 'result', id, wavBuf, sampleRate: _sampleRate, noiseScale, lengthScale, noiseW },
                [wavBuf]
            );
        } catch (e) {
            self.postMessage({ type: 'error', id, message: e?.message ?? String(e) });
        }
    }
});
