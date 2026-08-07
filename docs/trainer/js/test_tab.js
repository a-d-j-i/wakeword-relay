'use strict';

// ── Test tab: TFLite inference (WAV + live mic) ────────────────────────────────
// Mirrors docs/index.html functionality; shares window.Module (WASM frontend).

tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/');
tf.setBackend('cpu');

// ── Constants ─────────────────────────────────────────────────────────────────
const _SR    = 16000;
const _STEP  = 160;   // 10 ms at 16 kHz
const _NMELS = 40;
const _COOLDOWN_FRAMES = 200;   // 2 s between detections

// ── Globals ───────────────────────────────────────────────────────────────────
let _testModel        = null;
let _testOutScale     = 1.0 / 256.0;
let _testModelQScale  = 256 / (25.6 * 26);
let _testModelQZp     = -128;
let _testQuantScale   = 256 / (25.6 * 26);
let _testQuantZp      = -128;
let _testEsp32Mode    = false;
let _testInferBuf     = null;
let _testInferFill    = 0;
let _testInferLast    = 0;
let _testInferCount   = 0;
let _micRunning       = false;
let _micCtx           = null;
let _micFE            = null;
let _micFeatPtr       = null;
let _micLeftover      = new Float32Array(0);
let _micScoreBuf      = new Array(5).fill(0);
let _micCooldown      = 0;

// ── Viridis colormap (for rolling live spectrogram) ───────────────────────────
const _VIR = (() => {
    const pts = [
        [0.00,  68,  1,  84],[0.13,  71, 40,122],[0.25,  59, 82,139],
        [0.38,  44,113,142],[0.50,  33,145,140],[0.63,  53,183,121],
        [0.75,  94,201, 98],[0.88, 177,220, 25],[1.00, 253,231, 37],
    ];
    const lut = new Uint8Array(256*3);
    for (let i = 0; i < 256; i++) {
        const t = i/255; let j=0;
        while (j < pts.length-2 && pts[j+1][0] <= t) j++;
        const [t0,r0,g0,b0]=pts[j],[t1,r1,g1,b1]=pts[j+1];
        const f = t1>t0?(t-t0)/(t1-t0):0;
        lut[i*3]=Math.round(r0+(r1-r0)*f); lut[i*3+1]=Math.round(g0+(g1-g0)*f); lut[i*3+2]=Math.round(b0+(b1-b0)*f);
    }
    return lut;
})();

// ── Rolling spectrogram ───────────────────────────────────────────────────────
const _SPEC_COLS = 600;
let _micSpecCtx  = null;

function _initMicSpec() {
    const canvas = document.getElementById('test-mic-spec');
    canvas.width  = _SPEC_COLS;
    canvas.height = _NMELS;
    _micSpecCtx = canvas.getContext('2d');
    _micSpecCtx.fillStyle = '#0d0d0d';
    _micSpecCtx.fillRect(0, 0, _SPEC_COLS, _NMELS);
}

function _appendSpecFrame(rawFrame) {
    if (!_micSpecCtx) return;
    _micSpecCtx.drawImage(_micSpecCtx.canvas, -1, 0);
    const col = _micSpecCtx.createImageData(1, _NMELS);
    let lo = rawFrame[0], hi = rawFrame[0];
    for (let f = 1; f < _NMELS; f++) {
        if (rawFrame[f] < lo) lo = rawFrame[f];
        if (rawFrame[f] > hi) hi = rawFrame[f];
    }
    const span = hi > lo ? hi - lo : 1;
    for (let f = 0; f < _NMELS; f++) {
        const norm  = Math.round((rawFrame[f] - lo) / span * 255);
        const yFlip = _NMELS - 1 - f;
        const px    = yFlip * 4;
        col.data[px]=_VIR[norm*3]; col.data[px+1]=_VIR[norm*3+1]; col.data[px+2]=_VIR[norm*3+2]; col.data[px+3]=255;
    }
    _micSpecCtx.putImageData(col, _SPEC_COLS - 1, 0);
}

// ── FlatBuffer helpers ────────────────────────────────────────────────────────
function _makeFbHelpers(ab) {
    const dv  = new DataView(ab);
    const u32 = p => dv.getUint32(p, true);
    const i32 = p => dv.getInt32(p, true);
    const f32 = p => dv.getFloat32(p, true);
    const u16 = p => dv.getUint16(p, true);
    const u8  = p => dv.getUint8(p);
    function fld(tableBase, fieldIdx) {
        const vt   = tableBase - i32(tableBase);
        const vtSz = u16(vt);
        const fo   = 4 + fieldIdx * 2;
        if (fo >= vtSz) return 0;
        const off = u16(vt + fo);
        return off ? tableBase + off : 0;
    }
    const deref = pos => pos + u32(pos);
    function vec(pos) { const v = deref(pos); return { len: u32(v), base: v + 4 }; }
    return { u32, i32, f32, u16, u8, fld, deref, vec };
}

// Parse input tensor quantization from TFLite FlatBuffer. Returns {scale,zp} or null.
function _parseTFLiteInputQuant(ab) {
    try {
        const { u32, i32, f32, fld, deref, vec } = _makeFbHelpers(ab);
        function readTensorQuant(tens) {
            for (const fi of [5, 4]) {
                const qFld = fld(tens, fi); if (!qFld) continue;
                const q    = deref(qFld);
                const sFld = fld(q, 2);     if (!sFld) continue;
                const sv   = vec(sFld);     if (sv.len === 0) continue;
                const scale = f32(sv.base);
                if (!(scale > 0.001 && scale < 0.5)) continue;
                let zp = -128;
                const zFld = fld(q, 3);
                if (zFld) { const zv = vec(zFld); if (zv.len > 0) zp = i32(zv.base); }
                return { scale, zp };
            }
            return null;
        }
        function tensorLastDim(tens) {
            const sf = fld(tens, 0); if (!sf) return -1;
            const sv = vec(sf);      if (sv.len === 0) return -1;
            return i32(sv.base + (sv.len - 1) * 4);
        }
        const modelRoot = u32(0);
        const sgFld = fld(modelRoot, 2); if (!sgFld) return null;
        const sg    = vec(sgFld);        if (sg.len === 0) return null;
        const sg0   = deref(sg.base);
        const inFld = fld(sg0, 1); if (!inFld) return null;
        const inV   = vec(inFld);  if (inV.len === 0) return null;
        const tFld  = fld(sg0, 0); if (!tFld) return null;
        const tVec  = vec(tFld);
        for (let i = 0; i < inV.len; i++) {
            const idx = i32(inV.base + i * 4);
            if (idx < 0 || idx >= tVec.len) continue;
            const tens = deref(tVec.base + idx * 4);
            if (tensorLastDim(tens) !== _NMELS) continue;
            const q = readTensorQuant(tens);
            if (q) return q;
        }
        for (let i = 0; i < inV.len; i++) {
            const idx = i32(inV.base + i * 4);
            if (idx < 0 || idx >= tVec.len) continue;
            const tens = deref(tVec.base + idx * 4);
            const q = readTensorQuant(tens);
            if (q) return q;
        }
        return null;
    } catch(e) { console.warn('[parseTFLiteInputQuant]', e); return null; }
}

// Parse output tensor type. Returns e.g. "INT8", "UINT8", or null.
function _parseTFLiteOutputType(ab) {
    const TTYPE = {0:'FLOAT32',1:'FLOAT16',2:'INT32',3:'UINT8',4:'INT64',9:'INT8'};
    try {
        const { u32, i32, u8, fld, deref, vec } = _makeFbHelpers(ab);
        const modelRoot = u32(0);
        const sgFld = fld(modelRoot, 2); if (!sgFld) return null;
        const sg    = vec(sgFld);        if (sg.len === 0) return null;
        const sg0   = deref(sg.base);
        const outFld = fld(sg0, 2); if (!outFld) return null;
        const outV   = vec(outFld); if (outV.len === 0) return null;
        const outIdx = i32(outV.base);
        const tFld   = fld(sg0, 0); if (!tFld) return null;
        const tVec   = vec(tFld);   if (outIdx >= tVec.len) return null;
        const tens   = deref(tVec.base + outIdx * 4);
        const typFld = fld(tens, 1); if (!typFld) return null;
        return TTYPE[u8(typFld)] ?? `type=${u8(typFld)}`;
    } catch(e) { return null; }
}

// Parse the OUTPUT tensor's quantization (scale, zp). Returns { scale, zp } or null.
function _parseTFLiteOutputQuant(ab) {
    try {
        const { u32, i32, f32, fld, deref, vec } = _makeFbHelpers(ab);
        const modelRoot = u32(0);
        const sgFld = fld(modelRoot, 2); if (!sgFld) return null;
        const sg    = vec(sgFld);        if (sg.len === 0) return null;
        const sg0   = deref(sg.base);
        const outFld = fld(sg0, 2); if (!outFld) return null;
        const outV   = vec(outFld); if (outV.len === 0) return null;
        const outIdx = i32(outV.base);
        const tFld   = fld(sg0, 0); if (!tFld) return null;
        const tVec   = vec(tFld);   if (outIdx >= tVec.len) return null;
        const tens   = deref(tVec.base + outIdx * 4);
        for (const fi of [5, 4]) {
            const qFld = fld(tens, fi); if (!qFld) continue;
            const q    = deref(qFld);
            const sFld = fld(q, 2);     if (!sFld) continue;
            const sv   = vec(sFld);     if (sv.len === 0) continue;
            const scale = f32(sv.base);
            if (!(scale > 0 && scale < 1)) continue;
            let zp = 0;
            const zFld = fld(q, 3);
            if (zFld) { const zv = vec(zFld); if (zv.len > 0) zp = i32(zv.base); }
            return { scale, zp };
        }
        return null;
    } catch(e) { return null; }
}

// Feed ~60 frames of silence, return the model's stabilized raw output. Detects
// whether tf-tflite dequantizes the output (≈0 on silence) or returns the raw
// integer (≈1–2 for uint8). Also flushes streaming state to a clean silence init.
function _probeSilenceOutput() {
    if (!_testModel) return 0;
    const winT = _testModel.inputs[0].shape[1];
    const sil  = new Int32Array(winT * _NMELS).fill(_testQuantZp);
    let v = 0;
    for (let i = 0; i < 60; i++) {
        const inp = tf.tensor(sil, [1, winT, _NMELS], 'int32');
        try { const o = _testModel.predict(inp); v = o.dataSync()[0]; o.dispose(); } catch(e) {}
        inp.dispose();
    }
    return v;
}

// ── Quantization ──────────────────────────────────────────────────────────────
function _applyQuantMode() {
    const ESP32_QS = 256 / 666;
    if (_testEsp32Mode) { _testQuantScale = ESP32_QS; _testQuantZp = -128; }
    else                { _testQuantScale = _testModelQScale; _testQuantZp = _testModelQZp; }
    const el = document.getElementById('test-qinfo');
    if (el) el.textContent = `qs=${_testQuantScale.toFixed(4)} zp=${_testQuantZp}${_testEsp32Mode ? '  (256/666)' : ''}`;
}

function _quantizeFrames(raw, T) {
    const q = new Int32Array(T * _NMELS);
    for (let i = 0; i < T * _NMELS; i++)
        q[i] = Math.max(-128, Math.min(127, Math.round(raw[i] * _testQuantScale) + _testQuantZp));
    return q;
}

// ── Streaming inference ───────────────────────────────────────────────────────
function _resetInferBuf() {
    if (!_testModel) return;
    const T = _testModel.inputs[0].shape[1];
    _testInferBuf  = new Int32Array(T * _NMELS);
    _testInferFill = 0;
}

// Feed one NMELS-length Int32Array frame. Returns a score [0,1].
function _inferOneFrame(q) {
    const T = _testModel.inputs[0].shape[1];
    let buf;
    if (T === 1) {
        buf = new Int32Array(_NMELS);
        buf.set(q.subarray(0, _NMELS));
    } else {
        if (!_testInferBuf || _testInferBuf.length !== T * _NMELS) _resetInferBuf();
        _testInferBuf.set(q.subarray(0, _NMELS), _testInferFill * _NMELS);
        _testInferFill++;
        if (_testInferFill < T) return _testInferLast;
        _testInferFill = 0;
        buf = _testInferBuf;
    }
    const inp = tf.tensor(buf, [1, T, _NMELS], 'int32');
    let s = 0;
    try {
        const out = _testModel.predict(inp);
        const raw = out.dataSync()[0];
        _testInferCount++;
        if (_testInferCount === 1)
            console.log('[test] model in:', _testModel.inputs[0].shape, _testModel.inputs[0].dtype,
                        'out:', _testModel.outputs[0].shape);
        s = raw * _testOutScale;
        out.dispose();
    } catch(e) { console.error('[test] predict error:', e); }
    inp.dispose();
    _testInferLast = s;
    return s;
}

// Run inference over a full WAV (returns {scores, peakRaw}).
function _runInference(raw, T) {
    const frames = _quantizeFrames(raw, T);
    const winT   = _testModel.inputs[0].shape[1];
    if (winT > 1) {
        const silBuf = new Int32Array(winT * _NMELS).fill(_testQuantZp);
        for (let w = 0; w < 80; w++) {
            const inp = tf.tensor(silBuf, [1, winT, _NMELS], 'int32');
            try { const o = _testModel.predict(inp); o.dispose(); } catch(e) {}
            inp.dispose();
        }
    }
    const scores = [];
    let peakRaw  = null;
    if (winT === 1) {
        for (let i = 0; i < T; i++)
            scores.push(_inferOneFrame(frames.subarray(i * _NMELS, (i+1) * _NMELS)));
    } else {
        for (let i = 0; i + winT <= T; i += winT) {
            const chunk = frames.subarray(i * _NMELS, (i + winT) * _NMELS);
            const inp   = tf.tensor(chunk, [1, winT, _NMELS], 'int32');
            let s = 0;
            try {
                const out = _testModel.predict(inp);
                const r   = out.dataSync()[0];
                if (peakRaw === null || r > peakRaw) peakRaw = r;
                s = r * _testOutScale;
                out.dispose();
            } catch(e) { console.warn('[test wav] predict error chunk@frame', i, e); }
            inp.dispose();
            for (let k = 0; k < winT; k++) scores.push(s);
        }
    }
    return { scores, peakRaw };
}

// ── Score timeline HTML ───────────────────────────────────────────────────────
function _timelineHtml(scores, thr) {
    const NCOLS = 4, W = 14;
    const step   = Math.max(1, Math.floor(scores.length / (NCOLS * 40)));
    const thrPos = Math.round(thr * W);
    const rows = [];
    for (let i = 0; i < scores.length; i += step) {
        const t      = (i * _STEP / _SR).toFixed(2);
        const v      = scores[i];
        const filled = Math.round(v * W);
        let bar = '';
        for (let j = 0; j < W; j++) {
            if (j === thrPos)    bar += '┊';
            else if (j < filled) bar += '█';
            else                 bar += '░';
        }
        const color = v >= thr ? '#fa3' : '#666';
        rows.push(`<span style="color:${color}">${t.padStart(5)}s ${v.toFixed(3)} [${bar}]</span>`);
    }
    const c    = Math.ceil(rows.length / NCOLS);
    const cols = Array.from({length: NCOLS}, (_, i) => rows.slice(i*c, (i+1)*c));
    return `<div style="color:#555;margin-bottom:4px">score over time  ┊ = threshold (${thr.toFixed(2)})</div>` +
           `<div style="display:flex;gap:12px;white-space:pre;font-size:0.8em">` +
           cols.map(col => `<div style="flex:1">${col.join('\n')}</div>`).join('') +
           `</div>`;
}

// ── Model loading ─────────────────────────────────────────────────────────────
async function _loadTestModelBytes(name, ab) {
    const statusEl = document.getElementById('test-model-status');
    const infoEl   = document.getElementById('test-model-info');
    statusEl.textContent = 'Loading…';
    statusEl.className   = 'status';
    try {
        _testModel = await tflite.loadTFLiteModel(ab);
        const iq   = _parseTFLiteInputQuant(ab);
        const oty  = _parseTFLiteOutputType(ab);
        if (iq) { _testModelQScale = 1.0 / (25.6 * iq.scale); _testModelQZp = iq.zp; }
        else     { _testModelQScale = 1.0; _testModelQZp = -128; }
        _applyQuantMode();
        // tf-tflite's alpha build returns uint8 outputs as the RAW integer (not a
        // dequantized [0,1] float, despite older assumptions) — probe with silence
        // to pick the right scale automatically. (Matches docs/emulator/index.html.)
        const oq = _parseTFLiteOutputQuant(ab);
        const outQuantScale = (oq && oq.scale) ? oq.scale : 1.0 / 256.0;
        if (oty === 'FLOAT32') {
            _testOutScale = 1.0;
        } else {
            const probe = _probeSilenceOutput();   // also silence-primes streaming state
            _testOutScale = (probe > 0.5) ? outQuantScale : 1.0;
            console.log(`[test] output probe (silence) = ${probe.toPrecision(4)} → ` +
                        `${probe > 0.5 ? 'RAW int (scale by ' + outQuantScale.toExponential(2) + ')' : 'already [0,1]'}`);
        }
        _testInferBuf  = null;
        _testInferCount = 0;
        _testInferLast  = 0;

        statusEl.textContent = `Loaded: ${name}`;
        statusEl.className   = 'status ok';
        infoEl.textContent   =
            _testModel.inputs.map(t  => `in:  ${t.name} ${JSON.stringify(t.shape)} ${t.dtype}`).join('\n') + '\n' +
            _testModel.outputs.map(t => `out: ${t.name} ${JSON.stringify(t.shape)} ${t.dtype}`).join('\n') +
            (oty ? `  [${oty}]` : '') + '\n' +
            (iq ? `quant: scale=${iq.scale.toExponential(3)} zp=${iq.zp} → qs=${_testQuantScale.toFixed(4)}`
                : `quant: not found — using qs=1.0 zp=-128`);

        document.getElementById('test-mic-start').disabled = false;
        document.getElementById('test-wav-run').disabled   =
            !document.getElementById('test-wav-file').files.length;
    } catch(err) {
        statusEl.textContent = 'ERROR: ' + err.message;
        statusEl.className   = 'status err';
        console.error('[test] model load error:', err);
    }
}

// ── UI wiring ─────────────────────────────────────────────────────────────────
document.getElementById('test-model-file').addEventListener('change', async e => {
    const f = e.target.files[0];
    if (!f) return;
    const ab = await f.arrayBuffer();
    try { await tfliteCacheSave(ab.slice(0)); } catch(err) { console.warn('tflite cache save:', err); }
    await _loadTestModelBytes(f.name, ab);
});

document.getElementById('test-esp32-mode').addEventListener('change', e => {
    _testEsp32Mode = e.target.checked;
    _applyQuantMode();
});

const _thrSlider = document.getElementById('test-thr');
_thrSlider.addEventListener('input', () => {
    document.getElementById('test-thr-val').textContent = parseFloat(_thrSlider.value).toFixed(2);
});

document.getElementById('test-wav-file').addEventListener('change', e => {
    document.getElementById('test-wav-run').disabled = !(_testModel && e.target.files.length);
});

document.getElementById('test-wav-run').addEventListener('click', async () => {
    if (!_testModel || !window.Module) return;
    const f = document.getElementById('test-wav-file').files[0];
    if (!f) return;

    const outEl     = document.getElementById('test-wav-out');
    const specWrap  = document.getElementById('test-wav-spec-wrap');
    const specCanvas = document.getElementById('test-wav-spec');
    outEl.textContent = 'Processing…';
    specWrap.style.display = 'none';

    try {
        // Decode and resample to 16 kHz via AudioContext (handles any source format/rate).
        const ctx     = new AudioContext();
        const decoded = await ctx.decodeAudioData(await f.arrayBuffer());
        const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * _SR), _SR);
        const src = off.createBufferSource();
        src.buffer = decoded; src.connect(off.destination); src.start();
        const audio = (await off.startRendering()).getChannelData(0);
        ctx.close();

        // Pass sampleRate=_SR because audio is already at 16 kHz after offline rendering.
        const spec = audioToSpectrogram(audio, _SR);
        if (!spec) throw new Error('WASM not ready — wait and try again');
        const { features: raw, T } = spec;

        drawSpectrogram(specCanvas, raw, T);
        specWrap.style.display = '';

        const thr = parseFloat(_thrSlider.value);
        const { scores, peakRaw } = _runInference(raw, T);

        const maxScore = scores.length ? Math.max(...scores) : 0;
        const maxIdx   = scores.indexOf(maxScore);
        const det      = maxScore >= thr;
        const hdrColor = det ? '#fa3' : '#eee';
        const rawInfo  = peakRaw !== null
            ? `  raw_out: ${(Math.abs(peakRaw) < 0.0001 ? peakRaw.toExponential(3) : peakRaw.toPrecision(5))} × ${_testOutScale.toFixed(6)} = ${(peakRaw*_testOutScale).toPrecision(4)}`
            : '';
        outEl.innerHTML =
            `<div style="margin-bottom:4px">Duration: ${(audio.length/_SR).toFixed(2)}s  Frames: ${T}${rawInfo}\n` +
            `Peak: ${maxScore.toFixed(4)} at ${(maxIdx*_STEP/_SR).toFixed(3)}s  ` +
            `<span style="color:${hdrColor}">${det ? '✓ DETECTED' : '✗ not detected'}</span></div>` +
            _timelineHtml(scores, thr);
    } catch(err) { outEl.textContent = 'ERROR: ' + err.message; console.error(err); }
});

// ── Beep ──────────────────────────────────────────────────────────────────────
function _playBeep() {
    try {
        const ctx = new AudioContext(), osc = ctx.createOscillator(), g = ctx.createGain();
        osc.connect(g); g.connect(ctx.destination);
        osc.frequency.value = 880;
        g.gain.setValueAtTime(0.25, ctx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.18);
        osc.start(); osc.stop(ctx.currentTime + 0.18);
    } catch(e) {}
}

// ── Live mic ──────────────────────────────────────────────────────────────────
const _WORKLET_SRC = `
class MicProc extends AudioWorkletProcessor {
    process(inputs) { const ch=inputs[0][0]; if(ch) this.port.postMessage(ch.slice()); return true; }
}
registerProcessor('mic-proc', MicProc);
`;

function _micReset() {
    _micScoreBuf = new Array(5).fill(0);
    _micCooldown = 0;
    _micLeftover = new Float32Array(0);
    _testInferBuf = null;
    _initMicSpec();
}

function _micProcessStep(chunk) {
    if (!window.Module || !_micFE) return null;
    const int16 = new Int16Array(chunk.length);
    for (let i = 0; i < chunk.length; i++)
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(chunk[i] * 32767)));
    const pcmPtr  = Module._malloc(int16.byteLength);
    Module.HEAPU8.set(new Uint8Array(int16.buffer), pcmPtr);
    const nFrames = Module.ccall('frontend_process', 'number',
        ['number','number','number','number'],
        [_micFE, pcmPtr, int16.length, _micFeatPtr]);
    Module._free(pcmPtr);
    if (nFrames === 0) return null;
    const raw = new Uint16Array(_NMELS);
    const base = _micFeatPtr >> 1;
    for (let i = 0; i < _NMELS; i++) raw[i] = Module.HEAPU16[base + i];
    return raw;
}

function _micOnFrame(raw) {
    _appendSpecFrame(raw);
    const q     = _quantizeFrames(raw, 1);
    const score = _inferOneFrame(q);
    _micScoreBuf.shift(); _micScoreBuf.push(score);
    return Math.max(..._micScoreBuf);
}

function _updateMicUI(score) {
    const thr   = parseFloat(_thrSlider.value);
    const bar   = document.getElementById('test-mic-bar');
    document.getElementById('test-mic-score').textContent = score.toFixed(4);
    bar.style.width     = Math.min(100, score * 100).toFixed(1) + '%';
    bar.style.background = score >= thr ? '#16a34a' : '#2563eb';

    if (score >= thr && _micCooldown === 0) {
        _micCooldown = _COOLDOWN_FRAMES;
        const ts = new Date().toLocaleTimeString();
        document.getElementById('test-mic-detected').textContent = `DETECTED  ${ts}  (${score.toFixed(3)})`;
        _playBeep();
        const log = document.getElementById('test-mic-log');
        const el  = document.createElement('div');
        el.style.cssText = 'color:#16a34a;font-family:monospace';
        el.textContent   = `${ts}  score=${score.toFixed(4)}`;
        log.insertBefore(el, log.firstChild);
        while (log.children.length > 20) log.removeChild(log.lastChild);
    } else if (score < thr * 0.6) {
        document.getElementById('test-mic-detected').textContent = '';
    }
    if (_micCooldown > 0) _micCooldown--;
}

document.getElementById('test-mic-start').addEventListener('click', async () => {
    if (!_testModel || !window.Module) return;

    _micReset();
    _micFE      = Module._frontend_create();
    _micFeatPtr = Module._malloc(_NMELS * 2);

    const stream  = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    _micCtx       = new AudioContext({ sampleRate: _SR });
    const blob    = new Blob([_WORKLET_SRC], { type: 'application/javascript' });
    const blobUrl = URL.createObjectURL(blob);
    await _micCtx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    const src     = _micCtx.createMediaStreamSource(stream);
    const worklet = new AudioWorkletNode(_micCtx, 'mic-proc');

    worklet.port.onmessage = e => {
        if (!_micRunning) return;
        const tmp = new Float32Array(_micLeftover.length + e.data.length);
        tmp.set(_micLeftover); tmp.set(e.data, _micLeftover.length);
        _micLeftover = tmp;
        while (_micLeftover.length >= _STEP) {
            try {
                const raw = _micProcessStep(_micLeftover.subarray(0, _STEP));
                if (raw) _updateMicUI(_micOnFrame(raw));
            } catch(err) { console.error('[mic] frame error:', err); }
            _micLeftover = _micLeftover.slice(_STEP);
        }
    };

    src.connect(worklet);
    _micRunning = true;
    document.getElementById('test-mic-start').disabled = true;
    document.getElementById('test-mic-stop').disabled  = false;
});

document.getElementById('test-mic-stop').addEventListener('click', () => {
    _micRunning = false;
    if (_micCtx) { _micCtx.close(); _micCtx = null; }
    if (window.Module && _micFE)      { Module._frontend_destroy(_micFE); _micFE = null; }
    if (window.Module && _micFeatPtr) { Module._free(_micFeatPtr); _micFeatPtr = null; }
    document.getElementById('test-mic-stop').disabled  = true;
    document.getElementById('test-mic-start').disabled = false;
    document.getElementById('test-mic-score').textContent    = '—';
    document.getElementById('test-mic-bar').style.width      = '0%';
    document.getElementById('test-mic-detected').textContent = '';
});

// ── Startup: auto-load from IndexedDB cache ───────────────────────────────────
(async () => {
    _applyQuantMode();
    try {
        const has = await tfliteCacheHas();
        if (has) {
            const ab = await tfliteCacheLoad();
            await _loadTestModelBytes('wakeword.tflite (cached)', ab);
        }
    } catch(e) { console.log('[test] no cached TFLite:', e.message); }
})();
