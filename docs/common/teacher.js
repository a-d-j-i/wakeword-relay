// teacher.js — browser port of train/teacher_infer.py (openWakeWord-style teacher).
//
// Runs the same three ONNX files exported by train_teacher.py, in the browser via
// onnxruntime-web, so the emulator (and trainer) can score live mic audio with the
// teacher alongside the student .tflite (Distillation B5b):
//
//     audio 16 kHz float [-1,1]
//       -> melspectrogram.onnx   (10 ms hop, 32 mel bins)   -> [F, 32]
//       -> embedding_model.onnx  (76 mel frames -> 96-d, stride 8 / 80 ms) -> [N, 96]
//       -> head.onnx             (16 embeddings -> score)   -> [M]
//
// Kept byte-for-byte faithful to teacher_infer.py's scaling and windowing so the
// browser A/B matches what training saw. Depends on a global `ort` (onnxruntime-web),
// loaded from a CDN <script> by the host page. Exposes globals: `Teacher`, `encodeWav`.
//
// NOTE: browser mic ≠ ESP32 mic path, so teacher-vs-student here is a *relative*
// comparison only (accepted in the plan). Eval-only: nothing here trains.
'use strict';

// ── Contract constants (must match teacher_infer.py) ──────────────────────────
const T_SR                = 16000;
const T_MEL_HOP           = 160;   // samples per mel frame (10 ms)
const T_MEL_WINDOW_SAMPLES = 640;  // effective analysis window of the melspec ONNX
const T_EMB_MEL_WINDOW    = 76;    // mel frames consumed per embedding frame
const T_EMB_MEL_STRIDE    = 8;     // mel frames between embedding frames (80 ms)
const T_HEAD_EMB_WINDOW   = 16;    // embedding frames consumed by the head (~2 s)
const T_MEL_BINS          = 32;
const T_EMB_DIM           = 96;

// Shortest audio that yields one full head window (16 emb frames = 196 mel frames).
// 31840 samples ≈ 1.99 s. Mirrors teacher_infer.MIN_SAMPLES exactly.
const T_MIN_SAMPLES =
    T_MEL_WINDOW_SAMPLES +
    ((T_EMB_MEL_WINDOW + (T_HEAD_EMB_WINDOW - 1) * T_EMB_MEL_STRIDE) - 1) * T_MEL_HOP;

class Teacher {
    constructor() {
        this.mel = this.emb = this.head = null;
        this._melIn = this._embIn = this._headIn = null;
        this.loaded = false;
    }

    // Create the three ONNX sessions from raw file bytes (ArrayBuffer or Uint8Array).
    async load({ mel, emb, head }) {
        const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
        const toU8 = b => (b instanceof Uint8Array ? b : new Uint8Array(b));
        this.mel  = await ort.InferenceSession.create(toU8(mel),  opts);
        this.emb  = await ort.InferenceSession.create(toU8(emb),  opts);
        this.head = await ort.InferenceSession.create(toU8(head), opts);
        this._melIn  = this.mel.inputNames[0];
        this._embIn  = this.emb.inputNames[0];
        this._headIn = this.head.inputNames[0];
        this.loaded = true;
        return this;
    }

    // float [-1,1] audio -> Float32Array [F*32] (row-major [frame, mel]), plus F.
    // openWakeWord's melspec ONNX expects int16-range floats, then x/10 + 2.
    async _melspectrogram(audio) {
        const x = new Float32Array(audio.length);
        for (let i = 0; i < audio.length; i++) x[i] = audio[i] * 32767.0;
        const out = await this.mel.run({
            [this._melIn]: new ort.Tensor('float32', x, [1, x.length]),
        });
        const raw = out[this.mel.outputNames[0]];
        const data = raw.data;                    // squeeze -> [F, 32], row-major
        const F = data.length / T_MEL_BINS;
        const mel = new Float32Array(data.length);
        for (let i = 0; i < data.length; i++) mel[i] = data[i] / 10.0 + 2.0;
        return { mel, F };
    }

    // float [-1,1] audio -> Float32Array [N*96] (row-major [frame, dim]), plus N.
    async _embed(audio) {
        let { mel, F } = await this._melspectrogram(audio);
        // Front-pad mel frames if shorter than one embedding window.
        if (F < T_EMB_MEL_WINDOW) {
            const padded = new Float32Array(T_EMB_MEL_WINDOW * T_MEL_BINS);
            padded.set(mel, (T_EMB_MEL_WINDOW - F) * T_MEL_BINS);
            mel = padded; F = T_EMB_MEL_WINDOW;
        }
        const N = Math.floor((F - T_EMB_MEL_WINDOW) / T_EMB_MEL_STRIDE) + 1;
        // Build [N, 76, 32, 1] batch: window i starts at mel frame i*stride.
        const win = new Float32Array(N * T_EMB_MEL_WINDOW * T_MEL_BINS);
        const rowLen = T_EMB_MEL_WINDOW * T_MEL_BINS;
        for (let i = 0; i < N; i++) {
            const src = i * T_EMB_MEL_STRIDE * T_MEL_BINS;
            win.set(mel.subarray(src, src + rowLen), i * rowLen);
        }
        const out = await this.emb.run({
            [this._embIn]: new ort.Tensor(
                'float32', win, [N, T_EMB_MEL_WINDOW, T_MEL_BINS, 1]),
        });
        const emb = out[this.emb.outputNames[0]].data;   // [N, ...] -> reshape [N, 96]
        return { emb: emb instanceof Float32Array ? emb : Float32Array.from(emb), N };
    }

    // [N*96] embeddings -> Float32Array of scores, one per 16-frame sliding window.
    async _scoreWindows(emb, N) {
        if (N < T_HEAD_EMB_WINDOW) {
            const padded = new Float32Array(T_HEAD_EMB_WINDOW * T_EMB_DIM);
            padded.set(emb, (T_HEAD_EMB_WINDOW - N) * T_EMB_DIM);
            emb = padded; N = T_HEAD_EMB_WINDOW;
        }
        const M = N - T_HEAD_EMB_WINDOW + 1;
        const rowLen = T_HEAD_EMB_WINDOW * T_EMB_DIM;
        const win = new Float32Array(M * rowLen);
        for (let j = 0; j < M; j++) {
            const src = j * T_EMB_DIM;                    // stride 1 in emb frames
            win.set(emb.subarray(src, src + rowLen), j * rowLen);
        }
        const out = await this.head.run({
            [this._headIn]: new ort.Tensor(
                'float32', win, [M, T_HEAD_EMB_WINDOW, T_EMB_DIM]),
        });
        const s = out[this.head.outputNames[0]].data;
        return s instanceof Float32Array ? s : Float32Array.from(s);
    }

    // float [-1,1] audio -> max wake-word probability over the clip (mirrors
    // teacher_infer.Teacher.score_clip: front-pad audio to MIN_SAMPLES).
    async scoreClip(audio) {
        if (audio.length < T_MIN_SAMPLES) {
            const padded = new Float32Array(T_MIN_SAMPLES);
            padded.set(audio, T_MIN_SAMPLES - audio.length);
            audio = padded;
        }
        const { emb, N } = await this._embed(audio);
        const scores = await this._scoreWindows(emb, N);
        let mx = 0;
        for (let i = 0; i < scores.length; i++) if (scores[i] > mx) mx = scores[i];
        return mx;
    }

    // float [-1,1] audio -> { scores, peak }: per-window scores across the whole
    // clip (for WAV re-scoring / timelines), no front-pad beyond what windowing needs.
    async scoreTimeline(audio) {
        if (audio.length < T_MIN_SAMPLES) {
            const padded = new Float32Array(T_MIN_SAMPLES);
            padded.set(audio, T_MIN_SAMPLES - audio.length);
            audio = padded;
        }
        const { emb, N } = await this._embed(audio);
        const scores = await this._scoreWindows(emb, N);
        let peak = 0;
        for (let i = 0; i < scores.length; i++) if (scores[i] > peak) peak = scores[i];
        return { scores, peak };
    }
}

Teacher.MIN_SAMPLES = T_MIN_SAMPLES;
Teacher.SR = T_SR;

// ── WAV encode (16-bit PCM mono @ given sample rate) ──────────────────────────
// Float32 [-1,1] -> ArrayBuffer of a canonical .wav, for the clip bank / downloads.
function encodeWav(float32, sampleRate = T_SR) {
    const n = float32.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const dv = new DataView(buf);
    const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
    ws(0, 'RIFF');            dv.setUint32(4, 36 + n * 2, true);
    ws(8, 'WAVE');            ws(12, 'fmt ');
    dv.setUint32(16, 16, true);     dv.setUint16(20, 1, true);   // PCM
    dv.setUint16(22, 1, true);      dv.setUint32(24, sampleRate, true);
    dv.setUint32(28, sampleRate * 2, true);
    dv.setUint16(32, 2, true);      dv.setUint16(34, 16, true);
    ws(36, 'data');          dv.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        off += 2;
    }
    return buf;
}

if (typeof window !== 'undefined') { window.Teacher = Teacher; window.encodeWav = encodeWav; }
