// augment.js — Audio augmentation pipeline for the wake-word web trainer.
//
// All functions are pure (return new Float32Array; never mutate inputs).
// No external dependencies — runs in browser or Playwright headless.

'use strict';

// ── WAV loading ───────────────────────────────────────────────────────────────

// Parse a WAV ArrayBuffer → { samples: Float32Array, sampleRate }.
// Supports 16-bit PCM and 32-bit IEEE float, mono or multi-channel
// (only the first channel is returned).
function loadWav(buffer) {
    const v = new DataView(buffer);

    let numChannels, sampleRate, bitsPerSample;
    let dataOffset = 0, dataSize = 0;

    let offset = 12; // skip 'RIFF' + file size + 'WAVE'
    while (offset + 8 <= v.byteLength) {
        const id = String.fromCharCode(
            v.getUint8(offset),   v.getUint8(offset+1),
            v.getUint8(offset+2), v.getUint8(offset+3));
        const chunkSize = v.getUint32(offset + 4, true);

        if (id === 'fmt ') {
            numChannels  = v.getUint16(offset + 10, true);
            sampleRate   = v.getUint32(offset + 12, true);
            bitsPerSample = v.getUint16(offset + 22, true);
        } else if (id === 'data') {
            dataOffset = offset + 8;
            dataSize   = chunkSize;
            break;
        }
        offset += 8 + chunkSize + (chunkSize & 1); // WAV chunks are padded to even
    }

    if (!sampleRate || !dataOffset) throw new Error('Invalid or unsupported WAV file');

    const bytesPerSample = bitsPerSample / 8;
    const numSamples = Math.floor(dataSize / (bytesPerSample * numChannels));
    const samples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
        const byteOff = dataOffset + i * bytesPerSample * numChannels;
        if (bitsPerSample === 16) {
            samples[i] = v.getInt16(byteOff, true) / 32768.0;
        } else if (bitsPerSample === 32) {
            samples[i] = v.getFloat32(byteOff, true);
        }
        // Other bit depths produce silence (0)
    }

    return { samples, sampleRate };
}

// Fetch all bundled impulse responses from impulses/ under baseUrl.
// Returns an array of Float32Array (one per IR).
async function fetchImpulseResponses(baseUrl = '') {
    const names = [
        'Accoustic2_Impulse.wav',
        'Blatty%20Plate.wav',
        'Concrete%20Room.wav',
        'Derlon%20Sanctuary.wav',
        'Fat%20Bass.wav',
        'ir_bathroom1.wav',
        'Reverse%20Gate.wav',
        'Symphonic.wav',
    ];
    const irs = [];
    for (const name of names) {
        try {
            const buf = await fetch(baseUrl + 'impulses/' + name).then(r => {
                if (!r.ok) throw new Error(r.status);
                return r.arrayBuffer();
            });
            irs.push(loadWav(buf).samples);
        } catch (e) {
            console.warn('augment: skipping impulse', name, e.message);
        }
    }
    return irs;
}

// ── Pitch shift ───────────────────────────────────────────────────────────────

// Shift pitch by `semitones` (positive = up, negative = down) via linear
// resampling. Changes duration proportionally (pitch-only tools not needed
// for training-data diversity).
function pitchShift(samples, semitones) {
    if (semitones === 0) return new Float32Array(samples);
    const r = Math.pow(2, semitones / 12);
    const newLen = Math.round(samples.length / r);
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
        const pos = i * r;
        const i0 = Math.floor(pos);
        const i1 = Math.min(i0 + 1, samples.length - 1);
        const frac = pos - i0;
        out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
    }
    return out;
}

// ── Gain ──────────────────────────────────────────────────────────────────────

function applyGain(samples, gainDb) {
    const factor = Math.pow(10, gainDb / 20);
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) out[i] = samples[i] * factor;
    return out;
}

// ── 7-band parametric EQ ─────────────────────────────────────────────────────

// Center frequencies (Hz) for the 7 bands.
const _EQ_CENTERS = [32, 125, 500, 1000, 4000, 8000, 16000];
const _EQ_Q = 1.0;

// Apply one peaking biquad EQ band (Audio EQ Cookbook, RBJ).
function _peakingBiquad(samples, sampleRate, f0, Q, gainDb) {
    if (gainDb === 0) return new Float32Array(samples);
    const A    = Math.pow(10, gainDb / 40);
    const w0   = 2 * Math.PI * f0 / sampleRate;
    const cosw = Math.cos(w0);
    const sinw = Math.sin(w0);
    const alpha = sinw / (2 * Q);

    const a0 = 1 + alpha / A;
    const nb0 = (1 + alpha * A) / a0;
    const nb1 = (-2 * cosw)    / a0;
    const nb2 = (1 - alpha * A) / a0;
    const na1 = (-2 * cosw)    / a0;   // same as nb1 / a0, but a0 already applied
    const na2 = (1 - alpha / A) / a0;

    const out = new Float32Array(samples.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < samples.length; i++) {
        const x0 = samples[i];
        const y0 = nb0*x0 + nb1*x1 + nb2*x2 - na1*y1 - na2*y2;
        out[i] = y0;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;
    }
    return out;
}

// Apply 7-band parametric EQ.
// gainDbPerBand: 7-element array of gain values in dB (suggested range [-6, 6]).
function applyParametricEQ(samples, sampleRate, gainDbPerBand) {
    let out = samples;
    for (let b = 0; b < 7; b++) {
        const g = gainDbPerBand[b] || 0;
        if (Math.abs(g) > 0.001) {
            out = _peakingBiquad(out, sampleRate, _EQ_CENTERS[b], _EQ_Q, g);
        }
    }
    return out;
}

// ── FFT convolution ───────────────────────────────────────────────────────────

function _nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
}

// In-place radix-2 DIT FFT (Cooley-Tukey).
// re, im: Float64Array of length n (must be a power of 2).
function _fft(re, im) {
    const n = re.length;

    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            let t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }

    // Butterfly stages
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wRe = Math.cos(ang);
        const wIm = Math.sin(ang);
        for (let i = 0; i < n; i += len) {
            let uRe = 1, uIm = 0;
            for (let j = 0; j < (len >> 1); j++) {
                const k = i + j + (len >> 1);
                const vRe = re[k]*uRe - im[k]*uIm;
                const vIm = re[k]*uIm + im[k]*uRe;
                re[k] = re[i+j] - vRe;
                im[k] = im[i+j] - vIm;
                re[i+j] += vRe;
                im[i+j] += vIm;
                const nRe = uRe*wRe - uIm*wIm;
                uIm = uRe*wIm + uIm*wRe;
                uRe = nRe;
            }
        }
    }
}

// In-place IFFT via conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N.
function _ifft(re, im) {
    for (let i = 0; i < re.length; i++) im[i] = -im[i];
    _fft(re, im);
    const n = re.length;
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; }
}

// Linear convolution via FFT. Returns Float32Array of length signal.length + ir.length - 1.
function convolve(signal, ir) {
    const outLen = signal.length + ir.length - 1;
    const size = _nextPow2(outLen);

    const sRe = new Float64Array(size), sIm = new Float64Array(size);
    const hRe = new Float64Array(size), hIm = new Float64Array(size);
    for (let i = 0; i < signal.length; i++) sRe[i] = signal[i];
    for (let i = 0; i < ir.length;     i++) hRe[i] = ir[i];

    _fft(sRe, sIm);
    _fft(hRe, hIm);

    // Complex multiply
    for (let i = 0; i < size; i++) {
        const re = sRe[i]*hRe[i] - sIm[i]*hIm[i];
        const im = sRe[i]*hIm[i] + sIm[i]*hRe[i];
        sRe[i] = re; sIm[i] = im;
    }

    _ifft(sRe, sIm);

    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) out[i] = sRe[i];
    return out;
}

// ── Reverb ────────────────────────────────────────────────────────────────────

// Convolve samples with an impulse response, then normalize if clipping and
// trim back to the original length.
function applyReverb(samples, ir) {
    const wet = convolve(samples, ir);

    let maxAbs = 0;
    for (let i = 0; i < wet.length; i++) {
        const a = Math.abs(wet[i]);
        if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs > 1.0) {
        for (let i = 0; i < wet.length; i++) wet[i] /= maxAbs;
    }

    // Trim to original length (tail is discarded)
    return wet.length <= samples.length ? wet : wet.subarray(0, samples.length);
}

// ── Background noise mixing ───────────────────────────────────────────────────

function _rms(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
}

// Mix noise into signal at a target SNR (dB). Wraps noise if shorter than signal.
function mixNoise(signal, noise, snrDb) {
    const sigRms = _rms(signal);
    const noiseRms = _rms(noise);

    const out = new Float32Array(signal.length);
    if (noiseRms < 1e-8) {
        out.set(signal);
        return out;
    }

    const targetNoiseRms = sigRms < 1e-8
        ? 0.01  // signal is silence: add quiet noise
        : sigRms / Math.pow(10, snrDb / 20);
    const noiseScale = targetNoiseRms / noiseRms;

    let maxAbs = 0;
    for (let i = 0; i < signal.length; i++) {
        const v = signal[i] + noise[i % noise.length] * noiseScale;
        out[i] = v;
        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
    if (maxAbs > 1.0) {
        for (let i = 0; i < out.length; i++) out[i] /= maxAbs;
    }
    return out;
}

// ── Main augmentation function ────────────────────────────────────────────────

// Apply randomised augmentations to a Float32Array audio clip.
//
// options:
//   irs          — array of Float32Array impulse responses (for reverb)
//   noises       — array of Float32Array background sounds
//   pitchProb    — probability of pitch shift (default 0.5)
//   eqProb       — probability of parametric EQ (default 0.5)
//   reverbProb   — probability of reverb (default 0.5)
//   noiseProb    — probability of background noise (default 0.75)
//   gainRange    — [minDb, maxDb] for the unconditional gain step (default [-45, 0])
//   snrRange     — [minDb, maxDb] SNR for noise mixing (default [-10, 10])
//   semitoneRange — [min, max] semitones (default [-3, 3])
function augmentSample(samples, sampleRate, options = {}) {
    const {
        irs          = [],
        noises       = [],
        pitchProb    = 0.5,
        eqProb       = 0.5,
        reverbProb   = 0.5,
        noiseProb    = 0.75,
        gainRange    = [-45, 0],
        snrRange     = [-10, 10],
        semitoneRange = [-3, 3],
    } = options;

    let out = new Float32Array(samples);

    // Unconditional gain (matches Python's Gain(p=1.0))
    const gainDb = gainRange[0] + Math.random() * (gainRange[1] - gainRange[0]);
    out = applyGain(out, gainDb);

    // Pitch shift
    if (Math.random() < pitchProb) {
        const semitones = semitoneRange[0] +
            Math.random() * (semitoneRange[1] - semitoneRange[0]);
        out = pitchShift(out, semitones);
    }

    // 7-band parametric EQ
    if (Math.random() < eqProb) {
        const gains = new Float32Array(7);
        for (let b = 0; b < 7; b++) gains[b] = (Math.random() * 2 - 1) * 6;
        out = applyParametricEQ(out, sampleRate, gains);
    }

    // Background noise
    if (Math.random() < noiseProb && noises.length > 0) {
        const noise = noises[Math.floor(Math.random() * noises.length)];
        const snrDb = snrRange[0] + Math.random() * (snrRange[1] - snrRange[0]);
        out = mixNoise(out, noise, snrDb);
    }

    // Reverb
    if (Math.random() < reverbProb && irs.length > 0) {
        const ir = irs[Math.floor(Math.random() * irs.length)];
        out = applyReverb(out, ir);
    }

    // Normalize if clipping (matches Python's Normalize(apply_to="only_too_loud_sounds"))
    let maxAbs = 0;
    for (let i = 0; i < out.length; i++) {
        const a = Math.abs(out[i]);
        if (a > maxAbs) maxAbs = a;
    }
    if (maxAbs > 1.0) {
        for (let i = 0; i < out.length; i++) out[i] /= maxAbs;
    }

    return out;
}
