'use strict';

// Background audio clip bundle for in-browser wake-word augmentation.
//
// Bundle binary format (little-endian):
//   Magic:       4 bytes "MWWB"
//   Version:     uint8 = 1
//   SampleRate:  uint16 LE (always 16000)
//   ClipSamples: uint32 LE — samples per clip (e.g. 64000 for 4.0 s @ 16 kHz)
//   NumClips:    uint32 LE
//   Data:        NumClips × ClipSamples × int16 LE
//               (value / 32768.0 → float32 audio sample)
//
// Storage: Origin Private File System (OPFS) — survives page reloads.
// Build the bundle with: python3 web_trainer/tools/build_noise_bundle.py

const NOISE_OPFS_FILENAME = 'noise_bundle.bin';
const NOISE_INT16_SCALE   = 1.0 / 32768.0;

// ── Bundle parsing ────────────────────────────────────────────────────────────

// Parse a binary noise bundle ArrayBuffer.
// Returns { sampleRate, clipSamples, numClips, data: Int16Array }
function noiseParseBundle(buffer) {
    const bytes = buffer instanceof ArrayBuffer ? buffer : buffer.buffer;
    const view  = new DataView(bytes);

    const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== 'MWWB') throw new Error(`Invalid noise bundle (magic="${magic}")`);

    const version = view.getUint8(4);
    if (version !== 1) throw new Error(`Unsupported noise bundle version: ${version}`);

    const sampleRate  = view.getUint16(5, true);
    const clipSamples = view.getUint32(7, true);
    const numClips    = view.getUint32(11, true);
    const dataOff     = 15;

    const expectedBytes = numClips * clipSamples * 2;
    if (bytes.byteLength < dataOff + expectedBytes) {
        throw new Error(`Noise bundle truncated (expected ${dataOff + expectedBytes} bytes, got ${bytes.byteLength})`);
    }

    const data = new Int16Array(bytes.slice(dataOff, dataOff + expectedBytes));
    return { sampleRate, clipSamples, numClips, data };
}

// ── OPFS helpers ──────────────────────────────────────────────────────────────

async function _noiseOpfsRoot() {
    if (!navigator.storage || !navigator.storage.getDirectory) {
        throw new Error('Origin Private File System not supported in this browser');
    }
    return navigator.storage.getDirectory();
}

// Returns true if the noise bundle is cached in OPFS.
async function noiseHasBundle() {
    try {
        const root = await _noiseOpfsRoot();
        await root.getFileHandle(NOISE_OPFS_FILENAME);
        return true;
    } catch {
        return false;
    }
}

// Write a raw ArrayBuffer to OPFS.
async function noiseStoreToOPFS(buffer) {
    const root     = await _noiseOpfsRoot();
    const fh       = await root.getFileHandle(NOISE_OPFS_FILENAME, { create: true });
    const writable = await fh.createWritable();
    await writable.write(buffer);
    await writable.close();
}

// Read the raw noise bundle ArrayBuffer from OPFS.
async function noiseLoadFromOPFS() {
    const root = await _noiseOpfsRoot();
    const fh   = await root.getFileHandle(NOISE_OPFS_FILENAME);
    const file = await fh.getFile();
    return file.arrayBuffer();
}

// Delete the cached noise bundle from OPFS.
async function noiseClearOPFS() {
    try {
        const root = await _noiseOpfsRoot();
        await root.removeEntry(NOISE_OPFS_FILENAME);
    } catch { /* already absent */ }
}

// ── Public API ────────────────────────────────────────────────────────────────

// Return a random clip as a fresh Float32Array.
// parsed: result of noiseParseBundle().
function noiseGetSample(parsed) {
    if (!parsed || parsed.numClips === 0) throw new Error('No noise clips loaded');
    const i   = Math.floor(Math.random() * parsed.numClips);
    const off = i * parsed.clipSamples;
    const src = parsed.data.subarray(off, off + parsed.clipSamples);
    const out = new Float32Array(parsed.clipSamples);
    for (let j = 0; j < parsed.clipSamples; j++) out[j] = src[j] * NOISE_INT16_SCALE;
    return out;
}

// Download bundle from url, store in OPFS, return parsed bundle.
// onProgress(loaded, total) is called with byte counts during download.
async function noiseDownload(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching noise bundle`);

    const total  = parseInt(resp.headers.get('content-length') || '0');
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded   = 0;

    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        if (onProgress) onProgress(loaded, total);
    }

    const size   = chunks.reduce((s, c) => s + c.byteLength, 0);
    const merged = new Uint8Array(size);
    let pos = 0;
    for (const c of chunks) { merged.set(c, pos); pos += c.byteLength; }

    await noiseStoreToOPFS(merged.buffer);
    return noiseParseBundle(merged.buffer);
}

// Check OPFS and return parsed bundle if cached, otherwise null.
async function noiseInit() {
    if (!(await noiseHasBundle())) return null;
    const buf = await noiseLoadFromOPFS();
    return noiseParseBundle(buf);
}

// Expose as globals (plain <script> tags, no module system).
window.noiseParseBundle  = noiseParseBundle;
window.noiseHasBundle    = noiseHasBundle;
window.noiseStoreToOPFS  = noiseStoreToOPFS;
window.noiseLoadFromOPFS = noiseLoadFromOPFS;
window.noiseClearOPFS    = noiseClearOPFS;
window.noiseGetSample    = noiseGetSample;
window.noiseDownload     = noiseDownload;
window.noiseInit         = noiseInit;
