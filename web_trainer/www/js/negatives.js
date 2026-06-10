'use strict';

// Pre-computed negative spectrogram bundle for in-browser wake-word training.
//
// Bundle binary format (little-endian):
//   Magic:         4 bytes "MWWN"
//   Version:       uint8 = 1
//   NumFrames:     uint16 LE  — frames per sample (e.g. 160)
//   NumFeatures:   uint8      — filterbanks (40)
//   NumCategories: uint8
//   Per category:
//     NameLen:    uint8
//     Name:       NameLen bytes (UTF-8)
//     NumSamples: uint32 LE
//     Data:       NumSamples × NumFrames × NumFeatures × uint16 LE
//                 multiply by 0.0390625 → float32 spectrogram value
//
// Build the bundle with:  python3 web_trainer/tools/build_negatives_bundle.py

const NEG_DB_NAME    = 'ww-negatives';
const NEG_DB_VERSION = 1;
const NEG_DB_STORE   = 'categories';
const NEG_SCALE      = 0.0390625;   // uint16 → float32 (matches microwakeword data.py)

// ── IndexedDB helpers ─────────────────────────────────────────────────────────

function _negOpenDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(NEG_DB_NAME, NEG_DB_VERSION);
        req.onupgradeneeded = e => e.target.result.createObjectStore(NEG_DB_STORE);
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _negDbGet(db, key) {
    return new Promise((resolve, reject) => {
        const req = db.transaction(NEG_DB_STORE, 'readonly').objectStore(NEG_DB_STORE).get(key);
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _negDbPut(db, key, value) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(NEG_DB_STORE, 'readwrite');
        tx.objectStore(NEG_DB_STORE).put(value, key);
        tx.oncomplete = resolve;
        tx.onerror    = e => reject(e.target.error);
    });
}

function _negDbAllKeys(db) {
    return new Promise((resolve, reject) => {
        const req = db.transaction(NEG_DB_STORE, 'readonly').objectStore(NEG_DB_STORE).getAllKeys();
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _negDbCount(db) {
    return new Promise((resolve, reject) => {
        const req = db.transaction(NEG_DB_STORE, 'readonly').objectStore(NEG_DB_STORE).count();
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _negDbClear(db) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(NEG_DB_STORE, 'readwrite');
        tx.objectStore(NEG_DB_STORE).clear();
        tx.oncomplete = resolve;
        tx.onerror    = e => reject(e.target.error);
    });
}

// ── Bundle parsing ────────────────────────────────────────────────────────────

// Parse a binary negatives bundle ArrayBuffer.
// Returns: { numFrames, numFeatures, categories: { [name]: { numSamples, numFrames, numFeatures, data: Uint16Array } } }
function negParseBundle(buffer) {
    const bytes = buffer instanceof ArrayBuffer ? buffer : buffer.buffer;
    const view  = new DataView(bytes);
    let off     = 0;

    const magic = String.fromCharCode(
        view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== 'MWWN') throw new Error(`Invalid negatives bundle (magic="${magic}")`);
    off = 4;

    const version = view.getUint8(off++);
    if (version !== 1) throw new Error(`Unsupported bundle version: ${version}`);
    const numFrames     = view.getUint16(off, true); off += 2;
    const numFeatures   = view.getUint8(off++);
    const numCategories = view.getUint8(off++);

    const categories = {};

    for (let c = 0; c < numCategories; c++) {
        const nameLen  = view.getUint8(off++);
        const name     = new TextDecoder().decode(new Uint8Array(bytes, off, nameLen));
        off += nameLen;

        const numSamples = view.getUint32(off, true); off += 4;
        const numValues  = numSamples * numFrames * numFeatures;
        // slice() into a fresh aligned buffer before wrapping in Uint16Array
        const data = new Uint16Array(bytes.slice(off, off + numValues * 2));
        off += numValues * 2;

        categories[name] = { numSamples, numFrames, numFeatures, data };
    }

    return { numFrames, numFeatures, categories };
}

// ── Public API ────────────────────────────────────────────────────────────────

async function negOpenDB() {
    return _negOpenDB();
}

// Returns true if negatives have been cached in IndexedDB.
async function negHas(db) {
    return (await _negDbCount(db)) > 0;
}

// Returns a summary: { [categoryName]: { numSamples } }
async function negInfo(db) {
    const keys = await _negDbAllKeys(db);
    const info = {};
    for (const key of keys) {
        const cat = await _negDbGet(db, key);
        if (cat) info[key] = { numSamples: cat.numSamples };
    }
    return info;
}

// Store parsed bundle data into IndexedDB. Each category is one record.
async function negStore(db, parsed) {
    for (const [name, cat] of Object.entries(parsed.categories)) {
        await _negDbPut(db, name, {
            numSamples:  cat.numSamples,
            numFrames:   cat.numFrames,
            numFeatures: cat.numFeatures,
            data:        cat.data,
        });
    }
}

// Load all categories from IndexedDB into memory for fast random access.
// Returns: { [categoryName]: { numSamples, numFrames, numFeatures, data: Uint16Array } }
async function negLoad(db) {
    const keys = await _negDbAllKeys(db);
    const cats = {};
    for (const key of keys) {
        cats[key] = await _negDbGet(db, key);
    }
    return cats;
}

// Get a random negative sample from in-memory categories (result of negLoad).
// Returns: Float32Array of length numFrames × numFeatures (row-major, frames × features).
function negGetSample(cats, category) {
    const cat = cats[category];
    if (!cat) throw new Error(`Unknown category: ${category}`);
    const stride = cat.numFrames * cat.numFeatures;
    const idx    = Math.floor(Math.random() * cat.numSamples);
    const slice  = cat.data.subarray(idx * stride, (idx + 1) * stride);
    const out    = new Float32Array(stride);
    for (let i = 0; i < stride; i++) out[i] = slice[i] * NEG_SCALE;
    return out;
}

// Download bundle from url, parse, store in IndexedDB, return loaded categories.
// onProgress(loaded, total) is called with byte counts during download.
async function negDownload(url, db, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching negatives bundle`);

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

    const parsed = negParseBundle(merged.buffer);
    await negStore(db, parsed);
    return await negLoad(db);
}

// Clear the IndexedDB cache.
async function negClear(db) {
    return _negDbClear(db);
}

// Expose as globals (plain <script> tags, no module system).
window.negOpenDB      = negOpenDB;
window.negHas         = negHas;
window.negInfo        = negInfo;
window.negParseBundle = negParseBundle;
window.negStore       = negStore;
window.negLoad        = negLoad;
window.negGetSample   = negGetSample;
window.negDownload    = negDownload;
window.negClear       = negClear;
