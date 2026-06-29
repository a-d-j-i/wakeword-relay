'use strict';

// IndexedDB cache for the Piper voice model (.onnx + .onnx.json).
// Saves the user from re-picking the model file every session.
// IndexedDB is shared across tabs on the same origin.

const _DB_NAME    = 'piper_model_cache';
const _DB_VERSION = 1;
const _STORE      = 'model';
const _KEY_ONNX   = 'onnx';
const _KEY_CONFIG = 'config';

function _openDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(_DB_NAME, _DB_VERSION);
        req.onupgradeneeded = e => e.target.result.createObjectStore(_STORE);
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _put(db, key, value) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(_STORE, 'readwrite');
        tx.objectStore(_STORE).put(value, key);
        tx.oncomplete = resolve;
        tx.onerror    = e => reject(e.target.error);
    });
}

function _get(db, key) {
    return new Promise((resolve, reject) => {
        const req = db.transaction(_STORE, 'readonly').objectStore(_STORE).get(key);
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _clear(db) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(_STORE, 'readwrite');
        tx.objectStore(_STORE).clear();
        tx.oncomplete = resolve;
        tx.onerror    = e => reject(e.target.error);
    });
}

async function piperCacheHas() {
    try {
        const db  = await _openDB();
        const val = await _get(db, _KEY_ONNX);
        db.close();
        return val != null;
    } catch { return false; }
}

async function piperCacheSave(onnxBuf, configText) {
    const db = await _openDB();
    await _put(db, _KEY_ONNX,   onnxBuf instanceof ArrayBuffer ? onnxBuf : onnxBuf.slice().buffer);
    await _put(db, _KEY_CONFIG, configText);
    db.close();
}

async function piperCacheLoad() {
    const db         = await _openDB();
    const onnxBuf    = await _get(db, _KEY_ONNX);
    const configText = await _get(db, _KEY_CONFIG);
    db.close();
    if (!onnxBuf || !configText) throw new Error('No cached model found');
    return { onnxBuf, configText };
}

async function piperCacheClear() {
    try {
        const db = await _openDB();
        await _clear(db);
        db.close();
    } catch {}
}

window.piperCacheHas   = piperCacheHas;
window.piperCacheSave  = piperCacheSave;
window.piperCacheLoad  = piperCacheLoad;
window.piperCacheClear = piperCacheClear;
