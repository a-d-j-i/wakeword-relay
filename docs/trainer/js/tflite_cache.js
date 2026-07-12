'use strict';

// IndexedDB cache for the last trained TFLite model.
// Written by the Train tab after export; read by the Test tab on startup.

const _TFLITE_DB_NAME    = 'wakeword_tflite_cache';
const _TFLITE_DB_VERSION = 1;
const _TFLITE_STORE      = 'model';
const _TFLITE_KEY        = 'wakeword';

function _tfliteOpenDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(_TFLITE_DB_NAME, _TFLITE_DB_VERSION);
        req.onupgradeneeded = e => e.target.result.createObjectStore(_TFLITE_STORE);
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _tflitePut(db, value) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(_TFLITE_STORE, 'readwrite');
        tx.objectStore(_TFLITE_STORE).put(value, _TFLITE_KEY);
        tx.oncomplete = resolve;
        tx.onerror    = e => reject(e.target.error);
    });
}

function _tfliteGet(db) {
    return new Promise((resolve, reject) => {
        const req = db.transaction(_TFLITE_STORE, 'readonly').objectStore(_TFLITE_STORE).get(_TFLITE_KEY);
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

function _tfliteClear(db) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction(_TFLITE_STORE, 'readwrite');
        tx.objectStore(_TFLITE_STORE).clear();
        tx.oncomplete = resolve;
        tx.onerror    = e => reject(e.target.error);
    });
}

async function tfliteCacheHas() {
    try {
        const db  = await _tfliteOpenDB();
        const val = await _tfliteGet(db);
        db.close();
        return val != null;
    } catch { return false; }
}

async function tfliteCacheSave(bytes) {
    const buf = bytes instanceof ArrayBuffer ? bytes : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const db  = await _tfliteOpenDB();
    await _tflitePut(db, buf);
    db.close();
}

async function tfliteCacheLoad() {
    const db  = await _tfliteOpenDB();
    const buf = await _tfliteGet(db);
    db.close();
    if (!buf) throw new Error('No cached TFLite model');
    return buf instanceof ArrayBuffer ? buf : buf.buffer;
}

async function tfliteCacheClear() {
    try {
        const db = await _tfliteOpenDB();
        await _tfliteClear(db);
        db.close();
    } catch {}
}

window.tfliteCacheHas   = tfliteCacheHas;
window.tfliteCacheSave  = tfliteCacheSave;
window.tfliteCacheLoad  = tfliteCacheLoad;
window.tfliteCacheClear = tfliteCacheClear;
