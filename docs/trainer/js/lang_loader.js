'use strict';

// Lazy-loads espeak-ng language data files into the WASM virtual filesystem.
// English (en) and Spanish (es) are pre-packed in piper_phonemize.data;
// all other languages are fetched on demand from /espeak-ng-data/ static assets.

// Maps espeak language code → path under /espeak-ng-data/lang/
// (codes with no subfamily sit directly in lang/)
const _LANG_PATHS = {
    "af":               "gmw/af",
    "am":               "sem/am",
    "an":               "roa/an",
    "ar":               "sem/ar",
    "as":               "inc/as",
    "az":               "trk/az",
    "ba":               "trk/ba",
    "be":               "zle/be",
    "bg":               "zls/bg",
    "bn":               "inc/bn",
    "bpy":              "inc/bpy",
    "bs":               "zls/bs",
    "ca":               "roa/ca",
    "chr":              "iro/chr",
    "cmn":              "sit/cmn",
    "cmn-Latn-pinyin":  "sit/cmn-Latn-pinyin",
    "cs":               "zlw/cs",
    "cv":               "trk/cv",
    "cy":               "cel/cy",
    "da":               "gmq/da",
    "de":               "gmw/de",
    "el":               "grk/el",
    "en":               "gmw/en",
    "en-029":           "gmw/en-029",
    "en-GB-scotland":   "gmw/en-GB-scotland",
    "en-GB-x-gbclan":   "gmw/en-GB-x-gbclan",
    "en-GB-x-gbcwmd":   "gmw/en-GB-x-gbcwmd",
    "en-GB-x-rp":       "gmw/en-GB-x-rp",
    "en-US":            "gmw/en-US",
    "en-US-nyc":        "gmw/en-US-nyc",
    "eo":               "art/eo",
    "es":               "roa/es",
    "es-419":           "roa/es-419",
    "et":               "urj/et",
    "eu":               "eu",
    "fa":               "ira/fa",
    "fa-Latn":          "ira/fa-Latn",
    "fi":               "urj/fi",
    "fr":               "roa/fr",
    "fr-BE":            "roa/fr-BE",
    "fr-CH":            "roa/fr-CH",
    "ga":               "cel/ga",
    "gd":               "cel/gd",
    "gn":               "sai/gn",
    "grc":              "grk/grc",
    "gu":               "inc/gu",
    "hak":              "sit/hak",
    "haw":              "map/haw",
    "he":               "sem/he",
    "hi":               "inc/hi",
    "hr":               "zls/hr",
    "ht":               "roa/ht",
    "hu":               "urj/hu",
    "hy":               "ine/hy",
    "hyw":              "ine/hyw",
    "ia":               "art/ia",
    "id":               "poz/id",
    "io":               "art/io",
    "is":               "gmq/is",
    "it":               "roa/it",
    "ja":               "jpx/ja",
    "jbo":              "art/jbo",
    "ka":               "ccs/ka",
    "kk":               "trk/kk",
    "kl":               "esx/kl",
    "kn":               "dra/kn",
    "ko":               "ko",
    "kok":              "inc/kok",
    "ku":               "ira/ku",
    "ky":               "trk/ky",
    "la":               "itc/la",
    "lb":               "gmw/lb",
    "lfn":              "art/lfn",
    "lt":               "bat/lt",
    "ltg":              "bat/ltg",
    "lv":               "bat/lv",
    "mi":               "poz/mi",
    "mk":               "zls/mk",
    "ml":               "dra/ml",
    "mr":               "inc/mr",
    "ms":               "poz/ms",
    "mt":               "sem/mt",
    "my":               "sit/my",
    "nb":               "gmq/nb",
    "nci":              "azc/nci",
    "ne":               "inc/ne",
    "nl":               "gmw/nl",
    "nog":              "trk/nog",
    "om":               "cus/om",
    "or":               "inc/or",
    "pa":               "inc/pa",
    "pap":              "roa/pap",
    "piqd":             "art/piqd",
    "pl":               "zlw/pl",
    "pt":               "roa/pt",
    "pt-BR":            "roa/pt-BR",
    "py":               "art/py",
    "qdb":              "art/qdb",
    "qu":               "qu",
    "quc":              "myn/quc",
    "qya":              "art/qya",
    "ro":               "roa/ro",
    "ru":               "zle/ru",
    "ru-LV":            "zle/ru-LV",
    "sd":               "inc/sd",
    "shn":              "tai/shn",
    "si":               "inc/si",
    "sjn":              "art/sjn",
    "sk":               "zlw/sk",
    "sl":               "zls/sl",
    "smj":              "urj/smj",
    "sq":               "ine/sq",
    "sr":               "zls/sr",
    "sv":               "gmq/sv",
    "sw":               "bnt/sw",
    "ta":               "dra/ta",
    "te":               "dra/te",
    "th":               "tai/th",
    "tk":               "trk/tk",
    "tn":               "bnt/tn",
    "tr":               "trk/tr",
    "tt":               "trk/tt",
    "ug":               "trk/ug",
    "uk":               "zle/uk",
    "ur":               "inc/ur",
    "uz":               "trk/uz",
    "vi":               "aav/vi",
    "vi-VN-x-central":  "aav/vi-VN-x-central",
    "vi-VN-x-south":    "aav/vi-VN-x-south",
    "yue":              "sit/yue",
    "yue-Latn-jyutping":"sit/yue-Latn-jyutping",
};

// Preloaded by piper_phonemize.data — skip fetching these.
const _PRELOADED = new Set(['en', 'es']);

// Tracks which languages have been loaded this session.
const _loaded = new Set(['en', 'es']);

// _phonemizeModule reference — set by initLangLoader() after WASM init.
let _piperModule = null;

function initLangLoader(mod) {
    _piperModule = mod;
}

// Ensure the given espeak language code is available in the WASM FS.
// Returns a promise that resolves when the language is ready.
async function loadLanguage(lang) {
    // Use base code for lookup (e.g. 'es-419' → look up 'es-419', fallback 'es')
    const code = _LANG_PATHS[lang] ? lang : lang.split('-')[0];
    if (_loaded.has(code)) return;
    if (!_LANG_PATHS[code]) {
        console.warn('[lang_loader] unknown language code:', lang);
        return;
    }
    if (!_piperModule || !_piperModule.FS) {
        console.warn('[lang_loader] WASM module not ready');
        return;
    }

    const base = 'espeak-ng-data';
    const FS   = _piperModule.FS;

    // Fetch and write the dict file
    const dictResp = await fetch(`${base}/${code}_dict`);
    if (!dictResp.ok) throw new Error(`Failed to fetch ${code}_dict: ${dictResp.status}`);
    const dictData = new Uint8Array(await dictResp.arrayBuffer());
    FS.writeFile(`/espeak-ng-data/${code}_dict`, dictData);

    // Fetch and write the lang config file (may be in a subfamily dir)
    const langPath = _LANG_PATHS[code];
    const langResp = await fetch(`${base}/lang/${langPath}`);
    if (!langResp.ok) throw new Error(`Failed to fetch lang/${langPath}: ${langResp.status}`);
    const langText = await langResp.text();

    // Ensure the subfamily directory exists in WASM FS
    const parts = langPath.split('/');
    if (parts.length > 1) {
        const dir = `/espeak-ng-data/lang/${parts[0]}`;
        try { FS.mkdir(dir); } catch(e) { /* already exists */ }
    }
    FS.writeFile(`/espeak-ng-data/lang/${langPath}`, langText);

    _loaded.add(code);
    console.log(`[lang_loader] loaded: ${code}`);
}

window.initLangLoader  = initLangLoader;
window.loadLanguage    = loadLanguage;
window._LANG_PATHS     = _LANG_PATHS;
