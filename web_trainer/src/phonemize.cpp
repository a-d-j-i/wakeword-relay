#include <cstdlib>
#include <cstring>
#include <string>

#include <espeak-ng/speak_lib.h>
#include <espeak-ng/espeak_ng.h>

// IPA mode: espeakPHONEMES_IPA (0x02) gives UTF-8 IPA output.
// Space (0x20) in bits 8-23 is used as separator between phoneme symbols.
// Produces e.g. "h eɪ l uː m ə s" which JS can split on spaces → per-phoneme array.
static const int kPhonemeMode = espeakPHONEMES_IPA | (' ' << 8);

static bool g_initialized = false;

extern "C" {

// Call once before any phonemize() call.
// Returns 0 on success.
int phonemize_init() {
    if (g_initialized) return 0;
    int r = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, "/espeak-ng-data", 0);
    if (r < 0) return -1;
    g_initialized = true;
    return 0;
}

// Convert text to space-separated IPA phonemes for the given espeak language code.
// lang: "en" for English, "es" for Spanish.
// Returns a malloc'd UTF-8 string; caller must call phonemize_free().
// Returns NULL on error.
char* phonemize(const char* text, const char* lang) {
    if (!g_initialized) return nullptr;
    if (espeak_SetVoiceByName(lang) != EE_OK) return nullptr;

    std::string result;
    const void* ptr = (const void*)text;

    while (ptr != nullptr) {
        const char* chunk = espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, kPhonemeMode);
        if (chunk && *chunk) {
            if (!result.empty()) result += ' ';
            result += chunk;
        }
    }

    char* out = (char*)malloc(result.size() + 1);
    memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

void phonemize_free(char* ptr) {
    free(ptr);
}

} // extern "C"
