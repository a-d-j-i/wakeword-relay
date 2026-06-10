#include <cstdio>
#include <espeak-ng/speak_lib.h>
#include <espeak-ng/espeak_ng.h>

int main() {
    int r = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0,
                              "/usr/lib/x86_64-linux-gnu/espeak-ng-data", 0);
    if (r < 0) { fprintf(stderr, "init failed\n"); return 1; }

    static const int MODE = espeakPHONEMES_IPA | (' ' << 8);  // IPA + space separator

    struct Test { const char* text; const char* lang; };
    Test cases[] = {
        {"hey lumus",      "en"},
        {"chispa magica",  "es"},
        {"buenas noches",  "es"},
        {nullptr, nullptr}
    };

    for (int i = 0; cases[i].text; i++) {
        espeak_SetVoiceByName(cases[i].lang);
        const void* ptr = cases[i].text;
        printf("%-20s [%s] -> ", cases[i].text, cases[i].lang);
        while (ptr) {
            const char* ph = espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, MODE);
            if (ph && *ph) printf("%s", ph);
        }
        printf("\n");
    }
    return 0;
}
