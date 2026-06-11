// Native test: build a TFLite from random weights + template, verify it loads.
// Usage: ./test_tflite_export <template.tflite> [out.tflite]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "../src/mixednet.h"
#include "../src/tflite_export.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <template.tflite> [out.tflite]\n", argv[0]);
        return 1;
    }

    // Load template
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long tsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> tmpl((size_t)tsz);
    fread(tmpl.data(), 1, (size_t)tsz, f);
    fclose(f);
    printf("Template: %ld bytes\n", tsz);

    // Build MixedNet with random weights (pooled=0)
    MixedNet net;
    net.pooled = false;
    net.init_random(42);
    printf("MixedNet: %d params, min_frames=%d\n", net.num_params(), net.min_input_frames());

    // Export TFLite
    size_t out_size = 0;
    uint8_t* buf = mixednet_build_tflite(&net, tmpl.data(), tmpl.size(), 100, &out_size);
    if (!buf) { fprintf(stderr, "Export failed\n"); return 1; }
    printf("Output TFLite: %zu bytes\n", out_size);

    // Check magic
    if (memcmp(buf + 4, "TFL3", 4) == 0)
        printf("Magic OK: TFL3\n");
    else
        printf("Magic WRONG: %c%c%c%c\n", buf[4], buf[5], buf[6], buf[7]);

    // Optionally write output
    if (argc >= 3) {
        FILE* fo = fopen(argv[2], "wb");
        if (fo) { fwrite(buf, 1, out_size, fo); fclose(fo); printf("Written to %s\n", argv[2]); }
    }

    free(buf);
    return 0;
}
