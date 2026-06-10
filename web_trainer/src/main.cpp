#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

extern "C" {
#include "frontend.h"
#include "frontend_util.h"
}

#include "mixednet.h"

// MicroFrontend configuration matching microWakeWord / ESP32-S3 settings.
// These must match exactly what the model was trained with.
static constexpr int kSampleRate       = 16000;
static constexpr int kWindowSizeMs     = 30;
static constexpr int kWindowStepMs     = 10;
static constexpr int kNumFilterbanks   = 40;
static constexpr int kNumFeatures      = 40;  // one frame = 40 int8 values

struct Frontend {
    FrontendState state;
    FrontendConfig config;
};

extern "C" {

Frontend* frontend_create() {
    auto* fe = new Frontend();

    FrontendConfig& cfg = fe->config;
    cfg.window.size_ms       = kWindowSizeMs;
    cfg.window.step_size_ms  = kWindowStepMs;
    cfg.filterbank.num_channels = kNumFilterbanks;
    cfg.filterbank.lower_band_limit = 125.0f;
    cfg.filterbank.upper_band_limit = 7500.0f;
    cfg.noise_reduction.smoothing_bits        = 10;
    cfg.noise_reduction.even_smoothing        = 0.025f;
    cfg.noise_reduction.odd_smoothing         = 0.06f;
    cfg.noise_reduction.min_signal_remaining  = 0.05f;
    cfg.pcan_gain_control.enable_pcan         = 1;
    cfg.pcan_gain_control.strength            = 0.95f;
    cfg.pcan_gain_control.offset              = 80.0f;
    cfg.pcan_gain_control.gain_bits           = 21;
    cfg.log_scale.enable_log = 1;
    cfg.log_scale.scale_shift = 6;

    if (!FrontendPopulateState(&cfg, &fe->state, kSampleRate)) {
        delete fe;
        return nullptr;
    }
    return fe;
}

// Process one chunk of int16 PCM audio.
// Returns number of feature frames generated (0 or 1 per step-sized chunk).
// features_out must point to a buffer of at least (return_value * kNumFeatures) int8 values.
int frontend_process(Frontend* fe, const int16_t* pcm, int num_samples,
                     int8_t* features_out) {
    size_t num_samples_remaining = (size_t)num_samples;
    const int16_t* input = pcm;
    int frames = 0;

    while (num_samples_remaining > 0) {
        size_t num_read;
        FrontendOutput output = FrontendProcessSamples(
            &fe->state, input, num_samples_remaining, &num_read);
        input               += num_read;
        num_samples_remaining -= num_read;

        if (output.size > 0) {
            for (size_t i = 0; i < output.size; ++i)
                features_out[frames * kNumFeatures + i] =
                    static_cast<int8_t>(output.values[i]);
            ++frames;
        }
    }
    return frames;
}

void frontend_destroy(Frontend* fe) {
    FrontendFreeStateContents(&fe->state);
    delete fe;
}

int frontend_num_features() { return kNumFeatures; }

// ── MixedNet ─────────────────────────────────────────────────────────────

MixedNet* mixednet_create() {
    auto* m = new MixedNet();
    m->init_random(42);
    return m;
}

void mixednet_destroy(MixedNet* m) { delete m; }

void mixednet_init_random(MixedNet* m, uint32_t seed) { m->init_random(seed); }

int mixednet_num_params(MixedNet* m) { return m->num_params(); }

// Returns pointer to a freshly-malloc'd float buffer; caller must free().
float* mixednet_get_params(MixedNet* m) {
    int n = m->num_params();
    float* buf = (float*)malloc(n * sizeof(float));
    m->get_params(buf);
    return buf;
}

void mixednet_set_params(MixedNet* m, const float* src, int len) {
    (void)len;
    m->set_params(src);
}

// spectrogram: float[T * 40], scaled to the same range as the MicroFrontend output.
// Returns probability in [0, 1].
float mixednet_forward(MixedNet* m, const float* spectrogram, int T) {
    return m->forward(spectrogram, T);
}

int mixednet_min_frames() { return MixedNet::min_input_frames(); }

// ── Adam ──────────────────────────────────────────────────────────────────

Adam* adam_create(MixedNet* m, float lr) {
    auto* a = new Adam();
    a->init(m->num_params(), lr);
    return a;
}

void adam_destroy(Adam* a) { delete a; }

void adam_set_lr(Adam* a, float lr) { a->lr = lr; }

// ── Training ──────────────────────────────────────────────────────────────

// Run one forward+backward+Adam step.
// spectrogram: float[T * 40], label: 0.0 or 1.0
// Returns BCE loss.
float train_step(MixedNet* m, Adam* a,
                 const float* spectrogram, int T, float label) {
    ForwardCache cache;
    m->forward_cached(spectrogram, T, cache);

    int np = m->num_params();
    std::vector<float> grad(np, 0.0f);
    float loss = m->backward(spectrogram, T, label, cache, grad.data());

    std::vector<float> params(np);
    m->get_params(params.data());
    a->step(params.data(), grad.data(), np);
    m->set_params(params.data());

    return loss;
}

} // extern "C"

#ifndef __EMSCRIPTEN__
// Native smoke-test: train a few steps and verify loss decreases
int main() {
    int T = MixedNet::min_input_frames() + 10;
    int F = MixedNet::kInFeatures;

    MixedNet net;
    net.init_random(42);
    Adam opt;
    opt.init(net.num_params(), 1e-3f);

    // Fixed random spectrogram
    std::vector<float> spec(T * F);
    uint32_t seed = 123;
    for (float& v : spec) {
        seed = seed * 1664525u + 1013904223u;
        v = ((int32_t)seed) / 2147483648.0f * 0.5f + 0.5f;
    }

    float prev_loss = 1e9f;
    for (int i = 0; i < 20; ++i) {
        float loss = train_step(&net, &opt, spec.data(), T, 1.0f);
        printf("step %2d  loss=%.6f\n", i, loss);
        prev_loss = loss;
    }
    printf("Final loss=%.6f  (expect <0.4 after 20 steps)\n", prev_loss);
    return prev_loss < 0.4f ? 0 : 1;
}
#endif
