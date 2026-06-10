#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "frontend.h"
#include "frontend_util.h"
}

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

} // extern "C"
