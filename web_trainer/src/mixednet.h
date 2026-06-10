#pragma once
#include "nn.h"
#include <vector>
#include <cstdint>

// MixedNet matching microWakeWord production config:
//   first_conv(32, k=5, stride=3), 4 MixConv blocks (64 filters each),
//   global-avg-pool, Dense(1)+sigmoid.
//
// Block kernel configs (per-group depthwise kernels):
//   [0] = {5},  [1] = {7,11},  [2] = {9,15},  [3] = {23}
struct MixedNet {
    static constexpr int kInFeatures   = 40;
    static constexpr int kFirstFilters = 32;
    static constexpr int kFirstKernel  = 5;
    static constexpr int kFirstStride  = 3;
    static constexpr int kFilters      = 64;
    static constexpr int kNumBlocks    = 4;

    // First conv kernel: [kFirstKernel, kInFeatures, kFirstFilters]
    std::vector<float> first_conv;

    struct Block {
        std::vector<int> dw_ks;                   // kernel size per group
        int C_in;                                  // input channels
        std::vector<std::vector<float>> dw;        // dw[g]: kernel [dw_ks[g] * C_in/ngroups]
        std::vector<float> pw;                     // [C_in * kFilters]
        std::vector<float> bn_gamma, bn_beta;      // [kFilters]
        std::vector<float> bn_mean, bn_var;        // [kFilters]
    };
    Block blocks[kNumBlocks];

    // Dense: w[kFilters], b scalar
    std::vector<float> dense_w;
    float dense_b = 0.0f;

    // Initialize weights with He-scaled random values
    void init_random(uint32_t seed = 42);

    // Forward pass: spectrogram[T, kInFeatures] → probability in [0,1]
    // T must be >= min_input_frames() for the model to produce output.
    float forward(const float* spectrogram, int T) const;

    // Minimum number of input frames needed to produce at least one output frame
    static int min_input_frames();

    // Flat weight buffer (for serialization / JS transfer)
    int num_params() const;
    void get_params(float* dst) const;
    void set_params(const float* src);

    // Forward pass that caches activations needed for backprop
    float forward_cached(const float* spectrogram, int T, struct ForwardCache& cache) const;

    // Compute gradients for one (spectrogram, label) pair.
    // grad_flat must be zeroed by caller; layout matches get_params().
    // Returns BCE loss.
    float backward(const float* spectrogram, int T, float label,
                   const struct ForwardCache& cache, float* grad_flat) const;
};

// ── Activation cache ──────────────────────────────────────────────────────

struct BlockCache {
    int T_in = 0, T_out = 0;
    std::vector<std::vector<float>> gin;   // [ng][T_in * C_g] — depthwise group inputs
    std::vector<float> mixed;             // [T_out * C_in]   — after depthwise merge
    std::vector<float> pw_raw;           // [T_out * kFilters] — after pointwise, before BN
    std::vector<float> relu_out;         // [T_out * kFilters] — after BN+ReLU
};

struct ForwardCache {
    int T = 0, T1 = 0;
    std::vector<float> fc_relu_out;      // [T1 * MixedNet::kFirstFilters]
    BlockCache blk[MixedNet::kNumBlocks];
    std::vector<float> pooled;           // [MixedNet::kFilters]
    float logit = 0.f, prob = 0.f;
};

// ── Adam optimizer ────────────────────────────────────────────────────────

struct Adam {
    float lr     = 1e-3f;
    float beta1  = 0.9f;
    float beta2  = 0.999f;
    float eps_   = 1e-8f;
    int   t      = 0;
    std::vector<float> m, v;

    void init(int n_params, float lr = 1e-3f,
              float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    // One Adam step: updates params in-place, clears grad
    void step(float* params, const float* grad, int n);
};
