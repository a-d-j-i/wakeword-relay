#include "mixednet.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>

// Block depthwise kernel sizes (one vector per block)
static const std::vector<int> kBlkKernels[MixedNet::kNumBlocks] = {
    {5}, {7, 11}, {9, 15}, {23}
};
// Input channels per block (first block receives kFirstFilters, rest receive kFilters)
static const int kBlkCIn[MixedNet::kNumBlocks] = {
    MixedNet::kFirstFilters,
    MixedNet::kFilters,
    MixedNet::kFilters,
    MixedNet::kFilters,
};

// Simple LCG for reproducible random init
static float lcg_randn(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    // Uniform [0,1), then map to approx normal with Box-Muller skipped — use ~[-1,1]
    return ((int32_t)s) / 2147483648.0f;
}

void MixedNet::init_random(uint32_t seed) {
    // First conv — He init: std = sqrt(2 / fan_in)
    int fc_fan = kFirstKernel * kInFeatures;
    float fc_std = sqrtf(2.0f / fc_fan);
    first_conv.resize(kFirstKernel * kInFeatures * kFirstFilters);
    for (float& v : first_conv) v = lcg_randn(seed) * fc_std;

    for (int b = 0; b < kNumBlocks; ++b) {
        Block& blk = blocks[b];
        blk.dw_ks = kBlkKernels[b];
        blk.C_in  = kBlkCIn[b];
        int ng = (int)blk.dw_ks.size();
        int C_g = blk.C_in / ng;

        blk.dw.resize(ng);
        for (int g = 0; g < ng; ++g) {
            int ks = blk.dw_ks[g];
            float std_g = sqrtf(2.0f / ks);
            blk.dw[g].resize(ks * C_g);
            for (float& v : blk.dw[g]) v = lcg_randn(seed) * std_g;
        }

        float pw_std = sqrtf(2.0f / blk.C_in);
        blk.pw.resize(blk.C_in * kFilters);
        for (float& v : blk.pw) v = lcg_randn(seed) * pw_std;

        blk.bn_gamma.assign(kFilters, 1.0f);
        blk.bn_beta.assign(kFilters, 0.0f);
        blk.bn_mean.assign(kFilters, 0.0f);
        blk.bn_var.assign(kFilters, 1.0f);
    }

    float d_std = sqrtf(2.0f / kFilters);
    dense_w.resize(kFilters);
    for (float& v : dense_w) v = lcg_randn(seed) * d_std;
    dense_b = 0.0f;
}

int MixedNet::min_input_frames() {
    // Work backwards: each block needs at least max_kernel frames from previous
    int min_post_blk = 1;  // at least 1 output frame from last block
    for (int b = kNumBlocks - 1; b >= 0; --b) {
        int max_k = *std::max_element(kBlkKernels[b].begin(), kBlkKernels[b].end());
        min_post_blk += max_k - 1;
    }
    // Before first conv (stride): T >= min_post_blk * kFirstStride + kFirstKernel - 1
    return min_post_blk * kFirstStride + (kFirstKernel - 1);
}

float MixedNet::forward(const float* spectrogram, int T) const {
    // 1. First conv: [T, 40] → [T1, 32], ReLU
    int T1 = valid_out(T, kFirstKernel, kFirstStride);
    assert(T1 > 0);
    std::vector<float> x(T1 * kFirstFilters);
    conv1d_valid(spectrogram, T, kInFeatures, x.data(), kFirstFilters,
                 first_conv.data(), kFirstKernel, kFirstStride);
    relu_inplace(x.data(), T1 * kFirstFilters);

    // 2. Blocks
    int T_cur = T1, C_cur = kFirstFilters;

    for (int b = 0; b < kNumBlocks; ++b) {
        const Block& blk = blocks[b];
        int ng     = (int)blk.dw_ks.size();
        int max_ks = *std::max_element(blk.dw_ks.begin(), blk.dw_ks.end());
        int T_next = valid_out(T_cur, max_ks);  // stride=1
        assert(T_next > 0);

        int C_g = C_cur / ng;  // channels per group
        std::vector<float> mixed(T_next * C_cur, 0.0f);

        for (int g = 0; g < ng; ++g) {
            int ks    = blk.dw_ks[g];
            int T_g   = valid_out(T_cur, ks);
            int trim  = T_g - T_next;  // frames dropped from beginning (causal align)

            // Extract group input: x[T_cur, C_cur] → gin[T_cur, C_g]
            int g_off = g * C_g;
            std::vector<float> gin(T_cur * C_g);
            for (int t = 0; t < T_cur; ++t)
                memcpy(gin.data() + t * C_g, x.data() + t * C_cur + g_off,
                       C_g * sizeof(float));

            // Depthwise conv → dw_out[T_g, C_g]
            std::vector<float> dw_out(T_g * C_g);
            depthwise_conv1d_valid(gin.data(), T_cur, C_g,
                                   dw_out.data(), blk.dw[g].data(), ks);

            // Copy trimmed slice into mixed output at group channel offset
            for (int t = 0; t < T_next; ++t)
                memcpy(mixed.data() + t * C_cur + g_off,
                       dw_out.data() + (t + trim) * C_g,
                       C_g * sizeof(float));
        }

        // Pointwise conv → pw_out[T_next, kFilters]
        std::vector<float> pw_out(T_next * kFilters);
        pointwise_conv(mixed.data(), T_next, C_cur, pw_out.data(), kFilters,
                       blk.pw.data());

        // BatchNorm + ReLU (inference mode)
        batchnorm_infer(pw_out.data(), T_next, kFilters,
                        blk.bn_gamma.data(), blk.bn_beta.data(),
                        blk.bn_mean.data(), blk.bn_var.data());
        relu_inplace(pw_out.data(), T_next * kFilters);

        x     = std::move(pw_out);
        T_cur = T_next;
        C_cur = kFilters;
    }

    // 3. Global average pool: [T_cur, 64] → [64]
    std::vector<float> pooled(kFilters);
    avg_pool_global(x.data(), T_cur, kFilters, pooled.data());

    // 4. Dense(64→1) + sigmoid
    float logit = dense_b;
    for (int i = 0; i < kFilters; ++i) logit += pooled[i] * dense_w[i];
    return sigmoid_f(logit);
}

// ── Cached forward pass ────────────────────────────────────────────────────

float MixedNet::forward_cached(const float* spectrogram, int T, ForwardCache& cache) const {
    cache.T  = T;
    cache.T1 = valid_out(T, kFirstKernel, kFirstStride);
    assert(cache.T1 > 0);

    // 1. First conv: [T, 40] → [T1, 32], ReLU
    cache.fc_relu_out.resize(cache.T1 * kFirstFilters);
    conv1d_valid(spectrogram, T, kInFeatures,
                 cache.fc_relu_out.data(), kFirstFilters,
                 first_conv.data(), kFirstKernel, kFirstStride);
    relu_inplace(cache.fc_relu_out.data(), cache.T1 * kFirstFilters);

    // 2. Blocks
    const float* x_ptr = cache.fc_relu_out.data();
    int T_cur = cache.T1, C_cur = kFirstFilters;

    for (int b = 0; b < kNumBlocks; ++b) {
        const Block& blk = blocks[b];
        BlockCache&  bc  = cache.blk[b];
        int ng     = (int)blk.dw_ks.size();
        int max_ks = *std::max_element(blk.dw_ks.begin(), blk.dw_ks.end());
        int T_next = valid_out(T_cur, max_ks);
        assert(T_next > 0);

        bc.T_in  = T_cur;
        bc.T_out = T_next;
        int C_g = C_cur / ng;

        bc.gin.resize(ng);
        bc.mixed.assign(T_next * C_cur, 0.0f);

        for (int g = 0; g < ng; ++g) {
            int ks   = blk.dw_ks[g];
            int T_g  = valid_out(T_cur, ks);
            int trim = T_g - T_next;
            int g_off = g * C_g;

            // Save group input
            bc.gin[g].resize(T_cur * C_g);
            for (int t = 0; t < T_cur; ++t)
                memcpy(bc.gin[g].data() + t * C_g,
                       x_ptr + t * C_cur + g_off,
                       C_g * sizeof(float));

            // Depthwise conv
            std::vector<float> dw_out(T_g * C_g);
            depthwise_conv1d_valid(bc.gin[g].data(), T_cur, C_g,
                                   dw_out.data(), blk.dw[g].data(), ks);

            // Copy trimmed slice into mixed
            for (int t = 0; t < T_next; ++t)
                memcpy(bc.mixed.data() + t * C_cur + g_off,
                       dw_out.data() + (t + trim) * C_g,
                       C_g * sizeof(float));
        }

        // Pointwise conv
        bc.pw_raw.resize(T_next * kFilters);
        pointwise_conv(bc.mixed.data(), T_next, C_cur,
                       bc.pw_raw.data(), kFilters, blk.pw.data());

        // BatchNorm + ReLU
        bc.relu_out = bc.pw_raw;  // copy; BN runs in-place
        batchnorm_infer(bc.relu_out.data(), T_next, kFilters,
                        blk.bn_gamma.data(), blk.bn_beta.data(),
                        blk.bn_mean.data(), blk.bn_var.data());
        relu_inplace(bc.relu_out.data(), T_next * kFilters);

        x_ptr = bc.relu_out.data();
        T_cur = T_next;
        C_cur = kFilters;
    }

    // 3. Global average pool
    cache.pooled.resize(kFilters);
    avg_pool_global(x_ptr, T_cur, kFilters, cache.pooled.data());

    // 4. Dense + sigmoid
    cache.logit = dense_b;
    for (int i = 0; i < kFilters; ++i) cache.logit += cache.pooled[i] * dense_w[i];
    cache.prob = sigmoid_f(cache.logit);
    return cache.prob;
}

// ── Backward pass ──────────────────────────────────────────────────────────

float MixedNet::backward(const float* spectrogram, int T, float label,
                         const ForwardCache& cache, float* grad_flat) const {
    // grad_flat layout mirrors get_params():
    //   first_conv | blocks[b]: dw[g]... pw bn_gamma bn_beta bn_mean bn_var | dense_w dense_b

    // --- Build grad pointers into flat buffer (same walk as get_params) ---
    float* g_ptr = grad_flat;
    float* g_first_conv = g_ptr; g_ptr += first_conv.size();

    struct GBlock {
        std::vector<float*> dw;
        float* pw;
        float* bn_gamma;
        float* bn_beta;
        float* bn_mean;   // frozen — will stay zero
        float* bn_var;    // frozen — will stay zero
    };
    GBlock gb[kNumBlocks];
    for (int b = 0; b < kNumBlocks; ++b) {
        const Block& blk = blocks[b];
        int ng = (int)blk.dw_ks.size();
        gb[b].dw.resize(ng);
        for (int g = 0; g < ng; ++g) {
            gb[b].dw[g] = g_ptr;
            g_ptr += blk.dw[g].size();
        }
        gb[b].pw       = g_ptr; g_ptr += blk.pw.size();
        gb[b].bn_gamma = g_ptr; g_ptr += kFilters;
        gb[b].bn_beta  = g_ptr; g_ptr += kFilters;
        gb[b].bn_mean  = g_ptr; g_ptr += kFilters;
        gb[b].bn_var   = g_ptr; g_ptr += kFilters;
    }
    float* g_dense_w = g_ptr; g_ptr += kFilters;
    float& g_dense_b = *g_ptr;

    // --- 1. Loss + sigmoid: d_logit = prob - label ---
    float loss   = bce_loss(cache.prob, label);
    float d_logit = cache.prob - label;

    // --- 2. Dense grad ---
    for (int i = 0; i < kFilters; ++i) {
        g_dense_w[i] += d_logit * cache.pooled[i];
    }
    g_dense_b += d_logit;

    // d_pooled[i] = d_logit * dense_w[i]
    std::vector<float> d_pooled(kFilters);
    for (int i = 0; i < kFilters; ++i) d_pooled[i] = d_logit * dense_w[i];

    // --- 3. Avg pool backward → d_x for last block ---
    int T_last = cache.blk[kNumBlocks - 1].T_out;
    std::vector<float> d_x(T_last * kFilters, 0.0f);
    avg_pool_global_grad(d_pooled.data(), T_last, kFilters, d_x.data());

    // --- 4. Blocks backward (last to first) ---
    for (int b = kNumBlocks - 1; b >= 0; --b) {
        const Block& blk    = blocks[b];
        const BlockCache& bc = cache.blk[b];
        int ng  = (int)blk.dw_ks.size();
        int T_in  = bc.T_in;
        int T_out = bc.T_out;
        int C_in  = blk.C_in;
        int C_g   = C_in / ng;

        // 4a. ReLU backward (zero gradient where relu_out <= 0)
        relu_grad_inplace(d_x.data(), bc.relu_out.data(), T_out * kFilters);

        // 4b. BN backward → d_pw_raw
        std::vector<float> d_pw_raw(T_out * kFilters);
        batchnorm_grad(d_x.data(), bc.pw_raw.data(),
                       T_out, kFilters,
                       blk.bn_gamma.data(), blk.bn_mean.data(), blk.bn_var.data(),
                       1e-3f,
                       d_pw_raw.data(), gb[b].bn_gamma, gb[b].bn_beta);

        // 4c. Pointwise backward → d_mixed + grad for pw kernel
        std::vector<float> d_mixed(T_out * C_in, 0.0f);
        pointwise_grad_input(d_pw_raw.data(), T_out, kFilters,
                             blk.pw.data(), C_in, d_mixed.data());
        pointwise_grad_kernel(d_pw_raw.data(), T_out, kFilters,
                              bc.mixed.data(), C_in, gb[b].pw);

        // 4d. Depthwise backward per group → d_x_in
        std::vector<float> d_x_in(T_in * C_in, 0.0f);
        for (int g = 0; g < ng; ++g) {
            int ks   = blk.dw_ks[g];
            int T_g  = valid_out(T_in, ks);
            int trim = T_g - T_out;
            int g_off = g * C_g;

            // Un-trim: scatter d_mixed columns for this group into d_dw_full
            std::vector<float> d_dw_full(T_g * C_g, 0.0f);
            for (int t = 0; t < T_out; ++t)
                memcpy(d_dw_full.data() + (t + trim) * C_g,
                       d_mixed.data() + t * C_in + g_off,
                       C_g * sizeof(float));

            // Kernel gradient
            dw_conv1d_grad_kernel(d_dw_full.data(), T_g, C_g,
                                  bc.gin[g].data(), ks, gb[b].dw[g]);

            // Input gradient → into d_gin[g] then scatter to d_x_in
            std::vector<float> d_gin(T_in * C_g, 0.0f);
            dw_conv1d_grad_input(d_dw_full.data(), T_g, C_g,
                                 blk.dw[g].data(), ks, d_gin.data());
            for (int t = 0; t < T_in; ++t)
                memcpy(d_x_in.data() + t * C_in + g_off,
                       d_gin.data() + t * C_g,
                       C_g * sizeof(float));
        }

        // Prepare d_x for previous layer
        d_x = std::move(d_x_in);
    }

    // --- 5. First conv+relu backward ---
    // ReLU mask
    relu_grad_inplace(d_x.data(), cache.fc_relu_out.data(),
                      cache.T1 * kFirstFilters);

    // Conv kernel gradient
    conv1d_grad_kernel(d_x.data(), cache.T1, kFirstFilters,
                       spectrogram, kInFeatures,
                       kFirstKernel, kFirstStride, g_first_conv);

    return loss;
}

// ── Adam optimizer ─────────────────────────────────────────────────────────

void Adam::init(int n_params, float lr_, float beta1_, float beta2_, float eps) {
    lr    = lr_;
    beta1 = beta1_;
    beta2 = beta2_;
    eps_  = eps;
    t     = 0;
    m.assign(n_params, 0.0f);
    v.assign(n_params, 0.0f);
}

void Adam::step(float* params, const float* grad, int n) {
    ++t;
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    for (int i = 0; i < n; ++i) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps_);
    }
}

// ── Serialization ──────────────────────────────────────────────────────────

int MixedNet::num_params() const {
    int n = (int)first_conv.size();
    for (int b = 0; b < kNumBlocks; ++b) {
        const Block& blk = blocks[b];
        for (auto& k : blk.dw) n += (int)k.size();
        n += (int)blk.pw.size();
        n += 4 * kFilters;  // gamma, beta, mean, var
    }
    n += kFilters + 1;  // dense_w + dense_b
    return n;
}

void MixedNet::get_params(float* dst) const {
    auto copy = [&](const std::vector<float>& v) {
        memcpy(dst, v.data(), v.size() * sizeof(float));
        dst += v.size();
    };
    copy(first_conv);
    for (int b = 0; b < kNumBlocks; ++b) {
        const Block& blk = blocks[b];
        for (auto& k : blk.dw) copy(k);
        copy(blk.pw);
        copy(blk.bn_gamma); copy(blk.bn_beta);
        copy(blk.bn_mean);  copy(blk.bn_var);
    }
    copy(dense_w);
    *dst = dense_b;
}

void MixedNet::set_params(const float* src) {
    auto load = [&](std::vector<float>& v) {
        memcpy(v.data(), src, v.size() * sizeof(float));
        src += v.size();
    };
    load(first_conv);
    for (int b = 0; b < kNumBlocks; ++b) {
        Block& blk = blocks[b];
        for (auto& k : blk.dw) load(k);
        load(blk.pw);
        load(blk.bn_gamma); load(blk.bn_beta);
        load(blk.bn_mean);  load(blk.bn_var);
    }
    load(dense_w);
    dense_b = *src;
}
