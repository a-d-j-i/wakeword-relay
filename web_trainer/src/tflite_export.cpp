// Streaming INT8 TFLite exporter for browser-trained MixedNet weights.
//
// Uses a template TFLite (any pooled=0 output from train.py / quantize_from_browser.py)
// for the graph topology. Replaces weight buffers and quantization params.
//
// Template tensor→buffer mapping (fixed by architecture):
//   tensor N → buffer N+1, for N = 0..43
//   tensor 43: first_conv kernel  tensor 42: first_conv bias
//   tensor 41/40: blk0 dw kernel/bias  tensor 39/38: blk0 pw kernel/bias
//   tensor 37/36: blk1 dw-k7 kernel/bias  tensor 35/34: blk1 dw-k11 kernel/bias
//   tensor 33/32: blk1 pw kernel/bias
//   tensor 31/30: blk2 dw-k9  29/28: blk2 dw-k15  27/26: blk2 pw
//   tensor 25/24: blk3 dw-k23  23/22: blk3 pw  21: dense-w  20: dense-b
//
// Activation quantization groups (tensor indices share the same scale/zp):
//   input:         {0,50,53,54,55}  fixed: s=26/256, zp=-128
//   fc_relu:       {56,57,58,61}    calibrated
//   dw[0]:         {59}             calibrated (block 0)
//   pw_relu[0]:    {60,62,63,64,65,66,68,70}  calibrated (block 0 output)
//   dw[1]:         {69,71,72}       calibrated (block 1, combined k7+k11)
//   pw_relu[1]:    {67,73,74,75,76,77,78,80}  calibrated (block 1 output)
//   dw[2]:         {79,81,82}       calibrated (block 2, combined k9+k15)
//   pw_relu[2]:    {83,51,84,92}    calibrated (block 2 output)
//   dw[3]:         {85}             calibrated (block 3)
//   pw_relu[3]:    {86,52,87,88,91} calibrated (block 3 output)
//   dense_out:     {89}             calibrated
//   sigmoid:       {90}             fixed: s=1/256, zp=-128
//   output_uint8:  {93}             fixed: s=1/256, zp=0

#include "tflite_export.h"
#include "nn.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

// FlatBuffers + TFLite schema (vendored from esp-tflite-micro)
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "tflite_schema/schema_generated.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using MN = MixedNet;

// ── Calibration ──────────────────────────────────────────────────────────────

struct CalibStats {
    // All min/max initialised to extreme values; caller must reset before first use.
    float fc_relu_max = 0.f;          // always ≥ 0 (ReLU)
    float dw_min[4]  = {1e30f, 1e30f, 1e30f, 1e30f};
    float dw_max[4]  = {-1e30f,-1e30f,-1e30f,-1e30f};
    float pw_max[4]  = {0.f, 0.f, 0.f, 0.f};  // always ≥ 0 (ReLU)
    float dense_min  = 1e30f;
    float dense_max  = -1e30f;
};

static void update_range(float& mn, float& mx, const float* data, int n) {
    for (int i = 0; i < n; i++) {
        mn = std::min(mn, data[i]);
        mx = std::max(mx, data[i]);
    }
}

static void calib_one(const MixedNet& net, const float* spec, int T, CalibStats& s) {
    ForwardCache cache;
    net.forward_cached(spec, T, cache);

    // First conv ReLU
    s.fc_relu_max = std::max(s.fc_relu_max,
        *std::max_element(cache.fc_relu_out.begin(), cache.fc_relu_out.end()));

    // Per-block DW and PW stats
    for (int b = 0; b < MN::kNumBlocks; ++b) {
        const auto& blk = net.blocks[b];
        const auto& bc  = cache.blk[b];
        int ng   = (int)blk.dw_ks.size();
        int C_in = blk.C_in;
        int C_g  = C_in / ng;

        // Which dw_stat[] slot: blocks 0,1,2,3 → slots 0,1,2,3
        // For blocks with 2 groups, combine both groups into the same slot.
        for (int g = 0; g < ng; ++g) {
            int ks  = blk.dw_ks[g];
            int T_g = valid_out(bc.T_in, ks);
            std::vector<float> dw_out(T_g * C_g);
            depthwise_conv1d_valid(bc.gin[g].data(), bc.T_in, C_g,
                                   dw_out.data(), blk.dw[g].data(), ks);
            update_range(s.dw_min[b], s.dw_max[b], dw_out.data(), T_g * C_g);
        }

        // PW+BN+ReLU output (always ≥ 0)
        float blk_max = *std::max_element(bc.relu_out.begin(), bc.relu_out.end());
        s.pw_max[b] = std::max(s.pw_max[b], blk_max);
    }

    // Dense output (logit, can be any sign)
    s.dense_min = std::min(s.dense_min, cache.logit);
    s.dense_max = std::max(s.dense_max, cache.logit);
}

// Synthetic spectrogram: uniform in [0, 26] using a simple LCG
static void gen_spec(float* buf, int T, int F, uint32_t& seed) {
    for (int i = 0; i < T * F; i++) {
        seed = seed * 1664525u + 1013904223u;
        buf[i] = (float)(seed >> 8) / (float)(1u << 24) * 26.0f;
    }
}

static CalibStats run_calibration(const MixedNet& net, int n) {
    CalibStats s;
    int T = net.min_input_frames();
    std::vector<float> spec(T * MN::kInFeatures);
    uint32_t seed = 42;
    // Ensure extremes appear in the first spectrogram to anchor the range
    spec[0] = 0.f; spec[1] = 26.f;
    for (int i = 0; i < n; i++) {
        gen_spec(spec.data(), T, MN::kInFeatures, seed);
        if (i == 0) { spec[0] = 0.f; spec[1] = 26.f; }
        calib_one(net, spec.data(), T, s);
    }
    return s;
}

// ── BN folding ────────────────────────────────────────────────────────────────
// Fold gamma/beta/mean/var into pointwise conv weights + bias.
// Returns folded weights per block in the same layout as net.blocks[b].pw,
// and per-channel folded biases.

struct FoldedBlock {
    std::vector<float> pw;    // same layout as blocks[b].pw: [C_in * kFilters]
    std::vector<float> bias;  // [kFilters]
};

static FoldedBlock fold_bn(const MN::Block& blk, float eps = 1e-3f) {
    int F = MN::kFilters;
    int C = blk.C_in;
    FoldedBlock fb;
    fb.pw   = blk.pw;
    fb.bias.resize(F);
    for (int f = 0; f < F; ++f) {
        float inv_std = 1.0f / std::sqrt(blk.bn_var[f] + eps);
        float scale   = blk.bn_gamma[f] * inv_std;
        // Scale each row of pw corresponding to output channel f
        for (int c = 0; c < C; ++c)
            fb.pw[c * F + f] *= scale;
        fb.bias[f] = blk.bn_beta[f] - blk.bn_mean[f] * scale;
    }
    return fb;
}

// ── Quantization helpers ──────────────────────────────────────────────────────

// Asymmetric int8 scale/zp from [min, max].
static void act_quant(float lo, float hi, float& scale, int32_t& zp) {
    // Ensure range includes 0 so that the "zero" float maps correctly.
    lo = std::min(lo, 0.f);
    hi = std::max(hi, 0.f);
    if (hi == lo) hi = lo + 1e-6f;
    scale = (hi - lo) / 255.0f;
    // zero_point for int8 asymmetric: zp = round(-128 - lo/scale)
    float zp_f = std::round(-128.0f - lo / scale);
    zp = (int32_t)std::max(-128.f, std::min(127.f, zp_f));
}

// Per-channel symmetric weight quantization (|max| / 127, zp=0).
static std::vector<float> weight_scales_per_out_channel(
        const float* w, int C_out, int n_per_channel) {
    std::vector<float> scales(C_out);
    for (int c = 0; c < C_out; ++c) {
        float mx = 0.f;
        const float* row = w + c * n_per_channel;
        for (int i = 0; i < n_per_channel; ++i)
            mx = std::max(mx, std::abs(row[i]));
        scales[c] = (mx == 0.f) ? 1e-8f : mx / 127.0f;
    }
    return scales;
}

// Depthwise: quantized_dimension=3 (last dim = channels).
// In TFLite layout [1, K, 1, C_g], quantized over channel C_g.
// Each channel c corresponds to entries [t*C_g + c] for all t.
static std::vector<float> dw_weight_scales(const float* w, int kT, int C_g) {
    std::vector<float> scales(C_g);
    for (int c = 0; c < C_g; ++c) {
        float mx = 0.f;
        for (int t = 0; t < kT; ++t)
            mx = std::max(mx, std::abs(w[t * C_g + c]));
        scales[c] = (mx == 0.f) ? 1e-8f : mx / 127.0f;
    }
    return scales;
}

// Quantize float weight per-channel to int8 (symmetric, C_out channels last).
// For CONV_2D: w is [C_in, C_out], output layout [C_out, 1, 1, C_in].
static std::vector<int8_t> quantize_pw_kernel(
        const float* w, int C_in, int C_out, const std::vector<float>& scales) {
    std::vector<int8_t> out(C_out * C_in);
    for (int f = 0; f < C_out; ++f)
        for (int c = 0; c < C_in; ++c)
            out[f * C_in + c] = (int8_t)std::max(-127.f, std::min(127.f,
                std::round(w[c * C_out + f] / scales[f])));
    return out;
}

// First conv: C++ [kT, C_in, C_out] → TFLite [C_out, kT, 1, C_in]
static std::vector<int8_t> quantize_fc_kernel(
        const float* w, int kT, int C_in, int C_out,
        const std::vector<float>& scales) {
    std::vector<int8_t> out(C_out * kT * C_in);
    for (int co = 0; co < C_out; ++co)
        for (int t = 0; t < kT; ++t)
            for (int ci = 0; ci < C_in; ++ci)
                out[co * kT * C_in + t * C_in + ci] =
                    (int8_t)std::max(-127.f, std::min(127.f,
                        std::round(w[t * C_in * C_out + ci * C_out + co] / scales[co])));
    return out;
}

// Depthwise: C++ [kT, C_g] flat same as TFLite [1, kT, 1, C_g] (no reorder)
static std::vector<int8_t> quantize_dw_kernel(
        const float* w, int kT, int C_g,
        const std::vector<float>& scales) {
    std::vector<int8_t> out(kT * C_g);
    for (int i = 0; i < kT * C_g; ++i) {
        int c = i % C_g;
        out[i] = (int8_t)std::max(-127.f, std::min(127.f,
            std::round(w[i] / scales[c])));
    }
    return out;
}

// Int32 bias from float bias: bias_int32 = round(bias_float / (s_in * s_w[c]))
static std::vector<int32_t> quantize_bias(
        const float* bias, int n,
        float s_in, const std::vector<float>& s_w) {
    std::vector<int32_t> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = (int32_t)std::round(bias[i] / (s_in * s_w[i]));
    return out;
}

static std::vector<int32_t> zero_bias_int32(int n) {
    return std::vector<int32_t>(n, 0);
}

// ── FlatBuffer patching ────────────────────────────────────────────────────────

// Set scale+zp for all tensors in `indices` within the given subgraph.
static void set_quant(tflite::SubGraphT& sg,
                      std::initializer_list<int> indices,
                      float scale, int32_t zp) {
    for (int idx : indices) {
        auto& t = sg.tensors[idx];
        if (!t->quantization)
            t->quantization = std::make_unique<tflite::QuantizationParametersT>();
        t->quantization->scale      = {scale};
        t->quantization->zero_point = {(int64_t)zp};
    }
}

// Set per-channel scales (symmetric, zp=0) for a weight tensor.
static void set_per_channel_quant(tflite::SubGraphT& sg, int idx,
                                  const std::vector<float>& scales,
                                  int quantized_dim = 0) {
    auto& t = sg.tensors[idx];
    if (!t->quantization)
        t->quantization = std::make_unique<tflite::QuantizationParametersT>();
    t->quantization->scale.assign(scales.begin(), scales.end());
    t->quantization->zero_point.assign(scales.size(), 0LL);
    t->quantization->quantized_dimension = quantized_dim;
}

// Set per-channel bias quantization: scales = s_in * s_w[c], zp=0
static void set_bias_quant(tflite::SubGraphT& sg, int idx,
                           float s_in, const std::vector<float>& s_w) {
    auto& t = sg.tensors[idx];
    if (!t->quantization)
        t->quantization = std::make_unique<tflite::QuantizationParametersT>();
    int n = (int)s_w.size();
    t->quantization->scale.resize(n);
    t->quantization->zero_point.assign(n, 0LL);
    t->quantization->quantized_dimension = 0;
    for (int i = 0; i < n; ++i)
        t->quantization->scale[i] = s_in * s_w[i];
}

// Replace buffer data (buffer index = tensor_index + 1)
static void set_buffer_i8(tflite::ModelT& model, int tensor_idx,
                          const std::vector<int8_t>& data) {
    int buf_idx = tensor_idx + 1;
    model.buffers[buf_idx]->data.assign(
        reinterpret_cast<const uint8_t*>(data.data()),
        reinterpret_cast<const uint8_t*>(data.data()) + data.size());
}

static void set_buffer_i32(tflite::ModelT& model, int tensor_idx,
                           const std::vector<int32_t>& data) {
    int buf_idx = tensor_idx + 1;
    auto* src = reinterpret_cast<const uint8_t*>(data.data());
    model.buffers[buf_idx]->data.assign(src, src + data.size() * 4);
}

// ── Main export function ───────────────────────────────────────────────────────

uint8_t* mixednet_build_tflite(const MixedNet* netp,
                               const uint8_t* tmpl, size_t tmpl_size,
                               int calib_n,
                               size_t* out_size) {
    if (!netp || !tmpl || tmpl_size < 16 || !out_size) return nullptr;

    const MixedNet& net = *netp;

    // Only pooled=0 (production mode) is supported.
    if (net.pooled) return nullptr;

    // ── 1. Calibration ────────────────────────────────────────────────────
    CalibStats cs = run_calibration(net, calib_n);

    // Derive per-group activation scales/zps
    float s_input  = 26.0f / 256.0f;   int32_t zp_input  = -128;  // fixed

    float s_fc_relu;  int32_t zp_fc_relu;
    act_quant(0.f, cs.fc_relu_max, s_fc_relu, zp_fc_relu);  // ReLU → min=0

    float s_dw[4];   int32_t zp_dw[4];
    for (int i = 0; i < 4; ++i)
        act_quant(cs.dw_min[i], cs.dw_max[i], s_dw[i], zp_dw[i]);

    float s_pw[4];   int32_t zp_pw[4];
    for (int i = 0; i < 4; ++i)
        act_quant(0.f, cs.pw_max[i], s_pw[i], zp_pw[i]);   // ReLU → min=0

    float s_dense;   int32_t zp_dense;
    act_quant(cs.dense_min, cs.dense_max, s_dense, zp_dense);

    float s_sigmoid  = 1.0f / 256.0f;  int32_t zp_sigmoid  = -128;  // fixed
    float s_out_u8   = 1.0f / 256.0f;  int32_t zp_out_u8   = 0;     // fixed (uint8)

    // ── 2. BN folding ─────────────────────────────────────────────────────
    FoldedBlock fb[MN::kNumBlocks];
    for (int b = 0; b < MN::kNumBlocks; ++b)
        fb[b] = fold_bn(net.blocks[b]);

    // ── 3. Per-layer weight quantization ──────────────────────────────────

    // First conv: kernel [kFirstKernel, kInFeatures, kFirstFilters]
    //             TFLite [kFirstFilters, kFirstKernel, 1, kInFeatures]
    int FC_K = MN::kFirstKernel, FC_I = MN::kInFeatures, FC_O = MN::kFirstFilters;
    auto s_fc_w = weight_scales_per_out_channel(net.first_conv.data(), FC_O, FC_K * FC_I);
    auto fc_kernel_i8 = quantize_fc_kernel(net.first_conv.data(), FC_K, FC_I, FC_O, s_fc_w);
    auto fc_bias_i32  = zero_bias_int32(FC_O);  // first conv has no bias in C++

    // Blocks (in reverse order as stored in the TFLite: blk3=highest tensor, blk0=lowest)
    // blk0: tensor 41/40(dw), 39/38(pw)
    // blk1: tensor 37/36(dw-k7), 35/34(dw-k11), 33/32(pw)
    // blk2: tensor 31/30(dw-k9), 29/28(dw-k15), 27/26(pw)
    // blk3: tensor 25/24(dw), 23/22(pw)

    struct BlockQ {
        std::vector<std::vector<int8_t>>  dw_kernel_i8;   // per group
        std::vector<std::vector<int32_t>> dw_bias_i32;    // per group (zeros)
        std::vector<std::vector<float>>   dw_scales;      // per group
        std::vector<int8_t>  pw_kernel_i8;
        std::vector<int32_t> pw_bias_i32;
        std::vector<float>   pw_scales;     // per output channel
    };

    // s_dw_input[b]: scale of the tensor fed into block b's DW
    float s_dw_input[4] = { s_fc_relu, s_pw[0], s_pw[1], s_pw[2] };

    BlockQ bq[4];
    for (int b = 0; b < 4; ++b) {
        const auto& blk = net.blocks[b];
        int ng  = (int)blk.dw_ks.size();
        int C_in = blk.C_in;
        int C_g  = C_in / ng;

        bq[b].dw_kernel_i8.resize(ng);
        bq[b].dw_bias_i32.resize(ng);
        bq[b].dw_scales.resize(ng);

        for (int g = 0; g < ng; ++g) {
            int ks = blk.dw_ks[g];
            bq[b].dw_scales[g] = dw_weight_scales(blk.dw[g].data(), ks, C_g);
            bq[b].dw_kernel_i8[g] = quantize_dw_kernel(blk.dw[g].data(), ks, C_g, bq[b].dw_scales[g]);
            bq[b].dw_bias_i32[g]  = zero_bias_int32(C_g);
        }

        // PW conv quantization (input scale = s_dw[b] = DW output scale)
        int F = MN::kFilters;
        bq[b].pw_scales = weight_scales_per_out_channel(fb[b].pw.data(), F, C_in);
        bq[b].pw_kernel_i8 = quantize_pw_kernel(fb[b].pw.data(), C_in, F, bq[b].pw_scales);
        bq[b].pw_bias_i32  = quantize_bias(fb[b].bias.data(), F, s_dw[b], bq[b].pw_scales);
    }

    // Dense
    int dense_in = MN::kWindowFrames * MN::kFilters;  // 1088
    // Dense weight: [dense_in] flat; TFLite [1, dense_in] — single output channel
    std::vector<float> dense_w_as_row(net.dense_w.begin(), net.dense_w.end());
    float mx_dense = 0.f;
    for (float v : dense_w_as_row) mx_dense = std::max(mx_dense, std::abs(v));
    float s_dense_w = (mx_dense == 0.f) ? 1e-8f : mx_dense / 127.0f;
    std::vector<int8_t> dense_w_i8(dense_in);
    for (int i = 0; i < dense_in; ++i)
        dense_w_i8[i] = (int8_t)std::max(-127.f, std::min(127.f,
            std::round(dense_w_as_row[i] / s_dense_w)));
    // Dense input scale = s_pw[3] (block 3 PW+ReLU output)
    float s_dense_b = s_pw[3] * s_dense_w;
    int32_t dense_b_i32 = (int32_t)std::round(net.dense_b / s_dense_b);

    // ── 4. UnPack template ────────────────────────────────────────────────
    auto modelT = tflite::UnPackModel(tmpl);
    if (!modelT || modelT->subgraphs.empty()) return nullptr;

    tflite::SubGraphT& sg = *modelT->subgraphs[0];
    if ((int)sg.tensors.size() < 94) return nullptr;

    // ── 5. Patch activation quantization params ───────────────────────────

    // Input group (fixed)
    set_quant(sg, {0, 50, 53, 54, 55}, s_input, zp_input);
    // First conv ReLU
    set_quant(sg, {56, 57, 58, 61}, s_fc_relu, zp_fc_relu);
    // Block 0
    set_quant(sg, {59}, s_dw[0], zp_dw[0]);
    set_quant(sg, {60, 62, 63, 64, 65, 66, 68, 70}, s_pw[0], zp_pw[0]);
    // Block 1
    set_quant(sg, {69, 71, 72}, s_dw[1], zp_dw[1]);
    set_quant(sg, {67, 73, 74, 75, 76, 77, 78, 80}, s_pw[1], zp_pw[1]);
    // Block 2
    set_quant(sg, {79, 81, 82}, s_dw[2], zp_dw[2]);
    set_quant(sg, {83, 51, 84, 92}, s_pw[2], zp_pw[2]);
    // Block 3
    set_quant(sg, {85}, s_dw[3], zp_dw[3]);
    set_quant(sg, {86, 52, 87, 88, 91}, s_pw[3], zp_pw[3]);
    // Dense + sigmoid
    set_quant(sg, {89}, s_dense, zp_dense);
    set_quant(sg, {90}, s_sigmoid, zp_sigmoid);
    // Final uint8 output
    {
        auto& t = sg.tensors[93];
        if (!t->quantization) t->quantization = std::make_unique<tflite::QuantizationParametersT>();
        t->quantization->scale      = {s_out_u8};
        t->quantization->zero_point = {(int64_t)zp_out_u8};
    }

    // ── 6. Patch weight quantization params ───────────────────────────────

    // tensor 43: first conv kernel [32,5,1,40], qd=0 (per output channel)
    set_per_channel_quant(sg, 43, s_fc_w, 0);
    // tensor 42: first conv bias [32], qd=0
    {
        std::vector<float> fc_bias_scales(FC_O);
        for (int c = 0; c < FC_O; ++c) fc_bias_scales[c] = s_input * s_fc_w[c];
        set_bias_quant(sg, 42, 1.0f, fc_bias_scales);  // already s_input*s_w[c]
    }

    // Helper: blocks stored in TFLite at tensors (41,40,39,38), (37,36,35,34,33,32), (31,30,29,28,27,26), (25,24,23,22)
    // Using the mapping derived from the inspection:
    // blk0:  dw=[41,40], pw=[39,38]
    // blk1:  dw0=[37,36], dw1=[35,34], pw=[33,32]
    // blk2:  dw0=[31,30], dw1=[29,28], pw=[27,26]
    // blk3:  dw=[25,24], pw=[23,22]
    struct BlkTensors {
        int dw_kernel[2], dw_bias[2];  // up to 2 groups
        int ng;
        int pw_kernel, pw_bias;
    } bt[4] = {
        { {41,-1}, {40,-1}, 1, 39, 38 },
        { {37,35}, {36,34}, 2, 33, 32 },
        { {31,29}, {30,28}, 2, 27, 26 },
        { {25,-1}, {24,-1}, 1, 23, 22 },
    };

    for (int b = 0; b < 4; ++b) {
        const auto& blk = net.blocks[b];
        int C_in = blk.C_in;
        int C_g  = C_in / bt[b].ng;

        for (int g = 0; g < bt[b].ng; ++g) {
            int ks = blk.dw_ks[g];
            int kt = bt[b].dw_kernel[g];
            int kb = bt[b].dw_bias[g];
            // DW kernel: quantized_dimension=3 (channel dim, TFLite [1,K,1,C_g])
            set_per_channel_quant(sg, kt, bq[b].dw_scales[g], 3);
            // DW bias: scale[c] = s_dw_input[b] * s_dw_w[c]
            set_bias_quant(sg, kb, s_dw_input[b], bq[b].dw_scales[g]);
            (void)ks;
        }
        // PW kernel: quantized_dimension=0 (output channel)
        set_per_channel_quant(sg, bt[b].pw_kernel, bq[b].pw_scales, 0);
        // PW bias: scale = s_dw[b] * s_pw_w[c]
        set_bias_quant(sg, bt[b].pw_bias, s_dw[b], bq[b].pw_scales);
    }

    // Dense weight: tensor 21, qd=0 (single output channel)
    set_per_channel_quant(sg, 21, {s_dense_w}, 0);
    // Dense bias: tensor 20, scale = s_pw[3] * s_dense_w
    set_bias_quant(sg, 20, s_pw[3], {s_dense_w});

    // ── 7. Patch weight buffer data ───────────────────────────────────────

    // First conv
    set_buffer_i8 (*modelT, 43, fc_kernel_i8);
    set_buffer_i32(*modelT, 42, fc_bias_i32);

    // Blocks
    for (int b = 0; b < 4; ++b) {
        for (int g = 0; g < bt[b].ng; ++g) {
            set_buffer_i8 (*modelT, bt[b].dw_kernel[g], bq[b].dw_kernel_i8[g]);
            set_buffer_i32(*modelT, bt[b].dw_bias[g],   bq[b].dw_bias_i32[g]);
        }
        set_buffer_i8 (*modelT, bt[b].pw_kernel, bq[b].pw_kernel_i8);
        set_buffer_i32(*modelT, bt[b].pw_bias,   bq[b].pw_bias_i32);
    }

    // Dense
    set_buffer_i8 (*modelT, 21, dense_w_i8);
    {
        std::vector<int32_t> db = {dense_b_i32};
        set_buffer_i32(*modelT, 20, db);
    }

    // ── 8. Repack to FlatBuffer ───────────────────────────────────────────
    flatbuffers::FlatBufferBuilder fbb(tmpl_size + 65536);
    auto model_offset = tflite::Model::Pack(fbb, modelT.get());
    tflite::FinishModelBuffer(fbb, model_offset);

    size_t sz = fbb.GetSize();
    uint8_t* out = (uint8_t*)malloc(sz);
    if (!out) return nullptr;
    memcpy(out, fbb.GetBufferPointer(), sz);
    *out_size = sz;
    return out;
}
