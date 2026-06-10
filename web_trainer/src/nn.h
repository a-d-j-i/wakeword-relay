#pragma once
#include <cmath>

// Valid-padding output length for 1D conv
inline int valid_out(int T_in, int kT, int stride = 1) {
    return (T_in - kT) / stride + 1;
}

// in[T_in, C_in] × kernel[kT, C_in, C_out] → out[T_out, C_out]
// T_out = valid_out(T_in, kT, stride)
void conv1d_valid(const float* in, int T_in, int C_in,
                  float* out, int C_out,
                  const float* kernel, int kT, int stride = 1);

// in[T_in, C] × kernel[kT, C] → out[T_out, C]  (depthwise)
void depthwise_conv1d_valid(const float* in, int T_in, int C,
                            float* out, const float* kernel, int kT);

// in[T, C_in] × kernel[C_in, C_out] → out[T, C_out]  (pointwise / 1×1)
void pointwise_conv(const float* in, int T, int C_in,
                    float* out, int C_out, const float* kernel);

// y = gamma*(x-mean)/sqrt(var+eps)+beta, in-place on x[T, C]
void batchnorm_infer(float* x, int T, int C,
                     const float* gamma, const float* beta,
                     const float* mean, const float* var,
                     float eps = 1e-3f);

// in-place ReLU on n elements
void relu_inplace(float* x, int n);

// Global average pool: in[T, C] → out[C]
void avg_pool_global(const float* in, int T, int C, float* out);

inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// ── Backward primitives ────────────────────────────────────────────────────

// Grad w.r.t. input of conv1d_valid (additive into din, caller zeros first)
void conv1d_grad_input(const float* dout, int T_out, int C_out,
                       const float* kernel, int kT, int C_in, int stride,
                       float* din);

// Grad w.r.t. kernel of conv1d_valid (additive, caller zeros first)
void conv1d_grad_kernel(const float* dout, int T_out, int C_out,
                        const float* in, int C_in,
                        int kT, int stride, float* dkernel);

// Grad w.r.t. input of depthwise_conv1d_valid (additive, caller zeros first)
// din has size (T_out + kT - 1) * C
void dw_conv1d_grad_input(const float* dout, int T_out, int C,
                          const float* kernel, int kT, float* din);

// Grad w.r.t. kernel of depthwise_conv1d_valid (additive, caller zeros first)
void dw_conv1d_grad_kernel(const float* dout, int T_out, int C,
                           const float* in, int kT, float* dkernel);

// Grad w.r.t. input of pointwise_conv (additive, caller zeros first)
void pointwise_grad_input(const float* dout, int T, int C_out,
                          const float* kernel, int C_in, float* din);

// Grad w.r.t. kernel of pointwise_conv (additive, caller zeros first)
void pointwise_grad_kernel(const float* dout, int T, int C_out,
                           const float* in, int C_in, float* dkernel);

// Grad through batchnorm_infer.
// d_gamma and d_beta are additive; d_in is written (not additive).
void batchnorm_grad(const float* d_out, const float* pw_raw,
                    int T, int C,
                    const float* gamma, const float* mean, const float* var,
                    float eps, float* d_in, float* d_gamma, float* d_beta);

// Zero gradient where relu output is <= 0 (in-place on d, using relu_out as mask)
void relu_grad_inplace(float* d, const float* relu_out, int n);

// Distribute pooling gradient uniformly (additive into d_in)
void avg_pool_global_grad(const float* d_pool, int T, int C, float* d_in);

// Binary cross-entropy: -y*log(p+eps) - (1-y)*log(1-p+eps)
inline float bce_loss(float prob, float label) {
    const float eps = 1e-7f;
    return -(label * logf(prob + eps) + (1.0f - label) * logf(1.0f - prob + eps));
}
