#include "nn.h"
#include <cstring>

void conv1d_valid(const float* in, int T_in, int C_in,
                  float* out, int C_out,
                  const float* kernel, int kT, int stride) {
    int T_out = valid_out(T_in, kT, stride);
    memset(out, 0, T_out * C_out * sizeof(float));
    for (int t = 0; t < T_out; ++t) {
        float* o = out + t * C_out;
        for (int kt = 0; kt < kT; ++kt) {
            const float* ip = in + (t * stride + kt) * C_in;
            const float* kp = kernel + kt * (C_in * C_out);
            for (int ci = 0; ci < C_in; ++ci) {
                float v = ip[ci];
                for (int co = 0; co < C_out; ++co)
                    o[co] += v * kp[ci * C_out + co];
            }
        }
    }
}

void depthwise_conv1d_valid(const float* in, int T_in, int C,
                            float* out, const float* kernel, int kT) {
    int T_out = valid_out(T_in, kT);
    memset(out, 0, T_out * C * sizeof(float));
    for (int t = 0; t < T_out; ++t) {
        float* o = out + t * C;
        for (int kt = 0; kt < kT; ++kt) {
            const float* ip = in + (t + kt) * C;
            const float* kp = kernel + kt * C;
            for (int c = 0; c < C; ++c)
                o[c] += ip[c] * kp[c];
        }
    }
}

void pointwise_conv(const float* in, int T, int C_in,
                    float* out, int C_out, const float* kernel) {
    memset(out, 0, T * C_out * sizeof(float));
    for (int t = 0; t < T; ++t) {
        const float* ip = in + t * C_in;
        float* o = out + t * C_out;
        for (int ci = 0; ci < C_in; ++ci) {
            float v = ip[ci];
            for (int co = 0; co < C_out; ++co)
                o[co] += v * kernel[ci * C_out + co];
        }
    }
}

void batchnorm_infer(float* x, int T, int C,
                     const float* gamma, const float* beta,
                     const float* mean, const float* var, float eps) {
    for (int c = 0; c < C; ++c) {
        float scale = gamma[c] / sqrtf(var[c] + eps);
        float bias  = beta[c] - mean[c] * scale;
        for (int t = 0; t < T; ++t)
            x[t * C + c] = x[t * C + c] * scale + bias;
    }
}

void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

void avg_pool_global(const float* in, int T, int C, float* out) {
    memset(out, 0, C * sizeof(float));
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c)
            out[c] += in[t * C + c];
    float inv = 1.0f / T;
    for (int c = 0; c < C; ++c)
        out[c] *= inv;
}

// ── Backward primitives ────────────────────────────────────────────────────

void conv1d_grad_input(const float* dout, int T_out, int C_out,
                       const float* kernel, int kT, int C_in, int stride,
                       float* din) {
    for (int t = 0; t < T_out; ++t) {
        const float* do_ = dout + t * C_out;
        for (int kt = 0; kt < kT; ++kt) {
            float* di = din + (t * stride + kt) * C_in;
            const float* kp = kernel + kt * (C_in * C_out);
            for (int ci = 0; ci < C_in; ++ci)
                for (int co = 0; co < C_out; ++co)
                    di[ci] += do_[co] * kp[ci * C_out + co];
        }
    }
}

void conv1d_grad_kernel(const float* dout, int T_out, int C_out,
                        const float* in, int C_in,
                        int kT, int stride, float* dkernel) {
    for (int t = 0; t < T_out; ++t) {
        const float* do_ = dout + t * C_out;
        for (int kt = 0; kt < kT; ++kt) {
            const float* ip = in + (t * stride + kt) * C_in;
            float* dk = dkernel + kt * (C_in * C_out);
            for (int ci = 0; ci < C_in; ++ci) {
                float v = ip[ci];
                for (int co = 0; co < C_out; ++co)
                    dk[ci * C_out + co] += v * do_[co];
            }
        }
    }
}

void dw_conv1d_grad_input(const float* dout, int T_out, int C,
                          const float* kernel, int kT, float* din) {
    for (int t = 0; t < T_out; ++t) {
        const float* do_ = dout + t * C;
        for (int kt = 0; kt < kT; ++kt) {
            float* di = din + (t + kt) * C;
            const float* kp = kernel + kt * C;
            for (int c = 0; c < C; ++c)
                di[c] += do_[c] * kp[c];
        }
    }
}

void dw_conv1d_grad_kernel(const float* dout, int T_out, int C,
                           const float* in, int kT, float* dkernel) {
    for (int t = 0; t < T_out; ++t) {
        const float* do_ = dout + t * C;
        for (int kt = 0; kt < kT; ++kt) {
            const float* ip = in + (t + kt) * C;
            float* dk = dkernel + kt * C;
            for (int c = 0; c < C; ++c)
                dk[c] += do_[c] * ip[c];
        }
    }
}

void pointwise_grad_input(const float* dout, int T, int C_out,
                          const float* kernel, int C_in, float* din) {
    for (int t = 0; t < T; ++t) {
        const float* do_ = dout + t * C_out;
        float* di = din + t * C_in;
        for (int ci = 0; ci < C_in; ++ci)
            for (int co = 0; co < C_out; ++co)
                di[ci] += do_[co] * kernel[ci * C_out + co];
    }
}

void pointwise_grad_kernel(const float* dout, int T, int C_out,
                           const float* in, int C_in, float* dkernel) {
    for (int t = 0; t < T; ++t) {
        const float* do_ = dout + t * C_out;
        const float* ip = in + t * C_in;
        for (int ci = 0; ci < C_in; ++ci) {
            float v = ip[ci];
            for (int co = 0; co < C_out; ++co)
                dkernel[ci * C_out + co] += v * do_[co];
        }
    }
}

void batchnorm_grad(const float* d_out, const float* pw_raw,
                    int T, int C,
                    const float* gamma, const float* mean, const float* var,
                    float eps, float* d_in, float* d_gamma, float* d_beta) {
    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / sqrtf(var[c] + eps);
        float scale   = gamma[c] * inv_std;
        for (int t = 0; t < T; ++t) {
            float g = d_out[t * C + c];
            d_in[t * C + c]  = g * scale;
            d_gamma[c]      += g * (pw_raw[t * C + c] - mean[c]) * inv_std;
            d_beta[c]       += g;
        }
    }
}

void relu_grad_inplace(float* d, const float* relu_out, int n) {
    for (int i = 0; i < n; ++i)
        if (relu_out[i] <= 0.0f) d[i] = 0.0f;
}

void avg_pool_global_grad(const float* d_pool, int T, int C, float* d_in) {
    float inv_T = 1.0f / T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c)
            d_in[t * C + c] += d_pool[c] * inv_T;
}
