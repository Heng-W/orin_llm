#include "model.h"

#include <cmath>
#include <cstring>
#include "quant.h"

namespace llm {
namespace {

float silu(float x) { return x / (1.0f + std::exp(-x)); }

void rms_norm(const float* x, const float* w, int n, float eps, float* y) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    float r = 1.0f / std::sqrt((float) (ss / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
}

void softmax(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, x[i]);
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

void dequantize_row_q4_K(const void* vx, float* y, int64_t k) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* q   = x[i].qs;
        const float     d   = fp16_to_fp32(x[i].d);
        const float     min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        for (int j = 0; j < (int) QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

void dequantize_row_q5_K(const void* vx, float* y, int64_t k) {
    const block_q5_K* x = (const block_q5_K*) vx;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* ql  = x[i].qs;
        const uint8_t* qh  = x[i].qh;
        const float     d   = fp16_to_fp32(x[i].d);
        const float     min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < (int) QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row_q6_K(const void* vx, float* y, int64_t k) {
    const block_q6_K* x = (const block_q6_K*) vx;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        for (int n = 0; n < (int) QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dequantize_row_q8_0(const void* vx, float* y, int64_t k) {
    const block_q8_0* x = (const block_q8_0*) vx;
    const int64_t nb = k / 32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int l = 0; l < 32; ++l) y[l] = d * x[i].qs[l];
        y += 32;
    }
}

void dequantize_row(DType type, const void* x, float* y, int64_t k) {
    switch (type) {
        case DType::F32:  memcpy(y, x, sizeof(float) * k); break;
        case DType::F16: {
            const uint16_t* p = (const uint16_t*) x;
            for (int64_t i = 0; i < k; ++i) y[i] = fp16_to_fp32(p[i]);
            break;
        }
        case DType::Q8_0: dequantize_row_q8_0(x, y, k); break;
        case DType::Q4_K: dequantize_row_q4_K(x, y, k); break;
        case DType::Q5_K: dequantize_row_q5_K(x, y, k); break;
        case DType::Q6_K: dequantize_row_q6_K(x, y, k); break;
        default: LLM_ABORT("dequantize_row: unsupported dtype");
    }
}

void gemv(const Tensor & W, const float* x, float* y, std::vector<float> & row) {
    const uint64_t in = W.ne0, out = W.ne1;
    const uint64_t row_bytes = tensor_nbytes(W.type, in, 1);
    if (row.size() < in) row.resize(in);
    for (uint64_t j = 0; j < out; ++j) {
        dequantize_row(W.type, W.data + j * row_bytes, row.data(), (int64_t) in);
        y[j] = dot(row.data(), x, (int) in);
    }
}

void rope_neox(float* x, int n_rot, float theta, int pos, const float* freq_factors = nullptr) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        float freq = std::pow(theta, -2.0f * (float) i / (float) n_rot);
        if (freq_factors) freq /= freq_factors[i];
        float ang = (float) pos * freq;
        float c = std::cos(ang), s = std::sin(ang);
        float x0 = x[i], x1 = x[i + half];
        x[i]        = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

// Normal RoPE: pairs (0,1), (2,3), (4,5), ... used by Llama/Mistral/Yi
void rope_normal(float* x, int n_rot, float theta, int pos, const float* freq_factors = nullptr) {
    for (int i = 0; i < n_rot; i += 2) {
        float freq = std::pow(theta, -(float) i / (float) n_rot);
        if (freq_factors) freq /= freq_factors[i / 2];
        float ang = (float) pos * freq;
        float c = std::cos(ang), s = std::sin(ang);
        float x0 = x[i], x1 = x[i + 1];
        x[i]     = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

} // namespace

struct Model::impl {
    ModelWeights w;

    std::vector<float> k_cache_, v_cache_;
    std::vector<float> x_, xn_, q_, k_, v_, attn_, o_, xn2_, gate_, up_, act_;
    std::vector<float> score_, row_, logits_;

    void load(const GGUF& g, HParams& hparams);
    void init_buffers(HParams& hparams);
    const float* forward(HParams& hparams, int32_t token, int n_past);
    const float* forward_batch(HParams& hparams, const int32_t* tokens, int batch, int n_past);
    void kv_shift(HParams& hparams, int n_keep, int shift);
};

void Model::impl::load(const GGUF& g, HParams& hparams) {
    load_weights(g, hparams, w);
}

void Model::impl::init_buffers(HParams& hparams) {
    const int n_embd = hparams.n_embd, n_kv = hparams.n_head_kv, d = hparams.head_dim;
    const int per_layer = n_kv * hparams.n_ctx * d;

    k_cache_.assign((size_t) hparams.n_layer * per_layer, 0.0f);
    v_cache_.assign((size_t) hparams.n_layer * per_layer, 0.0f);

    x_.assign(n_embd, 0); xn_.assign(n_embd, 0);
    q_.assign(n_embd, 0); k_.assign(n_kv * d, 0); v_.assign(n_kv * d, 0);
    attn_.assign(n_embd, 0); o_.assign(n_embd, 0); xn2_.assign(n_embd, 0);
    gate_.assign(hparams.n_ff, 0); up_.assign(hparams.n_ff, 0); act_.assign(hparams.n_ff, 0);
    score_.assign(hparams.n_ctx, 0);
    row_.reserve((size_t) std::max(n_embd, hparams.n_ff));
}

const float* Model::impl::forward(HParams& hparams, int32_t token, int n_past) {
    const int n_embd = hparams.n_embd, n_head = hparams.n_head, n_kv = hparams.n_head_kv;
    const int d = hparams.head_dim, n_ctx = hparams.n_ctx, n_ff = hparams.n_ff;
    const int per_layer = n_kv * n_ctx * d;
    const float scale = 1.0f / std::sqrt((float) d);

    const uint64_t emb_row_bytes = tensor_nbytes(w.token_embd->type, n_embd, 1);
    dequantize_row(w.token_embd->type, w.token_embd->data + (uint64_t) token * emb_row_bytes,
                   x_.data(), n_embd);

    for (int l = 0; l < hparams.n_layer; ++l) {
        const Layer& layer = w.layers[l];
        float* kbase = k_cache_.data() + (size_t) l * per_layer;
        float* vbase = v_cache_.data() + (size_t) l * per_layer;

        rms_norm(x_.data(), layer.attn_norm, n_embd, hparams.rms_eps, xn_.data());
        gemv(*layer.wq, xn_.data(), q_.data(), row_);
        gemv(*layer.wk, xn_.data(), k_.data(), row_);
        gemv(*layer.wv, xn_.data(), v_.data(), row_);

        if (layer.bq) for (int i = 0; i < n_head * d; ++i) q_[i] += layer.bq[i];
        if (layer.bk) for (int i = 0; i < n_kv * d; ++i) k_[i] += layer.bk[i];
        if (layer.bv) for (int i = 0; i < n_kv * d; ++i) v_[i] += layer.bv[i];

        const float* ff = hparams.rope_freq_factors.empty() ? nullptr : hparams.rope_freq_factors.data();
        auto rope = hparams.rope_type == 1 ? rope_neox : rope_normal;
        for (int h = 0; h < n_head; ++h) {
            float* qh = q_.data() + h * d;
            if (layer.wq_norm) rms_norm(qh, layer.wq_norm, d, hparams.rms_eps, qh);
            rope(qh, hparams.n_rot, hparams.rope_theta, n_past, ff);
        }
        for (int h = 0; h < n_kv; ++h) {
            float* kh = k_.data() + h * d;
            if (layer.wk_norm) rms_norm(kh, layer.wk_norm, d, hparams.rms_eps, kh);
            rope(kh, hparams.n_rot, hparams.rope_theta, n_past, ff);
        }
        for (int h = 0; h < n_kv; ++h) {
            std::memcpy(kbase + (h * n_ctx + n_past) * d, k_.data() + h * d, d * sizeof(float));
            std::memcpy(vbase + (h * n_ctx + n_past) * d, v_.data() + h * d, d * sizeof(float));
        }

        const int npos = n_past + 1;
        for (int h = 0; h < n_head; ++h) {
            const int kv = h * n_kv / n_head;
            float* qh = q_.data() + h * d;
            for (int j = 0; j < npos; ++j) {
                score_[j] = scale * dot(qh, kbase + (kv * n_ctx + j) * d, d);
            }
            softmax(score_.data(), npos);
            float* ah = attn_.data() + h * d;
            for (int jd = 0; jd < d; ++jd) {
                float s = 0.0f;
                for (int j = 0; j < npos; ++j) s += score_[j] * vbase[(kv * n_ctx + j) * d + jd];
                ah[jd] = s;
            }
        }
        gemv(*layer.wo, attn_.data(), o_.data(), row_);
        for (int i = 0; i < n_embd; ++i) x_[i] += o_[i];

        rms_norm(x_.data(), layer.ffn_norm, n_embd, hparams.rms_eps, xn2_.data());
        gemv(*layer.ffn_gate, xn2_.data(), gate_.data(), row_);
        gemv(*layer.ffn_up,   xn2_.data(), up_.data(),   row_);
        for (int i = 0; i < n_ff; ++i) act_[i] = silu(gate_[i]) * up_[i];
        gemv(*layer.ffn_down, act_.data(), o_.data(), row_);
        for (int i = 0; i < n_embd; ++i) x_[i] += o_[i];
    }

    rms_norm(x_.data(), w.output_norm, n_embd, hparams.rms_eps, xn_.data());
    logits_.resize(hparams.n_vocab);
    gemv(*w.output, xn_.data(), logits_.data(), row_);

    return logits_.data();
}

const float* Model::impl::forward_batch(HParams& hparams, const int32_t* tokens, int batch, int n_past) {
    for (int b = 0; b < batch; ++b) {
        forward(hparams, tokens[b], n_past + b);
    }
    return logits_.data();
}

void Model::impl::kv_shift(HParams& hparams, int n_keep, int shift) {
    const int n_kv = hparams.n_head_kv, d = hparams.head_dim, n_ctx = hparams.n_ctx;
    const size_t per_layer = (size_t) n_kv * n_ctx * d;
    for (int l = 0; l < hparams.n_layer; ++l) {
        float* kbase = k_cache_.data() + (size_t) l * per_layer;
        float* vbase = v_cache_.data() + (size_t) l * per_layer;
        for (int h = 0; h < n_kv; ++h) {
            float* kh = kbase + (size_t) h * n_ctx * d;
            float* vh = vbase + (size_t) h * n_ctx * d;
            std::memmove(kh + (size_t) n_keep * d, kh + (size_t) (n_keep + shift) * d,
                         (size_t) (n_ctx - n_keep - shift) * d * sizeof(float));
            std::memmove(vh + (size_t) n_keep * d, vh + (size_t) (n_keep + shift) * d,
                         (size_t) (n_ctx - n_keep - shift) * d * sizeof(float));
            std::memset(kh + (size_t) (n_ctx - shift) * d, 0, (size_t) shift * d * sizeof(float));
            std::memset(vh + (size_t) (n_ctx - shift) * d, 0, (size_t) shift * d * sizeof(float));
        }
    }
}

// --- Model forwarding (pimpl) ---

Model::Model() : pimpl_(std::make_unique<impl>()) {}
Model::~Model() = default;

void Model::load(const GGUF& g, int /*weight_mode*/) { pimpl_->load(g, hparams); }

void Model::init_buffers(int n_ctx, int max_batch) { (void)max_batch; hparams.n_ctx = n_ctx; pimpl_->init_buffers(hparams); }

const float* Model::forward(int32_t token, int n_past) { return pimpl_->forward(hparams, token, n_past); }

const float* Model::forward_batch(const int32_t* tokens, int batch, int n_past) { return pimpl_->forward_batch(hparams, tokens, batch, n_past); }

void Model::kv_shift(int n_keep, int shift) { pimpl_->kv_shift(hparams, n_keep, shift); }

} // namespace llm
