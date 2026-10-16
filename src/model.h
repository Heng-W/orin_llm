#pragma once

#include <stdint.h>
#include <cmath>
#include <string>
#include <memory>
#include <vector>

#include "common.h"
#include "gguf.h"

namespace llm {

struct HParams {
    std::string arch;       // architecture (e.g. "qwen2", "llama")
    int32_t n_vocab   = 0;
    int32_t n_embd    = 0;
    int32_t n_layer   = 0;
    int32_t n_head    = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim  = 0;
    int32_t n_ff      = 0;
    int32_t n_rot     = 0;
    int32_t n_ctx     = 0;
    float   rms_eps   = 1e-6f;
    float   rope_theta = 10000.0f;
    // RoPE type: 0 = normal (pairs: 0,1  2,3  ...),  1 = neox (pairs: 0,half  1,half+1  ...)
    int rope_type = 0;
    // RoPE frequency factors, n_rot/2 elements, nullptr = no scaling (all factors = 1).
    const float* rope_freq_factors = nullptr;
};

struct Layer {
    const Tensor* wq       = nullptr;
    const Tensor* wk       = nullptr;
    const Tensor* wv       = nullptr;
    const Tensor* wo       = nullptr;
    const Tensor* ffn_gate = nullptr;
    const Tensor* ffn_up   = nullptr;
    const Tensor* ffn_down = nullptr;

    const Tensor* attn_norm = nullptr;  // [n_embd]
    const Tensor* ffn_norm  = nullptr;  // [n_embd]
    const Tensor* wq_norm   = nullptr;  // [head_dim] (Qwen2 QKNorm, optional)
    const Tensor* wk_norm   = nullptr;  // [head_dim]
    const Tensor* bq = nullptr;         // [n_head * head_dim] Q bias (optional)
    const Tensor* bk = nullptr;         // [n_head_kv * head_dim] K bias
    const Tensor* bv = nullptr;         // [n_head_kv * head_dim] V bias
};

// Shared weight loading result used by both CPU and GPU backends.
struct ModelWeights {
    const Tensor* token_embd  = nullptr;
    const Tensor* output      = nullptr;
    const Tensor* output_norm = nullptr;
    std::vector<Layer> layers;
};

// Parse hparams from GGUF metadata and load weight pointers.
void load_weights(const GGUF& g, HParams& hparams, ModelWeights& weights);

class Model {
public:
    HParams hparams;

    Model();
    ~Model();
    Model(const Model &) = delete;
    Model& operator=(const Model&) = delete;

    void load(const GGUF& g);
    void init_buffers(int n_ctx, int max_batch = 1);
    void print_memory() const;

    const float* forward(int32_t token, int n_past);
    const float* forward_batch(const int32_t* tokens, int batch, int n_past);
    void kv_shift(int n_keep, int shift);

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

} // namespace llm
