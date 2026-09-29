#include "model.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace llm {

void load_weights(const GGUF& g, HParams& hparams, ModelWeights& w) {
    hparams.arch = g.get_string("general.architecture");
    const std::string& arch = hparams.arch;

    w.token_embd = g.find_tensor("token_embd.weight");
    w.output     = g.find_tensor("output.weight");
    LLM_ASSERT(w.token_embd);
    if (!w.output) w.output = w.token_embd;

    hparams.n_vocab   = (int32_t) w.token_embd->ne1;
    hparams.n_embd    = (int32_t) w.token_embd->ne0;
    hparams.n_layer   = (int32_t) g.get_int(arch + ".block_count", 0);
    hparams.n_head    = (int32_t) g.get_int(arch + ".attention.head_count", 0);
    hparams.n_head_kv = (int32_t) g.get_int(arch + ".attention.head_count_kv", hparams.n_head);
    hparams.head_dim  = (int32_t) g.get_int(arch + ".attention.key_length", hparams.n_embd / hparams.n_head);
    hparams.n_rot     = (int32_t) g.get_int(arch + ".rope.dimension_count", hparams.head_dim);
    hparams.rope_theta = (float) g.get_float(arch + ".rope.freq_base", 10000.0);
    hparams.rms_eps   = (float) g.get_float(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
    hparams.n_ctx     = (int32_t) g.get_int(arch + ".context_length", 0);

    // RoPE type: NeoX (interleaved halves) vs normal (consecutive pairs)
    if (arch == "qwen2" || arch == "qwen3" || arch == "falcon" ||
        arch == "phi2" || arch == "phi3" || arch == "stablelm" ||
        arch == "gptneox" || arch == "starcoder2") {
        hparams.rope_type = 1;  // neox
    } else {
        hparams.rope_type = 0;  // normal
    }

    // RoPE frequency factors
    const Tensor* rope_freqs = g.find_tensor("rope_freqs.weight");
    if (!rope_freqs) rope_freqs = g.find_tensor("blk.0.rope_freqs.weight");
    if (rope_freqs && rope_freqs->ne0 == (uint64_t)(hparams.n_rot / 2)) {
        if (rope_freqs->type != DType::F32) {
            fprintf(stderr, "  rope_freqs.weight type=%d (expected F32=0)\n", (int)rope_freqs->type);
        }
        const float* ff = (const float*) rope_freqs->data;
        hparams.rope_freq_factors.assign(ff, ff + hparams.n_rot / 2);
        fprintf(stderr, "  rope_freq_factors loaded from tensor (%d values)\n", hparams.n_rot / 2);
    } else {
        const std::string scaling_type = g.find_meta(arch + ".rope.scaling.type")
            ? g.get_string(arch + ".rope.scaling.type") : "";
        if (scaling_type == "llama3" || scaling_type == "linear") {
            const float factor   = (float) g.get_float(arch + ".rope.scaling.factor", 1.0);
            const float lo_freq  = (float) g.get_float(arch + ".rope.scaling.low_freq_factor", 1.0);
            const float hi_freq  = (float) g.get_float(arch + ".rope.scaling.high_freq_factor", 4.0);
            const int64_t orig_ctx_i = g.get_int(arch + ".rope.scaling.original_context_length", 0);
            const float orig_ctx = orig_ctx_i > 0 ? (float) orig_ctx_i
                                 : (float) g.get_float(arch + ".rope.scaling.original_context_length", 8192.0);
            const int half = hparams.n_rot / 2;
            hparams.rope_freq_factors.resize(half);
            for (int i = 0; i < half; ++i) {
                float freq = std::pow(hparams.rope_theta, -2.0f * (float) i / (float) hparams.n_rot);
                float wavelen = 2.0f * 3.14159265f / freq;
                float lo_wavelen = orig_ctx / lo_freq;
                float hi_wavelen = orig_ctx / hi_freq;
                if (wavelen < hi_wavelen) {
                    hparams.rope_freq_factors[i] = 1.0f;
                } else if (wavelen > lo_wavelen) {
                    hparams.rope_freq_factors[i] = factor;
                } else {
                    float smooth = (orig_ctx / wavelen - lo_freq) / (hi_freq - lo_freq);
                    hparams.rope_freq_factors[i] = 1.0f / ((1.0f - smooth) / factor + smooth);
                }
            }
            fprintf(stderr, "  rope_scaling = %s (factor=%.1f)\n", scaling_type.c_str(), factor);
        }
    }

    LLM_ASSERT(hparams.n_layer > 0 && hparams.n_head > 0);
    LLM_ASSERT(hparams.n_head % hparams.n_head_kv == 0);

    w.layers.assign(hparams.n_layer, Layer{});
    for (int i = 0; i < hparams.n_layer; ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        Layer& layer = w.layers[i];
        layer.wq       = g.find_tensor(prefix + "attn_q.weight");
        layer.wk       = g.find_tensor(prefix + "attn_k.weight");
        layer.wv       = g.find_tensor(prefix + "attn_v.weight");
        layer.wo       = g.find_tensor(prefix + "attn_output.weight");
        layer.ffn_gate = g.find_tensor(prefix + "ffn_gate.weight");
        layer.ffn_up   = g.find_tensor(prefix + "ffn_up.weight");
        layer.ffn_down = g.find_tensor(prefix + "ffn_down.weight");

        const Tensor* attn_norm = g.find_tensor(prefix + "attn_norm.weight");
        const Tensor* ffn_norm  = g.find_tensor(prefix + "ffn_norm.weight");
        const Tensor* q_norm    = g.find_tensor(prefix + "attn_q_norm.weight");
        const Tensor* k_norm    = g.find_tensor(prefix + "attn_k_norm.weight");

        if (!layer.wq || !layer.wk || !layer.wv || !layer.wo ||
            !layer.ffn_gate || !layer.ffn_up || !layer.ffn_down ||
            !attn_norm || !ffn_norm) {
            LLM_ABORT("model: missing tensors for block %d", i);
        }

        layer.attn_norm = (const float*) attn_norm->data;
        layer.ffn_norm  = (const float*) ffn_norm->data;
        if (q_norm) layer.wq_norm = (const float*) q_norm->data;
        if (k_norm) layer.wk_norm = (const float*) k_norm->data;

        const Tensor* bq = g.find_tensor(prefix + "attn_q.bias");
        const Tensor* bk = g.find_tensor(prefix + "attn_k.bias");
        const Tensor* bv = g.find_tensor(prefix + "attn_v.bias");
        if (bq) layer.bq = (const float*) bq->data;
        if (bk) layer.bk = (const float*) bk->data;
        if (bv) layer.bv = (const float*) bv->data;
    }

    const Tensor* out_norm = g.find_tensor("output_norm.weight");
    LLM_ASSERT(out_norm);
    w.output_norm = (const float*) out_norm->data;

    hparams.n_ff = (int32_t) w.layers[0].ffn_gate->ne1;
}

} // namespace llm
