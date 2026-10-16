
#include "model.h"

#include <stdio.h>
#include <string>

namespace llm {

void load_weights(const GGUF& g, HParams& hparams, ModelWeights& w) {
    hparams.arch = g.get_string("general.architecture");
    const std::string& arch = hparams.arch;

    w.token_embd = g.find_tensor("token_embd.weight");
    w.output     = g.find_tensor("output.weight");
    LLM_ASSERT(w.token_embd);
    if (!w.output) w.output = w.token_embd;

    hparams.n_vocab    = (int32_t)w.token_embd->ne1;
    hparams.n_embd     = (int32_t)w.token_embd->ne0;
    hparams.n_layer    = (int32_t)g.get_int(arch + ".block_count", 0);
    hparams.n_head     = (int32_t)g.get_int(arch + ".attention.head_count", 0);
    hparams.n_head_kv  = (int32_t)g.get_int(arch + ".attention.head_count_kv", hparams.n_head);
    hparams.head_dim   = (int32_t)g.get_int(arch + ".attention.key_length", hparams.n_embd / hparams.n_head);
    hparams.n_rot      = (int32_t)g.get_int(arch + ".rope.dimension_count", hparams.head_dim);
    hparams.rope_theta = (float)g.get_float(arch + ".rope.freq_base", 10000.0);
    hparams.rms_eps    = (float)g.get_float(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
    hparams.n_ctx      = (int32_t)g.get_int(arch + ".context_length", 0);

    // RoPE type: NeoX (interleaved halves) vs normal (consecutive pairs)
    if (arch == "qwen2" || arch == "qwen3") {
        hparams.rope_type = 1;  // neox
    } else {
        hparams.rope_type = 0;  // normal
    }

    // RoPE frequency factors
    const Tensor* rope_freqs = g.find_tensor("rope_freqs.weight");
    if (rope_freqs && rope_freqs->ne0 == hparams.n_rot / 2) {
        LLM_ASSERT(rope_freqs->type == DType::F32);
        hparams.rope_freq_factors = (const float*)rope_freqs->data;
        printf("  rope_freq_factors loaded from tensor (%d values)\n", hparams.n_rot / 2);
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
        layer.attn_norm = g.find_tensor(prefix + "attn_norm.weight");
        layer.ffn_norm  = g.find_tensor(prefix + "ffn_norm.weight");
        layer.wq_norm   = g.find_tensor(prefix + "attn_q_norm.weight");
        layer.wk_norm   = g.find_tensor(prefix + "attn_k_norm.weight");
        layer.bq = g.find_tensor(prefix + "attn_q.bias");
        layer.bk = g.find_tensor(prefix + "attn_k.bias");
        layer.bv = g.find_tensor(prefix + "attn_v.bias");

        if (!layer.wq || !layer.wk || !layer.wv || !layer.wo ||
            !layer.ffn_gate || !layer.ffn_up || !layer.ffn_down ||
            !layer.attn_norm || !layer.ffn_norm) {
            LLM_ABORT("model: missing tensors for block %d", i);
        }
    }

    w.output_norm = g.find_tensor("output_norm.weight");
    LLM_ASSERT(w.output_norm);

    hparams.n_ff = (int32_t)w.layers[0].ffn_gate->ne1;
}

} // namespace llm
