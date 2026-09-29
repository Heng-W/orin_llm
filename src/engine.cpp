#include "orin_llm/engine.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include "common.h"
#include "gguf.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
namespace llm {

// ---- Chat template ----
// Auto-selects format based on special tokens in the vocabulary.

enum class ChatFormat {
    CHATML,     // Qwen2/2.5
    LLAMA3,     // Llama 3 / 3.1 / 3.2
    LLAMA2,     // Llama 2
    PLAIN,      // no template (plain text fallback)
};

struct ChatTemplate {
    ChatFormat fmt = ChatFormat::PLAIN;

    // special token IDs (-1 = not present)
    int32_t im_start     = -1;  // <|im_start|>
    int32_t im_end       = -1;  // <|im_end|>
    int32_t header_start = -1;  // <|start_header_id|>
    int32_t header_end   = -1;  // <|end_header_id|>
    int32_t eot          = -1;  // <|eot_id|>

    // Llama 3/3.1/3.2 require BOS (<|begin_of_text|>) at start
    bool needs_bos() const { return fmt == ChatFormat::LLAMA3 || fmt == ChatFormat::LLAMA2; }

    // detect format from vocabulary
    void detect(const Tokenizer& tok) {
        auto find = [&](const char* s) -> int32_t {
            auto it = tok.id_of.find(s);
            return it != tok.id_of.end() ? it->second : -1;
        };

        im_start     = find("<|im_start|>");
        im_end       = find("<|im_end|>");
        header_start = find("<|start_header_id|>");
        header_end   = find("<|end_header_id|>");
        eot          = find("<|eot_id|>");

        if (im_start >= 0 && im_end >= 0) {
            fmt = ChatFormat::CHATML;
        } else if (header_start >= 0 && header_end >= 0 && eot >= 0) {
            fmt = ChatFormat::LLAMA3;
        } else if (tok.id_of.count("[INST]")) {
            fmt = ChatFormat::LLAMA2;
        } else {
            fmt = ChatFormat::PLAIN;
        }
    }

    // close=true: append end-of-turn token; close=false: leave open for model to continue
    std::vector<int32_t> apply(const Tokenizer& tok,
                               const std::string& role,
                               const std::string& content,
                               bool close) const {
        switch (fmt) {
            case ChatFormat::CHATML:  return apply_chatml(tok, role, content, close);
            case ChatFormat::LLAMA3:  return apply_llama3(tok, role, content, close);
            case ChatFormat::LLAMA2:  return apply_llama2(tok, role, content, close);
            default:                  return apply_plain(tok, content, close);
        }
    }


    int32_t effective_eos(const Tokenizer& tok) const {
        switch (fmt) {
            case ChatFormat::CHATML:  return im_end;
            case ChatFormat::LLAMA3:  return eot;
            default:                  return tok.eos_id;
        }
    }

private:
    // ChatML: <|im_start|>role\ncontent<|im_end|>\n
    std::vector<int32_t> apply_chatml(const Tokenizer& tok,
                                      const std::string& role,
                                      const std::string& content,
                                      bool close) const {
        std::vector<int32_t> ids;
        ids.push_back(im_start);
        auto body = tok.encode(role + "\n" + content, false);
        ids.insert(ids.end(), body.begin(), body.end());
        if (close) {
            ids.push_back(im_end);
            ids.push_back(198);  // "\n"
        }
        return ids;
    }

    // Llama 3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    std::vector<int32_t> apply_llama3(const Tokenizer& tok,
                                      const std::string& role,
                                      const std::string& content,
                                      bool close) const {
        std::vector<int32_t> ids;
        ids.push_back(header_start);
        auto role_ids = tok.encode(role, false);
        ids.insert(ids.end(), role_ids.begin(), role_ids.end());
        ids.push_back(header_end);
        // \n\n
        auto nn = tok.encode("\n\n", false);
        ids.insert(ids.end(), nn.begin(), nn.end());
        if (!content.empty()) {
            auto body = tok.encode(content, false);
            ids.insert(ids.end(), body.begin(), body.end());
        }
        if (close) ids.push_back(eot);
        return ids;
    }

    // Llama 2: [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant</s>
    std::vector<int32_t> apply_llama2(const Tokenizer& tok,
                                      const std::string& role,
                                      const std::string& content,
                                      bool close) const {
        std::vector<int32_t> ids;
        if (role == "system") {
            auto sys = tok.encode("[INST] <<SYS>>\n" + content + "\n<</SYS>>\n\n", false);
            ids.insert(ids.end(), sys.begin(), sys.end());
        } else if (role == "user") {
            auto user = tok.encode(content + " [/INST]", false);
            ids.insert(ids.end(), user.begin(), user.end());
        } else if (role == "assistant") {
            if (!content.empty()) {
                auto asst = tok.encode(content, false);
                ids.insert(ids.end(), asst.begin(), asst.end());
            }
            if (close && tok.eos_id >= 0) {
                ids.push_back(tok.eos_id);
            }
        }
        return ids;
    }

    // Plain fallback
    std::vector<int32_t> apply_plain(const Tokenizer& tok,
                                     const std::string& content,
                                     bool close) const {
        auto ids = tok.encode(content + "\n", false);
        if (close && tok.eos_id >= 0)
            ids.push_back(tok.eos_id);
        return ids;
    }
};

// ---- Engine impl ----

struct Engine::impl {
    EngineParams   params;
    GGUF           gguf;
    Tokenizer      tok;
    Model          model;
    Sampler        sampler;
    ChatTemplate   tmpl;

    int          n_ctx          = 0;
    int          n_past         = 0;
    int          n_keep_clamped = 0;
    const float* logits         = nullptr;

    std::vector<int32_t> sys_ids;  // system prompt tokens (merged on first prefill)

    // chunked batch prefill with KV shift
    void eval(const std::vector<int32_t>& ids, on_prefill_done_t cb) {
        auto t0 = std::chrono::steady_clock::now();
        int total = 0;
        for (size_t off = 0; off < ids.size();) {
            if (n_past >= n_ctx) {
                model.kv_shift(n_keep_clamped, n_past - n_keep_clamped);
                n_past = n_keep_clamped;
            }
            int chunk = (int) std::min<size_t>((size_t)(n_ctx - n_past), ids.size() - off);
            chunk = std::min(chunk, params.n_batch);
            if (chunk <= 0) chunk = 1;
            logits = model.forward_batch(ids.data() + off, chunk, n_past);
            n_past += chunk;
            off += chunk;
            total += chunk;
        }
        if (cb) {
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            cb(total, ms);
        }
    }
};

// ---- Engine public methods ----

Engine::Engine() : pimpl_(std::make_unique<impl>()) {}
Engine::~Engine() = default;

void Engine::load(const std::string& model_path, const EngineParams& params) {
    auto& d = *pimpl_;
    d.params = params;

    load_gguf(d.gguf, model_path);

    if (!d.tok.load(d.gguf)) LLM_ABORT("failed to load tokenizer");

    d.model.load(d.gguf, d.params.weight_mode);

    d.tmpl.detect(d.tok);
    const char* fmt_names[] = {"ChatML", "Llama3", "Llama2", "Plain"};
    std::cerr << "  chat_fmt = " << fmt_names[(int)d.tmpl.fmt] << "\n";

    d.n_ctx = params.n_ctx;
    if (d.n_ctx <= 0) d.n_ctx = std::min(d.model.hparams.n_ctx, 4096);
    if (d.n_ctx <= 0) d.n_ctx = 2048;
    d.model.init_buffers(d.n_ctx, params.n_batch);

    d.sampler.temp  = params.temp;
    d.sampler.top_k = params.top_k;
    d.sampler.top_p = params.top_p;

    uint64_t seed = params.seed;
    if (seed == 0)
        seed = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    d.sampler.seed((uint32_t)(seed ^ (seed >> 32)));

    // tokenize system prompt (deferred until first send())
    if (!d.params.system_prompt.empty()) {
        d.sys_ids = d.tmpl.apply(d.tok, "system", d.params.system_prompt, true);
        d.n_keep_clamped = (d.params.n_keep >= 0)
                            ? std::min(d.params.n_keep, d.n_ctx - 1)
                            : std::min((int)d.sys_ids.size(), d.n_ctx - 1);
    }
}

void Engine::print_info() const {
    const auto& hp = pimpl_->model.hparams;
    std::cerr << "  arch     = " << hp.arch      << "\n"
              << "  layers   = " << hp.n_layer    << "\n"
              << "  embd     = " << hp.n_embd     << "\n"
              << "  heads    = " << hp.n_head      << "\n"
              << "  kv_heads = " << hp.n_head_kv   << "\n"
              << "  ctx      = " << pimpl_->n_ctx  << "\n"
              << "  vocab    = " << hp.n_vocab     << "\n";
}

void Engine::send(const std::string& user_input, on_prefill_done_t cb) {
    auto& d = *pimpl_;

    auto user_ids = d.tmpl.apply(d.tok, "user", user_input, true);
    auto asst_ids = d.tmpl.apply(d.tok, "assistant", "", false);

    std::vector<int32_t> ids;
    ids.reserve(1 + d.sys_ids.size() + user_ids.size() + asst_ids.size());
    // on first send, prepend BOS (if needed) and merge pending system prompt
    if (!d.sys_ids.empty()) {
        if (d.tmpl.needs_bos() && d.tok.bos_id >= 0)
            ids.push_back(d.tok.bos_id);
        ids.insert(ids.end(), d.sys_ids.begin(), d.sys_ids.end());
        d.sys_ids.clear();
    } else if (d.n_past == 0 && d.tmpl.needs_bos() && d.tok.bos_id >= 0) {
        ids.push_back(d.tok.bos_id);
    }
    ids.insert(ids.end(), user_ids.begin(), user_ids.end());
    ids.insert(ids.end(), asst_ids.begin(), asst_ids.end());

    d.eval(ids, cb);
}

int Engine::generate(on_new_token_t on_new_token, on_decode_step_t on_decode_step) {
    auto& d = *pimpl_;
    const int n_vocab = d.model.hparams.n_vocab;
    const int32_t eos = d.tmpl.effective_eos(d.tok);
    int n_gen = 0;

    while (n_gen < d.params.n_predict) {
        bool is_eos = false;
        int32_t next = d.sampler.sample(d.logits, n_vocab, eos, is_eos);
        if (is_eos || next == eos) break;

        std::string text = d.tok.decode(next);
        if (on_new_token && !on_new_token(text))
            break;

        if (d.n_past >= d.n_ctx) {
            d.model.kv_shift(d.n_keep_clamped, d.n_past - d.n_keep_clamped);
            d.n_past = d.n_keep_clamped;
        }

        auto t0 = std::chrono::steady_clock::now();
        d.logits = d.model.forward(next, d.n_past);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();

        ++d.n_past;
        ++n_gen;

        if (on_decode_step) on_decode_step(n_gen, ms);
    }
    return n_gen;
}

} // namespace llm
