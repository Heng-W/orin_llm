
#include "orin_llm/engine.h"

#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "common.h"
#include "model.h"
#include "gguf.h"
#include "sampler.h"
#include "tokenizer.h"

namespace llm {

class ChatTemplate {
public:
    // detect format from vocabulary
    void detect(const Tokenizer& tok) {
        auto find = [&](const char* s) -> int32_t {
            auto it = tok.id_of.find(s);
            return it != tok.id_of.end() ? it->second : -1;
        };

        im_start_     = find("<|im_start|>");
        im_end_       = find("<|im_end|>");
        header_start_ = find("<|start_header_id|>");
        header_end_   = find("<|end_header_id|>");
        eot_          = find("<|eot_id|>");

        if (im_start_ >= 0 && im_end_ >= 0) {
            fmt_ = ChatFormat::CHATML;
        } else if (header_start_ >= 0 && header_end_ >= 0 && eot_ >= 0) {
            fmt_ = ChatFormat::LLAMA3;
        } else if (tok.id_of.count("[INST]")) {
            fmt_ = ChatFormat::LLAMA2;
        } else {
            fmt_ = ChatFormat::PLAIN;
        }

        nl_ids_   = tok.encode("\n", false);
        nlnl_ids_ = tok.encode("\n\n", false);
    }

    // close=true: append end-of-turn token
    std::vector<int32_t> apply(const Tokenizer& tok, const std::string& role,
                               const std::string& content, bool close) const {
        switch (fmt_) {
            case ChatFormat::CHATML:  return apply_chatml(tok, role, content, close);
            case ChatFormat::LLAMA3:  return apply_llama3(tok, role, content, close);
            case ChatFormat::LLAMA2:  return apply_llama2(tok, role, content, close);
            default:                  return apply_plain(tok, content, close);
        }
    }

    int32_t effective_eos(const Tokenizer& tok) const {
        switch (fmt_) {
            case ChatFormat::CHATML:  return im_end_;
            case ChatFormat::LLAMA3:  return eot_;
            default:                  return tok.eos_id;
        }
    }

    const char* get_fmt_name() const { 
        switch (fmt_) {
            case ChatFormat::CHATML: return "ChatML";
            case ChatFormat::LLAMA3: return "Llama3";
            case ChatFormat::LLAMA2: return "Llama2";
            default: return "Plain";
        }
     }

    // Llama require BOS (<|begin_of_text|>) at start
    bool needs_bos() const { return fmt_ == ChatFormat::LLAMA3 || fmt_ == ChatFormat::LLAMA2; }

private:
    // ChatML: <|im_start|>role\ncontent<|im_end|>\n
    std::vector<int32_t> apply_chatml(const Tokenizer& tok, const std::string& role,
                                      const std::string& content, bool close) const {
        std::vector<int32_t> ids;
        ids.push_back(im_start_);
        auto body = tok.encode(role + "\n" + content, false);
        ids.insert(ids.end(), body.begin(), body.end());
        if (close) {
            ids.push_back(im_end_);
            ids.insert(ids.end(), nl_ids_.begin(), nl_ids_.end());
        }
        return ids;
    }

    // Llama 3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    std::vector<int32_t> apply_llama3(const Tokenizer& tok, const std::string& role,
                                      const std::string& content, bool close) const {
        std::vector<int32_t> ids;
        ids.push_back(header_start_);
        auto role_ids = tok.encode(role, false);
        ids.insert(ids.end(), role_ids.begin(), role_ids.end());
        ids.push_back(header_end_);
        ids.insert(ids.end(), nlnl_ids_.begin(), nlnl_ids_.end());
        if (!content.empty()) {
            auto body = tok.encode(content, false);
            ids.insert(ids.end(), body.begin(), body.end());
        }
        if (close) ids.push_back(eot_);
        return ids;
    }

    // Llama 2: [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant</s>
    std::vector<int32_t> apply_llama2(const Tokenizer& tok, const std::string& role,
                                      const std::string& content, bool close) const {
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
            if (close && tok.eos_id >= 0) ids.push_back(tok.eos_id);
        }
        return ids;
    }

    // no template, plain fallback
    std::vector<int32_t> apply_plain(const Tokenizer& tok, const std::string& content, bool close) const {
        auto ids = tok.encode(content + "\n", false);
        if (close && tok.eos_id >= 0) ids.push_back(tok.eos_id);
        return ids;
    }

    enum class ChatFormat {
        CHATML, LLAMA3, LLAMA2, PLAIN
    };

    ChatFormat fmt_ = ChatFormat::PLAIN;

    // special token IDs
    int32_t im_start_ = -1; // <|im_start|>
    int32_t im_end_ = -1;   // <|im_end|>
    int32_t header_start_ = -1; // <|start_header_id|>
    int32_t header_end_ = -1;   // <|end_header_id|>
    int32_t eot_ = -1;      // <|eot_id|>

    std::vector<int32_t> nl_ids_;    // cached encode("\n")
    std::vector<int32_t> nlnl_ids_;  // cached encode("\n\n")
};

struct Engine::impl {
public:
    void load(const std::string& model_path, const EngineParams& params);
    void print_info() const;

    int send(const std::string& prompt);
    int generate(on_new_token_t on_new_token);
    
private:
    EngineParams   params_;
    GGUF           gguf_;
    Tokenizer      tok_;
    Model          model_;
    Sampler        sampler_;
    ChatTemplate   tmpl_;

    int          n_ctx_          = 0;
    int          n_past_         = 0;
    int          n_keep_clamped_ = 0;
    const float* logits_         = nullptr;

    std::vector<int32_t> sys_ids_;
};

void Engine::impl::load(const std::string& model_path, const EngineParams& params) {
    params_ = params;

    gguf_.load(model_path);

    if (!tok_.load(gguf_)) LLM_ABORT("failed to load tokenizer");

    model_.load(gguf_);

    tmpl_.detect(tok_);
    std::cout << "  chat_fmt = " << tmpl_.get_fmt_name() << std::endl;

    n_ctx_ = params_.n_ctx;
    if (n_ctx_ <= 0) n_ctx_ = model_.hparams.n_ctx;
    model_.init_buffers(n_ctx_, params_.n_batch);

    sampler_.temp  = params_.temp;
    sampler_.top_k = params_.top_k;
    sampler_.top_p = params_.top_p;

    uint64_t seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    sampler_.seed((uint32_t)(seed ^ (seed >> 32)));

    if (params_.system_prompt && *params_.system_prompt) {
        sys_ids_ = tmpl_.apply(tok_, "system", params_.system_prompt, true);
        n_keep_clamped_ = std::min((int)sys_ids_.size(), n_ctx_ - 1);
    }
}

void Engine::impl::print_info() const {
    const auto& hp = model_.hparams;
    std::cout << "  arch     = " << hp.arch      << "\n"
              << "  layers   = " << hp.n_layer   << "\n"
              << "  embd     = " << hp.n_embd    << "\n"
              << "  heads    = " << hp.n_head    << "\n"
              << "  kv_heads = " << hp.n_head_kv << "\n"
              << "  ctx      = " << n_ctx_       << "\n"
              << "  vocab    = " << hp.n_vocab   << std::endl;
    model_.print_memory();
}

int Engine::impl::send(const std::string& prompt) {
    auto user_ids = tmpl_.apply(tok_, "user", prompt, true);
    auto asst_ids = tmpl_.apply(tok_, "assistant", "", false);

    std::vector<int32_t> ids;
    ids.reserve(1 + sys_ids_.size() + user_ids.size() + asst_ids.size());
    if (n_past_ == 0) {
        if (tmpl_.needs_bos() && tok_.bos_id >= 0) ids.push_back(tok_.bos_id);
        ids.insert(ids.end(), sys_ids_.begin(), sys_ids_.end());
        sys_ids_.clear();
    }
    ids.insert(ids.end(), user_ids.begin(), user_ids.end());
    ids.insert(ids.end(), asst_ids.begin(), asst_ids.end());

    int total = 0;
    for (size_t off = 0; off < ids.size();) {
        if (n_past_ >= n_ctx_) {
            int n_discard = (n_past_ - n_keep_clamped_) / 2;
            model_.kv_shift(n_keep_clamped_, n_discard); // shift kv cache
            n_past_ -= n_discard;
        }
        int chunk = std::min(n_ctx_ - n_past_, (int)(ids.size() - off));
        chunk = std::min(chunk, params_.n_batch);
        if (chunk <= 0) chunk = 1;
        logits_ = model_.forward_batch(ids.data() + off, chunk, n_past_);
        n_past_ += chunk;
        off += chunk;
        total += chunk;
    }
    return total;
}

int Engine::impl::generate(on_new_token_t on_new_token) {
    const int n_vocab = model_.hparams.n_vocab;
    const int32_t eos = tmpl_.effective_eos(tok_);
    int n_gen = 0;

    while (params_.n_predict < 0 || n_gen < params_.n_predict) {
        int32_t next = sampler_.sample(logits_, n_vocab);
        if (next == eos) break;

        std::string text = tok_.decode(next);
        if (on_new_token && !on_new_token(text)) break;
        if (n_past_ >= n_ctx_) {
            const int n_discard = (n_past_ - n_keep_clamped_) / 2;
            model_.kv_shift(n_keep_clamped_, n_discard);
            n_past_ -= n_discard;
        }
        logits_ = model_.forward(next, n_past_);
        ++n_past_;
        ++n_gen;
    }
    return n_gen;
}

// ------ forwarding ------

Engine::Engine() : pimpl_(std::make_unique<impl>()) {}
Engine::~Engine() = default;

void Engine::load(const std::string& model_path, const EngineParams& params) { pimpl_->load(model_path, params); }
void Engine::print_info() const { pimpl_->print_info(); }

int Engine::send(const std::string& prompt) { return pimpl_->send(prompt); }
int Engine::generate(on_new_token_t on_new_token) { return pimpl_->generate(on_new_token); }

} // namespace llm
