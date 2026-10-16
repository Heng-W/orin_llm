// LLM inference engine for Jetson Orin
//
// Supported models (GGUF, decoder-only Transformer + SwiGLU + RoPE, MoE model is not currently supported):
//   Model family        GGUF source (HuggingFace)
//   ----------------------------------------------------------
//   Qwen2 / 2.5        Qwen/Qwen2.5-*-Instruct-GGUF
//   Qwen2.5-Coder      Qwen/Qwen2.5-Coder-*-Instruct-GGUF
//   Llama 3 / 3.1      bartowski/Meta-Llama-3*-Instruct-GGUF
//   Llama 3.2          bartowski/Llama-3.2-*-Instruct-GGUF
//   Llama 2            TheBloke/Llama-2-*-Chat-GGUF
//
// Supported quants: Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, F16, F32
//   Not supported: Q2_K, Q3_K_*, Q4_0, Q4_1, Q5_0, Q5_1, IQ series
//

#pragma once

#include <stdint.h>
#include <string>
#include <memory>
#include <functional>

namespace llm {

// return false to stop generation early
using on_new_token_t = std::function<bool(const std::string& text)>;

struct EngineParams {
    int  n_ctx      = 0;   // 0 = use model metadata
    int  n_predict  = -1;  // -1 = unlimited (stop on EOS only)
    int  n_batch    = 512; // chunked prefill size

    float  temp  = 0.8f;
    int    top_k = 40;
    float  top_p = 0.95f;

    const char* system_prompt = nullptr;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void load(const std::string& model_path, const EngineParams& params = {});
    void print_info() const;

    int send(const std::string& prompt);
    int generate(on_new_token_t on_new_token = nullptr);

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

} // namespace llm
