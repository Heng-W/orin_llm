
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <chrono>

#include "orin_llm/engine.h"

static void usage(const char* proc) {
    std::cerr << "usage: " << proc << " <model.gguf> [options]\n"
        << "options:\n"
        << "  --temp F       temperature (default 0.8)\n"
        << "  --top-k N      top-k sampling (default 40)\n"
        << "  --top-p F      top-p sampling (default 0.95)\n"
        << "  --ctx N        context length (default: model metadata)\n"
        << "  --n-predict N  max generated tokens (default unlimited, stop on EOS only)\n"
        << "  --batch N      chunked prefill size (default 512)\n"
        << "  -s/--system S  system prompt\n"
        << "  -p/--prompt S  prompt (non-interactive mode)\n" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return -1;
    }

    llm::EngineParams params;
    std::string prompt;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << a << std::endl;
                exit(-1);
            }
            return argv[++i];
        };
        if      (a == "--temp")      params.temp  = (float)std::atof(next());
        else if (a == "--top-k")     params.top_k = (int)std::atoi(next());
        else if (a == "--top-p")     params.top_p = (float)std::atof(next());
        else if (a == "--ctx")       params.n_ctx = (int)std::atoi(next());
        else if (a == "--n-predict") params.n_predict = (int)std::atoi(next());
        else if (a == "--batch")     params.n_batch = (int)std::atoi(next());
        else if (a == "-s" || a == "--system") params.system_prompt = next();
        else if (a == "-p" || a == "--prompt") prompt = next();
        else {
            std::cerr << "unknown option: " << a << std::endl;
            usage(argv[0]);
            return -1;
        }
    }
    std::cout << std::endl;

    llm::Engine engine;
    engine.load(argv[1], params);
    engine.print_info();

    auto handle_prompt = [&](const std::string& prompt) {
        auto t_start = std::chrono::steady_clock::now();
        int n_prefill = engine.send(prompt);

        bool first_token = true;
        auto t_first_gen = t_start;
        auto on_new_token = [&](const std::string& text) -> bool {
            if (first_token) {
                first_token = false;
                t_first_gen = std::chrono::steady_clock::now();
                std::cout << '\n';
            }
            std::cout << text << std::flush;
            return true;
        };

        int n_gen = engine.generate(on_new_token);
        auto t_last_gen = std::chrono::steady_clock::now();

        if (n_gen > 0) {
            double first_gen_time = std::chrono::duration<double>(t_first_gen - t_start).count();
            double gen_time = std::chrono::duration<double>(t_last_gen - t_first_gen).count();
            double per_gen_time = n_gen > 1 ? gen_time / (n_gen - 1) : 1e-9;
            printf("\n\n[ Prompt: %d tokens in %.2f ms (%.2f tok/s)"
                " | Generation: %d tokens, each %.2f ms (%.2f tok/s) | Total: %.2f ms ]\n",
                n_prefill, first_gen_time * 1000, n_prefill / first_gen_time,
                n_gen, per_gen_time * 1000, 1 / per_gen_time, (first_gen_time + gen_time) * 1000);
        }
    };

    // non-interactive mode
    if (!prompt.empty()) {
        handle_prompt(prompt);
        return 0;
    }

    // interactive mode
    std::cout << "[ready] chat started\n> " << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty()) {
            if (line == "/quit" || line == "/exit") break;
            handle_prompt(line);
        }
        std::cout << "> " << std::flush;
    }
    return 0;
}
