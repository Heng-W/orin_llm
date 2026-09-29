#include "orin_llm/engine.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace llm;

static void usage(const char* prog) {
    std::cerr
        << "usage: " << prog << " <model.gguf> [options]\n"
        << "options:\n"
        << "  --temp F      temperature (default 0.8)\n"
        << "  --top-k N     top-k sampling (default 40)\n"
        << "  --top-p F     nucleus sampling (default 0.95)\n"
        << "  --ctx N       context length (default: model metadata)\n"
        << "  --system S    system prompt\n"
        << "  --n-predict N max generated tokens (default 512)\n"
        << "  --batch N     prefill batch size (default 512)\n"
        << "  --no-zero-copy  use cudaMalloc+copy instead of zero-copy\n"
        << "  -p S          prompt (non-interactive mode)\n"
        << "\n"
        << "type /quit or /exit to leave the chat.\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    EngineParams params;
    std::string prompt;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--temp")      params.temp  = (float) std::atof(next());
        else if (a == "--top-k")     params.top_k = (int) std::atoi(next());
        else if (a == "--top-p")     params.top_p = (float) std::atof(next());
        else if (a == "--ctx")       params.n_ctx = (int) std::atoi(next());
        else if (a == "--system")    params.system_prompt = next();
        else if (a == "--n-predict") params.n_predict = (int) std::atoi(next());
        else if (a == "--batch")     params.n_batch   = (int) std::atoi(next());
        else if (a == "--no-zero-copy") params.weight_mode = 1;
        else if (a == "-p")          prompt = next();
        else { std::cerr << "unknown option: " << a << "\n"; usage(argv[0]); return 1; }
    }

    auto print_token = [](const std::string& text) -> bool {
        std::cout << text << std::flush;
        return true;
    };

    double total_decode_ms = 0.0;
    auto accumulate_decode = [&](int /*step*/, double ms) {
        total_decode_ms += ms;
    };

    auto print_prefill = [](int n_tokens, double ms) {
        std::cerr << "[prefill] " << n_tokens << " tokens in "
                  << ms << " ms (" << (double) n_tokens / (ms / 1000.0) << " tok/s)\n";
    };

    Engine engine;
    engine.load(argv[1], params);
    engine.print_info();

    // non-interactive mode
    if (!prompt.empty()) {
        engine.send(prompt, print_prefill);

        auto t0 = std::chrono::steady_clock::now();
        total_decode_ms = 0.0;
        int n_gen = engine.generate(print_token, accumulate_decode);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::cout << "\n";
        if (dt > 0.0)
            std::cerr << "(" << n_gen << " tokens, "
                      << (double) n_gen / dt << " tok/s, decode only: "
                      << (double) n_gen / (total_decode_ms / 1000.0) << " tok/s)\n";
        return 0;
    }

    // interactive mode
    std::cerr << "[ready] chat started\n";

    std::string line;
    std::cout << "> " << std::flush;
    while (std::getline(std::cin, line)) {
        if (line.empty()) { std::cout << "> " << std::flush; continue; }
        if (line == "/quit" || line == "/exit") break;

        engine.send(line, print_prefill);

        auto t0 = std::chrono::steady_clock::now();
        total_decode_ms = 0.0;
        int n_gen = engine.generate(print_token, accumulate_decode);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        std::cout << "\n";
        if (dt > 0.0)
            std::cerr << "(" << n_gen << " tokens, "
                      << (double) n_gen / dt << " tok/s, decode only: "
                      << (double) n_gen / (total_decode_ms / 1000.0) << " tok/s)\n";
        std::cout << "> " << std::flush;
    }
    return 0;
}
