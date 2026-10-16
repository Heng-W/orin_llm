#pragma once

#include <stdint.h>
#include <vector>
#include <random>

namespace llm {

struct TokenData {
    int32_t id;
    float logit;
    float p; // probability, filled by softmax
};

struct Sampler {
    float  temp  = 0.8f;
    int    top_k = 40;
    float  top_p = 0.95f;
    float  min_p = 0.05f;

    // top-k -> top-p -> min-p -> temp -> dist
    int32_t sample(const float* logits, int n_vocab);

    void seed(uint32_t s) { rng_.seed(s); }

private:
    void apply_top_k (TokenData* data, size_t& size);
    void apply_top_p (TokenData* data, size_t& size);
    void apply_min_p (TokenData* data, size_t& size);
    void apply_temp  (TokenData* data, size_t  size);
    int  apply_dist  (TokenData* data, size_t  size);

    static void softmax(TokenData* data, size_t size);

    static void partial_sort_bucket(const TokenData* src, size_t src_size, int npartial, std::vector<TokenData>& res);

    std::mt19937 rng_{std::random_device{}()};
    std::vector<TokenData> cur_;
    std::vector<TokenData> buf_sort_;
};

} // namespace llm
