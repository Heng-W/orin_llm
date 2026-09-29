#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace llm {

void Sampler::seed(uint32_t s) {
    rng_.seed(s);
}

int32_t Sampler::sample(const float* logits, int n_vocab, int32_t eos_id, bool& is_eos) {
    if (n_vocab == 0) {
        is_eos = false;
        return -1;
    }

    cur_.resize(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        cur_[i] = { i, logits[i], 0.0f };
    }
    size_t size = (size_t) n_vocab;

    apply_top_k(cur_.data(), size);
    apply_top_p(cur_.data(), size);
    apply_min_p(cur_.data(), size);
    apply_temp(cur_.data(), size);
    int selected = apply_dist(cur_.data(), size);

    int32_t result = cur_[selected].id;
    is_eos = (result == eos_id);
    return result;
}

void Sampler::apply_temp(TokenData* data, size_t size) {
    if (temp <= 0.0f) {
        size_t max_i = 0;
        float  max_l = data[0].logit;
        for (size_t i = 1; i < size; ++i) {
            if (data[i].logit > max_l) {
                data[max_i].logit = -INFINITY;
                max_i = i;
                max_l = data[i].logit;
            } else {
                data[i].logit = -INFINITY;
            }
        }
        return;
    }

    for (size_t i = 0; i < size; ++i) {
        data[i].logit /= temp;
    }
}

void Sampler::partial_sort_bucket(const TokenData* src, size_t src_size, int npartial, std::vector<TokenData>& res) {
    static const auto comp = [](const TokenData& a, const TokenData& b) { return a.logit > b.logit; };
    constexpr int   nbuckets     = 128;
    constexpr float bucket_low   = -10.0f;
    constexpr float bucket_high  =  10.0f;
    constexpr float bucket_scale = nbuckets / (bucket_high - bucket_low);
    constexpr float bucket_inter = -bucket_low * bucket_scale;

    std::vector<int> bucket_idx;
    std::vector<int> histo(nbuckets, 0);
    std::vector<TokenData*> bucket_ptrs;

    bucket_idx.reserve(src_size);

    for (int i = 0; i < (int) src_size; ++i) {
        int ib = int(bucket_scale * src[i].logit + bucket_inter);
        ib = std::max(0, std::min(nbuckets - 1, ib));
        bucket_idx.push_back(ib);
        ++histo[ib];
    }

    int nhave = 0;
    int ib = nbuckets - 1;
    for ( ; ib >= 0; --ib) {
        nhave += histo[ib];
        if (nhave >= npartial) break;
    }

    res.resize(nhave);
    auto* ptr = res.data();
    bucket_ptrs.reserve(nbuckets - ib);
    for (int j = nbuckets - 1; j >= ib; --j) {
        bucket_ptrs.push_back(ptr);
        ptr += histo[j];
    }

    for (int i = 0; i < (int) src_size; ++i) {
        int j = bucket_idx[i];
        if (j >= ib) {
            *bucket_ptrs[nbuckets - 1 - j]++ = src[i];
        }
    }

    ptr = res.data();
    int ndone = 0;
    for (int j = nbuckets - 1; j > ib; --j) {
        std::sort(ptr, ptr + histo[j], comp);
        ptr += histo[j];
        ndone += histo[j];
    }
    std::partial_sort(ptr, ptr + npartial - ndone, ptr + histo[ib], comp);
}

void Sampler::apply_top_k(TokenData* data, size_t& size) {
    if (top_k <= 0) return;

    int k = std::min(top_k, (int) size);

    static const auto comp = [](const TokenData& a, const TokenData& b) { return a.logit > b.logit; };
    if (k <= 128) {
        std::partial_sort(data, data + k, data + size, comp);
    } else {
        partial_sort_bucket(data, size, k, buf_sort_);
        std::copy(buf_sort_.data(), buf_sort_.data() + k, data);
    }

    size = (size_t)k;
}

void Sampler::softmax(TokenData* data, size_t size) {
    float max_l = data[0].logit;
    for (size_t i = 1; i < size; ++i) {
        max_l = std::max(max_l, data[i].logit);
    }

    float cum_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float p = std::exp(data[i].logit - max_l);
        data[i].p = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < size; ++i) {
        data[i].p /= cum_sum;
    }
}

void Sampler::apply_top_p(TokenData* data, size_t& size) {
    if (top_p >= 1.0f) return;

    softmax(data, size);

    constexpr size_t TOP_P_SORT_THRESHOLD = 1024;
    constexpr size_t TOP_P_INITIAL_K      = 256;

    static const auto comp = [](const TokenData& a, const TokenData& b) { return a.p > b.p; };

    // adaptive partial sort: for large candidate sets, try sorting only
    // the initial top-k first; expand to full sort only if needed
    size_t k = size;
    if (size > TOP_P_SORT_THRESHOLD) {
        k = std::min<size_t>(TOP_P_INITIAL_K, size);
        std::partial_sort(data, data + k, data + size, comp);
    } else {
        std::sort(data, data + size, comp);
    }

    float cum_sum = 0.0f;
    size_t last_idx = size;
    for (size_t i = 0; i < size; ++i) {
        cum_sum += data[i].p;
        if (cum_sum >= top_p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
        // exceeded partial sort range, fall back to full sort
        if (i == k - 1 && k < size) {
            std::sort(data, data + size, comp);
            k = size;
        }
    }

    size = last_idx;
}

void Sampler::apply_min_p(TokenData* data, size_t& size) {
    if (min_p <= 0.0f || size == 0) return;

    float max_logit = data[0].logit;
    for (size_t i = 1; i < size; ++i) {
        max_logit = std::max(max_logit, data[i].logit);
    }
    const float min_logit = max_logit + std::log(min_p);

    size_t j = 0;
    for (size_t i = 0; i < size; ++i) {
        if (data[i].logit >= min_logit) {
            data[j++] = data[i];
        }
    }

    if (j >= min_keep) size = j;
}

int Sampler::apply_dist(TokenData* data, size_t size) {
    if (size == 0) return 0;

    if (size == 1) {
        data[0].p = 1.0f;
        return 0;
    }

    float max_l = data[0].logit;
    for (size_t i = 1; i < size; ++i) {
        max_l = std::max(max_l, data[i].logit);
    }

    double cum_sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float p = std::exp(data[i].logit - max_l);
        data[i].p = p;
        cum_sum += p;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double rnd = dist(rng_);

    double running_sum = 0.0;
    const double target_sum  = cum_sum * rnd;

    int selected = (int) size - 1;
    for (size_t i = 0; i < size; ++i) {
        running_sum += data[i].p;
        data[i].p = (float)(data[i].p / cum_sum);
        if (running_sum >= target_sum && selected == (int) size - 1) {
            selected = (int) i;
        }
    }

    return selected;
}

} // namespace llm
