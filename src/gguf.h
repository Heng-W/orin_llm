#pragma once

#include <vector>
#include <unordered_map>
#include "common.h"

namespace llm {

struct GGUFValue {
    GGUFType type     = GGUFType::UINT8;
    bool     is_array = false;
    uint64_t count    = 1;

    std::vector<std::string> str;
    std::vector<int64_t>     ints;
    std::vector<double>      floats;
};

// parsed GGUF file
struct GGUF {
    uint32_t version   = 0;
    size_t   alignment = 32;

    std::unordered_map<std::string, GGUFValue> meta;
    std::vector<Tensor> tensors;
    std::unordered_map<std::string, size_t> tensor_index;  // name -> index in tensors

    const uint8_t* data = nullptr;
    size_t          size = 0;
    size_t          data_offset = 0;
    int             fd   = -1;

    GGUF() = default;
    ~GGUF();
    GGUF(const GGUF &) = delete;
    GGUF(GGUF &&) = delete;
    GGUF& operator=(const GGUF&) = delete;
    GGUF& operator=(GGUF&&) = delete;

    const Tensor* find_tensor(const std::string& name) const;
    const GGUFValue* find_meta(const std::string& key) const;

    int64_t     get_int(const std::string& key, int64_t def = 0) const;
    double      get_float(const std::string& key, double def = 0) const;
    std::string get_string(const std::string& key) const;
};

void load_gguf(GGUF& out, const std::string& path);

} // namespace llm
