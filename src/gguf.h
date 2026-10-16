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

class GGUF {
public:
    GGUF() = default;
    ~GGUF();
    GGUF(const GGUF&) = delete;
    GGUF& operator=(const GGUF&) = delete;

    void load(const std::string& path);

    const Tensor* find_tensor(const std::string& name) const {
        auto it = tensor_index_.find(name);
        if (it != tensor_index_.end()) return &tensors_[it->second];
        return nullptr;
    }

    const GGUFValue* find_meta(const std::string& key) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }

    int64_t get_int(const std::string& key, int64_t def = 0) const {
        const GGUFValue* v = find_meta(key);
        return (v && !v->ints.empty()) ? v->ints[0] : def;
    }

    double get_float(const std::string& key, double def = 0) const {
        const GGUFValue* v = find_meta(key);
        return (v && !v->floats.empty()) ? v->floats[0] : def;
    }

    std::string get_string(const std::string& key) const {
        const GGUFValue* v = find_meta(key);
        return (v && !v->str.empty()) ? v->str[0] : std::string{};
    }

    const uint8_t* data_start() const { return data_ + data_offset_; }
    const uint8_t* data_end() const { return data_ + size_; }

private:
    std::unordered_map<std::string, GGUFValue> meta_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, size_t> tensor_index_; // name -> index in tensors

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t data_offset_ = 0;
    int fd_ = -1;
};

} // namespace llm
