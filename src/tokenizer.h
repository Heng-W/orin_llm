#pragma once
// Byte-level BPE tokenizer (GPT-2 style).
// Encode: UTF-8 bytes -> byte-level unicode mapping -> BPE merges.
// Decode: reverse byte lookup -> incremental UTF-8 output.

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common.h"
#include "gguf.h"

namespace llm {

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h = std::hash<std::string>{};
        return h(p.first) ^ (h(p.second) << 1);
    }
};

class Tokenizer {
public:
    std::vector<std::string>        tokens;
    std::unordered_map<std::string, int32_t> id_of;
    std::vector<int32_t>            token_type;

    int32_t bos_id = -1;
    int32_t eos_id = -1;

    // Load vocabulary from GGUF metadata; returns true on success.
    bool load(const GGUF& gguf);

    std::vector<int32_t> encode(const std::string& text, bool add_bos) const;

    // Incremental single-token decode; skips control/special tokens.
    // Multi-byte UTF-8 characters spanning tokens are buffered internally.
    std::string decode(int32_t id);

private:
    // BPE merge rules: (left, right) -> rank (lower = higher priority).
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merges;
    // Special token strings, sorted by length descending for longest-match-first.
    std::vector<std::string> special_strings;

    // GPT-2 byte <-> unicode mapping.
    std::array<std::string, 256> byte_to_str;
    std::unordered_map<std::string, uint8_t> str_to_byte;

    // Incremental UTF-8 decode buffer.
    std::string pending_bytes_;

    void bpe_ids(const std::string& piece, std::vector<int32_t>& out) const;
    void flush_utf8(std::string& out);
};

} // namespace llm
