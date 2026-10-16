
#include "tokenizer.h"
#include <queue>
#include <algorithm>

namespace llm {

namespace {

std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

uint32_t utf8_decode_next(const std::string& s, size_t& pos) {
    uint8_t c = (uint8_t)s[pos];
    uint32_t cp = 0;
    int len = 1;
    if (c < 0x80) {
        cp = c;
    } else if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        len = 3;
    } else {
        cp = c & 0x07;
        len = 4;
    }
    for (int k = 1; k < len; ++k) cp = (cp << 6) | ((uint8_t)s[pos + k] & 0x3F);
    pos += len;
    return cp;
}

// codepoint classification
enum CpClass : int { CP_LETTER = 0, CP_NUMBER = 1, CP_PUNCT = 2, CP_SPACE = -1 };

// Range table format, searched via binary search.
struct CpRange {
    uint32_t lo, hi;
    int cls; // CpClass
};

// Covers common ranges needed by BPE pre-tokenization, derived from llama.cpp unicode-data.cpp.
// Unmapped codepoints >= 0x80 default to CP_LETTER.
static const CpRange codepoint_ranges[] = {
    // ASCII
    {0x0009, 0x000D, CP_SPACE},
    {0x0020, 0x0020, CP_SPACE},
    {0x0021, 0x002F, CP_PUNCT},
    {0x0030, 0x0039, CP_NUMBER},
    {0x003A, 0x0040, CP_PUNCT},
    {0x0041, 0x005A, CP_LETTER},
    {0x005B, 0x0060, CP_PUNCT},
    {0x0061, 0x007A, CP_LETTER},
    {0x007B, 0x007E, CP_PUNCT},

    // Latin, Greek, Cyrillic, Arabic, Devanagari, Thai, Georgian, etc.
    {0x00A0, 0x00A0, CP_SPACE},
    {0x00C0, 0x024F, CP_LETTER},
    {0x0250, 0x02AF, CP_LETTER},
    {0x0370, 0x03FF, CP_LETTER},
    {0x0400, 0x04FF, CP_LETTER},
    {0x0500, 0x052F, CP_LETTER},
    {0x0600, 0x065F, CP_LETTER},
    {0x0660, 0x0669, CP_NUMBER},
    {0x066A, 0x06EF, CP_LETTER},
    {0x06F0, 0x06F9, CP_NUMBER},
    {0x06FA, 0x06FF, CP_LETTER},
    {0x0900, 0x0965, CP_LETTER},
    {0x0966, 0x096F, CP_NUMBER},
    {0x0970, 0x097F, CP_LETTER},
    {0x09E6, 0x09EF, CP_NUMBER},
    {0x0E00, 0x0E4F, CP_LETTER},
    {0x0E50, 0x0E59, CP_NUMBER},
    {0x0E5A, 0x0E7F, CP_LETTER},
    {0x10A0, 0x10FF, CP_LETTER},
    {0x1100, 0x11FF, CP_LETTER},

    // Unicode whitespace
    {0x1680, 0x1680, CP_SPACE},
    {0x1E00, 0x1EFF, CP_LETTER},
    {0x2000, 0x200A, CP_SPACE},
    {0x2028, 0x2029, CP_SPACE},
    {0x202F, 0x202F, CP_SPACE},
    {0x205F, 0x205F, CP_SPACE},

    // CJK punctuation, Hiragana, Katakana
    {0x3000, 0x3000, CP_SPACE},
    {0x3001, 0x3004, CP_PUNCT},
    {0x3005, 0x3005, CP_LETTER},
    {0x3006, 0x3006, CP_PUNCT},
    {0x3007, 0x3007, CP_NUMBER},
    {0x3008, 0x3020, CP_PUNCT},
    {0x3021, 0x3029, CP_NUMBER},
    {0x302A, 0x302F, CP_PUNCT},
    {0x3030, 0x3030, CP_PUNCT},
    {0x3031, 0x3035, CP_LETTER},
    {0x3036, 0x3037, CP_PUNCT},
    {0x3038, 0x303A, CP_NUMBER},
    {0x303B, 0x303B, CP_LETTER},
    {0x303C, 0x303F, CP_PUNCT},
    {0x3041, 0x3096, CP_LETTER},
    {0x3099, 0x309A, CP_PUNCT},
    {0x309D, 0x309F, CP_LETTER},
    {0x30A0, 0x30A0, CP_PUNCT},
    {0x30A1, 0x30FA, CP_LETTER},
    {0x30FB, 0x30FB, CP_PUNCT},
    {0x30FC, 0x30FF, CP_LETTER},
    {0x3130, 0x318F, CP_LETTER},
    {0x31F0, 0x31FF, CP_LETTER},

    // CJK Unified Ideographs
    {0x3400, 0x4DBF, CP_LETTER},
    {0x4E00, 0x9FFF, CP_LETTER},
    {0xAC00, 0xD7A3, CP_LETTER},
    {0xF900, 0xFAFF, CP_LETTER},
    {0xFEFF, 0xFEFF, CP_SPACE},

    // Fullwidth / Halfwidth forms
    {0xFF01, 0xFF0F, CP_PUNCT},
    {0xFF10, 0xFF19, CP_NUMBER},
    {0xFF1A, 0xFF20, CP_PUNCT},
    {0xFF21, 0xFF3A, CP_LETTER},
    {0xFF3B, 0xFF40, CP_PUNCT},
    {0xFF41, 0xFF5A, CP_LETTER},
    {0xFF5B, 0xFF65, CP_PUNCT},
    {0xFF66, 0xFFBE, CP_LETTER},
    {0xFFC2, 0xFFC7, CP_LETTER},
    {0xFFCA, 0xFFCF, CP_LETTER},
    {0xFFD2, 0xFFD7, CP_LETTER},
    {0xFFDA, 0xFFDC, CP_LETTER},
    {0xFFE0, 0xFFE6, CP_PUNCT},

    // CJK extensions
    {0x20000, 0x2A6DF, CP_LETTER},
    {0x2A700, 0x2B73F, CP_LETTER},
    {0x2B740, 0x2B81F, CP_LETTER},
    {0x2B820, 0x2CEAF, CP_LETTER},
    {0x2F800, 0x2FA1F, CP_LETTER},
};

static constexpr int NUM_CP_RANGES = sizeof(codepoint_ranges) / sizeof(codepoint_ranges[0]);

int codepoint_class(uint32_t cp) {
    // ASCII fast path
    if (cp < 0x80) {
        if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return CP_SPACE;
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return CP_LETTER;
        if (cp >= '0' && cp <= '9') return CP_NUMBER;
        return CP_PUNCT;
    }
    // binary search the range table
    int lo = 0, hi = NUM_CP_RANGES - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < codepoint_ranges[mid].lo) hi = mid - 1;
        else if (cp > codepoint_ranges[mid].hi) lo = mid + 1;
        else return codepoint_ranges[mid].cls;
    }
    return CP_LETTER; // unknown codepoints >= 0x80 default to letter
}

// Match English contractions ('s/'t/'re/'ve/'m/'ll/'d), case-insensitive.
size_t contraction_len(const std::string& s, size_t i) {
    if (s[i] != '\'') return 0;
    size_t n = s.size();
    auto to_lower = [](char c) -> char { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; };
    auto match_next = [&](char c) { return i + 1 < n && to_lower(s[i + 1]) == c; };
    auto match_next2 = [&](char a, char b) { return i + 2 < n && to_lower(s[i + 1]) == a && to_lower(s[i + 2]) == b; };
    if (match_next2('r', 'e') || match_next2('v', 'e') || match_next2('l', 'l')) return 3;
    if (match_next('s') || match_next('t') || match_next('m') || match_next('d')) return 2;
    return 0;
}

inline uint32_t peek_codepoint(const std::string& s, size_t pos) {
    size_t tmp = pos;
    return utf8_decode_next(s, tmp);
}

// Consume consecutive letter codepoints from pos.
size_t consume_letters(const std::string& s, size_t pos) {
    size_t k = pos;
    while (k < s.size()) {
        size_t prev = k;
        if (codepoint_class(utf8_decode_next(s, k)) != CP_LETTER) return prev;
    }
    return k;
}

} // namespace

bool Tokenizer::load(const GGUF& gguf) {
    const GGUFValue* tokv = gguf.find_meta("tokenizer.ggml.tokens");
    if (!tokv || tokv->str.empty()) return false;
    tokens = tokv->str;
    for (int32_t i = 0; i < (int32_t)tokens.size(); ++i) id_of[tokens[i]] = i;

    // Token types determine which tokens are control/special.
    if (const GGUFValue* tt = gguf.find_meta("tokenizer.ggml.token_type")) {
        for (int64_t v : tt->ints) token_type.push_back((int32_t)v);
    }
    if (token_type.empty()) token_type.assign(tokens.size(), 1);

    // Merge rules: "a b" strings split into pairs; index order = priority.
    if (const GGUFValue* m = gguf.find_meta("tokenizer.ggml.merges")) {
        for (size_t i = 0; i < m->str.size(); ++i) {
            const std::string& rule = m->str[i];
            auto sp = rule.find(' ');
            if (sp != std::string::npos) {
                merges[{rule.substr(0, sp), rule.substr(sp + 1)}] = (int32_t)i;
            }
        }
    }

    bos_id = (int32_t)gguf.get_int("tokenizer.ggml.bos_token_id", -1);
    eos_id = (int32_t)gguf.get_int("tokenizer.ggml.eos_token_id", -1);

    // GPT-2 bytes_to_unicode table
    auto printable = [](uint32_t b) {
        return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int n = 0;
    for (uint32_t b = 0; b <= 255; ++b) {
        uint32_t cp = printable(b) ? b : (256 + n++);
        std::string s = utf8_encode(cp);
        byte_to_str[b] = s;
        str_to_byte[s] = b;
    }

    // Special tokens: control / user-defined types matched as whole strings.
    for (size_t i = 0; i < tokens.size(); ++i) {
        int type = token_type[i];
        if (type == 3 || type == 4) special_strings.push_back(tokens[i]);
    }
    std::sort(special_strings.begin(), special_strings.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

    return true;
}

void Tokenizer::bpe_ids(const std::string& piece, std::vector<int32_t>& out) const {
    if (piece.empty()) return;

    // Byte-level unicode mapping
    std::string encoded;
    encoded.reserve(piece.size() * 2);
    for (unsigned char b : piece) encoded += byte_to_str[b];

    // Split into one symbol per unicode character, linked-list style
    struct Symbol {
        std::string text;
        int prev;
        int next;
    };

    std::vector<Symbol> syms;
    size_t p = 0;
    while (p < encoded.size()) {
        size_t start = p;
        utf8_decode_next(encoded, p);
        int idx = (int)syms.size();
        syms.push_back({encoded.substr(start, p - start), idx - 1, -1});
        if (idx > 0) syms[idx - 1].next = idx;
    }

    if (syms.empty()) return;
    if (syms.size() == 1) {
        auto it = id_of.find(syms[0].text);
        if (it != id_of.end()) out.push_back(it->second);
        return;
    }

    // Min-heap: lower rank = higher priority; ties broken by leftmost position
    struct Bigram {
        int left;
        int right;
        int rank;
        std::string text;
    };
    auto cmp = [](const Bigram& a, const Bigram& b) {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> queue(cmp);

    auto try_add_bigram = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        auto it = merges.find({syms[left].text, syms[right].text});
        if (it != merges.end()) {
            queue.push({left, right, it->second, syms[left].text + syms[right].text});
        }
    };

    for (int i = 1; i < (int)syms.size(); ++i) {
        try_add_bigram(i - 1, i);
    }

    // BPE merge loop
    while (!queue.empty()) {
        auto bigram = queue.top();
        queue.pop();

        auto& left_sym  = syms[bigram.left];
        auto& right_sym = syms[bigram.right];

        if (left_sym.text.empty() || right_sym.text.empty()) continue; // skip invalidated symbols
        if (left_sym.text + right_sym.text != bigram.text) continue; // skip stale bigrams

        // Merge right symbol into left
        left_sym.text += right_sym.text;
        right_sym.text.clear();

        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) {
            syms[right_sym.next].prev = bigram.left;
        }

        try_add_bigram(left_sym.prev, bigram.left);
        try_add_bigram(bigram.left, left_sym.next);
    }

    // Collect results by walking the linked list
    for (int i = 0; i >= 0; i = syms[i].next) {
        if (syms[i].text.empty()) continue;
        auto it = id_of.find(syms[i].text);
        if (it != id_of.end()) {
            out.push_back(it->second);
        } else {
            // Byte-level fallback: look up individual byte tokens
            for (char c : syms[i].text) {
                auto bt = id_of.find(std::string(1, c));
                if (bt != id_of.end()) out.push_back(bt->second);
            }
        }
    }
}

// Pre-tokenization regex pattern:
//   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_id >= 0) ids.push_back(bos_id);
    size_t i = 0, n = text.size();
    while (i < n) {
        // 1. special tokens
        bool matched = false;
        for (const std::string& sp : special_strings) {
            if (text.compare(i, sp.size(), sp) == 0) {
                auto it = id_of.find(sp);
                if (it != id_of.end()) ids.push_back(it->second);
                i += sp.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        uint32_t cp = peek_codepoint(text, i);
        int cls = codepoint_class(cp);

        // 2. contractions
        size_t clen = contraction_len(text, i);
        if (clen) {
            bpe_ids(text.substr(i, clen), ids);
            i += clen;
            continue;
        }
        // 3. letter string: [^\r\n\p{L}\p{N}]?\p{L}+
        if (cls == CP_LETTER) {
            size_t k = consume_letters(text, i);
            bpe_ids(text.substr(i, k - i), ids);
            i = k;
            continue;
        }
        if (cp != '\r' && cp != '\n' && cls != CP_LETTER && cls != CP_NUMBER) {
            size_t next_pos = i;
            utf8_decode_next(text, next_pos);
            if (next_pos < n && codepoint_class(peek_codepoint(text, next_pos)) == CP_LETTER) {
                size_t k = consume_letters(text, next_pos);
                bpe_ids(text.substr(i, k - i), ids);
                i = k;
                continue;
            }
        }
        // 4. number
        if (cls == CP_NUMBER) {
            size_t next_pos = i;
            utf8_decode_next(text, next_pos);
            bpe_ids(text.substr(i, next_pos - i), ids);
            i = next_pos;
            continue;
        }
        // 5. punct string:  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (cp == ' ' && j + 1 < n && codepoint_class(peek_codepoint(text, j + 1)) == CP_PUNCT) {
                j = j + 1;
            }
            int jcls = (j == i) ? cls : codepoint_class(peek_codepoint(text, j));
            if (jcls == CP_PUNCT) {
                size_t k = j;
                while (k < n) {
                    size_t prev_k = k;
                    uint32_t c = utf8_decode_next(text, k);
                    if (codepoint_class(c) != CP_PUNCT) {
                        k = prev_k;
                        break;
                    }
                }
                while (k < n && (text[k] == '\r' || text[k] == '\n')) ++k;
                bpe_ids(text.substr(i, k - i), ids);
                i = k;
                continue;
            }
        }
        // 6. whitespace
        if (cls == CP_SPACE) {
            size_t num_ws = 0;
            size_t last_newline_end = 0;
            size_t prev_end = i;
            size_t scan = i;
            while (scan < n) {
                size_t prev_scan = scan;
                uint32_t c = utf8_decode_next(text, scan);
                if (codepoint_class(c) != CP_SPACE) {
                    scan = prev_scan;
                    break;
                }
                num_ws++;
                if (c == '\r' || c == '\n') last_newline_end = scan;
                if (num_ws >= 2) prev_end = prev_scan;
            }
            // \s*[\r\n]+
            if (last_newline_end > 0) {
                bpe_ids(text.substr(i, last_newline_end - i), ids);
                i = last_newline_end;
                continue;
            }
            // \s+(?!\S)
            if (num_ws > 1 && scan < n) {
                bpe_ids(text.substr(i, prev_end - i), ids);
                i = prev_end;
                continue;
            }
            // \s+
            if (num_ws > 0) {
                bpe_ids(text.substr(i, scan - i), ids);
                i = scan;
                continue;
            }
        }
        // fallback
        {
            size_t next_pos = i;
            utf8_decode_next(text, next_pos);
            bpe_ids(text.substr(i, next_pos - i), ids);
            i = next_pos;
        }
    }
    return ids;
}

void Tokenizer::flush_utf8(std::string& out) {
    size_t i = 0, n = pending_bytes_.size();
    while (i < n) {
        uint8_t c = (uint8_t)pending_bytes_[i];
        int len;
        if (c < 0x80) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { ++i; continue; }
        if (i + len > n) break;
        out.append(pending_bytes_, i, len);
        i += len;
    }
    pending_bytes_.erase(0, i);
}

std::string Tokenizer::decode(int32_t id) {
    if (id < 0 || id >= (int32_t)tokens.size()) return "";
    if (token_type[id] == 3 || token_type[id] == 4) return "";

    const std::string& tok = tokens[id];
    std::string bytes;
    bytes.reserve(tok.size());
    size_t p = 0;
    while (p < tok.size()) {
        size_t start = p;
        utf8_decode_next(tok, p);
        auto it = str_to_byte.find(tok.substr(start, p - start));
        if (it != str_to_byte.end()) bytes += (char)it->second;
    }

    pending_bytes_ += bytes;
    std::string out;
    flush_utf8(out);
    return out;
}

} // namespace llm
