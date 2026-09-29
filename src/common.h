#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define LLM_ABORT(...) llm::llm_abort(__FILE__, __LINE__, __VA_ARGS__)
#define LLM_ASSERT(x) do { if (!(x)) LLM_ABORT("LLM_ASSERT(%s) failed", #x); } while(0)

namespace llm {

[[noreturn]] inline void llm_abort(const char* file, int line, const char* fmt, ...) {
    fflush(stdout);
    char buf[2048];
    int off = snprintf(buf, sizeof(buf), "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf + off, sizeof(buf) - off, fmt, args);
    va_end(args);
    fprintf(stderr, "%s\n", buf);
    abort();
}

// matches ggml_type numbering
enum class DType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
};


struct type_traits {
    uint32_t elems_per_block;
    uint32_t bytes_per_block;
};

inline type_traits type_info(DType t) {
    switch (t) {
        case DType::F32:  return { 1,   4 };
        case DType::F16:  return { 1,   2 };
        case DType::Q8_0: return { 32,  34 };
        case DType::Q4_K: return { 256, 144 };
        case DType::Q5_K: return { 256, 176 };
        case DType::Q6_K: return { 256, 210 };
        default: LLM_ABORT("unsupported dtype");
    }
}


inline size_t tensor_nbytes(DType t, uint64_t ne0, uint64_t ne1) {
    type_traits ti = type_info(t);
    uint64_t nb = (ne0 * ne1) / ti.elems_per_block;
    return (size_t)(nb * ti.bytes_per_block);
}

// fp16 -> fp32 (branchless, from llama.cpp/ggml)
inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } conv;
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = 0x1.0p-112f;
    conv.u = (two_w >> 4) + exp_offset;
    const float normalized_value = conv.f * exp_scale;

    const uint32_t magic_mask = 126u << 23;
    conv.u = (two_w >> 17) | magic_mask;
    const float denormalized_value = conv.f - 0.5f;

    const uint32_t denormalized_cutoff = 1u << 27;
    conv.f = (two_w < denormalized_cutoff) ? denormalized_value : normalized_value;
    conv.u |= sign;
    return conv.f;
}


enum class GGUFType : uint32_t {
    UINT8   = 0, INT8    = 1, UINT16  = 2,  INT16   = 3,
    UINT32  = 4, INT32   = 5, FLOAT32 = 6,  BOOL    = 7,
    STRING  = 8, ARRAY   = 9, UINT64  = 10, INT64   = 11,
    FLOAT64 = 12,
};

// 1D/2D tensor descriptor
struct Tensor {
    std::string    name;
    DType          type   = DType::F32;
    uint64_t       ne0    = 0;
    uint64_t       ne1    = 1;
    size_t         offset = 0;
    const uint8_t* data   = nullptr;

    uint64_t elems() const { return ne0 * ne1; }
};

} // namespace llm
