#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string>

#define LLM_ABORT(...) llm::llm_abort(__FILE__, __LINE__, __VA_ARGS__)
#define LLM_ASSERT(x) do { if (!(x)) LLM_ABORT("LLM_ASSERT(%s) failed", #x); } while(0)

#ifdef __CUDACC__
#define CUDA_CHECK(err)                                                       \
    do {                                                                      \
        cudaError_t err_ = (err);                                             \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s in function %s at %s:%d\n",        \
                    cudaGetErrorString(err_), __func__, __FILE__, __LINE__);  \
            abort();                                                          \
        }                                                                     \
    } while (0)

#include <cuda_fp16.h>
typedef __half   fp16;
typedef __half2  fp16x2;
#else
typedef uint16_t fp16;
typedef uint32_t fp16x2;
#endif

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

// match ggml_type number
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

inline size_t tensor_nbytes(DType type, int ne0, int ne1) {
    type_traits t = type_info(type);
    size_t nb = (size_t)ne0 * ne1 / t.elems_per_block;
    return nb * t.bytes_per_block;
}

enum class GGUFType : uint32_t {
    UINT8   = 0, INT8  = 1, UINT16  = 2,  INT16 = 3,
    UINT32  = 4, INT32 = 5, FLOAT32 = 6,  BOOL  = 7,
    STRING  = 8, ARRAY = 9, UINT64  = 10, INT64 = 11,
    FLOAT64 = 12,
};

// 1D/2D tensor descriptor
struct Tensor {
    std::string    name;
    DType          type   = DType::F32;
    int            ne0    = 0;
    int            ne1    = 1;
    size_t         offset = 0;
    const uint8_t* data   = nullptr;

    int64_t elems() const { return (int64_t)ne0 * ne1; }
};

} // namespace llm
