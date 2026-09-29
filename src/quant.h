#pragma once
// K-quant block definitions

#include "common.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
typedef __half   fp16;
typedef __half2  fp16x2;
#else
typedef uint16_t fp16;
typedef uint32_t fp16x2;
#endif

namespace llm {

// super-block size and scale packing size
constexpr uint32_t QK_K          = 256;
constexpr uint32_t K_SCALE_SIZE  = 12;

// Q4_K: dequant formula x = d*scale*q - dmin*min
struct block_q4_K {
    fp16 d;
    fp16 dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "q4_K block size");

// Q5_K: low 4 bits in qs, high 1 bit in qh
struct block_q5_K {
    fp16 d;
    fp16 dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 176, "q5_K block size");

// Q6_K: low 4 bits in ql, high 2 bits in qh, signed 8-bit scale per 16 elements
struct block_q6_K {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    fp16 d;
};
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

struct block_q8_0 {
    fp16 d;
    int8_t   qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 block size");

} // namespace llm