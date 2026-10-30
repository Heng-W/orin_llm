// CUDA kernel implementations.
#include "kernels.h"

#include <assert.h>
#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "quant.h"

namespace llm::cuda {

// --------------- Utility functions ----------------

static uint3 init_fastdiv(uint32_t d) {
    uint32_t L = 0;
    while (L < 32 && (uint32_t{1} << L) < d) L++;
    uint32_t mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fd) {
    return (__umulhi(n, fd.x) + n) >> fd.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fd) {
    return n - fastdiv(n, fd) * fd.z;
}

static __device__ __forceinline__ float low2float(uint32_t val) {
    uint16_t low = (uint16_t)(val & 0xFFFF);
    return __half2float(*reinterpret_cast<const __half*>(&low));
}

static __device__ __forceinline__ float high2float(uint32_t val) {
    uint16_t high = (uint16_t)(val >> 16);
    return __half2float(*reinterpret_cast<const __half*>(&high));
}

static __device__ __forceinline__ int get_int_b2(const void* x, const int i32) {
    const uint16_t* p = (const uint16_t*)x;
    return (p[2 * i32 + 1] << 16) | p[2 * i32];
}

// --- dequant helpers ---

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* __restrict q, uint8_t* __restrict d, uint8_t* __restrict m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float dequantize_q4_K(const block_q4_K& b, int index) {
    uint8_t scale, min;
    get_scale_min_k4(index / 32, b.scales, &scale, &min);
    const float dd = (float)b.d * scale, mm = (float)b.dmin * min;
    const int qs_idx = index / 64 * 32 + (index & 31);
    const int qs_shift = ((index & 32) >> 5) * 4;
    const int q = (b.qs[qs_idx] >> qs_shift) & 0xF;
    return dd * q - mm;
}

__device__ __forceinline__ float dequantize_q5_K(const block_q5_K& b, int index) {
    uint8_t scale, min;
    const int group = index / 32;
    get_scale_min_k4(group, b.scales, &scale, &min);
    const float dd = (float)b.d * scale, mm = (float)b.dmin * min;
    const int qs_idx = index / 64 * 32 + (index & 31);
    const int qs_shift = ((index & 32) >> 5) * 4;
    const int q = (b.qs[qs_idx] >> qs_shift) & 0xF;
    const int high = (b.qh[index & 31] >> group) & 1;
    return dd * (q | (high << 4)) - mm;
}

__device__ __forceinline__ float dequantize_q6_K(const block_q6_K& b, int index) {
    const int ql_idx = index / 128 * 64 + (index & 63);
    const int qh_idx = index / 128 * 32 + (index & 31);
    const int ql_shift = ((index & 64) >> 6) * 4;
    const int qh_shift = ((index & 127) >> 5) * 2;
    const int low  = (b.ql[ql_idx] >> ql_shift) & 0xF;
    const int high = (b.qh[qh_idx] >> qh_shift) & 3;
    return (float)b.d * b.scales[index / 16] * ((low | (high << 4)) - 32);
}

// --- block reduce ---

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset; offset >>= 1) val += __shfl_xor_sync(0xffffffffu, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset; offset >>= 1) val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, offset));
    return val;
}

enum class block_reduce_method { SUM, MAX };

template <block_reduce_method method>
struct block_reduce_policy;

template <> struct block_reduce_policy<block_reduce_method::SUM> {
    static __device__ float reduce(float val) { return warp_reduce_sum(val); }
    static __device__ float sentinel() { return 0.0f; }
};

template <> struct block_reduce_policy<block_reduce_method::MAX> {
    static __device__ float reduce(float val) { return warp_reduce_max(val); }
    static __device__ float sentinel() { return -INFINITY; }
};

template <block_reduce_method method>
__device__ __forceinline__ float block_reduce(float val) {
    __shared__ float shared_vals[32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    val = block_reduce_policy<method>::reduce(val);
    if (lane == 0) shared_vals[warp] = val;
    __syncthreads();
    val = (lane < blockDim.x / 32) ? shared_vals[lane] : block_reduce_policy<method>::sentinel();
    return block_reduce_policy<method>::reduce(val);
}

__device__ __forceinline__ float block_reduce_sum(float val) { return block_reduce<block_reduce_method::SUM>(val); }
__device__ __forceinline__ float block_reduce_max(float val) { return block_reduce<block_reduce_method::MAX>(val); }

// ------------- Activation quantization kernels -------------------

// Quantize activations to q8_1 so that __dp4a can be used, reducing activation read bandwidth to 1/4 of fp32.
__global__ void quantize_q8_1_k(const float* x, int8_t* qs, float* scale, int n_elem) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int group = idx / 32;
    if (group * 32 >= n_elem) return;
    const float val = (idx < n_elem) ? x[idx] : 0.0f;
    const float amax = warp_reduce_max(fabsf(val));
    const float d = amax / 127.0f;
    qs[idx] = amax == 0.0f ? 0 : roundf(val / d);
    if ((threadIdx.x & 31) == 0) scale[group] = d;
}

// Activation quantization to block_q8_1_mmq format, including 128 elements
struct block_q8_1_mmq {
    union {
        float   d4[4];  // D4 layout (Q6K: scale only)
        __half2 ds4[4]; // DS4 layout (Q4K: half2(d, sum)), Q4K needs sum to correct min
    };
    int8_t qs[4 * 32];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "Unexpected block_q8_1_mmq size");

// Output stored transposed: y[k_chunk * ncols_y + token]
// Grid: (ncols_y=batch, ceil(ne0/(4*256))), Block: 256, each thread handles 4 floats
template<bool DS4>
__global__ void quantize_mmq_q8_1_k(const float* __restrict__ X, void* __restrict__ vy,
                                     int ne00, int ne0, int ncols_y) {
    const int i0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    const int i1 = blockIdx.x;
    const float4* x4 = (const float4*)X;
    block_q8_1_mmq* y = (block_q8_1_mmq*)vy;

    const int ib = (i0 / (4 * 32)) * ncols_y + i1; // output block index
    const int iqs = i0 % (4 * 32);

    const float4 xi = (i0 < ne00) ? x4[((size_t)i1 * ne00 + i0) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) { 
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);

    char4* yqs4 = (char4*)y[ib].qs;
    yqs4[iqs / 4] = q;

    if constexpr (DS4) {
        float sum = xi.x + xi.y + xi.z + xi.w;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
        }
        if (iqs % 32 == 0) {
            const float d = 1.0f / d_inv;
            y[ib].ds4[iqs / 32] = __halves2half2(__float2half(d), __float2half(sum));
        }
    } else {
        if (iqs % 32 == 0) y[ib].d4[iqs / 32] = 1.0f / d_inv;
    }
}

// --------- GEMV kernels (dp4a, fused gate+up+swiglu, plain) ---------

// Q4_K: 16 threads per block, Q6_K: 32 threads per block
// Q5_K/Q8_0: pair grid-stride
template <DType TYPE>
__device__ __forceinline__ float gemv_dp4a_dot(const void* __restrict__ W, int row,
                                               const int8_t* __restrict__ xq, const float* __restrict__ xs,
                                               int n_in, int tid, int nthreads) {
    float acc = 0.0f;

    if constexpr (TYPE == DType::Q8_0) {
        const int nblk = n_in / 32;
        const int total_pairs = nblk * 4; // 4 pairs each block, 8 elements each pair
        for (int pair = tid; pair < total_pairs; pair += nthreads) {
            const int blk = pair / 4;
            const int offset = pair % 4 * 2; // in int32 units (each pair = 2 int32)
            const block_q8_0* bq = (const block_q8_0*)W + (size_t)row * nblk + blk;
            const int w0 = get_int_b2(bq->qs, offset);
            const int w1 = get_int_b2(bq->qs, offset + 1);
            const int* x_base = (const int*)(xq + blk * 32 + offset * 4);
            const int x0 = *x_base, x1 = *(x_base + 1);
            const int dot = __dp4a(w1, x1, __dp4a(w0, x0, 0));
            acc += (float)bq->d * xs[blk] * dot;
        }
    } else if constexpr (TYPE == DType::Q4_K) {
        const int nblk = n_in / QK_K;
        const int blocks_per_iter = nthreads / 16; // 16 threads cooperate on one Q4_K block
        const int kbx_start = tid / 16;
        const int iqs = 2 * (tid % 16); // 0,2,4,...,30. qs index in int32 array
        const int bq8_offset = 2 * (iqs / 8); // {0,2,4,6}, pair of q8_1 sub-blocks

        for (int kbx = kbx_start; kbx < nblk; kbx += blocks_per_iter) {
            const block_q4_K* bq = (const block_q4_K*)W + (size_t)row * nblk + kbx;
            // qs has 32 int32, we want tid0(0,4) tid1(1,5) ... tid15(27,31)
            const int* q4 = (const int*)(bq->qs + 16 * bq8_offset + 4 * (tid & 3));
            const int w0 = q4[0];
            const int w1 = q4[4]; // stride 16 bytes apart

            // decode scales
            const uint16_t* scales = (const uint16_t*)bq->scales;
            uint16_t aux[2];
            const int j = bq8_offset / 2;
            if (j < 2) {
                aux[0] = scales[j + 0] & 0x3f3f;
                aux[1] = scales[j + 2] & 0x3f3f;
            } else {
                aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
                aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
            }
            const uint8_t* scale_arr = (const uint8_t*)aux;
            const uint8_t* min_arr = scale_arr + 2;

            float sumf_d = 0.0f, sumf_m = 0.0f;
            const int q8_base = kbx * 8 + bq8_offset;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int w0i = (w0 >> (4 * i)) & 0x0F0F0F0F;
                const int w1i = (w1 >> (4 * i)) & 0x0F0F0F0F;

                const float d8 = xs[q8_base + i];
                const int* x_base = (const int*)(xq + (q8_base + i) * 32 + 4 * (tid & 3));
                const int x0i = x_base[0], x1i = x_base[4];

                const int dot  = __dp4a(w1i, x1i, __dp4a(w0i, x0i, 0));
                const int xsum = __dp4a(0x01010101, x1i, __dp4a(0x01010101, x0i, 0));

                sumf_d += d8 * (dot * scale_arr[i]);
                sumf_m += d8 * (xsum * min_arr[i]);
            }
            const uint32_t dm = *reinterpret_cast<const uint32_t*>(&bq->d);
            acc += low2float(dm) * sumf_d - high2float(dm) * sumf_m;
        }
    } else if constexpr (TYPE == DType::Q5_K) {
        const int nblk = n_in / QK_K;
        const int total_pairs = nblk * 32; // 32 pairs each block, 8 elements each pair
        for (int pair = tid; pair < total_pairs; pair += nthreads) {
            const int blk = pair / 32;
            const int offset = pair % 32 * 8;
            const block_q5_K* bq = (const block_q5_K*)W + (size_t)row * nblk + blk;
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int idx = offset + 4 * k;
                const int group = idx / 32;
                const float d8 = xs[blk * 8 + group];
                const int xint = *(const int*)(xq + blk * QK_K + idx);
                int wint = *(const int*)(bq->qs + (group / 2 * 32 + (idx & 31)));
                if (group & 1) wint >>= 4;
                wint &= 0x0F0F0F0F;
                const int qh4 = *(const int*)(bq->qh + (idx & 31));
                #pragma unroll
                for (int e = 0; e < 4; ++e) wint |= ((qh4 >> (8 * e + group)) & 1) << (8 * e + 4);
                uint8_t scale, min;
                get_scale_min_k4(group, bq->scales, &scale, &min);
                const int dot  = __dp4a(wint, xint, 0);
                const int xsum = __dp4a(0x01010101, xint, 0);
                acc += (float)bq->d * (scale * d8 * dot) - (float)bq->dmin * (min * d8 * xsum);
            }
        }
    } else if constexpr (TYPE == DType::Q6_K) {
        const int nblk = n_in / QK_K;
        const int blocks_per_iter = nthreads / 32;
        const int kbx_start = tid / 32;
        const int iqs = tid % 32;

        const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8; // {0,1,4,5}, each thread handles 2 q8_1 sub-blocks, stride 2
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4; // {0,1,2,3,8,9,10,11}, stride 4
        const int vh_shift = 2 * ((iqs % 16) / 8); // {0,2}
        const int qh_iqs = 8 * (iqs / 16) + iqs % 8;

        for (int kbx = kbx_start; kbx < nblk; kbx += blocks_per_iter) {
            const block_q6_K* bq = (const block_q6_K*)W + (size_t)row * nblk + kbx;

            // Load 4 bytes of ql/qh
            const int vl = get_int_b2(bq->ql, iqs);
            const int vh = get_int_b2(bq->qh, qh_iqs) >> vh_shift;

            const int8_t* scales = bq->scales + scale_offset;
            const float d = (float)bq->d;

            float sumf = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q8_blk = bq8_offset + 2 * i;
                const int q8_base = kbx * 8 + q8_blk;
                const float d8 = xs[q8_base];
                const int scale = scales[4 * i];
                // Read 4 bytes of activation
                const int xint = *(const int*)(xq + q8_base * 32 + 4 * (iqs % 8));
                
                const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
                const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                const int vi = __vsubss4(vil | vih, 0x20202020); // vi = (vil | vih) - 32

                sumf += d8 * (__dp4a(vi, xint, 0) * scale);
            }
            acc += d * sumf;
        }
    }
    return acc;
}

enum class GemvWrite { SET, ADD, BIAS };

template <DType TYPE, bool ADD>
__launch_bounds__(128, 1)
__global__ void gemv_dp4a_k(const void* __restrict__ W, const int8_t* __restrict__ xq,
                            const float* __restrict__ xs, float* __restrict__ y, int n_in) {
    const int warp_id = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    constexpr int nwarps = 4;
    const float acc = gemv_dp4a_dot<TYPE>(W, row, xq, xs, n_in, threadIdx.x, 128);

    __shared__ float tmp_shared[nwarps - 1][32];
    if (warp_id > 0) tmp_shared[warp_id - 1][lane] = acc;
    __syncthreads();
    if (warp_id > 0) return;

    float sum = acc;
    #pragma unroll
    for (int w = 0; w < nwarps - 1; ++w) {
        sum += tmp_shared[w][lane];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        if constexpr (ADD) y[row] += sum;
        else               y[row] = sum;
    }
}

// small_k version: each warp independently processes multiple rows (grid-stride); one block has 4 warps.
template <DType TYPE, GemvWrite WM>
__launch_bounds__(128, 1)
__global__ void gemv_dp4a_smallk_k(const void* __restrict__ W, const int8_t* __restrict__ xq,
                                    const float* __restrict__ xs,
                                    const float* __restrict__ bias, float* __restrict__ y, int n_in, int n_out) {
    const int warp_id = threadIdx.x / 32, lane = threadIdx.x & 31;
    for (int row = blockIdx.x * 4 + warp_id; row < n_out; row += gridDim.x * 4) {
        const float acc = gemv_dp4a_dot<TYPE>(W, row, xq, xs, n_in, lane, 32);
        const float s = warp_reduce_sum(acc);
        if (lane == 0) {
            if constexpr (WM == GemvWrite::SET)  y[row] = s;
            if constexpr (WM == GemvWrite::ADD)  y[row] += s;
            if constexpr (WM == GemvWrite::BIAS) y[row] = s + bias[row];
        }
    }
}

// Fused gate+up+swiglu kernel (small_k): activation is shared between gate and up
// accumulate gate=dot(Wg[j], x), up=dot(Wu[j], x), then do silu(gate)*up
template <DType TYPE>
__launch_bounds__(128, 1)
__global__ void gemv_fused_gate_up_swiglu_k(const void* Wg, const void* Wu,
                                             const int8_t* __restrict__ xq,
                                             const float* __restrict__ xs,
                                             float* out, int n_in, int n_out) {
    const int warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * 4 + warp_id;
    if (row >= n_out) return;

    float gate_acc = 0.0f, up_acc = 0.0f;

    if constexpr (TYPE == DType::Q4_K) {
        const int nblk = n_in / QK_K;
        const int blocks_per_iter = 32 / 16;
        const int kbx_start = lane / 16;
        const int iqs = 2 * (lane % 16);
        const int bq8_offset = 2 * (iqs / 8);

        for (int kbx = kbx_start; kbx < nblk; kbx += blocks_per_iter) {
            // Load activation once - shared between gate and up
            const int q8_base = kbx * 8 + bq8_offset;
            float d8[2];
            int x0[2], x1[2], xsum[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int q8_idx = q8_base + i;
                d8[i] = xs[q8_idx];
                const int* x_base = (const int*)(xq + q8_idx * 32 + 4 * (lane & 3));
                x0[i] = x_base[0];
                x1[i] = x_base[4];
                xsum[i] = __dp4a(0x01010101, x1[i], __dp4a(0x01010101, x0[i], 0));
            }
            const block_q4_K* bq_g = (const block_q4_K*)Wg + (size_t)row * nblk + kbx;
            const block_q4_K* bq_u = (const block_q4_K*)Wu + (size_t)row * nblk + kbx;

            const int* q4_g = (const int*)(bq_g->qs + 16 * bq8_offset + 4 * (lane & 3));
            const int w0_g = q4_g[0], w1_g = q4_g[4];
            const int* q4_u = (const int*)(bq_u->qs + 16 * bq8_offset + 4 * (lane & 3));
            const int w0_u = q4_u[0], w1_u = q4_u[4];

            // decode scales
            const uint16_t* scales_g = (const uint16_t*)bq_g->scales;
            const uint16_t* scales_u = (const uint16_t*)bq_u->scales;
            uint16_t aux_g[2], aux_u[2];
            const int j = bq8_offset / 2;
            if (j < 2) {
                aux_g[0] = scales_g[j + 0] & 0x3f3f;
                aux_g[1] = scales_g[j + 2] & 0x3f3f;
                aux_u[0] = scales_u[j + 0] & 0x3f3f;
                aux_u[1] = scales_u[j + 2] & 0x3f3f;
            } else {
                aux_g[0] = ((scales_g[j + 2] >> 0) & 0x0f0f) | ((scales_g[j - 2] & 0xc0c0) >> 2);
                aux_g[1] = ((scales_g[j + 2] >> 4) & 0x0f0f) | ((scales_g[j - 0] & 0xc0c0) >> 2);
                aux_u[0] = ((scales_u[j + 2] >> 0) & 0x0f0f) | ((scales_u[j - 2] & 0xc0c0) >> 2);
                aux_u[1] = ((scales_u[j + 2] >> 4) & 0x0f0f) | ((scales_u[j - 0] & 0xc0c0) >> 2);
            }
            const uint8_t* scale_arr_g = (const uint8_t*)aux_g;
            const uint8_t* min_arr_g   = scale_arr_g + 2;
            const uint8_t* scale_arr_u = (const uint8_t*)aux_u;
            const uint8_t* min_arr_u   = scale_arr_u + 2;

            float sumf_d_g = 0.0f, sumf_m_g = 0.0f;
            float sumf_d_u = 0.0f, sumf_m_u = 0.0f;
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int w0i_g = (w0_g >> (4 * i)) & 0x0F0F0F0F;
                const int w1i_g = (w1_g >> (4 * i)) & 0x0F0F0F0F;
                const int dot_g = __dp4a(w1i_g, x1[i], __dp4a(w0i_g, x0[i], 0));

                const int w0i_u = (w0_u >> (4 * i)) & 0x0F0F0F0F;
                const int w1i_u = (w1_u >> (4 * i)) & 0x0F0F0F0F;
                const int dot_u = __dp4a(w1i_u, x1[i], __dp4a(w0i_u, x0[i], 0));

                sumf_d_g += d8[i] * (dot_g * scale_arr_g[i]);
                sumf_m_g += d8[i] * (xsum[i] * min_arr_g[i]);
                sumf_d_u += d8[i] * (dot_u * scale_arr_u[i]);
                sumf_m_u += d8[i] * (xsum[i] * min_arr_u[i]);
            }
            const uint32_t dm_g = *reinterpret_cast<const uint32_t*>(&bq_g->d);
            const uint32_t dm_u = *reinterpret_cast<const uint32_t*>(&bq_u->d);
            gate_acc += low2float(dm_g) * sumf_d_g - high2float(dm_g) * sumf_m_g;
            up_acc   += low2float(dm_u) * sumf_d_u - high2float(dm_u) * sumf_m_u;
        }
    } else {
        gate_acc = gemv_dp4a_dot<TYPE>(Wg, row, xq, xs, n_in, lane, 32);
        up_acc   = gemv_dp4a_dot<TYPE>(Wu, row, xq, xs, n_in, lane, 32);
    }

    const float gate_val = warp_reduce_sum(gate_acc);
    const float up_val   = warp_reduce_sum(up_acc);
    if (lane == 0) {
        const float g = gate_val;
        out[row] = g / (1.0f + __expf(-g)) * up_val;
    }
}

// Regular-type GEMV (F32/F16); grid: n_out, block: 256 threads; grid-stride
template <DType TYPE>
__global__ void gemv_plain_k(const void* W, const float* x, float* y, int n_in) {
    const int row = blockIdx.x;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) {
        float v;
        if constexpr (TYPE == DType::F32) {
            v = ((const float*) W)[(size_t) row * n_in + i];
        } else if constexpr (TYPE == DType::F16) {
            v = __half2float(((const __half*) W)[(size_t) row * n_in + i]);
        }
        acc += v * x[i];
    }
    const float s = block_reduce_sum(acc);
    if (threadIdx.x == 0) y[row] = s;
}

// --------- MMA GEMM kernels (Q4K/Q6K, prefill stream-K) -----------
//
// Y input: block_q8_1_mmq format, pre-quantized by quantize_mmq_q8_1_k
// Stream-K: each CUDA block processes a range [kbc, kbc_stop) in continuous ijk space.
// The last partial tile writes to tmp_fixup; mul_mat_mmq_fixup sums them.

constexpr int MMQ_Y         = 128;
constexpr int MMQ_X_STRIDE  = 76;   // 64(qx) + 1(dx) + 4(scales) + 7(padding)
constexpr int MMQ_TILE_Y_K  = 36;   // 32(qy) + 4(dy) = 36 ints per block_q8_1_mmq
constexpr int MMQ_QI8_1     = 8;

static __device__ __forceinline__ void ldmatrix_x2(int (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "l"(p));
}

static __device__ __forceinline__ void ldmatrix_x4(int (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

static __device__ __forceinline__ void ldmatrix_x4_trans(int (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[2]), "=r"(r[1]), "=r"(r[3]) : "l"(p));
}

static __device__ __forceinline__ int movmatrix_trans(int src) {
    int dst;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(dst) : "r"(src));
    return dst;
}

static __device__ __forceinline__ void mma_s8s8_m16n8k16(int (&C)[4], int A0, int A1, int B) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
        : "r"(A0), "r"(A1), "r"(B));
}

static __device__ __forceinline__ void mma_s8s8_m16n8k32(int (&C)[4], const int (&A)[4], int B0, int B1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B0), "r"(B1));
}

__device__ __forceinline__ int unpack_scales_q45_K(const int* scales, const int ksc) {
    // ksc==0: sc[0-3],  ksc==1: sc[4-7],  ksc==2: m[0-3],  ksc==3: m[4-7]
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, bool need_check>
__device__ __forceinline__ void load_tiles_q4_K(
    const void* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = 8;
    constexpr int warp_size = 32;

    int*   x_qs = x_tile;
    half2* x_dm = (half2*)(x_qs + 64);

    constexpr int threads_per_row = 256 / 8;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = threadIdx.x % threads_per_row;

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);

        if (need_check) i = min(i, i_max);

        const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i * stride;
        const int qs0 = ((const int*)(bxi->qs))[txi];

        x_qs[i * MMQ_X_STRIDE + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i * MMQ_X_STRIDE + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    constexpr int rows_per_warp = warp_size / 2;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + threadIdx.x / 2) % mmq_y;

        if (need_check) i = min(i, i_max);

        const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i * stride;
        const int* scales = (const int*) bxi->scales;
        const int ksc = threadIdx.x % 2;

        const int sc32 = unpack_scales_q45_K(scales, ksc + 0); // d[0-3] or d[4-7]
        const int  m32 = unpack_scales_q45_K(scales, ksc + 2); // m[0-3] or m[4-7]

        const uint8_t* sc8 = (const uint8_t*)&sc32;
        const uint8_t* m8 = (const uint8_t*)&m32;

        const __half2 dm = __hmul2(*(const __half2*)&bxi->d, make_half2(1.0f, -1.0f));

        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            half2 val = __hmul2(dm, make_half2(sc8[l], m8[l]));
            x_dm[i * MMQ_X_STRIDE + 4 * ksc + l] = val;
        }
    }
}

template <int mmq_x, int mmq_y>
__device__ __forceinline__ void vec_dot_q4k_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {

    constexpr int granularity = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / 16; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (8 * MMQ_TILE_Y_K); // 8 cols in batch dim

    const int* x_qs = (const int*)x;
    const half2* x_dm = (const half2*)x_qs + 64;
    const int* y_qs = (const int*)y + 4;
    const half2* y_dm = (const half2*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp; // row index for this warp
    const int lane = threadIdx.x;

    int A_frag[ntx][32/MMQ_QI8_1][4]; // [mini-tile][Kchunk][4 regs]
    float2 dmA[ntx][2][32/MMQ_QI8_1]; // output C layout, [mini-tile][group(8 rows)][Kchunk] (d*scale, -dmin*min)

    #pragma unroll
    for (int n = 0; n < ntx; ++n) {
        #pragma unroll
        for (int k01 = 0; k01 < 32; k01 += MMQ_QI8_1) { // process 128 k elements for 4 iterations
            const int k0 = k00 + k01;
            ldmatrix_x4(A_frag[n][k01/MMQ_QI8_1], x_qs + (i0 + n * 16 + lane % 16) * MMQ_X_STRIDE + (lane / 16) * 4 + k0);
        }

        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int i = i0 + n * 16 + (l * 8 + lane / 4);
            #pragma unroll
            for (int k01 = 0; k01 < 32; k01 += MMQ_QI8_1) {
                const int k0 = k00 + k01;
                dmA[n][l][k01/MMQ_QI8_1] = __half22float2(x_dm[i*MMQ_X_STRIDE + k0/MMQ_QI8_1]);
            }
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * 8) {
        #pragma unroll
        for (int k01 = 0; k01 < 32; k01 += MMQ_QI8_1) {
            const int kq = k01 / MMQ_QI8_1;
            const int y_base = (j0 + lane / 4) * MMQ_TILE_Y_K + (k01 + lane % 4);
            const int B0 = y_qs[y_base];
            const int B1 = y_qs[y_base + 4];

            float2 dsB[2]; // output C layout, from 2t/2t+1 col
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                dsB[l] = __half22float2(y_dm[(j0 + (lane % 4) * 2 + l) * MMQ_TILE_Y_K + kq]);
            }

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                int C[4] = {0, 0, 0, 0};
                mma_s8s8_m16n8k32(C, A_frag[n][kq], B0, B1);

                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const float2 dm = dmA[n][l/2][kq];
                    const float2 ds = dsB[l%2];
                    const int idx = (j0/8 + n) * 4 + l;
                    sum[idx] += dm.x * ds.x * (float)C[l];
                    sum[idx] += dm.y * ds.y;
                }
            }
        }
    }
}

template <int mmq_y, bool need_check>
__device__ __forceinline__ void load_tiles_q6_K(
    const void* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = 8;
    constexpr int warp_size = 32;

    constexpr int threads_per_row = 256 / 8;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = threadIdx.x % threads_per_row;

    int* x_qs = x_tile;
    float* x_df = (float*)(x_qs + 64);
    int* x_sc = (int*)(x_df + 1);

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);
        if (need_check) i = min(i, i_max);

        const block_q6_K* bxi = (const block_q6_K*)x + kbx0 + i * stride;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, 8 * (txi / 16) + txi % 8);
        const int shift = (txi & 0x08) >> 2;
        const int qh0 = ((qh >> shift) << 4) & 0x30303030;
        const int qh1 = (qh >> shift) & 0x30303030;

        const int kq0 = txi + (txi & ~15);
        const int kq1 = kq0 + 16;

        x_qs[i * MMQ_X_STRIDE + kq0] = __vsubss4(ql0 | qh0, 0x20202020); // sub 32
        x_qs[i * MMQ_X_STRIDE + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * warp_size) {
        int i = (i0 + threadIdx.y * warp_size + threadIdx.x) % mmq_y;
        if (need_check) i = min(i, i_max);

        const block_q6_K* bxi = (const block_q6_K*)x + kbx0 + i * stride;
        x_df[i * MMQ_X_STRIDE] = (float)bxi->d;
    }

    constexpr int rows_per_warp = warp_size / 4;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = (i0 + threadIdx.y * rows_per_warp + threadIdx.x / 4) % mmq_y;

        if (need_check) i = min(i, i_max);

        const block_q6_K* bxi = (const block_q6_K*)x + kbx0 + i * stride + (threadIdx.x % 4) / 4;
        x_sc[i * MMQ_X_STRIDE + threadIdx.x % 4] = get_int_b2(bxi->scales, threadIdx.x % 4);
    }
}

template <int mmq_x, int mmq_y>
__device__ __forceinline__ void vec_dot_q6k_mma(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {

    constexpr int granularity = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / 16; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (8 * MMQ_TILE_Y_K); // 8 cols in batch dim

    const int* x_qs = x;
    const float* x_df = (const float*)(x_qs + 64);
    const int* x_sc = (const int*)(x_df + 1);
    const int* y_qs = y + 4;
    const float* y_df = (const float*)y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp; // row index for this warp
    const int lane = threadIdx.x;

    int A_frag[ntx][8][2]; // [mini-tile][Kchunk][2 regs]
    int scA[ntx][2][8]; // output C layout, [mini-tile][group(8 rows)][Kchunk]
    float dA[ntx][2]; // output C layout, [mini-tile][group(8 rows)]

    #pragma unroll
    for (int n = 0; n < ntx; ++n) {
        #pragma unroll
        for (int k01 = 0; k01 < 32; k01 += 8) {
            const int k0 = k00 + k01;
            const int* x_qs_base = x_qs + (i0 + n * 16 + lane % 16) * MMQ_X_STRIDE + k0;
            ldmatrix_x2(A_frag[n][k01/4], x_qs_base);
            ldmatrix_x2(A_frag[n][k01/4 + 1], x_qs_base + 4);
        }

        #pragma unroll
        for (int k01 = 0; k01 < 32; k01 += 16) {
            const int k0 = k00 + k01;
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int i = i0 + n * 16 + (l * 8 + lane / 4);
                const int sc_packed = x_sc[i * MMQ_X_STRIDE + k0 / 16];
                const int8_t* sc = (const int8_t*)&sc_packed;
                #pragma unroll
                for (int ksc = 0; ksc < 4; ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            dA[n][l] = x_df[(i0 + n * 16 + l * 8 + lane / 4) * MMQ_X_STRIDE];
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * 8) {
        float tmp[ntx][4] = {}; // [mini-tile][4 results]

        #pragma unroll
        for (int k01 = 0; k01 < 32; k01 += 8) {
            const int y_base = (j0 + lane / 4) * MMQ_TILE_Y_K + (k01 + lane % 4);
            const int B0 = y_qs[y_base];
            const int B1 = y_qs[y_base + 4];

            float dB[2]; // output C layout, from 2t/2t+1 col
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                dB[l] = y_df[(j0 + (lane % 4) * 2 + l) * MMQ_TILE_Y_K + k01 / MMQ_QI8_1];
            }

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                int C0[4] = {0, 0, 0, 0};
                int C1[4] = {0, 0, 0, 0};
                mma_s8s8_m16n8k16(C0, A_frag[n][k01/4 + 0][0], A_frag[n][k01/4 + 0][1], B0);
                mma_s8s8_m16n8k16(C1, A_frag[n][k01/4 + 1][0], A_frag[n][k01/4 + 1][1], B1);

                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    tmp[n][l] += (float)(C0[l] * scA[n][l/2][k01/4 + 0]
                                       + C1[l] * scA[n][l/2][k01/4 + 1]) * dB[l % 2];
                }
            }
        }

        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                sum[(j0/8 + n) * 4 + l] += tmp[n][l] * dA[n][l/2];
            }
        }
    }
}

template <DType type> struct MmqTrait;

template <> struct MmqTrait<DType::Q4_K> {
    template <int mmq_y, bool need_check>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
        load_tiles_q4_K<mmq_y, need_check>(x, x_tile, kbx0, i_max, stride);
    }
    template <int mmq_x, int mmq_y>
    static __device__ __forceinline__ void vec_dot_mma(
        const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {
        vec_dot_q4k_mma<mmq_x, mmq_y>(x, y, sum, k00);
    }
};

template <> struct MmqTrait<DType::Q6_K> {
    template <int mmq_y, bool need_check>
    static __device__ __forceinline__ void load_tiles(
        const void* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
        load_tiles_q6_K<mmq_y, need_check>(x, x_tile, kbx0, i_max, stride);
    }
    template <int mmq_x, int mmq_y>
    static __device__ __forceinline__ void vec_dot_mma(
        const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {
        vec_dot_q6k_mma<mmq_x, mmq_y>(x, y, sum, k00);
    }
};

template <int mmq_x, int mmq_y, bool need_check>
__device__ __forceinline__ void mmq_write_back(
    const float* __restrict__ sum, float* __restrict__ dst, const int stride, const int i_max, const int j_max) {

    constexpr int tile_I = 16, tile_J = 8, tile_ne = 4;
    constexpr int granularity = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / tile_I;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp; // row index for this warp

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_ne; ++l) {
                const int i = i0 + n * tile_I + (l / 2) * 8 + threadIdx.x / 4;
                const int j = j0 + (threadIdx.y % ntx) * tile_J + (threadIdx.x % 4) * 2 + l % 2;

                if (j > j_max) continue;
                if (need_check && i > i_max) continue;

                dst[(size_t)j * stride + i] = sum[(j0/tile_J + n) * tile_ne + l]; // column-major
            }
        }
    }
}

template <DType type, int mmq_x, int mmq_y, bool need_check, bool fixup>
__device__ __forceinline__ void mmq_process_tile(
    const void* __restrict__ W, const int* __restrict__ y,
    float* __restrict__ dst, float* __restrict__ tmp_fixup,
    const int stride_row_x, const int ncols_y, const int nrows_x,
    const int tile_x_max_i, const int tile_y_max_j,
    const int it, const int jt, const int kb0_start, const int kb0_stop) {

    constexpr int nwarps = 8, warp_size = 32;
    constexpr int nthreads = nwarps * warp_size;

    extern __shared__ int smem_mmq[]; // tile_y(B) + tile_x(A)
    int* tile_y = smem_mmq;
    constexpr int tile_y_ints = (mmq_x * MMQ_TILE_Y_K + nthreads - 1) / nthreads * nthreads;
    int* tile_x = tile_y + tile_y_ints;

    float sum[mmq_x * mmq_y / nthreads] = {0.0f};

    const int offset_x = it * mmq_y * stride_row_x;

    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        MmqTrait<type>::template load_tiles<mmq_y, need_check>((const char*)W, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

        #pragma unroll
        for (int kpart = 0; kpart < 2; ++kpart) { // divide 256 k-dim elements into 2 parts
            const int* by0 = y + ncols_y * (kb0 * 2 + kpart) * MMQ_TILE_Y_K;
            #pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nthreads) {
                int l = l0 + threadIdx.y * warp_size + threadIdx.x;
                tile_y[l] = by0[l];
            }
            __syncthreads();

            MmqTrait<type>::template vec_dot_mma<mmq_x, mmq_y>(tile_x, tile_y, sum, kpart * 32);
            __syncthreads();
        }
    }

    if (fixup) {
        mmq_write_back<mmq_x, mmq_y, false>(
            sum, tmp_fixup + blockIdx.x * (mmq_x * mmq_y), mmq_y, mmq_y - 1, mmq_x - 1);
    } else {
        mmq_write_back<mmq_x, mmq_y, need_check>(
            sum, dst + (size_t)jt * mmq_x * nrows_x + it * mmq_y, nrows_x, tile_x_max_i, tile_y_max_j);
    }
}

template <DType type, int mmq_x, bool need_check>
__global__ void __launch_bounds__(256, 1)
mul_mat_mmq(const void* __restrict__ W, const int* __restrict__ Y_q8,
            float* __restrict__ dst, float* __restrict__ tmp_fixup,
            int nrows_x, int ncols_x, int ncols_y, int stride_row_x,
            uint3 blocks_per_ne00_fd, uint3 ntx_tiles_fd, int nsm) {
    const int blocks_per_ne00 = blocks_per_ne00_fd.z;
    const int nty_tiles = (nrows_x + MMQ_Y - 1) / MMQ_Y;
    const int ntx_tiles = ntx_tiles_fd.z;

    const int total_tiles = ntx_tiles * nty_tiles;
    const int total_k_blocks = total_tiles * blocks_per_ne00;

    // block_i: [kbc, kbc_stop)
    int kbc      = (int)((int64_t)blockIdx.x       * total_k_blocks / gridDim.x);
    int kbc_stop = (int)((int64_t)(blockIdx.x + 1) * total_k_blocks / gridDim.x);

    int kb0_start = fastmodulo(kbc, blocks_per_ne00_fd);
    int kb0_stop  = min(blocks_per_ne00, kb0_start + (kbc_stop - kbc));

    auto dispatch = [&](int kbc_cur, int kb0_s, int kb0_e, auto fixup_tag) {
        const int tile_idx = fastdiv(kbc_cur, blocks_per_ne00_fd);
        const int it = fastdiv(tile_idx, ntx_tiles_fd);
        const int jt = tile_idx - it * ntx_tiles;
        const int offset_y = jt * mmq_x * MMQ_TILE_Y_K;
        mmq_process_tile<type, mmq_x, MMQ_Y, need_check, decltype(fixup_tag)::value>(
            W, Y_q8 + offset_y, dst, tmp_fixup,
            stride_row_x, ncols_y, nrows_x,
            nrows_x - it * MMQ_Y - 1, ncols_y - jt * mmq_x - 1,
            it, jt, kb0_s, kb0_e);
    };

    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        dispatch(kbc, kb0_start, kb0_stop, std::false_type{});

        kbc = (fastdiv(kbc, blocks_per_ne00_fd) + 1) * blocks_per_ne00; // process next tile
        kb0_start = 0;
        kb0_stop = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc < kbc_stop) dispatch(kbc, kb0_start, kb0_stop, std::true_type{}); // write to tmp_fixup
}

template <int mmq_x, int mmq_y, bool need_check>
__global__ void __launch_bounds__(128, 1)
mul_mat_mmq_fixup(float* __restrict__ dst, float* __restrict__ tmp_fixup,
                   int nrows_x, int ncols_y, uint3 blocks_per_ne00_fd, uint3 ntx_tiles_fd, int nsm) {
    constexpr int nwarps = 4;
    constexpr int warp_size = 32;
    const int blocks_per_ne00 = blocks_per_ne00_fd.z;

    const int nty = (nrows_x + mmq_y - 1) / mmq_y;
    const int total_tiles = ((ncols_y + mmq_x - 1) / mmq_x) * nty;
    const int total_k_blocks = total_tiles * blocks_per_ne00;

    float sum[mmq_x / nwarps] = {0.0f};
    const int i = blockIdx.y * warp_size + threadIdx.x;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = (int)((int64_t)bidx0       * total_k_blocks / gridDim.x);
    int kbc0_stop = (int)((int64_t)(bidx0 + 1) * total_k_blocks / gridDim.x);

    const int tile_idx_kbc0 = fastdiv(kbc0, blocks_per_ne00_fd);

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00_fd) == 0;
    const bool did_not_write_last      = tile_idx_kbc0 == fastdiv(kbc0_stop, blocks_per_ne00_fd) && fastmodulo(kbc0_stop, blocks_per_ne00_fd) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) return;

    bool any_fixup = false;
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while (true) {
        int kbc = (int)((int64_t)bidx * total_k_blocks / gridDim.x);

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            sum[j0 / nwarps] += tmp_fixup[bidx * (mmq_x * mmq_y) + j * mmq_y + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00_fd) == 0 || fastdiv(kbc, blocks_per_ne00_fd) < tile_idx_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) return;

    const int it = fastdiv(tile_idx_kbc0, ntx_tiles_fd);
    const int jt = tile_idx_kbc0 - it * ntx_tiles_fd.z;

    const int i_max = nrows_x - it * mmq_y - 1;
    const int j_max = ncols_y - jt * mmq_x - 1;

    if (need_check && i > i_max) return;

    const int offset_dst = jt * mmq_x * nrows_x + it * mmq_y;

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) return;
        dst[offset_dst + (size_t)j * nrows_x + i] += sum[j0 / nwarps];
    }
}

// ---------------- Flash attention kernels --------------------

static __device__ __forceinline__ void fa_cp_async_16(void* smem, const void* gmem) {
    unsigned int s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}

static __device__ __forceinline__ void fa_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

static __device__ __forceinline__ void fa_cp_async_wait() {
    asm volatile("cp.async.wait_all;");
}

// MMA m16n8k16 with f32 accumulation
static __device__ __forceinline__ void fa_mma16816(float* D, const int (&A)[4], const int* B) {
    int* d = (int*) D;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

static __device__ __forceinline__ void fa_exp_to_mma_b(int* B, const float* e) {
    half2 h0 = make_half2(__float2half(e[0]), __float2half(e[1]));
    half2 h1 = make_half2(__float2half(e[2]), __float2half(e[3]));
    B[0] = movmatrix_trans(*(int*)&h0);
    B[1] = movmatrix_trans(*(int*)&h1);
}

// Q-column index for two m16n8k16 MMA C fragment elements: {0,1,0,1,2,3,2,3}.
static __device__ __forceinline__ int fa_col8(int c) {
    return (c / 4) * 2 + (c % 2);
}

static constexpr float SOFTMAX_FTZ_THRESHOLD = -20.0f; // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.
static constexpr float FA_KQ_MAX_OFFSET = 3.0f * 0.6931f; // log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
static constexpr int FA_PREFILL_NBATCH = 64; // KV positions per main-loop iteration

// MMA-based flash attention decode
// Grid = n_kv blocks, block = 4 warps. 4 warps cooperating on same Q columns.
template <int HDIM>
__launch_bounds__(128, 1)
__global__ void flash_attention_decode_mma_k(
        const float* __restrict__ q,
        const uint16_t* __restrict__ kc,
        const uint16_t* __restrict__ vc,
        float* __restrict__ out,
        int n_head, int n_kv, int head_dim,
        int n_ctx, const int* __restrict__ d_pos, float scale) {

    constexpr int NCOLS    = 8;   // Q columns per block (padded GQA ratio)
    constexpr int NBATCH   = 128; // KV positions per main-loop iteration
    constexpr int DITER    = HDIM / 16;
    constexpr int FA_STRIDE = HDIM / 2 + 4; // row interval of shared memory, +4 is bank-conflict padding
    constexpr int NWARPS   = 4;
    constexpr int WARP_KV  = NBATCH / NWARPS; // 32 KV positions per warp
    constexpr int NTILES   = 32 / 16;   // MMA tiles per warp

    const int kv_head    = blockIdx.x;
    const int lane       = threadIdx.x;  // 0..31
    const int warp_id    = threadIdx.y;  // 0..3
    const int tid        = warp_id * 32 + lane;
    const int npos       = *d_pos + 1;
    const int gqa_ratio  = n_head / n_kv;
    const int q_head0    = kv_head * gqa_ratio;

    const __half* K_base = (const __half*)kc + (size_t)kv_head * n_ctx * HDIM;
    const __half* V_base = (const __half*)vc + (size_t)kv_head * n_ctx * HDIM;

    extern __shared__ char fa_shared_raw[];
    half2* tile_K = (half2*)fa_shared_raw;
    half2* tile_V = tile_K + NBATCH * FA_STRIDE;
    half2* tile_Q = tile_V; // alias for Q loading

    // Load Q into shared memory
    {
        const __half2 sc = __float2half2_rn(scale);
        for (int idx = tid; idx < NCOLS * (HDIM / 2); idx += 128) {
            const int col = idx / (HDIM / 2); // Q col 0...7
            const int k   = idx % (HDIM / 2); // Q half2 offset 0...HDIM/2
            half2 val;
            if (col < gqa_ratio) {
                const float* qptr = q + (size_t)(q_head0 + col) * HDIM + k * 2;
                val = __hmul2(sc, make_half2(__float2half(qptr[0]), __float2half(qptr[1])));
            } else {
                val = __float2half2_rn(0.0f);
            }
            tile_Q[col * FA_STRIDE + k] = val;
        }
    }
    __syncthreads();

    // Load Q into registers
    int Q_B[DITER][2]; // DITER MMA tiles per warp
    #pragma unroll
    for (int d = 0; d < DITER; ++d) {
        const int* tile_base = (const int*)tile_Q + d * 8;
        ldmatrix_x2(Q_B[d], tile_base + (lane % 8) * FA_STRIDE + (lane / 8) * 4);
    }
    __syncthreads();

    float S_max[2]    = {-FLT_MAX/2, -FLT_MAX/2};
    float S_rowsum[2] = {0.0f, 0.0f};
    float VKQ[DITER][4] = {}; // each warp outputs [HDIM, 8], DITER MMA tiles(16 x 8)

    auto load_tile_async = [&](half2* tile, const __half* base, int j0_arg) {
        constexpr int CHUNKS = HDIM / 8;
        #pragma unroll
        for (int idx = tid; idx < NBATCH * CHUNKS; idx += 128) {
            const int row = idx / CHUNKS, chunk = idx % CHUNKS;
            half2* dst = tile + row * FA_STRIDE + chunk * 4;
            fa_cp_async_16(dst, base + (size_t)(j0_arg + row) * HDIM + chunk * 8);
        }
        fa_cp_async_commit();
    };

    load_tile_async(tile_K, K_base, 0); // async load 128 row

    for (int j0 = 0; j0 < npos; j0 += NBATCH) {
        const bool last_iter = (j0 + NBATCH >= npos);

        fa_cp_async_wait();
        __syncthreads();

        load_tile_async(tile_V, V_base, j0);

        // KQ = K * Q^T (2 MMA tiles per warp)
        float KQ_C[NTILES][4] = {};
        #pragma unroll
        for (int d = 0; d < DITER; ++d) {
            #pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                int K_A[4];
                const int* tile_ka_base = (const int*)tile_K + (warp_id * WARP_KV + t * 16) * FA_STRIDE + d * 8;
                ldmatrix_x4(K_A, tile_ka_base + (lane % 16) * FA_STRIDE + (lane / 16) * 4);
                fa_mma16816(KQ_C[t], K_A, Q_B[d]);
            }
        }

        // Mask OOB positions
        #pragma unroll
        for (int t = 0; t < NTILES; ++t) {
            const int base_kv = j0 + warp_id * WARP_KV + t * 16;
            const int g = lane / 4; // MMA group index
            if (base_kv + g >= npos) KQ_C[t][0] = KQ_C[t][1] = -INFINITY;
            if (base_kv + g + 8 >= npos) KQ_C[t][2] = KQ_C[t][3] = -INFINITY;
        }

        // Online softmax
        float new_max[2] = {S_max[0], S_max[1]};
        #pragma unroll
        for (int t = 0; t < NTILES; ++t)
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                new_max[l % 2] = fmaxf(new_max[l % 2], KQ_C[t][l] + FA_KQ_MAX_OFFSET);

        #pragma unroll
        for (int off = 16; off >= 4; off >>= 1)
            #pragma unroll
            for (int col = 0; col < 2; ++col)
                new_max[col] = fmaxf(new_max[col], __shfl_xor_sync(0xFFFFFFFF, new_max[col], off));

        int sKQ_B[NTILES][2];
        float KQ_rowsum_add[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int t = 0; t < NTILES; ++t) {
            float KQ_exp[4];
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                KQ_exp[l] = __expf(KQ_C[t][l] - new_max[l % 2]);
                KQ_rowsum_add[l % 2] += KQ_exp[l];
            }
            fa_exp_to_mma_b(sKQ_B[t], KQ_exp); // convert softmax(KQ^T) to MMA B operand via movmatrix
        }

        float rescale[2];
        #pragma unroll
        for (int col = 0; col < 2; ++col) {
            const float diff = S_max[col] - new_max[col];
            rescale[col] = __expf(diff);
            *(uint32_t*)&rescale[col] *= diff >= SOFTMAX_FTZ_THRESHOLD;
            S_max[col] = new_max[col];
            S_rowsum[col] = rescale[col] * S_rowsum[col] + KQ_rowsum_add[col];
        }

        #pragma unroll
        for (int l = 0; l < DITER * 4; ++l)
            ((float*)VKQ)[l] *= rescale[l % 2];

        fa_cp_async_wait(); // wait for V
        __syncthreads();

        if (!last_iter) load_tile_async(tile_K, K_base, j0 + NBATCH); // start next K load

        // VKQ += V^T * softmax(KQ)
        #pragma unroll
        for (int d = 0; d < DITER; ++d) {
            #pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                int V_A[4];
                const int* tile_va_base = (const int*)tile_V + (warp_id * WARP_KV + t * 16) * FA_STRIDE + d * 8;
                ldmatrix_x4_trans(V_A, tile_va_base + (lane % 16) * FA_STRIDE + (lane / 16) * 4);
                fa_mma16816(VKQ[d], V_A, sKQ_B[t]);
            }
        }
    }

    // Softmax rowsum reduction within warp
    #pragma unroll
    for (int off = 16; off >= 4; off >>= 1)
        #pragma unroll
        for (int col = 0; col < 2; ++col)
            S_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, S_rowsum[col], off);

    // Cross-warp reduction
    {
        float* shared = (float*)fa_shared_raw;
        if (lane / 4 == 0) { // 4 lanes write 8 S_max/S_rowsum to shared memory
            const int qcol0 = (lane % 4) * 2;
            #pragma unroll
            for (int col = 0; col < 2; ++col) {
                shared[warp_id * 8 + qcol0 + col] = S_max[col];
                shared[32 + warp_id * 8 + qcol0 + col] = S_rowsum[col];
            }
        }

        float* sm_vkq = shared + 64; // layout [warp,lane,d,l]
        #pragma unroll
        for (int d = 0; d < DITER; ++d)
            #pragma unroll
            for (int l = 0; l < 4; ++l)
                sm_vkq[warp_id * (32 * DITER * 4) + lane * (DITER * 4) + d * 4 + l] = VKQ[d][l];

        __syncthreads();

        if (warp_id == 0) {
            const int qcol0 = (lane % 4) * 2; // 0,2,4,6,0,2,4,6...

            float KQ_max_all[2] = {shared[qcol0], shared[qcol0 + 1]};
            #pragma unroll
            for (int w = 1; w < NWARPS; ++w)
                #pragma unroll
                for (int col = 0; col < 2; ++col)
                    KQ_max_all[col] = fmaxf(KQ_max_all[col], shared[w * 8 + qcol0 + col]);

            float KQ_rowsum_all[2] = {0.0f, 0.0f};
            float KQ_rescale[NWARPS][2];
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                #pragma unroll
                for (int col = 0; col < 2; ++col) {
                    const float diff = shared[w * 8 + qcol0 + col] - KQ_max_all[col];
                    KQ_rescale[w][col] = __expf(diff);
                    *(uint32_t*)&KQ_rescale[w][col] *= diff >= SOFTMAX_FTZ_THRESHOLD;
                    KQ_rowsum_all[col] += shared[32 + w * 8 + qcol0 + col] * KQ_rescale[w][col];
                }
            }

            float KQ_rowsum_inv[2];
            #pragma unroll
            for (int col = 0; col < 2; ++col)
                KQ_rowsum_inv[col] = (KQ_rowsum_all[col] > 0.0f) ? (1.0f / KQ_rowsum_all[col]) : 0.0f;

            #pragma unroll
            for (int d = 0; d < DITER; ++d) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int col = l % 2;
                    float VKQ_acc = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < NWARPS; ++w)
                        VKQ_acc += sm_vkq[w * (32 * DITER * 4) + lane * (DITER * 4) + d * 4 + l] * KQ_rescale[w][col];

                    const int vdim = d * 16 + lane / 4 + (l / 2) * 8; // convert from MMA output index
                    if (qcol0 + col < gqa_ratio)
                        out[(size_t)(q_head0 + qcol0 + col) * HDIM + vdim] = VKQ_acc * KQ_rowsum_inv[col];
                }
            }
        }
    }
}

// MMA-based flash attention prefill. Each warp handles 16 Q columns = 2 Q tokens x 8 GQA heads (padded), handles all KV positions independently.
// Grid = ceil(batch/8) * n_kv, block = 4 warps. 4 warps = 8 Q tokens per block. 
template <int HDIM>
__launch_bounds__(128, 1)
__global__ void flash_attention_prefill_mma_k(
        const float* __restrict__ q,
        const uint16_t* __restrict__ kc,
        const uint16_t* __restrict__ vc,
        float* __restrict__ out,
        int n_head, int n_kv, int head_dim,
        int n_ctx, int n_past, int batch, float scale) {

    constexpr int QTOKENS_PER_BLOCK = 8;   // Q tokens per block (4 warps * 2)
    constexpr int QTOKENS_PER_WARP  = 2;   // Q tokens per warp
    constexpr int NCOLS_WARP        = 16;  // MMA columns per warp = QTOKENS_PER_WARP * 8
    constexpr int NBATCH    = FA_PREFILL_NBATCH;
    constexpr int DITER     = HDIM / 16;
    constexpr int FA_STRIDE = HDIM / 2 + 4;
    constexpr int NWARPS    = 4;
    constexpr int NTILES    = NBATCH / 16;

    const int kv_head     = blockIdx.x % n_kv;
    const int tile_q_idx  = blockIdx.x / n_kv; // 8-token block index
    const int tile_q_base = tile_q_idx * QTOKENS_PER_BLOCK; // first Q token global index in this block
    const int lane        = threadIdx.x;
    const int warp_id     = threadIdx.y;
    const int tid         = warp_id * 32 + lane;
    const int gqa_ratio   = n_head / n_kv;
    const int q_head0     = kv_head * gqa_ratio;

    const int warp_q0 = tile_q_base + warp_id * QTOKENS_PER_WARP; // first Q token global index in this warp
    const int max_npos = n_past + min(tile_q_base + QTOKENS_PER_BLOCK, batch);
    const int npos0 = (warp_q0 < batch) ? (n_past + warp_q0 + 1) : 0; // visible KV count for Q token 0
    const int npos1 = (warp_q0 + 1 < batch) ? (n_past + warp_q0 + 2) : 0; // visible KV count for Q token 1

    const __half* K_base = (const __half*)kc + (size_t)kv_head * n_ctx * HDIM;
    const __half* V_base = (const __half*)vc + (size_t)kv_head * n_ctx * HDIM;

    // Shared memory: tile_K + tile_V + tile_Q_all
    extern __shared__ char fa_prefill_shared_raw[];
    half2* tile_K     = (half2*)fa_prefill_shared_raw;
    half2* tile_V     = tile_K + NBATCH * FA_STRIDE;
    half2* tile_Q_all = tile_V + NBATCH * FA_STRIDE;

    // Load Q for all warps
    {
        const __half2 sc = __float2half2_rn(scale);
        for (int idx = tid; idx < NWARPS * NCOLS_WARP * (HDIM / 2); idx += 128) {
            const int w   = idx / (NCOLS_WARP * (HDIM / 2));
            const int rem = idx % (NCOLS_WARP * (HDIM / 2));
            const int col = rem / (HDIM / 2); // 0,1...,15. Each warp has 16 Q columns (2 tokens x 8 head).
            const int k   = rem % (HDIM / 2);
            const int qtoken  = col / 8;
            const int gqa_col = col % 8;
            const int global_q = tile_q_base + w * QTOKENS_PER_WARP + qtoken;
            half2 val;
            if (global_q < batch && gqa_col < gqa_ratio) {
                const float* qptr = q + ((size_t)global_q * n_head + q_head0 + gqa_col) * HDIM + k * 2;
                val = __hmul2(sc, make_half2(__float2half(qptr[0]), __float2half(qptr[1])));
            } else {
                val = __float2half2_rn(0.0f);
            }
            tile_Q_all[(w * NCOLS_WARP + col) * FA_STRIDE + k] = val;
        }
    }
    __syncthreads();

    // Load Q into registers (16 cols per warp)
    int Q_B[DITER][4];
    half2* my_tile_Q = tile_Q_all + warp_id * NCOLS_WARP * FA_STRIDE;
    #pragma unroll
    for (int d = 0; d < DITER; ++d) {
        const int col_in_tile = (lane % 8) + ((lane / 16) * 8);
        const int k_off = ((lane / 8) % 2) * 4;
        const int* base = (const int*)(my_tile_Q + col_in_tile * FA_STRIDE + d * 8) + k_off;
        ldmatrix_x4(Q_B[d], base);
    }
    __syncthreads();

    float S_max[4] = {-FLT_MAX/2, -FLT_MAX/2, -FLT_MAX/2, -FLT_MAX/2};
    float S_rowsum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float VKQ[DITER][8] = {};

    auto load_tile_async = [&](half2* tile, const __half* base, int j0_arg) {
        constexpr int CHUNKS = HDIM / 8;
        #pragma unroll
        for (int idx = tid; idx < NBATCH * CHUNKS; idx += 128) {
            const int row = idx / CHUNKS, chunk = idx % CHUNKS;
            half2* dst = tile + row * FA_STRIDE + chunk * 4;
            fa_cp_async_16(dst, base + (size_t)(j0_arg + row) * HDIM + chunk * 8);
        }
        fa_cp_async_commit();
    };

    load_tile_async(tile_K, K_base, 0);

    for (int j0 = 0; j0 < max_npos; j0 += NBATCH) {
        const bool last_iter = (j0 + NBATCH >= max_npos);

        fa_cp_async_wait();
        __syncthreads();

        load_tile_async(tile_V, V_base, j0);

        // KQ = K * Q^T (16x16 output)
        float KQ_C[NTILES][8] = {};
        #pragma unroll
        for (int d = 0; d < DITER; ++d) {
            #pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                int K_A[4];
                const int* tile_ka_base = (const int*)tile_K + t * 16 * FA_STRIDE + d * 8;
                ldmatrix_x4(K_A, tile_ka_base + (lane % 16) * FA_STRIDE + (lane / 16) * 4);
                fa_mma16816(&KQ_C[t][0], K_A, &Q_B[d][0]);
                fa_mma16816(&KQ_C[t][4], K_A, &Q_B[d][2]);
            }
        }

        // Mask OOB (per Q token)
        const int npos_tab[2] = {npos0, npos1};
        #pragma unroll
        for (int t = 0; t < NTILES; ++t) {
            const int base_kv = j0 + t * 16;
            const int g = lane / 4;
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int row = (l % 4 < 2) ? g : g + 8;
                if (base_kv + row >= npos_tab[fa_col8(l) / 2]) KQ_C[t][l] = -INFINITY;
            }
        }

        // Online softmax (4 column groups)
        float new_max[4] = {S_max[0], S_max[1], S_max[2], S_max[3]};
        #pragma unroll
        for (int t = 0; t < NTILES; ++t)
            #pragma unroll
            for (int l = 0; l < 8; ++l)
                new_max[fa_col8(l)] = fmaxf(new_max[fa_col8(l)], KQ_C[t][l] + FA_KQ_MAX_OFFSET);

        #pragma unroll
        for (int off = 16; off >= 4; off >>= 1)
            #pragma unroll
            for (int col = 0; col < 4; ++col)
                new_max[col] = fmaxf(new_max[col], __shfl_xor_sync(0xFFFFFFFF, new_max[col], off));

        int sKQ_B[NTILES][4];
        float KQ_rowsum_add[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int t = 0; t < NTILES; ++t) {
            float KQ_exp[8];
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                KQ_exp[l] = __expf(KQ_C[t][l] - new_max[fa_col8(l)]);
                KQ_rowsum_add[fa_col8(l)] += KQ_exp[l];
            }
            fa_exp_to_mma_b(&sKQ_B[t][0], &KQ_exp[0]);
            fa_exp_to_mma_b(&sKQ_B[t][2], &KQ_exp[4]);
        }

        float rescale[4];
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            const float diff = S_max[col] - new_max[col];
            rescale[col] = __expf(diff);
            *(uint32_t*)&rescale[col] *= diff >= SOFTMAX_FTZ_THRESHOLD;
            S_max[col] = new_max[col];
            S_rowsum[col] = rescale[col] * S_rowsum[col] + KQ_rowsum_add[col];
        }

        #pragma unroll
        for (int d = 0; d < DITER; ++d)
            #pragma unroll
            for (int l = 0; l < 8; ++l)
                VKQ[d][l] *= rescale[fa_col8(l)];

        fa_cp_async_wait();
        __syncthreads();

        if (!last_iter) load_tile_async(tile_K, K_base, j0 + NBATCH);

        // VKQ += V^T * softmax(KQ)
        #pragma unroll
        for (int d = 0; d < DITER; ++d) {
            #pragma unroll
            for (int t = 0; t < NTILES; ++t) {
                int V_A[4];
                const int* tile_va_base = (const int*)tile_V + t * 16 * FA_STRIDE + d * 8;
                ldmatrix_x4_trans(V_A, tile_va_base + (lane % 16) * FA_STRIDE + (lane / 16) * 4);
                fa_mma16816(&VKQ[d][0], V_A, &sKQ_B[t][0]);
                fa_mma16816(&VKQ[d][4], V_A, &sKQ_B[t][2]);
            }
        }
    }

    // Softmax rowsum reduction within warp
    #pragma unroll
    for (int off = 16; off >= 4; off >>= 1)
        #pragma unroll
        for (int col = 0; col < 4; ++col)
            S_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, S_rowsum[col], off);

    // Output, each warp writes directly
    {
        float inv[4];
        #pragma unroll
        for (int col = 0; col < 4; ++col)
            inv[col] = (S_rowsum[col] > 0.0f) ? (1.0f / S_rowsum[col]) : 0.0f;

        #pragma unroll
        for (int d = 0; d < DITER; ++d) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int qtoken = l / 4;
                const int vdim = d * 16 + (lane / 4) + (((l % 4) / 2) * 8);
                const int qcol = (lane % 4) * 2 + (l % 2);
                if (qcol < gqa_ratio && vdim < HDIM && (warp_q0 + qtoken) < batch) {
                    out[((size_t)(warp_q0 + qtoken) * n_head + q_head0 + qcol) * HDIM + vdim] = VKQ[d][l] * inv[qtoken * 2 + (l % 2)];
                }
            }
        }
    }
}

// ----  Elementwise & misc kernels (dequant, norm, rope, kv, swiglu, add) ---

template <DType TYPE>
__device__ __forceinline__ float dequant_element(const void* src, int e) {
    if constexpr (TYPE == DType::Q4_K)       return dequantize_q4_K(((const block_q4_K*)src)[e / 256], e % 256);
    else if constexpr (TYPE == DType::Q5_K)  return dequantize_q5_K(((const block_q5_K*)src)[e / 256], e % 256);
    else if constexpr (TYPE == DType::Q6_K)  return dequantize_q6_K(((const block_q6_K*)src)[e / 256], e % 256);
    else if constexpr (TYPE == DType::Q8_0)  return (float)((const block_q8_0*)src)[e / 32].d * ((const block_q8_0*)src)[e / 32].qs[e % 32];
    else if constexpr (TYPE == DType::F16)   return __half2float(((const __half*)src)[e]);
    else if constexpr (TYPE == DType::F32)   return ((const float*)src)[e];
}

template <DType TYPE>
__global__ void dequant_row_k(const void* src, float* y, int n) {
    for (int e = threadIdx.x; e < n; e += blockDim.x) y[e] = dequant_element<TYPE>(src, e);
}

template <DType TYPE>
__global__ void dequant_rows_k(const void* src, const int32_t* tokens, float* y,
                               int n, int64_t row_bytes) {
    const int idx = blockIdx.x;
    const void* row = (const uint8_t*)src + tokens[idx] * row_bytes;
    y += idx * n;
    for (int e = threadIdx.x; e < n; e += blockDim.x) y[e] = dequant_element<TYPE>(row, e);
}

__global__ void rms_norm_k(const float* x, const float* w, int ncols, float eps, float* dst) {
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        const float t = x[i];
        lsum += t * t;
    }
    const float r = rsqrtf(block_reduce_sum(lsum) / (float)ncols + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) dst[i] = x[i] * r * w[i];
}

__global__ void rms_norm_rows_k(const float* x, const float* w, int ncols, float eps, float* dst) {
    const int row = blockIdx.x;
    x += row * ncols;
    dst += row * ncols;
    float lsum = 0.0f;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        const float t = x[i];
        lsum += t * t;
    }
    const float r = rsqrtf(block_reduce_sum(lsum) / (float)ncols + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) dst[i] = x[i] * r * w[i];
}

enum class PosMode { INDIRECT, BATCH };

template <PosMode MODE>
__global__ void rope_neox_k(float* x, int head_dim, int n_rot, float theta,
                            int n_past, const int* d_pos, int heads, const float* freq_factors) {
    const int head = blockIdx.x, i = threadIdx.x;
    const int half = n_rot / 2;
    if (i >= half) return;
    int pos;
    if constexpr (MODE == PosMode::INDIRECT) pos = *d_pos;
    if constexpr (MODE == PosMode::BATCH)    pos = n_past + head / heads;
    x += head * head_dim;
    float freq = __powf(theta, -(float)(2 * i) / n_rot);
    if (freq_factors) freq /= freq_factors[i];
    const float theta_base = pos * freq;
    const float cos_theta = __cosf(theta_base), sin_theta = __sinf(theta_base);
    const float x0 = x[i], x1 = x[i + half];
    x[i]        = x0 * cos_theta - x1 * sin_theta;
    x[i + half] = x0 * sin_theta + x1 * cos_theta;
}

template <PosMode MODE>
__global__ void rope_normal_k(float* x, int head_dim, int n_rot, float theta,
                              int n_past, const int* d_pos, int heads, const float* freq_factors) {
    const int head = blockIdx.x, i = threadIdx.x;
    if (i >= n_rot / 2) return;
    int pos;
    if constexpr (MODE == PosMode::INDIRECT) pos = *d_pos;
    if constexpr (MODE == PosMode::BATCH)    pos = n_past + head / heads;
    x += head * head_dim;
    float freq = __powf(theta, -(float)(2 * i) / n_rot);
    if (freq_factors) freq /= freq_factors[i];
    const float theta_base = pos * freq;
    const float cos_theta = __cosf(theta_base), sin_theta = __sinf(theta_base);
    const int idx = 2 * i;
    const float x0 = x[idx], x1 = x[idx + 1];
    x[idx]     = x0 * cos_theta - x1 * sin_theta;
    x[idx + 1] = x0 * sin_theta + x1 * cos_theta;
}

template <PosMode MODE>
__global__ void kv_store_k(uint16_t* kc, uint16_t* vc, const float* k, const float* v,
                           int heads, int head_dim, int n_ctx, int pos_base, const int* d_pos) {
    const int i = threadIdx.x;
    if (i >= head_dim) return;
    int head, pos;
    size_t src_off;
    if constexpr (MODE == PosMode::BATCH) {
        head = blockIdx.x % heads;
        pos = pos_base + blockIdx.x / heads;
        src_off = (size_t)blockIdx.x * head_dim + i;
    } else {
        head = blockIdx.x;
        pos = *d_pos;
        src_off = (size_t)head * head_dim + i;
    }
    const size_t off = (size_t)head * n_ctx * head_dim + pos * head_dim + i; // KV cache layout [head][pos][head_dim]
    kc[off] = __half_as_ushort(__float2half(k[src_off]));
    vc[off] = __half_as_ushort(__float2half(v[src_off]));
}

// KV cache rolling: shift [n_keep+shift, n_ctx) to [n_keep, n_ctx-shift).
// grid: (n_layer, n_kv), block: head_dim
__global__ void kv_shift_k(uint16_t* kc, uint16_t* vc, int n_kv, int head_dim,
                           int n_ctx, int n_keep, int shift) {
    const int layer = blockIdx.x, head = blockIdx.y;
    const int i = threadIdx.x;
    if (i >= head_dim) return;
    const size_t per_layer_size = (size_t)n_kv * n_ctx * head_dim;
    const size_t off = layer * per_layer_size + (size_t)head * n_ctx * head_dim + i;
    kc += off;
    vc += off;
    // forward shift
    for (int pos = n_keep; pos < n_ctx - shift; ++pos) {
        kc[(size_t)pos * head_dim] = kc[(size_t)(pos + shift) * head_dim];
        vc[(size_t)pos * head_dim] = vc[(size_t)(pos + shift) * head_dim];
    }
    for (int pos = n_ctx - shift; pos < n_ctx; ++pos) {
        kc[(size_t)pos * head_dim] = 0;
        vc[(size_t)pos * head_dim] = 0;
    }
}

__global__ void swiglu_k(float* act, const float* gate, const float* up, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float g = gate[i];
        act[i] = g / (1.0f + __expf(-g)) * up[i];
    }
}

__global__ void add_inplace_k(float* a, const float* b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) a[i] += b[i];
}

__global__ void add_inplace_broadcast_k(float* a, const float* b, int stride, int batch) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride * batch; i += blockDim.x * gridDim.x)
        a[i] += b[i % stride];
}

// ----------------- Host wrappers -----------------------

constexpr int SM_COUNT = 8;
constexpr size_t SHARED_MEMORY_LIMIT = 100 * 1024; // 100KB, opt-in

constexpr int div_ceil(int n, int d)  { return (n + d - 1) / d; }
constexpr int align_up(int n, int align)  { return div_ceil(n, align) * align; }

static cudaStream_t s_stream = nullptr;

void set_stream(void* stream) { s_stream = (cudaStream_t)stream; }

int init() {
    cudaError_t e = cudaSetDevice(0);
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda::init] cudaSetDevice(0) failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

void device_info() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("[cuda] device: %s (sm_%d%d, %.1f GB)\n",
        p.name, p.major, p.minor, (double)p.totalGlobalMem / (1u << 30));
}

// Buffers for activation quantization, allocated once.
static int8_t* g_q8 = nullptr;
static float*  g_q8_scale = nullptr;

static void*  g_q8_mmq = nullptr;
static float* g_mmq_fixup = nullptr;

size_t scratch_size(int max_in, int max_batch) {
    const size_t n = align_up(max_in, 256);
    const size_t scale_bytes = (n / 32) * sizeof(float);
    const size_t mmq_bytes = (size_t)div_ceil(n, 128) * max_batch * sizeof(block_q8_1_mmq);
    constexpr int max_mmq_x = 128;
    const size_t fixup_bytes = (size_t)SM_COUNT * MMQ_Y * max_mmq_x * sizeof(float);

    return n + align_up(scale_bytes, 256) + align_up(mmq_bytes, 256) + fixup_bytes;
}

void scratch_assign(void* base_ptr, int max_in, int max_batch) {
    int8_t* base = (int8_t*)base_ptr;
    const size_t n = align_up(max_in, 256);
    const size_t scale_bytes = (n / 32) * sizeof(float);
    const size_t mmq_bytes = (size_t)div_ceil(n, 128) * max_batch * sizeof(block_q8_1_mmq);

    g_q8 = base;
    base += align_up(n, 256);
    g_q8_scale = (float*)base;
    base += align_up(scale_bytes, 256);
    g_q8_mmq = base;
    base += align_up(mmq_bytes, 256);
    g_mmq_fixup = (float*)base;
}

void quantize_activation(const float* x, int n_in) {
    const int ngroups = div_ceil(n_in, 32);
    const int warps_per_block = 8;
    const int grid = div_ceil(ngroups, warps_per_block);
    quantize_q8_1_k<<<grid, warps_per_block * 32, 0, s_stream>>>(x, g_q8, g_q8_scale, n_in);
    CUDA_CHECK(cudaGetLastError());
}

template <GemvWrite MODE, bool ADD_K>
static void gemv_q8_impl(DType type, const void* W, const float* extra, float* y, int n_in, int n_out) {
    const bool small_k = (n_in < 2048);
    if (small_k || MODE == GemvWrite::BIAS) {
        const int grid = div_ceil(n_out, 4);
        switch (type) {
            case DType::Q8_0: gemv_dp4a_smallk_k<DType::Q8_0, MODE><<<grid, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, extra, y, n_in, n_out); break;
            case DType::Q4_K: gemv_dp4a_smallk_k<DType::Q4_K, MODE><<<grid, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, extra, y, n_in, n_out); break;
            case DType::Q5_K: gemv_dp4a_smallk_k<DType::Q5_K, MODE><<<grid, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, extra, y, n_in, n_out); break;
            case DType::Q6_K: gemv_dp4a_smallk_k<DType::Q6_K, MODE><<<grid, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, extra, y, n_in, n_out); break;
        }
    } else {
        switch (type) {
            case DType::Q8_0: gemv_dp4a_k<DType::Q8_0, ADD_K><<<n_out, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, y, n_in); break;
            case DType::Q4_K: gemv_dp4a_k<DType::Q4_K, ADD_K><<<n_out, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, y, n_in); break;
            case DType::Q5_K: gemv_dp4a_k<DType::Q5_K, ADD_K><<<n_out, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, y, n_in); break;
            case DType::Q6_K: gemv_dp4a_k<DType::Q6_K, ADD_K><<<n_out, 128, 0, s_stream>>>(W, g_q8, g_q8_scale, y, n_in); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv(DType type, const void* W, const float* x, float* y, int n_in, int n_out) {
    if (type == DType::Q8_0 || type == DType::Q4_K || type == DType::Q5_K || type == DType::Q6_K) {
        const int ngroups = div_ceil(n_in, 32);
        const int warps_per_block = 8;
        const int grid_q = div_ceil(ngroups, warps_per_block);
        quantize_q8_1_k<<<grid_q, warps_per_block * 32, 0, s_stream>>>(x, g_q8, g_q8_scale, n_in);
        gemv_q8_impl<GemvWrite::SET, false>(type, W, nullptr, y, n_in, n_out);
    } else {
        switch (type) {
            case DType::F16:  gemv_plain_k<DType::F16><<<n_out, 256, 0, s_stream>>>(W, x, y, n_in); break;
            case DType::F32:  gemv_plain_k<DType::F32><<<n_out, 256, 0, s_stream>>>(W, x, y, n_in); break;
            default: fprintf(stderr, "CUDA gemv: unsupported type %d\n", (int)type); return;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv_q8(DType type, const void* W, float* y, int n_in, int n_out) {
    gemv_q8_impl<GemvWrite::SET, false>(type, W, nullptr, y, n_in, n_out);
}

void gemv_q8_bias(DType type, const void* W, const float* bias, float* y, int n_in, int n_out) {
    gemv_q8_impl<GemvWrite::BIAS, false>(type, W, bias, y, n_in, n_out);
}

void gemv_q8_add(DType type, const void* W, float* y, int n_in, int n_out) {
    gemv_q8_impl<GemvWrite::ADD, true>(type, W, nullptr, y, n_in, n_out);
}

// Unified MMA GEMM with stream-K for Q4K/Q6K
template <DType TYPE>
static void mmq_batch_launch(const void* W, const float* X, float* Y,
                            int n_in, int n_out, int batch) {
    // Quantize activation to q8_1
    {
        const int n_in_aligned = align_up(n_in, 128);
        constexpr int QUANT_BLK = 256;
        const int block_num_y = div_ceil(n_in_aligned, 4 * QUANT_BLK);
        constexpr bool ds4 = (TYPE == DType::Q4_K);
        quantize_mmq_q8_1_k<ds4><<<dim3(batch, block_num_y), QUANT_BLK, 0, s_stream>>>(
            X, g_q8_mmq, n_in, n_in_aligned, batch);
        CUDA_CHECK(cudaGetLastError());
    }

    // Select tile width (batch dim) that minimizes tile count within shared memory limit
    int best_n = 8;
    int best_tiles = div_ceil(batch, 8);
    for (int n = 16; n <= 128 && best_tiles > 1; n += 8) {
        const int granularity = (n >= 48) ? 16 : 8;
        if (n % granularity != 0) continue;
        const int tile_y_ints = align_up(n * MMQ_TILE_Y_K, 256);
        const size_t smem_bytes = (MMQ_Y * MMQ_X_STRIDE + tile_y_ints) * sizeof(int);
        if (smem_bytes > SHARED_MEMORY_LIMIT) continue;
        int tiles = div_ceil(batch, n);
        if (tiles < best_tiles) {
            best_tiles = tiles;
            best_n = n;
        }
    }

    // Stream-K schedule
    const int nb = n_in / QK_K;
    const int nty = div_ceil(n_out, MMQ_Y); // tile count of n_out dim
    const int ntx_tiles = div_ceil(batch, best_n); // tile count of batch dim
    const int total_tiles = ntx_tiles * nty;

    const int waves = div_ceil(total_tiles, SM_COUNT);
    const int efficiency = 100 * total_tiles / (SM_COUNT * waves);
    const int grid_x = efficiency >= 90 ? total_tiles : SM_COUNT; // tile-based schedule if high gpu efficiency, otherwise use Stream-K
    const bool fixup_needed = total_tiles % grid_x != 0;

    const uint3 nb_fd = init_fastdiv(nb);
    const uint3 ntx_fd = init_fastdiv(ntx_tiles);
    const bool need_check = (n_out % MMQ_Y != 0);

    auto launch = [&](auto tile_n_tag) {
        constexpr int TN = decltype(tile_n_tag)::value;
        constexpr int tile_y_ints = align_up(TN * MMQ_TILE_Y_K, 256);
        const size_t smem_bytes = (MMQ_Y * MMQ_X_STRIDE + tile_y_ints) * sizeof(int);

        auto launch_mm = [&](auto check_tag) {
            constexpr bool CHK = decltype(check_tag)::value;
            static bool smem_raised = false;
            if (smem_bytes > 48 * 1024 && !smem_raised) {
                smem_raised = true;
                CUDA_CHECK(cudaFuncSetAttribute(mul_mat_mmq<TYPE, TN, CHK>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
            }
            mul_mat_mmq<TYPE, TN, CHK><<<grid_x, dim3(32, 8), smem_bytes, s_stream>>>(
                W, (const int*)g_q8_mmq, Y, g_mmq_fixup,
                n_out, n_in, batch, nb, nb_fd, ntx_fd, SM_COUNT);
        };
        if (need_check) launch_mm(std::true_type{});
        else            launch_mm(std::false_type{});
        CUDA_CHECK(cudaGetLastError());

        if (fixup_needed) {
            auto launch_fixup = [&](auto check_tag) {
                constexpr bool CHK = decltype(check_tag)::value;
                mul_mat_mmq_fixup<TN, MMQ_Y, CHK><<<dim3(grid_x, 4), dim3(32, 4), 0, s_stream>>>(
                    Y, g_mmq_fixup, n_out, batch, nb_fd, ntx_fd, SM_COUNT);
            };
            if (need_check) launch_fixup(std::true_type{});
            else            launch_fixup(std::false_type{});
            CUDA_CHECK(cudaGetLastError());
        }
    };

    switch (best_n) {
        case  8: launch(std::integral_constant<int,  8>{}); break;
        case 16: launch(std::integral_constant<int, 16>{}); break;
        case 24: launch(std::integral_constant<int, 24>{}); break;
        case 32: launch(std::integral_constant<int, 32>{}); break;
        case 40: launch(std::integral_constant<int, 40>{}); break;
        case 48: launch(std::integral_constant<int, 48>{}); break;
        case 64: launch(std::integral_constant<int, 64>{}); break;
        case 80: launch(std::integral_constant<int, 80>{}); break;
        case 96: launch(std::integral_constant<int, 96>{}); break;
        case 112: launch(std::integral_constant<int, 112>{}); break;
        case 128: launch(std::integral_constant<int, 128>{}); break;
        default: launch(std::integral_constant<int,  8>{}); break;
    }
}

void gemv_batch(DType type, const void* W, const float* X, float* Y, int n_in, int n_out, int batch) {
    switch (type) {
        case DType::Q4_K: mmq_batch_launch<DType::Q4_K>(W, X, Y, n_in, n_out, batch); break;
        case DType::Q6_K: mmq_batch_launch<DType::Q6_K>(W, X, Y, n_in, n_out, batch); break;
        default: {
            for (int b = 0; b < batch; ++b) gemv(type, W, X + b * n_in, Y + b * n_out, n_in, n_out);
            return;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void gemv_fused_gate_up_swiglu(DType type, const void* Wg, const void* Wu,
                                float* out, int n_in, int n_out) {
    const int grid = div_ceil(n_out, 4);
    switch (type) {
        case DType::Q8_0: gemv_fused_gate_up_swiglu_k<DType::Q8_0><<<grid, 128, 0, s_stream>>>(Wg, Wu, g_q8, g_q8_scale, out, n_in, n_out); break;
        case DType::Q4_K: gemv_fused_gate_up_swiglu_k<DType::Q4_K><<<grid, 128, 0, s_stream>>>(Wg, Wu, g_q8, g_q8_scale, out, n_in, n_out); break;
        case DType::Q5_K: gemv_fused_gate_up_swiglu_k<DType::Q5_K><<<grid, 128, 0, s_stream>>>(Wg, Wu, g_q8, g_q8_scale, out, n_in, n_out); break;
        case DType::Q6_K: gemv_fused_gate_up_swiglu_k<DType::Q6_K><<<grid, 128, 0, s_stream>>>(Wg, Wu, g_q8, g_q8_scale, out, n_in, n_out); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// --- Flash attention ---

template <int HDIM>
static void flash_attention_prefill_launch(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                                           int n_head, int n_kv, int head_dim, int n_ctx, int n_past, int batch, float scale) {
    constexpr int QTOKENS_PER_BLOCK = 8;
    constexpr int NBATCH = FA_PREFILL_NBATCH;
    constexpr int FA_STRIDE = HDIM / 2 + 4;
    constexpr int NCOLS_WARP = 16;
    constexpr int NWARPS = 4;
    constexpr int smem_bytes = (2 * NBATCH + NWARPS * NCOLS_WARP) * FA_STRIDE * (int)sizeof(half2);
    const int grid_x = div_ceil(batch, QTOKENS_PER_BLOCK) * n_kv;
    cudaFuncSetAttribute(flash_attention_prefill_mma_k<HDIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    flash_attention_prefill_mma_k<HDIM><<<grid_x, dim3(32, 4), smem_bytes, s_stream>>>(
        q, kc, vc, out, n_head, n_kv, head_dim, n_ctx, n_past, batch, scale);
    CUDA_CHECK(cudaGetLastError());
}

void flash_attention(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                     int n_head, int n_kv, int head_dim, int n_ctx, int n_past, int batch, float scale) {
    switch (head_dim) {
        case  64: flash_attention_prefill_launch< 64>(q, kc, vc, out, n_head, n_kv, head_dim, n_ctx, n_past, batch, scale); break;
        case 128: flash_attention_prefill_launch<128>(q, kc, vc, out, n_head, n_kv, head_dim, n_ctx, n_past, batch, scale); break;
        default: assert(false && "unsupported head_dim");
    }
}

template <int HDIM>
static void flash_attention_decode_launch(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                                          int n_head, int n_kv, int head_dim, int n_ctx, const int* d_pos, float scale) {
    constexpr int NBATCH = 128, FA_STRIDE = HDIM / 2 + 4;
    constexpr int smem_bytes = 2 * NBATCH * FA_STRIDE * (int)sizeof(half2);
    cudaFuncSetAttribute(flash_attention_decode_mma_k<HDIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    flash_attention_decode_mma_k<HDIM><<<n_kv, dim3(32, 4), smem_bytes, s_stream>>>(
        q, kc, vc, out, n_head, n_kv, head_dim, n_ctx, d_pos, scale);
}

void flash_attention_decode_ind(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                                int n_head, int n_kv, int head_dim, int n_ctx, const int* d_pos, float scale) {
    switch (head_dim) {
        case  64: flash_attention_decode_launch< 64>(q, kc, vc, out, n_head, n_kv, head_dim, n_ctx, d_pos, scale); break;
        case 128: flash_attention_decode_launch<128>(q, kc, vc, out, n_head, n_kv, head_dim, n_ctx, d_pos, scale); break;
        default: assert(false && "unsupported head_dim");
    }
}

// --- Elementwise & misc ---

void dequant_row(DType type, const void* src, float* y, int n) {
    switch (type) {
        case DType::F32:  dequant_row_k<DType::F32> <<<1, 256, 0, s_stream>>>(src, y, n); break;
        case DType::F16:  dequant_row_k<DType::F16> <<<1, 256, 0, s_stream>>>(src, y, n); break;
        case DType::Q8_0: dequant_row_k<DType::Q8_0><<<1, 256, 0, s_stream>>>(src, y, n); break;
        case DType::Q4_K: dequant_row_k<DType::Q4_K><<<1, 256, 0, s_stream>>>(src, y, n); break;
        case DType::Q5_K: dequant_row_k<DType::Q5_K><<<1, 256, 0, s_stream>>>(src, y, n); break;
        case DType::Q6_K: dequant_row_k<DType::Q6_K><<<1, 256, 0, s_stream>>>(src, y, n); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void dequant_rows(DType type, const void* src, const int32_t* tokens, float* y, int n, int batch) {
    const int64_t row_bytes = (int64_t)tensor_nbytes(type, n, 1);
    switch (type) {
        case DType::F32:  dequant_rows_k<DType::F32> <<<batch, 256, 0, s_stream>>>(src, tokens, y, n, row_bytes); break;
        case DType::F16:  dequant_rows_k<DType::F16> <<<batch, 256, 0, s_stream>>>(src, tokens, y, n, row_bytes); break;
        case DType::Q8_0: dequant_rows_k<DType::Q8_0> <<<batch, 256, 0, s_stream>>>(src, tokens, y, n, row_bytes); break;
        case DType::Q4_K: dequant_rows_k<DType::Q4_K> <<<batch, 256, 0, s_stream>>>(src, tokens, y, n, row_bytes); break;
        case DType::Q5_K: dequant_rows_k<DType::Q5_K> <<<batch, 256, 0, s_stream>>>(src, tokens, y, n, row_bytes); break;
        case DType::Q6_K: dequant_rows_k<DType::Q6_K> <<<batch, 256, 0, s_stream>>>(src, tokens, y, n, row_bytes); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void rms_norm(const float* x, const float* w, int n, float eps, float* y) {
    const int block = (n >= 1024) ? 1024 : 256;
    rms_norm_k<<<1, block, 0, s_stream>>>(x, w, n, eps, y);
    CUDA_CHECK(cudaGetLastError());
}

void rms_norm_rows(const float* x, const float* w, int rows, int cols, float eps, float* y) {
    const int block = (cols >= 1024) ? 1024 : 256;
    rms_norm_rows_k<<<rows, block, 0, s_stream>>>(x, w, cols, eps, y);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu(float* act, const float* gate, const float* up, int n) {
    swiglu_k<<<div_ceil(n, 256), 256, 0, s_stream>>>(act, gate, up, n);
    CUDA_CHECK(cudaGetLastError());
}

void add_inplace(float* a, const float* b, int n) {
    add_inplace_k<<<div_ceil(n, 256), 256, 0, s_stream>>>(a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

void add_inplace_broadcast(float* a, const float* b, int stride, int batch) {
    const int total = stride * batch;
    add_inplace_broadcast_k<<<div_ceil(total, 256), 256, 0, s_stream>>>(a, b, stride, batch);
    CUDA_CHECK(cudaGetLastError());
}

// --- RoPE ---

void rope_neox_batch(float* x, int rows, int head_dim, int n_rot, float theta, int n_past, int heads, const float* freq_factors) {
    rope_neox_k<PosMode::BATCH><<<rows, n_rot / 2, 0, s_stream>>>(x, head_dim, n_rot, theta, n_past, nullptr, heads, freq_factors);
    CUDA_CHECK(cudaGetLastError());
}

void rope_normal_batch(float* x, int rows, int head_dim, int n_rot, float theta, int n_past, int heads, const float* freq_factors) {
    rope_normal_k<PosMode::BATCH><<<rows, n_rot / 2, 0, s_stream>>>(x, head_dim, n_rot, theta, n_past, nullptr, heads, freq_factors);
    CUDA_CHECK(cudaGetLastError());
}

void rope_neox_ind(float* x, int rows, int head_dim, int n_rot, float theta, const int* d_pos, const float* freq_factors) {
    rope_neox_k<PosMode::INDIRECT><<<rows, n_rot / 2, 0, s_stream>>>(x, head_dim, n_rot, theta, 0, d_pos, 0, freq_factors);
}

void rope_normal_ind(float* x, int rows, int head_dim, int n_rot, float theta, const int* d_pos, const float* freq_factors) {
    rope_normal_k<PosMode::INDIRECT><<<rows, n_rot / 2, 0, s_stream>>>(x, head_dim, n_rot, theta, 0, d_pos, 0, freq_factors);
}

// --- KV cache ---

void kv_store_batch(uint16_t* kc, uint16_t* vc, const float* k, const float* v,
                    int heads, int head_dim, int n_ctx, int pos_base, int batch) {
    kv_store_k<PosMode::BATCH><<<batch * heads, 256, 0, s_stream>>>(kc, vc, k, v, heads, head_dim, n_ctx, pos_base, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void kv_store_ind(uint16_t* kc, uint16_t* vc, const float* k, const float* v,
                  int heads, int head_dim, int n_ctx, const int* d_pos) {
    kv_store_k<PosMode::INDIRECT><<<heads, 256, 0, s_stream>>>(kc, vc, k, v, 0, head_dim, n_ctx, 0, d_pos);
}

void kv_shift(uint16_t* kc, uint16_t* vc, int n_layer, int n_kv, int head_dim, int n_ctx, int n_keep, int shift) {
    kv_shift_k<<<dim3(n_layer, n_kv), head_dim, 0, s_stream>>>(kc, vc, n_kv, head_dim, n_ctx, n_keep, shift);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llm::cuda
