#pragma once

#include <cstddef>
#include <cstdint>

namespace llm {
namespace cuda {

// --- Init ---

int init();
void device_info();
void set_stream(void* stream);
void init_buffers(int max_in, int max_batch);

// --- Activation quantization ---

// Quantize fp32 activation [n_in] to int8 + scale for subsequent gemv_q8 calls.
// Result stored in internal scratch buffer (overwritten by next call).
void quantize_activation(const float* x, int n_in);

// --- GEMV (decode + batch + fused) ---

// Fused dequant + matrix-vector multiply: y = W^T @ x.
void gemv(int type, const void* W, const float* x, float* y, int n_in, int n_out);

// GEMV using pre-quantized activation. Requires prior quantize_activation call.
void gemv_q8(int type, const void* W, float* y, int n_in, int n_out);

// y[i] = dot(W[i], xq) + bias[i]
void gemv_q8_bias(int type, const void* W, const float* bias, float* y, int n_in, int n_out);

// y[i] += dot(W[i], xq)
void gemv_q8_add(int type, const void* W, float* y, int n_in, int n_out);

// Batched GEMV: Y[b][m] = sum_k W[m][k] * X[b][k].
void gemv_batch(int type, const void* W, const float* X, float* Y, int n_in, int n_out, int batch);

// Fused gate+up GEMV + SwiGLU in a single kernel.
void gemv_fused_gate_up_swiglu(int type, const void* Wg, const void* Wu,
                                float* out, int n_in, int n_out);

// --- Elementwise & misc ---

// Dequantize a single row (n elements) to fp32. Used for token embedding lookup.
void dequant_row(int type, const void* src, float* y, int n);

// Batched token embedding lookup: dequantize batch rows from embedding table.
void dequant_rows(int type, const void* src, const int32_t* tokens, float* y, int n, int batch);

void rms_norm(const float* x, const float* w, int n, float eps, float* y);
void rms_norm_rows(const float* x, const float* w, int rows, int cols, float eps, float* y);

// act[i] = silu(gate[i]) * up[i]
void swiglu(float* act, const float* gate, const float* up, int n);

void add_inplace(float* a, const float* b, int n);
// a[i] += b[i % stride], for stride * batch elements.
void add_inplace_broadcast(float* a, const float* b, int stride, int batch);

// --- RoPE ---

void rope_neox_batch(float* x, int rows, int head_dim, int n_rot, float theta, int n_past, int heads,
                     const float* freq_factors = nullptr);

void rope_normal_batch(float* x, int rows, int head_dim, int n_rot, float theta, int n_past, int heads,
                       const float* freq_factors = nullptr);

// Indirect-pos variants: position read from device memory (for CUDA Graph).
void rope_neox_ind(float* x, int rows, int head_dim, int n_rot, float theta, const int* d_pos,
                   const float* freq_factors = nullptr);

void rope_normal_ind(float* x, int rows, int head_dim, int n_rot, float theta, const int* d_pos,
                     const float* freq_factors = nullptr);

// --- KV cache ---

void kv_store_batch(uint16_t* kc, uint16_t* vc, const float* k, const float* v,
                    int heads, int head_dim, int n_ctx, int pos_base, int batch);

// Indirect-pos variant: position read from device memory.
void kv_store_ind(uint16_t* kc, uint16_t* vc, const float* k, const float* v,
                  int heads, int head_dim, int n_ctx, const int* d_pos);

// Discard tokens [n_keep, n_keep+shift), shift remaining left, zero tail.
void kv_shift(uint16_t* kc, uint16_t* vc, int n_layer, int n_kv, int head_dim, int n_ctx, int n_keep, int shift);

// --- Flash attention ---

void flash_attention(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                    int n_head, int n_kv, int head_dim, int n_ctx, int n_past, int batch, float scale);

// Indirect-pos variant for single-token decode.
void flash_attention_decode_ind(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                                int n_head, int n_kv, int head_dim, int n_ctx, const int* d_pos, float scale);

} // namespace cuda
} // namespace llm
