
#include "model.h"

#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <iostream>

#include <cuda_runtime.h>
#include "quant.h"
#include "kernels.h"

namespace llm {

class Model::impl {
public:
    explicit impl(HParams& hp) : hp_(hp) {}
    ~impl();

    void load(const GGUF& g);
    void init_buffers(int n_ctx, int max_batch);
    void print_memory() const;

    const float* forward(int32_t token, int n_past);
    const float* forward_batch(const int32_t* tokens, int batch, int n_past);

    void kv_shift(int n_keep, int shift);

private:
    void decode_layer(int l);
    void forward_capture();

    ModelWeights w_;
    HParams& hp_;

    void* mapped_ptr_ = nullptr;
    size_t mapped_size_ = 0;
    size_t weight_bytes_ = 0;

    // device weights
    const uint8_t* d_token_embd_ = nullptr;
    const uint8_t* d_output_ = nullptr;

    struct DevLayer {
        const uint8_t* wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        const uint8_t* ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
        float* attn_norm = nullptr, * ffn_norm = nullptr;
        float* wq_norm = nullptr, * wk_norm = nullptr;
        float* bq = nullptr, * bk = nullptr, * bv = nullptr; // q/k/v bias
    };
    std::vector<DevLayer> dev_layers_;
    float* d_output_norm_ = nullptr;

    // device activation + kernel scratch buffers (merged allocation, d_x_ is base)
    float* d_x_ = nullptr, * d_xn_ = nullptr, * d_xn2_ = nullptr;
    float* d_q_ = nullptr, * d_k_ = nullptr, * d_v_ = nullptr;
    float* d_attn_ = nullptr, * d_o_ = nullptr;
    float* d_gate_ = nullptr, * d_up_ = nullptr, * d_act_ = nullptr;
    float* d_scores_ = nullptr, * d_logits_ = nullptr;

    float* h_logits_ = nullptr;

    // fp16 KV cache
    uint16_t* d_kc_ = nullptr, * d_vc_ = nullptr;

    // batch
    int32_t* d_tokens_ = nullptr;
    int max_batch_ = 1;

    // memory accounting
    size_t buf_bytes_ = 0;
    size_t kv_bytes_ = 0;

    // CUDA Graph
    int* d_pos_ = nullptr;
    uint8_t* d_emb_row_ = nullptr;
    float* d_rope_ff_ = nullptr;  // rope freq_factors
    void* graph_stream_ = nullptr;
    void* graph_ = nullptr;
    void* graph_exec_ = nullptr;
    bool graph_captured_ = false;
};

void Model::impl::load(const GGUF& g) {
    load_weights(g, hp_, w_);

    if (cuda::init() != 0) LLM_ABORT("CUDA init failed, check CUDA runtime");

    cuda::device_info();

    const uintptr_t page = sysconf(_SC_PAGESIZE);
    uintptr_t data_start = (uintptr_t)(g.data_start()) & ~(page - 1);
    uintptr_t data_end   = (uintptr_t)(g.data_end() + page - 1) & ~(page - 1);
    mapped_ptr_  = (void*)data_start;
    mapped_size_ = (size_t)(data_end - data_start);
    CUDA_CHECK(cudaHostRegister(mapped_ptr_, mapped_size_, cudaHostRegisterMapped));

    struct Slot {
        const void* src;
        void** dst;
        size_t nbytes;
    };
    std::vector<Slot> slots;

    auto add_tensor = [&](const Tensor* t, auto* dst_ptr) {
        if (t) slots.push_back({t->data, (void**)dst_ptr, tensor_nbytes(t->type, t->ne0, t->ne1)});
    };

    // weight tensors
    add_tensor(w_.token_embd, &d_token_embd_);
    if (w_.output != w_.token_embd) add_tensor(w_.output, &d_output_);

    dev_layers_.resize(hp_.n_layer);

    for (int i = 0; i < hp_.n_layer; ++i) {
        const Layer& L = w_.layers[i];
        DevLayer& D = dev_layers_[i];
        add_tensor(L.wq, &D.wq);
        add_tensor(L.wk, &D.wk);
        add_tensor(L.wv, &D.wv);
        add_tensor(L.wo, &D.wo);
        add_tensor(L.ffn_gate, &D.ffn_gate);
        add_tensor(L.ffn_up,   &D.ffn_up);
        add_tensor(L.ffn_down, &D.ffn_down);
        add_tensor(L.attn_norm, &D.attn_norm);
        add_tensor(L.ffn_norm,  &D.ffn_norm);
        add_tensor(L.wq_norm,   &D.wq_norm);
        add_tensor(L.wk_norm,   &D.wk_norm);
        add_tensor(L.bq, &D.bq);
        add_tensor(L.bk, &D.bk);
        add_tensor(L.bv, &D.bv);
    }
    add_tensor(w_.output_norm, &d_output_norm_);

    if (hp_.rope_freq_factors) slots.push_back({hp_.rope_freq_factors, (void**)&d_rope_ff_, (hp_.n_rot / 2) * sizeof(float)});

    // memory accounting + zero-copy mapping
    weight_bytes_ = 0;
    for (auto& s : slots) {
        weight_bytes_ += s.nbytes;
        void* dp = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&dp, (void*)s.src, 0));
        *s.dst = dp;
    }
    if (w_.output == w_.token_embd) d_output_ = d_token_embd_;
}

void Model::impl::init_buffers(int n_ctx, int max_batch) {
    max_batch_ = max_batch;

    const int ne = hp_.n_embd, nhead = hp_.n_head, nkv = hp_.n_head_kv;
    const int hdim = hp_.head_dim, nff = hp_.n_ff;
    const int batch = max_batch_;

    const size_t qa_sz = batch * nhead * hdim * sizeof(float);
    const size_t ff_sz = batch * nff * sizeof(float);
    const size_t big = std::max(qa_sz, ff_sz);
    const size_t x_bytes = batch * ne * sizeof(float);
    const size_t kv_dim_bytes = batch * nkv * hdim * sizeof(float);
    const size_t o_bytes = batch * ne * sizeof(float);
    const size_t tok_bytes = batch * sizeof(int32_t);
    const size_t pos_bytes = sizeof(int);
    const size_t logits_bytes = hp_.n_vocab * sizeof(float);
    const size_t row_bytes = tensor_nbytes(w_.token_embd->type, ne, 1);

    auto align = [](size_t s) -> size_t { return (s + 255) & ~(size_t)255; };

    // activation + kernel scratch: single malloc, 256-byte aligned sub-regions
    const int scratch_max_in = std::max(nff, ne);
    const size_t scratch_bytes = cuda::scratch_size(scratch_max_in, batch);
    const size_t buf_bytes = align(x_bytes) * 3      // d_x_, d_xn_, d_xn2_
                           + align(big) * 3          // d_q_, d_attn_, d_act_
                           + align(kv_dim_bytes) * 2 // d_k_, d_v_
                           + align(o_bytes)          // d_o_
                           + align(tok_bytes)        // d_tokens_
                           + align(pos_bytes)        // d_pos_
                           + align(row_bytes)        // d_emb_row_
                           + scratch_bytes;          // kernel scratch
    char* base = nullptr;
    CUDA_CHECK(cudaMalloc(&base, buf_bytes));

    auto carve = [&base, &align](size_t bytes) -> float* {
        float* p = (float*)base;
        base += align(bytes);
        return p;
    };

    d_x_      = carve(x_bytes);
    d_xn_     = carve(x_bytes);
    d_xn2_    = carve(x_bytes);
    d_q_      = carve(big);
    d_k_      = carve(kv_dim_bytes);
    d_v_      = carve(kv_dim_bytes);
    d_attn_   = carve(big);
    d_o_      = carve(o_bytes);
    d_act_    = carve(big);
    d_gate_   = d_q_;
    d_up_     = d_attn_;
    d_tokens_ = (int32_t*)carve(tok_bytes);
    d_pos_    = (int*)carve(pos_bytes);
    d_emb_row_ = (uint8_t*)base;
    base += align(row_bytes);
    cuda::scratch_assign(base, scratch_max_in, batch);

    CUDA_CHECK(cudaHostAlloc((void**)&h_logits_, logits_bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_logits_, h_logits_, 0));
    buf_bytes_ = buf_bytes + logits_bytes;

    // fit n_ctx: query free memory after all non-KV allocations
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t kv_per_ctx = (size_t)hp_.n_layer * nkv * hdim * sizeof(uint16_t) * 2;
    int max_ctx = (int)(free_mem / kv_per_ctx);
    constexpr int MIN_CTX = 4096;
    max_ctx = max_ctx / MIN_CTX * MIN_CTX;
    if (n_ctx > max_ctx) {
        std::cerr << "  [fit] reducing n_ctx from " << n_ctx << " to " << max_ctx << std::endl;
        n_ctx = max_ctx;
    }
    if (n_ctx < MIN_CTX) LLM_ABORT("not enough memory for n_ctx=%d (minimum %d)", n_ctx, MIN_CTX);

    hp_.n_ctx = n_ctx;
    const size_t k_bytes = (size_t)hp_.n_layer * nkv * n_ctx * hdim * sizeof(uint16_t);
    kv_bytes_ = k_bytes * 2;
 
    char* kv_base = nullptr;
    CUDA_CHECK(cudaMalloc(&kv_base, kv_bytes_));
    d_kc_ = (uint16_t*)kv_base;
    d_vc_ = (uint16_t*)(kv_base + k_bytes);

    CUDA_CHECK(cudaStreamCreateWithFlags((cudaStream_t*) &graph_stream_, cudaStreamNonBlocking));
}

void Model::impl::print_memory() const {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    auto mb = [](size_t b) { return (double)b / (1024 * 1024); };
    char buf[256];
    snprintf(buf, sizeof(buf),
             "  memory: %.1f MB weights (mapped), %.1f MB buffers + %.1f MB KV (n_ctx=%d) | %.1f MB free / %.1f MB total",
             mb(weight_bytes_), mb(buf_bytes_), mb(kv_bytes_), hp_.n_ctx, mb(free_mem), mb(total_mem));
    std::cout << buf << std::endl;
}

void Model::impl::decode_layer(int l) {
    const int ne = hp_.n_embd, nhead = hp_.n_head, hdim = hp_.head_dim;
    const int nkv = hp_.n_head_kv, nff = hp_.n_ff, n_ctx = hp_.n_ctx;
    const float scale = 1.0f / std::sqrt((float)hdim);
    const size_t per_layer = (size_t)nkv * n_ctx * hdim;

    const Layer& L = w_.layers[l];
    const DevLayer& D = dev_layers_[l];
    uint16_t* kbase = d_kc_ + l * per_layer;
    uint16_t* vbase = d_vc_ + l * per_layer;

    cuda::rms_norm(d_x_, D.attn_norm, ne, hp_.rms_eps, d_xn_);
    cuda::quantize_activation(d_xn_, ne);
    if (D.bq) {
        cuda::gemv_q8_bias(L.wq->type, D.wq, D.bq, d_q_, L.wq->ne0, L.wq->ne1);
        cuda::gemv_q8_bias(L.wk->type, D.wk, D.bk, d_k_, L.wk->ne0, L.wk->ne1);
        cuda::gemv_q8_bias(L.wv->type, D.wv, D.bv, d_v_, L.wv->ne0, L.wv->ne1);
    } else {
        cuda::gemv_q8(L.wq->type, D.wq, d_q_, L.wq->ne0, L.wq->ne1);
        cuda::gemv_q8(L.wk->type, D.wk, d_k_, L.wk->ne0, L.wk->ne1);
        cuda::gemv_q8(L.wv->type, D.wv, d_v_, L.wv->ne0, L.wv->ne1);
    }

    if (D.wq_norm) cuda::rms_norm_rows(d_q_, D.wq_norm, nhead, hdim, hp_.rms_eps, d_q_);
    if (D.wk_norm) cuda::rms_norm_rows(d_k_, D.wk_norm, nkv, hdim, hp_.rms_eps, d_k_);

    auto rope_ind = hp_.rope_type == 1 ? cuda::rope_neox_ind : cuda::rope_normal_ind;
    rope_ind(d_q_, nhead, hdim, hp_.n_rot, hp_.rope_theta, d_pos_, d_rope_ff_);
    rope_ind(d_k_, nkv, hdim, hp_.n_rot, hp_.rope_theta, d_pos_, d_rope_ff_);

    cuda::kv_store_ind(kbase, vbase, d_k_, d_v_, nkv, hdim, n_ctx, d_pos_);
    cuda::flash_attention_decode_ind(d_q_, kbase, vbase, d_attn_, nhead, nkv, hdim, n_ctx, d_pos_, scale);

    cuda::quantize_activation(d_attn_, ne);
    cuda::gemv_q8_add(L.wo->type, D.wo, d_x_, L.wo->ne0, L.wo->ne1);

    // FFN
    cuda::rms_norm(d_x_, D.ffn_norm, ne, hp_.rms_eps, d_xn2_);
    cuda::quantize_activation(d_xn2_, ne);
    cuda::gemv_fused_gate_up_swiglu(L.ffn_gate->type, D.ffn_gate, D.ffn_up, d_act_, L.ffn_gate->ne0, nff);
    cuda::quantize_activation(d_act_, nff);
    cuda::gemv_q8_add(L.ffn_down->type, D.ffn_down, d_x_, L.ffn_down->ne0, L.ffn_down->ne1);
}

void Model::impl::forward_capture() {
    const int ne = hp_.n_embd;
    cudaStream_t stream = (cudaStream_t)graph_stream_;

    cuda::set_stream(stream);
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

    cuda::dequant_row(w_.token_embd->type, d_emb_row_, d_x_, ne);
    for (int l = 0; l < hp_.n_layer; ++l) decode_layer(l);
    cuda::rms_norm(d_x_, d_output_norm_, ne, hp_.rms_eps, d_xn_);
    cuda::gemv(w_.output->type, d_output_, d_xn_, d_logits_, w_.output->ne0, w_.output->ne1);

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cuda::set_stream(nullptr);
    graph_ = graph;

    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    graph_exec_ = exec;
    graph_captured_ = true;
}

const float* Model::impl::forward(int32_t token, int n_past) {
    const int ne = hp_.n_embd;
    const uint64_t row_bytes = tensor_nbytes(w_.token_embd->type, ne, 1);
    cudaStream_t stream = (cudaStream_t) graph_stream_;

    CUDA_CHECK(cudaMemcpyAsync(d_emb_row_, d_token_embd_ + (uint64_t)token * row_bytes,
                               row_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, &n_past, sizeof(int), cudaMemcpyHostToDevice, stream));

    if (!graph_captured_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (getenv("NO_GRAPH")) {
            cuda::set_stream(stream);
            cuda::dequant_row(w_.token_embd->type, d_emb_row_, d_x_, ne);
            for (int l = 0; l < hp_.n_layer; ++l) decode_layer(l);
            cuda::rms_norm(d_x_, d_output_norm_, ne, hp_.rms_eps, d_xn_);
            cuda::gemv(w_.output->type, d_output_, d_xn_, d_logits_, w_.output->ne0, w_.output->ne1);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return h_logits_;
        }
        forward_capture();
    }

    CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t) graph_exec_, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_logits_;
}

const float* Model::impl::forward_batch(const int32_t* tokens, int batch, int n_past) {
    const int ne = hp_.n_embd, nhead = hp_.n_head, hdim = hp_.head_dim;
    const int nkv = hp_.n_head_kv, nff = hp_.n_ff, n_ctx = hp_.n_ctx;
    const float scale = 1.0f / std::sqrt((float)hdim);

    auto matmul = [batch](const Tensor* t, const uint8_t* dw, const float* in, float* out) {
        cuda::gemv_batch(t->type, dw, in, out, t->ne0, t->ne1, batch);
    };

    cudaMemcpy(d_tokens_, tokens, batch * sizeof(int32_t), cudaMemcpyHostToDevice);
    cuda::dequant_rows(w_.token_embd->type, d_token_embd_, d_tokens_, d_x_, ne, batch);

    const size_t per_layer = (size_t)nkv * n_ctx * hdim;
    float* d_last_x = d_x_ + (size_t)(batch - 1) * ne;
    for (int l = 0; l < hp_.n_layer; ++l) {
        const Layer& L = w_.layers[l];
        const DevLayer& D = dev_layers_[l];
        uint16_t* kbase = d_kc_ + l * per_layer;
        uint16_t* vbase = d_vc_ + l * per_layer;

        // attention
        cuda::rms_norm_rows(d_x_, D.attn_norm, batch, ne, hp_.rms_eps, d_xn_);
        matmul(L.wq, D.wq, d_xn_, d_q_);
        matmul(L.wk, D.wk, d_xn_, d_k_);
        matmul(L.wv, D.wv, d_xn_, d_v_);

        if (D.bq) cuda::add_inplace_broadcast(d_q_, D.bq, nhead * hdim, batch);
        if (D.bk) cuda::add_inplace_broadcast(d_k_, D.bk, nkv * hdim, batch);
        if (D.bv) cuda::add_inplace_broadcast(d_v_, D.bv, nkv * hdim, batch);

        if (D.wq_norm) cuda::rms_norm_rows(d_q_, D.wq_norm, batch * nhead, hdim, hp_.rms_eps, d_q_);
        if (D.wk_norm) cuda::rms_norm_rows(d_k_, D.wk_norm, batch * nkv, hdim, hp_.rms_eps, d_k_);

        auto rope_batch = hp_.rope_type == 1 ? cuda::rope_neox_batch : cuda::rope_normal_batch;
        rope_batch(d_q_, batch * nhead, hdim, hp_.n_rot, hp_.rope_theta, n_past, nhead, d_rope_ff_);
        rope_batch(d_k_, batch * nkv, hdim, hp_.n_rot, hp_.rope_theta, n_past, nkv, d_rope_ff_);

        cuda::kv_store_batch(kbase, vbase, d_k_, d_v_, nkv, hdim, n_ctx, n_past, batch);
        cuda::flash_attention(d_q_, kbase, vbase, d_attn_, nhead, nkv, hdim, n_ctx, n_past, batch, scale);

        matmul(L.wo, D.wo, d_attn_, d_o_);
        cuda::add_inplace(d_x_, d_o_, batch * ne);

        // FFN
        if (l < hp_.n_layer - 1) {
            cuda::rms_norm_rows(d_x_, D.ffn_norm, batch, ne, hp_.rms_eps, d_xn2_);
            matmul(L.ffn_gate, D.ffn_gate, d_xn2_, d_gate_);
            matmul(L.ffn_up,   D.ffn_up,   d_xn2_, d_up_);
            cuda::swiglu(d_act_, d_gate_, d_up_, batch * nff);
            matmul(L.ffn_down, D.ffn_down, d_act_, d_o_);
            cuda::add_inplace(d_x_, d_o_, batch * ne);
        } else { // only the last token needs final FFN
            cuda::rms_norm(d_last_x, D.ffn_norm, ne, hp_.rms_eps, d_xn2_);
            cuda::quantize_activation(d_xn2_, ne);
            cuda::gemv_fused_gate_up_swiglu(L.ffn_gate->type, D.ffn_gate, D.ffn_up, d_act_, L.ffn_gate->ne0, nff);
            cuda::quantize_activation(d_act_, nff);
            cuda::gemv_q8_add(L.ffn_down->type, D.ffn_down, d_last_x, L.ffn_down->ne0, L.ffn_down->ne1);
        }
    }

    cuda::rms_norm(d_last_x, d_output_norm_, ne, hp_.rms_eps, d_xn_);
    cuda::gemv(w_.output->type, d_output_, d_xn_, d_logits_, w_.output->ne0, w_.output->ne1);

    CUDA_CHECK(cudaDeviceSynchronize());
    return h_logits_;
}

void Model::impl::kv_shift(int n_keep, int shift) {
    if (shift <= 0 || shift > hp_.n_ctx) return;
    cuda::kv_shift(d_kc_, d_vc_, hp_.n_layer, hp_.n_head_kv, hp_.head_dim, hp_.n_ctx, n_keep, shift);
}

Model::impl::~impl() {
    if (mapped_ptr_) cudaHostUnregister(mapped_ptr_);
    // free buffers
    cudaFree(d_x_);
    cudaFree(d_kc_);
    cudaFreeHost(h_logits_);
    // free graph
    if (graph_exec_) cudaGraphExecDestroy((cudaGraphExec_t)graph_exec_);
    if (graph_) cudaGraphDestroy((cudaGraph_t)graph_);
    if (graph_stream_) cudaStreamDestroy((cudaStream_t)graph_stream_);
}

// ---- forwarding ----

Model::Model() : pimpl_(std::make_unique<impl>(hparams)) {}
Model::~Model() = default;

void Model::load(const GGUF& g) { pimpl_->load(g); }
void Model::init_buffers(int n_ctx, int max_batch) { pimpl_->init_buffers(n_ctx, max_batch); }
void Model::print_memory() const { pimpl_->print_memory(); }

const float* Model::forward(int32_t token, int n_past) { return pimpl_->forward(token, n_past); }
const float* Model::forward_batch(const int32_t* tokens, int batch, int n_past) { return pimpl_->forward_batch(tokens, batch, n_past); }

void Model::kv_shift(int n_keep, int shift) { pimpl_->kv_shift(n_keep, shift); }

} // namespace llm
