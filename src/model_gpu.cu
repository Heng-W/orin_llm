
#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <utility>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include "kernels.h"
#include "quant.h"

namespace llm {

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t e_ = (call);                                   \
        if (e_ != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n",            \
                    cudaGetErrorString(e_), __FILE__, __LINE__);   \
            abort();                                               \
        }                                                          \
    } while (0)

static float* dev_alloc(size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return (float*) p;
}

static float* dev_copy(const float* src, size_t n) {
    if (!src) return nullptr;
    float* p = dev_alloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(p, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}

static const uint8_t* dev_copy_tensor(const Tensor* t) {
    if (!t) return nullptr;
    const size_t bytes = tensor_nbytes(t->type, t->ne0, t->ne1);
    uint8_t* p = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&p, bytes));
    CUDA_CHECK(cudaMemcpy(p, t->data, bytes, cudaMemcpyHostToDevice));
    return p;
}

static void dev_free(const void* p) {
    cudaFree(const_cast<void*>(p));
}

struct Model::impl {
    // GGUF parsed weights (host-side pointers)
    ModelWeights w;
    HParams* hp_ = nullptr;

    // device weights
    const uint8_t* d_token_embd_ = nullptr;
    const uint8_t* d_output_     = nullptr;
    int weight_mode_ = 0;  // 0=cudaHostRegister (zero-copy), 1=cudaMalloc+copy
    void* zc_reg_ptr_ = nullptr;
    size_t zc_reg_size_ = 0;

    struct DevLayer {
        const uint8_t* wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        const uint8_t* ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
    };
    std::vector<DevLayer> dev_layers_;

    // device activation buffers
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

    // device norms
    std::vector<float*> d_attn_norm_, d_ffn_norm_, d_wq_norm_, d_wk_norm_;
    std::vector<float*> d_bq_, d_bk_, d_bv_;
    float* d_output_norm_ = nullptr;

    // CUDA Graph
    int* d_pos_ = nullptr;
    uint8_t* d_emb_row_ = nullptr;
    float* d_rope_ff_ = nullptr;  // device copy of rope freq_factors (nullptr if no scaling)
    void* graph_stream_ = nullptr;
    void* graph_ = nullptr;
    void* graph_exec_ = nullptr;
    bool graph_captured_ = false;

    // Per-layer decode kernel sequence
    void decode_layer(int l);

    void load(const GGUF& g, HParams& hparams, int weight_mode);
    void init_buffers(HParams& hparams, int max_batch);
    const float* forward(HParams& hparams, int32_t token, int n_past);
    const float* forward_batch(HParams& hparams, const int32_t* tokens, int batch, int n_past);
    void kv_shift(HParams& hparams, int n_keep, int shift);

    void forward_capture(HParams& hparams);

    void free_graph();
    void free_weights();
    void free_buffers();
    void free_norms();
};

// ---------------------------------------------------------------------------
// Per-layer decode kernel sequence - the single source of truth
// ---------------------------------------------------------------------------
void Model::impl::decode_layer(int l) {
    const auto& hp = *hp_;
    const int ne = hp.n_embd, nhead = hp.n_head, nkv = hp.n_head_kv, d = hp.head_dim;
    const int nff = hp.n_ff, n_ctx = hp.n_ctx;
    const float scale = 1.0f / std::sqrt((float) d);
    const size_t per_layer = (size_t) nkv * n_ctx * d;

    const Layer& L = w.layers[l];
    const DevLayer& D = dev_layers_[l];
    uint16_t* kbase = d_kc_ + (size_t) l * per_layer;
    uint16_t* vbase = d_vc_ + (size_t) l * per_layer;

    cuda::rms_norm(d_x_, d_attn_norm_[l], ne, hp.rms_eps, d_xn_);
    cuda::quantize_activation(d_xn_, ne);
    if (d_bq_[l]) {
        cuda::gemv_q8_bias((int) L.wq->type, D.wq, d_bq_[l], d_q_, (int) L.wq->ne0, (int) L.wq->ne1);
        cuda::gemv_q8_bias((int) L.wk->type, D.wk, d_bk_[l], d_k_, (int) L.wk->ne0, (int) L.wk->ne1);
        cuda::gemv_q8_bias((int) L.wv->type, D.wv, d_bv_[l], d_v_, (int) L.wv->ne0, (int) L.wv->ne1);
    } else {
        cuda::gemv_q8((int) L.wq->type, D.wq, d_q_, (int) L.wq->ne0, (int) L.wq->ne1);
        cuda::gemv_q8((int) L.wk->type, D.wk, d_k_, (int) L.wk->ne0, (int) L.wk->ne1);
        cuda::gemv_q8((int) L.wv->type, D.wv, d_v_, (int) L.wv->ne0, (int) L.wv->ne1);
    }

    if (d_wq_norm_[l]) cuda::rms_norm_rows(d_q_, d_wq_norm_[l], nhead, d, hp.rms_eps, d_q_);
    if (d_wk_norm_[l]) cuda::rms_norm_rows(d_k_, d_wk_norm_[l], nkv, d, hp.rms_eps, d_k_);

    if (hp.rope_type == 1) {
        cuda::rope_neox_ind(d_q_, nhead, d, hp.n_rot, hp.rope_theta, d_pos_, d_rope_ff_);
        cuda::rope_neox_ind(d_k_, nkv, d, hp.n_rot, hp.rope_theta, d_pos_, d_rope_ff_);
    } else {
        cuda::rope_normal_ind(d_q_, nhead, d, hp.n_rot, hp.rope_theta, d_pos_, d_rope_ff_);
        cuda::rope_normal_ind(d_k_, nkv, d, hp.n_rot, hp.rope_theta, d_pos_, d_rope_ff_);
    }
    cuda::kv_store_ind(kbase, vbase, d_k_, d_v_, nkv, d, n_ctx, d_pos_);
    cuda::flash_attention_decode_ind(d_q_, kbase, vbase, d_attn_, nhead, nkv, d, n_ctx, d_pos_, scale);

    cuda::quantize_activation(d_attn_, ne);
    cuda::gemv_q8_add((int) L.wo->type, D.wo, d_x_, (int) L.wo->ne0, (int) L.wo->ne1);

    cuda::rms_norm(d_x_, d_ffn_norm_[l], ne, hp.rms_eps, d_xn2_);
    cuda::quantize_activation(d_xn2_, ne);
    cuda::gemv_fused_gate_up_swiglu((int) L.ffn_gate->type, D.ffn_gate, D.ffn_up, d_act_, (int) L.ffn_gate->ne0, nff);
    cuda::quantize_activation(d_act_, nff);
    cuda::gemv_q8_add((int) L.ffn_down->type, D.ffn_down, d_x_, (int) L.ffn_down->ne0, (int) L.ffn_down->ne1);
}

// ---------------------------------------------------------------------------
// Resource cleanup
// ---------------------------------------------------------------------------
void Model::impl::free_weights() {
    if (weight_mode_ == 1) {
        for (DevLayer& D : dev_layers_) {
            dev_free(D.wq); dev_free(D.wk); dev_free(D.wv); dev_free(D.wo);
            dev_free(D.ffn_gate); dev_free(D.ffn_up); dev_free(D.ffn_down);
        }
        if (d_output_ && d_output_ != d_token_embd_) dev_free(d_output_);
        dev_free(d_token_embd_);
    } else if (weight_mode_ == 0 && zc_reg_ptr_) {
        cudaHostUnregister(zc_reg_ptr_);
    }
    zc_reg_ptr_ = nullptr;
    zc_reg_size_ = 0;
    weight_mode_ = 0;
    dev_layers_.clear();
    d_token_embd_ = d_output_ = nullptr;
}

void Model::impl::free_norms() {
    for (float* p : d_attn_norm_) dev_free(p);
    for (float* p : d_ffn_norm_) dev_free(p);
    for (float* p : d_wq_norm_) dev_free(p);
    for (float* p : d_wk_norm_) dev_free(p);
    for (float* p : d_bq_) dev_free(p);
    for (float* p : d_bk_) dev_free(p);
    for (float* p : d_bv_) dev_free(p);
    dev_free(d_output_norm_);
    d_attn_norm_.clear();
    d_ffn_norm_.clear();
    d_wq_norm_.clear();
    d_wk_norm_.clear();
    d_bq_.clear();
    d_bk_.clear();
    d_bv_.clear();
    d_output_norm_ = nullptr;
}

void Model::impl::free_buffers() {
    dev_free(d_x_); dev_free(d_xn_); dev_free(d_xn2_); dev_free(d_q_); dev_free(d_k_); dev_free(d_v_);
    // d_gate_ and d_up_ are aliases of d_q_ and d_attn_ -- do not free
    dev_free(d_attn_); dev_free(d_o_); dev_free(d_act_);
    dev_free(d_kc_); dev_free(d_vc_); dev_free(d_tokens_);
    dev_free(d_pos_); dev_free(d_emb_row_); dev_free(d_rope_ff_);
    if (h_logits_) { cudaFreeHost(h_logits_); h_logits_ = nullptr; }
    d_x_ = d_xn_ = d_xn2_ = d_q_ = d_k_ = d_v_ = nullptr;
    d_attn_ = d_o_ = d_gate_ = d_up_ = d_act_ = nullptr;
    d_scores_ = d_logits_ = nullptr;
    d_kc_ = d_vc_ = nullptr;
    d_tokens_ = nullptr;
    d_pos_ = nullptr;
    d_rope_ff_ = nullptr;
    d_emb_row_ = nullptr;
    free_graph();
}

void Model::impl::free_graph() {
    if (graph_exec_) {
        cudaGraphExecDestroy((cudaGraphExec_t)graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy((cudaGraph_t)graph_);
        graph_ = nullptr;
    }
    if (graph_stream_) {
        cudaStreamDestroy((cudaStream_t)graph_stream_);
        graph_stream_ = nullptr;
    }
    graph_captured_ = false;
}

// ---------------------------------------------------------------------------
// Impl methods: contain the actual logic
// ---------------------------------------------------------------------------

void Model::impl::load(const GGUF& g, HParams& hparams, int weight_mode) {
    hp_ = &hparams;
    load_weights(g, hparams, w);

    if (cuda::init() != 0) LLM_ABORT("CUDA init failed, check CUDA runtime");

    cuda::device_info();

    free_weights();
    weight_mode_ = weight_mode;

    if (weight_mode == 0) {
        const uintptr_t page = sysconf(_SC_PAGESIZE);
        uintptr_t reg_start = (uintptr_t)(g.data + g.data_offset) & ~(page - 1);
        uintptr_t reg_end   = ((uintptr_t) g.data + g.size + page - 1) & ~(page - 1);
        zc_reg_ptr_  = (void*) reg_start;
        zc_reg_size_ = (size_t)(reg_end - reg_start);
        CUDA_CHECK(cudaHostRegister(zc_reg_ptr_, zc_reg_size_, cudaHostRegisterMapped));
    }

    auto dev_map = [](const Tensor* t) -> const uint8_t* {
        if (!t) return nullptr;
        void* dp = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&dp, (void*)t->data, 0));
        return (const uint8_t*)dp;
    };

    auto load_tensor = [&](const Tensor* t) -> const uint8_t* {
        if (weight_mode == 0) return dev_map(t);
        return dev_copy_tensor(t);
    };

    d_token_embd_ = load_tensor(w.token_embd);
    d_output_ = (w.output == w.token_embd) ? d_token_embd_ : load_tensor(w.output);

    dev_layers_.resize(hparams.n_layer);
    for (int i = 0; i < hparams.n_layer; ++i) {
        const Layer& L = w.layers[i];
        DevLayer& D = dev_layers_[i];
        D.wq       = load_tensor(L.wq);
        D.wk       = load_tensor(L.wk);
        D.wv       = load_tensor(L.wv);
        D.wo       = load_tensor(L.wo);
        D.ffn_gate = load_tensor(L.ffn_gate);
        D.ffn_up   = load_tensor(L.ffn_up);
        D.ffn_down = load_tensor(L.ffn_down);
    }

    free_norms();
    for (int i = 0; i < hparams.n_layer; ++i) {
        const Layer& L = w.layers[i];
        d_attn_norm_.push_back(dev_copy(L.attn_norm, hparams.n_embd));
        d_ffn_norm_.push_back(dev_copy(L.ffn_norm, hparams.n_embd));
        d_wq_norm_.push_back(dev_copy(L.wq_norm, hparams.head_dim));
        d_wk_norm_.push_back(dev_copy(L.wk_norm, hparams.head_dim));
        d_bq_.push_back(dev_copy(L.bq, hparams.n_head * hparams.head_dim));
        d_bk_.push_back(dev_copy(L.bk, hparams.n_head_kv * hparams.head_dim));
        d_bv_.push_back(dev_copy(L.bv, hparams.n_head_kv * hparams.head_dim));
    }
    d_output_norm_ = dev_copy(w.output_norm, hparams.n_embd);

    if (weight_mode_ == 1 && g.data_offset < g.size) {
        madvise((void*)(g.data + g.data_offset), g.size - g.data_offset, MADV_DONTNEED);
    }
}

void Model::impl::init_buffers(HParams& hparams, int max_batch) {
    max_batch_ = max_batch < 1 ? 1 : max_batch;
    free_buffers();

    const int ne = hparams.n_embd, nhead = hparams.n_head, nkv = hparams.n_head_kv;
    const int hd = hparams.head_dim, nff = hparams.n_ff;
    const int B = max_batch_;

    d_x_   = dev_alloc((size_t) B * ne * 4);
    d_xn_  = dev_alloc((size_t) B * ne * 4);
    d_xn2_ = dev_alloc((size_t) B * ne * 4);

    const size_t qa_sz = (size_t) B * nhead * hd * 4;
    const size_t ff_sz = (size_t) B * nff * 4;
    const size_t big = std::max(qa_sz, ff_sz);

    d_q_    = dev_alloc(big);
    d_k_    = dev_alloc((size_t) B * nkv * hd * 4);
    d_v_    = dev_alloc((size_t) B * nkv * hd * 4);
    d_attn_ = dev_alloc(big);
    d_o_    = dev_alloc((size_t) B * ne * 4);
    d_act_  = dev_alloc(big);
    d_gate_ = d_q_;
    d_up_   = d_attn_;

    CUDA_CHECK(cudaHostAlloc((void**) &h_logits_, (size_t) hparams.n_vocab * 4, cudaHostAllocMapped));
    void* d_logits_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_logits_ptr, h_logits_, 0));
    d_logits_ = (float*) d_logits_ptr;

    d_tokens_ = (int32_t*) dev_alloc((size_t) B * sizeof(int32_t));

    const size_t kv_bytes = (size_t) hparams.n_layer * nkv * hparams.n_ctx * hd * 2;
    d_kc_ = (uint16_t*) dev_alloc(kv_bytes);
    d_vc_ = (uint16_t*) dev_alloc(kv_bytes);

    d_pos_ = (int*) dev_alloc(sizeof(int));
    const uint64_t row_bytes = tensor_nbytes(w.token_embd->type, ne, 1);
    CUDA_CHECK(cudaMalloc((void**) &d_emb_row_, row_bytes));
    CUDA_CHECK(cudaStreamCreateWithFlags((cudaStream_t*) &graph_stream_, cudaStreamNonBlocking));

    if (!hparams.rope_freq_factors.empty()) {
        const size_t ff_bytes = hparams.rope_freq_factors.size() * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**) &d_rope_ff_, ff_bytes));
        CUDA_CHECK(cudaMemcpy(d_rope_ff_, hparams.rope_freq_factors.data(), ff_bytes, cudaMemcpyHostToDevice));
    }

    const int max_in = nff > ne ? nff : ne;
    cuda::init_buffers(max_in, B);
}

void Model::impl::forward_capture(HParams& hparams) {
    const int ne = hparams.n_embd;
    cudaStream_t stream = (cudaStream_t) graph_stream_;

    cuda::set_stream(stream);
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    cuda::dequant_row((int) w.token_embd->type, d_emb_row_, d_x_, ne);
    for (int l = 0; l < hparams.n_layer; ++l) decode_layer(l);
    cuda::rms_norm(d_x_, d_output_norm_, ne, hparams.rms_eps, d_xn_);
    cuda::gemv((int) w.output->type, d_output_, d_xn_, d_logits_, (int) w.output->ne0, (int) w.output->ne1);

    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cuda::set_stream(nullptr);
    graph_ = graph;

    cudaGraphExec_t exec;
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    graph_exec_ = exec;
    graph_captured_ = true;

    // fprintf(stderr, "[cuda] CUDA Graph captured (%d layers, ~%d kernels)\n",
    //         hparams.n_layer, hparams.n_layer * 20 + 3);
}

const float* Model::impl::forward(HParams& hparams, int32_t token, int n_past) {
    hp_ = &hparams;
    const int ne = hparams.n_embd;
    const uint64_t row_bytes = tensor_nbytes(w.token_embd->type, ne, 1);
    cudaStream_t stream = (cudaStream_t) graph_stream_;

    CUDA_CHECK(cudaMemcpyAsync(d_emb_row_, d_token_embd_ + (uint64_t) token * row_bytes,
                               row_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, &n_past, sizeof(int), cudaMemcpyHostToDevice, stream));

    if (!graph_captured_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (getenv("NO_GRAPH")) {
            cuda::set_stream(stream);
            cuda::dequant_row((int) w.token_embd->type, d_emb_row_, d_x_, ne);
            for (int l = 0; l < hparams.n_layer; ++l) decode_layer(l);
            cuda::rms_norm(d_x_, d_output_norm_, ne, hparams.rms_eps, d_xn_);
            cuda::gemv((int) w.output->type, d_output_, d_xn_, d_logits_, (int) w.output->ne0, (int) w.output->ne1);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return h_logits_;
        }

        forward_capture(hparams);
    }

    CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t) graph_exec_, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_logits_;
}

const float* Model::impl::forward_batch(HParams& hparams, const int32_t* tokens, int batch, int n_past) {
    hp_ = &hparams;
    const int ne = hparams.n_embd, nhead = hparams.n_head, nkv = hparams.n_head_kv;
    const int hd = hparams.head_dim, nff = hparams.n_ff, n_ctx = hparams.n_ctx;
    const float scale = 1.0f / std::sqrt((float) hd);

    cudaMemcpy(d_tokens_, tokens, (size_t) batch * sizeof(int32_t), cudaMemcpyHostToDevice);
    cuda::dequant_rows((int) w.token_embd->type, d_token_embd_, d_tokens_, d_x_, ne, batch);

    const size_t per_layer = (size_t) nkv * n_ctx * hd;
    for (int l = 0; l < hparams.n_layer; ++l) {
        const Layer& L = w.layers[l];
        const DevLayer& D = dev_layers_[l];
        uint16_t* kbase = d_kc_ + (size_t) l * per_layer;
        uint16_t* vbase = d_vc_ + (size_t) l * per_layer;

        cuda::rms_norm_rows(d_x_, d_attn_norm_[l], batch, ne, hparams.rms_eps, d_xn_);
        cuda::gemv_batch((int) L.wq->type, D.wq, d_xn_, d_q_, (int) L.wq->ne0, (int) L.wq->ne1, batch);
        cuda::gemv_batch((int) L.wk->type, D.wk, d_xn_, d_k_, (int) L.wk->ne0, (int) L.wk->ne1, batch);
        cuda::gemv_batch((int) L.wv->type, D.wv, d_xn_, d_v_, (int) L.wv->ne0, (int) L.wv->ne1, batch);

        if (d_bq_[l]) cuda::add_inplace_broadcast(d_q_, d_bq_[l], nhead * hd, batch);
        if (d_bk_[l]) cuda::add_inplace_broadcast(d_k_, d_bk_[l], nkv * hd, batch);
        if (d_bv_[l]) cuda::add_inplace_broadcast(d_v_, d_bv_[l], nkv * hd, batch);

        if (d_wq_norm_[l]) cuda::rms_norm_rows(d_q_, d_wq_norm_[l], batch * nhead, hd, hparams.rms_eps, d_q_);
        if (d_wk_norm_[l]) cuda::rms_norm_rows(d_k_, d_wk_norm_[l], batch * nkv, hd, hparams.rms_eps, d_k_);
        if (hparams.rope_type == 1) {
            cuda::rope_neox_batch(d_q_, batch * nhead, hd, hparams.n_rot, hparams.rope_theta, n_past, nhead, d_rope_ff_);
            cuda::rope_neox_batch(d_k_, batch * nkv, hd, hparams.n_rot, hparams.rope_theta, n_past, nkv, d_rope_ff_);
        } else {
            cuda::rope_normal_batch(d_q_, batch * nhead, hd, hparams.n_rot, hparams.rope_theta, n_past, nhead, d_rope_ff_);
            cuda::rope_normal_batch(d_k_, batch * nkv, hd, hparams.n_rot, hparams.rope_theta, n_past, nkv, d_rope_ff_);
        }

        cuda::kv_store_batch(kbase, vbase, d_k_, d_v_, nkv, hd, n_ctx, n_past, batch);
        cuda::flash_attention(d_q_, kbase, vbase, d_attn_, nhead, nkv, hd, n_ctx, n_past, batch, scale);

        cuda::gemv_batch((int) L.wo->type, D.wo, d_attn_, d_o_, (int) L.wo->ne0, (int) L.wo->ne1, batch);
        cuda::add_inplace(d_x_, d_o_, batch * ne);

        cuda::rms_norm_rows(d_x_, d_ffn_norm_[l], batch, ne, hparams.rms_eps, d_xn2_);
        cuda::gemv_batch((int) L.ffn_gate->type, D.ffn_gate, d_xn2_, d_gate_, (int) L.ffn_gate->ne0, (int) L.ffn_gate->ne1, batch);
        cuda::gemv_batch((int) L.ffn_up->type,   D.ffn_up,   d_xn2_, d_up_,   (int) L.ffn_up->ne0,   (int) L.ffn_up->ne1, batch);
        cuda::swiglu(d_act_, d_gate_, d_up_, batch * nff);
        cuda::gemv_batch((int) L.ffn_down->type, D.ffn_down, d_act_, d_o_, (int) L.ffn_down->ne0, (int) L.ffn_down->ne1, batch);
        cuda::add_inplace(d_x_, d_o_, batch * ne);
    }

    cuda::rms_norm(d_x_ + (size_t)(batch - 1) * ne, d_output_norm_, ne, hparams.rms_eps, d_xn_);
    cuda::gemv((int) w.output->type, d_output_, d_xn_, d_logits_, (int) w.output->ne0, (int) w.output->ne1);

    CUDA_CHECK(cudaDeviceSynchronize());
    return h_logits_;
}

void Model::impl::kv_shift(HParams& hparams, int n_keep, int shift) {
    if (shift <= 0 || shift > hparams.n_ctx) return;
    cuda::kv_shift(d_kc_, d_vc_, hparams.n_layer, hparams.n_head_kv, hparams.head_dim, hparams.n_ctx, n_keep, shift);
}


// --- Model forwarding (pimpl) ---

Model::Model() : pimpl_(std::make_unique<impl>()) {}

Model::~Model() {
    pimpl_->free_buffers();
    pimpl_->free_norms();
    pimpl_->free_weights();
}

void Model::load(const GGUF& g, int weight_mode) { pimpl_->load(g, hparams, weight_mode); }

void Model::init_buffers(int n_ctx, int max_batch) { hparams.n_ctx = n_ctx; pimpl_->init_buffers(hparams, max_batch); }

const float* Model::forward(int32_t token, int n_past) { return pimpl_->forward(hparams, token, n_past); }

const float* Model::forward_batch(const int32_t* tokens, int batch, int n_past) { return pimpl_->forward_batch(hparams, tokens, batch, n_past); }

void Model::kv_shift(int n_keep, int shift) { pimpl_->kv_shift(hparams, n_keep, shift); }

} // namespace llm
