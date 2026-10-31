# orin\_llm

轻量级高性能 LLM 推理引擎，针对 NVIDIA Jetson Orin 硬件平台优化。针对 Orin 的统一内存、SM 8.7 架构、SM 数量少、有限带宽、端侧资源受限等硬件特性，优化 GGUF 量化模型（Qwen2、LLaMA 等）推理，实现性能对齐并在部分场景下超越 llama.cpp，如 1.5B Q4\_K\_M 模型 decode 性能提高 1.6 倍。

约 5000 行代码量，参考 llama.cpp 最小化实现核心推理逻辑，无外部框架依赖，仅需 CUDA Runtime。

## 特性

- GGUF 模型加载：Q4\_K\_M / Q6\_K / Q8\_0，支持 Qwen2/2.5 和 LLaMA 架构，已验证 8B 及以下参数的部分模型
- 融合反量化 + GEMV kernel，使用 `dp4a` 指令，按矩阵尺寸分 smallK/largeK 两套 kernel
- MMA GEMM + Stream-K，用于 prefill
- Flash Attention MMA 实现（GQA）
- 融合 gate + up 投影 + SwiGLU 于单个 kernel
- Orin 硬件调优（统一内存零拷贝、SM、带宽等）

## 构建

已在 NVIDIA Jetson Orin Nano Super 8g 平台上测试

```sh
make                 # CUDA 构建 (sm_87)
make USE_CPU_ONLY=1  # 纯 CPU 构建，仅用于调试正确性
```

## 使用

```sh
# 交互式对话
./chat_cli <model.gguf> --system "You are a helpful assistant."

# 单次推理
./chat_cli <model.gguf> -p "Hello, please tell me a story." --n-predict 128
```

| 参数              | 默认值   | 说明                           |
| --------------- | ----- | ---------------------------- |
| `-p S`          | -     | 提示词（非交互模式）                   |
| `--system S`    | -     | 系统提示词                        |
| `--ctx N`       | 模型元数据 | 上下文长度（物理内存紧张时自动缩小，最小 4096） |
| `--n-predict N` | -1    | 无限制，遇到 EOS 时停止生成             |
| `--batch N`     | 512   | Chunked Prefill 批大小          |
| `--temp F`      | 0.8   | 采样温度                         |
| `--top-k N`     | 40    | Top-k 采样                     |
| `--top-p F`     | 0.95  | Top-p 采样                     |
| `--bench N`     | -     | 重复 N 次跑 benchmark（第一次是 warmup） |
| `--dump-logits F`  | -     | dump 最后一个 token 计算的 logits 到文件，用于正确性验证 |


![demo](https://gitee.com/heng_w/imagebed/raw/master/prj/orin_llm/demo.png)

## 架构

```
chat_cli.cpp          CLI 前端，对话循环
include/orin_llm/
    engine.h          引擎 API（加载模型文件 / 生成 token）
src/
    model_gpu.cu      GPU 推理：权重加载、runtime 调度
    model_cpu.cpp     CPU 推理（仅调试用途）
    model.h           模型接口（CPU/GPU 后端）
    kernels.cu/h      全部 CUDA kernel 及 host wrapper（约 2000 行）
    gguf.cpp/h        GGUF 文件解析（基于 mmap）
    tokenizer.cpp/h   BPE 分词器，最小堆算法合并
    sampler.cpp/h     采样器，实现温度 / top-k / top-p / min-p 采样
```

## 支持的模型

支持 Qwen2/2.5 和 LLaMA 架构的 GGUF 模型。已测试：

- Meta-Llama-3-8B-Instruct.Q4\_K\_M.gguf
- Llama-3.2-3B-Instruct-Q4\_K\_M.gguf
- qwen2.5-7b-instruct-q4\_k\_m.gguf
- qwen2.5-3b-instruct-q4\_k\_m.gguf
- qwen2.5-1.5b-instruct-q4\_k\_m.gguf
- qwen2.5-1.5b-instruct-q8\_0.gguf

## 性能测试

测试环境：Jetson Orin Nano Super（8 GB LPDDR5，8 SM, sm\_87），Qwen2.5-1.5B-Instruct Q4\_K\_M

### orin\_llm
上图中的测试命令：
```sh
./chat_cli ../llama.cpp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p "Hello, what's your name? Please tell me a simple and short story." \
  --system "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." --n-predict 128
```
数据：
```
[cuda] device: Orin (sm_87, 7.4 GB)
  memory: 1059.9 MB weights (mapped), 71.5 MB buffers + 896.0 MB KV (n_ctx=32768) | 3823.1 MB free / 7619.9 MB total

[ Prompt: 45 tokens in 47.72 ms (943.03 tok/s) | Generation: 128 tokens, each 15.20 ms (65.81 tok/s) | Total: 1977.50 ms ]
```

### llama.cpp (commit 9e3b928fd, 2026-06-07, CUDA, 全部层 GPU offload)

构建命令：

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
```

测试命令：

```sh
./build/bin/llama-cli -m ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p "Hello, what's your name? Please tell me a simple and short story." \
  -n 128 -ngl 99 --verbose-prompt --verbose
```
输出
![llama.cpp 输出](https://gitee.com/heng_w/imagebed/raw/master/prj/orin_llm/llama_cpp_demo.png)
数据：
```
prompt eval time =      58.28 ms /    45 tokens (    1.30 ms per token,   772.08 tokens per second)
       eval time =    3110.88 ms /   128 tokens (   24.30 ms per token,    41.15 tokens per second)
      total time =    3169.16 ms /   173 tokens
```

### 对比

|             | orin\_llm | llama.cpp | 加速比  |
| ----------- | --------- | --------- | ----- |
| TTFT (ms)   | 47.72     | 58.28     | 1.22x |
| TPOT (ms)   | 15.20     | 24.30     | 1.60x |

## 算子级对比（nsys profiling）

测试模型 Qwen2.5-1.5B Q4\_K\_M，prefill 分别测 45 / 512 token，decode 测 128 token 生成，取 5-6 轮平均。加速比使用舍入前的耗时计算，详细 kernel 数据见 [docs/nsys\_report.md](docs/nsys_report.md)。

### Prefill（45 token）

| 算子                       | orin\_llm  | llama.cpp  | 加速比      |
| ------------------------ | --------- | --------- | --------- |
| Q4\_K MMQ                | 25.09 ms  | 25.65 ms  | 1.02x     |
| Q6\_K MMQ                | 5.27 ms   | 5.05 ms   | 0.96x     |
| MMQ fixup                | 1.02 ms   | 1.05 ms   | 1.02x     |
| 最后一层 FFN GEMV          | 0.38 ms   | 0.60 ms   | **1.58x** |
| Q6\_K GEMV（output head）   | 2.89 ms   | 5.12 ms   | **1.77x** |
| Flash Attention          | 0.49 ms   | 0.89 ms   | **1.82x** |
| SwiGLU               | 1.57 ms   | 1.70 ms   | 1.08x     |
| RMSNorm                  | 0.90 ms   | 1.18 ms   | 1.32x     |
| RoPE                     | 0.30 ms   | 0.55 ms   | **1.86x** |
| Add                     | 1.09 ms   | 0.99 ms   | 0.91x     |
| Q8\_1 量化                | 1.61 ms   | 1.61 ms   | 1.01x     |
| KV Store                 | 0.13 ms   | 0.11 ms   | 0.82x     |
| 其他                       | 0.01 ms   | 0.01 ms   | --        |
| **Kernel 总计**            | **40.74 ms** | **44.51 ms** | **1.09x** |

### Prefill（512 token）

| 算子                       | orin\_llm   | llama.cpp   | 加速比      |
| ------------------------ | ---------- | ---------- | --------- |
| Q4\_K MMQ                | 180.03 ms  | 178.04 ms  | 0.99x     |
| Q6\_K MMQ                | 34.49 ms   | 34.54 ms   | 1.00x     |
| 最后一层 FFN GEMV          | 0.37 ms    | 0.60 ms    | **1.60x** |
| Q6\_K GEMV（output head）   | 2.89 ms    | 5.14 ms    | **1.78x** |
| Flash Attention          | 9.08 ms    | 12.23 ms   | **1.35x** |
| SwiGLU               | 17.18 ms   | 19.04 ms   | 1.11x     |
| RMSNorm                  | 11.36 ms   | 15.17 ms   | **1.34x** |
| RoPE                     | 2.64 ms    | 5.91 ms    | **2.24x** |
| Add                    | 10.18 ms   | 8.61 ms    | 0.85x     |
| Q8\_1 量化                | 15.33 ms   | 14.99 ms   | 0.98x     |
| KV Store                 | 0.85 ms    | 0.58 ms    | 0.68x     |
| 其他                       | 0.09 ms    | 0.01 ms    | --        |
| **Kernel 总计**            | **284.49 ms** | **294.85 ms** | **1.04x** |

### Decode（128 token，每 token kernel 耗时）

| 算子              | orin\_llm    | llama.cpp    | 加速比       |
| --------------- | ----------- | ----------- | --------- |
| GEMV            | 14.22 ms    | 23.23 ms    | **1.63x** |
| Flash Attention | 0.31 ms     | 0.36 ms     | 1.17x     |
| RMSNorm         | 0.23 ms     | 0.33 ms     | 1.46x     |
| Q8 量化           | 0.21 ms     | 0.36 ms     | 1.69x     |
| RoPE            | 0.09 ms     | 0.15 ms     | 1.61x     |
| KV Store        | 0.07 ms     | 0.08 ms     | 1.12x     |
| **总计**          | **15.14 ms** | **24.52 ms** | **1.62x** |

### 分析

**GEMM**：矩阵乘 MMQ 占 Prefill kernel 耗时的 75% 左右（其中最后一层 FFN 仅处理末尾 token，不用 GEMM）。两边都用 MMA + Stream-K，性能接近，orin_llm 的 kernel 总耗时略短。

**GEMV**：Decode 90% 以上耗时在 GEMV，两边都使用 dp4a、smallK/largeK 分派和 gate+up+SwiGLU 融合。
1.63 倍加速主要来自 smallK 路径的归约策略差异：orin\_llm smallK 路径每 warp 独立处理一行，仅 warp 内归约；llama.cpp 路径用 4 warps 协作 + shared memory 归约，在主流大 GPU 上可能开销不高，但在 Orin 这种资源受限平台上成为瓶颈。
对于 Q6\_K output head（K=1536, 6 个量化 block），llama.cpp 因 smallK 阈值不满足而走了更慢的 largeK 路径。
进行对比实验：改用共享内存归约后 TPOT +37%，强制 Q6\_K output head 走 largeK 后 TPOT +12%。

**Flash Attention**：MMA (m16n8k16) + online softmax，prefill/decode 各一个专用 kernel。
Prefill 快 1.3-1.8 倍：每 warp 独立处理 2 个 Q token（无跨 warp 归约），因果掩码内联计算（llama.cpp 需支持多种 mask 模式，从全局内存加载通用 mask）。
Decode 快 1.17 倍，FA kernel 本身耗时接近（~11 us），优势来自 orin\_llm 不需要 Stream-K 调度，省掉 fixup kernel（~2.1 us）；llama.cpp 在 Orin 上 Decode 阶段 tile 数少于 SM 数，触发 Stream-K + fixup。

**辅助算子**：RMSNorm 快 1.3-1.5 倍，RoPE 快 1.6-2.2 倍，Q8 量化在 decode 时快 1.7 倍。这些算子精简了 llama.cpp 中为通用性所需的多维 stride 索引、类型转换等逻辑，仅保留 Orin 推理所需的最小路径。


## 精度分析

使用上图中的原始提示词输入（不加聊天模板，16 个 token），核对两边分词后的 token IDs 一致后，与 llama-debug 比较全部层的中间张量。Embedding 和第一层 attention norm 数值一致，后续张量差异如下：

| 张量 | 最大绝对差值 | 平均绝对差值 | 相对 L2 差值 |
|---|---:|---:|---:|
| 第一层 Q RoPE | 7.82e-5 | 1.20e-7 | 1.78e-7 |
| 第一层 K RoPE | 2.29e-5 | 8.74e-8 | 7.41e-9 |
| 第一层 FlashAttention | 1.67e-3 | 2.26e-5 | 1.66e-4 |
| 第一层 FFN norm | 1.40e-2 | 1.77e-3 | 4.63e-3 |
| 第一层 FFN SwiGLU | 8.58e-2 | 1.65e-3 | 1.17e-2 |
| 第一层 FFN down 输出 | 4.35e-2 | 3.47e-3 | 1.27e-2 |
| 最后一层 FFN norm，末 token | 2.24e-1 | 4.84e-2 | 5.45e-2 |
| 最后一层 FFN SwiGLU，末 token | 6.55e-1 | 2.45e-2 | 5.10e-2 |
| Final norm，末 token | 1.18 | 1.15e-1 | 3.36e-2 |
| Logits，末 token | 7.35e-1 | 1.18e-1 | 4.67e-2 |

**RoPE**：两边的数学公式等价，但计算舍入路径存在小的差异。实验对齐这一路径后，第一层 Q/K 结果完全一致。

**FlashAttention**：主要差异来自 VKQ（`softmax(QK^T) * V`）累加精度：本项目为 FP32，llama.cpp 路径为 FP16。对齐 RoPE 使 Q/K 一致后，临时改用 FP16 累加进行实验，第一层 FA 的最大绝对差从 `1.67e-3` 降至 `4.77e-7`，平均绝对差降至 `4.31e-9`，相对 L2 差降至 `5.09e-8`。实验发现剩余微小差异仍可能导致 Q8 量化取整产生差异，在后续层放大。

**FFN**：对第一层注入 llama.cpp 的相同 FFN 输入后进行实验，该层 FFN 及残差输出全部一致，说明该层原有差异来自上游传播。最后一层在相同输入下仍有差异，来源是 GEMV 激活量化 scale 的存储精度（本项目为 FP32，llama.cpp 为 FP16）。将两处 scale 按 FP16 舍入后进行实验，该层 FFN 及残差输出则全部一致。

**输出指标**：两边预测的首选 token 相同，top-5 候选集合也相同，KL 散度值为 `3.15e-3`。对于此输入样本来说，两边 logits 对应的概率分布差异很小。


## 许可

MIT License，详见 [LICENSE](LICENSE)。
