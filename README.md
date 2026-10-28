# orin\_llm

轻量级高性能 LLM 推理引擎，针对 NVIDIA Jetson Orin 硬件平台优化。针对 Orin 的统一内存、SM 8.7 架构、SM 数量少、有限带宽、端侧资源受限等硬件特性，优化 GGUF 量化模型（Qwen2、LLaMA 等）推理，实现性能对齐并在部分场景下超越 llama.cpp，其中 decode 性能提高 1.6 倍。

约 5000 行代码量，参考 llama.cpp 最小化实现核心推理逻辑，无外部框架依赖，仅需 CUDA Runtime。

## 特性

- GGUF 模型加载：Q4\_K\_M / Q6\_K / Q8\_0，支持 Qwen2/2.5 和 LLaMA 架构
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

测试模型 Qwen2.5-1.5B Q4\_K\_M，prefill 分别测 45 / 512 token，decode 测 128 token 生成，取 5-6 轮平均。详细 kernel 数据见 [docs/nsys\_report.md](docs/nsys_report.md)。

### Prefill（45 token）

| 算子                       | orin\_llm  | llama.cpp  | 加速比      |
| ------------------------ | --------- | --------- | --------- |
| Q4\_K matmul              | 25.72 ms  | 26.04 ms  | 1.01x     |
| Q6\_K matmul              | 5.66 ms   | 5.26 ms   | 0.93x     |
| MMQ fixup                | 1.01 ms   | 1.05 ms   | 1.04x       |
| Q6\_K GEMV（output head）   | 2.90 ms   | 5.12 ms   | **1.77x** |
| Flash Attention          | 0.48 ms   | 0.89 ms   | **1.85x** |
| SwiGLU                   | 1.61 ms   | 1.70 ms   | 1.06x     |
| RMSNorm                  | 0.91 ms   | 1.18 ms   | 1.30x     |
| RoPE                     | 0.30 ms   | 0.55 ms   | **1.83x** |
| Residual Add             | 1.07 ms   | 0.99 ms   | 0.92x     |
| Q8\_1 量化                  | 1.54 ms   | 1.61 ms   | 1.05x       |
| KV Store                 | 0.13 ms   | 0.11 ms   | 0.85x       |
| 其他                       | 0.02 ms   | 0.01 ms   | --        |
| **Kernel 总计**            | **41.35 ms** | **44.51 ms** | **1.08x** |

### Prefill（512 token）

| 算子                       | orin\_llm   | llama.cpp   | 加速比      |
| ------------------------ | ---------- | ---------- | --------- |
| Q4\_K matmul              | 184.47 ms  | 178.43 ms  | 0.97x     |
| Q6\_K matmul              | 37.08 ms   | 34.75 ms   | 0.94x     |
| Q6\_K GEMV（output head）   | 2.89 ms    | 5.14 ms    | **1.78x** |
| Flash Attention          | 9.56 ms    | 12.23 ms   | **1.28x** |
| SwiGLU                   | 17.87 ms   | 19.04 ms   | 1.07x     |
| RMSNorm                  | 11.32 ms   | 15.17 ms   | **1.34x** |
| RoPE                     | 2.66 ms    | 5.91 ms    | **2.22x** |
| Residual Add             | 10.38 ms   | 8.61 ms    | 0.83x     |
| Q8\_1 量化                | 15.49 ms   | 14.98 ms   | 0.97x    |
| KV Store                 | 0.86 ms    | 0.58 ms    | 0.67x     |
| 其他                       | 0.10 ms    | 0.02 ms    | --        |
| **Kernel 总计**            | **292.68 ms** | **294.86 ms** | **1.01x** |

### Decode（128 token）

| 算子              | orin\_llm    | llama.cpp    | 加速比       |
| --------------- | ----------- | ----------- | --------- |
| GEMV            | 14.22 ms    | 23.23 ms    | **1.63x** |
| Flash Attention | 0.31 ms     | 0.36 ms     | 1.17x     |
| RMSNorm         | 0.23 ms     | 0.33 ms     | 1.45x     |
| Q8 量化           | 0.21 ms     | 0.36 ms     | 1.69x     |
| RoPE            | 0.09 ms     | 0.15 ms     | 1.62x     |
| KV Store        | 0.07 ms     | 0.08 ms     | 1.12x     |
| **总计**          | **15.14 ms** | **24.52 ms** | **1.62x** |

### 分析

**GEMM**：矩阵乘占 Prefill 的 75% 左右，两边都用 MMA + Stream-K，性能接近，kernel 总时间基本持平。

**GEMV**：Decode 90% 以上耗时在 GEMV，两边都使用 dp4a、smallK/largeK 分派和 gate+up+SwiGLU 融合。
1.63 倍加速主要来自 smallK 路径的归约策略差异：orin\_llm smallK 路径每 warp 独立处理一行，仅 warp 内归约；llama.cpp 用 4 warps 协作 + shared memory 归约，在主流大 GPU 上开销可忽略，但在 Orin 这种资源受限平台上成为瓶颈。
对于 Q6\_K output head（K=1536, 6 个量化 block），llama.cpp 因 smallK 阈值不满足而走了更慢的 largeK 路径。
进行对比实验：改用共享内存归约后 TPOT +37%，强制 Q6\_K output head 走 largeK 后 TPOT +12%。

**Flash Attention**：MMA (m16n8k16) + online softmax，prefill/decode 各一个专用 kernel。
Prefill 快 1.28-1.87 倍：每 warp 独立处理 2 个 Q token（无跨 warp 归约），因果掩码内联计算（llama.cpp 需支持多种 mask 模式，从全局内存加载通用 mask）。
Decode 快 1.17 倍，FA kernel 本身耗时接近（~11 us），优势来自 orin\_llm 不需要 Stream-K 调度，省掉 fixup kernel（~2.1 us）；llama.cpp 在 Orin 上 Decode 阶段 tile 数少于 SM 数，触发 Stream-K + fixup。

**辅助算子**：RMSNorm 快 1.3-1.5 倍，Q8 量化快 1.7 倍，RoPE 快 1.6-2.2 倍。这些算子去掉了 llama.cpp 中为通用性所需的多维 stride 索引、类型转换等逻辑，仅保留 Orin 推理所需的最小逻辑。

## 许可

MIT License，详见 [LICENSE](LICENSE)。
