# nsys profile report

hardware: NVIDIA Jetson Orin Nano Super 8GB (SM 8.7, 8 SM)

model: Qwen2.5-1.5B-Instruct Q4\_K\_M

## 1. Prefill pp45 (6 loops)

### orin_llm

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 61.6 | 150,519,104 | 996 | 151,123.6 | 58,864.0 | 16,128 | 318,624 | 133,793.4 | void llm::cuda::mul_mat_mmq\<(llm::DType)12, (int)48, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 12.9 | 31,638,208 | 162 | 195,297.6 | 22,032.0 | 19,616 | 386,080 | 181,585.4 | void llm::cuda::mul_mat_mmq\<(llm::DType)14, (int)48, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 7.1 | 17,354,784 | 6 | 2,892,464.0 | 2,892,032.0 | 2,884,288 | 2,905,696 | 7,608.8 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)14, (llm::cuda::GemvWrite)0\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 3.9 | 9,424,352 | 162 | 58,175.0 | 58,352.0 | 56,768 | 60,768 | 926.8 | llm::cuda::swiglu_k(float *, const float *, const float *, int) |
| 3.0 | 7,269,504 | 996 | 7,298.7 | 5,728.0 | 5,152 | 24,352 | 5,021.5 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)1\>(const float *, void *, int, int, int) |
| 2.5 | 6,144,032 | 834 | 7,366.9 | 7,232.0 | 5,984 | 14,848 | 591.2 | void llm::cuda::mul_mat_mmq_fixup\<(int)48, (int)128, (bool)0\>(float *, float *, int, int, uint3, uint3, int) |
| 2.2 | 5,314,560 | 330 | 16,104.7 | 16,064.0 | 15,008 | 24,512 | 667.2 | llm::cuda::rms_norm_rows_k(const float *, const float *, int, float, float *) |
| 1.4 | 3,497,600 | 330 | 10,598.8 | 10,240.0 | 7,936 | 12,608 | 1,431.0 | llm::cuda::add_inplace_k(float *, const float *, int) |
| 1.2 | 3,049,792 | 504 | 6,051.2 | 4,736.0 | 3,840 | 13,280 | 2,210.8 | llm::cuda::add_inplace_broadcast_k(float *, const float *, int, int) |
| 1.2 | 2,921,056 | 168 | 17,387.2 | 17,424.0 | 16,384 | 26,656 | 806.8 | void llm::cuda::flash_attention_prefill_mma_k\<(int)128\>(const float *, const unsigned short *, const unsigned short *, float *, int, int, int, int, int, int, float) |
| 0.9 | 2,305,824 | 162 | 14,233.5 | 5,728.0 | 5,184 | 24,512 | 9,061.4 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)0\>(const float *, void *, int, int, int) |
| 0.7 | 1,781,792 | 336 | 5,303.0 | 5,696.0 | 3,168 | 9,920 | 1,706.7 | void llm::cuda::rope_neox_k\<(llm::cuda::PosMode)1\>(float *, int, int, float, int, const int *, int, int, float) |
| 0.5 | 1,219,936 | 6 | 203,322.7 | 203,376.0 | 202,752 | 203,872 | 431.9 | void llm::cuda::gemv_fused_gate_up_swiglu_k\<(llm::DType)12\>(const void *, const void *, const signed char *, const float *, float *, int, int) |
| 0.4 | 1,038,208 | 6 | 173,034.7 | 171,024.0 | 170,688 | 183,328 | 5,048.3 | void llm::cuda::gemv_dp4a_k\<(llm::DType)14, (bool)1\>(const void *, const signed char *, const float *, float *, int) |
| 0.3 | 785,952 | 168 | 4,678.3 | 4,656.0 | 4,256 | 7,392 | 344.5 | void llm::cuda::kv_store_k\<(llm::cuda::PosMode)1\>(unsigned short *, unsigned short *, const float *, int, int, int, const int *, int) |
| 0.0 | 74,912 | 6 | 12,485.3 | 12,208.0 | 11,520 | 14,272 | 1,032.4 | void llm::cuda::dequant_rows_k\<(llm::DType)12\>(const void *, const int *, float *, int, long) |
| 0.0 | 61,184 | 18 | 3,399.1 | 3,136.0 | 3,008 | 4,704 | 478.9 | llm::cuda::quantize_q8_1_k(const float *, signed char *, float *, int) |
| 0.0 | 59,648 | 12 | 4,970.7 | 4,800.0 | 4,384 | 6,560 | 554.8 | llm::cuda::rms_norm_k(const float *, const float *, int, float, float *) |

### llama.cpp

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 57.6 | 153,911,267 | 996 | 154,529.4 | 59,551.0 | 16,096 | 331,264 | 137,430.9 | void mul_mat_q\<(ggml_type)12, (int)48, (bool)0\>(const char *, const int *, const int *, const int *...) |
| 11.5 | 30,706,816 | 6 | 5,117,802.7 | 5,109,200.0 | 5,107,776 | 5,137,088 | 14,342.3 | void mul_mat_vec_q\<(ggml_type)14, (int)1, (bool)0, (bool)0\>(const void *, const void *, const int *...) |
| 11.3 | 30,305,694 | 162 | 187,072.2 | 23,728.0 | 20,512 | 375,136 | 172,457.5 | void mul_mat_q\<(ggml_type)14, (int)48, (bool)0\>(const char *, const int *, const int *, const int *...) |
| 3.8 | 10,214,873 | 162 | 63,054.8 | 62,495.0 | 61,823 | 68,320 | 1,337.7 | void unary_gated_op_kernel\<&op_silu, float\>(const T2 *, const T2 *, T2 *, long, long, long, long) |
| 2.7 | 7,221,809 | 996 | 7,250.8 | 5,344.0 | 4,768 | 27,072 | 5,249.9 | void quantize_mmq_q8_1\<(mmq_q8_1_ds_layout)1\>(const float *, const int *, void *, long, long, long,...) |
| 2.6 | 7,072,443 | 342 | 20,679.7 | 20,800.0 | 5,056 | 32,736 | 3,108.6 | void rms_norm_f32\<(int)1024, (bool)1, (bool)0\>(const float *, float *, int, long, long, long, float...) |
| 2.2 | 5,967,324 | 834 | 7,155.1 | 8,416.0 | 2,048 | 13,440 | 3,018.8 | void k_bin_bcast\<&op_add, float, float, float, const float *\>(const T2 *, const T3 *, T4 *, int, in...) |
| 2.0 | 5,323,904 | 168 | 31,689.9 | 31,568.0 | 28,800 | 42,912 | 2,202.7 | void flash_attn_ext_f16\<(int)128, (int)128, (int)8, (int)8, (bool)0, (bool)0\>(const char *, const c...) |
| 1.8 | 4,778,134 | 672 | 7,110.3 | 6,592.0 | 5,472 | 14,400 | 1,996.6 | void mul_mat_q_stream_k_fixup\<(ggml_type)12, (int)48, (bool)0\>(const int *, const int *, float *, f...) |
| 0.9 | 2,427,008 | 168 | 14,446.5 | 14,112.0 | 13,600 | 23,200 | 944.9 | void rope_neox\<(bool)1, (bool)0, float, float\>(const T3 *, T4 *, int, int, int, int, int, int, int,...) |
| 0.9 | 2,416,702 | 162 | 14,917.9 | 8,031.5 | 5,792 | 26,816 | 8,590.1 | void quantize_mmq_q8_1\<(mmq_q8_1_ds_layout)0\>(const float *, const int *, void *, long, long, long,...) |
| 0.9 | 2,307,835 | 6 | 384,639.2 | 383,807.0 | 383,423 | 387,264 | 1,590.1 | void mul_mat_vec_q\<(ggml_type)12, (int)1, (bool)1, (bool)1\>(const void *, const void *, const int *...) |
| 0.6 | 1,498,877 | 162 | 9,252.3 | 8,400.0 | 6,688 | 16,448 | 2,357.8 | void mul_mat_q_stream_k_fixup\<(ggml_type)14, (int)48, (bool)0\>(const int *, const int *, float *, f...) |
| 0.5 | 1,270,619 | 6 | 211,769.8 | 210,975.0 | 210,303 | 215,552 | 1,970.7 | void mul_mat_vec_q\<(ggml_type)14, (int)1, (bool)1, (bool)0\>(const void *, const void *, const int *...) |
| 0.3 | 889,374 | 168 | 5,293.9 | 5,056.0 | 4,864 | 7,808 | 545.4 | void rope_neox\<(bool)1, (bool)0, float, __half\>(const T3 *, T4 *, int, int, int, int, int, int, int,...) |
| 0.2 | 644,446 | 168 | 3,836.0 | 3,584.0 | 3,456 | 6,592 | 577.6 | void k_set_rows\<float, long, __half\>(const T1 *, const T2 *, T3 *, long, long, long, long, long, lo...) |
| 0.0 | 50,459 | 18 | 2,803.3 | 2,000.0 | 1,824 | 6,080 | 1,190.3 | quantize_q8_1(const float *, void *, long, long, long, long, long, unsigned int, uint3) |
| 0.0 | 40,544 | 12 | 3,378.7 | 2,912.0 | 2,784 | 6,400 | 1,212.4 | void k_get_rows_float\<float, float\>(const T1 *, const int *, T2 *, long, long, uint3, unsigned long...) |

---

## 2. Prefill pp512 (6 loops)

### orin_llm

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 63.3 | 1,080,165,344 | 996 | 1,084,503.4 | 402,784.0 | 72,320 | 2,341,472 | 998,061.1 | void llm::cuda::mul_mat_mmq\<(llm::DType)12, (int)128, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 12.1 | 206,948,896 | 162 | 1,277,462.3 | 91,648.0 | 86,464 | 2,571,008 | 1,236,486.7 | void llm::cuda::mul_mat_mmq\<(llm::DType)14, (int)128, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 6.0 | 103,071,040 | 162 | 636,241.0 | 635,600.0 | 633,824 | 661,792 | 3,247.4 | llm::cuda::swiglu_k(float *, const float *, const float *, int) |
| 4.0 | 68,095,552 | 330 | 206,350.2 | 206,304.0 | 203,008 | 214,688 | 1,094.8 | llm::cuda::rms_norm_rows_k(const float *, const float *, int, float, float *) |
| 4.0 | 67,923,200 | 996 | 68,196.0 | 51,136.0 | 47,808 | 258,560 | 56,517.3 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)1\>(const float *, void *, int, int, int) |
| 3.2 | 54,468,416 | 168 | 324,216.8 | 324,208.0 | 321,824 | 328,032 | 1,061.2 | void llm::cuda::flash_attention_prefill_mma_k\<(int)128\>(const float *, const unsigned short *, const unsigned short *, float *, int, int, int, int, int, int, float) |
| 2.1 | 36,155,872 | 330 | 109,563.2 | 109,520.0 | 104,832 | 114,912 | 867.2 | llm::cuda::add_inplace_k(float *, const float *, int) |
| 1.5 | 24,931,552 | 504 | 49,467.4 | 22,368.0 | 21,152 | 105,152 | 38,819.4 | llm::cuda::add_inplace_broadcast_k(float *, const float *, int, int) |
| 1.4 | 23,985,920 | 162 | 148,061.2 | 50,400.0 | 46,688 | 256,832 | 102,270.6 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)0\>(const float *, void *, int, int, int) |
| 1.0 | 17,325,184 | 6 | 2,887,530.7 | 2,887,408.0 | 2,884,288 | 2,891,168 | 2,766.9 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)14, (llm::cuda::GemvWrite)0\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 0.9 | 15,852,608 | 336 | 47,180.4 | 46,688.0 | 17,024 | 80,064 | 29,773.0 | void llm::cuda::rope_neox_k\<(llm::cuda::PosMode)1\>(float *, int, int, float, int, const int *, int, int, float) |
| 0.3 | 5,127,008 | 168 | 30,517.9 | 30,496.0 | 29,792 | 38,176 | 652.3 | void llm::cuda::kv_store_k\<(llm::cuda::PosMode)1\>(unsigned short *, unsigned short *, const float *, int, int, int, const int *, int) |
| 0.1 | 1,214,272 | 6 | 202,378.7 | 202,256.0 | 201,408 | 203,712 | 864.0 | void llm::cuda::gemv_fused_gate_up_swiglu_k\<(llm::DType)12\>(const void *, const void *, const signed char *, const float *, float *, int, int) |
| 0.1 | 1,034,912 | 6 | 172,485.3 | 169,936.0 | 169,664 | 185,344 | 6,301.3 | void llm::cuda::gemv_dp4a_k\<(llm::DType)14, (bool)1\>(const void *, const signed char *, const float *, float *, int) |
| 0.0 | 515,168 | 6 | 85,861.3 | 85,072.0 | 84,544 | 90,144 | 2,138.0 | void llm::cuda::dequant_rows_k\<(llm::DType)12\>(const void *, const int *, float *, int, long) |
| 0.0 | 67,840 | 12 | 5,653.3 | 5,824.0 | 4,352 | 6,592 | 911.3 | llm::cuda::rms_norm_k(const float *, const float *, int, float, float *) |
| 0.0 | 67,136 | 18 | 3,729.8 | 3,632.0 | 2,944 | 4,672 | 573.6 | llm::cuda::quantize_q8_1_k(const float *, signed char *, float *, int) |

### llama.cpp

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 60.4 | 1,068,243,410 | 996 | 1,072,533.5 | 394,895.5 | 66,975 | 2,329,599 | 991,387.8 | void mul_mat_q\<(ggml_type)12, (int)128, (bool)0\>(const char *, const int *, const int *, const int *...) |
| 11.7 | 207,257,885 | 162 | 1,279,369.7 | 84,480.0 | 79,487 | 2,577,792 | 1,247,609.7 | void mul_mat_q\<(ggml_type)14, (int)128, (bool)0\>(const char *, const int *, const int *, const int *...) |
| 6.5 | 114,225,785 | 162 | 705,097.4 | 702,495.0 | 700,671 | 788,288 | 9,222.3 | void unary_gated_op_kernel\<&op_silu, float\>(const T2 *, const T2 *, T2 *, long, long, long, long) |
| 5.1 | 91,037,644 | 342 | 266,191.9 | 275,023.0 | 5,216 | 295,519 | 49,649.5 | void rms_norm_f32\<(int)1024, (bool)1, (bool)0\>(const float *, float *, int, long, long, long, float...) |
| 4.1 | 73,362,356 | 168 | 436,680.7 | 436,943.5 | 427,359 | 448,128 | 3,335.3 | void flash_attn_ext_f16\<(int)128, (int)128, (int)8, (int)8, (bool)0, (bool)0\>(const char *, const c...) |
| 3.7 | 65,628,005 | 996 | 65,891.6 | 47,327.0 | 46,303 | 268,672 | 59,642.9 | void quantize_mmq_q8_1\<(mmq_q8_1_ds_layout)1\>(const float *, const int *, void *, long, long, long,...) |
| 2.9 | 51,655,328 | 834 | 61,936.8 | 82,367.0 | 1,984 | 112,448 | 41,765.5 | void k_bin_bcast\<&op_add, float, float, float, const float *\>(const T2 *, const T3 *, T4 *, int, in...) |
| 1.7 | 30,811,680 | 6 | 5,135,280.0 | 5,133,344.0 | 5,131,808 | 5,142,816 | 4,240.2 | void mul_mat_vec_q\<(ggml_type)14, (int)1, (bool)0, (bool)0\>(const void *, const void *, const int *...) |
| 1.5 | 27,332,341 | 168 | 162,692.5 | 161,983.0 | 161,375 | 194,144 | 2,803.9 | void rope_neox\<(bool)1, (bool)0, float, float\>(const T3 *, T4 *, int, int, int, int, int, int, int,...) |
| 1.4 | 24,255,459 | 162 | 149,725.1 | 50,768.0 | 46,751 | 272,672 | 106,167.4 | void quantize_mmq_q8_1\<(mmq_q8_1_ds_layout)0\>(const float *, const int *, void *, long, long, long,...) |
| 0.5 | 8,126,302 | 168 | 48,370.8 | 48,800.0 | 38,560 | 51,104 | 2,348.9 | void rope_neox\<(bool)1, (bool)0, float, __half\>(const T3 *, T4 *, int, int, int, int, int, int, int,...) |
| 0.2 | 3,482,841 | 168 | 20,731.2 | 20,384.0 | 20,032 | 25,472 | 846.3 | void k_set_rows\<float, long, __half\>(const T1 *, const T2 *, T3 *, long, long, long, long, long, lo...) |
| 0.1 | 2,324,027 | 6 | 387,337.8 | 387,087.0 | 385,375 | 390,623 | 1,986.6 | void mul_mat_vec_q\<(ggml_type)12, (int)1, (bool)1, (bool)1\>(const void *, const void *, const int *...) |
| 0.1 | 1,270,011 | 6 | 211,668.5 | 211,487.0 | 210,943 | 213,312 | 836.7 | void mul_mat_vec_q\<(ggml_type)14, (int)1, (bool)1, (bool)0\>(const void *, const void *, const int *...) |
| 0.0 | 50,299 | 18 | 2,794.4 | 2,032.0 | 1,824 | 6,720 | 1,273.6 | quantize_q8_1(const float *, void *, long, long, long, long, long, unsigned int, uint3) |
| 0.0 | 41,851 | 12 | 3,487.6 | 3,503.0 | 2,784 | 5,184 | 778.2 | void k_get_rows_float\<float, float\>(const T1 *, const int *, T2 *, long, long, uint3, unsigned long...) |

---

## 3. Decode tg128 (5 loops)

### orin_llm

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 36.9 | 3,584,006,047 | 17,948 | 199,688.3 | 199,584.0 | 198,336 | 225,760 | 599.5 | void llm::cuda::gemv_fused_gate_up_swiglu_k\<(llm::DType)12\>(const void *, const void *, const signed char *, const float *, float *, int, int, int) |
| 19.0 | 1,848,036,768 | 641 | 2,883,052.7 | 2,883,456.0 | 2,827,264 | 2,977,408 | 9,882.7 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)14, (llm::cuda::GemvWrite)0\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 15.6 | 1,512,669,184 | 8,974 | 168,561.3 | 168,512.0 | 167,552 | 182,400 | 434.6 | void llm::cuda::gemv_dp4a_k\<(llm::DType)14, (bool)1\>(const void *, const signed char *, const float *, float *, int, int, int, const float *) |
| 10.6 | 1,032,573,853 | 8,974 | 115,062.8 | 115,040.0 | 113,312 | 132,800 | 609.3 | void llm::cuda::gemv_dp4a_k\<(llm::DType)12, (bool)1\>(const void *, const signed char *, const float *, float *, int, int, int, const float *) |
| 6.5 | 633,553,439 | 44,870 | 14,119.8 | 6,624.0 | 6,144 | 28,576 | 9,286.5 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)12, (llm::cuda::GemvWrite)2\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 4.4 | 425,023,519 | 17,948 | 23,680.8 | 23,616.0 | 22,880 | 28,672 | 345.7 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)12, (llm::cuda::GemvWrite)1\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 2.0 | 198,643,775 | 17,948 | 11,067.7 | 11,040.0 | 10,784 | 12,224 | 98.4 | void llm::cuda::flash_attention_decode_mma_k\<(int)128\>(const float *, const unsigned short *, const unsigned short *, float *, int, int, int, int, float) |
| 1.5 | 146,579,455 | 36,537 | 4,011.8 | 3,776.0 | 3,680 | 6,944 | 466.5 | llm::cuda::rms_norm_k(const float *, const float *, int, float, float *) |
| 1.4 | 137,263,390 | 72,433 | 1,895.0 | 1,760.0 | 1,664 | 3,808 | 227.1 | llm::cuda::quantize_q8_1_k(const float *, signed char *, float *, int) |
| 0.8 | 78,858,944 | 8,974 | 8,787.5 | 8,800.0 | 8,128 | 9,536 | 218.3 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)14, (llm::cuda::GemvWrite)2\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 0.6 | 59,241,216 | 35,896 | 1,650.4 | 1,600.0 | 1,504 | 2,560 | 167.9 | void llm::cuda::rope_neox_k\<(llm::cuda::PosMode)0\>(float *, int, int, float, int, const int *, int, int, float) |
| 0.5 | 43,968,800 | 17,948 | 2,449.8 | 2,592.0 | 2,048 | 3,488 | 279.0 | void llm::cuda::kv_store_k\<(llm::cuda::PosMode)0\>(unsigned short *, unsigned short *, const float *, int, int, int, const int *, int) |
| 0.0 | 3,528,256 | 641 | 5,504.3 | 5,632.0 | 4,800 | 8,576 | 471.8 | void llm::cuda::dequant_row_k\<(llm::DType)12\>(const void *, float *, int) |

### llama.cpp

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 52.9 | 8,317,060,605 | 80,125 | 103,801.1 | 35,328.0 | 8,000 | 410,687 | 148,982.0 | void mul_mat_vec_q\<(ggml_type)12, (int)1, (bool)1, (bool)1\>(const void *, const void *, const int *...) |
| 20.8 | 3,273,825,504 | 641 | 5,107,372.1 | 5,107,040.0 | 5,102,656 | 5,128,352 | 2,617.3 | void mul_mat_vec_q\<(ggml_type)14, (int)1, (bool)0, (bool)0\>(const void *, const void *, const int *...) |
| 12.8 | 2,004,182,560 | 17,948 | 111,666.1 | 112,255.5 | 12,736 | 213,184 | 98,458.5 | void mul_mat_vec_q\<(ggml_type)14, (int)1, (bool)1, (bool)0\>(const void *, const void *, const int *...) |
| 8.2 | 1,281,261,408 | 8,974 | 142,774.8 | 142,751.0 | 141,247 | 147,807 | 516.0 | void mul_mat_vec_q\<(ggml_type)12, (int)1, (bool)1, (bool)0\>(const void *, const void *, const int *...) |
| 1.5 | 231,645,530 | 108,329 | 2,138.4 | 1,888.0 | 1,792 | 7,008 | 494.4 | quantize_q8_1(const float *, void *, long, long, long, long, long, unsigned int, uint3) |
| 1.4 | 213,479,999 | 36,537 | 5,842.8 | 5,983.0 | 4,992 | 10,528 | 690.2 | void rms_norm_f32\<(int)1024, (bool)1, (bool)0\>(const float *, float *, int, long, long, long, float...) |
| 1.2 | 193,999,647 | 17,948 | 10,809.0 | 10,624.0 | 9,728 | 15,488 | 517.4 | void flash_attn_ext_f16\<(int)128, (int)128, (int)1, (int)8, (bool)0, (bool)0\>(const char *, const c...) |
| 0.3 | 51,225,374 | 17,948 | 2,854.1 | 2,944.0 | 2,400 | 5,280 | 207.5 | void rope_neox\<(bool)1, (bool)0, float, __half\>(const T3 *, T4 *, int, int, int, int, int, int, int,...) |
| 0.3 | 49,057,600 | 17,948 | 2,733.3 | 2,688.0 | 2,624 | 5,696 | 137.6 | void k_set_rows\<float, long, __half\>(const T1 *, const T2 *, T3 *, long, long, long, long, long, lo...) |
| 0.3 | 44,048,314 | 17,948 | 2,454.2 | 2,400.0 | 2,304 | 5,472 | 207.2 | void rope_neox\<(bool)1, (bool)0, float, float\>(const T3 *, T4 *, int, int, int, int, int, int, int,...) |
| 0.2 | 38,033,853 | 17,948 | 2,119.1 | 2,112.0 | 1,952 | 6,592 | 82.8 | void flash_attn_stream_k_fixup_uniform\<(int)128, (int)1, (int)8\>(float *, const float2 *, int, int,...) |
| 0.1 | 15,740,896 | 641 | 24,556.8 | 24,544.0 | 23,968 | 27,456 | 224.1 | void mul_mat_vec_q\<(ggml_type)12, (int)1, (bool)0, (bool)1\>(const void *, const void *, const int *...) |
| 0.0 | 3,671,040 | 1,282 | 2,863.5 | 2,848.0 | 2,784 | 5,120 | 89.4 | void k_get_rows_float\<float, float\>(const T1 *, const int *, T2 *, long, long, uint3, unsigned long...) |
| 0.0 | 1,299,296 | 641 | 2,027.0 | 2,016.0 | 1,952 | 5,472 | 139.3 | void k_bin_bcast\<&op_add, float, float, float, const float *\>(const T2 *, const T3 *, T4 *, int, in...) |
