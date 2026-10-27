# nsys profile report

hardware: NVIDIA Jetson Orin Nano Super 8GB (SM 8.7, 8 SM)

model: Qwen2.5-1.5B-Instruct Q4\_K\_M

## 1. Prefill pp45 (6 loops)

### orin_llm

| Time (%) | Total Time (ns) | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | StdDev (ns) | Name |
|----------|-----------------|-----------|----------|----------|----------|----------|-------------|------|
| 62.2 | 154,349,472 | 1,008 | 153,124.5 | 58,880.0 | 16,192 | 319,872 | 134,239.5 | void llm::cuda::mul_mat_mmq\<(llm::DType)12, (int)48, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 13.7 | 33,942,624 | 168 | 202,039.4 | 201,136.0 | 19,776 | 386,976 | 181,757.9 | void llm::cuda::mul_mat_mmq\<(llm::DType)14, (int)48, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 7.0 | 17,397,632 | 6 | 2,899,605.3 | 2,900,736.0 | 2,887,552 | 2,907,008 | 6,906.1 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)14, (llm::cuda::GemvWrite)0\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 3.9 | 9,658,240 | 168 | 57,489.5 | 57,440.0 | 56,672 | 59,872 | 511.7 | llm::cuda::swiglu_k(float *, const float *, const float *, int) |
| 2.8 | 6,930,784 | 1,008 | 6,875.8 | 5,536.0 | 4,960 | 22,624 | 4,585.3 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)1\>(const float *, void *, int, int, int) |
| 2.4 | 6,038,240 | 840 | 7,188.4 | 7,104.0 | 5,856 | 14,720 | 678.1 | void llm::cuda::mul_mat_mmq_fixup\<(int)48, (int)128, (bool)0\>(float *, float *, int, int, uint3, uint3, int) |
| 2.2 | 5,451,360 | 336 | 16,224.3 | 16,096.0 | 15,360 | 24,608 | 662.1 | llm::cuda::rms_norm_rows_k(const float *, const float *, int, float, float *) |
| 1.4 | 3,419,712 | 336 | 10,177.7 | 11,168.0 | 8,064 | 12,352 | 1,450.9 | llm::cuda::add_inplace_k(float *, const float *, int) |
| 1.2 | 2,998,368 | 504 | 5,949.1 | 4,384.0 | 3,936 | 12,896 | 2,353.3 | llm::cuda::add_inplace_broadcast_k(float *, const float *, int, int) |
| 1.1 | 2,852,672 | 168 | 16,980.2 | 16,960.0 | 16,256 | 26,272 | 766.0 | void llm::cuda::flash_attention_prefill_mma_k\<(int)128\>(const float *, const unsigned short *, const unsigned short *, float *, int, int, int, int, int, int, float) |
| 0.9 | 2,305,280 | 168 | 13,721.9 | 14,640.0 | 4,960 | 22,720 | 8,300.4 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)0\>(const float *, void *, int, int, int) |
| 0.7 | 1,804,480 | 336 | 5,370.5 | 5,680.0 | 3,136 | 9,856 | 1,788.8 | void llm::cuda::rope_neox_k\<(llm::cuda::PosMode)1\>(float *, int, int, float, int, const int *, int, int, float) |
| 0.3 | 788,384 | 168 | 4,692.8 | 4,688.0 | 4,224 | 7,328 | 331.6 | void llm::cuda::kv_store_k\<(llm::cuda::PosMode)1\>(unsigned short *, unsigned short *, const float *, int, int, int, const int *, int) |
| 0.0 | 73,440 | 6 | 12,240.0 | 11,920.0 | 11,328 | 14,400 | 1,176.8 | void llm::cuda::dequant_rows_k\<(llm::DType)12\>(const void *, const int *, float *, int, long) |
| 0.0 | 31,168 | 6 | 5,194.7 | 4,944.0 | 4,864 | 6,464 | 625.3 | llm::cuda::rms_norm_k(const float *, const float *, int, float, float *) |
| 0.0 | 19,744 | 6 | 3,290.7 | 3,008.0 | 2,944 | 4,736 | 709.3 | llm::cuda::quantize_q8_1_k(const float *, signed char *, float *, int) |

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
| 63.0 | 1,106,813,344 | 1,008 | 1,098,029.1 | 403,696.0 | 72,896 | 2,354,720 | 999,012.0 | void llm::cuda::mul_mat_mmq\<(llm::DType)12, (int)128, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 12.7 | 222,471,296 | 168 | 1,324,233.9 | 1,319,264.0 | 87,264 | 2,578,944 | 1,238,265.2 | void llm::cuda::mul_mat_mmq\<(llm::DType)14, (int)128, (bool)0\>(const void *, const int *, float *, float *, int, int, int, int, uint3, uint3, int) |
| 6.1 | 107,200,320 | 168 | 638,097.1 | 637,360.0 | 635,168 | 670,752 | 3,636.9 | llm::cuda::swiglu_k(float *, const float *, const float *, int) |
| 3.9 | 67,904,128 | 336 | 202,095.6 | 201,968.0 | 200,288 | 205,792 | 963.4 | llm::cuda::rms_norm_rows_k(const float *, const float *, int, float, float *) |
| 3.8 | 67,548,192 | 1,008 | 67,012.1 | 50,496.0 | 46,560 | 260,672 | 56,770.7 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)1\>(const float *, void *, int, int, int) |
| 3.3 | 57,386,560 | 168 | 341,586.7 | 341,360.0 | 339,040 | 346,048 | 1,057.8 | void llm::cuda::flash_attention_prefill_mma_k\<(int)128\>(const float *, const unsigned short *, const unsigned short *, float *, int, int, int, int, int, int, float) |
| 2.1 | 37,310,336 | 336 | 111,042.7 | 110,976.0 | 107,264 | 115,776 | 753.9 | llm::cuda::add_inplace_k(float *, const float *, int) |
| 1.4 | 25,363,616 | 168 | 150,973.9 | 149,808.0 | 45,664 | 269,760 | 104,345.1 | void llm::cuda::quantize_mmq_q8_1_k\<(bool)0\>(const float *, void *, int, int, int) |
| 1.4 | 24,940,768 | 504 | 49,485.7 | 22,336.0 | 21,024 | 105,120 | 38,900.1 | llm::cuda::add_inplace_broadcast_k(float *, const float *, int, int) |
| 1.0 | 17,364,448 | 6 | 2,894,074.7 | 2,890,000.0 | 2,879,616 | 2,928,544 | 17,434.3 | void llm::cuda::gemv_dp4a_smallk_k\<(llm::DType)14, (llm::cuda::GemvWrite)0\>(const void *, const signed char *, const float *, float *, int, int, int) |
| 0.9 | 15,977,472 | 336 | 47,552.0 | 47,040.0 | 17,024 | 78,656 | 30,091.3 | void llm::cuda::rope_neox_k\<(llm::cuda::PosMode)1\>(float *, int, int, float, int, const int *, int, int, float) |
| 0.3 | 5,162,592 | 168 | 30,729.7 | 30,688.0 | 30,016 | 38,048 | 638.5 | void llm::cuda::kv_store_k\<(llm::cuda::PosMode)1\>(unsigned short *, unsigned short *, const float *, int, int, int, const int *, int) |
| 0.0 | 517,408 | 6 | 86,234.7 | 85,472.0 | 84,704 | 90,624 | 2,181.4 | void llm::cuda::dequant_rows_k\<(llm::DType)12\>(const void *, const int *, float *, int, long) |
| 0.0 | 37,664 | 6 | 6,277.3 | 6,224.0 | 6,080 | 6,528 | 193.5 | llm::cuda::rms_norm_k(const float *, const float *, int, float, float *) |
| 0.0 | 25,952 | 6 | 4,325.3 | 4,240.0 | 4,192 | 4,672 | 182.6 | llm::cuda::quantize_q8_1_k(const float *, signed char *, float *, int) |

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
