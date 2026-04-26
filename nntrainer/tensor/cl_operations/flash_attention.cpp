// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file	flash_attention.cpp
 * @date	25 March 2026
 * @brief	Common flash attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Anup Tiwari <anup.tiwari@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "flash_attention.h"
#include <algorithm>
#include <cl_kernels/flash_attention_fp32.h>
#include <cl_kernels/flash_attention_prefill_fp32.h>
#include <cl_kernels/flash_attention_prefill_fp32_adreno.h>
#include <cl_kernels/flash_attention_decode_fp32.h>
#ifdef ENABLE_FP16
#include <cl_kernels/flash_attention_fp16.h>
#include <cl_kernels/flash_attention_prefill_fp16.h>
#include <cl_kernels/flash_attention_prefill_fp16_adreno.h>
#include <cl_kernels/flash_attention_decode_fp16.h>
#endif

namespace nntrainer {


void flash_attention_cpu(const float *query, const float *key, 
                                   const float *value, float *output,
                                   int seqlen_q, int seqlen_k, int head_dim, 
                                   int num_heads_q, int num_heads_kv, int batch,
                                   float scale) {
  for (int b = 0; b < batch; b++) {
    for (int q_head = 0; q_head < num_heads_q; q_head++) {
      // Map query head to corresponding key/value head
      int kv_head = q_head * num_heads_kv / num_heads_q;
      
      // Calculate offsets
      int query_batch_offset = b * num_heads_q * seqlen_q * head_dim;
      int query_head_offset = query_batch_offset + q_head * seqlen_q * head_dim;
      
      int kv_batch_offset = b * num_heads_kv * seqlen_k * head_dim;
      int kv_head_offset = kv_batch_offset + kv_head * seqlen_k * head_dim;
      
      for (int q = 0; q < seqlen_q; q++) {
        int q_offset = query_head_offset + q * head_dim;
        
        // Compute attention scores
        std::vector<float> scores(seqlen_k, 0.0f);
        float max_score = -1e9f;
        
        for (int k = 0; k < seqlen_k; k++) {
          float sum = 0.0f;
          int k_offset = kv_head_offset + k * head_dim;
          
          for (int d = 0; d < head_dim; d++) {
            float q_val = query[q_offset + d];
            float k_val = key[k_offset + d];
            sum += q_val * k_val * scale;
          }
          scores[k] = sum;
          max_score = std::max(max_score, sum);
        }
        
        // Compute softmax
        std::vector<float> attn_weights(seqlen_k, 0.0f);
        float exp_sum = 0.0f;
        for (int k = 0; k < seqlen_k; k++) {
          float exp_val = std::exp(scores[k] - max_score);
          attn_weights[k] = exp_val;
          exp_sum += exp_val;
        }
        
        // Normalize
        for (int k = 0; k < seqlen_k; k++) {
          attn_weights[k] /= exp_sum;
        }
        
        // Compute output
        for (int d = 0; d < head_dim; d++) {
          float sum = 0.0f;
          for (int k = 0; k < seqlen_k; k++) {
            int v_offset = kv_head_offset + k * head_dim;
            sum += attn_weights[k] * value[v_offset + d];
          }
          output[q_offset + d] = sum;
        }
      }
    }
  }
}

/**
 * @brief Threshold for distinguishing decode (seqlen_q == 1) from prefill
 * @detail When seqlen_q <= DECODE_SEQLEN_THRESHOLD, the decode-optimized kernel
 *         is used. When seqlen_q > DECODE_SEQLEN_THRESHOLD, the prefill-optimized
 *         kernel is used. Currently set to 1 (single token = decode).
 */
static const unsigned int DECODE_SEQLEN_THRESHOLD = 1;

/**
 * @brief Maximum NCOLS2 (Q heads per KV head) supported by the kernel
 * @detail The FP16 kernel is compiled with NCOLS2=2, so we can group up to 2 Q heads
 *         per work-group. The actual ncols2 used at runtime may be less.
 *         For GQA-4+ models, multiple dispatches handle the remaining Q heads.
 */
static const unsigned int MAX_NCOLS2 = 2;

/**
 * @brief Maximum NCOLS1 (Q tokens per work-group) supported by the kernel
 * @detail The kernel is compiled with NCOLS1=4, which determines the local memory
 *         allocation. The actual ncols1 used at runtime may be less (adaptive sizing).
 */
static const unsigned int MAX_NCOLS1 = 4;

/**
 * @brief Maximum NCOLS1 for Adreno GPUs (64+ KB local memory allows larger tiles)
 * @detail Adreno has 64+ KB local memory vs Mali's 32 KB, allowing NCOLS1=8
 *         which doubles Q rows per work-group for better K/V tile reuse.
 */
static const unsigned int MAX_NCOLS1_ADRENO = 8;

/**
 * @brief Select adaptive ncols1 (Q tokens per work-group) based on seqlen_q and ncols2
 * @detail Phase 5 optimization: larger ncols1 for longer sequences reduces kernel launch
 *         overhead and improves work-item utilization. Smaller ncols1 for short sequences
 *         avoids wasted local memory and compute on padding rows.
 *         The selection also accounts for ncols2 (GQA grouping) to keep total Q rows
 *         per work-group (ncols1 * ncols2) reasonable for local memory constraints.
 * @param seqlen_q Query sequence length
 * @param ncols2 Number of Q heads per KV head in this dispatch
 * @return Selected ncols1 value (must be <= MAX_NCOLS1)
 */
static unsigned int select_ncols1(unsigned int seqlen_q, unsigned int ncols2) {
  // Adaptive sizing: larger ncols1 for longer sequences
  // Account for ncols2 to keep total Q rows (ncols1 * ncols2) manageable
  // Local memory usage scales with ncols1 * ncols2, so reduce ncols1 when ncols2 > 1
  unsigned int ncols1;
  if (seqlen_q > 16 / ncols2) {
    ncols1 = 32;
  } else if (seqlen_q > 8 / ncols2) {
    ncols1 = 16;
  } else if (seqlen_q > 4 / ncols2) {
    ncols1 = 8;
  } else if (seqlen_q > 2 / ncols2) {
    ncols1 = 4;
  } else {
    ncols1 = 2;
  }
  // Cap at MAX_NCOLS1 (kernel compile-time limit)
  if (ncols1 > MAX_NCOLS1) {
    ncols1 = MAX_NCOLS1;
  }
  return ncols1;
}

/**
 * @brief Select adaptive ncols1 for Adreno GPUs (larger local memory allows NCOLS1=8)
 * @detail Same logic as select_ncols1() but with MAX_NCOLS1_ADRENO cap (8 vs 4).
 *         Adreno's 64+ KB local memory can accommodate NCOLS1=8 with ~27 KB usage,
 *         allowing 2 work-groups to fit simultaneously for good occupancy.
 * @param seqlen_q Query sequence length
 * @param ncols2 Number of Q heads per KV head in this dispatch
 * @return Selected ncols1 value (must be <= MAX_NCOLS1_ADRENO)
 */
static unsigned int select_ncols1_adreno(unsigned int seqlen_q, unsigned int ncols2) {
  unsigned int ncols1;
  if (seqlen_q > 32 / ncols2) {
    ncols1 = 16;
  } else if (seqlen_q > 16 / ncols2) {
    ncols1 = 8;
  } else if (seqlen_q > 8 / ncols2) {
    ncols1 = 4;
  } else {
    ncols1 = 2;
  }
  if (ncols1 > MAX_NCOLS1_ADRENO) {
    ncols1 = MAX_NCOLS1_ADRENO;
  }
  return ncols1;
}

// Forward declarations for Adreno helper functions (defined after flash_attention_prefill_fp32_cl)
static bool is_adreno_device();
static bool supports_subgroups();
static std::string get_subgroup_compile_option();

void flash_attention_fp32_cl(float *query, float *key, float *value, float *output,
                             unsigned int seqlen_q, unsigned int seqlen_k,
                             unsigned int head_dim, unsigned int num_heads_q,
                             unsigned int num_heads_kv, unsigned int batch,
                             float scale) {
  // Dispatch to prefill or decode kernel based on query sequence length
  if (seqlen_q <= DECODE_SEQLEN_THRESHOLD) {
    flash_attention_decode_fp32_cl(query, key, value, output,
                                  seqlen_q, seqlen_k, head_dim,
                                  num_heads_q, num_heads_kv, batch, scale);
  } else {
    // On Adreno GPUs, use the Adreno-optimized prefill kernel
    if (is_adreno_device()) {
      flash_attention_prefill_fp32_adreno_cl(query, key, value, output,
                                             seqlen_q, seqlen_k, head_dim,
                                             num_heads_q, num_heads_kv, batch, scale);
    } else {
      flash_attention_prefill_fp32_cl(query, key, value, output,
                                     seqlen_q, seqlen_k, head_dim,
                                     num_heads_q, num_heads_kv, batch, scale);
    }
  }
}

void flash_attention_prefill_fp32_cl(float *query, float *key, float *value, float *output,
                                     unsigned int seqlen_q, unsigned int seqlen_k,
                                     unsigned int head_dim, unsigned int num_heads_q,
                                     unsigned int num_heads_kv, unsigned int batch,
                                     float scale) {
  // For very small workloads, use CPU implementation to avoid GPU overhead
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  // Threshold for switching to CPU - tune based on empirical testing
  if (total_work_items < 32 || total_elements < 4096) {
    flash_attention_cpu(query, key, value, output, 
                                      seqlen_q, seqlen_k, head_dim, 
                                      num_heads_q, num_heads_kv, batch, scale);
    return;
  }

  // FP32 prefill: Use the original row-by-row kernel which is well-optimized
  // for this device (40 ms vs 152 ms with tiled GEMM on Mali).
  // The tiled GEMM kernel has low work-item utilization and high local memory
  // usage for FP32, making the row-by-row approach faster.
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_fp32_kernel, "flash_attention_fp32");
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_prefill_fp32");
    return;
  }

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_prefill_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_prefill_fp32");

  // Dispatch: original row-by-row kernel — one work-item per (batch, head, q_token)
  const int work_groups_count[3] = {(int)total_work_items, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch kernel for flash_attention_prefill_fp32");
    return;
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output, 
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(float),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_prefill_fp32");
    return;
  }
}

/**
 * @brief Check if the current GPU device is an Adreno (Qualcomm) GPU
 * @detail Queries the OpenCL device vendor to determine if the GPU is Adreno.
 *         Adreno GPUs support features not available on Mali (e.g., parallel
 *         softmax with barriers, sub-groups, larger local memory).
 * @return true if the device is an Adreno GPU, false otherwise
 */
static bool is_adreno_device() {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  const auto *device_info = blas_cc->context_inst_.getDeviceInfo();
  if (!device_info) {
    return false;
  }

  const std::string &vendor = device_info->getDeviceVendor();
  std::string vendor_lower = vendor;
  std::transform(vendor_lower.begin(), vendor_lower.end(), vendor_lower.begin(), ::tolower);
  return (vendor_lower.find("qualcomm") != std::string::npos ||
          vendor_lower.find("adreno") != std::string::npos);
}

/**
 * @brief Check if the device supports sub-group operations
 */
static bool supports_subgroups() {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  const auto *device_info = blas_cc->context_inst_.getDeviceInfo();
  if (!device_info) {
    return false;
  }

  const std::string &extensions = device_info->getDeviceExtensions();
  return extensions.find("cl_qcom_subgroups") != std::string::npos ||
         extensions.find("cl_khr_subgroups") != std::string::npos;
}

/**
 * @brief Select optimal work-group size based on device properties
 * @detail Opt 11: Runtime work-group size optimization. Adreno GPUs with 64+ KB
 *         local memory can run 128 work-items per group. For devices with less
 *         local memory or where occupancy is limited, 64 may be better.
 *         The work-group size must be a multiple of the sub-group size (32 or 64).
 * @param local_mem_used Estimated local memory usage per work-group (bytes)
 * @return Selected work-group size (64 or 128)
 */
static unsigned int select_work_group_size(unsigned int local_mem_used) {
  // Opt 11: Runtime work-group size optimization based on device type
  // Adreno GPUs have 64+ KB local memory — 128 work-items per group is optimal
  // For non-Adreno (Mali with 32 KB), use 64 if local memory is tight
  if (is_adreno_device()) {
    // Adreno has 64+ KB local memory — 128 is always optimal
    return 128;
  }
  
  // For Mali and other GPUs with 32 KB local memory:
  // If local memory usage exceeds 16 KB, only 1 work-group fits → use 64
  // to allow 2 work-groups and improve occupancy
  if (local_mem_used > 16 * 1024) {
    ml_logi("Opt 11: Reducing work-group size to 64 (non-Adreno, used=%u KB > 16 KB)",
            local_mem_used / 1024);
    return 64;
  }
  
  return 128;  // Default — sufficient local memory for 2+ work-groups
}

/**
 * @brief Get the compile option to enable sub-group support
 */
static std::string get_subgroup_compile_option() {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  const auto *device_info = blas_cc->context_inst_.getDeviceInfo();
  if (!device_info) {
    return "";
  }

  const std::string &extensions = device_info->getDeviceExtensions();
  if (extensions.find("cl_qcom_subgroups") != std::string::npos) {
    return "-DCL_QCOM_SUBGROUPS";
  }
  if (extensions.find("cl_khr_subgroups") != std::string::npos) {
    return "-DCL_KHR_SUBGROUPS";
  }
  return "";
}

void flash_attention_prefill_fp32_adreno_cl(float *query, float *key, float *value,
                                            float *output, unsigned int seqlen_q,
                                            unsigned int seqlen_k, unsigned int head_dim,
                                            unsigned int num_heads_q,
                                            unsigned int num_heads_kv,
                                            unsigned int batch, float scale) {
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;

  if (total_work_items < 32 || total_elements < 4096) {
    flash_attention_cpu(query, key, value, output,
                        seqlen_q, seqlen_k, head_dim,
                        num_heads_q, num_heads_kv, batch, scale);
    return;
  }

  const unsigned int gqa_ratio = num_heads_q / num_heads_kv;
  unsigned int ncols2 = 1;
  if (gqa_ratio >= 4 && seqlen_q > 4) {
    ncols2 = 2;
  }
  if (ncols2 > MAX_NCOLS2) {
    ncols2 = MAX_NCOLS2;
  }

  // Opt 9: Use Adreno-specific ncols1 selection (MAX_NCOLS1_ADRENO=8 vs MAX_NCOLS1=4)
  const unsigned int ncols1 = select_ncols1_adreno(seqlen_q, ncols2);
  static const unsigned int PREFILL_WORK_GROUP_SIZE = 128;
  const unsigned int num_q_groups = (seqlen_q + ncols1 - 1) / ncols1;
  const unsigned int num_dispatches = (gqa_ratio + ncols2 - 1) / ncols2;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  std::string adreno_compile_options = get_subgroup_compile_option();
  // Opt 9: Compile with NCOLS1=8 for Adreno (64+ KB local memory allows larger tiles)
  adreno_compile_options += " -DNCOLS1=8";
  if (supports_subgroups()) {
    ml_logi("Adreno FP32 prefill: sub-group extension detected (%s), "
            "enabling hardware sub-group reductions",
            adreno_compile_options.c_str() + 2);
  } else {
    ml_logi("Adreno FP32 prefill: no sub-group extension detected, "
            "using manual barrier-based reductions");
  }

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_prefill_fp32_adreno_kernel, "flash_attention_prefill_fp32_adreno",
    adreno_compile_options);
  if (!kernel_ptr) {
    // Fallback to base (Mali) kernel if Adreno kernel fails to compile
    // This can happen on some Adreno devices where the compiler runs out of
    // host memory (CL_OUT_OF_HOST_MEMORY) compiling the FP32 tiled GEMM kernel
    ml_logi("Adreno FP32 prefill: kernel compilation failed, "
            "falling back to base row-by-row kernel");
    flash_attention_prefill_fp32_cl(query, key, value, output,
                                   seqlen_q, seqlen_k, head_dim,
                                   num_heads_q, num_heads_kv, batch, scale);
    return;
  }

  for (unsigned int dispatch = 0; dispatch < num_dispatches; dispatch++) {
    const unsigned int head_group_offset = dispatch * ncols2;
    const unsigned int dispatch_ncols2 = std::min(ncols2, gqa_ratio - head_group_offset);
    const unsigned int total_groups = batch * num_heads_kv * num_q_groups;

    int arg = 0;
    bool result = false;

    result = kernel_ptr->SetKernelSVMArguments(arg++, query);
    result = kernel_ptr->SetKernelSVMArguments(arg++, key);
    result = kernel_ptr->SetKernelSVMArguments(arg++, value);
    result = kernel_ptr->SetKernelSVMArguments(arg++, output);
    result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
    result = kernel_ptr->SetKernelArguments(arg++, &dispatch_ncols2, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &head_group_offset, sizeof(int));
    result = kernel_ptr->SetKernelArguments(arg++, &ncols1, sizeof(int));

    const int work_groups_count[3] = {(int)(total_groups * PREFILL_WORK_GROUP_SIZE), 1, 1};
    const int work_group_size[3] = {(int)PREFILL_WORK_GROUP_SIZE, 1, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error("Failed to dispatch kernel for flash_attention_prefill_fp32_adreno");
      return;
    }
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output,
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(float),
                                             true);
}

void flash_attention_decode_fp32_cl(float *query, float *key, float *value,
                                    float *output, unsigned int seqlen_q,
                                    unsigned int seqlen_k, unsigned int head_dim,
                                    unsigned int num_heads_q,
                                    unsigned int num_heads_kv,
                                    unsigned int batch, float scale) {
  // Phase 6: Decode-specific kernel with split-KV approach
  // For very small workloads (typical in decode), CPU fallback may be faster.
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  // For decode with small seqlen_k, use CPU or simple kernel
  // The split-KV approach is beneficial when seqlen_k is large enough to warrant parallelization
  static const unsigned int KV_TILE_SIZE = 64;
  const unsigned int num_kv_tiles = (seqlen_k + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
  
  // For small KV sequences or small batches, use CPU fallback
  if (total_work_items < 32 || total_elements < 4096 || num_kv_tiles < 2) {
    flash_attention_cpu(query, key, value, output,
                        seqlen_q, seqlen_k, head_dim,
                        num_heads_q, num_heads_kv, batch, scale);
    return;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  // Allocate partial buffers for split-KV reduction
  // partials_max: [batch, num_heads_q, num_kv_tiles]
  // partials_sum: [batch, num_heads_q, num_kv_tiles]
  // partials_vkq: [batch, num_heads_q, num_kv_tiles, head_dim]
  const size_t partials_max_sum_size = batch * num_heads_q * num_kv_tiles * sizeof(float);
  const size_t partials_vkq_size = batch * num_heads_q * num_kv_tiles * head_dim * sizeof(float);
  
  std::vector<float> partials_max(partials_max_sum_size / sizeof(float));
  std::vector<float> partials_sum(partials_max_sum_size / sizeof(float));
  std::vector<float> partials_vkq(partials_vkq_size / sizeof(float));
  
  // Allocate SVM buffers for partials
  float *partials_max_buf = (float *)blas_cc->context_inst_.createSVMRegion(partials_max_sum_size);
  float *partials_sum_buf = (float *)blas_cc->context_inst_.createSVMRegion(partials_max_sum_size);
  float *partials_vkq_buf = (float *)blas_cc->context_inst_.createSVMRegion(partials_vkq_size);
  
  if (!partials_max_buf || !partials_sum_buf || !partials_vkq_buf) {
    // Fallback to CPU if allocation fails
    flash_attention_cpu(query, key, value, output,
                        seqlen_q, seqlen_k, head_dim,
                        num_heads_q, num_heads_kv, batch, scale);
    return;
  }

  // Kernel 1: Decode kernel - each work-group processes a KV tile
  ClContext::SharedPtrClKernel decode_kernel_ptr = blas_cc->registerClKernel(
    flash_attention_decode_fp32_kernel, "flash_attention_decode_fp32");
  if (!decode_kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_decode_fp32");
    return;
  }

  // kv_max = seqlen_k (all KV positions are valid)
  const int kv_max = seqlen_k;
  
  // Work-group size for decode kernel
  static const unsigned int DECODE_WORK_GROUP_SIZE = 64;
  const unsigned int total_decode_groups = batch * num_heads_q * num_kv_tiles;
  
  int arg = 0;
  bool result = false;

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, partials_max_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, partials_sum_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, partials_vkq_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 11 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 12 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 13 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &kv_max, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 14 for flash_attention_decode_fp32");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &num_kv_tiles, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 15 for flash_attention_decode_fp32");

  const int decode_work_groups_count[3] = {(int)(total_decode_groups * DECODE_WORK_GROUP_SIZE), 1, 1};
  const int decode_work_group_size[3] = {(int)DECODE_WORK_GROUP_SIZE, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    decode_kernel_ptr, decode_work_groups_count, decode_work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch decode kernel for flash_attention_decode_fp32");
    return;
  }

  // Kernel 2: Reduction kernel - combine partials using log-sum-exp
  // Note: Both decode and reduce kernels are in the same .cl file, so we use the same kernel string
  ClContext::SharedPtrClKernel reduce_kernel_ptr = blas_cc->registerClKernel(
    flash_attention_decode_fp32_kernel, "flash_attention_decode_reduce_fp32");
  if (!reduce_kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_decode_reduce_fp32");
    return;
  }

  arg = 0;
  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, partials_max_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_decode_reduce_fp32");

  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, partials_sum_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_decode_reduce_fp32");

  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, partials_vkq_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_decode_reduce_fp32");

  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_decode_reduce_fp32");

  result = reduce_kernel_ptr->SetKernelArguments(arg++, &num_kv_tiles, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_decode_reduce_fp32");

  result = reduce_kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_decode_reduce_fp32");

  result = reduce_kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_decode_reduce_fp32");

  // One work-group per (batch, head) pair
  const int reduce_work_groups_count[3] = {(int)(batch * num_heads_q * DECODE_WORK_GROUP_SIZE), 1, 1};
  const int reduce_work_group_size[3] = {(int)DECODE_WORK_GROUP_SIZE, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    reduce_kernel_ptr, reduce_work_groups_count, reduce_work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch reduce kernel for flash_attention_decode_fp32");
    return;
  }

  // Read back output
  blas_cc->command_queue_inst_.enqueueSVMMap(output,
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(float),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_decode_fp32");
    return;
  }
  
  // Free partial buffers
  // Note: SVM buffers are freed automatically when the context is destroyed
}

#ifdef ENABLE_FP16

/**
 * @brief Helper to convert FP16 buffers to FP32 for CPU fallback
 */
static void fp16_to_fp32(const _FP16 *src, float *dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
}

/**
 * @brief Helper to convert FP32 buffers to FP16 for writing back
 */
static void fp32_to_fp16(const float *src, _FP16 *dst, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = static_cast<_FP16>(src[i]);
  }
}

void flash_attention_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value, _FP16 *output,
                             unsigned int seqlen_q, unsigned int seqlen_k,
                             unsigned int head_dim, unsigned int num_heads_q,
                             unsigned int num_heads_kv, unsigned int batch,
                             float scale) {
  // Dispatch to prefill or decode kernel based on query sequence length
  if (seqlen_q <= DECODE_SEQLEN_THRESHOLD) {
    flash_attention_decode_fp16_cl(query, key, value, output,
                                  seqlen_q, seqlen_k, head_dim,
                                  num_heads_q, num_heads_kv, batch, scale);
  } else {
    // On Adreno GPUs, use the Adreno-optimized prefill kernel with parallel softmax
    if (is_adreno_device()) {
      flash_attention_prefill_fp16_adreno_cl(query, key, value, output,
                                             seqlen_q, seqlen_k, head_dim,
                                             num_heads_q, num_heads_kv, batch, scale);
    } else {
      flash_attention_prefill_fp16_cl(query, key, value, output,
                                     seqlen_q, seqlen_k, head_dim,
                                     num_heads_q, num_heads_kv, batch, scale);
    }
  }
}

void flash_attention_prefill_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value,
                                     _FP16 *output, unsigned int seqlen_q,
                                     unsigned int seqlen_k, unsigned int head_dim,
                                     unsigned int num_heads_q,
                                     unsigned int num_heads_kv,
                                     unsigned int batch, float scale) {
  // For very small workloads, use CPU implementation to avoid GPU overhead
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  // Threshold for switching to CPU - tune based on empirical testing
  if (total_work_items < 32 || total_elements < 4096) {
    // Convert FP16 to FP32 for CPU computation
    const size_t kv_elements = batch * num_heads_kv * seqlen_k * head_dim;
    std::vector<float> query_fp32(total_elements);
    std::vector<float> key_fp32(kv_elements);
    std::vector<float> value_fp32(kv_elements);
    std::vector<float> output_fp32(total_elements);
    
    fp16_to_fp32(query, query_fp32.data(), total_elements);
    fp16_to_fp32(key, key_fp32.data(), kv_elements);
    fp16_to_fp32(value, value_fp32.data(), kv_elements);
    
    // Compute on CPU
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                       output_fp32.data(), seqlen_q, seqlen_k, head_dim,
                       num_heads_q, num_heads_kv, batch, scale);
    
    // Convert output back to FP16
    fp32_to_fp16(output_fp32.data(), output, total_elements);
    return;
  }

  // Phase 4: GQA grouping — compute adaptive ncols2
  // ncols2 = number of Q heads per KV head processed together in one work-group
  // This reuses K/V tiles across multiple Q heads, saving global memory bandwidth
  const unsigned int gqa_ratio = num_heads_q / num_heads_kv;

  // Adaptive ncols2 selection based on GQA ratio and sequence length
  // Only use GQA grouping for ratio >= 4 — ratio 2 doesn't benefit enough
  // from K/V reuse to offset the reduced parallelism and increased local memory
  unsigned int ncols2 = 1;
  if (gqa_ratio >= 4 && seqlen_q > 4) {
    ncols2 = 2;  // Group 2 Q heads per work-group (with multi-dispatch for ratio > 2)
  }
  // Cap at MAX_NCOLS2 (kernel compile-time limit)
  if (ncols2 > MAX_NCOLS2) {
    ncols2 = MAX_NCOLS2;
  }

  // Phase 5: Adaptive cols_per_block — select ncols1 based on seqlen_q and ncols2
  // Larger ncols1 for longer sequences reduces kernel launch overhead and improves
  // work-item utilization. Smaller ncols1 for short sequences avoids wasted compute.
  const unsigned int ncols1 = select_ncols1(seqlen_q, ncols2);

  // Phase 2: Tiled GEMM prefill kernel configuration
  // NBATCH_FA: KV rows per tile
  static const unsigned int NBATCH_FA = 16;

  // Work-group size: must be large enough to cooperatively load tiles
  static const unsigned int PREFILL_WORK_GROUP_SIZE = 128;

  // Number of Q groups (tiles along the sequence dimension) — uses runtime ncols1
  const unsigned int num_q_groups = (seqlen_q + ncols1 - 1) / ncols1;

  // Phase 4: We may need multiple dispatches if gqa_ratio > ncols2
  // Each dispatch handles `ncols2` Q heads per KV head
  // Number of dispatches = ceil(gqa_ratio / ncols2)
  const unsigned int num_dispatches = (gqa_ratio + ncols2 - 1) / ncols2;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_prefill_fp16_kernel, "flash_attention_prefill_fp16");
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_prefill_fp16");
    return;
  }

  // Dispatch multiple times if gqa_ratio > ncols2
  // Each dispatch processes a subset of Q heads for each KV head
  for (unsigned int dispatch = 0; dispatch < num_dispatches; dispatch++) {
    const unsigned int head_group_offset = dispatch * ncols2;
    // Actual ncols2 for this dispatch (may be less at boundary)
    const unsigned int dispatch_ncols2 = std::min(ncols2, gqa_ratio - head_group_offset);

    // Total work-groups: batch * num_heads_kv * num_q_groups
    // (Note: iterate over KV heads, not Q heads — each work-group handles dispatch_ncols2 Q heads)
    const unsigned int total_groups = batch * num_heads_kv * num_q_groups;

    int arg = 0;
    bool result = false;

    result = kernel_ptr->SetKernelSVMArguments(arg++, query);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelSVMArguments(arg++, key);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelSVMArguments(arg++, value);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelSVMArguments(arg++, output);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_prefill_fp16");

    // Phase 4: Kernel arguments for GQA grouping
    result = kernel_ptr->SetKernelArguments(arg++, &dispatch_ncols2, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 11 (ncols2) for flash_attention_prefill_fp16");

    result = kernel_ptr->SetKernelArguments(arg++, &head_group_offset, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 12 (head_group_offset) for flash_attention_prefill_fp16");

    // Phase 5: Runtime ncols1 argument (adaptive cols_per_block)
    result = kernel_ptr->SetKernelArguments(arg++, &ncols1, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 13 (ncols1) for flash_attention_prefill_fp16");

    // Dispatch: total_groups work-groups, each with PREFILL_WORK_GROUP_SIZE work-items
    const int work_groups_count[3] = {(int)(total_groups * PREFILL_WORK_GROUP_SIZE), 1, 1};
    const int work_group_size[3] = {(int)PREFILL_WORK_GROUP_SIZE, 1, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error("Failed to dispatch kernel for flash_attention_prefill_fp16");
      return;
    }
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output, 
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16),
                                             true);
}

void flash_attention_prefill_fp16_adreno_cl(_FP16 *query, _FP16 *key, _FP16 *value,
                                            _FP16 *output, unsigned int seqlen_q,
                                            unsigned int seqlen_k, unsigned int head_dim,
                                            unsigned int num_heads_q,
                                            unsigned int num_heads_kv,
                                            unsigned int batch, float scale) {
  // For very small workloads, use CPU implementation to avoid GPU overhead
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  if (total_work_items < 32 || total_elements < 4096) {
    const size_t kv_elements = batch * num_heads_kv * seqlen_k * head_dim;
    std::vector<float> query_fp32(total_elements);
    std::vector<float> key_fp32(kv_elements);
    std::vector<float> value_fp32(kv_elements);
    std::vector<float> output_fp32(total_elements);
    
    fp16_to_fp32(query, query_fp32.data(), total_elements);
    fp16_to_fp32(key, key_fp32.data(), kv_elements);
    fp16_to_fp32(value, value_fp32.data(), kv_elements);
    
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                       output_fp32.data(), seqlen_q, seqlen_k, head_dim,
                       num_heads_q, num_heads_kv, batch, scale);
    
    fp32_to_fp16(output_fp32.data(), output, total_elements);
    return;
  }

  // GQA grouping — same logic as Mali kernel
  const unsigned int gqa_ratio = num_heads_q / num_heads_kv;
  unsigned int ncols2 = 1;
  if (gqa_ratio >= 4 && seqlen_q > 4) {
    ncols2 = 2;
  }
  if (ncols2 > MAX_NCOLS2) {
    ncols2 = MAX_NCOLS2;
  }

  // Opt 9: FP16 Adreno — keep NCOLS1=4 (NCOLS1=8 caused 15% regression due to
  // local memory pressure reducing occupancy: 39ms vs 34ms with NCOLS1=4)
  const unsigned int ncols1 = select_ncols1(seqlen_q, ncols2);

  // Adreno kernel uses larger NBATCH_FA (32 vs 16 for Mali)
  static const unsigned int NBATCH_FA_ADRENO = 32;
  // Opt 11: Runtime work-group size selection based on device local memory
  // FP16 Adreno local memory: ~20 KB (NCOLS1=4, NBATCH_FA=32) — 128 is optimal
  const unsigned int PREFILL_WORK_GROUP_SIZE = select_work_group_size(20 * 1024);

  const unsigned int num_q_groups = (seqlen_q + ncols1 - 1) / ncols1;
  const unsigned int num_dispatches = (gqa_ratio + ncols2 - 1) / ncols2;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  // Build compile options: enable sub-group support if the device supports it
  // Adreno 6xx uses cl_qcom_subgroups, Adreno 7xx+/8xx uses cl_khr_subgroups
  // Both provide hardware-accelerated sub-group reductions (5-8% faster softmax)
  std::string adreno_compile_options = get_subgroup_compile_option();
  if (supports_subgroups()) {
    ml_logi("Adreno prefill: sub-group extension detected (%s), "
            "enabling hardware sub-group reductions",
            adreno_compile_options.c_str() + 2);  // Skip "-D" prefix for logging
  } else {
    ml_logi("Adreno prefill: no sub-group extension detected, "
            "using manual barrier-based reductions");
  }

  // Register the Adreno-optimized kernel
  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_prefill_fp16_adreno_kernel, "flash_attention_prefill_fp16_adreno",
    adreno_compile_options);
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_prefill_fp16_adreno");
    return;
  }

  for (unsigned int dispatch = 0; dispatch < num_dispatches; dispatch++) {
    const unsigned int head_group_offset = dispatch * ncols2;
    const unsigned int dispatch_ncols2 = std::min(ncols2, gqa_ratio - head_group_offset);
    const unsigned int total_groups = batch * num_heads_kv * num_q_groups;

    int arg = 0;
    bool result = false;

    result = kernel_ptr->SetKernelSVMArguments(arg++, query);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelSVMArguments(arg++, key);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelSVMArguments(arg++, value);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelSVMArguments(arg++, output);
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &dispatch_ncols2, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 11 (ncols2) for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &head_group_offset, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 12 (head_group_offset) for flash_attention_prefill_fp16_adreno");

    result = kernel_ptr->SetKernelArguments(arg++, &ncols1, sizeof(int));
    if (!result)
      throw std::runtime_error("Failed to set kernel argument 13 (ncols1) for flash_attention_prefill_fp16_adreno");

    const int work_groups_count[3] = {(int)(total_groups * PREFILL_WORK_GROUP_SIZE), 1, 1};
    const int work_group_size[3] = {(int)PREFILL_WORK_GROUP_SIZE, 1, 1};

    result = blas_cc->command_queue_inst_.DispatchCommand(
      kernel_ptr, work_groups_count, work_group_size);
    if (!result) {
      throw std::runtime_error("Failed to dispatch kernel for flash_attention_prefill_fp16_adreno");
      return;
    }
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output, 
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16),
                                             true);
}

void flash_attention_decode_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value,
                                    _FP16 *output, unsigned int seqlen_q,
                                    unsigned int seqlen_k, unsigned int head_dim,
                                    unsigned int num_heads_q,
                                    unsigned int num_heads_kv,
                                    unsigned int batch, float scale) {
  // Phase 6: Decode-specific kernel with split-KV approach
  // For very small workloads (typical in decode), CPU fallback may be faster.
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  // For decode with small seqlen_k, use CPU or simple kernel
  // The split-KV approach is beneficial when seqlen_k is large enough to warrant parallelization
  static const unsigned int KV_TILE_SIZE = 64;
  const unsigned int num_kv_tiles = (seqlen_k + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
  
  // For small KV sequences or small batches, use CPU fallback
  if (total_work_items < 32 || total_elements < 4096 || num_kv_tiles < 2) {
    // Convert FP16 to FP32 for CPU computation
    const size_t kv_elements = batch * num_heads_kv * seqlen_k * head_dim;
    std::vector<float> query_fp32(total_elements);
    std::vector<float> key_fp32(kv_elements);
    std::vector<float> value_fp32(kv_elements);
    std::vector<float> output_fp32(total_elements);
    
    fp16_to_fp32(query, query_fp32.data(), total_elements);
    fp16_to_fp32(key, key_fp32.data(), kv_elements);
    fp16_to_fp32(value, value_fp32.data(), kv_elements);
    
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                       output_fp32.data(), seqlen_q, seqlen_k, head_dim,
                       num_heads_q, num_heads_kv, batch, scale);
    
    fp32_to_fp16(output_fp32.data(), output, total_elements);
    return;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  // Allocate partial buffers for split-KV reduction
  // Note: partials are stored in FP32 for numerical stability
  // partials_max: [batch, num_heads_q, num_kv_tiles]
  // partials_sum: [batch, num_heads_q, num_kv_tiles]
  // partials_vkq: [batch, num_heads_q, num_kv_tiles, head_dim]
  const size_t partials_max_sum_size = batch * num_heads_q * num_kv_tiles * sizeof(float);
  const size_t partials_vkq_size = batch * num_heads_q * num_kv_tiles * head_dim * sizeof(float);
  
  // Allocate SVM buffers for partials (FP32 for numerical stability)
  float *partials_max_buf = (float *)blas_cc->context_inst_.createSVMRegion(partials_max_sum_size);
  float *partials_sum_buf = (float *)blas_cc->context_inst_.createSVMRegion(partials_max_sum_size);
  float *partials_vkq_buf = (float *)blas_cc->context_inst_.createSVMRegion(partials_vkq_size);
  
  if (!partials_max_buf || !partials_sum_buf || !partials_vkq_buf) {
    // Fallback to CPU if allocation fails
    const size_t kv_elements = batch * num_heads_kv * seqlen_k * head_dim;
    std::vector<float> query_fp32(total_elements);
    std::vector<float> key_fp32(kv_elements);
    std::vector<float> value_fp32(kv_elements);
    std::vector<float> output_fp32(total_elements);
    
    fp16_to_fp32(query, query_fp32.data(), total_elements);
    fp16_to_fp32(key, key_fp32.data(), kv_elements);
    fp16_to_fp32(value, value_fp32.data(), kv_elements);
    
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                       output_fp32.data(), seqlen_q, seqlen_k, head_dim,
                       num_heads_q, num_heads_kv, batch, scale);
    
    fp32_to_fp16(output_fp32.data(), output, total_elements);
    return;
  }

  // Kernel 1: Decode kernel - each work-group processes a KV tile
  ClContext::SharedPtrClKernel decode_kernel_ptr = blas_cc->registerClKernel(
    flash_attention_decode_fp16_kernel, "flash_attention_decode_fp16");
  if (!decode_kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_decode_fp16");
    return;
  }

  // kv_max = seqlen_k (all KV positions are valid)
  const int kv_max = seqlen_k;
  
  // Work-group size for decode kernel
  static const unsigned int DECODE_WORK_GROUP_SIZE = 64;
  const unsigned int total_decode_groups = batch * num_heads_q * num_kv_tiles;
  
  int arg = 0;
  bool result = false;

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, partials_max_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, partials_sum_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelSVMArguments(arg++, partials_vkq_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 11 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 12 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 13 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &kv_max, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 14 for flash_attention_decode_fp16");

  result = decode_kernel_ptr->SetKernelArguments(arg++, &num_kv_tiles, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 15 for flash_attention_decode_fp16");

  const int decode_work_groups_count[3] = {(int)(total_decode_groups * DECODE_WORK_GROUP_SIZE), 1, 1};
  const int decode_work_group_size[3] = {(int)DECODE_WORK_GROUP_SIZE, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    decode_kernel_ptr, decode_work_groups_count, decode_work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch decode kernel for flash_attention_decode_fp16");
    return;
  }

  // Kernel 2: Reduction kernel - combine partials using log-sum-exp
  // Note: Both decode and reduce kernels are in the same .cl file, so we use the same kernel string
  ClContext::SharedPtrClKernel reduce_kernel_ptr = blas_cc->registerClKernel(
    flash_attention_decode_fp16_kernel, "flash_attention_decode_reduce_fp16");
  if (!reduce_kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_decode_reduce_fp16");
    return;
  }

  arg = 0;
  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, partials_max_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_decode_reduce_fp16");

  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, partials_sum_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_decode_reduce_fp16");

  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, partials_vkq_buf);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_decode_reduce_fp16");

  result = reduce_kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_decode_reduce_fp16");

  result = reduce_kernel_ptr->SetKernelArguments(arg++, &num_kv_tiles, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_decode_reduce_fp16");

  result = reduce_kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_decode_reduce_fp16");

  result = reduce_kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_decode_reduce_fp16");

  // One work-group per (batch, head) pair
  const int reduce_work_groups_count[3] = {(int)(batch * num_heads_q * DECODE_WORK_GROUP_SIZE), 1, 1};
  const int reduce_work_group_size[3] = {(int)DECODE_WORK_GROUP_SIZE, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    reduce_kernel_ptr, reduce_work_groups_count, reduce_work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch reduce kernel for flash_attention_decode_fp16");
    return;
  }

  // Read back output
  blas_cc->command_queue_inst_.enqueueSVMMap(output,
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_decode_fp16");
    return;
  }
  
  // Free partial buffers
  // Note: SVM buffers are freed automatically when the context is destroyed
}
#endif /* ENABLE_FP16 */

} // namespace nntrainer