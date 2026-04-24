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
#include <cl_kernels/flash_attention_fp32.h>
#include <cl_kernels/flash_attention_prefill_fp32.h>
#ifdef ENABLE_FP16
#include <cl_kernels/flash_attention_fp16.h>
#include <cl_kernels/flash_attention_prefill_fp16.h>
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
    flash_attention_prefill_fp32_cl(query, key, value, output,
                                   seqlen_q, seqlen_k, head_dim,
                                   num_heads_q, num_heads_kv, batch, scale);
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

void flash_attention_decode_fp32_cl(float *query, float *key, float *value,
                                    float *output, unsigned int seqlen_q,
                                    unsigned int seqlen_k, unsigned int head_dim,
                                    unsigned int num_heads_q,
                                    unsigned int num_heads_kv,
                                    unsigned int batch, float scale) {
  // Phase 1: Decode uses the same existing kernel as prefill.
  // This will be replaced with a dedicated split-KV decode kernel in Phase 6.
  // For very small workloads (typical in decode), CPU fallback may be faster.
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  if (total_work_items < 32 || total_elements < 4096) {
    flash_attention_cpu(query, key, value, output,
                        seqlen_q, seqlen_k, head_dim,
                        num_heads_q, num_heads_kv, batch, scale);
    return;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_fp32_kernel, "flash_attention_fp32");
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_fp32 (decode)");
    return;
  }

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_fp32 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_fp32 (decode)");

  const int work_groups_count[3] = {(int)total_work_items, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch kernel for flash_attention_fp32 (decode)");
    return;
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output,
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(float),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_fp32 (decode)");
    return;
  }
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
    flash_attention_prefill_fp16_cl(query, key, value, output,
                                   seqlen_q, seqlen_k, head_dim,
                                   num_heads_q, num_heads_kv, batch, scale);
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

void flash_attention_decode_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value,
                                    _FP16 *output, unsigned int seqlen_q,
                                    unsigned int seqlen_k, unsigned int head_dim,
                                    unsigned int num_heads_q,
                                    unsigned int num_heads_kv,
                                    unsigned int batch, float scale) {
  // Phase 1: Decode uses the same existing kernel as prefill.
  // This will be replaced with a dedicated split-KV decode kernel in Phase 6.
  // For very small workloads (typical in decode), CPU fallback may be faster.
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
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
    
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                       output_fp32.data(), seqlen_q, seqlen_k, head_dim,
                       num_heads_q, num_heads_kv, batch, scale);
    
    fp32_to_fp16(output_fp32.data(), output, total_elements);
    return;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_fp16_kernel, "flash_attention_fp16");
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_fp16 (decode)");
    return;
  }

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_fp16 (decode)");

  result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_fp16 (decode)");

  const int work_groups_count[3] = {(int)total_work_items, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch kernel for flash_attention_fp16 (decode)");
    return;
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output,
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_fp16 (decode)");
    return;
  }
}
#endif /* ENABLE_FP16 */

} // namespace nntrainer