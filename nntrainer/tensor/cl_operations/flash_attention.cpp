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
#ifdef ENABLE_FP16
#include <cl_kernels/flash_attention_fp16.h>
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

void flash_attention_fp32_cl(float *query, float *key, float *value, float *output,
                             unsigned int seqlen_q, unsigned int seqlen_k,
                             unsigned int head_dim, unsigned int num_heads_q,
                             unsigned int num_heads_kv, unsigned int batch,
                             float scale) {
  // For very small workloads, use CPU implementation to avoid GPU overhead
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  // Threshold for switching to CPU - tune based on empirical testing
  if (total_work_items < 32 || total_elements < 4096) {
    // Fall back to CPU implementation for small workloads
    // This is a simplified fallback - in practice, you might want to implement
    // a more sophisticated CPU version or use a different GPU strategy
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
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_fp32");
    return;
  }

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_fp32");

  result = kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_fp32");

  result = kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_fp32");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_fp32");

  result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_fp32");

  const int work_groups_count[3] = {(int)total_work_items, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch kernel for flash_attention_fp32");
    return;
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output, 
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(float),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_fp32");
    return;
  }
}

#ifdef ENABLE_FP16
void flash_attention_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value, _FP16 *output,
                             unsigned int seqlen_q, unsigned int seqlen_k,
                             unsigned int head_dim, unsigned int num_heads_q,
                             unsigned int num_heads_kv, unsigned int batch,
                             float scale) {
  // For very small workloads, use CPU implementation to avoid GPU overhead
  const unsigned int total_elements = batch * num_heads_q * seqlen_q * head_dim;
  const unsigned int total_work_items = batch * num_heads_q * seqlen_q;
  
  // Threshold for switching to CPU - tune based on empirical testing
  if (total_work_items < 32 || total_elements < 4096) {
    // Convert FP16 to FP32 for CPU computation
    std::vector<float> query_fp32(total_elements);
    std::vector<float> key_fp32(batch * num_heads_kv * seqlen_k * head_dim);
    std::vector<float> value_fp32(batch * num_heads_kv * seqlen_k * head_dim);
    std::vector<float> output_fp32(total_elements);
    
    // Convert inputs to FP32
    for (size_t i = 0; i < total_elements; ++i) {
      query_fp32[i] = static_cast<float>(query[i]);
    }
    for (size_t i = 0; i < batch * num_heads_kv * seqlen_k * head_dim; ++i) {
      key_fp32[i] = static_cast<float>(key[i]);
      value_fp32[i] = static_cast<float>(value[i]);
    }
    
    // Compute on CPU
    flash_attention_cpu(query_fp32.data(), key_fp32.data(), value_fp32.data(),
                       output_fp32.data(), seqlen_q, seqlen_k, head_dim,
                       num_heads_q, num_heads_kv, batch, scale);
    
    // Convert output back to FP16
    for (size_t i = 0; i < total_elements; ++i) {
      output[i] = static_cast<_FP16>(output_fp32[i]);
    }
    return;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = blas_cc->registerClKernel(
    flash_attention_fp16_kernel, "flash_attention_fp16");
  if (!kernel_ptr) {
    throw std::runtime_error("Failed to get kernel_ptr for flash_attention_fp16");
    return;
  }

  int arg = 0;
  bool result = false;

  result = kernel_ptr->SetKernelSVMArguments(arg++, query);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 0 for flash_attention_fp16");

  result = kernel_ptr->SetKernelSVMArguments(arg++, key);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 1 for flash_attention_fp16");

  result = kernel_ptr->SetKernelSVMArguments(arg++, value);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 2 for flash_attention_fp16");

  result = kernel_ptr->SetKernelSVMArguments(arg++, output);
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 3 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 4 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &seqlen_k, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 5 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &head_dim, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 6 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_q, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 7 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &num_heads_kv, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 8 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &batch, sizeof(int));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 9 for flash_attention_fp16");

  result = kernel_ptr->SetKernelArguments(arg++, &scale, sizeof(float));
  if (!result)
    throw std::runtime_error("Failed to set kernel argument 10 for flash_attention_fp16");

  const int work_groups_count[3] = {(int)total_work_items, 1, 1};
  const int work_group_size[3] = {64, 1, 1};

  result = blas_cc->command_queue_inst_.DispatchCommand(
    kernel_ptr, work_groups_count, work_group_size);
  if (!result) {
    throw std::runtime_error("Failed to dispatch kernel for flash_attention_fp16");
    return;
  }

  blas_cc->command_queue_inst_.enqueueSVMMap(output, 
                                             batch * num_heads_q * seqlen_q * head_dim * sizeof(_FP16),
                                             true);
  if (!result) {
    throw std::runtime_error("Failed to read output data for flash_attention_fp16");
    return;
  }
}
#endif /* ENABLE_FP16 */

} // namespace nntrainer
