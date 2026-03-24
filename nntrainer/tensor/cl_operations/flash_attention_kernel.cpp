// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Kumar Tiwari(anup.tiwari@samsung.com)
 *
 * @file	flash_attention_kernel.cpp
 * @date	23 March 2026
 * @brief	Flash attention OpenCL kernel implementation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Anup Kumar Tiwari(anup.tiwari@samsung.com)
 * @bug		No known bugs except for NYI items
 *
 */

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer_manager.h>
#include <opencl_kernel.h>
#include <opencl_program.h>
#include <tensor.h>

#include "flash_attention_kernel.h"
#include <cl_kernels/flash_attention.h>

namespace nntrainer {

template <typename T>
inline static void flash_attention_cl_internal(
  ClContext::SharedPtrClKernel kernel, const T *query, const T *key, const T *value,
  T *output, const T *attention_mask, int batch_size, int num_heads, int seq_len,
  int head_dim, T scale) {
  
  bool result = false;

  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &cl_buffer_manager = ClBufferManager::Global();

  do {
    // Calculate buffer sizes
    size_t query_size = batch_size * num_heads * seq_len * head_dim * sizeof(T);
    size_t key_size = batch_size * num_heads * seq_len * head_dim * sizeof(T);
    size_t value_size = batch_size * num_heads * seq_len * head_dim * sizeof(T);
    size_t output_size = batch_size * num_heads * seq_len * head_dim * sizeof(T);
    size_t mask_size = attention_mask ? 
      batch_size * seq_len * seq_len * sizeof(T) : 0;

    // Write data to buffers
    result = cl_buffer_manager.getInBufferA()->WriteDataRegion(
      cl_context->command_queue_inst_, query_size, query);
    if (!result) {
      printf("Failed to write query data\n");
      break;
    }

    result = cl_buffer_manager.getInBufferB()->WriteDataRegion(
      cl_context->command_queue_inst_, key_size, key);
    if (!result) {
      printf("Failed to write key data\n");
      break;
    }

    result = cl_buffer_manager.getInBufferC()->WriteDataRegion(
      cl_context->command_queue_inst_, value_size, value);
    if (!result) {
      printf("Failed to write value data\n");
      break;
    }

    result = cl_buffer_manager.getOutBufferA()->WriteDataRegion(
      cl_context->command_queue_inst_, output_size, output);
    if (!result) {
      printf("Failed to write output data\n");
      break;
    }

    if (attention_mask != nullptr) {
      result = cl_buffer_manager.getOutBufferB()->WriteDataRegion(
        cl_context->command_queue_inst_, mask_size, attention_mask);
      if (!result) {
        printf("Failed to write attention mask data\n");
        break;
      }
    }

    // Set kernel arguments
    result = kernel->SetKernelArguments(0, cl_buffer_manager.getInBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      printf("Failed to set query argument\n");
      break;
    }

    result = kernel->SetKernelArguments(1, cl_buffer_manager.getInBufferB(),
                                        sizeof(cl_mem));
    if (!result) {
      printf("Failed to set key argument\n");
      break;
    }

    result = kernel->SetKernelArguments(2, cl_buffer_manager.getInBufferC(),
                                        sizeof(cl_mem));
    if (!result) {
      printf("Failed to set value argument\n");
      break;
    }

    result = kernel->SetKernelArguments(3, cl_buffer_manager.getOutBufferA(),
                                        sizeof(cl_mem));
    if (!result) {
      printf("Failed to set output argument\n");
      break;
    }

    cl_mem mask_buffer = attention_mask ? cl_buffer_manager.getOutBufferB()->GetBuffer() : nullptr;
    result = kernel->SetKernelArguments(4, &mask_buffer, sizeof(cl_mem));
    if (!result) {
      printf("Failed to set attention mask argument\n");
      break;
    }

    result = kernel->SetKernelArguments(5, &batch_size, sizeof(int));
    if (!result) {
      printf("Failed to set batch_size argument\n");
      break;
    }

    result = kernel->SetKernelArguments(6, &num_heads, sizeof(int));
    if (!result) {
      printf("Failed to set num_heads argument\n");
      break;
    }

    result = kernel->SetKernelArguments(7, &seq_len, sizeof(int));
    if (!result) {
      printf("Failed to set seq_len argument\n");
      break;
    }

    result = kernel->SetKernelArguments(8, &head_dim, sizeof(int));
    if (!result) {
      printf("Failed to set head_dim argument\n");
      break;
    }

    result = kernel->SetKernelArguments(9, &scale, sizeof(T));
    if (!result) {
      printf("Failed to set scale argument\n");
      break;
    }

    // Calculate work group sizes
    const int work_groups_count[3] = {batch_size, num_heads, seq_len};
    const int work_group_size[3] = {1, 1, 1};
    
    result = cl_context->command_queue_inst_.DispatchCommand(
      kernel, work_groups_count, work_group_size);
    if (!result) {
      printf("Failed to dispatch command\n");
      break;
    }

    // Read output data
    result = cl_buffer_manager.getOutBufferA()->ReadDataRegion(
      cl_context->command_queue_inst_, output_size, output);
    if (!result) {
      printf("Failed to read output data\n");
      break;
    }

  } while (false);
}

void flash_attention_cl(const Tensor &query, const Tensor &key, const Tensor &value,
                        Tensor &output, const Tensor *attention_mask,
                        int batch_size, int num_heads, int seq_len, int head_dim,
                        float scale) {
  
  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = 
    cl_context->registerClKernel(flash_attention_kernel, "flash_attention_cl");
  if (!kernel_ptr) {
    printf("Failed to register flash attention kernel\n");
    return;
  }

  const float *query_data = query.getData<float>();
  const float *key_data = key.getData<float>();
  const float *value_data = value.getData<float>();
  float *output_data = output.getData<float>();
  const float *mask_data = attention_mask ? attention_mask->getData<float>() : nullptr;

  flash_attention_cl_internal<float>(
    kernel_ptr, query_data, key_data, value_data, output_data, mask_data,
    batch_size, num_heads, seq_len, head_dim, scale);
}

#ifdef ENABLE_FP16
#include <cl_kernels/flash_attention_fp16.h>

void flash_attention_cl_fp16(const Tensor &query, const Tensor &key, const Tensor &value,
                             Tensor &output, const Tensor *attention_mask,
                             int batch_size, int num_heads, int seq_len, int head_dim,
                             float scale) {
  
  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_ptr = 
    cl_context->registerClKernel(flash_attention_fp16_kernel, "flash_attention_cl_fp16");
  if (!kernel_ptr) {
    printf("Failed to register flash attention FP16 kernel\n");
    return;
  }

  const _FP16 *query_data = query.getData<_FP16>();
  const _FP16 *key_data = key.getData<_FP16>();
  const _FP16 *value_data = value.getData<_FP16>();
  _FP16 *output_data = output.getData<_FP16>();
  const _FP16 *mask_data = attention_mask ? attention_mask->getData<_FP16>() : nullptr;

  flash_attention_cl_internal<_FP16>(
    kernel_ptr, query_data, key_data, value_data, output_data, mask_data,
    batch_size, num_heads, seq_len, head_dim, static_cast<_FP16>(scale));
}
#endif

} // namespace nntrainer