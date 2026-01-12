// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file    cuda_buffer_manager.cpp
 * @date    11 Dec 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Donghak Jung <dk11.jung@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them for CUDA
 */

#include <cstring>
#include <vector>

#include "cuda_buffer_manager.h"

namespace nntrainer {

void CudaBufferManager::initBuffers() {
  // Assuming allocDeviceMemory creates a device memory allocation suitable for
  // general use Since we don't have a specific Buffer wrapper for CUDA yet, we
  // treat all as device pointers.

  inBufferA = context_inst_.allocateDeviceMemory(buffer_size_bytes);
  inBufferB = context_inst_.allocateDeviceMemory(buffer_size_bytes);
  inBufferC = context_inst_.allocateDeviceMemory(unused_buffer_bytes);
  outBufferA = context_inst_.allocateDeviceMemory(buffer_size_bytes);
  outBufferB = context_inst_.allocateDeviceMemory(unused_buffer_bytes);

  data_input = context_inst_.allocateDeviceMemory(buffer_size_bytes);
  for (unsigned int i = 0; i < max_qs; ++i) {
    scale_vec.push_back(context_inst_.allocateDeviceMemory(scale_q4_0_size));
    quant_vec.push_back(context_inst_.allocateDeviceMemory(quant_q4_0_size));
    output_vec.push_back(context_inst_.allocateDeviceMemory(buffer_size_bytes));
  }

  ml_logi("CudaBufferManager: Buffers initialized");
}

CudaBufferManager::~CudaBufferManager() {
  context_inst_.releaseDeviceMemory(inBufferA);
  context_inst_.releaseDeviceMemory(inBufferB);
  context_inst_.releaseDeviceMemory(inBufferC);
  context_inst_.releaseDeviceMemory(outBufferA);
  context_inst_.releaseDeviceMemory(outBufferB);

  context_inst_.releaseDeviceMemory(data_input);
  for (unsigned int i = 0; i < max_qs; ++i) {
    context_inst_.releaseDeviceMemory(scale_vec[i]);
    context_inst_.releaseDeviceMemory(quant_vec[i]);
    context_inst_.releaseDeviceMemory(output_vec[i]);
  }

  ml_logi("CudaBufferManager: Buffers destroyed");
}

} // namespace nntrainer
