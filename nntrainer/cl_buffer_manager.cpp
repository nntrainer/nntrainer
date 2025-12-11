// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_buffer_manager.cpp
 * @date    01 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#include <cstring>

#include <cl_buffer_manager.h>
#include <opencl_loader.h>

namespace nntrainer {

// to-do: Implementation to be updated with array of Buffer objects if required
// fp16 Buffer objects to be added in future
void ClBufferManager::initBuffers() {
  inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);

  data_input = context_inst_.createSVMRegion(buffer_size_bytes);
  for (unsigned int i = 0; i < max_qs; ++i) {
    scale_vec.push_back(context_inst_.createSVMRegion(scale_q4_0_size));
    quant_vec.push_back(context_inst_.createSVMRegion(quant_q4_0_size));
    output_vec.push_back(context_inst_.createSVMRegion(buffer_size_bytes));
  }

  ml_logi("ClBufferManager: Buffers & images initialized");
}

ClBufferManager::~ClBufferManager() {
  delete inBufferA;
  delete inBufferB;
  delete inBufferC;
  delete outBufferA;
  delete outBufferB;

  context_inst_.releaseSVMRegion(data_input);
  for (unsigned int i = 0; i < max_qs; ++i) {
    context_inst_.releaseSVMRegion(scale_vec[i]);
    context_inst_.releaseSVMRegion(quant_vec[i]);
    context_inst_.releaseSVMRegion(output_vec[i]);
  }

  ml_logi("ClBufferManager: Buffers destroyed");
}

} // namespace nntrainer
