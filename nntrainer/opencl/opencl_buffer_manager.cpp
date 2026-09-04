// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_buffer_manager.cpp
 * @date    01 Dec 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#include <cstring>

#include <opencl_buffer_manager.h>
#include <opencl_loader.h>

namespace nntrainer {

void ClBufferManager::initBuffers() {
  data_input = context_inst_.createSVMRegion(buffer_size_bytes);
  for (unsigned int i = 0; i < max_qs; ++i) {
    scale_vec.push_back(context_inst_.createSVMRegion(scale_q4_0_size));
    quant_vec.push_back(context_inst_.createSVMRegion(quant_q4_0_size));
    output_vec.push_back(context_inst_.createSVMRegion(buffer_size_bytes));
  }
}

opencl::Buffer *ClBufferManager::getInBufferA() {
  if (inBufferA == nullptr) {
    inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferA;
}

opencl::Buffer *ClBufferManager::getInBufferB() {
  if (inBufferB == nullptr) {
    inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferB;
}

opencl::Buffer *ClBufferManager::getInBufferC() {
  if (inBufferC == nullptr) {
    inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferC;
}

opencl::Buffer *ClBufferManager::getOutBufferA() {
  if (outBufferA == nullptr) {
    outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferA;
}

opencl::Buffer *ClBufferManager::getOutBufferB() {
  if (outBufferB == nullptr) {
    outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferB;
}

void *ClBufferManager::getSVMInput() { return data_input; }

void *ClBufferManager::getSVMScale(unsigned int idx) {
  if (idx >= scale_vec.size())
    return nullptr;

  return scale_vec[idx];
}

void *ClBufferManager::getSVMQuant(unsigned int idx) {
  if (idx >= quant_vec.size())
    return nullptr;

  return quant_vec[idx];
}

void *ClBufferManager::getSVMOutput(unsigned int idx) {
  if (idx >= output_vec.size())
    return nullptr;

  return output_vec[idx];
}

void ClBufferManager::releaseBuffers() {
  delete inBufferA;
  inBufferA = nullptr;
  delete inBufferB;
  inBufferB = nullptr;
  delete inBufferC;
  inBufferC = nullptr;
  delete outBufferA;
  outBufferA = nullptr;
  delete outBufferB;
  outBufferB = nullptr;

  if (data_input) {
    context_inst_.releaseSVMRegion(data_input);
    data_input = nullptr;
  }
  for (auto &ptr : scale_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  scale_vec.clear();
  for (auto &ptr : quant_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  quant_vec.clear();
  for (auto &ptr : output_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  output_vec.clear();
}

ClBufferManager::~ClBufferManager() { releaseBuffers(); }

} // namespace nntrainer
