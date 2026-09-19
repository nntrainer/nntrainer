// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_kernel.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA kernel (CUfunction) management implementation.
 */

#include "cuda_kernel.h"
#include "cuda_common.h"
#include "cuda_module.h"

namespace nntrainer::cuda {

bool Kernel::CreateKernelFromModule(const Module &module,
                                    const std::string &function_name) {
  arg_storage_.clear();
  if (!module.valid()) {
    ml_loge("[CUDA] CreateKernelFromModule: module not loaded (%s)",
            function_name.c_str());
    return false;
  }
  return cuCheck(
    cuModuleGetFunction(&function_, module.GetModule(), function_name.c_str()),
    "cuModuleGetFunction");
}

bool Kernel::SetKernelArguments(unsigned int arg_index, const void *arg_value,
                                size_t size) {
  if (arg_storage_.size() <= arg_index)
    arg_storage_.resize(arg_index + 1);
  const char *p = static_cast<const char *>(arg_value);
  arg_storage_[arg_index].assign(p, p + size);
  return true;
}

std::vector<void *> Kernel::getKernelParams() {
  std::vector<void *> params;
  params.reserve(arg_storage_.size());
  for (auto &a : arg_storage_)
    params.push_back(a.data());
  return params;
}

} // namespace nntrainer::cuda
