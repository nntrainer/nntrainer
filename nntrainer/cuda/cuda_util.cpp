// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file    util.cpp
 * @date    11 Dec 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Donghak Jung <dk11.jung@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA utility functions implementation
 *
 */

#include "cuda_util.h"
#include <nntrainer_log.h>
#include <nntrainer_error.h>
#include <iostream>

namespace nntrainer::cuda {

void cuda_check_error(cudaError_t status, const char *file, int line) {
  if (status != cudaSuccess) {
    std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(status)) +
                            " at " + std::string(file) + ":" + std::to_string(line);
    ml_loge("%s", error_msg.c_str());
    throw std::runtime_error(error_msg);
  }
}

void cu_check_error(CUresult status, const char *file, int line) {
  if (status != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(status, &err_str);
    std::string error_msg = "CUDA Driver Error: " + std::string(err_str) +
                            " at " + std::string(file) + ":" + std::to_string(line);
    ml_loge("%s", error_msg.c_str());
    throw std::runtime_error(error_msg);
  }
}

} // namespace nntrainer::cuda
