// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file    util.h
 * @date    11 Dec 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Donghak Jung <dk11.jung@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA utility functions
 *
 */

#ifndef __NNTRAINER_CUDA_UTIL_H__
#define __NNTRAINER_CUDA_UTIL_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#define CUDA_CHECK(status)                                                     \
  nntrainer::cuda::cuda_check_error(status, __FILE__, __LINE__)
#define CU_CHECK(status)                                                       \
  nntrainer::cuda::cu_check_error(status, __FILE__, __LINE__)

namespace nntrainer::cuda {

/**
 * @brief   Check CUDA runtime API status and log error if failed
 *
 * @param   status cudaError_t to check
 * @param   file file name where error occurred
 * @param   line line number where error occurred
 * @return  void
 */
void cuda_check_error(cudaError_t status, const char *file, int line);

/**
 * @brief   Check CUDA driver API status and log error if failed
 *
 * @param   status CUresult to check
 * @param   file file name where error occurred
 * @param   line line number where error occurred
 * @return  void
 */
void cu_check_error(CUresult status, const char *file, int line);

} // namespace nntrainer::cuda

#endif // __NNTRAINER_CUDA_UTIL_H__
