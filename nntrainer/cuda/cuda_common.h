// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_common.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Shared error-checking helpers for the additive CUDA backend.
 *          Mirrors the role of OpenCLErrorCodeToString in opencl_loader for the
 *          three CUDA API surfaces we use: Runtime (cudart), Driver, and NVRTC.
 */

#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#include <cuda.h>         // driver API: CUcontext/CUmodule/cuLaunchKernel
#include <cuda_runtime.h> // runtime API: cudaMalloc/cudaMemcpy/cudaStream_t
#include <nvrtc.h>        // runtime kernel compilation

#include <nntrainer_log.h>

namespace nntrainer::cuda {

/**
 * @brief check a CUDA Runtime API return, logging on failure
 * @return true on cudaSuccess
 */
inline bool cudaCheck(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    ml_loge("[CUDA] %s failed: %s", what, cudaGetErrorString(e));
    return false;
  }
  return true;
}

/**
 * @brief check a CUDA Driver API return, logging on failure
 * @return true on CUDA_SUCCESS
 */
inline bool cuCheck(CUresult e, const char *what) {
  if (e != CUDA_SUCCESS) {
    const char *s = nullptr;
    cuGetErrorString(e, &s);
    ml_loge("[CUDA-drv] %s failed: %s", what, s ? s : "unknown");
    return false;
  }
  return true;
}

/**
 * @brief check an NVRTC return, logging on failure
 * @return true on NVRTC_SUCCESS
 */
inline bool nvrtcCheck(nvrtcResult e, const char *what) {
  if (e != NVRTC_SUCCESS) {
    ml_loge("[NVRTC] %s failed: %s", what, nvrtcGetErrorString(e));
    return false;
  }
  return true;
}

} // namespace nntrainer::cuda

#endif // __CUDA_COMMON_H__
