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
 * @brief Can a row-wise fp16 kernel read/write this row 4 halves at a time?
 *
 * The vectorized norm/quant kernels move a row as uint2 (4 halves, 8 bytes),
 * which needs the feature size to be a multiple of 4 (so every row starts on a
 * vector boundary too) and every buffer to be 8-byte aligned. Pool-suballocated
 * activations are not guaranteed either, so the caller picks the scalar kernel
 * when this is false rather than risking a misaligned access.
 *
 * @param width feature size (the vectorized dimension)
 * @param a,b,c row bases to check; nullptr entries are ignored
 */
inline bool cuda_vec4_rows_ok(unsigned int width, const void *a,
                              const void *b = nullptr,
                              const void *c = nullptr) {
  if ((width & 3u) != 0u)
    return false;
  auto aligned = [](const void *p) {
    return p == nullptr || ((reinterpret_cast<uintptr_t>(p) & 7u) == 0u);
  };
  return aligned(a) && aligned(b) && aligned(c);
}

/**
 * @brief Row count at or below which the vectorized row kernels are used.
 *
 * They carry the decoded row in registers to avoid a second global pass, which
 * is a clear win for the one-row decode launch and a clear LOSS at prefill row
 * counts, where the extra registers cut how many blocks an SM can hold and the
 * kernel is bandwidth-bound anyway (measured: ~5% prefill TPS). Same threshold
 * the device norm already uses to mean "decode-shaped".
 */
inline bool cuda_vec4_rows_small(unsigned int rows) { return rows <= 32u; }

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
