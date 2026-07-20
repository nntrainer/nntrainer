// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_blas_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Owns the process-lifetime cuBLAS handle, bound to the CUDA backend
 *          stream. Kept separate from StreamManager so cublas_v2.h stays out of
 *          the core runtime headers. Consumed by cuda_fc_qint4.cpp via
 *          CudaComputeOps::fc.
 */

#ifndef __CUDA_BLAS_MANAGER_H__
#define __CUDA_BLAS_MANAGER_H__

#include <cublas_v2.h>

#include "singleton.h"

namespace nntrainer::cuda {

/**
 * @class BlasManager
 * @brief Singleton cuBLAS handle bound to the backend stream.
 */
class BlasManager : public Singleton<BlasManager> {
public:
  /**
   * @brief raw cuBLAS handle (nullptr if init failed)
   */
  cublasHandle_t handle() const { return handle_; }

  /**
   * @brief Row-major INT8 IMMA GEMM: C[M,N] = A[M,K] * B[K,N], int8 in /
   *        int32 out, on the Tensor Cores (cublasGemmEx CUDA_R_8I /
   *        CUBLAS_COMPUTE_32I). A = per-row int8 activation [M,K], B = int8
   *        weight [K,N] (the int4 weight unpacked once into int8). The int32
   *        accumulate is exact, so C is bit-identical to the __dp4a path; the
   *        dequant (act scale/zp + weight scale) is applied in a separate
   *        epilogue. This is the w4a8 Tensor-Core prefill FC (dp4a is the int
   *        ALU fallback, ~10x slower).
   *
   * @return true on CUBLAS_STATUS_SUCCESS
   */
  bool igemmRowMajor(int M, int N, int K, const signed char *A,
                     const signed char *B, int *C);

  /**
   * @brief Destroy the cuBLAS handle.
   */
  ~BlasManager() override;

protected:
  /**
   * @brief Singleton hook: create the handle and bind it to the backend stream.
   */
  void initialize() noexcept override;

private:
  cublasHandle_t handle_{nullptr};
  bool ok_{false};
};

} // namespace nntrainer::cuda

#endif // __CUDA_BLAS_MANAGER_H__
