// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 NNtrainer contributors
 *
 * @file   cl_compute_ops.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL ComputeOps table with BLAS-standard wrapper functions.
 *         Each wrapper adapts the GPU-specific sgemm_cl/sgemv_cl/dot_cl
 *         signatures to the standard ComputeOps function pointer signatures.
 */

#include <blas_kernels.h>
#include <clblast_interface.h>
#include <compute_ops.h>

namespace nntrainer {

// =========================================================================
// Wrapper functions: adapt GPU signatures to BLAS-standard ComputeOps
// =========================================================================

/// sgemm wrapper: BLAS standard → sgemm_cl
static void cl_sgemm_fp32(const unsigned int /*TStorageOrder*/, bool TransA,
                           bool TransB, const unsigned int M,
                           const unsigned int N, const unsigned int K,
                           const float /*alpha*/, const float *A,
                           const unsigned int lda, const float *B,
                           const unsigned int ldb, const float /*beta*/,
                           float *C, const unsigned int ldc) {
  /// @note sgemm_cl ignores alpha/beta/StorageOrder (always row-major, 1.0, 0.0)
  sgemm_cl(TransA, TransB, A, B, C, M, N, K, lda, ldb, ldc);
}

/// sgemv wrapper: BLAS standard → gemv_cl (CLBlast)
static void cl_sgemv_fp32(const unsigned int /*TStorageOrder*/, bool TransA,
                           const unsigned int M, const unsigned int N,
                           const float alpha, const float *A,
                           const unsigned int lda, const float *X,
                           const unsigned int incX, const float beta, float *Y,
                           const unsigned int incY) {
  gemv_cl(0, TransA, M, N, alpha, A, lda, X, beta, Y, incX, incY);
}

/// sdot wrapper: BLAS standard → dot_cl (CLBlast)
static float cl_sdot_fp32(const unsigned int N, const float *X,
                           const unsigned int incX, const float *Y,
                           const unsigned int incY) {
  return dot_cl(N, X, Y, incX, incY);
}

/// saxpy wrapper
static void cl_saxpy_fp32(const unsigned int N, const float alpha,
                           const float *X, const unsigned int incX, float *Y,
                           const unsigned int incY) {
  axpy_cl(N, alpha, X, Y);
}

/// sscal wrapper
static void cl_sscal_fp32(const unsigned int N, const float alpha, float *X,
                           const unsigned int /*incX*/) {
  scal_cl(N, alpha, X);
}

/// copy wrapper
static void cl_scopy_fp32(const unsigned int N, const float *X,
                           const unsigned int /*incX*/, float *Y,
                           const unsigned int /*incY*/) {
  copy_cl(N, X, Y);
}

// =========================================================================
// OpenCL ComputeOps table
// =========================================================================

static ComputeOps opencl_ops = {
  // FP32 BLAS
  .sgemm_fp32 = cl_sgemm_fp32,
  .sgemv_fp32 = cl_sgemv_fp32,
  .sdot_fp32 = cl_sdot_fp32,
  .saxpy_fp32 = cl_saxpy_fp32,
  .scopy_fp32 = cl_scopy_fp32,
  .sscal_fp32 = cl_sscal_fp32,
  .snrm2_fp32 = nullptr, // TODO: implement via nrm2_cl
  .isamax_fp32 = nullptr, // TODO: implement via amax_cl

  // FP32 element-wise — TODO: implement via addition_cl etc.
  .ele_mul_fp32 = nullptr,
  .ele_add_fp32 = nullptr,
  .ele_sub_fp32 = nullptr,
  .ele_div_fp32 = nullptr,

  // FP32 activation / special — nullptr (handled at layer level)
  .swiglu_fp32 = nullptr,
  .swiglu_alpha_fp32 = nullptr,
  .tanh_gelu_fp32 = nullptr,
  .tanh_gelu_v2_fp32 = nullptr,
  .tanh_gelu_mul_fp32 = nullptr,
  .tanh_gelu_v2_mul_fp32 = nullptr,
  .max_val_fp32 = nullptr,
  .softmax_fp32 = nullptr,
  .is_valid_fp32 = nullptr,

  // FP32 matrix ops
  .transpose_matrix_fp32 = nullptr, // TODO: implement via transpose_cl_axis

  // FP32 data conversion
  .scopy_u8 = nullptr,
  .scopy_s8 = nullptr,
  .scopy_int4_to_float32 = nullptr,
  .copy_s16_fp32 = nullptr,
  .copy_u16_fp32 = nullptr,
  .copy_fp32_u32 = nullptr,
  .copy_fp32_u16 = nullptr,
  .copy_fp32_u8 = nullptr,
  .copy_fp32_s16 = nullptr,
  .copy_fp32_s8 = nullptr,

  // Quantized GEMM
  .gemm_q4_0_fp32 = nullptr, // handled via gemm_q4_0_accel_fp32 below
  .gemm_q4_K_fp32 = nullptr,
  .gemm_q6_K_fp32 = nullptr,

  // Quantized weight packing
  .unpack_q4_0 = nullptr,
  .unpack_q4_0x8_transpose16 = nullptr,

  // GPU-accelerated quantized ops
  .gemm_q4_0_batch_fp32 = gemm_q4_0_async_cl,
  .gemm_q4_0_accel_fp32 = gemm_q4_0_cl,
  .gemv_int4_batch_fp32 = nullptr, // FP32 overload not available
  .gemm_int4_batch_fp32 = gemm_int4_async_cl,
  .gemv_int4_accel_fp32 = nullptr, // FP32 overload
  .sgemm_int4_accel_fp32 = sgemm_int4_cl,

#ifdef ENABLE_FP16
  // FP16 BLAS — TODO: wrap FP16 OpenCL kernels
  .sgemm_fp16 = nullptr,
  .sgemv_fp16 = nullptr,
  .sdot_fp16 = nullptr,
  .saxpy_fp16 = nullptr,
  .scopy_fp16 = nullptr,
  .scopy_fp32_to_fp16 = nullptr,
  .scopy_fp16_to_fp32 = nullptr,
  .sscal_fp16 = nullptr,
  .snrm2_fp16 = nullptr,
  .isamax_fp16 = nullptr,

  .ele_mul_fp16 = nullptr,
  .ele_add_fp16 = nullptr,
  .ele_sub_fp16 = nullptr,
  .ele_div_fp16 = nullptr,

  .swiglu_fp16 = nullptr,
  .max_val_fp16 = nullptr,
  .softmax_fp16 = nullptr,
  .is_valid_fp16 = nullptr,
  .inv_sqrt_inplace_fp16 = nullptr,

  .transpose_matrix_fp16 = nullptr,

  .scopy_int4_to_float16 = nullptr,
  .scopy_int8_to_float16_u = nullptr,
  .scopy_int8_to_float16_s = nullptr,

  .shgemm = nullptr,
  .shgemv = nullptr,
  .hsgemm = nullptr,
  .hsgemv = nullptr,

  .gemm_q4_0_fp16 = nullptr,
  .gemm_q6_K_fp16 = nullptr,

  .compute_rotary_embedding_value = nullptr,
#endif
};

ComputeOps *get_opencl_ops() { return &opencl_ops; }

} // namespace nntrainer
