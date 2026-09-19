// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels_fp16.cpp
 * @date	29 May 2024
 * @brief	Common blas OpenCL fp16 kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "blas_kernels_templates.h"
#include <cl_kernels/cl_kernels.h>

namespace nntrainer {

void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda, bool out_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemv_fp16_ptr;
  if (TransA) {
    kernel_sgemv_fp16_ptr =
      blas_cc->registerClKernel(hgemv_kernel, "sgemv_cl_fp16");
  } else {
    kernel_sgemv_fp16_ptr =
      blas_cc->registerClKernel(hgemv_no_trans_kernel, "sgemv_cl_noTrans_fp16");
  }

  NNTR_CL_BLAS_REQUIRE(kernel_sgemv_fp16_ptr, "sgemv_cl<fp16>",
                       "kernel register", dim1, dim2, lda);

  sgemv_cl_internal<_FP16>(kernel_sgemv_fp16_ptr, matAdata, vecXdata, vecYdata,
                           dim1, dim2, lda, out_svm);
}

_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_dot_fp16_ptr =
    blas_cc->registerClKernel(dot_fp16_kernel, "dot_cl_fp16");

  if (!kernel_dot_fp16_ptr) {
    return {};
  }

  return dot_cl_internal<_FP16>(kernel_dot_fp16_ptr, vecAdata, vecXdata, dim1);
}

void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              bool out_svm) {
  std::string kernel_func_;
  std::string sgemm_cl_kernel_fp16_;
  if (!TransA && !TransB) {
    kernel_func_ = "sgemm_cl_noTrans_fp16";
    sgemm_cl_kernel_fp16_ = hgemm_no_trans_kernel;
  } else if (TransA && !TransB) {
    kernel_func_ = "sgemm_cl_transA_fp16";
    sgemm_cl_kernel_fp16_ = hgemm_trans_a_kernel;
  } else if (!TransA && TransB) {
    kernel_func_ = "sgemm_cl_transB_fp16";
    sgemm_cl_kernel_fp16_ = hgemm_trans_b_kernel;
  } else {
    kernel_func_ = "sgemm_cl_transAB_fp16";
    sgemm_cl_kernel_fp16_ = hgemm_trans_ab_kernel;
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_sgemm_fp16_ptr =
    blas_cc->registerClKernel(sgemm_cl_kernel_fp16_, kernel_func_);
  NNTR_CL_BLAS_REQUIRE(kernel_sgemm_fp16_ptr, "sgemm_cl<fp16>",
                       "kernel register", M, N, K);

  sgemm_cl_internal<_FP16>(kernel_sgemm_fp16_ptr, TransA, TransB, A, B, C, M, N,
                           K, lda, ldb, ldc, out_svm);
}

void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res, bool use_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_addition_fp16_ptr =
    blas_cc->registerClKernel(addition_fp16_kernel, "addition_cl_fp16");
  if (!kernel_addition_fp16_ptr) {
    return;
  }

  addition_cl_internal<_FP16>(kernel_addition_fp16_ptr, input, res, size_input,
                              size_res, use_svm);
}

void sscal_cl(_FP16 *X, const unsigned int N, const float alpha) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  ClContext::SharedPtrClKernel kernel_sscal_fp16_ptr =
    blas_cc->registerClKernel(hscal_kernel, "sscal_cl_fp16");

  if (!kernel_sscal_fp16_ptr) {
    return;
  }

  sscal_cl_internal<_FP16>(kernel_sscal_fp16_ptr, X, N, alpha);
}

void transpose_cl_axis(const _FP16 *in, _FP16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_transpose_fp_16_ptr;
  switch (axis) {
  case 0:
    kernel_transpose_fp_16_ptr = blas_cc->registerClKernel(
      transpose_axis_0_fp16_kernel, "transpose_cl_fp16_axis0");
    break;
  case 1:
    kernel_transpose_fp_16_ptr = blas_cc->registerClKernel(
      transpose_axis_1_fp16_kernel, "transpose_cl_fp16_axis1");
    break;
  case 2:
    kernel_transpose_fp_16_ptr = blas_cc->registerClKernel(
      transpose_axis_2_fp16_kernel, "transpose_cl_fp16_axis2");
    break;
  default:
    throw std::invalid_argument("failed to register CL kernel");
    break;
  }

  if (!kernel_transpose_fp_16_ptr) {
    return;
  }

  transpose_cl_axis_internal<_FP16>(kernel_transpose_fp_16_ptr, in, res,
                                    input_batch_size, input_channels,
                                    input_height, input_width, axis);
}
} // namespace nntrainer
