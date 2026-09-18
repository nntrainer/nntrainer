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

  if (!kernel_sgemv_fp16_ptr) {
    return;
  }

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
  if (!kernel_sgemm_fp16_ptr) {
    return;
  }

  sgemm_cl_internal<_FP16>(kernel_sgemm_fp16_ptr, TransA, TransB, A, B, C, M, N,
                           K, lda, ldb, ldc, out_svm);
}

void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_addition_fp16_ptr =
    blas_cc->registerClKernel(addition_fp16_kernel, "addition_cl_fp16");
  if (!kernel_addition_fp16_ptr) {
    return;
  }

  addition_cl_internal<_FP16>(kernel_addition_fp16_ptr, input, res, size_input,
                              size_res);
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

static const std::string scalar_mul_fp16_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void scalar_mul_cl_fp16(__global const half *in, __global half *out,
                                 float s, int n, int row_off) {
  int i = get_global_id(0);
  if (i < n)
    out[i + row_off] = convert_half(convert_float(in[i + row_off]) * s);
}
)CL";

void scalar_mul_cl_fp16(const _FP16 *input, _FP16 *result, float scalar,
                        unsigned int n, bool use_svm, void *out_clmem,
                        void *in_clmem, unsigned int row_off) {
  // Both operands have to be device visible, because a plain host pointer is
  // not a kernel operand at all. Refusing here is the difference between
  // doing nothing and dispatching with arguments 0 and 1 unbound, which the
  // driver rejects as CL_INVALID_KERNEL_ARGS -- a failure that reads as a
  // defect in a kernel that registered and compiled perfectly well.
  if (!use_svm)
    return;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(scalar_mul_fp16_kernel, "scalar_mul_cl_fp16");
  if (!kp)
    return;

  cl_mem out_cl = static_cast<cl_mem>(out_clmem);
  cl_mem in_cl = static_cast<cl_mem>(in_clmem);
  const bool from_clmem = in_cl != nullptr;
  const bool to_clmem = out_cl != nullptr && use_svm;

  bool ok = true;
  if (from_clmem) {
    ok = ok && kp->SetKernelArguments(0, &in_cl, sizeof(cl_mem));
  } else if (use_svm) {
    blas_cc->command_queue_inst_.enqueueSVMUnmap(const_cast<_FP16 *>(input));
    ok = ok && kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input));
  }
  if (to_clmem) {
    ok = ok && kp->SetKernelArguments(1, &out_cl, sizeof(cl_mem));
  } else if (use_svm) {
    blas_cc->command_queue_inst_.enqueueSVMUnmap(result);
    ok = ok && kp->SetKernelSVMArguments(1, result);
  }
  int ni = (int)n;
  // A device sub-buffer is bound whole, so the row window is applied in the
  // kernel; SVM and host pointers are pre-offset by the caller, so row_off
  // stays 0 for them.
  int kern_row_off = (from_clmem && to_clmem) ? (int)row_off : 0;
  ok = ok && kp->SetKernelArguments(2, &scalar, sizeof(float));
  ok = ok && kp->SetKernelArguments(3, &ni, sizeof(int));
  ok = ok && kp->SetKernelArguments(4, &kern_row_off, sizeof(int));
  if (!ok)
    return;

  const int lws = 64;
  const int gws = ((ni + lws - 1) / lws) * lws;
  const int wgc[3] = {gws, 1, 1};
  const int wgs[3] = {lws, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wgc, wgs))
    return;

  if (!to_clmem && use_svm)
    blas_cc->command_queue_inst_.enqueueSVMMap(result,
                                               (size_t)n * sizeof(_FP16), true);
}

} // namespace nntrainer
