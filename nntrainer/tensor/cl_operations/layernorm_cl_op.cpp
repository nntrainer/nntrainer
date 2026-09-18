// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   layernorm_cl_op.cpp
 * @date   28 July 2026
 * @brief  OpenCL LayerNorm whole-op kernel dispatch.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The kernel-argument binding lives in this dedicated whole-op file rather
 * than in blas_kernels.cpp, which is the BLAS/GEMM file and only ever carried
 * a norm because rmsnorm_cl was written before the op table existed.
 */

#include "layernorm_cl_op.h"

#include <stdexcept>
#include <string>

#include <cl_kernels/layernorm.h>
#include <engine.h> // Engine::Global().getRegisteredContext("gpu")
#include <nntrainer_log.h>
#include <tensor.h>
#ifdef ENABLE_FP16
#include <cl_kernels/layernorm_fp16.h>
#endif

namespace nntrainer {

namespace {

enum Kernels { LAYERNORM_CL, LAYERNORM_CL_FP16 }; /** kernels enum */

/**
 * @brief kernel objects registered for this op
 */
std::vector<ClContext::SharedPtrClKernel> &getOpKernelPtrs() {
  static std::vector<ClContext::SharedPtrClKernel> op_kernel_ptrs;
  return op_kernel_ptrs;
}

// One workgroup (one sub-group) per row; the kernel collapses the two per-row
// reductions with sub_group_reduce_add. Kernel arg order: 0=input, 1=output,
// 2=gamma, 3=beta, 4=epsilon(float), 5=H(int), 6=W(int) -- shifted by one vs
// rmsnorm to make room for the beta operand (arg 3, an SVM/buffer arg like
// gamma). epsilon is always bound as a float (both the fp32 and fp16 kernels
// take a float epsilon).
//
// There is deliberately no row-count gate that falls back to a host loop: a
// tensor on this context may live in shared virtual memory that the host has
// unmapped, where a host loop is not merely slower but wrong.
template <typename T = float>
void layernorm_cl_internal(ClContext::SharedPtrClKernel kernel, const T *input,
                           const T *gamma, const T *beta, T *result,
                           const float epsilon, unsigned int height,
                           unsigned int width, const bool use_svm) {
  unsigned dim_in = height * width;
  unsigned dim_w = width;
  unsigned size_in = dim_in * sizeof(T);
  unsigned size_w = dim_w * sizeof(T);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  if (use_svm) {
    if (!kernel->SetKernelSVMArguments(0, input)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(1, result)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(2, gamma)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(3, beta)) {
      return;
    }
  } else {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getInBufferA()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_in, input)) {
      return;
    }
    if (!clbuffInstance.getInBufferB()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_w, gamma)) {
      return;
    }
    if (!clbuffInstance.getInBufferC()->WriteDataRegion(
          blas_cc->command_queue_inst_, size_w, beta)) {
      return;
    }

    if (!kernel->SetKernelArguments(
          0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          1, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          2, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
    if (!kernel->SetKernelArguments(
          3, &clbuffInstance.getInBufferC()->GetBuffer(), sizeof(cl_mem))) {
      return;
    }
  }

  if (!kernel->SetKernelArguments(4, &epsilon, sizeof(float))) {
    return;
  }
  if (!kernel->SetKernelArguments(5, &height, sizeof(int))) {
    return;
  }
  if (!kernel->SetKernelArguments(6, &width, sizeof(int))) {
    return;
  }
#ifdef __ANDROID__
  constexpr int SUBGROUP_SIZE = 64;
#else
  constexpr int SUBGROUP_SIZE = 32;
#endif
  const int work_groups_count[3] = {static_cast<int>(height) * SUBGROUP_SIZE, 1,
                                    1};

  const int work_group_size[3] = {SUBGROUP_SIZE, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel, work_groups_count,
                                                    work_group_size)) {
    return;
  }

  if (!use_svm) {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
          blas_cc->command_queue_inst_, size_in, result)) {
      return;
    }
  } else {
    blas_cc->command_queue_inst_.enqueueSVMMap(result, size_in, false);
  }
}

} // namespace

bool registerLayerNormClKernels(ClContext &cl_context) {
  auto &op_kernel_ptrs = getOpKernelPtrs();

  // check if the kernels are already registered.
  if (!op_kernel_ptrs.empty()) {
    ml_loge("kernels for layernorm_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      cl_context.registerClKernel(layernorm_kernel, "layernorm_cl");

    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register layernorm_cl kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(kernel_ptr);

#ifdef ENABLE_FP16
    kernel_ptr =
      cl_context.registerClKernel(layernorm_fp16_kernel, "layernorm_cl_fp16");

    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register layernorm_cl_fp16 kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(kernel_ptr);
#endif

    return true;
  } while (false);

  // clear all registered kernels if any error occurs during registration
  op_kernel_ptrs.clear();

  return false;
}

void layernorm_cl_op(const Tensor &in, Tensor &out, const Tensor &gamma,
                     const Tensor &beta, float epsilon,
                     unsigned int active_rows, unsigned int row_offset) {
  if (active_rows == 0)
    return;

  if (getOpKernelPtrs().empty())
    throw std::runtime_error(
      "layernorm_cl_op: the OpenCL kernels are not registered (kernel build "
      "failed at ClContext init)");

  const unsigned int width = in.width();
  // Element offset into the pointers so the kernel processes rows
  // [row_offset, row_offset + active_rows). Both planes are addressed by
  // pointer (SVM) or bounced through the shared host buffers, so the offset
  // rides on the pointer and the kernel's own H is the live-row count.
  const size_t elem_off = (size_t)row_offset * width;

  // Bind the pointers directly only when the operands are device visible. On
  // the default host pool getData() returns plain host pointers, and handing
  // one to clSetKernelArgSVMPointer produces garbage rather than an error.
  const auto md = in.getMemoryData();
  const bool use_svm = md && md->isSVM();

  const auto dt = in.getDataType();
  if (gamma.getDataType() != dt || beta.getDataType() != dt) {
    throw std::invalid_argument(
      "layernorm_cl_op: gamma/beta dtype must match the activation dtype on "
      "the gpu engine (the OpenCL kernels are single-dtype); use engine=cpu "
      "for a mixed activation/weight dtype LayerNorm");
  }

  if (dt == ml::train::TensorDim::DataType::FP32) {
    layernorm_cl_internal<float>(
      getOpKernelPtrs()[Kernels::LAYERNORM_CL], in.getData<float>() + elem_off,
      gamma.getData<float>(), beta.getData<float>(),
      out.getData<float>() + elem_off, epsilon, active_rows, width, use_svm);
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    layernorm_cl_internal<_FP16>(getOpKernelPtrs()[Kernels::LAYERNORM_CL_FP16],
                                 in.getData<_FP16>() + elem_off,
                                 gamma.getData<_FP16>(), beta.getData<_FP16>(),
                                 out.getData<_FP16>() + elem_off, epsilon,
                                 active_rows, width, use_svm);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument("layernorm_cl_op: unsupported data type");
  }
}

} // namespace nntrainer
