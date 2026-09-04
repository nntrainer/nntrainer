// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gelu_cl_op.cpp
 * @date   28 July 2026
 * @brief  OpenCL GELU whole-op kernel dispatch (elementwise activation).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The kernel-argument binding lives in this dedicated whole-op file rather
 * than in blas_kernels.cpp, which is the BLAS/GEMM file.
 */

#include "gelu_cl_op.h"

#include <stdexcept>

#include <cl_kernels/gelu.h>
#include <engine.h> // Engine::Global().getRegisteredContext("gpu")
#include <nntrainer_log.h>
#include <tensor.h>
#ifdef ENABLE_FP16
#include <cl_kernels/gelu_fp16.h>
#endif

namespace nntrainer {

namespace {

enum Kernels { GELU_CL, GELU_CL_FP16 }; /** kernels enum */

/**
 * @brief kernel objects registered for this op
 */
std::vector<ClContext::SharedPtrClKernel> &getOpKernelPtrs() {
  static std::vector<ClContext::SharedPtrClKernel> op_kernel_ptrs;
  return op_kernel_ptrs;
}

// Elementwise GELU: one work item per element, no per-row reduction. Operands
// are input (argument 0) and output (argument 1); arguments 2 and 3 are the
// mode and the element count. Device-visible pointers bind directly; host
// memory is bounced through the shared staging buffers.
template <typename T = float>
void gelu_cl_internal(ClContext::SharedPtrClKernel kernel, const T *input,
                      T *output, int mode, unsigned int num_elems,
                      bool use_svm) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  size_t size = (size_t)num_elems * sizeof(T);

  if (use_svm) {
    if (!kernel->SetKernelSVMArguments(0, input)) {
      return;
    }
    if (!kernel->SetKernelSVMArguments(1, output)) {
      return;
    }
  } else {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getInBufferA()->WriteDataRegion(
          blas_cc->command_queue_inst_, size, input)) {
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
  }

  if (!kernel->SetKernelArguments(2, &mode, sizeof(int))) {
    return;
  }
  if (!kernel->SetKernelArguments(3, &num_elems, sizeof(int))) {
    return;
  }

  // Plain 1-D global range; the kernel's `if (i >= N) return;` guards the
  // rounded-up tail work-items.
  const int gelu_lws = 64;
  const int gelu_gws = (((int)num_elems + gelu_lws - 1) / gelu_lws) * gelu_lws;
  const int work_groups_count[3] = {gelu_gws, 1, 1};
  const int work_group_size[3] = {gelu_lws, 1, 1};

  if (!blas_cc->command_queue_inst_.DispatchCommand(kernel, work_groups_count,
                                                    work_group_size)) {
    return;
  }

  if (!use_svm) {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
          blas_cc->command_queue_inst_, size, output)) {
      return;
    }
  } else {
    blas_cc->command_queue_inst_.enqueueSVMMap(output, size, false);
  }
}

} // namespace

bool registerGeluClKernels(ClContext &cl_context) {
  auto &op_kernel_ptrs = getOpKernelPtrs();

  // check if the kernels are already registered.
  if (!op_kernel_ptrs.empty()) {
    ml_loge("kernels for gelu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      cl_context.registerClKernel(gelu_kernel, "gelu_cl");

    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register gelu_cl kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(kernel_ptr);

#ifdef ENABLE_FP16
    kernel_ptr = cl_context.registerClKernel(gelu_fp16_kernel, "gelu_cl_fp16");

    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register gelu_cl_fp16 kernel");
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

void gelu_cl_op(const Tensor &in, Tensor &out, int mode,
                unsigned int active_rows, unsigned int row_offset) {
  if (active_rows == 0)
    return;

  if (getOpKernelPtrs().empty())
    throw std::runtime_error(
      "gelu_cl_op: the OpenCL kernels are not registered (kernel build "
      "failed at ClContext init)");

  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  const unsigned int num_elems = active_rows * width;

  // Bind the pointer directly only when the tensor is device visible. GELU is
  // elementwise, so the row window is just a shifted base plus a count.
  const auto md = in.getMemoryData();
  const bool use_svm = md && md->isSVM();

  const auto dt = in.getDataType();
  if (dt == ml::train::TensorDim::DataType::FP32) {
    gelu_cl_internal<float>(
      getOpKernelPtrs()[Kernels::GELU_CL], in.getData<float>() + elem_off,
      out.getData<float>() + elem_off, mode, num_elems, use_svm);
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    gelu_cl_internal<_FP16>(
      getOpKernelPtrs()[Kernels::GELU_CL_FP16], in.getData<_FP16>() + elem_off,
      out.getData<_FP16>() + elem_off, mode, num_elems, use_svm);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument("gelu_cl_op: unsupported data type");
  }
}

} // namespace nntrainer
