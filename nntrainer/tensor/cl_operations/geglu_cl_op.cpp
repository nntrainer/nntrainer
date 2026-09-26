// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_cl_op.cpp
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL GeGLU whole-op kernel dispatch (gelu_tanh(gate) * up).
 */

#include "geglu_cl_op.h"

#include <stdexcept>
#include <vector>

#include <cl_gated_op.h>
#include <cl_kernels/geglu.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <tensor.h>
#ifdef ENABLE_FP16
#include <cl_kernels/geglu_fp16.h>
#endif

namespace nntrainer {

namespace {

enum Kernels { GEGLU_CL, GEGLU_CL_FP16 }; /**< kernels enum */

/**
 * @brief kernel objects registered for this op
 */
std::vector<ClContext::SharedPtrClKernel> &getOpKernelPtrs() {
  static std::vector<ClContext::SharedPtrClKernel> op_kernel_ptrs;
  return op_kernel_ptrs;
}

} // namespace

bool registerGeGLUClKernels(ClContext &cl_context) {
  auto &op_kernel_ptrs = getOpKernelPtrs();

  if (!op_kernel_ptrs.empty()) {
    ml_loge("kernels for geglu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_ptr =
      cl_context.registerClKernel(geglu_kernel, "geglu_cl");
    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_cl kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(kernel_ptr);

#ifdef ENABLE_FP16
    kernel_ptr =
      cl_context.registerClKernel(geglu_fp16_kernel, "geglu_cl_fp16");
    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_cl_fp16 kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(kernel_ptr);
#endif

    return true;
  } while (false);

  // drop every kernel registered so far if any of them failed
  op_kernel_ptrs.clear();

  return false;
}

void geglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                 unsigned int active_rows, unsigned int row_offset) {
  if (active_rows == 0)
    return;

  if (getOpKernelPtrs().empty())
    throw std::runtime_error("geglu_cl_op: the OpenCL kernels are not "
                             "registered (the kernel build failed at "
                             "ClContext init)");

  const unsigned int width = in1.width();
  const size_t elem_off = (size_t)row_offset * width;
  const unsigned int num_elems = active_rows * width;

  // Bind the pointers directly only when the operands are device visible. On
  // the default host pool getData() returns plain host pointers, and handing
  // one to clSetKernelArgSVMPointer produces garbage rather than an error.
  const auto md = in1.getMemoryData();
  const bool use_svm = md && md->isSVM();

  const auto dt = in1.getDataType();
  if (dt == ml::train::TensorDim::DataType::FP32) {
    dispatchGatedClKernel<float>(
      getOpKernelPtrs()[Kernels::GEGLU_CL], in1.getData<float>() + elem_off,
      in2.getData<float>() + elem_off, out.getData<float>() + elem_off,
      num_elems, use_svm);
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    dispatchGatedClKernel<_FP16>(
      getOpKernelPtrs()[Kernels::GEGLU_CL_FP16],
      in1.getData<_FP16>() + elem_off, in2.getData<_FP16>() + elem_off,
      out.getData<_FP16>() + elem_off, num_elems, use_svm);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument("geglu_cl_op: unsupported data type");
  }
}

} // namespace nntrainer
