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

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <blas_kernel_interface.h>
#include <cl_gated_op.h>
#include <cl_kernels/geglu.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <opencl_kernel.h>
#include <tensor.h>
#ifdef ENABLE_FP16
#include <cl_kernels/geglu_fp16.h>
#include <cl_kernels/geglu_q_fp16.h>
#endif

namespace nntrainer {

namespace {

enum Kernels {
  GEGLU_CL,
  GEGLU_ERF_CL,
  GEGLU_CL_FP16,
  GEGLU_ERF_CL_FP16
}; /**< kernels enum */

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

    kernel_ptr = cl_context.registerClKernel(geglu_kernel, "geglu_erf_cl");
    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_erf_cl kernel");
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

    kernel_ptr =
      cl_context.registerClKernel(geglu_fp16_kernel, "geglu_erf_cl_fp16");
    if (!kernel_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_erf_cl_fp16 kernel");
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

namespace {

#ifdef ENABLE_FP16
/**
 * @brief NNTR_FUSE_GEGLU_QUANT=0 keeps the plain GeGLU kernel on every call
 *        (an A/B switch: the fused kernel writes the same bytes). Read once.
 */
bool gegluQuantOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_FUSE_GEGLU_QUANT");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

/**
 * @brief The decode-shaped FP16 GeGLU whose row feeds a v8c FC (the MLP down
 *        projection): one kernel writes the GeGLU row and the int8
 *        quantisation the FC would compute from it, and publishes that
 *        through the norm->FC handoff, so the FC skips its own quant launch.
 *
 * Same contract as the fused norm (rmsnorm_cl_fp16): v8cNormQuantBegin admits
 * only few-row calls and stops fusing at a site whose quantisation nothing
 * claims, and the FC takes the handoff only when the write log says the row
 * is still the one quantised. All three operands must be on the device plane
 * (the FC's handoff test is keyed by the device buffer it reads).
 *
 * @return true when the fused kernel was enqueued (the GeGLU is done)
 */
bool gegluQuantHandoff(const GatedClOperands &dev, const _FP16 *in1,
                       const _FP16 *in2, _FP16 *out, unsigned int rows,
                       unsigned int width, bool exact) {
  // The v8c quantiser the FC would run reads K % 64 == 0 rows; the fused
  // kernel mirrors that one, so other widths keep the plain GeGLU.
  if (!gegluQuantOn() || dev.in1 == nullptr || dev.in2 == nullptr ||
      dev.out == nullptr || rows == 0 || width % 64 != 0)
    return false;
  // A site key per GeGLU form: the FC that claims the handoff is what keeps
  // the fusion on, and every layer's GeGLU feeds its down projection alike.
  static const char site_tanh = 0, site_erf = 0;
  void *q_i8 = nullptr, *q_scale = nullptr, *q_zp = nullptr, *q_rs = nullptr;
  if (!v8cNormQuantBegin(exact ? &site_erf : &site_tanh, rows, width, &q_i8,
                         &q_scale, &q_zp, &q_rs))
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kq = cc->registerClKernel(
    geglu_q_fp16_kernel, exact ? "geglu_erf_cl_fp16_q" : "geglu_cl_fp16_q");
  cl_mem a = static_cast<cl_mem>(dev.in1), b = static_cast<cl_mem>(dev.in2),
         o = static_cast<cl_mem>(dev.out), i8 = static_cast<cl_mem>(q_i8),
         sca = static_cast<cl_mem>(q_scale), zpb = static_cast<cl_mem>(q_zp),
         rsb = static_cast<cl_mem>(q_rs);
  const int n_rows = (int)rows, w = (int)width;
  bool ok = kq != nullptr && kq->SetKernelArguments(0, &a, sizeof(cl_mem)) &&
            kq->SetKernelArguments(1, &b, sizeof(cl_mem)) &&
            kq->SetKernelArguments(2, &o, sizeof(cl_mem)) &&
            kq->SetKernelArguments(3, &n_rows, sizeof(int)) &&
            kq->SetKernelArguments(4, &w, sizeof(int)) &&
            kq->SetKernelArguments(5, &i8, sizeof(cl_mem)) &&
            kq->SetKernelArguments(6, &sca, sizeof(cl_mem)) &&
            kq->SetKernelArguments(7, &zpb, sizeof(cl_mem)) &&
            kq->SetKernelArguments(8, &rsb, sizeof(cl_mem));
  // One work-group per row, as the fused norm: at one row the group is the
  // dispatch, and 256 lanes is the width the FC's own decode quantiser uses.
  constexpr int GEGLUQ_LWS = 256;
  const int gws[3] = {GEGLUQ_LWS * n_rows, 1, 1};
  const int lws[3] = {GEGLUQ_LWS, 1, 1};
  if (ok) {
    // The GeGLU row is the one graph tensor this writes; the int8 scratch is
    // the FC's own.
    opencl::Kernel::noteDispatchWrites(dev.out);
    ok = cc->command_queue_inst_.DispatchCommand(kq, gws, lws);
  }
  if (!ok) {
    v8cNormQuantAbort();
    return false;
  }
  v8cNormQuantCommit(dev.out, rows, width);
  return true;
}
#endif

void geglu_cl_dispatch(const Tensor &in1, const Tensor &in2, Tensor &out,
                       unsigned int active_rows, unsigned int row_offset,
                       bool exact) {
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

  // Bind whatever the residency planner placed in device memory on that
  // plane. Without this an operand the producing FC wrote into its device
  // buffer is read from a shared-plane shadow nothing ever wrote -- silently,
  // since the kernel still runs and still produces a number.
  const GatedClOperands dev{gatedClOperand(in1), gatedClOperand(in2),
                            gatedClOperand(out)};
  // A device buffer covers the WHOLE tensor and is bound from its base, so a
  // row window carried as a pointer offset cannot be expressed on it. Every
  // caller passes row_offset 0 (see geglu_layer.cpp); fail loudly rather than
  // process the wrong rows if that ever changes.
  if (dev.any() && elem_off != 0)
    throw std::runtime_error("geglu_cl_op: a device-plane operand cannot be "
                             "windowed by a row offset");

  const auto dt = in1.getDataType();
  if (dt == ml::train::TensorDim::DataType::FP32) {
    dispatchGatedClKernel<float>(
      getOpKernelPtrs()[exact ? Kernels::GEGLU_ERF_CL : Kernels::GEGLU_CL],
      in1.getData<float>() + elem_off, in2.getData<float>() + elem_off,
      out.getData<float>() + elem_off, num_elems, use_svm, dev);
  } else if (dt == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (use_svm &&
        gegluQuantHandoff(dev, in1.getData<_FP16>(), in2.getData<_FP16>(),
                          out.getData<_FP16>(), active_rows, width, exact))
      return;
    dispatchGatedClKernel<_FP16>(
      getOpKernelPtrs()[exact ? Kernels::GEGLU_ERF_CL_FP16
                              : Kernels::GEGLU_CL_FP16],
      in1.getData<_FP16>() + elem_off, in2.getData<_FP16>() + elem_off,
      out.getData<_FP16>() + elem_off, num_elems, use_svm, dev);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument("geglu_cl_op: unsupported data type");
  }
}

} // namespace

void geglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                 unsigned int active_rows, unsigned int row_offset) {
  geglu_cl_dispatch(in1, in2, out, active_rows, row_offset, false);
}

void geglu_erf_cl_op(const Tensor &in1, const Tensor &in2, Tensor &out,
                     unsigned int active_rows, unsigned int row_offset) {
  geglu_cl_dispatch(in1, in2, out, active_rows, row_offset, true);
}

} // namespace nntrainer
