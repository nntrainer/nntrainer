// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding_pool_cl_op.cpp
 * @date   28 July 2026
 * @brief  OpenCL whole-op dispatch for the sentence-embedding pooling /
 *         normalize tail (ComputeOps::mean_rows / l2_normalize_rows).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Structure mirrors swiglu_cl_op.cpp, the whole-op dispatch template: a
 * register*ClKernels() entry point plus free functions that the ClComputeOps
 * subclass forwards to, with the three residency shapes the CL backend can be
 * handed -- planner cl_mem, SVM, or a plain host pointer.
 *
 * NOTE on the host-bounce path: the read-back destination is always a host
 * heap pointer, never a coarse-grain SVM pointer. Reading a staging buffer
 * back into an SVM destination returns CL_SUCCESS but never lands in the host
 * view on some drivers (the defect behind the FP32 GEMM divergence), so the
 * SVM case is handled by binding SVM kernel arguments instead of bouncing.
 */

#include "embedding_pool_cl_op.h"

#include <vector>

#include <cl_kernels/embedding_pool.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <tensor.h>

namespace nntrainer {

namespace {

enum Kernels { L2_NORMALIZE_ROWS_CL, MEAN_ROWS_CL };

std::vector<ClContext::SharedPtrClKernel> &getOpKernelPtrs() {
  static std::vector<ClContext::SharedPtrClKernel> op_kernel_ptrs;
  return op_kernel_ptrs;
}

/** local work-group size for the cooperative row reduction. Must be a power of
 *  two: the kernel's tree reduction halves the stride each step. */
constexpr int kLocalSize = 64;

/** true when the tensor's memory is SVM (device-visible host pointer). */
bool isSvmTensor(const Tensor &t) {
  const auto md = t.getMemoryData();
  return md && md->isSVM();
}

} // namespace

bool registerEmbeddingPoolClKernels(ClContext &cl_context) {
  auto &op_kernel_ptrs = getOpKernelPtrs();

  if (!op_kernel_ptrs.empty()) {
    ml_loge("kernels for embedding_pool are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel k = cl_context.registerClKernel(
      embedding_pool_kernel, "l2_normalize_rows_cl");
    if (!k) {
      ml_loge("OpenCL Error: Fail to register l2_normalize_rows_cl kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(k);

    k = cl_context.registerClKernel(embedding_pool_kernel, "mean_rows_cl");
    if (!k) {
      ml_loge("OpenCL Error: Fail to register mean_rows_cl kernel");
      break;
    }
    op_kernel_ptrs.emplace_back(k);

    return true;
  } while (false);

  op_kernel_ptrs.clear();
  return false;
}

void l2_normalize_rows_cl_op(const Tensor &in, Tensor &out, float epsilon) {
  if (in.getDataType() != ml::train::TensorDim::DataType::FP32)
    throw std::invalid_argument(
      "l2_normalize_rows_cl_op: only FP32 is supported");

  auto *cl_ctx =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuf = ClBufferManager::Global();

  const unsigned int W = in.width();
  if (W == 0)
    return;
  const int rows = (int)(in.size() / W);
  const size_t bytes = (size_t)rows * W * sizeof(float);

  const float *idata = in.getData();
  float *odata = out.getData();

  const auto &kernel = getOpKernelPtrs()[Kernels::L2_NORMALIZE_ROWS_CL];

  // Residency: device plane > SVM > host. getData() is only host-addressable
  // in the last case, which is why the host branch below has to stage through
  // the shared buffers instead of binding the caller's pointer -- and why a
  // tensor the planner put on the device plane has to bind its own sub-buffer
  // rather than a shared-plane pointer that shadows it.
  const bool svm = isSvmTensor(in);
  void *in_clmem = (svm && in.isClMem()) ? in.getClMem() : nullptr;
  void *out_clmem = (svm && out.isClMem()) ? out.getClMem() : nullptr;

  do {
    if (!svm) {
      if (!clbuf.getInBufferA()->WriteDataRegion(cl_ctx->command_queue_inst_,
                                                 bytes, idata))
        break;
      auto bufIn = clbuf.getInBufferA()->GetBuffer();
      auto bufOut = clbuf.getOutBufferA()->GetBuffer();
      if (!kernel->SetKernelArguments(0, &bufIn, sizeof(cl_mem)) ||
          !kernel->SetKernelArguments(1, &bufOut, sizeof(cl_mem)))
        break;
    } else {
      // Unmap the INPUT only, then bind the args and re-map the output after
      // the dispatch -- the exact sequence the gated ops use. A device-bound
      // operand must NOT be SVM-unmapped: its shared-plane shadow was never
      // written, so unmapping it would publish stale bytes.
      if (in_clmem == nullptr &&
          !cl_ctx->command_queue_inst_.enqueueSVMUnmap((void *)idata)) {
        ml_loge("l2_normalize_rows: failed to unmap svm input");
        break;
      }
      bool bound = true;
      if (in_clmem != nullptr) {
        cl_mem b = static_cast<cl_mem>(in_clmem);
        bound &= kernel->SetKernelArguments(0, &b, sizeof(cl_mem));
      } else {
        bound &= kernel->SetKernelSVMArguments(0, idata);
      }
      if (out_clmem != nullptr) {
        cl_mem b = static_cast<cl_mem>(out_clmem);
        bound &= kernel->SetKernelArguments(1, &b, sizeof(cl_mem));
      } else {
        bound &= kernel->SetKernelSVMArguments(1, odata);
      }
      if (!bound) {
        ml_loge("l2_normalize_rows: failed to set svm/clmem args");
        break;
      }
    }

    const int w_arg = (int)W;
    if (!kernel->SetKernelArguments(2, &epsilon, sizeof(float)) ||
        !kernel->SetKernelArguments(3, &w_arg, sizeof(int)) ||
        // __local scratch for the tree reduction
        !kernel->SetKernelArguments(4, nullptr, kLocalSize * sizeof(float)))
      break;

    const int work_groups_count[3] = {rows * kLocalSize, 1, 1};
    const int work_group_size[3] = {kLocalSize, 1, 1};
    if (!cl_ctx->command_queue_inst_.DispatchCommand(kernel, work_groups_count,
                                                     work_group_size)) {
      ml_loge("l2_normalize_rows: dispatch failed");
      break;
    }

    if (!svm) {
      if (!clbuf.getOutBufferA()->ReadDataRegion(cl_ctx->command_queue_inst_,
                                                 bytes, odata))
        break;
    } else if (out_clmem == nullptr &&
               !cl_ctx->command_queue_inst_.enqueueSVMMap(odata, bytes,
                                                          /*read_only=*/true)) {
      // A device-bound output stays device-owned; only a shared-plane one is
      // mapped back for its host reader.
      ml_loge("l2_normalize_rows: failed to map svm output");
      break;
    }
  } while (false);
}

void mean_rows_cl_op(const Tensor &in, Tensor &out, unsigned int active_rows,
                     unsigned int row_offset) {
  if (in.getDataType() != ml::train::TensorDim::DataType::FP32)
    throw std::invalid_argument("mean_rows_cl_op: only FP32 is supported");

  auto *cl_ctx =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuf = ClBufferManager::Global();

  const unsigned int W = in.width();
  if (W == 0 || active_rows == 0)
    return;

  // Row window: the kernel reads rows [0, active_rows) from the pointer it is
  // given, so fold row_offset into the base pointer.
  const size_t elem_off = (size_t)row_offset * W;
  const size_t in_bytes = (size_t)active_rows * W * sizeof(float);
  const size_t out_bytes = (size_t)W * sizeof(float);

  const float *idata = in.getData() + elem_off;
  float *odata = out.getData();

  const auto &kernel = getOpKernelPtrs()[Kernels::MEAN_ROWS_CL];

  const bool svm = isSvmTensor(in);
  // Bind a device sub-buffer only for an offset-0 view: getClMem() hands back
  // the whole buffer with no offset applied, so a windowed read has to stay on
  // the shared plane, where the offset rides on the pointer.
  void *in_clmem =
    (svm && elem_off == 0 && in.isClMem()) ? in.getClMem() : nullptr;
  void *out_clmem = (svm && out.isClMem()) ? out.getClMem() : nullptr;

  do {
    if (!svm) {
      if (!clbuf.getInBufferA()->WriteDataRegion(cl_ctx->command_queue_inst_,
                                                 in_bytes, idata))
        break;
      auto bufIn = clbuf.getInBufferA()->GetBuffer();
      auto bufOut = clbuf.getOutBufferA()->GetBuffer();
      if (!kernel->SetKernelArguments(0, &bufIn, sizeof(cl_mem)) ||
          !kernel->SetKernelArguments(1, &bufOut, sizeof(cl_mem)))
        break;
    } else {
      // Input-only unmap, as in l2_normalize_rows_cl_op above (swiglu pattern).
      if (in_clmem == nullptr &&
          !cl_ctx->command_queue_inst_.enqueueSVMUnmap((void *)idata)) {
        ml_loge("mean_rows: failed to unmap svm input");
        break;
      }
      bool bound = true;
      if (in_clmem != nullptr) {
        cl_mem b = static_cast<cl_mem>(in_clmem);
        bound &= kernel->SetKernelArguments(0, &b, sizeof(cl_mem));
      } else {
        bound &= kernel->SetKernelSVMArguments(0, idata);
      }
      if (out_clmem != nullptr) {
        cl_mem b = static_cast<cl_mem>(out_clmem);
        bound &= kernel->SetKernelArguments(1, &b, sizeof(cl_mem));
      } else {
        bound &= kernel->SetKernelSVMArguments(1, odata);
      }
      if (!bound) {
        ml_loge("mean_rows: failed to set svm/clmem args");
        break;
      }
    }

    const int rows_arg = (int)active_rows;
    const int w_arg = (int)W;
    if (!kernel->SetKernelArguments(2, &rows_arg, sizeof(int)) ||
        !kernel->SetKernelArguments(3, &w_arg, sizeof(int)))
      break;

    // one work-item per output column, global rounded up to the local size
    const int global = (int)(((W + kLocalSize - 1) / kLocalSize) * kLocalSize);
    const int work_groups_count[3] = {global, 1, 1};
    const int work_group_size[3] = {kLocalSize, 1, 1};
    if (!cl_ctx->command_queue_inst_.DispatchCommand(kernel, work_groups_count,
                                                     work_group_size)) {
      ml_loge("mean_rows: dispatch failed");
      break;
    }

    if (!svm) {
      if (!clbuf.getOutBufferA()->ReadDataRegion(cl_ctx->command_queue_inst_,
                                                 out_bytes, odata))
        break;
    } else if (out_clmem == nullptr &&
               !cl_ctx->command_queue_inst_.enqueueSVMMap(odata, out_bytes,
                                                          /*read_only=*/true)) {
      ml_loge("mean_rows: failed to map svm output");
      break;
    }
  } while (false);
}

} // namespace nntrainer
