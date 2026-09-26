// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_gated_op.h
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Shared kernel-argument binding for the two-operand elementwise
 *         OpenCL whole-ops (GeGLU / SwiGLU).
 *
 * @details Both have the same kernel shape -- `(in1, in2) -> out`, one work
 * item per element, arguments 0/1/2 -- and the same residency question, so
 * they share one binder instead of a copy of it each. Only the kernel object
 * and the maths inside it differ.
 */

#ifndef __CL_GATED_OP_H__
#define __CL_GATED_OP_H__

#include <cl_context.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief The device buffer each operand of a two-operand elementwise op binds,
 *        or null for an operand that stays on the shared plane.
 *
 * @details A tensor the residency planner placed in device memory has no
 * host-visible copy of what its producer wrote: getData() returns a shared
 * plane shadow the device never touched. Binding that shadow reads stale
 * bytes, which is silent -- the kernel runs and writes a plausible result --
 * so each operand has to be bound on the plane the planner actually placed it
 * in. Mixing the two within one dispatch is valid and expected: a boundary
 * tensor is on one plane while the operand next to it is on the other.
 */
struct GatedClOperands {
  void *in1 = nullptr; /**< device buffer for operand 1, or null */
  void *in2 = nullptr; /**< device buffer for operand 2, or null */
  void *out = nullptr; /**< device buffer for the result, or null */

  /** @brief true when at least one operand is on the device plane */
  bool any() const {
    return in1 != nullptr || in2 != nullptr || out != nullptr;
  }
};

/**
 * @brief the device buffer @a t is placed in, or null when it is on the
 *        shared plane.
 * @param t tensor to ask
 * @return the cl_mem as void*, or nullptr
 */
inline void *gatedClOperand(const Tensor &t) {
  return t.isClMem() ? t.getClMem() : nullptr;
}

/**
 * @brief Bind and dispatch a two-operand elementwise OpenCL kernel.
 *
 * @param kernel kernel object, taking (in1, in2, out) at argument 0, 1, 2
 * @param in1 first operand (the gate for every current caller)
 * @param in2 second operand
 * @param out result, may alias neither operand
 * @param num_elems number of elements to process
 * @param use_svm bind the pointers directly (the operands are device visible)
 *                rather than bouncing them through the shared staging buffers
 * @param dev per-operand device buffers; an operand with a null entry is bound
 *            from its @a in1 / @a in2 / @a out pointer instead
 *
 * @note the row window is applied by the caller as a pointer offset: these
 * kernels are elementwise, so a window is just a shifted base plus a count.
 */
template <typename T>
void dispatchGatedClKernel(const ClContext::SharedPtrClKernel &kernel,
                           const T *in1, const T *in2, T *out,
                           unsigned int num_elems, bool use_svm,
                           const GatedClOperands &dev = GatedClOperands{}) {
  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &buffers = ClBufferManager::Global();

  const size_t bytes = (size_t)num_elems * sizeof(T);

  if (use_svm) {
    // Each operand is bound on the plane the planner placed it in. A
    // shared-memory one hands the device the very same pointer, after
    // releasing the host's mapping of it; a device-plane one binds its buffer
    // and needs no unmap at all -- its shared-plane shadow was never written,
    // and unmapping it would flip the host's SVM state one-sidedly.
    if ((dev.in1 == nullptr && !cl_context->command_queue_inst_.enqueueSVMUnmap(
                                 const_cast<T *>(in1))) ||
        (dev.in2 == nullptr && !cl_context->command_queue_inst_.enqueueSVMUnmap(
                                 const_cast<T *>(in2)))) {
      ml_loge("gated op: failed to unmap the SVM operands");
      return;
    }
    auto bind = [&kernel](unsigned int idx, void *device, const void *shared) {
      if (device != nullptr) {
        cl_mem buf = static_cast<cl_mem>(device);
        return kernel->SetKernelArguments(idx, &buf, sizeof(cl_mem));
      }
      return kernel->SetKernelSVMArguments(idx, shared);
    };
    if (!bind(0, dev.in1, in1) || !bind(1, dev.in2, in2) ||
        !bind(2, dev.out, out)) {
      ml_loge("gated op: failed to set the kernel arguments");
      return;
    }
  } else {
    // Host memory: the device cannot read it, so the operands go through the
    // shared staging buffers and the result comes back the same way.
    if (!buffers.getInBufferA()->WriteDataRegion(
          cl_context->command_queue_inst_, bytes, in1) ||
        !buffers.getInBufferB()->WriteDataRegion(
          cl_context->command_queue_inst_, bytes, in2)) {
      ml_loge("gated op: failed to stage the operands");
      return;
    }
    auto buffer_in_a = buffers.getInBufferA()->GetBuffer();
    auto buffer_in_b = buffers.getInBufferB()->GetBuffer();
    auto buffer_out_a = buffers.getOutBufferA()->GetBuffer();
    if (!kernel->SetKernelArguments(0, &buffer_in_a, sizeof(cl_mem)) ||
        !kernel->SetKernelArguments(1, &buffer_in_b, sizeof(cl_mem)) ||
        !kernel->SetKernelArguments(2, &buffer_out_a, sizeof(cl_mem))) {
      ml_loge("gated op: failed to set the buffer kernel arguments");
      return;
    }
  }

  const int elems = (int)num_elems;
  // The local size cannot exceed the global one.
  const int desired_local = 64;
  const int local = elems >= desired_local ? desired_local : elems;
  const int work_groups_count[3] = {elems, 1, 1};
  const int work_group_size[3] = {local, 1, 1};

  if (!cl_context->command_queue_inst_.DispatchCommand(
        kernel, work_groups_count, work_group_size)) {
    ml_loge("gated op: failed to dispatch the kernel");
    return;
  }

  if (use_svm) {
    // Only a shared-plane result comes back to the host; a device-plane one is
    // read by the next kernel from the buffer it was just written to.
    if (dev.out == nullptr &&
        !cl_context->command_queue_inst_.enqueueSVMMap(out, bytes, true))
      ml_loge("gated op: failed to map the SVM result");
  } else if (!buffers.getOutBufferA()->ReadDataRegion(
               cl_context->command_queue_inst_, bytes, out)) {
    ml_loge("gated op: failed to read the result back");
  }
}

} // namespace nntrainer

#endif // __CL_GATED_OP_H__
