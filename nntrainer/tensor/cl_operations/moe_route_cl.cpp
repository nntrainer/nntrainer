// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   moe_route_cl.cpp
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL data movement for a mixture-of-experts layer (gather,
 *         weighted scatter-add, zero) and the expert-weight FC seam.
 */

#include "moe_route_cl.h"

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cl_context.h>
#include <compute_ops.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <opencl_loader.h>
#include <tensor.h>
#include <tq2cx_cl.h>

namespace nntrainer {

namespace {

// Typed as unsigned integers of the element width rather than float/half for
// the copies: nothing is computed, and vload/vstore on the integer type only
// asks for element alignment, which a pooled tensor base always has.
//
// The scatter-add widens to float, scales, adds and narrows back once per
// call, which is the rounding sequence of the host saxpy loop it replaces.
const std::string moe_route_kernels = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

__kernel void moe_gather_rows_f32(__global const uint *in, __global uint *out,
                                  __global const int *src_row,
                                  const int width, const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / width;
  const int j = gid - r * width;
  out[gid] = in[(long)src_row[r] * width + j];
}

__kernel void moe_gather_rows_f16(__global const ushort *in,
                                  __global ushort *out,
                                  __global const int *src_row,
                                  const int width, const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / width;
  const int j = gid - r * width;
  out[gid] = in[(long)src_row[r] * width + j];
}

__kernel void moe_scatter_add_rows_f32(__global const float *expert_out,
                                       __global float *out,
                                       __global const int *dst_row,
                                       __global const float *weight,
                                       const int width, const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / width;
  const int j = gid - r * width;
  const long dst = (long)dst_row[r] * width + j;
  out[dst] = out[dst] + weight[r] * expert_out[gid];
}

__kernel void moe_scatter_add_rows_f16(__global const half *expert_out,
                                       __global half *out,
                                       __global const int *dst_row,
                                       __global const float *weight,
                                       const int width, const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / width;
  const int j = gid - r * width;
  const long dst = (long)dst_row[r] * width + j;
  out[dst] = (half)((float)out[dst] + weight[r] * (float)expert_out[gid]);
}

__kernel void moe_zero_rows_u32(__global uint *out, const int total) {
  const int gid = get_global_id(0);
  if (gid < total) out[gid] = 0u;
}

__kernel void moe_zero_rows_u16(__global ushort *out, const int total) {
  const int gid = get_global_id(0);
  if (gid < total) out[gid] = (ushort)0;
}
)CL";

/**
 * @brief The OpenCL context, or null when no gpu context is registered.
 */
ClContext *moeClContext() {
  return static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
}

/**
 * @brief A tensor these kernels can bind: FP32/FP16 on the shared plane.
 */
bool moeSvmOperand(const Tensor &t) {
  const auto dt = t.getDataType();
  if (dt != ml::train::TensorDim::DataType::FP32 &&
      dt != ml::train::TensorDim::DataType::FP16)
    return false;
  if (t.isClMem())
    return false;
  const auto md = t.getMemoryData();
  return md && md->isSVM() && t.getData<uint8_t>() != nullptr;
}

size_t moeElemSize(const Tensor &t) {
  return t.getDataType() == ml::train::TensorDim::DataType::FP16
           ? sizeof(uint16_t)
           : sizeof(float);
}

unsigned int moeRows(const Tensor &t) {
  return t.batch() * t.channel() * t.height();
}

/**
 * @brief A small per-call operand, copied into device memory at create time,
 *        so the host array may go out of scope as soon as this returns.
 */
cl_mem moeUpload(ClContext *cc, const void *host, size_t bytes) {
  cl_int err = CL_SUCCESS;
  cl_mem b = opencl::clCreateBufferT(cc->context_inst_.GetContextNoRetain(),
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     bytes, const_cast<void *>(host), &err);
  return err == CL_SUCCESS ? b : nullptr;
}

/**
 * @brief Enqueue a 1-D dispatch over @a total work items.
 */
void moeDispatch(ClContext *cc, const ClContext::SharedPtrClKernel &kp,
                 size_t total) {
  constexpr size_t LWS = 64;
  std::array<size_t, 3> gws = {((total + LWS - 1) / LWS) * LWS, 1, 1};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                        lws.data());
}

/**
 * @brief Shared-plane hand-off around a dispatch, the protocol every SVM op in
 *        this directory follows: unmap each operand for the kernel, and map
 *        it back afterwards without waiting. A host reader maps it itself.
 */
void moeUnmap(ClContext *cc, const Tensor &t) {
  cc->command_queue_inst_.enqueueSVMUnmap(
    const_cast<uint8_t *>(t.getData<uint8_t>()));
}
void moeMapBack(ClContext *cc, const Tensor &t, size_t bytes, bool read_only) {
  cc->command_queue_inst_.enqueueSVMMap(
    const_cast<uint8_t *>(t.getData<uint8_t>()), bytes, read_only,
    /*event=*/nullptr, /*async=*/true);
}

std::mutex &expertFcMutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<int, MoeExpertFcFn> &expertFcTable() {
  // The expert-only formats this tree ships a device FC for. Anything else
  // registers through registerMoeExpertFcCl.
  static std::unordered_map<int, MoeExpertFcFn> table = {
    {static_cast<int>(ml::train::TensorDim::DataType::TQ2CX),
     &tq2cx_expert_fc_cl},
  };
  return table;
}

std::unordered_map<int, MoeExpertGroupFn> &expertGroupTable() {
  static std::unordered_map<int, MoeExpertGroupFn> table = {
    {static_cast<int>(ml::train::TensorDim::DataType::TQ2CX),
     &tq2cx_moe_experts_cl},
  };
  return table;
}

} // namespace

bool moe_gather_rows_cl(const Tensor &in, Tensor &out,
                        const unsigned int *src_rows, unsigned int rows) {
  if (rows == 0 || src_rows == nullptr || !moeSvmOperand(in) ||
      !moeSvmOperand(out) || in.getDataType() != out.getDataType() ||
      in.width() == 0 || in.width() != out.width() || moeRows(out) < rows)
    return false;
  const unsigned int in_rows = moeRows(in);
  for (unsigned int r = 0; r < rows; ++r)
    if (src_rows[r] >= in_rows)
      return false;

  ClContext *cc = moeClContext();
  if (cc == nullptr)
    return false;
  const bool fp16 = in.getDataType() == ml::train::TensorDim::DataType::FP16;
  auto kp = cc->registerClKernel(
    moe_route_kernels, fp16 ? "moe_gather_rows_f16" : "moe_gather_rows_f32");
  if (!kp)
    return false;

  std::vector<int> idx(src_rows, src_rows + rows);
  cl_mem idx_buf = moeUpload(cc, idx.data(), sizeof(int) * rows);
  if (idx_buf == nullptr)
    return false;

  const unsigned int width = in.width();
  const size_t total = (size_t)rows * width;
  const int wi = (int)width;
  const int ti = (int)total;
  moeUnmap(cc, in);
  moeUnmap(cc, out);
  bool ok = kp->SetKernelSVMArguments(0, in.getData<uint8_t>()) &&
            kp->SetKernelSVMArguments(1, out.getData<uint8_t>()) &&
            kp->SetKernelArguments(2, &idx_buf, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &wi, sizeof(int)) &&
            kp->SetKernelArguments(4, &ti, sizeof(int));
  if (ok)
    moeDispatch(cc, kp, total);
  // The enqueued command retains the buffer; this drops our reference only.
  opencl::clReleaseMemObjectT(idx_buf);

  const size_t elem = moeElemSize(in);
  moeMapBack(cc, in, (size_t)in_rows * width * elem, /*read_only=*/true);
  moeMapBack(cc, out, total * elem, /*read_only=*/false);
  if (!ok)
    ml_loge("moe_gather_rows_cl: failed to bind the kernel arguments");
  return ok;
}

bool moe_scatter_add_rows_cl(const Tensor &expert_out, Tensor &out,
                             const unsigned int *dst_rows, const float *weights,
                             unsigned int rows) {
  if (rows == 0 || dst_rows == nullptr || weights == nullptr ||
      !moeSvmOperand(expert_out) || !moeSvmOperand(out) ||
      expert_out.getDataType() != out.getDataType() || out.width() == 0 ||
      expert_out.width() != out.width() || moeRows(expert_out) < rows)
    return false;
  const unsigned int out_rows = moeRows(out);
  for (unsigned int r = 0; r < rows; ++r)
    if (dst_rows[r] >= out_rows)
      return false;

  ClContext *cc = moeClContext();
  if (cc == nullptr)
    return false;
  const bool fp16 = out.getDataType() == ml::train::TensorDim::DataType::FP16;
  auto kp =
    cc->registerClKernel(moe_route_kernels, fp16 ? "moe_scatter_add_rows_f16"
                                                 : "moe_scatter_add_rows_f32");
  if (!kp)
    return false;

  std::vector<int> idx(dst_rows, dst_rows + rows);
  cl_mem idx_buf = moeUpload(cc, idx.data(), sizeof(int) * rows);
  cl_mem w_buf = moeUpload(cc, weights, sizeof(float) * rows);
  if (idx_buf == nullptr || w_buf == nullptr) {
    if (idx_buf != nullptr)
      opencl::clReleaseMemObjectT(idx_buf);
    if (w_buf != nullptr)
      opencl::clReleaseMemObjectT(w_buf);
    return false;
  }

  const unsigned int width = out.width();
  const size_t total = (size_t)rows * width;
  const int wi = (int)width;
  const int ti = (int)total;
  moeUnmap(cc, expert_out);
  moeUnmap(cc, out);
  bool ok = kp->SetKernelSVMArguments(0, expert_out.getData<uint8_t>()) &&
            kp->SetKernelSVMArguments(1, out.getData<uint8_t>()) &&
            kp->SetKernelArguments(2, &idx_buf, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &w_buf, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &wi, sizeof(int)) &&
            kp->SetKernelArguments(5, &ti, sizeof(int));
  if (ok)
    moeDispatch(cc, kp, total);
  opencl::clReleaseMemObjectT(idx_buf);
  opencl::clReleaseMemObjectT(w_buf);

  const size_t elem = moeElemSize(out);
  moeMapBack(cc, expert_out, total * elem, /*read_only=*/true);
  moeMapBack(cc, out, (size_t)out_rows * width * elem, /*read_only=*/false);
  if (!ok)
    ml_loge("moe_scatter_add_rows_cl: failed to bind the kernel arguments");
  return ok;
}

bool moe_zero_rows_cl(Tensor &out, unsigned int rows) {
  if (rows == 0 || !moeSvmOperand(out) || moeRows(out) < rows ||
      out.width() == 0)
    return false;
  ClContext *cc = moeClContext();
  if (cc == nullptr)
    return false;
  const bool fp16 = out.getDataType() == ml::train::TensorDim::DataType::FP16;
  auto kp = cc->registerClKernel(moe_route_kernels, fp16 ? "moe_zero_rows_u16"
                                                         : "moe_zero_rows_u32");
  if (!kp)
    return false;

  const size_t total = (size_t)rows * out.width();
  const int ti = (int)total;
  moeUnmap(cc, out);
  bool ok = kp->SetKernelSVMArguments(0, out.getData<uint8_t>()) &&
            kp->SetKernelArguments(1, &ti, sizeof(int));
  if (ok)
    moeDispatch(cc, kp, total);
  moeMapBack(cc, out, total * moeElemSize(out), /*read_only=*/false);
  if (!ok)
    ml_loge("moe_zero_rows_cl: failed to bind the kernel arguments");
  return ok;
}

void registerMoeExpertFcCl(ml::train::TensorDim::DataType dtype,
                           MoeExpertFcFn fn) {
  std::lock_guard<std::mutex> lk(expertFcMutex());
  if (fn == nullptr)
    expertFcTable().erase(static_cast<int>(dtype));
  else
    expertFcTable()[static_cast<int>(dtype)] = fn;
}

void moe_expert_fc_cl(Tensor &input, Tensor &weight, Tensor &output) {
  MoeExpertFcFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(expertFcMutex());
    auto it = expertFcTable().find(static_cast<int>(weight.getDataType()));
    if (it != expertFcTable().end())
      fn = it->second;
  }
  if (fn != nullptr) {
    fn(input, weight, output);
    return;
  }
  get_cl_ops()->fc(input, weight, output);
}

void registerMoeExpertGroupCl(ml::train::TensorDim::DataType dtype,
                              MoeExpertGroupFn fn) {
  std::lock_guard<std::mutex> lk(expertFcMutex());
  if (fn == nullptr)
    expertGroupTable().erase(static_cast<int>(dtype));
  else
    expertGroupTable()[static_cast<int>(dtype)] = fn;
}

bool moe_experts_cl(const MoeExpertGroup &group) {
  if (group.gate.empty() || group.gate[0] == nullptr)
    return false;
  MoeExpertGroupFn fn = nullptr;
  {
    std::lock_guard<std::mutex> lk(expertFcMutex());
    auto it =
      expertGroupTable().find(static_cast<int>(group.gate[0]->getDataType()));
    if (it != expertGroupTable().end())
      fn = it->second;
  }
  return fn != nullptr && fn(group);
}

bool moe_experts_device_served_cl(ml::train::TensorDim::DataType expert_dtype) {
  if (expert_dtype != ml::train::TensorDim::DataType::TQ2CX)
    return false;
  ClContext *cc = moeClContext();
  return cc != nullptr && (cc->caps().dpas || cc->caps().int_dot_product);
}

bool moe_route_cl(const MoeRouter &router, unsigned int tokens, uint32_t *ids,
                  float *weights) {
  return tq2cx_moe_route_cl(router, tokens, ids, weights);
}

bool moe_experts_prebuild_cl(const MoeExpertGroup &group) {
  if (group.gate.empty() || group.gate[0] == nullptr ||
      group.gate[0]->getDataType() != ml::train::TensorDim::DataType::TQ2CX)
    return false;
  return tq2cx_moe_experts_prebuild_cl(group);
}

bool moe_experts_prefetch_cl(const MoeExpertGroup &group) {
  if (group.gate.empty() || group.gate[0] == nullptr ||
      group.gate[0]->getDataType() != ml::train::TensorDim::DataType::TQ2CX)
    return false;
  return tq2cx_moe_experts_prefetch_cl(group);
}

void moe_experts_join_cl(const MoeExpertGroup &group) {
  tq2cx_moe_experts_join_cl(group);
}

void moe_experts_flush_deferred_cl(const Tensor &t) {
  tq2cx_moe_experts_flush_deferred_cl(t);
}

} // namespace nntrainer
