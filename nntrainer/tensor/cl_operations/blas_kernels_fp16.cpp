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

#include <cstring>
#include <string>

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

// Fused RMSNorm and residual add: out = rmsnorm(input) * gamma + residual in a
// single kernel, which removes the separate residual-add dispatch and the idle
// between the two at a sandwich-norm boundary. Mirrors rmsnorm_cl_fp16_coop
// (LWS = 64, FP32 accumulation, LDS-tree reduction) and then folds the add.
//
// The result is bit-identical to running the norm and the add separately: `nv`
// is a named half8 intermediate, so norm * gamma rounds to FP16 once before the
// residual add rounds again -- exactly the two roundings of the separate pair
// (the norm kernel stores FP16 and the add kernel re-rounds). That holds when
// the build does not contract the multiply-add. The kernel source is inline
// here rather than in a .cl file because it exists only for this dispatch.
static const std::string rmsnorm_add_fp16_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void rmsnorm_cl_fp16_coop_add(__global const half *input,
                                       __global half *output,
                                       __global const half *alpha,
                                       half epsilon, int n_rows, int W,
                                       __global const half *residual) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);   // 0..63
  if (row >= n_rows)
    return;
  const long base = (long)row * (long)W;
  const int W8 = W >> 3;
  __global const half8 *in8 = (__global const half8 *)(input + base);
  float partial = 0.0f;
  for (int i = tid; i < W8; i += 64) {
    const float8 v = convert_float8(in8[i]);
    partial += dot(v.lo, v.lo) + dot(v.hi, v.hi);
  }
  __local float lsum[64];
  lsum[tid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 32; s > 0; s >>= 1) {
    if (tid < s)
      lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = lsum[0] / (float)W;
  const float scale = rsqrt(mean + (float)epsilon);
  __global half8 *out8 = (__global half8 *)(output + base);
  __global const half8 *r8 = (__global const half8 *)(residual + base);
  for (int i = tid; i < W8; i += 64) {
    // Large-gamma FP16 overflow guard (see rmsnorm_fp16.cl): apply gamma in
    // FP32 and clamp to the FP16-safe range before the residual add. gamma
    // (alpha) is a model-weight pointer with no 16-byte alignment guarantee,
    // so it is loaded per element -- a half8 vector load would read garbage.
    const float8 v = convert_float8(in8[i]) * scale;
    const int gi = i << 3;
    const float8 a = (float8)(
      (float)alpha[gi + 0], (float)alpha[gi + 1], (float)alpha[gi + 2],
      (float)alpha[gi + 3], (float)alpha[gi + 4], (float)alpha[gi + 5],
      (float)alpha[gi + 6], (float)alpha[gi + 7]);
    const float8 nv = clamp(v * a, -60000.0f, 60000.0f);
    out8[i] = convert_half8(nv) + r8[i];
  }
}
)CL";

bool rmsnorm_add_cl_fp16(const _FP16 *input, const _FP16 *gamma,
                         const _FP16 *residual, _FP16 *result, float epsilon,
                         unsigned int height, unsigned int width, bool use_svm,
                         void *out_clmem, void *in_clmem, void *resid_clmem) {
  if ((width % 8u) != 0u || !use_svm)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return false;
  cl_half eps_h = 0;
  const _FP16 eps_f = static_cast<_FP16>(epsilon);
  std::memcpy(&eps_h, &eps_f, sizeof(cl_half));
  cl_mem out_cl = static_cast<cl_mem>(out_clmem);
  cl_mem in_cl = static_cast<cl_mem>(in_clmem);
  cl_mem resid_cl = static_cast<cl_mem>(resid_clmem);
  const int n_rows = (int)height, w = (int)width;

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rmsnorm_add_fp16_kernel, "rmsnorm_cl_fp16_coop_add");
  if (!kp)
    return false;
  // Argument order mirrors rmsnorm_cl_fp16_coop (in, out, alpha, eps, n_rows,
  // W) with the residual pointer appended last.
  bool a0 = in_cl ? kp->SetKernelArguments(0, &in_cl, sizeof(cl_mem))
                  : kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input));
  bool a1 = out_cl ? kp->SetKernelArguments(1, &out_cl, sizeof(cl_mem))
                   : kp->SetKernelSVMArguments(1, result);
  bool a2 = kp->SetKernelSVMArguments(2, const_cast<_FP16 *>(gamma));
  bool a3 = kp->SetKernelArguments(3, &eps_h, sizeof(cl_half));
  bool a4 = kp->SetKernelArguments(4, &n_rows, sizeof(int));
  bool a5 = kp->SetKernelArguments(5, &w, sizeof(int));
  bool a6 = resid_cl
              ? kp->SetKernelArguments(6, &resid_cl, sizeof(cl_mem))
              : kp->SetKernelSVMArguments(6, const_cast<_FP16 *>(residual));
  if (!(a0 && a1 && a2 && a3 && a4 && a5 && a6))
    return false;
  constexpr int RMSN_LWS = 64;
  const int gws[3] = {RMSN_LWS * n_rows, 1, 1};
  const int lws[3] = {RMSN_LWS, 1, 1};
  return blas_cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
}

void rmsnorm_cl_fp16(const _FP16 *input, const _FP16 *gamma, _FP16 *result,
                     const float epsilon, unsigned int height,
                     unsigned int width, const bool use_svm, void *out_clmem,
                     void *in_clmem) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  cl_half eps_h = 0;
  const _FP16 eps_f = static_cast<_FP16>(epsilon);
  std::memcpy(&eps_h, &eps_f, sizeof(cl_half));
  const size_t in_bytes = (size_t)height * width * sizeof(_FP16);

  // Device-resident binding: when out_clmem is set the normed output goes to
  // that device buffer (the tensor's residency-plane sub-buffer) with no host
  // map, so a device-direct consumer reads it without the map/unmap pair that
  // dominates a long prefill. When in_clmem is set the input is read from a
  // device-resident producer's buffer; mixing a cl_mem argument with SVM
  // arguments in one kernel is valid. Cooperative path only (width % 8 == 0).
  cl_mem out_cl = static_cast<cl_mem>(out_clmem);
  cl_mem in_cl = static_cast<cl_mem>(in_clmem);
  const bool to_clmem = (out_cl != nullptr) && use_svm && (width % 8u == 0u);
  const bool from_clmem = (in_cl != nullptr) && (width % 8u == 0u);

  // The scalar rmsnorm_cl_fp16 kernel gives one work item the whole row, so a
  // dispatch that sized the local work group by W would exceed
  // CL_DEVICE_MAX_WORK_GROUP_SIZE for a full hidden width and fail silently,
  // leaving the output stale. Use the cooperative kernel (64 work items per
  // row, half8-vectorized, FP32 accumulation, gamma folded in) whenever
  // width % 8 == 0, and keep the scalar kernel -- with a valid local size --
  // only for the rare width that is not a multiple of 8.
  const int n_rows = (int)height;
  const int w = (int)width;
  const bool use_coop = (width % 8u == 0u);

  if (use_coop) {
    constexpr int RMSN_LWS = 64;
    // Gamma-free variant (a norm whose layer has no learned scale, so gamma is
    // nullptr): the _ng kernel keeps the same six-argument signature but never
    // reads alpha, so bind argument 2 to a valid-but-unread pointer rather
    // than a null SVM argument. That keeps such a norm on the cooperative
    // kernel (FP32 reduction, overflow-safe) instead of a host fallback.
    const bool no_gamma = (gamma == nullptr);
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      rmsnorm_fp16_kernel,
      no_gamma ? "rmsnorm_cl_fp16_coop_ng" : "rmsnorm_cl_fp16_coop");
    if (!kp)
      return;
    if (to_clmem || from_clmem) {
      // Mixed bind: each of input and output takes its own tensor's plane (a
      // device sub-buffer when that tensor is device-resident, SVM otherwise);
      // gamma stays SVM.
      bool ok = true;
      if (from_clmem)
        ok = ok && kp->SetKernelArguments(0, &in_cl, sizeof(cl_mem));
      else
        ok = ok && kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input));
      if (to_clmem)
        ok = ok && kp->SetKernelArguments(1, &out_cl, sizeof(cl_mem));
      else
        ok = ok && kp->SetKernelSVMArguments(1, result);
      ok = ok && kp->SetKernelSVMArguments(
                   2, const_cast<_FP16 *>(no_gamma ? input : gamma));
      if (!ok)
        return;
    } else if (use_svm) {
      if (!kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input)) ||
          !kp->SetKernelSVMArguments(1, result) ||
          !kp->SetKernelSVMArguments(
            2, const_cast<_FP16 *>(no_gamma ? input : gamma)))
        return;
    } else {
      auto &clbuf = ClBufferManager::Global();
      // A gamma-free norm only reaches here from a non-SVM caller; the _ng
      // kernel ignores argument 2, so skip the null-gamma write and bind the
      // input buffer for it rather than dereferencing a null source.
      if (!clbuf.getInBufferA()->WriteDataRegion(blas_cc->command_queue_inst_,
                                                 in_bytes, input) ||
          (!no_gamma &&
           !clbuf.getInBufferB()->WriteDataRegion(
             blas_cc->command_queue_inst_, width * sizeof(_FP16), gamma)))
        return;
      if (!kp->SetKernelArguments(0, &clbuf.getInBufferA()->GetBuffer(),
                                  sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &clbuf.getOutBufferA()->GetBuffer(),
                                  sizeof(cl_mem)) ||
          !kp->SetKernelArguments(
            2,
            &(no_gamma ? clbuf.getInBufferA() : clbuf.getInBufferB())
               ->GetBuffer(),
            sizeof(cl_mem)))
        return;
    }
    if (!kp->SetKernelArguments(3, &eps_h, sizeof(cl_half)) ||
        !kp->SetKernelArguments(4, &n_rows, sizeof(int)) ||
        !kp->SetKernelArguments(5, &w, sizeof(int)))
      return;
    const int work_groups_count[3] = {RMSN_LWS * n_rows, 1, 1};
    const int work_group_size[3] = {RMSN_LWS, 1, 1};
    if (!blas_cc->command_queue_inst_.DispatchCommand(kp, work_groups_count,
                                                      work_group_size))
      return;
  } else {
    // Scalar fallback (width % 8 != 0): one work item per row; the local size
    // must be a legal one, not W. Binds the eight-argument [B, C, H, W] form.
    ClContext::SharedPtrClKernel kp =
      blas_cc->registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16");
    if (!kp)
      return;
    const int b = 1, c = 1, h = (int)height;
    if (use_svm) {
      if (!kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input)) ||
          !kp->SetKernelSVMArguments(1, result) ||
          !kp->SetKernelSVMArguments(2, const_cast<_FP16 *>(gamma)))
        return;
    } else {
      auto &clbuf = ClBufferManager::Global();
      if (!clbuf.getInBufferA()->WriteDataRegion(blas_cc->command_queue_inst_,
                                                 in_bytes, input) ||
          !clbuf.getInBufferB()->WriteDataRegion(blas_cc->command_queue_inst_,
                                                 width * sizeof(_FP16), gamma))
        return;
      if (!kp->SetKernelArguments(0, &clbuf.getInBufferA()->GetBuffer(),
                                  sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &clbuf.getOutBufferA()->GetBuffer(),
                                  sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &clbuf.getInBufferB()->GetBuffer(),
                                  sizeof(cl_mem)))
        return;
    }
    if (!kp->SetKernelArguments(3, &eps_h, sizeof(cl_half)) ||
        !kp->SetKernelArguments(4, &b, sizeof(int)) ||
        !kp->SetKernelArguments(5, &c, sizeof(int)) ||
        !kp->SetKernelArguments(6, &h, sizeof(int)) ||
        !kp->SetKernelArguments(7, &w, sizeof(int)))
      return;
    const int work_groups_count[3] = {b * c, h, 1};
    const int work_group_size[3] = {1, 1, 1};
    if (!blas_cc->command_queue_inst_.DispatchCommand(kp, work_groups_count,
                                                      work_group_size))
      return;
  }

  if (use_svm) {
    // The map is deliberately BLOCKING. The normed output feeds a residual add
    // that has a host fast path, and an asynchronous map there races with that
    // host read and corrupts the output (measured). Even an all-GPU FP16 chain
    // is not safe to make asynchronous here: it measured 5% faster and the
    // generated text diverged, i.e. a real coherence race. Closing that gap
    // needs the host map/unmap pair removed on both producer and consumer, not
    // this call flipped to asynchronous.
    // When the output went to a device buffer instead there is no host map to
    // restore -- the consumer reads it on the device.
    if (!to_clmem)
      blas_cc->command_queue_inst_.enqueueSVMMap(result, in_bytes, false);
  } else {
    auto &clbuf = ClBufferManager::Global();
    clbuf.getOutBufferA()->ReadDataRegion(blas_cc->command_queue_inst_,
                                          in_bytes, result);
  }
}

void rms_reverse_norm_cl_fp16(const _FP16 *input, const _FP16 *weight,
                              _FP16 out_scale, _FP16 *result,
                              const float epsilon, unsigned int height,
                              unsigned int width, bool use_svm, void *out_clmem,
                              void *in_clmem) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  cl_half eps_h = 0;
  const _FP16 eps_f = static_cast<_FP16>(epsilon);
  std::memcpy(&eps_h, &eps_f, sizeof(cl_half));
  const size_t in_bytes = (size_t)height * width * sizeof(_FP16);
  cl_mem out_cl = static_cast<cl_mem>(out_clmem);
  cl_mem in_cl = static_cast<cl_mem>(in_clmem);
  const bool to_clmem = (out_cl != nullptr) && use_svm;
  const bool from_clmem = (in_cl != nullptr);
  const int n_rows = (int)height;
  const int w = (int)width;
  constexpr int RMSN_LWS = 64;

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rmsnorm_fp16_kernel, "rms_reverse_norm_cl_fp16_coop");
  if (!kp)
    return;

  if (to_clmem || from_clmem) {
    bool ok = true;
    if (from_clmem)
      ok = ok && kp->SetKernelArguments(0, &in_cl, sizeof(cl_mem));
    else
      ok = ok && kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input));
    if (to_clmem)
      ok = ok && kp->SetKernelArguments(1, &out_cl, sizeof(cl_mem));
    else
      ok = ok && kp->SetKernelSVMArguments(1, result);
    ok = ok && kp->SetKernelSVMArguments(2, const_cast<_FP16 *>(weight));
    if (!ok)
      return;
  } else if (use_svm) {
    if (!kp->SetKernelSVMArguments(0, const_cast<_FP16 *>(input)) ||
        !kp->SetKernelSVMArguments(1, result) ||
        !kp->SetKernelSVMArguments(2, const_cast<_FP16 *>(weight)))
      return;
  } else {
    auto &clbuf = ClBufferManager::Global();
    if (!clbuf.getInBufferA()->WriteDataRegion(blas_cc->command_queue_inst_,
                                               in_bytes, input) ||
        !clbuf.getInBufferB()->WriteDataRegion(blas_cc->command_queue_inst_,
                                               width * sizeof(_FP16), weight))
      return;
    if (!kp->SetKernelArguments(0, &clbuf.getInBufferA()->GetBuffer(),
                                sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &clbuf.getOutBufferA()->GetBuffer(),
                                sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &clbuf.getInBufferB()->GetBuffer(),
                                sizeof(cl_mem)))
      return;
  }
  // out_scale is already an FP16 value: pass its two bytes straight through as
  // a `half` kernel argument, the same way epsilon is passed. Casting it to
  // cl_half would convert the VALUE to an unsigned integer (0.0292 -> 0) and
  // zero the scale, producing an all-zero output.
  if (!kp->SetKernelArguments(3, &out_scale, sizeof(cl_half)) ||
      !kp->SetKernelArguments(4, &eps_h, sizeof(cl_half)) ||
      !kp->SetKernelArguments(5, &n_rows, sizeof(int)) ||
      !kp->SetKernelArguments(6, &w, sizeof(int)))
    return;
  const int work_groups_count[3] = {RMSN_LWS * n_rows, 1, 1};
  const int work_group_size[3] = {RMSN_LWS, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, work_groups_count,
                                                    work_group_size))
    return;
  if (use_svm) {
    if (!to_clmem)
      blas_cc->command_queue_inst_.enqueueSVMMap(result, in_bytes, false);
  } else {
    auto &clbuf = ClBufferManager::Global();
    clbuf.getOutBufferA()->ReadDataRegion(blas_cc->command_queue_inst_,
                                          in_bytes, result);
  }
}

// out[r * fs + j] = in[r * in_width + off + j] for r in [0, rows), j in [0,
// fs). Gathers one layer's slice (off = layer_index * fs) from a packed
// per-layer input tensor. Device-resident when both planes are device buffers,
// else SVM.
static const std::string per_layer_slice_fp16_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void per_layer_slice_cl_fp16(__global const half *in,
                                      __global half *out, int fs, int in_width,
                                      int off, int rows) {
  int gid = get_global_id(0);
  int total = rows * fs;
  if (gid >= total)
    return;
  int r = gid / fs;
  int j = gid - r * fs;
  out[gid] = in[r * in_width + off + j];
}
)CL";

void per_layer_slice_cl_fp16(const _FP16 *input, _FP16 *result,
                             unsigned int rows, unsigned int fs,
                             unsigned int in_width, unsigned int off,
                             bool use_svm, void *out_clmem, void *in_clmem) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    per_layer_slice_fp16_kernel, "per_layer_slice_cl_fp16");
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
  int fsi = (int)fs, iwi = (int)in_width, offi = (int)off, rowsi = (int)rows;
  ok = ok && kp->SetKernelArguments(2, &fsi, sizeof(int));
  ok = ok && kp->SetKernelArguments(3, &iwi, sizeof(int));
  ok = ok && kp->SetKernelArguments(4, &offi, sizeof(int));
  ok = ok && kp->SetKernelArguments(5, &rowsi, sizeof(int));
  if (!ok)
    return;

  const int total = (int)rows * (int)fs;
  const int lws = 64;
  const int gws = ((total + lws - 1) / lws) * lws;
  const int wgc[3] = {gws, 1, 1};
  const int wgs[3] = {lws, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wgc, wgs))
    return;

  if (!to_clmem && use_svm)
    blas_cc->command_queue_inst_.enqueueSVMMap(
      result, (size_t)total * sizeof(_FP16), true);
}

} // namespace nntrainer
