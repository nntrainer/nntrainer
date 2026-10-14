// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   tq2cx_cl.cpp
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL FC for a TQ2CX (ternary) weight: one expert projection.
 */

#include "tq2cx_cl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#endif

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cl_kernels/int8_int4_gemm_v8c.h>
#include <cl_kernels/tq2cx_gemm.h>
#include <cl_kernels/tq2cx_gemm_portable.h>
#include <cl_kernels/tq2moe_fsu_flags.h>
#include <cl_kernels/tq2moe_glu_quant.h>
#include <cl_kernels/tq2moe_route.h>
#include <engine.h>
#include <expert_fsu_flags.h>
#include <expert_reader_pool.h>
#include <expert_stage_cache.h>
#include <fp16.h>
#include <moe_route_det.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <opencl_loader.h>
#include <tensor.h>
#include <ternary_tensor.h>

namespace nntrainer {

namespace {

/**
 * @brief Build options of the TQ2CX programs: CL 3.0 (Intel NEO exposes the
 *        packed integer dot product only there) and the stored layout the
 *        kernels index the records with (tq2cx_layout.h).
 */
std::string tq2Options(const std::string &extra = "") {
  const Tq2cxLayout lay = tq2cxStoredLayout();
  std::string o = "-cl-std=CL3.0";
  if (lay.tiled())
    o += " -DTQ2_LAYR=" + std::to_string(lay.R) +
         " -DTQ2_LAYV=" + std::to_string(lay.V);
  return extra.empty() ? o : o + " " + extra;
}

/** rows of one work-item in tq2cx_gemm_dp4a (the kernel's default) */
constexpr unsigned int kTq2TM = 8;

/**
 * @brief The device copy of one TQ2CX record: the masks and the scales.
 */
struct Tq2DeviceWeight {
  cl_mem masks = nullptr;  /**< [N][K/32] (pos, neg) uint32 pairs */
  cl_mem scales = nullptr; /**< [N] fp32 */
};

std::mutex &tq2Mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<const void *, Tq2DeviceWeight> &tq2Weights() {
  static std::unordered_map<const void *, Tq2DeviceWeight> w;
  return w;
}

cl_mem tq2Upload(ClContext *cc, const void *host, size_t bytes,
                 cl_mem_flags flags = CL_MEM_READ_ONLY) {
  cl_int err = CL_SUCCESS;
  cl_mem b = opencl::clCreateBufferT(cc->context_inst_.GetContextNoRetain(),
                                     flags | CL_MEM_COPY_HOST_PTR, bytes,
                                     const_cast<void *>(host), &err);
  NNTR_THROW_IF(err != CL_SUCCESS || b == nullptr, std::runtime_error)
    << "tq2cx_expert_fc_cl: clCreateBuffer failed (" << err << ")";
  return b;
}

/**
 * @brief The resident device copy of @a weight, uploaded on first use.
 */
const Tq2DeviceWeight &tq2DeviceWeight(ClContext *cc, const Tensor &weight,
                                       unsigned int N, unsigned int K) {
  const uint8_t *rec = weight.getData<uint8_t>();
  std::lock_guard<std::mutex> lk(tq2Mutex());
  auto it = tq2Weights().find(rec);
  if (it != tq2Weights().end())
    return it->second;
  refuseIfTq2cxPayloadReleased(rec, "tq2cx_expert_fc_cl");
  // the masks in the stored layout (padded rows included), then the scales
  const size_t mask_bytes = tq2cxMaskBytes(tq2cxStoredLayout(), N, K);
  Tq2DeviceWeight w;
  w.masks = tq2Upload(cc, rec, mask_bytes);
  w.scales = tq2Upload(cc, rec + mask_bytes, (size_t)N * sizeof(float));
  return tq2Weights().emplace(rec, w).first->second;
}

/**
 * @brief Per-row asymmetric int8 quantization, the scheme of the v8c
 *        activation quantizer: x ~= (q - zp) * scale.
 */
void tq2QuantizeRows(const float *x, unsigned int M, unsigned int K,
                     std::vector<int8_t> &q, std::vector<float> &scale,
                     std::vector<int> &zp) {
  q.resize((size_t)M * K);
  scale.resize(M);
  zp.resize(M);
  for (unsigned int i = 0; i < M; ++i) {
    const float *row = x + (size_t)i * K;
    float fmn = 0.0f, fmx = 0.0f;
    for (unsigned int k = 0; k < K; ++k) {
      fmn = std::fmin(fmn, row[k]);
      fmx = std::fmax(fmx, row[k]);
    }
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = fmx - fmn;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = fmn * scale_q, dmax = fmx * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    zp_f = std::min(qmax, std::max(qmin, zp_f));
    const int z = (int)std::nearbyint(zp_f);
    scale[i] = recip;
    zp[i] = z;
    for (unsigned int k = 0; k < K; ++k) {
      const int v = (int)std::nearbyint(row[k] * scale_q) + z;
      q[(size_t)i * K + k] = (int8_t)std::min(127, std::max(-128, v));
    }
  }
}

/**
 * @brief Host fallback of one TQ2CX expert projection for FP16 activations:
 *        fp32 accumulation over the widened rows, the result narrowed with a
 *        clamp to the half range. Correctness path only; the grouped launch
 *        (tq2cx_moe_experts_cl) is the device path for FP16.
 */
void tq2cxExpertFcHostFp16(Tensor &input, Tensor &weight, Tensor &output) {
  const unsigned int K = input.width();
  const unsigned int M = input.batch() * input.channel() * input.height();
  const unsigned int N = weight.width();
  NNTR_THROW_IF(weight.height() != K || K % 32 != 0 || output.width() != N,
                std::invalid_argument)
    << "tq2cx_expert_fc_cl: shape mismatch";
  if (M == 0)
    return;
  const bool busy = cl_queue_may_be_busy();
  auto svm = [](const Tensor &t) {
    const auto md = t.getMemoryData();
    return md && md->isSVM() && !t.isClMem();
  };
  NNTR_THROW_IF(!svm(input) || !svm(output), std::invalid_argument)
    << "tq2cx_expert_fc_cl: FP16 operands must be on the shared plane";
  void *in_ptr = input.getData<uint8_t>();
  void *out_ptr = output.getData<uint8_t>();
  std::vector<float> x((size_t)M * K), y((size_t)M * N);
  if (busy)
    cl_svm_map_force(in_ptr, (size_t)M * K * sizeof(uint16_t), true);
  const uint16_t *src = static_cast<const uint16_t *>(in_ptr);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = compute_fp16_to_fp32(src[i]);
  if (busy)
    cl_svm_unmap_force(in_ptr);
  refuseIfTq2cxPayloadReleased(weight.getData<uint8_t>(),
                               "tq2cx_expert_fc_cl (FP16 host fallback)");
  tq2cx_gemm_ref_fp32(M, N, K, x.data(), K, weight.getData<uint8_t>(), y.data(),
                      N, 0.0f);
  const size_t out_bytes = (size_t)M * N * sizeof(uint16_t);
  cl_svm_host_write_begin(out_ptr, out_bytes);
  uint16_t *dst = static_cast<uint16_t *>(out_ptr);
  for (size_t i = 0; i < y.size(); ++i)
    dst[i] =
      compute_fp32_to_fp16(std::min(65504.0f, std::max(-65504.0f, y[i])));
  cl_svm_host_write_end(out_ptr, out_bytes);
}

} // namespace

void tq2cx_expert_fc_cl(Tensor &input, Tensor &weight, Tensor &output) {
  NNTR_THROW_IF(weight.getDataType() != ml::train::TensorDim::DataType::TQ2CX,
                std::invalid_argument)
    << "tq2cx_expert_fc_cl: the weight is not TQ2CX";
  const bool fp16 =
    input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
    output.getDataType() == ml::train::TensorDim::DataType::FP16;
  NNTR_THROW_IF(
    !fp16 && (input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
              output.getDataType() != ml::train::TensorDim::DataType::FP32),
    std::invalid_argument)
    << "tq2cx_expert_fc_cl: FP32 or FP16 activations, the same on both sides";
  if (fp16) {
    tq2cxExpertFcHostFp16(input, weight, output);
    return;
  }
  const unsigned int K = input.width();
  const unsigned int M = input.batch() * input.channel() * input.height();
  const unsigned int N = weight.width();
  NNTR_THROW_IF(weight.height() != K || K % 32 != 0, std::invalid_argument)
    << "tq2cx_expert_fc_cl: weight IN " << weight.height()
    << " vs activation width " << K << " (must match, multiple of 32)";
  const auto out_md = output.getMemoryData();
  NNTR_THROW_IF(!out_md || !out_md->isSVM() || output.isClMem(),
                std::invalid_argument)
    << "tq2cx_expert_fc_cl: the output must be on the shared plane";
  NNTR_THROW_IF((size_t)output.batch() * output.channel() * output.height() <
                    M ||
                  output.width() != N,
                std::invalid_argument)
    << "tq2cx_expert_fc_cl: output shape mismatch";
  if (M == 0)
    return;

  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  NNTR_THROW_IF(cc == nullptr, std::runtime_error)
    << "tq2cx_expert_fc_cl: no gpu context";
  const Tq2DeviceWeight &w = tq2DeviceWeight(cc, weight, N, K);

  // The activation is quantized on the host, so it has to be read there: a
  // kernel may still be writing it (the row gather in front of this FC).
  std::vector<int8_t> xq;
  std::vector<float> xs;
  std::vector<int> xz;
  {
    const auto in_md = input.getMemoryData();
    void *in_ptr = input.getData<uint8_t>();
    const bool map =
      in_md && in_md->isSVM() && !input.isClMem() && cl_queue_may_be_busy();
    if (map)
      cl_svm_map_force(in_ptr, (size_t)M * K * sizeof(float), true);
    tq2QuantizeRows(input.getData<float>(), M, K, xq, xs, xz);
    if (map)
      cl_svm_unmap_force(in_ptr);
  }
  cl_mem d_x = tq2Upload(cc, xq.data(), xq.size());
  cl_mem d_s = tq2Upload(cc, xs.data(), xs.size() * sizeof(float));
  cl_mem d_z = tq2Upload(cc, xz.data(), xz.size() * sizeof(int));

  const bool gemv = (M == 1);
  auto kp = cc->registerClKernel(tq2cx_gemm_kernel,
                                 gemv ? "tq2cx_gemv_row" : "tq2cx_gemm_dp4a",
                                 tq2Options());
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_expert_fc_cl: the TQ2CX kernels did not build";

  float *out = output.getData<float>();
  auto &q = cc->command_queue_inst_;
  q.enqueueSVMUnmap(out);
  const int Ni = (int)N, Ki = (int)K, Mi = (int)M;
  bool ok = kp->SetKernelArguments(0, &w.masks, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &w.scales, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &d_x, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &d_s, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &d_z, sizeof(cl_mem)) &&
            kp->SetKernelSVMArguments(5, out);
  if (ok && gemv)
    ok = kp->SetKernelArguments(6, &Ni, sizeof(int)) &&
         kp->SetKernelArguments(7, &Ki, sizeof(int));
  else if (ok)
    ok = kp->SetKernelArguments(6, &Mi, sizeof(int)) &&
         kp->SetKernelArguments(7, &Ni, sizeof(int)) &&
         kp->SetKernelArguments(8, &Ki, sizeof(int));
  if (ok) {
    constexpr size_t LWS = 64;
    const size_t gn = ((size_t)N + LWS - 1) / LWS * LWS;
    const size_t gm = gemv ? 1 : ((size_t)M + kTq2TM - 1) / kTq2TM;
    std::array<size_t, 3> gws = {gn, gm, 1};
    std::array<size_t, 3> lws = {LWS, 1, 1};
    q.enqueueKernel(kp->GetKernel(), 2, gws.data(), lws.data());
  }
  // The enqueued command retains these; this drops our references only.
  opencl::clReleaseMemObjectT(d_x);
  opencl::clReleaseMemObjectT(d_s);
  opencl::clReleaseMemObjectT(d_z);
  q.enqueueSVMMap(out, (size_t)M * N * sizeof(float), /*read_only=*/true,
                  /*event=*/nullptr, /*async=*/true);
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_expert_fc_cl: failed to bind the kernel arguments";
}

namespace {

// ---------------------------------------------------------------------------
// Grouped TQ2CX experts (tq2cx_moe_experts_cl)
// ---------------------------------------------------------------------------

/** helper kernels of the grouped path: row gather, GeGLU, weighted combine */
const std::string kTq2MoeAux = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
__kernel void tq2moe_gather_f32(__global const float *in, __global float *out,
                                __global const int *src_row, const int width,
                                const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / width;
  out[gid] = in[(long)src_row[r] * width + (gid - r * width)];
}

__kernel void tq2moe_gather_f16(__global const half *in, __global float *out,
                                __global const int *src_row, const int width,
                                const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int r = gid / width;
  out[gid] = (float)in[(long)src_row[r] * width + (gid - r * width)];
}

__kernel void tq2moe_glu_f32(__global const float *gate,
                             __global const float *up, __global float *out,
                             const int total, const int exact,
                             const int gate_off, const int up_off,
                             const int out_off) {
  const int i = get_global_id(0);
  if (i >= total) return;
  const float x = gate[gate_off + i];
  float g;
  if (exact) {
    g = 0.5f * x * (1.0f + erf(x * 0.7071067811865476f));
  } else {
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    g = 0.5f * x * (1.0f + tanh(inner));
  }
  out[out_off + i] = g * up[up_off + i];
}

// out[t][j] = sum over c < counts[t] of weight[t*max_k+c] *
//             Y[src_row[t*max_k+c]][j], contributions in table order
// (ascending expert id); written, not accumulated.
__kernel void tq2moe_combine_f32(__global const float *Y, __global float *out,
                                 __global const int *src_row,
                                 __global const float *weight,
                                 __global const int *counts, const int width,
                                 const int max_k, const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int t = gid / width;
  const int j = gid - t * width;
  const int n = counts[t];
  float acc = 0.0f;
  for (int c = 0; c < n; ++c)
    acc = acc + weight[t * max_k + c] *
                  Y[(long)src_row[t * max_k + c] * width + j];
  out[gid] = acc;
}

// The same combine into a half output: fp32 sum, narrowed with a clamp to
// the half range.
__kernel void tq2moe_combine_f16(__global const float *Y, __global half *out,
                                 __global const int *src_row,
                                 __global const float *weight,
                                 __global const int *counts, const int width,
                                 const int max_k, const int total) {
  const int gid = get_global_id(0);
  if (gid >= total) return;
  const int t = gid / width;
  const int j = gid - t * width;
  const int n = counts[t];
  float acc = 0.0f;
  for (int c = 0; c < n; ++c)
    acc = acc + weight[t * max_k + c] *
                  Y[(long)src_row[t * max_k + c] * width + j];
  out[gid] = (half)clamp(acc, -65504.0f, 65504.0f);
}
)CL";

/**
 * The prefill gather quantizers of the grouped path, fused with the gather
 * that fed the quantizer (one launch instead of two, no fp32 copy of the
 * input): tq2moe_gather_quant_f16 / _f32 quantize gathered row r straight
 * from the layer input row src_row[r] (fp16 widened exactly to fp32, or
 * fp32) -- the row tq2moe_gather_* used to copy into an fp32 buffer for
 * v8c_act_quant_f32_par. (The expert GeGLU + quantize of the grouped path
 * is tq2moe_glu_act_quant_f32 in cl_kernels/tq2moe_glu_quant.cl, the kernel
 * the decode GEMVs use too.)
 *
 * Both are v8c_act_quant_f32_par (int8_int4_gemm_v8c.cl) term for term: one
 * 64-lane work-group per row, the same strided min / max and row-sum trees,
 * the same scale / zero-point code (tq2q_row_params, kept above the
 * FP_CONTRACT pragma like the quantizer it copies) and the same rounding, so
 * the int8 rows, scales, zero points and row sums are those of the two-launch
 * form bit for bit. Built with the quantizer's options (tq2QuantOptions),
 * plus -DV8C_AQ_PERM to store each row in TQ2 order like the v8c quantizer
 * (the expert GEMM tq2cx_gemm_tiles_v2_portable reads that order).
 */
const std::string kTq2MoeQuant = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TQ2Q_LWS 64
/* V8C_AQ_PERM: each row stored in TQ2 order (int8_int4_gemm_v8c.cl) */
#ifdef V8C_AQ_PERM
#define V8C_AQ_IDX(k) (((k) & ~31) | (((k) & 7) << 2) | (((k) >> 3) & 3))
#else
#define V8C_AQ_IDX(k) (k)
#endif

/* scale / zero point of a row from its min and max (v8c_act_quant_*_par) */
inline void tq2q_row_params(const float fmn, const float fmx,
                            __private float *scale_q_out,
                            __private float *recip_out,
                            __private int *zp_out) {
  const float rmin = fmn < 0.0f ? fmn : 0.0f;
  const float rmax = fmx > 0.0f ? fmx : 0.0f;
  const float qmin = -128.0f, qmax = 127.0f;
  const float range = rmax - rmin;
  const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
  const float recip = range > 0.0f ? range / 255.0f : 1.0f;
  const float dmin = rmin * scale_q, dmax = rmax * scale_q;
  const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
  float zp_f =
    (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
  if (zp_f < qmin) zp_f = qmin;
  if (zp_f > qmax) zp_f = qmax;
  *scale_q_out = scale_q;
  *recip_out = recip;
  *zp_out = (int)rint(zp_f);
}

#pragma OPENCL FP_CONTRACT OFF

/** the min / max tree, the row parameters and the row-sum tree of
 * v8c_act_quant_f32_par around a value source LOAD(k) */
#define TQ2Q_ROW(LOAD, PASS1_STORE)                                            \
  const int row = get_group_id(0);                                             \
  if (row >= M) return;                                                        \
  const int tid = get_local_id(0);                                             \
  __local float lmin[TQ2Q_LWS];                                                \
  __local float lmax[TQ2Q_LWS];                                                \
  __local int   lsum[TQ2Q_LWS];                                                \
  __local float l_scale_q;                                                     \
  __local int   l_zp;                                                          \
  float pmin = 0.0f, pmax = 0.0f;                                              \
  for (int k = tid; k < K; k += TQ2Q_LWS) {                                    \
    float v = LOAD(k);                                                         \
    PASS1_STORE(k, v);                                                         \
    pmin = fmin(pmin, v);                                                      \
    pmax = fmax(pmax, v);                                                      \
  }                                                                            \
  lmin[tid] = pmin;                                                            \
  lmax[tid] = pmax;                                                            \
  barrier(CLK_LOCAL_MEM_FENCE);                                                \
  for (int s = TQ2Q_LWS / 2; s > 0; s >>= 1) {                                 \
    if (tid < s) {                                                             \
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);                              \
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);                              \
    }                                                                          \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
  }                                                                            \
  if (tid == 0) {                                                              \
    float sq, rc;                                                              \
    int zq;                                                                    \
    tq2q_row_params(lmin[0], lmax[0], &sq, &rc, &zq);                          \
    l_scale_q = sq;                                                            \
    l_zp = zq;                                                                 \
    scale_per_row[row] = rc;                                                   \
    zp_per_row[row] = zq;                                                      \
  }                                                                            \
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);                         \
  const float scale_q = l_scale_q;                                             \
  const int zp = l_zp;                                                         \
  int psum = 0;                                                                \
  for (int k = tid; k < K; k += TQ2Q_LWS) {                                    \
    int q = (int)rint(LOAD2(k) * scale_q) + zp;                                \
    if (q < -128) q = -128;                                                    \
    if (q > 127) q = 127;                                                      \
    act_int8[(long)row * K + V8C_AQ_IDX(k)] = (char)q;                         \
    psum += q;                                                                 \
  }                                                                            \
  lsum[tid] = psum;                                                            \
  barrier(CLK_LOCAL_MEM_FENCE);                                                \
  for (int s = TQ2Q_LWS / 2; s > 0; s >>= 1) {                                 \
    if (tid < s) lsum[tid] += lsum[tid + s];                                   \
    barrier(CLK_LOCAL_MEM_FENCE);                                              \
  }                                                                            \
  if (tid == 0) row_sum_act[row] = lsum[0];

#define TQ2Q_NO_STORE(k, v)

__attribute__((reqd_work_group_size(TQ2Q_LWS, 1, 1)))
__kernel void tq2moe_gather_quant_f32(
    __global const float *in, __global const int *src_row,
    __global char *act_int8, __global float *scale_per_row,
    __global int *zp_per_row, __global int *row_sum_act, const int M,
    const int K) {
#define LOAD(k) in[(long)src_row[row] * K + (k)]
#define LOAD2(k) LOAD(k)
  TQ2Q_ROW(LOAD, TQ2Q_NO_STORE)
#undef LOAD
#undef LOAD2
}

__attribute__((reqd_work_group_size(TQ2Q_LWS, 1, 1)))
__kernel void tq2moe_gather_quant_f16(
    __global const half *in, __global const int *src_row,
    __global char *act_int8, __global float *scale_per_row,
    __global int *zp_per_row, __global int *row_sum_act, const int M,
    const int K) {
#define LOAD(k) ((float)in[(long)src_row[row] * K + (k)])
#define LOAD2(k) LOAD(k)
  TQ2Q_ROW(LOAD, TQ2Q_NO_STORE)
#undef LOAD
#undef LOAD2
}

// The hit / miss split of a streamed decode step (one lane per selected
// expert): res[e] != 0 when expert e is resident. The GEMVs take an
// activation row per record and skip a record whose row is -1, so the
// launch over the hits gets the miss rows as -1 and the launch over the
// misses the hit rows: gate / up rows a and K + a read row 0, down row a
// reads row a.
__kernel void tq2moe_route_split(__global const uint *ids,
                                 __global const uint *res, const int K,
                                 __global int *gu_hit, __global int *gu_miss,
                                 __global int *d_hit, __global int *d_miss) {
  const int a = get_global_id(0);
  if (a >= K)
    return;
  const int hit = res[ids[a]] != 0u;
  gu_hit[a] = hit ? 0 : -1;
  gu_hit[K + a] = hit ? 0 : -1;
  gu_miss[a] = hit ? -1 : 0;
  gu_miss[K + a] = hit ? -1 : 0;
  d_hit[a] = hit ? a : -1;
  d_miss[a] = hit ? -1 : a;
}
)CL";

// The device router kernels (decode and prefill): cl_kernels/tq2moe_route.cl.
constexpr unsigned int kTq2RouteSeg = 32; /**< RT_SEG */
static_assert(kTq2RouteSeg == kMoeRouteSeg,
              "the device router segments as moe_route_det.h does");
constexpr unsigned int kTq2RouteMaxH = 4096; /**< RT_MAXH */
constexpr unsigned int kTq2RouteMaxE = 256, kTq2RouteMaxK = 16;
/** prefill router rows per segment-partial chunk: [rows][nseg][E] fp32
 *  scratch, 11.5 MB at H 2816 / E 128 */
constexpr unsigned int kTq2RoutePartRows = 256;
/** prefill router: with a hardware fp32 fma, from this many lanes (E x
 *  ceil(rows / 4)) the logits take one lane per expert and four rows over the
 *  whole hidden axis (tq2moe_route_chain_rows) instead of the segment
 *  partials and their sums. Xe3, H 2816 E 128: chain 0.33 vs 0.38 ms at 128
 *  rows (4096 lanes), 0.24 vs 0.19 ms at 64; with the emulated fma the
 *  segment form is as fast or faster at every row count measured. */
constexpr size_t kTq2RouteChainLanes = 4096;
/** ranks after the top-k the select kernel reports for streamed experts
 *  (RT_EXT): the recency hint of the slot residency */
constexpr unsigned int kTq2RouteExt = 5;

/** v8c activation quantizer build options: bit-identical to the CPU port on a
 *  DPAS device; elsewhere the options the v8c FC path itself builds the
 *  quantizer with on that device (some drivers reject the correctly-rounded
 *  divide option), so the experts quantize like the dense FCs there */
const std::string &tq2QuantOptions(ClContext *cc) {
  static const std::string dpas =
    "-DV8C_BUFFER_ONLY -cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt";
  static const std::string buffer = "-DV8C_BUFFER_ONLY -cl-std=CL3.0";
  static const std::string image = "";
  if (cc->caps().dpas)
    return dpas;
  return cc->caps().image_v8c ? image : buffer;
}
constexpr size_t kV8cQuantLws = 64; /**< V8C_QUANT_LWS of the .cl */
constexpr size_t kTq2GemvSgs = 4;   /**< TQ2_SGS of the .cl (default) */
constexpr size_t kTq2GemvLws = 16 * kTq2GemvSgs; /**< work-group size */
constexpr size_t kTq2GemvCpl = 1; /**< TQ2_CPL of the .cl (default) */
constexpr size_t kTq2pLws = 128;  /**< portable GEMV: TQ2P_LPC * TQ2P_CPW */
constexpr size_t kTq2pGLws = 64;  /**< portable grouped GEMM work-group */
constexpr unsigned int kTq2pMaxK = 8192; /**< portable GEMV: TQ2P_MAXK */
constexpr size_t kTq2pCpw = 32;          /**< portable GEMV: TQ2P_CPW */
constexpr size_t kTq2pTm = 8;            /**< portable grouped GEMM: TQ2P_TM */

/**
 * @brief Output channels per work-item of the portable grouped GEMM
 *        (TQ2P_NB of the .cl): 2 by default, NNTR_TQ2CX_GROUPED_NB=1 keeps
 *        one channel per work-item (an A/B switch; same outputs bit for bit).
 *        Read once.
 */
unsigned int tq2GroupedNb() {
  static const unsigned int nb = []() {
    const char *e = std::getenv("NNTR_TQ2CX_GROUPED_NB");
    return (e != nullptr && e[0] == '1' && e[1] == '\0') ? 1u : 2u;
  }();
  return nb;
}

/**
 * @brief NNTR_TQ2CX_FORCE_PORTABLE=1 (a test switch): take the portable
 *        kernels on a DPAS device too, so the Adreno path can be checked on
 *        an Intel host. Read once.
 */
bool tq2ForcePortable() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_TQ2CX_FORCE_PORTABLE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

/**
 * @brief The router select kernel: tq2moe_route_select by default,
 *        NNTR_G4MOE_RSEL=0 takes the one-lane reference form
 *        tq2moe_route_select_ref (an A/B switch; same outputs bit for bit).
 *        Read once.
 */
const char *tq2RouteSelectKernel() {
  static const char *name = []() {
    const char *e = std::getenv("NNTR_G4MOE_RSEL");
    return (e != nullptr && e[0] == '0') ? "tq2moe_route_select_ref"
                                         : "tq2moe_route_select";
  }();
  return name;
}

/** the subgroup / XMX kernels serve this device (else the portable ones) */
bool tq2Dpas(ClContext *cc) { return cc->caps().dpas && !tq2ForcePortable(); }

/**
 * @brief Geometry of the Intel subgroup GEMV (tq2cx_gemv_batch) and the order
 *        its activation row is stored in. The default reads the row in TQ2
 *        order (TQ2_ACT_PERM: the weight unpack is a shift and a mask, no
 *        nibble spread) with one lane per channel walking one unit at a time
 *        (TQ2_LPC=1, TQ2_UPL=1); the decode activation quantizer then stores
 *        that order directly (V8C_AQ_PERM), so no extra launch.
 *        NNTR_TQ2CX_DOT_PERM=0 keeps the earlier plain-order build (LPC 4,
 *        UPL 2), an A/B switch: the accumulation is int32 either way, so the
 *        outputs are the same bit for bit. Read once.
 */
struct Tq2GemvGeom {
  bool perm;        /**< TQ2_ACT_PERM build + permuted activation rows */
  unsigned int lpc; /**< TQ2_LPC: lanes per output channel */
  unsigned int upl; /**< TQ2_UPL: units per lane per step */
  /** channels per work-group: TQ2_CPW = 16 / LPC * CPL * SGS */
  size_t cpw() const { return 16 / lpc * kTq2GemvCpl * kTq2GemvSgs; }
  /** build options of tq2cx_gemv_batch */
  std::string options() const {
    return "-DTQ2_LPC=" + std::to_string(lpc) +
           " -DTQ2_UPL=" + std::to_string(upl) +
           (perm ? " -DTQ2_ACT_PERM" : "");
  }
};
const Tq2GemvGeom &tq2GemvGeom() {
  static const Tq2GemvGeom g = []() {
    const char *e = std::getenv("NNTR_TQ2CX_DOT_PERM");
    const bool off = e != nullptr && e[0] == '0';
    return off ? Tq2GemvGeom{false, 4, 2} : Tq2GemvGeom{true, 1, 1};
  }();
  return g;
}

/** device state of one MoE layer: its stacked expert records */
struct Tq2Stack {
  Tq2cxLayout layout; /**< stored layout of the records */
  cl_mem buf = nullptr;
  cl_mem off_tab = nullptr; /**< [3][E] record offsets: gate, up, down */
  std::vector<uint32_t> off_gate, off_up, off_down; /**< word offsets */
  unsigned int H = 0, I = 0;
  /** buf is the layer's partition of the expert slot arena (not owned):
   *  an expert's records sit in its slot, at slot * slot_words */
  bool arena = false;
  size_t gu_bytes = 0, dn_bytes = 0; /**< 64-byte rounded record bytes */
  size_t slot_bytes = 0;             /**< 2 * gu_bytes + dn_bytes */
  /** bytes from one slot to the next in the partition: slot_bytes, or with
   *  NNTR_G4MOE_FSU_FLAGS the 4 KiB pages that hold any record's O_DIRECT
   *  window (tq2SlotStride) */
  size_t slot_stride = 0;
  /** NNTR_G4MOE_FSU_FLAGS: per expert, the byte offset of its record inside
   *  its slot (its file offset % 4096: an O_DIRECT window read to the slot's
   *  start puts the record there); empty = 0 for every expert */
  std::vector<uint32_t> shift;
  /** off_tab describes the slot map of a complete partition (device
   *  routing); built once, when the layer first reports it complete */
  bool tab_complete = false;
  /** the layer's decode step whose experts are still being made resident
   *  off the host thread (Tq2AsyncJob); the next call waits for it */
  std::shared_ptr<struct Tq2AsyncJob> async;
  /** the next-layer prediction of this layer's last decode step
   *  (NNTR_G4MOE_FSU_PREGATE): its readback of pg_ext must be done before
   *  the step writes pg_ext again (the host waits for that, never a reader) */
  std::shared_ptr<struct Tq2Prediction> pg;
  cl_mem pg_ext = nullptr; /**< [K + RT_EXT] the next layer's predicted ids */
  /** hit / miss split (tq2SplitOn): [E] 1 for a resident expert, read by
   *  tq2moe_route_split; with off_tab it describes the slot map tab_slots
   *  (the async job keeps both current) */
  cl_mem res_flag = nullptr;
  std::vector<unsigned int> tab_slots;
};

/** a grow-only device scratch buffer */
struct Tq2Scratch {
  cl_mem buf = nullptr;
  size_t bytes = 0;
  cl_mem get(ClContext *cc, size_t need) {
    if (need <= bytes)
      return buf;
    if (buf != nullptr)
      opencl::clReleaseMemObjectT(buf);
    cl_int err = CL_SUCCESS;
    opencl::ClMemAcctScope acct("tq2cx_moe_scratch");
    buf = opencl::clCreateBufferT(cc->context_inst_.GetContextNoRetain(),
                                  CL_MEM_READ_WRITE, need, nullptr, &err);
    NNTR_THROW_IF(err != CL_SUCCESS || buf == nullptr, std::runtime_error)
      << "tq2cx_moe_experts_cl: scratch clCreateBuffer(" << need << ") failed ("
      << err << ")";
    bytes = need;
    return buf;
  }
};

struct Tq2Workspace {
  Tq2Scratch xf;  /**< gathered fp32 rows [R][H] */
  Tq2Scratch xq;  /**< int8 rows [R][max(H, I)] */
  Tq2Scratch xs;  /**< per-row scale */
  Tq2Scratch xz;  /**< per-row zero point */
  Tq2Scratch xrs; /**< per-row sum (quantizer output; read by
                       tq2cx_gemm_tiles_v2_portable) */
  Tq2Scratch yg;  /**< gate [R][I] (decode: gate and up, [2k][I]) */
  Tq2Scratch yu;  /**< up [R][I] */
  Tq2Scratch act; /**< GeGLU [R][I] */
  Tq2Scratch yd;  /**< down [R][H] */
  // device routing of a decode row (tq2moe_route_logits / _select)
  Tq2Scratch r_ids;    /**< [K] selected expert ids, ascending */
  Tq2Scratch r_w;      /**< [K] their routing weights */
  Tq2Scratch r_off_gu; /**< [2K] gate then up record offsets */
  Tq2Scratch r_off_d;  /**< [K] down record offsets */
  Tq2Scratch r_part;   /**< [segments][E] partial router logits */
  Tq2Scratch r_y;      /**< [H] normalised, scaled router row */
  Tq2Scratch r_ext;    /**< [K + RT_EXT] ids by rank (streamed) */
  // hit / miss split of a streamed decode step (tq2moe_route_split)
  Tq2Scratch s_gu_hit, s_gu_miss; /**< [2K] act rows, -1 = skip */
  Tq2Scratch s_d_hit, s_d_miss;   /**< [K] act rows, -1 = skip */
  Tq2Scratch s_off_gu, s_off_d;   /**< the misses' offsets (the job's) */
  /** the next-layer prediction's select outputs (not read) */
  Tq2Scratch p_ids, p_w, p_off_gu, p_off_d;
  // device routing of prefill rows (tq2moe_route_*_rows)
  Tq2Scratch r_acc;         /**< [T][32] lane sums of squares */
  Tq2Scratch r_yrows;       /**< [T][H] normalised, scaled router rows */
  Tq2Scratch r_logits;      /**< [T][E] logits, then probabilities */
  Tq2Scratch r_ex;          /**< [T][E] exponentials */
  Tq2Scratch r_ppart;       /**< [chunk rows][nseg][E] segment partials */
  Tq2Scratch r_out;         /**< [T][2K] ids then weight bits, rank order */
  unsigned int const_k = 0; /**< K the constants below were built for */
  cl_mem c_zero = nullptr;  /**< [2K] zeros (the one activation row) */
  cl_mem c_iota = nullptr;  /**< [K] 0..K-1 */
  cl_mem c_count = nullptr; /**< [1] K */
  /** image views of the int8 rows xq for tq2cx_gemm_tiles_v2_portable, one
   *  per row length K (gate / up read H, down reads I) */
  struct XqView {
    cl_mem buf = nullptr; /**< the xq buffer the view was made over */
    unsigned int K = 0;
    int form = 0;    /**< 1 image2d, 2 image1d_buffer */
    size_t rows = 0; /**< activation rows the view covers */
    cl_mem img = nullptr;
  };
  XqView xq_view[2];
  /** release the image views of xq (each one keeps its buffer alive) */
  void releaseXqViews() {
    for (auto &v : xq_view) {
      if (v.img != nullptr)
        opencl::clReleaseMemObjectT(v.img);
      v = XqView{};
    }
  }
  /** xq with at least @a need bytes; growing it drops the views of the old
   *  buffer first, so that buffer is freed now and not when a later prefill
   *  replaces the view. The workspace is a process-lifetime static with no
   *  teardown hook (nor do its scratch buffers have one). */
  cl_mem xqGet(ClContext *cc, size_t need) {
    if (need > xq.bytes)
      releaseXqViews();
    return xq.get(cc, need);
  }
};

std::unordered_map<const void *, std::unique_ptr<Tq2Stack>> &tq2Stacks() {
  static std::unordered_map<const void *, std::unique_ptr<Tq2Stack>> m;
  return m;
}

Tq2Workspace &tq2Workspace() {
  static Tq2Workspace w;
  return w;
}

/** the quantized GeGLU rows of a split decode step: kept apart from the
 *  workspace's, whose row 0 (the step's input) the misses' gate / up GEMVs
 *  still read after the hits' down GEMV */
Tq2Workspace &tq2WorkspaceDown() {
  static Tq2Workspace w;
  return w;
}

size_t tq2RecordBytes(unsigned int N, unsigned int K) {
  return tq2cxRecordBytes(tq2cxStoredLayout(), N, K);
}

/**
 * @brief The layer's stacked expert records: built at load by
 *        tq2cx_moe_experts_prebuild_cl, else on first use. The bytes come
 *        from g.records when given (the file mapping), else the weights.
 */
std::unique_ptr<Tq2Stack> tq2BuildStack(ClContext *cc,
                                        const MoeExpertGroup &g) {
  auto st = std::make_unique<Tq2Stack>();
  st->layout = tq2cxStoredLayout();
  const size_t E = g.gate.size();
  st->H = g.gate[0]->height();
  st->I = g.gate[0]->width();
  const size_t gu_bytes = (tq2RecordBytes(st->I, st->H) + 63) / 64 * 64;
  const size_t dn_bytes = (tq2RecordBytes(st->H, st->I) + 63) / 64 * 64;
  const size_t total = E * (2 * gu_bytes + dn_bytes);
  st->off_gate.resize(E);
  st->off_up.resize(E);
  st->off_down.resize(E);

  cl_int err = CL_SUCCESS;
  {
    opencl::ClMemAcctScope acct("tq2cx_moe_experts");
    st->buf = opencl::clCreateBufferT(cc->context_inst_.GetContextNoRetain(),
                                      CL_MEM_READ_ONLY, total, nullptr, &err);
  }
  NNTR_THROW_IF(err != CL_SUCCESS || st->buf == nullptr, std::runtime_error)
    << "tq2cx_moe_experts_cl: expert stack clCreateBuffer(" << total
    << ") failed (" << err << ")";
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  size_t pos = 0;
  auto put = [&](const Tensor *w, size_t bytes, size_t slot) {
    const size_t rec = tq2RecordBytes(w->width(), w->height());
    const void *src = g.records != nullptr
                        ? (*g.records)[slot]
                        : static_cast<const void *>(w->getData<uint8_t>());
    if (g.records == nullptr)
      refuseIfTq2cxPayloadReleased(src, "tq2cx_moe_experts_cl");
    err = opencl::clEnqueueWriteBuffer(q, st->buf, CL_TRUE, pos, rec, src, 0,
                                       nullptr, nullptr);
#if !defined(_WIN32)
    // A record read from the file mapping is not needed on the host again:
    // let its (clean, re-faultable) pages go now rather than when the loader
    // unmaps the whole file, so the uploads do not stack up in the RSS.
    if (g.records != nullptr && err == CL_SUCCESS) {
      const uintptr_t page = 4096;
      const uintptr_t lo = ((uintptr_t)src + page - 1) & ~(page - 1);
      const uintptr_t hi = ((uintptr_t)src + rec) & ~(page - 1);
      if (hi > lo)
        (void)::posix_madvise((void *)lo, (size_t)(hi - lo),
                              POSIX_MADV_DONTNEED);
    }
#endif
    NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
      << "tq2cx_moe_experts_cl: expert upload failed (" << err << ")";
    const uint32_t off = static_cast<uint32_t>(pos / sizeof(uint32_t));
    pos += bytes;
    return off;
  };
  for (size_t e = 0; e < E; ++e) {
    st->off_gate[e] = put(g.gate[e], gu_bytes, 3 * e);
    st->off_up[e] = put(g.up[e], gu_bytes, 3 * e + 1);
    st->off_down[e] = put(g.down[e], dn_bytes, 3 * e + 2);
  }
  std::vector<uint32_t> tab;
  tab.reserve(3 * E);
  tab.insert(tab.end(), st->off_gate.begin(), st->off_gate.end());
  tab.insert(tab.end(), st->off_up.begin(), st->off_up.end());
  tab.insert(tab.end(), st->off_down.begin(), st->off_down.end());
  st->off_tab = tq2Upload(cc, tab.data(), tab.size() * sizeof(uint32_t));
  return st;
}

// ---------------------------------------------------------------------------
// NNTR_G4MOE_FSU_FLAGS: the streamed decode without the host round trip
// ---------------------------------------------------------------------------

ClContext::SharedPtrClKernel tq2FlagsKernel(ClContext *cc, const char *name);
double tq2FlagsCalibratePoll(ClContext *cc);

/** ns per gate poll on this device (tq2FlagsCalibratePoll at the decision;
 *  0 when not measured): the spin account's polls in ms */
std::atomic<double> g_tq2_flags_poll_ns{0.0};

/** the decision of tq2FsuFlagsDecide: -1 not made yet, 0 off, 1 on, 2 on
 *  in the coarse-grain test mode */
std::atomic<int> g_tq2_fsu_flags{-1};

/** NNTR_G4MOE_FSU_FLAGS is on (decided, see tq2FsuFlagsDecide) */
bool tq2FsuFlagsOn() {
  return g_tq2_fsu_flags.load(std::memory_order_acquire) >= 1;
}

/** ... in its coarse-grain test mode (NNTR_G4MOE_FSU_FLAGS=coarse) */
bool tq2FsuFlagsCoarse() {
  return g_tq2_fsu_flags.load(std::memory_order_acquire) == 2;
}

/** O_DIRECT granule of the flags protocol's slot reads */
constexpr size_t kTq2FlagsPage = 4096;

/** the alignment a record needs in its slot under NNTR_G4MOE_FSU_FLAGS: the
 *  word the portable GEMVs read (4), 64 for the DPAS kernels; set with the
 *  decision */
std::atomic<size_t> g_tq2_flags_align{64};

/** the auto rule of NNTR_G4MOE_FSU_FLAGS: on from this many cache experts
 *  per streamed layer (Adreno 840 phones, 1K decode: a gain at C32 / C40,
 *  and at C16 with 3 readers -- 60.7 vs 73.8 ms/token with the flags off) */
constexpr unsigned int kTq2FlagsAutoMinSlots = 16;

/** NNTR_G4MOE_FSU_FLAGS_AUTO_MIN_C=n (default kTq2FlagsAutoMinSlots): the
 *  auto rule's least cache experts per layer. Read once. */
unsigned int tq2FlagsAutoMinSlots() {
  static const unsigned int v = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_FLAGS_AUTO_MIN_C");
    if (e == nullptr || *e == '\0')
      return kTq2FlagsAutoMinSlots;
    char *end = nullptr;
    const unsigned long n = std::strtoul(e, &end, 10);
    return (end != e && *end == '\0' && n >= 1 && n <= 4096)
             ? (unsigned int)n
             : kTq2FlagsAutoMinSlots;
  }();
  return v;
}

/** the reader threads the flags protocol wants on Android when
 *  NNTR_MOE_READERS is unset (2 readers queue the demand reads behind the
 *  reads ahead; 3 gave the best flags arm on two Adreno 840 phones) */
constexpr unsigned int kTq2FlagsAndroidReaders = 3;

/**
 * @brief NNTR_G4MOE_FSU_FLAGS=auto|0|1|force|coarse (default auto): the
 *        streamed experts' slots are fine-grain SVM with atomics and the
 *        decode step publishes its misses to the host through it (see
 *        tq2cx_moe_experts_cl). Decided once, at the first streamed layer
 *        (before any slot exists: the slot memory and stride depend on it),
 *        from @a g's shapes and offsets.
 * @details auto (unset): on when @a g streams (fewer slots than experts)
 * with at least kTq2FlagsAutoMinSlots slots, the device is an Adreno GPU and
 * reports fine-grain buffer SVM with SVM atomics, and the conditions of 1
 * below hold; off otherwise (one log line when @a g streams). 1: on unless
 * refused (logged, the existing path kept): the device must be an Adreno GPU
 * (force: any vendor), report fine-grain buffer SVM with SVM atomics, the
 * loader must have clSVMAlloc and clSetKernelArgSVMPointer, and the expert
 * records must sit in the file as they sit in a slot (gate, up, down 64-byte
 * multiples, back to back). Xe3 (NEO 26.22) has coarse-grain SVM only:
 * refused there, and auto is off.
 * When it is on on Android and NNTR_MOE_READERS is unset, the expert reader
 * pool grows to kTq2FlagsAndroidReaders threads.
 * The records' file offsets need no alignment: an O_DIRECT read puts a
 * record at its slot + (offset % 4096); one whose offset is not a multiple
 * of the alignment the GEMVs need (4 bytes for the portable kernels; the
 * 26B pack's layer l starts at 2 l mod 64, so its odd layers are 2 mod 4)
 * is moved down by offset % that alignment on the reader thread after its
 * read (tq2SlotShift / Tq2Flags::issue).
 * coarse: a TEST MODE for a host without fine-grain SVM -- the same protocol
 * on coarse-grain SVM, which the host and a running kernel then share
 * outside the OpenCL memory model (an Intel iGPU shares the memory
 * coherently in practice); for checking the protocol end to end on such a
 * host, never a default.
 * @return the decision
 */
bool tq2FsuFlagsDecide(ClContext *cc, const MoeExpertGroup &g) {
  static std::once_flag once;
  std::call_once(once, [cc, &g]() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_FLAGS");
    const std::string v = e != nullptr && e[0] != '\0' ? e : "auto";
    if (v != "auto" && v != "1" && v != "force" && v != "coarse") {
      if (v != "0")
        std::fprintf(stderr,
                     "[TQ2CX-FSU-FLAGS] NNTR_G4MOE_FSU_FLAGS=%s: not "
                     "auto|0|1|force|coarse, off\n",
                     e);
      g_tq2_fsu_flags.store(0, std::memory_order_release);
      return;
    }
    const bool automatic = v == "auto";
    const bool streams = g.fsu != nullptr && g.fsu->slots < g.gate.size();
    if (automatic && !streams) {
      // every expert resident: nothing to stream, the slots stay as they are
      g_tq2_fsu_flags.store(0, std::memory_order_release);
      return;
    }
    std::string why;
    const cl_device_id dev = cc->context_inst_.GetDeviceId();
    constexpr uint32_t qualcomm_vendor_id = 0x5143;
    cl_device_svm_capabilities caps = 0;
    if (opencl::clGetDeviceInfo == nullptr ||
        opencl::clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(caps),
                                &caps, nullptr) != CL_SUCCESS)
      caps = 0;
    const bool coarse = v == "coarse";
    const cl_device_svm_capabilities need =
      coarse ? CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
             : CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS;
    if ((v == "1" || automatic) && cc->caps().vendor_id != qualcomm_vendor_id) {
      char b[96];
      std::snprintf(b, sizeof(b), "not an Adreno GPU (vendor 0x%x)",
                    (unsigned int)cc->caps().vendor_id);
      why = b;
    } else if (automatic && g.fsu->slots < tq2FlagsAutoMinSlots()) {
      char b[96];
      std::snprintf(b, sizeof(b), "%u cache experts per layer (auto needs %u)",
                    g.fsu->slots, tq2FlagsAutoMinSlots());
      why = b;
    } else if ((caps & need) != need) {
      char b[96];
      std::snprintf(
        b, sizeof(b), "svm caps 0x%llx lack %s", (unsigned long long)caps,
        coarse ? "buffer SVM" : "fine-grain buffer SVM with atomics");
      why = b;
    } else if (opencl::clSVMAlloc == nullptr ||
               opencl::clSetKernelArgSVMPointer == nullptr) {
      why = "the OpenCL loader has no clSVMAlloc / clSetKernelArgSVMPointer";
    } else if (g.fsu == nullptr || g.fsu->offset == nullptr || g.gate.empty()) {
      why = "no streamed expert descriptor";
    } else {
      const unsigned int H = g.gate[0]->height(), I = g.gate[0]->width();
      const size_t rec_gu = tq2RecordBytes(I, H), rec_dn = tq2RecordBytes(H, I);
      if (rec_gu % 64 != 0 || rec_dn % 64 != 0 ||
          g.fsu->record_bytes != 2 * rec_gu + rec_dn)
        why = "the expert records are not laid out as a slot";
    }
    if (why.empty() && (!tq2FlagsKernel(cc, "tq2moe_fsu_publish") ||
                        !tq2FlagsKernel(cc, "tq2moe_fsu_spec") ||
                        !tq2FlagsKernel(cc, "tq2moe_fsu_gate")))
      why = "the flags kernels do not build (OpenCL C 2.0 / 3.0 atomics)";
    if (!why.empty()) {
      if (automatic)
        std::fprintf(stderr,
                     "[TQ2CX-FSU-FLAGS] auto: off (%s); the existing decode "
                     "path is kept\n",
                     why.c_str());
      else
        std::fprintf(stderr,
                     "[TQ2CX-FSU-FLAGS] refused: %s; the existing decode "
                     "path is kept\n",
                     why.c_str());
      g_tq2_fsu_flags.store(0, std::memory_order_release);
      return;
    }
    if (coarse)
      std::fprintf(stderr,
                   "[TQ2CX-FSU-FLAGS] on, TEST MODE: coarse-grain SVM (svm "
                   "caps 0x%llx) shared with running kernels outside the "
                   "OpenCL memory model; each record at its O_DIRECT offset "
                   "in a 4 KiB-page slot\n",
                   (unsigned long long)caps);
    else
      std::fprintf(stderr,
                   "[TQ2CX-FSU-FLAGS] on%s: expert slots in fine-grain SVM "
                   "with atomics (svm caps 0x%llx), each record at its "
                   "O_DIRECT offset in a 4 KiB-page slot\n",
                   automatic ? " (auto)" : "", (unsigned long long)caps);
    if (!coarse) {
      // the gates' spin account is in polls: their time on this device
      g_tq2_flags_poll_ns.store(tq2FlagsCalibratePoll(cc));
      std::fprintf(stderr,
                   "[TQ2CX-FSU-FLAGS] gate poll %.1f ns (measured at load; "
                   "0 = not measured)\n",
                   g_tq2_flags_poll_ns.load());
    }
#if defined(__ANDROID__)
    if (std::getenv("NNTR_MOE_READERS") == nullptr) {
      ExpertReaderPool &pool = ExpertReaderPool::global();
      pool.ensureReaders(kTq2FlagsAndroidReaders);
      std::fprintf(stderr,
                   "[TQ2CX-FSU-FLAGS] expert readers %u (flags default on "
                   "Android; NNTR_MOE_READERS sets it)\n",
                   pool.readers());
    }
#endif
    g_tq2_flags_align.store(tq2Dpas(cc) ? 64 : 4);
    g_tq2_fsu_flags.store(coarse ? 2 : 1, std::memory_order_release);
  });
  return tq2FsuFlagsOn();
}

/**
 * @brief Bytes from one slot to the next for records of @a slot_bytes:
 *        @a slot_bytes, or with NNTR_G4MOE_FSU_FLAGS the 4 KiB pages that
 *        hold the O_DIRECT window of a record at any file offset (it starts
 *        at most 4095 bytes into its first page; 369 pages for the 26B
 *        pack's 1,503,744-byte records).
 */
size_t tq2SlotStride(size_t slot_bytes) {
  if (!tq2FsuFlagsOn())
    return slot_bytes;
  return (slot_bytes + 2 * (kTq2FlagsPage - 1)) / kTq2FlagsPage * kTq2FlagsPage;
}

/** the offset of a record inside its slot with NNTR_G4MOE_FSU_FLAGS: where
 *  an O_DIRECT window read to the slot's start puts it (file offset @a off
 *  % 4096), rounded down to the alignment the GEMVs need (the reader then
 *  moves it down by tq2SlotMove); 0 without */
size_t tq2SlotShift(uint64_t off) {
  return tq2FsuFlagsOn()
           ? (size_t)(off % kTq2FlagsPage) & ~(g_tq2_flags_align.load() - 1)
           : 0;
}

/** the bytes a record read to its slot's start must move down to its place
 *  (tq2SlotShift): file offset @a off % the GEMVs' alignment */
size_t tq2SlotMove(uint64_t off) {
  return tq2FsuFlagsOn() ? (size_t)(off % g_tq2_flags_align.load()) : 0;
}

/** tq2SlotShift of expert @a e of descriptor @a d */
size_t tq2SlotShift(const MoeExpertFsu &d, unsigned int e) {
  return d.offset != nullptr && e < d.offset->size()
           ? tq2SlotShift((*d.offset)[e])
           : 0;
}

// ---------------------------------------------------------------------------
// Expert slot arena: experts read from the weight file at run time
// ---------------------------------------------------------------------------

/** an event several slots may share (the last-use marker of one call) */
struct Tq2Event {
  cl_event ev = nullptr;
  explicit Tq2Event(cl_event e) : ev(e) {}
  ~Tq2Event() {
    if (ev != nullptr)
      opencl::clReleaseEvent(ev);
  }
  Tq2Event(const Tq2Event &) = delete;
  Tq2Event &operator=(const Tq2Event &) = delete;
};
using Tq2EventRef = std::shared_ptr<Tq2Event>;

/** a host staging buffer of the fill path and the write still reading it */
struct Tq2Bounce {
  AlignedBuffer buf;
  Tq2EventRef write;
};

/**
 * @brief The expert slots of the MoE layers whose experts are read from the
 *        weight file (MoeExpertGroup::fsu): one partition of C slots per
 *        layer, each a CL_MEM_READ_ONLY buffer of C * slot_bytes.
 *
 * @details An expert's gate, up and down records sit in its slot exactly as
 * in a resident stack (tq2BuildStack), so the kernels are the resident ones
 * with the partition buffer and slot word offsets. A partition is one buffer
 * so that every launch of a layer binds one cl_mem; the arena is therefore
 * split across buffers at partition granularity and each partition must fit
 * CL_DEVICE_MAX_MEM_ALLOC_SIZE.
 *
 * A fill is staged: the reader pool reads the record (O_DIRECT superset where
 * possible) into a page-aligned host bounce, then a non-blocking
 * clEnqueueWriteBuffer on a separate in-order I/O queue copies it into the
 * slot. Two fences per slot:
 *  - ready: the write's event. The compute queue waits on it (a barrier
 *    with the event in its wait list, ahead of the call's kernels) before a
 *    kernel reads the slot; the in-order compute queue then keeps every
 *    later kernel behind that barrier, so the event is dropped.
 *  - last_use: a marker on the compute queue after the last kernel of a call
 *    that read the slot (the down projection). A write into the slot carries
 *    it in its wait list, so an evicted slot is never overwritten while a
 *    queued kernel still reads it. Recorded only for a partition with fewer
 *    slots than experts (one that evicts).
 */
class Tq2ExpertArena {
public:
  struct Partition {
    cl_mem buf = nullptr;
    unsigned int slots = 0;
    size_t slot_bytes = 0;
    unsigned int index = UINT32_MAX; /**< MoeExpertFsu::layer_index */
    /** a sub-buffer partition: the block buffer it is a window of (not
     *  owned) and its origin there; null for a buffer of its own */
    cl_mem block = nullptr;
    size_t block_origin = 0;
    /** NNTR_G4MOE_FSU_ARENA=svm: the partition's SVM memory (buf is a
     *  CL_MEM_USE_HOST_PTR buffer over it); slot writes are CPU copies into
     *  it (slotWrite), coarse-grain ones between a map and an unmap of the
     *  written range. Null for a device buffer (clmem) */
    uint8_t *host = nullptr;
    bool fine = false;                 /**< fine-grain SVM: no map / unmap */
    std::vector<Tq2EventRef> ready;    /**< per slot: write not yet waited */
    std::vector<Tq2EventRef> last_use; /**< per slot: last reader marker */
    /** guards last_use: the host thread records it, a reader thread that
     *  writes a read-ahead record into a reserved slot (a short prefill's
     *  next-layer read ahead) reads it */
    std::mutex lu_mu;
    Tq2EventRef lastUse(unsigned int slot) {
      std::lock_guard<std::mutex> lk(lu_mu);
      return slot < last_use.size() ? last_use[slot] : Tq2EventRef();
    }
    std::vector<Tq2EventRef> lastUses() {
      std::lock_guard<std::mutex> lk(lu_mu);
      return last_use;
    }
  };

  /** the process's arena; never destroyed (its buffers and events belong
   *  to the OpenCL context, which may be gone at static destruction) */
  static Tq2ExpertArena &get() {
    static Tq2ExpertArena *a = new Tq2ExpertArena();
    return *a;
  }

  /**
   * @brief The partition of @a key, created with @a slots slots of
   *        @a slot_bytes on first use.
   *
   * @details With @a experts (the layer's E > slots) and a layer index below
   * 2 * ceil(E / slots), the partition is a sub-buffer of one of two block
   * buffers of ceil(E / slots) partitions each, in layer index order: the
   * partitions the prefill stream borrows (streamLend: the lowest indices)
   * then form two buffers in which expert e of a streamed layer sits in slot
   * e, so a streamed prefill call binds one buffer for all its experts and
   * runs the resident launch sequence (flat tiles, tiles-v2, fused
   * quantizers). The memory is the same: a block buffer replaces the
   * partitions it holds. Falls back to a buffer per partition when the
   * block exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE, a partition origin is not a
   * multiple of CL_DEVICE_MEM_BASE_ADDR_ALIGN, the driver refuses the
   * sub-buffer, or NNTR_G4MOE_FSU_BLOCK=0 (the per-partition chunk path of
   * the stream).
   */
  Partition &partition(ClContext *cc, const void *key, unsigned int slots,
                       size_t slot_bytes, unsigned int index = UINT32_MAX,
                       unsigned int experts = 0) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = parts_.find(key);
    if (it != parts_.end()) {
      NNTR_THROW_IF(it->second->slots != slots ||
                      it->second->slot_bytes != slot_bytes,
                    std::runtime_error)
        << "tq2cx expert arena: a layer asked for " << slots << " x "
        << slot_bytes << " B slots, it has " << it->second->slots << " x "
        << it->second->slot_bytes;
      return *it->second;
    }
    const size_t bytes = (size_t)slots * slot_bytes;
    const opencl::DeviceInfo *info =
      opencl::ContextManager::Global().getDeviceInfo();
    const size_t max_alloc =
      info != nullptr ? (size_t)info->getDeviceMaxMemAllocSize() : 0;
    NNTR_THROW_IF(max_alloc != 0 && bytes > max_alloc, std::runtime_error)
      << "tq2cx expert arena: " << slots << " slots of " << slot_bytes << " B ("
      << bytes << " B) exceed CL_DEVICE_MAX_MEM_ALLOC_SIZE " << max_alloc
      << "; lower the expert cache size";
    auto p = std::make_unique<Partition>();
    p->slots = slots;
    p->slot_bytes = slot_bytes;
    p->index = index;
    p->ready.resize(slots);
    p->last_use.resize(slots);
    cl_int err = CL_SUCCESS;
    if (experts > slots && index != UINT32_MAX && blockOn()) {
      const unsigned int n = (experts + slots - 1) / slots;
      if (index < 2 * n)
        p->buf = blockPartition(cc, slots, slot_bytes, n, index, max_alloc,
                                &p->block, &p->block_origin, &p->host);
    }
    if (p->buf == nullptr) {
      opencl::ClMemAcctScope acct("tq2cx_moe_arena");
      p->buf = allocArena(cc, bytes, &p->host, &err);
    }
    p->fine = p->host != nullptr && svmKind(cc) == 2;
    NNTR_THROW_IF(err != CL_SUCCESS || p->buf == nullptr, std::runtime_error)
      << "tq2cx expert arena: clCreateBuffer(" << bytes << ") failed (" << err
      << ")";
    total_slots_ += slots;
    Partition &ref = *parts_.emplace(key, std::move(p)).first->second;
    if (index != UINT32_MAX)
      by_index_[index] = &ref;
    return ref;
  }

  /** the partition of layer index @a index, or null */
  Partition *byIndex(unsigned int index) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = by_index_.find(index);
    return it == by_index_.end() ? nullptr : it->second;
  }

  /**
   * @brief A second in-order I/O queue: the slot writes of the read-ahead
   *        records (a short prefill's next-layer read ahead), so a step's
   *        own writes on ioQueue() never queue behind one. Nothing names an
   *        event of one queue on the other.
   */
  cl_command_queue ioQueueLow(ClContext *cc) {
    std::lock_guard<std::mutex> lk(mu_);
    if (io_low_ == nullptr) {
      cl_int err = CL_SUCCESS;
      io_low_ =
        opencl::clCreateCommandQueue(cc->context_inst_.GetContextNoRetain(),
                                     cc->context_inst_.GetDeviceId(), 0, &err);
      NNTR_THROW_IF(err != CL_SUCCESS || io_low_ == nullptr, std::runtime_error)
        << "tq2cx expert arena: low I/O queue creation failed (" << err << ")";
    }
    return io_low_;
  }

  /** the separate in-order queue the fills are written on */
  cl_command_queue ioQueue(ClContext *cc) {
    std::lock_guard<std::mutex> lk(mu_);
    if (io_q_ == nullptr) {
      cl_int err = CL_SUCCESS;
      io_q_ =
        opencl::clCreateCommandQueue(cc->context_inst_.GetContextNoRetain(),
                                     cc->context_inst_.GetDeviceId(), 0, &err);
      NNTR_THROW_IF(err != CL_SUCCESS || io_q_ == nullptr, std::runtime_error)
        << "tq2cx expert arena: I/O queue creation failed (" << err << ")";
    }
    return io_q_;
  }

  /**
   * @brief Read @a fills ((expert, slot) pairs) of @a part from the file and
   *        write them into their slots. Returns once every write is queued;
   *        each slot's ready event is set. @a bounces is the staging set (at
   *        least two); a bounce is reused only after its previous write ran.
   *        With @a wait_all the call also waits for every write (and leaves
   *        no ready event behind): the bounces may be freed afterwards.
   */
  void fill(ClContext *cc, Partition &part, const Tq2Stack &st,
            const MoeExpertFsu &fsu,
            const std::vector<std::pair<unsigned int, unsigned int>> &fills,
            std::vector<Tq2Bounce> &bounces, bool wait_all) {
    if (fills.empty())
      return;
    NNTR_THROW_IF(bounces.size() < 2, std::invalid_argument)
      << "tq2cx expert arena: at least two staging buffers";
    const size_t rec = fsu.record_bytes;
    const size_t rec_gu = tq2RecordBytes(st.I, st.H);
    const bool packed = rec == st.slot_bytes; // file record == slot layout
    cl_command_queue io = ioQueue(cc);
    cl_command_queue cq = cc->command_queue_inst_.GetCommandQueue();
    ExpertReaderPool &pool = ExpertReaderPool::global();
    const size_t half = bounces.size() / 2;
    for (auto &b : bounces)
      b.buf.reserve(ExpertReaderPool::bounceBytes(rec));

    // a write follows the slot's last reader, a marker on the compute
    // queue: hand that queue to the device first (the marker is waited for
    // on the host below, never named in a wait list)
    opencl::clFlush(cq);

    if (part.host != nullptr) {
      // svm arena: each reader copies its record into the slot itself (the
      // CPU copies run in parallel, not one by one on this thread); every
      // slot's last reader is waited for here first, the writes name nothing
      std::vector<ExpertRead> reads(fills.size());
      for (size_t i = 0; i < fills.size(); ++i) {
        const unsigned int e = fills[i].first, slot = fills[i].second;
        NNTR_THROW_IF(e >= fsu.offset->size() || slot >= part.slots,
                      std::out_of_range)
          << "tq2cx expert arena: expert " << e << " slot " << slot;
        const Tq2EventRef lu = part.lastUse(slot);
        if (lu && lu->ev != nullptr) {
          cl_event ev = lu->ev;
          const cl_int werr = opencl::clWaitForEvents(1, &ev);
          NNTR_THROW_IF(werr != CL_SUCCESS, std::runtime_error)
            << "tq2cx expert arena: slot " << slot << "'s last reader failed ("
            << werr << ")";
        }
        ExpertRead &r = reads[i];
        r.fd = fsu.fd;
        r.offset = (*fsu.offset)[e];
        r.bytes = rec;
        Partition *pp = &part;
        const size_t at = (size_t)slot * part.slot_bytes + tq2SlotShift(fsu, e);
        const size_t gu = st.gu_bytes;
        r.consume = [io, pp, at, packed, rec_gu, gu](const uint8_t *src,
                                                     const ExpertRead &q) {
          if (packed)
            return slotWrite(io, *pp, at, q.bytes, src) == CL_SUCCESS ? 0 : EIO;
          const size_t len[3] = {rec_gu, rec_gu, q.bytes - 2 * rec_gu};
          const size_t dst[3] = {0, gu, 2 * gu};
          const size_t from[3] = {0, rec_gu, 2 * rec_gu};
          for (int k = 0; k < 3; ++k)
            if (slotWrite(io, *pp, at + dst[k], len[k], src + from[k]) !=
                CL_SUCCESS)
              return EIO;
          return 0;
        };
      }
      pool.wait(pool.submit(&reads), /*help=*/true); // throws on a failure
      for (const auto &f : fills) {
        part.ready[f.second].reset();
        std::lock_guard<std::mutex> lk(part.lu_mu);
        part.last_use[f.second].reset();
      }
      filled_ += fills.size();
      return;
    }

    // two banks of staging buffers: the reads of the next group run while
    // this group is written. The groups stay in place (the pool holds a
    // pointer to their read lists until they are waited).
    struct Group {
      size_t first = 0, count = 0;
      std::vector<ExpertRead> reads;
      ExpertReaderPool::Ticket ticket = 0;
      bool pending = false;
    };
    Group grp[2];
    auto issue = [&](size_t bank, size_t first) {
      Group &g = grp[bank];
      g.first = first;
      g.count = std::min(half, fills.size() - first);
      g.reads.assign(g.count, ExpertRead());
      for (size_t i = 0; i < g.count; ++i) {
        Tq2Bounce &b = bounces[bank * half + i];
        if (b.write) { // the previous write from this bounce must be done
          opencl::clWaitForEvents(1, &b.write->ev);
          b.write.reset();
        }
        const unsigned int e = fills[first + i].first;
        NNTR_THROW_IF(e >= fsu.offset->size(), std::out_of_range)
          << "tq2cx expert arena: expert " << e;
        ExpertRead &r = g.reads[i];
        r.fd = fsu.fd;
        r.offset = (*fsu.offset)[e];
        r.bytes = rec;
        r.bounce = &b.buf;
      }
      g.ticket = pool.submit(&g.reads);
      g.pending = true;
    };
    auto write = [&](size_t bank) {
      Group &g = grp[bank];
      g.pending = false;
      pool.wait(g.ticket); // throws on a failed read
      for (size_t i = 0; i < g.count; ++i) {
        const unsigned int slot = fills[g.first + i].second;
        NNTR_THROW_IF(slot >= part.slots, std::out_of_range)
          << "tq2cx expert arena: slot " << slot << " of " << part.slots;
        Tq2Bounce &b = bounces[bank * half + i];
        const uint8_t *src = b.buf.data() + g.reads[i].data_offset;
        const size_t at = (size_t)slot * part.slot_bytes +
                          tq2SlotShift(fsu, fills[g.first + i].first);
        // No cross-queue wait list (the rule of the decode job and of the
        // prefill stream): a write on the I/O queue that names a compute-
        // queue event returns on Adreno only once the compute queue has run
        // everything runnable (a probe self-test), and with a user event
        // pending on the compute queue it did not return at all (a decode
        // deadlock). The slot's last reader is therefore waited
        // for here, on the host, and the write names nothing. That wait
        // ends: this path (the load, the blocking ASYNC=0 step, a prefill's
        // chunks) runs with no user event pending on the compute queue --
        // the steps before it ended with their join or the token's blocking
        // logits read -- and the queue was flushed above; the marker is
        // normally complete already (a status read, no wait).
        const Tq2EventRef lu_ref = part.lastUse(slot);
        if (lu_ref && lu_ref->ev != nullptr) {
          cl_event lu = lu_ref->ev;
          cl_int est = CL_COMPLETE;
          if (opencl::clGetEventInfo == nullptr ||
              opencl::clGetEventInfo(lu, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                     sizeof(est), &est,
                                     nullptr) != CL_SUCCESS ||
              est > CL_COMPLETE) {
            const cl_int werr = opencl::clWaitForEvents(1, &lu);
            NNTR_THROW_IF(werr != CL_SUCCESS, std::runtime_error)
              << "tq2cx expert arena: slot " << slot
              << "'s last reader failed (" << werr << ")";
            fence_waits_++;
          }
          NNTR_THROW_IF(est < 0, std::runtime_error)
            << "tq2cx expert arena: slot " << slot << "'s last reader failed ("
            << est << ")";
        }
        cl_event ev = nullptr;
        cl_int err = CL_SUCCESS;
        if (part.host != nullptr) {
          // svm arena: CPU copies, done when this returns (no event)
          if (packed) {
            err = slotWrite(io, part, at, rec, src);
          } else {
            const size_t rec_dn = rec - 2 * rec_gu;
            const size_t dst[3] = {0, st.gu_bytes, 2 * st.gu_bytes};
            const size_t from[3] = {0, rec_gu, 2 * rec_gu};
            const size_t len[3] = {rec_gu, rec_gu, rec_dn};
            for (int p = 0; p < 3 && err == CL_SUCCESS; ++p)
              err = slotWrite(io, part, at + dst[p], len[p], src + from[p]);
          }
          NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
            << "tq2cx expert arena: slot write failed (" << err << ")";
          b.write.reset();
          part.ready[slot].reset();
          std::lock_guard<std::mutex> lk(part.lu_mu);
          part.last_use[slot].reset();
          continue;
        }
        if (packed) {
          err = opencl::clEnqueueWriteBuffer(io, part.buf, CL_FALSE, at, rec,
                                             src, 0, nullptr, &ev);
        } else {
          // records not 64-byte multiples: gate, up, down at their rounded
          // places in the slot (in-order queue: the last event covers all)
          const size_t rec_dn = rec - 2 * rec_gu;
          const size_t dst[3] = {0, st.gu_bytes, 2 * st.gu_bytes};
          const size_t from[3] = {0, rec_gu, 2 * rec_gu};
          const size_t len[3] = {rec_gu, rec_gu, rec_dn};
          for (int p = 0; p < 3 && err == CL_SUCCESS; ++p)
            err = opencl::clEnqueueWriteBuffer(
              io, part.buf, CL_FALSE, at + dst[p], len[p], src + from[p], 0,
              nullptr, p == 2 ? &ev : nullptr);
        }
        NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
          << "tq2cx expert arena: slot write failed (" << err << ")";
        auto ref = std::make_shared<Tq2Event>(ev);
        b.write = ref;
        part.ready[slot] = ref;
        {
          std::lock_guard<std::mutex> lk(part.lu_mu);
          part.last_use[slot].reset(); // ordered behind it on the I/O queue
        }
      }
      opencl::clFlush(io);
    };

    try {
      size_t first = 0, bank = 0;
      issue(bank, first);
      first += grp[bank].count;
      while (true) {
        const bool more = first < fills.size();
        if (more) {
          issue(bank ^ 1, first);
          first += grp[bank ^ 1].count;
        }
        write(bank);
        if (!more)
          break;
        bank ^= 1;
      }
    } catch (...) {
      // the readers still hold the other group's read list: let them finish
      for (auto &g : grp)
        if (g.pending) {
          try {
            pool.wait(g.ticket);
          } catch (...) {
          }
        }
      // and the writes queued so far still read their staging buffers
      for (auto &b : bounces)
        if (b.write)
          opencl::clWaitForEvents(1, &b.write->ev);
      throw;
    }
    filled_ += fills.size();
    if (wait_all) {
      for (auto &b : bounces)
        if (b.write) {
          opencl::clWaitForEvents(1, &b.write->ev);
          b.write.reset();
        }
      for (const auto &f : fills)
        part.ready[f.second].reset();
    }
  }

  /**
   * @brief Keep the compute queue's next kernels behind the writes of
   *        @a slots (those still pending) and drop those events: the host
   *        waits for the writes. Not a barrier on the compute queue naming
   *        the I/O queue's events: the no-cross-queue-wait-list rule holds
   *        both ways on the I/O / compute pair (see fill()). The writes are
   *        flushed and wait for nothing, so they end; the callers (the load,
   *        the blocking ASYNC=0 step, a complete partition's first decode)
   *        wait for their reads on the host already.
   */
  void waitReady(ClContext *cc, Partition &part,
                 const std::vector<unsigned int> &slots) {
    std::vector<cl_event> evs;
    std::vector<Tq2EventRef> keep;
    for (unsigned int s : slots)
      if (s < part.slots && part.ready[s]) {
        evs.push_back(part.ready[s]->ev);
        keep.push_back(part.ready[s]);
        part.ready[s].reset();
      }
    if (evs.empty())
      return;
    opencl::clFlush(ioQueue(cc));
    const cl_int err = opencl::clWaitForEvents((cl_uint)evs.size(), evs.data());
    NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
      << "tq2cx expert arena: slot writes failed (" << err << ")";
  }

  /** @brief every slot of @a part with a pending write */
  std::vector<unsigned int> pendingSlots(const Partition &part) const {
    std::vector<unsigned int> s;
    for (unsigned int i = 0; i < part.slots; ++i)
      if (part.ready[i])
        s.push_back(i);
    return s;
  }

  /**
   * @brief Mark @a slots as read by everything enqueued so far on the
   *        compute queue (call after the last kernel that reads them).
   */
  void markUse(ClContext *cc, Partition &part,
               const std::vector<unsigned int> &slots) {
    if (slots.empty())
      return;
    cl_command_queue cq = cc->command_queue_inst_.GetCommandQueue();
    cl_event ev = nullptr;
    const cl_int err =
      opencl::clEnqueueBarrierWithWaitList(cq, 0, nullptr, &ev);
    NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
      << "tq2cx expert arena: last-use marker failed (" << err << ")";
    auto ref = std::make_shared<Tq2Event>(ev);
    std::lock_guard<std::mutex> lk(part.lu_mu);
    for (unsigned int s : slots)
      if (s < part.slots)
        part.last_use[s] = ref;
  }

  /**
   * @brief markUse of several partitions with one marker: @a uses holds,
   *        per partition, the slots everything enqueued so far has read.
   */
  void markUseParts(
    ClContext *cc,
    const std::vector<std::pair<Partition *, std::vector<unsigned int>>>
      &uses) {
    bool any = false;
    for (const auto &u : uses)
      any = any || !u.second.empty();
    if (!any)
      return;
    cl_command_queue cq = cc->command_queue_inst_.GetCommandQueue();
    cl_event ev = nullptr;
    const cl_int err =
      opencl::clEnqueueBarrierWithWaitList(cq, 0, nullptr, &ev);
    NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
      << "tq2cx expert arena: last-use marker failed (" << err << ")";
    auto ref = std::make_shared<Tq2Event>(ev);
    for (const auto &u : uses) {
      std::lock_guard<std::mutex> lk(u.first->lu_mu);
      for (unsigned int s : u.second)
        if (s < u.first->slots)
          u.first->last_use[s] = ref;
    }
  }

  /**
   * @brief A blocking write of @a len bytes from @a src at byte @a at of
   *        partition @a p: clEnqueueWriteBuffer on @a q for a device buffer;
   *        for an SVM partition a CPU copy (coarse-grain: between a blocking
   *        map of the range on @a q and its unmap, waited for on the host).
   *        No wait list either way.
   */
  static cl_int slotWrite(cl_command_queue q, Partition &p, size_t at,
                          size_t len, const void *src) {
    if (p.host == nullptr)
      return opencl::clEnqueueWriteBuffer(q, p.buf, CL_TRUE, at, len, src, 0,
                                          nullptr, nullptr);
    const auto t0 = std::chrono::steady_clock::now();
    uint8_t *dst = p.host + at;
    cl_int err = CL_SUCCESS;
    // the maps / unmaps go on a queue of their own: on the I/O queue they
    // slowed the job's next small table writes 25x (Xe3: 0.011 -> 0.25-0.39
    // ms per step)
    if (svm_q_ != nullptr)
      q = svm_q_;
    if (!p.fine)
      err = opencl::clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                    dst, len, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
      return err;
    std::memcpy(dst, src, len);
    if (!p.fine) {
      cl_event ev = nullptr;
      err = opencl::clEnqueueSVMUnmap(q, dst, 0, nullptr, &ev);
      if (err == CL_SUCCESS && ev != nullptr)
        err = opencl::clWaitForEvents(1, &ev);
      if (ev != nullptr)
        opencl::clReleaseEvent(ev);
    }
    svm_writes_++;
    svm_write_us_ +=
      (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0)
        .count();
    return err;
  }

  /** svm arena totals (slotWrite), printed at exit when used */
  struct SvmStats {
    ~SvmStats() {
      const unsigned long long n = svm_writes_.load();
      if (n != 0)
        std::fprintf(stderr,
                     "[TQ2CX-FSU-ARENA] svm slot writes=%llu "
                     "ms_per_write=%.4f\n",
                     n, svm_write_us_.load() / 1000.0 / n);
    }
  };

  /** staging set of the run-time fills (the caller holds tq2Mutex) */
  std::vector<Tq2Bounce> &runBounces() { return run_bounces_; }

  size_t filled() const { return filled_; }

  // ---- prefill block stream (every call below: the caller holds tq2Mutex)

  /**
   * @brief One of the two block buffers of the prefill stream: the
   *        partitions lent to it (expert e of the block it holds sits in
   *        slot e % C of parts[e / C]) and the reads of that block.
   */
  struct StreamSet {
    std::vector<Partition *> parts;
    /** the block buffer the parts are consecutive windows of (expert e in
     *  slot e), or null: the per-part chunk path */
    cl_mem block = nullptr;
    unsigned int layer = UINT32_MAX; /**< layer index of the block */
    int fd = -1;
    std::vector<uint64_t> first, last; /**< per part: file range read */
    std::vector<std::vector<ExpertRead>> reads; /**< per part */
    std::vector<ExpertReaderPool::Ticket> ticket;
    std::vector<bool> pending;
  };

  /**
   * @brief Lend 2 x ceil(@a experts / @a slots) partitions of this shape to
   *        the stream (the lowest layer indices; the same ones every time).
   * @return false when the arena has fewer such partitions
   */
  bool streamLend(unsigned int slots, size_t slot_bytes, unsigned int experts) {
    const unsigned int n = (experts + slots - 1) / slots;
    if (!sets_[0].parts.empty() && sets_[0].parts.size() == n &&
        sets_[0].parts[0]->slots == slots &&
        sets_[0].parts[0]->slot_bytes == slot_bytes)
      return true;
    std::vector<Partition *> fit;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto &kv : by_index_)
        if (kv.second->slots == slots && kv.second->slot_bytes == slot_bytes)
          fit.push_back(kv.second);
    }
    if (fit.size() < 2 * (size_t)n)
      return false;
    for (int x = 0; x < 2; ++x) {
      endSet(sets_[x]);
      StreamSet &S = sets_[x];
      S.parts.assign(fit.begin() + x * n, fit.begin() + (x + 1) * n);
      S.block = S.parts[0]->block;
      for (unsigned int j = 0; j < n && S.block != nullptr; ++j)
        if (S.parts[j]->block != S.block ||
            S.parts[j]->block_origin != (size_t)j * slots * slot_bytes)
          S.block = nullptr;
      S.first.assign(n, 0);
      S.last.assign(n, 0);
      S.reads.assign(n, std::vector<ExpertRead>());
      S.ticket.assign(n, 0);
      S.pending.assign(n, false);
    }
    return true;
  }

  /**
   * @brief Start reading the whole expert block of layer @a d (every expert,
   *        its record into its slot of set @a x) unless set @a x holds or
   *        reads that block already. Each record is written into its slot by
   *        the reader that read it (a blocking write on the I/O queue that
   *        waits for the slot's last reader), so a part is in device memory
   *        once its ticket completes.
   */
  void streamIssue(ClContext *cc, unsigned int x, const MoeExpertFsu &d,
                   const Tq2Stack &shape) {
    StreamSet &S = sets_[x];
    if (S.layer == d.layer_index)
      return;
    endSet(S);
    const size_t rec = d.record_bytes;
    const size_t rec_gu = tq2RecordBytes(shape.I, shape.H);
    const bool packed = rec == shape.slot_bytes;
    const size_t gu_bytes = shape.gu_bytes;
    const unsigned int E = (unsigned int)d.offset->size();
    const unsigned int C = S.parts[0]->slots;
    cl_command_queue io = ioQueue(cc);
    // the writes wait on markers of the compute queue: hand it over first
    opencl::clFlush(cc->command_queue_inst_.GetCommandQueue());
    ExpertReaderPool &pool = ExpertReaderPool::global();
    S.fd = d.fd;
    for (size_t j = 0; j < S.parts.size(); ++j) {
      Partition &p = *S.parts[j];
      std::vector<ExpertRead> &rd = S.reads[j];
      rd.clear();
      for (unsigned int sl = 0; sl < C && j * C + sl < E; ++sl) {
        const unsigned int e = (unsigned int)(j * C + sl);
        Tq2EventRef wait;
        {
          std::lock_guard<std::mutex> lk(p.lu_mu);
          wait = p.last_use[sl];
          p.last_use[sl].reset();
        }
        p.ready[sl].reset();
        ExpertRead r;
        r.fd = d.fd;
        r.offset = (*d.offset)[e];
        r.bytes = rec;
        const size_t at = (size_t)sl * p.slot_bytes + tq2SlotShift(d, e);
        cl_mem buf = p.buf;
        Partition *pp = &p;
        r.consume = [io, buf, at, wait, packed, rec_gu, gu_bytes,
                     pp](const uint8_t *src, const ExpertRead &q) -> int {
          const auto w0 = std::chrono::steady_clock::now();
          struct WriteClock {
            std::chrono::steady_clock::time_point t0;
            ~WriteClock() {
              stream_stats_.write_us +=
                (unsigned long long)
                  std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
            }
          } clock{w0};
          cl_event w = wait ? wait->ev : nullptr;
          if (w != nullptr) {
            // The slot's last reader is a marker of the compute queue. The
            // write does not name it: on Adreno 840 a blocking write on the
            // I/O queue whose wait list names a compute-queue event returns
            // only once the compute queue has run everything runnable (a
            // probe self-test: 3 writes per layer against 100-400 without
            // the list), which throttled the stream to the compute queue's
            // pace; and with a user event pending on the compute queue it
            // does not return at all (a decode deadlock).
            // The reader waits for the marker on the host instead (it is
            // normally long complete: the block's previous layer, two
            // layers back) and then writes with no wait list.
            cl_int est = CL_COMPLETE;
            if (opencl::clGetEventInfo == nullptr ||
                opencl::clGetEventInfo(w, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                       sizeof(est), &est,
                                       nullptr) != CL_SUCCESS ||
                est > CL_COMPLETE) {
              const auto f0 = std::chrono::steady_clock::now();
              const cl_int werr = opencl::clWaitForEvents(1, &w);
              stream_stats_.fence_us +=
                (unsigned long long)
                  std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - f0)
                    .count();
              stream_stats_.fence_waits++;
              if (werr != CL_SUCCESS)
                return EIO;
            } else if (est < 0) {
              return EIO; // the last reader failed
            }
          }
          cl_int err;
          if (pp->host != nullptr) {
            // svm arena: CPU copies into the slot
            const size_t len[3] = {rec_gu, rec_gu, q.bytes - 2 * rec_gu};
            const size_t dst[3] = {0, gu_bytes, 2 * gu_bytes};
            const size_t from[3] = {0, rec_gu, 2 * rec_gu};
            err = packed ? slotWrite(io, *pp, at, q.bytes, src) : CL_SUCCESS;
            for (int pc = 0; !packed && pc < 3 && err == CL_SUCCESS; ++pc)
              err = slotWrite(io, *pp, at + dst[pc], len[pc], src + from[pc]);
          } else if (packed) {
            err = opencl::clEnqueueWriteBuffer(io, buf, CL_TRUE, at, q.bytes,
                                               src, 0, nullptr, nullptr);
          } else {
            // gate, up, down at their rounded places; the in-order queue
            // completes the first two before the blocking third returns
            const size_t len[3] = {rec_gu, rec_gu, q.bytes - 2 * rec_gu};
            const size_t dst[3] = {0, gu_bytes, 2 * gu_bytes};
            const size_t from[3] = {0, rec_gu, 2 * rec_gu};
            err = CL_SUCCESS;
            for (int pp = 0; pp < 3 && err == CL_SUCCESS; ++pp)
              err = opencl::clEnqueueWriteBuffer(
                io, buf, pp == 2 ? CL_TRUE : CL_FALSE, at + dst[pp], len[pp],
                src + from[pp], 0, nullptr, nullptr);
          }
          return err == CL_SUCCESS ? 0 : EIO;
        };
        rd.push_back(std::move(r));
      }
      if (rd.empty()) {
        S.pending[j] = false;
        continue;
      }
      S.first[j] = rd.front().offset;
      S.last[j] = rd.back().offset + rec;
      S.ticket[j] = pool.submit(&rd);
      S.pending[j] = true;
    }
    S.layer = d.layer_index;
    stream_stats_.blocks++;
    stream_stats_.records += E;
    stream_stats_.bytes += (double)E * rec;
  }

  /**
   * @brief Wait until part @a j of set @a x is in device memory; then drop
   *        its file range from the page cache when @a dontneed.
   */
  void streamWait(unsigned int x, size_t j, bool dontneed) {
    StreamSet &S = sets_[x];
    if (!S.pending[j])
      return;
    S.pending[j] = false;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      ExpertReaderPool::global().wait(S.ticket[j]);
    } catch (...) {
      S.layer = UINT32_MAX; // the block is incomplete
      throw;
    }
    stream_stats_.wait_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
#if defined(POSIX_FADV_DONTNEED) && !defined(_WIN32)
    if (dontneed && S.last[j] > S.first[j])
      (void)::posix_fadvise(S.fd, (off_t)S.first[j],
                            (off_t)(S.last[j] - S.first[j]),
                            POSIX_FADV_DONTNEED);
#else
    (void)dontneed;
#endif
  }

  StreamSet &streamSet(unsigned int x) { return sets_[x]; }

  /** @brief forget both blocks (a call that does not stream): wait out any
   *         read still in flight, since it writes into lent partitions */
  void streamEnd() {
    endSet(sets_[0]);
    endSet(sets_[1]);
  }

  /** totals of the prefill stream, printed at exit when it ran */
  struct StreamStats {
    unsigned long long blocks = 0, records = 0;
    double bytes = 0.0, wait_ms = 0.0;
    std::atomic<unsigned long long> write_us{0}; /**< readers, summed */
    /** host waits of the readers on a slot's last reader (summed) */
    std::atomic<unsigned long long> fence_us{0}, fence_waits{0};
    ~StreamStats() {
      if (blocks == 0)
        return;
      std::fprintf(stderr,
                   "[TQ2CX-FSU-STREAM] blocks=%llu records=%llu read_mib=%.1f "
                   "compute_wait_ms=%.1f reader_write_ms=%.1f "
                   "reader_fence_ms=%.1f fence_waits=%llu readers=%u\n",
                   blocks, records, bytes / 1048576.0, wait_ms,
                   write_us.load() / 1000.0, fence_us.load() / 1000.0,
                   fence_waits.load(), ExpertReaderPool::global().readers());
    }
  };

private:
  /** NNTR_G4MOE_FSU_BLOCK=0: a buffer per partition (read once) */
  static bool blockOn() {
    static const bool on = []() {
      const char *e = std::getenv("NNTR_G4MOE_FSU_BLOCK");
      return !(e != nullptr && e[0] == '0');
    }();
    return on;
  }

  /**
   * @brief Partition @a index (< 2 @a n) as a window of block buffer
   *        index / @a n (created on first use, @a n partitions of @a slots
   *        x @a slot_bytes): the sub-buffer, its block and origin; null when
   *        the block cannot be made (the caller makes a buffer of its own).
   *        Caller holds mu_.
   */
  cl_mem blockPartition(ClContext *cc, unsigned int slots, size_t slot_bytes,
                        unsigned int n, unsigned int index, size_t max_alloc,
                        cl_mem *block, size_t *origin, uint8_t **host) {
    const size_t part_bytes = (size_t)slots * slot_bytes;
    const size_t bytes = (size_t)n * part_bytes;
    if (max_alloc != 0 && bytes > max_alloc)
      return nullptr;
    const cl_device_id dev = cc->context_inst_.GetDeviceId();
    cl_uint align_bits = 0;
    if (opencl::clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                                sizeof(align_bits), &align_bits,
                                nullptr) != CL_SUCCESS ||
        align_bits == 0 || part_bytes % (align_bits / 8 ? align_bits / 8 : 1))
      return nullptr;
    const auto key = std::make_tuple(slots, slot_bytes, index / n);
    cl_mem &blk = blocks_[key];
    uint8_t *&blk_host = block_host_[key];
    cl_int err = CL_SUCCESS;
    if (blk == nullptr) {
      opencl::ClMemAcctScope acct("tq2cx_moe_arena");
      blk = allocArena(cc, bytes, &blk_host, &err);
      if (err != CL_SUCCESS || blk == nullptr) {
        blk = nullptr;
        return nullptr;
      }
    }
    cl_buffer_region r;
    r.origin = (size_t)(index % n) * part_bytes;
    r.size = part_bytes;
    cl_mem sub = opencl::clCreateSubBufferT(
      blk, 0, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
    if (err != CL_SUCCESS || sub == nullptr) {
      static bool warned = false;
      if (!warned)
        ml_logw("tq2cx expert arena: sub-buffer refused (%d): a buffer per "
                "partition (the stream runs per partition)",
                (int)err);
      warned = true;
      return nullptr;
    }
    *block = blk;
    *origin = r.origin;
    *host = blk_host != nullptr ? blk_host + r.origin : nullptr;
    return sub;
  }

  /**
   * @brief NNTR_G4MOE_FSU_ARENA=auto|clmem|svm: what the slots are.
   *  - clmem: device buffers (CL_MEM_READ_ONLY), filled by
   *    clEnqueueWriteBuffer.
   *  - svm: shared virtual memory (fine-grain when the device has it, else
   *    coarse-grain) under a CL_MEM_USE_HOST_PTR buffer, so the kernels are
   *    unchanged; filled by CPU copies (coarse-grain: between a map and an
   *    unmap of the written range, on a queue of their own).
   *  - auto (default): svm on Intel GPUs, clmem elsewhere. Xe3, C=16 cold,
   *    same tokens (P1 / P2-256 / P4 1K, C=16 and 32): the prefill stream's
   *    slot writes 3351 -> 624 ms per prefill, its compute wait 397 -> 24
   *    ms, P4 1K prefill 1706 -> 1366 ms (the resident one is 1332), decode
   *    equal (50.2 / 50.5 ms per token; P2-256 55.4 / 57.2, within the run
   *    spread). An O_DIRECT read straight into SVM memory is refused there
   *    (EFAULT), so the bytes still pass a host bounce; the GPU reads the
   *    SVM arena at the device buffer's bandwidth (112 vs 110 GB/s) and the
   *    USE_HOST_PTR view allocates nothing. Adreno: not measured yet (the
   *    svm_probe microbench is staged for the phones), so clmem.
   *    Note: with coarse-grain SVM a written range is mapped while kernels
   *    may read other ranges of the same buffer, which the OpenCL rules
   *    leave undefined (fine-grain is inside them); the slot written is
   *    never one a queued kernel reads (its last reader is complete).
   *  NNTR_G4MOE_FSU_FLAGS on: fine-grain SVM with SVM atomics whatever
   *  NNTR_G4MOE_FSU_ARENA says (the protocol reads records straight into
   *  the slots and hands them over through it).
   * @return 0 clmem, 1 coarse-grain svm, 2 fine-grain svm. Read once.
   */
  static int svmKind(ClContext *cc) {
    static const int kind = [cc]() {
      if (tq2FsuFlagsOn()) {
        std::fprintf(stderr,
                     "[TQ2CX-FSU-ARENA] svm, %s (NNTR_G4MOE_FSU_FLAGS)\n",
                     tq2FsuFlagsCoarse() ? "coarse-grain, test mode"
                                         : "fine-grain with atomics");
        return tq2FsuFlagsCoarse() ? 1 : 2;
      }
      const char *e = std::getenv("NNTR_G4MOE_FSU_ARENA");
      const std::string v = e != nullptr ? e : "auto";
      const cl_device_id dev = cc->context_inst_.GetDeviceId();
      bool want = v == "svm";
      if (v == "auto" || v.empty()) {
        cl_uint vendor = 0;
        want = opencl::clGetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(vendor),
                                       &vendor, nullptr) == CL_SUCCESS &&
               vendor == 0x8086;
      } else if (v != "svm" && v != "clmem") {
        std::fprintf(stderr,
                     "[TQ2CX-FSU-ARENA] NNTR_G4MOE_FSU_ARENA=%s: not "
                     "auto|clmem|svm, auto kept\n",
                     e);
      }
      if (!want)
        return 0;
      cl_device_svm_capabilities caps = 0;
      if (opencl::clSVMAlloc == nullptr ||
          opencl::clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(caps),
                                  &caps, nullptr) != CL_SUCCESS)
        caps = 0;
      const int k = (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)     ? 2
                    : (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) ? 1
                                                                 : 0;
      std::fprintf(stderr, "[TQ2CX-FSU-ARENA] %s (%s, svm caps 0x%llx)\n",
                   k == 2   ? "svm, fine-grain"
                   : k == 1 ? "svm, coarse-grain"
                            : "clmem (no buffer SVM)",
                   v.c_str(), (unsigned long long)caps);
      return k;
    }();
    return kind;
  }

  /** an arena buffer of @a bytes: a device buffer, or (svmKind) an SVM
   *  allocation (*host) under a CL_MEM_USE_HOST_PTR buffer */
  cl_mem allocArena(ClContext *cc, size_t bytes, uint8_t **host, cl_int *err) {
    cl_context ctx = cc->context_inst_.GetContextNoRetain();
    *host = nullptr;
    const int kind = svmKind(cc);
    if (kind == 1 && svm_q_ == nullptr) {
      cl_int qerr = CL_SUCCESS;
      svm_q_ = opencl::clCreateCommandQueue(
        ctx, cc->context_inst_.GetDeviceId(), 0, &qerr);
      if (qerr != CL_SUCCESS)
        svm_q_ = nullptr;
    }
    if (kind != 0) {
      const bool flags = tq2FsuFlagsOn();
      const cl_svm_mem_flags f =
        CL_MEM_READ_WRITE | (kind == 2 ? CL_MEM_SVM_FINE_GRAIN_BUFFER : 0) |
        (flags && kind == 2 ? CL_MEM_SVM_ATOMICS : 0);
      void *p = opencl::clSVMAllocT(ctx, f, bytes, 0);
      // the flags protocol reads O_DIRECT windows to slot starts: the
      // allocation must start on a page (clSVMAlloc alignment 0 gives 4 KiB
      // on Adreno 0842.19)
      NNTR_THROW_IF(
        flags &&
          (p == nullptr || reinterpret_cast<uintptr_t>(p) % kTq2FlagsPage != 0),
        std::runtime_error)
        << "tq2cx expert arena: NNTR_G4MOE_FSU_FLAGS needs a 4 KiB-aligned "
           "fine-grain SVM allocation of "
        << bytes << " B (got " << p << ")";
      // the coarse-grain test mode writes slots from reader threads while
      // kernels run: fault every page in now (see Tq2Flags::issue)
      if (p != nullptr && flags && kind == 1)
        std::memset(p, 0, bytes);
      if (p != nullptr) {
        // a view of the SVM allocation (no memory of its own: Xe3 MemAvailable
        // and DRM totals unchanged by it): counted once in the GPU ledger
        opencl::ClMemAcctViewScope view;
        cl_mem m = opencl::clCreateBufferT(
          ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes, p, err);
        if (*err == CL_SUCCESS && m != nullptr) {
          *host = static_cast<uint8_t *>(p);
          return m;
        }
        opencl::clSVMFreeT(ctx, p);
      }
      NNTR_THROW_IF(flags, std::runtime_error)
        << "tq2cx expert arena: NNTR_G4MOE_FSU_FLAGS: no buffer over the SVM "
           "arena of "
        << bytes << " B (" << *err << ")";
      ml_logw("tq2cx expert arena: SVM arena of %zu B refused; a device "
              "buffer",
              bytes);
    }
    return opencl::clCreateBufferT(ctx, CL_MEM_READ_ONLY, bytes, nullptr, err);
  }

  void endSet(StreamSet &S) {
    for (size_t j = 0; j < S.pending.size(); ++j)
      if (S.pending[j]) {
        S.pending[j] = false;
        try {
          ExpertReaderPool::global().wait(S.ticket[j]);
        } catch (...) {
        }
      }
    S.layer = UINT32_MAX;
  }

  Tq2ExpertArena() : run_bounces_(16) {}
  std::mutex mu_;
  std::unordered_map<const void *, std::unique_ptr<Partition>> parts_;
  std::map<unsigned int, Partition *> by_index_;
  /** block buffers of the stream's partitions: (slots, slot bytes, group) */
  std::map<std::tuple<unsigned int, size_t, unsigned int>, cl_mem> blocks_;
  std::map<std::tuple<unsigned int, size_t, unsigned int>, uint8_t *>
    block_host_;
  StreamSet sets_[2];
  static StreamStats stream_stats_;
  static std::atomic<unsigned long long> svm_writes_, svm_write_us_;
  static cl_command_queue svm_q_; /**< svm arena: the maps' queue */
  static SvmStats svm_stats_;
  cl_command_queue io_q_ = nullptr;
  cl_command_queue io_low_ = nullptr;
  std::vector<Tq2Bounce> run_bounces_;
  size_t total_slots_ = 0;
  std::atomic<size_t> filled_{0};
  /** fill(): slot writes whose last reader the host had to wait for */
  std::atomic<size_t> fence_waits_{0};
};

std::atomic<unsigned long long> Tq2ExpertArena::svm_writes_{0};
std::atomic<unsigned long long> Tq2ExpertArena::svm_write_us_{0};
cl_command_queue Tq2ExpertArena::svm_q_ = nullptr;
Tq2ExpertArena::StreamStats Tq2ExpertArena::stream_stats_;
Tq2ExpertArena::SvmStats Tq2ExpertArena::svm_stats_;

/** words of one 64-byte rounded record */
inline size_t tq2Words(size_t bytes) { return bytes / sizeof(uint32_t); }

/**
 * @brief The empty arena-backed stack of an FSU layer: its partition of
 *        fsu.slots slots, no expert read yet. Declines (null) when the file
 *        record is not gate | up | down of this layer's shapes.
 */
std::unique_ptr<Tq2Stack> tq2ArenaStack(ClContext *cc, const MoeExpertGroup &g,
                                        Tq2ExpertArena::Partition **part) {
  const MoeExpertFsu &fsu = *g.fsu;
  auto st = std::make_unique<Tq2Stack>();
  st->layout = tq2cxStoredLayout();
  st->H = g.gate[0]->height();
  st->I = g.gate[0]->width();
  const size_t rec_gu = tq2RecordBytes(st->I, st->H);
  const size_t rec_dn = tq2RecordBytes(st->H, st->I);
  st->gu_bytes = (rec_gu + 63) / 64 * 64;
  st->dn_bytes = (rec_dn + 63) / 64 * 64;
  st->slot_bytes = 2 * st->gu_bytes + st->dn_bytes;
  st->arena = true;
  tq2FsuFlagsDecide(cc, g);
  st->slot_stride = tq2SlotStride(st->slot_bytes);
  if (fsu.fd < 0 || fsu.offset == nullptr ||
      fsu.offset->size() != g.gate.size() || fsu.slots == 0 ||
      fsu.slots > g.gate.size() || fsu.record_bytes != 2 * rec_gu + rec_dn) {
    ml_logw("tq2cx expert arena: declined (fd %d, %zu offsets for %zu "
            "experts, %u slots, record %zu B vs %zu B expected)",
            fsu.fd, fsu.offset ? fsu.offset->size() : (size_t)0, g.gate.size(),
            fsu.slots, fsu.record_bytes, 2 * rec_gu + rec_dn);
    return nullptr;
  }
  if (tq2FsuFlagsOn()) {
    st->shift.resize(g.gate.size());
    for (size_t e = 0; e < g.gate.size(); ++e)
      st->shift[e] = (uint32_t)tq2SlotShift((*fsu.offset)[e]);
  }
  Tq2ExpertArena::Partition &p = Tq2ExpertArena::get().partition(
    cc, g.layer_key, fsu.slots, st->slot_stride, fsu.layer_index,
    (unsigned int)g.gate.size());
  st->buf = p.buf;
  *part = &p;
  return st;
}

/** gate, up and down word offsets of every expert from a slot map (an
 *  expert that is not resident gets slot 0's: never read) */
void tq2ArenaOffsets(Tq2Stack &st, const std::vector<unsigned int> &slot_of) {
  const size_t E = slot_of.size();
  st.off_gate.resize(E);
  st.off_up.resize(E);
  st.off_down.resize(E);
  for (size_t e = 0; e < E; ++e) {
    const size_t s =
      slot_of[e] == std::numeric_limits<unsigned int>::max() ? 0 : slot_of[e];
    const size_t base =
      s * st.slot_stride + (e < st.shift.size() ? st.shift[e] : 0);
    st.off_gate[e] = (uint32_t)tq2Words(base);
    st.off_up[e] = (uint32_t)tq2Words(base + st.gu_bytes);
    st.off_down[e] = (uint32_t)tq2Words(base + 2 * st.gu_bytes);
  }
}

/** (re)build the stack's [3][E] offset table from its offsets */
void tq2ArenaTable(ClContext *cc, Tq2Stack &st) {
  std::vector<uint32_t> tab;
  tab.reserve(3 * st.off_gate.size());
  tab.insert(tab.end(), st.off_gate.begin(), st.off_gate.end());
  tab.insert(tab.end(), st.off_up.begin(), st.off_up.end());
  tab.insert(tab.end(), st.off_down.begin(), st.off_down.end());
  if (st.off_tab != nullptr)
    opencl::clReleaseMemObjectT(st.off_tab);
  st.off_tab = tq2Upload(cc, tab.data(), tab.size() * sizeof(uint32_t));
}

/** the partition of an arena-backed stack */
Tq2ExpertArena::Partition &tq2Partition(ClContext *cc, const Tq2Stack &st,
                                        const void *key, unsigned int slots) {
  return Tq2ExpertArena::get().partition(cc, key, slots, st.slot_stride);
}

/**
 * @brief The layer's stack; built here on first use (caller holds tq2Mutex):
 *        the resident stack, or for a layer that reads its experts from the
 *        file (g.fsu) its empty partition of the slot arena. Null when an
 *        arena stack is declined (nothing was enqueued).
 */
Tq2Stack *tq2Stack(ClContext *cc, const MoeExpertGroup &g) {
  auto it = tq2Stacks().find(g.layer_key);
  if (it != tq2Stacks().end()) {
    const Tq2cxLayout kl = tq2cxStoredLayout();
    NNTR_THROW_IF(it->second->layout != kl, std::runtime_error)
      << "tq2cx_moe_experts_cl: pack layout "
      << tq2cxLayoutName(it->second->layout) << ", kernel layout "
      << tq2cxLayoutName(kl);
    // an arena stack filled completely at load serves resident calls too
    NNTR_THROW_IF(
      (g.fsu != nullptr && !it->second->arena) ||
        (g.fsu == nullptr && it->second->arena && !it->second->tab_complete),
      std::runtime_error)
      << "tq2cx_moe_experts_cl: a layer switched between resident and "
         "streamed experts";
    return it->second.get();
  }
  std::unique_ptr<Tq2Stack> st;
  if (g.fsu != nullptr) {
    Tq2ExpertArena::Partition *part = nullptr;
    st = tq2ArenaStack(cc, g, &part);
    if (!st)
      return nullptr;
    tq2ArenaOffsets(
      *st, std::vector<unsigned int>(g.gate.size(),
                                     std::numeric_limits<unsigned>::max()));
  } else {
    st = tq2BuildStack(cc, g);
  }
  Tq2Stack *ref = st.get();
  tq2Stacks().emplace(g.layer_key, std::move(st));
  return ref;
}

/** small per-call operand, copied at create time */
cl_mem tq2Small(ClContext *cc, const void *host, size_t bytes) {
  return tq2Upload(cc, host, bytes, CL_MEM_READ_ONLY);
}

/** the device's largest work-group (256 when the device was not queried) */
size_t tq2MaxWorkGroup() {
  const opencl::DeviceInfo *info =
    opencl::ContextManager::Global().getDeviceInfo();
  const size_t n = info != nullptr ? info->getDeviceMaxWorkGroupSize() : 0;
  return n != 0 ? n : 256;
}

void tq2Dispatch(ClContext *cc, const ClContext::SharedPtrClKernel &kp,
                 cl_uint dims, const std::array<size_t, 3> &gws,
                 const std::array<size_t, 3> &lws) {
  cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), dims, gws.data(),
                                        lws.data());
}

/** quantize M rows of K floats (halves with @a fp16) from @a x into the
 *  given int8 rows, scale, zero point and row sum; with @a perm each row is
 *  stored in TQ2 order (V8C_AQ_PERM, K % 32 == 0) */
void tq2QuantizeTo(ClContext *cc, cl_mem x, void *x_svm, unsigned int M,
                   unsigned int K, bool fp16, bool perm, cl_mem xq, cl_mem xs,
                   cl_mem xz, cl_mem xr) {
  auto kp = cc->registerClKernel(
    int8_int4_gemm_v8c_kernel,
    fp16 ? "v8c_act_quant_f16_par" : "v8c_act_quant_f32_par",
    perm ? tq2QuantOptions(cc) + " -DV8C_AQ_PERM" : tq2QuantOptions(cc));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: activation quantizer did not build";
  const int Mi = (int)M, Ki = (int)K;
  bool ok =
    (x_svm != nullptr ? kp->SetKernelSVMArguments(0, x_svm)
                      : kp->SetKernelArguments(0, &x, sizeof(cl_mem))) &&
    kp->SetKernelArguments(1, &xq, sizeof(cl_mem)) &&
    kp->SetKernelArguments(2, &xs, sizeof(cl_mem)) &&
    kp->SetKernelArguments(3, &xz, sizeof(cl_mem)) &&
    kp->SetKernelArguments(4, &xr, sizeof(cl_mem)) &&
    kp->SetKernelArguments(5, &Mi, sizeof(int)) &&
    kp->SetKernelArguments(6, &Ki, sizeof(int));
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: quantizer arguments";
  tq2Dispatch(cc, kp, 1, {(size_t)M * kV8cQuantLws, 1, 1},
              {kV8cQuantLws, 1, 1});
}

/** quantize M rows of K floats from @a x into the workspace int8 rows; with
 *  @a perm each row is stored in TQ2 order (V8C_AQ_PERM, K % 32 == 0) */
void tq2Quantize(ClContext *cc, Tq2Workspace &ws, cl_mem x, void *x_svm,
                 unsigned int M, unsigned int K, bool fp16 = false,
                 bool perm = false) {
  tq2QuantizeTo(cc, x, x_svm, M, K, fp16, perm, ws.xqGet(cc, (size_t)M * K),
                ws.xs.get(cc, (size_t)M * sizeof(float)),
                ws.xz.get(cc, (size_t)M * sizeof(int)),
                ws.xrs.get(cc, (size_t)M * sizeof(int)));
}

/** quantize rows [@a r0, @a r0 + @a n) of the fp32 rows @a x (K wide) into
 *  the same rows of the workspace int8 rows; each row exactly as
 *  tq2Quantize quantizes it (the other rows are left as they are) */
void tq2QuantizeRowRange(ClContext *cc, Tq2Workspace &ws, cl_mem x,
                         unsigned int r0, unsigned int n, unsigned int K) {
  if (n == 0)
    return;
  auto kp =
    cc->registerClKernel(int8_int4_gemm_v8c_kernel, "v8c_act_quant_f32_par",
                         tq2QuantOptions(cc) + " -DV8C_AQ_ROW0");
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: activation quantizer (row range) did not build";
  const unsigned int M = r0 + n;
  cl_mem xq = ws.xqGet(cc, (size_t)M * K);
  cl_mem xs = ws.xs.get(cc, (size_t)M * sizeof(float));
  cl_mem xz = ws.xz.get(cc, (size_t)M * sizeof(int));
  cl_mem xr = ws.xrs.get(cc, (size_t)M * sizeof(int));
  const int Mi = (int)M, Ki = (int)K, r0i = (int)r0;
  bool ok = kp->SetKernelArguments(0, &x, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &xq, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &xs, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &xz, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &xr, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &Mi, sizeof(int)) &&
            kp->SetKernelArguments(6, &Ki, sizeof(int)) &&
            kp->SetKernelArguments(7, &r0i, sizeof(int));
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: quantizer (row range) arguments";
  tq2Dispatch(cc, kp, 1, {(size_t)n * kV8cQuantLws, 1, 1},
              {kV8cQuantLws, 1, 1});
}

/** the decode GEMVs read activation rows in TQ2 order (tq2GemvGeom): the
 *  quantizer feeding tq2GemvBatchDev must then store that order */
bool tq2GemvPermAct(ClContext *cc) { return tq2Dpas(cc) && tq2GemvGeom().perm; }

/** quantize the decode GEMV's activation rows in the order it reads them */
void tq2QuantizeGemv(ClContext *cc, Tq2Workspace &ws, cl_mem x, void *x_svm,
                     unsigned int M, unsigned int K, bool fp16 = false) {
  tq2Quantize(cc, ws, x, x_svm, M, K, fp16, tq2GemvPermAct(cc));
}

/** int8 activation rows held outside the workspace (a claimed norm
 *  handoff); all null = the workspace rows */
struct Tq2QRows {
  cl_mem xq = nullptr, xs = nullptr, xz = nullptr;
};

/** the batched GEMV with the record offsets and activation rows on the
 *  device (Intel subgroup kernel on DPAS devices, the portable one else) */
void tq2GemvBatchDev(ClContext *cc, Tq2Workspace &ws, const Tq2Stack &st,
                     cl_mem d_off, cl_mem d_ar, unsigned int B, cl_mem Y,
                     unsigned int N, unsigned int K,
                     const Tq2QRows &rows = Tq2QRows{}) {
  const bool portable = !tq2Dpas(cc);
  const Tq2GemvGeom &geom = tq2GemvGeom();
  auto kp = portable
              ? cc->registerClKernel(tq2cx_gemm_portable_kernel,
                                     "tq2cx_gemv_batch_portable", tq2Options())
              : cc->registerClKernel(tq2cx_gemm_kernel, "tq2cx_gemv_batch",
                                     tq2Options(geom.options()));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: the batched GEMV did not build";
  const bool own = rows.xq != nullptr;
  cl_mem xq = own ? rows.xq : ws.xq.buf, xs = own ? rows.xs : ws.xs.buf,
         xz = own ? rows.xz : ws.xz.buf;
  const int Ni = (int)N, Ki = (int)K;
  bool ok = kp->SetKernelArguments(0, &st.buf, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &d_off, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &xq, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &xs, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &xz, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &d_ar, sizeof(cl_mem)) &&
            kp->SetKernelArguments(6, &Y, sizeof(cl_mem)) &&
            kp->SetKernelArguments(7, &Ni, sizeof(int)) &&
            kp->SetKernelArguments(8, &Ki, sizeof(int));
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: gemv_batch arguments";
  if (portable)
    tq2Dispatch(
      cc, kp, 2,
      {((size_t)N + kTq2pCpw - 1) / kTq2pCpw * kTq2pLws, (size_t)B, 1},
      {kTq2pLws, 1, 1});
  else
    tq2Dispatch(
      cc, kp, 2,
      {((size_t)N + geom.cpw() - 1) / geom.cpw() * kTq2GemvLws, (size_t)B, 1},
      {kTq2GemvLws, 1, 1});
}

/** B weights of one shape against the quantized rows, Y[B][N] */
void tq2GemvBatch(ClContext *cc, Tq2Workspace &ws, const Tq2Stack &st,
                  const std::vector<uint32_t> &rec_off,
                  const std::vector<int> &act_row, cl_mem Y, unsigned int N,
                  unsigned int K, const Tq2QRows &rows = Tq2QRows{}) {
  const unsigned int B = (unsigned int)rec_off.size();
  cl_mem d_off = tq2Small(cc, rec_off.data(), B * sizeof(uint32_t));
  cl_mem d_ar = tq2Small(cc, act_row.data(), B * sizeof(int));
  tq2GemvBatchDev(cc, ws, st, d_off, d_ar, B, Y, N, K, rows);
  opencl::clReleaseMemObjectT(d_off);
  opencl::clReleaseMemObjectT(d_ar);
}

/** all active experts' rows in one portable dp4a launch, Y[R][N] */
void tq2GroupedGemmPortable(ClContext *cc, Tq2Workspace &ws, const Tq2Stack &st,
                            const std::vector<uint32_t> &rec_off,
                            const std::vector<int> &row_off,
                            const std::vector<int> &row_cnt,
                            unsigned int max_rows, cl_mem Y, unsigned int N,
                            unsigned int K) {
  // NB = 2 is the .cl default, so that program is the GEMV's one
  const unsigned int nb = tq2GroupedNb();
  auto kp = cc->registerClKernel(
    tq2cx_gemm_portable_kernel, "tq2cx_gemm_grouped_portable",
    nb == 2 ? tq2Options() : tq2Options("-DTQ2P_NB=" + std::to_string(nb)));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: tq2cx_gemm_grouped_portable did not build";
  const unsigned int E = (unsigned int)rec_off.size();
  cl_mem d_off = tq2Small(cc, rec_off.data(), E * sizeof(uint32_t));
  cl_mem d_ro = tq2Small(cc, row_off.data(), E * sizeof(int));
  cl_mem d_rc = tq2Small(cc, row_cnt.data(), E * sizeof(int));
  cl_mem xq = ws.xq.buf, xs = ws.xs.buf, xz = ws.xz.buf;
  const int Ni = (int)N, Ki = (int)K;
  bool ok = kp->SetKernelArguments(0, &xq, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &st.buf, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &d_off, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &d_ro, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &d_rc, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &xs, sizeof(cl_mem)) &&
            kp->SetKernelArguments(6, &xz, sizeof(cl_mem)) &&
            kp->SetKernelArguments(7, &Y, sizeof(cl_mem)) &&
            kp->SetKernelArguments(8, &Ni, sizeof(int)) &&
            kp->SetKernelArguments(9, &Ki, sizeof(int));
  if (ok)
    tq2Dispatch(
      cc, kp, 3,
      {((size_t)N + kTq2pGLws * nb - 1) / (kTq2pGLws * nb) * kTq2pGLws,
       ((size_t)max_rows + kTq2pTm - 1) / kTq2pTm, (size_t)E},
      {kTq2pGLws, 1, 1});
  opencl::clReleaseMemObjectT(d_off);
  opencl::clReleaseMemObjectT(d_ro);
  opencl::clReleaseMemObjectT(d_rc);
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: portable grouped GEMM arguments";
}

/** all active experts' rows in one XMX launch, Y[R][N] */
void tq2GroupedGemm(ClContext *cc, Tq2Workspace &ws, const Tq2Stack &st,
                    const std::vector<uint32_t> &rec_off,
                    const std::vector<int> &row_off,
                    const std::vector<int> &row_cnt, unsigned int max_rows,
                    cl_mem Y, unsigned int N, unsigned int K) {
  if (!tq2Dpas(cc)) {
    tq2GroupedGemmPortable(cc, ws, st, rec_off, row_off, row_cnt, max_rows, Y,
                           N, K);
    return;
  }
  // tile height by the largest per-expert row count: 8-row tiles for a
  // decode-like spread, 32-row tiles otherwise
  const bool small = max_rows <= 8;
  const unsigned int mt = small ? 1 : 4, nt = 2, sgm = 1;
  const std::string opts =
    tq2Options(std::string("-DTQ2_XMX -cl-intel-256-GRF-per-thread") +
               (small ? " -DTQ2_MT=1" : ""));
  auto kp =
    cc->registerClKernel(tq2cx_gemm_kernel, "tq2cx_gemm_xmx_grouped", opts);
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: tq2cx_gemm_xmx_grouped did not build";
  const unsigned int E = (unsigned int)rec_off.size();
  cl_mem d_off = tq2Small(cc, rec_off.data(), E * sizeof(uint32_t));
  cl_mem d_ro = tq2Small(cc, row_off.data(), E * sizeof(int));
  cl_mem d_rc = tq2Small(cc, row_cnt.data(), E * sizeof(int));
  cl_mem xq = ws.xq.buf, xs = ws.xs.buf, xz = ws.xz.buf;
  const int Ni = (int)N, Ki = (int)K;
  bool ok = kp->SetKernelArguments(0, &xq, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &st.buf, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &d_off, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &d_ro, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &d_rc, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &xs, sizeof(cl_mem)) &&
            kp->SetKernelArguments(6, &xz, sizeof(cl_mem)) &&
            kp->SetKernelArguments(7, &Y, sizeof(cl_mem)) &&
            kp->SetKernelArguments(8, &Ni, sizeof(int)) &&
            kp->SetKernelArguments(9, &Ki, sizeof(int));
  const unsigned int mpw = mt * 8 * sgm;
  if (ok)
    tq2Dispatch(cc, kp, 3,
                {((size_t)N + nt * 16 - 1) / (nt * 16) * 16,
                 ((size_t)max_rows + mpw - 1) / mpw * sgm, (size_t)E},
                {16, (size_t)sgm, 1});
  opencl::clReleaseMemObjectT(d_off);
  opencl::clReleaseMemObjectT(d_ro);
  opencl::clReleaseMemObjectT(d_rc);
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: grouped GEMM arguments";
}

void tq2Glu(ClContext *cc, cl_mem gate, size_t gate_off, cl_mem up,
            size_t up_off, cl_mem out, size_t total, bool exact,
            size_t out_off = 0) {
  auto kp = cc->registerClKernel(kTq2MoeAux, "tq2moe_glu_f32");
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: aux kernels did not build";
  const int ti = (int)total, ex = exact ? 1 : 0;
  const int go = (int)gate_off, uo = (int)up_off, oo = (int)out_off;
  bool ok = kp->SetKernelArguments(0, &gate, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &up, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &out, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &ti, sizeof(int)) &&
            kp->SetKernelArguments(4, &ex, sizeof(int)) &&
            kp->SetKernelArguments(5, &go, sizeof(int)) &&
            kp->SetKernelArguments(6, &uo, sizeof(int)) &&
            kp->SetKernelArguments(7, &oo, sizeof(int));
  if (ok)
    tq2Dispatch(cc, kp, 1, {(total + 63) / 64 * 64, 1, 1}, {64, 1, 1});
  NNTR_THROW_IF(!ok, std::runtime_error) << "tq2cx_moe_experts_cl: glu args";
}

/**
 * @brief NNTR_TQ2CX_GLU_QUANT=0 keeps the separate GeGLU and down-input
 *        quantiser (an A/B switch: the fused kernel writes the same bytes).
 *        Read once.
 */
bool tq2GluQuantOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_TQ2CX_GLU_QUANT");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

/**
 * @brief The expert GeGLU of M rows of K and the quantisation of the result
 *        for the down projection: act[M][K] as tq2Glu writes it, and the
 *        workspace int8 rows / scale / zero point / row sum as tq2Quantize
 *        (act, M, K, perm) writes them -- in one launch
 *        (tq2moe_glu_act_quant_f32) instead of two.
 */
void tq2GluQuantize(ClContext *cc, Tq2Workspace &ws, cl_mem gate,
                    size_t gate_off, cl_mem up, size_t up_off, cl_mem act,
                    unsigned int M, unsigned int K, bool exact, bool perm) {
  if (!tq2GluQuantOn()) {
    tq2Glu(cc, gate, gate_off, up, up_off, act, (size_t)M * K, exact);
    tq2Quantize(cc, ws, act, nullptr, M, K, false, perm);
    return;
  }
  // the quantiser's program options, so the quant half is that kernel's
  auto kp = cc->registerClKernel(
    tq2moe_glu_quant_kernel, "tq2moe_glu_act_quant_f32",
    perm ? tq2QuantOptions(cc) + " -DV8C_AQ_PERM" : tq2QuantOptions(cc));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: the GeGLU quantizer did not build";
  cl_mem xq = ws.xqGet(cc, (size_t)M * K);
  cl_mem xs = ws.xs.get(cc, (size_t)M * sizeof(float));
  cl_mem xz = ws.xz.get(cc, (size_t)M * sizeof(int));
  cl_mem xr = ws.xrs.get(cc, (size_t)M * sizeof(int));
  const int Mi = (int)M, Ki = (int)K, ex = exact ? 1 : 0;
  const int go = (int)gate_off, uo = (int)up_off;
  bool ok = kp->SetKernelArguments(0, &gate, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &up, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &act, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &xq, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &xs, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &xz, sizeof(cl_mem)) &&
            kp->SetKernelArguments(6, &xr, sizeof(cl_mem)) &&
            kp->SetKernelArguments(7, &Mi, sizeof(int)) &&
            kp->SetKernelArguments(8, &Ki, sizeof(int)) &&
            kp->SetKernelArguments(9, &ex, sizeof(int)) &&
            kp->SetKernelArguments(10, &go, sizeof(int)) &&
            kp->SetKernelArguments(11, &uo, sizeof(int));
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: GeGLU quantizer arguments";
  tq2Dispatch(cc, kp, 1, {(size_t)M * kV8cQuantLws, 1, 1},
              {kV8cQuantLws, 1, 1});
}

/**
 * @brief The gpu context when the grouped kernels serve this layer's expert
 *        weights (format, shapes, device), else null. Activations aside.
 */
ClContext *tq2GroupContext(const MoeExpertGroup &g) {
  using DT = ml::train::TensorDim::DataType;
  if (g.gate.empty() || g.up.size() != g.gate.size() ||
      g.down.size() != g.gate.size())
    return nullptr;
  for (size_t e = 0; e < g.gate.size(); ++e)
    if (g.gate[e] == nullptr || g.up[e] == nullptr || g.down[e] == nullptr ||
        g.gate[e]->getDataType() != DT::TQ2CX ||
        g.up[e]->getDataType() != DT::TQ2CX ||
        g.down[e]->getDataType() != DT::TQ2CX)
      return nullptr;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  // Intel DPAS devices take the subgroup / XMX kernels; any other device
  // with the packed int8 dot product takes the portable ones
  if (cc == nullptr || (!cc->caps().dpas && !cc->caps().int_dot_product))
    return nullptr;
  const unsigned int H = g.gate[0]->height(), I = g.gate[0]->width();
  // XMX block-io rules on both reduction sizes; K % 32 for the masks
  auto xmx_k = [](unsigned int K) {
    return K % 32 == 0 && K >= 64 && (K / 4) % 16 == 0;
  };
  if (!xmx_k(H) || !xmx_k(I))
    return nullptr;
  // the portable GEMV stages one activation row in local memory
  if (!tq2Dpas(cc) && (H > kTq2pMaxK || I > kTq2pMaxK))
    return nullptr;
  for (size_t e = 0; e < g.gate.size(); ++e)
    if (g.gate[e]->height() != H || g.gate[e]->width() != I ||
        g.up[e]->height() != H || g.up[e]->width() != I ||
        g.down[e]->height() != I || g.down[e]->width() != H)
      return nullptr;
  return cc;
}

} // namespace

namespace {
/** decide the async decode mode of @a cc now (see tq2AsyncMode) */
void tq2AsyncModeAtLoad(ClContext *cc);
} // namespace

bool tq2cx_moe_experts_prebuild_cl(const MoeExpertGroup &g) {
  if (g.layer_key == nullptr ||
      (g.records != nullptr && g.records->size() != 3 * g.gate.size()))
    return false;
  if (g.fsu != nullptr &&
      (g.fsu->slot_of == nullptr || g.fsu->slot_of->size() != g.gate.size()))
    return false;
  ClContext *cc = tq2GroupContext(g);
  if (cc == nullptr) {
    auto *c =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    ml_logw("tq2cx_moe_experts_prebuild_cl: not served here (%s)",
            c ? c->caps().toString().c_str() : "no gpu context");
    return false;
  }
  // a layer whose experts do not all stay resident decodes through the
  // async step: run its device probe at load, not in the first decode step
  if (g.fsu != nullptr && g.fsu->slots < g.gate.size()) {
    // before any slot exists (their memory); the flags protocol needs no
    // async mode (no callback, no user event)
    if (!tq2FsuFlagsDecide(cc, g))
      tq2AsyncModeAtLoad(cc);
  }
  {
    std::lock_guard<std::mutex> lk(tq2Mutex());
    if (tq2Stacks().count(g.layer_key))
      return true;
  }
  // The loader reads the layers on parallel threads: build (and upload)
  // outside the map lock so a few layers' uploads overlap, but at most
  // kTq2Uploaders at once -- every upload in flight holds its source pages
  // (the file mapping) resident. One layer key is built by one thread.
  static std::mutex slot_mu;
  static std::condition_variable slot_cv;
  static unsigned int in_flight = 0;
  constexpr unsigned int kTq2Uploaders = 4;
  {
    std::unique_lock<std::mutex> slk(slot_mu);
    slot_cv.wait(slk, [] { return in_flight < kTq2Uploaders; });
    ++in_flight;
  }
  std::unique_ptr<Tq2Stack> st;
  try {
    if (g.fsu != nullptr) {
      // the layer's partition of the slot arena, every listed expert read
      // from the file now through the reader pool and the I/O queue
      Tq2ExpertArena::Partition *part = nullptr;
      st = tq2ArenaStack(cc, g, &part);
      if (st) {
        std::vector<Tq2Bounce> bounces(8);
        static const std::vector<std::pair<unsigned int, unsigned int>> none;
        Tq2ExpertArena::get().fill(cc, *part, *st, *g.fsu,
                                   g.fsu->fill ? *g.fsu->fill : none, bounces,
                                   /*wait_all=*/true);
        tq2ArenaOffsets(*st, *g.fsu->slot_of);
        if (g.fsu->complete) {
          tq2ArenaTable(cc, *st);
          st->tab_complete = true;
        }
      }
    } else {
      st = tq2BuildStack(cc, g);
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> slk(slot_mu);
      --in_flight;
    }
    slot_cv.notify_one();
    throw;
  }
  {
    std::lock_guard<std::mutex> slk(slot_mu);
    --in_flight;
  }
  slot_cv.notify_one();
  if (!st)
    return false;
  std::lock_guard<std::mutex> lk(tq2Mutex());
  tq2Stacks().emplace(g.layer_key, std::move(st));
  return true;
}

namespace {

/**
 * @brief An activation operand of the grouped path: a shared-plane pointer,
 *        or the tensor's device buffer when it is device-resident.
 */
struct Tq2Act {
  void *svm = nullptr;  /**< shared-memory pointer (null when device-only) */
  cl_mem mem = nullptr; /**< device buffer (null when shared) */
  size_t bytes = 0;     /**< bytes the call touches */

  bool bind(const ClContext::SharedPtrClKernel &kp, unsigned int idx) const {
    return svm != nullptr ? kp->SetKernelSVMArguments(idx, svm)
                          : kp->SetKernelArguments(idx, &mem, sizeof(cl_mem));
  }
};

/** @a t as an operand, or false when it is on neither plane */
bool tq2Act(const Tensor &t, size_t bytes, Tq2Act &a) {
  const auto md = t.getMemoryData();
  if (!md)
    return false;
  a.bytes = bytes;
  if (t.isClMem()) {
    a.mem = static_cast<cl_mem>(t.getClMem());
    return a.mem != nullptr;
  }
  if (!md->isSVM())
    return false;
  a.svm = const_cast<uint8_t *>(t.getData<uint8_t>());
  return a.svm != nullptr;
}

/**
 * @brief NNTR_TQ2CX_NORM_HANDOFF=1 lets the pre-FFN norm's quantisation serve
 *        the experts' input where it could (default off: the experts keep
 *        their own input quantiser). Read once.
 * @details Opt-in until an in-model run proves it: the handoff is reachable
 *          only off DPAS, so no Xe3 run (FORCE_PORTABLE included) executes
 *          it; its evidence is a kernel-level harness cell. Turn it on by
 *          default once the Adreno harness and goldens pass with it.
 */
bool tq2NormHandoffOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_TQ2CX_NORM_HANDOFF");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

/**
 * @brief Do the router kernels use the device's fma()? False on a device
 *        without a hardware fp32 fma (DeviceCaps::fp32_fma), and with
 *        NNTR_TQ2CX_ROUTER_SOFT_FMA=1 (a test lane: the emulated fma of such
 *        a device, -DRT_SOFT_FMA, on any device; the same bits either way).
 */
bool tq2RouterHwFma(ClContext *cc) {
  static const bool forced = []() {
    const char *e = std::getenv("NNTR_TQ2CX_ROUTER_SOFT_FMA");
    return e != nullptr && e[0] == '1';
  }();
  return cc->caps().fp32_fma && !forced;
}

/**
 * @brief The decode expert input's int8 row from the norm that wrote it
 *        (v8cNormQuantClaim), when the experts would quantise exactly as that
 *        norm did, else all null and the caller quantises.
 * @details The fused norm quantises with the v8c FC's quantiser arithmetic
 *          under the FC's program options, in plain order. The experts'
 *          quantiser is the same kernel source; it matches only where it is
 *          built with the same options and stores plain order -- not on a
 *          DPAS device, which builds it with correctly rounded division and
 *          stores TQ2 order (tq2QuantOptions, tq2GemvPermAct).
 */
Tq2QRows tq2ClaimNormRow(ClContext *cc, Tq2Workspace &ws, const Tq2Act &in,
                         bool fp16, unsigned int H) {
  Tq2QRows r;
  if (!tq2NormHandoffOn() || !fp16 || in.mem == nullptr || cc->caps().dpas ||
      tq2GemvPermAct(cc))
    return r;
  void *i8 = nullptr, *sc = nullptr, *zp = nullptr, *rs = nullptr;
  if (!v8cNormQuantClaim(in.mem, 1, H, &i8, &sc, &zp, &rs))
    return r;
  r.xq = static_cast<cl_mem>(i8);
  r.xs = static_cast<cl_mem>(sc);
  r.xz = static_cast<cl_mem>(zp);
  return r;
}

/** a host-readable fp32 weight, or null */
float *tq2HostFloat(const Tensor *t) {
  if (t == nullptr || t->getDataType() != ml::train::TensorDim::DataType::FP32)
    return nullptr;
  const auto md = t->getMemoryData();
  if (!md || t->isClMem() || t->getData<float>() == nullptr)
    return nullptr;
  return const_cast<float *>(t->getData<float>());
}

/**
 * @brief The device copy of a router operand (fp32 weight), uploaded on
 *        first use and kept for the process.
 * @details Not bound in place on the shared weight plane: measured on Xe3, a
 * kernel reading the per-expert scale straight from the weight pool saw the
 * value the weight held before the load (the pool pages are written on the
 * host and never handed to the device), while the copy reads correctly.
 */
cl_mem tq2RouterOperand(ClContext *cc, const float *host, size_t count) {
  static std::unordered_map<const void *, cl_mem> copies;
  auto it = copies.find(host);
  if (it != copies.end())
    return it->second;
  cl_mem m = nullptr;
  {
    opencl::ClMemAcctScope acct("tq2cx_router");
    m = tq2Upload(cc, host, count * sizeof(float));
  }
  return copies.emplace(host, m).first->second;
}

/**
 * @brief The device copy of the router weight as bf16 (the high 16 bits of
 *        each fp32 value), uploaded on first use and kept for the process;
 *        nullptr when a value has non-zero low 16 bits (the caller keeps the
 *        fp32 copy) or NNTR_ROUTER_BF16=0.
 * @details Only exact: the kernel widens (bits << 16) back to the very same
 * fp32 value, so the logits are bit-identical to the fp32 operand and the
 * weight stream per layer halves (H x E x 2 bytes). Checked per weight.
 */
cl_mem tq2RouterWeightBf16(ClContext *cc, const float *host, size_t count) {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_ROUTER_BF16");
    return e == nullptr || e[0] != '0';
  }();
  if (!on)
    return nullptr;
  static std::unordered_map<const void *, cl_mem> copies;
  auto it = copies.find(host);
  if (it != copies.end())
    return it->second;
  std::vector<uint16_t> hi(count);
  bool exact = true;
  for (size_t i = 0; i < count && exact; ++i) {
    uint32_t b;
    std::memcpy(&b, host + i, sizeof(b));
    exact = (b & 0xFFFFu) == 0u;
    hi[i] = static_cast<uint16_t>(b >> 16);
  }
  cl_mem m = nullptr;
  if (exact) {
    opencl::ClMemAcctScope acct("tq2cx_router");
    m = tq2Upload(cc, hi.data(), count * sizeof(uint16_t));
  }
  static bool logged = false;
  if (!logged) {
    logged = true;
    ml_logi("tq2cx router weight on the device: %s",
            exact ? "bf16 (exact)" : "fp32 (not exact in bf16)");
  }
  return copies.emplace(host, m).first->second;
}

/** the decode constants for @a K selected experts */
void tq2RouteConstants(ClContext *cc, Tq2Workspace &ws, unsigned int K) {
  if (ws.const_k == K)
    return;
  for (cl_mem *m : {&ws.c_zero, &ws.c_iota, &ws.c_count})
    if (*m != nullptr) {
      opencl::clReleaseMemObjectT(*m);
      *m = nullptr;
    }
  std::vector<int> zero(2 * K, 0), iota(K);
  for (unsigned int a = 0; a < K; ++a)
    iota[a] = (int)a;
  const int count = (int)K;
  ws.c_zero = tq2Upload(cc, zero.data(), zero.size() * sizeof(int));
  ws.c_iota = tq2Upload(cc, iota.data(), iota.size() * sizeof(int));
  ws.c_count = tq2Upload(cc, &count, sizeof(int));
  ws.const_k = K;
}

/** an unsigned environment knob, @a def when unset or not a number */
unsigned int tq2EnvUint(const char *name, unsigned int def) {
  const char *e = std::getenv(name);
  if (e == nullptr || *e == '\0')
    return def;
  char *end = nullptr;
  const unsigned long v = std::strtoul(e, &end, 10);
  return (end != e && *end == '\0' && v <= 1000000ul) ? (unsigned int)v : def;
}

/**
 * @brief How the portable prefill GEMMs split the routed rows (read once).
 *        Every choice gives the same outputs bit for bit (integer sums, the
 *        reference epilogue); this is an A/B switch.
 *  - NNTR_TQ2CX_FLAT_TILES=0: the one grouped launch over every expert's
 *    max_rows / TQ2P_TM m-blocks (tq2cx_gemm_grouped_portable); default:
 *    a list of only the tiles that hold rows (tq2cx_gemm_tiles_portable).
 *    At a 1006-token prompt (30 layers, Xe3 with the portable kernels) that
 *    is 1.08 M work-groups instead of 8.98 M and 1164 instead of 1563 ms.
 */
struct Tq2TileCfg {
  bool flat = true;
};
const Tq2TileCfg &tq2TileCfg() {
  static const Tq2TileCfg c = []() {
    Tq2TileCfg r;
    const char *f = std::getenv("NNTR_TQ2CX_FLAT_TILES");
    r.flat = !(f != nullptr && f[0] == '0');
    return r;
  }();
  return c;
}

/**
 * @brief Fused prefill quantizers (tq2moe_gather_quant_*, tq2GluQuantize)
 *        by default; NNTR_TQ2CX_QUANT_FUSE=0 keeps gather / GeGLU and the
 *        quantizer as separate launches (an A/B switch; same bytes). At a
 *        1006-token prompt on Xe3 the four launches take 156 ms of GPU time
 *        over 30 layers, the two fused ones 61 ms, and the fp32 gather
 *        buffer (R x H floats, 90 MB) is no longer allocated. Read once.
 */
bool tq2QuantFuse() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_TQ2CX_QUANT_FUSE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

/**
 * @brief tq2cx_gemm_tiles_v2_portable for the portable prefill GEMMs: the
 *        8-row tile list of tq2cx_gemm_tiles_portable, 8 rows x 4 channels
 *        per work-item, TQ2-order activation rows (the prefill quantizers then
 *        store that order), the biased unpack and the row-sum correction.
 *        Same integers and epilogue, so the same outputs bit for bit.
 *        Measured on Adreno 840 (routed 1K prompt, 30 layers of gate / up /
 *        down, harness): 667 ms against 1584 ms for the P-TILES kernel.
 *  - NNTR_TQ2CX_TILES_V2: unset = on for Qualcomm Adreno (vendor id
 *    0x5143, no DPAS), off on every other device; 1 = on
 *    (a non-image device takes the buffer form, e.g. Xe3 under
 *    NNTR_TQ2CX_FORCE_PORTABLE=1, for exactness checks); 0 = off.
 *  - NNTR_TQ2CX_TILES_V2_ACT=0|1|2 forces the activation fetch: 0 = buffer
 *    only, 1 = image2d (the default on image devices), 2 = image1d_buffer
 *    (the image2d form falls back to it above CL_DEVICE_IMAGE2D_MAX_HEIGHT
 *    rows).
 *  Read once.
 */
struct Tq2V2Cfg {
  int on = -1;  /**< -1 = device default */
  int act = -1; /**< -1 = device default */
};
const Tq2V2Cfg &tq2V2Cfg() {
  static const Tq2V2Cfg c = []() {
    Tq2V2Cfg r;
    const char *e = std::getenv("NNTR_TQ2CX_TILES_V2");
    if (e != nullptr && (e[0] == '0' || e[0] == '1'))
      r.on = e[0] - '0';
    const char *a = std::getenv("NNTR_TQ2CX_TILES_V2_ACT");
    if (a != nullptr && a[0] >= '0' && a[0] <= '2' && a[1] == '\0')
      r.act = a[0] - '0';
    return r;
  }();
  return c;
}

const char *tq2V2FormName(int form) {
  return form == 1 ? "image2d" : form == 2 ? "image1d_buffer" : "buffer";
}

/**
 * @brief The v2 kernel of one activation form (0 buffer, 1 image2d,
 *        2 image1d_buffer), or null when it does not build or the compiler
 *        gave it local memory (it declares none: a non-zero size means
 *        private arrays were placed there, the slow and once wrong-result
 *        code shape of the Adreno 0842 compiler). Checked and logged once per
 *        form; the caller holds tq2Mutex.
 */
ClContext::SharedPtrClKernel tq2V2Kernel(ClContext *cc, int form) {
  static int state[3] = {0, 0, 0}; /**< 0 unknown, 1 good, -1 refused */
  if (form < 0 || form > 2 || state[form] < 0)
    return nullptr;
  auto kp = cc->registerClKernel(
    tq2cx_gemm_portable_kernel, "tq2cx_gemm_tiles_v2_portable",
    tq2Options("-DTQ2P_V2=" + std::to_string(form)));
  if (state[form] == 0) {
    cl_ulong lmem = 0;
    cl_int err = CL_INVALID_KERNEL;
    if (kp && opencl::clGetKernelWorkGroupInfo != nullptr)
      err = opencl::clGetKernelWorkGroupInfo(
        kp->GetKernel(), cc->context_inst_.GetDeviceId(),
        CL_KERNEL_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
    const bool good = kp && err == CL_SUCCESS && lmem == 0;
    std::fprintf(stderr,
                 "[TQ2-V2] tq2cx_gemm_tiles_v2_portable form=%s built=%d "
                 "local_mem=%llu B%s\n",
                 tq2V2FormName(form), kp ? 1 : 0, (unsigned long long)lmem,
                 good ? "" : " -> not used");
    state[form] = good ? 1 : -1;
  }
  return state[form] > 0 ? kp : nullptr;
}

/** image limits of the device: image2d max height, max width, and the
 *  image1d_buffer max texels */
struct Tq2ImgLimits {
  size_t h2 = 0, w2 = 0, t1 = 0;
};
const Tq2ImgLimits &tq2ImgLimits(ClContext *cc) {
  static const Tq2ImgLimits l = [cc]() {
    Tq2ImgLimits r;
    const cl_device_id d = cc->context_inst_.GetDeviceId();
    if (opencl::clGetDeviceInfo(d, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(r.h2),
                                &r.h2, nullptr) != CL_SUCCESS)
      r.h2 = 0;
    if (opencl::clGetDeviceInfo(d, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(r.w2),
                                &r.w2, nullptr) != CL_SUCCESS)
      r.w2 = 0;
    if (opencl::clGetDeviceInfo(d, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                                sizeof(r.t1), &r.t1, nullptr) != CL_SUCCESS)
      r.t1 = 0;
    return r;
  }();
  return l;
}

/**
 * @brief The v2 form the prefill GEMMs of a layer take, -1 = the P-TILES
 *        kernel. Decided before the quantizers run, since v2 reads the rows
 *        in TQ2 order. Only the flat tile plan, records
 *        whose unit pairs are adjacent (row-major or TQ2_LAYV == 2) and
 *        K % 64 == 0 for both K (an even unit count).
 */
int tq2V2Plan(ClContext *cc, bool tiles, unsigned int H, unsigned int I,
              unsigned int R) {
  const Tq2V2Cfg &c = tq2V2Cfg();
  if (!tiles || c.on == 0)
    return -1;
  const bool img = cc->caps().image_v8c;
  // default on only on Qualcomm Adreno, the device it was measured on;
  // caps.image_v8c (vendor != Intel) would also take Mali / AMD / NVIDIA
  // OpenCL. NNTR_TQ2CX_TILES_V2=1 still forces it on any device.
  constexpr uint32_t qualcomm_vendor_id = 0x5143;
  if (c.on < 0 &&
      (cc->caps().dpas || cc->caps().vendor_id != qualcomm_vendor_id))
    return -1;
  const Tq2cxLayout lay = tq2cxStoredLayout();
  if ((lay.tiled() && lay.V != 2) || H % 64 != 0 || I % 64 != 0)
    return -1;
  int form = c.act >= 0 ? c.act : (img ? 1 : 0);
  // more rows than an image2d holds: the image1d_buffer form, else buffer
  const Tq2ImgLimits &lim = tq2ImgLimits(cc);
  if (form == 1 && R > lim.h2)
    form = 2;
  if (form == 2 && (size_t)R * (std::max(H, I) / 16) > lim.t1)
    form = 0;
  if (tq2V2Kernel(cc, form))
    return form;
  return form != 0 && tq2V2Kernel(cc, 0) ? 0 : -1;
}

/**
 * @brief An image view (form 1 image2d, 2 image1d_buffer) of the workspace
 *        int8 rows xq as rows of K bytes covering at least R rows (RGBA
 *        uint32 texels: K / 16 per row), or null when the device limits or
 *        the driver refuse it. The view covers the whole buffer, so it is
 *        made again only when xq grows (or K / the form changes).
 */
cl_mem tq2XqView(ClContext *cc, Tq2Workspace &ws, unsigned int K,
                 unsigned int R, int form) {
  const size_t tpr = K / 16; /**< texels per row */
  const size_t cap = ws.xq.bytes / K;
  const Tq2ImgLimits &lim = tq2ImgLimits(cc);
  size_t rows = 0;
  if (form == 1 && tpr <= lim.w2)
    rows = std::min(cap, lim.h2);
  else if (form == 2 && tpr > 0)
    rows = std::min(cap, lim.t1 / tpr);
  if (rows < R || ws.xq.buf == nullptr)
    return nullptr;
  Tq2Workspace::XqView *slot = nullptr;
  for (auto &v : ws.xq_view)
    if (v.img != nullptr && v.buf == ws.xq.buf && v.K == K && v.form == form &&
        v.rows >= R)
      return v.img;
  for (auto &v : ws.xq_view)
    if (slot == nullptr && (v.img == nullptr || v.buf != ws.xq.buf || v.K == K))
      slot = &v;
  if (slot == nullptr)
    slot = &ws.xq_view[0];
  if (slot->img != nullptr)
    opencl::clReleaseMemObjectT(slot->img);
  *slot = Tq2Workspace::XqView{};
  cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc d;
  std::memset(&d, 0, sizeof(d));
  if (form == 1) {
    // Pitch = K bytes (K / 16 RGBA32UI pixels), the rows are packed as the
    // quantizer writes them. H % 64 == I % 64 == 0 (tq2V2Plan), so the pitch
    // is a multiple of 64 B = 4 pixels: the same assumption as the v8c
    // activation image (blas_kernel_interface, pitch K). It is NOT a multiple
    // of what every driver reports: Adreno 840 reports
    // CL_DEVICE_IMAGE_PITCH_ALIGNMENT 64 px and still accepts pitch K (44 /
    // 176 px, in a kernel test and in the v8c path). So the reported alignment
    // is logged, not enforced; a driver that enforces it refuses
    // clCreateImage, and the caller falls back to image1d_buffer (no pitch),
    // then the buffer form, then the P-TILES kernel.
    static const cl_uint align = [cc]() {
      cl_uint a = 0;
      if (opencl::clGetDeviceInfo(cc->context_inst_.GetDeviceId(),
                                  CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(a),
                                  &a, nullptr) != CL_SUCCESS)
        a = 0;
      return a;
    }();
    static bool noted = false;
    if (align > 1 && tpr % align != 0 && !noted) {
      noted = true;
      std::fprintf(stderr,
                   "[TQ2-V2] image2d view pitch %u B = %zu px is not a "
                   "multiple of CL_DEVICE_IMAGE_PITCH_ALIGNMENT %u px: "
                   "relying on the driver (a refusal falls back)\n",
                   K, tpr, (unsigned)align);
    }
    d.image_type = CL_MEM_OBJECT_IMAGE2D;
    d.image_width = tpr;
    d.image_height = rows;
    d.image_row_pitch = K;
  } else {
    d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    d.image_width = rows * tpr;
  }
  d.buffer = ws.xq.buf;
  cl_int err = CL_SUCCESS;
  cl_mem img =
    opencl::clCreateImageT(cc->context_inst_.GetContextNoRetain(),
                           CL_MEM_READ_ONLY, &fmt, &d, nullptr, &err);
  if (err != CL_SUCCESS || img == nullptr) {
    static bool warned = false;
    if (!warned)
      std::fprintf(stderr,
                   "[TQ2-V2] %s view of the int8 rows (K %u, %zu rows) "
                   "refused (%d): another form is used\n",
                   tq2V2FormName(form), K, rows, err);
    warned = true;
    return nullptr;
  }
  slot->buf = ws.xq.buf;
  slot->K = K;
  slot->form = form;
  slot->rows = rows;
  slot->img = img;
  return img;
}

/** quantize gathered row r of the layer input (row src_row[r]) into the
 *  workspace int8 rows, M rows of K values; with @a perm each row is stored
 *  in TQ2 order (V8C_AQ_PERM, K % 32 == 0) */
void tq2GatherQuantize(ClContext *cc, Tq2Workspace &ws, const Tq2Act &in,
                       bool fp16, cl_mem src_row, unsigned int M,
                       unsigned int K, bool perm) {
  auto kp = cc->registerClKernel(
    kTq2MoeQuant, fp16 ? "tq2moe_gather_quant_f16" : "tq2moe_gather_quant_f32",
    perm ? tq2QuantOptions(cc) + " -DV8C_AQ_PERM" : tq2QuantOptions(cc));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: the fused quantizers did not build";
  cl_mem xq = ws.xqGet(cc, (size_t)M * K);
  cl_mem xs = ws.xs.get(cc, (size_t)M * sizeof(float));
  cl_mem xz = ws.xz.get(cc, (size_t)M * sizeof(int));
  cl_mem xr = ws.xrs.get(cc, (size_t)M * sizeof(int));
  const int Mi = (int)M, Ki = (int)K;
  bool ok = in.bind(kp, 0) &&
            kp->SetKernelArguments(1, &src_row, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &xq, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &xs, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &xz, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &xr, sizeof(cl_mem)) &&
            kp->SetKernelArguments(6, &Mi, sizeof(int)) &&
            kp->SetKernelArguments(7, &Ki, sizeof(int));
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: gather quantizer arguments";
  tq2Dispatch(cc, kp, 1, {(size_t)M * kV8cQuantLws, 1, 1},
              {kV8cQuantLws, 1, 1});
}

/**
 * @brief The portable prefill GEMMs' split of one layer's routed rows,
 *        shared by its gate, up and down GEMMs: the device tile list.
 */
struct Tq2Tiles {
  unsigned int n_tiles = 0; /**< TQ2P_TM-row tiles */
  cl_mem d_ro = nullptr, d_rc = nullptr, d_tiles = nullptr;
  ~Tq2Tiles() {
    for (cl_mem m : {d_ro, d_rc, d_tiles})
      if (m != nullptr)
        opencl::clReleaseMemObjectT(m);
  }
};

/**
 * @brief Plan the tiles of the routed rows: experts by row count, largest
 *        first.
 */
void tq2PlanTiles(ClContext *cc, const std::vector<int> &row_off,
                  const std::vector<int> &row_cnt, Tq2Tiles &t) {
  const unsigned int k = (unsigned int)row_cnt.size();
  std::vector<unsigned int> order(k);
  for (unsigned int a = 0; a < k; ++a)
    order[a] = a;
  std::stable_sort(
    order.begin(), order.end(),
    [&](unsigned int x, unsigned int y) { return row_cnt[x] > row_cnt[y]; });
  std::vector<int> tiles;
  for (const unsigned int a : order) {
    const int c = row_cnt[a];
    for (int m0 = 0; m0 < c; m0 += (int)kTq2pTm) {
      tiles.push_back((int)a);
      tiles.push_back(m0);
    }
  }
  t.n_tiles = (unsigned int)(tiles.size() / 2);
  t.d_ro = tq2Small(cc, row_off.data(), k * sizeof(int));
  t.d_rc = tq2Small(cc, row_cnt.data(), k * sizeof(int));
  if (t.n_tiles)
    t.d_tiles = tq2Small(cc, tiles.data(), tiles.size() * sizeof(int));
}

/** one portable prefill GEMM over the planned tiles, Y[R][N], R rows; with
 *  @a v2 >= 0 (tq2V2Plan) the 8-row tiles take tq2cx_gemm_tiles_v2_portable
 *  in that activation form (the rows are then in TQ2 order).
 *  @return false (nothing launched) when v2 was planned but no form can run
 *          now (image views refused and the buffer form refused): the rows
 *          are in TQ2 order, so the caller quantizes them again in plain
 *          order and calls again with v2 = -1 (the P-TILES kernel) */
bool tq2GroupedGemmTiles(ClContext *cc, Tq2Workspace &ws, const Tq2Stack &st,
                         const Tq2Tiles &t,
                         const std::vector<uint32_t> &rec_off, cl_mem Y,
                         unsigned int N, unsigned int K, unsigned int R,
                         int v2) {
  const unsigned int E = (unsigned int)rec_off.size();
  cl_mem d_off = tq2Small(cc, rec_off.data(), E * sizeof(uint32_t));
  cl_mem xq = ws.xq.buf, xs = ws.xs.buf, xz = ws.xz.buf;
  const int Ni = (int)N, Ki = (int)K;
  auto bind = [&](const ClContext::SharedPtrClKernel &kp, cl_mem tiles) {
    return kp->SetKernelArguments(0, &xq, sizeof(cl_mem)) &&
           kp->SetKernelArguments(1, &st.buf, sizeof(cl_mem)) &&
           kp->SetKernelArguments(2, &d_off, sizeof(cl_mem)) &&
           kp->SetKernelArguments(3, &t.d_ro, sizeof(cl_mem)) &&
           kp->SetKernelArguments(4, &t.d_rc, sizeof(cl_mem)) &&
           kp->SetKernelArguments(5, &xs, sizeof(cl_mem)) &&
           kp->SetKernelArguments(6, &xz, sizeof(cl_mem)) &&
           kp->SetKernelArguments(7, &Y, sizeof(cl_mem)) &&
           kp->SetKernelArguments(8, &Ni, sizeof(int)) &&
           kp->SetKernelArguments(9, &Ki, sizeof(int)) &&
           kp->SetKernelArguments(10, &tiles, sizeof(cl_mem));
  };
  bool ok = true;
  if (t.n_tiles && v2 >= 0) {
    // the planned form, else image1d_buffer (rows past the image2d height or
    // a refused image2d view), else the buffer form (no view)
    int form = -1;
    cl_mem view = nullptr;
    ClContext::SharedPtrClKernel kp;
    for (const int f : {v2 == 1 ? 1 : -1, v2 >= 1 ? 2 : -1, 0}) {
      if (f < 0 || !(kp = tq2V2Kernel(cc, f)))
        continue;
      view = f > 0 ? tq2XqView(cc, ws, K, R, f) : nullptr;
      if (f == 0 || view != nullptr) {
        form = f;
        break;
      }
    }
    if (form < 0) {
      static bool warned = false;
      if (!warned)
        std::fprintf(stderr,
                     "[TQ2-V2] no v2 form usable at GEMM time (K %u, "
                     "%u rows): the P-TILES kernel is used\n",
                     K, R);
      warned = true;
      opencl::clReleaseMemObjectT(d_off);
      return false;
    }
    cl_mem xr = ws.xrs.buf;
    ok = bind(kp, t.d_tiles) &&
         kp->SetKernelArguments(11, &xr, sizeof(cl_mem)) &&
         kp->SetKernelArguments(12, form > 0 ? &view : &xq, sizeof(cl_mem));
    if (ok)
      tq2Dispatch(
        cc, kp, 2,
        {((size_t)N + 4 * kTq2pGLws - 1) / (4 * kTq2pGLws) * kTq2pGLws,
         (size_t)t.n_tiles, 1},
        {kTq2pGLws, 1, 1});
  } else if (t.n_tiles) {
    const unsigned int nb = tq2GroupedNb();
    auto kp = cc->registerClKernel(
      tq2cx_gemm_portable_kernel, "tq2cx_gemm_tiles_portable",
      nb == 2 ? tq2Options() : tq2Options("-DTQ2P_NB=" + std::to_string(nb)));
    NNTR_THROW_IF(!kp, std::runtime_error)
      << "tq2cx_moe_experts_cl: tq2cx_gemm_tiles_portable did not build";
    ok = ok && bind(kp, t.d_tiles);
    if (ok)
      tq2Dispatch(
        cc, kp, 2,
        {((size_t)N + kTq2pGLws * nb - 1) / (kTq2pGLws * nb) * kTq2pGLws,
         (size_t)t.n_tiles, 1},
        {kTq2pGLws, 1, 1});
  }
  opencl::clReleaseMemObjectT(d_off);
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: portable tile GEMM arguments";
  return true;
}

} // namespace

namespace {

/** how a device-routed decode step of a streamed layer waits for its misses
 *  (NNTR_G4MOE_FSU_ASYNC, tq2AsyncMode) */
enum class Tq2AsyncMode {
  off,   /**< the host waits for the selection and reads the misses itself */
  event, /**< reader threads complete a user event the GEMVs wait on */
  join,  /**< reader threads make them resident; the host waits for that
              before it enqueues the GEMVs (no user event on the queue) */
};

/**
 * @brief The miss handling of one device-routed decode step of a streamed
 *        layer, off the host thread.
 *
 * @details The host enqueues the router and a marker after its select kernel
 * and moves on; the marker's callback posts this job on the reader pool. The
 * job reads the selected ids back (on the I/O queue), makes them resident
 * through the layer's acquire hook (the reads and the slot writes run on the
 * readers, this one included), writes the slots' offsets the GEMVs read, and
 * refreshes the recency. Then
 *  - event: completes the user event the step's GEMVs wait on and flushes the
 *    compute queue: a command behind a user event is not submitted after the
 *    event completes until something flushes its queue (measured on Adreno
 *    0842.19.8: without the flush every token stalled);
 *  - join: only signals; the host waits (Tq2AsyncJob::wait) before it
 *    enqueues the GEMVs.
 * A failure completes the event with an error status; the layer's next
 * join rethrows it.
 */
struct Tq2AsyncJob {
  ClContext *cc = nullptr;
  cl_command_queue io = nullptr; /**< the arena's I/O queue */
  cl_command_queue cq = nullptr; /**< the compute queue (flushed in event) */
  cl_mem ext = nullptr, off_gu = nullptr, off_d = nullptr;
  unsigned int K = 0, X = 0, E = 0;
  Tq2ExpertArena::Partition *part = nullptr;
  Tq2Stack *st = nullptr;
  // the descriptor's parts the job uses (the call's descriptor is gone)
  int fd = -1;
  const std::vector<uint64_t> *offset = nullptr;
  size_t record_bytes = 0;
  const std::vector<unsigned int> *slot_of = nullptr;
  std::function<void(const std::vector<unsigned int> &,
                     std::vector<std::pair<unsigned int, unsigned int>> &)>
    acquire;
  std::function<void(const std::vector<std::pair<unsigned int, unsigned int>> &,
                     bool)>
    filled;
  std::function<void(const std::vector<unsigned int> &)> refresh;
  /** the slots' last readers when the step was enqueued (compute-queue
   *  markers ahead of the select: complete when the job runs, checked) */
  std::vector<Tq2EventRef> last_use;
  /** event mode: the job's reference to the user event (released in run) */
  cl_event user = nullptr;
  /** hit / miss split: off_gu / off_d are the misses' offset tables; the
   *  hits' GEMVs already run from the select's (from the device slot table
   *  st->off_tab / st->res_flag, which the job then updates) */
  bool split = false;
  /** the misses this step read (valid once done) */
  size_t misses = 0;
  /** the layer (MoeExpertFsu::layer_index), for the trace */
  unsigned int layer = 0;
  /** the next layer's prediction this step enqueued (NNTR_G4MOE_FSU_PREGATE)
   *  or null; attached by the host after the job may already run, so under
   *  mu: the job hands it on (handPrediction) once it is done, and a host
   *  that attaches it later does that for the job (attachPrediction) */
  std::shared_ptr<struct Tq2Prediction> pred;
  bool pred_closed = false;
  std::chrono::steady_clock::time_point posted;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  std::string error;

  void run();
  /**
   * @brief The select marker ended with @a status (an error, not
   *        CL_COMPLETE): the routing was not written, so nothing is read or
   *        made resident. Logs, completes the user event with an error (the
   *        GEMVs behind it fail) and leaves @a status in error: the layer's
   *        next join throws it.
   */
  void fail(cl_int status);
  /** complete the user event with @a status, flush, mark done */
  void finish(cl_int status);
  /** after finish: the job's arrival at its prediction (if attached) */
  void handPrediction();
  /** host: attach @a p, or arrive for the job when it is done already */
  void attachPrediction(const std::shared_ptr<struct Tq2Prediction> &p);
  /** run() / fail() is done (does not wait) */
  bool ready() {
    std::lock_guard<std::mutex> lk(mu);
    return done;
  }
  /** wait until run() / fail() is done */
  void wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this] { return done; });
  }
  /** wait, then rethrow the job's failure */
  void join() {
    wait();
    NNTR_THROW_IF(!error.empty(), std::runtime_error)
      << "tq2cx_moe_experts_cl: streamed decode step failed: " << error;
  }
};

unsigned int tq2PregateM();

/** async decode totals, printed at exit when used */
struct Tq2AsyncStats {
  std::atomic<unsigned long long> jobs{0}, misses{0}, job_us{0}, lag_us{0},
    tails{0}, deferred{0}, host_wait_us{0}, failed{0}, split{0}, no_miss{0},
    table_uploads{0}, event_skipped{0};
  /** the job's phases, summed: routing readback, the misses' reads and slot
   *  writes (from the first read submitted to the last write returned), the
   *  offset / slot table writes */
  std::atomic<unsigned long long> rb_us{0}, io_us{0}, tab_us{0};
  /** next-layer prefetch: predictions issued, the decode tokens (steps of
   *  layer 0), misses served from the read-ahead records (ready or after a
   *  wait for their running read), misses whose queued read-ahead the step
   *  took over, and the time the jobs spent waiting for running reads */
  std::atomic<unsigned long long> pg_tasks{0}, tokens{0}, pg_served{0},
    pg_waited{0}, pg_taken{0}, pg_wait_us{0};
  /** the host -> slot copies of read-ahead records on the step's path
   *  (every served record), and the time the job spent in them; the misses'
   *  own slot writes (from the readers, summed) */
  std::atomic<unsigned long long> pg_copies{0}, pg_copy_us{0}, miss_writes{0},
    miss_write_us{0};
  ~Tq2AsyncStats() {
    if (failed.load() != 0)
      std::fprintf(stderr,
                   "[TQ2CX-FSU-ASYNC] ERROR: %llu decode steps failed "
                   "(routing step status not CL_COMPLETE)\n",
                   failed.load());
    const unsigned long long n = jobs.load();
    if (n == 0)
      return;
    const unsigned long long t = std::max(1ULL, tails.load());
    std::fprintf(
      stderr,
      "[TQ2CX-FSU-ASYNC] jobs=%llu misses=%llu job_ms_avg=%.3f "
      "callback_lag_ms_avg=%.3f tails=%llu deferred=%llu "
      "tail_host_wait_ms_avg=%.3f split=%llu no_miss_tails=%llu "
      "table_uploads=%llu event_skipped=%llu readback_ms_avg=%.3f "
      "io_ms_avg=%.3f tables_ms_avg=%.3f "
      "readers=%u pregate=%u pg_tasks=%llu pg_staged=%llu "
      "pg_served=%llu pg_wasted=%llu pg_failed=%llu miss_writes=%llu "
      "miss_write_ms_per_rec=%.3f\n",
      n, misses.load(), job_us.load() / 1000.0 / (double)n,
      lag_us.load() / 1000.0 / (double)n, tails.load(), deferred.load(),
      host_wait_us.load() / 1000.0 / (double)t, split.load(), no_miss.load(),
      table_uploads.load(), event_skipped.load(),
      rb_us.load() / 1000.0 / (double)n, io_us.load() / 1000.0 / (double)n,
      tab_us.load() / 1000.0 / (double)n, ExpertReaderPool::global().readers(),
      tq2PregateM(), pg_tasks.load(), pg_staged(), pg_served.load(),
      pg_wasted(), pg_failed(), miss_writes.load(),
      miss_writes.load() ? miss_write_us.load() / 1000.0 / miss_writes.load()
                         : 0.0);
    if (tq2PregateM() > 0)
      printPregate(n);
  }
  unsigned long long pg_staged() const;
  unsigned long long pg_wasted() const;
  unsigned long long pg_failed() const;
  /** [TQ2CX-FSU-PREGATE]: per decode token */
  void printPregate(unsigned long long n) const;
};
Tq2AsyncStats &tq2AsyncStats() {
  static Tq2AsyncStats s;
  return s;
}

unsigned long long tq2UsSince(std::chrono::steady_clock::time_point t0) {
  return (unsigned long long)
    std::chrono::duration_cast<std::chrono::microseconds>(
           std::chrono::steady_clock::now() - t0)
      .count();
}

/**
 * @brief NNTR_G4MOE_FSU_PREGATE=m (0..K + RT_EXT; default 8, 0 = off):
 *        next-layer prefetch. A streamed layer's device-routed decode step
 *        also runs the next streamed layer's router on its own router row
 *        (pre-gating, a lookahead probe's best variant: 0.66 of the
 *        actual top-8 at m = 8, 0.79 at 12) and the top-m predicted experts
 *        that the next layer does not hold are read ahead at low priority.
 *        The next layer's step takes a miss found there instead of reading
 *        it. Routing, residency decisions and outputs are unchanged. 8 by
 *        default on every platform: measured faster than 0 with the same
 *        tokens on Xe3 (P4 1K C=16 steady 52.7 -> 46.2 ms/token, P2-256
 *        59.7 -> 52.3) and on Adreno 840 (two phones, P4 1K C=16 115.3 ->
 *        93.5 ms/token with LRFU, P2-256 103.5 -> 82.7); 12 reads 1.7x the
 *        bytes for +1 served record per token and was slower on both.
 *        Read once.
 */
// (m = 10 in the same sweep: P4 1K 69.7, P2-256 80.3 ms per token against
// 67.6 / 74.3 at 8; 1.6x the wasted reads for <= 2 more served per token)
unsigned int tq2PregateM() {
  static const unsigned int m = []() {
    const unsigned int v = tq2EnvUint("NNTR_G4MOE_FSU_PREGATE", 8);
    return std::min(v, kTq2RouteMaxK + kTq2RouteExt);
  }();
  return m;
}

/**
 * @brief NNTR_G4MOE_FSU_PREGATE_LAYERS=a-b (default 5-28): the layers
 *        (MoeExpertFsu::layer_index) whose experts are predicted. A routing
 *        probe: the next-layer prediction of layers 0-4 and 29 is poor (and
 *        12 / 16 / 17 / 19 weaker); a layer outside gets no prediction, so
 *        no router launches and no reads for it. Kept by a sweep
 *        (Xe3, C=16 cold, 2 runs, same tokens; steady ms per token P4 1K /
 *        P2-256): 5-28 67.6 / 74.3, 3-29 69.4 / 76.5, 1-29 70.2 / 74.1 --
 *        the wider ranges serve 1.5-4.8 more records per token (served
 *        fraction 0.39 -> 0.42-0.46) for 10 to 18 more wasted reads, and are
 *        not faster; the trace simulation's ceiling at 5-28 (every read in
 *        time) is 25.1 served of 58.5 misses (0.43), which the phones
 *        already reach (24.7). Read once.
 */
bool tq2PregateTarget(unsigned int layer) {
  static const std::pair<unsigned int, unsigned int> r = []() {
    std::pair<unsigned int, unsigned int> d{5, 28};
    const char *e = std::getenv("NNTR_G4MOE_FSU_PREGATE_LAYERS");
    if (e == nullptr || e[0] == '\0')
      return d;
    unsigned int a = 0, b = 0;
    if (std::sscanf(e, "%u-%u", &a, &b) == 2 && a <= b)
      return std::make_pair(a, b);
    std::fprintf(stderr,
                 "[TQ2CX-FSU-PREGATE] NNTR_G4MOE_FSU_PREGATE_LAYERS=%s: not "
                 "a-b, 5-28 kept\n",
                 e);
    return d;
  }();
  return layer >= r.first && layer <= r.second;
}

/**
 * @brief NNTR_G4MOE_FSU_PREGATE_INFLIGHT=n (1..8, default 2): read-ahead
 *        reads of one layer submitted at once (the next goes when one ends).
 *        Kept by a sweep (Xe3, C=16 cold, 2 runs, same tokens;
 *        steady ms per token P4 1K / P2-256): 2: 67.6 / 74.3, 3: 72.0 /
 *        78.5, 4: 69.2 / 82.7 -- a wider window serves more (P4 22.8 -> 24.7
 *        per token, P2 34.1 -> 38.6, fewer taken over) but its reads crowd
 *        the step's own (job 0.63 -> 0.68-0.75 ms). Read once.
 */
unsigned int tq2PregateInflight() {
  static const unsigned int n = std::max(
    1u, std::min(8u, tq2EnvUint("NNTR_G4MOE_FSU_PREGATE_INFLIGHT", 2)));
  return n;
}

/** the process's read-ahead records; never destroyed (a reader thread may
 *  end a read into it during exit) */
ExpertStageCache &tq2StageCache() {
  static ExpertStageCache *c =
    new ExpertStageCache(ExpertReaderPool::global(), tq2PregateInflight());
  return *c;
}

unsigned long long Tq2AsyncStats::pg_staged() const {
  return tq2PregateM() > 0 ? tq2StageCache().stats().staged.load() : 0;
}
unsigned long long Tq2AsyncStats::pg_wasted() const {
  return tq2PregateM() > 0 ? tq2StageCache().stats().wasted.load() : 0;
}
unsigned long long Tq2AsyncStats::pg_failed() const {
  return tq2PregateM() > 0 ? tq2StageCache().stats().failed.load() : 0;
}
void Tq2AsyncStats::printPregate(unsigned long long n) const {
  const auto &c = tq2StageCache().stats();
  const double tok = std::max(1.0, (double)tokens.load());
  const unsigned long long miss = misses.load(), srv = pg_served.load();
  std::fprintf(
    stderr,
    "[TQ2CX-FSU-PREGATE] m=%u inflight=%u tokens=%llu predictions=%llu "
    "predicted_per_tok=%.2f staged_per_tok=%.2f reads_per_tok=%.2f "
    "served_per_tok=%.2f served_after_wait=%llu taken_over_per_tok=%.2f "
    "wasted_per_tok=%.2f cancelled_per_tok=%.2f late=%llu failed=%llu "
    "misses_per_tok=%.2f crit_misses_per_tok=%.2f served_frac=%.3f "
    "wait_ms_avg=%.3f job_ms_avg=%.3f\n",
    tq2PregateM(), tq2PregateInflight(), tokens.load(), c.predictions.load(),
    c.predicted.load() / tok, c.staged.load() / tok, c.reads.load() / tok,
    srv / tok, pg_waited.load(), pg_taken.load() / tok, c.wasted.load() / tok,
    c.cancelled.load() / tok, c.late.load(), c.failed.load(), miss / tok,
    (double)(miss - std::min(miss, srv)) / tok,
    miss ? (double)srv / (double)miss : 0.0,
    pg_waited.load() ? pg_wait_us.load() / 1000.0 / pg_waited.load() : 0.0,
    job_us.load() / 1000.0 / (double)std::max(1ULL, n));
  const unsigned long long cp = pg_copies.load();
  std::fprintf(stderr,
               "[TQ2CX-FSU-PREGATE-COPY] copies_per_tok=%.2f "
               "copy_ms_per_tok=%.3f copy_ms_per_rec=%.4f\n",
               cp / tok, pg_copy_us.load() / 1000.0 / tok,
               cp ? pg_copy_us.load() / 1000.0 / cp : 0.0);
}

/**
 * @brief NNTR_G4MOE_FSU_PREFILL_LOOKAHEAD=0|1 (default 1): a short prefill
 *        call (fewer rows than MoeExpertFsu::spec_min_tokens, so its layers
 *        are not streamed) reads the next layer's predicted experts
 *        (MoeExpertFsu::next_ahead) into that layer's own slots once its own
 *        experts are enqueued, and runs its chunks resident experts first.
 *        Read once.
 */
bool tq2PrefillAheadOn() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_PREFILL_LOOKAHEAD");
    return e == nullptr || e[0] != '0';
  }();
  return v;
}

/**
 * @brief The short prefill's next-layer read-ahead in flight (one at a
 *        time: the next layer's call joins it first). Under tq2Mutex.
 */
struct Tq2PrefillAhead {
  bool pending = false;
  unsigned int layer = UINT32_MAX;     /**< the next layer's index */
  const MoeExpertFsu *nx = nullptr;    /**< its descriptor (the layer's own) */
  std::vector<unsigned int> predicted; /**< the whole prediction */
  /** (expert, reserved slot) per read */
  std::vector<std::pair<unsigned int, unsigned int>> issued;
  std::vector<ExpertRead> reads;
  ExpertReaderPool::Ticket ticket = 0;
  std::chrono::steady_clock::time_point t_issue;
  /** joined: the next layer's call compares with its routing (stats) */
  bool stats_due = false;
  std::vector<unsigned int> admitted; /**< written and admitted */
};

Tq2PrefillAhead &tq2PrefillAhead() {
  static Tq2PrefillAhead a;
  return a;
}

/** [TQ2CX-FSU-PREFILL-AHEAD] totals, printed at exit when used */
struct Tq2PrefillAheadStats {
  std::atomic<unsigned long long> calls{0}, predicted{0}, active{0}, covered{0},
    issued{0}, used{0}, failed{0}, join_wait_us{0}, ahead_us{0},
    reserve_stop{0}, resident{0};
  ~Tq2PrefillAheadStats() {
    const unsigned long long n = calls.load();
    if (n == 0)
      return;
    const double a = std::max(1ULL, active.load());
    std::fprintf(
      stderr,
      "[TQ2CX-FSU-PREFILL-AHEAD] layer_calls=%llu active=%.1f/call "
      "predicted=%.1f/call covered=%llu (%.1f%% of active) "
      "resident_at_issue=%llu issued=%llu used=%llu (%.1f%% of issued, "
      "%.1f%% of active) failed=%llu reserve_stops=%llu "
      "issue_to_join_ms_avg=%.2f join_wait_ms_avg=%.2f\n",
      n, active.load() / (double)n, predicted.load() / (double)n,
      covered.load(), 100.0 * covered.load() / a, resident.load(),
      issued.load(), used.load(),
      issued.load() ? 100.0 * used.load() / issued.load() : 0.0,
      100.0 * used.load() / a, failed.load(), reserve_stop.load(),
      ahead_us.load() / 1000.0 / (double)n,
      join_wait_us.load() / 1000.0 / (double)n);
  }
};

Tq2PrefillAheadStats &tq2PrefillAheadStats() {
  static Tq2PrefillAheadStats s;
  return s;
}

/**
 * @brief Wait for the read-ahead in flight, if any, and end its
 *        reservations: a record written is admitted into its slot (resident,
 *        no access counted), a failed one frees its slot. Before anything
 *        else touches the slots or the policies of the streamed layers.
 *        Caller holds tq2Mutex.
 */
void tq2PrefillAheadJoin() {
  Tq2PrefillAhead &a = tq2PrefillAhead();
  if (!a.pending)
    return;
  a.pending = false;
  const auto t0 = std::chrono::steady_clock::now();
  try {
    ExpertReaderPool::global().wait(a.ticket, /*help=*/true);
  } catch (...) {
    // per read below: a failed one is not admitted
  }
  auto &s = tq2PrefillAheadStats();
  s.join_wait_us += tq2UsSince(t0);
  s.ahead_us +=
    (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
      t0 - a.t_issue)
      .count();
  Tq2ExpertArena::Partition *np = Tq2ExpertArena::get().byIndex(a.layer);
  a.admitted.clear();
  for (size_t i = 0; i < a.issued.size(); ++i) {
    const bool ok = a.reads[i].error == 0;
    const unsigned int e = a.issued[i].first, slot = a.issued[i].second;
    if (ok && np != nullptr && slot < np->slots) {
      np->ready[slot].reset();
      std::lock_guard<std::mutex> lk(np->lu_mu);
      np->last_use[slot].reset();
    }
    if (!ok)
      s.failed++;
    a.nx->release(e, ok);
    if (ok)
      a.admitted.push_back(e);
  }
  a.reads.clear();
  a.issued.clear();
  a.stats_due = true;
}

/**
 * @brief The next layer's call: compare the prediction with its routing
 *        (stats only). Caller holds tq2Mutex.
 */
void tq2PrefillAheadTally(const MoeExpertFsu &fsu,
                          const std::vector<unsigned int> &active) {
  Tq2PrefillAhead &a = tq2PrefillAhead();
  if (!a.stats_due)
    return;
  a.stats_due = false;
  if (a.layer != fsu.layer_index)
    return;
  auto in = [&active](unsigned int e) {
    return std::binary_search(active.begin(), active.end(), e);
  };
  auto &s = tq2PrefillAheadStats();
  s.calls++;
  s.active += active.size();
  s.predicted += a.predicted.size();
  for (unsigned int e : a.predicted)
    s.covered += in(e) ? 1 : 0;
  s.issued += a.admitted.size();
  for (unsigned int e : a.admitted)
    s.used += in(e) ? 1 : 0;
}

/**
 * @brief Read the predicted experts of the next layer (fsu.next_ahead) that
 *        it does not hold into its own slots, on the reader pool, and return
 *        at once (the next layer's call joins them). A slot comes from the
 *        next layer's policy (reserve: a free slot or a victim outside the
 *        prediction, never more than its slots minus a call's top-k); the
 *        reader waits for the slot's last reader (a marker of the layer's
 *        previous call, a token or a turn back: complete), then writes the
 *        record. No staging memory beyond the readers' own buffers. Caller
 *        holds tq2Mutex, after this call's own experts are enqueued.
 */
void tq2PrefillAheadIssue(ClContext *cc, const MoeExpertFsu &fsu,
                          const Tq2Stack &shape) {
  const MoeExpertFsu *nx = fsu.next;
  if (!tq2PrefillAheadOn() || fsu.next_ahead == nullptr ||
      fsu.next_ahead->empty() || nx == nullptr || !nx->reserve ||
      !nx->release || nx->fd < 0 || nx->offset == nullptr ||
      nx->layer_index == UINT32_MAX || nx->record_bytes != fsu.record_bytes ||
      nx->slots >= nx->offset->size())
    return;
  Tq2ExpertArena &arena = Tq2ExpertArena::get();
  Tq2ExpertArena::Partition *np = arena.byIndex(nx->layer_index);
  if (np == nullptr || np->slots != nx->slots ||
      np->slot_bytes != shape.slot_stride)
    return;
  tq2PrefillAheadJoin(); // none in flight normally (joined at the call)
  // a decode step of the next layer from the previous token may still run
  // its hooks on a reader thread: done before its policy is touched
  for (auto &kv : tq2Stacks())
    if (kv.second->async) {
      auto job = std::move(kv.second->async);
      job->join();
    }
  Tq2PrefillAhead &a = tq2PrefillAhead();
  a = Tq2PrefillAhead();
  a.layer = nx->layer_index;
  a.nx = nx;
  a.predicted = *fsu.next_ahead;
  auto &s = tq2PrefillAheadStats();
  for (unsigned int e : a.predicted) {
    if (e >= nx->offset->size())
      continue;
    if (nx->resident && nx->resident(e)) {
      s.resident++;
      continue;
    }
    const unsigned int slot = nx->reserve(e, a.predicted);
    if (slot == std::numeric_limits<unsigned int>::max() || slot >= np->slots) {
      if (slot != std::numeric_limits<unsigned int>::max())
        nx->release(e, false);
      s.reserve_stop++;
      break; // no room left (at most slots - top-k reserved)
    }
    a.issued.emplace_back(e, slot);
  }
  a.t_issue = std::chrono::steady_clock::now();
  a.stats_due = true; // a call that issues nothing still tallies
  a.admitted.clear();
  if (a.issued.empty())
    return;
  // the slots' last readers are compute-queue markers: hand the queue over
  opencl::clFlush(cc->command_queue_inst_.GetCommandQueue());
  cl_command_queue lowq = arena.ioQueueLow(cc);
  const size_t rec = nx->record_bytes;
  const size_t rec_gu = tq2RecordBytes(shape.I, shape.H);
  const size_t gu = shape.gu_bytes;
  const bool packed = rec == shape.slot_bytes;
  a.reads.resize(a.issued.size());
  for (size_t i = 0; i < a.issued.size(); ++i) {
    const unsigned int e = a.issued[i].first, slot = a.issued[i].second;
    ExpertRead &r = a.reads[i];
    r.fd = nx->fd;
    r.offset = (*nx->offset)[e];
    r.bytes = rec;
    const size_t at = (size_t)slot * np->slot_bytes + tq2SlotShift(*nx, e);
    r.consume = [lowq, np, slot, at, packed, rec_gu,
                 gu](const uint8_t *src, const ExpertRead &q) -> int {
      const Tq2EventRef lu = np->lastUse(slot);
      if (lu && lu->ev != nullptr) {
        cl_event ev = lu->ev;
        if (opencl::clWaitForEvents(1, &ev) != CL_SUCCESS)
          return EIO;
      }
      if (packed)
        return Tq2ExpertArena::slotWrite(lowq, *np, at, q.bytes, src) ==
                   CL_SUCCESS
                 ? 0
                 : EIO;
      const size_t len[3] = {rec_gu, rec_gu, q.bytes - 2 * rec_gu};
      const size_t dst[3] = {0, gu, 2 * gu};
      const size_t from[3] = {0, rec_gu, 2 * rec_gu};
      for (int k = 0; k < 3; ++k)
        if (Tq2ExpertArena::slotWrite(lowq, *np, at + dst[k], len[k],
                                      src + from[k]) != CL_SUCCESS)
          return EIO;
      return 0;
    };
  }
  a.ticket = ExpertReaderPool::global().submit(&a.reads);
  a.pending = true;
}

/**
 * @brief One next-layer prediction (pre-gating) of a decode step. It needs
 *        two things: its marker (the next layer's router kernels wrote ext)
 *        and the predicting step's own job (its misses go first). Whichever
 *        comes last issues it: the marker's callback posts issue() on the
 *        reader pool, the job runs it on its way out. issue() never waits on
 *        another task: one small blocking readback of the ids on the I/O
 *        queue (no wait list; the marker is complete), then low-priority
 *        detached reads (ExpertStageCache::stage). Replaces an earlier
 *        prefetch task, which was posted on the pool and blocked a reader
 *        waiting for the step's job: with 2 readers (Android) the step's
 *        misses then ran at queue depth 1, and with one reader it could
 *        deadlock (a helping wait skips posted tasks).
 */
struct Tq2Prediction : std::enable_shared_from_this<Tq2Prediction> {
  cl_command_queue io = nullptr;
  cl_mem ext = nullptr; /**< [X] predicted ids, rank order */
  unsigned int X = 0, m = 0, E = 0;
  unsigned int layer = 0; /**< the next (predicted) layer's index */
  uint64_t gen = 0;       /**< its stage-cache generation at enqueue */
  ExpertStageCache::Source src;
  std::vector<unsigned int> slot_of; /**< the next layer's, at enqueue */
  std::atomic<int> arrivals{2};
  std::atomic<bool> marker_ok{false};
  std::mutex mu;
  std::condition_variable cv;
  bool issued = false;

  void arrive(bool from_callback) {
    if (arrivals.fetch_sub(1) != 1)
      return;
    if (from_callback) {
      // a driver callback thread: no blocking call here
      auto self = shared_from_this();
      ExpertReaderPool::global().post([self]() { self->issue(); });
    } else {
      issue();
    }
  }
  void issue() {
    if (marker_ok.load()) {
      std::vector<uint32_t> ranked(X);
      const cl_int err =
        opencl::clEnqueueReadBuffer(io, ext, CL_TRUE, 0, X * sizeof(uint32_t),
                                    ranked.data(), 0, nullptr, nullptr);
      if (err == CL_SUCCESS) {
        constexpr unsigned int kNo = std::numeric_limits<unsigned int>::max();
        std::vector<unsigned int> ids;
        for (unsigned int i = 0; i < std::min(m, X); ++i) {
          const unsigned int e = ranked[i];
          if (e < E && e < slot_of.size() && slot_of[e] == kNo)
            ids.push_back(e);
        }
        try {
          tq2StageCache().stage(layer, gen, src, ids);
        } catch (...) {
        }
      }
      tq2AsyncStats().pg_tasks++;
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      issued = true;
    }
    cv.notify_all();
  }
  /** host: wait until issue() ran (its readback of ext is done) */
  void waitIssued() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this] { return issued; });
  }
};

/** the prediction marker's callback: one of the prediction's two arrivals */
void CL_CALLBACK tq2PregateReady(cl_event ev, cl_int status, void *data) {
  auto *holder = static_cast<std::shared_ptr<Tq2Prediction> *>(data);
  std::shared_ptr<Tq2Prediction> p = std::move(*holder);
  delete holder;
  p->marker_ok = status == CL_COMPLETE;
  p->arrive(/*from_callback=*/true);
  opencl::clReleaseEvent(ev);
}

void Tq2AsyncJob::handPrediction() {
  std::shared_ptr<Tq2Prediction> p;
  {
    std::lock_guard<std::mutex> lk(mu);
    p = std::move(pred);
    pred_closed = true;
  }
  if (p)
    p->arrive(/*from_callback=*/false);
}

void Tq2AsyncJob::attachPrediction(const std::shared_ptr<Tq2Prediction> &p) {
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!pred_closed) {
      pred = p;
      return;
    }
  }
  p->arrive(/*from_callback=*/false); // the job is done: its arrival
}

void Tq2AsyncJob::run() {
  const auto t0 = std::chrono::steady_clock::now();
  tq2AsyncStats().lag_us += tq2UsSince(posted);
  std::vector<std::pair<unsigned int, unsigned int>> fills;
  bool acquired = false;
  cl_int status = CL_COMPLETE;
  try {
    std::vector<uint32_t> ranked(X);
    cl_int err =
      opencl::clEnqueueReadBuffer(io, ext, CL_TRUE, 0, X * sizeof(uint32_t),
                                  ranked.data(), 0, nullptr, nullptr);
    tq2AsyncStats().rb_us += tq2UsSince(t0);
    NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
      << "routing readback failed (" << err << ")";
    std::vector<unsigned int> ids(ranked.begin(), ranked.begin() + K);
    std::sort(ids.begin(), ids.end()); // the order ids[] / w[] are in
    for (unsigned int e : ids)
      NNTR_THROW_IF(e >= E, std::runtime_error) << "routed expert " << e;
    acquire(ids, fills);
    acquired = true;
    misses = fills.size();
    if (split) {
      // the device split the experts by the slot map it was given
      // (st->tab_slots); the policy must agree, or a "hit" GEMV has read a
      // slot that does not hold its expert
      constexpr unsigned int kNo = std::numeric_limits<unsigned int>::max();
      std::vector<unsigned int> dev_miss, pol_miss;
      for (unsigned int e : ids)
        if (e >= st->tab_slots.size() || st->tab_slots[e] == kNo)
          dev_miss.push_back(e);
      for (const auto &f : fills)
        pol_miss.push_back(f.first);
      std::sort(pol_miss.begin(), pol_miss.end());
      NNTR_THROW_IF(dev_miss != pol_miss, std::runtime_error)
        << "hit / miss split: the device found " << dev_miss.size()
        << " misses, the residency policy " << pol_miss.size();
    }
    // the misses: read and written into their slots by the readers; this
    // job runs on one of them and reads alongside (wait with help)
    if (!fills.empty()) {
      // A slot's last reader is a marker the compute queue got before this
      // step's select: the queue is in order, so it completed before the
      // select did, and the select's completion started this job. The
      // writes therefore take no wait list. They must not: a blocking write
      // on the I/O queue that names a compute-queue event, even a complete
      // one, does not return on Adreno (0842.19) while the compute queue
      // holds commands behind a user event not yet set -- the GEMVs behind
      // this job's own event -- so the job never sets it. Checked, not
      // assumed: a last reader not complete fails the step.
      for (const auto &f : fills) {
        const unsigned int sl = f.second;
        NNTR_THROW_IF(sl >= part->slots || f.first >= offset->size(),
                      std::runtime_error)
          << "slot " << sl << " expert " << f.first;
        const Tq2EventRef &lu = last_use[sl];
        cl_int est = CL_COMPLETE;
        if (lu && lu->ev != nullptr && opencl::clGetEventInfo != nullptr &&
            opencl::clGetEventInfo(lu->ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                   sizeof(est), &est, nullptr) != CL_SUCCESS)
          est = CL_COMPLETE; // no status: the queue order still holds
        NNTR_THROW_IF(est != CL_COMPLETE, std::runtime_error)
          << "slot " << sl << ": its last reader (status " << est
          << ") was not complete when the step's select was";
      }
      const size_t slot_bytes = part->slot_bytes;
      std::vector<ExpertRead> reads;
      reads.reserve(fills.size());
      // misses the previous layer's step read ahead (tq2PregateM): a record
      // read is copied from host memory below; one being read is waited for
      // after this step's own reads (a running read, never a task); one
      // whose read nobody started is taken over (that read is cancelled) and
      // read here like any miss
      using Staged =
        std::pair<std::shared_ptr<ExpertStageCache::Entry>, size_t>;
      std::vector<Staged> staged, flying;
      const bool pg = tq2PregateM() > 0;
      for (size_t i = 0; i < fills.size(); ++i) {
        const unsigned int e = fills[i].first, sl = fills[i].second;
        // where the record goes in the partition
        const size_t rec_at =
          (size_t)sl * slot_bytes + tq2SlotShift((*offset)[e]);
        if (pg) {
          auto tk = tq2StageCache().claim(layer, e);
          if (tk.kind == ExpertStageCache::Claim::ready) {
            staged.emplace_back(std::move(tk.entry), rec_at);
            continue;
          }
          if (tk.kind == ExpertStageCache::Claim::in_flight) {
            flying.emplace_back(std::move(tk.entry), rec_at);
            continue;
          }
          if (tk.kind == ExpertStageCache::Claim::take_over)
            tq2AsyncStats().pg_taken++;
        }
        Tq2ExpertArena::Partition *pp = part;
        cl_command_queue q = io;
        const size_t at = rec_at;
        reads.emplace_back();
        ExpertRead &r = reads.back();
        r.fd = fd;
        r.offset = (*offset)[e];
        r.bytes = record_bytes;
        r.consume = [q, pp, at](const uint8_t *src,
                                const ExpertRead &rd) -> int {
          const auto w0 = std::chrono::steady_clock::now();
          const cl_int err =
            Tq2ExpertArena::slotWrite(q, *pp, at, rd.bytes, src);
          tq2AsyncStats().miss_write_us += tq2UsSince(w0);
          tq2AsyncStats().miss_writes++;
          return err == CL_SUCCESS ? 0 : EIO;
        };
      }
      ExpertReaderPool &pool = ExpertReaderPool::global();
      const auto t_io = std::chrono::steady_clock::now();
      const ExpertReaderPool::Ticket tk = pool.submit(&reads);
      // the staged misses, while the readers read the rest: a host -> slot
      // copy each, on this step's path (timed: pg_copy_us)
      for (const Staged &sg : staged) {
        const auto c0 = std::chrono::steady_clock::now();
        const cl_int werr = Tq2ExpertArena::slotWrite(
          io, *part, sg.second, record_bytes, sg.first->data());
        tq2AsyncStats().pg_copy_us += tq2UsSince(c0);
        tq2AsyncStats().pg_copies++;
        tq2StageCache().release(sg.first);
        NNTR_THROW_IF(werr != CL_SUCCESS, std::runtime_error)
          << "staged slot write failed (" << werr << ")";
        tq2AsyncStats().pg_served++;
      }
      pool.wait(tk, /*help=*/true);
      // the records being read ahead when the step claimed them: a failed
      // one is read now
      std::vector<ExpertRead> late;
      for (const Staged &sg : flying) {
        const auto w0 = std::chrono::steady_clock::now();
        const bool ok = tq2StageCache().waitReady(*sg.first);
        tq2AsyncStats().pg_wait_us += tq2UsSince(w0);
        if (ok) {
          const auto c0 = std::chrono::steady_clock::now();
          const cl_int werr = Tq2ExpertArena::slotWrite(
            io, *part, sg.second, record_bytes, sg.first->data());
          tq2AsyncStats().pg_copy_us += tq2UsSince(c0);
          tq2AsyncStats().pg_copies++;
          tq2StageCache().release(sg.first);
          NNTR_THROW_IF(werr != CL_SUCCESS, std::runtime_error)
            << "staged slot write failed (" << werr << ")";
          tq2AsyncStats().pg_served++;
          tq2AsyncStats().pg_waited++;
          continue;
        }
        tq2StageCache().release(sg.first);
        late.emplace_back();
        ExpertRead &r = late.back();
        r.fd = fd;
        r.offset = (*offset)[sg.first->expert];
        r.bytes = record_bytes;
        Tq2ExpertArena::Partition *pp = part;
        cl_command_queue q = io;
        const size_t at = sg.second;
        r.consume = [q, pp, at](const uint8_t *src,
                                const ExpertRead &rd) -> int {
          return Tq2ExpertArena::slotWrite(q, *pp, at, rd.bytes, src) ==
                     CL_SUCCESS
                   ? 0
                   : EIO;
        };
      }
      if (!late.empty())
        pool.wait(pool.submit(&late), /*help=*/true);
      tq2AsyncStats().io_us += tq2UsSince(t_io);
      tq2AsyncStats().misses += fills.size();
    }
    const auto t_tab = std::chrono::steady_clock::now();
    filled(fills, true);
    acquired = false;
    // split without a miss: the slot map did not change and the misses'
    // GEMVs have no row to run
    if (!split || !fills.empty()) {
      tq2ArenaOffsets(*st, *slot_of);
      std::vector<uint32_t> ogu(2 * K), od(K);
      for (unsigned int a = 0; a < K; ++a) {
        ogu[a] = st->off_gate[ids[a]];
        ogu[K + a] = st->off_up[ids[a]];
        od[a] = st->off_down[ids[a]];
      }
      err = opencl::clEnqueueWriteBuffer(io, off_gu, CL_TRUE, 0,
                                         ogu.size() * sizeof(uint32_t),
                                         ogu.data(), 0, nullptr, nullptr);
      if (err == CL_SUCCESS)
        err = opencl::clEnqueueWriteBuffer(io, off_d, CL_TRUE, 0,
                                           od.size() * sizeof(uint32_t),
                                           od.data(), 0, nullptr, nullptr);
      NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
        << "slot offsets write failed (" << err << ")";
    }
    if (split && !fills.empty()) {
      // the device slot table of the layer's next step: the misses are
      // resident now, their victims are not. This step's select has read it
      // (the job runs after it); the next one runs after the next join.
      std::vector<uint32_t> tab;
      tab.reserve(3 * E);
      tab.insert(tab.end(), st->off_gate.begin(), st->off_gate.end());
      tab.insert(tab.end(), st->off_up.begin(), st->off_up.end());
      tab.insert(tab.end(), st->off_down.begin(), st->off_down.end());
      std::vector<uint32_t> res(E);
      for (unsigned int e = 0; e < E; ++e)
        res[e] = (*slot_of)[e] != std::numeric_limits<unsigned int>::max();
      err = opencl::clEnqueueWriteBuffer(io, st->off_tab, CL_TRUE, 0,
                                         tab.size() * sizeof(uint32_t),
                                         tab.data(), 0, nullptr, nullptr);
      if (err == CL_SUCCESS)
        err = opencl::clEnqueueWriteBuffer(io, st->res_flag, CL_TRUE, 0,
                                           res.size() * sizeof(uint32_t),
                                           res.data(), 0, nullptr, nullptr);
      NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
        << "slot table write failed (" << err << ")";
      st->tab_slots = *slot_of;
    }
    tq2AsyncStats().tab_us += tq2UsSince(t_tab);
    // recency hint: the extended ranks, least to most recent
    if (refresh)
      refresh(std::vector<unsigned int>(ranked.rbegin(), ranked.rend()));
  } catch (const std::exception &ex) {
    error = ex.what();
    status = -1;
  } catch (...) {
    error = "unknown failure";
    status = -1;
  }
  if (acquired) {
    try {
      filled(fills, false);
    } catch (...) {
    }
  }
  // what was read ahead for this step and not used is dropped, and a
  // prediction for it staged from now on is refused (before the user event:
  // nothing downstream of this step runs before its layer is closed)
  if (tq2PregateM() > 0)
    tq2StageCache().close(layer);
  if (layer == 0)
    tq2AsyncStats().tokens++;
  tq2AsyncStats().jobs++;
  tq2AsyncStats().job_us += tq2UsSince(t0);
  finish(status);
  // on the way out: the next layer's prediction (read ahead after this
  // step's misses, at low priority; nothing here waits for it)
  handPrediction();
}

void Tq2AsyncJob::finish(cl_int status) {
  if (user != nullptr) {
    // the GEMVs behind the event may run now (or fail with the error); the
    // flush submits them
    opencl::clSetUserEventStatus(user, status);
    opencl::clFlush(cq);
    opencl::clReleaseEvent(user);
    user = nullptr;
  }
  {
    std::lock_guard<std::mutex> lk(mu);
    done = true;
  }
  cv.notify_all();
}

void Tq2AsyncJob::fail(cl_int status) {
  error = "the routing step ended with status " + std::to_string(status) +
          " (the select kernel or a command before it failed)";
  std::fprintf(stderr, "[TQ2CX-FSU-ASYNC] ERROR: %s\n", error.c_str());
  tq2AsyncStats().failed++;
  if (tq2PregateM() > 0)
    tq2StageCache().close(layer);
  finish(status < 0 ? status : -1);
  handPrediction();
}

/** the select marker's callback: post the job on the reader pool. Any
 *  status but CL_COMPLETE means the routing (ext) was never written: the
 *  job must not read it back (an independent queue would return the
 *  previous step's ids), it fails the step instead. */
void CL_CALLBACK tq2AsyncSelected(cl_event ev, cl_int status, void *data) {
  auto *holder = static_cast<std::shared_ptr<Tq2AsyncJob> *>(data);
  std::shared_ptr<Tq2AsyncJob> job = std::move(*holder);
  delete holder;
  job->posted = std::chrono::steady_clock::now();
  if (status == CL_COMPLETE)
    ExpertReaderPool::global().post([job]() { job->run(); });
  else
    ExpertReaderPool::global().post([job, status]() { job->fail(status); });
  opencl::clReleaseEvent(ev);
}

/** a deferred decode-step tail: the kernels that write the output @a key */
struct Tq2Tail {
  const void *key = nullptr;
  std::function<void()> run;
};

/** pending tails, oldest first; guarded by tq2Mutex */
std::vector<Tq2Tail> &tq2Tails() {
  static std::vector<Tq2Tail> v;
  return v;
}

/** enqueue every pending tail, oldest first (caller holds tq2Mutex) */
void tq2RunTails() {
  auto &v = tq2Tails();
  if (v.empty())
    return;
  std::vector<Tq2Tail> run;
  run.swap(v);
  setDeferredOutputsPending(false);
  for (auto &t : run)
    t.run();
}

/** the key of an operand: its device buffer or its shared-memory pointer */
const void *tq2ActKey(const Tq2Act &a) {
  return a.mem != nullptr ? static_cast<const void *>(a.mem) : a.svm;
}

/** one layer of the probe, handed to its worker by the layer's marker */
struct Tq2ProbeTask {
  cl_event user = nullptr; /**< the layer's user event (worker's ref) */
  cl_event busy = nullptr; /**< marker after the layer's busy kernel (ref) */
  unsigned int stage = 0;  /**< the layer's slot in the slot buffer */
  float value = 0.0f;      /**< what the worker writes there */
};

/** the probe's shared state (see tq2AsyncProbe); leaked, with the probe's
 *  queues and buffers, when a probe thread is still inside the driver at the
 *  end (a failed probe) */
struct Tq2Probe {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Tq2ProbeTask> to_run; /**< layers whose marker fired */
  bool stop = false;
  unsigned int fired = 0;        /**< marker callbacks */
  unsigned int taken = 0;        /**< layers the worker started */
  unsigned int set = 0;          /**< layers the worker finished */
  unsigned int reads = 0;        /**< host blocking reads that returned */
  unsigned long long writes = 0; /**< worker writes, all layers */
  int io_errors = 0;
  cl_int status = CL_COMPLETE;
  std::chrono::steady_clock::time_point at;
};

/** a marker callback of the probe: hand the layer to the worker (like the
 *  decode step's callback posting its job) */
struct Tq2ProbeStage {
  Tq2Probe *p = nullptr;
  Tq2ProbeTask task; /**< user null for the round-trip marker */
};

void CL_CALLBACK tq2ProbeFired(cl_event ev, cl_int st, void *d) {
  auto *s = static_cast<Tq2ProbeStage *>(d);
  Tq2Probe *p = s->p;
  {
    std::lock_guard<std::mutex> lk(p->mu);
    ++p->fired;
    if (st != CL_COMPLETE)
      p->status = st;
    p->at = std::chrono::steady_clock::now();
    if (s->task.user != nullptr)
      p->to_run.push_back(s->task);
  }
  p->cv.notify_all();
  delete s;
  opencl::clReleaseEvent(ev);
}

/** what tq2AsyncProbe found */
struct Tq2ProbeResult {
  bool callbacks = false; /**< marker callbacks fire */
  bool events = false;    /**< the event pipeline keeps its order, no stall */
};

/**
 * @brief Do marker callbacks, user events and the decode step's pattern
 *        work on this device? On two scratch queues (compute, I/O):
 *        (1) a marker's callback round trip; (2) the decode token in its
 *        order: kReps tokens of kStages layers enqueued ahead, each layer =
 *        kernel, marker (its callback hands the layer to a worker thread, as
 *        the select's callback posts the step's job), a busy kernel that
 *        reads the buffer the worker writes (the hit GEMVs over the slot
 *        arena, and the dense MLP, the queue runs while the job reads), a
 *        marker, a barrier on the layer's user event, a kernel that reads
 *        the slot the worker writes and one that reads the written buffer
 *        (the misses' GEMVs), a shared-memory map / unmap (the activation
 *        protocol), kernel. The worker does what the job does: a
 *        blocking read on the I/O queue of a buffer the compute queue wrote
 *        (the routing readback), then blocking writes on the I/O queue (the
 *        expert slot writes) until the busy kernel is done and two more --
 *        so they span the compute queue going idle with commands held
 *        behind the user event, the moment an earlier decode deadlock
 *        needed -- then the slot, then it completes the user event and
 *        flushes the compute queue. Meanwhile the host sits in a blocking
 *        read on the compute queue (the token's logits readback; a clFinish
 *        on Adreno), and the expert writes run on two threads (the job and
 *        the reader helping it). Every kernel adds to every element of the
 *        checked buffer and the sum must be exact; every token must finish
 *        within kTokenBound. A stalled token completes its user events with
 *        an error to drain; a thread still in the driver then is left behind
 *        (and the probe's objects leaked: the process may not recover).
 *
 *        What it does not show: that every driver race is absent. On
 *        Adreno 840, writes that named a long-complete compute-queue marker
 *        in their wait list still passed the probe while the decode with
 *        that ordering hung. The probe answers "does this device run the
 *        step's ordering", not "is every other ordering safe": the step
 *        itself must keep to the ordering the probe runs. Logs one line.
 */
Tq2ProbeResult tq2AsyncProbe(ClContext *cc) {
  Tq2ProbeResult r;
  if (opencl::clCreateUserEvent == nullptr ||
      opencl::clSetUserEventStatus == nullptr ||
      opencl::clSetEventCallback == nullptr ||
      opencl::clGetEventInfo == nullptr)
    return r;
  const auto t_start = std::chrono::steady_clock::now();
  cl_int err = CL_SUCCESS;
  cl_context ctx = cc->context_inst_.GetContextNoRetain();
  cl_device_id dev = cc->context_inst_.GetDeviceId();
  cl_command_queue q = opencl::clCreateCommandQueue(ctx, dev, 0, &err);
  if (err != CL_SUCCESS || q == nullptr)
    return r;
  cl_command_queue qi = opencl::clCreateCommandQueue(ctx, dev, 0, &err);
  if (err != CL_SUCCESS || qi == nullptr) {
    opencl::clReleaseCommandQueue(q);
    return r;
  }
  auto *pr = new Tq2Probe();
  bool leak = false; // a callback or a probe thread may still come / run
  double rt_us = -1.0;
  {
    cl_event m = nullptr;
    err = opencl::clEnqueueBarrierWithWaitList(q, 0, nullptr, &m);
    const auto t0 = std::chrono::steady_clock::now();
    if (err == CL_SUCCESS)
      err = opencl::clSetEventCallback(m, CL_COMPLETE, tq2ProbeFired,
                                       new Tq2ProbeStage{pr, {}});
    opencl::clFlush(q);
    std::unique_lock<std::mutex> lk(pr->mu);
    r.callbacks = err == CL_SUCCESS &&
                  pr->cv.wait_for(lk, std::chrono::seconds(2),
                                  [pr] { return pr->fired == 1; }) &&
                  pr->status == CL_COMPLETE;
    if (r.callbacks)
      rt_us = std::chrono::duration<double, std::micro>(pr->at - t0).count();
    else
      leak = true;
  }
  constexpr unsigned int kReps = 3, kStages = 6, kMaxWrites = 64;
  constexpr int kBusyIters = 512;
  constexpr size_t kChunk = 64 * 1024, kWriteBuf = 16 * kChunk;
  constexpr auto kTokenBound = std::chrono::milliseconds(1000);
  long lost = -1;
  unsigned int stalls = 0;
  if (r.callbacks) {
    static const char *src =
      "__kernel void tq2_probe_add1(__global float *a, int n) {"
      " int i = get_global_id(0); if (i < n) a[i] += 1.0f; }"
      "__kernel void tq2_probe_add_slot(__global float *a,"
      " __global const float *s, int l, int n) {"
      " int i = get_global_id(0); if (i < n) a[i] += s[l]; }"
      "__kernel void tq2_probe_busy(__global float *c,"
      " __global const float *w, int nw, int iters) {"
      " int i = get_global_id(0); float x = c[i] + w[i % nw];"
      " for (int k = 0; k < iters; ++k) x = fma(x, 0.999999f, 1e-7f);"
      " c[i] = x; }";
    cl_program prg =
      opencl::clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    cl_kernel k = nullptr, ks = nullptr, kb = nullptr, kt = nullptr;
    if (err == CL_SUCCESS && opencl::clBuildProgram(prg, 1, &dev, "", nullptr,
                                                    nullptr) == CL_SUCCESS) {
      k = opencl::clCreateKernel(prg, "tq2_probe_add1", &err);
      ks = opencl::clCreateKernel(prg, "tq2_probe_add_slot", &err);
      kb = opencl::clCreateKernel(prg, "tq2_probe_busy", &err);
      kt = opencl::clCreateKernel(prg, "tq2_probe_busy", &err);
    }
    const int n = 1 << 14, nb = 1 << 16, iters = kBusyIters, one = 1;
    const int nwf = (int)(kWriteBuf / sizeof(float));
    cl_mem b = nullptr, sb = nullptr, cb = nullptr, wb = nullptr;
    if (k != nullptr && ks != nullptr && kb != nullptr && kt != nullptr) {
      b = opencl::clCreateBuffer(ctx, CL_MEM_READ_WRITE, n * sizeof(float),
                                 nullptr, &err);
      sb = opencl::clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                  kStages * sizeof(float), nullptr, &err);
      cb = opencl::clCreateBuffer(ctx, CL_MEM_READ_WRITE, nb * sizeof(float),
                                  nullptr, &err);
      wb =
        opencl::clCreateBuffer(ctx, CL_MEM_READ_ONLY, kWriteBuf, nullptr, &err);
    }
    void *svm = opencl::clSVMAlloc(ctx, CL_MEM_READ_WRITE, 4096, 0);
    const size_t gsz = n, gbz = nb;
    std::vector<std::thread> stuck; // probe threads left in the driver
    if (b != nullptr && sb != nullptr && cb != nullptr && wb != nullptr &&
        svm != nullptr) {
      const float zero = 0.0f;
      opencl::clEnqueueFillBuffer(q, b, &zero, sizeof(float), 0,
                                  n * sizeof(float), 0, nullptr, nullptr);
      opencl::clEnqueueFillBuffer(q, sb, &zero, sizeof(float), 0,
                                  kStages * sizeof(float), 0, nullptr, nullptr);
      opencl::clEnqueueFillBuffer(q, cb, &zero, sizeof(float), 0,
                                  nb * sizeof(float), 0, nullptr, nullptr);
      opencl::clEnqueueFillBuffer(q, wb, &zero, sizeof(float), 0, kWriteBuf, 0,
                                  nullptr, nullptr);
      opencl::clFinish(q);
      bool ok =
        opencl::clSetKernelArg(k, 0, sizeof(cl_mem), &b) == CL_SUCCESS &&
        opencl::clSetKernelArg(k, 1, sizeof(int), &n) == CL_SUCCESS &&
        opencl::clSetKernelArg(ks, 0, sizeof(cl_mem), &b) == CL_SUCCESS &&
        opencl::clSetKernelArg(ks, 1, sizeof(cl_mem), &sb) == CL_SUCCESS &&
        opencl::clSetKernelArg(ks, 3, sizeof(int), &n) == CL_SUCCESS &&
        opencl::clSetKernelArg(kb, 0, sizeof(cl_mem), &cb) == CL_SUCCESS &&
        opencl::clSetKernelArg(kb, 1, sizeof(cl_mem), &wb) == CL_SUCCESS &&
        opencl::clSetKernelArg(kb, 2, sizeof(int), &nwf) == CL_SUCCESS &&
        opencl::clSetKernelArg(kb, 3, sizeof(int), &iters) == CL_SUCCESS &&
        opencl::clSetKernelArg(kt, 0, sizeof(cl_mem), &cb) == CL_SUCCESS &&
        opencl::clSetKernelArg(kt, 1, sizeof(cl_mem), &wb) == CL_SUCCESS &&
        opencl::clSetKernelArg(kt, 2, sizeof(int), &nwf) == CL_SUCCESS &&
        opencl::clSetKernelArg(kt, 3, sizeof(int), &one) == CL_SUCCESS;
      // the worker: the job's I/O, then the user event and the flush
      std::thread worker([pr, q, qi, b, sb, wb]() {
        std::vector<float> rb(64);
        for (;;) {
          Tq2ProbeTask t;
          {
            std::unique_lock<std::mutex> lk(pr->mu);
            pr->cv.wait(lk, [pr] { return pr->stop || !pr->to_run.empty(); });
            if (pr->to_run.empty())
              return;
            t = pr->to_run.front();
            pr->to_run.pop_front();
            ++pr->taken;
          }
          cl_int e = opencl::clEnqueueReadBuffer(
            qi, b, CL_TRUE, 0, rb.size() * sizeof(float), rb.data(), 0, nullptr,
            nullptr);
          // the slot writes, on two threads as the job and the reader that
          // helps it: until the busy kernel is done, and two more each
          auto writes = [&t, qi, wb](size_t base, unsigned int &w) {
            std::vector<uint8_t> data(kChunk, 0);
            unsigned int after = 0;
            cl_int err = CL_SUCCESS;
            while (err == CL_SUCCESS && after < 2 && w < kMaxWrites) {
              cl_int bst = CL_COMPLETE;
              opencl::clGetEventInfo(t.busy, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                     sizeof(bst), &bst, nullptr);
              if (bst <= CL_COMPLETE)
                ++after;
              const size_t at = base + (w % (kWriteBuf / kChunk / 2)) * kChunk;
              err = opencl::clEnqueueWriteBuffer(
                qi, wb, CL_TRUE, at, kChunk, data.data(), 0, nullptr, nullptr);
              ++w;
            }
            return err;
          };
          unsigned int w = 0, w2 = 0;
          cl_int e2 = CL_SUCCESS;
          if (e == CL_SUCCESS) {
            std::thread helper([&]() { e2 = writes(kWriteBuf / 2, w2); });
            e = writes(0, w);
            helper.join();
            if (e == CL_SUCCESS)
              e = e2;
            w += w2;
          }
          if (e == CL_SUCCESS)
            e = opencl::clEnqueueWriteBuffer(
              qi, sb, CL_TRUE, t.stage * sizeof(float), sizeof(float), &t.value,
              0, nullptr, nullptr);
          opencl::clReleaseEvent(t.busy);
          opencl::clSetUserEventStatus(t.user,
                                       e == CL_SUCCESS ? CL_COMPLETE : -1);
          opencl::clFlush(q);
          opencl::clReleaseEvent(t.user);
          {
            std::lock_guard<std::mutex> lk(pr->mu);
            ++pr->set;
            pr->writes += w;
            if (e != CL_SUCCESS)
              ++pr->io_errors;
          }
          pr->cv.notify_all();
        }
      });
      auto add1 = [&]() {
        return opencl::clEnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, nullptr,
                                              0, nullptr,
                                              nullptr) == CL_SUCCESS;
      };
      double expect = 0.0;
      for (unsigned int rep = 0; rep < kReps && ok; ++rep) {
        std::vector<cl_event> users;
        unsigned int set0 = 0, reads0 = 0, fired0 = 0;
        {
          std::lock_guard<std::mutex> lk(pr->mu);
          set0 = pr->set;
          reads0 = pr->reads;
          fired0 = pr->fired;
        }
        for (unsigned int l = 0; l < kStages && ok; ++l) {
          cl_event u = opencl::clCreateUserEvent(ctx, &err);
          if (err != CL_SUCCESS || u == nullptr) {
            ok = false;
            break;
          }
          users.push_back(u);
          const float v = (float)(rep * kStages + l + 1);
          const int li = (int)l;
          cl_event m = nullptr, busy = nullptr;
          ok =
            add1() &&
            opencl::clEnqueueBarrierWithWaitList(q, 0, nullptr, &m) ==
              CL_SUCCESS &&
            opencl::clEnqueueNDRangeKernel(q, kb, 1, nullptr, &gbz, nullptr, 0,
                                           nullptr, nullptr) == CL_SUCCESS &&
            opencl::clEnqueueBarrierWithWaitList(q, 0, nullptr, &busy) ==
              CL_SUCCESS;
          if (ok) {
            opencl::clRetainEvent(u); // the worker's reference
            ok = opencl::clSetEventCallback(
                   m, CL_COMPLETE, tq2ProbeFired,
                   new Tq2ProbeStage{pr, {u, busy, l, v}}) == CL_SUCCESS;
            if (!ok) { // never handed to the worker
              opencl::clReleaseEvent(u);
              opencl::clReleaseEvent(busy);
            }
          }
          if (!ok)
            break;
          ok =
            opencl::clEnqueueBarrierWithWaitList(q, 1, &u, nullptr) ==
              CL_SUCCESS &&
            opencl::clSetKernelArg(ks, 2, sizeof(int), &li) == CL_SUCCESS &&
            opencl::clEnqueueNDRangeKernel(q, ks, 1, nullptr, &gsz, nullptr, 0,
                                           nullptr, nullptr) == CL_SUCCESS &&
            opencl::clEnqueueNDRangeKernel(q, kt, 1, nullptr, &gbz, nullptr, 0,
                                           nullptr, nullptr) == CL_SUCCESS &&
            opencl::clEnqueueSVMMap(q, CL_FALSE, CL_MAP_READ, svm, 4096, 0,
                                    nullptr, nullptr) == CL_SUCCESS &&
            opencl::clEnqueueSVMUnmap(q, svm, 0, nullptr, nullptr) ==
              CL_SUCCESS &&
            add1();
          expect += 2.0 + v;
        }
        opencl::clFlush(q); // once per token, as the decode step does
        // the token's readback: a blocking read on the compute queue, on a
        // thread of its own so that a deadlock is a timeout here
        std::thread reader([pr, q, b]() {
          float x = 0.0f;
          const cl_int e = opencl::clEnqueueReadBuffer(
            q, b, CL_TRUE, 0, sizeof(float), &x, 0, nullptr, nullptr);
          {
            std::lock_guard<std::mutex> lk(pr->mu);
            ++pr->reads;
            if (e != CL_SUCCESS)
              ++pr->io_errors;
          }
          pr->cv.notify_all();
        });
        const unsigned int want = (unsigned int)users.size();
        bool finished = false;
        {
          std::unique_lock<std::mutex> lk(pr->mu);
          finished = pr->cv.wait_for(lk, kTokenBound, [&] {
            return pr->set - set0 == want && pr->reads - reads0 == 1;
          });
        }
        if (!finished) {
          // a stall: complete every event of the token with an error so the
          // queues drain, give the threads a moment, else leave them
          ++stalls;
          ok = false;
          for (cl_event u : users)
            opencl::clSetUserEventStatus(u, -1);
          opencl::clFlush(q);
          std::unique_lock<std::mutex> lk(pr->mu);
          finished = pr->cv.wait_for(lk, kTokenBound, [&] {
            return pr->fired - fired0 == want && pr->to_run.empty() &&
                   pr->set == pr->taken && pr->reads - reads0 == 1;
          });
        }
        for (cl_event u : users)
          opencl::clReleaseEvent(u); // the host's reference
        if (finished) {
          reader.join();
        } else {
          stuck.push_back(std::move(reader));
          leak = true;
        }
      }
      {
        std::lock_guard<std::mutex> lk(pr->mu);
        pr->stop = true;
      }
      pr->cv.notify_all();
      if (leak) {
        stuck.push_back(std::move(worker));
      } else {
        worker.join();
        opencl::clFinish(q);
        std::vector<float> v(n, -1.0f);
        if (ok && pr->io_errors == 0 &&
            opencl::clEnqueueReadBuffer(q, b, CL_TRUE, 0, n * sizeof(float),
                                        v.data(), 0, nullptr,
                                        nullptr) == CL_SUCCESS) {
          lost = 0;
          for (int i = 0; i < n; ++i)
            lost += (long)std::lround(expect - (double)v[i]);
        }
      }
      r.events = ok && stalls == 0 && lost == 0 && pr->io_errors == 0;
    }
    for (auto &t : stuck)
      t.detach(); // inside the driver; the probe's objects stay alive
    if (!leak) {
      for (cl_mem m : {b, sb, cb, wb})
        if (m != nullptr)
          opencl::clReleaseMemObjectT(m);
      if (svm != nullptr)
        opencl::clSVMFreeT(ctx, svm);
    }
    // the scratch program and kernels are kept (the loader has no release
    // entries for them); a few hundred bytes, once per process
  }
  const unsigned long long writes = pr->writes;
  if (!leak) {
    opencl::clReleaseCommandQueue(qi);
    opencl::clReleaseCommandQueue(q);
    delete pr;
  }
  std::fprintf(stderr,
               "[TQ2CX-FSU-ASYNC] probe: callbacks %s (round trip %.1f us), "
               "event pipeline %s (%u tokens x %u layers, %llu I/O-queue "
               "writes across the compute queue going idle behind user events, "
               "host blocking read; stalls %u, lost updates %ld)%s, %.1f ms\n",
               r.callbacks ? "ok" : "FAILED", rt_us, r.events ? "ok" : "FAILED",
               kReps, kStages, writes, stalls, lost,
               leak ? " (probe threads left in the driver)" : "",
               std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t_start)
                 .count());
  return r;
}

/**
 * @brief NNTR_G4MOE_FSU_ASYNC=0|1|event|join|probe|auto (default auto): how a
 *        device-routed decode step of a streamed layer waits for its misses
 *        (Tq2AsyncMode). 0 = off (the host waits); 1 / event = event without
 *        the probe (testing a driver; selectable on Adreno); join = join
 *        without the probe; auto = join on Qualcomm Adreno when the probe's
 *        callbacks work (else off), on other non-Intel devices event when
 *        the device passes tq2AsyncProbe, join when only its callbacks work,
 *        else off; probe = run the probe and take its answer (event / join /
 *        off) on any non-Intel device, Adreno included (on Intel it still
 *        takes join; for testing the probe). Adreno: join because it was
 *        measured faster on the phone (an Adreno 840 phone, steady ms per
 *        token join vs event: P1 103.7 vs 112.4, P2-256 104.7 vs 107.4,
 *        P2-1024 120.1 vs 122.4; P4 113.1 vs 110.6, the one cell event won).
 *        Intel (NEO): join without the probe, as commands enqueued behind
 *        a user event lose their order across shared-memory map / unmap
 *        there (a standalone repro loses 10 to 14 of 4000 kernel updates on
 *        Xe3) and a decode step behind one faulted in the driver. Decided
 *        once per device, at load (tq2AsyncModeAtLoad): the probe must not
 *        land in the first decode step. It takes tens of milliseconds (an
 *        earlier probe took 22-24 s on Adreno: its host waited out a 2 s
 *        bound per token for a notify the worker never sent).
 */
Tq2AsyncMode tq2AsyncMode(ClContext *cc) {
  static std::mutex mu;
  static std::map<cl_device_id, Tq2AsyncMode> modes;
  // held across the probe: a second loader thread of the same device waits
  // for the answer instead of probing again
  std::lock_guard<std::mutex> lk(mu);
  const cl_device_id dev = cc->context_inst_.GetDeviceId();
  auto it = modes.find(dev);
  if (it != modes.end())
    return it->second;
  const Tq2AsyncMode mode = [cc]() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_ASYNC");
    const std::string v = e != nullptr ? e : "auto";
    if (v == "0")
      return Tq2AsyncMode::off;
    if (v == "1" || v == "event") {
      std::fprintf(stderr, "[TQ2CX-FSU-ASYNC] mode event (forced, no probe)\n");
      return Tq2AsyncMode::event;
    }
    if (v == "join") {
      std::fprintf(stderr, "[TQ2CX-FSU-ASYNC] mode join (forced, no probe)\n");
      return Tq2AsyncMode::join;
    }
    const bool intel = cc->caps().vendor_id == 0x8086;
    if (intel && v != "probe") {
      std::fprintf(stderr, "[TQ2CX-FSU-ASYNC] mode join (Intel: no user "
                           "events on the compute queue)\n");
      return Tq2AsyncMode::join;
    }
    const Tq2ProbeResult p = tq2AsyncProbe(cc);
    if (intel) { // probe: run it (testing it), keep the Intel rule
      std::fprintf(stderr, "[TQ2CX-FSU-ASYNC] mode join (Intel: no user "
                           "events on the compute queue; probe only run)\n");
      return Tq2AsyncMode::join;
    }
    // Adreno, auto: join (measured); the probe still decides whether the
    // callbacks join needs work at all
    constexpr uint32_t qualcomm_vendor_id = 0x5143;
    const bool adreno = cc->caps().vendor_id == qualcomm_vendor_id;
    const bool events = p.events && !(adreno && v != "probe");
    const Tq2AsyncMode m = events        ? Tq2AsyncMode::event
                           : p.callbacks ? Tq2AsyncMode::join
                                         : Tq2AsyncMode::off;
    std::fprintf(stderr, "[TQ2CX-FSU-ASYNC] mode %s%s\n",
                 m == Tq2AsyncMode::event  ? "event"
                 : m == Tq2AsyncMode::join ? "join"
                                           : "off",
                 adreno && v != "probe" && m == Tq2AsyncMode::join
                   ? " (Adreno: join measured faster; event = "
                     "NNTR_G4MOE_FSU_ASYNC=event)"
                   : "");
    return m;
  }();
  modes.emplace(dev, mode);
  return mode;
}

void tq2AsyncModeAtLoad(ClContext *cc) { (void)tq2AsyncMode(cc); }

/**
 * @brief NNTR_G4MOE_FSU_DEFER=0|1 (default 1): an async decode step leaves
 *        its expert GEMVs and combine for the output's first reader
 *        (ComputeOps::flush_deferred), so the work the graph enqueues in
 *        between (the dense MLP) runs while the misses are read.
 */
bool tq2DeferOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_DEFER");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

/**
 * @brief NNTR_G4MOE_FSU_SPLIT=0|1|auto (default auto): an async decode step
 *        runs the GEMVs of its resident (hit) experts as soon as it is routed
 *        and leaves only the misses' behind their reads (the GEMV skips a
 *        record whose activation row is -1). Bit-identical: a row is
 *        computed the same in either launch and the combine order is fixed.
 *        auto = in event mode only, where everything behind the user event
 *        waits for the reads; in join mode the host waits for the job
 *        anyway and the extra launches cost a little (Xe3: 0 to -2 % decode,
 *        same tokens).
 */
bool tq2SplitOn(Tq2AsyncMode mode) {
  static const int on = []() {
    const char *e = std::getenv("NNTR_G4MOE_FSU_SPLIT");
    if (e == nullptr || std::string(e) == "auto")
      return -1;
    return e[0] != '0' ? 1 : 0;
  }();
  return on < 0 ? mode == Tq2AsyncMode::event : on == 1;
}

/** the weighted combine into the output rows (ascending expert id) */
bool tq2Combine(ClContext *cc, bool fp16, cl_mem yd, const Tq2Act &out,
                cl_mem d_src, cl_mem d_w, cl_mem d_cnt, unsigned int H,
                unsigned int max_k, unsigned int T) {
  auto kp = cc->registerClKernel(kTq2MoeAux, fp16 ? "tq2moe_combine_f16"
                                                  : "tq2moe_combine_f32");
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: aux kernels did not build";
  const int Hi = (int)H, mk = (int)max_k, tot = (int)((size_t)T * H);
  bool ok = kp->SetKernelArguments(0, &yd, sizeof(cl_mem)) && out.bind(kp, 1) &&
            kp->SetKernelArguments(2, &d_src, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &d_w, sizeof(cl_mem)) &&
            kp->SetKernelArguments(4, &d_cnt, sizeof(cl_mem)) &&
            kp->SetKernelArguments(5, &Hi, sizeof(int)) &&
            kp->SetKernelArguments(6, &mk, sizeof(int)) &&
            kp->SetKernelArguments(7, &tot, sizeof(int));
  if (ok)
    tq2Dispatch(cc, kp, 1, {((size_t)tot + 63) / 64 * 64, 1, 1}, {64, 1, 1});
  return ok;
}

/**
 * @brief Does a call of @a T rows on this streamed layer stream whole
 *        layers (MoeExpertFsu::spec_min_tokens)? Lends the stream its
 *        partitions on first use. Caller holds tq2Mutex.
 */
bool tq2StreamOn(const MoeExpertFsu *fsu, const Tq2ExpertArena::Partition *part,
                 unsigned int E, unsigned int T) {
  if (fsu == nullptr || part == nullptr || part->slots >= E ||
      T < fsu->spec_min_tokens || fsu->layer_index == UINT32_MAX ||
      fsu->offset == nullptr || fsu->offset->size() != E)
    return false;
  return Tq2ExpertArena::get().streamLend(part->slots, part->slot_bytes, E);
}

/** start the block reads of this layer (unless read ahead) and the next */
void tq2StreamIssue(ClContext *cc, const MoeExpertFsu &fsu, const Tq2Stack &st,
                    unsigned int E) {
  Tq2ExpertArena &arena = Tq2ExpertArena::get();
  arena.streamIssue(cc, fsu.layer_index % 2, fsu, st);
  const MoeExpertFsu *nx = fsu.next;
  if (nx != nullptr && nx->fd >= 0 && nx->offset != nullptr &&
      nx->offset->size() == E && nx->record_bytes == fsu.record_bytes &&
      nx->slots == fsu.slots && nx->layer_index == fsu.layer_index + 1)
    arena.streamIssue(cc, nx->layer_index % 2, *nx, st);
}

// ---------------------------------------------------------------------------
// NNTR_G4MOE_FSU_FLAGS: the backend's side of the protocol
// (expert_fsu_flags.h, tq2moe_fsu_flags.cl)
// ---------------------------------------------------------------------------

/**
 * @brief NNTR_G4MOE_FSU_FLAGS_SPIN: polls a gate (and a publish waiting for
 *        ring room) spins at most before it gives up with the error word.
 *        Default 1,000,000: ~170 ms at a probe's 172 ns per
 *        system-scope acquire on Adreno 840 (a record read is ~0.6-0.8 ms;
 *        the driver let a 5 s spin run). Read once.
 */
cl_ulong tq2FlagsSpinCap() {
  static const cl_ulong v = []() -> cl_ulong {
    const char *e = std::getenv("NNTR_G4MOE_FSU_FLAGS_SPIN");
    const unsigned long long n =
      e != nullptr ? std::strtoull(e, nullptr, 10) : 1000000ull;
    return n >= 1000ull ? (cl_ulong)n : (cl_ulong)1000000ull;
  }();
  return v;
}

/**
 * @brief The flags kernel @a name, OpenCL C 2.0 (Adreno; the probe's form)
 *        else 3.0 (a device that has no 2.0, e.g. Xe3), with the protocol's
 *        layout; null when neither builds.
 */
ClContext::SharedPtrClKernel tq2FlagsKernel(ClContext *cc, const char *name) {
  static const std::string defs = fsuflags::clDefines();
  for (const char *std_opt : {"-cl-std=CL2.0", "-cl-std=CL3.0"}) {
    auto k =
      cc->registerClKernel(tq2moe_fsu_flags_kernel, name, std_opt + defs);
    if (k)
      return k;
  }
  return nullptr;
}

/**
 * @brief The time of one gate poll on @a cc's device, in ns: the gate
 *        kernel on scratch blocks of its own (a tag nobody stores, so its
 *        lane spins to the cap) at two caps, on a queue of its own, timed
 *        on the host; the difference cancels the launch and the finish.
 *        Fine-grain SVM with atomics when the device has it, else
 *        coarse-grain (the blocks are zeroed through map / unmap between
 *        runs, never while a kernel runs). 0 when it cannot run.
 * @details A system-scope acquire per poll, as in the decode's gates (a
 * probe: ~172 ns on Adreno 840); the gates' spin account multiplies
 * its polls by this. ~10 ms, once, at the decision (load).
 */
double tq2FlagsCalibratePoll(ClContext *cc) {
  auto gk = tq2FlagsKernel(cc, "tq2moe_fsu_gate");
  if (!gk || opencl::clSVMAlloc == nullptr ||
      opencl::clSetKernelArgSVMPointer == nullptr ||
      opencl::clCreateCommandQueue == nullptr ||
      opencl::clEnqueueSVMMap == nullptr ||
      opencl::clEnqueueSVMUnmap == nullptr)
    return 0.0;
  const cl_context ctx = cc->context_inst_.GetContextNoRetain();
  const cl_device_id dev = cc->context_inst_.GetDeviceId();
  cl_device_svm_capabilities caps = 0;
  if (opencl::clGetDeviceInfo == nullptr ||
      opencl::clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(caps),
                              &caps, nullptr) != CL_SUCCESS)
    return 0.0;
  const cl_device_svm_capabilities fine =
    CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS;
  cl_svm_mem_flags flags = CL_MEM_READ_WRITE;
  if ((caps & fine) == fine)
    flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS;
  else if ((caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) == 0)
    return 0.0;
  cl_int err = CL_SUCCESS;
  cl_command_queue q = opencl::clCreateCommandQueue(ctx, dev, 0, &err);
  if (q == nullptr || err != CL_SUCCESS)
    return 0.0;
  const size_t bytes = fsuflags::kCtlRing * sizeof(uint32_t);
  auto *ctl =
    static_cast<uint32_t *>(opencl::clSVMAllocT(ctx, flags, bytes, 0));
  auto *lay =
    static_cast<uint32_t *>(opencl::clSVMAllocT(ctx, flags, bytes, 0));
  uint32_t wait_v[2] = {0u, 1u}; // slot 0, tag 1: never stored
  float w_v = 1.0f;
  cl_mem wait = opencl::clCreateBufferT(
    ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(wait_v), wait_v, &err);
  cl_mem w = opencl::clCreateBufferT(
    ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(w_v), &w_v, &err);
  const cl_kernel k = gk->GetKernel();
  cl_uint layer = 0, C = 1, K = 1;
  // one gate of cap polls (its lane spins cap + 1): host ms, or < 0
  auto run = [&](cl_ulong cap) -> double {
    for (uint32_t *blk : {ctl, lay}) {
      if (opencl::clEnqueueSVMMap(q, CL_TRUE, CL_MAP_WRITE, blk, bytes, 0,
                                  nullptr, nullptr) != CL_SUCCESS)
        return -1.0;
      std::memset(blk, 0, bytes);
      opencl::clEnqueueSVMUnmap(q, blk, 0, nullptr, nullptr);
    }
    opencl::clFinish(q);
    if (opencl::clSetKernelArgSVMPointer(k, 0, ctl) != CL_SUCCESS ||
        opencl::clSetKernelArgSVMPointer(k, 1, lay) != CL_SUCCESS ||
        !gk->SetKernelArguments(2, &wait, sizeof(cl_mem)) ||
        !gk->SetKernelArguments(3, &w, sizeof(cl_mem)) ||
        !gk->SetKernelArguments(4, &layer, sizeof(cl_uint)) ||
        !gk->SetKernelArguments(5, &C, sizeof(cl_uint)) ||
        !gk->SetKernelArguments(6, &K, sizeof(cl_uint)) ||
        !gk->SetKernelArguments(7, &cap, sizeof(cl_ulong)))
      return -1.0;
    const size_t lws = fsuflags::kGateLanes;
    const auto t0 = std::chrono::steady_clock::now();
    if (opencl::clEnqueueNDRangeKernel(q, k, 1, nullptr, &lws, &lws, 0, nullptr,
                                       nullptr) != CL_SUCCESS)
      return -1.0;
    opencl::clFinish(q);
    return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
  };
  double ns = 0.0;
  if (ctl != nullptr && lay != nullptr && wait != nullptr && w != nullptr) {
    constexpr cl_ulong lo = 4000, hi = 44000;
    const double warm = run(1000), a = run(lo), b = run(hi);
    if (warm >= 0.0 && a >= 0.0 && b > a)
      ns = (b - a) * 1e6 / (double)(hi - lo);
  }
  if (wait != nullptr)
    opencl::clReleaseMemObjectT(wait);
  if (w != nullptr)
    opencl::clReleaseMemObjectT(w);
  if (ctl != nullptr)
    opencl::clSVMFreeT(ctx, ctl);
  if (lay != nullptr)
    opencl::clSVMFreeT(ctx, lay);
  if (opencl::clReleaseCommandQueue != nullptr)
    opencl::clReleaseCommandQueue(q);
  return ns;
}

/**
 * @brief How the flags protocol's poller waits for the device's entries
 *        (FsuFlagsHost::PollConfig), read once:
 *        NNTR_G4MOE_FSU_FLAGS_POLL=adaptive|spin (default adaptive). spin:
 *        it spins NNTR_G4MOE_FSU_FLAGS_POLL_US (default 2000) after the last
 *        entry, then polls every 20 us -- during a decode it never stops
 *        spinning (a core at 100 %). adaptive (two Adreno 840 phones: faster
 *        on both, 0.23 instead of 0.6 of a core): it spins
 *        NNTR_G4MOE_FSU_FLAGS_POLL_US (default 100) after an entry (the
 *        read ahead follows the step's entry closely), then sleeps 1 us
 *        doubling up to NNTR_G4MOE_FSU_FLAGS_POLL_MAX_US (default 50; timer
 *        slack 1 ns), and spins again from NNTR_G4MOE_FSU_FLAGS_POLL_LEAD
 *        percent (default 80) of the usual step-to-step interval after
 *        the last step entry, through its arrival.
 */
fsuflags::FsuFlagsHost::PollConfig tq2FlagsPoll() {
  fsuflags::FsuFlagsHost::PollConfig c;
  const char *e = std::getenv("NNTR_G4MOE_FSU_FLAGS_POLL");
  c.adaptive = e == nullptr || std::string(e) != "spin";
  c.spin_us =
    tq2EnvUint("NNTR_G4MOE_FSU_FLAGS_POLL_US", c.adaptive ? 100u : 2000u);
  c.max_sleep_us =
    std::max(1u, tq2EnvUint("NNTR_G4MOE_FSU_FLAGS_POLL_MAX_US", 50u));
  c.lead = tq2EnvUint("NNTR_G4MOE_FSU_FLAGS_POLL_LEAD", 80u) / 100.0f;
  return c;
}

/**
 * @brief The process's flags protocol state: the control block, a block per
 *        streamed layer, the host side (FsuFlagsHost: prepare, poller,
 *        replay) and the reads it starts (straight into the slots, O_DIRECT,
 *        on the reader pool). Never destroyed (its SVM belongs to the OpenCL
 *        context, which may be gone at static destruction); the poller is a
 *        thread of its own.
 */
class Tq2Flags {
public:
  /** a registered layer: its block and its reads' source / destination */
  struct Layer {
    unsigned int index = 0, C = 0, E = 0, K = 0;
    uint32_t *lay = nullptr; /**< fine-grain SVM, LayerLayout(C, E) */
    cl_mem shift = nullptr;  /**< [E] record shift in its slot, words */
    cl_mem wait = nullptr;   /**< [2 kMaxItems] the gate's wait list */
    int fd = -1;
    const std::vector<uint64_t> *offset = nullptr;
    size_t record_bytes = 0;
    uint8_t *base = nullptr; /**< the partition's SVM */
    size_t stride = 0;       /**< bytes between slots */
    cl_uint stride_w = 0, gu_w = 0;
    Tq2ExpertArena::Partition *part = nullptr;
    cl_command_queue io = nullptr; /**< coarse test mode: slot writes */
  };

  /** the state, created on first use (host thread, caller holds tq2Mutex) */
  static Tq2Flags &get(ClContext *cc) {
    static Tq2Flags *f = new Tq2Flags(cc);
    return *f;
  }
  /** the state if it exists */
  static Tq2Flags *existing() { return instance_.load(); }

  fsuflags::FsuFlagsHost &host() { return *host_; }
  uint32_t *ctl() { return ctl_; }

  /** the expert record bytes of the registered layers (the first's) */
  size_t recordBytes() {
    std::lock_guard<std::mutex> lk(mu_);
    return layers_.empty() ? 0 : layers_.begin()->second->record_bytes;
  }

  /**
   * @brief Layer @a d.layer_index, registered on first use: its block (the
   *        device table, empty), its shift table and wait list, its hooks.
   *        @a part is its partition, @a gu_bytes the gate / up record bytes
   *        of a slot, @a K the experts a step selects.
   */
  Layer &layer(ClContext *cc, const MoeExpertFsu &d,
               Tq2ExpertArena::Partition &part, size_t gu_bytes, unsigned int E,
               unsigned int K) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = layers_.find(d.layer_index);
      if (it != layers_.end())
        return *it->second;
    }
    NNTR_THROW_IF(!d.victim_order || !d.place || !d.admit || !d.forget ||
                    !d.refresh || d.slot_of == nullptr || d.offset == nullptr ||
                    d.offset->size() != E || part.host == nullptr ||
                    d.layer_index == UINT32_MAX,
                  std::runtime_error)
      << "NNTR_G4MOE_FSU_FLAGS: streamed layer " << d.layer_index
      << " has no flags hooks / SVM slots";
    auto l = std::make_unique<Layer>();
    l->index = d.layer_index;
    l->C = part.slots;
    l->E = E;
    l->K = K;
    l->fd = d.fd;
    l->offset = d.offset;
    l->record_bytes = d.record_bytes;
    l->base = part.host;
    l->stride = part.slot_bytes;
    l->part = &part;
    l->io = Tq2ExpertArena::get().ioQueue(cc);
    l->stride_w = (cl_uint)(part.slot_bytes / sizeof(uint32_t));
    l->gu_w = (cl_uint)(gu_bytes / sizeof(uint32_t));
    const fsuflags::LayerLayout L(l->C, E);
    l->lay = static_cast<uint32_t *>(svmAlloc(cc, L.words * sizeof(uint32_t)));
    std::vector<uint32_t> sh(E);
    for (unsigned int e = 0; e < E; ++e)
      sh[e] = (uint32_t)(tq2SlotShift((*d.offset)[e]) / sizeof(uint32_t));
    {
      opencl::ClMemAcctScope acct("tq2cx_router");
      l->shift = tq2Upload(cc, sh.data(), E * sizeof(uint32_t));
      l->wait = tq2Upload(
        cc, std::vector<uint32_t>(2 * fsuflags::kMaxItems, 0).data(),
        2 * fsuflags::kMaxItems * sizeof(uint32_t), CL_MEM_READ_WRITE);
    }
    fsuflags::FsuFlagsHost::Hooks h;
    h.victim_order = d.victim_order;
    h.place = d.place;
    h.admit = d.admit;
    h.forget = d.forget;
    h.refresh = d.refresh;
    h.tally = d.tally;
    const std::vector<unsigned int> *map = d.slot_of;
    h.slot_map = [map]() { return *map; };
    Layer &ref = *l;
    {
      std::lock_guard<std::mutex> lk(mu_);
      layers_.emplace(d.layer_index, std::move(l));
    }
    host_->addLayer(ref.index, ref.lay, ref.C, E, K, std::move(h));
    return ref;
  }

  /**
   * @brief The coarse-grain test mode's hand-over of the control and layer
   *        blocks: coarseMap (a blocking map of each on the compute queue:
   *        the device's stores there reach the host, before the host writes
   *        the table) and coarseUnmap (the host's stores reach the device,
   *        before a kernel that reads them). No-ops otherwise.
   */
  void coarseMap(ClContext *cc, const Layer &l) { coarse(cc, l, true); }
  void coarseUnmap(ClContext *cc, const Layer &l) { coarse(cc, l, false); }

  /** the registered layer @a index (any thread), or null */
  Layer *find(unsigned int index) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = layers_.find(index);
    return it == layers_.end() ? nullptr : it->second.get();
  }

  /**
   * @brief Before anything else writes the slots or changes a policy (a
   *        prefill): hand the queued work to the device, wait for every
   *        publish, read and replay (FsuFlagsHost::drain), then for the
   *        compute queue (the decode GEMVs that read the slots).
   */
  void drain(ClContext *cc) {
    if (host_->lastSeq() == 0)
      return; // no decode step yet
    cl_command_queue cq = cc->command_queue_inst_.GetCommandQueue();
    opencl::clFlush(cq);
    if (tq2FsuFlagsCoarse()) {
      // the test mode's host-thread consumption (see the decode tail)
      opencl::clFinish(cq);
      const size_t bytes = fsuflags::kCtlWords * sizeof(uint32_t);
      NNTR_THROW_IF(
        opencl::clEnqueueSVMMap(cq, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, ctl_,
                                bytes, 0, nullptr, nullptr) != CL_SUCCESS,
        std::runtime_error)
        << "NNTR_G4MOE_FSU_FLAGS=coarse: map failed";
      while (!host_->consumed(host_->lastSeq()))
        NNTR_THROW_IF(!host_->pollOnce(), std::runtime_error)
          << "NNTR_G4MOE_FSU_FLAGS=coarse: a publish is missing from the "
             "ring after the queue finished";
      opencl::clEnqueueSVMUnmap(cq, ctl_, 0, nullptr, nullptr);
    }
    host_->drain();
    opencl::clFinish(cq);
  }

private:
  explicit Tq2Flags(ClContext *cc) {
    ctl_ = static_cast<uint32_t *>(
      svmAlloc(cc, fsuflags::kCtlWords * sizeof(uint32_t)));
    host_ = std::make_unique<fsuflags::FsuFlagsHost>(
      ctl_, [this](const fsuflags::FsuFlagsHost::Read &r,
                   std::shared_ptr<std::atomic<bool>> once,
                   bool low) { issue(r, std::move(once), low); });
    // the coarse-grain test mode consumes the ring on the host thread, after
    // each step's kernels completed (see the decode tail)
    if (!tq2FsuFlagsCoarse())
      host_->startPoller(tq2FlagsPoll());
    instance_.store(this);
    std::fprintf(stderr,
                 "[TQ2CX-FSU-FLAGS] decode without the host round trip: "
                 "ring %u x %u words, gate spin cap %llu polls, readers %u, "
                 "poll %s (spin %u us, sleeps up to %u us, lead %.2f)\n",
                 fsuflags::kRingEntries, fsuflags::kEntryWords,
                 (unsigned long long)tq2FlagsSpinCap(),
                 ExpertReaderPool::global().readers(),
                 tq2FsuFlagsCoarse()       ? "host (coarse)"
                 : tq2FlagsPoll().adaptive ? "adaptive"
                                           : "spin",
                 tq2FlagsPoll().spin_us, tq2FlagsPoll().max_sleep_us,
                 (double)tq2FlagsPoll().lead);
  }

  void coarse(ClContext *cc, const Layer &l, bool map) {
    if (!tq2FsuFlagsCoarse())
      return;
    cl_command_queue cq = cc->command_queue_inst_.GetCommandQueue();
    const size_t lay_bytes =
      fsuflags::LayerLayout(l.C, l.E).words * sizeof(uint32_t);
    for (auto blk :
         {std::make_pair((void *)ctl_, fsuflags::kCtlWords * sizeof(uint32_t)),
          std::make_pair((void *)l.lay, lay_bytes)}) {
      const cl_int e =
        map
          ? opencl::clEnqueueSVMMap(cq, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                    blk.first, blk.second, 0, nullptr, nullptr)
          : opencl::clEnqueueSVMUnmap(cq, blk.first, 0, nullptr, nullptr);
      NNTR_THROW_IF(e != CL_SUCCESS, std::runtime_error)
        << "NNTR_G4MOE_FSU_FLAGS=coarse: map / unmap failed (" << e << ")";
    }
  }

  /** zeroed fine-grain SVM with atomics, page aligned */
  static void *svmAlloc(ClContext *cc, size_t bytes) {
    const size_t sz =
      (bytes + kTq2FlagsPage - 1) / kTq2FlagsPage * kTq2FlagsPage;
    void *p = opencl::clSVMAllocT(
      cc->context_inst_.GetContextNoRetain(),
      tq2FsuFlagsCoarse()
        ? CL_MEM_READ_WRITE
        : CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
      sz, 0);
    NNTR_THROW_IF(p == nullptr, std::runtime_error)
      << "NNTR_G4MOE_FSU_FLAGS: fine-grain SVM of " << sz << " B refused";
    std::memset(p, 0, sz);
    return p;
  }

  /** a read the poller starts: the record's O_DIRECT window straight into
   *  its slot (the record lands at slot + offset % 4096; one not at the
   *  GEMVs' alignment is moved down to slot + its shift), on the reader
   *  pool, detached; its end reported to the host */
  void issue(const fsuflags::FsuFlagsHost::Read &r,
             std::shared_ptr<std::atomic<bool>> once, bool low) {
    Layer *l = find(r.layer);
    fsuflags::FsuFlagsHost *h = host_.get();
    if (l == nullptr || r.slot >= l->C || r.expert >= l->E) {
      h->readDone(r, EINVAL);
      return;
    }
    auto batch = std::make_shared<std::vector<ExpertRead>>(1);
    ExpertRead &x = (*batch)[0];
    x.fd = l->fd;
    x.offset = (*l->offset)[r.expert];
    x.bytes = l->record_bytes;
    x.dst = l->base + (size_t)r.slot * l->stride;
    x.once = once.get();
    const size_t move = tq2SlotMove(x.offset);
    if (tq2FsuFlagsCoarse()) {
      // test mode (Intel NEO coarse-grain SVM): a read straight into the
      // slot stalls while a kernel runs (Xe3: 90-170 ms against a spinning
      // gate), and a CPU store there is not seen by the device without a
      // map / unmap: the reader stages the record and writes it with the
      // svm arena's coarse-grain slot write (map, copy, unmap)
      const size_t at = (size_t)r.slot * l->stride + tq2SlotShift(x.offset);
      Tq2ExpertArena::Partition *pp = l->part;
      cl_command_queue io = l->io;
      x.dst = nullptr;
      x.consume = [pp, io, at](const uint8_t *rec,
                               const ExpertRead &rd) -> int {
        return Tq2ExpertArena::slotWrite(io, *pp, at, rd.bytes, rec) ==
                   CL_SUCCESS
                 ? 0
                 : EIO;
      };
    } else if (move != 0) {
      x.consume = [move](const uint8_t *rec, const ExpertRead &rd) -> int {
        const auto t0 = std::chrono::steady_clock::now();
        uint8_t *p = const_cast<uint8_t *>(rec);
        std::memmove(p - move, p, rd.bytes);
        moves_++;
        move_us_ += tq2UsSince(t0);
        return 0;
      };
    }
    x.finished = [h, r, once](ExpertRead &rd) { h->readDone(r, rd.error); };
    ExpertReaderPool::global().submitDetached(std::move(batch), low);
  }

public:
  /** reads whose record was moved down to its alignment, and their time */
  static std::atomic<unsigned long long> moves_, move_us_;

private:
  static std::atomic<Tq2Flags *> instance_;
  uint32_t *ctl_ = nullptr;
  std::unique_ptr<fsuflags::FsuFlagsHost> host_;
  std::mutex mu_;
  std::unordered_map<unsigned int, std::unique_ptr<Layer>> layers_;
};

std::atomic<Tq2Flags *> Tq2Flags::instance_{nullptr};
std::atomic<unsigned long long> Tq2Flags::moves_{0};
std::atomic<unsigned long long> Tq2Flags::move_us_{0};

/** the flags protocol's totals, printed at exit when it ran */
struct Tq2FlagsStats {
  ~Tq2FlagsStats() {
    Tq2Flags *f = Tq2Flags::existing();
    if (f == nullptr)
      return;
    // the steps the policies never replayed (the last token's) into the
    // model's totals ([G4MOE-FSU]), which print after this
    const uint64_t tallied = f->host().tallyUnreplayed();
    const auto s = f->host().stats();
    if (s.steps == 0)
      return;
    // tokens: steps of the first streamed layer
    const double tok = (double)std::max<uint64_t>(1, tokens.load());
    std::vector<uint32_t> rd = s.miss_read_us;
    std::sort(rd.begin(), rd.end());
    auto pct = [&rd](double p) {
      return rd.empty()
               ? 0.0
               : rd[std::min(rd.size() - 1, (size_t)(p * rd.size()))] / 1000.0;
    };
    double rsum = 0.0;
    for (uint32_t v : rd)
      rsum += v;
    const uint32_t err = f->host().errorCode();
    // every decode token's steps (the replay into the policies, and with it
    // the model's [G4MOE-FSU-STATS] line, misses the last token's)
    const double hit_rate =
      s.step_items ? 1.0 - (double)s.misses / (double)s.step_items : 0.0;
    const double miss_mib =
      (double)s.misses * (double)f->recordBytes() / 1048576.0 / tok;
    // the gates' spin account (device): the slowest lane's polls per gate
    const fsuflags::SpinAccount sp = fsuflags::spinAccount(f->ctl());
    const double poll_ns = g_tq2_flags_poll_ns.load();
    // the poller's waits: last empty poll -> the poll that took the entry
    std::vector<uint32_t> gp = s.poll_gap_us;
    std::sort(gp.begin(), gp.end());
    auto gpct = [&gp](double p) -> unsigned int {
      return gp.empty() ? 0u
                        : gp[std::min(gp.size() - 1, (size_t)(p * gp.size()))];
    };
    std::fprintf(
      stderr,
      "[TQ2CX-FSU-FLAGS] tokens=%.0f steps=%llu hit_rate=%.4f "
      "misses_per_token=%.2f miss_mib_per_token=%.2f "
      "ready_at_publish_per_token=%.2f specs=%llu spec_placed_per_token=%.2f "
      "spec_hits_per_token=%.2f spec_promoted=%llu spec_dropped_per_token=%.2f "
      "reads=%llu failed=%llu miss_read_ms_avg=%.3f p50=%.3f p90=%.3f "
      "p99=%.3f moves=%llu move_ms_avg=%.3f prepares=%llu "
      "prepare_ms_avg=%.4f prepare_wait_ms_avg=%.4f "
      "room_waits=%llu adopts=%llu drains=%llu gates=%llu "
      "gates_waited_per_token=%.2f gate_spin_polls_per_token=%.0f "
      "gate_spin_max_polls=%u gate_poll_ns=%.1f gate_spin_ms_per_token=%.3f "
      "gate_spin_ms_max=%.3f poll=%s poll_gap_us_p50=%u p90=%u p99=%u "
      "poll_waits=%zu poll_sleeps=%llu poller_cpu_ms_per_token=%.3f "
      "poller_cpu_share=%.3f unreplayed_steps_tallied=%llu error=%s\n",
      tok, (unsigned long long)s.steps, hit_rate, s.misses / tok, miss_mib,
      s.ready_at_publish / tok, (unsigned long long)s.specs,
      s.spec_placed / tok, s.spec_hits / tok,
      (unsigned long long)s.spec_promoted, s.spec_cancelled / tok,
      (unsigned long long)s.reads_done, (unsigned long long)s.reads_failed,
      rd.empty() ? 0.0 : rsum / rd.size() / 1000.0, pct(0.5), pct(0.9),
      pct(0.99), Tq2Flags::moves_.load(),
      Tq2Flags::moves_.load()
        ? Tq2Flags::move_us_.load() / 1000.0 / Tq2Flags::moves_.load()
        : 0.0,
      (unsigned long long)s.prepares,
      s.prepares ? s.prepare_us / 1000.0 / s.prepares : 0.0,
      s.prepares ? s.prepare_wait_us / 1000.0 / s.prepares : 0.0,
      (unsigned long long)s.room_waits, (unsigned long long)s.adopts,
      (unsigned long long)s.drains, (unsigned long long)sp.gates,
      sp.waited / tok, sp.polls / tok, sp.max_polls, poll_ns,
      sp.polls * poll_ns / 1e6 / tok, sp.max_polls * poll_ns / 1e6,
      tq2FlagsPoll().adaptive ? "adaptive" : "spin", gpct(0.5), gpct(0.9),
      gpct(0.99), gp.size(), (unsigned long long)s.poll_sleeps,
      s.poller_cpu_us / 1000.0 / tok,
      s.poller_wall_us ? (double)s.poller_cpu_us / s.poller_wall_us : 0.0,
      (unsigned long long)tallied, fsuflags::errorName(err));
  }
  std::atomic<unsigned long long> tokens{0};
};
Tq2FlagsStats &tq2FlagsStats() {
  static Tq2FlagsStats s;
  return s;
}

} // namespace

bool tq2cx_moe_experts_prefetch_cl(const MoeExpertGroup &g) {
  const MoeExpertFsu *fsu = g.fsu;
  if (fsu == nullptr || g.layer_key == nullptr || g.tokens == 0 ||
      fsu->slot_of == nullptr || fsu->slot_of->size() != g.gate.size())
    return false;
  ClContext *cc = tq2GroupContext(g);
  if (cc == nullptr)
    return false;
  std::lock_guard<std::mutex> lk(tq2Mutex());
  Tq2Stack *st = tq2Stack(cc, g);
  if (st == nullptr || !st->arena)
    return false;
  tq2RunTails();
  // the stream writes lent slots: no read of the flags protocol may run
  if (Tq2Flags *fl = Tq2Flags::existing())
    fl->drain(cc);
  if (st->async) {
    auto job = std::move(st->async);
    job->join();
  }
  tq2PrefillAheadJoin();
  Tq2ExpertArena::Partition &part =
    tq2Partition(cc, *st, g.layer_key, fsu->slots);
  const unsigned int E = (unsigned int)g.gate.size();
  if (!tq2StreamOn(fsu, &part, E, g.tokens))
    return false;
  tq2StreamIssue(cc, *fsu, *st, E);
  return true;
}

void tq2cx_moe_experts_join_cl(const MoeExpertGroup &g) {
  std::lock_guard<std::mutex> lk(tq2Mutex());
  tq2RunTails();
  tq2PrefillAheadJoin();
  if (Tq2Flags *fl = Tq2Flags::existing()) {
    // NNTR_G4MOE_FSU_FLAGS: a failure of an earlier step surfaces here; a
    // call of many rows (a prefill: it changes the policies and writes the
    // slots itself) first lets every decode step finish and replays them
    fl->host().throwIfFailed();
    if (g.tokens > 1) {
      auto *cc =
        static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
      if (cc != nullptr)
        fl->drain(cc);
    }
  }
  auto it = tq2Stacks().find(g.layer_key);
  if (it == tq2Stacks().end() || !it->second->async)
    return;
  auto job = std::move(it->second->async);
  job->join();
}

double tq2cx_fsu_gate_poll_ns_cl() {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  return cc != nullptr ? tq2FlagsCalibratePoll(cc) : 0.0;
}

void tq2cx_moe_experts_flush_deferred_cl(const Tensor &t) {
  if (!deferredOutputsPending())
    return;
  Tq2Act a;
  if (!tq2Act(t, 0, a))
    return;
  std::lock_guard<std::mutex> lk(tq2Mutex());
  const void *key = tq2ActKey(a);
  for (const auto &tl : tq2Tails()) {
    if (tl.key == key) {
      tq2RunTails();
      return;
    }
  }
}

bool tq2cx_moe_experts_cl(const MoeExpertGroup &g) {
  using DT = ml::train::TensorDim::DataType;
  const MoeRouter *rt = g.router;
  const bool device_route = rt != nullptr;
  if (g.input == nullptr || g.output == nullptr || g.gate.empty() ||
      g.tokens == 0)
    return false;
  if (!device_route && (g.assignments == nullptr || g.active.empty()))
    return false;
  if (device_route && g.tokens != 1)
    return false; // the device router routes one decode row
  // experts read from the file: the device router looks the offsets of a
  // complete slot map up on the device; otherwise the host reads the
  // selected ids back and the acquire hook makes them resident first (so a
  // device-routed call needs the hook). A caller without an acquire hook
  // must have made every active expert resident.
  const MoeExpertFsu *fsu = g.fsu;
  if (fsu != nullptr &&
      (fsu->slot_of == nullptr || fsu->slot_of->size() != g.gate.size() ||
       (device_route && !fsu->complete && !fsu->acquire) ||
       (!fsu->acquire && g.active.size() > fsu->slots)))
    return false;
  ClContext *cc = tq2GroupContext(g);
  if (cc == nullptr)
    return false;
  const DT act_dt = g.input->getDataType();
  if ((act_dt != DT::FP32 && act_dt != DT::FP16) ||
      g.output->getDataType() != act_dt)
    return false;
  const bool fp16 = act_dt == DT::FP16;
  const size_t act_bytes = fp16 ? sizeof(uint16_t) : sizeof(float);
  const unsigned int H = g.gate[0]->height(), I = g.gate[0]->width();
  if (g.input->width() != H)
    return false;
  const unsigned int T = g.tokens;
  const size_t in_rows =
    (size_t)g.input->batch() * g.input->channel() * g.input->height();
  Tq2Act in, out, rin;
  if (!tq2Act(*g.input, in_rows * H * act_bytes, in) ||
      !tq2Act(*g.output, (size_t)T * H * act_bytes, out))
    return false;

  // device routing: every operand must be bindable before anything is
  // enqueued, so a decline leaves the queue untouched
  float *r_w = nullptr, *r_scale = nullptr, *r_norm = nullptr, *r_pes = nullptr;
  const unsigned int E = (unsigned int)g.gate.size();
  if (device_route) {
    const unsigned int K = rt->topk;
    if (rt->input == nullptr || K == 0 || K > kTq2RouteMaxK || K > E ||
        E > kTq2RouteMaxE || H > kTq2RouteMaxH || H % 8 != 0 ||
        rt->input->width() != H || rt->input->getDataType() != act_dt)
      return false;
    r_w = tq2HostFloat(rt->weight);
    r_scale = tq2HostFloat(rt->scale);
    r_pes = tq2HostFloat(rt->per_expert_scale);
    r_norm = tq2HostFloat(rt->norm);
    if (r_w == nullptr || r_scale == nullptr || r_pes == nullptr ||
        (rt->norm != nullptr && r_norm == nullptr) ||
        rt->weight->getDim().getDataLen() != (size_t)H * E)
      return false;
    if (!tq2Act(*rt->input, (size_t)H * act_bytes, rin))
      return false;
  }

  std::lock_guard<std::mutex> lk(tq2Mutex());
  Tq2Stack *stp = tq2Stack(cc, g);
  if (stp == nullptr)
    return false;
  // a tail nobody flushed (its output had no reader through the graph):
  // enqueue it ahead of this call
  tq2RunTails();
  // the previous layer's read-ahead into this layer's slots (a short
  // prefill): written and admitted before this call's acquire
  tq2PrefillAheadJoin();
  Tq2Stack &st = *stp;
  if (st.async) {
    // the layer's previous decode step may still be making experts resident
    auto job = std::move(st.async);
    job->join();
  }
  // a call routed on the host (a prefill) makes experts resident itself:
  // a read of the flags protocol must not write a slot meanwhile
  if (fsu != nullptr && !device_route) {
    if (Tq2Flags *fl = Tq2Flags::existing())
      fl->drain(cc);
  }
  Tq2Workspace &ws = tq2Workspace();
  // ---- experts read from the file into the layer's slots ----
  Tq2ExpertArena &arena = Tq2ExpertArena::get();
  Tq2ExpertArena::Partition *part =
    fsu != nullptr ? &tq2Partition(cc, st, g.layer_key, fsu->slots) : nullptr;
  // a partition with fewer slots than experts evicts: fence its slots
  const bool evicting = part != nullptr && part->slots < g.gate.size();
  // make @a experts resident (read the misses), refresh the offsets, and
  // keep the next kernels behind their writes; returns their slots
  // The policy holds the slots of the misses from its acquire until
  // fsu->filled: the reads are in flight in between.
  auto fsuAcquire = [&](const std::vector<unsigned int> &experts) {
    std::vector<std::pair<unsigned int, unsigned int>> fills;
    if (fsu->acquire)
      fsu->acquire(experts, fills);
    else if (fsu->fill != nullptr)
      fills = *fsu->fill;
    std::vector<unsigned int> slots;
    try {
      arena.fill(cc, *part, st, *fsu, fills, arena.runBounces(), false);
      tq2ArenaOffsets(st, *fsu->slot_of);
      slots.reserve(experts.size());
      for (unsigned int e : experts) {
        const unsigned int s = (*fsu->slot_of)[e];
        NNTR_THROW_IF(s >= part->slots, std::runtime_error)
          << "tq2cx_moe_experts_cl: expert " << e
          << " is not resident after its acquire";
        slots.push_back(s);
      }
      arena.waitReady(cc, *part, slots);
    } catch (...) {
      if (fsu->acquire && fsu->filled)
        fsu->filled(fills, false);
      throw;
    }
    if (fsu->acquire && fsu->filled)
      fsu->filled(fills, true);
    return slots;
  };
  auto &q = cc->command_queue_inst_;

  // shared-plane operands are handed to the device for the call
  if (in.svm)
    q.enqueueSVMUnmap(in.svm);
  if (out.svm)
    q.enqueueSVMUnmap(out.svm);
  if (rin.svm && rin.svm != in.svm)
    q.enqueueSVMUnmap(rin.svm);

  cl_mem yd = nullptr;
  bool ahead = false; // a short prefill that may read the next layer ahead
  // combine tables: device buffers (device routing) or per-call uploads
  cl_mem d_src = nullptr, d_w = nullptr, d_cnt = nullptr;
  bool own_tables = false;
  unsigned int max_k = 0;

  if (device_route) {
    // ---- decode, routed on the device ----
    const unsigned int K = rt->topk;
    // the input row's quantisation from the norm that wrote it, asked before
    // this call enqueues anything (the write log must reach back to the norm)
    const Tq2QRows in_q = tq2ClaimNormRow(cc, ws, in, fp16, H);
    if (part != nullptr)
      arena.streamEnd();
    // streamed experts not all resident: the select kernel also reports the
    // next ranks; the host waits for it, makes the selected experts resident
    // and writes their slots' offsets in place of the table's
    const bool fsu_route = part != nullptr && !fsu->complete;
    // NNTR_G4MOE_FSU_FLAGS: the device places the misses and the readers
    // hand them over through fine-grain SVM; the host never waits (below)
    const bool flags = fsu_route && tq2FsuFlagsOn() && part->host != nullptr;
    const Tq2AsyncMode amode =
      fsu_route && !flags ? tq2AsyncMode(cc) : Tq2AsyncMode::off;
    // the hit / miss split: the GEMVs over the resident experts run now, only
    // the misses' wait for their reads
    const bool split = amode != Tq2AsyncMode::off && tq2SplitOn(amode);
    if (part != nullptr && fsu->complete) {
      // a complete partition: its slot map is fixed, look it up on the
      // device like a resident stack's; the writes still pending first
      if (!st.tab_complete) {
        tq2ArenaOffsets(st, *fsu->slot_of);
        tq2ArenaTable(cc, st);
        st.tab_complete = true;
      }
      arena.waitReady(cc, *part, arena.pendingSlots(*part));
    } else if (fsu_route && (st.off_tab == nullptr ||
                             (split && (st.res_flag == nullptr ||
                                        st.tab_slots != *fsu->slot_of)))) {
      // the select kernel's table operand; its offsets are rewritten below.
      // Split: it also gives the hits their offsets and must describe the
      // slot map now (the async job keeps it current; anything else that
      // moved experts -- a prefill that lent the slots, the blocking step --
      // is caught here). The previous step's select has run (joined).
      tq2ArenaOffsets(st, *fsu->slot_of);
      tq2ArenaTable(cc, st);
      if (split) {
        std::vector<uint32_t> res(E);
        for (unsigned int e = 0; e < E; ++e)
          res[e] =
            (*fsu->slot_of)[e] != std::numeric_limits<unsigned int>::max();
        if (st.res_flag != nullptr)
          opencl::clReleaseMemObjectT(st.res_flag);
        st.res_flag =
          tq2Upload(cc, res.data(), E * sizeof(uint32_t), CL_MEM_READ_WRITE);
        st.tab_slots = *fsu->slot_of;
        tq2AsyncStats().table_uploads++;
      }
    }
    tq2RouteConstants(cc, ws, K);
    cl_mem ids = ws.r_ids.get(cc, K * sizeof(uint32_t));
    cl_mem wts = ws.r_w.get(cc, K * sizeof(float));
    cl_mem off_gu = ws.r_off_gu.get(cc, 2 * K * sizeof(uint32_t));
    cl_mem off_d = ws.r_off_d.get(cc, K * sizeof(uint32_t));
    // the kernels use only IEEE add / multiply / fma (moe_route_det.h) and
    // must be built with strict math (no relaxed-math defaults) to keep them
    const std::string ropts = std::string("-cl-std=CL3.0 -DNNTR_STRICT_FP") +
                              (tq2RouterHwFma(cc) ? "" : " -DRT_SOFT_FMA");
    // the router weight as exact bf16 when every value allows it, else fp32
    cl_mem d_rw16 = tq2RouterWeightBf16(cc, r_w, (size_t)H * E);
    auto nk =
      cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_norm", ropts);
    auto lk_ =
      cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_logits",
                           d_rw16 != nullptr ? ropts + " -DRT_W_BF16" : ropts);
    auto sk = cc->registerClKernel(
      tq2moe_route_kernel, tq2RouteSelectKernel(),
      fsu_route ? ropts + " -DRT_EXT=" + std::to_string(kTq2RouteExt) : ropts);
    NNTR_THROW_IF(!nk || !lk_ || !sk, std::runtime_error)
      << "tq2cx_moe_experts_cl: the device router did not build";
    const unsigned int nseg = (H + kTq2RouteSeg - 1) / kTq2RouteSeg;
    cl_mem partial = ws.r_part.get(cc, (size_t)nseg * E * sizeof(float));
    cl_mem ry = ws.r_y.get(cc, (size_t)H * sizeof(float));
    const int x_half = fp16 ? 1 : 0, has_norm = r_norm != nullptr ? 1 : 0;
    const float hscale = 1.0f / std::sqrt(static_cast<float>(H));
    const float eps = rt->epsilon;
    const int Hi = (int)H, Ei = (int)E, Ki = (int)K, nsegi = (int)nseg;
    cl_mem d_rw =
      d_rw16 != nullptr ? d_rw16 : tq2RouterOperand(cc, r_w, (size_t)H * E);
    cl_mem d_scale = tq2RouterOperand(cc, r_scale, H);
    cl_mem d_pes = tq2RouterOperand(cc, r_pes, E);
    cl_mem d_norm =
      r_norm != nullptr ? tq2RouterOperand(cc, r_norm, H) : d_scale;
    // the unused half / float input slot is bound to the same operand
    bool ok = rin.bind(nk, 0) && rin.bind(nk, 1) &&
              nk->SetKernelArguments(2, &x_half, sizeof(int)) &&
              nk->SetKernelArguments(3, &d_norm, sizeof(cl_mem)) &&
              nk->SetKernelArguments(4, &has_norm, sizeof(int)) &&
              nk->SetKernelArguments(5, &d_scale, sizeof(cl_mem)) &&
              nk->SetKernelArguments(6, &hscale, sizeof(float)) &&
              nk->SetKernelArguments(7, &eps, sizeof(float)) &&
              nk->SetKernelArguments(8, &Hi, sizeof(int)) &&
              nk->SetKernelArguments(9, &ry, sizeof(cl_mem)) &&
              lk_->SetKernelArguments(0, &ry, sizeof(cl_mem)) &&
              lk_->SetKernelArguments(1, &d_rw, sizeof(cl_mem)) &&
              lk_->SetKernelArguments(2, &Hi, sizeof(int)) &&
              lk_->SetKernelArguments(3, &Ei, sizeof(int)) &&
              lk_->SetKernelArguments(4, &partial, sizeof(cl_mem)) &&
              sk->SetKernelArguments(0, &partial, sizeof(cl_mem)) &&
              sk->SetKernelArguments(1, &nsegi, sizeof(int)) &&
              sk->SetKernelArguments(2, &d_pes, sizeof(cl_mem)) &&
              sk->SetKernelArguments(3, &st.off_tab, sizeof(cl_mem)) &&
              sk->SetKernelArguments(4, &Ei, sizeof(int)) &&
              sk->SetKernelArguments(5, &Ki, sizeof(int)) &&
              sk->SetKernelArguments(6, &ids, sizeof(cl_mem)) &&
              sk->SetKernelArguments(7, &wts, sizeof(cl_mem)) &&
              sk->SetKernelArguments(8, &off_gu, sizeof(cl_mem)) &&
              sk->SetKernelArguments(9, &off_d, sizeof(cl_mem));
    const unsigned int X = K + kTq2RouteExt;
    cl_mem ext = fsu_route ? ws.r_ext.get(cc, X * sizeof(uint32_t)) : nullptr;
    if (fsu_route)
      ok = ok && sk->SetKernelArguments(10, &ext, sizeof(cl_mem));
    NNTR_THROW_IF(!ok, std::runtime_error)
      << "tq2cx_moe_experts_cl: device router arguments";
    // the norm: one group, a lane per four row elements up to 512 lanes
    const size_t nlws = std::min<size_t>(
      std::min<size_t>(512, tq2MaxWorkGroup()), (H / 4 + 31) / 32 * 32);
    tq2Dispatch(cc, nk, 1, {nlws, 1, 1}, {nlws, 1, 1});
    // the projection: a lane per expert (rounded to the 16-wide subgroup),
    // a group per segment
    const size_t elws = (E + 15) / 16 * 16;
    tq2Dispatch(cc, lk_, 2, {elws, (size_t)nseg, 1}, {elws, 1, 1});
    tq2Dispatch(cc, sk, 1, {elws, 1, 1}, {elws, 1, 1});
    cl_mem s_gu_hit = nullptr, s_gu_miss = nullptr, s_d_hit = nullptr,
           s_d_miss = nullptr;
    if (split) {
      s_gu_hit = ws.s_gu_hit.get(cc, 2 * K * sizeof(int));
      s_gu_miss = ws.s_gu_miss.get(cc, 2 * K * sizeof(int));
      s_d_hit = ws.s_d_hit.get(cc, K * sizeof(int));
      s_d_miss = ws.s_d_miss.get(cc, K * sizeof(int));
      auto spk =
        cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_split", ropts);
      NNTR_THROW_IF(!spk, std::runtime_error)
        << "tq2cx_moe_experts_cl: the hit / miss split did not build";
      NNTR_THROW_IF(
        !(spk->SetKernelArguments(0, &ids, sizeof(cl_mem)) &&
          spk->SetKernelArguments(1, &st.res_flag, sizeof(cl_mem)) &&
          spk->SetKernelArguments(2, &Ki, sizeof(int)) &&
          spk->SetKernelArguments(3, &s_gu_hit, sizeof(cl_mem)) &&
          spk->SetKernelArguments(4, &s_gu_miss, sizeof(cl_mem)) &&
          spk->SetKernelArguments(5, &s_d_hit, sizeof(cl_mem)) &&
          spk->SetKernelArguments(6, &s_d_miss, sizeof(cl_mem))),
        std::runtime_error)
        << "tq2cx_moe_experts_cl: hit / miss split arguments";
      tq2Dispatch(cc, spk, 1, {16, 1, 1}, {16, 1, 1});
    }

    // the next streamed layer's router on this layer's router row (the
    // next-layer prefetch, NNTR_G4MOE_FSU_PREGATE): its host operands
    // exist, and the three router kernels with it, into st.pg_ext (the
    // next layer's ranked ids); false when an operand cannot be bound
    auto predictable = [&](const MoeRouter *nr) {
      const float *n_norm =
        nr->norm != nullptr ? tq2HostFloat(nr->norm) : nullptr;
      return tq2HostFloat(nr->weight) != nullptr &&
             tq2HostFloat(nr->scale) != nullptr &&
             tq2HostFloat(nr->per_expert_scale) != nullptr &&
             (nr->norm == nullptr || n_norm != nullptr) &&
             nr->weight->getDim().getDataLen() == (size_t)H * E;
    };
    auto predictInto = [&](const MoeRouter *nr) {
      const float *n_w = tq2HostFloat(nr->weight);
      const float *n_scale = tq2HostFloat(nr->scale);
      const float *n_pes = tq2HostFloat(nr->per_expert_scale);
      const float *n_norm =
        nr->norm != nullptr ? tq2HostFloat(nr->norm) : nullptr;
      if (st.pg_ext == nullptr) {
        opencl::ClMemAcctScope acct("tq2cx_router");
        st.pg_ext = tq2Upload(cc, std::vector<uint32_t>(X, 0).data(),
                              X * sizeof(uint32_t), CL_MEM_READ_WRITE);
      }
      cl_mem p_rw16 = tq2RouterWeightBf16(cc, n_w, (size_t)H * E);
      cl_mem p_rw =
        p_rw16 != nullptr ? p_rw16 : tq2RouterOperand(cc, n_w, (size_t)H * E);
      cl_mem p_scale = tq2RouterOperand(cc, n_scale, H);
      cl_mem p_pes = tq2RouterOperand(cc, n_pes, E);
      cl_mem p_norm =
        n_norm != nullptr ? tq2RouterOperand(cc, n_norm, H) : p_scale;
      const int p_has_norm = n_norm != nullptr ? 1 : 0;
      const float p_eps = nr->epsilon;
      auto plk = cc->registerClKernel(
        tq2moe_route_kernel, "tq2moe_route_logits",
        p_rw16 != nullptr ? ropts + " -DRT_W_BF16" : ropts);
      cl_mem p_ids = ws.p_ids.get(cc, K * sizeof(uint32_t));
      cl_mem p_w = ws.p_w.get(cc, K * sizeof(float));
      cl_mem p_ogu = ws.p_off_gu.get(cc, 2 * K * sizeof(uint32_t));
      cl_mem p_od = ws.p_off_d.get(cc, K * sizeof(uint32_t));
      const bool pok = plk && rin.bind(nk, 0) && rin.bind(nk, 1) &&
                       nk->SetKernelArguments(2, &x_half, sizeof(int)) &&
                       nk->SetKernelArguments(3, &p_norm, sizeof(cl_mem)) &&
                       nk->SetKernelArguments(4, &p_has_norm, sizeof(int)) &&
                       nk->SetKernelArguments(5, &p_scale, sizeof(cl_mem)) &&
                       nk->SetKernelArguments(6, &hscale, sizeof(float)) &&
                       nk->SetKernelArguments(7, &p_eps, sizeof(float)) &&
                       nk->SetKernelArguments(8, &Hi, sizeof(int)) &&
                       nk->SetKernelArguments(9, &ry, sizeof(cl_mem)) &&
                       plk->SetKernelArguments(0, &ry, sizeof(cl_mem)) &&
                       plk->SetKernelArguments(1, &p_rw, sizeof(cl_mem)) &&
                       plk->SetKernelArguments(2, &Hi, sizeof(int)) &&
                       plk->SetKernelArguments(3, &Ei, sizeof(int)) &&
                       plk->SetKernelArguments(4, &partial, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(0, &partial, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(1, &nsegi, sizeof(int)) &&
                       sk->SetKernelArguments(2, &p_pes, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(3, &st.off_tab, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(4, &Ei, sizeof(int)) &&
                       sk->SetKernelArguments(5, &Ki, sizeof(int)) &&
                       sk->SetKernelArguments(6, &p_ids, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(7, &p_w, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(8, &p_ogu, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(9, &p_od, sizeof(cl_mem)) &&
                       sk->SetKernelArguments(10, &st.pg_ext, sizeof(cl_mem));
      if (pok) {
        tq2Dispatch(cc, nk, 1, {nlws, 1, 1}, {nlws, 1, 1});
        tq2Dispatch(cc, plk, 2, {elws, (size_t)nseg, 1}, {elws, 1, 1});
        tq2Dispatch(cc, sk, 1, {elws, 1, 1}, {elws, 1, 1});
      }
      return pok;
    };

    if (flags) {
      // ---- decode without the host round trip (expert_fsu_flags.h) ----
      // The publish kernel places the misses from the victim queue the
      // host wrote at the layer's prepare, writes every selected expert's
      // record offsets (off_gu / off_d, over the select's) and the gate's
      // wait list, and publishes the misses on the ring; the poller starts
      // their reads straight into the slots; the gate ahead of the GEMVs
      // spins on the slots' ready words. Nothing here waits for the device.
      Tq2Flags &fl = Tq2Flags::get(cc);
      fsuflags::FsuFlagsHost &fh = fl.host();
      fh.throwIfFailed();
      Tq2Flags::Layer &fly = fl.layer(cc, *fsu, *part, st.gu_bytes, E, K);
      fl.coarseMap(cc, fly);
      if (fh.needsPrepare(fly.index))
        fh.prepare(fly.index);
      fl.coarseUnmap(cc, fly);
      if (fly.index == 0)
        tq2FlagsStats().tokens++;
      const cl_uint seq = fh.beginPublish(fly.index, false);
      const cl_uint layer_u = fly.index, C_u = fly.C, E_u = E, K_u = K, X_u = X;
      const cl_ulong cap = tq2FlagsSpinCap();
      auto pk = tq2FlagsKernel(cc, "tq2moe_fsu_publish");
      NNTR_THROW_IF(!pk, std::runtime_error)
        << "tq2cx_moe_experts_cl: the flags kernels did not build";
      const cl_kernel pkk = pk->GetKernel();
      bool pok =
        opencl::clSetKernelArgSVMPointer(pkk, 0, fl.ctl()) == CL_SUCCESS &&
        opencl::clSetKernelArgSVMPointer(pkk, 1, fly.lay) == CL_SUCCESS &&
        pk->SetKernelArguments(2, &ids, sizeof(cl_mem)) &&
        pk->SetKernelArguments(3, &ext, sizeof(cl_mem)) &&
        pk->SetKernelArguments(4, &off_gu, sizeof(cl_mem)) &&
        pk->SetKernelArguments(5, &off_d, sizeof(cl_mem)) &&
        pk->SetKernelArguments(6, &fly.shift, sizeof(cl_mem)) &&
        pk->SetKernelArguments(7, &fly.wait, sizeof(cl_mem)) &&
        pk->SetKernelArguments(8, &seq, sizeof(cl_uint)) &&
        pk->SetKernelArguments(9, &layer_u, sizeof(cl_uint)) &&
        pk->SetKernelArguments(10, &C_u, sizeof(cl_uint)) &&
        pk->SetKernelArguments(11, &E_u, sizeof(cl_uint)) &&
        pk->SetKernelArguments(12, &K_u, sizeof(cl_uint)) &&
        pk->SetKernelArguments(13, &X_u, sizeof(cl_uint)) &&
        pk->SetKernelArguments(14, &fly.stride_w, sizeof(cl_uint)) &&
        pk->SetKernelArguments(15, &fly.gu_w, sizeof(cl_uint)) &&
        pk->SetKernelArguments(16, &cap, sizeof(cl_ulong));
      NNTR_THROW_IF(!pok, std::runtime_error)
        << "tq2cx_moe_experts_cl: flags publish arguments";
      tq2Dispatch(cc, pk, 1, {1, 1, 1}, {1, 1, 1});
      cl_command_queue cq = q.GetCommandQueue();
      // the publish must reach the device for the reads to start
      opencl::clFlush(cq);
      // the activation row does not depend on the routing: quantize it now
      tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
      // next-layer prefetch through the ring (NNTR_G4MOE_FSU_PREGATE, the
      // first stage): the next streamed layer's router on this layer's
      // router row, then the read-ahead publish of its predicted experts
      // from that layer's victim queue; the poller reads them at low
      // priority straight into their slots, and that layer's own step finds
      // them (hit), promotes them (not started yet) or drops them
      const MoeExpertFsu *nx = fsu->next;
      const MoeRouter *nr = fsu->next_router;
      Tq2ExpertArena::Partition *np =
        nx != nullptr && nx->layer_index != UINT32_MAX
          ? arena.byIndex(nx->layer_index)
          : nullptr;
      if (tq2PregateM() > 0 && nr != nullptr && np != nullptr &&
          np->host != nullptr && np->slots == nx->slots && nx->fd >= 0 &&
          nx->offset != nullptr && nx->offset->size() == E &&
          nx->slot_of != nullptr && nx->slot_of->size() == E && nx->slots < E &&
          nx->record_bytes == fsu->record_bytes &&
          tq2PregateTarget(nx->layer_index) && predictable(nr)) {
        Tq2Flags::Layer &nly = fl.layer(cc, *nx, *np, st.gu_bytes, E, K);
        fl.coarseMap(cc, nly);
        if (fh.needsPrepare(nly.index))
          fh.prepare(nly.index);
        fl.coarseUnmap(cc, nly);
        if (predictInto(nr)) {
          const cl_uint s2 = fh.beginPublish(nly.index, true);
          const cl_uint m_u =
                          std::min<cl_uint>(std::min<cl_uint>(tq2PregateM(), X),
                                            fsuflags::kMaxItems),
                        keep_u = K, nl_u = nly.index, nC_u = nly.C;
          auto sk2 = tq2FlagsKernel(cc, "tq2moe_fsu_spec");
          NNTR_THROW_IF(!sk2, std::runtime_error)
            << "tq2cx_moe_experts_cl: the flags kernels did not build";
          const cl_kernel sk2k = sk2->GetKernel();
          const bool sok =
            opencl::clSetKernelArgSVMPointer(sk2k, 0, fl.ctl()) == CL_SUCCESS &&
            opencl::clSetKernelArgSVMPointer(sk2k, 1, nly.lay) == CL_SUCCESS &&
            sk2->SetKernelArguments(2, &st.pg_ext, sizeof(cl_mem)) &&
            sk2->SetKernelArguments(3, &s2, sizeof(cl_uint)) &&
            sk2->SetKernelArguments(4, &nl_u, sizeof(cl_uint)) &&
            sk2->SetKernelArguments(5, &nC_u, sizeof(cl_uint)) &&
            sk2->SetKernelArguments(6, &E_u, sizeof(cl_uint)) &&
            sk2->SetKernelArguments(7, &m_u, sizeof(cl_uint)) &&
            sk2->SetKernelArguments(8, &keep_u, sizeof(cl_uint)) &&
            sk2->SetKernelArguments(9, &cap, sizeof(cl_ulong));
          NNTR_THROW_IF(!sok, std::runtime_error)
            << "tq2cx_moe_experts_cl: flags read-ahead arguments";
          tq2Dispatch(cc, sk2, 1, {1, 1, 1}, {1, 1, 1});
          opencl::clFlush(cq);
        }
      }
      const bool defer = tq2DeferOn() && g.output_deferred != nullptr;
      if (defer && in.svm)
        q.enqueueSVMMap(in.svm, in.bytes, /*read_only=*/true, nullptr,
                        /*async=*/true);
      if (defer && rin.svm && rin.svm != in.svm)
        q.enqueueSVMMap(rin.svm, rin.bytes, /*read_only=*/true, nullptr,
                        /*async=*/true);
      Tq2Stack *stp = &st;
      uint32_t *ctl = fl.ctl();
      uint32_t *lay = fly.lay;
      cl_mem wait = fly.wait;
      // the tail: the gate (a work-group spinning on the slots' ready
      // words), then the GEMVs and the combine of the async step's tail
      auto tail = [cc, stp, ctl, lay, wait, layer_u, C_u, K_u, cap, K, I, H,
                   fp16, exact = g.exact_gelu, defer, in, out, off_gu, off_d,
                   wts, seq, &fh, &fl, &fly]() {
        auto &cq_mgr = cc->command_queue_inst_;
        if (tq2FsuFlagsCoarse()) {
          // the coarse-grain test mode: a running kernel does not see the
          // host's stores there (Xe3: the gates timed out, with or without
          // a map / unmap of the blocks) and the host sees a ring entry's
          // sequence number before its items (Xe3: an entry taken with two
          // items still zero). So, in OpenCL's coarse-grain rules: the
          // host waits for the step's kernels, maps the blocks, consumes
          // the ring itself (no poller), waits for the step's reads (the
          // slots' own coarse-grain writes make the records visible) and
          // unmaps the blocks; no gate
          cl_command_queue cq = cq_mgr.GetCommandQueue();
          opencl::clFinish(cq);
          fl.coarseMap(cc, fly);
          for (int spins = 0; !fh.consumed(seq); ++spins) {
            NNTR_THROW_IF(!fh.pollOnce() && spins > 1000, std::runtime_error)
              << "NNTR_G4MOE_FSU_FLAGS=coarse: publish " << seq
              << " not in the ring after its kernel completed";
          }
          fh.waitStepReads(layer_u, seq);
          fl.coarseUnmap(cc, fly);
        } else {
          auto gk = tq2FlagsKernel(cc, "tq2moe_fsu_gate");
          NNTR_THROW_IF(!gk, std::runtime_error)
            << "tq2cx_moe_experts_cl: the flags kernels did not build";
          const cl_kernel gkk = gk->GetKernel();
          cl_mem w_buf = wts, wait_buf = wait;
          const bool gok =
            opencl::clSetKernelArgSVMPointer(gkk, 0, ctl) == CL_SUCCESS &&
            opencl::clSetKernelArgSVMPointer(gkk, 1, lay) == CL_SUCCESS &&
            gk->SetKernelArguments(2, &wait_buf, sizeof(cl_mem)) &&
            gk->SetKernelArguments(3, &w_buf, sizeof(cl_mem)) &&
            gk->SetKernelArguments(4, &layer_u, sizeof(cl_uint)) &&
            gk->SetKernelArguments(5, &C_u, sizeof(cl_uint)) &&
            gk->SetKernelArguments(6, &K_u, sizeof(cl_uint)) &&
            gk->SetKernelArguments(7, &cap, sizeof(cl_ulong));
          NNTR_THROW_IF(!gok, std::runtime_error)
            << "tq2cx_moe_experts_cl: flags gate arguments";
          const size_t glws = std::max<size_t>(16, ((size_t)K + 15) / 16 * 16);
          tq2Dispatch(cc, gk, 1, {glws, 1, 1}, {glws, 1, 1});
        }
        Tq2Workspace &w = tq2Workspace();
        cl_mem ygu = w.yg.get(cc, (size_t)2 * K * I * sizeof(float));
        cl_mem act = w.act.get(cc, (size_t)K * I * sizeof(float));
        cl_mem yd = w.yd.get(cc, (size_t)K * H * sizeof(float));
        tq2GemvBatchDev(cc, w, *stp, off_gu, w.c_zero, 2 * K, ygu, I, H);
        tq2Glu(cc, ygu, 0, ygu, (size_t)K * I, act, (size_t)K * I, exact);
        tq2QuantizeGemv(cc, w, act, nullptr, K, I);
        tq2GemvBatchDev(cc, w, *stp, off_d, w.c_iota, K, yd, H, I);
        NNTR_THROW_IF(
          !tq2Combine(cc, fp16, yd, out, w.c_iota, wts, w.c_count, H, K, 1),
          std::runtime_error)
          << "tq2cx_moe_experts_cl: combine arguments";
        if (!defer) {
          if (in.svm)
            cq_mgr.enqueueSVMMap(in.svm, in.bytes, /*read_only=*/true, nullptr,
                                 /*async=*/true);
          if (out.svm)
            cq_mgr.enqueueSVMMap(out.svm, out.bytes, /*read_only=*/true,
                                 nullptr, /*async=*/true);
        }
        // the gate spins until the reads land: hand it over now
        opencl::clFlush(cq_mgr.GetCommandQueue());
      };
      if (defer) {
        tq2Tails().push_back({tq2ActKey(out), std::move(tail)});
        setDeferredOutputsPending(true);
        *g.output_deferred = true;
      } else {
        tail();
        if (rin.svm && rin.svm != in.svm)
          q.enqueueSVMMap(rin.svm, rin.bytes, /*read_only=*/true, nullptr,
                          /*async=*/true);
      }
      return true;
    }

    std::vector<unsigned int> route_used;
    if (amode != Tq2AsyncMode::off) {
      // non-blocking: a job on the reader pool makes the selected experts
      // resident once the select kernel is done (Tq2AsyncJob); the expert
      // GEMVs and the combine (the tail) run behind it, enqueued now or,
      // deferred, by the output's first reader, so the work the graph
      // enqueues in between (the dense MLP) runs while the misses are read
      cl_command_queue cq = q.GetCommandQueue();
      cl_context ctx = cc->context_inst_.GetContextNoRetain();
      cl_event sel = nullptr, user = nullptr;
      cl_int err = opencl::clEnqueueBarrierWithWaitList(cq, 0, nullptr, &sel);
      if (err == CL_SUCCESS && amode == Tq2AsyncMode::event) {
        user = opencl::clCreateUserEvent(ctx, &err);
        if (err == CL_SUCCESS)
          err = opencl::clRetainEvent(user); // the job's reference
      }
      NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
        << "tq2cx_moe_experts_cl: async decode fence failed (" << err << ")";
      auto job = std::make_shared<Tq2AsyncJob>();
      job->cc = cc;
      job->io = arena.ioQueue(cc);
      job->cq = cq;
      job->ext = ext;
      job->split = split;
      // split: the job writes the misses' tables, the hits read off_gu /
      // off_d (the select's) meanwhile
      job->off_gu =
        split ? ws.s_off_gu.get(cc, 2 * K * sizeof(uint32_t)) : off_gu;
      job->off_d = split ? ws.s_off_d.get(cc, K * sizeof(uint32_t)) : off_d;
      job->K = K;
      job->X = X;
      job->E = E;
      job->part = part;
      job->st = &st;
      job->fd = fsu->fd;
      job->offset = fsu->offset;
      job->record_bytes = fsu->record_bytes;
      job->slot_of = fsu->slot_of;
      job->acquire = fsu->acquire;
      job->filled = fsu->filled;
      job->refresh = fsu->refresh;
      job->last_use = part->lastUses();
      job->user = user;
      job->layer = fsu->layer_index;
      st.async = job;
      auto *holder = new std::shared_ptr<Tq2AsyncJob>(job);
      err =
        opencl::clSetEventCallback(sel, CL_COMPLETE, tq2AsyncSelected, holder);
      if (err != CL_SUCCESS) {
        delete holder;
        st.async.reset();
        if (user != nullptr) {
          opencl::clSetUserEventStatus(user, -1);
          opencl::clReleaseEvent(user);
          opencl::clReleaseEvent(user);
        }
        opencl::clReleaseEvent(sel);
        NNTR_THROW_IF(true, std::runtime_error)
          << "tq2cx_moe_experts_cl: select callback failed (" << err << ")";
      }
      opencl::clFlush(cq); // the select must run for the callback to come
      // the activation row does not depend on the routing: quantize it now,
      // and hand the inputs back as the end of the call does
      tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
      const bool exact = g.exact_gelu;
      if (split) {
        // the hits' half now, while the job reads the misses: gate / up and
        // down over the resident experts' records (the miss rows skip), the
        // GeGLU over every row (a miss row's is recomputed by the tail). A
        // row is the same computation in either launch, the combine reads
        // the rows in the same order: the same bits as one launch.
        cl_mem ygu = ws.yg.get(cc, (size_t)2 * K * I * sizeof(float));
        cl_mem act = ws.act.get(cc, (size_t)K * I * sizeof(float));
        cl_mem yd = ws.yd.get(cc, (size_t)K * H * sizeof(float));
        Tq2Workspace &wd = tq2WorkspaceDown();
        tq2GemvBatchDev(cc, ws, st, off_gu, s_gu_hit, 2 * K, ygu, I, H);
        tq2Glu(cc, ygu, 0, ygu, (size_t)K * I, act, (size_t)K * I, exact);
        tq2QuantizeGemv(cc, wd, act, nullptr, K, I);
        tq2GemvBatchDev(cc, wd, st, off_d, s_d_hit, K, yd, H, I);
        opencl::clFlush(cq);
        tq2AsyncStats().split++;
      }
      // next-layer prefetch (pre-gating): the next streamed layer's router
      // on this layer's router row; the predicted experts the next layer
      // does not hold are read ahead into host memory once this step's job
      // and the prediction are both done (Tq2Prediction)
      const MoeExpertFsu *nx = fsu->next;
      const MoeRouter *nr = fsu->next_router;
      if (tq2PregateM() > 0 && nx != nullptr && nr != nullptr && nx->fd >= 0 &&
          nx->offset != nullptr && nx->offset->size() == E &&
          nx->slot_of != nullptr && nx->slot_of->size() == E && nx->slots < E &&
          nx->record_bytes == fsu->record_bytes &&
          nx->layer_index != UINT32_MAX && tq2PregateTarget(nx->layer_index)) {
        if (predictable(nr)) {
          if (st.pg) {
            // the last prediction's readback of pg_ext is done (it was
            // issued by the end of the previous token's step of this layer)
            st.pg->waitIssued();
            st.pg.reset();
          }
          cl_event pev = nullptr;
          bool pok = predictInto(nr);
          if (pok)
            pok = opencl::clEnqueueBarrierWithWaitList(cq, 0, nullptr, &pev) ==
                  CL_SUCCESS;
          if (pok) {
            auto pred = std::make_shared<Tq2Prediction>();
            pred->io = arena.ioQueue(cc);
            pred->ext = st.pg_ext;
            pred->X = X;
            pred->m = tq2PregateM();
            pred->E = E;
            pred->layer = nx->layer_index;
            pred->gen = tq2StageCache().generation(nx->layer_index);
            pred->src.fd = nx->fd;
            pred->src.offset = nx->offset;
            pred->src.record_bytes = nx->record_bytes;
            pred->slot_of = *nx->slot_of;
            st.pg = pred;
            auto *ph = new std::shared_ptr<Tq2Prediction>(pred);
            if (opencl::clSetEventCallback(pev, CL_COMPLETE, tq2PregateReady,
                                           ph) != CL_SUCCESS) {
              delete ph;
              opencl::clReleaseEvent(pev);
              pred->arrive(/*from_callback=*/false); // no prediction
            }
            // the job (maybe running already) arrives when it is done
            job->attachPrediction(pred);
            opencl::clFlush(cq);
          }
        }
      }
      const bool defer = tq2DeferOn() && g.output_deferred != nullptr;
      if (defer && in.svm)
        q.enqueueSVMMap(in.svm, in.bytes, /*read_only=*/true, nullptr,
                        /*async=*/true);
      if (defer && rin.svm && rin.svm != in.svm)
        q.enqueueSVMMap(rin.svm, rin.bytes, /*read_only=*/true, nullptr,
                        /*async=*/true);
      Tq2Stack *stp = &st;
      // the tail: the GEMVs read the slots' offsets the job writes; every
      // slot's next writer waits for them (the job chose the slots)
      auto tail = [cc, stp, part, evicting, amode, job, user, K, I, H, fp16,
                   exact, defer, in, out, off_gu, off_d, wts, split, s_gu_miss,
                   s_d_miss]() {
        auto &cq_mgr = cc->command_queue_inst_;
        // event mode, the job already done (a deferred tail comes after the
        // dense MLP): its writes have completed, so the kernels below need
        // no barrier on its event, and without a miss no kernel at all
        const bool done_early = amode == Tq2AsyncMode::event && job->ready();
        if (done_early) {
          opencl::clReleaseEvent(user); // the host's reference
          tq2AsyncStats().event_skipped++;
          job->join(); // rethrow its failure
        } else if (amode == Tq2AsyncMode::join) {
          // the work enqueued since the call (the hits, the dense MLP) runs
          // while the host waits
          opencl::clFlush(cq_mgr.GetCommandQueue());
          const auto t0 = std::chrono::steady_clock::now();
          job->wait();
          tq2AsyncStats().host_wait_us += tq2UsSince(t0);
          job->join(); // rethrow its failure
        } else {
          const cl_int e = opencl::clEnqueueBarrierWithWaitList(
            cq_mgr.GetCommandQueue(), 1, &user, nullptr);
          opencl::clReleaseEvent(user); // the host's reference
          NNTR_THROW_IF(e != CL_SUCCESS, std::runtime_error)
            << "tq2cx_moe_experts_cl: async decode barrier failed (" << e
            << ")";
        }
        tq2AsyncStats().tails++;
        Tq2Workspace &w = tq2Workspace();
        cl_mem ygu = w.yg.get(cc, (size_t)2 * K * I * sizeof(float));
        cl_mem act = w.act.get(cc, (size_t)K * I * sizeof(float));
        cl_mem yd = w.yd.get(cc, (size_t)K * H * sizeof(float));
        if (split) {
          // the misses' half; once the job is known done, no miss = nothing
          if ((amode != Tq2AsyncMode::join && !done_early) ||
              job->misses != 0) {
            Tq2Workspace &wd = tq2WorkspaceDown();
            tq2GemvBatchDev(cc, w, *stp, job->off_gu, s_gu_miss, 2 * K, ygu, I,
                            H);
            tq2Glu(cc, ygu, 0, ygu, (size_t)K * I, act, (size_t)K * I, exact);
            tq2QuantizeGemv(cc, wd, act, nullptr, K, I);
            tq2GemvBatchDev(cc, wd, *stp, job->off_d, s_d_miss, K, yd, H, I);
          } else {
            tq2AsyncStats().no_miss++;
          }
        } else {
          tq2GemvBatchDev(cc, w, *stp, off_gu, w.c_zero, 2 * K, ygu, I, H);
          tq2Glu(cc, ygu, 0, ygu, (size_t)K * I, act, (size_t)K * I, exact);
          tq2QuantizeGemv(cc, w, act, nullptr, K, I);
          tq2GemvBatchDev(cc, w, *stp, off_d, w.c_iota, K, yd, H, I);
        }
        if (evicting) {
          std::vector<unsigned int> all(part->slots);
          for (unsigned int sl = 0; sl < part->slots; ++sl)
            all[sl] = sl;
          Tq2ExpertArena::get().markUse(cc, *part, all);
        }
        NNTR_THROW_IF(
          !tq2Combine(cc, fp16, yd, out, w.c_iota, wts, w.c_count, H, K, 1),
          std::runtime_error)
          << "tq2cx_moe_experts_cl: combine arguments";
        // deferred: the caller already handed the output to the device as
        // it was (unmapped); in place, the end of the call maps it like the
        // blocking step does
        if (!defer) {
          if (in.svm)
            cq_mgr.enqueueSVMMap(in.svm, in.bytes, /*read_only=*/true, nullptr,
                                 /*async=*/true);
          if (out.svm)
            cq_mgr.enqueueSVMMap(out.svm, out.bytes, /*read_only=*/true,
                                 nullptr, /*async=*/true);
        }
      };
      if (defer) {
        tq2Tails().push_back({tq2ActKey(out), std::move(tail)});
        setDeferredOutputsPending(true);
        *g.output_deferred = true;
        tq2AsyncStats().deferred++;
      } else {
        tail();
        if (rin.svm && rin.svm != in.svm)
          q.enqueueSVMMap(rin.svm, rin.bytes, /*read_only=*/true, nullptr,
                          /*async=*/true);
      }
      return true;
    } else if (fsu_route) {
      // wait for the selection, read the ids back (rank order)
      std::vector<uint32_t> ranked(X);
      cl_command_queue cq = q.GetCommandQueue();
      cl_int err =
        opencl::clEnqueueReadBuffer(cq, ext, CL_TRUE, 0, X * sizeof(uint32_t),
                                    ranked.data(), 0, nullptr, nullptr);
      NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
        << "tq2cx_moe_experts_cl: routing readback failed (" << err << ")";
      std::vector<unsigned int> ids(ranked.begin(), ranked.begin() + K);
      std::sort(ids.begin(), ids.end()); // the order ids[] / w[] are in
      for (unsigned int e : ids)
        NNTR_THROW_IF(e >= E, std::runtime_error)
          << "tq2cx_moe_experts_cl: routed expert " << e << " of " << E;
      route_used = fsuAcquire(ids);
      // recency hint: the extended ranks, least to most recent
      if (fsu->refresh)
        fsu->refresh(std::vector<unsigned int>(ranked.rbegin(), ranked.rend()));
      std::vector<uint32_t> ogu(2 * K), od(K);
      for (unsigned int a = 0; a < K; ++a) {
        ogu[a] = st.off_gate[ids[a]];
        ogu[K + a] = st.off_up[ids[a]];
        od[a] = st.off_down[ids[a]];
      }
      err = opencl::clEnqueueWriteBuffer(cq, off_gu, CL_TRUE, 0,
                                         ogu.size() * sizeof(uint32_t),
                                         ogu.data(), 0, nullptr, nullptr);
      if (err == CL_SUCCESS)
        err = opencl::clEnqueueWriteBuffer(cq, off_d, CL_TRUE, 0,
                                           od.size() * sizeof(uint32_t),
                                           od.data(), 0, nullptr, nullptr);
      NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
        << "tq2cx_moe_experts_cl: slot offsets write failed (" << err << ")";
    }

    cl_mem ygu = ws.yg.get(cc, (size_t)2 * K * I * sizeof(float));
    cl_mem act = ws.act.get(cc, (size_t)K * I * sizeof(float));
    yd = ws.yd.get(cc, (size_t)K * H * sizeof(float));
    if (in_q.xq == nullptr)
      tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
    tq2GemvBatchDev(cc, ws, st, off_gu, ws.c_zero, 2 * K, ygu, I, H, in_q);
    tq2GluQuantize(cc, ws, ygu, 0, ygu, (size_t)K * I, act, K, I, g.exact_gelu,
                   tq2GemvPermAct(cc));
    tq2GemvBatchDev(cc, ws, st, off_d, ws.c_iota, K, yd, H, I);
    if (evicting && fsu_route)
      arena.markUse(cc, *part, route_used);
    d_src = ws.c_iota;
    d_w = wts;
    d_cnt = ws.c_count;
    max_k = K;
  } else {
    const unsigned int k = (unsigned int)g.active.size();
    const auto &asg = *g.assignments;
    // Streamed experts, a call of many rows on an evicting layer: whole
    // layers stream through partitions the arena lends (the layer forgot its
    // resident experts first); the experts run per lent partition.
    const bool stream = T > 1 && tq2StreamOn(fsu, part, E, T);
    if (part != nullptr && !stream)
      arena.streamEnd();
    ahead = T > 1 && part != nullptr && !stream && fsu->next_ahead != nullptr;
    // A streamed layer whose block buffer holds every expert of the layer
    // (expert e in slot e of one cl_mem, Tq2ExpertArena::partition): the
    // resident launch sequence below, once the block is in device memory.
    Tq2ExpertArena::StreamSet *bset = nullptr;
    if (stream && arena.streamSet(fsu->layer_index % 2).block != nullptr)
      bset = &arena.streamSet(fsu->layer_index % 2);
    // Launch order of the active experts. Resident, streamed, or one chunk:
    // ascending id. Read in chunks (more active experts than slots): the
    // experts with the most rows first, so the largest launches start while
    // the reads of the rest are still to come. Each expert's rows are laid
    // out in launch order, so a chunk's rows are one contiguous range; the
    // combine still sums every token's contributions in ascending expert id,
    // so the order changes no value.
    std::vector<unsigned int> ord(k);
    for (unsigned int a = 0; a < k; ++a)
      ord[a] = a;
    const bool chunked = T > 1 && part != nullptr && !stream && k > part->slots;
    // a short prefill (NNTR_G4MOE_FSU_PREFILL_LOOKAHEAD): the chunks take
    // the resident experts first (those the previous layer read ahead among
    // them), so no chunk's misses evict an expert a later chunk needs and
    // the first chunk's kernels start without a read
    const bool res_first = chunked && tq2PrefillAheadOn() && fsu->resident;
    if (T > 1 && part != nullptr && !stream)
      tq2PrefillAheadTally(*fsu, g.active);
    std::vector<char> res(res_first ? k : 0, 0);
    for (unsigned int a = 0; res_first && a < k; ++a)
      res[a] = fsu->resident(g.active[a]) ? 1 : 0;
    if (chunked)
      std::stable_sort(
        ord.begin(), ord.end(), [&](unsigned int x, unsigned int y) {
          if (res_first && res[x] != res[y])
            return res[x] > res[y];
          return asg[g.active[x]].size() > asg[g.active[y]].size();
        });
    // per-token contribution table, ascending expert id (the active list
    // order): source row of the down output and the routing weight
    std::vector<int> row_off(k), row_cnt(k);
    unsigned int R = 0, max_rows = 0;
    for (unsigned int i = 0; i < k; ++i) {
      const unsigned int a = ord[i];
      row_off[a] = (int)R;
      row_cnt[a] = (int)asg[g.active[a]].size();
      R += row_cnt[a];
      max_rows = std::max(max_rows, (unsigned int)row_cnt[a]);
    }
    // gathered row layout: the launch order above (ascending id unless
    // chunked). Any layout gives the same results: rows are quantized
    // independently and the combine adds the contributions in table order
    // (ascending expert id), not row order. Streamed or chunked experts
    // (moe_fsu) run the per-chunk grouped GEMMs, not the tile list; a
    // streamed layer in a block buffer (bset) runs the one pass.
    const bool one_pass = (!stream && !chunked) || bset != nullptr;
    const bool tiles = T > 1 && one_pass && !tq2Dpas(cc) && tq2TileCfg().flat;
    std::vector<int> counts(T, 0);
    for (unsigned int a = 0; a < k; ++a)
      for (const auto &tw : asg[g.active[a]])
        counts[tw.first]++;
    for (unsigned int t = 0; t < T; ++t)
      max_k = std::max(max_k, (unsigned int)counts[t]);
    std::vector<int> src_row((size_t)T * max_k, 0);
    std::vector<float> weight((size_t)T * max_k, 0.0f);
    std::vector<int> fill(T, 0);
    std::vector<int> gather_src(R);
    for (unsigned int a = 0; a < k; ++a) {
      const auto &lst = asg[g.active[a]];
      for (size_t i = 0; i < lst.size(); ++i) {
        const unsigned int t = lst[i].first;
        const int r = row_off[a] + (int)i;
        gather_src[r] = (int)t;
        src_row[(size_t)t * max_k + fill[t]] = r;
        weight[(size_t)t * max_k + fill[t]] = lst[i].second;
        fill[t]++;
      }
    }

    if (T == 1) {
      // ---- decode: one activation row, all experts batched ----
      std::vector<unsigned int> used;
      if (part != nullptr)
        used = fsuAcquire(g.active);
      const Tq2QRows in_q = tq2ClaimNormRow(cc, ws, in, fp16, H);
      if (in_q.xq == nullptr)
        tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
      std::vector<uint32_t> off_gu(2 * k);
      std::vector<int> ar_gu(2 * k, 0);
      for (unsigned int a = 0; a < k; ++a) {
        off_gu[a] = st.off_gate[g.active[a]];
        off_gu[k + a] = st.off_up[g.active[a]];
      }
      cl_mem ygu = ws.yg.get(cc, (size_t)2 * k * I * sizeof(float));
      tq2GemvBatch(cc, ws, st, off_gu, ar_gu, ygu, I, H, in_q);
      cl_mem act = ws.act.get(cc, (size_t)k * I * sizeof(float));
      tq2GluQuantize(cc, ws, ygu, 0, ygu, (size_t)k * I, act, k, I,
                     g.exact_gelu, tq2GemvPermAct(cc));
      std::vector<uint32_t> off_d(k);
      std::vector<int> ar_d(k);
      for (unsigned int a = 0; a < k; ++a) {
        off_d[a] = st.off_down[g.active[a]];
        ar_d[a] = (int)a;
      }
      yd = ws.yd.get(cc, (size_t)k * H * sizeof(float));
      tq2GemvBatch(cc, ws, st, off_d, ar_d, yd, H, I);
      if (evicting)
        arena.markUse(cc, *part, used);
    } else {
      // ---- prefill: gather, grouped GEMMs over every expert's rows ----
      // tq2cx_gemm_tiles_v2_portable reads the int8 rows in TQ2 order: the
      // two quantizers below then store that order
      int v2 = tq2V2Plan(cc, tiles, H, I, R);
      cl_mem d_gsrc = tq2Small(cc, gather_src.data(), R * sizeof(int));
      if (one_pass) {
        // one launch sequence over every active expert (the resident ones,
        // or a streamed layer's block: expert e in slot e of the block)
        std::vector<unsigned int> used;
        Tq2Stack bst;
        const Tq2Stack *rs = &st;
        if (bset != nullptr) {
          tq2StreamIssue(cc, *fsu, st, E);
          bst.layout = st.layout;
          bst.H = st.H;
          bst.I = st.I;
          bst.arena = true;
          bst.gu_bytes = st.gu_bytes;
          bst.dn_bytes = st.dn_bytes;
          bst.slot_bytes = st.slot_bytes;
          bst.slot_stride = st.slot_stride;
          bst.shift = st.shift;
          bst.buf = bset->block;
          tq2ArenaOffsets(bst, [E]() {
            std::vector<unsigned int> v(E);
            for (unsigned int e = 0; e < E; ++e)
              v[e] = e;
            return v;
          }());
          rs = &bst;
        } else if (part != nullptr) {
          used = fsuAcquire(g.active);
        }
        auto quantize_in = [&](bool perm) {
          if (tq2QuantFuse()) {
            tq2GatherQuantize(cc, ws, in, fp16, d_gsrc, R, H, perm);
          } else {
            auto gk = cc->registerClKernel(
              kTq2MoeAux, fp16 ? "tq2moe_gather_f16" : "tq2moe_gather_f32");
            NNTR_THROW_IF(!gk, std::runtime_error)
              << "tq2cx_moe_experts_cl: aux kernels did not build";
            cl_mem xf = ws.xf.get(cc, (size_t)R * H * sizeof(float));
            const int Hi = (int)H, tot = (int)((size_t)R * H);
            bool ok = in.bind(gk, 0) &&
                      gk->SetKernelArguments(1, &xf, sizeof(cl_mem)) &&
                      gk->SetKernelArguments(2, &d_gsrc, sizeof(cl_mem)) &&
                      gk->SetKernelArguments(3, &Hi, sizeof(int)) &&
                      gk->SetKernelArguments(4, &tot, sizeof(int));
            if (ok)
              tq2Dispatch(cc, gk, 1, {((size_t)tot + 63) / 64 * 64, 1, 1},
                          {64, 1, 1});
            if (!ok)
              opencl::clReleaseMemObjectT(d_gsrc);
            NNTR_THROW_IF(!ok, std::runtime_error)
              << "tq2cx_moe_experts_cl: gather arguments";
            tq2Quantize(cc, ws, xf, nullptr, R, H, false, perm);
          }
        };
        quantize_in(v2 >= 0);
        std::vector<uint32_t> off_g(k), off_u(k), off_d(k);
        for (unsigned int a = 0; a < k; ++a) {
          off_g[a] = rs->off_gate[g.active[a]];
          off_u[a] = rs->off_up[g.active[a]];
          off_d[a] = rs->off_down[g.active[a]];
        }
        std::unique_ptr<Tq2Tiles> tl;
        if (tiles) {
          tl = std::make_unique<Tq2Tiles>();
          tq2PlanTiles(cc, row_off, row_cnt, *tl);
        }
        // a v2 GEMM that cannot run (no usable form at GEMM time): quantize
        // its rows again in plain order (requant) and take the P-TILES kernel
        // for the rest of the layer (same outputs bit for bit)
        auto gemm = [&](const std::vector<uint32_t> &off, cl_mem Y,
                        unsigned int N, unsigned int K, auto requant) {
          if (!tl) {
            tq2GroupedGemm(cc, ws, *rs, off, row_off, row_cnt, max_rows, Y, N,
                           K);
            return;
          }
          if (tq2GroupedGemmTiles(cc, ws, *rs, *tl, off, Y, N, K, R, v2))
            return;
          v2 = -1;
          requant();
          tq2GroupedGemmTiles(cc, ws, *rs, *tl, off, Y, N, K, R, v2);
        };
        if (bset != nullptr) {
          // the GEMMs read the block: wait until all of it is in device
          // memory (the input rows above are quantized meanwhile)
          const unsigned int x = fsu->layer_index % 2;
          for (size_t j = 0; j < bset->parts.size(); ++j)
            arena.streamWait(x, j, fsu->fadvise_dontneed);
        }
        auto requant_in = [&]() { quantize_in(false); };
        cl_mem yg = ws.yg.get(cc, (size_t)R * I * sizeof(float));
        cl_mem yu = ws.yu.get(cc, (size_t)R * I * sizeof(float));
        gemm(off_g, yg, I, H, requant_in);
        gemm(off_u, yu, I, H, requant_in);
        opencl::clReleaseMemObjectT(d_gsrc);
        cl_mem act = ws.act.get(cc, (size_t)R * I * sizeof(float));
        if (tq2QuantFuse()) {
          tq2GluQuantize(cc, ws, yg, 0, yu, 0, act, R, I, g.exact_gelu,
                         v2 >= 0);
        } else {
          tq2Glu(cc, yg, 0, yu, 0, act, (size_t)R * I, g.exact_gelu);
          tq2Quantize(cc, ws, act, nullptr, R, I, false, v2 >= 0);
        }
        yd = ws.yd.get(cc, (size_t)R * H * sizeof(float));
        // both GeGLU paths leave the activations in act
        gemm(off_d, yd, H, I,
             [&]() { tq2Quantize(cc, ws, act, nullptr, R, I, false, false); });
        if (bset != nullptr) {
          // the slots read, per lent partition (expert e: slot e % C of
          // part e / C), behind one marker
          const unsigned int C = part->slots;
          std::vector<
            std::pair<Tq2ExpertArena::Partition *, std::vector<unsigned int>>>
            uses(bset->parts.size());
          for (size_t j = 0; j < bset->parts.size(); ++j)
            uses[j].first = bset->parts[j];
          for (unsigned int a = 0; a < k; ++a) {
            const unsigned int e = g.active[a];
            NNTR_THROW_IF(e / C >= uses.size(), std::logic_error)
              << "tq2cx_moe_experts_cl: expert " << e
              << " outside the streamed block";
            uses[e / C].second.push_back(e % C);
          }
          arena.markUseParts(cc, uses);
        } else if (evicting) {
          arena.markUse(cc, *part, used);
        }
      } else {
        // streamed or chunked experts: gather the rows in fp32 (the
        // per-chunk quantizers read row ranges of it)
        auto gk = cc->registerClKernel(kTq2MoeAux, fp16 ? "tq2moe_gather_f16"
                                                        : "tq2moe_gather_f32");
        NNTR_THROW_IF(!gk, std::runtime_error)
          << "tq2cx_moe_experts_cl: aux kernels did not build";
        cl_mem xf = ws.xf.get(cc, (size_t)R * H * sizeof(float));
        const int Hi = (int)H, tot = (int)((size_t)R * H);
        bool ok = in.bind(gk, 0) &&
                  gk->SetKernelArguments(1, &xf, sizeof(cl_mem)) &&
                  gk->SetKernelArguments(2, &d_gsrc, sizeof(cl_mem)) &&
                  gk->SetKernelArguments(3, &Hi, sizeof(int)) &&
                  gk->SetKernelArguments(4, &tot, sizeof(int));
        if (ok)
          tq2Dispatch(cc, gk, 1, {((size_t)tot + 63) / 64 * 64, 1, 1},
                      {64, 1, 1});
        opencl::clReleaseMemObjectT(d_gsrc);
        NNTR_THROW_IF(!ok, std::runtime_error)
          << "tq2cx_moe_experts_cl: gather arguments";
        cl_mem yg = ws.yg.get(cc, (size_t)R * I * sizeof(float));
        cl_mem yu = ws.yu.get(cc, (size_t)R * I * sizeof(float));
        cl_mem act = ws.act.get(cc, (size_t)R * I * sizeof(float));
        yd = ws.yd.get(cc, (size_t)R * H * sizeof(float));
        // In chunks: a chunk's grouped GEMMs produce exactly its experts'
        // rows (row_off / row_cnt are absolute), and the quantizers and the
        // GeGLU work per row, restricted to the chunk's rows, so the output
        // does not depend on the chunking. The int8 rows of the down
        // projection reuse the input's buffer: in-order, a chunk's input
        // rows are consumed before its GeGLU rows are quantized.
        ws.xqGet(cc, (size_t)R * std::max(H, I));
        ws.xs.get(cc, (size_t)R * sizeof(float));
        ws.xz.get(cc, (size_t)R * sizeof(int));
        ws.xrs.get(cc, (size_t)R * sizeof(int));
        // experts [i0, i1) of the launch order on stack @a cs (offsets
        // @a og / @a ou / @a od, per expert of the chunk)
        auto runChunk = [&](const Tq2Stack &cs, unsigned int i0,
                            unsigned int i1, const std::vector<uint32_t> &og,
                            const std::vector<uint32_t> &ou,
                            const std::vector<uint32_t> &od) {
          const unsigned int n = i1 - i0;
          std::vector<int> c_off(n), c_cnt(n);
          unsigned int c_max = 0, rows = 0;
          for (unsigned int i = i0; i < i1; ++i) {
            const unsigned int a = ord[i];
            c_off[i - i0] = row_off[a];
            c_cnt[i - i0] = row_cnt[a];
            c_max = std::max(c_max, (unsigned int)row_cnt[a]);
            rows += row_cnt[a];
          }
          const unsigned int r0 = (unsigned int)row_off[ord[i0]];
          tq2QuantizeRowRange(cc, ws, xf, r0, rows, H);
          tq2GroupedGemm(cc, ws, cs, og, c_off, c_cnt, c_max, yg, I, H);
          tq2GroupedGemm(cc, ws, cs, ou, c_off, c_cnt, c_max, yu, I, H);
          tq2Glu(cc, yg, (size_t)r0 * I, yu, (size_t)r0 * I, act,
                 (size_t)rows * I, g.exact_gelu, (size_t)r0 * I);
          tq2QuantizeRowRange(cc, ws, act, r0, rows, I);
          tq2GroupedGemm(cc, ws, cs, od, c_off, c_cnt, c_max, yd, H, I);
        };
        if (stream) {
          // the lent partitions of this layer's block: expert e in slot
          // e % C of part e / C; one chunk per part, in read order
          const unsigned int x = fsu->layer_index % 2;
          tq2StreamIssue(cc, *fsu, st, E);
          Tq2ExpertArena::StreamSet &S = arena.streamSet(x);
          const unsigned int C = part->slots;
          unsigned int i = 0;
          for (size_t j = 0; j < S.parts.size(); ++j) {
            arena.streamWait(x, j, fsu->fadvise_dontneed);
            const unsigned int i0 = i;
            while (i < k && g.active[ord[i]] < (j + 1) * C)
              ++i;
            if (i == i0)
              continue;
            Tq2ExpertArena::Partition &p = *S.parts[j];
            Tq2Stack cs = st;
            cs.buf = p.buf;
            std::vector<uint32_t> og, ou, od;
            std::vector<unsigned int> used;
            for (unsigned int ii = i0; ii < i; ++ii) {
              const unsigned int sl = g.active[ord[ii]] - (unsigned int)j * C;
              const unsigned int ex = g.active[ord[ii]];
              const size_t base = (size_t)sl * p.slot_bytes +
                                  (ex < st.shift.size() ? st.shift[ex] : 0);
              og.push_back((uint32_t)tq2Words(base));
              ou.push_back((uint32_t)tq2Words(base + st.gu_bytes));
              od.push_back((uint32_t)tq2Words(base + 2 * st.gu_bytes));
              used.push_back(sl);
            }
            runChunk(cs, i0, i, og, ou, od);
            arena.markUse(cc, p, used);
          }
          NNTR_THROW_IF(i != k, std::logic_error)
            << "tq2cx_moe_experts_cl: " << (k - i)
            << " active experts outside the streamed block";
        } else {
          // read in chunks of at most the layer's slots, launch order
          for (unsigned int i0 = 0; i0 < k; i0 += part->slots) {
            const unsigned int i1 = std::min(k, i0 + part->slots);
            std::vector<unsigned int> experts;
            for (unsigned int i = i0; i < i1; ++i)
              experts.push_back(g.active[ord[i]]);
            const std::vector<unsigned int> used = fsuAcquire(experts);
            std::vector<uint32_t> og, ou, od;
            for (unsigned int e : experts) {
              og.push_back(st.off_gate[e]);
              ou.push_back(st.off_up[e]);
              od.push_back(st.off_down[e]);
            }
            runChunk(st, i0, i1, og, ou, od);
            if (evicting)
              arena.markUse(cc, *part, used);
          }
        }
      }
    }
    d_src = tq2Small(cc, src_row.data(), src_row.size() * sizeof(int));
    d_w = tq2Small(cc, weight.data(), weight.size() * sizeof(float));
    d_cnt = tq2Small(cc, counts.data(), counts.size() * sizeof(int));
    own_tables = true;
  }

  // ---- weighted combine into the output rows (ascending expert id) ----
  const bool ok = tq2Combine(cc, fp16, yd, out, d_src, d_w, d_cnt, H, max_k, T);
  if (own_tables) {
    opencl::clReleaseMemObjectT(d_src);
    opencl::clReleaseMemObjectT(d_w);
    opencl::clReleaseMemObjectT(d_cnt);
  }
  if (in.svm)
    q.enqueueSVMMap(in.svm, in.bytes, /*read_only=*/true, nullptr,
                    /*async=*/true);
  if (out.svm)
    q.enqueueSVMMap(out.svm, out.bytes, /*read_only=*/true, nullptr,
                    /*async=*/true);
  if (rin.svm && rin.svm != in.svm)
    q.enqueueSVMMap(rin.svm, rin.bytes, /*read_only=*/true, nullptr,
                    /*async=*/true);
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_experts_cl: combine arguments";

  // a short prefill: the next layer's predicted experts are read into its
  // slots while this layer's kernels run
  if (ahead)
    tq2PrefillAheadIssue(cc, *fsu, st);
  return true;
}

bool tq2cx_moe_route_cl(const MoeRouter &rt, unsigned int tokens, uint32_t *ids,
                        float *weights) {
  using DT = ml::train::TensorDim::DataType;
  if (rt.input == nullptr || rt.weight == nullptr || rt.scale == nullptr ||
      rt.per_expert_scale == nullptr || tokens == 0 || ids == nullptr ||
      weights == nullptr)
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (cc == nullptr)
    return false;
  const DT dt = rt.input->getDataType();
  if (dt != DT::FP32 && dt != DT::FP16)
    return false;
  const bool fp16 = dt == DT::FP16;
  const unsigned int H = rt.input->width();
  const unsigned int E = rt.per_expert_scale->getDim().getDataLen();
  const unsigned int K = rt.topk;
  const size_t in_rows =
    (size_t)rt.input->batch() * rt.input->channel() * rt.input->height();
  if (K == 0 || K > kTq2RouteMaxK || K > E || E == 0 || H == 0 || H % 8 != 0 ||
      in_rows < tokens || rt.weight->getDim().getDataLen() != (size_t)H * E)
    return false;
  float *r_w = tq2HostFloat(rt.weight);
  float *r_scale = tq2HostFloat(rt.scale);
  float *r_pes = tq2HostFloat(rt.per_expert_scale);
  float *r_norm = tq2HostFloat(rt.norm);
  if (r_w == nullptr || r_scale == nullptr || r_pes == nullptr ||
      (rt.norm != nullptr && r_norm == nullptr))
    return false;
  Tq2Act rin;
  if (!tq2Act(*rt.input, (size_t)tokens * H * (fp16 ? 2 : 4), rin))
    return false;

  std::lock_guard<std::mutex> lk(tq2Mutex());
  Tq2Workspace &ws = tq2Workspace();
  auto &q = cc->command_queue_inst_;
  // strict math (no relaxed-math defaults): see the decode router
  const std::string ropts = std::string("-cl-std=CL3.0 -DNNTR_STRICT_FP") +
                            (tq2RouterHwFma(cc) ? "" : " -DRT_SOFT_FMA");
  cl_mem d_rw16 = tq2RouterWeightBf16(cc, r_w, (size_t)H * E);
  auto sk =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_sumsq_rows", ropts);
  auto yk =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_y_rows", ropts);
  auto lk_ =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_seg_rows",
                         d_rw16 != nullptr ? ropts + " -DRT_W_BF16" : ropts);
  auto fk =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_fold_rows", ropts);
  const size_t elws = (E + 15) / 16 * 16;
  const bool whole =
    tq2RouterHwFma(cc) && (size_t)E * ((tokens + 3) / 4) >= kTq2RouteChainLanes;
  auto wk =
    whole
      ? cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_chain_rows",
                             d_rw16 != nullptr ? ropts + " -DRT_W_BF16" : ropts)
      : ClContext::SharedPtrClKernel();
  auto xk =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_exp_rows", ropts);
  auto pk =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_prob_rows", ropts);
  auto rk =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_rank_rows", ropts);
  auto nk_ =
    cc->registerClKernel(tq2moe_route_kernel, "tq2moe_route_fin_rows", ropts);
  NNTR_THROW_IF(!sk || !yk || !lk_ || !fk || !xk || !pk || !rk || !nk_ ||
                  (whole && !wk),
                std::runtime_error)
    << "tq2cx_moe_route_cl: the device router did not build";
  cl_mem d_rw =
    d_rw16 != nullptr ? d_rw16 : tq2RouterOperand(cc, r_w, (size_t)H * E);
  cl_mem d_scale = tq2RouterOperand(cc, r_scale, H);
  cl_mem d_pes = tq2RouterOperand(cc, r_pes, E);
  cl_mem d_norm = r_norm != nullptr ? tq2RouterOperand(cc, r_norm, H) : d_scale;
  const unsigned int nseg = (H + kTq2RouteSeg - 1) / kTq2RouteSeg;
  const unsigned int chunk = std::min(tokens, kTq2RoutePartRows);
  cl_mem acc = ws.r_acc.get(cc, (size_t)tokens * 32 * sizeof(float));
  cl_mem yrows = ws.r_yrows.get(cc, (size_t)tokens * H * sizeof(float));
  cl_mem logits = ws.r_logits.get(cc, (size_t)tokens * E * sizeof(float));
  cl_mem ex = ws.r_ex.get(cc, (size_t)tokens * E * sizeof(float));
  cl_mem part =
    whole ? nullptr
          : ws.r_ppart.get(cc, (size_t)chunk * nseg * E * sizeof(float));
  cl_mem out = ws.r_out.get(cc, (size_t)tokens * 2 * K * sizeof(uint32_t));
  const int x_half = fp16 ? 1 : 0, has_norm = r_norm != nullptr ? 1 : 0;
  const float hscale = 1.0f / std::sqrt(static_cast<float>(H));
  const float eps = rt.epsilon;
  const int Hi = (int)H, Ei = (int)E, Ki = (int)K, Ti = (int)tokens,
            nsegi = (int)nseg;
  // the unused half / float input slot is bound to the same operand; the
  // probabilities overwrite the logits (the exponentials read them first)
  bool ok = rin.bind(sk, 0) && rin.bind(sk, 1) &&
            sk->SetKernelArguments(2, &x_half, sizeof(int)) &&
            sk->SetKernelArguments(3, &Hi, sizeof(int)) &&
            sk->SetKernelArguments(4, &Ti, sizeof(int)) &&
            sk->SetKernelArguments(5, &acc, sizeof(cl_mem)) &&
            rin.bind(yk, 0) && rin.bind(yk, 1) &&
            yk->SetKernelArguments(2, &x_half, sizeof(int)) &&
            yk->SetKernelArguments(3, &acc, sizeof(cl_mem)) &&
            yk->SetKernelArguments(4, &d_norm, sizeof(cl_mem)) &&
            yk->SetKernelArguments(5, &has_norm, sizeof(int)) &&
            yk->SetKernelArguments(6, &d_scale, sizeof(cl_mem)) &&
            yk->SetKernelArguments(7, &hscale, sizeof(float)) &&
            yk->SetKernelArguments(8, &eps, sizeof(float)) &&
            yk->SetKernelArguments(9, &Hi, sizeof(int)) &&
            yk->SetKernelArguments(10, &Ti, sizeof(int)) &&
            yk->SetKernelArguments(11, &yrows, sizeof(cl_mem)) &&
            lk_->SetKernelArguments(0, &yrows, sizeof(cl_mem)) &&
            lk_->SetKernelArguments(1, &d_rw, sizeof(cl_mem)) &&
            lk_->SetKernelArguments(2, &Hi, sizeof(int)) &&
            lk_->SetKernelArguments(3, &Ei, sizeof(int)) &&
            lk_->SetKernelArguments(6, &nsegi, sizeof(int)) &&
            lk_->SetKernelArguments(7, &part, sizeof(cl_mem)) &&
            fk->SetKernelArguments(0, &part, sizeof(cl_mem)) &&
            fk->SetKernelArguments(1, &Ei, sizeof(int)) &&
            fk->SetKernelArguments(4, &nsegi, sizeof(int)) &&
            fk->SetKernelArguments(5, &logits, sizeof(cl_mem)) &&
            xk->SetKernelArguments(0, &logits, sizeof(cl_mem)) &&
            xk->SetKernelArguments(1, &Ei, sizeof(int)) &&
            xk->SetKernelArguments(2, &Ti, sizeof(int)) &&
            xk->SetKernelArguments(3, &ex, sizeof(cl_mem)) &&
            pk->SetKernelArguments(0, &ex, sizeof(cl_mem)) &&
            pk->SetKernelArguments(1, &Ei, sizeof(int)) &&
            pk->SetKernelArguments(2, &Ti, sizeof(int)) &&
            pk->SetKernelArguments(3, &logits, sizeof(cl_mem)) &&
            rk->SetKernelArguments(0, &logits, sizeof(cl_mem)) &&
            rk->SetKernelArguments(1, &Ei, sizeof(int)) &&
            rk->SetKernelArguments(2, &Ki, sizeof(int)) &&
            rk->SetKernelArguments(3, &Ti, sizeof(int)) &&
            rk->SetKernelArguments(4, &out, sizeof(cl_mem)) &&
            nk_->SetKernelArguments(0, &d_pes, sizeof(cl_mem)) &&
            nk_->SetKernelArguments(1, &Ki, sizeof(int)) &&
            nk_->SetKernelArguments(2, &Ti, sizeof(int)) &&
            nk_->SetKernelArguments(3, &out, sizeof(cl_mem));
  NNTR_THROW_IF(!ok, std::runtime_error)
    << "tq2cx_moe_route_cl: device router arguments";

  // a shared-plane input is handed to the device for the call
  if (rin.svm)
    q.enqueueSVMUnmap(rin.svm);
  // lane sums: 32 lanes per row, two rows per group
  tq2Dispatch(cc, sk, 2, {32, ((size_t)tokens + 1) / 2 * 2, 1}, {32, 2, 1});
  // the scaled rows: four elements per lane, 64-lane groups
  tq2Dispatch(cc, yk, 2, {((size_t)H / 4 + 63) / 64 * 64, (size_t)tokens, 1},
              {64, 1, 1});
  // logits: enough rows on a hardware fma, a lane per expert and four rows
  // over the whole hidden axis; else, in chunks of rows, segment partials (a
  // lane per expert, segment and eight rows), then their sums in segment
  // order (a lane per expert and row). The same chains and sums either way.
  if (whole) {
    const int zero = 0;
    NNTR_THROW_IF(!(wk->SetKernelArguments(0, &yrows, sizeof(cl_mem)) &&
                    wk->SetKernelArguments(1, &d_rw, sizeof(cl_mem)) &&
                    wk->SetKernelArguments(2, &Hi, sizeof(int)) &&
                    wk->SetKernelArguments(3, &Ei, sizeof(int)) &&
                    wk->SetKernelArguments(4, &zero, sizeof(int)) &&
                    wk->SetKernelArguments(5, &Ti, sizeof(int)) &&
                    wk->SetKernelArguments(6, &logits, sizeof(cl_mem))),
                  std::runtime_error)
      << "tq2cx_moe_route_cl: device router chain arguments";
    tq2Dispatch(cc, wk, 2, {elws, ((size_t)tokens + 3) / 4, 1}, {elws, 1, 1});
  }
  for (unsigned int r0 = 0; !whole && r0 < tokens; r0 += chunk) {
    const int r0i = (int)r0, ni = (int)std::min(chunk, tokens - r0);
    NNTR_THROW_IF(!(lk_->SetKernelArguments(4, &r0i, sizeof(int)) &&
                    lk_->SetKernelArguments(5, &ni, sizeof(int)) &&
                    fk->SetKernelArguments(2, &r0i, sizeof(int)) &&
                    fk->SetKernelArguments(3, &ni, sizeof(int))),
                  std::runtime_error)
      << "tq2cx_moe_route_cl: device router chunk arguments";
    tq2Dispatch(cc, lk_, 3, {elws, (size_t)nseg, ((size_t)ni + 7) / 8},
                {elws, 1, 1});
    tq2Dispatch(cc, fk, 2, {elws, (size_t)ni, 1}, {elws, 1, 1});
  }
  // softmax and selection: a lane per expert and row, then per row
  tq2Dispatch(cc, xk, 2, {elws, (size_t)tokens, 1}, {elws, 1, 1});
  tq2Dispatch(cc, pk, 2, {elws, (size_t)tokens, 1}, {elws, 1, 1});
  tq2Dispatch(cc, rk, 2, {elws, (size_t)tokens, 1}, {elws, 1, 1});
  tq2Dispatch(cc, nk_, 1, {((size_t)tokens + 63) / 64 * 64, 1, 1}, {64, 1, 1});
  if (rin.svm)
    q.enqueueSVMMap(rin.svm, rin.bytes, /*read_only=*/true, nullptr,
                    /*async=*/true);
  // the one read back: every row's ids and weights (it waits for the queue)
  std::vector<uint32_t> host((size_t)tokens * 2 * K);
  const cl_int err = opencl::clEnqueueReadBuffer(
    q.GetCommandQueue(), out, CL_TRUE, 0, host.size() * sizeof(uint32_t),
    host.data(), 0, nullptr, nullptr);
  NNTR_THROW_IF(err != CL_SUCCESS, std::runtime_error)
    << "tq2cx_moe_route_cl: routing readback failed (" << err << ")";
  for (size_t t = 0; t < tokens; ++t) {
    const uint32_t *o = host.data() + t * 2 * K;
    for (unsigned int j = 0; j < K; ++j) {
      ids[t * K + j] = o[j];
      std::memcpy(weights + t * K + j, o + K + j, sizeof(float));
    }
  }
  return true;
}

} // namespace nntrainer
