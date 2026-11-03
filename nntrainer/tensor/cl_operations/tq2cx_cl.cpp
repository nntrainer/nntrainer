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
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <cl_kernels/int8_int4_gemm_v8c.h>
#include <cl_kernels/tq2cx_gemm.h>
#include <cl_kernels/tq2cx_gemm_portable.h>
#include <cl_kernels/tq2moe_glu_quant.h>
#include <cl_kernels/tq2moe_route.h>
#include <engine.h>
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
                             const int gate_off, const int up_off) {
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
  out[i] = g * up[up_off + i];
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

/* the min / max tree, the row parameters and the row-sum tree of
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

/** the layer's stack; built here on first use (caller holds tq2Mutex) */
const Tq2Stack &tq2Stack(ClContext *cc, const MoeExpertGroup &g) {
  auto it = tq2Stacks().find(g.layer_key);
  if (it != tq2Stacks().end()) {
    const Tq2cxLayout kl = tq2cxStoredLayout();
    NNTR_THROW_IF(it->second->layout != kl, std::runtime_error)
      << "tq2cx_moe_experts_cl: pack layout "
      << tq2cxLayoutName(it->second->layout) << ", kernel layout "
      << tq2cxLayoutName(kl);
    return *it->second;
  }
  auto st = tq2BuildStack(cc, g);
  auto &ref = *st;
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
            size_t up_off, cl_mem out, size_t total, bool exact) {
  auto kp = cc->registerClKernel(kTq2MoeAux, "tq2moe_glu_f32");
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: aux kernels did not build";
  const int ti = (int)total, ex = exact ? 1 : 0;
  const int go = (int)gate_off, uo = (int)up_off;
  bool ok = kp->SetKernelArguments(0, &gate, sizeof(cl_mem)) &&
            kp->SetKernelArguments(1, &up, sizeof(cl_mem)) &&
            kp->SetKernelArguments(2, &out, sizeof(cl_mem)) &&
            kp->SetKernelArguments(3, &ti, sizeof(int)) &&
            kp->SetKernelArguments(4, &ex, sizeof(int)) &&
            kp->SetKernelArguments(5, &go, sizeof(int)) &&
            kp->SetKernelArguments(6, &uo, sizeof(int));
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

bool tq2cx_moe_experts_prebuild_cl(const MoeExpertGroup &g) {
  if (g.layer_key == nullptr ||
      (g.records != nullptr && g.records->size() != 3 * g.gate.size()))
    return false;
  ClContext *cc = tq2GroupContext(g);
  if (cc == nullptr) {
    auto *c =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    ml_logw("tq2cx_moe_experts_prebuild_cl: not served here (%s)",
            c ? c->caps().toString().c_str() : "no gpu context");
    return false;
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
    st = tq2BuildStack(cc, g);
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
  const Tq2Stack &st = tq2Stack(cc, g);
  Tq2Workspace &ws = tq2Workspace();
  auto &q = cc->command_queue_inst_;

  auto combine_kp = cc->registerClKernel(
    kTq2MoeAux, fp16 ? "tq2moe_combine_f16" : "tq2moe_combine_f32");
  NNTR_THROW_IF(!combine_kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: aux kernels did not build";

  // shared-plane operands are handed to the device for the call
  if (in.svm)
    q.enqueueSVMUnmap(in.svm);
  if (out.svm)
    q.enqueueSVMUnmap(out.svm);
  if (rin.svm && rin.svm != in.svm)
    q.enqueueSVMUnmap(rin.svm);

  cl_mem yd = nullptr;
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
    auto sk =
      cc->registerClKernel(tq2moe_route_kernel, tq2RouteSelectKernel(), ropts);
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

    cl_mem ygu = ws.yg.get(cc, (size_t)2 * K * I * sizeof(float));
    cl_mem act = ws.act.get(cc, (size_t)K * I * sizeof(float));
    yd = ws.yd.get(cc, (size_t)K * H * sizeof(float));
    if (in_q.xq == nullptr)
      tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
    tq2GemvBatchDev(cc, ws, st, off_gu, ws.c_zero, 2 * K, ygu, I, H, in_q);
    tq2GluQuantize(cc, ws, ygu, 0, ygu, (size_t)K * I, act, K, I, g.exact_gelu,
                   tq2GemvPermAct(cc));
    tq2GemvBatchDev(cc, ws, st, off_d, ws.c_iota, K, yd, H, I);
    d_src = ws.c_iota;
    d_w = wts;
    d_cnt = ws.c_count;
    max_k = K;
  } else {
    const unsigned int k = (unsigned int)g.active.size();
    const auto &asg = *g.assignments;
    // per-token contribution table, ascending expert id (the active list
    // order): source row of the down output and the routing weight
    std::vector<int> row_off(k), row_cnt(k);
    unsigned int R = 0, max_rows = 0;
    for (unsigned int a = 0; a < k; ++a) {
      row_off[a] = (int)R;
      row_cnt[a] = (int)asg[g.active[a]].size();
      R += row_cnt[a];
      max_rows = std::max(max_rows, (unsigned int)row_cnt[a]);
    }
    const bool tiles = T > 1 && !tq2Dpas(cc) && tq2TileCfg().flat;
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
    } else {
      // ---- prefill: gather, grouped GEMMs over every expert's rows ----
      // tq2cx_gemm_tiles_v2_portable reads the int8 rows in TQ2 order: the
      // two quantizers below then store that order
      int v2 = tq2V2Plan(cc, tiles, H, I, R);
      cl_mem d_gsrc = tq2Small(cc, gather_src.data(), R * sizeof(int));
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
        off_g[a] = st.off_gate[g.active[a]];
        off_u[a] = st.off_up[g.active[a]];
        off_d[a] = st.off_down[g.active[a]];
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
          tq2GroupedGemm(cc, ws, st, off, row_off, row_cnt, max_rows, Y, N, K);
          return;
        }
        if (tq2GroupedGemmTiles(cc, ws, st, *tl, off, Y, N, K, R, v2))
          return;
        v2 = -1;
        requant();
        tq2GroupedGemmTiles(cc, ws, st, *tl, off, Y, N, K, R, v2);
      };
      auto requant_in = [&]() { quantize_in(false); };
      cl_mem yg = ws.yg.get(cc, (size_t)R * I * sizeof(float));
      cl_mem yu = ws.yu.get(cc, (size_t)R * I * sizeof(float));
      gemm(off_g, yg, I, H, requant_in);
      gemm(off_u, yu, I, H, requant_in);
      opencl::clReleaseMemObjectT(d_gsrc);
      cl_mem act = ws.act.get(cc, (size_t)R * I * sizeof(float));
      if (tq2QuantFuse()) {
        tq2GluQuantize(cc, ws, yg, 0, yu, 0, act, R, I, g.exact_gelu, v2 >= 0);
      } else {
        tq2Glu(cc, yg, 0, yu, 0, act, (size_t)R * I, g.exact_gelu);
        tq2Quantize(cc, ws, act, nullptr, R, I, false, v2 >= 0);
      }
      yd = ws.yd.get(cc, (size_t)R * H * sizeof(float));
      // both GeGLU paths leave the activations in act
      gemm(off_d, yd, H, I,
           [&]() { tq2Quantize(cc, ws, act, nullptr, R, I, false, false); });
    }
    d_src = tq2Small(cc, src_row.data(), src_row.size() * sizeof(int));
    d_w = tq2Small(cc, weight.data(), weight.size() * sizeof(float));
    d_cnt = tq2Small(cc, counts.data(), counts.size() * sizeof(int));
    own_tables = true;
  }

  // ---- weighted combine into the output rows (ascending expert id) ----
  const int Hi = (int)H, mk = (int)max_k, tot = (int)((size_t)T * H);
  bool ok = combine_kp->SetKernelArguments(0, &yd, sizeof(cl_mem)) &&
            out.bind(combine_kp, 1) &&
            combine_kp->SetKernelArguments(2, &d_src, sizeof(cl_mem)) &&
            combine_kp->SetKernelArguments(3, &d_w, sizeof(cl_mem)) &&
            combine_kp->SetKernelArguments(4, &d_cnt, sizeof(cl_mem)) &&
            combine_kp->SetKernelArguments(5, &Hi, sizeof(int)) &&
            combine_kp->SetKernelArguments(6, &mk, sizeof(int)) &&
            combine_kp->SetKernelArguments(7, &tot, sizeof(int));
  if (ok)
    tq2Dispatch(cc, combine_kp, 1, {((size_t)tot + 63) / 64 * 64, 1, 1},
                {64, 1, 1});
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
