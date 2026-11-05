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

#include <blas_kernels.h>
#include <cl_context.h>
#include <cl_kernels/int8_int4_gemm_v8c.h>
#include <cl_kernels/tq2cx_gemm.h>
#include <cl_kernels/tq2cx_gemm_portable.h>
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
 * The decode router on the device, three launches, the deterministic router
 * of moe_route_det.h term for term (only IEEE add / multiply / fma and exact
 * operations in a fixed order), so it selects the same experts with the same
 * weights as the host router on any device.
 *
 * tq2moe_route_norm: one work-group: the rms scale of the row, then the
 * normalised, scaled row y[H] the projection reads.
 * tq2moe_route_logits: one lane per (expert, RT_SEG-wide segment of the
 * hidden axis): partial[seg][e] = the fma chain over the segment.
 * tq2moe_route_select: one work-group of E lanes: logits = partial sums in
 * segment order, softmax, top-k by rank, renormalisation, per-expert scale
 * (tq2moe_route_select_ref: the same in its one-lane reference form);
 * out ids[K] / w[K] in ascending expert id (the host combine order) and the
 * selected experts' record offsets for the batched GEMVs (off_gu = gate
 * offsets then up offsets, off_d = down offsets).
 *
 * Built with -DRT_SOFT_FMA on a device without a hardware fp32 fma
 * (DeviceCaps::fp32_fma), where fma() is a slow library routine: rt_fma then
 * builds the same correctly rounded result from multiplies and adds.
 *
 * tq2moe_route_logits built with -DRT_W_BF16 reads the router weight as the
 * high 16 bits of each fp32 value (tq2RouterWeightBf16: only when every low
 * half is zero), widened back exactly: the same fp32 operands, half the bytes.
 */
const std::string kTq2Route = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
#define RT_SEG 32
#define RT_MAXH 4096
#define RT_MAXE 256
#define RT_MAXK 16

// fma(a, b, c), correctly rounded, from fp32 multiplies and adds only (the
// Dekker product and two-sum are exact, the low parts are added rounded to
// odd, then the high part rounded to nearest: Boldo & Melquiond's emulated
// fma). Outside the range where those steps are exact (huge operands, a
// product below 2^-100) it calls fma().
inline float rt_fma(const float a, const float b, const float c) {
#ifdef RT_SOFT_FMA
  const float ph = a * b;
  const float m = fabs(ph);
  if (fabs(a) <= 0x1p100f && fabs(b) <= 0x1p100f && fabs(c) <= 0x1p120f &&
      m <= 0x1p100f && (m >= 0x1p-100f || a == 0.0f || b == 0.0f)) {
    const float ta = 4097.0f * a, tb = 4097.0f * b;
    const float ah = ta - (ta - a), al = a - ah;
    const float bh = tb - (tb - b), bl = b - bh;
    const float pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl;
    const float th = c + ph;
    const float tz = th - c;
    const float tl = (c - (th - tz)) + (ph - tz);
    const float v = tl + pl;
    if (v == 0.0f)
      return th;
    const float vz = v - tl;
    const float ve = (tl - (v - vz)) + (pl - vz);
    uint vu = as_uint(v);
    if (ve != 0.0f && (vu & 1u) == 0u)
      vu = ((ve > 0.0f) == (v > 0.0f)) ? vu + 1u : vu - 1u;
    return th + as_float(vu);
  }
#endif
  return fma(a, b, c);
}

// The deterministic primitives of moe_route_det.h, term for term: only IEEE
// add / multiply / fma and exact operations, so the host and every device
// produce the same bits whatever their division / sqrt / exp accuracy.
inline float rt_div(const float x, const float s) {
  float y = 1.0f / s;
  float e = rt_fma(-s, y, 1.0f);
  y = rt_fma(e, y, y);
  e = rt_fma(-s, y, 1.0f);
  y = rt_fma(e, y, y);
  const float q = x * y;
  const float r = rt_fma(-s, q, x);
  return r == 0.0f ? q : rt_fma(r, y, q);
}
inline float rt_rsqrt(const float a) {
  float y = as_float(0x5f375a86u - (as_uint(a) >> 1));
  const float h = 0.5f * a;
  for (int it = 0; it < 3; ++it) {
    const float hy = h * y;
    const float r = rt_fma(-hy, y, 1.5f);
    y = y * r;
  }
  return y;
}
inline float rt_exp(const float x) {
  if (x < -87.0f)
    return 0.0f;
  const float t = x * 1.44269504088896341f;
  const float n = rint(t);
  float r = rt_fma(-n, 0.693145751953125f, x);
  r = rt_fma(-n, 1.428606765330187e-06f, r);
  float p = 1.3888889e-3f;
  p = rt_fma(p, r, 8.3333333e-3f);
  p = rt_fma(p, r, 4.1666667e-2f);
  p = rt_fma(p, r, 1.6666667e-1f);
  p = rt_fma(p, r, 0.5f);
  p = rt_fma(p, r, 1.0f);
  p = rt_fma(p, r, 1.0f);
  return ldexp(p, (int)n);
}

__kernel void tq2moe_route_norm(__global const half *xh,
                                __global const float *xf, const int x_half,
                                __global const float *norm,
                                const int has_norm,
                                __global const float *rscale,
                                const float hscale, const float eps,
                                const int H, __global float *y) {
  __local float lx[RT_MAXH];
  __local float lacc[32];
  __local float lscale;
  const int tid = get_local_id(0);
  const int lws = get_local_size(0);
  for (int i = tid; 4 * i < H; i += lws) // H % 8 == 0
    vstore4(x_half ? vload_half4(i, xh) : vload4(i, xf), i, lx);
  barrier(CLK_LOCAL_MEM_FENCE);
  // sum of squares in the host order: lane l of 32 sums x*x over i = l,
  // l + 32, ... with fma; the 8-wide tail goes to lanes 0..7. The square of
  // an fp16 value is exact in fp32 (22 bits), so for a half row x*x + acc is
  // the same bits as fma(x, x, acc) without the fma.
  if (tid < 32) {
    const int n32 = H / 32 * 32;
    float acc = 0.0f;
    if (x_half) {
      for (int i = tid; i < n32; i += 32)
        acc = lx[i] * lx[i] + acc;
      if (tid < 8)
        for (int i = n32 + tid; i + 8 - tid <= H; i += 8)
          acc = lx[i] * lx[i] + acc;
    } else {
      for (int i = tid; i < n32; i += 32)
        acc = rt_fma(lx[i], lx[i], acc);
      if (tid < 8)
        for (int i = n32 + tid; i + 8 - tid <= H; i += 8)
          acc = rt_fma(lx[i], lx[i], acc);
    }
    lacc[tid] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    float h[4];
    for (int j = 0; j < 4; ++j) {
      __local const float *v = lacc + 8 * j;
      h[j] = ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]));
    }
    const float sumsq = ((h[0] + h[1]) + h[2]) + h[3];
    const float mean = rt_div(sumsq, (float)H);
    lscale = rt_rsqrt(mean + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale = lscale;
  for (int i = tid; i < H; i += lws) {
    float v = lx[i] * scale;
    if (has_norm)
      v = v * norm[i];
    y[i] = v * (rscale[i] * hscale);
  }
}

#ifdef RT_W_BF16
#define RT_WT ushort
#define RT_WLOAD(i) as_float(((uint)W[i]) << 16)
#else
#define RT_WT float
#define RT_WLOAD(i) W[i]
#endif

__kernel void tq2moe_route_logits(__global const float *y,
                                  __global const RT_WT *W, const int H,
                                  const int E, __global float *partial) {
  const int e = get_global_id(0);
  const int seg = get_global_id(1);
  if (e >= E)
    return;
  const int f0 = seg * RT_SEG;
  const int n = min(RT_SEG, H - f0);
  float acc = 0.0f;
#pragma unroll 8
  for (int f = 0; f < n; ++f)
    acc = rt_fma(y[f0 + f], RT_WLOAD((long)(f0 + f) * E + e), acc);
  partial[(long)seg * E + e] = acc;
}

__kernel void tq2moe_route_select_ref(__global const float *partial,
                                  const int nseg, __global const float *pes,
                                  __global const uint *off_tab, const int E,
                                  const int K, __global uint *ids,
                                  __global float *wout, __global uint *off_gu,
                                  __global uint *off_d) {
  __local float lp[RT_MAXE];
  __local float lred;
  __local int rank_id[RT_MAXK];
  __local float rank_p[RT_MAXK];
  const int e = get_local_id(0);
  float l = 0.0f;
  if (e < E) {
    l = partial[e];
    for (int s = 1; s < nseg; ++s)
      l = l + partial[(long)s * E + e];
    lp[e] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    float mx = lp[0];
    for (int i = 1; i < E; ++i)
      mx = fmax(mx, lp[i]);
    lred = mx;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float ex = e < E ? rt_exp(l - lred) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E)
    lp[e] = ex;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    float sum = 0.0f;
    for (int i = 0; i < E; ++i)
      sum = sum + lp[i];
    lred = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float p = e < E ? rt_div(ex, lred) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E)
    lp[e] = p;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E) {
    // rank among the experts: larger first, ties to the lower id
    int rank = 0;
    for (int i = 0; i < E; ++i)
      rank += (lp[i] > p || (lp[i] == p && i < e)) ? 1 : 0;
    if (rank < K) {
      rank_id[rank] = e;
      rank_p[rank] = p;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    int sel[RT_MAXK];
    float val[RT_MAXK];
    float s = 0.0f;
    for (int j = 0; j < K; ++j) {
      sel[j] = rank_id[j];
      val[j] = rank_p[j];
      s = s + val[j];
    }
    for (int j = 0; j < K; ++j)
      val[j] = rt_div(val[j], s) * pes[sel[j]];
    // ascending expert id: the order the host combine sums in
    for (int a = 1; a < K; ++a)
      for (int b = a; b > 0 && sel[b - 1] > sel[b]; --b) {
        const int ti = sel[b]; sel[b] = sel[b - 1]; sel[b - 1] = ti;
        const float tv = val[b]; val[b] = val[b - 1]; val[b - 1] = tv;
      }
    for (int a = 0; a < K; ++a) {
      ids[a] = (uint)sel[a];
      wout[a] = val[a];
      off_gu[a] = off_tab[sel[a]];
      off_gu[K + a] = off_tab[E + sel[a]];
      off_d[a] = off_tab[2 * E + sel[a]];
    }
  }
}

// The same selection as tq2moe_route_select_ref, bit for bit, with less
// latency: the segment loads are issued eight ahead (the adds keep the
// segment order), the max is a two-level reduction (fmax is exact in any
// order), every lane runs the softmax sum itself (the same strict
// i = 0..E-1 chain, read four at a time from local memory, so no barrier
// waits on one lane), the rank reads four probabilities per local load and
// the K selected experts are renormalised and put in ascending id order by K
// lanes at once (no per-lane arrays).
__kernel void tq2moe_route_select(__global const float *partial,
                                  const int nseg, __global const float *pes,
                                  __global const uint *off_tab, const int E,
                                  const int K, __global uint *ids,
                                  __global float *wout, __global uint *off_gu,
                                  __global uint *off_d) {
  __local float4 lp4[RT_MAXE / 4];
  __local float lm[16];
  __local int rank_id[RT_MAXK];
  __local float rank_p[RT_MAXK];
  __local float *lp = (__local float *)lp4;
  const int e = get_local_id(0);
  const int lws = get_local_size(0); // a multiple of 16, >= E
  float l = 0.0f;
  if (e < E) {
    __global const float *pe = partial + e;
    l = pe[0];
    int s = 1;
    for (; s + 8 <= nseg; s += 8) {
      const float v0 = pe[(long)(s + 0) * E], v1 = pe[(long)(s + 1) * E];
      const float v2 = pe[(long)(s + 2) * E], v3 = pe[(long)(s + 3) * E];
      const float v4 = pe[(long)(s + 4) * E], v5 = pe[(long)(s + 5) * E];
      const float v6 = pe[(long)(s + 6) * E], v7 = pe[(long)(s + 7) * E];
      l = l + v0;
      l = l + v1;
      l = l + v2;
      l = l + v3;
      l = l + v4;
      l = l + v5;
      l = l + v6;
      l = l + v7;
    }
    for (; s < nseg; ++s)
      l = l + pe[(long)s * E];
  }
  lp[e] = e < E ? l : -INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);
  // max: 16 lanes take lws / 16 values each, then every lane the 16
  if (e < 16) {
    const int g = lws / 16;
    float m = lp[e * g];
    for (int i = 1; i < g; ++i)
      m = fmax(m, lp[e * g + i]);
    lm[e] = m;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = lm[0];
  for (int i = 1; i < 16; ++i)
    mx = fmax(mx, lm[i]);
  const float ex = e < E ? rt_exp(l - mx) : 0.0f;
  lp[e] = ex; // lp was last read before the barrier above
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  {
    const int e4 = E / 4;
    for (int i = 0; i < e4; ++i) {
      const float4 v = lp4[i];
      sum = sum + v.s0;
      sum = sum + v.s1;
      sum = sum + v.s2;
      sum = sum + v.s3;
    }
    for (int i = 4 * e4; i < E; ++i)
      sum = sum + lp[i];
  }
  const float p = e < E ? rt_div(ex, sum) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  lp[e] = e < E ? p : -INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E) {
    // rank among the experts: larger first, ties to the lower id (the
    // padding is -inf and never counts)
    int rank = 0;
    const int n4 = (E + 3) / 4;
    for (int i = 0; i < n4; ++i) {
      const float4 v = lp4[i];
      const int b = 4 * i;
      rank += (v.s0 > p || (v.s0 == p && b < e)) ? 1 : 0;
      rank += (v.s1 > p || (v.s1 == p && b + 1 < e)) ? 1 : 0;
      rank += (v.s2 > p || (v.s2 == p && b + 2 < e)) ? 1 : 0;
      rank += (v.s3 > p || (v.s3 == p && b + 3 < e)) ? 1 : 0;
    }
    if (rank < K) {
      rank_id[rank] = e;
      rank_p[rank] = p;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < K) {
    float s = 0.0f;
    for (int j = 0; j < K; ++j)
      s = s + rank_p[j];
    const int id = rank_id[e];
    const float v = rt_div(rank_p[e], s) * pes[id];
    // position in ascending expert id (the ids are distinct)
    int pos = 0;
    for (int j = 0; j < K; ++j)
      pos += rank_id[j] < id ? 1 : 0;
    ids[pos] = (uint)id;
    wout[pos] = v;
    off_gu[pos] = off_tab[id];
    off_gu[K + pos] = off_tab[E + id];
    off_d[pos] = off_tab[2 * E + id];
  }
}
)CL";
constexpr unsigned int kTq2RouteSeg = 32; /**< RT_SEG */
static_assert(kTq2RouteSeg == kMoeRouteSeg,
              "the device router segments as moe_route_det.h does");
constexpr unsigned int kTq2RouteMaxH = 4096; /**< RT_MAXH */
constexpr unsigned int kTq2RouteMaxE = 256, kTq2RouteMaxK = 16;

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
  Tq2Scratch xrs; /**< per-row sum (quantizer output, unused here) */
  Tq2Scratch yg;  /**< gate [R][I] (decode: gate and up, [2k][I]) */
  Tq2Scratch yu;  /**< up [R][I] */
  Tq2Scratch act; /**< GeGLU [R][I] */
  Tq2Scratch yd;  /**< down [R][H] */
  // device routing of a decode row (tq2moe_route_logits / _select)
  Tq2Scratch r_ids;         /**< [K] selected expert ids, ascending */
  Tq2Scratch r_w;           /**< [K] their routing weights */
  Tq2Scratch r_off_gu;      /**< [2K] gate then up record offsets */
  Tq2Scratch r_off_d;       /**< [K] down record offsets */
  Tq2Scratch r_part;        /**< [segments][E] partial router logits */
  Tq2Scratch r_y;           /**< [H] normalised, scaled router row */
  unsigned int const_k = 0; /**< K the constants below were built for */
  cl_mem c_zero = nullptr;  /**< [2K] zeros (the one activation row) */
  cl_mem c_iota = nullptr;  /**< [K] 0..K-1 */
  cl_mem c_count = nullptr; /**< [1] K */
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

/** quantize M rows of K floats from @a x into the workspace int8 rows; with
 *  @a perm each row is stored in TQ2 order (V8C_AQ_PERM, K % 32 == 0) */
void tq2Quantize(ClContext *cc, Tq2Workspace &ws, cl_mem x, void *x_svm,
                 unsigned int M, unsigned int K, bool fp16 = false,
                 bool perm = false) {
  auto kp = cc->registerClKernel(
    int8_int4_gemm_v8c_kernel,
    fp16 ? "v8c_act_quant_f16_par" : "v8c_act_quant_f32_par",
    perm ? tq2QuantOptions(cc) + " -DV8C_AQ_PERM" : tq2QuantOptions(cc));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: activation quantizer did not build";
  cl_mem xq = ws.xq.get(cc, (size_t)M * K);
  cl_mem xs = ws.xs.get(cc, (size_t)M * sizeof(float));
  cl_mem xz = ws.xz.get(cc, (size_t)M * sizeof(int));
  cl_mem xr = ws.xrs.get(cc, (size_t)M * sizeof(int));
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

/** the decode GEMVs read activation rows in TQ2 order (tq2GemvGeom): the
 *  quantizer feeding tq2GemvBatchDev must then store that order */
bool tq2GemvPermAct(ClContext *cc) { return tq2Dpas(cc) && tq2GemvGeom().perm; }

/** quantize the decode GEMV's activation rows in the order it reads them */
void tq2QuantizeGemv(ClContext *cc, Tq2Workspace &ws, cl_mem x, void *x_svm,
                     unsigned int M, unsigned int K, bool fp16 = false) {
  tq2Quantize(cc, ws, x, x_svm, M, K, fp16, tq2GemvPermAct(cc));
}

/** the batched GEMV with the record offsets and activation rows on the
 *  device (Intel subgroup kernel on DPAS devices, the portable one else) */
void tq2GemvBatchDev(ClContext *cc, Tq2Workspace &ws, const Tq2Stack &st,
                     cl_mem d_off, cl_mem d_ar, unsigned int B, cl_mem Y,
                     unsigned int N, unsigned int K) {
  const bool portable = !tq2Dpas(cc);
  const Tq2GemvGeom &geom = tq2GemvGeom();
  auto kp = portable
              ? cc->registerClKernel(tq2cx_gemm_portable_kernel,
                                     "tq2cx_gemv_batch_portable", tq2Options())
              : cc->registerClKernel(tq2cx_gemm_kernel, "tq2cx_gemv_batch",
                                     tq2Options(geom.options()));
  NNTR_THROW_IF(!kp, std::runtime_error)
    << "tq2cx_moe_experts_cl: the batched GEMV did not build";
  cl_mem xq = ws.xq.buf, xs = ws.xs.buf, xz = ws.xz.buf;
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
                  unsigned int K) {
  const unsigned int B = (unsigned int)rec_off.size();
  cl_mem d_off = tq2Small(cc, rec_off.data(), B * sizeof(uint32_t));
  cl_mem d_ar = tq2Small(cc, act_row.data(), B * sizeof(int));
  tq2GemvBatchDev(cc, ws, st, d_off, d_ar, B, Y, N, K);
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
    tq2RouteConstants(cc, ws, K);
    cl_mem ids = ws.r_ids.get(cc, K * sizeof(uint32_t));
    cl_mem wts = ws.r_w.get(cc, K * sizeof(float));
    cl_mem off_gu = ws.r_off_gu.get(cc, 2 * K * sizeof(uint32_t));
    cl_mem off_d = ws.r_off_d.get(cc, K * sizeof(uint32_t));
    // the kernels use only IEEE add / multiply / fma (moe_route_det.h) and
    // must be built with strict math (no relaxed-math defaults) to keep them
    const std::string ropts = std::string("-cl-std=CL3.0 -DNNTR_STRICT_FP") +
                              (cc->caps().fp32_fma ? "" : " -DRT_SOFT_FMA");
    // the router weight as exact bf16 when every value allows it, else fp32
    cl_mem d_rw16 = tq2RouterWeightBf16(cc, r_w, (size_t)H * E);
    auto nk = cc->registerClKernel(kTq2Route, "tq2moe_route_norm", ropts);
    auto lk_ =
      cc->registerClKernel(kTq2Route, "tq2moe_route_logits",
                           d_rw16 != nullptr ? ropts + " -DRT_W_BF16" : ropts);
    auto sk = cc->registerClKernel(kTq2Route, tq2RouteSelectKernel(), ropts);
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
    tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
    tq2GemvBatchDev(cc, ws, st, off_gu, ws.c_zero, 2 * K, ygu, I, H);
    tq2Glu(cc, ygu, 0, ygu, (size_t)K * I, act, (size_t)K * I, g.exact_gelu);
    tq2QuantizeGemv(cc, ws, act, nullptr, K, I);
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
      tq2QuantizeGemv(cc, ws, in.mem, in.svm, 1, H, fp16);
      std::vector<uint32_t> off_gu(2 * k);
      std::vector<int> ar_gu(2 * k, 0);
      for (unsigned int a = 0; a < k; ++a) {
        off_gu[a] = st.off_gate[g.active[a]];
        off_gu[k + a] = st.off_up[g.active[a]];
      }
      cl_mem ygu = ws.yg.get(cc, (size_t)2 * k * I * sizeof(float));
      tq2GemvBatch(cc, ws, st, off_gu, ar_gu, ygu, I, H);
      cl_mem act = ws.act.get(cc, (size_t)k * I * sizeof(float));
      tq2Glu(cc, ygu, 0, ygu, (size_t)k * I, act, (size_t)k * I, g.exact_gelu);
      std::vector<uint32_t> off_d(k);
      std::vector<int> ar_d(k);
      for (unsigned int a = 0; a < k; ++a) {
        off_d[a] = st.off_down[g.active[a]];
        ar_d[a] = (int)a;
      }
      yd = ws.yd.get(cc, (size_t)k * H * sizeof(float));
      tq2QuantizeGemv(cc, ws, act, nullptr, k, I);
      tq2GemvBatch(cc, ws, st, off_d, ar_d, yd, H, I);
    } else {
      // ---- prefill: gather, grouped GEMMs over every expert's rows ----
      auto gk = cc->registerClKernel(kTq2MoeAux, fp16 ? "tq2moe_gather_f16"
                                                      : "tq2moe_gather_f32");
      NNTR_THROW_IF(!gk, std::runtime_error)
        << "tq2cx_moe_experts_cl: aux kernels did not build";
      cl_mem xf = ws.xf.get(cc, (size_t)R * H * sizeof(float));
      cl_mem d_gsrc = tq2Small(cc, gather_src.data(), R * sizeof(int));
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
      tq2Quantize(cc, ws, xf, nullptr, R, H);
      std::vector<uint32_t> off_g(k), off_u(k), off_d(k);
      for (unsigned int a = 0; a < k; ++a) {
        off_g[a] = st.off_gate[g.active[a]];
        off_u[a] = st.off_up[g.active[a]];
        off_d[a] = st.off_down[g.active[a]];
      }
      cl_mem yg = ws.yg.get(cc, (size_t)R * I * sizeof(float));
      cl_mem yu = ws.yu.get(cc, (size_t)R * I * sizeof(float));
      tq2GroupedGemm(cc, ws, st, off_g, row_off, row_cnt, max_rows, yg, I, H);
      tq2GroupedGemm(cc, ws, st, off_u, row_off, row_cnt, max_rows, yu, I, H);
      cl_mem act = ws.act.get(cc, (size_t)R * I * sizeof(float));
      tq2Glu(cc, yg, 0, yu, 0, act, (size_t)R * I, g.exact_gelu);
      tq2Quantize(cc, ws, act, nullptr, R, I);
      yd = ws.yd.get(cc, (size_t)R * H * sizeof(float));
      tq2GroupedGemm(cc, ws, st, off_d, row_off, row_cnt, max_rows, yd, H, I);
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

} // namespace nntrainer
