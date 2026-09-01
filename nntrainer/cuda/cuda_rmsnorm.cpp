// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rmsnorm.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RMSNorm op implementation (NVRTC kernel, validated math).
 */

#include "cuda_rmsnorm.h"

#include <cuda_common.h> // cuda_vec4_rows_ok
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_fc_qs4cx.h> // cuda_fc_qs4cx_fused_normq_enabled
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cstdlib>

namespace nntrainer::cuda {

// Post-op drain, skipped for a device-only (cudaMalloc) destination: host
// code cannot read it without a stream-ordered staging copy, so the sync-mode
// per-op drain is provably unnecessary -- and its WDDM submit+wait round-trip
// dominated the wide-row prefill norm cost (~0.65ms/call in the 1K lprof).
static inline void rms_maybe_finish(const void *out) {
  static const bool skip_dev_drain = []() {
    const char *e = std::getenv("NNTR_CUDA_DRAINSKIP_RMS");
    return e != nullptr && e[0] == '1';
  }();
  if (skip_dev_drain && out != nullptr && dev_only(out))
    return;
  StreamManager::Global().maybeFinish();
}

// One block per row; block-reduces the sum of squares in FP32; scales by
// rsqrt(mean+eps) and folds the raw gamma (no (1+gamma) bias). has_gamma=0
// skips the gamma read (gamma-free v_norm).
static const char *RMSNORM_FP32_SRC = R"CU(
extern "C" __global__ void rmsnorm_fp32(const float *x, const float *gamma,
                                        float *y, int width, float eps,
                                        int has_gamma) {
  int row = blockIdx.x;
  const float *xr = x + (size_t)row * width;
  float *yr = y + (size_t)row * width;
  __shared__ float sdata[256];
  float partial = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float v = xr[k];
    partial += v * v;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sdata[0] / (float)width + eps);
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float g = has_gamma ? gamma[k] : 1.0f;
    yr[k] = xr[k] * inv * g;
  }
}
)CU";

// FP16 I/O variant (gemma4 activations are fp16): reads fp16, accumulates the
// sum of squares in FP32 (overflow-safe), writes fp16. gamma fp16.
static const char *RMSNORM_FP16_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float rms_h2f(unsigned short h) {
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) o = s;
    else { int x=-1; do{m<<=1;x++;}while((m&0x400u)==0u); m&=0x3FFu;
           o = s | ((unsigned int)(127-15-x)<<23) | (m<<13); }
  } else if (e == 0x1Fu) o = s | 0x7F800000u | (m<<13);
  else o = s | ((e + (127u-15u))<<23) | (m<<13);
  return __int_as_float((int)o);
}
__device__ __forceinline__ unsigned short rms_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if (e==0xFF) return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if (exp>=0x1F) return (unsigned short)(s|0x7C00u);
  if (exp<=0){ if(exp<-10) return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u))) hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
}
__global__ void rmsnorm_fp16(const unsigned short *x, const unsigned short *gamma,
                             unsigned short *y, int width, float eps,
                             int has_gamma) {
  int row = blockIdx.x;
  const unsigned short *xr = x + (size_t)row * width;
  unsigned short *yr = y + (size_t)row * width;
  __shared__ float sdata[256];
  float partial = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float v = rms_h2f(xr[k]);
    partial += v * v;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sdata[0] / (float)width + eps);
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float g = has_gamma ? rms_h2f(gamma[k]) : 1.0f;
    yr[k] = rms_f2h(rms_h2f(xr[k]) * inv * g);
  }
}
// Vectorized RMSNorm: 4 halves per load/store, hardware half<->float (one
// instruction instead of the ~20-op software emulation above), and a
// warp-shuffle reduction. Same math, but the sum of squares is accumulated in
// a different ORDER (vector-of-4 per thread), so `inv` can move by an ulp --
// the deliberate, gated deviation of the fused norm/quant lever. Requires
// width % 4 == 0 and 8-byte-aligned rows; the caller checks both.
__device__ __forceinline__ float rms_h2f_hw(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short rms_f2h_hw(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
// gamma is a weight at an arbitrary 2-byte offset in the model blob, so it is
// routinely NOT vector-aligned even when the activation rows are; four scalar
// loads for it keep the activation traffic vectorized. has_gamma: 0 none,
// 1 vector-aligned, 2 scalar.
__device__ __forceinline__ float4 rms_gather4(const unsigned short *g) {
  return make_float4(rms_h2f_hw(g[0]), rms_h2f_hw(g[1]), rms_h2f_hw(g[2]),
                     rms_h2f_hw(g[3]));
}
__device__ __forceinline__ float4 rms_load4(uint2 r) {
  return make_float4(rms_h2f_hw((unsigned short)(r.x & 0xFFFFu)),
                     rms_h2f_hw((unsigned short)(r.x >> 16)),
                     rms_h2f_hw((unsigned short)(r.y & 0xFFFFu)),
                     rms_h2f_hw((unsigned short)(r.y >> 16)));
}
__global__ void rmsnorm_fp16_v4(const unsigned short *x,
                                const unsigned short *gamma, unsigned short *y,
                                int width, float eps, int has_gamma) {
  int row = blockIdx.x;
  const uint2 *xv = (const uint2 *)(x + (size_t)row * width);
  const uint2 *gv = (const uint2 *)gamma;
  uint2 *yv = (uint2 *)(y + (size_t)row * width);
  const int nv = width >> 2;
  __shared__ float ssq[32];
  float4 carry[8];
  int nc = 0;
  float p = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = rms_load4(xv[i]);
    if (nc < 8)
      carry[nc++] = f;
    p += f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
  }
  for (int o = 16; o > 0; o >>= 1)
    p += __shfl_down_sync(0xffffffffu, p, o);
  if ((threadIdx.x & 31) == 0)
    ssq[threadIdx.x >> 5] = p;
  __syncthreads();
  if (threadIdx.x < 32) {
    float a = (threadIdx.x < (blockDim.x >> 5)) ? ssq[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_down_sync(0xffffffffu, a, o);
    if (threadIdx.x == 0)
      ssq[0] = a;
  }
  __syncthreads();
  const float inv = rsqrtf(ssq[0] / (float)width + eps);
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = (nc < 8) ? carry[nc++] : rms_load4(xv[i]);
    float4 g = make_float4(1.f, 1.f, 1.f, 1.f);
    if (has_gamma == 1)
      g = rms_load4(gv[i]);
    else if (has_gamma == 2)
      g = rms_gather4(gamma + 4 * i);
    uint2 o;
    o.x = (unsigned int)rms_f2h_hw(f.x * inv * g.x) |
          ((unsigned int)rms_f2h_hw(f.y * inv * g.y) << 16);
    o.y = (unsigned int)rms_f2h_hw(f.z * inv * g.z) |
          ((unsigned int)rms_f2h_hw(f.w * inv * g.w) << 16);
    yv[i] = o;
  }
}
// PLE post-norm (ReverseRMSNorm): t = x * w applied BEFORE the norm,
// rms over t, then a SCALAR out_scale AFTER: y = (t / rms(t)) * out_scale.
// Same 1-block-per-row FP32-accumulate reduction as rmsnorm_fp16 above;
// out_scale is passed as a device pointer to one fp16 value (weights may be
// device-resident on the cuda pool).
__global__ void rms_reverse_norm_fp16(const unsigned short *x,
                                      const unsigned short *w,
                                      const unsigned short *out_scale,
                                      unsigned short *y, int width,
                                      float eps) {
  int row = blockIdx.x;
  const unsigned short *xr = x + (size_t)row * width;
  unsigned short *yr = y + (size_t)row * width;
  __shared__ float sdata[256];
  float partial = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float t = rms_h2f(xr[k]) * rms_h2f(w[k]);
    partial += t * t;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float scale = rsqrtf(sdata[0] / (float)width + eps) * rms_h2f(out_scale[0]);
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float t = rms_h2f(xr[k]) * rms_h2f(w[k]);
    yr[k] = rms_f2h(t * scale);
  }
}
}
)CU";

bool cuda_rmsnorm_fp16(const unsigned short *in, const unsigned short *gamma,
                       unsigned short *out, float eps, unsigned int rows,
                       unsigned int width) {
  if (rows == 0 || width == 0)
    return true;
  const bool vec4 = cuda_vec4_rows_small(rows) &&
                    cuda_vec4_rows_ok(width, in, out) &&
                    cuda_fc_qs4cx_fused_normq_enabled();
  auto kernel = CudaContext::Global().registerCudaKernel(
    RMSNORM_FP16_SRC, vec4 ? "rmsnorm_fp16_v4" : "rmsnorm_fp16");
  if (!kernel) {
    ml_loge("[CUDA] rmsnorm_fp16: kernel registration failed");
    return false;
  }
  int w = (int)width;
  int has_gamma = (gamma == nullptr)                       ? 0
                  : (!vec4 || cuda_vec4_rows_ok(4, gamma)) ? 1
                                                           : 2;
  const unsigned short *gamma_ptr = gamma;
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &gamma_ptr, sizeof(gamma_ptr));
  kernel->SetKernelArguments(2, &out, sizeof(out));
  kernel->SetKernelArguments(3, &w, sizeof(w));
  kernel->SetKernelArguments(4, &eps, sizeof(eps));
  kernel->SetKernelArguments(5, &has_gamma, sizeof(has_gamma));
  const int block[3] = {vec4 ? 512 : 256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  rms_maybe_finish(out);
  return true;
}

bool cuda_rms_reverse_norm_fp16(const unsigned short *in,
                                const unsigned short *w,
                                const unsigned short *out_scale,
                                unsigned short *out, float eps,
                                unsigned int rows, unsigned int width) {
  if (rows == 0 || width == 0)
    return true;
  auto kernel = CudaContext::Global().registerCudaKernel(
    RMSNORM_FP16_SRC, "rms_reverse_norm_fp16");
  if (!kernel) {
    ml_loge("[CUDA] rms_reverse_norm_fp16: kernel registration failed");
    return false;
  }
  int wi = (int)width;
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &w, sizeof(w));
  kernel->SetKernelArguments(2, &out_scale, sizeof(out_scale));
  kernel->SetKernelArguments(3, &out, sizeof(out));
  kernel->SetKernelArguments(4, &wi, sizeof(wi));
  kernel->SetKernelArguments(5, &eps, sizeof(eps));
  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  rms_maybe_finish(out);
  return true;
}

bool cuda_rmsnorm_fp32(const float *in, const float *gamma, float *out,
                       float eps, unsigned int rows, unsigned int width) {
  if (rows == 0 || width == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(RMSNORM_FP32_SRC, "rmsnorm_fp32");
  if (!kernel) {
    ml_loge("[CUDA] rmsnorm: kernel registration failed");
    return false;
  }

  int w = (int)width;
  int has_gamma = (gamma != nullptr) ? 1 : 0;
  const float *gamma_ptr = gamma; // may be null; never dereferenced when 0

  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &gamma_ptr, sizeof(gamma_ptr));
  kernel->SetKernelArguments(2, &out, sizeof(out));
  kernel->SetKernelArguments(3, &w, sizeof(w));
  kernel->SetKernelArguments(4, &eps, sizeof(eps));
  kernel->SetKernelArguments(5, &has_gamma, sizeof(has_gamma));

  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  rms_maybe_finish(out);
  return true;
}

} // namespace nntrainer::cuda
