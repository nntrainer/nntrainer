// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_layernorm.cpp
 * @date    27 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device LayerNorm op implementation (NVRTC kernel, validated math).
 */

#include "cuda_layernorm.h"

#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cstdlib>

namespace nntrainer::cuda {

// Post-op drain, skipped for a device-only (cudaMalloc) destination: host
// code cannot read it without a stream-ordered staging copy, so the sync-mode
// per-op drain is provably unnecessary (see cuda_rmsnorm.cpp rms_maybe_finish
// for the measured rationale).
static inline void ln_maybe_finish(const void *out) {
  static const bool skip_dev_drain = []() {
    const char *e = std::getenv("NNTR_CUDA_DRAINSKIP_LN");
    return e != nullptr && e[0] == '1';
  }();
  if (skip_dev_drain && out != nullptr && dev_only(out))
    return;
  StreamManager::Global().maybeFinish();
}

// One block per row; block-reduces mean then variance in FP32 (two passes),
// then scales by rsqrt(var+eps) and folds gamma/beta. LayerNorm always has
// both gamma and beta (unlike RMSNorm's optional gamma), so no has_gamma
// flag here.
static const char *LAYERNORM_FP32_SRC = R"CU(
extern "C" __global__ void layernorm_fp32(const float *x, const float *gamma,
                                          const float *beta, float *y,
                                          int width, float eps) {
  int row = blockIdx.x;
  const float *xr = x + (size_t)row * width;
  float *yr = y + (size_t)row * width;
  __shared__ float sdata[256];

  // pass 1: mean = (1/width) * sum(x)
  float partial = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    partial += xr[k];
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float mean = sdata[0] / (float)width;
  // Every thread is about to read sdata[0] into a register (the line above).
  // Without this barrier, a fast thread starting pass 2 below can overwrite
  // sdata[tid] (aliasing sdata[0] for tid==0) before a slower thread has read
  // it -- a shared-memory race that pass 1's own reduction does not have
  // (nothing writes sdata after its final read there).
  __syncthreads();

  // pass 2: variance = (1/width) * sum((x-mean)^2)
  float partial2 = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float d = xr[k] - mean;
    partial2 += d * d;
  }
  sdata[threadIdx.x] = partial2;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float scale = rsqrtf(sdata[0] / (float)width + eps);

  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    yr[k] = (xr[k] - mean) * scale * gamma[k] + beta[k];
  }
}
)CU";

// FP16 I/O variant: reads fp16, accumulates mean/variance in FP32
// (overflow-safe, same rationale as rmsnorm_fp16), writes fp16. gamma/beta
// fp16, both required.
static const char *LAYERNORM_FP16_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float ln_h2f(unsigned short h) {
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
__device__ __forceinline__ unsigned short ln_f2h(float f) {
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
__global__ void layernorm_fp16(const unsigned short *x,
                               const unsigned short *gamma,
                               const unsigned short *beta, unsigned short *y,
                               int width, float eps) {
  int row = blockIdx.x;
  const unsigned short *xr = x + (size_t)row * width;
  unsigned short *yr = y + (size_t)row * width;
  __shared__ float sdata[256];

  // pass 1: mean = (1/width) * sum(x)
  float partial = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    partial += ln_h2f(xr[k]);
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float mean = sdata[0] / (float)width;
  // Same clobber-before-read hazard as layernorm_fp32 above: pass 2's store
  // into sdata[tid] can race a slow thread's read of sdata[0] for the mean.
  __syncthreads();

  // pass 2: variance = (1/width) * sum((x-mean)^2)
  float partial2 = 0.f;
  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    float d = ln_h2f(xr[k]) - mean;
    partial2 += d * d;
  }
  sdata[threadIdx.x] = partial2;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float scale = rsqrtf(sdata[0] / (float)width + eps);

  for (int k = threadIdx.x; k < width; k += blockDim.x) {
    yr[k] = ln_f2h((ln_h2f(xr[k]) - mean) * scale * ln_h2f(gamma[k]) +
                   ln_h2f(beta[k]));
  }
}
}
)CU";

bool cuda_layernorm_fp32(const float *in, const float *gamma, const float *beta,
                         float *out, float eps, unsigned int rows,
                         unsigned int width) {
  if (rows == 0 || width == 0)
    return true;

  auto kernel = CudaContext::Global().registerCudaKernel(LAYERNORM_FP32_SRC,
                                                         "layernorm_fp32");
  if (!kernel) {
    ml_loge("[CUDA] layernorm_fp32: kernel registration failed");
    return false;
  }

  int w = (int)width;
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &gamma, sizeof(gamma));
  kernel->SetKernelArguments(2, &beta, sizeof(beta));
  kernel->SetKernelArguments(3, &out, sizeof(out));
  kernel->SetKernelArguments(4, &w, sizeof(w));
  kernel->SetKernelArguments(5, &eps, sizeof(eps));

  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  ln_maybe_finish(out);
  return true;
}

bool cuda_layernorm_fp16(const unsigned short *in, const unsigned short *gamma,
                         const unsigned short *beta, unsigned short *out,
                         float eps, unsigned int rows, unsigned int width) {
  if (rows == 0 || width == 0)
    return true;

  auto kernel = CudaContext::Global().registerCudaKernel(LAYERNORM_FP16_SRC,
                                                         "layernorm_fp16");
  if (!kernel) {
    ml_loge("[CUDA] layernorm_fp16: kernel registration failed");
    return false;
  }

  int w = (int)width;
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &gamma, sizeof(gamma));
  kernel->SetKernelArguments(2, &beta, sizeof(beta));
  kernel->SetKernelArguments(3, &out, sizeof(out));
  kernel->SetKernelArguments(4, &w, sizeof(w));
  kernel->SetKernelArguments(5, &eps, sizeof(eps));

  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  ln_maybe_finish(out);
  return true;
}

} // namespace nntrainer::cuda
