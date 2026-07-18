// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rmsnorm.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RMSNorm op implementation (NVRTC kernel).
 */

#include "cuda_rmsnorm.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

namespace nntrainer::cuda {

// One block per row; reads fp16, accumulates the sum of squares in FP32
// (overflow-safe), scales by rsqrt(mean+eps) and folds the raw gamma (no
// (1+gamma) bias). has_gamma=0 skips the gamma read (gamma-free norm).
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
}
)CU";

bool cuda_rmsnorm_fp16(const unsigned short *in, const unsigned short *gamma,
                       unsigned short *out, float eps, unsigned int rows,
                       unsigned int width) {
  if (rows == 0 || width == 0)
    return true;
  auto kernel =
    CudaContext::Global().registerCudaKernel(RMSNORM_FP16_SRC, "rmsnorm_fp16");
  if (!kernel) {
    ml_loge("[CUDA] rmsnorm_fp16: kernel registration failed");
    return false;
  }
  int w = (int)width;
  int has_gamma = (gamma != nullptr) ? 1 : 0;
  const unsigned short *gamma_ptr = gamma;
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
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda
