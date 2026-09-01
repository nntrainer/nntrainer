// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_gelu.cpp
 * @date    27 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device GELU op implementation (NVRTC kernel, validated math).
 */

#include "cuda_gelu.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

namespace nntrainer::cuda {

// Plain elementwise map, one thread per element. mode 0 = erf-exact GELU
// (ACT_GELU), mode 1 = tanh approximation (ACT_TANH_GELU); constants match
// the OpenCL gelu.cl kernel byte-for-byte.
static const char *GELU_FP32_SRC = R"CU(
extern "C" __global__ void gelu_fp32(const float *x, float *y, int mode,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  if (mode == 1) {
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(inner));
  } else {
    y[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
  }
}
)CU";

// FP16 I/O variant: reads fp16, computes in float, writes fp16.
static const char *GELU_FP16_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float gelu_h2f(unsigned short h) {
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
__device__ __forceinline__ unsigned short gelu_f2h(float f) {
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
__global__ void gelu_fp16(const unsigned short *x, unsigned short *y, int mode,
                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = gelu_h2f(x[i]);
  float r;
  if (mode == 1) {
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    r = 0.5f * v * (1.0f + tanhf(inner));
  } else {
    r = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
  }
  y[i] = gelu_f2h(r);
}
}
)CU";

bool cuda_gelu_fp32(const float *in, float *out, int mode, unsigned int n) {
  if (n == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(GELU_FP32_SRC, "gelu_fp32");
  if (!kernel) {
    ml_loge("[CUDA] gelu_fp32: kernel registration failed");
    return false;
  }

  int ni = (int)n;
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &mode, sizeof(mode));
  kernel->SetKernelArguments(3, &ni, sizeof(ni));

  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)((n + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_gelu_fp16(const unsigned short *in, unsigned short *out, int mode,
                    unsigned int n) {
  if (n == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(GELU_FP16_SRC, "gelu_fp16");
  if (!kernel) {
    ml_loge("[CUDA] gelu_fp16: kernel registration failed");
    return false;
  }

  int ni = (int)n;
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &mode, sizeof(mode));
  kernel->SetKernelArguments(3, &ni, sizeof(ni));

  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)((n + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda
