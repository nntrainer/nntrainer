// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_gemv_q6k.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Q6_K lm_head GEMV implementation (NVRTC kernel). Split out of
 *          cuda_fc_qint4.cpp -- see cuda_gemv_q6k.h.
 */

#include "cuda_gemv_q6k.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

namespace nntrainer::cuda {

// Q6_K lm_head GEMV: FP16 hidden x (managed) Q6_K weight -> FP16 logits, all on
// the device. this is the GPU path gemma4 gets from its QINT4 untied lm_head.
static const char *Q6K_GEMV_SRC = R"CU(
#define QK_K 256
typedef unsigned char  u8;
typedef signed char    s8;
typedef unsigned short u16;

typedef struct { u8 ql[128]; u8 qh[64]; s8 scales[16]; u16 d; } block_q6_K;

__device__ __forceinline__ float h2f(u16 h) {
  unsigned int s = (unsigned int)(h & 0x8000) << 16;
  unsigned int e = (h >> 10) & 0x1F;
  unsigned int m = h & 0x3FF;
  unsigned int out;
  if (e == 0) {
    if (m == 0) { out = s; }
    else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
           out = s | ((e + 112) << 23) | (m << 13); }
  } else if (e == 31) { out = s | 0x7F800000u | (m << 13); }
  else { out = s | ((e + 112) << 23) | (m << 13); }
  return __int_as_float((int)out);
}

__device__ __forceinline__ u16 f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
  unsigned int man = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return (u16)sign;
    man |= 0x800000u;
    unsigned int shift = (unsigned int)(14 - exp);
    unsigned int half = man >> shift;
    if ((man >> (shift - 1)) & 1u) half += 1; // round to nearest
    return (u16)(sign | half);
  } else if (exp >= 31) {
    return (u16)(sign | 0x7C00u);
  }
  unsigned int half = (unsigned int)(exp << 10) | (man >> 13);
  if ((man >> 12) & 1u) half += 1; // round to nearest
  return (u16)(sign | half);
}

extern "C" __global__ void q6k_gemv(const void *src0, const u16 *src1, u16 *dst,
                                    int ne00, int ne01) {
  const int N_SIMDWIDTH = 16;
  __shared__ float red[2][16];
  int nb = ne00 / QK_K;
  int row_group = threadIdx.x / N_SIMDWIDTH;
  int lane = threadIdx.x % N_SIMDWIDTH;
  int row = blockIdx.x * 2 + row_group;
  const block_q6_K *x = (const block_q6_K *)src0 + (long)row * nb;
  const u16 *yy = src1;
  u8 kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  int tid = lane;
  int ip = tid / 8, il = tid % 8, l0 = 4 * il;
  int is = 8 * ip + l0 / 16;
  int y_offset = 128 * ip + l0;
  int q_offset_l = 64 * ip + l0;
  int q_offset_h = 32 * ip + l0;
  float sumf = 0.0f;
  if (row < ne01) {
    for (int i = 0; i < nb; i++) {
      const u8 *q1 = x[i].ql + q_offset_l;
      const u8 *q2 = q1 + QK_K / 8;
      const u8 *qh = x[i].qh + q_offset_h;
      const s8 *sc = x[i].scales + is;
      const u16 *y = yy + i * QK_K + y_offset;
      float dall = h2f(x[i].d);
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (int j = 0; j < 4; j++) {
        s0 += h2f(y[j + 0])  * ((float)((q1[j] & 0xF) | ((qh[j] & kmask1) << 4)) - 32.f);
        s1 += h2f(y[j + 32]) * ((float)((q2[j] & 0xF) | ((qh[j] & kmask2) << 2)) - 32.f);
        s2 += h2f(y[j + 64]) * ((float)((q1[j] >> 4)  | ((qh[j] & kmask3) >> 0)) - 32.f);
        s3 += h2f(y[j + 96]) * ((float)((q2[j] >> 4)  | ((qh[j] & kmask4) >> 2)) - 32.f);
      }
      sumf += dall * (s0 * sc[0] + s1 * sc[2] + s2 * sc[4] + s3 * sc[6]);
    }
  }
  red[row_group][lane] = sumf;
  __syncthreads();
  for (int off = N_SIMDWIDTH / 2; off > 0; off >>= 1) {
    if (lane < off) red[row_group][lane] += red[row_group][lane + off];
    __syncthreads();
  }
  if (lane == 0 && row < ne01)
    dst[row] = f2h(red[row_group][0]);
}
)CU";

bool lmhead_gemv_q6_k_cuda(const void *w_q6k_dev,
                           const unsigned short *hidden_fp16_dev,
                           unsigned short *logits_fp16_dev, int vocab,
                           int hidden) {
  if (vocab <= 0 || hidden <= 0 || (hidden % 256) != 0)
    return false;
  auto kernel =
    CudaContext::Global().registerCudaKernel(Q6K_GEMV_SRC, "q6k_gemv");
  if (!kernel)
    return false;
  kernel->SetKernelArguments(0, &w_q6k_dev, sizeof(w_q6k_dev));
  kernel->SetKernelArguments(1, &hidden_fp16_dev, sizeof(hidden_fp16_dev));
  kernel->SetKernelArguments(2, &logits_fp16_dev, sizeof(logits_fp16_dev));
  kernel->SetKernelArguments(3, &hidden, sizeof(hidden));
  kernel->SetKernelArguments(4, &vocab, sizeof(vocab));
  const int block[3] = {32, 1, 1};
  const int grid[3] = {(vocab + 1) / 2, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda
