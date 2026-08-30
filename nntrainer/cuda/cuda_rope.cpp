// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rope.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RoPE op (NVRTC kernel) -- split-half, matches host math.
 */

#include "cuda_rope.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

namespace nntrainer::cuda {

// One block per (row, head); threads sweep the rotated-pair index k in
// [0,half). Per-row position is from + blockIdx.x, indexing the flat device
// cos/sin LUTs.
static const char *ROPE_FP16_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float rp_h2f(unsigned short h) {
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
__device__ __forceinline__ unsigned short rp_f2h(float f) {
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
__global__ void rope_fp16(const unsigned short *in, unsigned short *out,
                          const unsigned short *cos_lut,
                          const unsigned short *sin_lut, int num_heads,
                          int head_dim, int half, int from) {
  int row = blockIdx.x, head = blockIdx.y;
  long HD = (long)num_heads * head_dim;
  const unsigned short *xr = in + (long)row * HD + (long)head * head_dim;
  unsigned short *yr = out + (long)row * HD + (long)head * head_dim;
  const unsigned short *cosr = cos_lut + (long)(from + row) * half;
  const unsigned short *sinr = sin_lut + (long)(from + row) * half;
  for (int k = threadIdx.x; k < half; k += blockDim.x) {
    float a = rp_h2f(xr[k]);
    float b = rp_h2f(xr[k + half]);
    float c = rp_h2f(cosr[k]);
    float s = rp_h2f(sinr[k]);
    yr[k]        = rp_f2h(a * c - b * s);
    yr[k + half] = rp_f2h(a * s + b * c);
  }
}
// Device-pos variant: reads the RoPE position `from` from d_pos[0] (so the
// captured graph never bakes a stale int), and -- when out_slot_dpos != 0 --
// writes each input row to OUTPUT row (from+row) so the K-into-cache write lands
// at the live cache slot computed on-device (out is then the cache BASE pointer,
// not a host pre-offset slice). out_slot_dpos==0 keeps the row-relative output
// (Q, in-place). Math identical to rope_fp16.
// [kv-window-ring] ring_cap > 0 maps the slot-mode OUTPUT row to its physical
// ring row ((from+row) % ring_cap): a sliding layer's cache only has ring_cap
// physical rows, so the absolute row is outside it. The RoPE LUT index stays
// ABSOLUTE (from+row) -- only physical storage wraps, positions do not.
__global__ void rope_fp16_dpos(const unsigned short *in, unsigned short *out,
                               const unsigned short *cos_lut,
                               const unsigned short *sin_lut, int num_heads,
                               int head_dim, int half, const int *d_pos,
                               int out_slot_dpos, int ring_cap) {
  int from = d_pos[0];
  int row = blockIdx.x, head = blockIdx.y;
  long HD = (long)num_heads * head_dim;
  long out_row;
  if (out_slot_dpos) {
    long abs_row = (long)from + row;
    out_row = (ring_cap > 0) ? (abs_row % ring_cap) : abs_row;
  } else {
    out_row = (long)row;
  }
  const unsigned short *xr = in + (long)row * HD + (long)head * head_dim;
  unsigned short *yr = out + out_row * HD + (long)head * head_dim;
  const unsigned short *cosr = cos_lut + (long)(from + row) * half;
  const unsigned short *sinr = sin_lut + (long)(from + row) * half;
  for (int k = threadIdx.x; k < half; k += blockDim.x) {
    float a = rp_h2f(xr[k]);
    float b = rp_h2f(xr[k + half]);
    float c = rp_h2f(cosr[k]);
    float s = rp_h2f(sinr[k]);
    yr[k]        = rp_f2h(a * c - b * s);
    yr[k + half] = rp_f2h(a * s + b * c);
  }
}
}
)CU";

bool cuda_rope_fp16(const unsigned short *in, unsigned short *out,
                    const unsigned short *cos_lut,
                    const unsigned short *sin_lut, int num_heads, int head_dim,
                    int num_rows, int from) {
  if (num_heads == 0 || head_dim == 0 || num_rows == 0)
    return true;
  const int half = head_dim / 2;
  auto kernel =
    CudaContext::Global().registerCudaKernel(ROPE_FP16_SRC, "rope_fp16");
  if (!kernel) {
    ml_loge("[CUDA] rope_fp16: kernel registration failed");
    return false;
  }
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &cos_lut, sizeof(cos_lut));
  kernel->SetKernelArguments(3, &sin_lut, sizeof(sin_lut));
  kernel->SetKernelArguments(4, &num_heads, sizeof(num_heads));
  kernel->SetKernelArguments(5, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(6, &half, sizeof(half));
  kernel->SetKernelArguments(7, &from, sizeof(from));
  const int block[3] = {half < 256 ? half : 256, 1, 1};
  const int grid[3] = {num_rows, num_heads, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_rope_fp16_dpos(const unsigned short *in, unsigned short *out,
                         const unsigned short *cos_lut,
                         const unsigned short *sin_lut, int num_heads,
                         int head_dim, int num_rows, int out_slot_dpos,
                         int ring_cap) {
  if (num_heads == 0 || head_dim == 0 || num_rows == 0)
    return true;
  const int half = head_dim / 2;
  auto kernel =
    CudaContext::Global().registerCudaKernel(ROPE_FP16_SRC, "rope_fp16_dpos");
  if (!kernel) {
    ml_loge("[CUDA] rope_fp16_dpos: kernel registration failed");
    return false;
  }
  const int *d_pos = cuda_pos_buffer();
  kernel->SetKernelArguments(0, &in, sizeof(in));
  kernel->SetKernelArguments(1, &out, sizeof(out));
  kernel->SetKernelArguments(2, &cos_lut, sizeof(cos_lut));
  kernel->SetKernelArguments(3, &sin_lut, sizeof(sin_lut));
  kernel->SetKernelArguments(4, &num_heads, sizeof(num_heads));
  kernel->SetKernelArguments(5, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(6, &half, sizeof(half));
  kernel->SetKernelArguments(7, &d_pos, sizeof(d_pos));
  kernel->SetKernelArguments(8, &out_slot_dpos, sizeof(out_slot_dpos));
  kernel->SetKernelArguments(9, &ring_cap,
                             sizeof(ring_cap)); // [kv-window-ring]
  const int block[3] = {half < 256 ? half : 256, 1, 1};
  const int grid[3] = {num_rows, num_heads, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda
