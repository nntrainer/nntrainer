// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   lmhead_q6k_gemv_a4.cl
 * @brief  Four-row LM head matrix-vector product with aligned Q6_K weights.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void q6k_gemv_lmhead_a4(__global const uchar *W,  // [V][H/256][212 B]
                                 __global const float *x,   // [H]
                                 __global float *logits,    // [V]
                                 const int V, const int H) {
  const int t = get_local_id(0); // 0..63
  const int rl = t >> 4;         // row within the workgroup
  const int u = t & 15;
  const int row = get_group_id(0) * 4 + rl;
  const int nb = H >> 8;
  const int nh = u >> 3;
  const int q = u & 7;
  const int is = q >> 2;
  float sum = 0.0f;
  if (row < V) {
    __global const uint *rb =
      (__global const uint *)(W + (size_t)row * (size_t)(nb * 212));
    for (int bi = 0; bi < nb; ++bi) {
      __global const uint *blk = rb + bi * 53;
      const uint wl = blk[(nh << 4) + q];
      const uint wh = blk[(nh << 4) + 8 + q];
      const uint wq = blk[32 + (nh << 3) + q];
      const uint s01 = blk[48 + (nh << 1)] >> (is << 3);
      const uint s45 = blk[49 + (nh << 1)] >> (is << 3);
      const float d = vload_half(0, (__global const half *)(blk + 52));
      const float s0 = d * (float)(((int)(s01 << 24)) >> 24);
      const float s2 = d * (float)(((int)(s01 << 8)) >> 24);
      const float s4 = d * (float)(((int)(s45 << 24)) >> 24);
      const float s6 = d * (float)(((int)(s45 << 8)) >> 24);
      const int yb = (bi << 8) + (nh << 7) + (q << 2);
      float acc0 = 0.0f, acc2 = 0.0f, acc4 = 0.0f, acc6 = 0.0f;
      for (int j = 0; j < 4; ++j) {
        const int sh = j << 3;
        const int l = (int)((wl >> sh) & 0xFF);
        const int hh = (int)((wh >> sh) & 0xFF);
        const int qq = (int)((wq >> sh) & 0xFF);
        const int v0 = ((l & 0xF) | ((qq & 3) << 4)) - 32;
        const int v2 = ((hh & 0xF) | (((qq >> 2) & 3) << 4)) - 32;
        const int v4 = ((l >> 4) | (((qq >> 4) & 3) << 4)) - 32;
        const int v6 = ((hh >> 4) | (((qq >> 6) & 3) << 4)) - 32;
        acc0 += x[yb + j] * (float)v0;
        acc2 += x[yb + 32 + j] * (float)v2;
        acc4 += x[yb + 64 + j] * (float)v4;
        acc6 += x[yb + 96 + j] * (float)v6;
      }
      sum += s0 * acc0 + s2 * acc2 + s4 * acc4 + s6 * acc6;
    }
  }
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 8; off > 0; off >>= 1) {
    if (u < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (u == 0 && row < V) logits[row] = red[t];
}
