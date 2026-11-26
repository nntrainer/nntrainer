// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   lmhead_int4_v8c.cl
 * @brief  LM head matrix-vector product with packed INT4 weights.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void lmhead_int4_v8c_gemv(__global const uchar *W,    // [N][K/2] nibbles
                                   __global const half *x,      // [K] fp16 act
                                   __global const float *scale, // [N] fp32
                                   __global half *logits,       // [N]
                                   const int N, const int K) {
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  const int kblocks = K >> 5;    // K/32
  __global const uchar *wr = W + (size_t)row * (size_t)(K >> 1);
  float sum = 0.0f;
  for (int kb = t; kb < kblocks; kb += 64) {
    __global const uchar *blk = wr + (kb << 4); // 16 bytes
    __global const half *xb = x + (kb << 5);    // K base for this block
    // Vectorized: each 4-byte group c (uchar4) decodes to 8 nibbles whose K
    // indices are contiguous -- lo nibbles -> K=[c*8 .. c*8+3], hi nibbles ->
    // K=[c*8+4 .. c*8+7] -- so the activation is one contiguous half8 and the
    // whole group is two dot4s.
    for (int c = 0; c < 4; ++c) {
      const uchar4 by = vload4(0, blk + (c << 2));
      const float4 lo = convert_float4(convert_int4(by & (uchar4)0x0F) - 8);
      const float4 hi = convert_float4(convert_int4(by >> (uchar4)4) - 8);
      const float8 a = convert_float8(vload8(0, xb + (c << 3)));
      sum += dot(a.lo, lo) + dot(a.hi, hi);
    }
  }
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = (half)(red[0] * scale[row]);
}
