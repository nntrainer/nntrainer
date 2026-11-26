// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   lmhead_fp32w_gemv.cl
 * @brief  LM head matrix-vector product with FP32 weights.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void lmhead_gemv_fp32w(__global const float *W,  // [V*H] row-major
                                __global const half *x,    // [H] fp16 act
                                __global float *logits,    // [V]
                                const int V, const int H) {
  const int row = get_group_id(0);
  const int t = get_local_id(0); // 0..63
  __global const float *rb = W + (size_t)row * (size_t)H;
  float sum = 0.0f;
  for (int k = t; k < H; k += 64)
    sum += rb[k] * (float)x[k];
  __local float red[64];
  red[t] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = 32; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) logits[row] = red[0];
}
