// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   act_cast.cl
 * @brief  FP32/FP16 activation conversion kernels.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from attention_kernels.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void cast_f32_to_f16(__global const float *in, __global half *out,
                              const int N) {
  const int i = get_global_id(0);
  if (i < N)
    vstore_half_rte(in[i], i, out);
}
__kernel void cast_f16_to_f32(__global const half *in, __global float *out,
                              const int N) {
  const int i = get_global_id(0);
  if (i < N)
    out[i] = vload_half(i, in);
}
