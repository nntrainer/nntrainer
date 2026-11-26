// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   scalar_mul_fp16.cl
 * @brief  FP16 scalar multiplication with a row offset.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels_fp16.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void scalar_mul_cl_fp16(__global const half *in, __global half *out,
                                 float s, int n, int row_off) {
  int i = get_global_id(0);
  if (i < n)
    out[i + row_off] = convert_half(convert_float(in[i + row_off]) * s);
}
