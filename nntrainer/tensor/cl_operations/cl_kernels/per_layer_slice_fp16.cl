// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   per_layer_slice_fp16.cl
 * @brief  Per-layer row slicing of FP16 activations.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels_fp16.cpp.
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void per_layer_slice_cl_fp16(__global const half *in,
                                      __global half *out, int fs, int in_width,
                                      int off, int rows) {
  int gid = get_global_id(0);
  int total = rows * fs;
  if (gid >= total)
    return;
  int r = gid / fs;
  int j = gid - r * fs;
  out[gid] = in[r * in_width + off + j];
}
