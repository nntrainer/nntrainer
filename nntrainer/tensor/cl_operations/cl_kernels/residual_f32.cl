// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   residual_f32.cl
 * @brief  FP32 residual copy or accumulation kernel.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * @details Extracted from blas_kernels.cpp.
 */

__kernel void residual_f32(__global const float *in, __global float *out,
                           const int N, const int accumulate) {
  const int i = get_global_id(0);
  if (i < N)
    out[i] = accumulate ? out[i] + in[i] : in[i];
}
