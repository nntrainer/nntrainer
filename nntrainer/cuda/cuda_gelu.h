// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_gelu.h
 * @date    27 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device GELU op for the CUDA backend. Plain elementwise map:
 *          mode 0 = erf-exact GELU (0.5*x*(1+erf(x/sqrt2))), mode 1 = tanh
 *          approximation (gelu_pytorch_tanh). No per-row reduction, so unlike
 *          cuda_rmsnorm/cuda_layernorm this is a flat 1-D launch over the
 *          element count. Callers must pass device-accessible (UVM) pointers.
 */

#ifndef __CUDA_GELU_H__
#define __CUDA_GELU_H__

namespace nntrainer::cuda {

/**
 * @brief FP32 elementwise GELU on device (UVM) pointers.
 *
 * @param in    input buffer (device-accessible)
 * @param out   output buffer (device-accessible)
 * @param mode  0 = erf-based exact GELU, 1 = tanh approximation
 * @param n     element count
 * @return true on success
 */
bool cuda_gelu_fp32(const float *in, float *out, int mode, unsigned int n);

/** @brief fp16 I/O variant; math done in FP32. */
bool cuda_gelu_fp16(const unsigned short *in, unsigned short *out, int mode,
                    unsigned int n);

} // namespace nntrainer::cuda

#endif // __CUDA_GELU_H__
