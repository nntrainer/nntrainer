// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_layernorm.h
 * @date    27 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device LayerNorm op (NVRTC kernel) for the CUDA backend. Row-wise:
 *          y = (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta, i.e.
 *          RMSNorm plus a row-mean subtraction and a beta (shift) weight --
 *          gamma/beta are both required (LayerNorm always has both, unlike
 *          the optional-gamma RMSNorm). Mean and variance are accumulated in
 *          FP32. Callers must pass device-accessible (UVM) pointers.
 */

#ifndef __CUDA_LAYERNORM_H__
#define __CUDA_LAYERNORM_H__

namespace nntrainer::cuda {

/**
 * @brief FP32 row-wise LayerNorm on device (UVM) pointers.
 *
 * @param in     [rows, width] row-major input (device-accessible)
 * @param gamma  [width] per-feature scale
 * @param beta   [width] per-feature shift
 * @param out    [rows, width] row-major output (device-accessible)
 * @param eps    epsilon added to the variance
 * @param rows   number of rows (one block per row)
 * @param width  feature size (the normalized dimension)
 * @return true on success
 */
bool cuda_layernorm_fp32(const float *in, const float *gamma, const float *beta,
                         float *out, float eps, unsigned int rows,
                         unsigned int width);

/** @brief fp16 I/O variant; FP32 mean/variance accumulation. */
bool cuda_layernorm_fp16(const unsigned short *in, const unsigned short *gamma,
                         const unsigned short *beta, unsigned short *out,
                         float eps, unsigned int rows, unsigned int width);

} // namespace nntrainer::cuda

#endif // __CUDA_LAYERNORM_H__
