// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rmsnorm.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RMSNorm op for the CUDA backend. Row-wise:
 *          y = x * rsqrt(mean(x^2) + eps) * gamma  (gamma optional / raw, no
 *          (1+gamma) bias -- matches ReshapedRMSNormLayer). Sum of squares is
 *          accumulated in FP32. Callers must pass device-accessible (UVM)
 *          pointers.
 */

#ifndef __CUDA_RMSNORM_H__
#define __CUDA_RMSNORM_H__

namespace nntrainer::cuda {

/**
 * @brief FP32 row-wise RMSNorm on device (UVM) pointers.
 *
 * @param in     [rows, width] row-major input (device-accessible)
 * @param gamma  [width] per-feature scale, or nullptr for the gamma-free norm
 * @param out    [rows, width] row-major output (device-accessible)
 * @param eps    epsilon added to the mean of squares
 * @param rows   number of rows (one block per row)
 * @param width  feature size (the normalized dimension)
 * @return true on success
 */
bool cuda_rmsnorm_fp32(const float *in, const float *gamma, float *out,
                       float eps, unsigned int rows, unsigned int width);

/** @brief fp16 I/O variant (gemma4 activations); FP32 sum-of-squares. */
bool cuda_rmsnorm_fp16(const unsigned short *in, const unsigned short *gamma,
                       unsigned short *out, float eps, unsigned int rows,
                       unsigned int width);

/**
 * @brief ReverseRMSNorm on device: y = ((x*w)/rms(x*w)) * out_scale[0].
 */
bool cuda_rms_reverse_norm_fp16(const unsigned short *in,
                                const unsigned short *w,
                                const unsigned short *out_scale,
                                unsigned short *out, float eps,
                                unsigned int rows, unsigned int width);

} // namespace nntrainer::cuda

#endif // __CUDA_RMSNORM_H__
