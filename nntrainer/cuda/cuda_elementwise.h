// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_elementwise.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device element-wise ops (swiglu / scalar-mul / softcap) for the
 *          decode path -- the small per-op host loops that break the GPU chain.
 *          fp16 I/O, FP32 math; each collapses one op to a single kernel so the
 *          activation stays device-resident between the surrounding GEMMs.
 */

#ifndef __CUDA_ELEMENTWISE_H__
#define __CUDA_ELEMENTWISE_H__

namespace nntrainer::cuda {

/** @brief out[i] = silu(gate[i]) * up[i], silu(x) = x/(1+exp(-x)) (qwen3 FFN)
 */
bool cuda_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                      unsigned short *out, unsigned int n);

/** @brief out[i] = in[i] * scalar */
bool cuda_scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int n, float scalar);

/** @brief out[i] = cap * tanh(in[i] / cap) -- final logit softcapping */
bool cuda_softcap_fp16(const unsigned short *in, unsigned short *out,
                       unsigned int n, float cap);

} // namespace nntrainer::cuda

#endif // __CUDA_ELEMENTWISE_H__
