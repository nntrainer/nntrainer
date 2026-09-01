// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_elementwise.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device element-wise ops (geglu / add / scalar-mul / slice) for the
 *          gemma4 decode path -- the small host ops that break the GPU chain.
 *          fp16 I/O, FP32 math; all reduce per-op host work to one kernel.
 */

#ifndef __CUDA_ELEMENTWISE_H__
#define __CUDA_ELEMENTWISE_H__

namespace nntrainer::cuda {

/** @brief out[i] = gelu_tanh(gate[i]) * up[i], gelu_tanh = pytorch-tanh approx
 */
bool cuda_geglu_fp16(const unsigned short *gate, const unsigned short *up,
                     unsigned short *out, unsigned int n);

/** @brief out[i] = silu(gate[i]) * up[i], silu(x) = x/(1+exp(-x)) (qwen3 FFN)
 */
bool cuda_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                      unsigned short *out, unsigned int n);

/** @brief out[i] = a[i] + b[i] (residual add) */
bool cuda_add_fp16(const unsigned short *a, const unsigned short *b,
                   unsigned short *out, unsigned int n);

/** @brief out[i] = in[i] * scalar */
bool cuda_scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int n, float scalar);

/**
 * @brief Device-slot KV V-copy: out_base[d_pos[0]*width + i] = scalar * in[i],
 * with the cache slot read from the device cuda_pos_buffer() so a captured
 * graph writes V to the live (new-token) slot on every replay. @p out_base is
 *        the cache BASE (batch) pointer; @p width is the per-row element count.
 */
bool cuda_scalar_mul_fp16_slot(const unsigned short *in,
                               unsigned short *out_base, unsigned int n,
                               float scalar, int width);

/** @brief out[i] = cap * tanh(in[i] / cap) -- final logit softcapping */
bool cuda_softcap_fp16(const unsigned short *in, unsigned short *out,
                       unsigned int n, float cap);

/** @brief out[r*fs + f] = in[r*in_width + layer_off + f] (per-layer slice) */
bool cuda_slice_copy_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int rows, unsigned int in_width,
                          unsigned int layer_off, unsigned int fs);

/**
 * @brief On-GPU greedy argmax over device-resident fp32 logits [vocab].
 * @details Two-pass block reduction entirely on the GPU; only the 4-byte
 *          winning index is copied to the host (vs the full-vocab D->H pass +
 *          host std::max_element). Ties resolve to the LOWEST index, matching
 *          std::max_element. @p logits_dev must be device-accessible (UVM /
 *          managed or device). Returns false (caller falls back to the host
 *          path) on a null/zero arg, a non-device pointer, or under graph
 *          capture before the scratch is allocated.
 */
bool cuda_argmax_fp32(const float *logits_dev, unsigned int vocab,
                      unsigned int *token_out_host);

/** @brief fp16 variant of cuda_argmax_fp32 (logits decoded half->float). */
bool cuda_argmax_fp16(const unsigned short *logits_dev, unsigned int vocab,
                      unsigned int *token_out_host);

} // namespace nntrainer::cuda

#endif // __CUDA_ELEMENTWISE_H__
