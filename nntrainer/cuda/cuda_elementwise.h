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
                               float scalar, int width, int ring_cap = 0);

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

/**
 * @brief On-GPU greedy argmax over fp16 logits with the host sampling
 *        penalties folded in: repetition penalty over @p win_ids (the last
 *        repetition_window GENERATED token ids, duplicates and order
 *        preserved) and the bad-words -inf mask over @p bad_ids.
 * @details Replicates CausalLM's host path bit-for-bit: each id's
 *          first-occurrence thread replays applyRepetitionPenalty()'s
 *          per-id op chain sequentially in fp32 (fp16 read -> fp32
 *          v<0 ? v*p : v/p per occurrence, so k occurrences penalize k
 *          times; different ids never interact in the host loop, so per-id
 *          chains parallelize exactly) and applyBadWordsPenalty()'s
 *          -INFINITY overwrite lands last, into a sparse (id, value)
 *          override table; the
 *          argmax pass-1 substitutes the fp32 overrides for the raw logits so
 *          the reduction compares exactly the values host std::max_element
 *          would see (ties -> lowest index). The raw logits row is NOT
 *          modified -- no fp32->fp16 round-trip can perturb the winner. The
 *          ids are H2D'd per token from a pinned staging buffer on the
 *          backend stream (same pattern as cuda_set_pos), outside any graph
 *          capture; the caller runs this after the decode graph replay, like
 *          cuda_argmax_fp16.
 * @return false (caller falls back to the host path) on a null/zero arg, id
 *         counts over the scratch caps (win > 512 or bad > 64), scratch
 *         allocation under graph capture, or kernel failure.
 */
bool cuda_argmax_penalized_fp16(const unsigned short *logits_dev,
                                unsigned int vocab, const unsigned int *win_ids,
                                unsigned int n_win, const unsigned int *bad_ids,
                                unsigned int n_bad, float penalty,
                                unsigned int *token_out_host);

} // namespace nntrainer::cuda

#endif // __CUDA_ELEMENTWISE_H__
