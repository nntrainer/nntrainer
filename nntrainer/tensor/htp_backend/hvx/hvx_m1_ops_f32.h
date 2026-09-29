// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_m1_ops_f32.h
 * @date   27 Sep 2026
 * @brief  The M=1 small ops on HVX: RMSNorm (whole row / per head), RoPE at
 *         head_dim 64, causal conv1d L=3 + gate, the MoE router (#132) --
 *         each bit-identical to nntrainer/tensor/m1_ops_det.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The specification, its operation order and its domain live in
 * m1_ops_det.h; this file implements it in plain Vsf (no qf32, no sf FMA --
 * HVX has none), except the RMSNorm row scale: scalar-core sffma and the
 * spec's integer sqrt / reciprocal (#164). All four run on the calling
 * thread: no worker pool, no VTCM, no DMA, no heap. Every load and store
 * goes through HVX_UVector, because the FastRPC buffers the test entries
 * hand in, and the per-token entry's activation slots, carry no 128-byte
 * alignment.
 *
 * SHAPE CONTRACT: a call with a shape the kernel does not accept (the
 * parameter notes below) returns without writing anything -- there is no
 * status to return through the per-token op table. Callers validate first:
 * the skel entries return AEE_EINVALIDFORMAT, and #85's graph validator
 * owns the shapes once the ops are wired.
 *
 * test/htp/host/m1_ops_host_check.c compiles THIS source against an
 * intrinsic emulation and compares it with the spec by memcmp;
 * unittest_hvx_softmax.cpp's HvxM1Ops.* repeats that on the device.
 */

#ifndef __NNTRAINER_HVX_M1_OPS_F32_H__
#define __NNTRAINER_HVX_M1_OPS_F32_H__

#include <stdint.h>

/**
 * @brief y = (x * r) * gamma per chunk, r the Android CPU's RMSNorm scale
 *        RN(1 / RN(sqrt(s / chunk + eps))), s the sum of squares over 16
 *        fused chains reduced ((h0 + h1) + h2) + h3 (m1_ops_det.h, #164).
 *
 * @param x, y          n floats; may alias
 * @param gamma         chunk floats, shared by every chunk
 * @param n             a multiple of chunk
 * @param chunk         a power of two and a multiple of 32 (2048: the hidden
 *                      norm; 64: the per-head q/k norm)
 * @param row_scale_out n / chunk floats, r per chunk; NULL to skip
 */
void hvx_rmsnorm_f32(const float *x, const float *gamma, float *y, uint32_t n,
                     uint32_t chunk, float eps, float *row_scale_out);

/**
 * @brief RoPE in place on n_q q heads then n_k k heads, each 64 contiguous
 *        floats; cs = cos[32] | sin[32] for this position.
 */
void hvx_rope64_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                    const float *cs);

/**
 * @brief Causal depthwise conv1d (L=3) + gate for one token, through the
 *        prefill kernel hvx_conv_gate_f32 at m_count = 1.
 *
 * @param abc    [3C]: the in_proj split a | b | c
 * @param state3 [3C]: rows 0-1 are the layer's conv state x_{t-2} | x_{t-1}
 *                     (CausalConv1DLayer's layout) on entry and on return;
 *                     row 2 is this call's scratch for a*c. Three rows,
 *                     because the prefill kernel reads g[t-2..t] at one
 *                     stride and this wrapper allocates nothing.
 * @param conv_w [3C]: w0 (current), w1 (t-1), w2 (t-2)
 * @param out    [C]:  b * conv1d(a*c)
 * @param C      a multiple of 32
 */
void hvx_conv_gate_m1_f32(const float *abc, float *state3, const float *conv_w,
                          float *out, uint32_t C);

/**
 * @brief The MoE router of one token in the Android CPU's order
 *        (m1_router_cpu_det, #132 PR 2): logits, sigmoid, biased top-k
 *        with the lowest index winning a tie, and the normalized routing
 *        weights.
 *
 * The logits are E fused chains over k in order, run on the scalar core
 * with sffma (E independent chains in flight); the rest is the spec's own
 * m1_router_cpu_select. The weight read (K x 128 B, 256 KiB at LFM2.5)
 * keeps an l2fetch of the next 16 KiB ahead -- a hint that moves no bits.
 *
 * @param x       K floats
 * @param w32     K x 32 floats: the [K][E] gate weight padded to 32 columns
 *                (lanes >= E are read and ignored)
 * @param bias    E floats
 * @param K       >= 1
 * @param E       1..32
 * @param top_k   1..E
 * @param logits  E floats out
 * @param sel     top_k expert indices out, in selection order
 * @param weight  top_k routing weights out, in selection order
 */
void hvx_router_topk_f32(const float *x, const float *w32, const float *bias,
                         uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                         uint32_t *sel, float *weight);

#endif /* __NNTRAINER_HVX_M1_OPS_F32_H__ */
