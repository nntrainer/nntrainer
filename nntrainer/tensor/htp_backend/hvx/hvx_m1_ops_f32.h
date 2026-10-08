// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_m1_ops_f32.h
 * @date   27 Sep 2026
 * @brief  The M=1 small ops on HVX: RMSNorm (whole row / per head), RoPE at
 *         head_dim 64 (and any multiple of 64, plan 201 S4), causal conv1d L=3
 * + gate, the MoE router (#132) -- each bit-identical to
 * nntrainer/tensor/m1_ops_det.h
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

#include "hvx_worker_pool.h"

/**
 * @brief y = (x * r) * gamma per chunk, r the Android CPU's RMSNorm scale
 *        RN(1 / RN(sqrt(s / chunk + eps))), s the sum of squares over 16
 *        fused chains reduced ((h0 + h1) + h2) + h3 (m1_ops_det.h, #164).
 *
 * @param x, y          n floats; may alias
 * @param gamma         chunk floats, shared by every chunk; NULL: y = x * r
 *                      (Gemma's v norm)
 * @param n             a multiple of chunk
 * @param chunk         a multiple of 32 (2048: LFM's hidden norm; 64: the
 *                      per-head q/k norm; 2816: Gemma's hidden norm). The
 *                      CPU's 16 chains cover it, so there is no tail; any
 *                      other width is refused (the contract below)
 * @param row_scale_out n / chunk floats, r per chunk; NULL to skip
 */
void hvx_rmsnorm_f32(const float *x, const float *gamma, float *y, uint32_t n,
                     uint32_t chunk, float eps, float *row_scale_out);

/**
 * @brief [plan 201 S4] N1, opt-in: hvx_rmsnorm_f32's shape and output form,
 *        r from m1_rmsnorm_n1_chunk_det's order-free integer sum of squares
 *        (32 word lanes). Not the CPU's bits. DOMAIN: finite x.
 */
void hvx_rmsnorm_n1_f32(const float *x, const float *gamma, float *y,
                        uint32_t n, uint32_t chunk, float eps,
                        float *row_scale_out);

/**
 * @brief RoPE in place on n_q q heads then n_k k heads, each 64 contiguous
 *        floats; cs = cos[32] | sin[32] for this position.
 */
void hvx_rope64_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                    const float *cs);

/**
 * @brief [plan 201 S4] m1_rope_det on n_q q heads then n_k k heads, each
 *        @a head_dim contiguous floats (half a head a multiple of 32: 64,
 *        128, 256, 512); cs = cos[head_dim / 2] | sin[head_dim / 2] for
 *        this position. A head_dim outside that rule returns without
 *        writing (the validator refuses it first).
 */
void hvx_rope_f32(float *q, uint32_t n_q, float *k, uint32_t n_k,
                  const float *cs, uint32_t head_dim);

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
 * @brief [#132 Part B E5f] m1_swiglu_cpu_det over @a pool's lanes: the
 *        spec's exp_ps and add, the quotient by the Hexagon's IEEE scalar
 *        divide instead of the spec's integer division (the same RN result:
 *        HvxFcQ4.ScalarDivide and SmallOpsMatchSpec check it on the DSP;
 *        m1_ops_host_check on the host), then * z. n any; pool may be NULL.
 */
void hvx_swiglu_cpu_f32(const float *y, const float *z, float *out, uint32_t n,
                        hvx_worker_pool *pool);

/**
 * @brief [#132 Part B E5f] m1_argmax_first (std::max_element: the first
 *        maximum, -0 == +0) in one vector pass: an order-preserving integer
 *        key per lane, the lane maxima and the block each first reached
 *        its maximum at. Runs the spec itself when n % 32 != 0 or any
 *        element is a NaN.
 */
uint32_t hvx_argmax_first_f32(const float *x, uint32_t n);

/**
 * @brief [plan 201 S4] Gemma 4's dense GeGLU: out = geglu_det_one(gate,
 *        up) per element (swiglu_det.h, the MoE epilogue's spec), through
 *        hvx_geglu_det_sf a vector at a time; n any (the tail scalar).
 */
void hvx_geglu_f32(const float *gate, const float *up, float *out, uint32_t n);

/**
 * @brief [plan 201 S4] m1_softcap_det in place over @a pool's lanes: Gemma
 *        4's final logit soft-cap, cap a positive normal f32. n any (the
 *        tail scalar); pool may be NULL.
 */
void hvx_softcap_f32(float *x, uint32_t n, float cap, hvx_worker_pool *pool);

/** @brief [plan 201 S4] x[i] = x[i] * s, one IEEE multiply each (Gemma 4's
 *         layer_scalar on the residual); n any. */
void hvx_mul_scalar_f32(float *x, float s, uint32_t n);

/**
 * @brief The MoE router of one token in the Android CPU's order
 *        (m1_router_cpu_det, #132 PR 2): logits, sigmoid, biased top-k
 *        with the lowest index winning a tie, and the normalized routing
 *        weights.
 *
 * The logits are E fused chains over k in order, run on the scalar cores
 * with sffma, 8 register-resident chains per pool lane (so up to 4 lanes
 * at E = 32); the rest is the spec's own m1_router_cpu_sigmoid (in the lanes)
 * and m1_router_cpu_pick. The weight read (K x 128 B, 256 KiB at LFM2.5) keeps
 * an l2fetch of the next 16 KiB ahead -- a hint that moves no bits.
 *
 * @param x       K floats
 * @param w32     K x 32 floats: the [K][E] gate weight padded to 32 columns
 *                (lanes >= E are read and ignored), 8-byte aligned
 * @param bias    E floats
 * @param K       >= 1
 * @param E       1..32
 * @param top_k   1..E
 * @param logits  E floats out
 * @param sel     top_k expert indices out, in selection order
 * @param weight  top_k routing weights out, in selection order
 * @param pool    the lanes for the chains (NULL: the caller alone)
 */
void hvx_router_topk_f32(const float *x, const float *w32, const float *bias,
                         uint32_t K, uint32_t E, uint32_t top_k, float *logits,
                         uint32_t *sel, float *weight, hvx_worker_pool *pool);

/**
 * @brief [plan 201 S4] Gemma 4's router after its input norm
 *        (m1_router_softmax_det): logits as unfused Vsf chains over k, 32
 *        experts a vector and a vector per pool lane, then the spec's own
 *        m1_router_softmax_pick (softmax, top-k with the lowest index on a
 *        tie, renormalised, times the per-expert scale).
 *
 * @param x       K floats: rmsnorm(h) * m1_router_input_scale_det's g
 * @param wp      K x Ep floats: [K][E] padded to Ep = E rounded up to 32
 *                columns (lanes >= E are read and ignored)
 * @param pes     E floats, the per-expert scale
 * @param E       1..128
 * @param top_k   1..E
 * @param logits  E floats out; may alias x (written after the last read)
 * @param sel     top_k expert indices out, in selection order
 * @param weight  top_k routing weights out, in selection order
 * @param pool    the lanes for the chains (NULL: the caller alone)
 */
void hvx_router_softmax_topk_f32(const float *x, const float *wp,
                                 const float *pes, uint32_t K, uint32_t E,
                                 uint32_t top_k, float *logits, uint32_t *sel,
                                 float *weight, hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_M1_OPS_F32_H__ */
