// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_decode_hook.h
 * @date   28 Sep 2026
 * @brief  The CPU layers' one-line entry to the HTP per-token decode hook
 *         (#130): each resident op kind as one inline call, 0 without HTP
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * A layer calls its function at one decode row (to - from == 1, FP32)
 * before its own kernel and skips that kernel on 1. Everything about
 * the graph -- which op this call is, whether the parameter is bound,
 * the stretch the op belongs to -- is the backend's
 * (HtpComputeOps::decode_op_fp32); the layers only hand over the row and
 * their weights. Without ENABLE_HEXKL every call is 0 and the layers are
 * byte-for-byte what they were.
 */

#ifndef __CAUSALLM_HTP_DECODE_HOOK_H__
#define __CAUSALLM_HTP_DECODE_HOOK_H__

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_graph_desc.h>
#endif

namespace causallm {

#ifdef ENABLE_HEXKL
inline int htpDecodeOp(unsigned kind, unsigned pos, const float *in,
                       unsigned in_len, float *out, unsigned out_len,
                       const float *param, unsigned param_len,
                       const float *state, unsigned state_len, float eps) {
  return nntrainer::get_htp_ops()->decode_op_fp32(kind, pos, in, in_len, out,
                                                  out_len, param, param_len,
                                                  state, state_len, eps);
}
#else
inline int htpDecodeOp(unsigned, unsigned, const float *, unsigned, float *,
                       unsigned, const float *, unsigned, const float *,
                       unsigned, float) {
  return 0;
}
#define HTP_OP_RMSNORM 0
#define HTP_OP_CONV1D_GATE 2
#define HTP_OP_QK_NORM 3
#define HTP_OP_ATTN_M1 5
#define HTP_OP_ADD 6
#define HTP_OP_ROUTER_TOPK 7
#define HTP_OP_LM_HEAD 10
#endif

/** @brief RMSNorm of one row of W: y = norm(x) * gamma. */
inline int htpDecodeRmsNorm(unsigned pos, const float *x, float *y,
                            const float *gamma, unsigned W, float eps) {
  return htpDecodeOp(HTP_OP_RMSNORM, pos, x, W, y, W, gamma, W, nullptr, 0,
                     eps);
}

/** @brief Per-head q / k norm of the row q_raw | k_raw | v (len floats)
 *  with gamma = q_gamma | k_gamma (2 x head_dim). On 1 the DSP holds the
 *  normed row for the attention hook and the layer writes nothing. */
inline int htpDecodeQkNorm(unsigned pos, const float *row, unsigned len,
                           const float *gamma, unsigned gamma_len, float eps) {
  return htpDecodeOp(HTP_OP_QK_NORM, pos, row, len, nullptr, 0, gamma,
                     gamma_len, nullptr, 0, eps);
}

/** @brief Causal conv1d (L=3) + gate of one token: abc = a | b | c (3C),
 *  conv_w = w0 | w1 | w2 (3C), state = x_{t-2} | x_{t-1} (2C), y (C). */
inline int htpDecodeConvGate(unsigned pos, const float *abc, float *y,
                             unsigned C, const float *conv_w,
                             const float *state) {
  return htpDecodeOp(HTP_OP_CONV1D_GATE, pos, abc, 3 * C, y, C, conv_w, 3 * C,
                     state, 2 * C, 0.0f);
}

/** @brief Decode attention of one token: qkv = q | k | v (len floats,
 *  post-projection, pre-norm when QK_NORM is resident, else normed),
 *  out (n_heads x head_dim), rope = cos[32] | sin[32] per position. */
inline int htpDecodeAttn(unsigned pos, const float *qkv, unsigned len,
                         float *out, unsigned out_len, const float *rope,
                         unsigned rope_len) {
  return htpDecodeOp(HTP_OP_ATTN_M1, pos, qkv, len, out, out_len, rope,
                     rope_len, nullptr, 0, 0.0f);
}

/** @brief [#132] The residual add of one row: out = residual + addend
 *  (W floats). The DSP holds the residual (its slot 0), so only the
 *  addend travels; on 1 @a out is written only when the ADD ends a
 *  stretch, which the validator rules out -- the next norm hook writes. */
inline int htpDecodeAdd(unsigned pos, const float *addend, float *out,
                        unsigned W) {
  return htpDecodeOp(HTP_OP_ADD, pos, addend, W, out, W, nullptr, 0, nullptr, 0,
                     0.0f);
}

/** @brief [#132] The MoE router of one row: x (K floats, the ffn-normed
 *  row), gate_w the [K][E] gate weight, bias the E expert biases. On 1 the
 *  DSP routes and runs the experts in the same stretch; the layer skips
 *  its router, top-k and experts and writes nothing. */
inline int htpDecodeRouter(unsigned pos, const float *x, unsigned K,
                           const float *gate_w, unsigned E, const float *bias) {
  return htpDecodeOp(HTP_OP_ROUTER_TOPK, pos, x, K, nullptr, 0, gate_w, K * E,
                     bias, E, 0.0f);
}

/** @brief [#132 Part B] The tied lm_head of one row: x (K floats, the
 *  final-normed row), logits (vocab floats). Resident only with every
 *  kind (one stretch per token): this hook is its last op, so on 1 the
 *  DSP ran the whole token and @a logits holds its output. The weight was
 *  bound at load. */
inline int htpDecodeLmHead(unsigned pos, const float *x, unsigned K,
                           float *logits, unsigned vocab) {
  return htpDecodeOp(HTP_OP_LM_HEAD, pos, x, K, logits, vocab, nullptr, 0,
                     nullptr, 0, 0.0f);
}

/** @brief [#132 Part B] Whether the HTP runs the whole decode row at
 *  @a pos (every kind resident, the row handed over): a layer's FC GEMVs
 *  are then discarded work and it skips them. False without HTP. */
inline bool htpDecodeRowResident(unsigned pos) {
#ifdef ENABLE_HEXKL
  return nntrainer::get_htp_ops()->decode_row_resident(pos);
#else
  (void)pos;
  return false;
#endif
}

/** @brief Rows [0, n_rows) of the layer whose attention hook returned 2,
 *  [n_rows][n_kv x head_dim] f32 each. */
inline bool htpDecodeKvSeed(unsigned n_rows, const float *k_rows,
                            const float *v_rows) {
#ifdef ENABLE_HEXKL
  return nntrainer::get_htp_ops()->decode_kv_seed_fp32(n_rows, k_rows, v_rows);
#else
  (void)n_rows;
  (void)k_rows;
  (void)v_rows;
  return false;
#endif
}

} // namespace causallm

#endif /* __CAUSALLM_HTP_DECODE_HOOK_H__ */
