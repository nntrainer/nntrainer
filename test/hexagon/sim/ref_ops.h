// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ref_ops.h
 * @date	18 August 2026
 * @brief	Scalar C reference implementations for hexagon-sim primitive tests
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef REF_OPS_H
#define REF_OPS_H

#include <stdint.h>

/* __fp16 is an ARM/Hexagon spelling. On an x86 host the reference executor
 * is compiled as C++ and __fp16 becomes a conversion struct with the same
 * fp16 round trip (see ref_fp16_x86.h). */
#if defined(__cplusplus) && !defined(__hexagon__) && !defined(__aarch64__) &&  \
  !defined(__arm__)
#include "ref_fp16_x86.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Same formula as htp_quant_row_fp16, so results are bit-exact. */
float ref_quant_row(const __fp16 *x, int8_t *q, uint32_t k);

/* Naive scalar int32 dot product. */
int32_t ref_dot_i8(const int8_t *w, const int8_t *x, uint32_t k);

/* Reference for MATMUL_W8A8: x fp16[m][k], w int8[n][k], sw fp32[n],
 * y fp16[m][n]. Quantizes each x row with ref_quant_row, then dots. */
void ref_matmul_w8a8(const __fp16 *x, const int8_t *w, const float *sw,
                     __fp16 *y, uint32_t m, uint32_t k, uint32_t n);

/* Reference for MATMUL_W8A16: like W8A8 but x stays fp16 (no per-token
 * quantization), fp32 accumulation, y = (fp16)(dot * sw[n]). */
void ref_matmul_w8a16(const __fp16 *x, const int8_t *w, const float *sw,
                      __fp16 *y, uint32_t m, uint32_t k, uint32_t n);

/* Reference for RMSNORM: x/gamma/y fp16[m][n] (gamma repeats every `chunk`
 * elements, chunk == n for whole-row or head_dim for PER_HEAD QK-Norm).
 * Per row, per chunk: r = 1/sqrt(mean(x^2) + eps), y_i = x_i * r * gamma. */
void ref_rmsnorm(const __fp16 *x, const __fp16 *gamma, __fp16 *y, uint32_t m,
                 uint32_t n, uint32_t chunk, float eps);

/* Fills a ROPE cos/sin table: table[p][i] = cos(p*inv_freq_i),
 * table[p][64+i] = sin(p*inv_freq_i), inv_freq_i = theta^(-2i/128),
 * for p in [0,max_seq), i in [0,64). Row stride is 128 halves. */
void ref_rope_table_fill(__fp16 *table, uint32_t max_seq, float theta);

/* Reference for ROPE: x fp16[m][heads*128] in place, table fp16[*][128]
 * (row = cos[64] || sin[64]). Per token t, head, i<64: x0=x[i], x1=x[i+64]
 * rotated using the table row at pos+t (rotate-half pair (i, i+64)). */
void ref_rope(__fp16 *x, const __fp16 *table, uint32_t m, uint32_t heads,
              uint32_t pos);

/* Reference for ADD: y = a + b, fp16[count]. */
void ref_add(const __fp16 *a, const __fp16 *b, __fp16 *y, uint32_t count);

/* Reference for SILU_MUL: silu(x) = x / (1 + expf(-x)) in fp32,
 * y = (fp16)(silu(g) * u), fp16[count]. */
void ref_silu_mul(const __fp16 *g, const __fp16 *u, __fp16 *y, uint32_t count);

/* Reference for ATTN (fp32 scalar): q fp16[m][n_heads*hd],
 * k/v fp16[m][n_kv_heads*hd], out fp16[m][n_heads*hd]. kv is the fp16 KV
 * cache base: K region [n_layers][n_kv_heads][max_seq][hd] followed by an
 * identically sized V region (the kernel stores its K region transposed,
 * [n_layers][n_kv_heads][hd][max_seq]; the buffer is DSP-private). Appends
 * k/v at [layer][h][pos+t], then causal SDPA (token t attends to positions
 * [0, pos+t]) with GQA mapping h_kv = h_q / (n_heads/n_kv_heads) and
 * softmax(scale * q.K). */
void ref_attn(const __fp16 *q, const __fp16 *k, const __fp16 *v, __fp16 *kv,
              __fp16 *out, uint32_t m, uint32_t pos, uint32_t layer,
              uint32_t n_layers, uint32_t n_heads, uint32_t n_kv_heads,
              uint32_t hd, uint32_t max_seq, float scale);

/* Reference for MATMUL_LOGITS: x_last fp16[k] (the last token row),
 * w int8[n][k], sw fp32[n], out fp32[n]. Quantizes x_last with
 * ref_quant_row (same formula as the kernel = bit-exact), then dots. */
void ref_matmul_logits(const __fp16 *x_last, const int8_t *w, const float *sw,
                       float *out, uint32_t k, uint32_t n);

/* Reference for EMBED: tokens int32[m], w int8[vocab][k], scale fp32[vocab],
 * y fp16[m][k]. y[t][i] = (fp16)(w[tokens[t]][i] * scale[tokens[t]]). */
void ref_embed(const int32_t *tokens, const int8_t *w, const float *scale,
               __fp16 *y, uint32_t m, uint32_t k);

/* Scalar reference graph executor: interprets the same op-list bytes as
 * htp_graph_forward over caller-provided copies of the weights/kv/act
 * buffers, dispatching each op to the ref_* function above. Activations
 * round-trip through fp16 at every op boundary (the ref_* signatures are
 * already fp16 in/out). No validation: pass an op-list that
 * htp_graph_init already accepted. */
void ref_graph_forward(const uint8_t *oplist, uint8_t *weights, uint8_t *kv,
                       uint8_t *act, const int32_t *tokens, uint32_t n_tokens,
                       uint32_t pos, float *logits);

/* Same, but runs only ops [0, n_ops_limit) (n_ops_limit >= n_ops runs the
 * whole list). Used to dump intermediate activations. */
void ref_graph_forward_upto(const uint8_t *oplist, uint8_t *weights,
                            uint8_t *kv, uint8_t *act, const int32_t *tokens,
                            uint32_t n_tokens, uint32_t pos, float *logits,
                            uint32_t n_ops_limit);

#ifdef __cplusplus
}
#endif

#endif /* REF_OPS_H */
