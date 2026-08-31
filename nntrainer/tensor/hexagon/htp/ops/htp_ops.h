// SPDX-License-Identifier: Apache-2.0
/**
 * @file	htp_ops.h
 * @date	18 August 2026
 * @brief	Execution context and op entry points shared by all HTP op kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_OPS_H
#define NNTR_HTP_OPS_H

#include <stdint.h>

#include "nntr_htp_common.h"
#include "worker_pool.h"

/**
 * @brief Per-forward-call execution state passed to every op kernel.
 */
struct htp_exec_ctx {
  uint8_t *buf[NNTR_HTP_BUF_COUNT];
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  const struct nntr_htp_oplist_header *cfg;
  uint32_t n_tokens, pos;
  struct wp_pool *pool;
  int8_t *xq;          /**< per-token quant scratch [max_chunk][k_max] */
  float *xq_scale;     /**< [max_chunk] */
  float *attn_scratch; /**< [n_workers][max_seq] fp32 scores */
  uint8_t *vtcm;       /**< Task 7 onward, split per worker */
  uint32_t vtcm_size;
};

typedef void (*htp_op_fn)(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d);

/** @brief Resolve a tensor ref to its byte address inside ctx buffers. */
static inline uint8_t *htp_ref_ptr(struct htp_exec_ctx *c,
                                   struct nntr_htp_tensor_ref r) {
  return c->buf[r.buf] + r.offset;
}

/** @brief Op row count: d->m if set, else the current chunk's n_tokens. */
static inline uint32_t htp_m(struct htp_exec_ctx *c,
                             const struct nntr_htp_op_desc *d) {
  return d->m ? d->m : c->n_tokens;
}

void hvx_op_embed(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);
void hvx_op_rmsnorm(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);
void hvx_op_matmul_w8a8(struct htp_exec_ctx *c,
                        const struct nntr_htp_op_desc *d);
void hvx_op_rope(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);
void hvx_op_attn(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);
void hvx_op_silu_mul(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);
void hvx_op_add(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d);
void hvx_op_matmul_logits(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d);
void hvx_op_matmul_w8a16(struct htp_exec_ctx *c,
                         const struct nntr_htp_op_desc *d);

/* Defined in Task 14's htp_graph.c: htp_op_table[d->kind] dispatches. */
extern const htp_op_fn htp_op_table[NNTR_HTP_OP_KIND_COUNT];

#endif /* NNTR_HTP_OPS_H */
