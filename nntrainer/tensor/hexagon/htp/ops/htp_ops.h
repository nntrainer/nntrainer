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
#if HTP_HMX
#include "htp_hmx.h"
#endif

struct dma_queue_s; /* dma-queue.h; only hvx-matmul.c and htp_graph.c use it */

/** Ring depth of each worker's DMA queue (graph lifetime, one per worker):
 * at most the in-op "chunk c+1" kick plus one cross-op prefetch are ever
 * outstanding, so the 3 usable slots of a 4-deep ring suffice. Must be a
 * power of two (dma_queue_init rounds up regardless). */
#define HTP_MM_DMA_QUEUE_CAP 4u

/**
 * @brief Per-worker record of a weight chunk kicked ahead of its op (the
 *        cross-op prefetch of hvx-matmul.c, issue #25).
 */
struct htp_mm_prefetch {
  const struct nntr_htp_op_desc *desc; /**< op whose chunk 0 is in flight,
                                          NULL = nothing pending */
  uint8_t *buf;                        /**< VTCM half-slab the chunk lands in */
  uint32_t rows;                       /**< rows in flight */
  uint32_t hits; /**< diagnostics: ops that found their chunk 0 in flight */
};

/**
 * @brief Per-forward-call execution state passed to every op kernel.
 */
struct htp_exec_ctx {
  uint8_t *buf[NNTR_HTP_BUF_COUNT];
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  const struct nntr_htp_oplist_header *cfg;
  uint32_t n_tokens, pos;
  struct wp_pool *pool;
  int8_t *xq;          /**< per-token quant scratch: int8 [max_chunk][k_max] for
                          W8A8/LOGITS or int16 [max_chunk][k_max] for W8A16
                          (allocated 2*max_chunk*k_max bytes) */
  float *xq_scale;     /**< [max_chunk] */
  float *attn_scratch; /**< [n_workers][max_seq] fp32 scores */
  uint8_t *vtcm;       /**< per-worker weight double buffer for the tiled matmul
                          (hvx-matmul.c) */
  uint32_t vtcm_size;
  struct dma_queue_s **dmaq;  /**< [n_workers] graph-lifetime DMA queues,
                                 NULL when VTCM is absent
                                 (htp_graph_dma_init) */
  void *dmaq_mem;             /**< one block backing every queue */
  struct htp_mm_prefetch *pf; /**< [n_workers] pending cross-op prefetch */
  const struct nntr_htp_op_desc *next_mm; /**< the next tiled matmul
                                  (W8A8 / LOGITS) after the running op in
                                  this call, NULL if none; set per op by
                                  htp_graph_forward_upto */
  /** Per-op-kind profile, accumulated by htp_graph_forward_upto around each
   * op call; read/cleared through htp_graph_profile_get/reset. Sim-side
   * instrumentation only: not part of the RPC ABI. */
  uint64_t prof_cycles[NNTR_HTP_OP_KIND_COUNT];
  uint32_t prof_calls[NNTR_HTP_OP_KIND_COUNT];
  uint64_t *prof_op_cycles; /**< [n_ops] per-op-index pcycles, same window as
                                 prof_cycles (kind = sum over its ops). */
#if HTP_HMX
  struct htp_hmx hmx; /**< HMX arena above the HVX slabs, worker-0 lock,
                           HexKL version (hmx/htp_hmx.h); base == NULL
                           when the session has no HMX */
#endif
};

typedef void (*htp_op_fn)(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d);

/** @brief Resolve a tensor ref to its byte address inside ctx buffers. */
static inline uint8_t *htp_ref_ptr(struct htp_exec_ctx *c,
                                   struct nntr_htp_tensor_ref r) {
  return c->buf[r.buf] + r.offset;
}

/** @brief Op row count for this call (nntr_htp_op_rows: d->m or n_tokens). */
static inline uint32_t htp_m(struct htp_exec_ctx *c,
                             const struct nntr_htp_op_desc *d) {
  return nntr_htp_op_rows(d, c->n_tokens);
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

/* Defined in htp_graph.c: htp_op_table[d->kind] dispatches. */
extern const htp_op_fn htp_op_table[NNTR_HTP_OP_KIND_COUNT];

#endif /* NNTR_HTP_OPS_H */
