// SPDX-License-Identifier: Apache-2.0
/**
 * @file	htp_graph.h
 * @date	19 August 2026
 * @brief	Op-list graph executor: owns the worker pool, quant/attention
 *		scratch and the optional VTCM slab, and runs the validated
 *		op-list sequentially per forward call
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_GRAPH_H
#define NNTR_HTP_GRAPH_H

#include "htp_ops.h"

/**
 * @brief Graph executor state. Treat as opaque: init/forward/destroy only.
 */
struct htp_graph {
  struct nntr_htp_oplist_header cfg;  /**< header copy */
  const struct nntr_htp_op_desc *ops; /**< points into the caller's oplist */
  struct htp_exec_ctx ctx;            /**< owns pool + scratch pointers */
  unsigned vtcm_ctx_id;               /**< HAP compute-res id, 0 = no VTCM */
  uint64_t stream_bytes; /**< weight bytes every forward call streams: sum
                            of n*k over the W8A8 / LOGITS / W8A16 ops (the
                            denominator of the weight-stream GB/s) */
  uint32_t *next_mm;     /**< [n_ops] index of the next MATMUL_W8A8 / LOGITS op
                            after op i (UINT32_MAX = none), built at init for
                            the cross-op weight prefetch (hvx-matmul.c) */
};

/**
 * @brief Validate the op-list and set up pool, scratch and VTCM.
 * @return 0 ok, non-zero on validation or allocation failure
 */
int htp_graph_init(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                   uint8_t *weights, uint32_t wsize, uint8_t *kv,
                   uint32_t kvsize, uint8_t *act, uint32_t actsize);

/**
 * @brief Same as htp_graph_init, with an explicit worker-pool size.
 * @param n_workers <= 0 picks the HVX-unit count; larger values are clamped
 *        to it by wp_create (a worker without a unit would block forever on
 *        qurt_hvx_lock). test_graph exercises a forced 2-worker pool; the
 *        sim profile test passes the argv count through.
 */
int htp_graph_init_ex(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                      uint8_t *weights, uint32_t wsize, uint8_t *kv,
                      uint32_t kvsize, uint8_t *act, uint32_t actsize,
                      int n_workers);

/**
 * @brief Run one chunk of n_tokens starting at sequence position pos.
 * @return 0 ok, non-zero on bad runtime arguments
 */
int htp_graph_forward(struct htp_graph *g, const int32_t *tokens,
                      uint32_t n_tokens, uint32_t pos, float *logits,
                      uint32_t n_logits);

/**
 * @brief Release VTCM, scratch and the worker pool.
 */
/**
 * @brief Run ops [0, n_ops_limit) of one chunk. n_ops_limit == n_ops runs the
 *        whole list (identical to htp_graph_forward); larger is rejected.
 * @param pcycles optional out: HAP_perf_get_pcycles() delta over the op loop
 * @return 0 ok, non-zero on bad runtime arguments
 */
int htp_graph_forward_upto(struct htp_graph *g, const int32_t *tokens,
                           uint32_t n_tokens, uint32_t pos, float *logits,
                           uint32_t n_logits, uint32_t n_ops_limit,
                           uint64_t *pcycles);

/**
 * @brief Resolve a (buf id, offset, bytes) triple against the mapped
 *        WEIGHTS/KV/ACT buffers (TOKENS/LOGITS are per-call arguments).
 * @return pointer, or 0 when out of bounds / not a mapped buffer
 */
const uint8_t *htp_graph_buf_ref(const struct htp_graph *g, uint32_t buf,
                                 uint32_t offset, uint32_t bytes);

/**
 * @brief Zero the per-kind pcycle/call counters (see htp_exec_ctx).
 */
void htp_graph_profile_reset(struct htp_graph *g);

/**
 * @brief Copy out the per-kind pcycle and call counters accumulated by
 *        htp_graph_forward_upto since init or the last reset.
 */
void htp_graph_profile_get(const struct htp_graph *g,
                           uint64_t cycles[NNTR_HTP_OP_KIND_COUNT],
                           uint32_t calls[NNTR_HTP_OP_KIND_COUNT]);

/**
 * @brief Copy out the per-op-index pcycles (op-list order, n_ops entries)
 *        accumulated since init or the last reset. Same window as
 *        htp_graph_profile_get; a kind's counter is the sum of its ops.
 * @return 0 ok, 1 when n is smaller than the op count
 */
int htp_graph_profile_get_ops(const struct htp_graph *g, uint64_t *cycles,
                              uint32_t n);

void htp_graph_destroy(struct htp_graph *g);

/**
 * @brief Allocate one DMA queue (HTP_MM_DMA_QUEUE_CAP deep) and one prefetch
 *        record per worker on c->dmaq / c->pf, backed by c->dmaq_mem. Needs
 *        c->vtcm / c->vtcm_size set (the queues carry the VTCM range for the
 *        bypass decision). Called by htp_graph_init_ex; exported so
 *        test_matmul_dma allocates the same way.
 * @return 0 ok, 1 on allocation failure (nothing is left allocated)
 */
int htp_graph_dma_init(struct htp_exec_ctx *c, int n_workers);

/**
 * @brief Drain every worker's queue on the worker threads (one wp_run job):
 *        no descriptor may be in flight when VTCM is released or a test
 *        re-checks the pending-prefetch state. No-op without queues.
 */
void htp_graph_dma_flush(struct htp_exec_ctx *c);

/**
 * @brief Free what htp_graph_dma_init allocated (flush first).
 */
void htp_graph_dma_destroy(struct htp_exec_ctx *c);

#endif /* NNTR_HTP_GRAPH_H */
