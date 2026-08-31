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
};

/**
 * @brief Validate the op-list and set up pool, scratch and VTCM.
 * @return 0 ok, non-zero on validation or allocation failure
 */
int htp_graph_init(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                   uint8_t *weights, uint32_t wsize, uint8_t *kv,
                   uint32_t kvsize, uint8_t *act, uint32_t actsize);

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

void htp_graph_destroy(struct htp_graph *g);

#endif /* NNTR_HTP_GRAPH_H */
