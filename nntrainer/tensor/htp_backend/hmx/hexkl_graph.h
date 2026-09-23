// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_graph.h
 * @date   23 Sep 2026
 * @brief  The session's validated decode op table and its forward loop (#85)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * "The DSP owns the graph" (LEDGER section 4, hvx_impl's htp_graph.c
 * lifted to this tree's kernels): graph_init validates the op list once
 * (htp_graph_desc.h) and binds its weight handles against the session's
 * table; forward then runs `for op from start: if !resident break;
 * table[kind]()` and reports where it stopped. The kernel table has one
 * non-NULL slot in this issue, MOE, which calls hexkl_mm_u8i4_moe_layer_run
 * unchanged; #81 and #82 fill ATTN_M1 and the small ops.
 */

#ifndef __NNTRAINER_HEXKL_GRAPH_H__
#define __NNTRAINER_HEXKL_GRAPH_H__

#include <stdint.h>

#include "../htp_graph_desc.h" /* beside hmx/, on every include path */
#include "hexkl_mm_u8i4_moe.h"

/** @brief What a kernel needs from the session, handed per call so the
 *  graph holds no pointer into the session. */
typedef struct {
  hexkl_weight_u8i4_table *tbl;
  uint8_t *vtcm_base;
  uint32_t vtcm_size;
  uint32_t config_off;
  hvx_worker_pool *pool;
  hexkl_moe_scratch *scratch;
  uint32_t moe_flags;
} hexkl_graph_env;

/** @brief The start op's per-call side input while the router stays on
 *  the ARM: the MoE routing of this token, in mm_u8i4_moe_layer's layout.
 *  Empty (n_rows 0, n_experts 0) once ROUTER_TOPK is resident. */
typedef struct {
  const uint32_t *row_index;
  const uint32_t *row_count;
  const float *row_weight;
  uint32_t n_rows;
  uint32_t n_experts;
} hexkl_graph_routing;

typedef struct {
  uint32_t n_layers, n_ops, hidden, vocab, max_seq;
  uint32_t slot_words; /**< f32 per activation slot: the widest resident
                            op's in or out width */
  float *slots;        /**< HTP_GRAPH_N_SLOTS x slot_words, DSP heap */
  htp_graph_op ops[HTP_GRAPH_MAX_OPS];
  uint64_t op_pcycles[HTP_GRAPH_MAX_OPS]; /**< of the last forward */
} hexkl_graph;

/** @brief HTP_GRAPH_KIND_BIT mask of the kinds whose table slot is
 *  non-NULL in this build. */
uint32_t hexkl_graph_resident_kinds(void);

/**
 * @brief Validates the words, checks every resident MoE op's handles
 *        against @a tbl (in use, gate_up K x 2N, down N x N_out) and
 *        keeps a copy.
 * @return 0 or htp_graph_validate's code; HTP_GRAPH_E_INVHANDLE for a
 *         handle that is out of range, free or of the wrong shape;
 *         AEE_ENOMEMORY when the heap refuses
 */
int hexkl_graph_init(const uint32_t *words, uint32_t n_words,
                     const hexkl_weight_u8i4_table *tbl, hexkl_graph **out);

/** @brief Frees the table and the slots. Safe on NULL. */
void hexkl_graph_free(hexkl_graph *g);

/** @brief Whether any resident MoE op names @a handle: weight_release
 *  refuses such a handle with AEE_EBADSTATE while the graph lives. */
int hexkl_graph_uses_handle(const hexkl_graph *g, uint32_t handle);

/**
 * @brief Runs ops [start_op, ...) while they are resident, at most
 *        @a n_ops_limit of them.
 *
 * act_in (act_in_len == the start op's input width) is copied into the
 * start op's in_slot; when at least one op ran, the last op's out_slot is
 * copied to act_out (act_out_len == its output width) -- the f32 wrapper
 * of doc 45 section 3.1 that goes away when the ARM stops touching
 * activations. When no op ran (the start op is not resident) act_out is
 * untouched, *resume_at == start_op and every op_pcycles entry is 0.
 * @a routing is consumed by the first MoE op run; a second MoE op in the
 * same call, or a MoE op with no routing, fails with AEE_EBADSTATE.
 * @return 0, AEE_EBADSTATE (no graph, routing), HTP_GRAPH_E_BADITEM
 *         (start_op past the list, pos >= max_seq),
 *         HTP_GRAPH_E_INVALIDFORMAT (an act length or the routing's shape
 *         disagrees with the op), or the kernel's own code
 */
int hexkl_graph_forward(hexkl_graph *g, const hexkl_graph_env *env,
                        uint32_t start_op, uint32_t n_ops_limit, uint32_t pos,
                        const hexkl_graph_routing *routing, const float *act_in,
                        uint32_t act_in_len, float *act_out,
                        uint32_t act_out_len, uint32_t *resume_at);

#endif /* __NNTRAINER_HEXKL_GRAPH_H__ */
