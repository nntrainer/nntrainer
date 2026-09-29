// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_graph.h
 * @date   23 Sep 2026
 * @brief  The session's validated decode op table and its forward loop
 *         (#85), with the small ops and m=1 attention resident (#130),
 *         the residual add and the router (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * "The DSP owns the graph" (LEDGER section 4, hvx_impl's htp_graph.c
 * lifted to this tree's kernels): graph_init validates the op list once
 * (htp_graph_desc.h) and binds its weight handles against the session's
 * table; forward then runs `for op from start: if !resident break;
 * table[kind]()` and reports where it stopped. The kernel table holds
 * MOE (hexkl_mm_u8i4_moe_layer_run unchanged), RMSNORM / QK_NORM / ROPE /
 * CONV1D_GATE (#82's hvx_m1_ops_f32.c), ATTN_M1 (#81's
 * hvx_attn_m1_f32.c over the session's cache, borrowed through the env),
 * ADD (hvx_scale_add_rows_f32 at scale 1 into slot 0, the residual) and
 * ROUTER_TOPK (hvx_router_topk_f32, whose routing the next MOE op
 * consumes in the same call; #132). The small ops' parameters -- gammas,
 * conv weights, the conv state seed, the RoPE table, the router weight
 * and bias -- are bound once after init through
 * hexkl_graph_set_param (plan 130 section 3.1); forward refuses an op
 * whose parameter is missing with AEE_EBADSTATE.
 *
 * [#132 Part B] FC, DENSE_FFN and LM_HEAD: the Android CPU's M=1 Q4_0 FC
 * bit for bit (q4_gemv_cpu_det.h) -- hvx_q4m1_prep quantizes the op's
 * input once (hvx_intrin, = q8_0_quant_cpu_det) and the session's FC
 * runner (the env's fc callback: nntr_hvx_fc_q4.c's lanes and weight
 * feed) runs each Q4M1 part on it. DENSE_FFN is up, gate,
 * m1_swiglu_cpu_det (neon::swiglu's order), a second quantization and
 * down; LM_HEAD is its slices into the graph's logits buffer, then
 * m1_argmax_first (std::max_element) into lm_id. The weights are Q4M1
 * handles of the session's table bound in the op record before
 * graph_init (htp_graph_desc.h), checked there against the shapes the
 * caller passes.
 */

#ifndef __NNTRAINER_HEXKL_GRAPH_H__
#define __NNTRAINER_HEXKL_GRAPH_H__

#include <stdint.h>

#include "../htp_graph_desc.h" /* beside hmx/, on every include path */
#include "hexkl_mm_u8i4_moe.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_q4_gemv_f32.h"

/** @brief Largest K a Q4M1 op quantizes (the validator's rule; the model's
 *  widest is 7168, the dense down). */
#define HEXKL_GRAPH_Q4M1_MAX_K 8192u

/**
 * @brief The session's FC runner (#132 Part B): y (the slot's N floats)
 *        = weight @a h (a Q4M1 handle of the session's table) times the
 *        prepared activation @a a, with @a feed the op's (0 VTCM when it
 *        fits, else the L2 scratch; 1 the L2 scratch).
 * @return 0, or the runner's code (AEE_EEXPIRED: a weight DMA never came)
 */
typedef int (*hexkl_graph_fc_fn)(void *ctx, uint32_t h, uint32_t feed,
                                 const hvx_q4m1_act *a, float *y);

/** @brief One Q4M1 slot's shape for hexkl_graph_init (K 0 = free). */
typedef struct {
  uint32_t K, N;
} hexkl_graph_q4m1_shape;

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
  hvx_attn_m1_ctx *attn_m1; /**< the session's m=1 KV cache (#81), borrowed;
                                 NULL = none, and a resident ATTN_M1 op
                                 fails with AEE_EBADSTATE */
  hexkl_graph_fc_fn fc;     /**< [#132 Part B] the Q4M1 kinds' runner; NULL
                                 = none, and they fail with AEE_EBADSTATE */
  void *fc_ctx;
} hexkl_graph_env;

/** @brief The MoE routing of this token, in mm_u8i4_moe_layer's layout:
 *  the start op's per-call side input while the router stays on the ARM
 *  (empty -- n_rows 0, n_experts 0 -- otherwise), or the graph's own
 *  route_* record once a resident ROUTER_TOPK op has filled it (#132). */
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
  float *rope_cs;      /**< [max_seq][64] cos | sin, or NULL until bound */
  htp_graph_op ops[HTP_GRAPH_MAX_OPS];
  uint64_t op_pcycles[HTP_GRAPH_MAX_OPS]; /**< of the last forward */
  float *param[HTP_GRAPH_MAX_OPS];     /**< gamma or conv_w, NULL until bound */
  float *state[HTP_GRAPH_MAX_OPS];     /**< CONV1D_GATE: 3 x N (rows 0-1 the
                                            conv state, row 2 scratch) */
  uint32_t ordinal[HTP_GRAPH_MAX_OPS]; /**< ATTN_M1: the attention-layer
                                            index the cache is keyed by */
  /** The last ROUTER_TOPK op's routing (#132), in expert order: rewritten
   *  by every router op, read by the MOE op after it. */
  uint32_t route_idx[HTP_GRAPH_MAX_EXPERTS];
  uint32_t route_cnt[HTP_GRAPH_MAX_EXPERTS];
  float route_w[HTP_GRAPH_MAX_EXPERTS];
  /* [#132 Part B] the Q4M1 kinds' state, allocated at init only when one
     is resident: the part widths, the quantized activation, the dense
     FFN's up | gate | swiglu rows, the logits and their argmax */
  hexkl_graph_q4m1_shape *q4m1; /**< n_q4m1 slot shapes, copied at init */
  uint32_t n_q4m1;
  hvx_q4m1_act act; /**< its arrays in act_buf, HEXKL_GRAPH_Q4M1_MAX_K */
  uint8_t *act_buf;
  float *ffn;     /**< 3 x the widest DENSE_FFN N */
  float *logits;  /**< vocab floats: LM_HEAD's output, not a slot */
  uint32_t lm_id; /**< m1_argmax_first of the last LM_HEAD's logits */
} hexkl_graph;

/** @brief HTP_GRAPH_KIND_BIT mask of the kinds whose table slot is
 *  non-NULL in this build. */
uint32_t hexkl_graph_resident_kinds(void);

/**
 * @brief Validates the words, checks every resident MoE op's handles
 *        against @a tbl (in use, gate_up K x 2N, down N x N_out) and every
 *        resident Q4M1 op's against @a q4m1 (#132 Part B: each part in
 *        range and in use, K the op's, an FC's or LM_HEAD's part widths
 *        multiples of 32 summing to N, a DENSE_FFN's up and gate K x N and
 *        down N x N_out), and keeps a copy.
 * @param q4m1   the session's Q4M1 slot shapes (may be NULL when @a n_q4m1
 *               is 0: then a resident Q4M1 op is refused)
 * @return 0 or htp_graph_validate's code; HTP_GRAPH_E_INVHANDLE for a
 *         handle that is out of range, free or of the wrong shape;
 *         AEE_ENOMEMORY when the heap refuses
 */
int hexkl_graph_init(const uint32_t *words, uint32_t n_words,
                     const hexkl_weight_u8i4_table *tbl,
                     const hexkl_graph_q4m1_shape *q4m1, uint32_t n_q4m1,
                     hexkl_graph **out);

/** @brief Frees the table, the slots and every bound parameter. Safe on
 *  NULL. The KV cache is the session's, not the graph's. */
void hexkl_graph_free(hexkl_graph *g);

/**
 * @brief Binds one f32 parameter of op @a op (HTP_GRAPH_PARAM_*, the
 *        lengths in htp_graph_desc.h). Copies @a data to the DSP heap;
 *        a second call replaces (CONV_STATE: re-seeds rows 0-1).
 * @return 0, AEE_EBADSTATE (no graph), HTP_GRAPH_E_BADITEM (op or which
 *         out of range), HTP_GRAPH_E_INVALIDFORMAT (the op's kind does
 *         not take @a which, or @a n is not its length), AEE_ENOMEMORY
 */
int hexkl_graph_set_param(hexkl_graph *g, uint32_t op, uint32_t which,
                          const float *data, uint32_t n);

/** @brief Whether any resident MoE op names @a handle: weight_release
 *  refuses such a handle with AEE_EBADSTATE while the graph lives. */
int hexkl_graph_uses_handle(const hexkl_graph *g, uint32_t handle);

/** @brief [#132 Part B] The same for a Q4M1 handle (q4m1_release). */
int hexkl_graph_uses_q4m1(const hexkl_graph *g, uint32_t handle);

/**
 * @brief Runs ops [start_op, ...) while they are resident, at most
 *        @a n_ops_limit of them.
 *
 * act_in (act_in_len == the start op's input width) is copied into the
 * start op's in_slot; when at least one op ran, the last op's out_slot is
 * copied to act_out (act_out_len == its output width; an LM_HEAD's
 * logits come from the graph's logits buffer) -- the f32 wrapper
 * of doc 45 section 3.1 that goes away when the ARM stops touching
 * activations. When no op ran (the start op is not resident) act_out is
 * untouched, *resume_at == start_op and every op_pcycles entry is 0.
 * @a routing is consumed by the first MoE op run; a second MoE op in the
 * same call, or a MoE op with no routing, fails with AEE_EBADSTATE.
 * @return 0, AEE_EBADSTATE (no graph, routing; a RMSNORM / QK_NORM /
 *         CONV1D_GATE / ROUTER_TOPK op with no parameter or state bound, a ROPE
 * op with no table, an ATTN_M1 op with no cache in @a env, a Q4M1 op with
 * no fc runner in @a env, or the cache
 *         kernel's own hole), HTP_GRAPH_E_BADITEM (start_op past the
 *         list, pos >= max_seq), HTP_GRAPH_E_INVALIDFORMAT (an act length
 *         or the routing's shape disagrees with the op), or the kernel's
 *         own code
 */
int hexkl_graph_forward(hexkl_graph *g, const hexkl_graph_env *env,
                        uint32_t start_op, uint32_t n_ops_limit, uint32_t pos,
                        const hexkl_graph_routing *routing, const float *act_in,
                        uint32_t act_in_len, float *act_out,
                        uint32_t act_out_len, uint32_t *resume_at);

#endif /* __NNTRAINER_HEXKL_GRAPH_H__ */
