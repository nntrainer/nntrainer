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
 * [plan 201 S1] A MOE op's expert handles are its EXPERTS table, bound
 * after init like a parameter and replaceable at any time: the expert
 * pool's (layer, expert) -> handle map, HTP_GRAPH_NO_HANDLE for an expert
 * that is not resident. forward hands the table to the kernel as its
 * handle arrays (the kernel skips an expert with no rows before it reads
 * its handle) and refuses a token that routes to a non-resident expert.
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
 *
 * [#225] An FC or DENSE_FFN with HTP_GRAPH_FEED_WH reads WH handles of the
 * session's u8i4 table instead (the prefill's, from the FC WH sidecar):
 * the FC through hexkl_mm_u8i4_fc_m1_run, the DENSE_FFN through
 * hexkl_mm_u8i4_moe_layer_run as its chunks at weight 1 -- the kernels
 * the MOE op calls, so no fc runner is involved and the LM_HEAD alone
 * stays Q4M1.
 */

#ifndef __NNTRAINER_HEXKL_GRAPH_H__
#define __NNTRAINER_HEXKL_GRAPH_H__

#include <stdint.h>

#include "../htp_graph_desc.h" /* beside hmx/, on every include path */
#include "hexkl_mm_u8i4_moe.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_q4_gemv_f32.h"

/** @brief [#261] attn_m1_det.h's ATTN_M1_PROF_WORDS (hexkl_graph.c asserts
 *  they agree): the size of hexkl_graph.attn_prof's rows. */
#define HEXKL_GRAPH_ATTN_PROF_WORDS 14u

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

struct hexkl_graph_s;

/**
 * @brief [plan 201 S1] The session's in-place rebind of a pool slot (the
 *        miss path): the expert whose bytes the ARM just read into the
 *        arena at (@a arena, @a off_gu / @a off_dn) takes over the pair
 *        @a old_gu / @a old_dn (HTP_GRAPH_NO_HANDLE: a fresh pair), whose
 *        numbers come back in @a h_gu / @a h_dn
 *        (nntr_hvx_weight_swap_u8i4_arena). [plan 229] Non-NULL @a pal_gu /
 *        @a pal_dn (four int4 codes each) make the pair QS2CX_WH
 *        (nntr_hvx_weight_swap_batch_u2i4_arena).
 */
typedef int (*hexkl_graph_rebind_fn)(
  void *ctx, uint32_t old_gu, uint32_t old_dn, uint32_t K, uint32_t inter,
  uint32_t N_out, uint32_t arena, uint32_t off_gu, uint32_t off_dn,
  const int8_t *pal_gu, const int8_t *pal_dn, uint32_t *h_gu, uint32_t *h_dn);

/**
 * @brief [plan 201 S1] The expert pool's miss round, which the token driver
 *        provides (hexkl_token.c): @a post names MOE op @a op's routed
 *        experts and the ones its EXPERTS table lacks and returns at once;
 *        @a wait blocks until the pool's owner answers and applies the
 *        answer to @a g's tables. The MOE op runs its present experts
 *        between the two.
 */
typedef struct {
  int (*post)(void *ctx, uint32_t op, const uint32_t *routed, uint32_t n_routed,
              const uint32_t *miss, uint32_t n_miss);
  int (*wait)(void *ctx, struct hexkl_graph_s *g, uint32_t op);
  void *ctx;
  /** [#282 B] graph_predict's guess for ROUTER_TOPK op @a op, posted
   *  without a wait (htp_pred_hint); NULL = not posted */
  void (*hint)(void *ctx, uint32_t op, const uint32_t *ids, uint32_t n);
} hexkl_graph_miss;

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
  hvx_attn_m1_ctx *attn_m1;   /**< the session's m=1 KV cache (#81), borrowed;
                                   NULL = none, and a resident ATTN_M1 op
                                   fails with AEE_EBADSTATE */
  hvx_attn_m1_ctx *attn_m1_b; /**< [plan 201 S4] a second cache of another
                                   shape (Gemma 4's full layers beside its
                                   sliding ones), borrowed; an ATTN_M1 op
                                   reads the one of its (n_kv, gqa,
                                   head_dim), NULL = none */
  hexkl_graph_fc_fn fc;       /**< [#132 Part B] the Q4M1 kinds' runner; NULL
                                   = none, and they fail with AEE_EBADSTATE */
  void *fc_ctx;
  hexkl_graph_rebind_fn rebind; /**< [plan 201 S1] NULL = none */
  void *rebind_ctx;
  hexkl_graph_miss miss; /**< [plan 201 S1] post NULL = no miss path: a
                              token routed to a non-resident expert fails
                              with AEE_EBADSTATE */
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

/** @brief [plan 201 S1] The most experts one MOE op may route a token to
 *  (the M = 1 path's MOE_M1_MAX_EXPERTS), what a miss round names. */
#define HEXKL_GRAPH_MISS_MAX 16u

/** @brief [plan 201 S1] The token's routed sets, S1's response carries
 *  them to the pool's owner: per MOE op run, its count then its ids. */
#define HEXKL_GRAPH_ROUTE_LOG 320u

typedef struct hexkl_graph_s {
  uint32_t n_layers, n_ops, hidden, vocab, max_seq;
  uint32_t slot_words; /**< f32 per activation slot: the widest resident
                            op's in or out width */
  float *slots;        /**< HTP_GRAPH_N_SLOTS x slot_words, DSP heap */
  float *rope_cs;      /**< [max_seq][64] cos | sin, or NULL until bound;
                            [plan 201 S4] a ROPE op with its own table
                            (param[op], max_seq x head_dim) reads that */
  htp_graph_op ops[HTP_GRAPH_MAX_OPS];
  uint64_t op_pcycles[HTP_GRAPH_MAX_OPS]; /**< of the last forward */
  uint32_t op_qt[HTP_GRAPH_MAX_OPS];   /**< [#267 L0] the same brackets in wall
                                            QTimer ticks (ns on the host): the
                                            op's time at whatever clock it ran */
  float *param[HTP_GRAPH_MAX_OPS];     /**< gamma or conv_w, NULL until bound */
  float *state[HTP_GRAPH_MAX_OPS];     /**< CONV1D_GATE: 3 x N (rows 0-1 the
                                            conv state, row 2 scratch) */
  uint32_t ordinal[HTP_GRAPH_MAX_OPS]; /**< ATTN_M1: the attention-layer
                                            index the cache is keyed by
                                            ([plan 201 S4] counted within
                                            its shape's cache) */
  /** [plan 201 S1] MOE: the op's pool table, h_gu[0..E) then h_dn[0..E)
   *  (HTP_GRAPH_PARAM_EXPERTS; HTP_GRAPH_NO_HANDLE = not resident), NULL
   *  until bound */
  uint32_t *experts[HTP_GRAPH_MAX_OPS];
  /** the session's handle table graph_init checked against, which the
   *  EXPERTS tables are checked against too */
  const hexkl_weight_u8i4_table *tbl;
  /** [plan 201 S1] a miss round's per-expert outputs, top_k x N_out f32
   *  (the widest resident MOE op's), allocated at init */
  float *moe_rows;
  uint8_t route_log[HEXKL_GRAPH_ROUTE_LOG]; /**< cleared by the token driver */
  uint32_t route_log_n;
  /** [#266 S2] HTP_GRAPH_PARAM_PREDICT: 1 = each ROUTER_TOPK op also runs
   *  the next ROUTER_TOPK op's router on its own input */
  uint32_t predict;
  /** [#289] HTP_GRAPH_PARAM_DENSE_EARLY: 1 = a MOE op's miss round runs
   *  the dense branch after it (RMSNORM, DENSE_FFN, RMSNORM: it reads the
   *  residual, not the MoE output) into @a early while the ARM reads, and
   *  the forward skips those ops, copying @a early out at the last; the
   *  ops in [early_from, early_to] are done (early_to 0: none) */
  uint32_t dense_early;
  float *early; /**< slot_words, allocated when dense_early is set */
  uint32_t early_from, early_to;
  /** [#266 S2] per ROUTER_TOPK op run with predict set, in list order: the
   *  count (0: no next router, or its weights unbound), then the guessed
   *  ids best first; cleared by the token driver like route_log */
  uint8_t pred_log[HEXKL_GRAPH_ROUTE_LOG];
  uint32_t pred_log_n;
  /** [#267 L3] MOE kernel calls the MOE ops made, and the calls the miss
   *  path's one-expert-at-a-time form would have made for the same misses;
   *  cleared with route_log_n */
  uint32_t moe_calls, moe_calls_1x;
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
  uint32_t lm_id; /**< m1_argmax_first of the last LM_HEAD's logits, the
                       LM_BAN ids skipped */
  uint32_t ban[HTP_GRAPH_MAX_BAN]; /**< [#132 Part B E3] LM_BAN's ids */
  uint32_t n_ban;
  /** [#261] While attn_prof_on (the token driver sets it per token), each
   *  ATTN_M1 op runs the kernel's _prof entry and adds its phase words
   *  (attn_m1_det.h ATTN_M1_PROF_*) here: [0] env->attn_m1's calls,
   *  [1] env->attn_m1_b's (Gemma's full layers). Off: no timestamp. */
  uint32_t attn_prof_on;
  uint32_t attn_prof_calls[2];
  uint64_t attn_prof[2][HEXKL_GRAPH_ATTN_PROF_WORDS];
} hexkl_graph;

/** @brief HTP_GRAPH_KIND_BIT mask of the kinds whose table slot is
 *  non-NULL in this build. */
uint32_t hexkl_graph_resident_kinds(void);

/**
 * @brief Validates the words, checks every resident Q4M1 op's handles
 *        against @a q4m1 (#132 Part B: each part in
 *        range and in use, K the op's, an FC's or LM_HEAD's part widths
 *        multiples of 32 summing to N, a DENSE_FFN's up and gate K x N and
 *        down N x N_out), and keeps a copy and @a tbl (the MOE ops'
 *        EXPERTS tables are checked against it when bound).
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
 *         not take @a which, or @a n is not its length),
 *         HTP_GRAPH_E_INVHANDLE (an EXPERTS entry that is neither
 *         HTP_GRAPH_NO_HANDLE nor a registered weight of the op's shape),
 *         AEE_ENOMEMORY
 */
int hexkl_graph_set_param(hexkl_graph *g, uint32_t op, uint32_t which,
                          const float *data, uint32_t n);

/**
 * @brief [plan 201 S1] One EXPERTS entry: expert @a e of MOE op @a op to
 *        the pair @a h_gu / @a h_dn (both HTP_GRAPH_NO_HANDLE: evicted).
 * @return 0; AEE_EBADITEM (op or e out of range, not a MOE op);
 *         AEE_EBADSTATE (no table bound); HTP_GRAPH_E_INVHANDLE (a handle
 *         not registered at the op's shape)
 */
int hexkl_graph_pool_set(hexkl_graph *g, uint32_t op, uint32_t e, uint32_t h_gu,
                         uint32_t h_dn);

/** @brief Whether any MoE op's EXPERTS table, or [#225] a WH op, names
 *  @a handle:
 *  weight_release refuses such a handle with AEE_EBADSTATE while the graph
 *  lives. */
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
 * @return 0, AEE_EBADSTATE (no graph, routing; a MOE op with no EXPERTS
 *         table or one whose routed expert is not resident; a RMSNORM /
 *         QK_NORM /
 *         CONV1D_GATE / ROUTER_TOPK op with no parameter or state bound, a ROPE
 * op with no table, an ATTN_M1 op with no cache of its shape in @a env, a
 * Q4M1 op with
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
