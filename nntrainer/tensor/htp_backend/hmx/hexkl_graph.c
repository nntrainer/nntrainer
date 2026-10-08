// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_graph.c
 * @date   23 Sep 2026
 * @brief  The session's validated decode op table and its forward loop
 *         (#85), with the small ops and m=1 attention resident (#130),
 *         the residual add and the router (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Address space (LEDGER rule 8): one hexkl_graph is HTP_GRAPH_MAX_OPS x
 * (320 + 8 + 4 + 4 + 4 + 4) B of table, about 344 KiB at 1024 ops (plan
 * 201 S1; 88 KiB at the 256 before it), plus the MOE ops' EXPERTS tables
 * (8 B an expert: 5.5 KiB at LFM2.5, 30 KiB at Gemma's 30 x 128) and the
 * miss round's rows (top_k x N_out f32: 32 KiB at LFM2.5, 88 KiB at
 * Gemma), plus
 * HTP_GRAPH_N_SLOTS x slot_words f32 of activation slots (36 KiB at LFM2.5:
 * QK_NORM's 3072 words is the widest resident op; [plan 201 S4] 120 KiB at
 * Gemma 4's full-attention QK_NORM, 10 x 2 x 512 words), plus the parameters
 * bound
 * through hexkl_graph_set_param -- about 1.8 MiB at LFM2.5 (49 gammas of 8 KiB,
 * 18 x (24 + 24) KiB conv weight and state, a 512 KiB RoPE table at
 * max_seq 2048; plan 82 section 3.4) -- plus, with ROUTER_TOPK resident
 * (#132), 22 router weights padded to [2048][32] f32, 256 KiB each, 5.5
 * MiB -- all DSP heap, no arena, no VTCM, no DMA. The 24 MiB fp16 KV cache the
 * ATTN_M1 op reads is the session's (hvx_attn_m1_f32.h's budget note), borrowed
 * through the env. [#132 Part B] With a Q4M1 kind resident: the quantized
 * activation (about 13 KiB), the dense FFN's three rows (84 KiB at inter
 * 7168), the logits (vocab floats, 256 KiB at 65536; [plan 201 S4] 1 MiB
 * at Gemma's 262144) and the slot shape
 * copy (8 B a slot) -- about 0.35 MiB of heap. The Q4M1 weights are the
 * session's (nntr_hvx_fc_q4.c's note), not the graph's.
 * [plan 201 S4] Gemma's softmax router: 30 x [2816][128] f32 weights plus
 * 11 KiB of scales, 1.4 MiB a layer, 43 MiB (the ponytail note in
 * hvx_router_softmax_topk_f32). Its RoPE: one max_seq x head_dim f32 table
 * per distinct (theta, variant) -- equal tables are shared -- so 4 MiB
 * (sliding, 256) + 8 MiB (full, 512) at max_seq 4096.
 */

#include "hexkl_graph.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

#include "hvx_m1_ops_f32.h"
#include "hvx_q4_gemv_f32.h"
#include "hvx_scale_add_f32.h"
#include "m1_ops_det.h"

/* htp_graph_desc.h restates the SDK's codes so it can be built with no
   SDK; here both are in scope, so a drift is a build error. */
#if HTP_GRAPH_E_CLASSNOTSUPPORT != AEE_ECLASSNOTSUPPORT ||                     \
  HTP_GRAPH_E_BADSTATE != AEE_EBADSTATE ||                                     \
  HTP_GRAPH_E_BADITEM != AEE_EBADITEM ||                                       \
  HTP_GRAPH_E_INVALIDFORMAT != AEE_EINVALIDFORMAT ||                           \
  HTP_GRAPH_E_INCOMPLETEITEM != AEE_EINCOMPLETEITEM ||                         \
  HTP_GRAPH_E_UNSUPPORTED != AEE_EUNSUPPORTED ||                               \
  HTP_GRAPH_E_NOTYPE != AEE_ENOTYPE ||                                         \
  HTP_GRAPH_E_INVALIDITEM != AEE_EINVALIDITEM ||                               \
  HTP_GRAPH_E_INVHANDLE != AEE_EINVHANDLE ||                                   \
  HTP_GRAPH_E_SCHEMENOTSUPPORTED != AEE_ESCHEMENOTSUPPORTED ||                 \
  HTP_GRAPH_E_NOTALLOWED != AEE_ENOTALLOWED
#error "htp_graph_desc.h's error codes drifted from AEEStdErr.h"
#endif

/** @brief What one forward call carries to the kernels. */
typedef struct {
  const hexkl_graph_env *env;
  hexkl_graph_routing routing; /**< n_experts 0 once consumed */
  uint32_t pos;                /**< the token position (ROPE, ATTN_M1) */
} graph_call;

typedef int (*graph_kernel)(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out);

/* ---- MOE: hexkl_mm_u8i4_moe_layer_run at M = 1, unchanged -------------- */

/* The kernel over the n experts ids[] (ascending) with their weights w[],
   the token's one row each: a whole layer, or one expert of a miss round. */
static int graph_moe_run(const hexkl_graph_env *env, const htp_graph_op *op,
                         const uint32_t *h, const uint32_t *ids, const float *w,
                         uint32_t n, const float *in, float *out) {
  uint32_t cnt[HTP_GRAPH_MAX_EXPERTS] = {0}, idx[HEXKL_GRAPH_MISS_MAX] = {0};
  uint32_t i;
  for (i = 0; i < n; ++i) {
    cnt[ids[i]] = 1u;
  }
  return hexkl_mm_u8i4_moe_layer_run(
    env->tbl, env->vtcm_base, env->vtcm_size, env->config_off, 1u, op->K, op->N,
    op->N_out, op->n_experts, h, h + op->n_experts, idx, cnt, w, in, out,
    env->pool, env->scratch, env->moe_flags);
}

/*
 * [plan 201 S1] A token routed to experts the pool does not hold: the miss
 * is posted, the present experts run while the pool's owner reads the rest,
 * then the rest run. The kernel adds each expert's weighted row into a
 * zeroed output in expert order (hvx_scale_add_rows_f32: out + w * res,
 * two roundings), so the all-resident bits are kept by running the experts
 * before the first miss as one call and every later one alone into its own
 * row (0 + w * res: that expert's term exactly), then adding those rows in
 * expert order at scale 1 (exact products). ponytail: a later expert costs
 * a call of its own (its weight feed no longer overlaps its neighbour's);
 * only a layer with a miss pays it.
 */
static int graph_moe_miss(hexkl_graph *g, const htp_graph_op *op,
                          graph_call *call, const uint32_t *h,
                          const uint32_t *ids, const float *w, uint32_t n,
                          const float *in, float *out) {
  const hexkl_graph_env *env = call->env;
  const uint32_t op_i = (uint32_t)(op - g->ops);
  uint32_t miss[HEXKL_GRAPH_MISS_MAX], was_miss[HEXKL_GRAPH_MISS_MAX];
  uint32_t i, n_miss = 0, first = n;
  int rc;
  for (i = 0; i < n; ++i) {
    was_miss[i] = h[ids[i]] == HTP_GRAPH_NO_HANDLE ||
                  h[op->n_experts + ids[i]] == HTP_GRAPH_NO_HANDLE;
    if (was_miss[i]) {
      miss[n_miss++] = ids[i];
      first = first < i ? first : i;
    }
  }
  if (env->miss.post == NULL || g->moe_rows == NULL) {
    return AEE_EBADSTATE; /* no miss path: the one-session entry */
  }
  rc = env->miss.post(env->miss.ctx, op_i, ids, n, miss, n_miss);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (first != 0u) {
    rc = graph_moe_run(env, op, h, ids, w, first, in, out);
  } else {
    memset(out, 0, (size_t)op->N_out * sizeof(float));
  }
  for (i = first; rc == AEE_SUCCESS && i < n; ++i) {
    if (!was_miss[i]) {
      rc = graph_moe_run(env, op, h, &ids[i], &w[i], 1u, in,
                         g->moe_rows + (size_t)i * op->N_out);
    }
  }
  /* the wait comes even after a failure: the answer must not land on a
     later round's page */
  {
    const int wrc = env->miss.wait(env->miss.ctx, g, op_i);
    rc = rc != AEE_SUCCESS ? rc : wrc;
  }
  for (i = first; rc == AEE_SUCCESS && i < n; ++i) {
    if (was_miss[i]) {
      if (h[ids[i]] == HTP_GRAPH_NO_HANDLE ||
          h[op->n_experts + ids[i]] == HTP_GRAPH_NO_HANDLE) {
        return AEE_EBADSTATE; /* the answer did not bring it */
      }
      rc = graph_moe_run(env, op, h, &ids[i], &w[i], 1u, in,
                         g->moe_rows + (size_t)i * op->N_out);
    }
  }
  for (i = first; rc == AEE_SUCCESS && i < n; ++i) {
    hvx_scale_add_rows_f32(out, g->moe_rows + (size_t)i * op->N_out, 1.0f,
                           op->N_out);
  }
  return rc;
}

static int graph_op_moe(hexkl_graph *g, const htp_graph_op *op,
                        graph_call *call, const float *in, float *out) {
  const hexkl_graph_env *env = call->env;
  const hexkl_graph_routing *r = &call->routing;
  const uint32_t *h = g->experts[op - g->ops];
  uint32_t ids[HEXKL_GRAPH_MISS_MAX];
  uint32_t e, sum = 0, n_miss = 0;
  if (h == NULL) {
    return AEE_EBADSTATE;
  }
  /* The same checks as the per-layer entry (nntr_hvx_mm_u8i4.c
     check_moe_layer_args / check_moe_row_totals), with M fixed at 1: every
     row index is 0 and no expert sees the token twice. */
  if (r->n_experts == 0u) {
    return AEE_EBADSTATE;
  }
  if (r->n_experts != op->n_experts || r->n_rows == 0u ||
      r->n_rows > op->top_k || r->n_rows > HEXKL_GRAPH_MISS_MAX) {
    return AEE_EINVALIDFORMAT;
  }
  for (e = 0; e < r->n_experts; ++e) {
    if (r->row_count[e] > 1u) {
      return AEE_EINVALIDFORMAT;
    }
    if (r->row_count[e] != 0u) {
      if (sum < HEXKL_GRAPH_MISS_MAX) {
        ids[sum] = e;
      }
      /* [plan 201 S1] a routed expert the pool does not hold */
      n_miss += h[e] == HTP_GRAPH_NO_HANDLE ||
                h[op->n_experts + e] == HTP_GRAPH_NO_HANDLE;
    }
    sum += r->row_count[e];
  }
  if (sum != r->n_rows) {
    return AEE_EINVALIDFORMAT;
  }
  for (e = 0; e < r->n_rows; ++e) {
    if (r->row_index[e] != 0u) {
      return AEE_EINVALIDFORMAT;
    }
  }
  call->routing.n_experts = 0u; /* consumed */
  /* [plan 201 S1] the token's routed set, for the pool's recency */
  if (g->route_log_n + 1u + sum <= HEXKL_GRAPH_ROUTE_LOG) {
    g->route_log[g->route_log_n++] = (uint8_t)sum;
    for (e = 0; e < sum; ++e) {
      g->route_log[g->route_log_n++] = (uint8_t)ids[e];
    }
  }
  if (n_miss != 0u) {
    return graph_moe_miss(g, op, call, h, ids, r->row_weight, sum, in, out);
  }
  /* the table is the handle arrays: the kernel skips an expert with no
     rows before it reads its handle, so a NO_HANDLE there is never read */
  return hexkl_mm_u8i4_moe_layer_run(
    env->tbl, env->vtcm_base, env->vtcm_size, env->config_off, 1u, op->K, op->N,
    op->N_out, op->n_experts, h, h + op->n_experts, r->row_index, r->row_count,
    r->row_weight, in, out, env->pool, env->scratch, env->moe_flags);
}

/* ---- the small ops (#82) and m=1 attention (#81), as they are ---------- */

static float graph_eps(const htp_graph_op *op) {
  float eps;
  memcpy(&eps, &op->eps_bits, sizeof(eps));
  return eps;
}

static int graph_op_rmsnorm(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  const float *gamma = g->param[op - g->ops];
  (void)call;
  if (gamma == NULL) {
    return AEE_EBADSTATE;
  }
  ((op->feed & HTP_GRAPH_NORM_N1) != 0u ? hvx_rmsnorm_n1_f32 : hvx_rmsnorm_f32)(
    in, gamma, out, op->K, op->K, graph_eps(op), NULL);
  return AEE_SUCCESS;
}

/* q heads with gamma[0..head_dim), k heads with gamma[head_dim..2 head_dim),
   v copied through: the row stays q | k | v for ROPE and ATTN_M1.
   [plan 201 S4] Gemma 4 (#4296 gemma4_causallm.cpp:672-708): with
   HTP_GRAPH_QKNORM_V the v heads are normed with no gamma (v_norm), and
   with HTP_GRAPH_QKNORM_K_EQ_V v is the raw k (v = k before k_norm); v is
   written first, so in == out still reads the raw k. */
static int graph_op_qk_norm(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  const float *gamma = g->param[op - g->ops];
  const uint32_t hd = op->head_dim, n_q = op->gqa * op->n_kv * hd,
                 n_k = op->n_kv * hd;
  const float eps = graph_eps(op);
  const float *v =
    (op->feed & HTP_GRAPH_QKNORM_K_EQ_V) != 0u ? in + n_q : in + n_q + n_k;
  (void)call;
  if (gamma == NULL) {
    return AEE_EBADSTATE;
  }
  if ((op->feed & HTP_GRAPH_QKNORM_V) != 0u) {
    hvx_rmsnorm_f32(v, NULL, out + n_q + n_k, n_k, hd, eps, NULL);
  } else if (v != out + n_q + n_k) {
    memcpy(out + n_q + n_k, v, (size_t)n_k * sizeof(float));
  }
  hvx_rmsnorm_f32(in, gamma, out, n_q, hd, eps, NULL);
  hvx_rmsnorm_f32(in + n_q, gamma + hd, out + n_q, n_k, hd, eps, NULL);
  return AEE_SUCCESS;
}

/* [plan 201 S4] The op's own table (ROPE_TABLE bound on this op: max_seq
   x head_dim, Gemma's per-layer theta and partial factor) when it has one,
   else the session's shared head_dim 64 table (LFM2). */
static int graph_op_rope(hexkl_graph *g, const htp_graph_op *op,
                         graph_call *call, const float *in, float *out) {
  const uint32_t n_q = op->gqa * op->n_kv, hd = op->head_dim;
  const float *cs = g->param[op - g->ops];
  if (cs == NULL) {
    cs = hd == 64u ? g->rope_cs : NULL;
  }
  if (cs == NULL) {
    return AEE_EBADSTATE;
  }
  if (in != out) {
    memcpy(out, in, (size_t)op->K * sizeof(float));
  }
  hvx_rope_f32(out, n_q, out + (size_t)n_q * hd, op->n_kv,
               cs + (size_t)call->pos * hd, hd);
  return AEE_SUCCESS;
}

static int graph_op_conv1d_gate(hexkl_graph *g, const htp_graph_op *op,
                                graph_call *call, const float *in, float *out) {
  const uint32_t i = (uint32_t)(op - g->ops);
  (void)call;
  if (g->param[i] == NULL || g->state[i] == NULL) {
    return AEE_EBADSTATE;
  }
  hvx_conv_gate_m1_f32(in, g->state[i], g->param[i], out, op->N);
  return AEE_SUCCESS;
}

/** @brief The score scale: eps_bits' f32 when set ([plan 201 S4] Gemma
 *  4's 1.0, attn_m1_det.h's GEMMA note), else 1/sqrt(head_dim) -- at 64
 *  0.125f, exact, and the same bits as the fp16 CPU's `/ sqrt(64.f)`
 *  (attn_m1_det.h step 2). The kernel refuses a scale that is not an fp16
 *  value (1/sqrt(512) is not). */
static float graph_attn_scale(const htp_graph_op *op) {
  return op->eps_bits != 0u    ? graph_eps(op)
         : op->head_dim == 64u ? 0.125f
                               : 1.0f / sqrtf((float)op->head_dim);
}

/** @brief The session cache an ATTN_M1 op's shape names: env->attn_m1, or
 *  [plan 201 S4] env->attn_m1_b (Gemma's full layers beside its sliding
 *  ones), or NULL. */
static hvx_attn_m1_ctx *graph_attn_cache(const hexkl_graph_env *env,
                                         const htp_graph_op *op) {
  hvx_attn_m1_ctx *const c[2] = {env->attn_m1, env->attn_m1_b};
  uint32_t i;
  for (i = 0; i < 2u; ++i) {
    if (c[i] != NULL && c[i]->n_kv == op->n_kv && c[i]->gqa == op->gqa &&
        c[i]->head_dim == op->head_dim) {
      return c[i];
    }
  }
  return NULL;
}

/* [plan 201 S4] top_k is the sliding window (0: full causal, LFM2 and
   Gemma's full layers), eps_bits the scale (graph_attn_scale) */
static int graph_op_attn_m1(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  const uint32_t hd = op->head_dim, n_q = op->gqa * op->n_kv * hd,
                 n_k = op->n_kv * hd;
  hvx_attn_m1_ctx *c = graph_attn_cache(call->env, op);
  if (c == NULL) {
    return AEE_EBADSTATE;
  }
  return hvx_attn_m1_forward(c, g->ordinal[op - g->ops], call->pos, op->top_k,
                             graph_attn_scale(op), in, in + n_q, in + n_q + n_k,
                             out, NULL);
}

/* ---- #132: the residual add and the router ------------------------------ */

/* out += in * 1 (LFM2: out is slot 0, the residual). x * 1 is exact and
   the add rounds once, which is the CPU's copy + add_i bit for bit.
   [plan 201 S4] eps_bits set: then out *= s, Gemma 4's layer_scalar
   (#4296 gemma4_causallm.cpp:486-494, a scalar_multiply after the add). */
static int graph_op_add(hexkl_graph *g, const htp_graph_op *op,
                        graph_call *call, const float *in, float *out) {
  (void)g;
  (void)call;
  hvx_scale_add_rows_f32(out, in, 1.0f, op->N);
  if (op->eps_bits != 0u) {
    hvx_mul_scalar_f32(out, graph_eps(op), op->N);
  }
  return AEE_SUCCESS;
}

/* The logits go to the out slot (nothing reads them); the routing goes to
   g->route_* in ascending expert order -- tryMoeLayerOnAccelerator's
   grouping -- and the call's routing points at it for the MOE op next.
   [plan 201 S4] The softmax router (eps_bits set) norms its un-normed input
   into the out slot first (gamma = ROUTER_BIAS's g, the validator keeps
   in != out), then its logits overwrite it. */
static int graph_op_router_topk(hexkl_graph *g, const htp_graph_op *op,
                                graph_call *call, const float *in, float *out) {
  const uint32_t i = (uint32_t)(op - g->ops);
  uint32_t sel[HTP_GRAPH_MAX_EXPERTS], e, r, n = 0;
  float w[HTP_GRAPH_MAX_EXPERTS], by_expert[HTP_GRAPH_MAX_EXPERTS];
  if (g->param[i] == NULL || g->state[i] == NULL) {
    return AEE_EBADSTATE;
  }
  if (op->eps_bits != 0u) {
    hvx_rmsnorm_f32(in, g->state[i], out, op->K, op->K, graph_eps(op), NULL);
    hvx_router_softmax_topk_f32(out, g->param[i], g->state[i] + op->K, op->K,
                                op->n_experts, op->top_k, out, sel, w,
                                call->env->pool);
  } else {
    hvx_router_topk_f32(in, g->param[i], g->state[i], op->K, op->n_experts,
                        op->top_k, out, sel, w, call->env->pool);
  }
  memset(g->route_cnt, 0, sizeof(g->route_cnt));
  for (r = 0; r < op->top_k; ++r) {
    g->route_cnt[sel[r]] = 1u;
    by_expert[sel[r]] = w[r];
  }
  for (e = 0; e < op->n_experts; ++e) {
    if (g->route_cnt[e] != 0u) {
      g->route_idx[n] = 0u;
      g->route_w[n++] = by_expert[e];
    }
  }
  call->routing.row_index = g->route_idx;
  call->routing.row_count = g->route_cnt;
  call->routing.row_weight = g->route_w;
  call->routing.n_rows = n;
  call->routing.n_experts = op->n_experts;
  return AEE_SUCCESS;
}

/* ---- #132 Part B: the CPU-exact Q4_0 FC, the dense FFN, the lm_head ---- */

/* [#194 L1] The op's quantizer: the native one under the feed word's
   HTP_GRAPH_FEED_NATIVE (the runner then takes the native GEMV), else the
   CPU-exact one. */
static void graph_prep(const htp_graph_op *op, const float *x, uint32_t K,
                       hvx_q4m1_act *a) {
  if ((op->feed & HTP_GRAPH_FEED_NATIVE) != 0u) {
    hvx_q4m1_prep_vec(x, K, a);
  } else {
    hvx_q4m1_prep(x, K, a);
  }
}

/* Runs parts h[0..n) on the prepared activation, their outputs
   concatenated in y (q | k | v; the lm_head's slices). */
static int graph_q4m1_parts(hexkl_graph *g, const htp_graph_op *op,
                            const graph_call *call, const uint32_t *h,
                            uint32_t n, float *y) {
  uint32_t p;
  for (p = 0; p < n; ++p) {
    const int rc = call->env->fc(call->env->fc_ctx, h[p], op->feed, &g->act, y);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    y += g->q4m1[h[p]].N;
  }
  return AEE_SUCCESS;
}

/* [#225] A WH op's kernel flags: the session's MoE flags, with the VTCM
   feed off under the op's L2 bit (the weights then stay in the arena). */
_Static_assert(HEXKL_FC_M1_MAX_PARTS >= HTP_GRAPH_MAX_PARTS &&
                 HTP_GRAPH_WH_DENSE_MAX_CHUNKS <= HEXKL_GRAPH_MISS_MAX,
               "the WH ops' part and chunk limits drifted from the kernels'");
static uint32_t graph_wh_flags(const htp_graph_op *op, uint32_t flags) {
  if ((op->feed & HTP_GRAPH_FEED_L2) != 0u) {
    flags = (flags | HEXKL_MOE_FLAG_GEMV_FEED_SET) & ~HEXKL_MOE_FLAG_GEMV_FEED;
  }
  return flags;
}

/* [#225] HTP_GRAPH_FEED_WH: the parts are the session's u8i4 handles (the
   FC WH sidecar's), one u8 row quantization and the WH GEMV over them. */
static int graph_op_fc_wh(const htp_graph_op *op, const graph_call *call,
                          const float *in, float *out) {
  const hexkl_graph_env *env = call->env;
  return hexkl_mm_u8i4_fc_m1_run(env->tbl, env->vtcm_base, env->vtcm_size,
                                 env->config_off, op->K, op->n_experts,
                                 op->h_gu, in, out, env->pool, env->scratch,
                                 graph_wh_flags(op, env->moe_flags));
}

static int graph_op_fc(hexkl_graph *g, const htp_graph_op *op, graph_call *call,
                       const float *in, float *out) {
  if ((op->feed & HTP_GRAPH_FEED_WH) != 0u) {
    return graph_op_fc_wh(op, call, in, out);
  }
  if (call->env->fc == NULL) {
    return AEE_EBADSTATE;
  }
  graph_prep(op, in, op->K, &g->act);
  return graph_q4m1_parts(g, op, call, op->h_gu, op->n_experts, out);
}

/* up and gate on one quantization, silu(gate) * up in the CPU's order
   (m1_swiglu_cpu_det: swiglu layer input 0 is gate), the swiglu row
   quantized, down. [#132 E5f] The SwiGLU over the pool with the scalar
   IEEE divide (hvx_swiglu_cpu_f32; E5d read DENSE_FFN at 5.6 ms/token
   with the spec's integer division on one thread). [plan 201 S4] Under
   the session's HEXKL_MOE_FLAG_GEGLU (the model's activation, #209) it is
   gelu_tanh(gate) * up (geglu_det_one), Gemma 4's dense FFN (#4296
   gemma4_causallm.cpp:760-806: separate gate and up, tanh_gelu, multiply). */
static int graph_op_dense_ffn(hexkl_graph *g, const htp_graph_op *op,
                              graph_call *call, const float *in, float *out) {
  float *up, *gate, *act;
  int rc;
  if ((op->feed & HTP_GRAPH_FEED_WH) != 0u) {
    /* [#225] the n_experts chunks as experts of weight 1, the token's one
       row each: the MoE kernel's M = 1 pair path, its VTCM feed, its
       SwiGLU (GeGLU under the session's flag), the chunks' downs added in
       chunk order */
    const hexkl_graph_env *env = call->env;
    uint32_t idx[HTP_GRAPH_MAX_PARTS], cnt[HTP_GRAPH_MAX_PARTS];
    float w[HTP_GRAPH_MAX_PARTS];
    uint32_t c;
    for (c = 0; c < op->n_experts; ++c) {
      idx[c] = 0u;
      cnt[c] = 1u;
      w[c] = 1.0f;
    }
    return hexkl_mm_u8i4_moe_layer_run(
      env->tbl, env->vtcm_base, env->vtcm_size, env->config_off, 1u, op->K,
      op->N / op->n_experts, op->N_out, op->n_experts, op->h_gu, op->h_dn, idx,
      cnt, w, in, out, env->pool, env->scratch,
      graph_wh_flags(op, env->moe_flags));
  }
  if (call->env->fc == NULL) {
    return AEE_EBADSTATE;
  }
  /* g->ffn is NULL when every DENSE_FFN is WH (graph_init sizes it by the
     Q4M1 ones): set only on this path */
  up = g->ffn;
  gate = g->ffn + op->N;
  act = g->ffn + 2u * op->N;
  graph_prep(op, in, op->K, &g->act);
  rc = call->env->fc(call->env->fc_ctx, op->h_gu[0], op->feed, &g->act, up);
  if (rc == AEE_SUCCESS) {
    rc = call->env->fc(call->env->fc_ctx, op->h_gu[1], op->feed, &g->act, gate);
  }
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if ((call->env->moe_flags & HEXKL_MOE_FLAG_GEGLU) != 0u) {
    hvx_geglu_f32(gate, up, act, op->N);
  } else {
    hvx_swiglu_cpu_f32(gate, up, act, op->N, call->env->pool);
  }
  graph_prep(op, act, op->N, &g->act);
  return call->env->fc(call->env->fc_ctx, op->h_dn[0], op->feed, &g->act, out);
}

/* The slices into g->logits (forward hands them out as the op's output),
   then the first maximum, as the CPU's sampler picks it. [plan 201 S4]
   eps_bits set: the logits are soft-capped first (Gemma 4's
   final_logit_softcapping, #4296 gemma4_causallm.cpp:855-864), so the
   pick and the logits handed out are the capped ones, as on the CPU. */
static int graph_op_lm_head(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  int rc;
  (void)out;
  if (call->env->fc == NULL) {
    return AEE_EBADSTATE;
  }
  graph_prep(op, in, op->K, &g->act);
  rc = graph_q4m1_parts(g, op, call, op->h_gu, op->n_experts, g->logits);
  if (rc == AEE_SUCCESS && op->eps_bits != 0u) {
    hvx_softcap_f32(g->logits, op->N, graph_eps(op), call->env->pool);
  }
  if (rc == AEE_SUCCESS) {
    /* [#132 Part B E3] the banned ids at -inf for the pick only (the
       CPU's applyBadWordsPenalty), put back in reverse so a repeated id
       gets its own value again */
    float keep[HTP_GRAPH_MAX_BAN];
    uint32_t i;
    for (i = 0; i < g->n_ban; ++i) {
      keep[i] = g->logits[g->ban[i]];
      g->logits[g->ban[i]] = -INFINITY;
    }
    g->lm_id = hvx_argmax_first_f32(g->logits, op->N);
    for (i = g->n_ban; i-- > 0;) {
      g->logits[g->ban[i]] = keep[i];
    }
  }
  return rc;
}

/** @brief The kernel table: a NULL slot is a kind this build does not run
 *  (hvx_impl's htp_op_table rule); the validator refuses a resident bit on
 *  it with AEE_ECLASSNOTSUPPORT, so forward never reaches a NULL. */
static const graph_kernel kernels[HTP_OP_KIND_N] = {
  graph_op_rmsnorm,     /* RMSNORM      #82 */
  graph_op_fc,          /* FC           #132 Part B */
  graph_op_conv1d_gate, /* CONV1D_GATE  #82 */
  graph_op_qk_norm,     /* QK_NORM      #82 */
  graph_op_rope,        /* ROPE         #82 */
  graph_op_attn_m1,     /* ATTN_M1      #81 */
  graph_op_add,         /* ADD          #132 */
  graph_op_router_topk, /* ROUTER_TOPK  #132 */
  graph_op_moe,         /* MOE */
  graph_op_dense_ffn,   /* DENSE_FFN    #132 Part B */
  graph_op_lm_head,     /* LM_HEAD      #132 Part B */
};

uint32_t hexkl_graph_resident_kinds(void) {
  uint32_t mask = 0, k;
  for (k = 0; k < HTP_OP_KIND_N; ++k) {
    if (kernels[k] != NULL) {
      mask |= HTP_GRAPH_KIND_BIT(k);
    }
  }
  return mask;
}

static int graph_check_handle(const hexkl_weight_u8i4_table *tbl, uint32_t h,
                              uint32_t K, uint32_t N) {
  if (h >= HEXKL_MM_U8I4_MAX_WEIGHTS || !tbl->slots[h].in_use ||
      tbl->slots[h].K != K || tbl->slots[h].N != N) {
    return AEE_EINVHANDLE;
  }
  return AEE_SUCCESS;
}

/* A resident Q4M1 op's parts against the slot shapes (hexkl_graph.h). */
static int graph_check_q4m1(const htp_graph_op *op,
                            const hexkl_graph_q4m1_shape *q4m1,
                            uint32_t n_q4m1) {
  uint32_t p, sum = 0;
#define Q4M1_OK(h, k) ((h) < n_q4m1 && q4m1[h].K != 0u && q4m1[h].K == (k))
  if (op->kind == HTP_OP_DENSE_FFN) {
    return (op->n_experts == 3u && Q4M1_OK(op->h_gu[0], op->K) &&
            Q4M1_OK(op->h_gu[1], op->K) && Q4M1_OK(op->h_dn[0], op->N) &&
            q4m1[op->h_gu[0]].N == op->N && q4m1[op->h_gu[1]].N == op->N &&
            q4m1[op->h_dn[0]].N == op->N_out)
             ? AEE_SUCCESS
             : AEE_EINVHANDLE;
  }
  if (op->n_experts == 0u) {
    return AEE_EINVHANDLE;
  }
  for (p = 0; p < op->n_experts; ++p) {
    if (!Q4M1_OK(op->h_gu[p], op->K) || q4m1[op->h_gu[p]].N % 32u != 0u) {
      return AEE_EINVHANDLE;
    }
    sum += q4m1[op->h_gu[p]].N;
  }
#undef Q4M1_OK
  return sum == op->N ? AEE_SUCCESS : AEE_EINVHANDLE;
}

/* [#225] A resident op's WH handles (HTP_GRAPH_FEED_WH) against the
   session's table: an FC's parts K x N_p, N_p % 32, summing to N; a
   DENSE_FFN's n_experts chunks (<= HTP_GRAPH_WH_DENSE_MAX_CHUNKS) gate |
   up K x 2w and down w x N_out, w = N / n_experts. */
static int graph_check_wh(const htp_graph_op *op,
                          const hexkl_weight_u8i4_table *tbl) {
  uint32_t p, sum = 0;
  if (op->kind == HTP_OP_DENSE_FFN) {
    const uint32_t n = op->n_experts, w = n != 0u ? op->N / n : 0u;
    if (n == 0u || n > HTP_GRAPH_WH_DENSE_MAX_CHUNKS || w * n != op->N ||
        w % 32u != 0u) {
      return AEE_EINVHANDLE;
    }
    for (p = 0; p < n; ++p) {
      if (graph_check_handle(tbl, op->h_gu[p], op->K, 2u * w) != AEE_SUCCESS ||
          graph_check_handle(tbl, op->h_dn[p], w, op->N_out) != AEE_SUCCESS) {
        return AEE_EINVHANDLE;
      }
    }
    return AEE_SUCCESS;
  }
  if (op->kind != HTP_OP_FC || op->n_experts == 0u) {
    return AEE_EINVHANDLE;
  }
  for (p = 0; p < op->n_experts; ++p) {
    const uint32_t h = op->h_gu[p];
    /* [#234 P4] 4-bit only: hexkl_mm_u8i4_fc_m1_run refuses a 2-bit
       handle on every token, so it is refused here, once */
    if (h >= HEXKL_MM_U8I4_MAX_WEIGHTS || !tbl->slots[h].in_use ||
        tbl->slots[h].K != op->K || tbl->slots[h].N % 32u != 0u ||
        tbl->slots[h].bits == 2u) {
      return AEE_EINVHANDLE;
    }
    sum += tbl->slots[h].N;
  }
  return sum == op->N ? AEE_SUCCESS : AEE_EINVHANDLE;
}

int hexkl_graph_init(const uint32_t *words, uint32_t n_words,
                     const hexkl_weight_u8i4_table *tbl,
                     const hexkl_graph_q4m1_shape *q4m1, uint32_t n_q4m1,
                     hexkl_graph **out) {
  uint32_t n_ops = 0, i, slot_words = 0, ffn_n = 0;
  uint32_t q4m1_ops = 0, vocab_out = 0, moe_rows = 0;
  hexkl_graph *g;
  int rc;
  if (out == NULL || tbl == NULL) {
    return AEE_EBADSTATE;
  }
  rc = (int)htp_graph_validate(words, n_words, hexkl_graph_resident_kinds(),
                               &n_ops);
  if (rc != 0) {
    return rc;
  }
  for (i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words, i);
    if (!op->resident) {
      continue;
    }
    if ((op->feed & HTP_GRAPH_FEED_WH) != 0u) {
      if (graph_check_wh(op, tbl) != AEE_SUCCESS) {
        return AEE_EINVHANDLE;
      }
    } else if ((HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) != 0u) {
      if (q4m1 == NULL || graph_check_q4m1(op, q4m1, n_q4m1) != AEE_SUCCESS) {
        return AEE_EINVHANDLE;
      }
      ++q4m1_ops;
      if (op->kind == HTP_OP_DENSE_FFN && op->N > ffn_n) {
        ffn_n = op->N;
      }
    }
    if (op->kind == HTP_OP_MOE && op->top_k * op->N_out > moe_rows) {
      moe_rows = op->top_k * op->N_out; /* the miss round's rows */
    }
    if (htp_graph_op_in_words(op) > slot_words) {
      slot_words = htp_graph_op_in_words(op);
    }
    /* the logits are not a slot: g->logits below */
    if (op->kind == HTP_OP_LM_HEAD) {
      vocab_out = op->N;
    } else if (htp_graph_op_out_words(op) > slot_words) {
      slot_words = htp_graph_op_out_words(op);
    }
  }
  g = (hexkl_graph *)calloc(1, sizeof(*g));
  if (g == NULL) {
    return AEE_ENOMEMORY;
  }
  g->tbl = tbl;
  if (moe_rows != 0u) {
    g->moe_rows = (float *)malloc((size_t)moe_rows * sizeof(float));
    if (g->moe_rows == NULL) {
      free(g);
      return AEE_ENOMEMORY;
    }
  }
  g->n_layers = words[2];
  g->n_ops = n_ops;
  g->hidden = words[4];
  g->vocab = words[5];
  g->max_seq = words[6];
  g->slot_words = slot_words;
  if (slot_words != 0u) {
    g->slots =
      (float *)calloc((size_t)HTP_GRAPH_N_SLOTS * slot_words, sizeof(float));
    if (g->slots == NULL) {
      free(g->moe_rows);
      free(g);
      return AEE_ENOMEMORY;
    }
  }
  if (q4m1_ops != 0u) {
    /* q (K bytes, 128-aligned) then s8 / ma / ea / df / d per block */
    const size_t nb = HEXKL_GRAPH_Q4M1_MAX_K / 32u;
    g->act_buf = (uint8_t *)memalign(128, HEXKL_GRAPH_Q4M1_MAX_K + nb * 18u);
    g->q4m1 =
      (hexkl_graph_q4m1_shape *)malloc((size_t)n_q4m1 * sizeof(*g->q4m1));
    g->ffn = ffn_n ? (float *)malloc((size_t)3u * ffn_n * sizeof(float)) : NULL;
    g->logits =
      vocab_out ? (float *)malloc((size_t)vocab_out * sizeof(float)) : NULL;
    if (g->act_buf == NULL || g->q4m1 == NULL || (ffn_n && g->ffn == NULL) ||
        (vocab_out && g->logits == NULL)) {
      hexkl_graph_free(g);
      return AEE_ENOMEMORY;
    }
    memcpy(g->q4m1, q4m1, (size_t)n_q4m1 * sizeof(*g->q4m1));
    g->n_q4m1 = n_q4m1;
    g->act.q = (int8_t *)g->act_buf;
    g->act.s8 = (int32_t *)(g->act_buf + HEXKL_GRAPH_Q4M1_MAX_K);
    g->act.ma = g->act.s8 + nb;
    g->act.ea = g->act.ma + nb;
    g->act.df = (float *)(g->act.ea + nb);
    g->act.d = (uint16_t *)(g->act.df + nb);
  }
  for (i = 0; i < n_ops; ++i) {
    g->ops[i] = *htp_graph_op_cat(words, i);
    if (g->ops[i].kind == HTP_OP_ATTN_M1) {
      /* [plan 201 S4] the layer index within its shape's cache: the ATTN_M1
         ops before it of the same (n_kv, gqa, head_dim). The ARM side
         (htp_compute_ops.cpp attn_ordinal_, the kv seed) still counts all
         ATTN_M1 ops: the same numbers while there is one shape (LFM2); the
         Gemma hand-over that registers attn_m1_b counts per shape too. */
      uint32_t j;
      g->ordinal[i] = 0u;
      for (j = 0; j < i; ++j) {
        g->ordinal[i] += g->ops[j].kind == HTP_OP_ATTN_M1 &&
                         g->ops[j].n_kv == g->ops[i].n_kv &&
                         g->ops[j].gqa == g->ops[i].gqa &&
                         g->ops[j].head_dim == g->ops[i].head_dim;
      }
    }
  }
  *out = g;
  return AEE_SUCCESS;
}

/** @brief ROPE ops after @a i whose table is @a t (NULL: none). */
static uint32_t graph_rope_users_after(const hexkl_graph *g, const float *t,
                                       uint32_t i) {
  uint32_t j, n = 0;
  for (j = i + 1u; t != NULL && j < g->n_ops; ++j) {
    n += g->ops[j].kind == HTP_OP_ROPE && g->param[j] == t;
  }
  return n;
}

void hexkl_graph_free(hexkl_graph *g) {
  uint32_t i;
  if (g == NULL) {
    return;
  }
  for (i = 0; i < g->n_ops; ++i) {
    /* a shared ROPE table is freed by its last holder */
    if (g->ops[i].kind != HTP_OP_ROPE ||
        graph_rope_users_after(g, g->param[i], i) == 0u) {
      free(g->param[i]);
    }
    free(g->state[i]);
    free(g->experts[i]);
  }
  free(g->rope_cs);
  free(g->moe_rows);
  free(g->slots);
  free(g->act_buf);
  free(g->q4m1);
  free(g->ffn);
  free(g->logits);
  free(g);
}

/** @brief The parameter's length for (kind, which), 0 when the kind does
 *  not take it. */
static uint32_t graph_param_len(const htp_graph_op *op, uint32_t which,
                                uint32_t max_seq) {
  switch (which) {
  case HTP_GRAPH_PARAM_GAMMA:
    return op->kind == HTP_OP_RMSNORM   ? op->K
           : op->kind == HTP_OP_QK_NORM ? 2u * op->head_dim
                                        : 0u;
  case HTP_GRAPH_PARAM_CONV_W:
    return op->kind == HTP_OP_CONV1D_GATE ? 3u * op->N : 0u;
  case HTP_GRAPH_PARAM_CONV_STATE:
    return op->kind == HTP_OP_CONV1D_GATE ? 2u * op->N : 0u;
  case HTP_GRAPH_PARAM_ROUTER_W:
    return op->kind == HTP_OP_ROUTER_TOPK ? op->K * op->n_experts : 0u;
  case HTP_GRAPH_PARAM_ROPE_TABLE: /* [plan 201 S4] the op's own */
    return op->kind == HTP_OP_ROPE ? op->head_dim * max_seq : 0u;
  case HTP_GRAPH_PARAM_ROUTER_BIAS: /* softmax: g[K] | per-expert scale */
    return op->kind != HTP_OP_ROUTER_TOPK ? 0u
           : op->eps_bits != 0u           ? op->K + op->n_experts
                                          : op->n_experts;
  default:
    return 0u;
  }
}

/** @brief ROPE ops other than @a except whose table is @a t. */
static uint32_t graph_rope_users(const hexkl_graph *g, const float *t,
                                 uint32_t except) {
  uint32_t j, n = 0;
  for (j = 0; j < g->n_ops; ++j) {
    n += j != except && g->ops[j].kind == HTP_OP_ROPE && g->param[j] == t;
  }
  return n;
}

/* [plan 201 S4] A ROPE op's own table, max_seq x head_dim. A table equal to
   one another ROPE op of the same head_dim holds is shared, not copied
   (Gemma's 25 sliding layers bind one: 4 MiB at max_seq 4096 and head_dim
   256, not 100), and a shared table is never written in place. */
static int graph_set_rope(hexkl_graph *g, uint32_t op, const float *data,
                          uint32_t n) {
  float *mine, *t = NULL;
  uint32_t j;
  if (op >= g->n_ops) {
    return AEE_EBADITEM;
  }
  if (n == 0u || n != graph_param_len(&g->ops[op], HTP_GRAPH_PARAM_ROPE_TABLE,
                                      g->max_seq)) {
    return AEE_EINVALIDFORMAT;
  }
  for (j = 0; j < g->n_ops && t == NULL; ++j) {
    if (j != op && g->ops[j].kind == HTP_OP_ROPE && g->param[j] != NULL &&
        g->ops[j].head_dim == g->ops[op].head_dim &&
        memcmp(g->param[j], data, (size_t)n * sizeof(float)) == 0) {
      t = g->param[j];
    }
  }
  mine = g->param[op];
  if (t == NULL) {
    t = (mine != NULL && graph_rope_users(g, mine, op) == 0u)
          ? mine
          : (float *)malloc((size_t)n * sizeof(float));
    if (t == NULL) {
      return AEE_ENOMEMORY;
    }
    memcpy(t, data, (size_t)n * sizeof(float));
  }
  if (mine != NULL && mine != t && graph_rope_users(g, mine, op) == 0u) {
    free(mine);
  }
  g->param[op] = t;
  return AEE_SUCCESS;
}

int hexkl_graph_set_param(hexkl_graph *g, uint32_t op, uint32_t which,
                          const float *data, uint32_t n) {
  float **dst;
  uint32_t want, alloc;
  if (g == NULL) {
    return AEE_EBADSTATE;
  }
  if (which >= HTP_GRAPH_PARAM_N || data == NULL) {
    return AEE_EBADITEM;
  }
  if (which == HTP_GRAPH_PARAM_LM_BAN) {
    uint32_t i;
    if (op >= g->n_ops || g->ops[op].kind != HTP_OP_LM_HEAD) {
      return AEE_EBADITEM;
    }
    if (n > HTP_GRAPH_MAX_BAN) { /* 0 clears the list */
      return AEE_EINVALIDFORMAT;
    }
    for (i = 0; i < n; ++i) {
      uint32_t id;
      memcpy(&id, &data[i], sizeof(id));
      if (id >= g->ops[op].N) {
        return AEE_EINVALIDFORMAT;
      }
      g->ban[i] = id;
    }
    g->n_ban = n;
    return AEE_SUCCESS;
  }
  if (which == HTP_GRAPH_PARAM_EXPERTS) {
    /* [plan 201 S1] the MOE op's pool table: every entry checked before
       any is stored, so a refused table leaves the old one in place */
    const htp_graph_op *o;
    uint32_t i, *h;
    if (op >= g->n_ops) {
      return AEE_EBADITEM;
    }
    o = &g->ops[op];
    if (o->kind != HTP_OP_MOE || n != 2u * o->n_experts) {
      return AEE_EINVALIDFORMAT;
    }
    for (i = 0; i < n; ++i) {
      uint32_t v;
      memcpy(&v, &data[i], sizeof(v));
      if (v != HTP_GRAPH_NO_HANDLE &&
          (i < o->n_experts
             ? graph_check_handle(g->tbl, v, o->K, 2u * o->N)
             : graph_check_handle(g->tbl, v, o->N, o->N_out)) != AEE_SUCCESS) {
        return AEE_EINVHANDLE;
      }
    }
    h = g->experts[op];
    if (h == NULL) {
      h = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
      if (h == NULL) {
        return AEE_ENOMEMORY;
      }
      g->experts[op] = h;
    }
    memcpy(h, data, (size_t)n * sizeof(uint32_t));
    return AEE_SUCCESS;
  }
  if (which == HTP_GRAPH_PARAM_ROPE_TABLE && op != HTP_GRAPH_NO_OP) {
    return graph_set_rope(g, op, data, n);
  }
  if (which == HTP_GRAPH_PARAM_ROPE_TABLE) {
    want = alloc = g->max_seq * 64u;
    dst = &g->rope_cs;
  } else {
    if (op >= g->n_ops) {
      return AEE_EBADITEM;
    }
    want = graph_param_len(&g->ops[op], which, g->max_seq);
    if (want == 0u) {
      return AEE_EINVALIDFORMAT;
    }
    /* the conv state buffer is 3 rows: the kernel's scratch is row 2; the
       router bias shares the state pointer (a ROUTER_TOPK op has none) */
    alloc = (which == HTP_GRAPH_PARAM_CONV_STATE) ? 3u * g->ops[op].N : want;
    dst = (which == HTP_GRAPH_PARAM_CONV_STATE ||
           which == HTP_GRAPH_PARAM_ROUTER_BIAS)
            ? &g->state[op]
            : &g->param[op];
  }
  if (n != want) {
    return AEE_EINVALIDFORMAT;
  }
  if (which == HTP_GRAPH_PARAM_ROUTER_W) {
    /* [K][E] padded to [K][W], zero lanes: one vector a row for the
       sigmoid router, E rounded up to 32 for the softmax one */
    const uint32_t K = g->ops[op].K, E = g->ops[op].n_experts;
    const uint32_t W = g->ops[op].eps_bits != 0u ? (E + 31u) / 32u * 32u
                                                 : HTP_GRAPH_ROUTER_MAX_EXPERTS;
    uint32_t k;
    if (E > W) { /* the sigmoid router's width (htp_graph_desc.h) */
      return AEE_ESCHEMENOTSUPPORTED;
    }
    if (*dst == NULL) {
      *dst = (float *)memalign(128, (size_t)K * W * sizeof(float));
      if (*dst == NULL) {
        return AEE_ENOMEMORY;
      }
    }
    memset(*dst, 0, (size_t)K * W * sizeof(float));
    for (k = 0; k < K; ++k) {
      memcpy(*dst + (size_t)k * W, data + (size_t)k * E,
             (size_t)E * sizeof(float));
    }
    return AEE_SUCCESS;
  }
  if (*dst == NULL) {
    *dst = (float *)calloc(alloc, sizeof(float));
    if (*dst == NULL) {
      return AEE_ENOMEMORY;
    }
  }
  memcpy(*dst, data, (size_t)n * sizeof(float));
  return AEE_SUCCESS;
}

int hexkl_graph_pool_set(hexkl_graph *g, uint32_t op, uint32_t e, uint32_t h_gu,
                         uint32_t h_dn) {
  const htp_graph_op *o;
  if (g == NULL || op >= g->n_ops || g->ops[op].kind != HTP_OP_MOE ||
      e >= g->ops[op].n_experts) {
    return AEE_EBADITEM;
  }
  if (g->experts[op] == NULL) {
    return AEE_EBADSTATE;
  }
  o = &g->ops[op];
  if ((h_gu == HTP_GRAPH_NO_HANDLE) != (h_dn == HTP_GRAPH_NO_HANDLE) ||
      (h_gu != HTP_GRAPH_NO_HANDLE &&
       (graph_check_handle(g->tbl, h_gu, o->K, 2u * o->N) != AEE_SUCCESS ||
        graph_check_handle(g->tbl, h_dn, o->N, o->N_out) != AEE_SUCCESS))) {
    return AEE_EINVHANDLE;
  }
  g->experts[op][e] = h_gu;
  g->experts[op][o->n_experts + e] = h_dn;
  return AEE_SUCCESS;
}

int hexkl_graph_uses_handle(const hexkl_graph *g, uint32_t handle) {
  uint32_t i, e;
  if (g == NULL) {
    return 0;
  }
  for (i = 0; i < g->n_ops; ++i) {
    const htp_graph_op *op = &g->ops[i];
    if (op->resident && (op->feed & HTP_GRAPH_FEED_WH) != 0u) {
      /* [#225] an FC's parts, a DENSE_FFN's chunk pairs */
      for (e = 0; e < op->n_experts; ++e) {
        if (op->h_gu[e] == handle ||
            (op->kind == HTP_OP_DENSE_FFN && op->h_dn[e] == handle)) {
          return 1;
        }
      }
    }
    if (g->experts[i] == NULL) {
      continue;
    }
    for (e = 0; e < 2u * g->ops[i].n_experts; ++e) {
      if (g->experts[i][e] == handle) {
        return 1;
      }
    }
  }
  return 0;
}

int hexkl_graph_uses_q4m1(const hexkl_graph *g, uint32_t handle) {
  uint32_t i, p;
  if (g == NULL) {
    return 0;
  }
  for (i = 0; i < g->n_ops; ++i) {
    const htp_graph_op *op = &g->ops[i];
    if (!op->resident ||
        (HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) == 0u ||
        (op->feed & HTP_GRAPH_FEED_WH) != 0u) {
      continue;
    }
    if (op->kind == HTP_OP_DENSE_FFN) { /* up, gate, down */
      if (op->h_gu[0] == handle || op->h_gu[1] == handle ||
          op->h_dn[0] == handle) {
        return 1;
      }
      continue;
    }
    for (p = 0; p < op->n_experts; ++p) {
      if (op->h_gu[p] == handle) {
        return 1;
      }
    }
  }
  return 0;
}

int hexkl_graph_forward(hexkl_graph *g, const hexkl_graph_env *env,
                        uint32_t start_op, uint32_t n_ops_limit, uint32_t pos,
                        const hexkl_graph_routing *routing, const float *act_in,
                        uint32_t act_in_len, float *act_out,
                        uint32_t act_out_len, uint32_t *resume_at) {
  graph_call call;
  uint32_t i, n_run = 0;
  const htp_graph_op *last = NULL;
  if (g == NULL || env == NULL || resume_at == NULL) {
    return AEE_EBADSTATE;
  }
  memset(g->op_pcycles, 0, sizeof(g->op_pcycles));
  if (start_op >= g->n_ops || pos >= g->max_seq) {
    return AEE_EBADITEM;
  }
  if (act_in == NULL ||
      act_in_len != htp_graph_op_in_words(&g->ops[start_op])) {
    return AEE_EINVALIDFORMAT;
  }
  /* Which op will be the last one run is known before anything runs, so
     act_out is checked now rather than after the work was done. */
  for (i = start_op; i < g->n_ops && n_run < n_ops_limit && g->ops[i].resident;
       ++i, ++n_run) {
    last = &g->ops[i];
  }
  if (last != NULL &&
      (act_out == NULL || act_out_len != htp_graph_op_out_words(last))) {
    return AEE_EINVALIDFORMAT;
  }
  last = NULL;
  n_run = 0;
  call.env = env;
  call.pos = pos;
  if (routing != NULL) {
    call.routing = *routing;
  } else {
    memset(&call.routing, 0, sizeof(call.routing));
  }
  for (i = start_op; i < g->n_ops && n_run < n_ops_limit; ++i) {
    const htp_graph_op *op = &g->ops[i];
    const graph_kernel k = kernels[op->kind];
    float *in, *out;
    uint64_t t0;
    int rc;
    if (!op->resident) {
      break;
    }
    if (k == NULL) {
      return AEE_ECLASSNOTSUPPORT;
    }
    in = g->slots + (size_t)op->in_slot * g->slot_words;
    out = g->slots + (size_t)op->out_slot * g->slot_words;
    if (n_run == 0u) {
      memcpy(in, act_in, (size_t)act_in_len * sizeof(float));
    }
    t0 = HAP_perf_get_pcycles();
    rc = k(g, op, &call, in, out);
    g->op_pcycles[i] = HAP_perf_get_pcycles() - t0;
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    last = op;
    ++n_run;
  }
  *resume_at = i;
  if (last != NULL) {
    if (act_out == NULL || act_out_len != htp_graph_op_out_words(last)) {
      return AEE_EINVALIDFORMAT;
    }
    const float *src = last->kind == HTP_OP_LM_HEAD
                         ? g->logits
                         : g->slots + (size_t)last->out_slot * g->slot_words;
    /* the token driver may hand the logits buffer itself (hexkl_token.c) */
    if (src != act_out) {
      memcpy(act_out, src, (size_t)act_out_len * sizeof(float));
    }
  }
  return AEE_SUCCESS;
}
