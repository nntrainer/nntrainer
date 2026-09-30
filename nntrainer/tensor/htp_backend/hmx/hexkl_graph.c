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
 * (8 B an expert: 5.5 KiB at LFM2.5, 30 KiB at Gemma's 30 x 128), plus
 * HTP_GRAPH_N_SLOTS x slot_words f32 of activation slots (36 KiB at LFM2.5:
 * QK_NORM's 3072 words is the widest resident op), plus the parameters bound
 * through hexkl_graph_set_param -- about 1.8 MiB at LFM2.5 (49 gammas of 8 KiB,
 * 18 x (24 + 24) KiB conv weight and state, a 512 KiB RoPE table at
 * max_seq 2048; plan 82 section 3.4) -- plus, with ROUTER_TOPK resident
 * (#132), 22 router weights padded to [2048][32] f32, 256 KiB each, 5.5
 * MiB -- all DSP heap, no arena, no VTCM, no DMA. The 24 MiB fp16 KV cache the
 * ATTN_M1 op reads is the session's (hvx_attn_m1_f32.h's budget note), borrowed
 * through the env. [#132 Part B] With a Q4M1 kind resident: the quantized
 * activation (about 13 KiB), the dense FFN's three rows (84 KiB at inter
 * 7168), the logits (vocab floats, 256 KiB at 65536) and the slot shape
 * copy (8 B a slot) -- about 0.35 MiB of heap. The Q4M1 weights are the
 * session's (nntr_hvx_fc_q4.c's note), not the graph's.
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

static int graph_op_moe(hexkl_graph *g, const htp_graph_op *op,
                        graph_call *call, const float *in, float *out) {
  const hexkl_graph_env *env = call->env;
  const hexkl_graph_routing *r = &call->routing;
  const uint32_t *h = g->experts[op - g->ops];
  uint32_t e, sum = 0;
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
      r->n_rows > op->top_k) {
    return AEE_EINVALIDFORMAT;
  }
  for (e = 0; e < r->n_experts; ++e) {
    if (r->row_count[e] > 1u) {
      return AEE_EINVALIDFORMAT;
    }
    /* [plan 201 S1] a routed expert the pool does not hold */
    if (r->row_count[e] != 0u &&
        (h[e] == HTP_GRAPH_NO_HANDLE ||
         h[op->n_experts + e] == HTP_GRAPH_NO_HANDLE)) {
      return AEE_EBADSTATE;
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
  hvx_rmsnorm_f32(in, gamma, out, op->K, op->K, graph_eps(op), NULL);
  return AEE_SUCCESS;
}

/* q heads with gamma[0..head_dim), k heads with gamma[head_dim..2 head_dim),
   v copied through: the row stays q | k | v for ROPE and ATTN_M1. */
static int graph_op_qk_norm(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  const float *gamma = g->param[op - g->ops];
  const uint32_t hd = op->head_dim, n_q = op->gqa * op->n_kv * hd,
                 n_k = op->n_kv * hd;
  const float eps = graph_eps(op);
  (void)call;
  if (gamma == NULL) {
    return AEE_EBADSTATE;
  }
  hvx_rmsnorm_f32(in, gamma, out, n_q, hd, eps, NULL);
  hvx_rmsnorm_f32(in + n_q, gamma + hd, out + n_q, n_k, hd, eps, NULL);
  if (in != out) {
    memcpy(out + n_q + n_k, in + n_q + n_k, (size_t)n_k * sizeof(float));
  }
  return AEE_SUCCESS;
}

static int graph_op_rope(hexkl_graph *g, const htp_graph_op *op,
                         graph_call *call, const float *in, float *out) {
  const uint32_t n_q = op->gqa * op->n_kv;
  if (g->rope_cs == NULL) {
    return AEE_EBADSTATE;
  }
  if (in != out) {
    memcpy(out, in, (size_t)op->K * sizeof(float));
  }
  hvx_rope64_f32(out, n_q, out + (size_t)n_q * 64u, op->n_kv,
                 g->rope_cs + (size_t)call->pos * 64u);
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

/** @brief 1/sqrt(head_dim) at the one head_dim the validator admits for a
 *  resident ATTN_M1 (64 since #152): 0.125f, exact, and the same bits as
 *  the fp16 CPU's `/ sqrt(64.f)` (attn_m1_det.h step 2). */
static float graph_attn_scale(uint32_t head_dim) {
  (void)head_dim;
  return 0.125f;
}

static int graph_op_attn_m1(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  const uint32_t hd = op->head_dim, n_q = op->gqa * op->n_kv * hd,
                 n_k = op->n_kv * hd;
  if (call->env->attn_m1 == NULL) {
    return AEE_EBADSTATE;
  }
  return hvx_attn_m1_forward(call->env->attn_m1, g->ordinal[op - g->ops],
                             call->pos, graph_attn_scale(hd), in, in + n_q,
                             in + n_q + n_k, out, NULL);
}

/* ---- #132: the residual add and the router ------------------------------ */

/* out is slot 0, the residual (the validator's rule): slot 0 += in * 1.
   x * 1 is exact and the add rounds once, which is the CPU's copy +
   add_i bit for bit. */
static int graph_op_add(hexkl_graph *g, const htp_graph_op *op,
                        graph_call *call, const float *in, float *out) {
  (void)g;
  (void)call;
  hvx_scale_add_rows_f32(out, in, 1.0f, op->N);
  return AEE_SUCCESS;
}

/* The logits go to the out slot (nothing reads them); the routing goes to
   g->route_* in ascending expert order -- tryMoeLayerOnAccelerator's
   grouping -- and the call's routing points at it for the MOE op next. */
static int graph_op_router_topk(hexkl_graph *g, const htp_graph_op *op,
                                graph_call *call, const float *in, float *out) {
  const uint32_t i = (uint32_t)(op - g->ops);
  uint32_t sel[HTP_GRAPH_ROUTER_MAX_EXPERTS], e, r, n = 0;
  float w[HTP_GRAPH_ROUTER_MAX_EXPERTS],
    by_expert[HTP_GRAPH_ROUTER_MAX_EXPERTS];
  if (g->param[i] == NULL || g->state[i] == NULL) {
    return AEE_EBADSTATE;
  }
  hvx_router_topk_f32(in, g->param[i], g->state[i], op->K, op->n_experts,
                      op->top_k, out, sel, w, call->env->pool);
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

static int graph_op_fc(hexkl_graph *g, const htp_graph_op *op, graph_call *call,
                       const float *in, float *out) {
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
   with the spec's integer division on one thread). */
static int graph_op_dense_ffn(hexkl_graph *g, const htp_graph_op *op,
                              graph_call *call, const float *in, float *out) {
  float *up = g->ffn, *gate = g->ffn + op->N, *act = g->ffn + 2u * op->N;
  int rc;
  if (call->env->fc == NULL) {
    return AEE_EBADSTATE;
  }
  graph_prep(op, in, op->K, &g->act);
  rc = call->env->fc(call->env->fc_ctx, op->h_gu[0], op->feed, &g->act, up);
  if (rc == AEE_SUCCESS) {
    rc = call->env->fc(call->env->fc_ctx, op->h_gu[1], op->feed, &g->act, gate);
  }
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  hvx_swiglu_cpu_f32(gate, up, act, op->N, call->env->pool);
  graph_prep(op, act, op->N, &g->act);
  return call->env->fc(call->env->fc_ctx, op->h_dn[0], op->feed, &g->act, out);
}

/* The slices into g->logits (forward hands them out as the op's output),
   then the first maximum, as the CPU's sampler picks it. */
static int graph_op_lm_head(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out) {
  int rc;
  (void)out;
  if (call->env->fc == NULL) {
    return AEE_EBADSTATE;
  }
  graph_prep(op, in, op->K, &g->act);
  rc = graph_q4m1_parts(g, op, call, op->h_gu, op->n_experts, g->logits);
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

int hexkl_graph_init(const uint32_t *words, uint32_t n_words,
                     const hexkl_weight_u8i4_table *tbl,
                     const hexkl_graph_q4m1_shape *q4m1, uint32_t n_q4m1,
                     hexkl_graph **out) {
  uint32_t n_ops = 0, i, slot_words = 0, n_attn = 0, ffn_n = 0;
  uint32_t q4m1_ops = 0, vocab_out = 0;
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
    if ((HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) != 0u) {
      if (q4m1 == NULL || graph_check_q4m1(op, q4m1, n_q4m1) != AEE_SUCCESS) {
        return AEE_EINVHANDLE;
      }
      ++q4m1_ops;
      if (op->kind == HTP_OP_DENSE_FFN && op->N > ffn_n) {
        ffn_n = op->N;
      }
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
      g->ordinal[i] = n_attn++;
    }
  }
  *out = g;
  return AEE_SUCCESS;
}

void hexkl_graph_free(hexkl_graph *g) {
  uint32_t i;
  if (g == NULL) {
    return;
  }
  for (i = 0; i < g->n_ops; ++i) {
    free(g->param[i]);
    free(g->state[i]);
    free(g->experts[i]);
  }
  free(g->rope_cs);
  free(g->slots);
  free(g->act_buf);
  free(g->q4m1);
  free(g->ffn);
  free(g->logits);
  free(g);
}

/** @brief The parameter's length for (kind, which), 0 when the kind does
 *  not take it. */
static uint32_t graph_param_len(const htp_graph_op *op, uint32_t which) {
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
  case HTP_GRAPH_PARAM_ROUTER_BIAS:
    return op->kind == HTP_OP_ROUTER_TOPK ? op->n_experts : 0u;
  default:
    return 0u;
  }
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
  if (which == HTP_GRAPH_PARAM_ROPE_TABLE) {
    if (op != HTP_GRAPH_NO_OP) {
      return AEE_EBADITEM;
    }
    want = alloc = g->max_seq * 64u;
    dst = &g->rope_cs;
  } else {
    if (op >= g->n_ops) {
      return AEE_EBADITEM;
    }
    want = graph_param_len(&g->ops[op], which);
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
    /* one vector per weight row: [K][E] padded to [K][32], zero lanes */
    const uint32_t K = g->ops[op].K, E = g->ops[op].n_experts;
    const uint32_t W = HTP_GRAPH_ROUTER_MAX_EXPERTS;
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

int hexkl_graph_uses_handle(const hexkl_graph *g, uint32_t handle) {
  uint32_t i, e;
  if (g == NULL) {
    return 0;
  }
  for (i = 0; i < g->n_ops; ++i) {
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
        (HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) == 0u) {
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
