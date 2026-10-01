// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   graph_host_check.c
 * @date   23 Sep 2026
 * @brief  Host check of the per-token entry's op table and forward loop
 *         (#85), and of the small ops and m=1 attention wired into it
 *         (#130) against their scalar specs
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

/* htp_graph_desc.h's validator and LFM2 builder, and hexkl_graph.c's init /
   forward loop.

   Validator half: the LFM2.5-8B-A1B op list validates; each mutation of
   plan 85 section 1 fails with its own code (the table printed below), and
   none of them is AEE_EBADPARM, which stays the stale-skel symptom.

   Forward half, on the tiny fixture's shapes (hidden 64, inter 64, 4
   experts, top-2, unittest_causallm_lfm2_moe.cpp): with no op resident
   forward is the identity; with MOE resident it hands
   hexkl_mm_u8i4_moe_layer_run exactly what the per-layer entry hands it
   and its output is byte-equal to a direct call. The kernel itself is a
   stand-in here that records its arguments and mixes every input into
   its output -- the loop structure inside it is moe_layer_host_check's
   job, and bit identity on the real kernel holds by construction once the
   arguments are the same (same function, same session state).

   Stretch half (#130), on the hd64 fixture's shapes (hidden 128, 2 q
   heads, 1 kv head, head_dim 64, max_seq 64): the REAL hvx_m1_ops_f32.c,
   hvx_conv_gate_f32.c and hvx_attn_m1_f32.c on hvx_emu/, driven through
   graph_set_param and forward, memcmp'd against m1_ops_det.h and
   attn_m1_det.h stretch by stretch -- [RMSNORM], an 8-token
   [CONV1D_GATE] chain from a seeded state, [QK_NORM ROPE ATTN_M1] at four
   positions after a kv_append seed -- plus resume_at, the builder's slot
   routing, the per-op pcycles and the forward-time refusals (a missing
   parameter, state, table or cache, and the cache's hole).

   #132, same shapes with MOE on the stand-in: [ADD RMSNORM] after an op-0
   RMSNORM seeded slot 0, and [ADD RMSNORM ROUTER_TOPK MOE ADD RMSNORM]
   across a layer boundary, against the scalar composition -- slot 0 as
   the residual across calls, the router's routing reaching the MOE op in
   the same call, and a router with no weights refused.

   #132 Part B, the hd64 shape with every kind resident: the REAL
   hvx_q4_gemv_f32.c (quantizer and GEMV) on hvx_emu/ behind a host fc
   runner, driven through FC (q | k | v, three parts), DENSE_FFN and
   LM_HEAD (two slices, the argmax, a tie), and [RMSNORM LM_HEAD] as a
   stretch, memcmp'd against the CPU-order specs (q4_gemv_cpu_det.h,
   m1_swiglu_cpu_det, m1_argmax_first); init's handle refusals; the feed
   word reaching the runner; and the two sessions' complementary masks on
   the LFM2.5 list (each validates, together they are every kind, and
   their stretches alternate at the MoE hops). run_host_checks.sh then
   builds this file against mutants of hexkl_graph.c, each of which must
   fail it. */
#include "hexkl_graph.h"
#include "htp_graph_desc.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attn_m1_det.h"
#include "hvx_q4_gemv_f32.h"
#include "hvx_worker_pool.h"
#include "m1_ops_det.h"
#include "q4_gemv_native_det.h"

/** @brief A real pthread pool (hvx_worker_pool.c on stub/qurt.h) for the
 *  env: since #132 PR 2 the router op runs its chains on it. */
static hvx_worker_pool *real_pool(void) {
  static hvx_worker_pool *p;
  if (p == NULL) {
    p = hvx_worker_pool_create(3u);
  }
  return p;
}

static int g_fail;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);                              \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      ++g_fail;                                                                \
    }                                                                          \
  } while (0)

/* ---- the kernel stand-in ---------------------------------------------- */
typedef struct {
  const void *tbl, *vtcm, *pool, *scratch;
  uint32_t vtcm_size, config_off, M, K, inter, N_out, n_experts, flags;
  uint32_t h_gu[HTP_GRAPH_MAX_EXPERTS], h_dn[HTP_GRAPH_MAX_EXPERTS];
  uint32_t row_count[HTP_GRAPH_MAX_EXPERTS];
  uint32_t n_calls;
} moe_args;
static moe_args g_last;

int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch, uint32_t flags) {
  uint32_t e, c, r = 0;
  moe_args *a = &g_last;
  memset(a, 0, sizeof(*a));
  a->tbl = tbl;
  a->vtcm = vtcm_base;
  a->pool = pool;
  a->scratch = scratch;
  a->vtcm_size = vtcm_size;
  a->config_off = config_off;
  a->M = M;
  a->K = K;
  a->inter = inter;
  a->N_out = N_out;
  a->n_experts = n_experts;
  a->flags = flags;
  memcpy(a->h_gu, h_gate_up, n_experts * sizeof(uint32_t));
  memcpy(a->h_dn, h_down, n_experts * sizeof(uint32_t));
  memcpy(a->row_count, row_count, n_experts * sizeof(uint32_t));
  a->n_calls = 1;
  memset(out_f32, 0, (size_t)M * N_out * sizeof(float));
  for (e = 0; e < n_experts; ++e) {
    uint32_t i;
    for (i = 0; i < row_count[e]; ++i, ++r) {
      const float *x = act_f32 + (size_t)row_index[r] * K;
      float *y = out_f32 + (size_t)row_index[r] * N_out;
      const float wsum =
        (float)(tbl->slots[h_gate_up[e]].N + tbl->slots[h_down[e]].K +
                7u * h_gate_up[e] + 3u * h_down[e] + flags + inter);
      for (c = 0; c < N_out; ++c)
        y[c] += row_weight[r] * (x[c % K] * wsum + (float)c);
    }
  }
  return AEE_SUCCESS;
}

/* ---- shapes ------------------------------------------------------------ */
static const char *const kLfm25Layers = "CCACCCACCCACCCACCCACCACC";
static const htp_graph_lfm2_shape kLfm25 = {24, 2, 2048, 7168,   1792, 32,   4,
                                            32, 8, 64,   128000, 2048, 1e-5f};
/* unittest_causallm_lfm2_moe.cpp:85-113, layer_types [attention, conv] */
static const htp_graph_lfm2_shape kTiny = {2, 0, 64, 64, 64, 4,    2,
                                           8, 4, 8,  32, 8,  1e-6f};
/* lfm2_moe_tiny_hd64 (plan 130 section 3.4): what the attention kinds
   need -- head_dim 64, a gqa of 2, max_seq a multiple of 32 */
static const htp_graph_lfm2_shape kHd64 = {3, 1, 128, 64, 32, 4,    2,
                                           2, 1, 64,  32, 64, 1e-6f};
#define ALL_KINDS                                                              \
  (HTP_GRAPH_KIND_BIT(HTP_OP_MOE) | HTP_GRAPH_KIND_BIT(HTP_OP_RMSNORM) |       \
   HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) | HTP_GRAPH_KIND_BIT(HTP_OP_ROPE) |      \
   HTP_GRAPH_KIND_BIT(HTP_OP_CONV1D_GATE) |                                    \
   HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1))
/* #132's D mask: the six plus the residual add and the router */
#define D_KINDS                                                                \
  (ALL_KINDS | HTP_GRAPH_KIND_BIT(HTP_OP_ADD) |                                \
   HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK))

static uint32_t build(uint32_t *w, uint32_t cap, const htp_graph_lfm2_shape *s,
                      const char *layers, uint32_t resident) {
  uint8_t attn[HTP_GRAPH_MAX_LAYERS];
  uint32_t l;
  for (l = 0; l < s->n_layers; ++l)
    attn[l] = layers[l] == 'A';
  return htp_graph_lfm2_build(w, cap, s, attn, resident);
}

static uint32_t count_kind(const uint32_t *w, uint32_t kind) {
  uint32_t i, n = 0;
  for (i = 0; i < w[3]; ++i)
    n += htp_graph_op_cat(w, i)->kind == kind;
  return n;
}
static uint32_t nth_op(const uint32_t *w, uint32_t kind, uint32_t nth) {
  uint32_t i;
  for (i = 0; i < w[3]; ++i)
    if (htp_graph_op_cat(w, i)->kind == kind && nth-- == 0u)
      return i;
  return HTP_GRAPH_NO_OP;
}

/* ---- validator half ---------------------------------------------------- */
typedef void (*mutate_fn)(uint32_t *w, uint32_t *n_words);
static void mut_shape(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_MOE, 3))->K += 32u;
  (void)n;
}
static void mut_kind(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, 5)->kind = 99u;
  (void)n;
}
static void mut_attn_in_conv(uint32_t *w, uint32_t *n) {
  /* layer 0 is conv; its CONV1D_GATE has N == hidden, so only the layer
     check can refuse an ATTN_M1 there */
  htp_graph_op_at(w, nth_op(w, HTP_OP_CONV1D_GATE, 0))->kind = HTP_OP_ATTN_M1;
  (void)n;
}
static void mut_next_mm_back(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, 6)->next_mm = 1u; /* op 1 is an FC, but behind */
  (void)n;
}
static void mut_next_mm_not_mm(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, 1)->next_mm = 2u; /* 2 is CONV1D_GATE, no weights */
  (void)n;
}
static void mut_resident_no_kernel(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, 1)->resident = 1u; /* op 1 is an FC */
  (void)n;
}
/* #130: the v2 record's rules, one negative per new kind */
static void mut_eps_zero(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, 0)->eps_bits = 0u;
  (void)n;
}
static void mut_rope_before_qk_norm(uint32_t *w, uint32_t *n) {
  /* the two records differ only in kind, so swapping the kinds is the
     order mutation and nothing else */
  const uint32_t q = nth_op(w, HTP_OP_QK_NORM, 0);
  htp_graph_op_at(w, q)->kind = HTP_OP_ROPE;
  htp_graph_op_at(w, q + 1u)->kind = HTP_OP_QK_NORM;
  (void)n;
}
static void mut_rope_head_dim_32(uint32_t *w, uint32_t *n) {
  /* K = 3072 = (6 + 2) x 12 x 32 keeps the record consistent, so only
     the kernel's head_dim 64 rule can refuse it */
  htp_graph_op *op = htp_graph_op_at(w, nth_op(w, HTP_OP_ROPE, 2));
  op->head_dim = 32u;
  op->n_kv = 12u;
  op->gqa = 6u;
  (void)n;
}
static void mut_attn_head_dim_32(uint32_t *w, uint32_t *n) {
  /* K = 3072 = (4 + 2) x 16 x 32 and N = 2048 = 4 x 16 x 32: consistent,
     so only #152's head_dim 64 rule (the fp16 CPU order) can refuse it */
  htp_graph_op *op = htp_graph_op_at(w, nth_op(w, HTP_OP_ATTN_M1, 2));
  op->head_dim = 32u;
  op->n_kv = 16u;
  op->gqa = 4u;
  (void)n;
}
/* [plan 201 S4] head_dim 128 is a kernel shape now (K = 3072 = (4 + 2) x 4
   x 128, N = 2048 = 4 x 4 x 128), but 1/sqrt(128) is not an fp16 value:
   without a scale in eps_bits it is refused; and a scale that is not one */
static void mut_attn_hd128_no_scale(uint32_t *w, uint32_t *n) {
  htp_graph_op *op = htp_graph_op_at(w, nth_op(w, HTP_OP_ATTN_M1, 2));
  op->head_dim = 128u;
  op->n_kv = 4u;
  op->gqa = 4u;
  op = htp_graph_op_at(w, nth_op(w, HTP_OP_ROPE, 2));
  op->head_dim = 128u;
  op->n_kv = 4u;
  op->gqa = 4u;
  op = htp_graph_op_at(w, nth_op(w, HTP_OP_QK_NORM, 2));
  op->head_dim = 128u;
  op->n_kv = 4u;
  op->gqa = 4u;
  (void)n;
}
static void mut_attn_scale_not_fp16(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_ATTN_M1, 2))->eps_bits =
    0x3DCCCCCDu; /* 0.1f */
  (void)n;
}
static void mut_qk_norm_head_dim_96(uint32_t *w, uint32_t *n) {
  /* K = 3072 = (2 + 2) x 8 x 96: consistent, and the per-head norm's
     chunk would not be a power of two */
  htp_graph_op *op = htp_graph_op_at(w, nth_op(w, HTP_OP_QK_NORM, 1));
  op->head_dim = 96u;
  op->gqa = 2u;
  (void)n;
}
static void mut_attn_without_rope(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_ROPE, 4))->resident = 0u;
  (void)n;
}
static void mut_rmsnorm_k_not_32(uint32_t *w, uint32_t *n) {
  /* the final norm: K == hidden is not a validator rule for RMSNORM, so
     only the kernel's width rule can refuse 2048 + 24 (plan 201 S4: any
     multiple of 32 is taken, 2080 included) */
  htp_graph_op_at(w, w[3] - 2u)->K = 2072u;
  htp_graph_op_at(w, w[3] - 2u)->N = 2072u;
  (void)n;
}
/* [plan 201 S4] an RMSNORM feed bit other than N1; a softmax router (eps
   set) norming over its own input slot */
static void mut_rmsnorm_feed_2(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_RMSNORM, 2))->feed = 2u;
  (void)n;
}
static void mut_softmax_router_in_is_out(uint32_t *w, uint32_t *n) {
  htp_graph_op *op = htp_graph_op_at(w, nth_op(w, HTP_OP_ROUTER_TOPK, 1));
  op->eps_bits = 0x358637BDu; /* 1e-6f */
  op->out_slot = op->in_slot;
  (void)n;
}
/* #132: the residual add's and the router's rules */
static void mut_add_rmsnorm_cpu(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_RMSNORM, 7))->resident = 0u;
  (void)n;
}
static void mut_router_moe_cpu(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_MOE, 5))->resident = 0u;
  (void)n;
}
static void mut_add_out_slot_1(uint32_t *w, uint32_t *n) {
  htp_graph_op_at(w, nth_op(w, HTP_OP_ADD, 3))->out_slot = 1u;
  (void)n;
}
static void mut_truncated(uint32_t *w, uint32_t *n) {
  (void)w;
  *n -= 1u;
}
static void mut_magic(uint32_t *w, uint32_t *n) {
  w[0] ^= 1u;
  (void)n;
}
static void mut_version(uint32_t *w, uint32_t *n) {
  w[1] += 1u;
  (void)n;
}
static void mut_lm_head_not_last(uint32_t *w, uint32_t *n) {
  /* every op after the last MoE names it as next_mm; those are dropped
     too, or BADITEM fires first */
  uint32_t i;
  htp_graph_op_at(w, w[3] - 1u)->kind = HTP_OP_RMSNORM;
  for (i = 0; i < w[3]; ++i)
    if (htp_graph_op_at(w, i)->next_mm == w[3] - 1u)
      htp_graph_op_at(w, i)->next_mm = HTP_GRAPH_NO_OP;
  (void)n;
}
static const struct {
  const char *name;
  mutate_fn fn;
  uint32_t want;
} kMutations[] = {
  {"wrong shape (MoE K != hidden)", mut_shape, HTP_GRAPH_E_INVALIDFORMAT},
  {"unknown kind", mut_kind, HTP_GRAPH_E_NOTYPE},
  {"attention op in a conv layer", mut_attn_in_conv, HTP_GRAPH_E_INVALIDITEM},
  {"next_mm points backwards", mut_next_mm_back, HTP_GRAPH_E_BADITEM},
  {"next_mm names a non-weight op", mut_next_mm_not_mm, HTP_GRAPH_E_BADITEM},
  {"resident bit on a kind with no kernel", mut_resident_no_kernel,
   HTP_GRAPH_E_CLASSNOTSUPPORT},
  {"truncated list", mut_truncated, HTP_GRAPH_E_INCOMPLETEITEM},
  {"bad magic", mut_magic, HTP_GRAPH_E_INVALIDFORMAT},
  {"wrong version", mut_version, HTP_GRAPH_E_UNSUPPORTED},
  {"lm_head not last", mut_lm_head_not_last, HTP_GRAPH_E_INVALIDITEM},
  {"RMSNORM eps_bits 0", mut_eps_zero, HTP_GRAPH_E_INVALIDFORMAT},
  {"ROPE before QK_NORM", mut_rope_before_qk_norm, HTP_GRAPH_E_INVALIDITEM},
  {"ROPE resident at head_dim 32", mut_rope_head_dim_32,
   HTP_GRAPH_E_SCHEMENOTSUPPORTED},
  {"ATTN_M1 resident at head_dim 32", mut_attn_head_dim_32,
   HTP_GRAPH_E_SCHEMENOTSUPPORTED},
  {"ATTN_M1 resident, its ROPE not", mut_attn_without_rope,
   HTP_GRAPH_E_NOTALLOWED},
  {"QK_NORM resident at head_dim 96", mut_qk_norm_head_dim_96,
   HTP_GRAPH_E_SCHEMENOTSUPPORTED},
  {"ATTN_M1 resident at head_dim 128, no scale", mut_attn_hd128_no_scale,
   HTP_GRAPH_E_SCHEMENOTSUPPORTED},
  {"ATTN_M1 scale 0.1 (not fp16)", mut_attn_scale_not_fp16,
   HTP_GRAPH_E_SCHEMENOTSUPPORTED},
  {"RMSNORM resident at K 2072", mut_rmsnorm_k_not_32,
   HTP_GRAPH_E_SCHEMENOTSUPPORTED},
  {"RMSNORM feed 2 (only N1 is a norm bit)", mut_rmsnorm_feed_2,
   HTP_GRAPH_E_INVALIDFORMAT},
  {"softmax ROUTER_TOPK in_slot == out_slot", mut_softmax_router_in_is_out,
   HTP_GRAPH_E_INVALIDFORMAT},
  {"ADD resident, a RMSNORM not", mut_add_rmsnorm_cpu, HTP_GRAPH_E_NOTALLOWED},
  {"ROUTER_TOPK resident, its MOE not", mut_router_moe_cpu,
   HTP_GRAPH_E_NOTALLOWED},
  {"ADD out_slot 1", mut_add_out_slot_1, HTP_GRAPH_E_INVALIDFORMAT},
};

static void check_validator(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  static uint32_t m[sizeof(w) / sizeof(w[0])];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  const uint32_t resident_ok = D_KINDS;
  uint32_t n = build(w, cap, &kLfm25, kLfm25Layers, resident_ok);
  uint32_t n_ops = 0, i, rc;
  char names[128];
  CHECK(n != 0u, "LFM2.5 build refused");
  CHECK(HTP_GRAPH_OP_WORDS == 80u && HTP_GRAPH_VERSION == 2u, "wire v2");
  rc = htp_graph_validate(w, n, resident_ok, &n_ops);
  CHECK(rc == 0u, "LFM2.5 validate: %s", htp_graph_err_name(rc));
  CHECK(n_ops == 228u, "LFM2.5 n_ops %u (want 228)", n_ops);
  CHECK(count_kind(w, HTP_OP_MOE) == 22u, "MoE ops %u",
        count_kind(w, HTP_OP_MOE));
  CHECK(count_kind(w, HTP_OP_DENSE_FFN) == 2u, "dense FFN ops");
  CHECK(count_kind(w, HTP_OP_ATTN_M1) == 6u, "attention ops");
  CHECK(count_kind(w, HTP_OP_CONV1D_GATE) == 18u, "conv ops");
  CHECK(count_kind(w, HTP_OP_LM_HEAD) == 1u, "lm_head");
  /* MoE ops in layer order, resident, and every next_mm forward */
  for (i = 0; i < 22u; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(w, nth_op(w, HTP_OP_MOE, i));
    CHECK(op->layer == i + 2u && op->resident == 1u && op->n_experts == 32u &&
            op->top_k == 4u && op->K == 2048u && op->N == 1792u &&
            op->N_out == 2048u,
          "MoE op %u record", i);
  }
  for (i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(w, i);
    CHECK(op->next_mm == HTP_GRAPH_NO_OP || op->next_mm > i, "next_mm %u", i);
    CHECK(op->resident == ((resident_ok & HTP_GRAPH_KIND_BIT(op->kind)) != 0u),
          "resident bit %u", i);
    if (op->kind == HTP_OP_ATTN_M1)
      CHECK(op->n_kv == 8u && op->gqa == 4u && op->head_dim == 64u &&
              op->K == 3072u && op->N == 2048u,
            "ATTN_M1 record %u", i);
    if (op->kind == HTP_OP_RMSNORM || op->kind == HTP_OP_QK_NORM)
      CHECK(op->eps_bits == 0x3727C5ACu, "eps_bits %u = 0x%x", i, op->eps_bits);
  }
  CHECK(htp_graph_op_cat(w, 1)->next_mm == 3u, "op 1 (in_proj) -> op 3");
  CHECK(htp_graph_op_cat(w, n_ops - 1u)->next_mm == HTP_GRAPH_NO_OP,
        "lm_head has no next");
  printf("GRAPH VALIDATOR OK: LFM2.5-8B-A1B n_ops=%u moe=22 words=%u "
         "resident=%s\n",
         n_ops, n, htp_graph_kinds_str(resident_ok, names, sizeof(names)));
  /* the mask parser both sides read NNTR_HTP_FORWARD_KINDS with */
  CHECK(htp_graph_kinds_parse("MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1") ==
          ALL_KINDS,
        "kinds parse (six)");
  CHECK(htp_graph_kinds_parse("MOE,RMSNORM,QK_NORM,ROPE,CONV1D_GATE,ATTN_M1,"
                              "ADD,ROUTER_TOPK") == resident_ok,
        "kinds parse (D)");
  CHECK(htp_graph_kinds_parse("MOE") == HTP_GRAPH_KIND_BIT(HTP_OP_MOE) &&
          htp_graph_kinds_parse("MOE,BOGUS") == 0u &&
          htp_graph_kinds_parse("") == 0u,
        "kinds parse (one, bad, empty)");
  /* the hd8 fixture with the attention kinds resident: the kernels'
     head_dim rule, end to end through the validator */
  {
    static uint32_t t[sizeof(w) / sizeof(w[0])];
    const uint32_t nt = build(t, cap, &kTiny, "AC", resident_ok);
    rc = htp_graph_validate(t, nt, resident_ok, NULL);
    CHECK(rc == HTP_GRAPH_E_SCHEMENOTSUPPORTED, "tiny hd8 all kinds: %s",
          htp_graph_err_name(rc));
    printf("  tiny fixture (head_dim 8), all kinds resident -> %s (0x%x)\n",
           htp_graph_err_name(rc), rc);
  }

  printf("  mutation                                  -> code\n");
  for (i = 0; i < sizeof(kMutations) / sizeof(kMutations[0]); ++i) {
    uint32_t nm = n;
    memcpy(m, w, n * sizeof(uint32_t));
    kMutations[i].fn(m, &nm);
    rc = htp_graph_validate(m, nm, resident_ok, NULL);
    printf("  %-41s -> %s (0x%x)\n", kMutations[i].name, htp_graph_err_name(rc),
           rc);
    CHECK(rc == kMutations[i].want, "%s: got %s want %s", kMutations[i].name,
          htp_graph_err_name(rc), htp_graph_err_name(kMutations[i].want));
    CHECK(rc != (uint32_t)AEE_EBADPARM, "%s returned AEE_EBADPARM",
          kMutations[i].name);
  }
  /* The header's codes are the SDK's (this side: offset 0). */
  CHECK(HTP_GRAPH_E_INVALIDFORMAT == (uint32_t)AEE_EINVALIDFORMAT &&
          HTP_GRAPH_E_INVHANDLE == (uint32_t)AEE_EINVHANDLE &&
          HTP_GRAPH_E_NOTYPE == (uint32_t)AEE_ENOTYPE &&
          HTP_GRAPH_E_INVALIDITEM == (uint32_t)AEE_EINVALIDITEM &&
          HTP_GRAPH_E_BADITEM == (uint32_t)AEE_EBADITEM &&
          HTP_GRAPH_E_CLASSNOTSUPPORT == (uint32_t)AEE_ECLASSNOTSUPPORT &&
          HTP_GRAPH_E_INCOMPLETEITEM == (uint32_t)AEE_EINCOMPLETEITEM &&
          HTP_GRAPH_E_BADSTATE == (uint32_t)AEE_EBADSTATE &&
          HTP_GRAPH_E_SCHEMENOTSUPPORTED == (uint32_t)AEE_ESCHEMENOTSUPPORTED &&
          HTP_GRAPH_E_NOTALLOWED == (uint32_t)AEE_ENOTALLOWED,
        "error code values drifted from AEEStdErr.h");
  CHECK(hexkl_graph_resident_kinds() == HTP_GRAPH_KINDS_ALL,
        "kernel table: resident kinds 0x%x", hexkl_graph_resident_kinds());

  /* #132 Part B: the same list for two sessions, complementary masks */
  {
    static uint32_t s1[sizeof(w) / sizeof(w[0])], s2[sizeof(w) / sizeof(w[0])];
    uint32_t n_s1 = 0, n_s2 = 0, hops = 0, bad = 0;
    n = build(w, cap, &kLfm25, kLfm25Layers, HTP_GRAPH_KINDS_ALL);
    rc = htp_graph_validate(w, n, HTP_GRAPH_KINDS_ALL, NULL);
    CHECK(rc == 0u, "LFM2.5 all kinds: %s", htp_graph_err_name(rc));
    memcpy(s1, w, n * sizeof(uint32_t));
    memcpy(s2, w, n * sizeof(uint32_t));
    htp_graph_set_resident(s1, HTP_GRAPH_KINDS_S1);
    htp_graph_set_resident(s2, HTP_GRAPH_KINDS_S2);
    CHECK((HTP_GRAPH_KINDS_S1 & HTP_GRAPH_KINDS_S2) == 0u &&
            (HTP_GRAPH_KINDS_S1 | HTP_GRAPH_KINDS_S2) == HTP_GRAPH_KINDS_ALL,
          "masks do not partition the kinds");
    rc = htp_graph_validate(s1, n, HTP_GRAPH_KINDS_ALL, NULL);
    CHECK(rc == 0u, "S1 mask: %s", htp_graph_err_name(rc));
    rc = htp_graph_validate(s2, n, HTP_GRAPH_KINDS_ALL, NULL);
    CHECK(rc == 0u, "S2 mask: %s", htp_graph_err_name(rc));
    /* every op resident in exactly one session; S1's stretches are
       exactly [ROUTER_TOPK MOE] and each is a hop between two of S2's */
    for (i = 0; i < n_ops; ++i) {
      const htp_graph_op *a = htp_graph_op_cat(s1, i),
                         *b = htp_graph_op_cat(s2, i);
      const int starts_s1 =
        a->resident && (i == 0u || !htp_graph_op_cat(s1, i - 1u)->resident);
      const int starts_s2 =
        b->resident && (i == 0u || !htp_graph_op_cat(s2, i - 1u)->resident);
      bad += a->resident == b->resident;
      if (starts_s1) {
        ++n_s1;
        bad += a->kind != HTP_OP_ROUTER_TOPK || i + 2u >= n_ops ||
               htp_graph_op_cat(s1, i + 1u)->kind != HTP_OP_MOE ||
               htp_graph_op_cat(s1, i + 2u)->resident;
      }
      n_s2 += starts_s2;
      hops += (starts_s1 || starts_s2) && i != 0u;
    }
    CHECK(bad == 0u && n_s1 == 22u && n_s2 == 23u && hops == 44u &&
            htp_graph_op_cat(s2, 0)->resident &&
            htp_graph_op_cat(s2, n_ops - 1u)->resident,
          "session split: bad=%u S1 stretches=%u S2 stretches=%u hops=%u", bad,
          n_s1, n_s2, hops);
    printf("GRAPH SESSION MASKS OK: S1=%s S2 = the other %u kinds; LFM2.5 "
           "S2 stretches=%u S1 stretches=%u hops/token=%u\n",
           htp_graph_kinds_str(HTP_GRAPH_KINDS_S1, names, sizeof(names)),
           HTP_OP_KIND_N - 2u, n_s2, n_s1, hops);
  }
}

/* ---- forward half ------------------------------------------------------ */
static hexkl_weight_u8i4_table g_tbl;
static uint8_t g_vtcm[64];
static hexkl_moe_scratch g_scratch;

static void register_weight(uint32_t h, uint32_t K, uint32_t N) {
  g_tbl.slots[h].in_use = 1;
  g_tbl.slots[h].K = K;
  g_tbl.slots[h].N = N;
}
/** @brief [plan 201 S1] Binds MoE op @a op's EXPERTS table: h_gu[0..E)
 *  then h_dn[0..E), as graph_set_param's f32 words. */
static int set_experts(hexkl_graph *g, uint32_t op, const uint32_t *h_gu,
                       const uint32_t *h_dn, uint32_t E) {
  float t[2u * HTP_GRAPH_MAX_EXPERTS];
  memcpy(t, h_gu, E * sizeof(uint32_t));
  memcpy(t + E, h_dn, E * sizeof(uint32_t));
  return hexkl_graph_set_param(g, op, HTP_GRAPH_PARAM_EXPERTS, t, 2u * E);
}

/* Layer l's expert e: gate_up handle 10 + 40 l + e, down 30 + 40 l + e. */
static uint32_t g_tiny_gu[2][4], g_tiny_dn[2][4];
static void bind_tiny(void) {
  uint32_t l, e;
  for (l = 0; l < 2u; ++l) {
    for (e = 0; e < 4u; ++e) {
      g_tiny_gu[l][e] = 10u + 40u * l + e;
      g_tiny_dn[l][e] = 30u + 40u * l + e;
      register_weight(g_tiny_gu[l][e], 64u, 128u);
      register_weight(g_tiny_dn[l][e], 64u, 64u);
    }
  }
}

static float frand(uint32_t *s) {
  *s = *s * 1664525u + 1013904223u;
  return (float)(*s >> 8) / 16777216.0f * 4.0f - 2.0f;
}

static void check_forward(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  const uint32_t resident_ok = HTP_GRAPH_KIND_BIT(HTP_OP_MOE);
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  uint32_t n, i, rc, resume, seed = 12345u;
  float act[64], out[64], ref[64], sentinel[64];
  /* routing: experts 1 and 3, the two rows of the one token */
  const uint32_t row_index[2] = {0u, 0u};
  const uint32_t row_count[4] = {0u, 1u, 0u, 1u};
  const float row_weight[2] = {0.7f, 0.3f};
  hexkl_graph_routing routing = {row_index, row_count, row_weight, 2u, 4u};
  const uint32_t flag_cases[2] = {0u, HEXKL_MOE_FLAG_M1_GEMV};
  uint32_t f, l;

  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  env.vtcm_base = g_vtcm;
  env.vtcm_size = sizeof(g_vtcm);
  env.config_off = 32u;
  env.pool = real_pool(); /* the stand-in passes it through */
  env.scratch = &g_scratch;
  for (i = 0; i < 64u; ++i) {
    act[i] = frand(&seed);
    sentinel[i] = -12345.0f;
  }

  /* (1) every op non-resident: the identity at every start op */
  n = build(w, cap, &kTiny, "AC", 0u);
  CHECK(n != 0u, "tiny build");
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "init (none resident): %s",
        htp_graph_err_name(rc));
  if (g != NULL) {
    CHECK(g->n_ops == 22u && g->slot_words == 0u, "tiny n_ops %u slot_words %u",
          g->n_ops, g->slot_words);
    for (i = 0; i < g->n_ops; ++i) {
      const uint32_t in_len = htp_graph_op_in_words(&g->ops[i]);
      float in[512];
      uint32_t j;
      memcpy(out, sentinel, sizeof(out));
      resume = 99u;
      g_last.n_calls = 0;
      rc = (uint32_t)hexkl_graph_forward(g, &env, i, 1000u, 0u, &routing, in,
                                         in_len, out, 64u, &resume);
      CHECK(rc == 0u && resume == i, "identity at op %u: rc %s resume %u", i,
            htp_graph_err_name(rc), resume);
      CHECK(memcmp(out, sentinel, sizeof(out)) == 0, "identity touched out");
      CHECK(g_last.n_calls == 0u, "identity called the kernel");
      for (j = 0; j < HTP_GRAPH_MAX_OPS; ++j)
        CHECK(g->op_pcycles[j] == 0u, "identity pcycles[%u] != 0", j);
    }
    CHECK(hexkl_graph_uses_handle(g, 10u) == 0, "non-resident uses handle");
    hexkl_graph_free(g);
    g = NULL;
  }
  printf("GRAPH FORWARD IDENTITY OK (22 start ops, no op resident)\n");

  /* (2) MOE resident: the handles are each op's EXPERTS table (plan 201
     S1), bound after init and checked there, not in the record */
  n = build(w, cap, &kTiny, "AC", resident_ok);
  bind_tiny();
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "init (MOE resident): %s",
        htp_graph_err_name(rc));
  if (g == NULL)
    return;
  CHECK(g->slot_words == 64u, "slot_words %u", g->slot_words);
  {
    const uint32_t m0 = nth_op(w, HTP_OP_MOE, 0), m1 = nth_op(w, HTP_OP_MOE, 1);
    rc = (uint32_t)hexkl_graph_forward(g, &env, m0, 1000u, 0u, &routing, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_BADSTATE, "no EXPERTS table: %s",
          htp_graph_err_name(rc));
    printf("  MOE op with no EXPERTS table              -> %s (0x%x)\n",
           htp_graph_err_name(rc), rc);
    g_tiny_dn[1][2] = 7u; /* a free slot */
    rc = (uint32_t)set_experts(g, m1, g_tiny_gu[1], g_tiny_dn[1], 4u);
    CHECK(rc == HTP_GRAPH_E_INVHANDLE, "free handle: %s",
          htp_graph_err_name(rc));
    printf("  missing weight handle                     -> %s (0x%x)\n",
           htp_graph_err_name(rc), rc);
    g_tiny_dn[1][2] = 72u;
    g_tbl.slots[31].N = 96u; /* layer 0 expert 1's down: inter differs */
    rc = (uint32_t)set_experts(g, m0, g_tiny_gu[0], g_tiny_dn[0], 4u);
    CHECK(rc == HTP_GRAPH_E_INVHANDLE, "mis-shaped handle: %s",
          htp_graph_err_name(rc));
    printf("  weight handle of another shape            -> %s (0x%x)\n",
           htp_graph_err_name(rc), rc);
    g_tbl.slots[31].N = 64u;
    CHECK(g->experts[m0] == NULL && g->experts[m1] == NULL,
          "a refused table was stored");
    rc = (uint32_t)set_experts(g, m0, g_tiny_gu[0], g_tiny_dn[0], 3u);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "EXPERTS length: %s",
          htp_graph_err_name(rc));
    rc = (uint32_t)set_experts(g, nth_op(w, HTP_OP_RMSNORM, 0), g_tiny_gu[0],
                               g_tiny_dn[0], 4u);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "EXPERTS on a norm: %s",
          htp_graph_err_name(rc));
    rc = (uint32_t)set_experts(g, m0, g_tiny_gu[0], g_tiny_dn[0], 4u) |
         (uint32_t)set_experts(g, m1, g_tiny_gu[1], g_tiny_dn[1], 4u);
    CHECK(rc == 0u, "EXPERTS bind: %s", htp_graph_err_name(rc));
  }
  CHECK(hexkl_graph_uses_handle(g, 10u) && hexkl_graph_uses_handle(g, 53u) &&
          !hexkl_graph_uses_handle(g, 7u) && !hexkl_graph_uses_handle(g, 99u),
        "uses_handle");

  /* (3) bit identity against a direct call, both flag values, both layers */
  for (f = 0; f < 2u; ++f) {
    env.moe_flags = flag_cases[f];
    for (l = 0; l < 2u; ++l) {
      const uint32_t start = nth_op(w, HTP_OP_MOE, l);
      moe_args direct;
      uint32_t j;
      hexkl_mm_u8i4_moe_layer_run(&g_tbl, g_vtcm, sizeof(g_vtcm), 32u, 1u, 64u,
                                  64u, 64u, 4u, g_tiny_gu[l], g_tiny_dn[l],
                                  row_index, row_count, row_weight, act, ref,
                                  env.pool, &g_scratch, env.moe_flags);
      direct = g_last;
      memset(out, 0, sizeof(out));
      memset(&g_last, 0, sizeof(g_last));
      resume = 99u;
      rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 3u, &routing,
                                         act, 64u, out, 64u, &resume);
      CHECK(rc == 0u, "forward flags=%u layer %u: %s", env.moe_flags, l,
            htp_graph_err_name(rc));
      CHECK(resume == start + 1u, "resume %u (start %u)", resume, start);
      CHECK(memcmp(out, ref, sizeof(out)) == 0,
            "output differs (flags=%u l=%u)", env.moe_flags, l);
      CHECK(g_last.n_calls == 1u, "kernel calls %u", g_last.n_calls);
      /* the same arguments as the direct call, pointers included, except
         act/out, which are the graph's slots */
      CHECK(memcmp(&g_last, &direct, sizeof(direct)) == 0,
            "kernel arguments differ (flags=%u l=%u)", env.moe_flags, l);
      CHECK(g_last.flags == env.moe_flags, "flags passed %u", g_last.flags);
      for (j = 0; j < HTP_GRAPH_MAX_OPS; ++j)
        CHECK((g->op_pcycles[j] != 0u) == (j == start), "pcycles[%u]=%llu", j,
              (unsigned long long)g->op_pcycles[j]);
    }
  }
  printf("GRAPH FORWARD BIT-IDENTICAL: MoE op vs direct layer_run, flags 0x0 "
         "and 0x%x, layers 0-1, resume_at = start + 1\n",
         HEXKL_MOE_FLAG_M1_GEMV);

  /* (4) the forward entry's own refusals, none of them AEE_EBADPARM */
  {
    const uint32_t start = nth_op(w, HTP_OP_MOE, 0);
    hexkl_graph_routing bad = routing;
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 8u, &routing, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_BADITEM, "pos >= max_seq: %s",
          htp_graph_err_name(rc));
    rc = (uint32_t)hexkl_graph_forward(g, &env, 22u, 1000u, 0u, &routing, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_BADITEM, "start past the list: %s",
          htp_graph_err_name(rc));
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, &routing, act,
                                       63u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "act_in length: %s",
          htp_graph_err_name(rc));
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, &routing, act,
                                       64u, out, 32u, &resume);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "act_out length: %s",
          htp_graph_err_name(rc));
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, NULL, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_BADSTATE, "no routing: %s", htp_graph_err_name(rc));
    bad.n_experts = 3u;
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, &bad, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "routing n_experts: %s",
          htp_graph_err_name(rc));
    bad = routing;
    bad.n_rows = 3u;
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, &bad, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "routing row total: %s",
          htp_graph_err_name(rc));
    /* n_ops_limit 0: nothing runs, the identity again */
    memcpy(out, sentinel, sizeof(out));
    rc = (uint32_t)hexkl_graph_forward(g, &env, start, 0u, 0u, &routing, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == 0u && resume == start &&
            memcmp(out, sentinel, sizeof(out)) == 0,
          "n_ops_limit 0");
    /* a non-resident start op next to a resident one: identity */
    rc = (uint32_t)hexkl_graph_forward(g, &env, start - 1u, 1000u, 0u, &routing,
                                       act, 64u, out, 4u, &resume);
    CHECK(rc == 0u && resume == start - 1u, "router op is not resident");
    /* [plan 201 S1] a pool that does not hold experts 0 and 2 still runs
       a token routed to 1 and 3, with the full table's bits; one that
       does not hold 3 refuses it */
    {
      uint32_t gu[4], dn[4];
      memcpy(gu, g_tiny_gu[0], sizeof(gu));
      memcpy(dn, g_tiny_dn[0], sizeof(dn));
      gu[0] = dn[0] = gu[2] = dn[2] = HTP_GRAPH_NO_HANDLE;
      rc = (uint32_t)set_experts(g, start, gu, dn, 4u);
      CHECK(rc == 0u, "EXPERTS with holes: %s", htp_graph_err_name(rc));
      env.moe_flags = 0u;
      hexkl_mm_u8i4_moe_layer_run(&g_tbl, g_vtcm, sizeof(g_vtcm), 32u, 1u, 64u,
                                  64u, 64u, 4u, g_tiny_gu[0], g_tiny_dn[0],
                                  row_index, row_count, row_weight, act, ref,
                                  env.pool, &g_scratch, 0u);
      rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, &routing,
                                         act, 64u, out, 64u, &resume);
      CHECK(rc == 0u && memcmp(out, ref, sizeof(out)) == 0,
            "holes on unrouted experts: %s", htp_graph_err_name(rc));
      dn[3] = HTP_GRAPH_NO_HANDLE;
      rc = (uint32_t)set_experts(g, start, gu, dn, 4u);
      CHECK(rc == 0u, "EXPERTS with a routed hole: %s", htp_graph_err_name(rc));
      rc = (uint32_t)hexkl_graph_forward(g, &env, start, 1000u, 0u, &routing,
                                         act, 64u, out, 64u, &resume);
      CHECK(rc == HTP_GRAPH_E_BADSTATE, "routed to a non-resident expert: %s",
            htp_graph_err_name(rc));
      CHECK(!hexkl_graph_uses_handle(g, 10u) && hexkl_graph_uses_handle(g, 11u),
            "uses_handle follows the table");
    }
  }
  hexkl_graph_free(g);
  hexkl_graph_free(NULL);
  printf("GRAPH FORWARD REFUSALS OK\n");
}

/* ---- limits half (plan 201 S1): sized for Gemma-4-26B-A4B -------------- */
/* A miss round answered at once: every missing odd expert e loaded on the
   pair 3000 + e / 3200 + e, the order of events recorded. */
typedef struct {
  uint32_t posts, waits, kernel_calls_at_post;
  uint32_t miss[HEXKL_GRAPH_MISS_MAX], n_miss;
} fake_owner;
static fake_owner g_owner;
static int fake_post(void *ctx, uint32_t op, const uint32_t *routed,
                     uint32_t n_routed, const uint32_t *miss, uint32_t n_miss) {
  (void)ctx, (void)op, (void)routed, (void)n_routed;
  ++g_owner.posts;
  memcpy(g_owner.miss, miss, n_miss * sizeof(uint32_t));
  g_owner.n_miss = n_miss;
  return AEE_SUCCESS;
}
static int fake_wait(void *ctx, struct hexkl_graph_s *g, uint32_t op) {
  uint32_t i;
  int rc = AEE_SUCCESS;
  (void)ctx;
  ++g_owner.waits;
  for (i = 0; rc == AEE_SUCCESS && i < g_owner.n_miss; ++i)
    rc = hexkl_graph_pool_set(g, op, g_owner.miss[i], 3000u + g_owner.miss[i],
                              3200u + g_owner.miss[i]);
  return rc;
}

static void check_limits(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  /* 60 conv layers, 128 experts, top-8: 60 x 9 + 2 = 542 ops, over the
     256 before; the handles at the top of the 4096-entry table */
  static const htp_graph_lfm2_shape big = {60, 0, 64, 64, 64, 128,  8,
                                           8,  4, 8,  32, 8,  1e-6f};
  char layers[61];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  static const uint32_t routed[8] = {2u, 6u, 18u, 40u, 64u, 90u, 100u, 126u};
  uint32_t gu[128], dn[128], row_count[128], row_index[8], e, n, rc, m0, resume,
    seed = 201u;
  float row_weight[8], act[64], out[64], ref[64];
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  hexkl_graph_routing routing;

  memset(layers, 'C', 60);
  layers[60] = '\0';
  n = build(w, cap, &big, layers, HTP_GRAPH_KIND_BIT(HTP_OP_MOE));
  CHECK(n != 0u && w[3] == 542u, "60-layer build: %u words, %u ops", n, w[3]);
  rc = htp_graph_validate(w, n, HTP_GRAPH_KINDS_ALL, NULL);
  CHECK(rc == 0u, "542 ops x 128 experts: %s", htp_graph_err_name(rc));
  htp_graph_op_at(w, nth_op(w, HTP_OP_ROUTER_TOPK, 0))->resident = 1u;
  rc = htp_graph_validate(w, n, HTP_GRAPH_KINDS_ALL, NULL);
  CHECK(rc == HTP_GRAPH_E_SCHEMENOTSUPPORTED,
        "resident router over 128 experts: %s", htp_graph_err_name(rc));
  /* [plan 201 S4] the softmax router (eps set) takes 128 */
  htp_graph_op_at(w, nth_op(w, HTP_OP_ROUTER_TOPK, 0))->eps_bits = 0x358637BDu;
  rc = htp_graph_validate(w, n, HTP_GRAPH_KINDS_ALL, NULL);
  CHECK(rc == 0u, "resident softmax router over 128 experts: %s",
        htp_graph_err_name(rc));
  htp_graph_op_at(w, nth_op(w, HTP_OP_ROUTER_TOPK, 0))->eps_bits = 0u;
  htp_graph_op_at(w, nth_op(w, HTP_OP_ROUTER_TOPK, 0))->resident = 0u;

  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "init 542 ops: %s", htp_graph_err_name(rc));
  if (g == NULL)
    return;
  m0 = nth_op(w, HTP_OP_MOE, 0);
  /* the pool holds the even experts only */
  for (e = 0; e < 128u; ++e) {
    gu[e] = (e & 1u) ? HTP_GRAPH_NO_HANDLE : 3969u + e;
    dn[e] = (e & 1u) ? HTP_GRAPH_NO_HANDLE : 3840u + e;
    if (!(e & 1u)) {
      register_weight(gu[e], 64u, 128u);
      register_weight(dn[e], 64u, 64u);
    }
    row_count[e] = 0u;
  }
  rc = (uint32_t)set_experts(g, m0, gu, dn, 128u);
  CHECK(rc == 0u && gu[126] == HEXKL_MM_U8I4_MAX_WEIGHTS - 1u,
        "EXPERTS 128 up to the last handle: %s", htp_graph_err_name(rc));
  for (e = 0; e < 8u; ++e) {
    row_count[routed[e]] = 1u;
    row_index[e] = 0u;
    row_weight[e] = 0.125f * (float)(e + 1u);
  }
  for (e = 0; e < 64u; ++e)
    act[e] = frand(&seed);
  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  env.vtcm_base = g_vtcm;
  env.vtcm_size = sizeof(g_vtcm);
  env.config_off = 32u;
  env.pool = real_pool();
  env.scratch = &g_scratch;
  hexkl_mm_u8i4_moe_layer_run(&g_tbl, g_vtcm, sizeof(g_vtcm), 32u, 1u, 64u, 64u,
                              64u, 128u, gu, dn, row_index, row_count,
                              row_weight, act, ref, env.pool, &g_scratch, 0u);
  routing.row_index = row_index;
  routing.row_count = row_count;
  routing.row_weight = row_weight;
  routing.n_rows = 8u;
  routing.n_experts = 128u;
  rc = (uint32_t)hexkl_graph_forward(g, &env, m0, 1000u, 0u, &routing, act, 64u,
                                     out, 64u, &resume);
  CHECK(rc == 0u && memcmp(out, ref, sizeof(out)) == 0 &&
          g_last.n_experts == 128u,
        "128 experts, top-8, half the pool: %s", htp_graph_err_name(rc));
  row_count[routed[7]] = 0u;
  row_count[127] = 1u; /* odd: not in the pool */
  routing.n_experts = 128u;
  rc = (uint32_t)hexkl_graph_forward(g, &env, m0, 1000u, 0u, &routing, act, 64u,
                                     out, 64u, &resume);
  CHECK(rc == HTP_GRAPH_E_BADSTATE, "routed outside the pool: %s",
        htp_graph_err_name(rc));
  /* the miss round (plan 201 S1): top-8 over experts 2, 5, 6, 9, 40, 63,
     64, 126 -- misses 5, 9, 63 after a hit, in the middle and late -- equals
     the direct call on the loaded table bit for bit */
  {
    static const uint32_t rt[8] = {2u, 5u, 6u, 9u, 40u, 63u, 64u, 126u};
    uint32_t full_gu[128], full_dn[128], k;
    for (e = 0; e < 128u; ++e) {
      row_count[e] = 0u;
      if (e & 1u) {
        register_weight(3000u + e, 64u, 128u);
        register_weight(3200u + e, 64u, 64u);
      }
      full_gu[e] = (e & 1u) ? 3000u + e : gu[e];
      full_dn[e] = (e & 1u) ? 3200u + e : dn[e];
    }
    for (k = 0; k < 8u; ++k)
      row_count[rt[k]] = 1u;
    hexkl_mm_u8i4_moe_layer_run(&g_tbl, g_vtcm, sizeof(g_vtcm), 32u, 1u, 64u,
                                64u, 64u, 128u, full_gu, full_dn, row_index,
                                row_count, row_weight, act, ref, env.pool,
                                &g_scratch, 0u);
    env.miss.post = fake_post;
    env.miss.wait = fake_wait;
    memset(&g_owner, 0, sizeof(g_owner));
    routing.n_experts = 128u;
    rc = (uint32_t)hexkl_graph_forward(g, &env, m0, 1000u, 0u, &routing, act,
                                       64u, out, 64u, &resume);
    CHECK(rc == 0u && memcmp(out, ref, sizeof(out)) == 0 &&
            g_owner.posts == 1u && g_owner.waits == 1u && g_owner.n_miss == 3u,
          "miss round: %s posts %u waits %u misses %u", htp_graph_err_name(rc),
          g_owner.posts, g_owner.waits, g_owner.n_miss);
    CHECK(g->experts[m0][5] == 3005u && g->experts[m0][128u + 63u] == 3263u,
          "the answer did not reach the table");
    CHECK(g->route_log_n >= 9u && g->route_log[g->route_log_n - 9u] == 8u &&
            g->route_log[g->route_log_n - 1u] == 126u,
          "route log");
    env.miss.post = NULL;
    env.miss.wait = NULL;
  }
  hexkl_graph_free(g);
  printf("GRAPH LIMITS OK: 542 ops, 128 experts top-8, handles up to %u, a "
         "pool of the even experts bit-identical to the direct call; a miss "
         "round (3 of 8 routed missing) bit-identical to the loaded table\n",
         gu[126]);
}

/* ---- stretch half (#130): the real kernels vs the scalar specs --------- */
#define HD 64u
#define N_Q 2u
#define N_KV 1u
#define HID 128u
#define MAX_SEQ 64u

static void fill(float *p, uint32_t n, uint32_t *seed) {
  uint32_t i;
  for (i = 0; i < n; ++i)
    p[i] = frand(seed);
}

/** @brief Every op run has a pcycle bracket, no other op does. */
static void check_pcycles(const hexkl_graph *g, uint32_t s, uint32_t e,
                          const char *what) {
  uint32_t j;
  for (j = 0; j < HTP_GRAPH_MAX_OPS; ++j)
    CHECK((g->op_pcycles[j] != 0u) == (j >= s && j < e), "%s pcycles[%u]", what,
          j);
}

static void check_stretches(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  const uint32_t mask = ALL_KINDS & ~HTP_GRAPH_KIND_BIT(HTP_OP_MOE);
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  uint32_t n, rc, resume, seed = 777u, t, i, p;
  static float gamma[HID], in[3u * HID], out[HID], ref[HID];
  static float conv_w[3u * HID], state[2u * HID], state_ref[2u * HID];
  static float qk_gamma[2u * HD], cs[MAX_SEQ * HD], qn[N_Q * HD], kn[HD];
  static float k_rows[MAX_SEQ * N_KV * HD], v_rows[MAX_SEQ * N_KV * HD];
  static float kt[N_KV * HD * MAX_SEQ], vv[N_KV * MAX_SEQ * HD], e[MAX_SEQ];
  const uint32_t positions[4] = {0u, 31u, 32u, 63u};
  uint32_t op_rms, op_conv, op_qk;
  int err = 0;

  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  n = build(w, cap, &kHd64, "CAC", mask);
  CHECK(n != 0u, "hd64 build");
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "hd64 init: %s", htp_graph_err_name(rc));
  if (g == NULL)
    return;
  CHECK(g->n_ops == 30u && g->slot_words == 3u * HID, "hd64 n_ops %u slots %u",
        g->n_ops, g->slot_words);
  op_rms = nth_op(w, HTP_OP_RMSNORM, 0);
  op_conv = nth_op(w, HTP_OP_CONV1D_GATE, 0);
  op_qk = nth_op(w, HTP_OP_QK_NORM, 0);
  CHECK(op_rms == 0u && op_conv == 2u && op_qk == 10u &&
          g->ops[op_qk + 1u].kind == HTP_OP_ROPE &&
          g->ops[op_qk + 2u].kind == HTP_OP_ATTN_M1 &&
          g->ordinal[op_qk + 2u] == 0u,
        "hd64 op indices %u %u %u", op_rms, op_conv, op_qk);

  /* (1) [RMSNORM]: no gamma -> EBADSTATE; then bit-identical to the spec */
  fill(in, HID, &seed);
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_rms, 1000u, 0u, NULL, in, HID,
                                     out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "RMSNORM without gamma: %s",
        htp_graph_err_name(rc));
  printf("  RMSNORM forward with no gamma bound       -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  fill(gamma, HID, &seed);
  rc = (uint32_t)hexkl_graph_set_param(g, op_rms, HTP_GRAPH_PARAM_GAMMA, gamma,
                                       HID - 1u);
  CHECK(rc == (uint32_t)AEE_EINVALIDFORMAT, "gamma length: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(g, op_conv, HTP_GRAPH_PARAM_GAMMA, gamma,
                                       HID);
  CHECK(rc == (uint32_t)AEE_EINVALIDFORMAT, "gamma on a conv op: %s",
        htp_graph_err_name(rc));
  rc =
    (uint32_t)hexkl_graph_set_param(g, 99u, HTP_GRAPH_PARAM_GAMMA, gamma, HID);
  CHECK(rc == (uint32_t)AEE_EBADITEM, "gamma op out of range: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(g, op_rms, HTP_GRAPH_PARAM_GAMMA, gamma,
                                       HID);
  CHECK(rc == 0u, "set gamma: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_rms, 1000u, 0u, NULL, in, HID,
                                     out, HID, &resume);
  CHECK(rc == 0u && resume == op_rms + 1u, "RMSNORM forward: %s resume %u",
        htp_graph_err_name(rc), resume);
  m1_rmsnorm_det(in, gamma, ref, HID, HID, kHd64.eps, NULL);
  CHECK(memcmp(out, ref, HID * sizeof(float)) == 0, "RMSNORM differs");
  check_pcycles(g, op_rms, op_rms + 1u, "RMSNORM");
  err |= memcmp(out, ref, HID * sizeof(float)) != 0;

  /* (2) [CONV1D_GATE]: no conv_w / state -> EBADSTATE; an 8-token chain
     from a seeded state; the state re-sent mid-chain */
  fill(in, 3u * HID, &seed);
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_conv, 1000u, 0u, NULL, in,
                                     3u * HID, out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "CONV1D_GATE without conv_w: %s",
        htp_graph_err_name(rc));
  printf("  CONV1D_GATE forward with no conv_w bound  -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  fill(conv_w, 3u * HID, &seed);
  fill(state, 2u * HID, &seed);
  memcpy(state_ref, state, sizeof(state_ref));
  rc = (uint32_t)hexkl_graph_set_param(g, op_conv, HTP_GRAPH_PARAM_CONV_W,
                                       conv_w, 3u * HID);
  CHECK(rc == 0u, "set conv_w: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_conv, 1000u, 0u, NULL, in,
                                     3u * HID, out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "CONV1D_GATE without state: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(g, op_conv, HTP_GRAPH_PARAM_CONV_STATE,
                                       state, 2u * HID);
  CHECK(rc == 0u, "set conv state: %s", htp_graph_err_name(rc));
  for (t = 0; t < 12u; ++t) {
    if (t == 8u) {
      /* re-seed: the DSP state must follow the sent one, not its own */
      fill(state, 2u * HID, &seed);
      memcpy(state_ref, state, sizeof(state_ref));
      rc = (uint32_t)hexkl_graph_set_param(
        g, op_conv, HTP_GRAPH_PARAM_CONV_STATE, state, 2u * HID);
      CHECK(rc == 0u, "re-send conv state: %s", htp_graph_err_name(rc));
    }
    fill(in, 3u * HID, &seed);
    rc = (uint32_t)hexkl_graph_forward(g, &env, op_conv, 1000u, t, NULL, in,
                                       3u * HID, out, HID, &resume);
    CHECK(rc == 0u && resume == op_conv + 1u, "conv t=%u: %s resume %u", t,
          htp_graph_err_name(rc), resume);
    m1_conv_gate_det(in, state_ref, conv_w, ref, HID);
    CHECK(memcmp(out, ref, HID * sizeof(float)) == 0, "conv t=%u differs", t);
    err |= memcmp(out, ref, HID * sizeof(float)) != 0;
  }
  check_pcycles(g, op_conv, op_conv + 1u, "CONV1D_GATE");
  CHECK(memcmp(g->state[op_conv], state_ref, sizeof(state_ref)) == 0,
        "conv state after the chain");

  /* (3) [QK_NORM ROPE ATTN_M1]: no gamma / table / cache -> EBADSTATE;
     then, at four positions after a kv_append seed, bit-identical to
     rmsnorm_det -> rope64_det -> attn_m1_det */
  fill(in, (N_Q + 2u * N_KV) * HD, &seed);
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_qk, 1000u, 0u, NULL, in,
                                     (N_Q + 2u * N_KV) * HD, out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "QK_NORM without gamma: %s",
        htp_graph_err_name(rc));
  fill(qk_gamma, 2u * HD, &seed);
  rc = (uint32_t)hexkl_graph_set_param(g, op_qk, HTP_GRAPH_PARAM_GAMMA,
                                       qk_gamma, 2u * HD);
  CHECK(rc == 0u, "set qk gamma: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_qk, 1000u, 0u, NULL, in,
                                     (N_Q + 2u * N_KV) * HD, out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "ROPE without table: %s",
        htp_graph_err_name(rc));
  printf("  ROPE forward with no table bound          -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  fill(cs, MAX_SEQ * HD, &seed);
  rc = (uint32_t)hexkl_graph_set_param(g, op_qk, HTP_GRAPH_PARAM_ROPE_TABLE, cs,
                                       MAX_SEQ * HD);
  /* [plan 201 S4] a ROPE op may hold its own table; any other kind takes
     none, the length rule every other parameter has */
  CHECK(rc == (uint32_t)AEE_EINVALIDFORMAT, "rope table on a QK_NORM op: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(
    g, HTP_GRAPH_NO_OP, HTP_GRAPH_PARAM_ROPE_TABLE, cs, MAX_SEQ * HD);
  CHECK(rc == 0u, "set rope table: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_qk, 1000u, 0u, NULL, in,
                                     (N_Q + 2u * N_KV) * HD, out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "ATTN_M1 without cache: %s",
        htp_graph_err_name(rc));
  printf("  ATTN_M1 forward with no cache registered  -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  fill(k_rows, MAX_SEQ * N_KV * HD, &seed);
  fill(v_rows, MAX_SEQ * N_KV * HD, &seed);
  for (p = 0; p < 4u; ++p) {
    const uint32_t pos = positions[p];
    int cerr = 0;
    env.attn_m1 =
      hvx_attn_m1_create(1u, N_KV, N_Q / N_KV, HD, MAX_SEQ, NULL, &cerr);
    CHECK(env.attn_m1 != NULL, "attn_m1_create: %d", cerr);
    if (env.attn_m1 == NULL)
      break;
    memset(kt, 0, sizeof(kt));
    memset(vv, 0, sizeof(vv));
    if (pos != 0u) {
      rc = (uint32_t)hvx_attn_m1_kv_append(env.attn_m1, 0u, 0u, pos, k_rows,
                                           v_rows);
      CHECK(rc == 0u, "kv_append %u rows: %s", pos, htp_graph_err_name(rc));
      for (i = 0; i < pos; ++i)
        attn_m1_det_append(kt, vv, HD, MAX_SEQ, i, k_rows + (size_t)i * HD,
                           v_rows + (size_t)i * HD);
    }
    fill(in, (N_Q + 2u * N_KV) * HD, &seed);
    memset(out, 0, sizeof(out));
    rc =
      (uint32_t)hexkl_graph_forward(g, &env, op_qk, 1000u, pos, NULL, in,
                                    (N_Q + 2u * N_KV) * HD, out, HID, &resume);
    CHECK(rc == 0u && resume == op_qk + 3u, "attn pos %u: %s resume %u", pos,
          htp_graph_err_name(rc), resume);
    /* the spec: per-head norm, RoPE on q then k heads, append, attend */
    m1_rmsnorm_det(in, qk_gamma, qn, N_Q * HD, HD, kHd64.eps, NULL);
    m1_rmsnorm_det(in + N_Q * HD, qk_gamma + HD, kn, HD, HD, kHd64.eps, NULL);
    for (i = 0; i < N_Q; ++i)
      m1_rope64_det(qn + i * HD, cs + (size_t)pos * HD);
    m1_rope64_det(kn, cs + (size_t)pos * HD);
    attn_m1_det_append(kt, vv, HD, MAX_SEQ, pos, kn, in + (N_Q + 1u) * HD);
    attn_m1_det_forward(qn, kt, vv, N_KV, N_Q / N_KV, HD, MAX_SEQ, pos + 1u,
                        0.125f, e, ref, NULL);
    CHECK(memcmp(out, ref, HID * sizeof(float)) == 0, "attn pos %u differs",
          pos);
    err |= memcmp(out, ref, HID * sizeof(float)) != 0;
    /* slot routing: QK_NORM and ROPE ran in place in slot 2 (the roped
       q | k and the untouched v), ATTN_M1 wrote slot 1 */
    CHECK(memcmp(g->slots + 2u * g->slot_words, qn, N_Q * HD * sizeof(float)) ==
              0 &&
            memcmp(g->slots + 2u * g->slot_words + N_Q * HD, kn,
                   HD * sizeof(float)) == 0 &&
            memcmp(g->slots + 2u * g->slot_words + (N_Q + 1u) * HD,
                   in + (N_Q + 1u) * HD, HD * sizeof(float)) == 0 &&
            memcmp(g->slots + 1u * g->slot_words, out, HID * sizeof(float)) ==
              0,
          "slot routing at pos %u", pos);
    check_pcycles(g, op_qk, op_qk + 3u, "attention stretch");
    if (p == 3u) {
      /* the cache's own hole check passes through: pos past the length */
      rc = (uint32_t)hexkl_graph_forward(g, &env, op_qk, 1000u, 10u, NULL, in,
                                         (N_Q + 2u * N_KV) * HD, out, HID,
                                         &resume);
      CHECK(rc == 0u, "rewind to pos 10 (kernel allows): %s",
            htp_graph_err_name(rc));
      rc = (uint32_t)hexkl_graph_forward(g, &env, op_qk, 1000u, 20u, NULL, in,
                                         (N_Q + 2u * N_KV) * HD, out, HID,
                                         &resume);
      CHECK(rc == (uint32_t)AEE_EBADSTATE, "hole at pos 20: %s",
            htp_graph_err_name(rc));
      printf("  ATTN_M1 forward at a hole (pos > kv_len)  -> %s (0x%x)\n",
             htp_graph_err_name(rc), rc);
    }
    hvx_attn_m1_free(env.attn_m1);
    env.attn_m1 = NULL;
  }
  hexkl_graph_free(g);
  if (err == 0)
    printf("GRAPH STRETCH BIT-IDENTICAL: RMSNORM CONV1D_GATE "
           "QK_NORM+ROPE+ATTN_M1 (hd64 shape, pos 0/31/32/63, conv chain "
           "of 12 with a re-seed)\n");
}

/* ---- #132: [ADD RMSNORM] and [ADD RMSNORM ROUTER_TOPK MOE ADD RMSNORM] -- */
#define HD64_E 4u
#define HD64_TOP 2u
#define HD64_INTER 32u

/* hd64's two MoE layers on the stand-in: gate_up 200 + 10 m + e, down
   250 + 10 m + e (the tiny fixture's handles stay 10..73). */
/* MoE op m's expert e: gate_up 200 + 10 m + e, down 250 + 10 m + e */
static uint32_t g_hd64_gu[2][HD64_E], g_hd64_dn[2][HD64_E];
static void bind_hd64(uint32_t *w) {
  uint32_t m, e;
  (void)w;
  for (m = 0; m < 2u; ++m) {
    for (e = 0; e < HD64_E; ++e) {
      g_hd64_gu[m][e] = 200u + 10u * m + e;
      g_hd64_dn[m][e] = 250u + 10u * m + e;
      register_weight(g_hd64_gu[m][e], HID, 2u * HD64_INTER);
      register_weight(g_hd64_dn[m][e], HD64_INTER, HID);
    }
  }
}

/* [plan 201 S4] gemma: the same stretch with the router op made Gemma's
   softmax router (eps_bits set, ROUTER_BIAS = g | per-expert scale) and
   the norm before it on N1 (feed HTP_GRAPH_NORM_N1). */
static void check_add_router(int gemma) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  uint32_t n, rc, resume, seed = 132u, i, e, r, nr = 0;
  static float x[HID], a[HID], a2[HID], out[HID], ref[HID], h[HID], nrm[HID];
  static float gam[4][HID], rw[HID * HD64_E], moe[HID];
  static float rbias[HID + HD64_E], rsc[HID];
  float lg[HD64_E], wt[HD64_TOP];
  const uint32_t n_bias = gemma ? HID + HD64_E : HD64_E;
  uint32_t sel[HD64_TOP], r_idx[HD64_TOP], r_cnt[HD64_E] = {0};
  float r_w[HD64_TOP], by_e[HD64_E];
  uint32_t op_norm[4], op_add0, op_add1, op_router;
  int err = 0;

  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  env.vtcm_base = g_vtcm;
  env.vtcm_size = sizeof(g_vtcm);
  env.config_off = 32u;
  env.pool = real_pool(); /* ROUTER_TOPK hands it its chains (#132 PR 2;
                            one group at E = 4: m1_ops_host_check runs
                            the multi-lane split) */
  env.scratch = &g_scratch;
  n = build(w, cap, &kHd64, "CAC", D_KINDS);
  bind_hd64(w);
  if (gemma) {
    memcpy(&htp_graph_op_at(w, nth_op(w, HTP_OP_ROUTER_TOPK, 0))->eps_bits,
           &kHd64.eps, sizeof(float));
    htp_graph_op_at(w, nth_op(w, HTP_OP_RMSNORM, 3))->feed = HTP_GRAPH_NORM_N1;
  }
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "hd64 D init: %s", htp_graph_err_name(rc));
  if (g == NULL)
    return;
  for (i = 0; i < 2u; ++i) {
    rc = (uint32_t)set_experts(g, nth_op(w, HTP_OP_MOE, i), g_hd64_gu[i],
                               g_hd64_dn[i], HD64_E);
    CHECK(rc == 0u, "hd64 EXPERTS %u: %s", i, htp_graph_err_name(rc));
  }
  /* layer 0's norms (0, 5), layer 1's ffn norm (15), layer 2's operator
     norm (19); layer 0's ffn-side ADD (4) and layer 1's first ADD (14) */
  op_norm[0] = nth_op(w, HTP_OP_RMSNORM, 0);
  op_norm[1] = nth_op(w, HTP_OP_RMSNORM, 1);
  op_norm[2] = nth_op(w, HTP_OP_RMSNORM, 3);
  op_norm[3] = nth_op(w, HTP_OP_RMSNORM, 4);
  op_add0 = nth_op(w, HTP_OP_ADD, 0);
  op_add1 = nth_op(w, HTP_OP_ADD, 2);
  op_router = nth_op(w, HTP_OP_ROUTER_TOPK, 0);
  CHECK(op_norm[1] == op_add0 + 1u && op_add1 == op_router - 2u &&
          op_norm[2] == op_router - 1u && op_norm[3] == op_router + 3u &&
          g->ops[op_router + 1u].kind == HTP_OP_MOE &&
          g->ops[op_router + 2u].kind == HTP_OP_ADD &&
          g->ops[op_norm[3] + 1u].kind == HTP_OP_FC &&
          g->ops[op_norm[1] + 1u].kind == HTP_OP_DENSE_FFN,
        "hd64 D op indices");
  for (i = 0; i < 4u; ++i) {
    fill(gam[i], HID, &seed);
    rc = (uint32_t)hexkl_graph_set_param(g, op_norm[i], HTP_GRAPH_PARAM_GAMMA,
                                         gam[i], HID);
    CHECK(rc == 0u, "gamma %u: %s", i, htp_graph_err_name(rc));
  }
  fill(x, HID, &seed);
  fill(a, HID, &seed);
  fill(a2, HID, &seed);
  fill(rw, HID * HD64_E, &seed);
  fill(rbias, HD64_E, &seed);
  if (gemma) { /* g from a router_scale, then the per-expert scale */
    fill(rsc, HID, &seed);
    m1_router_input_scale_det(rsc, HID, rbias);
    fill(rbias + HID, HD64_E, &seed);
  }

  /* (1) a router with no weights bound: AEE_EBADSTATE (ADD and the norm
     ran first and moved slot 0; op 0 below re-seeds it) */
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_add1, 1000u, 0u, NULL, a, HID,
                                     out, HID, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "ROUTER_TOPK without weights: %s",
        htp_graph_err_name(rc));
  printf("  ROUTER_TOPK forward with no weights bound -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  rc = (uint32_t)hexkl_graph_set_param(g, op_router, HTP_GRAPH_PARAM_ROUTER_W,
                                       rw, HID * HD64_E - 1u);
  CHECK(rc == (uint32_t)AEE_EINVALIDFORMAT, "router W length: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(
    g, op_norm[0], HTP_GRAPH_PARAM_ROUTER_BIAS, rbias, HD64_E);
  CHECK(rc == (uint32_t)AEE_EINVALIDFORMAT, "router bias on a norm: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(g, op_router, HTP_GRAPH_PARAM_ROUTER_W,
                                       rw, HID * HD64_E);
  CHECK(rc == 0u, "set router W: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(
    g, op_router, HTP_GRAPH_PARAM_ROUTER_BIAS, rbias, n_bias);
  CHECK(rc == 0u, "set router bias: %s", htp_graph_err_name(rc));

  /* (2) op 0 seeds slot 0 with the embedding row x */
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_norm[0], 1000u, 0u, NULL, x,
                                     HID, out, HID, &resume);
  CHECK(rc == 0u && resume == op_norm[0] + 1u, "op 0: %s resume %u",
        htp_graph_err_name(rc), resume);

  /* (3) [ADD RMSNORM]: out = rmsnorm(x + a), slot 0 = x + a */
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_add0, 1000u, 0u, NULL, a, HID,
                                     out, HID, &resume);
  CHECK(rc == 0u && resume == op_norm[1] + 1u, "[ADD RMSNORM]: %s resume %u",
        htp_graph_err_name(rc), resume);
  for (i = 0; i < HID; ++i)
    h[i] = m1_det_add(x[i], a[i]);
  m1_rmsnorm_det(h, gam[1], ref, HID, HID, kHd64.eps, NULL);
  CHECK(memcmp(out, ref, sizeof(ref)) == 0, "[ADD RMSNORM] differs");
  CHECK(memcmp(g->slots, h, sizeof(h)) == 0, "slot 0 != x + a");
  check_pcycles(g, op_add0, op_norm[1] + 1u, "[ADD RMSNORM]");
  err |= memcmp(out, ref, sizeof(ref)) != 0 || memcmp(g->slots, h, sizeof(h));

  if (gemma) {
    /* an a2 whose row N1 norms to another r than the f32 norm does, so the
       stretch tells the two apart (they agree on many rows) */
    static float t1[HID], t2[HID];
    int differs = 0;
    for (r = 0; r < 64u && !differs; ++r) {
      fill(a2, HID, &seed);
      for (i = 0; i < HID; ++i)
        t1[i] = m1_det_add(h[i], a2[i]);
      differs = m1_rmsnorm_n1_chunk_det(t1, gam[2], t2, HID, kHd64.eps) !=
                m1_rmsnorm_chunk_det(t1, gam[2], t2, HID, kHd64.eps);
    }
    CHECK(differs, "no a2 in 64 rows tells N1 from the f32 norm");
  }

  /* (4) [ADD RMSNORM ROUTER_TOPK MOE ADD RMSNORM], slot 0 carried over:
     h = s0 + a2, n = rmsnorm(h), routing = router_topk_det(n),
     h2 = h + standin(n, routing), out = rmsnorm(h2) */
  memset(&g_last, 0, sizeof(g_last));
  rc = (uint32_t)hexkl_graph_forward(g, &env, op_add1, 1000u, 0u, NULL, a2, HID,
                                     out, HID, &resume);
  CHECK(rc == 0u && resume == op_norm[3] + 1u, "[ADD .. RMSNORM]: %s resume %u",
        htp_graph_err_name(rc), resume);
  for (i = 0; i < HID; ++i)
    h[i] = m1_det_add(h[i], a2[i]);
  if (gemma) {
    static float xs[HID];
    m1_rmsnorm_n1_chunk_det(h, gam[2], nrm, HID, kHd64.eps);
    m1_rmsnorm_det(nrm, rbias, xs, HID, HID, kHd64.eps, NULL);
    m1_router_softmax_det(xs, rw, rbias + HID, HID, HD64_E, HD64_TOP, lg, sel,
                          wt);
  } else {
    m1_rmsnorm_det(h, gam[2], nrm, HID, HID, kHd64.eps, NULL);
    m1_router_cpu_det(nrm, rw, rbias, HID, HD64_E, HD64_TOP, lg, sel, wt);
  }
  for (r = 0; r < HD64_TOP; ++r) {
    r_cnt[sel[r]] = 1u;
    by_e[sel[r]] = wt[r];
  }
  for (e = 0; e < HD64_E; ++e) {
    if (r_cnt[e]) {
      r_idx[nr] = 0u;
      r_w[nr++] = by_e[e];
    }
  }
  CHECK(g_last.n_calls == 1u &&
          memcmp(g_last.row_count, r_cnt, sizeof(r_cnt)) == 0,
        "the MOE op did not get the router's routing");
  hexkl_mm_u8i4_moe_layer_run(&g_tbl, g_vtcm, sizeof(g_vtcm), 32u, 1u, HID,
                              HD64_INTER, HID, HD64_E, g_hd64_gu[0],
                              g_hd64_dn[0], r_idx, r_cnt, r_w, nrm, moe,
                              env.pool, &g_scratch, 0u);
  for (i = 0; i < HID; ++i)
    h[i] = m1_det_add(h[i], moe[i]);
  m1_rmsnorm_det(h, gam[3], ref, HID, HID, kHd64.eps, NULL);
  CHECK(memcmp(out, ref, sizeof(ref)) == 0,
        "[ADD RMSNORM ROUTER_TOPK MOE ADD RMSNORM] differs");
  CHECK(memcmp(g->slots, h, sizeof(h)) == 0, "slot 0 != h2");
  check_pcycles(g, op_add1, op_norm[3] + 1u, "[ADD .. RMSNORM]");
  err |= memcmp(out, ref, sizeof(ref)) != 0 || memcmp(g->slots, h, sizeof(h));
  hexkl_graph_free(g);
  if (err == 0)
    printf("GRAPH STRETCH BIT-IDENTICAL%s: ADD+RMSNORM "
           "ADD+RMSNORM+ROUTER_TOPK+MOE+ADD+RMSNORM (hd64 shape, MOE on the "
           "stand-in, slot 0 carried across calls, resume_at the next FC)\n",
           gemma ? " (Gemma softmax router, N1 norm before it)" : "");
}

/* ---- #132 Part B: the Q4M1 kinds against the CPU-order specs ---------- */
#define Q_SLOTS 16u
static uint8_t *g_qw[Q_SLOTS]; /* Q4M1 bytes the host runner reads */
static uint8_t *g_qc[Q_SLOTS]; /* the same weight, canonical block_q4_0 */
static hexkl_graph_q4m1_shape g_qs[Q_SLOTS];
static uint32_t g_fc_calls, g_fc_feed;

/* The session runner's host twin: the real GEMV on the Q4M1 bytes, no
   feed (the DMA feed moves no bits; the in-process E2E runs the real
   runner on its DMA stand-in). */
static int host_fc(void *ctx, uint32_t h, uint32_t feed, const hvx_q4m1_act *a,
                   float *y) {
  (void)ctx;
  ++g_fc_calls;
  g_fc_feed = feed;
  if (h >= Q_SLOTS || g_qw[h] == NULL)
    return AEE_EBADITEM;
  ((feed & HTP_GRAPH_FEED_NATIVE) != 0u
     ? hvx_q4m1_gemv_groups_native
     : hvx_q4m1_gemv_groups)(g_qw[h], g_qs[h].K, g_qs[h].N / Q4M1_GROUP, a, y);
  return AEE_SUCCESS;
}

static void q_block_rand(uint8_t *blk, uint32_t *seed) {
  uint32_t j;
  const uint16_t d =
    (uint16_t)(0x2000u |
               (((*seed = *seed * 1664525u + 1013904223u) >> 8) & 0x7ffu));
  blk[0] = (uint8_t)(d & 0xffu);
  blk[1] = (uint8_t)(d >> 8);
  for (j = 0; j < 16u; ++j) {
    *seed = *seed * 1664525u + 1013904223u;
    blk[2 + j] = (uint8_t)(*seed >> 24);
  }
}

static void q_register(uint32_t h, uint32_t K, uint32_t N, uint32_t *seed) {
  const size_t bytes = (size_t)N * (K / 32u) * Q4_CPU_BLOCK_BYTES;
  size_t b;
  g_qc[h] = realloc(g_qc[h], bytes);
  free(g_qw[h]);
  g_qw[h] = aligned_alloc(128, (q4m1_bytes(K, N) + 127u) & ~(size_t)127u);
  for (b = 0; b < bytes / Q4_CPU_BLOCK_BYTES; ++b)
    q_block_rand(g_qc[h] + b * Q4_CPU_BLOCK_BYTES, seed);
  q4m1_from_q4_0(g_qc[h], K, N, g_qw[h]);
  g_qs[h].K = K;
  g_qs[h].N = N;
}

/* the CPU's FC of x: the Q8_0 quantizer, then the fused chain; with
   g_native [#194 L1] q4_gemv_native_det.h's pair instead */
static int g_native;
static void spec_fc(uint32_t h, const float *x, float *y) {
  static int8_t q[8192];
  static uint16_t d[256];
  if (g_native) {
    q8_0_quant_native_det(x, g_qs[h].K, q, d);
    q4_gemv_native_det(g_qc[h], q, d, g_qs[h].K, g_qs[h].N, y);
    return;
  }
  q8_0_quant_cpu_det(x, g_qs[h].K, q, d);
  q4_gemv_cpu_det(g_qc[h], q, d, g_qs[h].K, g_qs[h].N, y);
}

static void check_q4m1(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  htp_graph_lfm2_shape shape = kHd64;
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  uint32_t n, rc, resume, seed = 0x1320u, i, fc_qkv, ffn, lm, fin, best, r2;
  static float x[HID], out[3u * HID], ref[3u * HID], up[64], gate[64], act[64],
    gam[HID], nrm[HID];
  htp_graph_op *op;
  int err = 0;

  shape.vocab = 64u; /* two lm_head slices of 32 rows */
  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  env.vtcm_base = g_vtcm;
  env.vtcm_size = sizeof(g_vtcm);
  env.config_off = 32u;
  env.pool = real_pool();
  env.scratch = &g_scratch;
  n = build(w, cap, &shape, "CAC", HTP_GRAPH_KINDS_ALL);
  CHECK(n != 0u, "hd64 all-kinds build");
  bind_hd64(w);
  fc_qkv = nth_op(w, HTP_OP_FC, 2); /* layer 0: in, out; layer 1: qkv */
  ffn = nth_op(w, HTP_OP_DENSE_FFN, 0);
  lm = nth_op(w, HTP_OP_LM_HEAD, 0);
  fin = lm - 1u;
  CHECK(htp_graph_op_cat(w, fc_qkv)->N == 256u && lm == w[3] - 1u &&
          htp_graph_op_cat(w, fin)->kind == HTP_OP_RMSNORM,
        "hd64 all-kinds op indices");

  /* (1) init: a resident FC op with no parts is refused */
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE && g == NULL, "unbound FC: %s",
        htp_graph_err_name(rc));
  /* the FC parts: every FC op one part except qkv (q 128 | k 64 | v 64) */
  {
    uint32_t h = 0, f = 0, k;
    for (i = 0; i < w[3]; ++i) {
      op = htp_graph_op_at(w, i);
      if (op->kind == HTP_OP_FC && i != fc_qkv) {
        q_register(h, op->K, op->N, &seed);
        op->h_gu[0] = h++;
        op->n_experts = 1u;
        ++f;
      }
    }
    op = htp_graph_op_at(w, fc_qkv);
    for (k = 0; k < 3u; ++k) {
      q_register(h, HID, k == 0u ? 128u : 64u, &seed);
      op->h_gu[k] = h++;
    }
    op->n_experts = 3u;
    op = htp_graph_op_at(w, ffn); /* up, gate, down */
    q_register(h, HID, 64u, &seed);
    op->h_gu[0] = h++;
    q_register(h, HID, 64u, &seed);
    op->h_gu[1] = h++;
    q_register(h, 64u, HID, &seed);
    op->h_dn[0] = h++;
    op->n_experts = 3u;
    op = htp_graph_op_at(w, lm);
    q_register(h, HID, 32u, &seed);
    op->h_gu[0] = h++;
    q_register(h, HID, 32u, &seed);
    op->h_gu[1] = h++;
    op->n_experts = 2u;
    CHECK(f == 5u && h <= Q_SLOTS, "hd64 FC count %u, slots %u", f, h);
  }
  /* (2) init's shape refusals: a part of the wrong K, parts that do not
     sum to N, a down of the wrong shape, no shape table */
  op = htp_graph_op_at(w, fc_qkv);
  op->n_experts = 2u;
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE, "qkv parts short of N: %s",
        htp_graph_err_name(rc));
  op->n_experts = 3u;
  g_qs[op->h_gu[1]].K = 64u;
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE, "part of another K: %s",
        htp_graph_err_name(rc));
  g_qs[op->h_gu[1]].K = HID;
  op = htp_graph_op_at(w, ffn);
  {
    const uint32_t keep = op->h_dn[0];
    op->h_dn[0] = op->h_gu[0]; /* 128 x 64 where 64 x 128 belongs */
    rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &g);
    CHECK(rc == HTP_GRAPH_E_INVHANDLE, "down of another shape: %s",
          htp_graph_err_name(rc));
    op->h_dn[0] = keep;
  }
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE, "no shape table: %s",
        htp_graph_err_name(rc));
  printf("  Q4M1 op: no parts / short / K / down / no table -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  htp_graph_op_at(w, fc_qkv)->feed = 1u; /* the L2 feed, for (4) */
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &g);
  CHECK(rc == 0u && g != NULL, "hd64 all-kinds init: %s",
        htp_graph_err_name(rc));
  if (g == NULL)
    return;
  CHECK(g->logits != NULL && g->ffn != NULL && g->slot_words == 3u * HID,
        "Q4M1 buffers / slots %u", g->slot_words);
  /* the dense FFN's h_gu[2] is 0 and names nothing: slot 0 is the first
     FC's, so park that FC on another slot for this one probe */
  {
    const uint32_t fc0 = nth_op(w, HTP_OP_FC, 0), keep = g->ops[fc0].h_gu[0];
    g->ops[fc0].h_gu[0] = g->ops[fc_qkv].h_gu[0];
    CHECK(hexkl_graph_uses_q4m1(g, g->ops[ffn].h_dn[0]) &&
            hexkl_graph_uses_q4m1(g, g->ops[ffn].h_gu[1]) &&
            hexkl_graph_uses_q4m1(g, g->ops[lm].h_gu[1]) &&
            !hexkl_graph_uses_q4m1(g, 0u) &&
            !hexkl_graph_uses_q4m1(g, Q_SLOTS - 1u),
          "uses_q4m1");
    g->ops[fc0].h_gu[0] = keep;
  }

  /* (3) no runner in the env: AEE_EBADSTATE */
  fill(x, HID, &seed);
  rc = (uint32_t)hexkl_graph_forward(g, &env, fc_qkv, 1u, 0u, NULL, x, HID, out,
                                     256u, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "FC without a runner: %s",
        htp_graph_err_name(rc));
  env.fc = host_fc;

  /* (4) [FC] q | k | v: three parts, one quantization, the op's feed */
  g_fc_calls = 0;
  rc = (uint32_t)hexkl_graph_forward(g, &env, fc_qkv, 1u, 0u, NULL, x, HID, out,
                                     256u, &resume);
  CHECK(rc == 0u && resume == fc_qkv + 1u && g_fc_calls == 3u &&
          g_fc_feed == 1u,
        "[FC qkv]: %s resume %u calls %u feed %u", htp_graph_err_name(rc),
        resume, g_fc_calls, g_fc_feed);
  op = &g->ops[fc_qkv];
  spec_fc(op->h_gu[0], x, ref);
  spec_fc(op->h_gu[1], x, ref + 128);
  spec_fc(op->h_gu[2], x, ref + 192);
  CHECK(memcmp(out, ref, 256u * sizeof(float)) == 0, "[FC qkv] differs");
  err |= memcmp(out, ref, 256u * sizeof(float)) != 0;

  /* (5) [DENSE_FFN] up, gate, silu(gate) * up, down */
  rc = (uint32_t)hexkl_graph_forward(g, &env, ffn, 1u, 0u, NULL, x, HID, out,
                                     HID, &resume);
  CHECK(rc == 0u && resume == ffn + 1u, "[DENSE_FFN]: %s",
        htp_graph_err_name(rc));
  op = &g->ops[ffn];
  spec_fc(op->h_gu[0], x, up);
  spec_fc(op->h_gu[1], x, gate);
  m1_swiglu_cpu_det(gate, up, act, 64u);
  spec_fc(op->h_dn[0], act, ref);
  CHECK(memcmp(out, ref, HID * sizeof(float)) == 0, "[DENSE_FFN] differs");
  err |= memcmp(out, ref, HID * sizeof(float)) != 0;

  /* (6) [RMSNORM LM_HEAD]: the final norm from slot 0, the two slices,
     the logits as the stretch's output, the first maximum */
  fill(gam, HID, &seed);
  rc = (uint32_t)hexkl_graph_set_param(g, fin, HTP_GRAPH_PARAM_GAMMA, gam, HID);
  CHECK(rc == 0u, "final gamma: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_forward(g, &env, fin, 1000u, 0u, NULL, x, HID, out,
                                     64u, &resume);
  CHECK(rc == 0u && resume == w[3], "[RMSNORM LM_HEAD]: %s resume %u",
        htp_graph_err_name(rc), resume);
  m1_rmsnorm_det(x, gam, nrm, HID, HID, shape.eps, NULL);
  op = &g->ops[lm];
  spec_fc(op->h_gu[0], nrm, ref);
  spec_fc(op->h_gu[1], nrm, ref + 32);
  best = m1_argmax_first(ref, 64u);
  CHECK(memcmp(out, ref, 64u * sizeof(float)) == 0 && g->lm_id == best,
        "[RMSNORM LM_HEAD] differs (lm_id %u, want %u)", g->lm_id, best);
  err |= memcmp(out, ref, 64u * sizeof(float)) != 0 || g->lm_id != best;
  /* a tie: row best's weights copied into row r2 = best + 37 mod 64, so
     the first of the two equal maxima must win, as std::max_element */
  r2 = (best + 37u) % 64u;
  {
    const uint32_t hb = op->h_gu[best / 32u], h2 = op->h_gu[r2 / 32u];
    const size_t row = (HID / 32u) * Q4_CPU_BLOCK_BYTES;
    memcpy(g_qc[h2] + (r2 % 32u) * row, g_qc[hb] + (best % 32u) * row, row);
    q4m1_from_q4_0(g_qc[h2], HID, 32u, g_qw[h2]);
  }
  rc = (uint32_t)hexkl_graph_forward(g, &env, fin, 1000u, 0u, NULL, x, HID, out,
                                     64u, &resume);
  CHECK(rc == 0u && memcmp(&out[best], &out[r2], sizeof(float)) == 0 &&
          g->lm_id == (best < r2 ? best : r2),
        "argmax tie: rows %u and %u, lm_id %u", best, r2, g->lm_id);
  err |= g->lm_id != (best < r2 ? best : r2);
  /* [#132 Part B E3] LM_BAN: the pick skips the banned ids as the CPU's
     bad-words penalty (-inf) does, the logits stay raw, a repeated id is
     harmless; an id past N or a non-LM_HEAD op is refused */
  {
    float ban[3], lg[64], tmp[64];
    uint32_t id, want;
    memcpy(lg, out, sizeof(lg));
    id = best < r2 ? best : r2;
    memcpy(&ban[0], &id, 4u);
    memcpy(&ban[1], &id, 4u);
    id = (id + 5u) % 64u;
    memcpy(&ban[2], &id, 4u);
    rc =
      (uint32_t)hexkl_graph_set_param(g, lm, HTP_GRAPH_PARAM_LM_BAN, ban, 3u);
    CHECK(rc == 0u, "LM_BAN: %s", htp_graph_err_name(rc));
    rc = (uint32_t)hexkl_graph_forward(g, &env, fin, 1000u, 0u, NULL, x, HID,
                                       out, 64u, &resume);
    memcpy(tmp, lg, sizeof(tmp));
    tmp[best < r2 ? best : r2] = -INFINITY;
    tmp[id] = -INFINITY;
    want = m1_argmax_first(tmp, 64u);
    CHECK(rc == 0u && memcmp(out, lg, sizeof(lg)) == 0 && g->lm_id == want &&
            want == (best < r2 ? r2 : best),
          "LM_BAN: lm_id %u, want %u (the tie's second), logits %s", g->lm_id,
          want, memcmp(out, lg, sizeof(lg)) == 0 ? "raw" : "CHANGED");
    err |= rc != 0u || memcmp(out, lg, sizeof(lg)) != 0 || g->lm_id != want ||
           want != (best < r2 ? r2 : best);
    id = 64u;
    memcpy(&ban[0], &id, 4u);
    rc =
      (uint32_t)hexkl_graph_set_param(g, lm, HTP_GRAPH_PARAM_LM_BAN, ban, 1u);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "LM_BAN id 64 of 64: %s",
          htp_graph_err_name(rc));
    err |= rc != HTP_GRAPH_E_INVALIDFORMAT;
    rc =
      (uint32_t)hexkl_graph_set_param(g, fin, HTP_GRAPH_PARAM_LM_BAN, ban, 1u);
    CHECK(rc == HTP_GRAPH_E_BADITEM, "LM_BAN on RMSNORM: %s",
          htp_graph_err_name(rc));
    err |= rc != HTP_GRAPH_E_BADITEM;
  }
  /* (7) [#194 L1] the native bit on the three kinds' feed words: the
     validator takes it (and still refuses an unknown bit), the runner
     sees it, and FC / DENSE_FFN / LM_HEAD equal the native specs */
  rc = (uint32_t)hexkl_graph_set_param(g, lm, HTP_GRAPH_PARAM_LM_BAN, x, 0u);
  CHECK(rc == 0u, "LM_BAN clear: %s", htp_graph_err_name(rc));
  {
    uint32_t n_v = 0, keep = htp_graph_op_at(w, ffn)->feed;
    htp_graph_op_at(w, ffn)->feed = keep | HTP_GRAPH_FEED_NATIVE;
    rc = htp_graph_validate(w, n, hexkl_graph_resident_kinds(), &n_v);
    CHECK(rc == 0u, "native feed bit refused: %s", htp_graph_err_name(rc));
    err |= rc != 0u;
    htp_graph_op_at(w, ffn)->feed = keep | (1u << 17);
    rc = htp_graph_validate(w, n, hexkl_graph_resident_kinds(), &n_v);
    CHECK(rc == HTP_GRAPH_E_INVALIDFORMAT, "feed bit 17 taken: %s",
          htp_graph_err_name(rc));
    err |= rc != HTP_GRAPH_E_INVALIDFORMAT;
    htp_graph_op_at(w, ffn)->feed = keep;
  }
  g->ops[fc_qkv].feed |= HTP_GRAPH_FEED_NATIVE;
  g->ops[ffn].feed |= HTP_GRAPH_FEED_NATIVE;
  g->ops[lm].feed |= HTP_GRAPH_FEED_NATIVE;
  g_native = 1;
  {
    int nerr = 0;
    rc = (uint32_t)hexkl_graph_forward(g, &env, fc_qkv, 1u, 0u, NULL, x, HID,
                                       out, 256u, &resume);
    op = &g->ops[fc_qkv];
    spec_fc(op->h_gu[0], x, ref);
    spec_fc(op->h_gu[1], x, ref + 128);
    spec_fc(op->h_gu[2], x, ref + 192);
    nerr |= rc != 0u || g_fc_feed != (1u | HTP_GRAPH_FEED_NATIVE) ||
            memcmp(out, ref, 256u * sizeof(float)) != 0;
    rc = (uint32_t)hexkl_graph_forward(g, &env, ffn, 1u, 0u, NULL, x, HID, out,
                                       HID, &resume);
    op = &g->ops[ffn];
    spec_fc(op->h_gu[0], x, up);
    spec_fc(op->h_gu[1], x, gate);
    m1_swiglu_cpu_det(gate, up, act, 64u);
    spec_fc(op->h_dn[0], act, ref);
    nerr |= rc != 0u || memcmp(out, ref, HID * sizeof(float)) != 0;
    rc = (uint32_t)hexkl_graph_forward(g, &env, fin, 1000u, 0u, NULL, x, HID,
                                       out, 64u, &resume);
    op = &g->ops[lm];
    spec_fc(op->h_gu[0], nrm, ref);
    spec_fc(op->h_gu[1], nrm, ref + 32);
    nerr |= rc != 0u || memcmp(out, ref, 64u * sizeof(float)) != 0 ||
            g->lm_id != m1_argmax_first(ref, 64u);
    CHECK(nerr == 0, "[#194 L1] native FC / DENSE_FFN / LM_HEAD differ from "
                     "q4_gemv_native_det");
    err |= nerr;
    if (nerr == 0)
      printf("GRAPH Q4M1 NATIVE BIT-IDENTICAL: FC q|k|v DENSE_FFN "
             "RMSNORM+LM_HEAD with HTP_GRAPH_FEED_NATIVE vs "
             "q4_gemv_native_det (bit 17 refused)\n");
  }
  g_native = 0;
  hexkl_graph_free(g);
  if (err == 0)
    printf("GRAPH Q4M1 BIT-IDENTICAL: FC q|k|v (3 parts, feed passed) "
           "DENSE_FFN (up gate swiglu down) RMSNORM+LM_HEAD (2 slices, "
           "argmax first of a tie, LM_BAN skips its ids) vs q4_gemv_cpu_det / "
           "m1_swiglu_cpu_det / "
           "m1_argmax_first (hd64 shape, the real HVX kernel on hvx_emu)\n");
}

/* ---- [plan 201 S4] Gemma 4's attention shapes -------------------------- */

/* google/gemma-4-26B-A4B: 16 q heads; sliding layers head_dim 256 over 8 kv
   heads, full layers head_dim 512 over 2 (num_global_key_value_heads). Three
   layers, sliding / full / sliding, at a max_seq of two 64-position tiles. */
#define G4_HID 2816u
#define G4_SEQ 128u
/** @brief The sliding layers' window here: Gemma's 1024 does not fit
 *  G4_SEQ, 48 crosses the first tile boundary mid-tile (attn_m1_host_check
 *  runs 1024 itself). */
#define G4_WIN 48u
static const struct {
  uint32_t n_kv, gqa, hd;
} kG4[3] = {{8u, 2u, 256u}, {2u, 8u, 512u}, {8u, 2u, 256u}};

static void g4_op(uint32_t *w, uint32_t i, uint32_t kind, uint32_t layer,
                  uint32_t resident, uint32_t K, uint32_t N) {
  htp_graph_op *op = htp_graph_op_at(w, i);
  memset(op, 0, sizeof(*op));
  op->kind = kind;
  op->layer = layer;
  op->resident = resident;
  op->K = K;
  op->N = N;
  op->next_mm = HTP_GRAPH_NO_OP;
  op->in_slot = 2u;
  op->out_slot = kind == HTP_OP_ATTN_M1 ? 1u : 2u;
  if (layer < 3u) {
    op->n_kv = kG4[layer].n_kv;
    op->gqa = kG4[layer].gqa;
    op->head_dim = kG4[layer].hd;
  }
  if (kind == HTP_OP_QK_NORM || kind == HTP_OP_RMSNORM) {
    op->eps_bits = 0x358637BDu; /* 1e-6f */
  }
}

/** @brief [QK_NORM ROPE ATTN_M1] x 3 then [RMSNORM LM_HEAD]; ROPE resident,
 *  QK_NORM and the tail not, ATTN_M1 as @a attn. @return the words. */
static uint32_t g4_list(uint32_t *w, uint32_t attn) {
  const uint32_t n_ops = 3u * 3u + 2u;
  uint32_t l, i = 0;
  w[0] = HTP_GRAPH_MAGIC;
  w[1] = HTP_GRAPH_VERSION;
  w[2] = 3u;
  w[3] = n_ops;
  w[4] = G4_HID;
  w[5] = 256u;
  w[6] = G4_SEQ;
  for (l = 0; l < 3u; ++l) {
    w[HTP_GRAPH_HEADER_WORDS + l] = HTP_GRAPH_LAYER_ATTN;
    w[HTP_GRAPH_HEADER_WORDS + 3u + l] = HTP_GRAPH_FFN_DENSE;
  }
  for (l = 0; l < 3u; ++l) {
    const uint32_t q = kG4[l].gqa * kG4[l].n_kv * kG4[l].hd;
    const uint32_t K = q + 2u * kG4[l].n_kv * kG4[l].hd;
    g4_op(w, i++, HTP_OP_QK_NORM, l, 0u, K, K);
    g4_op(w, i++, HTP_OP_ROPE, l, 1u, K, K);
    g4_op(w, i++, HTP_OP_ATTN_M1, l, attn, K, q);
    /* scale 1.0 (Gemma4TextAttention's scaling), the window on sliding
       layers only */
    htp_graph_op_at(w, i - 1u)->eps_bits = 0x3F800000u;
    htp_graph_op_at(w, i - 1u)->top_k = l == 1u ? 0u : G4_WIN;
  }
  g4_op(w, i++, HTP_OP_RMSNORM, 3u, 0u, G4_HID, G4_HID);
  g4_op(w, i++, HTP_OP_LM_HEAD, 3u, 0u, G4_HID, 256u);
  return htp_graph_words_for(3u, n_ops);
}

/** @brief Gemma's cos | sin table, max_seq x hd: mha_core's default (all
 *  angles) or proportional (the first @a partial of them, 0 above). */
static void g4_table(float *cs, uint32_t hd, double theta, double partial) {
  const uint32_t h = hd / 2u, angles = (uint32_t)(partial * hd / 2.0);
  uint32_t p, i;
  for (p = 0; p < G4_SEQ; ++p) {
    for (i = 0; i < h; ++i) {
      const double a =
        i < angles ? (double)p * pow(theta, -(2.0 * i) / (double)hd) : 0.0;
      cs[(size_t)p * hd + i] = (float)cos(a);
      cs[(size_t)p * hd + h + i] = (float)sin(a);
    }
  }
}

/* (1) The list validates with ROPE resident at head_dim 256 / 512; (2) a
   ROPE op with no table of its own and no head_dim 64 -> EBADSTATE (the
   shared hd64 table is not read at another head_dim); (3) tables bound per
   op, the two sliding layers' equal ones shared and the full one apart, a
   wrong length refused; (4) each ROPE op's forward == m1_rope_det on its
   q and k heads with its own table, v untouched. */
static void check_gemma_rope(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  static float cs_s[G4_SEQ * 256u], cs_f[G4_SEQ * 512u], cs64[G4_SEQ * 64u];
  static float in[10240u], out[10240u], ref[10240u];
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  uint32_t n, rc, resume, seed = 4296u, l, i, p;
  int err = 0;
  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  n = g4_list(w, 0u);
  rc = htp_graph_validate(w, n, ALL_KINDS, NULL);
  CHECK(rc == 0u, "gemma rope list: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "gemma rope init: %s", htp_graph_err_name(rc));
  if (g == NULL)
    return;
  fill(in, 10240u, &seed);
  fill(cs64, G4_SEQ * 64u, &seed);
  rc = (uint32_t)hexkl_graph_set_param(
    g, HTP_GRAPH_NO_OP, HTP_GRAPH_PARAM_ROPE_TABLE, cs64, G4_SEQ * 64u);
  CHECK(rc == 0u, "gemma: shared hd64 table: %s", htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_forward(g, &env, 1u, 1u, 5u, NULL, in, 8192u, out,
                                     8192u, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "gemma ROPE hd256, no own table: %s",
        htp_graph_err_name(rc));
  g4_table(cs_s, 256u, 1e4, 1.0);
  g4_table(cs_f, 512u, 1e6, 0.25);
  rc = (uint32_t)hexkl_graph_set_param(g, 1u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_s,
                                       G4_SEQ * 256u - 1u);
  CHECK(rc == (uint32_t)AEE_EINVALIDFORMAT, "gemma: short table: %s",
        htp_graph_err_name(rc));
  rc = (uint32_t)hexkl_graph_set_param(g, 1u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_s,
                                       G4_SEQ * 256u);
  rc |= (uint32_t)hexkl_graph_set_param(g, 4u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_f,
                                        G4_SEQ * 512u);
  rc |= (uint32_t)hexkl_graph_set_param(g, 7u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_s,
                                        G4_SEQ * 256u);
  CHECK(rc == 0u, "gemma: per-op tables: %s", htp_graph_err_name(rc));
  CHECK(g->param[1] != NULL && g->param[1] == g->param[7] &&
          g->param[4] != g->param[1],
        "gemma: the sliding tables are not shared (or the full one is)");
  /* re-binding a shared table with other bytes splits it, never writes the
     other op's copy */
  rc = (uint32_t)hexkl_graph_set_param(g, 7u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_f,
                                       G4_SEQ * 256u);
  CHECK(rc == 0u && g->param[7] != g->param[1] &&
          memcmp(g->param[1], cs_s, sizeof(cs_s)) == 0,
        "gemma: re-binding a shared table");
  rc = (uint32_t)hexkl_graph_set_param(g, 7u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_s,
                                       G4_SEQ * 256u);
  CHECK(rc == 0u && g->param[7] == g->param[1], "gemma: re-sharing");
  for (l = 0; l < 3u; ++l) {
    const uint32_t hd = kG4[l].hd, nq = kG4[l].gqa * kG4[l].n_kv,
                   nk = kG4[l].n_kv, K = (nq + 2u * nk) * hd;
    const float *cs = l == 1u ? cs_f : cs_s;
    for (p = 0; p < G4_SEQ; p += 37u) {
      fill(in, K, &seed);
      rc = (uint32_t)hexkl_graph_forward(g, &env, 3u * l + 1u, 1000u, p, NULL,
                                         in, K, out, K, &resume);
      CHECK(rc == 0u && resume == 3u * l + 2u, "gemma ROPE l%u pos %u: %s", l,
            p, htp_graph_err_name(rc));
      memcpy(ref, in, K * sizeof(float));
      for (i = 0; i < nq + nk; ++i)
        m1_rope_det(ref + i * hd, hd, cs + (size_t)p * hd);
      if (memcmp(out, ref, K * sizeof(float)) != 0) {
        CHECK(0, "gemma ROPE l%u (hd %u) pos %u differs", l, hd, p);
        err = 1;
      }
    }
  }
  hexkl_graph_free(g); /* the shared table freed once (ASan / valgrind) */
  if (err == 0)
    printf("GRAPH GEMMA ROPE BIT-IDENTICAL: hd256 default 1e4 (2 layers, one "
           "shared table) and hd512 proportional 0.25 1e6, own tables, 4 "
           "positions each, vs m1_rope_det\n");
}

/* [plan 201 S4] [ROPE ATTN_M1] resident on the three Gemma layers, a
   sliding cache (8 x 256, two layers) and a full one (2 x 512) in the env:
   (1) the ordinals count within each shape's cache; (2) a shape with no
   cache -> EBADSTATE; (3) a chain of 100 tokens through every layer, each
   output bit-identical to m1_rope_det -> attn_m1_det_forward_win with the
   op's window (48 on the sliding layers) and scale 1.0. */
static void check_gemma_attn(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  static float cs_s[G4_SEQ * 256u], cs_f[G4_SEQ * 512u];
  static float in[10240u], out[8192u], ref[8192u], qk[10240u], e[G4_SEQ];
  static float kt[3][8u * 256u * G4_SEQ], vv[3][8u * 256u * G4_SEQ];
  hexkl_graph_env env;
  hexkl_graph *g = NULL;
  uint32_t n, rc, resume, seed = 4296u, l, i, p;
  int err = 0, cerr = 0;
  memset(&env, 0, sizeof(env));
  env.tbl = &g_tbl;
  n = g4_list(w, 1u);
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, NULL, 0u, &g);
  CHECK(rc == 0u && g != NULL, "gemma attn init: %s", htp_graph_err_name(rc));
  if (g == NULL)
    return;
  CHECK(g->ordinal[2] == 0u && g->ordinal[5] == 0u && g->ordinal[8] == 1u,
        "gemma ordinals %u %u %u", g->ordinal[2], g->ordinal[5], g->ordinal[8]);
  g4_table(cs_s, 256u, 1e4, 1.0);
  g4_table(cs_f, 512u, 1e6, 0.25);
  rc = (uint32_t)hexkl_graph_set_param(g, 1u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_s,
                                       G4_SEQ * 256u);
  rc |= (uint32_t)hexkl_graph_set_param(g, 4u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_f,
                                        G4_SEQ * 512u);
  rc |= (uint32_t)hexkl_graph_set_param(g, 7u, HTP_GRAPH_PARAM_ROPE_TABLE, cs_s,
                                        G4_SEQ * 256u);
  CHECK(rc == 0u, "gemma attn tables: %s", htp_graph_err_name(rc));
  env.attn_m1 = hvx_attn_m1_create(2u, 8u, 2u, 256u, G4_SEQ, NULL, &cerr);
  CHECK(env.attn_m1 != NULL, "sliding cache: %d", cerr);
  fill(in, 10240u, &seed);
  rc = (uint32_t)hexkl_graph_forward(g, &env, 4u, 1000u, 0u, NULL, in, 10240u,
                                     out, 8192u, &resume);
  CHECK(rc == (uint32_t)AEE_EBADSTATE, "full layer with no hd512 cache: %s",
        htp_graph_err_name(rc));
  printf("  ATTN_M1 hd512 with only an hd256 cache    -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  env.attn_m1_b = hvx_attn_m1_create(1u, 2u, 8u, 512u, G4_SEQ, NULL, &cerr);
  CHECK(env.attn_m1_b != NULL, "full cache: %d", cerr);
  if (env.attn_m1 == NULL || env.attn_m1_b == NULL) {
    err = 1;
    p = 100u;
  } else {
    p = 0u;
  }
  memset(kt, 0, sizeof(kt));
  memset(vv, 0, sizeof(vv));
  for (; p < 100u; ++p) {
    for (l = 0; l < 3u; ++l) {
      const uint32_t hd = kG4[l].hd, nk = kG4[l].n_kv, nq = kG4[l].gqa * nk;
      const uint32_t K = (nq + 2u * nk) * hd, win = l == 1u ? 0u : G4_WIN;
      const float *cs = (l == 1u ? cs_f : cs_s) + (size_t)p * hd;
      fill(in, K, &seed);
      rc = (uint32_t)hexkl_graph_forward(g, &env, 3u * l + 1u, 1000u, p, NULL,
                                         in, K, out, nq * hd, &resume);
      if (rc != 0u || resume != 3u * l + 3u) {
        CHECK(0, "gemma attn l%u pos %u: %s resume %u", l, p,
              htp_graph_err_name(rc), resume);
        err = 1;
        continue;
      }
      /* the spec: RoPE on q then k heads, append k | v, attend */
      memcpy(qk, in, K * sizeof(float));
      for (i = 0; i < nq + nk; ++i)
        m1_rope_det(qk + i * hd, hd, cs);
      for (i = 0; i < nk; ++i)
        attn_m1_det_append(kt[l] + (size_t)i * hd * G4_SEQ,
                           vv[l] + (size_t)i * G4_SEQ * hd, hd, G4_SEQ, p,
                           qk + (nq + i) * hd, in + (nq + nk + i) * hd);
      attn_m1_det_forward_win(qk, kt[l], vv[l], nk, nq / nk, hd, G4_SEQ, p + 1u,
                              win, 1.0f, e, ref, NULL);
      if (memcmp(out, ref, nq * hd * sizeof(float)) != 0) {
        CHECK(0, "gemma attn l%u (hd %u) pos %u differs", l, hd, p);
        err = 1;
      }
    }
  }
  hvx_attn_m1_free(env.attn_m1);
  hvx_attn_m1_free(env.attn_m1_b);
  hexkl_graph_free(g);
  if (err == 0)
    printf("GRAPH GEMMA ATTN BIT-IDENTICAL: ROPE+ATTN_M1 on 3 layers (hd256 "
           "window %u x 2 in one cache, hd512 full in a second), scale 1.0, "
           "100 tokens, vs m1_rope_det -> attn_m1_det_forward_win\n",
           G4_WIN);
}

int main(void) {
  check_validator();
  check_forward();
  check_limits();
  check_stretches();
  check_add_router(0);
  check_add_router(1);
  check_gemma_rope();
  check_gemma_attn();
  check_q4m1();
  if (g_fail) {
    printf("GRAPH CHECKS FAILED (%d)\n", g_fail);
    return 1;
  }
  printf("GRAPH CHECKS PASS\n");
  return 0;
}
