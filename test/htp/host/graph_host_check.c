// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   graph_host_check.c
 * @date   23 Sep 2026
 * @brief  Host check of the per-token entry skeleton's op table and forward
 *         loop (#85)
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
   arguments are the same (same function, same session state). */
#include "hexkl_graph.h"
#include "htp_graph_desc.h"
#include <AEEStdErr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
static const htp_graph_lfm2_shape kLfm25 = {24, 2,  2048, 7168, 1792,   32,
                                            4,  32, 8,    64,   128000, 2048};
/* unittest_causallm_lfm2_moe.cpp:85-113, layer_types [attention, conv] */
static const htp_graph_lfm2_shape kTiny = {2, 0, 64, 64, 64, 4,
                                           2, 8, 4,  8,  32, 8};

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
  htp_graph_op_at(w, 0)->resident = 1u;
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
};

static void check_validator(void) {
  static uint32_t w[HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +
                    HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS];
  static uint32_t m[sizeof(w) / sizeof(w[0])];
  const uint32_t cap = (uint32_t)(sizeof(w) / sizeof(w[0]));
  const uint32_t resident_ok = HTP_GRAPH_KIND_BIT(HTP_OP_MOE);
  uint32_t n = build(w, cap, &kLfm25, kLfm25Layers, resident_ok);
  uint32_t n_ops = 0, i, rc;
  CHECK(n != 0u, "LFM2.5 build refused");
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
    CHECK(op->resident == (op->kind == HTP_OP_MOE), "resident bit %u", i);
  }
  CHECK(htp_graph_op_cat(w, 1)->next_mm == 3u, "op 1 (in_proj) -> op 3");
  CHECK(htp_graph_op_cat(w, n_ops - 1u)->next_mm == HTP_GRAPH_NO_OP,
        "lm_head has no next");
  printf("GRAPH VALIDATOR OK: LFM2.5-8B-A1B n_ops=%u moe=22 words=%u\n", n_ops,
         n);

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
          HTP_GRAPH_E_BADSTATE == (uint32_t)AEE_EBADSTATE,
        "error code values drifted from AEEStdErr.h");
  CHECK(hexkl_graph_resident_kinds() == resident_ok,
        "kernel table: resident kinds 0x%x", hexkl_graph_resident_kinds());
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
/* Layer l's expert e: gate_up handle 10 + 40 l + e, down 30 + 40 l + e. */
static void bind_tiny(uint32_t *w) {
  uint32_t l, e;
  for (l = 0; l < 2u; ++l) {
    htp_graph_op *op = htp_graph_op_at(w, nth_op(w, HTP_OP_MOE, l));
    for (e = 0; e < 4u; ++e) {
      op->h_gu[e] = 10u + 40u * l + e;
      op->h_dn[e] = 30u + 40u * l + e;
      register_weight(op->h_gu[e], 64u, 128u);
      register_weight(op->h_dn[e], 64u, 64u);
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
  env.pool = (hvx_worker_pool *)&env; /* any non-NULL, only passed through */
  env.scratch = &g_scratch;
  for (i = 0; i < 64u; ++i) {
    act[i] = frand(&seed);
    sentinel[i] = -12345.0f;
  }

  /* (1) every op non-resident: the identity at every start op */
  n = build(w, cap, &kTiny, "AC", 0u);
  CHECK(n != 0u, "tiny build");
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, &g);
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

  /* (2) MOE resident: init refuses a free or mis-shaped handle */
  n = build(w, cap, &kTiny, "AC", resident_ok);
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE && g == NULL, "unbound handles: %s",
        htp_graph_err_name(rc));
  bind_tiny(w);
  htp_graph_op_at(w, nth_op(w, HTP_OP_MOE, 1))->h_dn[2] = 7u; /* free slot */
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE, "free handle: %s", htp_graph_err_name(rc));
  printf("  missing weight handle                     -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  bind_tiny(w);
  g_tbl.slots[31].N = 96u; /* layer 0 expert 1's down: inter differs */
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, &g);
  CHECK(rc == HTP_GRAPH_E_INVHANDLE, "mis-shaped handle: %s",
        htp_graph_err_name(rc));
  printf("  weight handle of another shape            -> %s (0x%x)\n",
         htp_graph_err_name(rc), rc);
  g_tbl.slots[31].N = 64u;
  rc = (uint32_t)hexkl_graph_init(w, n, &g_tbl, &g);
  CHECK(rc == 0u && g != NULL, "init (MOE resident): %s",
        htp_graph_err_name(rc));
  if (g == NULL)
    return;
  CHECK(g->slot_words == 64u, "slot_words %u", g->slot_words);
  CHECK(hexkl_graph_uses_handle(g, 10u) && hexkl_graph_uses_handle(g, 53u) &&
          !hexkl_graph_uses_handle(g, 7u) && !hexkl_graph_uses_handle(g, 99u),
        "uses_handle");

  /* (3) bit identity against a direct call, both flag values, both layers */
  for (f = 0; f < 2u; ++f) {
    env.moe_flags = flag_cases[f];
    for (l = 0; l < 2u; ++l) {
      const uint32_t start = nth_op(w, HTP_OP_MOE, l);
      const htp_graph_op *op = htp_graph_op_cat(w, start);
      moe_args direct;
      uint32_t j;
      hexkl_mm_u8i4_moe_layer_run(&g_tbl, g_vtcm, sizeof(g_vtcm), 32u, 1u, 64u,
                                  64u, 64u, 4u, op->h_gu, op->h_dn, row_index,
                                  row_count, row_weight, act, ref, env.pool,
                                  &g_scratch, env.moe_flags);
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
  }
  hexkl_graph_free(g);
  hexkl_graph_free(NULL);
  printf("GRAPH FORWARD REFUSALS OK\n");
}

int main(void) {
  check_validator();
  check_forward();
  if (g_fail) {
    printf("GRAPH CHECKS FAILED (%d)\n", g_fail);
    return 1;
  }
  printf("GRAPH CHECKS PASS\n");
  return 0;
}
