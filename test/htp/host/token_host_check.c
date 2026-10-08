// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   token_host_check.c
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2, #211] Host check of the one-PD token driver
 *         (hmx/hexkl_token.c): one session with every kind resident,
 *         against the one-session forward
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The hd64 fixture's list (3 layers C A C, layer 0 dense, two MoE layers,
 * vocab 64 in two lm_head slices) with every kind resident. Reference:
 * hexkl_graph_forward from op 0 per token. Under test: hexkl_token_main on
 * a second graph of the same words, its own slots, pool, KV cache and conv
 * state, with a malloc'd page for the miss lines. Same parameters and
 * weights in both, TOKENS tokens at pos = tok % max_seq (the KV cache
 * rewinds at 0, the conv state runs on).
 *
 * Kernels: the REAL small ops, router, m=1 attention, residual add and
 * CPU-exact Q4_0 FC (hvx_q4_gemv_f32.c) on hvx_emu/, as graph_host_check;
 * the MoE is a stand-in that mixes its input into its output under the
 * routing and records a hash of every call's input and output (the dump's
 * analog). Gated: every token's logits memcmp-equal and ids equal, every
 * MoE call's in / out hash equal, timeouts 0, stale 0. Then the
 * pool (miss rounds against an owner pthread) and the miss round's
 * failure paths: no owner (AEE_EEXPIRED after the 1 s window, not a
 * hang), a stale answer (HEXKL_TOKEN_E_STALE), the owner's code. What this
 * cannot show: the DSP's caches (the clean / flush-invalidate calls are
 * fences here).
 */
#include "hexkl_graph.h"
#include "hexkl_token.h"
#include "htp_graph_desc.h"
#include <AEEStdErr.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hvx_attn_m1_f32.h"
#include "hvx_q4_gemv_f32.h"
#include "hvx_worker_pool.h"
#include "q4_gemv_cpu_det.h"

#define TOKENS 10000u
#define HID 128u
#define VOCAB 64u
#define MAXW                                                                   \
  (HTP_GRAPH_HEADER_WORDS + 2u * HTP_GRAPH_MAX_LAYERS +                        \
   HTP_GRAPH_MAX_OPS * HTP_GRAPH_OP_WORDS)
#define Q_SLOTS 16u
#define SPIN_US 20000u

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

/* ---- the MoE stand-in: deterministic, routing-dependent, recorded ---- */
static uint64_t *g_moe_log; /* in hash, out hash per call */
static uint32_t g_moe_calls, g_moe_cap;

static uint64_t fnv(const void *p, size_t n) {
  const uint8_t *b = (const uint8_t *)p;
  uint64_t h = 1469598103934665603ull;
  while (n--)
    h = (h ^ *b++) * 1099511628211ull;
  return h;
}

/* [#225] hexkl_graph.c's FC on WH handles: no op of this check's list
   carries the WH bit (graph_host_check holds that path). */
int hexkl_mm_u8i4_fc_m1_run(const hexkl_weight_u8i4_table *tbl,
                            uint8_t *vtcm_base, uint32_t vtcm_size,
                            uint32_t config_off, uint32_t K, uint32_t n_parts,
                            const uint32_t *h, const float *act_f32,
                            float *out_f32, hvx_worker_pool *pool,
                            hexkl_moe_scratch *scratch, uint32_t flags) {
  (void)tbl, (void)vtcm_base, (void)vtcm_size, (void)config_off, (void)K;
  (void)n_parts, (void)h, (void)act_f32, (void)out_f32, (void)pool;
  (void)scratch, (void)flags;
  return AEE_EUNSUPPORTED;
}

int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch, uint32_t flags) {
  uint32_t e, c, r = 0;
  (void)vtcm_base, (void)vtcm_size, (void)config_off, (void)pool;
  (void)scratch, (void)flags, (void)inter, (void)h_down;
  memset(out_f32, 0, (size_t)M * N_out * sizeof(float));
  for (e = 0; e < n_experts; ++e) {
    uint32_t i;
    for (i = 0; i < row_count[e]; ++i, ++r) {
      const float *x = act_f32 + (size_t)row_index[r] * K;
      float *y = out_f32 + (size_t)row_index[r] * N_out;
      /* the expert is what its gate_up's bytes say, not its handle: the
         pool check moves experts between handles */
      const uint32_t id = *(const uint32_t *)tbl->slots[h_gate_up[e]].wh_bytes;
      for (c = 0; c < N_out; ++c)
        y[c] +=
          row_weight[r] *
          (x[(c + 7u * e) % K] * (float)(id + 1u) * 0.125f + 0.01f * (float)c);
    }
  }
  if (g_moe_calls < g_moe_cap) {
    g_moe_log[2u * g_moe_calls] = fnv(act_f32, (size_t)K * sizeof(float));
    g_moe_log[2u * g_moe_calls + 1u] =
      fnv(out_f32, (size_t)N_out * sizeof(float));
  }
  ++g_moe_calls;
  return AEE_SUCCESS;
}

/* ---- weights and parameters ------------------------------------------ */
static hexkl_weight_u8i4_table g_tbl;
/* each u8i4 handle's "bytes": the id of the expert it holds (10 m + e) */
static uint32_t g_wh[64];
static uint8_t g_vtcm[64];
static uint8_t *g_qw[Q_SLOTS];
static hexkl_graph_q4m1_shape g_qs[Q_SLOTS];

static int host_fc(void *ctx, uint32_t h, uint32_t feed, const hvx_q4m1_act *a,
                   float *y) {
  (void)ctx;
  if (h >= Q_SLOTS || g_qw[h] == NULL)
    return AEE_EBADITEM;
  ((feed & HTP_GRAPH_FEED_NATIVE) != 0u
     ? hvx_q4m1_gemv_groups_native
     : hvx_q4m1_gemv_groups)(g_qw[h], g_qs[h].K, g_qs[h].N / Q4M1_GROUP, a, y);
  return AEE_SUCCESS;
}

static uint32_t lcg(uint32_t *s) { return *s = *s * 1664525u + 1013904223u; }
static float frand(uint32_t *s) {
  return (float)(lcg(s) >> 8) / 16777216.0f * 2.0f - 1.0f;
}
static void fill(float *p, uint32_t n, uint32_t *s) {
  while (n--)
    *p++ = frand(s);
}

static uint32_t q_register(uint32_t h, uint32_t K, uint32_t N, uint32_t *s) {
  const size_t bytes = (size_t)N * (K / 32u) * Q4_CPU_BLOCK_BYTES;
  uint8_t *c = malloc(bytes);
  size_t b;
  uint32_t j;
  for (b = 0; b < bytes / Q4_CPU_BLOCK_BYTES; ++b) {
    uint8_t *blk = c + b * Q4_CPU_BLOCK_BYTES;
    const uint16_t d = (uint16_t)(0x2000u | ((lcg(s) >> 8) & 0x7ffu));
    blk[0] = (uint8_t)d;
    blk[1] = (uint8_t)(d >> 8);
    for (j = 0; j < 16u; ++j)
      blk[2 + j] = (uint8_t)(lcg(s) >> 24);
  }
  g_qw[h] = aligned_alloc(128, (q4m1_bytes(K, N) + 127u) & ~(size_t)127u);
  q4m1_from_q4_0(c, K, N, g_qw[h]);
  free(c);
  g_qs[h].K = K;
  g_qs[h].N = N;
  return h;
}

/* The hd64 list, every kind resident, its weights bound. */
static uint32_t build_words(uint32_t *w) {
  static const htp_graph_lfm2_shape shape = {3, 1, 128, 64,    32, 4,    2,
                                             2, 1, 64,  VOCAB, 64, 1e-6f};
  const uint8_t attn[3] = {0, 1, 0};
  uint32_t n = htp_graph_lfm2_build(w, MAXW, &shape, attn, HTP_GRAPH_KINDS_ALL);
  uint32_t i, e, h = 0, m = 0, seed = 0x132e2u;
  for (i = 0; i < w[3]; ++i) {
    htp_graph_op *op = htp_graph_op_at(w, i);
    if (op->kind == HTP_OP_MOE) {
      /* MoE op m's expert e: gate_up 10 m + e, down 10 m + 5 + e, bound
         as its EXPERTS table (bind_params) */
      for (e = 0; e < op->n_experts; ++e) {
        g_wh[10u * m + e] = g_wh[10u * m + 5u + e] = 10u * m + e;
        g_tbl.slots[10u * m + e].wh_bytes = (uint8_t *)&g_wh[10u * m + e];
        g_tbl.slots[10u * m + 5u + e].wh_bytes =
          (uint8_t *)&g_wh[10u * m + 5u + e];
        g_tbl.slots[10u * m + e].in_use = 1;
        g_tbl.slots[10u * m + e].K = op->K;
        g_tbl.slots[10u * m + e].N = 2u * op->N;
        g_tbl.slots[10u * m + 5u + e].in_use = 1;
        g_tbl.slots[10u * m + 5u + e].K = op->N;
        g_tbl.slots[10u * m + 5u + e].N = op->N_out;
      }
      ++m;
    } else if (op->kind == HTP_OP_DENSE_FFN) {
      op->h_gu[0] = q_register(h++, op->K, op->N, &seed);
      op->h_gu[1] = q_register(h++, op->K, op->N, &seed);
      op->h_dn[0] = q_register(h++, op->N, op->N_out, &seed);
      op->n_experts = 3u;
    } else if (op->kind == HTP_OP_LM_HEAD) {
      op->h_gu[0] = q_register(h++, op->K, op->N / 2u, &seed);
      op->h_gu[1] = q_register(h++, op->K, op->N / 2u, &seed);
      op->n_experts = 2u;
    } else if (op->kind == HTP_OP_FC) {
      op->h_gu[0] = q_register(h++, op->K, op->N, &seed);
      op->n_experts = 1u;
    }
  }
  CHECK(n != 0u && m == 2u && h <= Q_SLOTS, "build: n %u moe %u slots %u", n, m,
        h);
  return n;
}

/* Every parameter of every op, the same bytes on every graph. */
static void bind_params(hexkl_graph *g) {
  static float buf[64u * 64u * 4u];
  uint32_t i, e, s = 0x51u, m = 0;
  int rc = 0;
  for (i = 0; i < g->n_ops; ++i) {
    const htp_graph_op *op = &g->ops[i];
    switch (op->kind) {
    case HTP_OP_RMSNORM:
      fill(buf, op->K, &s);
      rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_GAMMA, buf, op->K);
      break;
    case HTP_OP_QK_NORM:
      fill(buf, 2u * op->head_dim, &s);
      rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_GAMMA, buf,
                                  2u * op->head_dim);
      break;
    case HTP_OP_CONV1D_GATE:
      fill(buf, 3u * op->N, &s);
      rc |=
        hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_CONV_W, buf, 3u * op->N);
      fill(buf, 2u * op->N, &s);
      rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_CONV_STATE, buf,
                                  2u * op->N);
      break;
    case HTP_OP_ROUTER_TOPK:
      fill(buf, op->K * op->n_experts, &s);
      rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_ROUTER_W, buf,
                                  op->K * op->n_experts);
      fill(buf, op->n_experts, &s);
      rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_ROUTER_BIAS, buf,
                                  op->n_experts);
      break;
    case HTP_OP_MOE:
      for (e = 0; e < 2u * op->n_experts; ++e) {
        const uint32_t h =
          e < op->n_experts ? 10u * m + e : 10u * m + 5u + e - op->n_experts;
        memcpy(&buf[e], &h, sizeof(h));
      }
      rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_EXPERTS, buf,
                                  2u * op->n_experts);
      ++m;
      break;
    default:
      break;
    }
  }
  fill(buf, g->max_seq * 64u, &s);
  rc |= hexkl_graph_set_param(g, HTP_GRAPH_NO_OP, HTP_GRAPH_PARAM_ROPE_TABLE,
                              buf, g->max_seq * 64u);
  CHECK(rc == 0, "bind_params: 0x%x", (unsigned)rc);
}

/* One session: its graph, its env (own pool, own KV cache). */
typedef struct {
  hexkl_graph *g;
  hexkl_graph_env env;
  hexkl_moe_scratch scratch;
} session;

static void open_session(session *s, const uint32_t *words, uint32_t n) {
  static uint32_t w[MAXW];
  int err = 0;
  memcpy(w, words, n * sizeof(uint32_t));
  memset(s, 0, sizeof(*s));
  CHECK(hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &s->g) == 0, "init");
  s->env.tbl = &g_tbl;
  s->env.vtcm_base = g_vtcm;
  s->env.vtcm_size = sizeof(g_vtcm);
  s->env.config_off = 32u;
  s->env.pool = hvx_worker_pool_create(3u);
  s->env.scratch = &s->scratch;
  s->env.attn_m1 =
    hvx_attn_m1_create(1u, 1u, 2u, 64u, s->g->max_seq, s->env.pool, &err);
  s->env.fc = host_fc;
  CHECK(s->env.attn_m1 != NULL, "attn_m1_create: %d", err);
  bind_params(s->g);
}

static void close_session(session *s) {
  hexkl_graph_free(s->g);
  hvx_attn_m1_free(s->env.attn_m1);
  hvx_worker_pool_destroy(s->env.pool);
}

static void emb_row(uint32_t tok, float *x) {
  uint32_t s = 0x9e3779b9u ^ tok;
  fill(x, HID, &s);
}

static uint64_t now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
}

static uint8_t *new_page(void) {
  uint8_t *p = aligned_alloc(4096, 20480);
  memset(p, 0, 20480);
  return p;
}

/* The one-session forward: every token's logits and id, and (when
   g_moe_log is set) its MoE calls' hashes. */
static void reference(const uint32_t *words, uint32_t n,
                      float (*ref_logits)[VOCAB], uint32_t *ref_id) {
  static float x[HID];
  session ref;
  uint32_t t, resume;
  int rc;
  open_session(&ref, words, n);
  for (t = 0; t < TOKENS; ++t) {
    emb_row(t, x);
    rc = hexkl_graph_forward(ref.g, &ref.env, 0u, HTP_GRAPH_MAX_OPS,
                             t % ref.g->max_seq, NULL, x, HID, ref_logits[t],
                             VOCAB, &resume);
    if (rc != 0 || resume != ref.g->n_ops) {
      CHECK(0, "reference token %u: 0x%x resume %u", t, (unsigned)rc, resume);
      break;
    }
    ref_id[t] = ref.g->lm_id;
  }
  close_session(&ref);
}

static void check_bit_identical(const uint32_t *words, uint32_t n) {
  static float ref_logits[TOKENS][VOCAB], logits[VOCAB], x[HID];
  static uint32_t ref_id[TOKENS];
  const uint32_t moe_per_token = 2u;
  uint64_t *ref_log;
  uint8_t *page = new_page();
  session s;
  hexkl_token_stats st;
  uint32_t t, id, same_logits = 0, same_id = 0, same_moe = 0;
  uint32_t distinct = 0, seen[VOCAB] = {0};
  int rc;

  g_moe_cap = TOKENS * moe_per_token;
  g_moe_log = calloc(2u * g_moe_cap, sizeof(uint64_t));
  ref_log = calloc(2u * g_moe_cap, sizeof(uint64_t));

  g_moe_calls = 0;
  reference(words, n, ref_logits, ref_id);
  for (t = 0; t < TOKENS; ++t)
    distinct += ref_id[t] < VOCAB && seen[ref_id[t]]++ == 0u;
  /* not a constant model: the ids move with the input */
  CHECK(distinct >= 8u, "reference ids: %u distinct", distinct);
  CHECK(g_moe_calls == g_moe_cap, "reference MoE calls %u", g_moe_calls);
  memcpy(ref_log, g_moe_log, 2u * g_moe_cap * sizeof(uint64_t));

  open_session(&s, words, n);
  memset(&st, 0, sizeof(st));
  g_moe_calls = 0;
  for (t = 0; t < TOKENS; ++t) {
    emb_row(t, x);
    rc = hexkl_token_main(s.g, &s.env, page, t, t % s.g->max_seq, x, HID,
                          t % 2u ? logits : NULL, VOCAB, SPIN_US, &st, &id);
    if (rc != AEE_SUCCESS) {
      CHECK(0, "token %u: 0x%x", t, (unsigned)rc);
      break;
    }
    /* odd tokens hand a logits buffer, even ones read the graph's own */
    same_logits +=
      memcmp(t % 2u ? logits : s.g->logits, ref_logits[t], sizeof(logits)) == 0;
    same_id += id == ref_id[t];
  }
  for (t = 0; t < g_moe_cap && t < g_moe_calls; ++t)
    same_moe +=
      memcmp(&g_moe_log[2u * t], &ref_log[2u * t], 2u * sizeof(uint64_t)) == 0;
  CHECK(st.tokens == TOKENS, "tokens %u", st.tokens);
  CHECK(same_logits == TOKENS && same_id == TOKENS, "logits %u / ids %u of %u",
        same_logits, same_id, TOKENS);
  CHECK(g_moe_calls == g_moe_cap && same_moe == g_moe_cap,
        "MoE calls %u, %u equal of %u", g_moe_calls, same_moe, g_moe_cap);
  CHECK(st.timeouts + st.stale + st.misses == 0u,
        "timeouts %u stale %u misses %u", st.timeouts, st.stale, st.misses);
  if (g_fail == 0)
    printf("TOKEN DRIVER BIT-IDENTICAL: tokens %u/%u (%u distinct ids) "
           "logits bit_identical=1 "
           "moe_calls %u/%u in+out bit_identical=1 timeouts=0 stale=0 "
           "(hd64 C A C, one session, vs the one-session forward)\n",
           same_id, TOKENS, distinct, same_moe, g_moe_cap);
  close_session(&s);
  free(page);
  free(ref_log);
  free(g_moe_log);
  g_moe_log = NULL;
  g_moe_cap = 0;
}

/* ---- [plan 201 S1] the pool: the miss rounds against an owner thread --- */
#define POOL_OPS 2u /* the hd64 list's MoE ops */
#define POOL_E 4u   /* experts a layer; the pool holds half of them */

/* The owner (the ARM's side, here a pthread): per MoE op, which expert
   each of its two slots holds and a recency stamp, and the op indices. */
typedef struct {
  uint8_t *page;
  uint32_t op[POOL_OPS];
  uint32_t holder[POOL_OPS][2]; /* expert in slot s */
  uint32_t used[POOL_OPS][2];   /* recency */
  uint32_t clock, served, loads, bad;
  volatile int stop;
} pool_owner;

static uint32_t pool_handle(uint32_t m, uint32_t slot, int dn) {
  return 10u * m + (dn ? 5u : 0u) + slot; /* the pair slot s owns */
}

/* The rebind stub: the pair stays (its bytes were rewritten in place). */
static int host_rebind(void *ctx, uint32_t old_gu, uint32_t old_dn, uint32_t K,
                       uint32_t inter, uint32_t N_out, uint32_t arena,
                       uint32_t off_gu, uint32_t off_dn, const int8_t *pal_gu,
                       const int8_t *pal_dn, uint32_t *h_gu, uint32_t *h_dn) {
  (void)ctx, (void)K, (void)inter, (void)N_out, (void)arena, (void)off_gu;
  (void)off_dn, (void)pal_gu, (void)pal_dn;
  if (old_gu == HTP_GRAPH_NO_HANDLE)
    return AEE_EBADPARM;
  *h_gu = old_gu;
  *h_dn = old_dn;
  return AEE_SUCCESS;
}

/* Every MoE op's EXPERTS table: experts [0, held) in slots 0.., the rest
   absent; the ops' indices into o. */
static void set_pool(session *s, pool_owner *o, uint32_t held) {
  uint32_t m, t, e;
  s->env.rebind = host_rebind;
  for (m = 0; m < POOL_OPS; ++m) {
    float tab[2u * POOL_E];
    uint32_t h;
    o->op[m] = HTP_GRAPH_NO_OP;
    for (t = 0, e = 0; t < s->g->n_ops; ++t)
      if (s->g->ops[t].kind == HTP_OP_MOE && e++ == m)
        o->op[m] = t;
    for (e = 0; e < POOL_E; ++e) {
      h = e < held ? pool_handle(m, e, 0) : HTP_GRAPH_NO_HANDLE;
      memcpy(&tab[e], &h, sizeof(h));
      h = e < held ? pool_handle(m, e, 1) : HTP_GRAPH_NO_HANDLE;
      memcpy(&tab[POOL_E + e], &h, sizeof(h));
      if (e < held)
        o->holder[m][e] = e;
    }
    CHECK(hexkl_graph_set_param(s->g, o->op[m], HTP_GRAPH_PARAM_EXPERTS, tab,
                                2u * POOL_E) == 0,
          "pool table %u", m);
  }
}

static void pool_serve(pool_owner *o, const htp_miss_req *q) {
  htp_miss_ans *a = (htp_miss_ans *)(o->page + HEXKL_MBOX_MISS_ANS);
  uint32_t m, i, j, s;
  for (m = 0; m < POOL_OPS && o->op[m] != q->op; ++m) {
  }
  memset(a, 0, sizeof(*a));
  if (m == POOL_OPS) {
    a->rc = AEE_EBADITEM;
    ++o->bad;
  } else {
    /* the routed stay (they are touched first), the least recent other
       goes: the ARM's ExpertLru rule */
    for (i = 0; i < q->n_routed; ++i)
      for (s = 0; s < 2u; ++s)
        if (o->holder[m][s] == q->routed[i])
          o->used[m][s] = ++o->clock;
    for (i = 0; i < q->n_miss; ++i) {
      const uint32_t e = q->miss[i];
      uint32_t v = 2u, routed[2] = {0u, 0u};
      for (s = 0; s < 2u; ++s) /* never a routed expert */
        for (j = 0; j < q->n_routed; ++j)
          routed[s] |= o->holder[m][s] == q->routed[j];
      for (s = 0; s < 2u; ++s)
        if (!routed[s] && (v == 2u || o->used[m][s] < o->used[m][v]))
          v = s;
      if (v == 2u) { /* a pool smaller than the routed set */
        a->rc = AEE_ENOMEMORY;
        ++o->bad;
        break;
      }
      a->evict[a->n_evict][0] = q->op;
      a->evict[a->n_evict++][1] = o->holder[m][v];
      /* "pread": the slot's bytes become expert e's */
      g_wh[pool_handle(m, v, 0)] = g_wh[pool_handle(m, v, 1)] = 10u * m + e;
      a->load[a->n_load].e = e;
      a->load[a->n_load].old_gu = pool_handle(m, v, 0);
      a->load[a->n_load].old_dn = pool_handle(m, v, 1);
      a->load[a->n_load].h_gu = a->load[a->n_load].h_dn = HTP_GRAPH_NO_HANDLE;
      ++a->n_load;
      o->holder[m][v] = e;
      o->used[m][v] = ++o->clock;
      ++o->loads;
    }
  }
  a->seq2 = q->seq;
  __atomic_store_n(&a->seq, q->seq, __ATOMIC_RELEASE);
  ++o->served;
}

static void *owner_thread(void *arg) {
  pool_owner *o = (pool_owner *)arg;
  const htp_miss_req *q = (const htp_miss_req *)(o->page + HEXKL_MBOX_MISS_REQ);
  uint32_t last = 0;
  while (!o->stop) {
    const uint32_t seq = __atomic_load_n(&q->seq, __ATOMIC_ACQUIRE);
    if (seq != last && seq != 0u && q->seq2 == seq) {
      pool_serve(o, q);
      last = seq;
    } else {
      sched_yield();
    }
  }
  return NULL;
}

static void check_pool(const uint32_t *words, uint32_t n) {
  static float ref_logits[TOKENS][VOCAB], x[HID];
  static uint32_t ref_id[TOKENS];
  session s;
  pool_owner o;
  pthread_t tho;
  hexkl_token_stats st;
  uint32_t t, id, same_logits = 0, same_id = 0;
  int rc;

  reference(words, n, ref_logits, ref_id);
  open_session(&s, words, n);
  memset(&o, 0, sizeof(o));
  o.page = new_page();
  set_pool(&s, &o, 2u); /* experts 0 and 1 of each layer */
  memset(&st, 0, sizeof(st));
  pthread_create(&tho, NULL, owner_thread, &o);
  for (t = 0; t < TOKENS; ++t) {
    emb_row(t, x);
    rc = hexkl_token_main(s.g, &s.env, o.page, t, t % s.g->max_seq, x, HID,
                          NULL, VOCAB, SPIN_US, &st, &id);
    if (rc != AEE_SUCCESS) {
      CHECK(0, "pool token %u: 0x%x", t, (unsigned)rc);
      break;
    }
    same_logits +=
      memcmp(s.g->logits, ref_logits[t], VOCAB * sizeof(float)) == 0;
    same_id += id == ref_id[t];
  }
  o.stop = 1;
  pthread_join(tho, NULL);
  CHECK(st.tokens == TOKENS, "pool tokens %u", st.tokens);
  CHECK(same_logits == TOKENS && same_id == TOKENS,
        "pool logits %u / ids %u of %u", same_logits, same_id, TOKENS);
  CHECK(st.misses == o.loads && o.loads > TOKENS / 4u && o.bad == 0u,
        "pool misses %u owner %u bad %u", st.misses, o.loads, o.bad);
  CHECK(st.timeouts + st.stale == 0u, "pool timeouts %u stale %u", st.timeouts,
        st.stale);
  CHECK(s.g->route_log_n == POOL_OPS * 3u, "route log %u bytes",
        s.g->route_log_n);
  if (g_fail == 0)
    printf("TOKEN POOL BIT-IDENTICAL: tokens %u/%u logits bit_identical=1, a "
           "pool of %u of %u experts a layer, misses=%u (%.2f/token) in %u "
           "rounds, timeouts=0 stale=0 (the miss rounds against an owner "
           "pthread; vs the one-session forward)\n",
           same_id, TOKENS, 2u, POOL_E, o.loads, (double)o.loads / TOKENS,
           o.served);
  close_session(&s);
  free(o.page);
}

/* ---- the miss round's failure paths ------------------------------------ */
static void check_failures(const uint32_t *words, uint32_t n) {
  static float x[HID];
  session s;
  pool_owner o;
  hexkl_token_stats st;
  htp_miss_ans *a;
  uint64_t t0, dt;
  uint32_t id, seq;
  int rc;

  /* no expert held: the token's first MoE op misses at once */
  open_session(&s, words, n);
  memset(&o, 0, sizeof(o));
  o.page = new_page();
  a = (htp_miss_ans *)(o.page + HEXKL_MBOX_MISS_ANS);
  set_pool(&s, &o, 0u);
  emb_row(0, x);

  /* (1) no owner: the request is never answered */
  memset(&st, 0, sizeof(st));
  t0 = now_us();
  rc = hexkl_token_main(s.g, &s.env, o.page, 1u, 0u, x, HID, NULL, 0u, 0u, &st,
                        &id);
  dt = now_us() - t0;
  CHECK(rc == AEE_EEXPIRED && st.timeouts == 1u &&
          dt >= HEXKL_TOKEN_TIMEOUT_US && dt < 3u * HEXKL_TOKEN_TIMEOUT_US,
        "no owner: 0x%x timeouts %u after %llu us", (unsigned)rc, st.timeouts,
        (unsigned long long)dt);
  printf(
    "  no owner                -> AEE_EEXPIRED after %.2f s, timeouts=%u\n",
    dt / 1e6, st.timeouts);

  /* (2) the answer word carries this round's seq over an earlier answer
     (its body's clean did not land): refused */
  seq = hexkl_token_seq(2u, 0u);
  memset(a, 0, sizeof(*a));
  a->seq2 = seq - 1u;
  a->seq = seq;
  memset(&st, 0, sizeof(st));
  rc = hexkl_token_main(s.g, &s.env, o.page, 2u, 0u, x, HID, NULL, 0u, 0u, &st,
                        &id);
  CHECK(rc == HEXKL_TOKEN_E_STALE && st.stale == 1u && st.timeouts == 0u,
        "stale answer: 0x%x stale %u", (unsigned)rc, st.stale);
  printf("  a stale answer          -> 0x%x (stale=%u)\n", (unsigned)rc,
         st.stale);

  /* (3) the owner fails the round: its code, not a timeout */
  seq = hexkl_token_seq(3u, 0u);
  memset(a, 0, sizeof(*a));
  a->rc = AEE_ENOMEMORY;
  a->seq2 = seq;
  a->seq = seq;
  memset(&st, 0, sizeof(st));
  t0 = now_us();
  rc = hexkl_token_main(s.g, &s.env, o.page, 3u, 0u, x, HID, NULL, 0u, 0u, &st,
                        &id);
  dt = now_us() - t0;
  CHECK(rc == AEE_ENOMEMORY && st.timeouts == 0u &&
          dt < HEXKL_TOKEN_TIMEOUT_US / 2u,
        "owner failure: 0x%x after %llu us", (unsigned)rc,
        (unsigned long long)dt);
  printf("  the owner fails         -> 0x%x after %llu us\n", (unsigned)rc,
         (unsigned long long)dt);
  if (g_fail == 0)
    printf("TOKEN DRIVER FAILURE PATHS OK: no owner -> AEE_EEXPIRED, stale "
           "answer refused, the owner's code reaches the token\n");
  close_session(&s);
  free(o.page);
}

int main(void) {
  static uint32_t words[MAXW];
  const uint32_t n = build_words(words);
  check_bit_identical(words, n);
  check_pool(words, n);
  check_failures(words, n);
  if (g_fail) {
    printf("TOKEN CHECKS FAILED (%d)\n", g_fail);
    return 1;
  }
  printf("TOKEN CHECKS PASS\n");
  return 0;
}
