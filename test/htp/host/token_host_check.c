// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   token_host_check.c
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2] Host check of the two-session token driver
 *         (hmx/hexkl_token.c): S1 and S2 on two pthreads over one
 *         description split by mask, against the one-session run
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The hd64 fixture's list (3 layers C A C, layer 0 dense, two MoE layers,
 * vocab 64 in two lm_head slices) with every kind resident. Reference:
 * ONE graph with the full mask, forward from op 0 per token (the E1
 * one-session run). Under test: TWO graphs of the same words with the
 * masks HTP_GRAPH_KINDS_S1 / _S2, each with its own slots, pool, KV cache
 * and conv state, S1 served on a pthread (hexkl_token_serve) and S2 on
 * the main thread (hexkl_token_main), every MoE input and output crossing
 * a malloc'd page. Same parameters and weights in both, TOKENS tokens at
 * pos = tok % max_seq (the KV cache rewinds at 0, the conv state runs on).
 *
 * Kernels: the REAL small ops, router, m=1 attention, residual add and
 * CPU-exact Q4_0 FC (hvx_q4_gemv_f32.c) on hvx_emu/, as graph_host_check;
 * the MoE is a stand-in that mixes its input into its output under the
 * routing and records a hash of every call's input and output (the dump's
 * analog). Gated: every token's logits memcmp-equal and ids equal, every
 * MoE call's in / out hash equal, hops = 2 x rounds x tokens on each
 * side, timeouts 0, stale 0. Then the failure paths: a lost post on
 * either side (AEE_EEXPIRED after the 1 s window, not a hang), a stale
 * header and a stale trailer (HEXKL_TOKEN_E_STALE, S1 posting its code
 * back), and S1's forward failing (S2 returns S1's code well inside the
 * window). What this cannot show: the DSP's caches (the clean /
 * flush-invalidate calls are fences here), hop time, two PDs.
 */
#include "hexkl_graph.h"
#include "hexkl_token.h"
#include "htp_graph_desc.h"
#include <AEEStdErr.h>
#include <pthread.h>
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

int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch, uint32_t flags) {
  uint32_t e, c, r = 0;
  (void)tbl, (void)vtcm_base, (void)vtcm_size, (void)config_off, (void)pool;
  (void)scratch, (void)flags, (void)inter, (void)h_down;
  memset(out_f32, 0, (size_t)M * N_out * sizeof(float));
  for (e = 0; e < n_experts; ++e) {
    uint32_t i;
    for (i = 0; i < row_count[e]; ++i, ++r) {
      const float *x = act_f32 + (size_t)row_index[r] * K;
      float *y = out_f32 + (size_t)row_index[r] * N_out;
      for (c = 0; c < N_out; ++c)
        y[c] += row_weight[r] *
                (x[(c + 7u * e) % K] * (float)(h_gate_up[e] + 1u) * 0.125f +
                 0.01f * (float)c);
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
      for (e = 0; e < op->n_experts; ++e) {
        op->h_gu[e] = 10u * m + e;
        op->h_dn[e] = 10u * m + 5u + e;
        g_tbl.slots[op->h_gu[e]].in_use = 1;
        g_tbl.slots[op->h_gu[e]].K = op->K;
        g_tbl.slots[op->h_gu[e]].N = 2u * op->N;
        g_tbl.slots[op->h_dn[e]].in_use = 1;
        g_tbl.slots[op->h_dn[e]].K = op->N;
        g_tbl.slots[op->h_dn[e]].N = op->N_out;
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

/* Every parameter of every op, the same bytes on every graph (a
   non-resident op's are never read). skip_router leaves the router bias
   unbound (S1's forward then fails). */
static void bind_params(hexkl_graph *g, int skip_router) {
  static float buf[64u * 64u * 4u];
  uint32_t i, s = 0x51u;
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
      if (!skip_router)
        rc |= hexkl_graph_set_param(g, i, HTP_GRAPH_PARAM_ROUTER_BIAS, buf,
                                    op->n_experts);
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

static void open_session(session *s, const uint32_t *words, uint32_t n,
                         uint32_t mask, int skip_router) {
  static uint32_t w[MAXW];
  int err = 0;
  memcpy(w, words, n * sizeof(uint32_t));
  htp_graph_set_resident(w, mask);
  memset(s, 0, sizeof(*s));
  CHECK(hexkl_graph_init(w, n, &g_tbl, g_qs, Q_SLOTS, &s->g) == 0,
        "init mask 0x%x", mask);
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
  bind_params(s->g, skip_router);
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

/* ---- the S1 thread --------------------------------------------------- */
typedef struct {
  session *s;
  uint8_t *page;
  uint32_t first, n;
  hexkl_token_stats st;
  int rc;
} serve_args;

static void *serve_thread(void *arg) {
  serve_args *a = (serve_args *)arg;
  uint32_t t;
  for (t = a->first; t < a->first + a->n; ++t) {
    a->rc = hexkl_token_serve(a->s->g, &a->s->env, a->page, t,
                              t % a->s->g->max_seq, SPIN_US, &a->st);
    if (a->rc != AEE_SUCCESS)
      break;
  }
  return NULL;
}

static uint8_t *new_page(void) {
  uint8_t *p = aligned_alloc(4096, 20480);
  memset(p, 0, 20480);
  return p;
}

static void check_bit_identical(const uint32_t *words, uint32_t n) {
  static float ref_logits[TOKENS][VOCAB], logits[VOCAB], x[HID];
  static uint32_t ref_id[TOKENS];
  const uint32_t moe_per_token = 2u;
  uint64_t *ref_log;
  session ref, s1, s2;
  serve_args sa;
  pthread_t th;
  hexkl_token_stats st;
  uint32_t t, resume, id, rounds, same_logits = 0, same_id = 0, same_moe = 0;
  uint32_t distinct = 0, seen[VOCAB] = {0};
  int rc;

  g_moe_cap = TOKENS * moe_per_token;
  g_moe_log = calloc(2u * g_moe_cap, sizeof(uint64_t));
  ref_log = calloc(2u * g_moe_cap, sizeof(uint64_t));

  /* the one-session all-resident run */
  open_session(&ref, words, n, HTP_GRAPH_KINDS_ALL, 0);
  g_moe_calls = 0;
  for (t = 0; t < TOKENS; ++t) {
    emb_row(t, x);
    rc = hexkl_graph_forward(ref.g, &ref.env, 0u, HTP_GRAPH_MAX_OPS,
                             t % ref.g->max_seq, NULL, x, HID, ref_logits[t],
                             VOCAB, &resume);
    if (rc != 0 || resume != ref.g->n_ops) {
      CHECK(0, "reference token %u: 0x%x resume %u", t, (unsigned)rc, resume);
      return;
    }
    ref_id[t] = ref.g->lm_id;
    distinct += ref_id[t] < VOCAB && seen[ref_id[t]]++ == 0u;
  }
  /* not a constant model: the ids move with the input */
  CHECK(distinct >= 8u, "reference ids: %u distinct", distinct);
  CHECK(g_moe_calls == g_moe_cap, "reference MoE calls %u", g_moe_calls);
  memcpy(ref_log, g_moe_log, 2u * g_moe_cap * sizeof(uint64_t));
  close_session(&ref);

  /* the two sessions */
  open_session(&s1, words, n, HTP_GRAPH_KINDS_S1, 0);
  open_session(&s2, words, n, HTP_GRAPH_KINDS_S2, 0);
  rounds = hexkl_token_rounds(s1.g);
  CHECK(rounds == moe_per_token && hexkl_token_rounds(s2.g) == rounds + 1u,
        "rounds S1 %u S2 %u", rounds, hexkl_token_rounds(s2.g));
  memset(&sa, 0, sizeof(sa));
  sa.s = &s1;
  sa.page = new_page();
  sa.first = 0;
  sa.n = TOKENS;
  memset(&st, 0, sizeof(st));
  g_moe_calls = 0;
  pthread_create(&th, NULL, serve_thread, &sa);
  for (t = 0; t < TOKENS; ++t) {
    emb_row(t, x);
    rc = hexkl_token_main(s2.g, &s2.env, sa.page, t, t % s2.g->max_seq, x, HID,
                          t % 2u ? logits : NULL, VOCAB, SPIN_US, &st, &id);
    if (rc != AEE_SUCCESS) {
      CHECK(0, "S2 token %u: 0x%x", t, (unsigned)rc);
      break;
    }
    /* odd tokens hand a logits buffer, even ones read the graph's own */
    same_logits += memcmp(t % 2u ? logits : s2.g->logits, ref_logits[t],
                          sizeof(logits)) == 0;
    same_id += id == ref_id[t];
  }
  pthread_join(th, NULL);
  for (t = 0; t < g_moe_cap && t < g_moe_calls; ++t)
    same_moe +=
      memcmp(&g_moe_log[2u * t], &ref_log[2u * t], 2u * sizeof(uint64_t)) == 0;
  CHECK(sa.rc == 0 && sa.st.tokens == TOKENS && st.tokens == TOKENS,
        "tokens S1 %u (rc 0x%x) S2 %u", sa.st.tokens, (unsigned)sa.rc,
        st.tokens);
  CHECK(same_logits == TOKENS && same_id == TOKENS, "logits %u / ids %u of %u",
        same_logits, same_id, TOKENS);
  CHECK(g_moe_calls == g_moe_cap && same_moe == g_moe_cap,
        "MoE calls %u, %u equal of %u", g_moe_calls, same_moe, g_moe_cap);
  /* [#194 L0] the hop latency is a part of the wait, and not all of it */
  CHECK(st.hop_us <= st.wait_us && sa.st.hop_us <= sa.st.wait_us &&
          st.hop_us > 0u && sa.st.hop_us > 0u,
        "hop_us S2 %u of wait %u, S1 %u of wait %u", st.hop_us, st.wait_us,
        sa.st.hop_us, sa.st.wait_us);
  CHECK(st.hops == 2u * rounds * TOKENS && sa.st.hops == st.hops &&
          st.timeouts + sa.st.timeouts + st.stale + sa.st.stale == 0u,
        "hops S2 %u S1 %u timeouts %u %u stale %u %u", st.hops, sa.st.hops,
        st.timeouts, sa.st.timeouts, st.stale, sa.st.stale);
  if (g_fail == 0)
    printf("TOKEN DRIVER BIT-IDENTICAL: tokens %u/%u (%u distinct ids) "
           "logits bit_identical=1 "
           "moe_calls %u/%u in+out bit_identical=1 hops=%u (%u x tokens) "
           "timeouts=0 stale=0 hop_us<=wait_us (hd64 C A C, S1 = "
           "ROUTER_TOPK|MOE on a pthread, S2 = the rest; vs the one-session "
           "all-resident run)\n",
           same_id, TOKENS, distinct, same_moe, g_moe_cap, st.hops,
           2u * rounds);
  close_session(&s1);
  close_session(&s2);
  free(sa.page);
  free(ref_log);
  free(g_moe_log);
  g_moe_log = NULL;
  g_moe_cap = 0;
}

/* ---- the failure paths ----------------------------------------------- */
static void check_failures(const uint32_t *words, uint32_t n) {
  static float x[HID];
  session s1, s2, s1bad;
  hexkl_token_stats st;
  serve_args sa;
  pthread_t th;
  uint8_t *page = new_page();
  hexkl_mbox_hdr *h;
  uint64_t t0, dt;
  uint32_t id, seq, r1;
  int rc;

  open_session(&s1, words, n, HTP_GRAPH_KINDS_S1, 0);
  open_session(&s2, words, n, HTP_GRAPH_KINDS_S2, 0);
  for (r1 = 0; !s1.g->ops[r1].resident; ++r1) {
  }
  emb_row(0, x);

  /* (1) S2 alone: its first post is never answered */
  memset(&st, 0, sizeof(st));
  t0 = now_us();
  rc = hexkl_token_main(s2.g, &s2.env, page, 1u, 0u, x, HID, NULL, 0u, 0u, &st,
                        &id);
  dt = now_us() - t0;
  CHECK(rc == AEE_EEXPIRED && st.timeouts == 1u && st.hops == 1u &&
          dt >= HEXKL_TOKEN_TIMEOUT_US && dt < 3u * HEXKL_TOKEN_TIMEOUT_US,
        "S2 alone: 0x%x timeouts %u after %llu us", (unsigned)rc, st.timeouts,
        (unsigned long long)dt);
  printf("  S2 with no S1          -> AEE_EEXPIRED after %.2f s, timeouts=%u\n",
         dt / 1e6, st.timeouts);

  /* (2) S1 alone: no ping comes (the page still holds token 1's round 0,
     not token 2's) */
  memset(&st, 0, sizeof(st));
  t0 = now_us();
  rc = hexkl_token_serve(s1.g, &s1.env, page, 2u, 0u, 0u, &st);
  dt = now_us() - t0;
  CHECK(rc == AEE_EEXPIRED && st.timeouts == 1u && st.hops == 0u &&
          dt >= HEXKL_TOKEN_TIMEOUT_US && dt < 3u * HEXKL_TOKEN_TIMEOUT_US,
        "S1 alone: 0x%x timeouts %u after %llu us", (unsigned)rc, st.timeouts,
        (unsigned long long)dt);
  printf("  S1 with no S2          -> AEE_EEXPIRED after %.2f s, timeouts=%u\n",
         dt / 1e6, st.timeouts);

  /* (3) S1 reads a ping whose trailer is an earlier round's (the row's
     clean did not land): refused, and S1 posts the code back */
  seq = hexkl_token_seq(3u, 0u);
  memset(page, 0, 20480);
  h = (hexkl_mbox_hdr *)(page + HEXKL_MBOX_S2_SLOT);
  h->seq = seq;
  h->op = r1;
  h->n = HID;
  h->rc = 0;
  *(uint32_t *)(page + HEXKL_MBOX_S2_SLOT + HEXKL_MBOX_LINE + HID * 4u) =
    seq - 1u;
  *(uint32_t *)(page + HEXKL_MBOX_PING) = seq;
  memset(&st, 0, sizeof(st));
  rc = hexkl_token_serve(s1.g, &s1.env, page, 3u, 0u, 0u, &st);
  h = (hexkl_mbox_hdr *)(page + HEXKL_MBOX_S1_SLOT);
  CHECK(rc == HEXKL_TOKEN_E_STALE && st.stale == 1u &&
          *(uint32_t *)(page + HEXKL_MBOX_PONG) == seq && h->seq == seq &&
          h->rc == HEXKL_TOKEN_E_STALE,
        "S1 stale trailer: 0x%x stale %u, posted rc 0x%x", (unsigned)rc,
        st.stale, (unsigned)h->rc);
  printf("  S1 reads a stale trailer -> 0x%x (stale=%u), code posted to S2\n",
         (unsigned)rc, st.stale);

  /* (4) S2 finds pong == seq over a header of the previous round */
  seq = hexkl_token_seq(4u, 0u);
  memset(page, 0, 20480);
  h = (hexkl_mbox_hdr *)(page + HEXKL_MBOX_S1_SLOT);
  h->seq = seq - 1u;
  *(uint32_t *)(page + HEXKL_MBOX_PONG) = seq;
  memset(&st, 0, sizeof(st));
  rc = hexkl_token_main(s2.g, &s2.env, page, 4u, 0u, x, HID, NULL, 0u, 0u, &st,
                        &id);
  CHECK(rc == HEXKL_TOKEN_E_STALE && st.stale == 1u && st.timeouts == 0u,
        "S2 stale header: 0x%x stale %u", (unsigned)rc, st.stale);
  printf("  S2 reads a stale header  -> 0x%x (stale=%u)\n", (unsigned)rc,
         st.stale);

  /* (5) S1's forward fails (no router bias): S2 returns S1's code at
     once, not after the window */
  open_session(&s1bad, words, n, HTP_GRAPH_KINDS_S1, 1);
  memset(page, 0, 20480);
  memset(&sa, 0, sizeof(sa));
  sa.s = &s1bad;
  sa.page = page;
  sa.first = 5u;
  sa.n = 1u;
  memset(&st, 0, sizeof(st));
  pthread_create(&th, NULL, serve_thread, &sa);
  t0 = now_us();
  rc = hexkl_token_main(s2.g, &s2.env, page, 5u, 0u, x, HID, NULL, 0u, SPIN_US,
                        &st, &id);
  dt = now_us() - t0;
  pthread_join(th, NULL);
  CHECK(rc == AEE_EBADSTATE && sa.rc == AEE_EBADSTATE && st.timeouts == 0u &&
          dt < HEXKL_TOKEN_TIMEOUT_US / 2u,
        "S1 forward failure: S2 0x%x S1 0x%x after %llu us", (unsigned)rc,
        (unsigned)sa.rc, (unsigned long long)dt);
  printf("  S1's forward fails     -> S2 returns 0x%x after %llu us\n",
         (unsigned)rc, (unsigned long long)dt);
  if (g_fail == 0)
    printf("TOKEN DRIVER FAILURE PATHS OK: lost post (both sides) -> "
           "AEE_EEXPIRED, stale header / trailer refused, a failed side's "
           "code reaches the other\n");
  close_session(&s1bad);
  close_session(&s1);
  close_session(&s2);
  free(page);
}

int main(void) {
  static uint32_t words[MAXW];
  const uint32_t n = build_words(words);
  check_bit_identical(words, n);
  check_failures(words, n);
  if (g_fail) {
    printf("TOKEN CHECKS FAILED (%d)\n", g_fail);
    return 1;
  }
  printf("TOKEN CHECKS PASS\n");
  return 0;
}
