// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_token.c
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2, #211] The one-PD token driver over the mailbox
 *         page (hexkl_token.h)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plain C over volatile words, so test/htp/host/token_host_check.c runs
 * it against an owner pthread as it is: off the DSP the cache calls are
 * sequentially consistent fences and the poll sleep is nanosleep. The
 * waits, the posts and the stale checks are the #178 probe's
 * (nntr_hvx_mailbox.c), made per miss round of a token.
 */

#if !defined(__hexagon__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L /* nanosleep, clock_gettime */
#endif

#include "hexkl_token.h"

#include <string.h>

#include "hvx_worker_pool.h"

#include <AEEStdErr.h>

#if defined(__hexagon__)
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_memory.h>
#else
#include <time.h>
#endif

static uint64_t tk_now_us(void) {
#if defined(__hexagon__)
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
#endif
}

/** @brief [#267 L0] The core's pcycles; 0 on the host. */
static uint64_t tk_pcyc(void) {
#if defined(__hexagon__)
  return HAP_perf_get_pcycles();
#else
  return 0u;
#endif
}

/** @brief Our stores reach DDR, where the ARM reads them. */
static void tk_clean(void *p, uint32_t n) {
#if defined(__hexagon__)
  qurt_mem_cache_clean((qurt_addr_t)p, (qurt_size_t)n, QURT_MEM_CACHE_FLUSH,
                       QURT_MEM_DCACHE);
#else
  (void)p;
  (void)n;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/** @brief The ARM's lines are re-read from DDR. Flush-invalidate, never a
 *  bare invalidate: a bare invalidate could drop a not-yet-cleaned store
 *  of ours on a shared line (the probe's rule). */
static void tk_refresh(void *p, uint32_t n) {
#if defined(__hexagon__)
  qurt_mem_cache_clean((qurt_addr_t)p, (qurt_size_t)n,
                       QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
#else
  (void)p;
  (void)n;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static inline void tk_pause(void) {
#if defined(__hexagon__)
  asm volatile(" pause(#255)\n");
#endif
}

static void tk_sleep(void) {
#if defined(__hexagon__)
  qurt_timer_sleep(HEXKL_TOKEN_POLL_US);
#else
  const struct timespec t = {0, (long)HEXKL_TOKEN_POLL_US * 1000L};
  nanosleep(&t, NULL);
#endif
}

/** @brief Adds the op pcycles of the ops [s, e) forward just ran. */
static void tk_pcycles(const hexkl_graph *g, uint32_t s, uint32_t e,
                       hexkl_token_stats *st) {
  for (; s < e; ++s) {
    st->pcycles += g->op_pcycles[s];
    if (g->ops[s].kind < HTP_OP_KIND_N) {
      st->kind_pcycles[g->ops[s].kind] += g->op_pcycles[s];
      st->kind_qt[g->ops[s].kind] += g->op_qt[s];
    }
  }
}

/** @brief The first op at or after @a s that is not resident. */
static uint32_t tk_stretch_end(const hexkl_graph *g, uint32_t s) {
  while (s < g->n_ops && g->ops[s].resident) {
    ++s;
  }
  return s;
}

/* ---- [plan 201 S1] the expert pool's miss round (P-A) ------------------ */

/** @brief One token's miss rounds: the env its rebinds go through. */
typedef struct {
  const hexkl_graph_env *env;
  uint8_t *mbox;
  uint32_t tok, k, spin_us;
  hexkl_token_stats *st;
  uint32_t hk; /**< [#282 B] hints posted this token */
} tk_miss;

/** @brief [#282 B] graph_predict's guess to the page's hint slot: body,
 *  then seq. Never waits; a full token's worth is 30 of 255. */
static void tk_hint_post(void *ctx, uint32_t op, const uint32_t *ids,
                         uint32_t n) {
  tk_miss *m = (tk_miss *)ctx;
  htp_pred_hint *h = (htp_pred_hint *)(m->mbox + HTP_MBOX_HINT);
  const uint32_t seq = hexkl_token_seq(m->tok, m->hk);
  if (m->hk >= HEXKL_TOKEN_MAX_ROUNDS || n > HTP_MBOX_MISS_MAX) {
    return;
  }
  ++m->hk;
  h->op = op;
  h->n = n;
  memcpy(h->ids, ids, n * sizeof(uint32_t));
  h->seq2 = seq;
  tk_clean(h, sizeof(*h));
  *(volatile uint32_t *)&h->seq = seq;
  tk_clean(h, 4u);
}

static int tk_miss_post(void *ctx, uint32_t op, const uint32_t *routed,
                        uint32_t n_routed, const uint32_t *miss,
                        uint32_t n_miss) {
  tk_miss *m = (tk_miss *)ctx;
  htp_miss_req *q = (htp_miss_req *)(m->mbox + HEXKL_MBOX_MISS_REQ);
  const uint32_t seq = hexkl_token_seq(m->tok, m->k);
  if (m->k >= HEXKL_TOKEN_MAX_ROUNDS || n_routed > HEXKL_GRAPH_MISS_MAX ||
      n_miss > n_routed) {
    return AEE_EINVALIDFORMAT;
  }
  q->op = op;
  q->n_routed = n_routed;
  q->n_miss = n_miss;
  memcpy(q->routed, routed, n_routed * sizeof(uint32_t));
  memcpy(q->miss, miss, n_miss * sizeof(uint32_t));
  q->seq2 = seq;
  tk_clean(q, sizeof(*q));
  *(volatile uint32_t *)&q->seq = seq;
  tk_clean(q, 4u);
  return AEE_SUCCESS;
}

static int tk_miss_wait(void *ctx, struct hexkl_graph_s *g, uint32_t op) {
  tk_miss *m = (tk_miss *)ctx;
  htp_miss_ans *a = (htp_miss_ans *)(m->mbox + HEXKL_MBOX_MISS_ANS);
  volatile uint32_t *word = (volatile uint32_t *)&a->seq;
  const uint32_t seq = hexkl_token_seq(m->tok, m->k++);
  const htp_graph_op *o = &g->ops[op];
  const uint64_t t0 = tk_now_us();
  const uint64_t pc0 = tk_pcyc();
  uint32_t i;
  int rc = AEE_SUCCESS;
  for (;;) {
    uint64_t dt;
    tk_refresh((void *)word, 4u);
    if (*word == seq) {
      break;
    }
    dt = tk_now_us() - t0;
    if (dt >= (uint64_t)m->spin_us + HEXKL_TOKEN_TIMEOUT_US) {
      m->st->miss_us += (uint32_t)dt;
      m->st->miss_pcyc += tk_pcyc() - pc0;
      ++m->st->timeouts;
      return AEE_EEXPIRED;
    }
    if (dt >= m->spin_us) {
      tk_sleep();
    } else {
      tk_pause();
    }
  }
  m->st->miss_us += (uint32_t)(tk_now_us() - t0);
  m->st->miss_pcyc += tk_pcyc() - pc0;
  tk_refresh(a, sizeof(*a));
  if (a->seq2 != seq || a->n_evict > HEXKL_GRAPH_MISS_MAX ||
      a->n_load > HEXKL_GRAPH_MISS_MAX) {
    ++m->st->stale;
    return HEXKL_TOKEN_E_STALE;
  }
  if (a->rc != AEE_SUCCESS) {
    return a->rc;
  }
  /* evictions first: a load may take an evicted expert's pair */
  for (i = 0; rc == AEE_SUCCESS && i < a->n_evict; ++i) {
    rc = hexkl_graph_pool_set(g, a->evict[i][0], a->evict[i][1],
                              HTP_GRAPH_NO_HANDLE, HTP_GRAPH_NO_HANDLE);
  }
  for (i = 0; rc == AEE_SUCCESS && i < a->n_load; ++i) {
    htp_miss_load *l = &a->load[i];
    uint32_t hg = HTP_GRAPH_NO_HANDLE, hd = HTP_GRAPH_NO_HANDLE;
    rc = m->env->rebind == NULL
           ? AEE_EBADSTATE
           : m->env->rebind(m->env->rebind_ctx, l->old_gu, l->old_dn, o->K,
                            o->N, o->N_out, l->arena, l->off_gu, l->off_dn,
                            l->bits == 2u ? l->pal_gu : NULL,
                            l->bits == 2u ? l->pal_dn : NULL, &hg, &hd);
    if (rc == AEE_SUCCESS) {
      rc = hexkl_graph_pool_set(g, op, l->e, hg, hd);
    }
    l->h_gu = hg; /* the owner files these */
    l->h_dn = hd;
    m->st->misses += rc == AEE_SUCCESS;
  }
  tk_clean(a, sizeof(*a));
  return rc;
}

int hexkl_token_main(hexkl_graph *g, const hexkl_graph_env *env, uint8_t *mbox,
                     uint32_t tok, uint32_t pos, const float *act_in,
                     uint32_t act_len, float *logits, uint32_t logits_len,
                     uint32_t spin_us, hexkl_token_stats *st, uint32_t *id) {
  /* [plan 201 S1] the MOE ops' miss rounds go through the page */
  tk_miss miss = {env, mbox, tok, 0u, spin_us, st, 0u};
  hexkl_graph_env menv = *env;
  const htp_graph_op *last;
  uint32_t resume;
  int rc;
  menv.miss.post = tk_miss_post;
  menv.miss.wait = tk_miss_wait;
  menv.miss.ctx = &miss;
  menv.miss.hint = tk_hint_post; /* [#282 B] posted only under predict */
  g->route_log_n = 0u;
  g->pred_log_n = 0u; /* [#266 S2] */
  g->moe_calls = g->moe_calls_1x = 0u;
  *id = 0u;
  if (g->n_ops == 0u || tk_stretch_end(g, 0u) != g->n_ops) {
    return AEE_EBADSTATE;
  }
  last = &g->ops[g->n_ops - 1u];
  if (logits == NULL) {
    if (last->kind != HTP_OP_LM_HEAD) {
      return AEE_EINVALIDFORMAT;
    }
    logits = g->logits; /* forward skips the copy onto itself */
    logits_len = last->N;
  }
  rc = hexkl_graph_forward(g, &menv, 0u, g->n_ops, pos, NULL, act_in, act_len,
                           logits, logits_len, &resume);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  hvx_worker_pool_park(env->pool); /* the token is done */
  tk_pcycles(g, 0u, g->n_ops, st);
  *id = last->kind == HTP_OP_LM_HEAD ? g->lm_id : 0u;
  ++st->tokens;
  return AEE_SUCCESS;
}
