// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_token.c
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2] The two-session token driver over the mailbox
 *         page (hexkl_token.h)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plain C over volatile words, so test/htp/host/token_host_check.c runs
 * both sides on two pthreads against this file as it is: off the DSP the
 * cache calls are sequentially consistent fences and the poll sleep is
 * nanosleep. The waits, the posts and the stale checks are the #178
 * probe's (nntr_hvx_mailbox.c on origin/htp/178-probe, 0.33 / 2.4 us a
 * hop on silicon), made per round of a token.
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

/** @brief Our stores reach DDR, where the other PD reads them. */
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

/** @brief The other side's lines are re-read from DDR. Flush-invalidate,
 *  never a bare invalidate: if the two PDs share a physical line, a bare
 *  invalidate could drop the writer's not-yet-cleaned store (the probe's
 *  rule). */
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

/** @brief The row's padded size: the trailer line follows it. */
static uint32_t tk_row_bytes(uint32_t n) {
  return (n * 4u + HEXKL_MBOX_LINE - 1u) & ~(HEXKL_MBOX_LINE - 1u);
}

static float *tk_row(uint8_t *slot) {
  return (float *)(slot + HEXKL_MBOX_LINE);
}

/** @brief Writes a slot's header and trailer around the row the caller
 *  already wrote, cleans the slot, then posts @a seq on @a word. */
static void tk_post(uint8_t *slot, volatile uint32_t *word, uint32_t seq,
                    uint32_t op, uint32_t n, int rc) {
  hexkl_mbox_hdr *h = (hexkl_mbox_hdr *)slot;
  const uint32_t row = rc == 0 ? tk_row_bytes(n) : 0u;
  h->seq = seq;
  h->op = op;
  h->n = rc == 0 ? n : 0u;
  h->rc = rc;
  h->post_us = (uint32_t)tk_now_us();
  *(uint32_t *)(slot + HEXKL_MBOX_LINE + row) = seq;
  tk_clean(slot, 2u * HEXKL_MBOX_LINE + row);
  *word = seq;
  tk_clean((void *)word, 4u);
}

/** @brief Waits for @a word == @a seq, then reads the slot's header and
 *  checks it and the trailer against @a seq.
 *  @return 0, AEE_EEXPIRED or HEXKL_TOKEN_E_STALE */
static int tk_take(uint8_t *slot, volatile uint32_t *word, uint32_t seq,
                   uint32_t spin_us, hexkl_token_stats *st,
                   hexkl_mbox_hdr *out) {
  const uint64_t t0 = tk_now_us();
  uint64_t seen;
  uint32_t row;
  for (;;) {
    tk_refresh((void *)word, 4u);
    if (*word == seq) {
      break;
    }
    const uint64_t dt = tk_now_us() - t0;
    if (dt >= (uint64_t)spin_us + HEXKL_TOKEN_TIMEOUT_US) {
      st->wait_us += (uint32_t)dt;
      ++st->timeouts;
      return AEE_EEXPIRED;
    }
    if (dt >= spin_us) {
      tk_sleep(); /* one read of the word per wake */
    } else {
      tk_pause();
    }
  }
  seen = tk_now_us();
  st->wait_us += (uint32_t)(seen - t0);
  tk_refresh(slot, HEXKL_MBOX_LINE);
  memcpy(out, slot, sizeof(*out));
  if (out->seq != seq || out->n * 4u > HEXKL_MBOX_ROW_MAX) {
    ++st->stale;
    return HEXKL_TOKEN_E_STALE;
  }
  /* [#194 L0] the hop's own latency: from the post, or from the start of
     this wait if the post came first, to the read that saw it (the poll
     sleep's granularity, the cache maintenance); never more than the wait */
  st->hop_us +=
    (uint32_t)seen -
    ((int32_t)(out->post_us - (uint32_t)t0) > 0 ? out->post_us : (uint32_t)t0);
  row = tk_row_bytes(out->n);
  tk_refresh(slot + HEXKL_MBOX_LINE, row + HEXKL_MBOX_LINE);
  if (*(const uint32_t *)(slot + HEXKL_MBOX_LINE + row) != seq) {
    ++st->stale;
    return HEXKL_TOKEN_E_STALE;
  }
  return AEE_SUCCESS;
}

/** @brief Adds the op pcycles of the stretch [s, e) forward just ran. */
static void tk_pcycles(const hexkl_graph *g, uint32_t s, uint32_t e,
                       hexkl_token_stats *st) {
  for (; s < e; ++s) {
    st->pcycles += g->op_pcycles[s];
    if (g->ops[s].kind < HTP_OP_KIND_N) {
      st->kind_pcycles[g->ops[s].kind] += g->op_pcycles[s];
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

uint32_t hexkl_token_rounds(const hexkl_graph *g) {
  uint32_t i, n = 0;
  for (i = 0; i < g->n_ops; ++i) {
    n += g->ops[i].resident && (i == 0u || !g->ops[i - 1u].resident);
  }
  return n;
}

/* ---- [plan 201 S1] the expert pool's miss round (P-A) ------------------ */

/** @brief One S1 token's miss rounds: the env its rebinds go through. */
typedef struct {
  const hexkl_graph_env *env;
  uint8_t *mbox;
  uint32_t tok, k, spin_us;
  hexkl_token_stats *st;
} tk_miss;

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
    rc =
      m->env->rebind == NULL
        ? AEE_EBADSTATE
        : m->env->rebind(m->env->rebind_ctx, l->old_gu, l->old_dn, o->K, o->N,
                         o->N_out, l->arena, l->off_gu, l->off_dn, &hg, &hd);
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
  uint8_t *const mine = mbox + HEXKL_MBOX_S2_SLOT;
  uint8_t *const theirs = mbox + HEXKL_MBOX_S1_SLOT;
  volatile uint32_t *const ping = (volatile uint32_t *)(mbox + HEXKL_MBOX_PING);
  volatile uint32_t *const pong = (volatile uint32_t *)(mbox + HEXKL_MBOX_PONG);
  const float *in = act_in;
  uint32_t in_len = act_len, start = 0, round = 0, end, resume;
  hexkl_mbox_hdr h;
  /* [plan 201 S1] one PD: this side runs the MOE ops too */
  tk_miss miss = {env, mbox, tok, 0u, spin_us, st};
  hexkl_graph_env menv = *env;
  int rc;
  menv.miss.post = tk_miss_post;
  menv.miss.wait = tk_miss_wait;
  menv.miss.ctx = &miss;
  env = &menv;
  g->route_log_n = 0u;
  *id = 0u;
  if (g->n_ops == 0u || !g->ops[0].resident) {
    return AEE_EBADSTATE;
  }
  for (;;) {
    const htp_graph_op *last;
    uint32_t n_out, seq;
    end = tk_stretch_end(g, start);
    last = &g->ops[end - 1u];
    if (end == g->n_ops) {
      float *out = logits;
      uint32_t out_len = logits_len;
      if (out == NULL) {
        if (last->kind != HTP_OP_LM_HEAD) {
          return AEE_EINVALIDFORMAT;
        }
        out = g->logits; /* forward skips the copy onto itself */
        out_len = last->N;
      }
      rc = hexkl_graph_forward(g, env, start, end - start, pos, NULL, in,
                               in_len, out, out_len, &resume);
      if (rc != AEE_SUCCESS) {
        return rc; /* S1 has served its last round: nothing to abort */
      }
      hvx_worker_pool_park(env->pool); /* the token is done */
      tk_pcycles(g, start, end, st);
      *id = last->kind == HTP_OP_LM_HEAD ? g->lm_id : 0u;
      ++st->tokens;
      return AEE_SUCCESS;
    }
    n_out = htp_graph_op_out_words(last);
    seq = hexkl_token_seq(tok, round);
    if (round >= HEXKL_TOKEN_MAX_ROUNDS || n_out * 4u > HEXKL_MBOX_ROW_MAX) {
      rc = AEE_EINVALIDFORMAT;
    } else {
      rc = hexkl_graph_forward(g, env, start, end - start, pos, NULL, in,
                               in_len, tk_row(mine), n_out, &resume);
    }
    hvx_worker_pool_park(env->pool); /* S1 computes next */
    /* posted either way: a failure ends S1's token at once */
    tk_post(mine, ping, seq, end, n_out, rc);
    ++st->hops;
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    tk_pcycles(g, start, end, st);
    rc = tk_take(theirs, pong, seq, spin_us, st, &h);
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    ++st->hops;
    if (h.rc != AEE_SUCCESS) {
      return h.rc;
    }
    if (h.op <= end || h.op >= g->n_ops || !g->ops[h.op].resident) {
      return AEE_EBADSTATE;
    }
    start = h.op;
    in = tk_row(theirs);
    in_len = h.n;
    ++round;
  }
}

int hexkl_token_serve(hexkl_graph *g, const hexkl_graph_env *env, uint8_t *mbox,
                      uint32_t tok, uint32_t pos, uint32_t spin_us,
                      hexkl_token_stats *st) {
  uint8_t *const theirs = mbox + HEXKL_MBOX_S2_SLOT;
  uint8_t *const mine = mbox + HEXKL_MBOX_S1_SLOT;
  volatile uint32_t *const ping = (volatile uint32_t *)(mbox + HEXKL_MBOX_PING);
  volatile uint32_t *const pong = (volatile uint32_t *)(mbox + HEXKL_MBOX_PONG);
  const uint32_t rounds = hexkl_token_rounds(g);
  tk_miss miss = {env, mbox, tok, 0u, spin_us, st};
  hexkl_graph_env menv = *env;
  uint32_t r;
  menv.miss.post = tk_miss_post;
  menv.miss.wait = tk_miss_wait;
  menv.miss.ctx = &miss;
  env = &menv;
  g->route_log_n = 0u;
  if (rounds == 0u) {
    return AEE_EBADSTATE;
  }
  if (rounds > HEXKL_TOKEN_MAX_ROUNDS) {
    return AEE_EINVALIDFORMAT;
  }
  for (r = 0; r < rounds; ++r) {
    const uint32_t seq = hexkl_token_seq(tok, r);
    uint32_t end = 0, n_out = 0, resume = 0;
    hexkl_mbox_hdr h;
    int rc = tk_take(theirs, ping, seq, spin_us, st, &h);
    if (rc == AEE_EEXPIRED) {
      return rc; /* S2 is gone or on another token: nobody to tell */
    }
    if (rc == AEE_SUCCESS) {
      ++st->hops;
      if (h.rc != AEE_SUCCESS) {
        return h.rc; /* S2 failed and said so */
      }
      if (h.op >= g->n_ops || !g->ops[h.op].resident) {
        rc = AEE_EBADSTATE;
      } else {
        end = tk_stretch_end(g, h.op);
        n_out = htp_graph_op_out_words(&g->ops[end - 1u]);
        rc = n_out * 4u > HEXKL_MBOX_ROW_MAX
               ? AEE_EINVALIDFORMAT
               : hexkl_graph_forward(g, env, h.op, end - h.op, pos, NULL,
                                     tk_row(theirs), h.n, tk_row(mine), n_out,
                                     &resume);
      }
    }
    hvx_worker_pool_park(env->pool); /* S2 computes next */
    tk_post(mine, pong, seq, resume, n_out, rc);
    ++st->hops;
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    tk_pcycles(g, h.op, end, st);
  }
  ++st->tokens;
  return AEE_SUCCESS;
}
