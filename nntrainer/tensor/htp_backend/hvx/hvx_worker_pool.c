// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_worker_pool.c
 * @date   06 Aug 2026
 * @brief  Fixed-size QuRT thread pool for splitting HVX-bound work by index
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The seqn/barrier/futex protocol below is not novel -- it is
 * llama.cpp ggml-hexagon's htp/work-queue.c pattern (same QuRT primitives,
 * same target), trimmed to a single-job-in-flight fork/join: this pool has
 * no multi-slot queue. Two additions since: submit/wait, the same job run
 * without the caller as worker 0, and the background lane (submit_bg), a
 * unit counter workers claim from when the foreground lane has nothing
 * for them. Both keep the one-owner, one-job-per-lane shape.
 */

#include "hvx_worker_pool.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include <qurt.h>

/** @brief Hexagon SMT pause hint for the calling thread's spin-wait below --
 *         same instruction ggml-hexagon's work-queue.c uses while a job is
 *         in flight. */
static inline void hvx_worker_pool_pause(void) {
  /* The host check (test/htp/host) builds this file with a pthread stand-in
     for QuRT; a plain spin is fine there. */
#if defined(__hexagon__)
  asm volatile(" pause(#255)\n");
#endif
}

/** @brief Per-worker stack. The tail-block units (hvx_gemm_u8i4_wh.c and
 *         the epilogues they call) keep a dozen HVX vectors live, so 32 KB
 *         rather than the 16 the quant kernels needed. */
#define HVX_WORKER_POOL_STACK_SIZE (32u * 1024u)

typedef struct hvx_worker_pool_s hvx_worker_pool;

typedef struct {
  hvx_worker_pool *pool;
  uint32_t id; /**< 1..n_workers; 0 is reserved for the calling thread */
} hvx_worker_ctx;

struct hvx_worker_pool_s {
  _Atomic uint32_t seqn;    /**< wake counter: bumped by every publish */
  _Atomic uint32_t fg_id;   /**< bumped once per run/submit: the job a
                                 worker has not served yet */
  _Atomic uint32_t barrier; /**< participating workers still running */
  _Atomic int killed;
  hvx_worker_pool_func func;
  void *ctx;
  uint32_t n_threads; /**< participants for the current job: main + workers
                           for run, workers only for submit */
  /** Nonzero while a submit's job is in flight. Workers read it under the
      fg_id acquire, so a job's mode is fixed before it is published: in a
      submitted job the caller is not worker 0, and worker id k acts as
      index k-1 of n_threads instead of index k. */
  int async;
  int outstanding; /**< caller-side: a submit not yet waited for */

  /** The background lane: a ring of caller-owned jobs in submit order.
      Workers scan from head; a job's units are claimable only once every
      job before it is complete, and a claim is a CAS on the job's own
      counter, so a job's fields are fixed from before its publication
      (the release store of bg_tail) and never reused. */
  hvx_bg_job *bg_ring[HVX_WORKER_POOL_BG_DEPTH];
  _Atomic uint32_t bg_head; /**< oldest job not yet popped by the caller */
  _Atomic uint32_t bg_tail; /**< next slot; head == tail means empty */

  uint32_t n_workers;
  qurt_thread_t *tids;
  hvx_worker_ctx *worker_ctx;
  unsigned char *stack_blob;
};

/** @brief Claims and runs one unit of @a job; 0 when it has none left. */
static int hvx_worker_pool_bg_take_from(hvx_bg_job *job) {
  uint32_t u = atomic_load_explicit(&job->next, memory_order_relaxed);
  for (;;) {
    if (u >= job->n_units) {
      return 0;
    }
    if (atomic_compare_exchange_weak_explicit(
          &job->next, &u, u + 1u, memory_order_acq_rel, memory_order_relaxed)) {
      break;
    }
  }
  job->func(job->n_units, u, job->ctx);
  atomic_store_explicit((_Atomic uint8_t *)&job->done[u], 1,
                        memory_order_release);
  if (atomic_fetch_add_explicit(&job->n_done, 1u, memory_order_acq_rel) + 1u ==
      job->n_units) {
    atomic_store_explicit(&job->complete, 1, memory_order_release);
  }
  return 1;
}

/** @brief Claims and runs one background unit from the oldest job that has
 *         one, stopping at the first incomplete job that has none left --
 *         the jobs behind it are gated on it. 0 when there is nothing. */
static int hvx_worker_pool_bg_take_one(hvx_worker_pool *pool) {
  const uint32_t tail =
    atomic_load_explicit(&pool->bg_tail, memory_order_acquire);
  for (uint32_t i = atomic_load_explicit(&pool->bg_head, memory_order_relaxed);
       i != tail; ++i) {
    hvx_bg_job *job = pool->bg_ring[i % HVX_WORKER_POOL_BG_DEPTH];
    if (atomic_load_explicit(&job->complete, memory_order_acquire)) {
      continue;
    }
    return hvx_worker_pool_bg_take_from(job);
  }
  return 0;
}

static void hvx_worker_pool_thread_entry(void *arg) {
  hvx_worker_ctx *me = (hvx_worker_ctx *)arg;
  hvx_worker_pool *pool = me->pool;
  uint32_t prev_fg = 0;

  for (;;) {
    if (atomic_load_explicit(&pool->killed, memory_order_relaxed)) {
      qurt_thread_exit(0);
    }

    /* seqn first: a publish between this read and the futex wait below
       changes it, and qurt_futex_wait returns at once on a mismatch, so
       no wake is lost. */
    const uint32_t seqn =
      atomic_load_explicit(&pool->seqn, memory_order_acquire);
    const uint32_t fg =
      atomic_load_explicit(&pool->fg_id, memory_order_acquire);
    if (fg != prev_fg) {
      prev_fg = fg;
      const uint32_t idx = pool->async ? (me->id - 1u) : me->id;
      if (idx < pool->n_threads) {
        pool->func(pool->n_threads, idx, pool->ctx);
        atomic_fetch_sub_explicit(&pool->barrier, 1, memory_order_release);
      }
      // me->id >= pool->n_threads: this run didn't need this worker.
      continue;
    }
    /* One background unit, then back to the top: a foreground job that
       arrived meanwhile is served before the next unit. */
    if (hvx_worker_pool_bg_take_one(pool)) {
      continue;
    }
    qurt_futex_wait(&pool->seqn, (int)seqn);
  }
}

hvx_worker_pool *hvx_worker_pool_create(uint32_t n_workers) {
  hvx_worker_pool *pool = (hvx_worker_pool *)calloc(1, sizeof(*pool));
  if (!pool) {
    return NULL;
  }
  atomic_init(&pool->seqn, 0);
  atomic_init(&pool->fg_id, 0);
  atomic_init(&pool->barrier, 0);
  atomic_init(&pool->killed, 0);
  atomic_init(&pool->bg_head, 0);
  atomic_init(&pool->bg_tail, 0);
  pool->n_workers = n_workers;

  if (n_workers == 0) {
    return pool;
  }

  pool->tids = (qurt_thread_t *)calloc(n_workers, sizeof(qurt_thread_t));
  pool->worker_ctx =
    (hvx_worker_ctx *)calloc(n_workers, sizeof(hvx_worker_ctx));
  pool->stack_blob =
    (unsigned char *)malloc((size_t)n_workers * HVX_WORKER_POOL_STACK_SIZE);
  if (!pool->tids || !pool->worker_ctx || !pool->stack_blob) {
    hvx_worker_pool_destroy(pool);
    return NULL;
  }

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_size(&attr, HVX_WORKER_POOL_STACK_SIZE);

  // Match the creating thread's priority, same as ggml-hexagon's
  // work_queue_init -- these workers only ever run while the FastRPC
  // thread that owns this session is waiting on them, so there is no
  // reason for them to run at a different priority.
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  if (prio < 1) {
    prio = 1;
  }
  qurt_thread_attr_set_priority(&attr, (unsigned short)prio);

  for (uint32_t i = 0; i < n_workers; ++i) {
    pool->worker_ctx[i].pool = pool;
    pool->worker_ctx[i].id = i + 1;
    qurt_thread_attr_set_stack_addr(
      &attr, pool->stack_blob + (size_t)i * HVX_WORKER_POOL_STACK_SIZE);

    char name[16];
    snprintf(name, sizeof(name), "hvxpool:%u", (unsigned)i);
    qurt_thread_attr_set_name(&attr, name);

    if (qurt_thread_create(&pool->tids[i], &attr, hvx_worker_pool_thread_entry,
                           &pool->worker_ctx[i]) != 0) {
      // Only i threads actually started; limit teardown's join loop to
      // those.
      pool->n_workers = i;
      hvx_worker_pool_destroy(pool);
      return NULL;
    }
  }

  return pool;
}

void hvx_worker_pool_destroy(hvx_worker_pool *pool) {
  if (!pool) {
    return;
  }
  if (pool->n_workers > 0 && pool->tids) {
    atomic_store_explicit(&pool->killed, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
    qurt_futex_wake(&pool->seqn, (int)pool->n_workers);
    for (uint32_t i = 0; i < pool->n_workers; ++i) {
      int status;
      qurt_thread_join(pool->tids[i], &status);
    }
  }
  free(pool->tids);
  free(pool->worker_ctx);
  free(pool->stack_blob);
  free(pool);
}

void hvx_worker_pool_wait(hvx_worker_pool *pool) {
  if (!pool || !pool->outstanding) {
    return;
  }
  while (atomic_load_explicit(&pool->barrier, memory_order_relaxed) > 0) {
    hvx_worker_pool_pause();
  }
  atomic_thread_fence(memory_order_acquire);
  pool->outstanding = 0;
}

void hvx_worker_pool_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                            void *ctx, uint32_t n_units) {
  if (n_units == 0u) {
    return;
  }
  if (!pool || pool->n_workers == 0) {
    func(1u, 0, ctx); /* see run: one slice, the whole range */
    return;
  }
  hvx_worker_pool_wait(pool);

  uint32_t n = n_units;
  if (n > pool->n_workers) {
    n = pool->n_workers;
  }
  pool->func = func;
  pool->ctx = ctx;
  pool->n_threads = n;
  pool->async = 1;
  pool->outstanding = 1;
  atomic_store_explicit(&pool->barrier, n, memory_order_relaxed);
  /* Publish, then wake everyone -- run() explains why everyone. */
  atomic_fetch_add_explicit(&pool->fg_id, 1, memory_order_release);
  atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
  qurt_futex_wake(&pool->seqn, (int)pool->n_workers);
}

/** @brief Pops complete jobs off the head of the ring. Caller-side. */
static void hvx_worker_pool_bg_pop(hvx_worker_pool *pool) {
  uint32_t head = atomic_load_explicit(&pool->bg_head, memory_order_relaxed);
  const uint32_t tail =
    atomic_load_explicit(&pool->bg_tail, memory_order_relaxed);
  while (head != tail &&
         atomic_load_explicit(
           &pool->bg_ring[head % HVX_WORKER_POOL_BG_DEPTH]->complete,
           memory_order_acquire)) {
    ++head;
  }
  atomic_store_explicit(&pool->bg_head, head, memory_order_release);
}

void hvx_worker_pool_submit_bg(hvx_worker_pool *pool, hvx_bg_job *job) {
  if (!job || job->n_units == 0u || !job->done) {
    return;
  }
  for (uint32_t u = 0; u < job->n_units; ++u) {
    job->done[u] = 0;
  }
  atomic_init(&job->next, 0);
  atomic_init(&job->n_done, 0);
  atomic_init(&job->complete, 0);
  job->low = 0;
  if (!pool || pool->n_workers == 0) {
    for (uint32_t u = 0; u < job->n_units; ++u) {
      job->func(job->n_units, u, job->ctx);
      job->done[u] = 1;
    }
    atomic_store_explicit(&job->n_done, job->n_units, memory_order_relaxed);
    atomic_store_explicit(&job->complete, 1, memory_order_relaxed);
    return;
  }
  hvx_worker_pool_bg_pop(pool);
  /* Ring full: the oldest job has to finish first. Helping while waiting
     is what makes that finish. */
  while (atomic_load_explicit(&pool->bg_tail, memory_order_relaxed) -
           atomic_load_explicit(&pool->bg_head, memory_order_relaxed) >=
         HVX_WORKER_POOL_BG_DEPTH) {
    hvx_worker_pool_wait_bg(
      pool,
      pool->bg_ring[atomic_load_explicit(&pool->bg_head, memory_order_relaxed) %
                    HVX_WORKER_POOL_BG_DEPTH],
      UINT32_MAX);
    hvx_worker_pool_bg_pop(pool);
  }
  const uint32_t tail =
    atomic_load_explicit(&pool->bg_tail, memory_order_relaxed);
  pool->bg_ring[tail % HVX_WORKER_POOL_BG_DEPTH] = job;
  /* The job and its slot are visible before the tail that announces it. */
  atomic_store_explicit(&pool->bg_tail, tail + 1u, memory_order_release);
  atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
  qurt_futex_wake(&pool->seqn, (int)pool->n_workers);
}

void hvx_worker_pool_wait_bg(hvx_worker_pool *pool, hvx_bg_job *job,
                             uint32_t n) {
  if (!pool || !job || pool->n_workers == 0) {
    return; /* inline jobs are complete at submit */
  }
  if (n >= job->n_units) {
    while (!atomic_load_explicit(&job->complete, memory_order_acquire)) {
      if (!hvx_worker_pool_bg_take_one(pool)) {
        hvx_worker_pool_pause();
      }
    }
  } else {
    while (job->low < n) {
      if (atomic_load_explicit((_Atomic uint8_t *)&job->done[job->low],
                               memory_order_acquire)) {
        job->low++;
      } else if (!hvx_worker_pool_bg_take_one(pool)) {
        hvx_worker_pool_pause();
      }
    }
  }
  hvx_worker_pool_bg_pop(pool);
}

void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  hvx_worker_pool_wait(pool);
  if (!pool || pool->n_workers == 0 || n_units <= 1) {
    /* ONE slice, not n_units of them. func(n_units, 0, ctx) means "act as
       worker 0 of n_units", so with no pool it ran 1/n_units of the work
       and silently dropped the rest -- doc 43 section 7's L1 correction is
       an entry about exactly that, where a test used NULL as a "serial
       reference", compared against a weight image that was 1/n baked, and
       got a revert out of it. The trap was left in place and documented;
       this removes it. Callers that pass NULL now get the whole range on
       the calling thread, which is what every one of them meant. */
    func(1u, 0, ctx);
    return;
  }

  uint32_t n = n_units;
  if (n > pool->n_workers + 1u) {
    n = pool->n_workers + 1u;
  }

  pool->func = func;
  pool->ctx = ctx;
  pool->n_threads = n;
  pool->async = 0;
  atomic_store_explicit(&pool->barrier, n - 1u, memory_order_relaxed);

  // Publish the job, then wake every worker -- not just the n-1 that will
  // participate. qurt_futex_wake(addr, k) wakes k ARBITRARY sleepers on
  // addr, not k specific ones by id: waking only n-1 risked waking
  // non-participants while the actual participants stayed asleep forever
  // (found on-device: the very first call deadlocked here). A
  // non-participant that wakes just rechecks its id against n_threads,
  // finds it doesn't apply, and goes back to sleep -- harmless.
  atomic_fetch_add_explicit(&pool->fg_id, 1, memory_order_release);
  atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
  qurt_futex_wake(&pool->seqn, (int)pool->n_workers);

  func(n, 0, ctx); // the calling thread is worker 0

  while (atomic_load_explicit(&pool->barrier, memory_order_relaxed) > 0) {
    hvx_worker_pool_pause();
  }
  // Pairs with each worker's release store to barrier: makes every
  // worker's writes to ctx visible to the calling thread from here on.
  atomic_thread_fence(memory_order_acquire);
}
