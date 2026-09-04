// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hvx_worker_pool.c
 * @date 06 Aug 2026
 * @brief Fixed-size QuRT thread pool for splitting HVX-bound work by index
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * The seqn/barrier/futex protocol below is not novel it is
 * llama.cpp ggml-hexagon's htp/work-queue.c pattern (same QuRT primitives,
 * same target), trimmed to a single-job-in-flight fork/join: this pool has
 * no run_async and no multi-slot queue, since every caller here blocks
 * until its job is done anyway. That is a smaller, already-proven surface
 * for what is otherwise this codebase's first genuinely concurrent code
 * path. */

#include "hvx_worker_pool.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include <qurt.h>

/** @brief Hexagon SMT pause hint for the calling thread's spin-wait below *
 * same instruction ggml-hexagon's work-queue.c uses while a job is in flight.
 */
static inline void hvx_worker_pool_pause(void) {
  asm volatile(" pause(#255)\n");
}

/** @brief Per-worker stack, generous for the quant kernels this pool
 * currently runs (a handful of HVX vector locals, no recursion). */
#define HVX_WORKER_POOL_STACK_SIZE (16u * 1024u)

typedef struct hvx_worker_pool_s hvx_worker_pool;

typedef struct {
  hvx_worker_pool *pool;
  uint32_t id; /**< 1..n_workers; 0 is reserved for the calling thread */
} hvx_worker_ctx;

struct hvx_worker_pool_s {
  _Atomic uint32_t seqn;    /**< bumped once per hvx_worker_pool_run call */
  _Atomic uint32_t barrier; /**< participating workers still running */
  _Atomic int killed;
  hvx_worker_pool_func func;
  void *ctx;
  uint32_t n_threads; /**< participants (main + workers) for the current run */
  uint32_t n_workers;
  qurt_thread_t *tids;
  hvx_worker_ctx *worker_ctx;
  unsigned char *stack_blob;
};

static void hvx_worker_pool_thread_entry(void *arg) {
  hvx_worker_ctx *me = (hvx_worker_ctx *)arg;
  hvx_worker_pool *pool = me->pool;
  uint32_t prev_seqn = 0;

  for (;;) {
    if (atomic_load_explicit(&pool->killed, memory_order_relaxed)) {
      qurt_thread_exit(0);
    }

    uint32_t seqn = atomic_load_explicit(&pool->seqn, memory_order_acquire);
    if (seqn == prev_seqn) {
      qurt_futex_wait(&pool->seqn, (int)prev_seqn);
      continue;
    }
    prev_seqn = seqn;

    if (me->id < pool->n_threads) {
      pool->func(pool->n_threads, me->id, pool->ctx);
      atomic_fetch_sub_explicit(&pool->barrier, 1, memory_order_release);
    }
    // me->id >= pool->n_threads: this run didn't need this worker; loop
    // back and wait for the next one.
  }
}

hvx_worker_pool *hvx_worker_pool_create(uint32_t n_workers) {
  hvx_worker_pool *pool = (hvx_worker_pool *)calloc(1, sizeof(*pool));
  if (!pool) {
    return NULL;
  }
  atomic_init(&pool->seqn, 0);
  atomic_init(&pool->barrier, 0);
  atomic_init(&pool->killed, 0);
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
  // work_queue_init these workers only ever run while the FastRPC
  // thread that owns this session is waiting on them, so there is no
  // reason for them to run at a different priority.
  int prio = qurt_thread_get_priority(qurt_thread_get_id);
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

void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  if (!pool || pool->n_workers == 0 || n_units <= 1) {
    func(n_units == 0 ? 1u : n_units, 0, ctx);
    return;
  }

  uint32_t n = n_units;
  if (n > pool->n_workers + 1u) {
    n = pool->n_workers + 1u;
  }

  pool->func = func;
  pool->ctx = ctx;
  pool->n_threads = n;
  atomic_store_explicit(&pool->barrier, n - 1u, memory_order_relaxed);

  // Publish the job, then wake every worker not just the n-1 that will
  // participate. qurt_futex_wake(addr, k) wakes k ARBITRARY sleepers on
  // addr, not k specific ones by id: waking only n-1 risked waking
  // non-participants while the actual participants stayed asleep forever
  // (found on-device: the very first call deadlocked here). A
  // non-participant that wakes just rechecks its id against n_threads,
  // finds it doesn't apply, and goes back to sleep harmless.
  atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
  qurt_futex_wake(&pool->seqn, (int)pool->n_workers);

  func(n, 0, ctx); // the calling thread is worker 0

  while (atomic_load_explicit(&pool->barrier, memory_order_relaxed) > 0) {
    hvx_worker_pool_pause;
  }
  // Pairs with each worker's release store to barrier: makes every
  // worker's writes to ctx visible to the calling thread from here on.
  atomic_thread_fence(memory_order_acquire);
}
