// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_worker_pool.h
 * @date   06 Aug 2026
 * @brief  Fixed-size QuRT thread pool for splitting HVX-bound work by index
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_WORKER_POOL_H__
#define __NNTRAINER_HVX_WORKER_POOL_H__

#include <stdint.h>

/**
 * @brief One unit of parallel work: the callee sees which of @a n_threads
 *        it is (@a i) and computes its own slice from that -- the same
 *        shape as llama.cpp ggml-hexagon's work_queue_func_t, so a second
 *        task type (e.g. a future dequant fusion) can share this pool later
 *        by writing one more function with this signature, with nothing
 *        below it needing to change.
 */
typedef void (*hvx_worker_pool_func)(uint32_t n_threads, uint32_t i, void *ctx);

typedef struct hvx_worker_pool_s hvx_worker_pool;

/**
 * @brief Starts @a n_workers QuRT threads, parked waiting for work.
 *
 * @param n_workers  worker thread count, NOT including the calling thread
 *                    (pass hwinfo's n_hvx - 1: the caller's own thread uses
 *                    one HVX context too). 0 is valid and makes every
 *                    hvx_worker_pool_run() call run inline on the caller.
 * @return NULL on allocation or thread-creation failure.
 */
hvx_worker_pool *hvx_worker_pool_create(uint32_t n_workers);

/** @brief Signals every worker to exit and joins them. Safe to call on NULL. */
void hvx_worker_pool_destroy(hvx_worker_pool *pool);

/**
 * @brief Runs func(n, i, ctx) for i in [0, n) and blocks until all n calls
 *        finish -- i == 0 on the calling thread, the rest on pool workers.
 *
 * n = min(n_units, pool's worker count + 1). If @a pool is NULL or has no
 * workers, or n_units <= 1, func runs once inline with no thread handoff.
 *
 * Not safe to call concurrently from two threads on the same pool -- same
 * single-owner assumption as the HMX lock this pool lives alongside.
 */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units);

/**
 * @brief Starts func(n, i, ctx) for i in [0, n) on the WORKERS ONLY and
 *        returns at once; the calling thread is free to do other work --
 *        issue HMX tiles -- until hvx_worker_pool_wait.
 *
 * n = min(n_units, pool's worker count): the caller is not worker 0 here,
 * so a pool with no workers (or NULL, or n_units == 0) runs func once
 * inline before returning, and the wait is then a no-op. @a ctx must stay
 * valid until the wait. One job in flight at a time: a submit while one is
 * outstanding waits for it first, as does hvx_worker_pool_run.
 *
 * Same single-owner rule as run.
 */
void hvx_worker_pool_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                            void *ctx, uint32_t n_units);

/** @brief Blocks until the job from hvx_worker_pool_submit has finished and
 *         its writes are visible to the caller. No-op with nothing in
 *         flight. */
void hvx_worker_pool_wait(hvx_worker_pool *pool);

/**
 * @brief A background job: func(n_units, u, ctx) for u in [0, n_units),
 *        claimed one unit at a time by whichever worker has no run/submit
 *        job to serve. The caller owns the struct and @a done (n_units
 *        bytes) and keeps both alive until the job is complete; it fills
 *        func, ctx, n_units and done and leaves the rest to the pool.
 *
 * Jobs form a QUEUE in submit order, and a job's units are claimable only
 * once every job before it is complete -- that is the whole dependency
 * mechanism: a pipeline (matmul units, then a requantization, then more
 * matmul units) is three jobs in a row. Nothing spins inside a unit.
 */
typedef struct {
  hvx_worker_pool_func func;
  void *ctx;
  uint32_t n_units;
  uint8_t *done; /**< one byte per unit, set to 1 as it finishes */
  /* pool-owned from submit on */
  _Atomic uint32_t next;   /**< claim counter */
  _Atomic uint32_t n_done; /**< finished units */
  _Atomic int complete;    /**< n_done == n_units */
  uint32_t low;            /**< caller-side: units [0, low) seen done */
} hvx_bg_job;

/** @brief How many background jobs can be queued before submit_bg has to
 *         wait for the oldest to complete. A MoE prefill call queues the
 *         pack and three jobs per tail block. */
#define HVX_WORKER_POOL_BG_DEPTH 128u

/**
 * @brief Queues a background job (see hvx_bg_job) and returns at once.
 *
 * The run/submit lane is one job in flight, retired at the next wait; a
 * worker that finishes its slice of it goes idle until the next job. The
 * MoE kernel's HMX issue between two epilogue submits is ~47 us and the
 * epilogue ~18 us, so the workers idle for most of every batch (doc 47
 * section 14). This lane fills that idle with work that has no place in
 * the batch order: the activation pack, an expert's tail block. A worker
 * takes one unit, then looks for a foreground job again, so a foreground
 * submit waits at most one unit -- size units accordingly (~10 us).
 *
 * With no workers (or NULL) every unit runs inline here, in order.
 */
void hvx_worker_pool_submit_bg(hvx_worker_pool *pool, hvx_bg_job *job);

/**
 * @brief Blocks until units [0, n) of @a job have finished and their writes
 *        are visible; n at or past the unit count waits for the job to be
 *        complete (UINT32_MAX therefore always does). While it waits the
 *        calling thread takes units itself -- of this job or an earlier
 *        one -- so a wait never idles the caller beside idle work.
 */
void hvx_worker_pool_wait_bg(hvx_worker_pool *pool, hvx_bg_job *job,
                             uint32_t n);

#endif /* __NNTRAINER_HVX_WORKER_POOL_H__ */
