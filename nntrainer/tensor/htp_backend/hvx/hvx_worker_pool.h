// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hvx_worker_pool.h
 * @date 06 Aug 2026
 * @brief Fixed-size QuRT thread pool for splitting HVX-bound work by index
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs except for NYI items */

#ifndef __NNTRAINER_HVX_WORKER_POOL_H__
#define __NNTRAINER_HVX_WORKER_POOL_H__

#include <stdint.h>

/**
 * @brief One unit of parallel work: the callee sees which of @a n_threads
 * it is (@a i) and computes its own slice from that the same
 * shape as llama.cpp ggml-hexagon's work_queue_func_t, so a second
 * task type (e.g. a future dequant fusion) can share this pool later
 * by writing one more function with this signature, with nothing
 * below it needing to change. */
typedef void (*hvx_worker_pool_func)(uint32_t n_threads, uint32_t i, void *ctx);

typedef struct hvx_worker_pool_s hvx_worker_pool;

/**
 * @brief Starts @a n_workers QuRT threads, parked waiting for work.
 *
 * @param n_workers worker thread count, NOT including the calling thread
 * (pass hwinfo's n_hvx - 1: the caller's own thread uses
 * one HVX context too). 0 is valid and makes every
 * hvx_worker_pool_run call run inline on the caller.
 * @return NULL on allocation or thread-creation failure. */
hvx_worker_pool *hvx_worker_pool_create(uint32_t n_workers);

/** @brief Signals every worker to exit and joins them. Safe to call on NULL. */
void hvx_worker_pool_destroy(hvx_worker_pool *pool);

/**
 * @brief Runs func(n, i, ctx) for i in [0, n) and blocks until all n calls
 * finish i == 0 on the calling thread, the rest on pool workers.
 *
 * n = min(n_units, pool's worker count + 1). If @a pool is NULL or has no
 * workers, or n_units <= 1, func runs once inline with no thread handoff.
 *
 * Not safe to call concurrently from two threads on the same pool same
 * single-owner assumption as the HMX lock this pool lives alongside. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units);

#endif /* __NNTRAINER_HVX_WORKER_POOL_H__ */
