// SPDX-License-Identifier: Apache-2.0
/**
 * @file	worker_pool.h
 * @date	18 August 2026
 * @brief	QuRT worker thread pool with per-op barrier dispatch
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_WORKER_POOL_H
#define NNTR_HTP_WORKER_POOL_H

struct wp_pool;

typedef void (*wp_job_fn)(void *arg, int worker_id, int n_workers);

/* n_workers <= 0 picks the HVX-unit count reported by qurt_hvx_get_units(). */
struct wp_pool *wp_create(int n_workers);

/* Runs fn on every worker and returns only after all of them finish. */
void wp_run(struct wp_pool *p, wp_job_fn fn, void *arg);

int wp_size(const struct wp_pool *p);

void wp_destroy(struct wp_pool *p);

#endif /* NNTR_HTP_WORKER_POOL_H */
