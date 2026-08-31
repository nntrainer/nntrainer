// SPDX-License-Identifier: Apache-2.0
/**
 * @file	worker_pool.c
 * @date	18 August 2026
 * @brief	QuRT worker thread pool: per-worker start semaphores plus a
 *		shared done semaphore, one barrier per wp_run() call. The
 *		calling thread is not a worker.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>

#include <qurt.h>

#include "worker_pool.h"

#define WP_STACK_SIZE (64 * 1024)

struct worker {
  qurt_thread_t tid;
  void *stack;
  int id;
  struct wp_pool *pool;
  qurt_sem_t start_sem; /* this worker's own credit; released once per
                         * wp_run()/shutdown so each worker runs the job
                         * exactly once (a shared counting sem would let a
                         * fast worker steal a second credit before a slow
                         * one wakes) */
};

struct wp_pool {
  int n;
  struct worker *workers;
  qurt_sem_t done_sem; /* shared; acquired n times per wp_run() (the barrier) */
  wp_job_fn fn;        /* NULL means "shut down" */
  void *arg;
};

static void worker_main(void *varg) {
  struct worker *w = varg;
  struct wp_pool *p = w->pool;

  qurt_hvx_lock(QURT_HVX_MODE_128B);
  for (;;) {
    qurt_sem_down(&w->start_sem);
    wp_job_fn fn = p->fn;
    if (!fn)
      break;
    fn(p->arg, w->id, p->n);
    qurt_sem_up(&p->done_sem);
  }
  qurt_hvx_unlock();
  qurt_thread_exit(0);
}

struct wp_pool *wp_create(int n_workers) {
  int n = n_workers;
  if (n <= 0) {
    int units = qurt_hvx_get_units();
    n = (units >> 8) & 0xFF; /* count of 128B-mode HVX units */
    if (n <= 0)
      n = 1;
  }

  struct wp_pool *p = malloc(sizeof(*p));
  struct worker *workers = malloc(sizeof(*workers) * (size_t)n);
  if (!p || !workers) {
    free(p);
    free(workers);
    return NULL;
  }
  p->n = n;
  p->workers = workers;
  p->fn = NULL;
  p->arg = NULL;
  qurt_sem_init_val(&p->done_sem, 0);

  for (int i = 0; i < n; ++i) {
    workers[i].id = i;
    workers[i].pool = p;
    qurt_sem_init_val(&workers[i].start_sem, 0);
    workers[i].stack = memalign(128, WP_STACK_SIZE);

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_name(&attr, "wp_worker");
    qurt_thread_attr_set_stack_addr(&attr, workers[i].stack);
    qurt_thread_attr_set_stack_size(&attr, WP_STACK_SIZE);
    qurt_thread_attr_set_detachstate(&attr, QURT_THREAD_ATTR_CREATE_JOINABLE);

    int rc = workers[i].stack ? qurt_thread_create(&workers[i].tid, &attr,
                                                   worker_main, &workers[i])
                              : -1;
    if (rc != QURT_EOK) {
      /* Roll back: shut down and free the i workers already started, then
       * this failed one, then the pool itself. */
      free(workers[i].stack);
      qurt_sem_destroy(&workers[i].start_sem);
      for (int k = 0; k < i; ++k)
        qurt_sem_up(&workers[k].start_sem); /* p->fn is still NULL: exit */
      int status;
      for (int k = 0; k < i; ++k) {
        qurt_thread_join(workers[k].tid, &status);
        free(workers[k].stack);
        qurt_sem_destroy(&workers[k].start_sem);
      }
      qurt_sem_destroy(&p->done_sem);
      free(workers);
      free(p);
      return NULL;
    }
  }
  return p;
}

void wp_run(struct wp_pool *p, wp_job_fn fn, void *arg) {
  p->fn = fn;
  p->arg = arg;
  /* Plain stores to p->fn/p->arg are ordered by the per-worker semaphore
   * release/acquire (QuRT semantics), no C11 atomics needed. */
  for (int i = 0; i < p->n; ++i)
    qurt_sem_up(&p->workers[i].start_sem);
  for (int i = 0; i < p->n; ++i)
    qurt_sem_down(&p->done_sem);
}

int wp_size(const struct wp_pool *p) { return p->n; }

void wp_destroy(struct wp_pool *p) {
  p->fn = NULL;
  p->arg = NULL;
  for (int i = 0; i < p->n; ++i)
    qurt_sem_up(&p->workers[i].start_sem); /* fn == NULL tells worker to exit */

  int status;
  for (int i = 0; i < p->n; ++i) {
    qurt_thread_join(p->workers[i].tid, &status);
    free(p->workers[i].stack);
    qurt_sem_destroy(&p->workers[i].start_sem);
  }
  qurt_sem_destroy(&p->done_sem);
  free(p->workers);
  free(p);
}
