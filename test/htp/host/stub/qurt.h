/* Host stand-in for the QuRT calls hvx_worker_pool.c makes, on pthreads:
   enough to run the pool's protocol on a desktop, not a QuRT emulation.
   futex_wait/wake are a mutex and a condition variable with the value
   check under the mutex, which is the one property the pool relies on --
   a wake between the check and the sleep is not lost. */
#ifndef HOST_STUB_QURT_H
#define HOST_STUB_QURT_H

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef pthread_t qurt_thread_t;
typedef struct {
  int unused;
} qurt_thread_attr_t;

static inline void qurt_thread_attr_init(qurt_thread_attr_t *a) {
  memset(a, 0, sizeof(*a));
}
static inline void qurt_thread_attr_set_stack_size(qurt_thread_attr_t *a,
                                                   unsigned s) {
  (void)a;
  (void)s;
}
static inline void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t *a,
                                                   void *p) {
  (void)a;
  (void)p;
}
static inline void qurt_thread_attr_set_priority(qurt_thread_attr_t *a,
                                                 unsigned short p) {
  (void)a;
  (void)p;
}
static inline void qurt_thread_attr_set_name(qurt_thread_attr_t *a,
                                             const char *n) {
  (void)a;
  (void)n;
}
static inline qurt_thread_t qurt_thread_get_id(void) { return pthread_self(); }
static inline int qurt_thread_get_priority(qurt_thread_t t) {
  (void)t;
  return 100;
}

typedef struct {
  void (*entry)(void *);
  void *arg;
} host_qurt_tramp;

static void *host_qurt_thread_main(void *v) {
  host_qurt_tramp t = *(host_qurt_tramp *)v;
  free(v);
  t.entry(t.arg);
  return NULL;
}

static inline int qurt_thread_create(qurt_thread_t *tid,
                                     qurt_thread_attr_t *attr,
                                     void (*entry)(void *), void *arg) {
  (void)attr;
  host_qurt_tramp *t = (host_qurt_tramp *)malloc(sizeof(*t));
  if (!t)
    return -1;
  t->entry = entry;
  t->arg = arg;
  return pthread_create(tid, NULL, host_qurt_thread_main, t);
}
static inline void qurt_thread_exit(int status) {
  (void)status;
  pthread_exit(NULL);
}
static inline int qurt_thread_join(qurt_thread_t tid, int *status) {
  if (status)
    *status = 0;
  return pthread_join(tid, NULL);
}

static pthread_mutex_t host_futex_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t host_futex_cv = PTHREAD_COND_INITIALIZER;

static inline int qurt_futex_wait(void *addr, int val) {
  pthread_mutex_lock(&host_futex_mu);
  if (*(volatile uint32_t *)addr == (uint32_t)val)
    pthread_cond_wait(&host_futex_cv, &host_futex_mu);
  pthread_mutex_unlock(&host_futex_mu);
  return 0;
}
static inline int qurt_futex_wake(void *addr, int n) {
  (void)addr;
  (void)n;
  pthread_mutex_lock(&host_futex_mu);
  pthread_cond_broadcast(&host_futex_cv);
  pthread_mutex_unlock(&host_futex_mu);
  return 0;
}

#endif /* HOST_STUB_QURT_H */
