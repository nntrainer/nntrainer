/* Host check for hvx_worker_pool's two lanes on pthreads (stub/qurt.h).
   What it checks: every foreground job runs its slices exactly once while
   a background job is in flight; every background unit runs exactly once;
   wait_bg(n) returns only when units [0, n) are done; a background job
   can follow a larger and a smaller one (the stale-bg_n path in
   bg_take_one); and the caller with no workers runs everything inline.
   Timing is not measured -- the device profile does that. */
#include "hvx_worker_pool.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      failures++;                                                              \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);                              \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
    }                                                                          \
  } while (0)

#define MAX_UNITS 512u

typedef struct {
  _Atomic uint32_t count[MAX_UNITS];
  uint32_t value[MAX_UNITS];
  uint32_t n;
} bg_ctx;

static void bg_unit(uint32_t n_units, uint32_t u, void *v) {
  bg_ctx *c = (bg_ctx *)v;
  if (n_units != c->n || u >= c->n) {
    failures++;
    printf("FAIL bg_unit: n_units=%u u=%u want n=%u\n", n_units, u, c->n);
    return;
  }
  /* A little work so units overlap across workers. */
  volatile uint32_t spin = 0;
  for (uint32_t i = 0; i < 200u + (u % 7u) * 300u; ++i)
    spin += i;
  c->value[u] = u * 3u + 1u;
  atomic_fetch_add(&c->count[u], 1u);
}

typedef struct {
  _Atomic uint32_t slices[8];
  uint32_t n_threads_seen;
} fg_ctx;

static void fg_slice(uint32_t n_threads, uint32_t i, void *v) {
  fg_ctx *c = (fg_ctx *)v;
  c->n_threads_seen = n_threads;
  atomic_fetch_add(&c->slices[i], 1u);
}

static void bg_round(hvx_worker_pool *pool, uint32_t n_units, uint8_t *done,
                     int with_fg) {
  bg_ctx *bg = (bg_ctx *)calloc(1, sizeof(*bg));
  bg->n = n_units;
  hvx_worker_pool_submit_bg(pool, bg_unit, bg, n_units, done);

  if (with_fg) {
    for (int j = 0; j < 40; ++j) {
      fg_ctx fg;
      for (int s = 0; s < 8; ++s)
        atomic_init(&fg.slices[s], 0);
      hvx_worker_pool_submit(pool, fg_slice, &fg, 3u);
      /* wait_bg between submit and wait: the caller helps with units while
         a foreground job is outstanding, as the kernel does. */
      hvx_worker_pool_wait_bg(pool, (uint32_t)(j * 3));
      for (uint32_t u = 0; u < (uint32_t)(j * 3) && u < n_units; ++u)
        CHECK(atomic_load(&bg->count[u]) == 1u,
              "wait_bg(%d) returned with unit %u not done", j * 3, u);
      hvx_worker_pool_wait(pool);
      /* With no workers the pool runs ONE slice covering the whole range
         (n_threads == 1); with workers, min(n_units, workers) slices. */
      const uint32_t want_fg = fg.n_threads_seen;
      CHECK(want_fg == 1u || want_fg == 3u, "fg n_threads %u", want_fg);
      for (uint32_t s = 0; s < 8u; ++s)
        CHECK(atomic_load(&fg.slices[s]) == (s < want_fg ? 1u : 0u),
              "fg slice %u ran %u times", s, atomic_load(&fg.slices[s]));
      /* and a synchronous run, as requant does mid-call */
      for (int s = 0; s < 8; ++s)
        atomic_init(&fg.slices[s], 0);
      hvx_worker_pool_run(pool, fg_slice, &fg, 4u);
      const uint32_t want_run = fg.n_threads_seen; /* 1 or workers+1 = 4 */
      CHECK(want_run == 1u || want_run == 4u, "run n_threads %u", want_run);
      for (uint32_t s = 0; s < 8u; ++s)
        CHECK(atomic_load(&fg.slices[s]) == (s < want_run ? 1u : 0u),
              "run slice %u ran %u times", s, atomic_load(&fg.slices[s]));
    }
  }
  hvx_worker_pool_wait_bg(pool, UINT32_MAX);
  for (uint32_t u = 0; u < n_units; ++u) {
    CHECK(atomic_load(&bg->count[u]) == 1u, "unit %u ran %u times", u,
          atomic_load(&bg->count[u]));
    CHECK(bg->value[u] == u * 3u + 1u, "unit %u value", u);
    CHECK(done[u] == 1u, "unit %u done byte", u);
  }
  for (uint32_t u = n_units; u < MAX_UNITS; ++u)
    CHECK(atomic_load(&bg->count[u]) == 0u, "unit %u past the end ran", u);
  free(bg);
}

int main(void) {
  uint8_t *done = (uint8_t *)malloc(MAX_UNITS);

  /* No workers: everything inline, waits are no-ops. */
  {
    hvx_worker_pool *p0 = hvx_worker_pool_create(0);
    bg_round(p0, 17u, done, 1);
    hvx_worker_pool_destroy(p0);
  }

  hvx_worker_pool *pool = hvx_worker_pool_create(3);
  CHECK(pool != NULL, "create");
  for (int iter = 0; iter < 30; ++iter) {
    bg_round(pool, 257u, done, 1);  /* big, with foreground traffic */
    bg_round(pool, 5u, done, 0);    /* smaller after bigger: stale bg_n */
    bg_round(pool, 300u, done, 0);  /* bigger after smaller */
    bg_round(pool, 1u, done, 1);
  }
  /* A wait_bg with nothing submitted, and one past the end. */
  hvx_worker_pool_wait_bg(pool, 10u);
  hvx_worker_pool_wait_bg(NULL, 10u);
  hvx_worker_pool_destroy(pool);
  free(done);

  if (failures) {
    printf("%d FAILURES\n", failures);
    return 1;
  }
  printf("WORKER POOL LANES OK\n");
  return 0;
}
