// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_pool.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the QuRT worker pool barrier semantics
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <string.h>

#include "worker_pool.h"

struct pjob {
  int slot[16];
  unsigned count;
};

static void pfn(void *arg, int wid, int nw) {
  struct pjob *j = arg;
  j->slot[wid] = wid + 1;
  __atomic_add_fetch(&j->count, 1, __ATOMIC_SEQ_CST);
  (void)nw;
}

int test_pool(void) {
  struct wp_pool *p = wp_create(0);
  if (!p)
    return 1;
  int n = wp_size(p);
  printf("SIM_TEST pool n_workers=%d\n", n);
  if (n < 1 || n > 16)
    return 1;
  struct pjob j;
  memset(&j, 0, sizeof(j));
  for (int it = 0; it < 100; ++it)
    wp_run(p, pfn, &j);
  if (j.count != (unsigned)(100 * n))
    return 1;
  for (int w = 0; w < n; ++w)
    if (j.slot[w] != w + 1)
      return 1;
  wp_destroy(p);
  printf("SIM_TEST pool PASS\n");
  return 0;
}
