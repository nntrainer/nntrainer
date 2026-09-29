// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   mailbox_host_check.c
 * @date   29 Sep 2026
 * @brief  [#178] The mailbox hop's protocol (nntr_hvx_mailbox.c, included
 *         as-is) with its two roles on two pthreads over one buffer
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * 10 000 exchanges at payload 0 and 8 KiB: no timeouts, no stale probe
 * word, both sides done n times, and each side's checksum equal to the sum
 * the other side's words must give. Then a lone role 0 (nobody answers)
 * must time out after its spin window plus the 1 s cap, not hang. What this
 * does not check: the DSP cache maintenance (compiled away here) and the
 * cost of a hop, which are the device's (probe Q2).
 */

#include <pthread.h>
#include <stdio.h>

#include "../nntr_hvx_mailbox.c"

typedef struct {
  uint8_t *base;
  uint32_t bytes, role, n, payload;
  uint32_t res[5];
  int rc;
} side;

static void *run_side(void *v) {
  side *s = (side *)v;
  s->rc = nntr_mailbox_loop(s->base, s->bytes, s->role, s->n, s->payload, 1000u,
                            s->res);
  return NULL;
}

/** @brief What role @a from's probe words sum to over n exchanges. */
static uint32_t want_sum(uint32_t from, uint32_t n) {
  uint32_t sum = 0;
  for (uint32_t seq = 1u; seq <= n; ++seq) {
    for (uint32_t k = 0; k < 3u; ++k) {
      sum += mb_word(from, seq, k);
    }
  }
  return sum;
}

int main(void) {
  const uint32_t bytes = 64u * 1024u, n = 10000u;
  int fail = 0;
  for (uint32_t payload = 0; payload <= 8192u; payload += 8192u) {
    uint8_t *page = (uint8_t *)calloc(1, bytes);
    side a = {page, bytes, 0u, n, payload, {0}, 0};
    side b = {page, bytes, 1u, n, payload, {0}, 0};
    pthread_t ta, tb;
    pthread_create(&tb, NULL, run_side, &b);
    pthread_create(&ta, NULL, run_side, &a);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    const uint32_t wa = payload ? want_sum(1u, n) : 0u;
    const uint32_t wb = payload ? want_sum(0u, n) : 0u;
    const int ok = a.rc == 0 && b.rc == 0 && a.res[1] == n && b.res[1] == n &&
                   a.res[2] == 0u && b.res[2] == 0u && a.res[4] == 0u &&
                   b.res[4] == 0u && a.res[3] == wa && b.res[3] == wb;
    printf("MAILBOX payload=%u done=%u/%u timeouts=%u/%u bad=%u/%u "
           "checksum_ok=%d us_per_hop=%.2f %s\n",
           payload, a.res[1], b.res[1], a.res[2], b.res[2], a.res[4], b.res[4],
           a.res[3] == wa && b.res[3] == wb, (double)a.res[0] / (2.0 * n),
           ok ? "ok" : "FAIL");
    fail |= !ok;
    free(page);
  }
  {
    uint8_t *page = (uint8_t *)calloc(1, bytes);
    uint32_t res[5];
    const uint64_t t0 = mb_now_us();
    const int rc = nntr_mailbox_loop(page, bytes, 0u, 1u, 0u, 1000u, res);
    const uint64_t dt = mb_now_us() - t0;
    const int ok = rc == 0 && res[1] == 0u && res[2] == 1u &&
                   dt >= MB_TIMEOUT_US && dt < 3u * MB_TIMEOUT_US;
    printf("MAILBOX lone role 0: timeouts=%u after %llu us %s\n", res[2],
           (unsigned long long)dt, ok ? "ok" : "FAIL");
    fail |= !ok;
    /* a page too small for the payload is refused, not overrun */
    fail |= nntr_mailbox_loop(page, 1024u, 0u, 1u, 8192u, 0u, res) != -1;
    free(page);
  }
  if (fail) {
    printf("MAILBOX FAIL\n");
    return 1;
  }
  printf("MAILBOX OK\n");
  return 0;
}
