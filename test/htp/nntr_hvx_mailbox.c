// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_mailbox.c
 * @date   29 Sep 2026
 * @brief  [#178] mailbox_run: one side of a DSP-to-DSP hop through a shared
 *         ION page that two sessions (two PDs) both map and poll
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * DEBUG ONLY: no model path calls it (plan 178 section 2, probe Q2). There
 * is no PD-to-PD signalling API, so a hop is a sequence word one side
 * writes and the other polls. Layout of the page: ping (role 0 writes) at
 * byte 0, pong (role 1 writes) at byte 128, each on its own 128-byte line;
 * role 0's payload at 256, role 1's after it, each rounded to 128 bytes.
 *
 * Exchange seq (1..n): role 0 copies its payload (probe words at first,
 * middle and last carry seq), cleans it and the ping line out of the DSP
 * caches, posts ping = seq, and waits for pong == seq; then it reads role
 * 1's payload. Role 1 waits for ping == seq, reads role 0's payload,
 * posts its own the same way. A read cleans+invalidates the other side's
 * lines first (flush-invalidate, never a bare invalidate: if the two PDs
 * share a physical cache line a bare invalidate could drop the writer's
 * not-yet-cleaned store). The checksum is the sum of the probe words
 * received; n_bad counts probe words that did not carry the expected seq
 * (a stale read). The loop itself is plain C over volatile words, so
 * test/htp/host/mailbox_host_check.c runs both roles on two pthreads by
 * including this file (the cache calls compile away off the DSP).
 *
 * Address space: a mapping of the caller's buffer (HAP_mmap_get) for the
 * call's duration and a heap copy source of @a payload bytes (at most
 * 64 KiB), both released before return.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__hexagon__)
#include <AEEStdErr.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_memory.h>
#include <remote.h>

#include "nntr_hvx.h"
#else
#include <time.h>
#include <unistd.h>
#endif

/** @brief Largest payload one side copies per exchange. */
#define MB_MAX_PAYLOAD (64u * 1024u)
/** @brief Byte offsets of the two sequence words and the payloads. */
#define MB_PING 0u
#define MB_PONG 128u
#define MB_DATA 256u
/** @brief A wait gives up this long after its spin window. */
#define MB_TIMEOUT_US 1000000u
/** @brief Poll period after the spin window. */
#define MB_SLEEP_US 50u

static uint64_t mb_now_us(void) {
#if defined(__hexagon__)
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
#else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
#endif
}

/** @brief Our store reaches DDR (and the other PD's view). */
static void mb_clean(void *p, uint32_t n) {
#if defined(__hexagon__)
  qurt_mem_cache_clean((qurt_addr_t)p, (qurt_size_t)n, QURT_MEM_CACHE_FLUSH,
                       QURT_MEM_DCACHE);
#else
  (void)p;
  (void)n;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/** @brief The other side's lines are re-read from memory. */
static void mb_refresh(void *p, uint32_t n) {
#if defined(__hexagon__)
  qurt_mem_cache_clean((qurt_addr_t)p, (qurt_size_t)n,
                       QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
#else
  (void)p;
  (void)n;
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static void mb_sleep(void) {
#if defined(__hexagon__)
  qurt_timer_sleep(MB_SLEEP_US);
#else
  usleep(MB_SLEEP_US);
#endif
}

/** @brief The probe word values of @a role's payload at exchange @a seq. */
static uint32_t mb_word(uint32_t role, uint32_t seq, uint32_t k) {
  return seq * 2654435761u + k + (role ? 0x5A5A0000u : 0u);
}

/** @brief Word indices of the probe words of a payload of @a words. */
static void mb_probe_idx(uint32_t words, uint32_t idx[3]) {
  idx[0] = 0u;
  idx[1] = words / 2u;
  idx[2] = words - 1u;
}

/** @brief Spins on @a w == @a seq for @a spin_us, then polls every 50 us.
 *  @return 1 when it came, 0 on timeout. */
static int mb_wait(volatile uint32_t *w, uint32_t seq, uint32_t spin_us) {
  const uint64_t t0 = mb_now_us();
  for (;;) {
    mb_refresh((void *)w, 4u);
    if (*w == seq) {
      return 1;
    }
    const uint64_t dt = mb_now_us() - t0;
    if (dt >= (uint64_t)spin_us + MB_TIMEOUT_US) {
      return 0;
    }
    if (dt >= spin_us) {
      mb_sleep();
    }
  }
}

/**
 * @brief One role's whole run over the page at @a base (@a bytes long).
 * @param res [us_total, n_done, n_timeouts, payload_checksum, n_bad]
 * @return 0, or -1 when the page is too small or the payload too big
 */
int nntr_mailbox_loop(uint8_t *base, uint32_t bytes, uint32_t role, uint32_t n,
                      uint32_t payload, uint32_t spin_us, uint32_t res[5]) {
  const uint32_t words = payload / 4u;
  const uint32_t slot = (payload + 127u) & ~127u;
  if (role > 1u || payload % 4u != 0u || payload > MB_MAX_PAYLOAD ||
      (uint64_t)MB_DATA + 2u * (uint64_t)slot > bytes) {
    return -1;
  }
  volatile uint32_t *mine =
    (volatile uint32_t *)(base + (role ? MB_PONG : MB_PING));
  volatile uint32_t *theirs =
    (volatile uint32_t *)(base + (role ? MB_PING : MB_PONG));
  uint32_t *out = (uint32_t *)(base + MB_DATA + (role ? slot : 0u));
  uint32_t *in = (uint32_t *)(base + MB_DATA + (role ? 0u : slot));
  uint32_t *src = NULL;
  uint32_t idx[3] = {0u, 0u, 0u};
  if (words) {
    src = (uint32_t *)malloc(payload);
    if (!src) {
      return -1;
    }
    for (uint32_t i = 0; i < words; ++i) {
      src[i] = i;
    }
    mb_probe_idx(words, idx);
  }
  memset(res, 0, 5u * sizeof(uint32_t));
  uint64_t t0 = 0;
  uint32_t sum = 0, bad = 0, done = 0, timeouts = 0;
  for (uint32_t seq = 1u; seq <= n; ++seq) {
    if (role == 1u) {
      if (!mb_wait(theirs, seq, spin_us)) {
        ++timeouts;
        break;
      }
      if (seq == 1u) {
        t0 = mb_now_us();
      }
    } else if (seq == 1u) {
      t0 = mb_now_us();
    }
    if (role == 1u && words) {
      mb_refresh(in, payload);
      for (uint32_t k = 0; k < 3u; ++k) {
        sum += in[idx[k]];
        bad += in[idx[k]] != mb_word(0u, seq, k);
      }
    }
    if (words) {
      for (uint32_t k = 0; k < 3u; ++k) {
        src[idx[k]] = mb_word(role, seq, k);
      }
      memcpy(out, src, payload);
      mb_clean(out, payload);
    }
    *mine = seq;
    mb_clean((void *)mine, 4u);
    if (role == 0u) {
      if (!mb_wait(theirs, seq, spin_us)) {
        ++timeouts;
        break;
      }
      if (words) {
        mb_refresh(in, payload);
        for (uint32_t k = 0; k < 3u; ++k) {
          sum += in[idx[k]];
          bad += in[idx[k]] != mb_word(1u, seq, k);
        }
      }
    }
    ++done;
  }
  res[0] = (uint32_t)(mb_now_us() - t0);
  res[1] = done;
  res[2] = timeouts;
  res[3] = sum;
  res[4] = bad;
  free(src);
  return 0;
}

#if defined(__hexagon__)
int nntr_hvx_mailbox_run(remote_handle64 handle, int32 fd, uint32 bytes,
                         uint32 role, uint32 n, uint32 payload, uint32 spin_us,
                         uint32 *res, int resLen) {
  void *va = NULL;
  uint64 pa = 0;
  if (!handle) {
    return AEE_EBADPARM;
  }
  if (resLen != 5) {
    return AEE_EINVALIDFORMAT;
  }
  const int rc = HAP_mmap_get((int)fd, &va, &pa);
  if (rc != 0 || va == NULL) {
    return rc != 0 ? rc : AEE_ENOMEMORY;
  }
  const int lrc =
    nntr_mailbox_loop((uint8_t *)va, bytes, role, n, payload, spin_us, res);
  HAP_mmap_put((int)fd);
  return lrc == 0 ? AEE_SUCCESS : AEE_EINVALIDFORMAT;
}
#endif
