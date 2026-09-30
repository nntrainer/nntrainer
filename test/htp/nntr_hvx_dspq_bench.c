// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_dspq_bench.c
 * @date   28 Sep 2026
 * @brief  [#141] DSP side of the dspqueue vs FastRPC round-trip microbench:
 *         one thread parked on a queue the ARM side exported, echoing a
 *         12 KiB payload between two rpcmem buffers the DSP already maps
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * DEBUG ONLY: no model path calls these entries (plan 141). The dspqueue
 * symbols are weak, as in the SDK's examples/profiling dspqperf_imp.c, so a
 * DSP image without dspqueue still loads this skel and start returns
 * AEE_EUNSUPPORTED; test/htp/build.sh fails the build on a non-weak
 * dspqueue_* import. AEE_EBADPARM stays the stale-skel symptom (rule 3).
 *
 * Request message {u32 op, u32 seq}, op 1 = echo, 2 = quit, with 0 or 2
 * buffer references (in, out). Response message {u32 seq, u32 status}; an
 * echo with 2 references copies in -> out and hands out back with
 * FLUSH_SENDER | INVALIDATE_RECIPIENT (dspqueue_sample_imp.c's pattern).
 * A malformed packet counts as bad and is still answered, so the ARM side
 * never hangs on it.
 *
 * Address space: 16 KiB of DSP heap for the thread stack while a bench
 * queue is open, freed by stop.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <qurt.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "dspqueue.h"
#include "nntr_hvx.h"

#pragma weak dspqueue_import
#pragma weak dspqueue_close
#pragma weak dspqueue_write
#pragma weak dspqueue_read
#pragma weak dspqueue_read_noblock

#define DSPQ_BENCH_STACK (16 * 1024)
#define DSPQ_OP_ECHO 1u
#define DSPQ_OP_QUIT 2u
#define DSPQ_READ_TIMEOUT_US 100000u

/* ponytail: one bench queue per PD in file-static state, so
 * nntr_hvx_session.h stays untouched; a thread leaked by a PD that dies
 * without stop dies with the PD. Step 2 of #141 moves this into the
 * session if dspqueue is adopted. */
static struct {
  dspqueue_t q;
  qurt_thread_t tid;
  void *stack;
  uint32_t mode;
  volatile int stop;
  uint32_t served, empty, bad;
} g_dspq;

static void dspq_bench_thread(void *arg) {
  (void)arg;
  while (!g_dspq.stop) {
    uint32_t flags = 0, nb = 0, len = 0;
    uint32_t msg[2] = {0, 0};
    struct dspqueue_buffer bufs[2];
    memset(bufs, 0, sizeof(bufs));
    int err;
    if (g_dspq.mode == 0) {
      err = dspqueue_read(g_dspq.q, &flags, 2, &nb, bufs, sizeof(msg), &len,
                          (uint8_t *)msg, DSPQ_READ_TIMEOUT_US);
      if (err == AEE_EEXPIRED) {
        continue;
      }
    } else {
      err = dspqueue_read_noblock(g_dspq.q, &flags, 2, &nb, bufs, sizeof(msg),
                                  &len, (uint8_t *)msg);
      if (err == AEE_EWOULDBLOCK) {
        if ((++g_dspq.empty & 4095u) == 0 && g_dspq.stop) {
          break;
        }
        continue;
      }
    }
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "dspq_bench: read failed: 0x%08x", (unsigned)err);
      ++g_dspq.bad;
      break;
    }
    if (len == sizeof(msg) && msg[0] == DSPQ_OP_QUIT) {
      break;
    }

    uint32_t status = 0;
    if (len != sizeof(msg) || msg[0] != DSPQ_OP_ECHO || nb == 1) {
      status = 1;
    } else if (nb == 2) {
      if (bufs[0].ptr == NULL || bufs[1].ptr == NULL ||
          bufs[0].size != bufs[1].size) {
        status = 2;
      } else {
        memcpy(bufs[1].ptr, bufs[0].ptr, bufs[0].size);
      }
    }
    if (status != 0) {
      ++g_dspq.bad;
    }
    // Every reference the request took is released, malformed or not.
    for (uint32_t i = 0; i < nb; ++i) {
      bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
    }
    if (nb == 2 && status == 0) {
      bufs[1].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                       DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    }
    const uint32_t resp[2] = {msg[1], status};
    err = dspqueue_write(g_dspq.q, 0, nb, bufs, sizeof(resp),
                         (const uint8_t *)resp, DSPQUEUE_TIMEOUT_NONE);
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "dspq_bench: write failed: 0x%08x", (unsigned)err);
      ++g_dspq.bad;
      break;
    }
    ++g_dspq.served;
  }
  qurt_thread_exit(QURT_EOK);
}

int nntr_hvx_dspq_bench_start(remote_handle64 handle, uint64 queue_id,
                              uint32 mode) {
  (void)handle;
  if (!dspqueue_import || !dspqueue_close || !dspqueue_write ||
      !dspqueue_read || !dspqueue_read_noblock) {
    return AEE_EUNSUPPORTED;
  }
  if (mode > 1) {
    return AEE_EINVALIDFORMAT;
  }
  if (g_dspq.q != NULL) {
    return AEE_EBADSTATE;
  }
  // No packet callback: blocking reads are refused once one is set.
  int err = dspqueue_import(queue_id, NULL, NULL, NULL, &g_dspq.q);
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "dspq_bench: dspqueue_import failed: 0x%08x", (unsigned)err);
    g_dspq.q = NULL;
    return err;
  }
  g_dspq.stack = malloc(DSPQ_BENCH_STACK);
  if (g_dspq.stack == NULL) {
    dspqueue_close(g_dspq.q);
    g_dspq.q = NULL;
    return AEE_ENOMEMORY;
  }
  g_dspq.mode = mode;
  g_dspq.stop = 0;
  g_dspq.served = g_dspq.empty = g_dspq.bad = 0;

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "dspq_bench");
  qurt_thread_attr_set_stack_addr(&attr, g_dspq.stack);
  qurt_thread_attr_set_stack_size(&attr, DSPQ_BENCH_STACK);
  int prio = qurt_thread_get_priority(qurt_thread_get_id()) + 1;
  qurt_thread_attr_set_priority(&attr,
                                (unsigned short)(prio > 254 ? 254 : prio));
  err = qurt_thread_create(&g_dspq.tid, &attr, dspq_bench_thread, NULL);
  if (err != QURT_EOK) {
    FARF(ERROR, "dspq_bench: qurt_thread_create failed: %d", err);
    dspqueue_close(g_dspq.q);
    g_dspq.q = NULL;
    free(g_dspq.stack);
    g_dspq.stack = NULL;
    return AEE_EQURTTHREADCREATE;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_dspq_bench_stop(remote_handle64 handle, uint32 *res, int resLen) {
  (void)handle;
  if (resLen < 4) {
    return AEE_EINVALIDFORMAT;
  }
  if (g_dspq.q == NULL) {
    return AEE_EBADSTATE;
  }
  g_dspq.stop = 1;
  int status;
  qurt_thread_join(g_dspq.tid, &status);
  const int err = dspqueue_close(g_dspq.q);
  g_dspq.q = NULL;
  free(g_dspq.stack);
  g_dspq.stack = NULL;
  res[0] = g_dspq.served;
  res[1] = g_dspq.empty;
  res[2] = g_dspq.bad;
  res[3] = g_dspq.mode;
  return err;
}
