// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_dspq.c
 * @date   28 Sep 2026
 * @brief  [#141] The M==1 MoE layer call over dspqueue: one DSP thread
 *         parked on a queue the ARM side exported, calling the FastRPC
 *         method's own C function (nntr_hvx_mm_u8i4_moe_layer[_timed])
 *         with the packet's arguments
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/141-dspq-moe.md sections 3.2-3.4; the packet is
 * htp_dspq_wire.h. [#132 Part B E2] The same thread answers
 * HTP_DSPQ_OP_TOKEN, one decode token of the session's token driver role
 * (nntr_hvx_token.c): on S1 it waits for S2's rows on the mailbox page
 * between its rounds, on S2 it runs the token's graph -- both sessions'
 * own threads, not pool lanes (plan 132-part-b section 3.2). Same entry
 * function, same argument bytes, same session state, so the arithmetic is the
 * FastRPC path's bit for bit: the kernel resets its DMA ring per call and the
 * pool splits jobs by index, so which thread is lane 0 does not enter the
 * result. The ARM side keeps at most one DSP entry in flight (its invoke mutex
 * spans write -> response), so this thread and the FastRPC threads never run a
 * kernel at once.
 *
 * Waiting: after each response the thread spins on read_noblock with
 * pause(#255) for spin_us (the decode gaps between two MoE calls), then
 * blocks in dspqueue_read with a 100 ms timeout that re-checks the stop
 * flag. spin_us = 0 always blocks.
 *
 * The dspqueue_* symbols are weak (nntr_hvx_dspq_bench.c's five): a DSP
 * image without dspqueue still loads the skel and start returns
 * AEE_EUNSUPPORTED, which the ARM side turns into its "dspq: off" line.
 *
 * Address space: a 64 KiB thread stack of DSP heap while a queue is open
 * (lane 0 of the M=1 GEMV runs on it; the pool's workers get 32 KiB), and
 * the session's struct nntr_hvx_dspq. Both are freed by stop or close.
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "dspqueue.h"
#include "htp_dspq_wire.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#pragma weak dspqueue_import
#pragma weak dspqueue_close
#pragma weak dspqueue_write
#pragma weak dspqueue_read
#pragma weak dspqueue_read_noblock

#define DSPQ_STACK (64u * 1024u)
#define DSPQ_BLOCK_TIMEOUT_US 100000u

/** @brief The session's queue: its thread and the counters stop returns. */
struct nntr_hvx_dspq {
  nntr_hvx_session *s;
  dspqueue_t q;
  qurt_thread_t tid;
  void *stack;
  uint32_t spin_us;
  volatile int stop;
  uint32_t served, bad, empty;
};

/** @brief The request message, readable as u32 and as f32 (row_weight). */
typedef union {
  uint32_t u[HTP_DSPQ_MAX_MSG / 4];
  float f[HTP_DSPQ_MAX_MSG / 4];
  htp_dspq_req_hdr h;
} dspq_msg;

static inline void dspq_pause(void) {
#if defined(__hexagon__)
  asm volatile(" pause(#255)\n");
#endif
}

static inline uint64_t dspq_now_us(void) {
  return HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
}

/** @brief The packet agrees with its own header (section 3.2 step 2); the
 *  kernel's own argument checks run after this, as under FastRPC. */
static int dspq_valid(const dspq_msg *m, uint32_t len, uint32_t nb,
                      const struct dspqueue_buffer *bufs) {
  const htp_dspq_req_hdr *h = &m->h;
  return len >= sizeof(*h) && h->op == HTP_DSPQ_OP_MOE &&
         (h->flags & ~HTP_DSPQ_FLAG_TIMED) == 0 &&
         len == htp_dspq_req_bytes(h->n_experts, h->n_rows) && nb == 2 &&
         bufs[0].ptr != NULL && bufs[1].ptr != NULL &&
         bufs[0].size == (uint64_t)h->M * h->K * sizeof(float) &&
         bufs[1].size == (uint64_t)h->M * h->N_out * sizeof(float);
}

/** @brief Runs one valid packet through the FastRPC method's function. */
static int dspq_run(struct nntr_hvx_dspq *d, const dspq_msg *m,
                    const struct dspqueue_buffer *bufs, htp_dspq_resp *resp) {
  const htp_dspq_req_hdr *h = &m->h;
  const uint32_t ne = h->n_experts, nr = h->n_rows;
  const uint32_t o = sizeof(*h) / 4;
  const uint32_t *h_gu = &m->u[o];
  const uint32_t *h_dn = &m->u[o + ne];
  const uint32_t *row_count = &m->u[o + 2 * ne];
  const uint32_t *row_index = &m->u[o + 3 * ne];
  const float *row_weight = &m->f[o + 3 * ne + nr];
  const remote_handle64 hs = (remote_handle64)d->s;
  const int act_len = (int)(h->M * h->K), out_len = (int)(h->M * h->N_out);
  if (h->flags & HTP_DSPQ_FLAG_TIMED) {
    return nntr_hvx_mm_u8i4_moe_layer_timed(
      hs, h->M, h->K, h->inter, h->N_out, h_gu, (int)ne, h_dn, (int)ne,
      row_index, (int)nr, row_count, (int)ne, row_weight, (int)nr,
      (const float *)bufs[0].ptr, act_len, (float *)bufs[1].ptr, out_len,
      resp->stage_us, (int)HTP_DSPQ_STAGES);
  }
  return nntr_hvx_mm_u8i4_moe_layer(
    hs, h->M, h->K, h->inter, h->N_out, h_gu, (int)ne, h_dn, (int)ne, row_index,
    (int)nr, row_count, (int)ne, row_weight, (int)nr,
    (const float *)bufs[0].ptr, act_len, (float *)bufs[1].ptr, out_len);
}

/** @brief [#132 Part B E2] Answers one HTP_DSPQ_OP_TOKEN packet: the
 *  session's token driver role on the packet's buffers (S2: 0 the
 *  embedding row, 1 the logits under HTP_DSPQ_TOKEN_LOGITS; S1: none).
 *  A malformed packet is answered AEE_EBADPARM, as a MoE packet is.
 *  @return dspqueue_write's code */
static int dspq_token(struct nntr_hvx_dspq *d, const dspq_msg *m, uint32_t len,
                      uint32_t nb, struct dspqueue_buffer *bufs) {
  const htp_dspq_token_req *q = (const htp_dspq_token_req *)m;
  htp_dspq_token_resp resp;
  uint32_t res[4] = {0, 0, 0, 0}, i;
  const int logits = len == sizeof(*q) && (q->flags & HTP_DSPQ_TOKEN_LOGITS);
  const int valid =
    len == sizeof(*q) && (q->flags & ~HTP_DSPQ_TOKEN_LOGITS) == 0u &&
    nb <= 2u && (nb == 2u) == (logits != 0) &&
    (nb < 1u || (bufs[0].ptr != NULL && bufs[0].size % 4u == 0u)) &&
    (nb < 2u || (bufs[1].ptr != NULL && bufs[1].size % 4u == 0u));
  memset(&resp, 0, sizeof(resp));
  resp.seq = len >= 8 ? m->u[1] : 0;
  if (valid) {
    resp.rc = nntr_hvx_token_run(
      d->s, q->seq, q->pos, nb >= 1u ? (const float *)bufs[0].ptr : NULL,
      nb >= 1u ? (uint32_t)(bufs[0].size / 4u) : 0u,
      nb == 2u ? (float *)bufs[1].ptr : NULL,
      nb == 2u ? (uint32_t)(bufs[1].size / 4u) : 0u, res);
  } else {
    resp.rc = AEE_EBADPARM;
    ++d->bad;
  }
  resp.id = res[0];
  resp.hops = res[1];
  resp.wait_us = res[2];
  resp.pcycles = res[3];
  for (i = 0; i < nb; ++i) {
    bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
  }
  if (nb == 2u) {
    bufs[1].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                     DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
  }
  return dspqueue_write(d->q, 0, nb, bufs, sizeof(resp), (const uint8_t *)&resp,
                        DSPQUEUE_TIMEOUT_NONE);
}

static void dspq_thread(void *arg) {
  struct nntr_hvx_dspq *d = (struct nntr_hvx_dspq *)arg;
  dspq_msg m;
  uint64_t spin_until = 0; /* 0: block on the next read */
  while (!d->stop) {
    uint32_t flags = 0, nb = 0, len = 0;
    struct dspqueue_buffer bufs[2];
    memset(bufs, 0, sizeof(bufs));
    int err;
    if (spin_until != 0) {
      err = dspqueue_read_noblock(d->q, &flags, 2, &nb, bufs, sizeof(m), &len,
                                  (uint8_t *)&m);
      if (err == AEE_EWOULDBLOCK) {
        ++d->empty;
        dspq_pause();
        if (dspq_now_us() >= spin_until) {
          spin_until = 0;
        }
        continue;
      }
    } else {
      err = dspqueue_read(d->q, &flags, 2, &nb, bufs, sizeof(m), &len,
                          (uint8_t *)&m, DSPQ_BLOCK_TIMEOUT_US);
      if (err == AEE_EEXPIRED) {
        continue;
      }
    }
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "dspq: read failed: 0x%08x", (unsigned)err);
      ++d->bad;
      break;
    }
    if (len >= 4 && m.u[0] == HTP_DSPQ_OP_QUIT && nb == 0) {
      break;
    }
    if (len >= 4 && m.u[0] == HTP_DSPQ_OP_TOKEN) {
      err = dspq_token(d, &m, len, nb, bufs);
      if (err != AEE_SUCCESS) {
        FARF(ERROR, "dspq: write failed: 0x%08x", (unsigned)err);
        ++d->bad;
        break;
      }
      ++d->served;
      spin_until = d->spin_us ? dspq_now_us() + d->spin_us : 0;
      continue;
    }

    // A malformed packet is answered too, so the ARM side never hangs on it.
    htp_dspq_resp resp;
    memset(&resp, 0, sizeof(resp));
    resp.seq = len >= 8 ? m.u[1] : 0;
    const int valid = dspq_valid(&m, len, nb, bufs);
    if (valid) {
      resp.rc = dspq_run(d, &m, bufs, &resp);
    } else {
      resp.rc = AEE_EBADPARM;
      ++d->bad;
    }
    const uint32_t rlen =
      HTP_DSPQ_RESP_BASE_BYTES + ((valid && (m.h.flags & HTP_DSPQ_FLAG_TIMED))
                                    ? HTP_DSPQ_STAGES * 4u
                                    : 0u);
    // Every reference the request took is released, malformed or not; the
    // output goes back flushed from the DSP's caches.
    for (uint32_t i = 0; i < nb; ++i) {
      bufs[i].flags = DSPQUEUE_BUFFER_FLAG_DEREF;
    }
    if (nb == 2) {
      bufs[1].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                       DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    }
    err = dspqueue_write(d->q, 0, nb, bufs, rlen, (const uint8_t *)&resp,
                         DSPQUEUE_TIMEOUT_NONE);
    if (err != AEE_SUCCESS) {
      FARF(ERROR, "dspq: write failed: 0x%08x", (unsigned)err);
      ++d->bad;
      break;
    }
    ++d->served;
    spin_until = d->spin_us ? dspq_now_us() + d->spin_us : 0;
  }
  qurt_thread_exit(0); /* QURT_EOK; the host stub has no qurt_error.h */
}

/** @brief Joins the thread, closes the queue, frees; NULL-safe. */
static int dspq_teardown(nntr_hvx_session *s, uint32_t res[4]) {
  struct nntr_hvx_dspq *d = s->dspq;
  if (d == NULL) {
    return AEE_EBADSTATE;
  }
  d->stop = 1;
  int status;
  qurt_thread_join(d->tid, &status);
  const int err = dspqueue_close(d->q);
  if (res != NULL) {
    res[0] = d->served;
    res[1] = d->bad;
    res[2] = d->empty;
    res[3] = d->spin_us;
  }
  free(d->stack);
  free(d);
  s->dspq = NULL;
  return err;
}

int nntr_hvx_dspq_start(remote_handle64 handle, uint64 queue_id,
                        uint32 spin_us) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!dspqueue_import || !dspqueue_close || !dspqueue_write ||
      !dspqueue_read || !dspqueue_read_noblock) {
    return AEE_EUNSUPPORTED;
  }
  if (s == NULL || s->dspq != NULL) {
    return AEE_EBADSTATE;
  }
  if (nntr_hvx_moe_stage_count() != HTP_DSPQ_STAGES) {
    FARF(ERROR, "dspq: stage slots %u, wire header says %u",
         (unsigned)nntr_hvx_moe_stage_count(), (unsigned)HTP_DSPQ_STAGES);
    return AEE_EINVALIDFORMAT;
  }
  struct nntr_hvx_dspq *d =
    (struct nntr_hvx_dspq *)calloc(1, sizeof(struct nntr_hvx_dspq));
  if (d == NULL) {
    return AEE_ENOMEMORY;
  }
  d->s = s;
  d->spin_us = spin_us;
  // No packet callback: blocking reads are refused once one is set.
  int err = dspqueue_import(queue_id, NULL, NULL, NULL, &d->q);
  if (err != AEE_SUCCESS) {
    FARF(ERROR, "dspq: dspqueue_import failed: 0x%08x", (unsigned)err);
    free(d);
    return err;
  }
  d->stack = malloc(DSPQ_STACK);
  if (d->stack == NULL) {
    dspqueue_close(d->q);
    free(d);
    return AEE_ENOMEMORY;
  }
  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_name(&attr, "nntr_dspq");
  qurt_thread_attr_set_stack_addr(&attr, d->stack);
  qurt_thread_attr_set_stack_size(&attr, DSPQ_STACK);
  // The caller's own priority: the one lane 0 runs at under FastRPC and
  // the one the worker pool was created with (hvx_worker_pool.c).
  qurt_thread_attr_set_priority(
    &attr, (unsigned short)qurt_thread_get_priority(qurt_thread_get_id()));
  err = qurt_thread_create(&d->tid, &attr, dspq_thread, d);
  if (err != 0) { /* QURT_EOK */
    FARF(ERROR, "dspq: qurt_thread_create failed: %d", err);
    dspqueue_close(d->q);
    free(d->stack);
    free(d);
    return AEE_EQURTTHREADCREATE;
  }
  s->dspq = d;
  return AEE_SUCCESS;
}

int nntr_hvx_dspq_stop(remote_handle64 handle, uint32 *res, int resLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (s == NULL || res == NULL || resLen < 4) {
    return AEE_EINVALIDFORMAT;
  }
  return dspq_teardown(s, res);
}

void nntr_hvx_dspq_shutdown(nntr_hvx_session *s) {
  if (s != NULL && s->dspq != NULL) {
    FARF(HIGH, "dspq: close() stops a queue the ARM side left open");
    dspq_teardown(s, NULL);
  }
}
