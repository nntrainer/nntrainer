// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   dspqueue_standin.c
 * @date   28 Sep 2026
 * @brief  [#141] Host stand-in for dspqueue on both ends: the ARM side's
 *         create / export / close / write / read, the DSP side's import,
 *         as one in-process object (in-process HTP build)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/141-dspq-moe.md section 3.6. Two fixed rings of 16
 * packets (ARM -> DSP and DSP -> ARM), each packet a 4 KiB message and up
 * to two buffer references, under one mutex and condition variable.
 * import(id) returns the DSP end of the object create made. A buffer
 * reference resolves fd -> address through rpc_standin.c and is refused
 * with AEE_ENOSUCHMAP when the fd was not fastrpc_mmap'd, the device's own
 * rule. Closing the ARM end while the DSP end is open is AEE_EBADPARM, as
 * dspqueue.h documents. NNTR_INPROC_NO_DSPQ=1 makes create fail, which
 * drives the ARM side's "dspq: off" path.
 *
 * What this proves: the packing, the unpacking, the checks, the spin ->
 * block switch, teardown and the off path, with the real nntr_hvx_dspq.c
 * thread on a pthread. What it cannot: transport time, DSP threads, cache
 * maintenance, power. Every symbol is exported with default visibility, as
 * rpc_standin.c's are, so HtpDspqApi's dlsym(RTLD_DEFAULT) finds them.
 */

#include <AEEStdErr.h>
#include <dspqueue.h>

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXPORT __attribute__((visibility("default")))
#define SQ_SLOTS 16u
#define SQ_MAX_MSG 4096u
#define SQ_MAX_BUFS 2u

void *rpc_standin_mapped_ptr(int fd); /* rpc_standin.c */

typedef struct {
  uint32_t len, nb;
  struct dspqueue_buffer bufs[SQ_MAX_BUFS];
  uint8_t msg[SQ_MAX_MSG];
} sq_packet;

typedef struct {
  sq_packet p[SQ_SLOTS];
  uint32_t head, count;
} sq_ring;

struct sq_queue;
/** @brief One end; side 0 = ARM (writes ring 0, reads ring 1), 1 = DSP. */
struct dspqueue {
  struct sq_queue *q;
  int side;
  int open;
};

struct sq_queue {
  pthread_mutex_t mu;
  pthread_cond_t cv;
  sq_ring ring[2];
  struct dspqueue end[2];
};

/** @brief Waits on the queue's condvar until timeout_us from start;
 *  returns ETIMEDOUT once it has passed. Caller holds mu. */
static int sq_wait(struct sq_queue *q, const struct timespec *deadline,
                   uint32_t timeout_us) {
  if (timeout_us == DSPQUEUE_TIMEOUT_NONE)
    return pthread_cond_wait(&q->cv, &q->mu);
  return pthread_cond_timedwait(&q->cv, &q->mu, deadline);
}

static struct timespec sq_deadline(uint32_t timeout_us) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += timeout_us / 1000000u;
  ts.tv_nsec += (long)(timeout_us % 1000000u) * 1000L;
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec += 1;
    ts.tv_nsec -= 1000000000L;
  }
  return ts;
}

EXPORT AEEResult dspqueue_create(int domain, uint32_t flags,
                                 uint32_t req_queue_size,
                                 uint32_t resp_queue_size,
                                 dspqueue_callback_t packet_callback,
                                 dspqueue_callback_t error_callback,
                                 void *callback_context, dspqueue_t *queue) {
  (void)domain, (void)flags, (void)req_queue_size, (void)resp_queue_size;
  (void)error_callback, (void)callback_context;
  const char *off = getenv("NNTR_INPROC_NO_DSPQ");
  if (off != NULL && atoi(off) == 1)
    return AEE_EUNSUPPORTED;
  if (packet_callback != NULL || queue == NULL)
    return AEE_EUNSUPPORTED; /* not modelled: nothing here uses one */
  struct sq_queue *q = (struct sq_queue *)calloc(1, sizeof(*q));
  if (q == NULL)
    return AEE_ENOMEMORY;
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->cv, NULL);
  for (int s = 0; s < 2; ++s) {
    q->end[s].q = q;
    q->end[s].side = s;
  }
  q->end[0].open = 1;
  *queue = &q->end[0];
  return AEE_SUCCESS;
}

EXPORT AEEResult dspqueue_export(dspqueue_t queue, uint64_t *queue_id) {
  if (queue == NULL || queue->side != 0 || queue_id == NULL)
    return AEE_EBADPARM;
  *queue_id = (uint64_t)(uintptr_t)queue->q;
  return AEE_SUCCESS;
}

EXPORT AEEResult dspqueue_import(uint64_t queue_id,
                                 dspqueue_callback_t packet_callback,
                                 dspqueue_callback_t error_callback,
                                 void *callback_context, dspqueue_t *queue) {
  (void)error_callback, (void)callback_context;
  struct sq_queue *q = (struct sq_queue *)(uintptr_t)queue_id;
  if (q == NULL || queue == NULL || packet_callback != NULL)
    return AEE_EBADPARM;
  pthread_mutex_lock(&q->mu);
  const int busy = q->end[1].open;
  q->end[1].open = 1;
  pthread_mutex_unlock(&q->mu);
  if (busy)
    return AEE_EITEMBUSY;
  *queue = &q->end[1];
  return AEE_SUCCESS;
}

EXPORT AEEResult dspqueue_close(dspqueue_t queue) {
  if (queue == NULL)
    return AEE_EBADPARM;
  struct sq_queue *q = queue->q;
  pthread_mutex_lock(&q->mu);
  if (queue->side == 0 && q->end[1].open) {
    pthread_mutex_unlock(&q->mu);
    return AEE_EBADPARM; /* dspqueue.h: still open on the DSP */
  }
  queue->open = 0;
  const int last = !q->end[0].open && !q->end[1].open;
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
  if (last) {
    pthread_cond_destroy(&q->cv);
    pthread_mutex_destroy(&q->mu);
    free(q);
  }
  return AEE_SUCCESS;
}

EXPORT AEEResult dspqueue_write(dspqueue_t queue, uint32_t flags,
                                uint32_t num_buffers,
                                struct dspqueue_buffer *buffers,
                                uint32_t message_length, const uint8_t *message,
                                uint32_t timeout_us) {
  (void)flags;
  if (queue == NULL || !queue->open || num_buffers > SQ_MAX_BUFS ||
      message_length > SQ_MAX_MSG || (num_buffers && buffers == NULL) ||
      (message_length && message == NULL))
    return AEE_EBADPARM;
  sq_packet pk;
  pk.len = message_length;
  pk.nb = num_buffers;
  for (uint32_t i = 0; i < num_buffers; ++i) {
    uint8_t *base = (uint8_t *)rpc_standin_mapped_ptr((int)buffers[i].fd);
    if (base == NULL)
      return AEE_ENOSUCHMAP;
    pk.bufs[i] = buffers[i];
    pk.bufs[i].ptr = base + buffers[i].offset; /* the recipient's address */
  }
  if (message_length)
    memcpy(pk.msg, message, message_length);

  struct sq_queue *q = queue->q;
  sq_ring *r = &q->ring[queue->side];
  const struct timespec dl = sq_deadline(timeout_us);
  int rc = AEE_SUCCESS;
  pthread_mutex_lock(&q->mu);
  while (r->count == SQ_SLOTS) {
    if (timeout_us == 0 || sq_wait(q, &dl, timeout_us) == ETIMEDOUT) {
      rc = timeout_us == 0 ? AEE_EWOULDBLOCK : AEE_EEXPIRED;
      break;
    }
  }
  if (rc == AEE_SUCCESS) {
    r->p[(r->head + r->count) % SQ_SLOTS] = pk;
    ++r->count;
    pthread_cond_broadcast(&q->cv);
  }
  pthread_mutex_unlock(&q->mu);
  return rc;
}

EXPORT AEEResult dspqueue_read(dspqueue_t queue, uint32_t *flags,
                               uint32_t max_buffers, uint32_t *num_buffers,
                               struct dspqueue_buffer *buffers,
                               uint32_t max_message_length,
                               uint32_t *message_length, uint8_t *message,
                               uint32_t timeout_us) {
  if (queue == NULL || !queue->open)
    return AEE_EBADPARM;
  struct sq_queue *q = queue->q;
  sq_ring *r = &q->ring[1 - queue->side];
  const struct timespec dl = sq_deadline(timeout_us);
  int rc = AEE_SUCCESS;
  pthread_mutex_lock(&q->mu);
  while (r->count == 0) {
    if (timeout_us == 0 || sq_wait(q, &dl, timeout_us) == ETIMEDOUT) {
      rc = timeout_us == 0 ? AEE_EWOULDBLOCK : AEE_EEXPIRED;
      break;
    }
  }
  if (rc == AEE_SUCCESS) {
    const sq_packet *pk = &r->p[r->head];
    if (pk->nb > max_buffers || pk->len > max_message_length) {
      rc = AEE_EBADPARM; /* left in the queue, as the device leaves it */
    } else {
      *flags = (pk->len ? DSPQUEUE_PACKET_FLAG_MESSAGE : 0u) |
               (pk->nb ? DSPQUEUE_PACKET_FLAG_BUFFERS : 0u);
      *num_buffers = pk->nb;
      if (pk->nb)
        memcpy(buffers, pk->bufs, pk->nb * sizeof(pk->bufs[0]));
      *message_length = pk->len;
      if (pk->len)
        memcpy(message, pk->msg, pk->len);
      r->head = (r->head + 1) % SQ_SLOTS;
      --r->count;
      pthread_cond_broadcast(&q->cv);
    }
  }
  pthread_mutex_unlock(&q->mu);
  return rc;
}

EXPORT AEEResult dspqueue_read_noblock(
  dspqueue_t queue, uint32_t *flags, uint32_t max_buffers,
  uint32_t *num_buffers, struct dspqueue_buffer *buffers,
  uint32_t max_message_length, uint32_t *message_length, uint8_t *message) {
  return dspqueue_read(queue, flags, max_buffers, num_buffers, buffers,
                       max_message_length, message_length, message, 0);
}

/* HtpDspqApi resolves it; nothing in the model path calls it. */
EXPORT AEEResult dspqueue_get_stat(dspqueue_t queue, enum dspqueue_stat stat,
                                   uint64_t *value) {
  (void)queue, (void)stat, (void)value;
  return AEE_EUNSUPPORTED;
}
