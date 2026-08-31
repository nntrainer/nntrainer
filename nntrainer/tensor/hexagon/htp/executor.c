// SPDX-License-Identifier: Apache-2.0
/**
 * @file	executor.c
 * @date	15 August 2026
 * @brief	DSP-side FastRPC glue: validates the op-list, persistently
 *		maps the init-time dma-buf fds, and delegates forward() to the
 *		htp_graph executor. An empty op-list (n_ops == 0) keeps the
 *		M1 deterministic dummy pattern for the round-trip test.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"

#include "htp_graph.h"
#include "nntr_htp.h"
#include "nntr_htp_common.h"

struct session {
  void *weights;
  void *kv;
  void *act;
  uint32 weights_size;
  uint32 kv_size;
  uint32 act_size;
  uint8 *oplist_copy; /* graph->ops points into this heap copy */
  struct htp_graph graph;
  int graph_ready;
};

static void destroy_graph(struct session *s) {
  if (s->graph_ready)
    htp_graph_destroy(&s->graph);
  s->graph_ready = 0;
  free(s->oplist_copy);
  s->oplist_copy = 0;
}

static void unmap_all(struct session *s) {
  if (s->weights)
    HAP_munmap(s->weights, (int)s->weights_size);
  if (s->kv)
    HAP_munmap(s->kv, (int)s->kv_size);
  if (s->act)
    HAP_munmap(s->act, (int)s->act_size);
  s->weights = s->kv = s->act = 0;
}

static void *map_fd(int32 fd, uint32 size) {
  void *p = HAP_mmap(0, (int)size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
  return (p == (void *)-1) ? 0 : p;
}

AEEResult nntr_htp_open(const char *uri, remote_handle64 *h) {
  struct session *s = calloc(1, sizeof(*s));
  (void)uri;
  if (!s)
    return AEE_ENOMEMORY;
  *h = (remote_handle64)(uintptr_t)s;
  FARF(ALWAYS, "nntr_htp: open, abi v%u", NNTR_HTP_ABI_VERSION);
  return AEE_SUCCESS;
}

AEEResult nntr_htp_close(remote_handle64 h) {
  struct session *s = (struct session *)(uintptr_t)h;
  destroy_graph(s); /* before unmap: the graph references the mappings */
  unmap_all(s);
  free(s);
  FARF(ALWAYS, "nntr_htp: close");
  return AEE_SUCCESS;
}

AEEResult nntr_htp_init(remote_handle64 h, const uint8 *oplist, int oplistLen,
                        const uint8 *weights, int weightsLen, int32 weights_fd,
                        int32 kv_fd, uint32 kv_size, int32 act_fd,
                        uint32 act_size, uint32 *dsp_abi_version) {
  struct session *s = (struct session *)(uintptr_t)h;
  struct nntr_htp_oplist_header hdr;
  int rc;

  (void)weights; /* in-sequence only forces the driver cache flush */
  *dsp_abi_version = NNTR_HTP_ABI_VERSION;
  if (oplistLen < 0 || weightsLen < (int)sizeof(int32_t))
    return AEE_EBADPARM;
  rc = nntr_htp_oplist_check(oplist, (uint32)oplistLen);
  if (rc != 0) {
    FARF(ERROR, "nntr_htp: op-list rejected (rc=%d)", rc);
    return rc == 3 ? AEE_EUNSUPPORTED : AEE_EBADPARM;
  }
  memcpy(&hdr, oplist, sizeof(hdr));

  destroy_graph(s); /* re-init replaces any previous graph and mapping */
  unmap_all(s);
  s->weights_size = (uint32)weightsLen;
  s->kv_size = kv_size;
  s->act_size = act_size;
  s->weights = map_fd(weights_fd, s->weights_size);
  s->kv = map_fd(kv_fd, kv_size);
  s->act = map_fd(act_fd, act_size);
  if (!s->weights || !s->kv || !s->act) {
    FARF(ERROR, "nntr_htp: HAP_mmap failed (w=%p kv=%p act=%p)", s->weights,
         s->kv, s->act);
    unmap_all(s);
    return AEE_ENOMEMORY;
  }

  /* n_ops == 0 keeps the M1 dummy forward path: no graph is built. */
  if (hdr.n_ops > 0u) {
    s->oplist_copy = malloc((size_t)oplistLen);
    if (!s->oplist_copy) {
      unmap_all(s);
      return AEE_ENOMEMORY;
    }
    memcpy(s->oplist_copy, oplist, (size_t)oplistLen);
    rc = htp_graph_init(&s->graph, s->oplist_copy, (uint32)oplistLen,
                        (uint8_t *)s->weights, s->weights_size,
                        (uint8_t *)s->kv, kv_size, (uint8_t *)s->act, act_size);
    if (rc != 0) {
      FARF(ERROR, "nntr_htp: graph init failed (rc=%d)", rc);
      destroy_graph(s); /* frees the copy; graph_ready is still 0 */
      unmap_all(s);
      return rc == 3 ? AEE_EUNSUPPORTED : AEE_EBADPARM;
    }
    s->graph_ready = 1;
  }
  FARF(ALWAYS, "nntr_htp: init ok, weights=%d kv=%u act=%u n_ops=%u",
       weightsLen, kv_size, act_size, hdr.n_ops);
  return AEE_SUCCESS;
}

AEEResult nntr_htp_forward(remote_handle64 h, const int32 *token_ids,
                           int token_idsLen, uint32 pos, float *logits,
                           int logitsLen, uint64 *dsp_pcycles) {
  struct session *s = (struct session *)(uintptr_t)h;
  int32_t w0;
  int i;

  *dsp_pcycles = 0;
  if (!s->weights)
    return AEE_EBADSTATE;
  if (token_idsLen <= 0 || logitsLen <= 0)
    return AEE_EBADPARM;

  if (s->graph_ready) {
    if (htp_graph_forward_upto(&s->graph, token_ids, (uint32)token_idsLen, pos,
                               logits, (uint32)logitsLen, s->graph.cfg.n_ops,
                               dsp_pcycles) != 0)
      return AEE_EBADPARM;
    return AEE_SUCCESS;
  }

  w0 = ((const int32_t *)s->weights)[0];

  // Dummy pattern - must match test/hexagon/hexagon_rpc_test.cpp:
  // logits[i] = token_ids[i % n] + pos + i + weights[0]
  for (i = 0; i < logitsLen; ++i)
    logits[i] = (float)(token_ids[i % token_idsLen] + (int32_t)pos + i + w0);
  return AEE_SUCCESS;
}

AEEResult nntr_htp_forward_debug(remote_handle64 h, const int32 *token_ids,
                                 int token_idsLen, uint32 pos,
                                 uint32 n_ops_limit, uint32 dump_buf,
                                 uint32 dump_offset, uint8 *dump, int dumpLen,
                                 uint64 *dsp_pcycles) {
  struct session *s = (struct session *)(uintptr_t)h;
  float *logits;
  const uint8_t *src;
  int rc;

  *dsp_pcycles = 0;
  if (!s->weights || !s->graph_ready)
    return AEE_EBADSTATE;
  if (token_idsLen <= 0 || dumpLen < 0)
    return AEE_EBADPARM;
  src = htp_graph_buf_ref(&s->graph, dump_buf, dump_offset, (uint32)dumpLen);
  if (!src || n_ops_limit > s->graph.cfg.n_ops)
    return AEE_EBADPARM;

  /* A partial run may stop before MATMUL_LOGITS; give it a scratch target
   * so the op-list's LOGITS ref stays valid either way. */
  logits = malloc((size_t)s->graph.cfg.vocab * sizeof(float));
  if (!logits)
    return AEE_ENOMEMORY;
  rc = htp_graph_forward_upto(&s->graph, token_ids, (uint32)token_idsLen, pos,
                              logits, s->graph.cfg.vocab, n_ops_limit,
                              dsp_pcycles);
  free(logits);
  if (rc != 0)
    return AEE_EBADPARM;
  memcpy(dump, src, (size_t)dumpLen);
  return AEE_SUCCESS;
}
