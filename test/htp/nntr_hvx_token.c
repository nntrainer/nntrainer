// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_token.c
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2] token_driver_start / token_driver_stop and the
 *         session's token call: the mailbox page mapped once, the role
 *         (0 = S2, the main session; 1 = S1, the MoE server) and the
 *         counters, over hexkl_token.c
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/132-part-b-two-session-e2e.md sections 3.2 and 3.3. The
 * ARM side allocates one ION page, fastrpc_mmap's it into both sessions
 * and calls token_driver_start on each with its role; every token is then
 * one HTP_DSPQ_OP_TOKEN packet per session (nntr_hvx_dspq.c), whose
 * thread calls nntr_hvx_token_run -- so the waits run on the sessions'
 * own dspq threads, never on a pool lane. The graph is the session's
 * (graph_init with the role's mask), read at each token.
 *
 * Address space: the page's mapping (HAP_mmap_get, at least
 * HEXKL_MBOX_BYTES = 16.5 KiB) and this struct, both released by stop or
 * close.
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <remote.h>

#include "hexkl_token.h"
#include "htp_dspq_wire.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief The session's driver: its page and its side's counters. */
struct nntr_hvx_token {
  int fd;
  uint8_t *page;
  uint32_t role, spin_us;
  hexkl_token_stats st;
};

int nntr_hvx_token_driver_start(remote_handle64 handle, int32 mbox_fd,
                                uint32 mbox_bytes, uint32 role,
                                uint32 spin_us) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  struct nntr_hvx_token *t;
  void *va = NULL;
  uint64 pa = 0;
  int rc;
  if (s == NULL) {
    return AEE_EBADSTATE;
  }
  if (s->token != NULL) {
    FARF(ERROR, "token_driver_start: a driver is already running");
    return AEE_EBADSTATE;
  }
  if (role > 1u || mbox_bytes < HEXKL_MBOX_BYTES) {
    FARF(ERROR, "token_driver_start: role %u, page %u B (want >= %u)",
         (unsigned)role, (unsigned)mbox_bytes, (unsigned)HEXKL_MBOX_BYTES);
    return AEE_EINVALIDFORMAT;
  }
  rc = HAP_mmap_get((int)mbox_fd, &va, &pa);
  if (rc != 0 || va == NULL) {
    FARF(ERROR, "token_driver_start: HAP_mmap_get(%d): 0x%08x", (int)mbox_fd,
         (unsigned)rc);
    return rc != 0 ? rc : AEE_ENOMEMORY;
  }
  t = (struct nntr_hvx_token *)calloc(1, sizeof(*t));
  if (t == NULL) {
    HAP_mmap_put((int)mbox_fd);
    return AEE_ENOMEMORY;
  }
  t->fd = (int)mbox_fd;
  t->page = (uint8_t *)va;
  t->role = role;
  t->spin_us = spin_us;
  s->token = t;
  FARF(HIGH, "[token] start role=%s spin_us=%u page=%u B", role ? "S1" : "S2",
       (unsigned)spin_us, (unsigned)mbox_bytes);
  return AEE_SUCCESS;
}

/** @brief Unmaps and frees; @a res (5) gets the counters when not NULL. */
static void token_teardown(nntr_hvx_session *s, uint32 *res) {
  struct nntr_hvx_token *t = s->token;
  if (res != NULL) {
    res[0] = t->st.tokens;
    res[1] = t->st.hops;
    res[2] = t->st.timeouts;
    res[3] = t->st.stale;
    res[4] = t->st.wait_us;
  }
  HAP_mmap_put(t->fd);
  free(t);
  s->token = NULL;
}

int nntr_hvx_token_driver_stop(remote_handle64 handle, uint32 *res,
                               int resLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (s == NULL || s->token == NULL) {
    return AEE_EBADSTATE;
  }
  if (res == NULL || resLen != 5) {
    return AEE_EINVALIDFORMAT;
  }
  token_teardown(s, res);
  return AEE_SUCCESS;
}

void nntr_hvx_token_shutdown(nntr_hvx_session *s) {
  if (s != NULL && s->token != NULL) {
    token_teardown(s, NULL);
  }
}

int nntr_hvx_token_run(nntr_hvx_session *s, uint32_t tok, uint32_t pos,
                       const float *act, uint32_t act_len, float *logits,
                       uint32_t logits_len, struct htp_dspq_token_resp_s *r) {
  struct nntr_hvx_token *t = s ? s->token : NULL;
  hexkl_graph_env env;
  uint32_t id = 0, k;
  int rc;
  if (t == NULL || s->graph == NULL) {
    return AEE_EBADSTATE;
  }
  const hexkl_token_stats before = t->st;
  const uint64_t us0 = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
  const uint64_t pc0 = HAP_perf_get_pcycles();
  nntr_hvx_graph_env(s, &env);
  if (t->role == 0u) {
    if (act == NULL) {
      return AEE_EINVALIDFORMAT;
    }
    rc = hexkl_token_main(s->graph, &env, t->page, tok, pos, act, act_len,
                          logits, logits_len, t->spin_us, &t->st, &id);
  } else {
    rc =
      hexkl_token_serve(s->graph, &env, t->page, tok, pos, t->spin_us, &t->st);
  }
  r->wall_pcyc = (uint32_t)(HAP_perf_get_pcycles() - pc0);
  r->wall_us =
    (uint32_t)(HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count()) - us0);
  r->id = id;
  r->hops = t->st.hops - before.hops;
  r->wait_us = t->st.wait_us - before.wait_us;
  r->hop_us = t->st.hop_us - before.hop_us;
  r->pcycles = (uint32_t)(t->st.pcycles - before.pcycles);
  for (k = 0; k < HTP_DSPQ_TOKEN_KINDS && k < HTP_OP_KIND_N; ++k) {
    r->kind_pcyc[k] =
      (uint32_t)(t->st.kind_pcycles[k] - before.kind_pcycles[k]);
  }
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "[token] %s tok=%u pos=%u: 0x%08x (timeouts %u stale %u)",
         t->role ? "S1" : "S2", (unsigned)tok, (unsigned)pos, (unsigned)rc,
         (unsigned)t->st.timeouts, (unsigned)t->st.stale);
  }
  return rc;
}
