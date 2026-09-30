// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_attn_m1.c
 * @date   27 Sep 2026
 * @brief  DSP-side entries for decode attention at m=1 with the session's
 *         fp16 KV cache (hvx_attn_m1_f32.c): register / release / kv_append
 *         / forward
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Each entry validates the FastRPC lengths the kernel cannot see, then
 * calls the kernel on the caller's buffers and the session's cache. A bad
 * shape is AEE_EINVALIDFORMAT, a hole or a missing cache AEE_EBADSTATE;
 * AEE_EBADPARM stays the stale-skel symptom (rule 3), so the gtest can
 * tell the two apart. One FARF line per register / release, since the
 * cache is the session's largest heap object (plan 81 section 3.1).
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "attn_m1_det.h"
#include "hvx_attn_m1_f32.h"

int nntr_hvx_attn_m1_register(remote_handle64 handle, uint32 n_layers,
                              uint32 n_kv, uint32 gqa, uint32 head_dim,
                              uint32 max_seq) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (s->attn_m1) {
    FARF(ERROR, "attn_m1_register: a cache is already registered");
    return AEE_EBADSTATE;
  }
  int err = AEE_SUCCESS;
  s->attn_m1 = hvx_attn_m1_create(n_layers, n_kv, gqa, head_dim, max_seq,
                                  s->quant_pool, &err);
  if (!s->attn_m1) {
    FARF(ERROR,
         "attn_m1_register: create failed 0x%08x (layers=%u kv=%u gqa=%u "
         "head_dim=%u max_seq=%u)",
         (unsigned)err, (unsigned)n_layers, (unsigned)n_kv, (unsigned)gqa,
         (unsigned)head_dim, (unsigned)max_seq);
    return err;
  }
  FARF(HIGH,
       "attn_m1_register: layers=%u kv=%u gqa=%u head_dim=%u max_seq=%u "
       "cache=%u KiB",
       (unsigned)n_layers, (unsigned)n_kv, (unsigned)gqa, (unsigned)head_dim,
       (unsigned)max_seq,
       (unsigned)(s->attn_m1->cache_halves * 2u * sizeof(uint16_t) / 1024u));
  return AEE_SUCCESS;
}

int nntr_hvx_attn_m1_release(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (!s->attn_m1) {
    return AEE_EBADSTATE;
  }
  hvx_attn_m1_free(s->attn_m1);
  s->attn_m1 = NULL;
  FARF(HIGH, "attn_m1_release: cache freed");
  return AEE_SUCCESS;
}

int nntr_hvx_attn_m1_kv_append(remote_handle64 handle, uint32 layer,
                               uint32 kv_from, uint32 n_rows,
                               const float *k_rows, int kLen,
                               const float *v_rows, int vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hvx_attn_m1_ctx *c = s->attn_m1;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint64_t want = (uint64_t)n_rows * c->n_kv * c->head_dim;
  if (kLen < 0 || vLen < 0 || (uint64_t)kLen != want ||
      (uint64_t)vLen != want) {
    FARF(ERROR, "attn_m1_kv_append: bad shape (n_rows=%u k=%d v=%d want=%u)",
         (unsigned)n_rows, kLen, vLen, (unsigned)want);
    return AEE_EINVALIDFORMAT;
  }
  return hvx_attn_m1_kv_append(s->attn_m1, layer, kv_from, n_rows, k_rows,
                               v_rows);
}

int nntr_hvx_attn_m1_forward(remote_handle64 handle, uint32 layer, uint32 pos,
                             float scale, const float *q, int qLen,
                             const float *k, int kLen, const float *v, int vLen,
                             float *y, int yLen, float *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hvx_attn_m1_ctx *c = s->attn_m1;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint32_t n_q = c->n_kv * c->gqa;
  if (qLen < 0 || kLen < 0 || vLen < 0 || yLen < 0 || statsLen < 0 ||
      (uint32_t)qLen != n_q * c->head_dim ||
      (uint32_t)kLen != c->n_kv * c->head_dim || vLen != kLen || yLen != qLen ||
      (statsLen != 0 && (uint32_t)statsLen != 2u * n_q &&
       (uint32_t)statsLen != 2u * n_q + ATTN_M1_PROF_WORDS)) {
    FARF(ERROR,
         "attn_m1_forward: bad shape (q=%d k=%d v=%d y=%d stats=%d; "
         "n_q=%u head_dim=%u)",
         qLen, kLen, vLen, yLen, statsLen, (unsigned)n_q,
         (unsigned)c->head_dim);
    return AEE_EINVALIDFORMAT;
  }
  if ((uint32_t)statsLen != 2u * n_q + ATTN_M1_PROF_WORDS) {
    return hvx_attn_m1_forward(s->attn_m1, layer, pos, scale, q, k, v, y,
                               statsLen ? stats : NULL);
  }
  /* The phase words (#146) follow the (m, l) pairs, bit-copied. */
  uint32_t prof[ATTN_M1_PROF_WORDS];
  const int rc = hvx_attn_m1_forward_prof(s->attn_m1, layer, pos, scale, q, k,
                                          v, y, stats, prof);
  if (rc == AEE_SUCCESS) {
    memcpy(stats + 2u * n_q, prof, sizeof(prof));
  }
  return rc;
}
