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
#include "htp_graph_desc.h"
#include "hvx_attn_m1_f32.h"

int nntr_hvx_attn_m1_register(remote_handle64 handle, uint32 n_layers,
                              uint32 n_kv, uint32 gqa, uint32 head_dim,
                              uint32 max_seq) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  /** [plan 201 S4] a second register of another shape is the second cache
     (Gemma 4's full layers beside its sliding ones); a third, or a second
     of the first one's shape, is refused as before */
  hvx_attn_m1_ctx **slot = &s->attn_m1;
  if (s->attn_m1) {
    if (s->attn_m1_b || (s->attn_m1->n_kv == n_kv && s->attn_m1->gqa == gqa &&
                         s->attn_m1->head_dim == head_dim)) {
      FARF(ERROR, "attn_m1_register: a cache is already registered");
      return AEE_EBADSTATE;
    }
    slot = &s->attn_m1_b;
  }
  int err = AEE_SUCCESS;
  *slot = hvx_attn_m1_create(n_layers, n_kv, gqa, head_dim, max_seq,
                             s->quant_pool, &err);
  if (!*slot) {
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
       (unsigned)((*slot)->cache_halves * 2u * sizeof(uint16_t) / 1024u));
  return AEE_SUCCESS;
}

int nntr_hvx_attn_m1_reserve(remote_handle64 handle, uint32 cache, uint32 rows,
                             uint32 drop, uint32 base) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || cache > 1u)
    return AEE_EBADPARM;
  hvx_attn_m1_ctx **slot = cache ? &s->attn_m1_b : &s->attn_m1;
  hvx_attn_m1_ctx *old = *slot;
  if (!old || !s->graph)
    return AEE_EBADSTATE;
  for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
    const htp_graph_op *op = &s->graph->ops[i];
    if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == old->n_kv &&
        op->gqa == old->gqa && op->head_dim == old->head_dim &&
        ((!op->top_k && (drop || base)) || base > s->graph->max_seq ||
         (drop &&
          (op->n_experts > UINT32_MAX - drop || base != op->n_experts + drop))))
      return AEE_EINVALIDFORMAT;
  }
  int err = AEE_SUCCESS;
  hvx_attn_m1_ctx *next = rows == old->max_seq && !drop
                            ? old
                            : hvx_attn_m1_reserve(old, rows, drop, &err);
  if (!next)
    return err;
  *slot = next;
  for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
    htp_graph_op *op = &s->graph->ops[i];
    if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == old->n_kv &&
        op->gqa == old->gqa && op->head_dim == old->head_dim)
      op->n_experts = base;
  }
  if (old != next)
    hvx_attn_m1_free(old);
  return AEE_SUCCESS;
}

static void snapshot_abort(nntr_hvx_session *s) {
  for (uint32_t c = 0; c < 2u; ++c) {
    hvx_attn_m1_free(s->kv_restore[c]);
    s->kv_restore[c] = NULL;
    s->kv_restore_next[c] = 0;
  }
  s->kv_restore_mask = 0;
}

int nntr_hvx_attn_m1_snapshot_control(remote_handle64 handle, uint32 action,
                                      const uint32 *metadata, int metadataLen,
                                      uint32 *result, int resultLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !s->graph || !s->attn_m1)
    return AEE_EBADSTATE;
  hvx_attn_m1_ctx *active[2] = {s->attn_m1, s->attn_m1_b};
  if (action == 4u) {
    snapshot_abort(s);
    return AEE_SUCCESS;
  }
  if (action == 0u) {
    if (!result || resultLen != 17)
      return AEE_EINVALIDFORMAT;
    memset(result, 0, 17u * sizeof(uint32_t));
    for (uint32_t c = 0; c < 2u; ++c) {
      const hvx_attn_m1_ctx *a = active[c];
      if (!a)
        continue;
      if (!a->q8)
        return AEE_EBADSTATE;
      result[0] |= 1u << c;
      uint32_t *r = result + 1u + 8u * c;
      r[0] = a->n_layers;
      r[1] = a->n_kv;
      r[2] = a->gqa;
      r[3] = a->head_dim;
      r[4] = a->max_seq;
      r[5] = HVX_ATTN_M1_Q8 | (a->fast ? HVX_ATTN_M1_FAST : 0u) |
             (a->qk_i8 ? HVX_ATTN_M1_QK_I8 : 0u);
      for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
        const htp_graph_op *op = &s->graph->ops[i];
        if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == a->n_kv &&
            op->gqa == a->gqa && op->head_dim == a->head_dim)
          r[6] = op->n_experts;
      }
    }
    return AEE_SUCCESS;
  }
  if (action == 1u) {
    if (s->kv_restore_mask)
      return AEE_EBADSTATE;
    hvx_attn_m1_park(active[0]);
    hvx_attn_m1_park(active[1]);
    return AEE_SUCCESS;
  }
  if (action == 2u) {
    snapshot_abort(s);
    if (!metadata || metadataLen != 17 || metadata[0] != (active[1] ? 3u : 1u))
      return AEE_EINVALIDFORMAT;
    int rc = AEE_SUCCESS;
    for (uint32_t c = 0; c < 2u; ++c) {
      if (!active[c])
        continue;
      const uint32_t *r = metadata + 1u + 8u * c;
      const hvx_attn_m1_ctx *a = active[c];
      const uint32_t flags = HVX_ATTN_M1_Q8 |
                             (a->fast ? HVX_ATTN_M1_FAST : 0u) |
                             (a->qk_i8 ? HVX_ATTN_M1_QK_I8 : 0u);
      if (r[0] != a->n_layers || r[1] != a->n_kv || r[2] != a->gqa ||
          r[3] != a->head_dim || r[5] != flags || r[7] || !r[4] ||
          r[4] > s->graph->max_seq || r[4] % 64u || r[6] > s->graph->max_seq) {
        rc = AEE_EINVALIDFORMAT;
        break;
      }
      s->kv_restore[c] =
        hvx_attn_m1_create(r[0], r[1], r[2], r[3],
                           r[4] | flags | HVX_ATTN_M1_LAZY, s->quant_pool, &rc);
      if (!s->kv_restore[c])
        break;
      s->kv_restore_base[c] = r[6];
    }
    if (rc != AEE_SUCCESS) {
      snapshot_abort(s);
      return rc;
    }
    s->kv_restore_mask = metadata[0];
    return AEE_SUCCESS;
  }
  if (action == 3u) {
    if (!s->kv_restore_mask)
      return AEE_EBADSTATE;
    for (uint32_t c = 0; c < 2u; ++c)
      if (s->kv_restore[c] &&
          s->kv_restore_next[c] != s->kv_restore[c]->n_layers)
        return AEE_EBADSTATE;
    // No failure points after validation: both shapes become visible together.
    for (uint32_t c = 0; c < 2u; ++c) {
      const hvx_attn_m1_ctx *a = active[c];
      if (!a)
        continue;
      for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
        htp_graph_op *op = &s->graph->ops[i];
        if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == a->n_kv &&
            op->gqa == a->gqa && op->head_dim == a->head_dim)
          op->n_experts = s->kv_restore_base[c];
      }
      if (c)
        s->attn_m1_b = s->kv_restore[c];
      else
        s->attn_m1 = s->kv_restore[c];
      s->kv_restore[c] = NULL;
      hvx_attn_m1_free(active[c]);
    }
    snapshot_abort(s);
    return AEE_SUCCESS;
  }
  return AEE_EINVALIDFORMAT;
}

int nntr_hvx_attn_m1_snapshot_layer(remote_handle64 handle, uint32 action,
                                    uint32 cache, uint32 layer,
                                    const uint8 *input, int inputLen,
                                    uint8 *output, int outputLen,
                                    uint32 *bytes) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || cache > 1u || !bytes || inputLen < 0 || outputLen < 0)
    return AEE_EINVALIDFORMAT;
  hvx_attn_m1_ctx *a =
    action == 2u ? s->kv_restore[cache] : (cache ? s->attn_m1_b : s->attn_m1);
  if (!a || layer >= a->n_layers)
    return AEE_EBADSTATE;
  if (action == 2u) {
    if (!s->kv_restore_mask || layer != s->kv_restore_next[cache])
      return AEE_EBADSTATE;
    const int rc =
      hvx_attn_m1_snapshot_import(a, layer, input, (size_t)inputLen);
    if (rc == AEE_SUCCESS)
      ++s->kv_restore_next[cache];
    *bytes = (uint32)inputLen;
    return rc;
  }
  *bytes = (uint32)hvx_attn_m1_snapshot_bytes(a, layer);
  if (action == 0u)
    return *bytes ? AEE_SUCCESS : AEE_EBADSTATE;
  if (action != 1u)
    return AEE_EINVALIDFORMAT;
  return hvx_attn_m1_snapshot_export(a, layer, output, (size_t)outputLen);
}

int nntr_hvx_attn_m1_release(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (!s->attn_m1) {
    return AEE_EBADSTATE;
  }
  snapshot_abort(s);
  hvx_attn_m1_free(s->attn_m1);
  s->attn_m1 = NULL;
  hvx_attn_m1_free(s->attn_m1_b);
  s->attn_m1_b = NULL;
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
  /* [plan 201 S4] HTP_ATTN_KV_CACHE_B in layer names the second cache */
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
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
  return hvx_attn_m1_kv_append(c, layer, kv_from, n_rows, k_rows, v_rows);
}

int nntr_hvx_attn_m1_kv_append_f16(remote_handle64 handle, uint32 layer,
                                   uint32 kv_from, uint32 n_rows,
                                   const uint16 *k_rows, int kLen,
                                   const uint16 *v_rows, int vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint64_t want = (uint64_t)n_rows * c->n_kv * c->head_dim;
  if (kLen < 0 || vLen < 0 || (uint64_t)kLen != want ||
      (uint64_t)vLen != want) {
    FARF(ERROR,
         "attn_m1_kv_append_f16: bad shape (n_rows=%u k=%d v=%d want=%u)",
         (unsigned)n_rows, kLen, vLen, (unsigned)want);
    return AEE_EINVALIDFORMAT;
  }
  return hvx_attn_m1_kv_append_f16(c, layer, kv_from, n_rows, k_rows, v_rows);
}

int nntr_hvx_attn_m1_kv_amax_floor(remote_handle64 handle, uint32 layer,
                                   const float *k_amax, int kLen,
                                   const float *v_amax, int vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint64_t want = (uint64_t)c->n_kv * c->head_dim;
  if (kLen < 0 || vLen < 0 || (uint64_t)kLen != want ||
      (uint64_t)vLen != want) {
    FARF(ERROR, "attn_m1_kv_amax_floor: bad shape (k=%d v=%d want=%u)", kLen,
         vLen, (unsigned)want);
    return AEE_EINVALIDFORMAT;
  }
  return hvx_attn_m1_kv_amax_floor(c, layer, k_amax, v_amax);
}

int nntr_hvx_attn_m1_seed_kvq(remote_handle64 handle, uint32 layer,
                              uint32 kv_handle, uint32 n_rows,
                              const float *k_rows, int k_rowsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!c || !kv || kv->kind != HEXKL_KV_Q8 || !kv->fixed ||
      !kv->plain_masters || kv->n_head_kv != c->n_kv ||
      kv->head_dim != c->head_dim || n_rows > kv->max_rows ||
      (k_rowsLen != 0 &&
       (uint64_t)k_rowsLen != (uint64_t)n_rows * c->n_kv * c->head_dim)) {
    FARF(ERROR, "attn_m1_seed_kvq: cache %u does not fit layer %u",
         (unsigned)kv_handle, (unsigned)layer);
    return AEE_EBADSTATE;
  }
  const int rc = hvx_attn_m1_seed_q8(
    c, layer, n_rows, kv->kt4, kv->v4, (size_t)kv->max_rows * kv->head_dim,
    kv->head_dim, kv->fs_k, kv->fs_v, kv->fs_v_inv,
    k_rowsLen != 0 ? k_rows : NULL);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  return hexkl_kv_q_release(&s->kv_q, kv_handle);
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
    return hvx_attn_m1_forward(s->attn_m1, layer, pos, 0u, scale, q, k, v, y,
                               statsLen ? stats : NULL);
  }
  /* The phase words (#146) follow the (m, l) pairs, bit-copied. */
  uint32_t prof[ATTN_M1_PROF_WORDS];
  const int rc = hvx_attn_m1_forward_prof(s->attn_m1, layer, pos, 0u, scale, q,
                                          k, v, y, stats, prof);
  if (rc == AEE_SUCCESS) {
    memcpy(stats + 2u * n_q, prof, sizeof(prof));
  }
  return rc;
}
