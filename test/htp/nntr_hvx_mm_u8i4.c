// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_mm_u8i4.c
 * @date   03 Aug 2026
 * @brief  FastRPC entry points for the HMX u8i4 accuracy harness
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include "hexkl_micro.h"
#include "hexkl_mm_u8i4.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief Rounds @a v up to a multiple of @a a. */
#define ROUND_UP(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/**
 * @brief Accuracy harness: the whole flow, quantization and dequantization
 *        on the DSP, with every intermediate buffer returned so each stage
 *        is checkable. Used by unittest_hvx_mm_u8i4's fixed shapes.
 *
 * hw_init and the HMX lock are session-scoped (nntr_hvx_open), not per
 * call, since doc15 §8 item 3 turned this from a call that stood alone into
 * one of several entry points sharing one session. The weight is still
 * baked fresh every call -- that is the point of this harness, checking the
 * bake -- unlike the resident-weight path in mm_u8i4_layer below.
 */
int nntr_hvx_mm_u8i4_from_f32(
  remote_handle64 handle, uint32 M, uint32 K, uint32 N, const float *act_f32,
  int act_f32Len, const int8 *w_i4_rm, int w_i4_rmLen, const float *w_scale,
  int w_scaleLen, const int32 *colsum_w, int colsum_wLen, const float *bias,
  int biasLen, uint8 *act_u8_ah, int act_u8_ahLen, float *act_scale,
  int act_scaleLen, int32 *act_zp, int act_zpLen, int32 *acc_i32,
  int acc_i32Len, float *out_f32, int out_f32Len) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }

  const uint32_t m_pad = ROUND_UP(M, HEXKL_HMX_INT8_BLOCK_N_ROW);

  if ((uint32_t)act_f32Len != M * K || (uint32_t)w_i4_rmLen != K * N ||
      (uint32_t)act_u8_ahLen != m_pad * K || (uint32_t)act_scaleLen != m_pad ||
      (uint32_t)act_zpLen != m_pad || (uint32_t)acc_i32Len != m_pad * N ||
      (uint32_t)w_scaleLen != N || (uint32_t)colsum_wLen != N ||
      (uint32_t)biasLen != N || (uint32_t)out_f32Len != M * N) {
    FARF(ERROR, "bad lengths (M=%u K=%u N=%u m_pad=%u)", (unsigned)M,
         (unsigned)K, (unsigned)N, (unsigned)m_pad);
    return AEE_EBADPARM;
  }

  hexkl_mm_u8i4_layout L;
  int res = hexkl_mm_u8i4_plan(s->vtcm_base, s->vtcm_size, m_pad, K, N, &L);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "plan failed: 0x%08x", res);
    return res;
  }

  // K1 then K2, writing the AH tiles straight into VTCM.
  hvx_quant_rows_u8_params(act_f32, M, m_pad, K, act_scale, act_zp,
                           s->quant_pool);
  res = hvx_quant_pack_u8_ah(act_f32, M, m_pad, K, act_scale, act_zp,
                             s->vtcm_base + L.act_base, s->quant_pool);
  if (res != AEE_SUCCESS) {
    return res;
  }
  memcpy(act_u8_ah, s->vtcm_base + L.act_base, (size_t)m_pad * K);

  // setup_acc_read_int32 already ran once in nntr_hvx_open for
  // s->config_off, which hexkl_mm_u8i4_plan recomputes identically here
  // (it is a pure function of vtcm_size) -- no need to call it again.
  res = hexkl_mm_u8i4_bake_weights(&L, w_i4_rm, K, N);
  if (res == AEE_SUCCESS) {
    res = hexkl_mm_u8i4_run(&L, m_pad, K, N, acc_i32);
  }

  if (res == AEE_SUCCESS) {
    hvx_dequant_i32_to_f32(acc_i32, M, m_pad, N, act_scale, act_zp, colsum_w,
                           w_scale, bias, out_f32, /*accumulate=*/0);
  }
  return res;
}

/**
 * @brief Bakes a K x N int4 weight once and keeps it resident until
 *        weight_release_u8i4 -- see hexkl_mm_u8i4_dma.h.
 */
int nntr_hvx_weight_register_u8i4(remote_handle64 handle, uint32 K, uint32 N,
                                  const int8 *w_i4_rm, int w_i4_rmLen,
                                  const float *w_scale, int w_scaleLen,
                                  const int32 *colsum_w, int colsum_wLen,
                                  const float *bias, int biasLen,
                                  uint32 *w_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)w_i4_rmLen != K * N || (uint32_t)w_scaleLen != N ||
      (uint32_t)colsum_wLen != N || (uint32_t)biasLen != N) {
    FARF(ERROR, "weight_register_u8i4: bad lengths (K=%u N=%u)", (unsigned)K,
         (unsigned)N);
    return AEE_EBADPARM;
  }
  return hexkl_weight_u8i4_register(&s->weights_u8i4, s->vtcm_base,
                                    s->vtcm_size, K, N, w_i4_rm, w_scale,
                                    colsum_w, bias, s->quant_pool, w_handle);
}

int nntr_hvx_weight_release_u8i4(remote_handle64 handle, uint32 w_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  return hexkl_weight_u8i4_release(&s->weights_u8i4, w_handle);
}

/**
 * @brief Runs a layer's worth of matmuls (Q/K/V, or gate/up) against one
 *        shared activation -- see hexkl_mm_u8i4_dma.h. This is the entry
 *        point PR③'s ComputeOps seam will call; nntr_hvx_mm_u8i4_from_f32
 *        above stays the accuracy harness.
 */
/**
 * @brief Slots mm_u8i4_layer_timed fills, in order.
 *
 * MUST match the same enum in nntr_hvx_mm_u8i8.c and the kStage list in
 * unittest_hvx_fc.cpp: the ARM side cannot include this header, so the entry
 * point rejects a stale count with AEE_EBADPARM rather than silently
 * truncating. Restated per width rather than shared, the same call this tree
 * already made for the two dma modules.
 */
enum {
  FC_T_DSP_TOTAL = 0, /**< the whole layer_run call, DSP clock */
  FC_T_QUANT,         /**< act f32 -> u8 AH in VTCM */
  FC_T_DEQUANT,       /**< i32 -> f32, in place on the VTCM tile */
  FC_T_ACC_READ,      /**< HMX accumulator -> VTCM, vendor */
  FC_T_ACC_COPY,      /**< VTCM -> DDR staging; 0 on the in-place path */
  FC_T_DRAIN,         /**< DMA waits */
  FC_T_ACC_STRIDE,    /**< not a time: the derived row stride, 0 if fallback */
  FC_N_STAGES
};

/**
 * @brief [doc 45 Gate 0b] The DSP PD's own allocatable memory.
 *
 * Gate 0 stopped at 1.89 GB with 0x8000040d, an error raised before
 * hexkl_weight_u8i4_register ran -- registration's own out-of-memory path
 * returns AEE_ENOMEMORY. So the PD could not service the call at all, and
 * what ran out was PD memory generally rather than the malloc heap the
 * baked weights live in. This probe carries no payload, so whatever it
 * reaches is the heap ceiling on its own, with none of the 3.67 MB input
 * buffer registration also needs in flight.
 *
 * Every chunk is touched one byte per 4 KB page: an allocator that reserves
 * lazily would otherwise report a ceiling that does not exist once the
 * pages are actually written.
 */
int nntr_hvx_mem_probe_dsp_heap(remote_handle64 handle, uint32 chunk_mb,
                                uint32 max_chunks, uint32 *chunks_ok,
                                int chunks_okLen, uint64 *touched_sum,
                                int touched_sumLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  void **blocks;
  uint32_t n = 0, i;
  uint64_t sum = 0;
  const size_t chunk = (size_t)chunk_mb * 1024u * 1024u;

  if (!s || chunks_okLen < 1 || touched_sumLen < 1) {
    return AEE_EBADPARM;
  }
  if (chunk_mb == 0 || max_chunks == 0) {
    return AEE_EBADPARM;
  }
  blocks = (void **)malloc(sizeof(void *) * max_chunks);
  if (!blocks) {
    return AEE_ENOMEMORY;
  }

  for (n = 0; n < max_chunks; ++n) {
    unsigned char *p = (unsigned char *)malloc(chunk);
    size_t off;
    if (!p) {
      break;
    }
    for (off = 0; off < chunk; off += 4096u) {
      p[off] = (unsigned char)(off + n);
      sum += p[off];
    }
    blocks[n] = p;
  }
  for (i = 0; i < n; ++i) {
    free(blocks[i]);
  }
  free(blocks);

  chunks_ok[0] = n;
  touched_sum[0] = sum;
  return AEE_SUCCESS;
}

/**
 * @brief [doc 45 Gate 0b] Whether the DSP can reach a host buffer of a
 *        given size.
 *
 * Called with rpcmem/ION memory, this is the measurement doc 45 section 8.4
 * rests on: the redesign puts one baked-weight arena in ION and has the DSP
 * read it directly, so what matters is the largest ION mapping the DSP can
 * touch, not how much it can malloc. Touching every page rather than the
 * first byte is what makes it a mapping test rather than an address test.
 */
int nntr_hvx_mem_probe_touch(remote_handle64 handle, const uint8 *buf,
                             int bufLen, uint64 *touched_sum,
                             int touched_sumLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint64_t sum = 0;
  int off;

  if (!s || !buf || bufLen <= 0 || touched_sumLen < 1) {
    return AEE_EBADPARM;
  }
  for (off = 0; off < bufLen; off += 4096) {
    sum += buf[off];
  }
  touched_sum[0] = sum;
  return AEE_SUCCESS;
}

/** @brief Shared by both entry points below, so they cannot drift apart on
 *         what they accept. */
static int check_layer_args(const nntr_hvx_session *s, uint32 M, uint32 K,
                            const uint32 *w_handles, int w_handlesLen,
                            int act_f32Len, int out_catLen) {
  uint32_t n_total = 0;
  int i;
  if (!s || w_handlesLen <= 0) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i4_layer: bad act_f32Len (M=%u K=%u)", (unsigned)M,
         (unsigned)K);
    return AEE_EBADPARM;
  }
  for (i = 0; i < w_handlesLen; ++i) {
    if (w_handles[i] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        !s->weights_u8i4.slots[w_handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    n_total += s->weights_u8i4.slots[w_handles[i]].N;
  }
  if ((uint32_t)out_catLen != M * n_total) {
    FARF(ERROR, "mm_u8i4_layer: bad out_catLen (M=%u n_total=%u)", (unsigned)M,
         (unsigned)n_total);
    return AEE_EBADPARM;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_layer(remote_handle64 handle, uint32 M, uint32 K,
                           const uint32 *w_handles, int w_handlesLen,
                           const float *act_f32, int act_f32Len, float *out_cat,
                           int out_catLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  int rc =
    check_layer_args(s, M, K, w_handles, w_handlesLen, act_f32Len, out_catLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  /** No probe reset here: this is the production path, and hexkl_probe_on stays
   * wherever the last timed call left it -- off, unless one ran. */
  return hexkl_mm_u8i4_layer_run(&s->weights_u8i4, s->vtcm_base, s->vtcm_size,
                                 s->config_off, M, K, w_handles,
                                 (uint32_t)w_handlesLen, act_f32, out_cat,
                                 &(const hexkl_mm_opts){.pool = s->quant_pool});
}

int nntr_hvx_mm_u8i4_layer_timed(remote_handle64 handle, uint32 M, uint32 K,
                                 const uint32 *w_handles, int w_handlesLen,
                                 const float *act_f32, int act_f32Len,
                                 float *out_cat, int out_catLen,
                                 uint32 *stage_us, int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint64_t t0, t1;
  int rc =
    check_layer_args(s, M, K, w_handles, w_handlesLen, act_f32Len, out_catLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (!stage_us || stage_usLen != FC_N_STAGES) {
    FARF(ERROR, "mm_u8i4_layer_timed: stage_usLen %d, expected %d", stage_usLen,
         (int)FC_N_STAGES);
    return AEE_EBADPARM;
  }

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now();
  rc = hexkl_mm_u8i4_layer_run(&s->weights_u8i4, s->vtcm_base, s->vtcm_size,
                               s->config_off, M, K, w_handles,
                               (uint32_t)w_handlesLen, act_f32, out_cat,
                               &(const hexkl_mm_opts){.pool = s->quant_pool});
  t1 = hexkl_probe_now();
  /** Left off again on the way out: the production entry point above shares
   * these globals, and an instrumented run must not make the next
   * uninstrumented one pay for the probes. */
  hexkl_probe_on = 0;

  stage_us[FC_T_DSP_TOTAL] = (uint32)(t1 - t0);
  stage_us[FC_T_QUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_QUANT];
  stage_us[FC_T_DEQUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_DEQUANT];
  stage_us[FC_T_ACC_READ] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_READ];
  stage_us[FC_T_ACC_COPY] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_COPY];
  stage_us[FC_T_DRAIN] = (uint32)hexkl_probe_us[HEXKL_PROBE_DRAIN];
  stage_us[FC_T_ACC_STRIDE] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE];
  return rc;
}

/** @brief Shared by both u8in entry points below, mirroring
 *         check_layer_args's role for the f32-activation pair. */
static int check_layer_args_u8in(const nntr_hvx_session *s, uint32 M, uint32 K,
                                 const uint32 *w_handles, int w_handlesLen,
                                 int act_ahLen, int act_scaleLen, int act_zpLen,
                                 int out_catLen) {
  uint32_t n_total = 0;
  int i;
  const uint32_t m_pad = ROUND_UP(M, HEXKL_HMX_INT8_BLOCK_N_ROW);
  if (!s || w_handlesLen <= 0) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_ahLen != m_pad * K || (uint32_t)act_scaleLen != m_pad ||
      (uint32_t)act_zpLen != m_pad) {
    FARF(ERROR, "mm_u8i4_layer_u8in: bad lengths (M=%u K=%u m_pad=%u)",
         (unsigned)M, (unsigned)K, (unsigned)m_pad);
    return AEE_EBADPARM;
  }
  for (i = 0; i < w_handlesLen; ++i) {
    if (w_handles[i] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        !s->weights_u8i4.slots[w_handles[i]].in_use) {
      return AEE_EBADPARM;
    }
    n_total += s->weights_u8i4.slots[w_handles[i]].N;
  }
  if ((uint32_t)out_catLen != M * n_total) {
    FARF(ERROR, "mm_u8i4_layer_u8in: bad out_catLen (M=%u n_total=%u)",
         (unsigned)M, (unsigned)n_total);
    return AEE_EBADPARM;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_layer_u8in(remote_handle64 handle, uint32 M, uint32 K,
                                const uint32 *w_handles, int w_handlesLen,
                                const uint8 *act_ah, int act_ahLen,
                                const float *act_scale, int act_scaleLen,
                                const int32 *act_zp, int act_zpLen,
                                float *out_cat, int out_catLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  int rc = check_layer_args_u8in(s, M, K, w_handles, w_handlesLen, act_ahLen,
                                 act_scaleLen, act_zpLen, out_catLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  return hexkl_mm_u8i4_layer_run(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
    w_handles, (uint32_t)w_handlesLen, /*act_f32=*/NULL, out_cat,
    &(const hexkl_mm_opts){.pool = s->quant_pool,
                           .act_scale = act_scale,
                           .act_zp = act_zp,
                           .act_ah_prepacked = act_ah});
}

int nntr_hvx_mm_u8i4_layer_u8in_timed(
  remote_handle64 handle, uint32 M, uint32 K, const uint32 *w_handles,
  int w_handlesLen, const uint8 *act_ah, int act_ahLen, const float *act_scale,
  int act_scaleLen, const int32 *act_zp, int act_zpLen, float *out_cat,
  int out_catLen, uint32 *stage_us, int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint64_t t0, t1;
  int rc = check_layer_args_u8in(s, M, K, w_handles, w_handlesLen, act_ahLen,
                                 act_scaleLen, act_zpLen, out_catLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (!stage_us || stage_usLen != FC_N_STAGES) {
    FARF(ERROR, "mm_u8i4_layer_u8in_timed: stage_usLen %d, expected %d",
         stage_usLen, (int)FC_N_STAGES);
    return AEE_EBADPARM;
  }

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now();
  rc = hexkl_mm_u8i4_layer_run(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
    w_handles, (uint32_t)w_handlesLen, /*act_f32=*/NULL, out_cat,
    &(const hexkl_mm_opts){.pool = s->quant_pool,
                           .act_scale = act_scale,
                           .act_zp = act_zp,
                           .act_ah_prepacked = act_ah});
  t1 = hexkl_probe_now();
  hexkl_probe_on = 0;

  stage_us[FC_T_DSP_TOTAL] = (uint32)(t1 - t0);
  stage_us[FC_T_QUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_QUANT];
  stage_us[FC_T_DEQUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_DEQUANT];
  stage_us[FC_T_ACC_READ] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_READ];
  stage_us[FC_T_ACC_COPY] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_COPY];
  stage_us[FC_T_DRAIN] = (uint32)hexkl_probe_us[HEXKL_PROBE_DRAIN];
  stage_us[FC_T_ACC_STRIDE] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE];
  return rc;
}

/**
 * @brief Slots mm_u8i4_layer_fused_timed fills, in order.
 *
 * NOT the FC_T_* layout: the fused call runs a second QUANT pass (the
 * SwiGLU output's requant) and a SwiGLU stage the unfused call does not
 * have, so the slots are restated here rather than reusing FC_N_STAGES --
 * the entry rejects a stale count with AEE_EBADPARM, same drift guard as
 * the FC pair.
 */
enum {
  FU_T_DSP_TOTAL = 0, /**< the whole fused_run call, DSP clock */
  FU_T_QUANT,         /**< act quant + the SwiGLU output's requant */
  FU_T_SWIGLU,        /**< silu(gate)*up, in VTCM */
  FU_T_DEQUANT,       /**< both matmuls' i32 -> f32 */
  FU_T_ACC_READ,      /**< HMX accumulator -> VTCM, vendor (both matmuls) */
  FU_T_ACC_COPY,      /**< always 0: fused_run requires the in-place layout */
  FU_T_DRAIN,         /**< DMA waits */
  FU_T_ACC_STRIDE,    /**< not a time: the derived row stride */
  FU_N_STAGES
};

/** @brief Shared by both fused entry points below. The SwiGLU contract
 *         (down.K == gate_up.N / 2) is checked here so a mismatching pair
 *         fails loudly instead of computing garbage. */
static int check_layer_args_fused(const nntr_hvx_session *s, uint32 M, uint32 K,
                                  const uint32 *w_handles, int w_handlesLen,
                                  int act_f32Len, int out_f32Len) {
  int i;
  if (!s || w_handlesLen != 2) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i4_layer_fused: bad act_f32Len (M=%u K=%u)", (unsigned)M,
         (unsigned)K);
    return AEE_EBADPARM;
  }
  for (i = 0; i < 2; ++i) {
    if (w_handles[i] >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
        !s->weights_u8i4.slots[w_handles[i]].in_use) {
      return AEE_EBADPARM;
    }
  }
  const hexkl_weight_u8i4 *gu = &s->weights_u8i4.slots[w_handles[0]];
  const hexkl_weight_u8i4 *dn = &s->weights_u8i4.slots[w_handles[1]];
  if (gu->K != K || gu->N != 2 * dn->K) {
    FARF(ERROR, "mm_u8i4_layer_fused: SwiGLU contract (gu K=%u N=%u, dn K=%u)",
         (unsigned)gu->K, (unsigned)gu->N, (unsigned)dn->K);
    return AEE_EBADPARM;
  }
  if ((uint32_t)out_f32Len != M * dn->N) {
    FARF(ERROR, "mm_u8i4_layer_fused: bad out_f32Len (M=%u N=%u)", (unsigned)M,
         (unsigned)dn->N);
    return AEE_EBADPARM;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_layer_fused(remote_handle64 handle, uint32 M, uint32 K,
                                 const uint32 *w_handles, int w_handlesLen,
                                 const float *act_f32, int act_f32Len,
                                 float *out_f32, int out_f32Len) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  int rc = check_layer_args_fused(s, M, K, w_handles, w_handlesLen, act_f32Len,
                                  out_f32Len);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  /** No probe reset here: this is the production path, same policy as
   * mm_u8i4_layer above. */
  return hexkl_mm_u8i4_fused_run(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
    w_handles, act_f32, out_f32, &(const hexkl_mm_opts){.pool = s->quant_pool});
}

int nntr_hvx_mm_u8i4_layer_fused_timed(remote_handle64 handle, uint32 M,
                                       uint32 K, const uint32 *w_handles,
                                       int w_handlesLen, const float *act_f32,
                                       int act_f32Len, float *out_f32,
                                       int out_f32Len, uint32 *stage_us,
                                       int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint64_t t0, t1;
  int rc = check_layer_args_fused(s, M, K, w_handles, w_handlesLen, act_f32Len,
                                  out_f32Len);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (!stage_us || stage_usLen != FU_N_STAGES) {
    FARF(ERROR, "mm_u8i4_layer_fused_timed: stage_usLen %d, expected %d",
         stage_usLen, (int)FU_N_STAGES);
    return AEE_EBADPARM;
  }

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now();
  rc = hexkl_mm_u8i4_fused_run(&s->weights_u8i4, s->vtcm_base, s->vtcm_size,
                               s->config_off, M, K, w_handles, act_f32, out_f32,
                               &(const hexkl_mm_opts){.pool = s->quant_pool});
  t1 = hexkl_probe_now();
  hexkl_probe_on = 0;

  stage_us[FU_T_DSP_TOTAL] = (uint32)(t1 - t0);
  stage_us[FU_T_QUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_QUANT];
  stage_us[FU_T_SWIGLU] = (uint32)hexkl_probe_us[HEXKL_PROBE_SWIGLU];
  stage_us[FU_T_DEQUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_DEQUANT];
  stage_us[FU_T_ACC_READ] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_READ];
  stage_us[FU_T_ACC_COPY] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_COPY];
  stage_us[FU_T_DRAIN] = (uint32)hexkl_probe_us[HEXKL_PROBE_DRAIN];
  stage_us[FU_T_ACC_STRIDE] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE];
  return rc;
}

/**
 * @brief Slots mm_u8i4_gate_up_swiglu_timed fills. Restated rather than
 *        reusing FU_T_*: this call has no down-side dequant/acc_read at
 *        all, so FU_T_DEQUANT/FU_T_ACC_READ would silently mean something
 *        narrower here than there.
 */
enum {
  GU_T_DSP_TOTAL = 0,
  GU_T_QUANT, /**< act quant + the SwiGLU output's requant */
  GU_T_SWIGLU,
  GU_T_DEQUANT, /**< gate_up matmul's i32 -> f32 split */
  GU_T_ACC_READ,
  GU_T_DRAIN,
  GU_T_ACC_STRIDE,
  GU_N_STAGES
};

static int check_gate_up_swiglu_args(const nntr_hvx_session *s, uint32 M,
                                     uint32 K, uint32 w_handle_gate_up,
                                     int act_f32Len, int out_ahLen,
                                     int out_scaleLen, int out_zpLen) {
  if (!s) {
    return AEE_EBADPARM;
  }
  if ((uint32_t)act_f32Len != M * K) {
    FARF(ERROR, "mm_u8i4_gate_up_swiglu: bad act_f32Len (M=%u K=%u)",
         (unsigned)M, (unsigned)K);
    return AEE_EBADPARM;
  }
  if (w_handle_gate_up >= HEXKL_MM_U8I4_MAX_WEIGHTS ||
      !s->weights_u8i4.slots[w_handle_gate_up].in_use) {
    return AEE_EBADPARM;
  }
  const hexkl_weight_u8i4 *gu = &s->weights_u8i4.slots[w_handle_gate_up];
  if (gu->K != K || (gu->N % 2) != 0) {
    FARF(ERROR, "mm_u8i4_gate_up_swiglu: bad shape (gu K=%u N=%u, want K=%u)",
         (unsigned)gu->K, (unsigned)gu->N, (unsigned)K);
    return AEE_EBADPARM;
  }
  const uint32_t inter = gu->N / 2;
  const uint32_t m_pad = ROUND_UP(M, HEXKL_HMX_INT8_BLOCK_N_ROW);
  if ((uint32_t)out_ahLen != m_pad * inter || (uint32_t)out_scaleLen != m_pad ||
      (uint32_t)out_zpLen != m_pad) {
    FARF(ERROR, "mm_u8i4_gate_up_swiglu: bad out lengths (M=%u inter=%u)",
         (unsigned)M, (unsigned)inter);
    return AEE_EBADPARM;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_mm_u8i4_gate_up_swiglu(remote_handle64 handle, uint32 M, uint32 K,
                                    uint32 w_handle_gate_up,
                                    const float *act_f32, int act_f32Len,
                                    uint8 *out_ah, int out_ahLen,
                                    float *out_scale, int out_scaleLen,
                                    int32 *out_zp, int out_zpLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  int rc = check_gate_up_swiglu_args(s, M, K, w_handle_gate_up, act_f32Len,
                                     out_ahLen, out_scaleLen, out_zpLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  return hexkl_mm_u8i4_gate_up_swiglu_run(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
    w_handle_gate_up, act_f32, out_ah, out_scale, out_zp, s->quant_pool);
}

int nntr_hvx_mm_u8i4_gate_up_swiglu_timed(remote_handle64 handle, uint32 M,
                                          uint32 K, uint32 w_handle_gate_up,
                                          const float *act_f32, int act_f32Len,
                                          uint8 *out_ah, int out_ahLen,
                                          float *out_scale, int out_scaleLen,
                                          int32 *out_zp, int out_zpLen,
                                          uint32 *stage_us, int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint64_t t0, t1;
  int rc = check_gate_up_swiglu_args(s, M, K, w_handle_gate_up, act_f32Len,
                                     out_ahLen, out_scaleLen, out_zpLen);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (!stage_us || stage_usLen != GU_N_STAGES) {
    FARF(ERROR, "mm_u8i4_gate_up_swiglu_timed: stage_usLen %d, expected %d",
         stage_usLen, (int)GU_N_STAGES);
    return AEE_EBADPARM;
  }

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now();
  rc = hexkl_mm_u8i4_gate_up_swiglu_run(
    &s->weights_u8i4, s->vtcm_base, s->vtcm_size, s->config_off, M, K,
    w_handle_gate_up, act_f32, out_ah, out_scale, out_zp, s->quant_pool);
  t1 = hexkl_probe_now();
  hexkl_probe_on = 0;

  stage_us[GU_T_DSP_TOTAL] = (uint32)(t1 - t0);
  stage_us[GU_T_QUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_QUANT];
  stage_us[GU_T_SWIGLU] = (uint32)hexkl_probe_us[HEXKL_PROBE_SWIGLU];
  stage_us[GU_T_DEQUANT] = (uint32)hexkl_probe_us[HEXKL_PROBE_DEQUANT];
  stage_us[GU_T_ACC_READ] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_READ];
  stage_us[GU_T_DRAIN] = (uint32)hexkl_probe_us[HEXKL_PROBE_DRAIN];
  stage_us[GU_T_ACC_STRIDE] = (uint32)hexkl_probe_us[HEXKL_PROBE_ACC_STRIDE];
  return rc;
}
