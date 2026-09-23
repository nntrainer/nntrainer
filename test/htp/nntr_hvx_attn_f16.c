// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   nntr_hvx_attn_f16.c
 * @date   14 Sep 2026
 * @brief  DSP-side fp16 attention entry points: layout probe first
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <math.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hexkl_attn_f16.h"
#include "hexkl_micro.h"
#include "hvx_attn_decode_f16.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_tile_f16.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/** @brief One 32x32 fp16 tile, also the activation alignment. */
#define TILE_BYTES HEXKL_HMX_ACTIVATION_ALIGNMENT
/** @brief fp16 elements per tile. */
#define TILE_ELEMS 1024u

/**
 * @brief Slot map for the probe's VTCM scratch. Everything is one tile and
 *        sits on a 2048-byte boundary, which satisfies both the activation
 *        (2048) and weight (128) alignment rules at once.
 */
enum {
  SLOT_FLAT = 0,    /**< row-major staging for HexKL's rm_to_* inputs */
  SLOT_AH_HEXKL,    /**< x as AH, by hexkl_micro_hmx_rm_to_ah_f16 */
  SLOT_AH_HAND,     /**< x as AH, by hvx_tile_f16_rows_to_tile */
  SLOT_WH_HEXKL,    /**< w as WH, by hexkl_micro_hmx_rm_to_wh_f16 */
  SLOT_WH_HAND,     /**< w by hvx_tile_f16_rows_to_tile (AH == WH check) */
  SLOT_WT_WH_HEXKL, /**< w^T as WH, by HexKL from the host-transposed wt */
  SLOT_WT_WH_HAND,  /**< w^T as WH, by hvx_tile_f16_rows_to_tile_transposed */
  SLOT_DIAG_HEXKL,  /**< diag(x row 0) as AH, by HexKL from a staged matrix */
  SLOT_DIAG_HAND,   /**< diag(x row 0) as AH, by hvx_tile_f16_diag */
  SLOT_F32_HAND,    /**< x as AH via the fused f32 -> f16 path, scale 1 */
  SLOT_ACC,         /**< acc_read_f16 destination */
  SLOT_ACC_RM,      /**< ah_to_rm_f16 destination */
  SLOT_COUNT
};

/** @brief flags[] indices reported to the host. */
enum {
  FLAG_AH_HAND_EQ_HEXKL = 0,
  FLAG_WH_EQ_AH,         /**< HexKL's WH of w == our AH-style tile of w */
  FLAG_WT_HAND_EQ_HEXKL, /**< transposed builder == HexKL WH of w^T */
  FLAG_DIAG_HAND_EQ_HEXKL,
  FLAG_ROWS_ROUNDTRIP, /**< tile_to_rows(rm_to_ah(x)) == x */
  FLAG_F32_FUSED_EQ_HEXKL,
  FLAG_WIDEN_EVEN_TO_LO, /**< Wqf32 = vmpy(tile vec, 1): low half == even
                              row -- the split the row sums and the O
                              store both rely on */
  FLAG_COUNT
};

static inline uint8_t *slot(const nntr_hvx_session *s, uint32_t i) {
  return s->vtcm_base + (uint32_t)i * TILE_BYTES;
}

static inline uint32_t slot_off(uint32_t i) { return i * TILE_BYTES; }

static uint32_t tiles_equal(const nntr_hvx_session *s, uint32_t a, uint32_t b) {
  return memcmp(slot(s, a), slot(s, b), TILE_BYTES) == 0 ? 1u : 0u;
}

/**
 * @brief Runs one 32x32 tile matmul on HMX and returns it row-major.
 *
 * @param ah_slot   activation tile
 * @param wh_slot   weight tile
 * @param hand_out  0: read back through HexKL's ah_to_rm_f16;
 *                  1: through hvx_tile_f16_tile_to_rows
 */
static int mm_tile_to_rows(const nntr_hvx_session *s, uint32_t cfg_off,
                           uint32_t ah_slot, uint32_t wh_slot, int hand_out,
                           uint16_t *out) {
  hexkl_micro_hmx_acc_clear_f16();
  int res =
    hexkl_micro_hmx_mm_f16(s->vtcm_base, slot_off(ah_slot), slot_off(wh_slot));
  if (res != AEE_SUCCESS) {
    return res;
  }
  res = hexkl_micro_hmx_acc_read_f16(s->vtcm_base, cfg_off, slot_off(SLOT_ACC));
  if (res != AEE_SUCCESS) {
    return res;
  }
  if (hand_out) {
    hvx_tile_f16_tile_to_rows(out, HVX_TILE_F16_COLS,
                              (const HVX_Vector *)slot(s, SLOT_ACC));
    return AEE_SUCCESS;
  }
  res = hexkl_micro_hmx_ah_to_rm_f16(s->vtcm_base, slot_off(SLOT_ACC_RM),
                                     slot_off(SLOT_ACC));
  if (res != AEE_SUCCESS) {
    return res;
  }
  // ah_to_rm's output is a plain contiguous 32x32 row-major tile.
  memcpy(out, slot(s, SLOT_ACC_RM), TILE_BYTES);
  return AEE_SUCCESS;
}

int nntr_hvx_probe_f16_layouts(remote_handle64 handle, const uint16 *x,
                               int xLen, const uint16 *w, int wLen,
                               const uint16 *wt, int wtLen, uint32 *flags,
                               int flagsLen, uint16 *y_hexkl, int y_hexklLen,
                               uint16 *y_hand, int y_handLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (s->hmx_fp16_rate == 0) {
    FARF(ERROR, "probe_f16_layouts: no fp16 HMX on this part");
    return AEE_EUNSUPPORTED;
  }
  if (xLen != (int)TILE_ELEMS || wLen != (int)TILE_ELEMS ||
      wtLen != (int)TILE_ELEMS || flagsLen < FLAG_COUNT ||
      y_hexklLen != (int)TILE_ELEMS || y_handLen != (int)TILE_ELEMS) {
    return AEE_EBADPARM;
  }

  // The f16 accumulator config lives below the session's int32 one, inside
  // this call's own scratch, and is (re)written every call: the u8 layer
  // paths bound their scratch only against the int32 region, so nothing
  // placed permanently below it would survive them.
  const uint32_t cfg_size = hexkl_micro_hmx_config_size();
  const uint32_t cfg_off =
    (s->config_off - cfg_size) & ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1u);
  if (slot_off(SLOT_COUNT) > cfg_off) {
    return AEE_ENOMEMORY;
  }
  int res = hexkl_micro_hmx_setup_acc_read_f16(s->vtcm_base, cfg_off);
  if (res != AEE_SUCCESS) {
    return res;
  }
  memset(flags, 0, sizeof(uint32) * (uint32_t)flagsLen);

  // --- x -> AH: HexKL (DDR -> flat VTCM -> AH) vs hand ---
  res = hexkl_micro_hmx_copy_submatrix_to_f16(
    s->vtcm_base, slot_off(SLOT_FLAT), (const _Float16 *)x, 0, 0, 32, 32);
  if (res != AEE_SUCCESS) {
    return res;
  }
  res = hexkl_micro_hmx_rm_to_ah_f16(s->vtcm_base, slot_off(SLOT_AH_HEXKL),
                                     slot_off(SLOT_FLAT));
  if (res != AEE_SUCCESS) {
    return res;
  }
  hvx_tile_f16_rows_to_tile((HVX_Vector *)slot(s, SLOT_AH_HAND), x,
                            HVX_TILE_F16_COLS);
  flags[FLAG_AH_HAND_EQ_HEXKL] = tiles_equal(s, SLOT_AH_HAND, SLOT_AH_HEXKL);

  // --- w -> WH by HexKL vs the same rows through the AH builder ---
  // If these match, fp16 AH and WH are one layout and an accumulator read
  // back as AH can be the next matmul's weight with no relayout.
  res = hexkl_micro_hmx_rm_to_wh_f16(s->vtcm_base, slot_off(SLOT_WH_HEXKL),
                                     (const _Float16 *)w, 0, 0, 32);
  if (res != AEE_SUCCESS) {
    return res;
  }
  hvx_tile_f16_rows_to_tile((HVX_Vector *)slot(s, SLOT_WH_HAND), w,
                            HVX_TILE_F16_COLS);
  flags[FLAG_WH_EQ_AH] = tiles_equal(s, SLOT_WH_HAND, SLOT_WH_HEXKL);

  // --- w^T -> WH: hand transposed builder from w vs HexKL from host wt ---
  res = hexkl_micro_hmx_rm_to_wh_f16(s->vtcm_base, slot_off(SLOT_WT_WH_HEXKL),
                                     (const _Float16 *)wt, 0, 0, 32);
  if (res != AEE_SUCCESS) {
    return res;
  }
  hvx_tile_f16_rows_to_tile_transposed((HVX_Vector *)slot(s, SLOT_WT_WH_HAND),
                                       w, HVX_TILE_F16_COLS);
  flags[FLAG_WT_HAND_EQ_HEXKL] =
    tiles_equal(s, SLOT_WT_WH_HAND, SLOT_WT_WH_HEXKL);

  // --- diag(x row 0): stage the full matrix for HexKL, build directly by hand
  {
    uint16_t *flat = (uint16_t *)slot(s, SLOT_FLAT);
    memset(flat, 0, TILE_BYTES);
    for (uint32_t r = 0; r < 32u; ++r) {
      flat[r * 32u + r] = x[r];
    }
    res = hexkl_micro_hmx_rm_to_ah_f16(s->vtcm_base, slot_off(SLOT_DIAG_HEXKL),
                                       slot_off(SLOT_FLAT));
    if (res != AEE_SUCCESS) {
      return res;
    }
    hvx_tile_f16_diag((HVX_Vector *)slot(s, SLOT_DIAG_HAND), x);
    flags[FLAG_DIAG_HAND_EQ_HEXKL] =
      tiles_equal(s, SLOT_DIAG_HAND, SLOT_DIAG_HEXKL);
  }

  // --- AH -> rows round trip through the hand de-interleave ---
  {
    uint16_t *flat = (uint16_t *)slot(s, SLOT_FLAT);
    hvx_tile_f16_tile_to_rows(flat, HVX_TILE_F16_COLS,
                              (const HVX_Vector *)slot(s, SLOT_AH_HEXKL));
    flags[FLAG_ROWS_ROUNDTRIP] = memcmp(flat, x, TILE_BYTES) == 0 ? 1u : 0u;
  }

  // --- fused f32 -> f16 AH path, scale 1.0, must reproduce rm_to_ah(x) ---
  {
    // Scalar f16 -> f32 widening is exact, so a scale of 1.0 makes the
    // fused path's only freedom the lane mapping this checks.
    float xf[TILE_ELEMS];
    for (uint32_t i = 0; i < TILE_ELEMS; ++i) {
      _Float16 h;
      memcpy(&h, &x[i], sizeof(h));
      xf[i] = (float)h;
    }
    hvx_tile_f16_rows_f32_to_tile((HVX_Vector *)slot(s, SLOT_F32_HAND), xf,
                                  HVX_TILE_F16_COLS, 1.0f);
    flags[FLAG_F32_FUSED_EQ_HEXKL] =
      tiles_equal(s, SLOT_F32_HAND, SLOT_AH_HEXKL);
  }

  // --- widening split: vector 0 of AH(x) holds rows 0 and 1 interleaved;
  // the qf32 pair from multiplying it by 1.0 must put row 0 in its low
  // half and row 1 in its high half, each in column order.
  {
    const HVX_Vector one_hf = Q6_Vh_vsplat_R(0x3C00);
    const HVX_Vector *ah = (const HVX_Vector *)slot(s, SLOT_AH_HEXKL);
    const HVX_VectorPair w = Q6_Wqf32_vmpy_VhfVhf(ah[0], one_hf);
    const HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(w));
    const HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(w));
    float rows[2][32];
    *(HVX_UVector *)rows[0] = lo;
    *(HVX_UVector *)rows[1] = hi;
    uint32_t ok = 1;
    for (uint32_t r = 0; r < 2u && ok; ++r) {
      for (uint32_t col = 0; col < 32u; ++col) {
        _Float16 h;
        memcpy(&h, &x[r * 32u + col], sizeof(h));
        if ((float)h != rows[r][col]) {
          ok = 0;
          break;
        }
      }
    }
    flags[FLAG_WIDEN_EVEN_TO_LO] = ok;
  }

  // --- x.w on HMX both ways; the host checks both against its reference ---
  res = mm_tile_to_rows(s, cfg_off, SLOT_AH_HEXKL, SLOT_WH_HEXKL, 0, y_hexkl);
  if (res != AEE_SUCCESS) {
    return res;
  }
  res = mm_tile_to_rows(s, cfg_off, SLOT_AH_HAND, SLOT_WH_HAND, 1, y_hand);
  if (res != AEE_SUCCESS) {
    return res;
  }

  FARF(ALWAYS,
       "probe_f16_layouts: ah=%u wh_eq_ah=%u wt=%u diag=%u roundtrip=%u "
       "f32=%u widen=%u",
       (unsigned)flags[0], (unsigned)flags[1], (unsigned)flags[2],
       (unsigned)flags[3], (unsigned)flags[4], (unsigned)flags[5],
       (unsigned)flags[6]);
  return AEE_SUCCESS;
}

int nntr_hvx_probe_f16_math(remote_handle64 handle, const float *x, int xLen,
                            uint16 *hf_x, int hf_xLen, uint16 *exp2_x,
                            int exp2_xLen, uint16 *softcap_x, int softcap_xLen,
                            float *misc, int miscLen) {
  (void)handle;
  if (xLen != 64 || hf_xLen != 64 || exp2_xLen != 64 || softcap_xLen != 64 ||
      miscLen < 9) {
    return AEE_EBADPARM;
  }
  const HVX_Vector lo = *(const HVX_UVector *)x;
  const HVX_Vector hi = *(const HVX_UVector *)(x + 32);
  const HVX_Vector h = hvx_attn_f32x64_to_hf(lo, hi);
  *(HVX_UVector *)hf_x = h;
  *(HVX_UVector *)exp2_x = hvx_attn_exp2_hf(h);

  const float cap = 30.0f * 1.4426950408889634f;
  const HVX_Vector cap_hf = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(cap));
  const HVX_Vector k_hf =
    Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(2.0f * 1.4426950408889634f / cap));
  *(HVX_UVector *)softcap_x = hvx_attn_softcap_hf(h, cap_hf, k_hf);

  misc[0] = hvx_attn_lane0_f32(hvx_attn_sum32_sf(lo));
  const HVX_Vector pm = hvx_attn_pairmax_hf(h);
  const uint32_t pmw = hvx_attn_word0(pm);
  misc[1] = hvx_attn_hf_to_f32((uint16_t)(pmw & 0xFFFFu));
  misc[2] = hvx_attn_hf_to_f32((uint16_t)(pmw >> 16));
  misc[3] = hvx_attn_hf_to_f32(
    (uint16_t)(hvx_attn_word0(hvx_attn_max64_hf(h)) & 0xFFFFu));
  const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(h, h);
  HVX_Vector acc = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
  misc[4] = hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(acc)));
  const HVX_Vector s0 = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(x[0]));
  misc[5] = hvx_attn_hf_to_f32(
    (uint16_t)(hvx_attn_word0(hvx_attn_exp2_hf(s0)) & 0xFFFFu));
  // The two kernels' shared scalar pieces: libm sqrtf on the DSP, and the
  // Q scale exactly as hexkl_attn_f16.c / hvx_attn_decode_f16.c form it.
  misc[6] = sqrtf(64.0f);
  misc[7] = 1.4426950408889634f / sqrtf(128.0f);
  {
    // Scale a 64-lane vector of x[1] by that scale the way the decode
    // kernel scales Q, then narrow; lane 0 back as f32.
    int bits;
    const float sc = 1.4426950408889634f / sqrtf(64.0f);
    memcpy(&bits, &sc, sizeof(bits));
    const HVX_Vector v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(
      Q6_V_vsplat_R(*(const int *)&x[1]), Q6_V_vsplat_R(bits)));
    const HVX_Vector hv = hvx_attn_f32x64_to_hf(v, v);
    misc[8] = hvx_attn_hf_to_f32((uint16_t)(hvx_attn_word0(hv) & 0xFFFFu));
  }
  return AEE_SUCCESS;
}

int nntr_hvx_probe_f16_sub(remote_handle64 handle, const float *x, int xLen,
                           float c, uint16 *sub_hf, int sub_hfLen,
                           uint16 *sub_qf16, int sub_qf16Len,
                           uint16 *exp2_sub_hf, int exp2_sub_hfLen) {
  (void)handle;
  if (xLen != 64 || sub_hfLen != 64 || sub_qf16Len != 64 ||
      exp2_sub_hfLen != 64) {
    return AEE_EBADPARM;
  }
  const HVX_Vector h = hvx_attn_f32x64_to_hf(*(const HVX_UVector *)x,
                                             *(const HVX_UVector *)(x + 32));
  const HVX_Vector cv = Q6_Vh_vsplat_R((int)hvx_attn_f32_to_hf(c));
  const HVX_Vector d_hf = Q6_Vhf_vsub_VhfVhf(h, cv);
  const HVX_Vector d_qf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(h, cv));
  *(HVX_UVector *)sub_hf = d_hf;
  *(HVX_UVector *)sub_qf16 = d_qf;
  *(HVX_UVector *)exp2_sub_hf = hvx_attn_exp2_hf(d_hf);
  return AEE_SUCCESS;
}

int nntr_hvx_probe_f16_decode_block(remote_handle64 handle, uint32 head_dim,
                                    uint32 kv, const float *q, int qLen,
                                    const uint16 *k, int kLen, float *sc_f32,
                                    int sc_f32Len, uint16 *sv_hf, int sv_hfLen,
                                    uint16 *p_hf, int p_hfLen, float *misc,
                                    int miscLen) {
  (void)handle;
  if (head_dim == 0 || head_dim > 128u || (head_dim % 32u) != 0 || kv == 0 ||
      kv > 64u || (uint32)qLen != head_dim || (uint32)kLen != kv * head_dim ||
      sc_f32Len != 64 || sv_hfLen != 64 || p_hfLen != 64 || miscLen < 3) {
    return AEE_EBADPARM;
  }
  // --- exactly hvx_attn_decode_f16.c's Q prep and score loop ---
  const float scale = 1.4426950408889634f / sqrtf((float)head_dim);
  const uint32_t n_chunks = (head_dim + 63u) / 64u;
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector one_hf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_ONE);
  const HVX_Vector neg_inf = Q6_Vh_vsplat_R((int)HVX_ATTN_HF_NEG_INF);
  HVX_Vector q_hf[2];
  {
    int bits;
    memcpy(&bits, &scale, sizeof(bits));
    const HVX_Vector vscale = Q6_V_vsplat_R(bits);
    for (uint32_t ch = 0; ch < n_chunks; ++ch) {
      float tmp[64];
      const uint32_t n_el =
        (head_dim - 64u * ch) < 64u ? (head_dim - 64u * ch) : 64u;
      memset(tmp, 0, sizeof(tmp));
      memcpy(tmp, q + 64u * ch, n_el * sizeof(float));
      const HVX_Vector a = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *)tmp, vscale));
      const HVX_Vector b = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *)(tmp + 32), vscale));
      q_hf[ch] = hvx_attn_f32x64_to_hf(a, b);
    }
  }
  const uint32_t row_bytes = head_dim * 2u;
  float sc[64];
  for (uint32_t j = 0; j < 64u; ++j) {
    if (j >= kv) {
      sc[j] = 0.0f;
      continue;
    }
    const uint16_t *krow = k + (size_t)j * head_dim;
    HVX_Vector acc = zero;
    for (uint32_t ch = 0; ch < n_chunks; ++ch) {
      HVX_Vector kvv;
      if (row_bytes - 128u * ch >= 128u) {
        kvv = *(const HVX_UVector *)(krow + 64u * ch);
      } else {
        kvv = zero;
        memcpy(&kvv, krow + 64u * ch, row_bytes - 128u * ch);
      }
      const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(kvv, q_hf[ch]);
      acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_lo_W(p));
      acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_V_hi_W(p));
    }
    sc[j] = hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(acc)));
  }
  memcpy(sc_f32, sc, sizeof(sc));
  HVX_Vector sv = hvx_attn_f32x64_to_hf(*(const HVX_UVector *)sc,
                                        *(const HVX_UVector *)(sc + 32));
  {
    uint16_t idx[64];
    for (uint32_t l = 0; l < 64u; ++l) {
      idx[l] = (uint16_t)l;
    }
    const HVX_Vector col_idx = *(const HVX_UVector *)idx;
    const HVX_VectorPred below_hi =
      Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R((int)kv), col_idx);
    sv = Q6_V_vmux_QVV(below_hi, sv, neg_inf);
  }
  *(HVX_UVector *)sv_hf = sv;
  const HVX_Vector m_new = hvx_attn_max64_hf(sv);
  const HVX_Vector pv = hvx_attn_exp2_hf(Q6_Vhf_vsub_VhfVhf(sv, m_new));
  *(HVX_UVector *)p_hf = pv;
  const HVX_VectorPair pw = Q6_Wqf32_vmpy_VhfVhf(pv, one_hf);
  misc[0] = hvx_attn_hf_to_f32((uint16_t)(hvx_attn_word0(m_new) & 0xFFFFu));
  misc[1] = hvx_attn_lane0_f32(hvx_attn_sum32_sf(Q6_Vsf_equals_Vqf32(
    Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(pw), Q6_V_hi_W(pw)))));
  misc[2] = scale;
  return AEE_SUCCESS;
}

/** @brief stats_us[] layout returned to the host. */
enum {
  STAT_QPREP = 0,
  STAT_DMA,
  STAT_TILE,
  STAT_QK,
  STAT_SOFTMAX,
  STAT_PV,
  STAT_STORE,
  STAT_N_BLOCKS,
  STAT_COUNT
};

int nntr_hvx_attn_f16_prefill(
  remote_handle64 handle, uint32 n_q, uint32 cache_from, uint32 cache_to,
  uint32 n_head_q, uint32 n_head_kv, uint32 head_dim, uint32 window, uint32 br,
  uint32 bc, float softcap, const float *q_f32, int q_f32Len,
  const uint16 *k_cache, int k_cacheLen, const uint16 *v_cache, int v_cacheLen,
  const float *sinks, int sinksLen, float *out_f32, int out_f32Len,
  uint32 *stats_us, int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (s->hmx_fp16_rate == 0) {
    FARF(ERROR, "attn_f16_prefill: no fp16 HMX on this part");
    return AEE_EUNSUPPORTED;
  }

  hexkl_attn_f16_shape shape = {n_q,       cache_from, cache_to, n_head_q,
                                n_head_kv, head_dim,   window,   softcap};
  hexkl_attn_f16_tiling tiling = {br, bc, 0, 0};
  if (br == 0 || bc == 0) {
    hexkl_attn_f16_choose_tiling(&shape, &tiling);
  }
  if (hexkl_attn_f16_tiling_init(&shape, &tiling) != HEXKL_ATTN_OK) {
    FARF(ERROR, "attn_f16_prefill: bad shape/tiling");
    return AEE_EBADPARM;
  }

  const uint64_t q_elems = (uint64_t)n_q * n_head_q * head_dim;
  const uint64_t kv_elems = (uint64_t)cache_to * n_head_kv * head_dim;
  if ((uint64_t)q_f32Len != q_elems || (uint64_t)k_cacheLen != kv_elems ||
      (uint64_t)v_cacheLen != kv_elems || (uint64_t)out_f32Len != q_elems ||
      stats_usLen < STAT_COUNT ||
      (sinksLen != 0 && (uint32)sinksLen != n_head_q)) {
    FARF(ERROR, "attn_f16_prefill: bad lengths");
    return AEE_EBADPARM;
  }

  hexkl_attn_f16_io io = {
    q_f32,   n_head_q * head_dim,  k_cache,
    v_cache, n_head_kv * head_dim, sinksLen ? sinks : NULL,
    out_f32, n_head_q * head_dim};
  hexkl_attn_f16_stats st;
  int res =
    hexkl_attn_f16_prefill(s->vtcm_base, s->config_off, s->hmx_fp16_rate,
                           &shape, &tiling, &io, s->quant_pool, &st);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_f16_prefill: kernel failed: 0x%08x", res);
    return res;
  }
  stats_us[STAT_QPREP] = (uint32)st.us_qprep;
  stats_us[STAT_DMA] = (uint32)st.us_dma;
  stats_us[STAT_TILE] = (uint32)st.us_tile;
  stats_us[STAT_QK] = (uint32)st.us_qk;
  stats_us[STAT_SOFTMAX] = (uint32)st.us_softmax;
  stats_us[STAT_PV] = (uint32)st.us_pv;
  stats_us[STAT_STORE] = (uint32)st.us_store;
  stats_us[STAT_N_BLOCKS] = st.n_blocks;
  return AEE_SUCCESS;
}

int nntr_hvx_attn_f16_decode(remote_handle64 handle, uint32 n_q,
                             uint32 cache_from, uint32 cache_to,
                             uint32 n_head_q, uint32 n_head_kv, uint32 head_dim,
                             uint32 window, float softcap, const float *q_f32,
                             int q_f32Len, const uint16 *k_cache,
                             int k_cacheLen, const uint16 *v_cache,
                             int v_cacheLen, const float *sinks, int sinksLen,
                             float *out_f32, int out_f32Len, uint32 *stats_us,
                             int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  // No HMX involved: this path is valid on parts without fp16 HMX too.
  hexkl_attn_f16_shape shape = {n_q,       cache_from, cache_to, n_head_q,
                                n_head_kv, head_dim,   window,   softcap};
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * head_dim;
  const uint64_t kv_elems = (uint64_t)cache_to * n_head_kv * head_dim;
  if (n_head_q == 0 || head_dim == 0 || (uint64_t)q_f32Len != q_elems ||
      (uint64_t)k_cacheLen != kv_elems || (uint64_t)v_cacheLen != kv_elems ||
      (uint64_t)out_f32Len != q_elems || stats_usLen < 1 ||
      (sinksLen != 0 && (uint32)sinksLen != n_head_q)) {
    FARF(ERROR, "attn_f16_decode: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_f16_io io = {
    q_f32,   n_head_q * head_dim,  k_cache,
    v_cache, n_head_kv * head_dim, sinksLen ? sinks : NULL,
    out_f32, n_head_q * head_dim};
  uint64_t us = 0;
  int res = hvx_attn_decode_f16(&shape, &io, s->quant_pool, &us);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_f16_decode: kernel failed: 0x%08x", res);
    return res;
  }
  stats_us[0] = (uint32)us;
  return AEE_SUCCESS;
}

/* --- resident KV tiles --------------------------------------------------- */

int nntr_hvx_kv_register_f16(remote_handle64 handle, uint32 max_rows,
                             uint32 n_head_kv, uint32 head_dim,
                             uint32 *kv_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !kv_handle) {
    return AEE_EBADPARM;
  }
  return hexkl_kv_tiles_f16_register(&s->kv_tiles, max_rows, n_head_kv,
                                     head_dim, kv_handle);
}

int nntr_hvx_kv_release_f16(remote_handle64 handle, uint32 kv_handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  return hexkl_kv_tiles_f16_release(&s->kv_tiles, kv_handle);
}

int nntr_hvx_kv_append_f16(remote_handle64 handle, uint32 kv_handle,
                           uint32 row0, const uint16 *k_rows, int k_rowsLen,
                           const uint16 *v_rows, int v_rowsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hexkl_kv_tiles_f16 *kv =
    hexkl_kv_tiles_f16_get(&s->kv_tiles, kv_handle);
  if (!kv) {
    return AEE_EBADPARM;
  }
  const uint32_t stride = kv->n_head_kv * kv->head_dim;
  if (k_rowsLen != v_rowsLen || k_rowsLen <= 0 ||
      ((uint32_t)k_rowsLen % stride) != 0) {
    FARF(ERROR, "kv_append_f16: bad lengths");
    return AEE_EBADPARM;
  }
  return hexkl_kv_tiles_f16_append(&s->kv_tiles, kv_handle, row0,
                                   (uint32_t)k_rowsLen / stride, k_rows,
                                   v_rows);
}

int nntr_hvx_attn_f16_prefill_resident(
  remote_handle64 handle, uint32 kv_handle, uint32 n_q, uint32 cache_from,
  uint32 cache_to, uint32 n_head_q, uint32 window, uint32 br, uint32 bc,
  float softcap, const float *q_f32, int q_f32Len, const float *sinks,
  int sinksLen, float *out_f32, int out_f32Len, uint32 *stats_us,
  int stats_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (s->hmx_fp16_rate == 0) {
    return AEE_EUNSUPPORTED;
  }
  const hexkl_kv_tiles_f16 *kv =
    hexkl_kv_tiles_f16_get(&s->kv_tiles, kv_handle);
  if (!kv || cache_to > kv->max_rows) {
    FARF(ERROR, "attn_f16_prefill_resident: bad handle or cache_to");
    return AEE_EBADPARM;
  }
  hexkl_attn_f16_shape shape = {n_q,           cache_from,   cache_to, n_head_q,
                                kv->n_head_kv, kv->head_dim, window,   softcap};
  hexkl_attn_f16_tiling tiling = {br, bc, 0, 0};
  if (br == 0 || bc == 0) {
    hexkl_attn_f16_choose_tiling(&shape, &tiling);
  }
  if (hexkl_attn_f16_tiling_init(&shape, &tiling) != HEXKL_ATTN_OK) {
    return AEE_EBADPARM;
  }
  const uint64_t q_elems = (uint64_t)n_q * n_head_q * kv->head_dim;
  if ((uint64_t)q_f32Len != q_elems || (uint64_t)out_f32Len != q_elems ||
      stats_usLen < STAT_COUNT ||
      (sinksLen != 0 && (uint32)sinksLen != n_head_q)) {
    FARF(ERROR, "attn_f16_prefill_resident: bad lengths");
    return AEE_EBADPARM;
  }
  hexkl_attn_f16_io io;
  memset(&io, 0, sizeof(io));
  io.q = q_f32;
  io.q_stride = n_head_q * kv->head_dim;
  io.sinks = sinksLen ? sinks : NULL;
  io.out = out_f32;
  io.out_stride = n_head_q * kv->head_dim;
  io.kt_tiles = kv->kt;
  io.v_tiles = kv->v;
  io.kv_tile_cols = kv->n_col_tiles;
  hexkl_attn_f16_stats st;
  int res =
    hexkl_attn_f16_prefill(s->vtcm_base, s->config_off, s->hmx_fp16_rate,
                           &shape, &tiling, &io, s->quant_pool, &st);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "attn_f16_prefill_resident: kernel failed: 0x%08x", res);
    return res;
  }
  stats_us[STAT_QPREP] = (uint32)st.us_qprep;
  stats_us[STAT_DMA] = (uint32)st.us_dma;
  stats_us[STAT_TILE] = (uint32)st.us_tile;
  stats_us[STAT_QK] = (uint32)st.us_qk;
  stats_us[STAT_SOFTMAX] = (uint32)st.us_softmax;
  stats_us[STAT_PV] = (uint32)st.us_pv;
  stats_us[STAT_STORE] = (uint32)st.us_store;
  stats_us[STAT_N_BLOCKS] = st.n_blocks;
  return AEE_SUCCESS;
}
