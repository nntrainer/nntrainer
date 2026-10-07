// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_softmax_q.c
 * @date   06 Oct 2026
 * @brief  Integer softmax over int16 log2-domain scores: scalar definition
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "hvx_softmax_q.h"

#include <math.h>
#include <string.h>

int hvx_softmax_q_scale(float alpha, uint32_t frac_bits, int *k,
                        uint16_t *rho_q15) {
  if (!(alpha > 0.0f) || frac_bits == 0 || frac_bits > 8 || !k || !rho_q15) {
    return -1;
  }
  /* rho = alpha * 2^F * 512 / 2^k in (0.5, 1]: k = ceil(log2(alpha*2^F*512)) */
  const double target = (double)alpha * (double)(1u << frac_bits) * 512.0;
  int kk = (int)ceil(log2(target));
  double rho = target / ldexp(1.0, kk);
  if (rho <= 0.5) { /* log2 landed exactly on a power of two from below */
    --kk;
    rho = target / ldexp(1.0, kk);
  }
  if (rho > 1.0) {
    ++kk;
    rho = target / ldexp(1.0, kk);
  }
  if (kk < -14 || kk > 15) {
    return -1; /* 2^k must be an fp16 normal */
  }
  const double q = rho * 32768.0;
  const long r = lround(q);
  *rho_q15 = (uint16_t)(r > 32767 ? 32767 : r);
  *k = kk;
  return 0;
}

/* vmpy(Vh,Vh):<<1:rnd:sat and its scalar twin: (a*b*2 + 2^15) >> 16, saturated.
 */
static inline int32_t q15_mul(int32_t a, int32_t b) {
  const int32_t v = (int32_t)(((int64_t)a * b * 2 + 32768) >> 16);
  return v > 32767 ? 32767 : v < -32768 ? -32768 : v;
}

static inline int32_t sat16(int32_t v) {
  return v > 32767 ? 32767 : v < -32768 ? -32768 : v;
}

uint16_t hvx_softmax_q_exp2_q14(uint32_t f, uint32_t frac_bits) {
  const int32_t x = (int32_t)(f << (15u - frac_bits)); /* Q15 in [0, 1) */
  int32_t p = sat16(q15_mul(x, HVX_SOFTMAX_Q_EXP2_C) + HVX_SOFTMAX_Q_EXP2_B);
  p = sat16(q15_mul(p, x) + HVX_SOFTMAX_Q_EXP2_A);
  const int32_t m = q15_mul(p, x); /* 2^x - 1, Q15, in [0, 32767] */
  return (uint16_t)(16384 + (m >> 1));
}

static inline int masked(const hvx_softmax_q_block *b, uint32_t r,
                         uint32_t col) {
  const uint32_t row = b->row0 + r;
  if (r >= b->n_rows || col >= b->col0 + b->n_cols || col > row) {
    return 1;
  }
  if (b->window != 0 && col + b->window <= row) {
    return 1;
  }
  return 0;
}

void hvx_softmax_q_ref(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                       const int16_t *corr, uint8_t *p_tiles, int32_t *rowsum) {
  const uint32_t ct = b->n_col_tiles;
  const uint32_t F = b->frac_bits;
  int16_t m[HVX_SOFTMAX_Q_ROWS];
  int32_t sum[HVX_SOFTMAX_Q_ROWS];
  for (uint32_t r = 0; r < HVX_SOFTMAX_Q_ROWS; ++r) {
    m[r] = INT16_MIN;
    sum[r] = 0;
  }
  /* Pass 1: corrected score and row max over unmasked entries. */
  for (uint32_t c = 0; c < ct; ++c) {
    for (uint32_t r = 0; r < HVX_SOFTMAX_Q_ROWS; ++r) {
      for (uint32_t j = 0; j < 32u; ++j) {
        if (masked(b, r, b->col0 + 32u * c + j)) {
          continue;
        }
        const uint16_t raw =
          (uint16_t)s_tiles[c * HVX_SOFTMAX_Q_TILE + r * 32u + j];
        const uint16_t cc = corr ? (uint16_t)corr[c * 32u + j] : 0u;
        const int16_t s = (int16_t)(uint16_t)(raw - cc);
        if (s > m[r]) {
          m[r] = s;
        }
      }
    }
  }
  /* Pass 2: t, u, P', row sum. */
  for (uint32_t c = 0; c < ct; ++c) {
    for (uint32_t r = 0; r < HVX_SOFTMAX_Q_ROWS; ++r) {
      for (uint32_t j = 0; j < 32u; ++j) {
        const uint32_t at = c * HVX_SOFTMAX_Q_TILE + r * 32u + j;
        if (masked(b, r, b->col0 + 32u * c + j)) {
          p_tiles[at] = 0;
          continue;
        }
        const uint16_t raw = (uint16_t)s_tiles[at];
        const uint16_t cc = corr ? (uint16_t)corr[c * 32u + j] : 0u;
        const int16_t s = (int16_t)(uint16_t)(raw - cc);
        int32_t t = (int32_t)s - (int32_t)m[r]; /* <= 0 */
        if (t < INT16_MIN) {
          t = INT16_MIN; /* vsub:sat */
        }
        const int32_t u = q15_mul(t, b->rho_q15); /* <= 0 */
        const int32_t e = u >> F;                 /* floor, <= 0 */
        const uint32_t f = (uint32_t)u & ((1u << F) - 1u);
        const int32_t T = hvx_softmax_q_exp2_q14(f, F); /* [16384, 32767] */
        /** T >> (5 - e) is 2^(u/2^F) * 512; the shift clamps at 15, where
         * everything that would round to 0 lands, and the halving below
         * rounds. e = 0 gives q = 512 -> 256 -> saturates to 255. */
        int32_t sh = 5 - e;
        if (sh > 15) {
          sh = 15;
        }
        const int32_t q = T >> sh;
        uint32_t p = (uint32_t)(q + 1) >> 1;
        if (p > 255u) {
          p = 255u;
        }
        p_tiles[at] = (uint8_t)p;
        sum[r] += (int32_t)p;
      }
    }
  }
  for (uint32_t r = 0; r < HVX_SOFTMAX_Q_ROWS; ++r) {
    rowsum[hvx_softmax_q_rowsum_index(r)] = sum[r];
  }
}

#if defined(__hexagon__)

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** Column index of each lane in a 64-lane int16 vector of a tile: two rows
 * of 32 columns. Loaded once per call from cached memory. */
static const int16_t lane_col[64] __attribute__((aligned(128))) = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

enum { TILE_FULL = 0, TILE_PARTIAL = 1, TILE_MASKED = 2 };

/** Whether every lane of tile c is visible to every row of the block, no
 * lane is, or it depends on the lane. Rows past n_rows make the whole
 * block partial, which only the last block of a sequence pays. */
static int tile_class(const hvx_softmax_q_block *b, uint32_t c) {
  const uint32_t cl = b->col0 + 32u * c, ch = cl + 31u;
  const uint32_t rl = b->row0, rh = b->row0 + HVX_SOFTMAX_Q_ROWS - 1u;
  const uint32_t cend = b->col0 + b->n_cols;
  if (cl > rh || cl >= cend || (b->window != 0 && ch + b->window <= rl)) {
    return TILE_MASKED;
  }
  if (ch <= rl && ch < cend && b->n_rows == HVX_SOFTMAX_Q_ROWS &&
      (b->window == 0 || cl + b->window > rh)) {
    return TILE_FULL;
  }
  return TILE_PARTIAL;
}

/* True where (row, col) is masked, for the vector v of tile c. */
static inline HVX_VectorPred mask_q(const hvx_softmax_q_block *b,
                                    HVX_Vector rowv, HVX_Vector colv) {
  HVX_VectorPred q = Q6_Q_vcmp_gt_VhVh(colv, rowv); /* col > row */
  q = Q6_Q_or_QQ(
    q, Q6_Q_vcmp_gt_VhVh(colv, Q6_Vh_vsplat_R((int)(b->col0 + b->n_cols) - 1)));
  q = Q6_Q_or_QQ(
    q, Q6_Q_vcmp_gt_VhVh(rowv, Q6_Vh_vsplat_R((int)(b->row0 + b->n_rows) - 1)));
  if (b->window != 0) {
    /* col < row - window + 1 */
    const HVX_Vector lim =
      Q6_Vh_vsub_VhVh(rowv, Q6_Vh_vsplat_R((int)b->window - 1));
    q = Q6_Q_or_QQ(q, Q6_Q_vcmp_gt_VhVh(lim, colv));
  }
  return q;
}

/** The per-column correction of tile c replicated into both halves. corr
 * is read 128 bytes at a time, so it must extend 32 entries past the last
 * tile. */
static inline HVX_Vector corr_vec(const int16_t *corr, uint32_t c,
                                  HVX_VectorPred qlow) {
  const HVX_Vector u = *(const HVX_UVector *)(corr + 32u * c);
  return Q6_V_vmux_QVV(qlow, u, Q6_V_vror_VR(u, 64));
}

/** Lanes 0..31 -> max over lanes 0..31, lanes 32..63 -> max over 32..63:
 * interleave the halves so a rotation by an even lane count keeps the
 * rows apart, five rotate-max steps, deal them back. */
static inline HVX_Vector rowmax_bcast(HVX_Vector m, HVX_VectorPred qlow) {
  HVX_Vector w = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(m, 64), m, -2));
  w = Q6_Vh_vmax_VhVh(w, Q6_V_vror_VR(w, 4));
  w = Q6_Vh_vmax_VhVh(w, Q6_V_vror_VR(w, 8));
  w = Q6_Vh_vmax_VhVh(w, Q6_V_vror_VR(w, 16));
  w = Q6_Vh_vmax_VhVh(w, Q6_V_vror_VR(w, 32));
  w = Q6_Vh_vmax_VhVh(w, Q6_V_vror_VR(w, 64));
  const HVX_VectorPair d = Q6_W_vdeal_VVR(w, w, -2);
  return Q6_V_vmux_QVV(qlow, Q6_V_lo_W(d), Q6_V_hi_W(d));
}

/** t (<= 0, int16) -> q = 2^(t*rho / 2^F) * 512 as int16, before the final
 * rounding halve. */
static inline HVX_Vector exp2_q(HVX_Vector t, int rho_pair, int F,
                                HVX_Vector fmask, HVX_Vector vA, HVX_Vector vB,
                                HVX_Vector v16384, HVX_Vector v5,
                                HVX_Vector v15) {
  const HVX_Vector u = Q6_Vh_vmpy_VhRh_s1_rnd_sat(t, rho_pair);
  const HVX_Vector e = Q6_Vh_vasr_VhR(u, F);
  const HVX_Vector x = Q6_Vh_vasl_VhR(Q6_V_vand_VV(u, fmask), 15 - F);
  HVX_Vector p = Q6_Vh_vmpy_VhRh_s1_rnd_sat(x, (HVX_SOFTMAX_Q_EXP2_C << 16) |
                                                 HVX_SOFTMAX_Q_EXP2_C);
  p = Q6_Vh_vadd_VhVh_sat(p, vB);
  p = Q6_Vh_vmpy_VhVh_s1_rnd_sat(p, x);
  p = Q6_Vh_vadd_VhVh_sat(p, vA);
  const HVX_Vector m = Q6_Vh_vmpy_VhVh_s1_rnd_sat(p, x);
  const HVX_Vector T = Q6_Vh_vadd_VhVh(v16384, Q6_Vh_vasr_VhR(m, 1));
  const HVX_Vector sh = Q6_Vh_vmin_VhVh(Q6_Vh_vsub_VhVh(v5, e), v15);
  return Q6_Vh_vasr_VhVh(T, sh);
}

void hvx_softmax_q(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                   const int16_t *corr, uint8_t *p_tiles, int32_t *rowsum,
                   void *scratch) {
  const uint32_t ct = b->n_col_tiles;
  const int F = b->frac_bits;
  const int rho_pair = ((int)b->rho_q15 << 16) | (int)b->rho_q15;
  HVX_Vector *M = (HVX_Vector *)scratch;             /* [32] running max */
  HVX_Vector *RS = (HVX_Vector *)scratch + 32;       /* [16] row sums   */
  const HVX_Vector *S = (const HVX_Vector *)s_tiles; /* [ct][32]        */
  HVX_Vector *P = (HVX_Vector *)p_tiles;             /* [ct][16]        */

  const HVX_VectorPred qlow = Q6_Q_vsetq_R(64);
  const HVX_Vector vmin = Q6_Vh_vsplat_R(-32768);
  const HVX_Vector colbase = Q6_Vh_vadd_VhVh(*(const HVX_Vector *)lane_col,
                                             Q6_Vh_vsplat_R((int)b->col0));
  const HVX_Vector rowbase = Q6_V_vmux_QVV(qlow, Q6_Vh_vsplat_R((int)b->row0),
                                           Q6_Vh_vsplat_R((int)b->row0 + 1));

  uint8_t cls[256];
  for (uint32_t c = 0; c < ct; ++c) {
    cls[c] = (uint8_t)tile_class(b, c);
  }

  /* Pass 1: row max of the corrected, masked scores. */
  for (uint32_t v = 0; v < 32u; ++v) {
    M[v] = vmin;
  }
  for (uint32_t c = 0; c < ct; ++c) {
    if (cls[c] == TILE_MASKED) {
      continue;
    }
    const HVX_Vector cv = corr ? corr_vec(corr, c, qlow) : Q6_V_vzero();
    const HVX_Vector *Sc = S + 32u * c;
    if (cls[c] == TILE_FULL) {
      for (uint32_t v = 0; v < 32u; ++v) {
        M[v] = Q6_Vh_vmax_VhVh(M[v], Q6_Vh_vsub_VhVh(Sc[v], cv));
      }
    } else {
      const HVX_Vector colv =
        Q6_Vh_vadd_VhVh(colbase, Q6_Vh_vsplat_R(32 * (int)c));
      for (uint32_t v = 0; v < 32u; ++v) {
        const HVX_Vector rowv =
          Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v));
        const HVX_Vector s = Q6_Vh_vsub_VhVh(Sc[v], cv);
        M[v] =
          Q6_Vh_vmax_VhVh(M[v], Q6_V_vmux_QVV(mask_q(b, rowv, colv), vmin, s));
      }
    }
  }
  for (uint32_t v = 0; v < 32u; ++v) {
    M[v] = rowmax_bcast(M[v], qlow);
  }

  /* Pass 2: P' and row sums. */
  const HVX_Vector fmask = Q6_Vh_vsplat_R((1 << F) - 1);
  const HVX_Vector vA = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_A);
  const HVX_Vector vB = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_B);
  const HVX_Vector v16384 = Q6_Vh_vsplat_R(16384);
  const HVX_Vector v5 = Q6_Vh_vsplat_R(5);
  const HVX_Vector v15 = Q6_Vh_vsplat_R(15);
  const HVX_Vector one = Q6_Vh_vsplat_R(1);
  for (uint32_t w = 0; w < 16u; ++w) {
    RS[w] = Q6_V_vzero();
  }
  for (uint32_t c = 0; c < ct; ++c) {
    HVX_Vector *Pc = P + 16u * c;
    if (cls[c] == TILE_MASKED) {
      for (uint32_t w = 0; w < 16u; ++w) {
        Pc[w] = Q6_V_vzero();
      }
      continue;
    }
    const HVX_Vector cv = corr ? corr_vec(corr, c, qlow) : Q6_V_vzero();
    const HVX_Vector *Sc = S + 32u * c;
    const HVX_Vector colv =
      Q6_Vh_vadd_VhVh(colbase, Q6_Vh_vsplat_R(32 * (int)c));
    for (uint32_t w = 0; w < 16u; ++w) {
      HVX_Vector q[2];
      for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t v = 2u * w + h;
        HVX_Vector t = Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(Sc[v], cv), M[v]);
        if (cls[c] == TILE_PARTIAL) {
          const HVX_Vector rowv =
            Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v));
          t = Q6_V_vmux_QVV(mask_q(b, rowv, colv), vmin, t);
        }
        HVX_Vector qq = exp2_q(t, rho_pair, F, fmask, vA, vB, v16384, v5, v15);
        q[h] = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(qq, one), 1);
      }
      /* Rows 4w..4w+3 as 128 uint8, saturating 256 -> 255. */
      const HVX_Vector p8 = Q6_Vub_vpack_VhVh_sat(q[1], q[0]);
      Pc[w] = p8;
      RS[w] = Q6_Vuw_vrmpyacc_VuwVubRub(RS[w], p8, 0x01010101u);
    }
  }
  /* Lane 8k of RS[w] <- sum of lanes 8k..8k+7: row 4w + k. */
  HVX_Vector *out = (HVX_Vector *)rowsum;
  for (uint32_t w = 0; w < 16u; ++w) {
    HVX_Vector r = RS[w];
    r = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 4));
    r = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 8));
    r = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 16));
    out[w] = r;
  }
}

#endif /* __hexagon__ */
