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

/** vmpy(Vh,Vh):<<1:rnd:sat as a scalar: (a*b*2 + 2^15) >> 16, saturated. */
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

#include <HAP_perf.h>
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

/**
 * Loop order. Pass 1 runs one row-pair vector v over every tile with the
 * running max in a register; pass 2 runs one 4-row group w over every
 * tile with the two max vectors and the row-sum accumulator in registers,
 * two tiles per iteration so the two exp2 chains overlap in the pipeline.
 * The per-tile correction vectors are staged once in scratch.
 */
void hvx_softmax_q(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                   const int16_t *corr, uint8_t *p_tiles, int32_t *rowsum,
                   void *scratch) {
  const uint32_t ct = b->n_col_tiles;
  const int F = b->frac_bits;
  const int rho_pair = ((int)b->rho_q15 << 16) | (int)b->rho_q15;
  HVX_Vector *M = (HVX_Vector *)scratch;             /* [32] row max    */
  HVX_Vector *RS = (HVX_Vector *)scratch + 32;       /* [16] row sums   */
  HVX_Vector *CV = (HVX_Vector *)scratch + 48;       /* [ct] corrections */
  const HVX_Vector *S = (const HVX_Vector *)s_tiles; /* [ct][32]        */
  HVX_Vector *P = (HVX_Vector *)p_tiles;             /* [ct][16]        */

  const HVX_VectorPred qlow = Q6_Q_vsetq_R(64);
  const HVX_Vector vmin = Q6_Vh_vsplat_R(-32768);
  const HVX_Vector colbase = Q6_Vh_vadd_VhVh(*(const HVX_Vector *)lane_col,
                                             Q6_Vh_vsplat_R((int)b->col0));
  const HVX_Vector rowbase = Q6_V_vmux_QVV(qlow, Q6_Vh_vsplat_R((int)b->row0),
                                           Q6_Vh_vsplat_R((int)b->row0 + 1));

  uint8_t cls[HVX_SOFTMAX_Q_MAX_TILES];
  for (uint32_t c = 0; c < ct; ++c) {
    cls[c] = (uint8_t)tile_class(b, c);
    CV[c] = corr ? corr_vec(corr, c, qlow) : Q6_V_vzero();
  }

  /** Pass 1: row max of the corrected, masked scores, four row-pair
   * vectors per sweep over the tiles so the branches and the 4 KiB stride
   * are paid once per four vectors. */
  for (uint32_t v = 0; v < 32u; v += 4u) {
    HVX_Vector m0 = vmin, m1 = vmin, m2 = vmin, m3 = vmin;
    const HVX_Vector rowv0 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v));
    const HVX_Vector rowv1 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v + 2));
    const HVX_Vector rowv2 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v + 4));
    const HVX_Vector rowv3 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v + 6));
    for (uint32_t c = 0; c < ct; ++c) {
      if (cls[c] == TILE_MASKED) {
        continue;
      }
      const HVX_Vector *Sc = S + 32u * c + v;
      const HVX_Vector cv = CV[c];
      HVX_Vector s0 = Q6_Vh_vsub_VhVh(Sc[0], cv);
      HVX_Vector s1 = Q6_Vh_vsub_VhVh(Sc[1], cv);
      HVX_Vector s2 = Q6_Vh_vsub_VhVh(Sc[2], cv);
      HVX_Vector s3 = Q6_Vh_vsub_VhVh(Sc[3], cv);
      if (cls[c] == TILE_PARTIAL) {
        const HVX_Vector colv =
          Q6_Vh_vadd_VhVh(colbase, Q6_Vh_vsplat_R(32 * (int)c));
        s0 = Q6_V_vmux_QVV(mask_q(b, rowv0, colv), vmin, s0);
        s1 = Q6_V_vmux_QVV(mask_q(b, rowv1, colv), vmin, s1);
        s2 = Q6_V_vmux_QVV(mask_q(b, rowv2, colv), vmin, s2);
        s3 = Q6_V_vmux_QVV(mask_q(b, rowv3, colv), vmin, s3);
      }
      m0 = Q6_Vh_vmax_VhVh(m0, s0);
      m1 = Q6_Vh_vmax_VhVh(m1, s1);
      m2 = Q6_Vh_vmax_VhVh(m2, s2);
      m3 = Q6_Vh_vmax_VhVh(m3, s3);
    }
    M[v] = rowmax_bcast(m0, qlow);
    M[v + 1u] = rowmax_bcast(m1, qlow);
    M[v + 2u] = rowmax_bcast(m2, qlow);
    M[v + 3u] = rowmax_bcast(m3, qlow);
  }

  /* Pass 2: P' and row sums. */
  const HVX_Vector fmask = Q6_Vh_vsplat_R((1 << F) - 1);
  const HVX_Vector vA = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_A);
  const HVX_Vector vB = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_B);
  const HVX_Vector v16384 = Q6_Vh_vsplat_R(16384);
  const HVX_Vector v5 = Q6_Vh_vsplat_R(5);
  const HVX_Vector v15 = Q6_Vh_vsplat_R(15);
  const HVX_Vector one = Q6_Vh_vsplat_R(1);
  const HVX_Vector zero = Q6_V_vzero();
  /* One full tile for the 4-row group w: t -> q -> P' as 128 uint8. */
#define SMX_TILE_FULL(Sc, cv, out_p8)                                          \
  do {                                                                         \
    const HVX_Vector t0_ =                                                     \
      Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh((Sc)[v0], (cv)), m0);                \
    const HVX_Vector t1_ =                                                     \
      Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh((Sc)[v0 + 1u], (cv)), m1);           \
    const HVX_Vector q0_ =                                                     \
      exp2_q(t0_, rho_pair, F, fmask, vA, vB, v16384, v5, v15);                \
    const HVX_Vector q1_ =                                                     \
      exp2_q(t1_, rho_pair, F, fmask, vA, vB, v16384, v5, v15);                \
    (out_p8) =                                                                 \
      Q6_Vub_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q1_, one), 1),      \
                            Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q0_, one), 1));     \
  } while (0)

  for (uint32_t w = 0; w < 16u; ++w) {
    const uint32_t v0 = 2u * w;
    const HVX_Vector m0 = M[v0], m1 = M[v0 + 1u];
    const HVX_Vector rowv0 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v0));
    const HVX_Vector rowv1 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v0 + 2));
    HVX_Vector rs = zero;
    uint32_t c = 0;
    while (c < ct) {
      /* Four full tiles: eight independent exp2 chains in flight. */
      if (c + 4u <= ct && cls[c] == TILE_FULL && cls[c + 1u] == TILE_FULL &&
          cls[c + 2u] == TILE_FULL && cls[c + 3u] == TILE_FULL) {
        const HVX_Vector *Sa = S + 32u * c;
        HVX_Vector pa, pb, pc, pd;
        SMX_TILE_FULL(Sa, CV[c], pa);
        SMX_TILE_FULL(Sa + 32u, CV[c + 1u], pb);
        SMX_TILE_FULL(Sa + 64u, CV[c + 2u], pc);
        SMX_TILE_FULL(Sa + 96u, CV[c + 3u], pd);
        P[16u * c + w] = pa;
        P[16u * (c + 1u) + w] = pb;
        P[16u * (c + 2u) + w] = pc;
        P[16u * (c + 3u) + w] = pd;
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, pa, 0x01010101u);
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, pb, 0x01010101u);
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, pc, 0x01010101u);
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, pd, 0x01010101u);
        c += 4u;
        continue;
      }
      if (c + 2u <= ct && cls[c] == TILE_FULL && cls[c + 1u] == TILE_FULL) {
        const HVX_Vector *Sa = S + 32u * c;
        HVX_Vector pa, pb;
        SMX_TILE_FULL(Sa, CV[c], pa);
        SMX_TILE_FULL(Sa + 32u, CV[c + 1u], pb);
        P[16u * c + w] = pa;
        P[16u * (c + 1u) + w] = pb;
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, pa, 0x01010101u);
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, pb, 0x01010101u);
        c += 2u;
        continue;
      }
      if (cls[c] == TILE_MASKED) {
        P[16u * c + w] = zero;
        ++c;
        continue;
      }
      /* One tile, full or partial. */
      const HVX_Vector *Sc = S + 32u * c;
      const HVX_Vector cv = CV[c];
      HVX_Vector t0 = Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(Sc[v0], cv), m0);
      HVX_Vector t1 = Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(Sc[v0 + 1u], cv), m1);
      if (cls[c] == TILE_PARTIAL) {
        const HVX_Vector colv =
          Q6_Vh_vadd_VhVh(colbase, Q6_Vh_vsplat_R(32 * (int)c));
        t0 = Q6_V_vmux_QVV(mask_q(b, rowv0, colv), vmin, t0);
        t1 = Q6_V_vmux_QVV(mask_q(b, rowv1, colv), vmin, t1);
      }
      const HVX_Vector q0 =
        exp2_q(t0, rho_pair, F, fmask, vA, vB, v16384, v5, v15);
      const HVX_Vector q1 =
        exp2_q(t1, rho_pair, F, fmask, vA, vB, v16384, v5, v15);
      const HVX_Vector p8 =
        Q6_Vub_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q1, one), 1),
                              Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q0, one), 1));
      P[16u * c + w] = p8;
      rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, p8, 0x01010101u);
      ++c;
    }
    RS[w] = rs;
  }
#undef SMX_TILE_FULL
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

void hvx_softmax_q_rate(uint32_t n, void *vtcm, void *vtcm_big,
                        uint32_t out[12]) {
  HVX_Vector *mem = (HVX_Vector *)vtcm;
  HVX_Vector a0 = Q6_Vh_vsplat_R(1), a1 = Q6_Vh_vsplat_R(2),
             a2 = Q6_Vh_vsplat_R(3), a3 = Q6_Vh_vsplat_R(4),
             a4 = Q6_Vh_vsplat_R(5), a5 = Q6_Vh_vsplat_R(6),
             a6 = Q6_Vh_vsplat_R(7), a7 = Q6_Vh_vsplat_R(8);
  const HVX_Vector k = Q6_Vh_vsplat_R(3);
  for (uint32_t i = 0; i < 8u; ++i) {
    mem[i] = Q6_Vh_vsplat_R((int)i - 300);
  }
  /* 8 independent vadd chains. */
  uint64_t c0 = HAP_perf_get_pcycles();
  for (uint32_t i = 0; i < n; ++i) {
    a0 = Q6_Vh_vadd_VhVh(a0, k);
    a1 = Q6_Vh_vadd_VhVh(a1, k);
    a2 = Q6_Vh_vadd_VhVh(a2, k);
    a3 = Q6_Vh_vadd_VhVh(a3, k);
    a4 = Q6_Vh_vadd_VhVh(a4, k);
    a5 = Q6_Vh_vadd_VhVh(a5, k);
    a6 = Q6_Vh_vadd_VhVh(a6, k);
    a7 = Q6_Vh_vadd_VhVh(a7, k);
  }
  uint64_t c1 = HAP_perf_get_pcycles();
  mem[8] = Q6_Vh_vadd_VhVh(
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a0, a1), Q6_Vh_vadd_VhVh(a2, a3)),
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a4, a5), Q6_Vh_vadd_VhVh(a6, a7)));
  out[0] = (uint32_t)((c1 - c0) / (8u * (uint64_t)n));

  /* 8 independent Q15 multiply chains. */
  const HVX_Vector km = Q6_Vh_vsplat_R(30000);
  c0 = HAP_perf_get_pcycles();
  for (uint32_t i = 0; i < n; ++i) {
    a0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a0, km);
    a1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a1, km);
    a2 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a2, km);
    a3 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a3, km);
    a4 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a4, km);
    a5 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a5, km);
    a6 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a6, km);
    a7 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a7, km);
  }
  c1 = HAP_perf_get_pcycles();
  mem[9] = Q6_Vh_vadd_VhVh(
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a0, a1), Q6_Vh_vadd_VhVh(a2, a3)),
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a4, a5), Q6_Vh_vadd_VhVh(a6, a7)));
  out[1] = (uint32_t)((c1 - c0) / (8u * (uint64_t)n));

  /* 4 independent exp2 chains, as pass 2 runs them. */
  const HVX_Vector fmask = Q6_Vh_vsplat_R(255),
                   vA = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_A),
                   vB = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_B),
                   v16384 = Q6_Vh_vsplat_R(16384), v5 = Q6_Vh_vsplat_R(5),
                   v15 = Q6_Vh_vsplat_R(15);
  const int rho = (30000 << 16) | 30000;
  HVX_Vector t0 = mem[0], t1 = mem[1], t2 = mem[2], t3 = mem[3];
  c0 = HAP_perf_get_pcycles();
  for (uint32_t i = 0; i < n; ++i) {
    t0 = Q6_Vh_vsub_VhVh(exp2_q(t0, rho, 8, fmask, vA, vB, v16384, v5, v15), k);
    t1 = Q6_Vh_vsub_VhVh(exp2_q(t1, rho, 8, fmask, vA, vB, v16384, v5, v15), k);
    t2 = Q6_Vh_vsub_VhVh(exp2_q(t2, rho, 8, fmask, vA, vB, v16384, v5, v15), k);
    t3 = Q6_Vh_vsub_VhVh(exp2_q(t3, rho, 8, fmask, vA, vB, v16384, v5, v15), k);
  }
  c1 = HAP_perf_get_pcycles();
  mem[10] = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(t0, t1), Q6_Vh_vadd_VhVh(t2, t3));
  out[2] = (uint32_t)((c1 - c0) / (4u * (uint64_t)n));

  /* 8 VTCM loads per iteration, accumulated. */
  a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = Q6_V_vzero();
  c0 = HAP_perf_get_pcycles();
  for (uint32_t i = 0; i < n; ++i) {
    a0 = Q6_Vh_vadd_VhVh(a0, mem[0]);
    a1 = Q6_Vh_vadd_VhVh(a1, mem[1]);
    a2 = Q6_Vh_vadd_VhVh(a2, mem[2]);
    a3 = Q6_Vh_vadd_VhVh(a3, mem[3]);
    a4 = Q6_Vh_vadd_VhVh(a4, mem[4]);
    a5 = Q6_Vh_vadd_VhVh(a5, mem[5]);
    a6 = Q6_Vh_vadd_VhVh(a6, mem[6]);
    a7 = Q6_Vh_vadd_VhVh(a7, mem[7]);
    mem[11] = a0; /* keep the loads from being hoisted */
  }
  c1 = HAP_perf_get_pcycles();
  mem[12] = Q6_Vh_vadd_VhVh(
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a0, a1), Q6_Vh_vadd_VhVh(a2, a3)),
    Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(a4, a5), Q6_Vh_vadd_VhVh(a6, a7)));
  out[3] = (uint32_t)((c1 - c0) / (8u * (uint64_t)n));

  /* Streaming over 512 KiB (4096 vectors): each load consumed at once. */
  const HVX_Vector *big = (const HVX_Vector *)vtcm_big;
  const uint32_t nb = 4096u;
  const uint32_t reps = n / 64u + 1u;
  a0 = Q6_V_vzero();
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t i = 0; i < nb; ++i) {
      a0 = Q6_Vh_vmax_VhVh(a0, Q6_Vh_vsub_VhVh(big[i], k));
    }
  }
  c1 = HAP_perf_get_pcycles();
  mem[13] = a0;
  out[4] = (uint32_t)((c1 - c0) / ((uint64_t)nb * reps));

  /* The same, loads issued one iteration ahead. */
  a0 = Q6_V_vzero();
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    HVX_Vector nxt = big[0];
    for (uint32_t i = 0; i + 1u < nb; ++i) {
      const HVX_Vector curv = nxt;
      nxt = big[i + 1u];
      a0 = Q6_Vh_vmax_VhVh(a0, Q6_Vh_vsub_VhVh(curv, k));
    }
    a0 = Q6_Vh_vmax_VhVh(a0, Q6_Vh_vsub_VhVh(nxt, k));
  }
  c1 = HAP_perf_get_pcycles();
  mem[14] = a0;
  out[5] = (uint32_t)((c1 - c0) / ((uint64_t)nb * reps));

  /** Pass-1 pattern: for each of 32 row-pair vectors, walk 128 tiles at a
   * 4 KiB stride, subtract a per-tile vector, running max. No class
   * checks. 4096 vector visits per rep. */
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t v = 0; v < 32u; ++v) {
      HVX_Vector m = Q6_Vh_vsplat_R(-32768);
      for (uint32_t c = 0; c < 128u; ++c) {
        m = Q6_Vh_vmax_VhVh(m, Q6_Vh_vsub_VhVh(big[32u * c + v], mem[c & 7u]));
      }
      mem[16u + (v & 7u)] = m;
    }
  }
  c1 = HAP_perf_get_pcycles();
  out[6] = (uint32_t)((c1 - c0) / ((uint64_t)nb * reps));

  /* The same visits in tile-major order (stride 128 B). */
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t c = 0; c < 128u; ++c) {
      HVX_Vector m = Q6_Vh_vsplat_R(-32768);
      const HVX_Vector cv = mem[c & 7u];
      for (uint32_t v = 0; v < 32u; ++v) {
        m = Q6_Vh_vmax_VhVh(m, Q6_Vh_vsub_VhVh(big[32u * c + v], cv));
      }
      mem[16u + (c & 7u)] = m;
    }
  }
  c1 = HAP_perf_get_pcycles();
  out[7] = (uint32_t)((c1 - c0) / ((uint64_t)nb * reps));

  /** Pass-2 body, two row-pair vectors per visit, over 64 tiles (the P'
   * store goes to the upper half of the region): w-major, 4 KiB stride. */
  HVX_Vector *pst = (HVX_Vector *)vtcm_big + 2048u;
  const uint32_t nt = 64u;
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t w = 0; w < 16u; ++w) {
      const HVX_Vector m0 = mem[w & 7u], m1 = mem[(w + 1u) & 7u];
      HVX_Vector rs = Q6_V_vzero();
      for (uint32_t c = 0; c < nt; ++c) {
        const HVX_Vector cv = mem[c & 7u];
        const HVX_Vector t0 =
          Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(big[32u * c + 2u * w], cv), m0);
        const HVX_Vector t1 = Q6_Vh_vsub_VhVh_sat(
          Q6_Vh_vsub_VhVh(big[32u * c + 2u * w + 1u], cv), m1);
        const HVX_Vector q0 =
          exp2_q(t0, rho, 8, fmask, vA, vB, v16384, v5, v15);
        const HVX_Vector q1 =
          exp2_q(t1, rho, 8, fmask, vA, vB, v16384, v5, v15);
        const HVX_Vector p8 =
          Q6_Vub_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q1, k), 1),
                                Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q0, k), 1));
        pst[16u * c + w] = p8;
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, p8, 0x01010101u);
      }
      mem[24u + (w & 7u)] = rs;
    }
  }
  c1 = HAP_perf_get_pcycles();
  out[8] = (uint32_t)((c1 - c0) / ((uint64_t)nt * 32u * reps));

  /* The same body, tile-major (stride 128 B loads, 128 B stores). */
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t c = 0; c < nt; ++c) {
      const HVX_Vector cv = mem[c & 7u];
      HVX_Vector rs = Q6_V_vzero();
      for (uint32_t w = 0; w < 16u; ++w) {
        const HVX_Vector m0 = mem[w & 7u], m1 = mem[(w + 1u) & 7u];
        const HVX_Vector t0 =
          Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(big[32u * c + 2u * w], cv), m0);
        const HVX_Vector t1 = Q6_Vh_vsub_VhVh_sat(
          Q6_Vh_vsub_VhVh(big[32u * c + 2u * w + 1u], cv), m1);
        const HVX_Vector q0 =
          exp2_q(t0, rho, 8, fmask, vA, vB, v16384, v5, v15);
        const HVX_Vector q1 =
          exp2_q(t1, rho, 8, fmask, vA, vB, v16384, v5, v15);
        const HVX_Vector p8 =
          Q6_Vub_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q1, k), 1),
                                Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q0, k), 1));
        pst[16u * c + w] = p8;
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, p8, 0x01010101u);
      }
      mem[24u + (c & 7u)] = rs;
    }
  }
  c1 = HAP_perf_get_pcycles();
  out[9] = (uint32_t)((c1 - c0) / ((uint64_t)nt * 32u * reps));

  /** w-major again with padded tile strides: 33 vectors between score
   * tiles, 17 between P' tiles, so consecutive tiles change VTCM bank. */
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t w = 0; w < 16u; ++w) {
      const HVX_Vector m0 = mem[w & 7u], m1 = mem[(w + 1u) & 7u];
      HVX_Vector rs = Q6_V_vzero();
      for (uint32_t c = 0; c < nt; ++c) {
        const HVX_Vector cv = mem[c & 7u];
        const HVX_Vector t0 =
          Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(big[33u * c + 2u * w], cv), m0);
        const HVX_Vector t1 = Q6_Vh_vsub_VhVh_sat(
          Q6_Vh_vsub_VhVh(big[33u * c + 2u * w + 1u], cv), m1);
        const HVX_Vector q0 =
          exp2_q(t0, rho, 8, fmask, vA, vB, v16384, v5, v15);
        const HVX_Vector q1 =
          exp2_q(t1, rho, 8, fmask, vA, vB, v16384, v5, v15);
        const HVX_Vector p8 =
          Q6_Vub_vpack_VhVh_sat(Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q1, k), 1),
                                Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(q0, k), 1));
        pst[17u * c + w] = p8;
        rs = Q6_Vuw_vrmpyacc_VuwVubRub(rs, p8, 0x01010101u);
      }
      mem[24u + (w & 7u)] = rs;
    }
  }
  c1 = HAP_perf_get_pcycles();
  out[10] = (uint32_t)((c1 - c0) / ((uint64_t)nt * 32u * reps));

  /* Pass-1 pattern at the padded stride. */
  c0 = HAP_perf_get_pcycles();
  for (uint32_t r = 0; r < reps; ++r) {
    for (uint32_t v = 0; v < 32u; ++v) {
      HVX_Vector m = Q6_Vh_vsplat_R(-32768);
      for (uint32_t c = 0; c < 120u; ++c) {
        m = Q6_Vh_vmax_VhVh(m, Q6_Vh_vsub_VhVh(big[33u * c + v], mem[c & 7u]));
      }
      mem[16u + (v & 7u)] = m;
    }
  }
  c1 = HAP_perf_get_pcycles();
  out[11] = (uint32_t)((c1 - c0) / ((uint64_t)120u * 32u * reps));
}

#endif /* __hexagon__ */
