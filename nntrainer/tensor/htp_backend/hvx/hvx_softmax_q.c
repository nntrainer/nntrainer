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

int hvx_softmax_q_scale_k(float alpha, uint32_t frac_bits, int k_min, int k_max,
                          int *k, uint16_t *rho_q15) {
  if (!(alpha > 0.0f) || frac_bits == 0 || frac_bits > 15 || !k || !rho_q15 ||
      k_min < -14 || k_max > 15 || k_min > k_max) {
    return -1; /* 2^k must be an fp16 normal */
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
  if (kk > k_max || kk < k_min) {
    return -1;
  }
  const double q = rho * 32768.0;
  const long r = lround(q);
  *rho_q15 = (uint16_t)(r > 32767 ? 32767 : r);
  *k = kk;
  return 0;
}

int hvx_softmax_q_scale(float alpha, uint32_t frac_bits, int *k,
                        uint16_t *rho_q15) {
  return hvx_softmax_q_scale_k(alpha, frac_bits, -14, 15, k, rho_q15);
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

/** P' = 2^(e + f) * 65536 as u16 from the Q14 2^f and the integer part
 * e <= 0: 65535 at e = 0 (t = 0), 2T at e = -1, T at e = -2, and below
 * that T >> (-e - 2) rounded via ((T >> (-e - 3)) + 1) >> 1, the shift
 * clamped at 15 where everything rounds to 0. */
static inline uint32_t p16_prime(int32_t T, int32_t e) {
  if (e == 0) {
    return 65535u;
  }
  if (e == -1) {
    return (uint32_t)T * 2u;
  }
  if (e == -2) {
    return (uint32_t)T;
  }
  int32_t sh = -e - 3;
  if (sh > 15) {
    sh = 15;
  }
  return ((uint32_t)(T >> sh) + 1u) >> 1;
}

/** 1 / x for a positive normal binary32: the 0x7EF311C7 seed and three
 * Newton steps r <- r (2 - x r), written as plain products and
 * differences so the vector code can repeat it operation for operation. */
static inline float recip_nr(float x) {
  union {
    float f;
    uint32_t u;
  } m;
  m.f = x;
  m.u = 0x7EF311C7u - m.u;
  float r = m.f;
  float xr = x * r;
  r = r * (2.0f - xr);
  xr = x * r;
  r = r * (2.0f - xr);
  xr = x * r;
  r = r * (2.0f - xr);
  return r;
}

/** R = rne(65535 * 65536 * recip(rowsum)), clamped to 65535; 0 for an
 * empty row. 65535 * 65536 = 2^32 - 2^16 is exact in binary32. */
static inline uint32_t p16_recip(uint32_t rowsum) {
  if (rowsum == 0) {
    return 0;
  }
  const float q = 4294901760.0f * recip_nr((float)rowsum);
  const float qr = rintf(q);
  return qr >= 65535.0f ? 65535u : (uint32_t)qr;
}

void hvx_softmax_q16_ref(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                         const int16_t *corr, uint8_t *p_lo, uint8_t *p_hi,
                         int32_t *rowsum) {
  const uint32_t ct = b->n_col_tiles;
  const uint32_t F = b->frac_bits;
  int16_t m[HVX_SOFTMAX_Q_ROWS];
  uint32_t sum[HVX_SOFTMAX_Q_ROWS];
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
  /* Pass 2: P' as u16 in the byte tiles, row sums of P'. */
  for (uint32_t c = 0; c < ct; ++c) {
    for (uint32_t r = 0; r < HVX_SOFTMAX_Q_ROWS; ++r) {
      for (uint32_t j = 0; j < 32u; ++j) {
        const uint32_t at = c * HVX_SOFTMAX_Q_TILE + r * 32u + j;
        if (masked(b, r, b->col0 + 32u * c + j)) {
          p_lo[at] = 0;
          p_hi[at] = 0;
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
        const uint32_t p = p16_prime(hvx_softmax_q_exp2_q14(f, F), e);
        p_lo[at] = (uint8_t)(p & 0xffu);
        p_hi[at] = (uint8_t)(p >> 8);
        sum[r] += p;
      }
    }
  }
  /* Pass 3: P16 = round(P' * R / 65536), in place. */
  for (uint32_t r = 0; r < HVX_SOFTMAX_Q_ROWS; ++r) {
    const uint32_t R = p16_recip(sum[r]);
    for (uint32_t c = 0; c < ct; ++c) {
      for (uint32_t j = 0; j < 32u; ++j) {
        const uint32_t at = c * HVX_SOFTMAX_Q_TILE + r * 32u + j;
        const uint32_t p = (uint32_t)p_lo[at] | ((uint32_t)p_hi[at] << 8);
        uint32_t q = (uint32_t)(((uint64_t)p * R + 32768u) >> 16);
        if (q > 65535u) {
          q = 65535u;
        }
        p_lo[at] = (uint8_t)(q & 0xffu);
        p_hi[at] = (uint8_t)(q >> 8);
      }
    }
    if (rowsum) {
      rowsum[hvx_softmax_q_rowsum_index(r)] = (int32_t)sum[r];
    }
  }
}

#if defined(__hexagon__)

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

/** Column index of each lane in a 64-lane int16 vector of a tile: two rows
 * of 32 columns. Loaded once per call from cached memory. */
static const int16_t lane_col[64] __attribute__((aligned(128))) = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

enum {
  TILE_FULL = HVX_SOFTMAX_Q_TILE_FULL,
  TILE_PARTIAL = HVX_SOFTMAX_Q_TILE_PARTIAL,
  TILE_MASKED = HVX_SOFTMAX_Q_TILE_MASKED
};

/** Whether every lane of tile c is visible to every row of the block, no
 * lane is, or it depends on the lane. Rows past n_rows make the whole
 * block partial, which only the last block of a sequence pays. */
int hvx_softmax_q_tile_class(const hvx_softmax_q_block *b, uint32_t c) {
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

/** t (<= 0, int16) -> u = t * rho, its integer part e (<= 0, in *e_out) and
 * T = 2^(frac(u)) as Q14 in [16384, 32767]. */
static inline HVX_Vector exp2_T(HVX_Vector t, int rho_pair, int F,
                                HVX_Vector fmask, HVX_Vector vA, HVX_Vector vB,
                                HVX_Vector v16384, HVX_Vector *e_out) {
  const HVX_Vector u = Q6_Vh_vmpy_VhRh_s1_rnd_sat(t, rho_pair);
  *e_out = Q6_Vh_vasr_VhR(u, F);
  const HVX_Vector x = Q6_Vh_vasl_VhR(Q6_V_vand_VV(u, fmask), 15 - F);
  HVX_Vector p = Q6_Vh_vmpy_VhRh_s1_rnd_sat(x, (HVX_SOFTMAX_Q_EXP2_C << 16) |
                                                 HVX_SOFTMAX_Q_EXP2_C);
  p = Q6_Vh_vadd_VhVh_sat(p, vB);
  p = Q6_Vh_vmpy_VhVh_s1_rnd_sat(p, x);
  p = Q6_Vh_vadd_VhVh_sat(p, vA);
  const HVX_Vector m = Q6_Vh_vmpy_VhVh_s1_rnd_sat(p, x);
  return Q6_Vh_vadd_VhVh(v16384, Q6_Vh_vasr_VhR(m, 1));
}

/** The vector p16_prime: 65535 at e = 0, 2T at e = -1, and for e <= -2
 * ((T >> (-e - 3)) + 1) >> 1. vasr(Vh, Vh) shifts left for the negative
 * amount e = -2 gives, so q = 2T there (up to 65534, past int16) and the
 * final halve must be a logical shift; e = -2 then needs no branch. */
static inline HVX_Vector p16_prime_v(HVX_Vector T, HVX_Vector e, HVX_Vector vm3,
                                     HVX_Vector vm1, HVX_Vector v15,
                                     HVX_Vector one, HVX_Vector zero) {
  const HVX_Vector sh = Q6_Vh_vmin_VhVh(Q6_Vh_vsub_VhVh(vm3, e), v15);
  const HVX_Vector q = Q6_Vh_vasr_VhVh(T, sh);
  HVX_Vector p = Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(q, one), 1);
  p = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(e, vm1), Q6_Vh_vadd_VhVh(T, T), p);
  return Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(e, zero), Q6_V_vnot_V(zero), p);
}

static void pass1_rowmax(const hvx_softmax_q_block *b, const HVX_Vector *S,
                         const HVX_Vector *CV, const uint8_t *cls, uint32_t c0,
                         uint32_t n, HVX_Vector *M, HVX_VectorPred qlow,
                         HVX_Vector vmin, HVX_Vector rowbase,
                         HVX_Vector colbase);

/** A vrmpy row-sum accumulator (row k of the 4-row group in words
 * 8k..8k+7) -> lane 8k holds row k's sum. */
static inline HVX_Vector rowsum_reduce(HVX_Vector r) {
  r = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 4));
  r = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 8));
  r = Q6_Vw_vadd_VwVw(r, Q6_V_vror_VR(r, 16));
  return r;
}

/* Scratch layout of the 16-bit softmax, in vectors. */
enum {
  SMX_M = 0,    /* [32] row maxima, broadcast per row                  */
  SMX_RSL = 32, /* [16] vrmpy accumulators of the low bytes            */
  SMX_RSH = 48, /* [16] of the high bytes                              */
  SMX_R = 64,   /* [16] per-row reciprocals in the row-sum layout      */
  SMX_CV = 80   /* [n]  staged corrections of the part's tiles         */
};

/** The vector recip_nr: every lane of @a x (as int32 row sums; lanes
 * other than 8k hold partial sums and are ignored downstream). */
static inline HVX_Vector recip_nr_v(HVX_Vector x_i32) {
  const HVX_Vector two = hvx_splat_sf(2.0f);
  const HVX_Vector x = Q6_Vsf_equals_Vw(x_i32);
  HVX_Vector r = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R((int)0x7EF311C7), x);
  HVX_Vector xr = Q6_Vsf_vmpy_VsfVsf(x, r);
  r = Q6_Vsf_vmpy_VsfVsf(r, Q6_Vsf_vsub_VsfVsf(two, xr));
  xr = Q6_Vsf_vmpy_VsfVsf(x, r);
  r = Q6_Vsf_vmpy_VsfVsf(r, Q6_Vsf_vsub_VsfVsf(two, xr));
  xr = Q6_Vsf_vmpy_VsfVsf(x, r);
  r = Q6_Vsf_vmpy_VsfVsf(r, Q6_Vsf_vsub_VsfVsf(two, xr));
  return r;
}

/** The vector p16_recip over a row-sum vector. */
static inline HVX_Vector p16_recip_v(HVX_Vector rowsum) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_Vector q =
    Q6_Vsf_vmpy_VsfVsf(hvx_splat_sf(4294901760.0f), recip_nr_v(rowsum));
  HVX_Vector R = Q6_Vw_vmin_VwVw(hvx_sf_to_w_rne(q), Q6_V_vsplat_R(65535));
  return Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(rowsum, zero), zero, R);
}

static inline void part_classes(const hvx_softmax_q_block *b, uint32_t c0,
                                uint32_t c1, uint8_t *cls) {
  for (uint32_t c = c0; c < c1; ++c) {
    cls[c - c0] = (uint8_t)hvx_softmax_q_tile_class(b, c);
  }
}

void hvx_softmax_q16_part_max(const hvx_softmax_q_block *b,
                              const int16_t *s_tiles, const int16_t *corr,
                              uint32_t c0, uint32_t c1, void *scratch) {
  HVX_Vector *M = (HVX_Vector *)scratch + SMX_M;
  HVX_Vector *CV = (HVX_Vector *)scratch + SMX_CV;
  const HVX_Vector *S = (const HVX_Vector *)s_tiles;
  const HVX_VectorPred qlow = Q6_Q_vsetq_R(64);
  const HVX_Vector vmin = Q6_Vh_vsplat_R(-32768);
  const HVX_Vector colbase = Q6_Vh_vadd_VhVh(*(const HVX_Vector *)lane_col,
                                             Q6_Vh_vsplat_R((int)b->col0));
  const HVX_Vector rowbase = Q6_V_vmux_QVV(qlow, Q6_Vh_vsplat_R((int)b->row0),
                                           Q6_Vh_vsplat_R((int)b->row0 + 1));
  uint8_t cls[HVX_SOFTMAX_Q_MAX_TILES];
  part_classes(b, c0, c1, cls);
  for (uint32_t c = c0; c < c1; ++c) {
    CV[c - c0] = corr ? corr_vec(corr, c, qlow) : Q6_V_vzero();
  }
  pass1_rowmax(b, S, CV, cls, c0, c1 - c0, M, qlow, vmin, rowbase, colbase);
}

void hvx_softmax_q16_merge_max(void *scratch, void *const *parts,
                               uint32_t n_parts) {
  HVX_Vector *M = (HVX_Vector *)scratch + SMX_M;
  /** Reading a part another thread has already merged is harmless: the
   * merged maximum is a fixed point of the merge. */
  for (uint32_t v = 0; v < 32u; ++v) {
    HVX_Vector m = M[v];
    for (uint32_t p = 0; p < n_parts; ++p) {
      m = Q6_Vh_vmax_VhVh(m, ((const HVX_Vector *)parts[p])[SMX_M + v]);
    }
    M[v] = m;
  }
}

void hvx_softmax_q16_row_max(const void *scratch,
                             int16_t max[HVX_SOFTMAX_Q_ROWS]) {
  /* M[v]: row 2v's maximum in lanes 0..31, row 2v + 1's in lanes 32..63 */
  const HVX_Vector *M = (const HVX_Vector *)scratch + SMX_M;
  for (uint32_t v = 0; v < 32u; ++v) {
    const int16_t *m = (const int16_t *)&M[v];
    max[2u * v] = m[0];
    max[2u * v + 1u] = m[32];
  }
}

void hvx_softmax_q16_part_exp(const hvx_softmax_q_block *b,
                              const int16_t *s_tiles, uint32_t c0, uint32_t c1,
                              uint8_t *p_lo, uint8_t *p_hi, void *scratch) {
  const int F = b->frac_bits;
  const int rho_pair = ((int)b->rho_q15 << 16) | (int)b->rho_q15;
  const HVX_Vector *M = (const HVX_Vector *)scratch + SMX_M;
  HVX_Vector *RSL = (HVX_Vector *)scratch + SMX_RSL;
  HVX_Vector *RSH = (HVX_Vector *)scratch + SMX_RSH;
  const HVX_Vector *CV = (const HVX_Vector *)scratch + SMX_CV;
  const HVX_Vector *S = (const HVX_Vector *)s_tiles; /* [ct][32] */
  HVX_Vector *PL = (HVX_Vector *)p_lo;               /* [ct][16] */
  HVX_Vector *PH = (HVX_Vector *)p_hi;               /* [ct][16] */
  const uint32_t n = c1 - c0;

  const HVX_VectorPred qlow = Q6_Q_vsetq_R(64);
  const HVX_Vector vmin = Q6_Vh_vsplat_R(-32768);
  const HVX_Vector colbase = Q6_Vh_vadd_VhVh(*(const HVX_Vector *)lane_col,
                                             Q6_Vh_vsplat_R((int)b->col0));
  const HVX_Vector rowbase = Q6_V_vmux_QVV(qlow, Q6_Vh_vsplat_R((int)b->row0),
                                           Q6_Vh_vsplat_R((int)b->row0 + 1));
  uint8_t cls[HVX_SOFTMAX_Q_MAX_TILES];
  part_classes(b, c0, c1, cls);

  const HVX_Vector fmask = Q6_Vh_vsplat_R((1 << F) - 1);
  const HVX_Vector vA = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_A);
  const HVX_Vector vB = Q6_Vh_vsplat_R(HVX_SOFTMAX_Q_EXP2_B);
  const HVX_Vector v16384 = Q6_Vh_vsplat_R(16384);
  const HVX_Vector vm3 = Q6_Vh_vsplat_R(-3);
  const HVX_Vector vm1 = Q6_Vh_vsplat_R(-1);
  const HVX_Vector v15 = Q6_Vh_vsplat_R(15);
  const HVX_Vector one = Q6_Vh_vsplat_R(1);
  const HVX_Vector zero = Q6_V_vzero();
  /* One full tile for the 4-row group w: t -> P' -> low / high byte tiles. */
#define SMX16_TILE_FULL(Sc, cv, out_lo, out_hi)                                \
  do {                                                                         \
    const HVX_Vector t0_ =                                                     \
      Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh((Sc)[v0], (cv)), m0);                \
    const HVX_Vector t1_ =                                                     \
      Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh((Sc)[v0 + 1u], (cv)), m1);           \
    HVX_Vector e0_, e1_;                                                       \
    const HVX_Vector T0_ =                                                     \
      exp2_T(t0_, rho_pair, F, fmask, vA, vB, v16384, &e0_);                   \
    const HVX_Vector T1_ =                                                     \
      exp2_T(t1_, rho_pair, F, fmask, vA, vB, v16384, &e1_);                   \
    const HVX_Vector p0_ = p16_prime_v(T0_, e0_, vm3, vm1, v15, one, zero);    \
    const HVX_Vector p1_ = p16_prime_v(T1_, e1_, vm3, vm1, v15, one, zero);    \
    (out_lo) = Q6_Vb_vpacke_VhVh(p1_, p0_);                                    \
    (out_hi) = Q6_Vb_vpacko_VhVh(p1_, p0_);                                    \
  } while (0)

  for (uint32_t w = 0; w < 16u; ++w) {
    const uint32_t v0 = 2u * w;
    const HVX_Vector m0 = M[v0], m1 = M[v0 + 1u];
    const HVX_Vector rowv0 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v0));
    const HVX_Vector rowv1 =
      Q6_Vh_vadd_VhVh(rowbase, Q6_Vh_vsplat_R(2 * (int)v0 + 2));
    HVX_Vector rsl = zero, rsh = zero;
    uint32_t c = 0;
    while (c < n) {
      const uint32_t cc = c0 + c;
      /* Two full tiles: four independent exp2 chains in flight. */
      if (c + 2u <= n && cls[c] == TILE_FULL && cls[c + 1u] == TILE_FULL) {
        const HVX_Vector *Sa = S + 32u * cc;
        HVX_Vector la, ha, lb, hb;
        SMX16_TILE_FULL(Sa, CV[c], la, ha);
        SMX16_TILE_FULL(Sa + 32u, CV[c + 1u], lb, hb);
        PL[16u * cc + w] = la;
        PH[16u * cc + w] = ha;
        PL[16u * (cc + 1u) + w] = lb;
        PH[16u * (cc + 1u) + w] = hb;
        rsl = Q6_Vuw_vrmpyacc_VuwVubRub(rsl, la, 0x01010101u);
        rsh = Q6_Vuw_vrmpyacc_VuwVubRub(rsh, ha, 0x01010101u);
        rsl = Q6_Vuw_vrmpyacc_VuwVubRub(rsl, lb, 0x01010101u);
        rsh = Q6_Vuw_vrmpyacc_VuwVubRub(rsh, hb, 0x01010101u);
        c += 2u;
        continue;
      }
      if (cls[c] == TILE_MASKED) {
        PL[16u * cc + w] = zero;
        PH[16u * cc + w] = zero;
        ++c;
        continue;
      }
      /* One tile, full or partial. */
      const HVX_Vector *Sc = S + 32u * cc;
      const HVX_Vector cv = CV[c];
      HVX_Vector t0 = Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(Sc[v0], cv), m0);
      HVX_Vector t1 = Q6_Vh_vsub_VhVh_sat(Q6_Vh_vsub_VhVh(Sc[v0 + 1u], cv), m1);
      if (cls[c] == TILE_PARTIAL) {
        const HVX_Vector colv =
          Q6_Vh_vadd_VhVh(colbase, Q6_Vh_vsplat_R(32 * (int)cc));
        t0 = Q6_V_vmux_QVV(mask_q(b, rowv0, colv), vmin, t0);
        t1 = Q6_V_vmux_QVV(mask_q(b, rowv1, colv), vmin, t1);
      }
      HVX_Vector e0, e1;
      const HVX_Vector T0 = exp2_T(t0, rho_pair, F, fmask, vA, vB, v16384, &e0);
      const HVX_Vector T1 = exp2_T(t1, rho_pair, F, fmask, vA, vB, v16384, &e1);
      const HVX_Vector p0 = p16_prime_v(T0, e0, vm3, vm1, v15, one, zero);
      const HVX_Vector p1 = p16_prime_v(T1, e1, vm3, vm1, v15, one, zero);
      const HVX_Vector lo = Q6_Vb_vpacke_VhVh(p1, p0);
      const HVX_Vector hi = Q6_Vb_vpacko_VhVh(p1, p0);
      PL[16u * cc + w] = lo;
      PH[16u * cc + w] = hi;
      rsl = Q6_Vuw_vrmpyacc_VuwVubRub(rsl, lo, 0x01010101u);
      rsh = Q6_Vuw_vrmpyacc_VuwVubRub(rsh, hi, 0x01010101u);
      ++c;
    }
    RSL[w] = rsl;
    RSH[w] = rsh;
  }
#undef SMX16_TILE_FULL
}

void hvx_softmax_q16_merge_sum(void *scratch, void *const *parts,
                               uint32_t n_parts, int32_t *rowsum) {
  /** Row sums = low + 256 * high over all parts, lane 8k <- row 4w + k;
   * the totals go over this scratch's dead row maxima. */
  HVX_Vector *T = (HVX_Vector *)scratch + SMX_M;
  for (uint32_t w = 0; w < 16u; ++w) {
    HVX_Vector l = Q6_V_vzero(), h = Q6_V_vzero();
    for (uint32_t p = 0; p < n_parts; ++p) {
      const HVX_Vector *P = (const HVX_Vector *)parts[p];
      l = Q6_Vw_vadd_VwVw(l, P[SMX_RSL + w]);
      h = Q6_Vw_vadd_VwVw(h, P[SMX_RSH + w]);
    }
    T[w] = rowsum_reduce(Q6_Vw_vadd_VwVw(l, Q6_Vw_vasl_VwR(h, 8)));
  }
  if (rowsum) {
    HVX_Vector *out = (HVX_Vector *)rowsum;
    for (uint32_t w = 0; w < 16u; ++w) {
      out[w] = T[w];
    }
  }
  HVX_Vector *R = (HVX_Vector *)scratch + SMX_R;
  for (uint32_t w = 0; w < 16u; ++w) {
    R[w] = p16_recip_v(T[w]);
  }
}

void hvx_softmax_q16_part_norm(const hvx_softmax_q_block *b, uint32_t c0,
                               uint32_t c1, uint8_t *p_lo, uint8_t *p_hi,
                               void *scratch) {
  const uint32_t *R = (const uint32_t *)((HVX_Vector *)scratch + SMX_R);
  HVX_Vector *PL = (HVX_Vector *)p_lo;
  HVX_Vector *PH = (HVX_Vector *)p_hi;
  const HVX_VectorPred qlow = Q6_Q_vsetq_R(64);
  uint8_t cls[HVX_SOFTMAX_Q_MAX_TILES];
  part_classes(b, c0, c1, cls);
  /* P16 = round(P' * R / 65536) in place, R per row (row-sum layout). */
  for (uint32_t w = 0; w < 16u; ++w) {
    const uint32_t R0 = R[hvx_softmax_q_rowsum_index(4u * w)];
    const uint32_t R1 = R[hvx_softmax_q_rowsum_index(4u * w + 1u)];
    const uint32_t R2 = R[hvx_softmax_q_rowsum_index(4u * w + 2u)];
    const uint32_t R3 = R[hvx_softmax_q_rowsum_index(4u * w + 3u)];
    const HVX_Vector M0 =
      Q6_V_vmux_QVV(qlow, Q6_Vh_vsplat_R((int)R0), Q6_Vh_vsplat_R((int)R1));
    const HVX_Vector M1 =
      Q6_V_vmux_QVV(qlow, Q6_Vh_vsplat_R((int)R2), Q6_Vh_vsplat_R((int)R3));
    for (uint32_t c = c0; c < c1; ++c) {
      if (cls[c - c0] == TILE_MASKED) {
        continue;
      }
      const HVX_Vector lo = PL[16u * c + w];
      const HVX_Vector hi = PH[16u * c + w];
      /* Byte interleave: (hi, lo) -> little-endian u16, rows 4w.. in order. */
      const HVX_VectorPair x = Q6_W_vshuff_VVR(hi, lo, -1);
      const HVX_VectorPair y0 = Q6_Wuw_vmpy_VuhVuh(Q6_V_lo_W(x), M0);
      const HVX_VectorPair y1 = Q6_Wuw_vmpy_VuhVuh(Q6_V_hi_W(x), M1);
      /** Even products in v0, odd in v1; the shift-pack interleaves them
       * back. Its shift amount is 4 bits, so (y + 2^15) >> 16 is done as
       * ((y >> 1) + 2^14) >> 15, which is the same floor. */
      const HVX_Vector q0 =
        Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(y0), 1),
                                    Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(y0), 1), 15);
      const HVX_Vector q1 =
        Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(y1), 1),
                                    Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(y1), 1), 15);
      PL[16u * c + w] = Q6_Vb_vpacke_VhVh(q1, q0);
      PH[16u * c + w] = Q6_Vb_vpacko_VhVh(q1, q0);
    }
  }
}

/**
 * @brief hvx_softmax_q16: the five phases with a single part.
 */
void hvx_softmax_q16(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                     const int16_t *corr, uint8_t *p_lo, uint8_t *p_hi,
                     int32_t *rowsum, void *scratch) {
  void *const parts[1] = {scratch};
  const uint32_t ct = b->n_col_tiles;
  hvx_softmax_q16_part_max(b, s_tiles, corr, 0, ct, scratch);
  hvx_softmax_q16_merge_max(scratch, parts, 1);
  hvx_softmax_q16_part_exp(b, s_tiles, 0, ct, p_lo, p_hi, scratch);
  hvx_softmax_q16_merge_sum(scratch, parts, 1, rowsum);
  hvx_softmax_q16_part_norm(b, 0, ct, p_lo, p_hi, scratch);
}

/** Pass 1 of both softmaxes: row max of the corrected, masked scores into
 * M[32] (row-pair vectors, each row's max broadcast over its 32 lanes),
 * four row-pair vectors per sweep over the tiles so the branches and the
 * 4 KiB stride are paid once per four vectors. */
static void pass1_rowmax(const hvx_softmax_q_block *b, const HVX_Vector *S,
                         const HVX_Vector *CV, const uint8_t *cls, uint32_t c0,
                         uint32_t n, HVX_Vector *M, HVX_VectorPred qlow,
                         HVX_Vector vmin, HVX_Vector rowbase,
                         HVX_Vector colbase) {
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
    for (uint32_t c = 0; c < n; ++c) {
      if (cls[c] == TILE_MASKED) {
        continue;
      }
      const HVX_Vector *Sc = S + 32u * (c0 + c) + v;
      const HVX_Vector cv = CV[c];
      HVX_Vector s0 = Q6_Vh_vsub_VhVh(Sc[0], cv);
      HVX_Vector s1 = Q6_Vh_vsub_VhVh(Sc[1], cv);
      HVX_Vector s2 = Q6_Vh_vsub_VhVh(Sc[2], cv);
      HVX_Vector s3 = Q6_Vh_vsub_VhVh(Sc[3], cv);
      if (cls[c] == TILE_PARTIAL) {
        const HVX_Vector colv =
          Q6_Vh_vadd_VhVh(colbase, Q6_Vh_vsplat_R(32 * (int)(c0 + c)));
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
}

#endif /* __hexagon__ */
