// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd.
 * @file hexkl_integer_hvx.h
 * @brief Exact widened integer HVX operations for FC and requantization
 * @bug No known bugs except for NYI items
 * Included by hexkl_integer.h after its public parameter types.
 */
#ifndef __NNTRAINER_HEXKL_INTEGER_HVX_H__
#define __NNTRAINER_HEXKL_INTEGER_HVX_H__
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

typedef struct {
  HVX_Vector lo, hi;
} hexkl_wide;

static inline HVX_Vector hexkl_carry(HVX_Vector a, HVX_Vector sum) {
  return Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(a, sum), 1);
}
static inline hexkl_wide hexkl_wide_add(hexkl_wide a, hexkl_wide b) {
  hexkl_wide r;
  r.lo = Q6_Vw_vadd_VwVw(a.lo, b.lo);
  r.hi = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a.hi, b.hi), hexkl_carry(a.lo, r.lo));
  return r;
}
static inline hexkl_wide hexkl_wide_sub(hexkl_wide a, hexkl_wide b) {
  hexkl_wide r;
  r.lo = Q6_Vw_vsub_VwVw(a.lo, b.lo);
  r.hi = Q6_Vw_vsub_VwVw(Q6_Vw_vsub_VwVw(a.hi, b.hi),
                         Q6_V_vand_QR(Q6_Q_vcmp_gt_VuwVuw(b.lo, a.lo), 1));
  return r;
}
static inline hexkl_wide hexkl_extend(HVX_Vector a) {
  hexkl_wide r = {a, Q6_Vw_vasr_VwR(a, 31)};
  return r;
}
/** Signed word times unsigned halfword, exact low/high words. */
static inline hexkl_wide hexkl_mul_u16(HVX_Vector a, uint16_t z) {
  const HVX_Vector vz = Q6_V_vsplat_R(z), mask = Q6_V_vsplat_R(65535);
  HVX_Vector p0 = Q6_Vw_vmpyie_VwVuh(Q6_V_vand_VV(a, mask), vz);
  HVX_Vector p1 = Q6_Vw_vmpyie_VwVuh(Q6_Vw_vasr_VwR(a, 16), vz);
  hexkl_wide x = {p0, Q6_V_vzero()};
  hexkl_wide y = {Q6_Vw_vasl_VwR(p1, 16), Q6_Vw_vasr_VwR(p1, 16)};
  return hexkl_wide_add(x, y);
}
static inline int hexkl_any_word(HVX_Vector v) {
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 64));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 32));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 16));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 8));
  v = Q6_V_vor_VV(v, Q6_V_vror_VR(v, 4));
  return Q6_R_vextract_VR(v, 0) != 0;
}
/** Four rows per aligned byte-plane load, as the existing FP32 epilogue. */
static inline void hexkl_integer_rows4(const uint8_t *p, uint32_t group,
                                       uint32_t stride, uint32_t planes,
                                       HVX_Vector out[4]) {
  if (!planes) {
    for (uint32_t r = 0; r < 4; ++r)
      out[r] =
        *(const HVX_UVector *)(p + (size_t)(4u * group + r) * stride * 4u);
    return;
  }
  p += (size_t)group * 128u;
  HVX_VectorPair h01 = Q6_W_vshuff_VVR(*(const HVX_Vector *)(p + 2048),
                                       *(const HVX_Vector *)p, -1);
  HVX_VectorPair h2 = planes == 3u
                        ? Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(p + 4096))
                        : Q6_W_vshuff_VVR(*(const HVX_Vector *)(p + 6144),
                                          *(const HVX_Vector *)(p + 4096), -1);
  HVX_VectorPair w01 = Q6_W_vshuff_VVR(Q6_V_lo_W(h2), Q6_V_lo_W(h01), -2);
  HVX_VectorPair w23 = Q6_W_vshuff_VVR(Q6_V_hi_W(h2), Q6_V_hi_W(h01), -2);
  out[0] = Q6_V_lo_W(w01);
  out[1] = Q6_V_hi_W(w01);
  out[2] = Q6_V_lo_W(w23);
  out[3] = Q6_V_hi_W(w23);
}
static inline HVX_Vector hexkl_requant_vector(HVX_Vector a, HVX_Vector mult,
                                              HVX_Vector shift, HVX_Vector zp);

/** @brief Exact saturation shortcut for a uniform shift <= 16.
 * If product+round exceeds UINT32_MAX then its shifted magnitude exceeds
 * UINT16_MAX, so it must saturate. Only the non-overflowing lanes need the
 * low product word. No accumulator range assumption is made. */
static inline HVX_Vector
hexkl_requant_small_shift(HVX_Vector x, HVX_Vector mult, HVX_Vector zp,
                          uint32_t shift, uint32_t multiplier, uint32_t safe) {
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(zero, x);
  const HVX_Vector mag = Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(zero, x), x);
  HVX_Vector prod = Q6_Vw_vmpyie_VwVuh(mag, mult);
  if (multiplier >> 16)
    prod = Q6_Vw_vadd_VwVw(
      prod, Q6_Vw_vasl_VwR(
              Q6_Vw_vmpyie_VwVuh(mag, Q6_V_vsplat_R(multiplier >> 16)), 16));
  if (shift)
    prod = Q6_Vuw_vlsr_VuwR(
      Q6_Vw_vadd_VwVw(prod, Q6_V_vsplat_R(1u << (shift - 1u))), shift);
  const HVX_Vector bound =
    Q6_V_vmux_QVV(neg, zp, Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(65535), zp));
  const HVX_VectorPred clip =
    Q6_Q_or_QQ(Q6_Q_vcmp_gt_VuwVuw(mag, Q6_V_vsplat_R(safe)),
               Q6_Q_vcmp_gt_VuwVuw(prod, bound));
  const HVX_Vector value = Q6_V_vmux_QVV(clip, bound, prod);
  return Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(zp, value),
                       Q6_Vw_vadd_VwVw(zp, value));
}

/** @brief Exactly 32 output codes; predication does not read adjacent bytes. */
static inline void hexkl_store_u16x32(uint16_t *dst, HVX_Vector codes) {
  const HVX_Vector packed = Q6_Vuh_vpack_VwVw_sat(codes, codes);
  if (((uintptr_t)dst & 63u) == 0u) {
    HVX_VectorPred mask = Q6_Q_vsetq_R(64);
    if ((uintptr_t)dst & 64u)
      mask = Q6_Q_not_Q(mask);
    Q6_vmem_QRIV_nt(mask, (HVX_Vector *)((uintptr_t)dst & ~(uintptr_t)127u),
                    packed);
  } else {
    memcpy(dst, &packed, 64u);
  }
}

static inline void hexkl_requant_params32(const hexkl_requant_u16 *p,
                                          HVX_Vector *vm, HVX_Vector *vs,
                                          HVX_Vector *vz) {
  int32_t m[32] __attribute__((aligned(128)));
  int32_t s[32] __attribute__((aligned(128)));
  int32_t z[32] __attribute__((aligned(128)));
  for (uint32_t i = 0; i < 32u; ++i) {
    m[i] = p[i].multiplier;
    s[i] = p[i].shift;
    z[i] = p[i].zero_point;
  }
  *vm = *(HVX_Vector *)m;
  *vs = *(HVX_Vector *)s;
  *vz = *(HVX_Vector *)z;
}

static inline void hexkl_store_i32x32(int32_t *dst, HVX_Vector value) {
  if (((uintptr_t)dst & 127u) == 0u)
    Q6_vmem_QnRIV_nt(Q6_Q_vcmp_gt_VwVw(value, value), (HVX_Vector *)dst, value);
  else
    *(HVX_UVector *)dst = value;
}

/** @brief A proven INT32 dot, plus an arbitrary widened correction. */
static inline HVX_Vector hexkl_correct_i32_dot(HVX_Vector hi, HVX_Vector lo,
                                               hexkl_wide correction,
                                               int narrow, HVX_Vector *bad) {
  const HVX_Vector dot = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(hi, 8), lo);
  HVX_Vector value;
  if (narrow) {
    value = Q6_Vw_vadd_VwVw(dot, correction.lo);
    *bad = Q6_V_vor_VV(
      *bad, Q6_V_vxor_VV(value, Q6_Vw_vadd_VwVw_sat(dot, correction.lo)));
  } else {
    const hexkl_wide x = hexkl_wide_add(hexkl_extend(dot), correction);
    value = x.lo;
    *bad = Q6_V_vor_VV(*bad, Q6_V_vxor_VV(x.hi, Q6_Vw_vasr_VwR(value, 31)));
  }
  return value;
}

/** @brief Pair neighboring tiles before moving to the next four rows.
 * Keeping the two requantized vectors together removes the full-row-band
 * temporary previously needed to coalesce half-line UINT16 stores. */
static inline uint32_t hexkl_i32_requant_pairs(hexkl_i32_rows_job *j,
                                               uint32_t begin, uint32_t end,
                                               uint32_t safe) {
  const uint32_t shift = j->requant[0].shift,
                 multiplier = j->requant[0].multiplier;
  const HVX_Vector vm = Q6_V_vsplat_R(multiplier);
  const HVX_Vector vz = Q6_V_vsplat_R(j->requant[0].zero_point);
  uint32_t t = 0;
  for (; t + 1u < j->n_tiles; t += 2u) {
    const uint32_t c = (j->nt0 + t) * 32u;
    hexkl_wide correction[2];
    int narrow[2];
    for (uint32_t h = 0; h < 2u; ++h) {
      const uint32_t col = c + 32u * h;
      const HVX_Vector b =
        j->bias ? *(const HVX_UVector *)(j->bias + col) : Q6_V_vzero();
      correction[h] = hexkl_wide_sub(
        hexkl_extend(b),
        hexkl_mul_u16(*(const HVX_UVector *)(j->colsum + col), j->zero_point));
      narrow[h] = !hexkl_any_word(
        Q6_V_vxor_VV(correction[h].hi, Q6_Vw_vasr_VwR(correction[h].lo, 31)));
    }
    const uint8_t *p = j->tiles + (size_t)t * j->tile_stride;
    for (uint32_t g = begin; g < end; ++g) {
      HVX_Vector hi0[4], lo0[4], hi1[4], lo1[4], bad = Q6_V_vzero();
      hexkl_integer_rows4(p, g, j->row_stride, j->plane_count, hi0);
      hexkl_integer_rows4(p + j->lo_off, g, j->row_stride, j->plane_count, lo0);
      hexkl_integer_rows4(p + j->tile_stride, g, j->row_stride, j->plane_count,
                          hi1);
      hexkl_integer_rows4(p + j->tile_stride + j->lo_off, g, j->row_stride,
                          j->plane_count, lo1);
#pragma clang loop unroll(full)
      for (uint32_t r = 0; r < 4u; ++r) {
        const uint32_t row = 4u * g + r;
        if (row >= j->rows)
          break;
        const HVX_Vector v0 =
          hexkl_correct_i32_dot(hi0[r], lo0[r], correction[0], narrow[0], &bad);
        const HVX_Vector v1 =
          hexkl_correct_i32_dot(hi1[r], lo1[r], correction[1], narrow[1], &bad);
        hexkl_store_i32x32(j->out + (size_t)row * j->cols + c, v0);
        hexkl_store_i32x32(j->out + (size_t)row * j->cols + c + 32u, v1);
        const HVX_Vector q0 =
          hexkl_requant_small_shift(v0, vm, vz, shift, multiplier, safe);
        const HVX_Vector q1 =
          hexkl_requant_small_shift(v1, vm, vz, shift, multiplier, safe);
        const HVX_Vector packed = Q6_Vuh_vpack_VwVw_sat(q1, q0);
        uint16_t *dst = j->out_u16 + (size_t)row * j->cols + c;
        if (((uintptr_t)dst & 127u) == 0u)
          Q6_vmem_QnRIV_nt(Q6_Q_vcmp_gt_VwVw(packed, packed), (HVX_Vector *)dst,
                           packed);
        else
          *(HVX_UVector *)dst = packed;
      }
      if (hexkl_any_word(bad))
        j->overflow[4u * g] = 1;
    }
  }
  return t;
}

/** @brief Four corrected rows go straight to the next FC's byte planes. */
static inline void hexkl_i32_rows_ah(hexkl_i32_rows_job *j, uint32_t begin,
                                     uint32_t end) {
  const int fast = j->n_requant == 1u && j->requant[0].shift <= 16u;
  const uint32_t shift = j->requant[0].shift;
  const uint32_t mult = j->requant[0].multiplier;
  const uint32_t round = fast && shift ? 1u << (shift - 1u) : 0u;
  const uint32_t safe = mult ? (UINT32_MAX - round) / mult : UINT32_MAX;
  for (uint32_t t = 0; t < j->n_tiles; ++t) {
    const uint32_t c = (j->nt0 + t) * 32u;
    HVX_Vector vm, vs, vz;
    if (j->n_requant == 1u) {
      vm = Q6_V_vsplat_R(mult);
      vs = Q6_V_vsplat_R(shift);
      vz = Q6_V_vsplat_R(j->requant[0].zero_point);
    } else {
      hexkl_requant_params32(j->requant + c, &vm, &vs, &vz);
    }
    const HVX_Vector b =
      j->bias ? *(const HVX_UVector *)(j->bias + c) : Q6_V_vzero();
    const hexkl_wide correction = hexkl_wide_sub(
      hexkl_extend(b),
      hexkl_mul_u16(*(const HVX_UVector *)(j->colsum + c), j->zero_point));
    const int narrow = !hexkl_any_word(
      Q6_V_vxor_VV(correction.hi, Q6_Vw_vasr_VwR(correction.lo, 31)));
    const uint8_t *p = j->tiles + (size_t)t * j->tile_stride;
    for (uint32_t g = begin; g < end; ++g) {
      HVX_Vector hi[4], lo[4], q[4], bad = Q6_V_vzero();
      hexkl_integer_rows4(p, g, j->row_stride, j->plane_count, hi);
      hexkl_integer_rows4(p + j->lo_off, g, j->row_stride, j->plane_count, lo);
#pragma clang loop unroll(full)
      for (uint32_t r = 0; r < 4u; ++r) {
        const uint32_t row = 4u * g + r;
        q[r] = Q6_V_vzero();
        if (row >= j->rows)
          continue;
        HVX_Vector value;
        if (j->sum_fits_i32) {
          value = hexkl_correct_i32_dot(hi[r], lo[r], correction, narrow, &bad);
        } else {
          const hexkl_wide high = {Q6_Vw_vasl_VwR(hi[r], 8),
                                   Q6_Vw_vasr_VwR(hi[r], 24)};
          const hexkl_wide x = hexkl_wide_add(
            hexkl_wide_add(high, hexkl_extend(lo[r])), correction);
          value = x.lo;
          bad = Q6_V_vor_VV(bad, Q6_V_vxor_VV(x.hi, Q6_Vw_vasr_VwR(value, 31)));
        }
        hexkl_store_i32x32(j->out + (size_t)row * j->cols + c, value);
        q[r] = fast
                 ? hexkl_requant_small_shift(value, vm, vz, shift, mult, safe)
                 : hexkl_requant_vector(value, vm, vs, vz);
      }
      const HVX_Vector h01 = Q6_Vuh_vpack_VwVw_sat(q[1], q[0]);
      const HVX_Vector h23 = Q6_Vuh_vpack_VwVw_sat(q[3], q[2]);
      const uint32_t row = j->ah_row + 4u * g, col = j->ah_col + c;
      const size_t d = (size_t)(row / 64u) * 64u * j->ah_cols +
                       (size_t)(col / 32u) * 2048u + (row % 64u) * 32u;
      const HVX_Vector low = Q6_Vb_vpacke_VhVh(h23, h01);
      const HVX_Vector high = Q6_Vb_vpacko_VhVh(h23, h01);
      const HVX_VectorPred none = Q6_Q_vcmp_gt_VwVw(low, low);
      Q6_vmem_QnRIV_nt(none, (HVX_Vector *)(j->ah_lo + d), low);
      Q6_vmem_QnRIV_nt(none, (HVX_Vector *)(j->ah_hi + d), high);
      if (hexkl_any_word(bad))
        j->overflow[4u * g] = 1;
    }
  }
}

static inline void hexkl_i32_rows_hvx(uint32_t threads, uint32_t lane,
                                      void *ctx) {
  hexkl_i32_rows_job *j = (hexkl_i32_rows_job *)ctx;
  HVX_Vector pending[64];
  uint32_t groups = (j->rows + 3u) / 4u;
  uint32_t begin = (uint32_t)((uint64_t)groups * lane / threads);
  uint32_t end = (uint32_t)((uint64_t)groups * (lane + 1u) / threads);
  if (j->ah_lo) {
    hexkl_i32_rows_ah(j, begin, end);
    return;
  }
  const int fast =
    j->out_u16 && j->n_requant == 1u && j->requant[0].shift <= 16u;
  uint32_t safe = 0, fast_shift = 0, fast_mult = 0;
  if (fast) {
    fast_shift = j->requant[0].shift;
    fast_mult = j->requant[0].multiplier;
    const uint32_t round = fast_shift ? 1u << (fast_shift - 1u) : 0u;
    safe = fast_mult ? (UINT32_MAX - round) / fast_mult : UINT32_MAX;
  }
  const uint32_t first_tile =
    fast && j->sum_fits_i32 ? hexkl_i32_requant_pairs(j, begin, end, safe) : 0u;
  for (uint32_t t = first_tile; t < j->n_tiles; ++t) {
    const uint32_t c = (j->nt0 + t) * 32u;
    const uint8_t *p = j->tiles + (size_t)t * j->tile_stride;
    HVX_Vector vm = Q6_V_vzero(), vs = Q6_V_vzero(), vz = Q6_V_vzero();
    if (j->out_u16) {
      if (j->n_requant == 1u) {
        vm = Q6_V_vsplat_R(j->requant[0].multiplier);
        vs = Q6_V_vsplat_R(j->requant[0].shift);
        vz = Q6_V_vsplat_R(j->requant[0].zero_point);
      } else {
        hexkl_requant_params32(j->requant + c, &vm, &vs, &vz);
      }
    }
    const HVX_Vector b =
      j->bias ? *(const HVX_UVector *)(j->bias + c) : Q6_V_vzero();
    const hexkl_wide correction = hexkl_wide_sub(
      hexkl_extend(b),
      hexkl_mul_u16(*(const HVX_UVector *)(j->colsum + c), j->zero_point));
    const int narrow =
      j->sum_fits_i32 && !hexkl_any_word(Q6_V_vxor_VV(
                           correction.hi, Q6_Vw_vasr_VwR(correction.lo, 31)));
    for (uint32_t g = begin; g < end; ++g) {
      HVX_Vector hi[4], lo[4], bad = Q6_V_vzero();
      hexkl_integer_rows4(p, g, j->row_stride, j->plane_count, hi);
      hexkl_integer_rows4(p + j->lo_off, g, j->row_stride, j->plane_count, lo);
      const uint32_t count = j->rows - 4u * g < 4u ? j->rows - 4u * g : 4u;
      for (uint32_t r = 0; r < count; ++r) {
        HVX_Vector value;
        if (narrow) {
          const HVX_Vector dot =
            Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(hi[r], 8), lo[r]);
          value = Q6_Vw_vadd_VwVw(dot, correction.lo);
          // Saturating and wrapping sums differ exactly on signed overflow.
          // This checks arbitrary bias without widened per-output arithmetic.
          bad = Q6_V_vor_VV(
            bad, Q6_V_vxor_VV(value, Q6_Vw_vadd_VwVw_sat(dot, correction.lo)));
        } else {
          hexkl_wide x;
          if (j->sum_fits_i32) {
            x = hexkl_extend(Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(hi[r], 8), lo[r]));
          } else {
            const hexkl_wide high = {Q6_Vw_vasl_VwR(hi[r], 8),
                                     Q6_Vw_vasr_VwR(hi[r], 24)};
            x = hexkl_wide_add(high, hexkl_extend(lo[r]));
          }
          x = hexkl_wide_add(x, correction);
          value = x.lo;
          bad = Q6_V_vor_VV(bad, Q6_V_vxor_VV(x.hi, Q6_Vw_vasr_VwR(value, 31)));
        }
        int32_t *dst32 = j->out + (size_t)(4u * g + r) * j->cols + c;
        if (((uintptr_t)dst32 & 127u) == 0u)
          Q6_vmem_QnRIV_nt(Q6_Q_vcmp_gt_VwVw(value, value), (HVX_Vector *)dst32,
                           value);
        else
          *(HVX_UVector *)dst32 = value;
        if (j->out_u16) {
          HVX_Vector q = fast ? hexkl_requant_small_shift(
                                  value, vm, vz, fast_shift, fast_mult, safe)
                              : hexkl_requant_vector(value, vm, vs, vz);
          const uint32_t row = 4u * g + r;
          if (!(t & 1u) && t + 1u < j->n_tiles) {
            pending[row] = q;
          } else if (t & 1u) {
            uint16_t *dst = j->out_u16 + (size_t)row * j->cols + c - 32u;
            HVX_Vector packed = Q6_Vuh_vpack_VwVw_sat(q, pending[row]);
            if (((uintptr_t)dst & 127u) == 0u)
              Q6_vmem_QnRIV_nt(Q6_Q_vcmp_gt_VwVw(packed, packed),
                               (HVX_Vector *)dst, packed);
            else
              *(HVX_UVector *)dst = packed;
          } else {
            hexkl_store_u16x32(j->out_u16 + (size_t)row * j->cols + c, q);
          }
        }
      }
      // Any error makes the whole FC output invalid. One flag per group
      // avoids vector-to-scalar reductions for each individual row.
      if (hexkl_any_word(bad))
        j->overflow[4u * g] = 1;
    }
  }
}

/** Magnitude (<= 2^31) times multiplier (<= INT32_MAX), exact 64 bits.
 * The sum of the two cross products fits one unsigned word. */
static inline hexkl_wide hexkl_umul(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPair ends = Q6_Wuw_vmpy_VuhVuh(a, b);
  HVX_Vector swapped = Q6_V_vdelta_VV(b, Q6_V_vsplat_R(0x02020202));
  HVX_VectorPair cross = Q6_Wuw_vmpy_VuhVuh(a, swapped);
  HVX_Vector mid = Q6_Vw_vadd_VwVw(Q6_V_lo_W(cross), Q6_V_hi_W(cross));
  hexkl_wide r;
  r.lo = Q6_Vw_vadd_VwVw(Q6_V_lo_W(ends), Q6_Vw_vasl_VwR(mid, 16));
  r.hi = Q6_Vw_vadd_VwVw(Q6_V_hi_W(ends), Q6_Vuw_vlsr_VuwR(mid, 16));
  r.hi = Q6_Vw_vadd_VwVw(r.hi, hexkl_carry(Q6_V_lo_W(ends), r.lo));
  return r;
}
static inline HVX_Vector hexkl_requant_vector(HVX_Vector a, HVX_Vector mult,
                                              HVX_Vector shift, HVX_Vector zp) {
  const HVX_Vector zero = Q6_V_vzero(), one = Q6_V_vsplat_R(1),
                   v31 = Q6_V_vsplat_R(31);
  const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(zero, a);
  const HVX_Vector mag = Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(zero, a), a);
  hexkl_wide prod = hexkl_umul(mag, mult);
  const HVX_VectorPred small = Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(32), shift);
  const HVX_VectorPred shzero = Q6_Q_vcmp_eq_VwVw(shift, zero);
  const HVX_Vector round_shift = Q6_V_vand_VV(Q6_Vw_vsub_VwVw(shift, one), v31);
  const HVX_Vector rounding = Q6_Vw_vasl_VwVw(one, round_shift);
  const HVX_VectorPred roundlo = Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(33), shift);
  hexkl_wide add = {Q6_V_vmux_QVV(roundlo, rounding, zero),
                    Q6_V_vmux_QVV(roundlo, zero, rounding)};
  add.lo = Q6_V_vmux_QVV(shzero, zero, add.lo);
  prod = hexkl_wide_add(prod, add);
  const HVX_Vector s = Q6_V_vand_VV(shift, v31);
  HVX_Vector cross =
    Q6_Vw_vasl_VwVw(prod.hi, Q6_V_vand_VV(Q6_Vw_vsub_VwVw(zero, s), v31));
  cross = Q6_V_vmux_QVV(shzero, zero, cross);
  HVX_Vector low =
    Q6_V_vmux_QVV(small, Q6_V_vor_VV(Q6_Vw_vlsr_VwVw(prod.lo, s), cross),
                  Q6_Vw_vlsr_VwVw(prod.hi, s));
  HVX_Vector high = Q6_V_vmux_QVV(small, Q6_Vw_vlsr_VwVw(prod.hi, s), zero);
  HVX_Vector bound =
    Q6_V_vmux_QVV(neg, zp, Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(65535), zp));
  HVX_Vector clipped = Q6_V_vmux_QVV(
    Q6_Q_vcmp_eq_VwVw(high, zero),
    Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VuwVuw(low, bound), bound, low), bound);
  return Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(zp, clipped),
                       Q6_Vw_vadd_VwVw(zp, clipped));
}
/** @brief Fetch the first 8 KiB, then 4 KiB blocks one block ahead. */
static inline void hexkl_requant_prefetch(const int32_t *src, uint32_t cols,
                                          uint32_t c) {
  if ((c & 1023u) == 0) {
    const uint32_t ahead = c ? c + 1024u : 0u;
    if (ahead < cols) {
      const uint32_t limit = c ? 1024u : 2048u;
      const uint32_t count = cols - ahead < limit ? cols - ahead : limit;
      // HVX streaming loads need explicit L2 prefetch. Stay within the row.
      Q6_l2fetch_AP((void *)(src + ahead), ((uint64_t)(count * 4u) << 32) |
                                             ((uint64_t)(count * 4u) << 16) |
                                             1u);
    }
  }
}

/** Vector body; scalar tail is handled by the caller. */
static inline uint32_t hexkl_requant_row_hvx(const int32_t *src, uint16_t *dst,
                                             uint32_t cols,
                                             const hexkl_requant_u16 *params,
                                             uint32_t n_params) {
  uint32_t c = 0;
  HVX_Vector vm = Q6_V_vsplat_R(params[0].multiplier);
  HVX_Vector vs = Q6_V_vsplat_R(params[0].shift);
  HVX_Vector vz = Q6_V_vsplat_R(params[0].zero_point);
  if (n_params == 1u) {
    for (; cols - c >= 64u; c += 64u) {
      hexkl_requant_prefetch(src, cols, c);
      HVX_Vector q0 =
        hexkl_requant_vector(*(const HVX_UVector *)(src + c), vm, vs, vz);
      HVX_Vector q1 =
        hexkl_requant_vector(*(const HVX_UVector *)(src + c + 32u), vm, vs, vz);
      *(HVX_UVector *)(dst + c) = Q6_Vuh_vpack_VwVw_sat(q1, q0);
    }
    return c;
  }
  for (; cols - c >= 64u; c += 64u) {
    hexkl_requant_prefetch(src, cols, c);
    HVX_Vector q[2];
    for (uint32_t half = 0; half < 2u; ++half) {
      hexkl_requant_params32(params + c + half * 32u, &vm, &vs, &vz);
      q[half] = hexkl_requant_vector(
        *(const HVX_UVector *)(src + c + half * 32u), vm, vs, vz);
    }
    *(HVX_UVector *)(dst + c) = Q6_Vuh_vpack_VwVw_sat(q[1], q[0]);
  }
  return c;
}
#endif
