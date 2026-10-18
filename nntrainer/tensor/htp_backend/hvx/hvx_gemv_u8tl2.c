// SPDX-License-Identifier: Apache-2.0
/**
 * @file hvx_gemv_u8tl2.c
 * @brief BitNet-style 3-weight/5-bit LUT GEMV for 128-byte Hexagon HVX.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "hvx_gemv_u8tl2.h"
#include <string.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static uint32_t blocks(uint32_t K) { return K / 24u + (K % 24u != 0); }

size_t hvx_tl2_weight_bytes(uint32_t K, uint32_t N) {
  if (!K || !N || K > INT32_MAX / 255u || N > UINT32_MAX - 127u)
    return 0;
  const uint64_t bytes =
    (uint64_t)blocks(K) * (N / 128u + (N % 128u != 0)) * 640u;
  return bytes > SIZE_MAX ? 0 : (size_t)bytes;
}

size_t hvx_tl2_table_bytes(uint32_t K) {
  if (!K || K > INT32_MAX / 255u)
    return 0;
  const uint64_t bytes = (uint64_t)blocks(K) * 512u;
  return bytes > SIZE_MAX ? 0 : (size_t)bytes;
}

uint32_t hvx_tl2_k_splits(uint32_t K, uint32_t N, uint32_t workers) {
  if (workers < 2u || workers > 8u || !hvx_tl2_weight_bytes(K, N))
    return 1u;
  const uint32_t panels = (N + 127u) / 128u;
  const uint32_t kb = blocks(K);
  uint64_t best = ((uint64_t)panels + workers - 1u) / workers * kb;
  uint32_t choice = 1u;
  for (uint32_t s = 2u; s <= 4u && kb >= 64u; s *= 2u) {
    const uint64_t cost =
      ((uint64_t)panels * s + workers - 1u) / workers * ((kb + s - 1u) / s);
    /* A small theoretical gain does not pay for the extra join/reduction. */
    if (cost * 10u < best * 9u) {
      choice = s;
      best = cost;
    }
  }
  return choice;
}

/** Balanced-base-3 digits of an unsigned magnitude (0..13). */
static void digits(unsigned m, int *a, int *b, int *c) {
  *a = (int)((m + 13u) / 9u) - 1;
  *b = (int)(((m + 13u) / 3u) % 3u) - 1;
  *c = (int)((m + 13u) % 3u) - 1;
}

#if defined(__hexagon__)
/** VLUT32 reads byte 2*i for index i. The low four index bits hold the
 * magnitude and bit four holds the sign, so the lookup produces the final
 * signed int4 nibble directly. Invalid magnitudes remain zero. */
#define TL2_DIGIT0(p, n)                                                       \
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p, p, p, p, p, p, p, p, p, p, p, p, p, p, p,   \
    p, p, p, 0, 0, 0, 0
#define TL2_DIGIT1(p, n)                                                       \
  0, 0, 0, 0, p, p, p, p, p, p, n, n, n, n, n, n, 0, 0, 0, 0, 0, 0, p, p, p,   \
    p, p, p, 0, 0, 0, 0
#define TL2_DIGIT2(p, n)                                                       \
  0, 0, p, p, n, n, 0, 0, p, p, n, n, 0, 0, p, p, n, n, 0, 0, p, p, n, n, 0,   \
    0, p, p, 0, 0, 0, 0
#define TL2_SIGNED_LUT(digit, p, n)                                            \
  { digit(p, n), digit(n, p), digit(p, n), digit(n, p) }
static const uint8_t tl2_expand_nibbles[2][3][128]
  __attribute__((aligned(128))) = {
    {TL2_SIGNED_LUT(TL2_DIGIT0, 1, 15), TL2_SIGNED_LUT(TL2_DIGIT1, 1, 15),
     TL2_SIGNED_LUT(TL2_DIGIT2, 1, 15)},
    {TL2_SIGNED_LUT(TL2_DIGIT0, 7, 9), TL2_SIGNED_LUT(TL2_DIGIT1, 7, 9),
     TL2_SIGNED_LUT(TL2_DIGIT2, 7, 9)}};
#undef TL2_SIGNED_LUT
#undef TL2_DIGIT2
#undef TL2_DIGIT1
#undef TL2_DIGIT0

static inline HVX_Vector tl2_expand_index(const uint8_t *src, uint32_t k) {
  const uint32_t group = k % 24u / 3u;
  const uint8_t *block = src + k / 24u * 640u;
  const HVX_Vector code_mask = Q6_V_vsplat_R(0x0f0f0f0f);
  HVX_Vector code = *(const HVX_UVector *)(block + group / 2u * 128u);
  if (group % 2u)
    code = Q6_Vuh_vlsr_VuhR(code, 4);
  code = Q6_V_vand_VV(code, code_mask);
  const HVX_Vector signs = *(const HVX_UVector *)(block + 512u);
  return Q6_V_vandor_VQR(code, Q6_Q_vand_VR(signs, 0x01010101u << group),
                         0x10101010);
}
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TL2_EXPAND_INLINE inline __attribute__((always_inline))
#else
#define TL2_EXPAND_INLINE inline
#endif

static TL2_EXPAND_INLINE void tl2_expand_wh96(const uint8_t *src, uint32_t rows,
                                              uint32_t columns,
                                              uint32_t row_stride,
                                              int magnitude, uint8_t *dst) {
#if defined(__hexagon__)
  if (magnitude == 1 || magnitude == 7) {
    const uint8_t(*table)[128] = tl2_expand_nibbles[magnitude == 7];
    const HVX_Vector low[3] = {*(const HVX_Vector *)table[0],
                               *(const HVX_Vector *)table[1],
                               *(const HVX_Vector *)table[2]};
    /** Shift the tables once and merge each high-nibble lookup with VLUT's
     * OR accumulator, instead of shifting and ORing every decoded row. */
    const HVX_Vector high[3] = {Q6_Vh_vasl_VhR(low[0], 4),
                                Q6_Vh_vasl_VhR(low[1], 4),
                                Q6_Vh_vasl_VhR(low[2], 4)};
    /** Constant row indices eliminate division and lookup-address work from
     * the inner loop, and keep the four transpose inputs in registers. */
#pragma unroll(12)
    for (uint32_t k = 0; k < 96u; k += 8u) {
      if (k >= rows / 32u * 32u)
        break;
      HVX_Vector bytes[4];
#pragma unroll(4)
      for (uint32_t r = 0; r < 4u; ++r) {
        const HVX_Vector lo = Q6_Vb_vlut32_VbVbI(tl2_expand_index(src, k + r),
                                                 low[(k + r) % 3u], 0);
        bytes[r] = Q6_Vb_vlut32or_VbVbVbI(lo, tl2_expand_index(src, k + r + 4u),
                                          high[(k + r + 4u) % 3u], 0);
      }
      /** Native byte j = 4*c + tile. Exchange the two tile bits with
       * the row bits in the four registers, leaving c in place. Each
       * result is one complete WH vector: byte 4*c + row. */
      const HVX_VectorPair a = Q6_W_vshuff_VVR(bytes[1], bytes[0], 1);
      const HVX_VectorPair b = Q6_W_vshuff_VVR(bytes[3], bytes[2], 1);
      const HVX_VectorPair even =
        Q6_W_vshuff_VVR(Q6_V_lo_W(b), Q6_V_lo_W(a), 2);
      const HVX_VectorPair odd = Q6_W_vshuff_VVR(Q6_V_hi_W(b), Q6_V_hi_W(a), 2);
      const HVX_Vector tiles[4] = {Q6_V_lo_W(even), Q6_V_lo_W(odd),
                                   Q6_V_hi_W(even), Q6_V_hi_W(odd)};
      uint8_t *out = dst + (size_t)(k / 32u) * row_stride + k % 32u / 8u * 128u;
#pragma unroll(4)
      for (uint32_t nt = 0; nt < 4u; ++nt)
        if (nt < columns / 32u)
          *(HVX_UVector *)(out + nt * 512u) = tiles[nt];
    }
    return;
  }
#endif
  /** Each magnitude names a balanced-base-three triple. This small inverse
     table also keeps integer division out of the weight loop. */
  static const int8_t trits[16][3] = {
    {0, 0, 0},   {0, 0, 1},  {0, 1, -1}, {0, 1, 0},  {0, 1, 1},
    {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0},
    {1, 0, 1},   {1, 1, -1}, {1, 1, 0},  {1, 1, 1}};
  for (uint32_t kt = 0; kt < rows / 32u; ++kt)
    for (uint32_t nt = 0; nt < columns / 32u; ++nt) {
      uint8_t *tile = dst + (size_t)kt * row_stride + nt * 512u;
      memset(tile, 0, 512u);
      for (uint32_t r = 0; r < 32u; ++r) {
        const uint32_t k = kt * 32u + r, group = k % 24u / 3u;
        const uint8_t *block = src + k / 24u * 640u;
        for (uint32_t c = 0; c < 32u; ++c) {
          const uint32_t j = c * 4u + nt;
          const unsigned m =
            (block[group / 2u * 128u + j] >> (4u * (group % 2u))) & 15u;
          const int sign = (block[512u + j] >> group) & 1u;
          const int q = (sign ? -magnitude : magnitude) * trits[m][k % 3u];
          tile[r / 8u * 128u + c * 4u + r % 4u] |= (uint8_t)(q & 15)
                                                   << (4u * (r / 4u % 2u));
        }
      }
    }
}

#undef TL2_EXPAND_INLINE

void hvx_tl2_expand_wh96(const uint8_t *src, uint32_t rows, uint32_t columns,
                         uint32_t row_stride, int magnitude, uint8_t *dst) {
  tl2_expand_wh96(src, rows, columns, row_stride, magnitude, dst);
}

void hvx_tl2_expand_wh96_disjoint(const uint8_t *__restrict src, uint32_t rows,
                                  uint32_t columns, uint32_t row_stride,
                                  int magnitude, uint8_t *__restrict dst) {
  tl2_expand_wh96(src, rows, columns, row_stride, magnitude, dst);
}

int hvx_tl2_pack(const int8_t *kn, uint32_t K, uint32_t N, uint8_t *packed) {
  const size_t bytes = hvx_tl2_weight_bytes(K, N);
  if (!kn || !packed || !bytes || (size_t)K > SIZE_MAX / N)
    return -1;
  for (size_t i = 0; i < (size_t)K * N; ++i)
    if (kn[i] < -1 || kn[i] > 1)
      return -1;
  memset(packed, 0, bytes);
  for (uint32_t p = 0; p < (N + 127u) / 128u; ++p)
    for (uint32_t kb = 0; kb < blocks(K); ++kb) {
      uint8_t *dst = packed + ((size_t)p * blocks(K) + kb) * 640u;
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint32_t n = p * 128u + (j % 4u) * 32u + j / 4u;
        for (uint32_t g = 0; g < 8u; ++g) {
          int value = 0;
          for (uint32_t t = 0; t < 3u; ++t) {
            const uint32_t k = kb * 24u + g * 3u + t;
            value = 3 * value + (k < K && n < N ? kn[(size_t)k * N + n] : 0);
          }
          const unsigned mag = value < 0 ? (unsigned)-value : (unsigned)value;
          dst[(g / 2u) * 128u + j] |= mag << (4u * (g % 2u));
          dst[512u + j] |= (value < 0) << g;
        }
      }
    }
  return 0;
}

int hvx_tl2_unpack(const uint8_t *packed, uint32_t K, uint32_t N, int8_t *kn) {
  if (!packed || !kn || !hvx_tl2_weight_bytes(K, N) || (size_t)K > SIZE_MAX / N)
    return -1;
  for (uint32_t p = 0; p < (N + 127u) / 128u; ++p)
    for (uint32_t kb = 0; kb < blocks(K); ++kb) {
      const uint8_t *src = packed + ((size_t)p * blocks(K) + kb) * 640u;
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint32_t n = p * 128u + (j % 4u) * 32u + j / 4u;
        for (uint32_t g = 0; g < 8u; ++g) {
          const unsigned mag =
            (src[(g / 2u) * 128u + j] >> (4u * (g % 2u))) & 15u;
          const int sign = (src[512u + j] >> g) & 1u;
          if (mag > 13u || (!mag && sign))
            return -1;
          int t[3];
          digits(mag, &t[0], &t[1], &t[2]);
          for (uint32_t i = 0; i < 3u; ++i) {
            const uint32_t k = kb * 24u + g * 3u + i;
            const int w = sign ? -t[i] : t[i];
            if (k < K && n < N)
              kn[(size_t)k * N + n] = (int8_t)w;
            else if (w)
              return -1;
          }
        }
      }
    }
  return 0;
}

#if defined(__hexagon__)
/* Two 16-entry magnitude tables per vector, duplicated halfwords for VLUT. */
static const int16_t tl2_digits[3][64] __attribute__((aligned(128))) = {
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
   1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
  {0,  0,  0,  0,  1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,  0,
   1,  1,  1,  1,  1, 1, 0, 0, 0, 0, 0,  0,  0,  0,  1,  1,  1, 1, 1, 1, -1, -1,
   -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1,  1,  1,  1,  1,  1,  0, 0, 0, 0},
  {0,  0,  1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1,  1,  -1, -1, 0, 0, 1,  1,
   -1, -1, 0, 0, 1,  1,  0, 0, 0, 0, 0,  0,  1, 1, -1, -1, 0,  0,  1, 1, -1, -1,
   0,  0,  1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 1,  1,  0,  0,  0, 0}};
#endif

void hvx_tl2_prepare(const uint8_t *act, uint32_t K, uint32_t begin,
                     uint32_t end, int16_t *tables) {
#if defined(__hexagon__)
  const HVX_Vector x = *(const HVX_Vector *)tl2_digits[0];
  const HVX_Vector y = *(const HVX_Vector *)tl2_digits[1];
  const HVX_Vector z = *(const HVX_Vector *)tl2_digits[2];
  const HVX_VectorPred lower = Q6_Q_vsetq_R(64);
  /** VLUT32 broadcasts each activation into the required halfword lanes.
   * Even index bytes select its low byte; odd indices start at 32, outside
   * this lookup's range, and therefore produce zero high bytes. One bounded
   * vector load supplies all four tables. Keep the scalar tail below when
   * fewer than 128 source bytes remain, including partitioned calls. */
  const uint32_t load_end = K < 128u ? 0u : (K - 128u) / 24u + 1u;
  const uint32_t vector_end = end < load_end ? end : load_end;
  if (begin < vector_end) {
    const HVX_Vector idx0 = Q6_V_vmux_QVV(lower, Q6_V_vsplat_R(0x20002000),
                                          Q6_V_vsplat_R(0x20032003));
    const HVX_Vector idx1 = Q6_Vb_vadd_VbVb(idx0, Q6_V_vsplat_R(0x01010101));
    const HVX_Vector idx2 = Q6_Vb_vadd_VbVb(idx0, Q6_V_vsplat_R(0x02020202));
    do {
      const HVX_Vector actv = Q6_V_lo_W(
        Q6_Wuh_vunpack_Vub(*(const HVX_UVector *)(act + begin * 24u)));
#pragma clang loop unroll(full)
      for (uint32_t pair = 0; pair < 4u; ++pair) {
        const HVX_Vector off = Q6_V_vsplat_R(0x06060606u * pair);
        const HVX_Vector va =
          Q6_Vb_vlut32_VbVbI(Q6_Vb_vadd_VbVb(idx0, off), actv, 0);
        const HVX_Vector vb =
          Q6_Vb_vlut32_VbVbI(Q6_Vb_vadd_VbVb(idx1, off), actv, 0);
        const HVX_Vector vc =
          Q6_Vb_vlut32_VbVbI(Q6_Vb_vadd_VbVb(idx2, off), actv, 0);
        *(HVX_Vector *)(tables + ((size_t)begin * 4u + pair) * 64u) =
          Q6_Vh_vmpyiacc_VhVhVh(
            Q6_Vh_vmpyiacc_VhVhVh(Q6_Vh_vmpyi_VhVh(va, x), vb, y), vc, z);
      }
      ++begin;
    } while (begin < vector_end);
  }
#endif
  for (uint32_t kb = begin; kb < end; ++kb)
    for (uint32_t pair = 0; pair < 4u; ++pair) {
      int16_t *tab = tables + ((size_t)kb * 4u + pair) * 64u;
#if defined(__hexagon__)
      const uint32_t k = kb * 24u + pair * 6u;
      int a[6];
      for (unsigned t = 0; t < 6; ++t)
        a[t] = k + t < K ? act[k + t] : 0;
      const HVX_Vector va =
        Q6_V_vmux_QVV(lower, Q6_Vh_vsplat_R(a[0]), Q6_Vh_vsplat_R(a[3]));
      const HVX_Vector vb =
        Q6_V_vmux_QVV(lower, Q6_Vh_vsplat_R(a[1]), Q6_Vh_vsplat_R(a[4]));
      const HVX_Vector vc =
        Q6_V_vmux_QVV(lower, Q6_Vh_vsplat_R(a[2]), Q6_Vh_vsplat_R(a[5]));
      *(HVX_Vector *)tab = Q6_Vh_vmpyiacc_VhVhVh(
        Q6_Vh_vmpyiacc_VhVhVh(Q6_Vh_vmpyi_VhVh(va, x), vb, y), vc, z);
#else
      for (uint32_t h = 0; h < 2u; ++h) {
        const uint32_t k = kb * 24u + (pair * 2u + h) * 3u;
        const int a = k < K ? act[k] : 0;
        const int b = k + 1u < K ? act[k + 1u] : 0;
        const int c = k + 2u < K ? act[k + 2u] : 0;
        for (uint32_t m = 0; m < 16u; ++m) {
          int x, y, z;
          digits(m < 14u ? m : 0u, &x, &y, &z);
          const int16_t v = (int16_t)(a * x + b * y + c * z);
          tab[h * 32u + 2u * m] = v;
          tab[h * 32u + 2u * m + 1u] = v;
        }
      }
#endif
    }
}

#ifndef NNTR_TL2_UNROLL
#define NNTR_TL2_UNROLL 4
#endif

#ifndef NNTR_TL2_PREDICATED_SIGN
#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 81
#define NNTR_TL2_PREDICATED_SIGN 1
#else
#define NNTR_TL2_PREDICATED_SIGN 0
#endif
#endif

void hvx_gemv_u8tl2_panel(const uint8_t *src, uint32_t K, const int16_t *tables,
                          int32_t *out) {
#if defined(__hexagon__)
  const HVX_Vector zero = Q6_V_vzero();
  HVX_VectorPair even = Q6_W_vcombine_VV(zero, zero), odd = even;
#if __HEXAGON_ARCH__ < 81
  const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector select = Q6_V_vsplat_R(0x10101010);
#endif
#if NNTR_TL2_PREDICATED_SIGN
  /* Keep adjacent K blocks from competing for the four predicate registers. */
#pragma clang loop pipeline(disable)
#endif
  for (uint32_t kb = 0; kb < blocks(K); ++kb) {
    const HVX_Vector signs =
      *(const HVX_Vector *)(src + (size_t)kb * 640u + 512u);
#if NNTR_TL2_PREDICATED_SIGN
    /** VLUT's low/high vectors correspond to even/odd index bytes. Duplicate
     * each sign byte across its halfword so both predicate bits agree. */
    const HVX_Vector se = Q6_Vb_vshuffe_VbVb(signs, signs);
    const HVX_Vector so = Q6_Vb_vshuffo_VbVb(signs, signs);
#endif
    HVX_Vector le = zero, lo = zero, he = zero, ho = zero;
#pragma clang loop unroll_count(NNTR_TL2_UNROLL)
    for (uint32_t g = 0; g < 4u; ++g) {
      const HVX_Vector codes =
        *(const HVX_Vector *)(src + (size_t)kb * 640u + g * 128u);
      const HVX_Vector tab =
        *(const HVX_Vector *)(tables + ((size_t)kb * 4u + g) * 64u);
#if __HEXAGON_ARCH__ >= 81
      const HVX_VectorPair l = Q6_Wh_vlut16_VbVhR_nomatch(codes, tab, 0);
      const HVX_VectorPair h =
        Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vuh_vlsr_VuhR(codes, 4), tab, 1);
#else
      const HVX_VectorPair l =
        Q6_Wh_vlut16_VbVhI(Q6_V_vand_VV(codes, mask), tab, 0);
      const HVX_VectorPair h = Q6_Wh_vlut16_VbVhI(
        Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), select),
        tab, 1);
#endif
#if NNTR_TL2_PREDICATED_SIGN
      const HVX_VectorPred sl0 = Q6_Q_vand_VR(se, 0x01010101u << (2 * g));
      const HVX_VectorPred sl1 = Q6_Q_vand_VR(so, 0x01010101u << (2 * g));
      const HVX_VectorPred sh0 = Q6_Q_vand_VR(se, 0x02020202u << (2 * g));
      const HVX_VectorPred sh1 = Q6_Q_vand_VR(so, 0x02020202u << (2 * g));
      le = Q6_Vh_condnac_QVhVh(sl0, Q6_Vh_condacc_QnVhVh(sl0, le, Q6_V_lo_W(l)),
                               Q6_V_lo_W(l));
      lo = Q6_Vh_condnac_QVhVh(sl1, Q6_Vh_condacc_QnVhVh(sl1, lo, Q6_V_hi_W(l)),
                               Q6_V_hi_W(l));
      he = Q6_Vh_condnac_QVhVh(sh0, Q6_Vh_condacc_QnVhVh(sh0, he, Q6_V_lo_W(h)),
                               Q6_V_lo_W(h));
      ho = Q6_Vh_condnac_QVhVh(sh1, Q6_Vh_condacc_QnVhVh(sh1, ho, Q6_V_hi_W(h)),
                               Q6_V_hi_W(h));
#else
      /** VLUT halfword results separate even/odd input bytes. Extract their
       * packed sign bits directly into full halfword masks; no unpack/shuffle.
       */
      const HVX_Vector sl0 =
        Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(signs, 15 - 2 * g), 15);
      const HVX_Vector sl1 =
        Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(signs, 7 - 2 * g), 15);
      const HVX_Vector sh0 =
        Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(signs, 14 - 2 * g), 15);
      const HVX_Vector sh1 =
        Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(signs, 6 - 2 * g), 15);
      le = Q6_Vh_vadd_VhVh(
        le, Q6_Vh_vsub_VhVh(Q6_V_vxor_VV(Q6_V_lo_W(l), sl0), sl0));
      lo = Q6_Vh_vadd_VhVh(
        lo, Q6_Vh_vsub_VhVh(Q6_V_vxor_VV(Q6_V_hi_W(l), sl1), sl1));
      he = Q6_Vh_vadd_VhVh(
        he, Q6_Vh_vsub_VhVh(Q6_V_vxor_VV(Q6_V_lo_W(h), sh0), sh0));
      ho = Q6_Vh_vadd_VhVh(
        ho, Q6_Vh_vsub_VhVh(Q6_V_vxor_VV(Q6_V_hi_W(h), sh1), sh1));
#endif
    }
    even = Q6_Ww_vaddacc_WwVhVh(even, le, he);
    odd = Q6_Ww_vaddacc_WwVhVh(odd, lo, ho);
  }
  *(HVX_Vector *)(out + 0u) = Q6_V_lo_W(even);
  *(HVX_Vector *)(out + 32u) = Q6_V_lo_W(odd);
  *(HVX_Vector *)(out + 64u) = Q6_V_hi_W(even);
  *(HVX_Vector *)(out + 96u) = Q6_V_hi_W(odd);
#else
  memset(out, 0, 128u * sizeof(*out));
  for (uint32_t kb = 0; kb < blocks(K); ++kb)
    for (uint32_t g = 0; g < 8u; ++g)
      for (uint32_t j = 0; j < 128u; ++j) {
        const uint8_t *s = src + (size_t)kb * 640u;
        const unsigned mag = (s[(g / 2u) * 128u + j] >> (4u * (g % 2u))) & 15u;
        const int16_t v =
          tables[((size_t)kb * 4u + g / 2u) * 64u + (g % 2u) * 32u + 2u * mag];
        out[(j % 4u) * 32u + j / 4u] += ((s[512u + j] >> g) & 1u) ? -v : v;
      }
#endif
}
