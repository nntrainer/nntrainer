// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   q4_gemv_cpu_det.h
 * @date   29 Sep 2026
 * @brief  The Android CPU's M=1 Q4_0 fully connected layer as a scalar
 *         spec: the Q8_0 activation quantizer, the per-column fused chain,
 *         the Q4M1 weight layout the DSP kernel reads, and the exact scalar
 *         helpers the CPU-order specs share (#132 PR 2)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * The decode's text must equal the CPU run's (LEDGER rule 45), so an FC the
 * HTP runs has to produce the Android CPU's bits, not merely close ones.
 * This header writes down what the shipped libnntrainer.so computes, read
 * off its aarch64 disassembly (plan 132-cpu-exact-fc-lmhead.md section 0):
 *
 *   nntr_quantize_row_q8_0, per block of 32:
 *     amax = max |x|;  d = RN(amax / 127)  (a true fdiv)
 *     id   = d != 0 ? RN(1 / d) : 0        (fdiv, fcmp, fcsel)
 *     q[j] = low byte of fcvtns(RN(x[j] * id))   (RN ties-even, saturating)
 *     the block stores fcvt h, s (d):  RN f32 -> f16, subnormals kept
 *   nntr_gemv_q4_0_4x8_q8_0, per output column, from acc = +0:
 *     isum_b = sum_k (w_k - 8) * q_k   (exact int32, sdot)
 *     s_b    = f16->f32(d_a) * f16->f32(d_w)   (exact: 11 x 11 bits)
 *     acc    = fma(isum_b, s_b, acc)   for b = 0 .. K/32 - 1 in order
 *
 * The thread split (16-column chunks) does not change a bit: one call owns
 * a whole column. The output is never -0 (acc starts at +0).
 *
 * THE HELPERS. Every f32 operation stores through a volatile so no compiler
 * contracts or reassociates it; cpu_det_fma is the one fused step (fmaf,
 * or the DSP's sffma). Division, f32 -> f16 and float -> int are done in
 * integers, correctly rounded, so the spec needs no libm and no divider
 * and gives the same bits on the host, the ARM and the DSP.
 *
 * DOMAIN. Finite inputs. A block whose amax / 127 overflows f16 (|x| above
 * about 8.3e6) stores d = inf, which the CPU would then multiply as inf;
 * the Q4M1 kernel reads exponent 31 as a finite number, so such a block is
 * outside the kernel's domain (no decode activation comes near it).
 *
 * THE Q4M1 LAYOUT (the DSP kernel's weight order; hvx_q4_gemv_f32.c). The
 * same nibbles and f16 scales as block_q4_0, reordered at registration:
 * for each group of 32 columns, for each pair of 32-blocks, 1152 bytes:
 *   [0, 512)     block 2p:   4 vectors j = 0..3 of 128 bytes; byte 4l + r
 *                (r < 4) of vector j holds column l's element 8j + r in its
 *                low nibble and element 8j + 4 + r in its high nibble
 *   [512, 1024)  block 2p+1, the same
 *   [1024, 1152) 64 f16 d: block 2p's 32 columns, then block 2p+1's
 * Nibbles are the stored q (0..15), not q - 8. K % 64 == 0, N % 32 == 0.
 */

#ifndef __NNTRAINER_Q4_GEMV_CPU_DET_H__
#define __NNTRAINER_Q4_GEMV_CPU_DET_H__

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__hexagon__)
#include <hexagon_protos.h>
#endif

/** @brief Elements per Q4_0 / Q8_0 block. */
#define Q4_CPU_QK 32u
/** @brief block_q4_0: f16 d, then 16 bytes of nibbles (low: k, high: k+16). */
#define Q4_CPU_BLOCK_BYTES 18u
/** @brief Columns per Q4M1 group: one vector of 32 f32 lanes. */
#define Q4M1_GROUP 32u
/** @brief One Q4M1 unit: 2 blocks x 512 nibble bytes + 64 f16 scales. */
#define Q4M1_PAIR_BYTES 1152u
/** @brief Offset of the 64 f16 scales in a unit. */
#define Q4M1_D_OFF 1024u

/* ---- one IEEE f32 operation each ---------------------------------------- */

static inline uint32_t cpu_det_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}
static inline float cpu_det_float(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}
static inline float cpu_det_mul(float a, float b) {
  volatile float r = a * b;
  return r;
}
static inline float cpu_det_add(float a, float b) {
  volatile float r = a + b;
  return r;
}
static inline float cpu_det_sub(float a, float b) {
  volatile float r = a - b;
  return r;
}
/** @brief a * b + c with ONE rounding: the CPU's fmla / fmadd. On the DSP
 *         the scalar sffma (a plain fmaf there is a libm call the skel may
 *         not import). */
static inline float cpu_det_fma(float a, float b, float c) {
#if defined(__hexagon__)
  volatile float r = Q6_R_sfmpyacc_RR(c, a, b);
#else
  volatile float r = fmaf(a, b, c);
#endif
  return r;
}

/* ---- exact integer rounding --------------------------------------------- */

/** @brief Bit length of @a m (0 for 0). */
static inline int cpu_det_bitlen(uint64_t m) {
  int n = 0;
  for (int s = 32; s > 0; s >>= 1) {
    if (m >> s) {
      m >>= s;
      n += s;
    }
  }
  return n + (int)(m != 0u);
}

/** @brief RN(m / 2^sh) ties to even; sh <= 0 shifts left (exact). */
static inline uint64_t cpu_det_round_shift(uint64_t m, int sh) {
  if (sh <= 0) {
    return m << -sh;
  }
  if (sh >= 64) {
    return 0u; /* m < 2^63 <= half the quantum */
  }
  const uint64_t r = m >> sh, rem = m & ((1ull << sh) - 1u);
  const uint64_t half = 1ull << (sh - 1);
  return r + (uint64_t)(rem > half || (rem == half && (r & 1u)));
}

/**
 * @brief RN of (-1)^s m 2^e as an IEEE binary format with @a mbits stored
 *        mantissa bits and exponent bias @a bias, subnormals kept, overflow
 *        to inf. m > 0, m < 2^63. Returns the encoding without the sign.
 */
static inline uint32_t cpu_det_pack(uint64_t m, int e, int mbits, int bias,
                                    uint32_t emax_field) {
  const int n = cpu_det_bitlen(m);
  const int lead = n - 1 + e; /* exponent of the leading bit */
  const int emin = 1 - bias;
  if (lead >= emin) {
    uint64_t r = cpu_det_round_shift(m, n - 1 - mbits);
    uint32_t be = (uint32_t)(lead + bias);
    if (r >> (mbits + 1)) {
      r >>= 1;
      ++be;
    }
    if (be >= emax_field) {
      return emax_field << mbits; /* inf */
    }
    return (be << mbits) | (uint32_t)(r & ((1ull << mbits) - 1u));
  }
  /* subnormal: quantum 2^(emin - mbits); r == 2^mbits is the smallest
     normal, which the encoding gives for free */
  return (uint32_t)cpu_det_round_shift(m, (emin - mbits) - e);
}

/** @brief The quiet NaN every NaN result of these helpers returns. */
#define CPU_DET_NAN 0x7fc00000u

/** @brief RN(a / b), IEEE division in integers (fdiv; the DSP has none). */
static inline float cpu_det_div_rn(float a, float b) {
  const uint32_t ua = cpu_det_bits(a), ub = cpu_det_bits(b);
  const uint32_t s = (ua ^ ub) & 0x80000000u;
  const uint32_t aa = ua & 0x7fffffffu, ab = ub & 0x7fffffffu;
  if (aa > 0x7f800000u || ab > 0x7f800000u) {
    return cpu_det_float(CPU_DET_NAN);
  }
  if (aa == 0x7f800000u) {
    return cpu_det_float(ab == 0x7f800000u ? CPU_DET_NAN : s | 0x7f800000u);
  }
  if (ab == 0x7f800000u) {
    return cpu_det_float(s);
  }
  if (ab == 0u) {
    return cpu_det_float(aa == 0u ? CPU_DET_NAN : s | 0x7f800000u);
  }
  if (aa == 0u) {
    return cpu_det_float(s);
  }
  uint64_t ma = (aa >> 23) ? ((aa & 0x7fffffu) | 0x800000u) : aa;
  uint64_t mb = (ab >> 23) ? ((ab & 0x7fffffu) | 0x800000u) : ab;
  int xa = (int)((aa >> 23) ? (aa >> 23) : 1u) - 150;
  int xb = (int)((ab >> 23) ? (ab >> 23) : 1u) - 150;
  while (ma < 0x800000u) {
    ma <<= 1;
    --xa;
  }
  while (mb < 0x800000u) {
    mb <<= 1;
    --xb;
  }
  /* q has 40 or 41 bits; the remainder is a sticky bit below them, far
     under any rounding position (>= 16 bits are dropped) */
  const uint64_t num = ma << 40;
  const uint64_t q = ((num / mb) << 1) | (uint64_t)(num % mb != 0u);
  return cpu_det_float(s | cpu_det_pack(q, xa - xb - 41, 23, 127, 255u));
}

/** @brief fcvt h, s: RN f32 -> f16, subnormals kept, overflow to inf. */
static inline uint16_t cpu_det_f32_to_f16(float f) {
  const uint32_t u = cpu_det_bits(f);
  const uint16_t s = (uint16_t)((u >> 16) & 0x8000u);
  const uint32_t a = u & 0x7fffffffu;
  if (a >= 0x7f800000u) {
    return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
  }
  if (a == 0u) {
    return s;
  }
  const uint64_t m = (a >> 23) ? ((a & 0x7fffffu) | 0x800000u) : a;
  const int e = (int)((a >> 23) ? (a >> 23) : 1u) - 150;
  return (uint16_t)(s | cpu_det_pack(m, e, 10, 15, 31u));
}

/** @brief fcvtl: f16 -> f32, exact. */
static inline float cpu_det_f16_to_f32(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 31u, m = h & 0x3ffu;
  if (e == 31u) {
    return cpu_det_float(s | 0x7f800000u | (m << 13));
  }
  if (e == 0u) {
    if (m == 0u) {
      return cpu_det_float(s);
    }
    e = 1u;
    while (!(m & 0x400u)) {
      m <<= 1;
      --e;
    }
    m &= 0x3ffu;
  }
  return cpu_det_float(s | ((e + 112u) << 23) | (m << 13));
}

/** @brief fcvtns: RN ties-even to int32, saturating, NaN -> 0. */
static inline int32_t cpu_det_fcvtns(float f) {
  const uint32_t u = cpu_det_bits(f), a = u & 0x7fffffffu;
  if (a > 0x7f800000u) {
    return 0;
  }
  if (a >= 0x4f000000u) { /* |f| >= 2^31 */
    return (u >> 31) ? INT32_MIN : INT32_MAX;
  }
  if (a == 0u) {
    return 0;
  }
  const uint64_t m = (a >> 23) ? ((a & 0x7fffffu) | 0x800000u) : a;
  const int e = (int)((a >> 23) ? (a >> 23) : 1u) - 150;
  const uint64_t r = cpu_det_round_shift(m, -e);
  return (u >> 31) ? -(int32_t)r : (int32_t)r;
}

/* ---- the FC ------------------------------------------------------------- */

/**
 * @brief nntr_quantize_row_q8_0: K floats (K % 32 == 0) into K int8 and
 *        K / 32 f16 scales (the q8_0 block's d).
 */
static inline void q8_0_quant_cpu_det(const float *x, uint32_t K, int8_t *q,
                                      uint16_t *d) {
  for (uint32_t b = 0; b < K / Q4_CPU_QK; ++b) {
    const float *xb = x + (size_t)b * Q4_CPU_QK;
    float amax = 0.0f;
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      const float v = cpu_det_float(cpu_det_bits(xb[j]) & 0x7fffffffu);
      amax = v > amax ? v : amax;
    }
    const float db = cpu_det_div_rn(amax, 127.0f);
    const float id = db != 0.0f ? cpu_det_div_rn(1.0f, db) : 0.0f;
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      q[(size_t)b * Q4_CPU_QK + j] =
        (int8_t)(uint8_t)(cpu_det_fcvtns(cpu_det_mul(xb[j], id)) & 0xff);
    }
    d[b] = cpu_det_f32_to_f16(db);
  }
}

/** @brief The exact int32 dot of one block: sum (w - 8) * q. @a qs is a
 *         block_q4_0's 16 nibble bytes. */
static inline int32_t q4_cpu_block_isum(const uint8_t *qs, const int8_t *q) {
  int32_t isum = 0;
  for (uint32_t j = 0; j < 16u; ++j) {
    isum += ((int32_t)(qs[j] & 15u) - 8) * q[j];
    isum += ((int32_t)(qs[j] >> 4) - 8) * q[j + 16u];
  }
  return isum;
}

/** @brief A block_q4_0's f16 d (little endian, the file's byte order). */
static inline uint16_t q4_cpu_block_d(const uint8_t *blk) {
  return (uint16_t)(blk[0] | (blk[1] << 8));
}

/**
 * @brief nntr_gemv_q4_0_4x8_q8_0 per column over canonical block_q4_0
 *        rows: column n's K / 32 blocks at w + (n K/32 + b) * 18.
 */
static inline void q4_gemv_cpu_det(const uint8_t *w, const int8_t *q,
                                   const uint16_t *da, uint32_t K, uint32_t N,
                                   float *y) {
  const uint32_t nb = K / Q4_CPU_QK;
  for (uint32_t n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
      const uint8_t *blk = w + ((size_t)n * nb + b) * Q4_CPU_BLOCK_BYTES;
      const int32_t isum =
        q4_cpu_block_isum(blk + 2, q + (size_t)b * Q4_CPU_QK);
      const float s = cpu_det_mul(cpu_det_f16_to_f32(da[b]),
                                  cpu_det_f16_to_f32(q4_cpu_block_d(blk)));
      acc = cpu_det_fma((float)isum, s, acc);
    }
    y[n] = acc;
  }
}

/* ---- the Q4M1 layout ---------------------------------------------------- */

/** @brief Bytes of a K x N weight in Q4M1 (the same as block_q4_0). */
static inline size_t q4m1_bytes(uint32_t K, uint32_t N) {
  return (size_t)(N / Q4M1_GROUP) * (K / 64u) * Q4M1_PAIR_BYTES;
}

/** @brief Canonical block_q4_0 [N][K/32] -> Q4M1 (layout above). */
static inline void q4m1_from_q4_0(const uint8_t *w, uint32_t K, uint32_t N,
                                  uint8_t *out) {
  const uint32_t nb = K / Q4_CPU_QK, np = nb / 2u;
  for (uint32_t g = 0; g < N / Q4M1_GROUP; ++g) {
    for (uint32_t p = 0; p < np; ++p) {
      uint8_t *unit = out + ((size_t)g * np + p) * Q4M1_PAIR_BYTES;
      memset(unit, 0, Q4M1_D_OFF);
      for (uint32_t hb = 0; hb < 2u; ++hb) {
        for (uint32_t l = 0; l < Q4M1_GROUP; ++l) {
          const uint8_t *blk =
            w + ((size_t)(g * Q4M1_GROUP + l) * nb + 2u * p + hb) *
                  Q4_CPU_BLOCK_BYTES;
          uint8_t *dp = unit + Q4M1_D_OFF + 2u * (hb * Q4M1_GROUP + l);
          dp[0] = blk[0];
          dp[1] = blk[1];
          for (uint32_t k = 0; k < Q4_CPU_QK; ++k) {
            const uint32_t v =
              k < 16u ? (blk[2 + k] & 15u) : (blk[2 + k - 16u] >> 4);
            const uint32_t j = k / 8u, r = k % 8u;
            uint8_t *byte = unit + hb * 512u + j * 128u + 4u * l + (r & 3u);
            *byte |= (uint8_t)(r < 4u ? v : v << 4);
          }
        }
      }
    }
  }
}

/**
 * @brief The ARM's repacked q4_0x4 (nntr_make_block_q4_0x4, interleave 8,
 *        nibbles XOR 0x88) back to canonical block_q4_0 [N][K/32]: per 4
 *        columns and block, 72 bytes = d[4] then qs[64], column c's byte j
 *        at qs[((j / 8) * 4 + c) * 8 + j % 8].
 */
static inline void q4_0_from_q4_0x4(const uint8_t *x4, uint32_t K, uint32_t N,
                                    uint8_t *out) {
  const uint32_t nb = K / Q4_CPU_QK;
  for (uint32_t n4 = 0; n4 < N / 4u; ++n4) {
    for (uint32_t b = 0; b < nb; ++b) {
      const uint8_t *src = x4 + ((size_t)n4 * nb + b) * 72u;
      for (uint32_t c = 0; c < 4u; ++c) {
        uint8_t *blk =
          out + ((size_t)(4u * n4 + c) * nb + b) * Q4_CPU_BLOCK_BYTES;
        blk[0] = src[2u * c];
        blk[1] = src[2u * c + 1u];
        for (uint32_t j = 0; j < 16u; ++j) {
          blk[2 + j] =
            (uint8_t)(src[8u + ((j / 8u) * 4u + c) * 8u + j % 8u] ^ 0x88u);
        }
      }
    }
  }
}

#endif /* __NNTRAINER_Q4_GEMV_CPU_DET_H__ */
