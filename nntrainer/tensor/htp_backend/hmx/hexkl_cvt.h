// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hexkl_cvt.h
 * @date   06 Oct 2026
 * @brief  One pass of the HMX convert unit, issued inline
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The only way out of an HMX accumulator is the convert unit: it applies a
 * per-output-channel scale (and a small additive bias) to the 37-bit
 * accumulator and writes one byte per element to VTCM. HexKL's
 * hexkl_micro_hmx_acc_read_int32 runs it four times with scales 512, 2,
 * 2^-7 and 0 to extract the four bytes of the int32, then re-interleaves
 * them with an HVX shuffle that costs more than the passes. A quantized
 * kernel does not want the int32: it wants the accumulator already scaled
 * to its next operand's precision, which is what the convert unit is for.
 *
 * What the unit computes, measured on v81 (22_attention_vs_qnn_report.md,
 * D6): for output channel c, out[p] = floor(acc[p] * scale[c] / 512),
 * truncated to its low 8 bits, with byte p of the 2 KiB destination being
 * spatial p/32, channel p%32 -- the uint8 activation tile layout. Two
 * passes with scales s and s/256 therefore yield the low and high bytes of
 * the low 16 bits of floor(acc * s / 512), exactly, including two's
 * complement wraparound.
 *
 * The bias block the unit reads is laid out as HexKL's setup writes it:
 * 64 32-bit words, word c (and c+32) for channel c, fp16 scale in bits
 * 15..0, the additive bias in bits 31..16, then zeros to 1 KiB. The two
 * instructions are the ones hexkl_micro_hmx_acc_read_int32 issues, read
 * out of libhexkl_micro.a:
 *
 *   bias = mxmem(Rs)                             load the bias block
 *   mxmem(Rd, #0):after:retain:cm.ub = acc       convert; keep the acc
 *
 * Both operate only on memory this code allocates: the block and the
 * destination tile are the caller's VTCM.
 */

#ifndef __NNTRAINER_HEXKL_CVT_H__
#define __NNTRAINER_HEXKL_CVT_H__

#include <stdint.h>
#include <string.h>

/** @brief Bytes of one bias block; also its alignment. */
#define HEXKL_CVT_BLOCK_BYTES 1024u
/** @brief Bytes one convert pass writes: a 64x32 uint8 tile. */
#define HEXKL_CVT_PLANE_BYTES 2048u
/** @brief Alignment of a destination plane (the activation alignment). */
#define HEXKL_CVT_PLANE_ALIGN 2048u

/**
 * @brief Fills a bias block: @a scale_hf[c] is the fp16 bit pattern of the
 *        scale of output channel c, @a bias_q[c] the 16-bit additive field
 *        (0 for none).
 *
 * @param block  HEXKL_CVT_BLOCK_BYTES bytes, HEXKL_CVT_BLOCK_BYTES aligned
 */
static inline void hexkl_cvt_block_set(void *block, const uint16_t *scale_hf,
                                       const uint16_t *bias_q) {
  uint32_t *w = (uint32_t *)block;
  memset(block, 0, HEXKL_CVT_BLOCK_BYTES);
  for (uint32_t c = 0; c < 32u; ++c) {
    const uint32_t word =
      (uint32_t)scale_hf[c] | ((uint32_t)(bias_q ? bias_q[c] : 0u) << 16);
    w[c] = word;
    w[c + 32u] = word;
  }
}

/** @brief A block with the same scale on every channel and no bias. */
static inline void hexkl_cvt_block_set_uniform(void *block, uint16_t scale_hf) {
  uint16_t s[32];
  for (uint32_t c = 0; c < 32u; ++c) {
    s[c] = scale_hf;
  }
  hexkl_cvt_block_set(block, s, (const uint16_t *)0);
}

/**
 * @brief fp16 bit pattern of @a x (round to nearest even). Host-side
 *        helper for building scale tables; not for hot loops.
 */
static inline uint16_t hexkl_cvt_f32_to_hf(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = u & 0x7fffffu;
  if (((u >> 23) & 0xffu) == 0xffu) {
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  }
  if (exp >= 31) {
    return (uint16_t)(sign | 0x7c00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return (uint16_t)sign;
    }
    mant |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u))) {
      ++half;
    }
    return (uint16_t)(sign | half);
  }
  uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) {
    ++half;
  }
  return (uint16_t)(sign | half);
}

/** @brief The float an fp16 bit pattern denotes. */
static inline float hexkl_cvt_hf_to_f32(uint16_t h) {
  const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t mant = h & 0x3ffu;
  uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {
      // subnormal: value = mant * 2^-24
      float f = (float)mant * (1.0f / 16777216.0f);
      memcpy(&u, &f, 4);
      u |= sign;
    }
  } else if (exp == 31) {
    u = sign | 0x7f800000u | (mant << 13);
  } else {
    u = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &u, 4);
  return f;
}

/**
 * @brief Host model of one pass: the byte the unit writes for accumulator
 *        value @a acc under fp16 scale @a scale_hf. floor(acc * s / 512)
 *        is exact in double for |acc| < 2^42.
 */
static inline uint8_t hexkl_cvt_model(int32_t acc, uint16_t scale_hf) {
  const double v = (double)acc * (double)hexkl_cvt_hf_to_f32(scale_hf) / 512.0;
  double fl = (double)(int64_t)v;
  if (fl > v) {
    fl -= 1.0;
  }
  return (uint8_t)((int64_t)fl & 0xff);
}

#if defined(__hexagon__)

#include <hexagon_protos.h>
#include <hexagon_types.h>

/**
 * @brief Two byte planes of one tile -> one int16 tile, same element
 *        order: dst[p] = lo[p] | hi[p] << 8. 16 vshuff for 2048 elements.
 *
 * @param lo, hi  HEXKL_CVT_PLANE_BYTES each, 128-byte aligned, VTCM
 * @param dst     2 * HEXKL_CVT_PLANE_BYTES, 128-byte aligned
 */
static inline void hexkl_cvt_zip_i16(const uint8_t *lo, const uint8_t *hi,
                                     int16_t *dst) {
  const HVX_Vector *vl = (const HVX_Vector *)lo;
  const HVX_Vector *vh = (const HVX_Vector *)hi;
  HVX_Vector *vd = (HVX_Vector *)dst;
  for (uint32_t i = 0; i < HEXKL_CVT_PLANE_BYTES / 128u; ++i) {
    // Byte interleave: element 2j from the second operand, 2j+1 from the
    // first, so (hi, lo) gives little-endian int16.
    const HVX_VectorPair w = Q6_W_vshuff_VVR(vh[i], vl[i], -1);
    vd[2u * i] = Q6_V_lo_W(w);
    vd[2u * i + 1u] = Q6_V_hi_W(w);
  }
}

/**
 * @brief One convert pass: loads the bias block, writes the current
 *        accumulator as 64x32 uint8 to @a dst, keeps the accumulator.
 *
 * Requires the HMX lock and a multiply or clear issued before it.
 *
 * @param block  HEXKL_CVT_BLOCK_BYTES aligned, in VTCM
 * @param dst    HEXKL_CVT_PLANE_ALIGN aligned, HEXKL_CVT_PLANE_BYTES, VTCM
 */
static inline void hexkl_cvt_issue(const void *block, void *dst) {
  __asm__ __volatile__("{ bias = mxmem(%0) }\n\t"
                       "{ mxmem(%1,%2):after:retain:cm.ub = acc }"
                       :
                       : "r"(block), "r"(dst), "r"(0)
                       : "memory");
}

#else

static inline void hexkl_cvt_zip_i16(const uint8_t *lo, const uint8_t *hi,
                                     int16_t *dst) {
  for (uint32_t p = 0; p < HEXKL_CVT_PLANE_BYTES; ++p) {
    dst[p] = (int16_t)(uint16_t)(lo[p] | ((uint16_t)hi[p] << 8));
  }
}

static inline void hexkl_cvt_issue(const void *block, void *dst) {
  (void)block;
  (void)dst;
}

#endif /* __hexagon__ */

#endif /* __NNTRAINER_HEXKL_CVT_H__ */
