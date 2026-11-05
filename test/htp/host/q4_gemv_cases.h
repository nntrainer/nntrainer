// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   q4_gemv_cases.h
 * @date   29 Sep 2026
 * @brief  The CPU-exact Q4_0 FC's test inputs (#132 PR 2), shared by the
 *         host check q4_gemv_host_check.c and the device gtest HvxFcQ4.*
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Weights: random block_q4_0 with special columns (negative, f16-subnormal
 * and zero d_w, power-of-two d_w for exact sums, a block that cancels the
 * one before). Rows: block magnitudes 2^-20 .. 2^20, zero and tiny blocks,
 * and quantizer-rounding blocks (exact ties, and x where amax / 127 and
 * amax * (1/127) round q differently). Deterministic from the seed.
 */

#ifndef __Q4_GEMV_CASES_H__
#define __Q4_GEMV_CASES_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "q4_gemv_cpu_det.h"

static uint64_t g_seed = 0x132132132ull;
static inline uint32_t rnd(void) {
  g_seed = g_seed * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_seed >> 32);
}
static inline float frand(float lo, float hi) {
  return lo + (hi - lo) * ((float)(rnd() >> 8) / 16777216.0f);
}

/* ---- inputs -------------------------------------------------------------- */

/** @brief A random f16 scale around 2^e (e in [lo, lo + span)), sign
 *         random if @a neg. */
static inline uint16_t rand_h(int lo, int span, int neg) {
  const uint32_t e = (uint32_t)(lo + (int)(rnd() % (uint32_t)span));
  return (uint16_t)(((neg && (rnd() & 1u)) ? 0x8000u : 0u) | (e << 10) |
                    (rnd() & 1023u));
}

/** @brief Canonical block_q4_0 weights [N][K/32]. Columns 0..3 of each 32
 *  are special: 0 all-negative d_w, 1 f16-subnormal d_w, 2 d_w = 0 in odd
 *  blocks, 3 power-of-two d_w (exact sums, so ties are frequent), 4 block
 *  1 the negation of block 0 (w - 8 negated, same d_w): with make_row's
 *  kind 1, which repeats block 0's activations in block 1, the chain
 *  cancels to exactly 0. */
static inline void make_weights(uint8_t *w, uint32_t K, uint32_t N) {
  const uint32_t nb = K / Q4_CPU_QK;
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t b = 0; b < nb; ++b) {
      uint8_t *blk = w + ((size_t)n * nb + b) * Q4_CPU_BLOCK_BYTES;
      uint16_t d = rand_h(4, 10, 1);
      switch (n % 32u) {
      case 0:
        d |= 0x8000u;
        break;
      case 1:
        d = (uint16_t)((rnd() & 0x8000u) | (rnd() & 1023u));
        break;
      case 2:
        d = (b & 1u) ? 0u : d;
        break;
      case 3:
        d = (uint16_t)((rnd() & 0x8000u) | ((6u + rnd() % 12u) << 10));
        break;
      default:
        break;
      }
      blk[0] = (uint8_t)(d & 0xff);
      blk[1] = (uint8_t)(d >> 8);
      for (uint32_t j = 0; j < 16u; ++j) {
        blk[2 + j] = (uint8_t)(rnd() | 0x11u); /* no zero nibble */
      }
      if (n % 32u == 4u && b == 1u) {
        const uint8_t *b0 = blk - Q4_CPU_BLOCK_BYTES;
        blk[0] = b0[0];
        blk[1] = b0[1];
        for (uint32_t j = 0; j < 16u; ++j) {
          blk[2 + j] = (uint8_t)(((16u - (b0[2 + j] & 15u)) & 15u) |
                                 ((16u - (b0[2 + j] >> 4)) << 4));
        }
      }
    }
  }
}

/** @brief A quantizer-rounding block: element 0 is amax; the others sit
 *  where RN(x id) is decided by the last bit of id. Even @a b: amax =
 *  127 2^e, so d = 2^e and x = (k + 1/2) 2^e are exact ties (ties-even vs
 *  ties-away). Odd @a b: an amax whose amax / 127 and amax * (1/127)
 *  differ, and x a few ulps around (k + 1/2) d where the two ids round
 *  x id differently (found by search). */
static inline void quant_edge_block(float *x, uint32_t b, int e) {
  if ((b & 1u) == 0u) {
    x[0] = ldexpf(127.0f, e);
    for (uint32_t j = 1; j < Q4_CPU_QK; ++j) {
      x[j] = ldexpf((float)((int)(rnd() % 252u) - 126) + 0.5f, e);
    }
    return;
  }
  /* the spec's exact helpers, so -ffast-math cannot fold the two ids */
  const float r127 = cpu_det_div_rn(1.0f, 127.0f);
  float amax = 0.0f, d1 = 0.0f, d2 = 0.0f;
  for (int t = 0; t < 1000 && d1 == d2; ++t) {
    amax = ldexpf(frand(64.0f, 127.0f), e);
    d1 = cpu_det_div_rn(amax, 127.0f);
    d2 = cpu_det_mul(amax, r127);
  }
  const float id1 = cpu_det_div_rn(1.0f, d1), id2 = cpu_det_div_rn(1.0f, d2);
  x[0] = amax;
  uint32_t j = 1;
  for (int t = 0; j < Q4_CPU_QK && t < 100000; ++t) {
    const float c = cpu_det_mul((float)((int)(rnd() % 250u) - 125) + 0.5f, d1);
    const float v = cpu_det_float(cpu_det_bits(c) + (rnd() % 9u) - 4u);
    if (cpu_det_fcvtns(cpu_det_mul(v, id1)) !=
          cpu_det_fcvtns(cpu_det_mul(v, id2)) &&
        fabsf(v) < amax) {
      x[j++] = v;
    }
  }
  for (; j < Q4_CPU_QK; ++j) {
    x[j] = 0.0f;
  }
}

/** @brief One activation row: block b at magnitude 2^(e_b), e_b cycling
 *  -20..20; kind 1 zero blocks every 5th; kind 2 tiny (d_a f16-subnormal
 *  or zero) blocks every 3rd; kind 3 quant_edge_block everywhere. */
static inline void make_row(float *x, uint32_t K, int kind, int row) {
  for (uint32_t b = 0; b < K / Q4_CPU_QK; ++b) {
    const int e = (int)((b + (uint32_t)row * 7u) % 41u) - 20;
    if (kind == 3) {
      /* d = 2^e stays below f16's 65504 (the domain) */
      quant_edge_block(x + b * Q4_CPU_QK, b, e < 12 ? e : e - 30);
      continue;
    }
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      float v = ldexpf(frand(-1.0f, 1.0f), e);
      if (kind == 1 && b % 5u == 4u) {
        v = 0.0f;
      } else if (kind == 2 && b % 3u == 0u) {
        v = ldexpf(frand(-1.0f, 1.0f), -30 - (int)(b % 12u));
      }
      x[b * Q4_CPU_QK + j] = v;
    }
    if (kind == 1 && b == 1u) {
      memcpy(x + Q4_CPU_QK, x, Q4_CPU_QK * sizeof(float));
    }
  }
}

#endif /* __Q4_GEMV_CASES_H__ */
