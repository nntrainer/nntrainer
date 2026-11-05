// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   fc_m1_epilogue_host_check.c
 * @date   12 Oct 2026
 * @brief  Decode M=1 packing and three/four-column epilogue parity on real HVX
 *         sources with host instruction emulation
 * @author NNTrainer contributors
 * @bug    No known bugs
 */

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_dequant_i32.h"
#include "hvx_quant_u8.h"

enum { GUARD = 128, DQ_N = 128 };
static uint32_t rng = 17u;
static unsigned failures;

static uint32_t next_u32(void) {
  rng = rng * 1664525u + 1013904223u;
  return rng;
}

static void check_equal(const void *got, const void *want, size_t bytes,
                        const char *what, uint32_t dim, uint32_t mode,
                        uint32_t offset) {
  if (memcmp(got, want, bytes) != 0) {
    fprintf(stderr, "FAIL: %s dim=%u mode=%u offset=%u\n", what, dim, mode,
            offset);
    ++failures;
  }
}

static void *alloc_bytes(size_t bytes) {
  void *p = malloc(bytes);
  if (!p) {
    fprintf(stderr, "allocation failed\n");
    exit(2);
  }
  return p;
}

static void check_pack(uint32_t k, uint32_t mode, uint32_t offset) {
  const size_t span = (size_t)k * 64u;
  const size_t bytes = GUARD + offset + span + GUARD;
  uint8_t *got = alloc_bytes(bytes), *old = alloc_bytes(bytes);
  uint8_t *spec = alloc_bytes(bytes);
  float *storage = alloc_bytes((4u * (size_t)k + 64u) * sizeof(float));
  float *x = storage + offset / sizeof(float);
  float scale[4] = {1.f, 1.f, 1.f, 1.f};
  int32_t zp[4] = {0, 0, 0, 0};
  const uint32_t map[4] = {0u, 0u, 0u, 0u};
  for (uint32_t j = 0; j < k; ++j) {
    switch (mode) {
    case 0: /* min/max parameters, signs and many different fractions */
      x[j] = (float)((int32_t)(next_u32() % 200001u) - 100000) / 193.f;
      break;
    case 1:
      x[j] = 0.f;
      break;
    case 2:
      x[j] = -3.75f;
      break;
    case 3: /* exact half-integers exercise round-to-nearest-even */
      x[j] = ((float)((int32_t)(j % 641u) - 320) + 0.5f) * 0.125f;
      break;
    default: /* representable neighbors on both sides of rounding ties */
      x[j] = nextafterf(((float)((int32_t)(j % 513u) - 256) + 0.5f) * 0.125f,
                        (j & 1u) ? INFINITY : -INFINITY);
      break;
    }
  }
  if (mode < 3u) {
    hvx_quant_rows_u8_params(x, 1u, 1u, k, scale, zp, NULL);
  } else {
    scale[0] = 0.125f;
    zp[0] = 127;
  }
  for (uint32_t r = 1; r < 4u; ++r) {
    scale[r] = scale[0];
    zp[r] = zp[0];
  }
  memset(got, 0xa5, bytes);
  memset(old, 0xa5, bytes);
  memset(spec, 0xa5, bytes);
  hvx_quant_pack_u8_ah_m1_repeat(x, k, scale[0], zp[0], got + GUARD + offset);
  hvx_quant_pack_u8_ah_rows(x, map, 0u, 4u, k, scale, zp, old + GUARD + offset);
  volatile float inv = 1.f / scale[0];
  for (uint32_t kt = 0; kt < k / 32u; ++kt) {
    for (uint32_t j = 0; j < 32u; ++j) {
      volatile float prod = x[kt * 32u + j] * inv;
      int32_t q = (int32_t)nearbyintf(prod) + zp[0];
      q = q < 0 ? 0 : q > 255 ? 255 : q;
      for (uint32_t r = 0; r < 4u; ++r)
        spec[GUARD + offset + (size_t)kt * 2048u + r * 32u + j] = (uint8_t)q;
    }
  }
  check_equal(got, old, bytes, "pack vs four-row HVX", k, mode, offset);
  check_equal(got, spec, bytes, "pack bytes, padding and guards", k, mode,
              offset);
  /** The dense FFN's former pack zeroed three padded float rows. Its live
   * row must also equal the repeat pack; padded rows are never consumed. */
  memset(x + k, 0, (size_t)3u * k * sizeof(float));
  for (uint32_t r = 1u; r < 4u; ++r) {
    scale[r] = 1.f;
    zp[r] = 0;
  }
  hvx_quant_pack_u8_ah_rows(x, NULL, 0u, 4u, k, scale, zp,
                            old + GUARD + offset);
  for (uint32_t kt = 0u; kt < k / 32u; ++kt)
    check_equal(got + GUARD + offset + (size_t)kt * 2048u,
                old + GUARD + offset + (size_t)kt * 2048u, 32u,
                "live row vs zero-padded HVX", k, mode, offset);
  free(storage);
  free(spec);
  free(old);
  free(got);
}

static float dequant_spec(int32_t a, float scale, int32_t zp, int32_t cs,
                          float ws, float bias) {
  /** Each volatile is one Vsf operation. Integer correction before
     conversion is a different formula once products exceed 2^24. */
  volatile float correction = (float)zp * (float)cs;
  volatile float centered = (float)a - correction;
  volatile float scaled = centered * scale;
  volatile float weighted = scaled * ws;
  return weighted + bias;
}

static void check_dequant(uint32_t mode, uint32_t offset, uint32_t tiles) {
  const uint32_t n = tiles * 32u;
  const size_t bytes = GUARD + offset + n * sizeof(float) + GUARD;
  uint8_t *got = alloc_bytes(bytes), *old = alloc_bytes(bytes);
  uint8_t *spec = alloc_bytes(bytes);
  int32_t acc_buf[DQ_N + 32], cs_buf[DQ_N + 32];
  float ws_buf[DQ_N + 32], bias_buf[DQ_N + 32];
  int32_t *acc = acc_buf + offset / 4u, *cs = cs_buf + offset / 4u;
  float *ws = ws_buf + offset / 4u, *bias = bias_buf + offset / 4u;
  float as = mode == 3u ? FLT_MIN : mode == 4u ? 0.f : 0.013713f;
  int32_t az = mode == 0u ? 0 : mode == 1u ? 127 : 255;
  for (uint32_t j = 0; j < n; ++j) {
    acc[j] = (int32_t)(next_u32() % 33423361u) - 16711680;
    cs[j] = (int32_t)(next_u32() % 131073u) - 65536;
    ws[j] = (float)((int32_t)(next_u32() % 8193u) - 4096) / 7919.f;
    bias[j] = (float)((int32_t)(next_u32() % 1025u) - 512) / 1009.f;
    if (mode == 2u) {
      /* Cancellation, signed zeros and offsets neighboring exact zero. */
      acc[j] = az * cs[j] + (int32_t)(j % 3u) - 1;
      bias[j] = (j & 1u) ? -0.f : 0.f;
    } else if (mode == 3u) {
      acc[j] = (int32_t)(j % 3u) - 1;
      cs[j] = 0;
      ws[j] = (j & 1u) ? -0.25f : 0.25f;
      bias[j] = 0.f;
    } else if (mode == 5u) {
      /* Conversion rounding is visible beyond the exact-int float range. */
      acc[j] = (int32_t)(0x01000001u + j * 199u);
      cs[j] = 70001 + (int32_t)j;
    }
  }
  memset(got, 0x5a, bytes);
  memset(old, 0x5a, bytes);
  memset(spec, 0x5a, bytes);
  float *gout = (float *)(got + GUARD + offset);
  float *oout = (float *)(old + GUARD + offset);
  float *sout = (float *)(spec + GUARD + offset);
  if (tiles == 3u)
    hvx_dequant_acc_3tiles_m1_f32(acc, as, az, cs, ws, bias, gout);
  else
    hvx_dequant_acc_4tiles_m1_f32(acc, as, az, cs, ws, bias, gout);
  for (uint32_t j = 0; j < tiles; ++j)
    hvx_dequant_acc_tile_to_f32(acc + j * 32u, 32u, 1u, &as, &az, cs + j * 32u,
                                ws + j * 32u, bias + j * 32u, oout + j * 32u,
                                DQ_N, 0);
  for (uint32_t j = 0; j < n; ++j)
    sout[j] = dequant_spec(acc[j], as, az, cs[j], ws[j], bias[j]);
  check_equal(got, old, bytes, "dequant vs separate HVX calls", n, mode,
              offset);
  check_equal(got, spec, bytes, "dequant values and guards", n, mode, offset);
  free(spec);
  free(old);
  free(got);
}

int main(void) {
  static const uint32_t shapes[] = {32u,  64u,   96u,   128u,
                                    704u, 2816u, 4096u, 8192u};
  static const uint32_t offsets[] = {0u, 4u, 28u, 124u};
  unsigned packs = 0, epilogues = 0;
  if (fesetround(FE_TONEAREST) != 0)
    return 2;
  for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); ++oi) {
    for (size_t ki = 0; ki < sizeof(shapes) / sizeof(shapes[0]); ++ki) {
      for (uint32_t mode = 0; mode < 5u; ++mode) {
        check_pack(shapes[ki], mode, offsets[oi]);
        ++packs;
      }
    }
    for (uint32_t mode = 0; mode < 6u; ++mode) {
      check_dequant(mode, offsets[oi], 3u);
      check_dequant(mode, offsets[oi], 4u);
      epilogues += 2u;
    }
  }
  printf("FC M1 EPILOGUE: pack_cases=%u dequant_cases=%u failures=%u\n", packs,
         epilogues, failures);
  return failures != 0u;
}
