// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_micro_standin.c
 * @date   27 Sep 2026
 * @brief  Host stand-in for HexKL's micro API around the tile calls
 *         (standin/hvx_scalar.c): VTCM, the HMX lock, the layout bake
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * "VTCM" is one static 8 MiB buffer, v79's size, so hexkl_mm_u8i4_moe_layout
 * and the DMA plan compute the device's numbers. The int8 and fp16 paths
 * are refused (no model path reaches them on the NPU configuration).
 */

#include "hexkl_micro.h"

#include <AEEStdErr.h>
#include <string.h>

#define VTCM_BYTES (8u << 20)
static uint8_t g_vtcm[VTCM_BYTES] __attribute__((aligned(2048)));
/** [#132 Part B E3] One session holds the VTCM and the HMX, as on the
 *  device (#178: a second session's hw_init is refused and it opens lite,
 *  with no VTCM); the full open's close gives them back (hmx_unlock). */
static int g_hw_held;

int hexkl_micro_hw_init(uint8_t **vtcm_base, uint32_t *vtcm_size,
                        uint32_t *hmx_fp16_rate) {
  if (g_hw_held)
    return AEE_EFAILED;
  g_hw_held = 1;
  *vtcm_base = g_vtcm;
  *vtcm_size = VTCM_BYTES;
  if (hmx_fp16_rate)
    *hmx_fp16_rate = 0;
  return AEE_SUCCESS;
}
int hexkl_micro_hmx_lock(void) { return AEE_SUCCESS; }
int hexkl_micro_hmx_unlock(void) {
  g_hw_held = 0;
  return AEE_SUCCESS;
}
uint32_t hexkl_micro_hmx_config_size(void) { return 2048u; }
int hexkl_micro_hmx_setup_acc_read_int32(uint8_t *b, uint32_t cfg) {
  (void)b;
  (void)cfg;
  return AEE_SUCCESS;
}

/* The bake: tile (tr, tc) of a row-major int4-in-int8 matrix of N columns
   into the 512-byte WH tile at b + off. The slot formula is
   htp_wh_layout.h's whSlot (the quantizer's, C++), restated here in C; it
   is the inverse of standin/hvx_scalar.c's wh_value, which every host
   reader of a tile goes through. */
int hexkl_micro_hmx_rm_to_wh_i4(uint8_t *b, uint32_t off, const int8_t *rm,
                                uint32_t tr, uint32_t tc, uint32_t N) {
  uint8_t *tile = b + off;
  memset(tile, 0, 512u);
  for (uint32_t r = 0; r < 32u; ++r) {
    const int8_t *row = rm + (size_t)(tr * 32u + r) * N + tc * 32u;
    for (uint32_t c = 0; c < 32u; ++c) {
      const uint32_t sl =
        (r / 8u) * 256u + c * 8u + (r % 4u) * 2u + ((r / 4u) % 2u);
      tile[sl / 2u] |= (uint8_t)((row[c] & 0x0F) << (4u * (sl % 2u)));
    }
  }
  return AEE_SUCCESS;
}
int hexkl_micro_hmx_rm_to_wh_i8(uint8_t *b, uint32_t off, const int8_t *rm,
                                uint32_t tr, uint32_t tc, uint32_t N) {
  (void)b, (void)off, (void)rm, (void)tr, (void)tc, (void)N;
  return AEE_EUNSUPPORTED;
}
int hexkl_micro_hmx_mm_u8i8(uint8_t *b, uint32_t a, uint32_t w) {
  (void)b, (void)a, (void)w;
  return AEE_EUNSUPPORTED;
}

/* The accumulator tile (64 x 32 int32, row stride 32 -- what
   hvx_scalar.c's acc_read writes, so hexkl_acc_tile.c's ramp probe finds
   base 0, stride 32) into output rows tile_row*64.. and columns
   tile_col*32.. of an output_rows x output_cols matrix. */
int hexkl_micro_hmx_copy_32b_to_submatrix(uint8_t *b, uint32_t off,
                                          int32_t *dst, uint32_t rb,
                                          uint32_t nt, uint32_t m_pad,
                                          uint32_t N) {
  const int32_t *tile = (const int32_t *)(b + off);
  for (uint32_t r = 0; r < 64u; ++r) {
    const uint32_t orow = rb * 64u + r;
    if (orow >= m_pad)
      break;
    for (uint32_t c = 0; c < 32u; ++c) {
      const uint32_t ocol = nt * 32u + c;
      if (ocol < N)
        dst[(size_t)orow * N + ocol] = tile[r * 32u + c];
    }
  }
  return AEE_SUCCESS;
}
