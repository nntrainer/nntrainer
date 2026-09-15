// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_wh_layout.h
 * @date   15 Sep 2026
 * @brief  The WH weight tile layout, as a formula rather than a device call
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_micro_hmx_rm_to_wh_i4 rearranges an int4 weight into the 32x32 tiles
 * the HMX unit reads, and it runs on the DSP. That put the conversion on the
 * model load path, where it is the larger half of a registration that costs
 * 48.2% of prefill (doc 46 section 20), and left the baked bytes on a DSP
 * heap that tops out at 1.89 GB against the 3.9 the model needs.
 *
 * It does not have to be there. A tile is 32x32 i4 = 1024 nibbles = 512
 * bytes = WEIGHT_TILE_BYTES_U8I4 exactly, so there is no room in the output
 * for anything but the values and the transform can only be a permutation of
 * nibbles. Reading that permutation off the device gives:
 *
 *     byte(r,c) = (r/8)*128 + c*4 + (r%4)       nibble = (r/4)%2
 *
 * One byte holds the two ROWS four apart in the same column -- two k values
 * per byte, which is what the reduction wants -- and tiles are k-major,
 * kt*(N/32) + nt, where hexkl_bake_u8i4_worker places each at t*512.
 *
 * Device-verified byte for byte against weight_bake_export at 2048x3584, the
 * model's largest weight, and at a non-square 64x128 so the tile order is
 * checked too (doc 46 section 35.3b). unittest_hvx_mm_u8i4's
 * WhPackReferenceMatchesDspBake is that check and has to keep passing: this
 * header is a copy of a hardware layout, and nothing but the device can say
 * it is still right.
 *
 * Not to be confused with sdkl_cpu_i4_rm_to_i4_wh, whose tile is this one
 * TRANSPOSED -- whSlot(c, r). The two APIs read their source with opposite
 * conventions, of a piece with hexkl_macro_i4_rm_to_i4_wh taking
 * (n_col, n_inner) where sdkl_cpu_i4_rm_to_i4_wh is documented
 * (wt_rows, wt_cols). Deriving this from the library instead of from the
 * kernel under test cost two device rounds, because a square probe hides
 * both the transpose and the argument order.
 */

#ifndef __NNTRAINER_HTP_WH_LAYOUT_H__
#define __NNTRAINER_HTP_WH_LAYOUT_H__

#include <cstddef>
#include <cstdint>

namespace nntrainer {

/** @brief Side of one WH tile, in weights. HEXKL_HMX_INT8_BLOCK_N_INNER and
 *  _N_COL, both 32; spelled here so this header needs no DSP headers. */
constexpr uint32_t WH_TILE = 32u;
/** @brief Bytes per tile: WH_TILE*WH_TILE i4 values, two per byte. */
constexpr uint32_t WH_TILE_BYTES = WH_TILE * WH_TILE / 2u;

/** @brief Bytes a K x N weight occupies in WH layout. */
inline size_t whBytes(uint32_t K, uint32_t N) {
  return static_cast<size_t>(K / WH_TILE) * (N / WH_TILE) * WH_TILE_BYTES;
}

/** @brief Which nibble of a tile's 512 bytes element (r, c) of that tile
 *  lands in. Slot s is the low half of byte s/2 for even s, the high half
 *  for odd. */
inline uint32_t whSlot(uint32_t r, uint32_t c) {
  return (r / 8u) * 256u + c * 8u + (r % 4u) * 2u + ((r / 4u) % 2u);
}

/**
 * @brief Packs a row-major i4 weight into WH layout.
 *
 * @param rm  one sign-extended int4 per int8, K rows of N, row stride N --
 *            what htp_qs4cx_from_packed produces and what the DSP bake takes
 * @param out whBytes(K, N) bytes, zeroed by this function
 *
 * K and N must both be multiples of WH_TILE; the caller checks, because the
 * shapes come from a model config and a partial tile here would be a
 * silently wrong matmul rather than a failure.
 */
inline void whPack(const int8_t *rm, uint32_t K, uint32_t N, uint8_t *out) {
  const uint32_t k_tiles = K / WH_TILE, n_tiles = N / WH_TILE;
  for (size_t i = 0, n = whBytes(K, N); i < n; ++i) {
    out[i] = 0;
  }
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
      uint8_t *tile = out + ((size_t)kt * n_tiles + nt) * WH_TILE_BYTES;
      for (uint32_t r = 0; r < WH_TILE; ++r) {
        const int8_t *row = rm + (size_t)(kt * WH_TILE + r) * N + nt * WH_TILE;
        for (uint32_t c = 0; c < WH_TILE; ++c) {
          const uint32_t sl = whSlot(r, c);
          tile[sl / 2] |=
            static_cast<uint8_t>((row[c] & 0x0F) << (4 * (sl % 2)));
        }
      }
    }
  }
}

} // namespace nntrainer

#endif // __NNTRAINER_HTP_WH_LAYOUT_H__
