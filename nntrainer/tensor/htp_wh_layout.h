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

/**
 * @brief whPack's inverse: WH bytes back to one sign-extended int4 per int8,
 *        K rows of N, row stride N.
 *
 * A host-side reader for the weights the model file keeps in tile order, so
 * that a reference path (NNTR_MOE_DIFF) can dequantize the bytes the DSP is
 * running without the DSP. Same K, N contract as whPack, and
 * nntrainer_cpu_backend_standalone.wh_pack_unpacks_like_the_load_path is the
 * round trip.
 */
inline void whUnpack(const uint8_t *wh, uint32_t K, uint32_t N, int8_t *rm) {
  const uint32_t k_tiles = K / WH_TILE, n_tiles = N / WH_TILE;
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
      const uint8_t *tile = wh + ((size_t)kt * n_tiles + nt) * WH_TILE_BYTES;
      for (uint32_t r = 0; r < WH_TILE; ++r) {
        int8_t *row = rm + (size_t)(kt * WH_TILE + r) * N + nt * WH_TILE;
        for (uint32_t c = 0; c < WH_TILE; ++c) {
          const uint32_t sl = whSlot(r, c);
          const uint8_t nib = (tile[sl / 2] >> (4 * (sl % 2))) & 0x0Fu;
          row[c] = static_cast<int8_t>(nib > 7u ? (int)nib - 16 : (int)nib);
        }
      }
    }
  }
}

/**
 * @brief The FC WH sidecar (#225): the Q4_0 model's FC weights quantized a
 *        second time, from the same f32, as QS4CX_WH images, so the HTP
 *        prefill reads them into the arena instead of re-quantizing the
 *        Q4_0 bytes at load.
 *
 * File: FCWH_MAGIC, uint32 version (FCWH_VERSION), uint32 count, count
 * FcWhEntry, zero padding to 4 KiB, then the images back to back, each a
 * QS4CX_WH tensor as nntr_quantize_stream writes it: whBytes(K, N) of
 * nibbles, N f32 scales, N f32 column sums. Little-endian, as written.
 * The packer writes nntr_config.json's fc_wh_format = FCWH_FORMAT; a
 * change to any of this bumps both.
 */
constexpr char FCWH_MAGIC[8] = {'N', 'N', 'T', 'R', 'F', 'C', 'W', 'H'};
constexpr uint32_t FCWH_VERSION = 1u;
constexpr const char *FCWH_FORMAT = "QS4CX_WH/1";

/** @brief One sidecar index entry. */
struct FcWhEntry {
  char name[64];   /**< the packer's tensor name, NUL-padded */
  uint32_t K, N;   /**< the weight as the matmul sees it, [K x N] */
  uint64_t key;    /**< fcWhKey of the weight's Q4_0 bytes in the main file */
  uint64_t q4_off; /**< where those Q4_0 bytes start in the main file */
  uint64_t off;    /**< where the image starts in this file */
  uint64_t bytes;  /**< the image's length */
};
static_assert(sizeof(FcWhEntry) == 104, "FcWhEntry is a file layout");

/** @brief Byte offset of the first image: the index rounded up to 4 KiB. */
inline uint64_t fcWhHeaderBytes(uint32_t count) {
  const uint64_t raw = 16u + static_cast<uint64_t>(count) * sizeof(FcWhEntry);
  return (raw + 4095u) & ~uint64_t(4095u);
}

/**
 * @brief Names a Q4_0 weight by its bytes: FNV-1a 64 over the length and
 *        the first and last 4 KiB.
 *
 * The loader finds a weight's image by this, with no layer or tensor names
 * to agree on between the packer's walk and the four graph forms that hold
 * an FC (fully_connected, qkv_layer, conv_block, dense_ffn), and a main
 * file re-quantized without its sidecar misses every entry instead of
 * loading another weight's image.
 * ponytail: 8 KiB of a weight, not all of it -- enough to tell 66 weights
 * and two quantizations apart; hashing all 255 MiB would cost the load
 * ~0.1 s for no case this misses.
 */
inline uint64_t fcWhKey(const void *q4_0, size_t len) {
  const uint8_t *p = static_cast<const uint8_t *>(q4_0);
  uint64_t h = 1469598103934665603ull;
  auto mix = [&h](uint8_t b) { h = (h ^ b) * 1099511628211ull; };
  for (int i = 0; i < 8; ++i)
    mix(static_cast<uint8_t>(static_cast<uint64_t>(len) >> (8 * i)));
  const size_t head = len < 4096u ? len : 4096u;
  for (size_t i = 0; i < head; ++i)
    mix(p[i]);
  for (size_t i = len - (len < 4096u ? len : 4096u); i < len; ++i)
    mix(p[i]);
  return h;
}

/**
 * @brief The whole pages of [src, src + len) -- what the arena copy can hand
 *        back to the OS once a weight's bytes are in the arena.
 *
 * This model's expert weights are 3.9 GB living in ONE contiguous allocation
 * (TensorPool gives every weight a slice of a single aligned_alloc, so
 * Tensor::deallocate() drops a pointer and frees nothing), and the arena
 * needs its own 3.9 GB copy of them. Both at once is more memory than the
 * device has. Dropping each weight's pages right after its copy keeps the
 * peak at one model instead of two. The pointer stays valid and mapped --
 * only the physical pages go, and a read would see zeros -- so nothing that
 * still holds it dangles, which is also why the handle cache may keep using
 * it as a key.
 *
 * Rounds INWARD. A weight's first and last partial page are shared with its
 * neighbours in the pool, and dropping a neighbour's live bytes would be a
 * silently wrong matmul rather than a failure. That keeps at most one page at
 * each end, against a 1.8 MB smallest weight here.
 *
 * @param page_size    OS page size, a power of two; a parameter rather than a
 *                     call so a host test can check the arithmetic
 * @param[out] out_len 0 when the range covers no whole page, in which case
 *                     *out_begin is unspecified and the caller does nothing
 */
inline void whSourcePageRange(const void *src, size_t len, size_t page_size,
                              uintptr_t *out_begin, size_t *out_len) {
  const uintptr_t mask = static_cast<uintptr_t>(page_size) - 1u;
  const uintptr_t addr = reinterpret_cast<uintptr_t>(src);
  const uintptr_t begin = (addr + mask) & ~mask;
  const uintptr_t end = (addr + len) & ~mask;
  *out_begin = begin;
  *out_len = end > begin ? static_cast<size_t>(end - begin) : 0u;
}

} // namespace nntrainer

#endif // __NNTRAINER_HTP_WH_LAYOUT_H__
