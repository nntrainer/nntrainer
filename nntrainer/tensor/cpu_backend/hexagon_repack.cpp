// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_repack.cpp
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  See hexagon_repack.h.
 */

#include <hexagon_repack.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nntrainer {

namespace {

constexpr unsigned int QK4_0 = 32;
constexpr size_t Q4_0_BLOCK_BYTES = 2 + QK4_0 / 2; // fp16 delta + 16 nibbles
constexpr unsigned int QK_Q4X4X2 = 256;             // 8 x QK4_0
constexpr unsigned int BLOCKS_PER_TILE = QK_Q4X4X2 / QK4_0; // 8
constexpr size_t TILE_QUANT_BYTES = QK_Q4X4X2 / 2;  // 128
constexpr size_t TILE_SCALE_BYTES = BLOCKS_PER_TILE * sizeof(uint16_t); // 16

// unpack one q4_0 block's 16 packed-nibble bytes into 32 individual
// 4-bit values, written at qs[bi*32 .. bi*32+31] (matches ggml-hexagon's
// unpack_q4_0_quants / block_q4_0 nibble convention: byte i holds element i
// in the low nibble and element i+16 in the high nibble).
inline void unpack_block_quants(uint8_t *qs, const uint8_t *block_nibbles,
                                 unsigned int bi) {
  for (unsigned int i = 0; i < QK4_0 / 2; ++i) {
    qs[bi * QK4_0 + i] = block_nibbles[i] & 0x0F;
    qs[bi * QK4_0 + i + QK4_0 / 2] = block_nibbles[i] >> 4;
  }
}

// inverse of unpack_block_quants
inline void pack_block_quants(uint8_t *block_nibbles_out, const uint8_t *qs,
                               unsigned int bi) {
  for (unsigned int i = 0; i < QK4_0 / 2; ++i) {
    block_nibbles_out[i] =
      qs[bi * QK4_0 + i] | (qs[bi * QK4_0 + i + QK4_0 / 2] << 4);
  }
}

void check_row_len(unsigned int N) {
  if (N % QK_Q4X4X2 != 0) {
    throw std::invalid_argument(
      "repack_q4_0_to_htp_q4x4x2: N must be divisible by 256 (got " +
      std::to_string(N) + "); non-multiple-of-256 rows are not supported yet");
  }
}

} // namespace

void repack_q4_0_to_htp_q4x4x2(void *dst, const void *src, size_t data_size,
                                unsigned int M, unsigned int N) {
  check_row_len(N);

  const unsigned int n_tiles = N / QK_Q4X4X2;
  const size_t row_bytes = (size_t)(N / QK4_0) * Q4_0_BLOCK_BYTES;
  const size_t out_row_bytes = n_tiles * (TILE_QUANT_BYTES + TILE_SCALE_BYTES);

  (void)data_size;

  const uint8_t *src_bytes = static_cast<const uint8_t *>(src);
  uint8_t *dst_bytes = static_cast<uint8_t *>(dst);

  std::vector<uint8_t> qs(QK_Q4X4X2);

  for (unsigned int m = 0; m < M; ++m) {
    const uint8_t *src_row = src_bytes + (size_t)m * row_bytes;
    uint8_t *dst_row = dst_bytes + (size_t)m * out_row_bytes;

    uint8_t *y_q = dst_row;
    uint8_t *y_d = dst_row + n_tiles * TILE_QUANT_BYTES;

    for (unsigned int t = 0; t < n_tiles; ++t) {
      const uint8_t *blocks = src_row + (size_t)t * BLOCKS_PER_TILE * Q4_0_BLOCK_BYTES;

      for (unsigned int b = 0; b < BLOCKS_PER_TILE; ++b) {
        const uint8_t *block = blocks + (size_t)b * Q4_0_BLOCK_BYTES;
        unpack_block_quants(qs.data(), block + 2 /* skip fp16 delta */, b);
      }

      uint8_t *q = y_q + (size_t)t * TILE_QUANT_BYTES;
      for (unsigned int j = 0; j < TILE_QUANT_BYTES; ++j) {
        q[j] = (qs[j + 128] << 4) | qs[j + 0];
      }

      uint16_t *d = reinterpret_cast<uint16_t *>(y_d + (size_t)t * TILE_SCALE_BYTES);
      for (unsigned int b = 0; b < BLOCKS_PER_TILE; ++b) {
        const uint8_t *block = blocks + (size_t)b * Q4_0_BLOCK_BYTES;
        uint16_t delta;
        std::memcpy(&delta, block, sizeof(uint16_t));
        d[b] = delta;
      }
    }
  }
}

void unpack_htp_q4x4x2_to_q4_0(void *dst, const void *src, size_t data_size,
                                unsigned int M, unsigned int N) {
  check_row_len(N);

  const unsigned int n_tiles = N / QK_Q4X4X2;
  const size_t out_row_bytes = (size_t)(N / QK4_0) * Q4_0_BLOCK_BYTES;
  const size_t in_row_bytes = n_tiles * (TILE_QUANT_BYTES + TILE_SCALE_BYTES);

  (void)data_size;

  const uint8_t *src_bytes = static_cast<const uint8_t *>(src);
  uint8_t *dst_bytes = static_cast<uint8_t *>(dst);

  std::vector<uint8_t> qs(QK_Q4X4X2);

  for (unsigned int m = 0; m < M; ++m) {
    const uint8_t *src_row = src_bytes + (size_t)m * in_row_bytes;
    uint8_t *dst_row = dst_bytes + (size_t)m * out_row_bytes;

    const uint8_t *y_q = src_row;
    const uint8_t *y_d = src_row + n_tiles * TILE_QUANT_BYTES;

    for (unsigned int t = 0; t < n_tiles; ++t) {
      const uint8_t *q = y_q + (size_t)t * TILE_QUANT_BYTES;
      for (unsigned int j = 0; j < TILE_QUANT_BYTES; ++j) {
        qs[j + 0] = q[j] & 0x0F;
        qs[j + 128] = q[j] >> 4;
      }

      uint8_t *blocks = dst_row + (size_t)t * BLOCKS_PER_TILE * Q4_0_BLOCK_BYTES;
      const uint16_t *d = reinterpret_cast<const uint16_t *>(
        y_d + (size_t)t * TILE_SCALE_BYTES);

      for (unsigned int b = 0; b < BLOCKS_PER_TILE; ++b) {
        uint8_t *block = blocks + (size_t)b * Q4_0_BLOCK_BYTES;
        std::memcpy(block, &d[b], sizeof(uint16_t));
        pack_block_quants(block + 2, qs.data(), b);
      }
    }
  }
}

} // namespace nntrainer
