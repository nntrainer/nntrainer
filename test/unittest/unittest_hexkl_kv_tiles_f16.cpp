// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hexkl_kv_tiles_f16.cpp
 * @date   14 Sep 2026
 * @brief  Host tests for the resident fp16 KV tile registry
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The registry writes tiles by scalar scatter; the kernel builds the same
 * tiles on HVX. The device test checks the two agree byte for byte. This
 * host test checks the registry against the tile layout as documented in
 * hvx_tile_f16.h -- the place a lane index or a stride would be slipped.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "hexkl_kv_tiles_f16.h"

namespace {

/** Documented AH/WH fp16 layout: vector i holds rows 2i, 2i+1 interleaved. */
inline size_t tile_elem(uint32_t row, uint32_t col) {
  return static_cast<size_t>(row / 2) * 64 + col * 2 + (row & 1);
}

} // namespace

TEST(HexklKvTilesF16, RegisterRoundsUpAndReleases) {
  hexkl_kv_tiles_f16_table tbl{};
  uint32_t h = 99;
  ASSERT_EQ(hexkl_kv_tiles_f16_register(&tbl, 100, 2, 64, &h), 0);
  const hexkl_kv_tiles_f16 *kv = hexkl_kv_tiles_f16_get(&tbl, h);
  ASSERT_NE(kv, nullptr);
  EXPECT_EQ(kv->max_rows, 128u);
  EXPECT_EQ(kv->n_col_tiles, 4u);
  EXPECT_EQ(kv->n_dot_tiles, 2u);
  EXPECT_EQ(hexkl_kv_tiles_f16_release(&tbl, h), 0);
  EXPECT_EQ(hexkl_kv_tiles_f16_get(&tbl, h), nullptr);
  EXPECT_NE(hexkl_kv_tiles_f16_release(&tbl, h), 0);
  // Bad head_dim.
  EXPECT_NE(hexkl_kv_tiles_f16_register(&tbl, 64, 2, 100, &h), 0);
}

TEST(HexklKvTilesF16, AppendWritesTheDocumentedLayout) {
  hexkl_kv_tiles_f16_table tbl{};
  uint32_t h = 0;
  const uint32_t n_kv = 2, hd = 64, rows = 70;
  ASSERT_EQ(hexkl_kv_tiles_f16_register(&tbl, rows, n_kv, hd, &h), 0);
  const hexkl_kv_tiles_f16 *kv = hexkl_kv_tiles_f16_get(&tbl, h);
  ASSERT_NE(kv, nullptr);

  // Distinct values everywhere: k = row*1000 + col, v = 0x8000 | same.
  const uint32_t stride = n_kv * hd;
  std::vector<uint16_t> k(static_cast<size_t>(rows) * stride),
    v(static_cast<size_t>(rows) * stride);
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < stride; ++c) {
      k[static_cast<size_t>(r) * stride + c] =
        static_cast<uint16_t>(r * 200 + c);
      v[static_cast<size_t>(r) * stride + c] =
        static_cast<uint16_t>(0x8000u | (r * 200 + c));
    }
  }
  // Two appends, the second starting mid-tile.
  ASSERT_EQ(hexkl_kv_tiles_f16_append(&tbl, h, 0, 37, k.data(), v.data()), 0);
  ASSERT_EQ(hexkl_kv_tiles_f16_append(&tbl, h, 37, rows - 37,
                                      k.data() + 37 * stride,
                                      v.data() + 37 * stride),
            0);

  for (uint32_t n = 0; n < n_kv; ++n) {
    for (uint32_t r = 0; r < kv->max_rows; ++r) {
      const uint32_t c = r / 32, rr = r % 32;
      for (uint32_t d = 0; d < kv->n_dot_tiles; ++d) {
        const uint16_t *kt = reinterpret_cast<const uint16_t *>(
          kv->kt + hexkl_kv_tiles_f16_off(kv, n, c, d));
        const uint16_t *vt = reinterpret_cast<const uint16_t *>(
          kv->v + hexkl_kv_tiles_f16_off(kv, n, c, d));
        for (uint32_t e = 0; e < 32; ++e) {
          const uint32_t dim = 32 * d + e;
          uint16_t want_k = 0, want_v = 0;
          if (r < rows) {
            want_k = k[static_cast<size_t>(r) * stride + n * hd + dim];
            want_v = v[static_cast<size_t>(r) * stride + n * hd + dim];
          }
          // K^T tile: row = dim (within the dot tile), col = cache row.
          EXPECT_EQ(kt[tile_elem(e, rr)], want_k)
            << "K^T n=" << n << " r=" << r << " dim=" << dim;
          // V tile: row = cache row, col = dim.
          EXPECT_EQ(vt[tile_elem(rr, e)], want_v)
            << "V n=" << n << " r=" << r << " dim=" << dim;
        }
      }
    }
  }
  EXPECT_EQ(hexkl_kv_tiles_f16_release(&tbl, h), 0);
}

TEST(HexklKvTilesF16, AppendRejectsOutOfRange) {
  hexkl_kv_tiles_f16_table tbl{};
  uint32_t h = 0;
  ASSERT_EQ(hexkl_kv_tiles_f16_register(&tbl, 64, 1, 32, &h), 0);
  std::vector<uint16_t> row(32, 1);
  EXPECT_NE(hexkl_kv_tiles_f16_append(&tbl, h, 64, 1, row.data(), row.data()),
            0);
  EXPECT_NE(hexkl_kv_tiles_f16_append(&tbl, h, 63, 2, row.data(), row.data()),
            0);
  EXPECT_EQ(hexkl_kv_tiles_f16_append(&tbl, h, 63, 1, row.data(), row.data()),
            0);
  EXPECT_NE(hexkl_kv_tiles_f16_append(&tbl, 7, 0, 1, row.data(), row.data()),
            0);
  EXPECT_EQ(hexkl_kv_tiles_f16_release(&tbl, h), 0);
}
