// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hexkl_attn_q_plan.cpp
 * @date   28 Sep 2026
 * @brief  Host tests for the quantized attention VTCM planner
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The block geometry is the fp16 planner's and is tested there; this pins
 * the 64-row tiling and the region arithmetic (alignment, no overlap, fit)
 * for both weight widths.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "hexkl_attn_q_plan.h"

namespace {

constexpr uint32_t TB = HEXKL_ATTN_TILE_BYTES;

hexkl_attn_f16_shape shape(uint32_t n_q, uint32_t from, uint32_t to,
                           uint32_t nh_q, uint32_t nh_kv, uint32_t hd) {
  hexkl_attn_f16_shape s{};
  s.n_q = n_q;
  s.cache_from = from;
  s.cache_to = to;
  s.n_head_q = nh_q;
  s.n_head_kv = nh_kv;
  s.head_dim = hd;
  return s;
}

hexkl_attn_f16_tiling tiling(uint32_t br, uint32_t bc) {
  hexkl_attn_f16_tiling t{};
  t.br = br;
  t.bc = bc;
  return t;
}

std::vector<std::pair<uint32_t, uint32_t>>
regions(const hexkl_attn_q_layout &L, const hexkl_attn_f16_shape &s,
        const hexkl_attn_f16_tiling &t) {
  const uint32_t rt = L.n_row_tiles, ct = L.n_col_tiles, dt = L.n_dot_tiles;
  const uint32_t wb = ct * dt * L.tile_bytes;
  std::vector<std::pair<uint32_t, uint32_t>> r;
  r.push_back({L.q_ah, L.q_ah + L.n_qb_chunk * rt * dt * TB});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.kt_wh[i], L.kt_wh[i] + wb});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.v_wh[i], L.v_wh[i] + wb});
  r.push_back({L.acc, L.acc + HEXKL_ATTN_Q_ACC_BYTES});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.s_hf[i], L.s_hf[i] + 2 * rt * ct * TB});
  r.push_back({L.p_ah, L.p_ah + dt * rt * ct * TB});
  r.push_back({L.o_f32, L.o_f32 + t.g_br * s.head_dim * 4});
  r.push_back({L.qx_f32, L.qx_f32 + L.n_qb_chunk * t.g_br * s.head_dim * 4});
  return r;
}

} // namespace

TEST(HexklAttnQPlan, TilingPadsRowsTo64) {
  auto s = shape(128, 0, 1024, 16, 4, 128);
  auto t = tiling(16, 128);
  ASSERT_EQ(hexkl_attn_q_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g, 4u);
  EXPECT_EQ(t.g_br, 64u);
  t = tiling(20, 128); // 80 rows -> 128
  ASSERT_EQ(hexkl_attn_q_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g_br, 128u);
  // The fp16 init still pads to 32.
  t = tiling(20, 128);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g_br, 96u);
  // The chooser's br lands on exactly one uint8 row tile for G <= 64.
  hexkl_attn_f16_tiling ch{};
  hexkl_attn_f16_choose_tiling(&s, &ch);
  ASSERT_EQ(hexkl_attn_q_tiling_init(&s, &ch), HEXKL_ATTN_OK);
  EXPECT_EQ(ch.g_br, 64u);
}

TEST(HexklAttnQPlan, LayoutRegionsAreAlignedDisjointAndFit) {
  const uint32_t arena = 8u << 20;
  for (uint32_t tile_bytes : {1024u, 512u}) {
    for (uint32_t chunk : {1u, 8u}) {
      for (auto [nq, to, hq, hkv, hd, br, bc] :
           {std::make_tuple(128u, 1024u, 16u, 4u, 128u, 16u, 256u),
            std::make_tuple(100u, 100u, 8u, 8u, 64u, 64u, 32u),
            std::make_tuple(7u, 4096u, 32u, 8u, 256u, 20u, 64u),
            std::make_tuple(1u, 64u, 1u, 1u, 32u, 1u, 32u)}) {
        auto s = shape(nq, to - nq, to, hq, hkv, hd);
        auto t = tiling(br, bc);
        ASSERT_EQ(hexkl_attn_q_tiling_init(&s, &t), HEXKL_ATTN_OK);
        hexkl_attn_q_layout L{};
        ASSERT_EQ(hexkl_attn_q_plan(&s, &t, arena, tile_bytes, chunk, &L),
                  HEXKL_ATTN_OK)
          << "hd " << hd << " tile_bytes " << tile_bytes;
        EXPECT_EQ(L.tile_bytes, tile_bytes);
        EXPECT_EQ(L.n_row_tiles, t.g_br / 64);
        auto r = regions(L, s, t);
        for (const auto &[b, e] : r) {
          EXPECT_EQ(b % TB, 0u) << "region at " << b;
          EXPECT_LE(e, L.total);
        }
        std::sort(r.begin(), r.end());
        for (size_t i = 1; i < r.size(); ++i) {
          EXPECT_LE(r[i - 1].second, r[i].first)
            << "overlap between regions ending " << r[i - 1].second
            << " and starting " << r[i].first;
        }
        EXPECT_EQ(L.n_qb_chunk, chunk);
        EXPECT_LE(L.total, arena);
      }
    }
  }
}

TEST(HexklAttnQPlan, LayoutRejectsBadInputAndReportsNoFit) {
  auto s = shape(128, 0, 1024, 16, 4, 128);
  auto t = tiling(16, 256);
  ASSERT_EQ(hexkl_attn_q_tiling_init(&s, &t), HEXKL_ATTN_OK);
  hexkl_attn_q_layout L{};
  EXPECT_EQ(hexkl_attn_q_plan(&s, &t, 8u << 20, 768u, 1, &L),
            HEXKL_ATTN_EBADPARM);
  EXPECT_EQ(hexkl_attn_q_plan(&s, &t, 8u << 20, 1024u, 0, &L),
            HEXKL_ATTN_EBADPARM);
  EXPECT_EQ(hexkl_attn_q_plan(&s, &t, 64u << 10, 1024u, 1, &L),
            HEXKL_ATTN_ENOMEM);
  // An fp16 tiling (g_br 96) is not a valid uint8 tiling.
  auto t32 = tiling(24, 256);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t32), HEXKL_ATTN_OK);
  EXPECT_EQ(hexkl_attn_q_plan(&s, &t32, 8u << 20, 1024u, 1, &L),
            HEXKL_ATTN_EBADPARM);
}
