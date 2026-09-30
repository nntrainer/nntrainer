// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hexkl_attn_f16_plan.cpp
 * @date   14 Sep 2026
 * @brief  Host tests for the fp16 attention planner and its online softmax
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Two things a device is not needed for and that are cheapest to get wrong
 * on one: the VTCM region arithmetic (overlap, alignment, fit) and the
 * causal/window block geometry the kernel uses instead of a mask tensor.
 * The third test pins the online-softmax recurrence itself -- running max,
 * running sum, rescale of the partial output -- against a plain two-pass
 * softmax in f32, so the algebra the HMX diagonal-rescale implements is
 * checked before it is ever expressed in tiles.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "hexkl_attn_f16_plan.h"

namespace {

constexpr uint32_t TB = HEXKL_ATTN_TILE_BYTES;

hexkl_attn_f16_shape shape(uint32_t n_q, uint32_t from, uint32_t to,
                           uint32_t nh_q, uint32_t nh_kv, uint32_t hd,
                           uint32_t window = 0) {
  hexkl_attn_f16_shape s{};
  s.n_q = n_q;
  s.cache_from = from;
  s.cache_to = to;
  s.n_head_q = nh_q;
  s.n_head_kv = nh_kv;
  s.head_dim = hd;
  s.window = window;
  return s;
}

hexkl_attn_f16_tiling tiling(uint32_t br, uint32_t bc) {
  hexkl_attn_f16_tiling t{};
  t.br = br;
  t.bc = bc;
  return t;
}

/** Every region as [begin, end) so overlap can be checked generically. */
std::vector<std::pair<uint32_t, uint32_t>>
regions(const hexkl_attn_f16_layout &L, const hexkl_attn_f16_shape &s,
        const hexkl_attn_f16_tiling &t, uint32_t cfg_size) {
  const uint32_t rt = L.n_row_tiles, ct = L.n_col_tiles, dt = L.n_dot_tiles;
  std::vector<std::pair<uint32_t, uint32_t>> r;
  r.push_back({L.q_ah, L.q_ah + rt * dt * TB});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.kt_wh[i], L.kt_wh[i] + dt * ct * TB});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.v_wh[i], L.v_wh[i] + ct * dt * TB});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.s_ah[i], L.s_ah[i] + rt * ct * TB});
  r.push_back({L.d_ah, L.d_ah + rt * TB});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.o_wh[i], L.o_wh[i] + dt * rt * TB});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.k_land[i], L.k_land[i] + t.bc * s.head_dim * 2});
  for (int i = 0; i < 2; ++i)
    r.push_back({L.v_land[i], L.v_land[i] + t.bc * s.head_dim * 2});
  r.push_back({L.ml, L.ml + 2 * t.g_br * 4});
  r.push_back({L.cfg_f16, L.cfg_f16 + cfg_size});
  return r;
}

} // namespace

TEST(HexklAttnF16Plan, TilingDerivesGAndPadsRowsTo32) {
  auto s = shape(128, 0, 128, 16, 4, 128);
  auto t = tiling(16, 128);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g, 4u);
  EXPECT_EQ(t.g_br, 64u);

  // G=3 with br=16 gives 48 rows, which is not a tile multiple: padded to 64.
  s = shape(128, 0, 128, 24, 8, 128);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g, 3u);
  EXPECT_EQ(t.g_br, 64u);

  // MHA, one head per KV head: br alone sets the row count.
  s = shape(1, 500, 501, 8, 8, 64);
  t = tiling(1, 64);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g_br, 32u);
}

TEST(HexklAttnF16Plan, TilingRejectsBadShapes) {
  auto t = tiling(16, 128);
  auto s = shape(128, 0, 128, 16, 4, 100); // head_dim not a tile multiple
  EXPECT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_EBADPARM);
  s = shape(128, 0, 128, 15, 4, 128); // n_head_q not a multiple of n_head_kv
  EXPECT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_EBADPARM);
  s = shape(128, 100, 200, 16, 4, 128); // cache_to < cache_from + n_q
  EXPECT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_EBADPARM);
  s = shape(128, 0, 128, 16, 4, 128);
  t = tiling(16, 100); // bc not a tile multiple
  EXPECT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_EBADPARM);
  t = tiling(16, 128);
  s = shape(128, 0, 128, 16, 4, 288); // head_dim > 256
  EXPECT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_EBADPARM);
}

TEST(HexklAttnF16Plan, LayoutRegionsAreAlignedDisjointAndFit) {
  const uint32_t cfg = 256;
  const uint32_t arena = 8u * 1024u * 1024u;
  struct Case {
    uint32_t nh_q, nh_kv, hd, br, bc;
  };
  const Case cases[] = {
    {16, 4, 128, 16, 128}, {8, 8, 64, 32, 64},   {32, 8, 128, 8, 256},
    {24, 8, 128, 16, 64},  {16, 2, 256, 4, 128}, {8, 8, 64, 1, 32},
  };
  for (const auto &c : cases) {
    auto s = shape(256, 0, 256, c.nh_q, c.nh_kv, c.hd);
    auto t = tiling(c.br, c.bc);
    ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
    hexkl_attn_f16_layout L{};
    ASSERT_EQ(hexkl_attn_f16_plan(&s, &t, arena, cfg, &L), HEXKL_ATTN_OK)
      << "nh_q=" << c.nh_q << " hd=" << c.hd << " br=" << c.br
      << " bc=" << c.bc;

    auto r = regions(L, s, t, cfg);
    for (size_t i = 0; i < r.size(); ++i) {
      // Everything but the config and the m/l vectors is an HMX tile
      // operand and must sit on the activation alignment.
      if (r[i].first != L.cfg_f16) {
        EXPECT_EQ(r[i].first % TB, 0u) << "region " << i;
      } else {
        EXPECT_EQ(r[i].first % HEXKL_ATTN_CFG_ALIGN, 0u);
      }
      EXPECT_LE(r[i].second, arena) << "region " << i;
      for (size_t j = i + 1; j < r.size(); ++j) {
        const bool disjoint =
          r[i].second <= r[j].first || r[j].second <= r[i].first;
        EXPECT_TRUE(disjoint) << "regions " << i << " and " << j << " overlap";
      }
    }
    EXPECT_LE(L.total, L.cfg_f16);
    EXPECT_EQ(L.n_row_tiles, t.g_br / 32);
    EXPECT_EQ(L.n_col_tiles, c.bc / 32);
    EXPECT_EQ(L.n_dot_tiles, c.hd / 32);
  }
}

TEST(HexklAttnF16Plan, LayoutReportsWhenItDoesNotFit) {
  auto s = shape(256, 0, 256, 16, 4, 128);
  auto t = tiling(16, 128);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  hexkl_attn_f16_layout L{};
  // The worked example in the plan is ~420 KB; 64 KB cannot hold it.
  EXPECT_EQ(hexkl_attn_f16_plan(&s, &t, 64u * 1024u, 256, &L),
            HEXKL_ATTN_ENOMEM);
  // And an arena smaller than the config itself is rejected outright.
  EXPECT_EQ(hexkl_attn_f16_plan(&s, &t, 128, 256, &L), HEXKL_ATTN_ENOMEM);
}

TEST(HexklAttnF16Plan, RowRangeIsCausalAndWindowed) {
  // Prefill from an empty cache: row q sees [0, q].
  auto s = shape(64, 0, 64, 8, 8, 64);
  uint32_t lo, hi;
  hexkl_attn_f16_row_range(&s, 0, &lo, &hi);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 1u);
  hexkl_attn_f16_row_range(&s, 63, &lo, &hi);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 64u);

  // Decode at position 500: the single row sees everything so far.
  s = shape(1, 500, 501, 8, 8, 64);
  hexkl_attn_f16_row_range(&s, 0, &lo, &hi);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 501u);

  // Sliding window of 128 at position 500: [373, 501).
  s = shape(1, 500, 501, 8, 8, 64, 128);
  hexkl_attn_f16_row_range(&s, 0, &lo, &hi);
  EXPECT_EQ(lo, 501u - 128u);
  EXPECT_EQ(hi, 501u);

  // Window larger than the history: clamps at 0, never wraps.
  s = shape(1, 10, 11, 8, 8, 64, 128);
  hexkl_attn_f16_row_range(&s, 0, &lo, &hi);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 11u);
}

TEST(HexklAttnF16Plan, KvBlockRangeSkipsTheFutureAndFlagsTheDiagonal) {
  // 128 prefill rows appended after 256 cached ones, br=16, bc=64.
  auto s = shape(128, 256, 384, 16, 4, 128);
  auto t = tiling(16, 64);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(hexkl_attn_f16_n_q_blocks(&s, &t), 8u);

  uint32_t lo, hi;
  // Query block 0 covers positions 256..271: cache blocks [0, 5) -- block 4
  // (rows 256..319) is the diagonal one, blocks 5 (320..383) are future.
  hexkl_attn_f16_kv_block_range(&s, &t, 0, &lo, &hi);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 5u);
  for (uint32_t kb = 0; kb < 4; ++kb) {
    EXPECT_TRUE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 0, kb)) << kb;
  }
  EXPECT_FALSE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 0, 4));

  // Last query block covers 368..383: all 6 blocks, only the last masked.
  hexkl_attn_f16_kv_block_range(&s, &t, 7, &lo, &hi);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 6u);
  EXPECT_TRUE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 7, 4));
  EXPECT_FALSE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 7, 5));

  // A query block whose rows exactly fill a cache block: positions 320..335
  // sit inside cache block 5, which is still masked (row 320 cannot see 335).
  hexkl_attn_f16_kv_block_range(&s, &t, 4, &lo, &hi);
  EXPECT_EQ(hi, 6u);
  EXPECT_FALSE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 4, 5));
}

TEST(HexklAttnF16Plan, KvBlockRangeHonoursTheWindowLowerBound) {
  // Window 96 with bc=32: query block at 256..271 sees [161, 272) for its
  // first row and [176, 272) for its last -> blocks [5, 9); block 5
  // (160..191) is only partially visible to the last row, so masked.
  auto s = shape(128, 256, 384, 8, 8, 64, 96);
  auto t = tiling(16, 32);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  uint32_t lo, hi;
  hexkl_attn_f16_kv_block_range(&s, &t, 0, &lo, &hi);
  EXPECT_EQ(lo, 5u);
  EXPECT_EQ(hi, 9u);
  EXPECT_FALSE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 0, 5));
  EXPECT_TRUE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 0, 6));
  EXPECT_TRUE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 0, 7));
  EXPECT_FALSE(hexkl_attn_f16_kv_block_unmasked(&s, &t, 0, 8)); // diagonal
}

TEST(HexklAttnF16Plan, ChooserGivesValidTilingsAcrossShapes) {
  struct Case {
    uint32_t n_q, cache_to, nh_q, nh_kv;
    uint32_t want_br, want_bc;
  };
  const Case cases[] = {
    {128, 1024, 16, 4, 16, 256}, // long cache: bc capped at 256
    {128, 128, 16, 4, 16, 64},   // 128/3 = 42 -> below 64 -> 64
    {1, 40, 16, 4, 1, 32},       // tiny cache: 32-row blocks
    {128, 700, 8, 8, 64, 192},   // MHA: br 64, 700/3 = 233 -> 192
    {40, 512, 24, 8, 21, 128},   // G=3: 21 rows -> 63 -> padded to 64
    {5, 5, 8, 2, 5, 32},         // fewer rows than br
  };
  for (const auto &c : cases) {
    auto s = shape(c.n_q, 0, c.cache_to, c.nh_q, c.nh_kv, 64);
    hexkl_attn_f16_tiling t{};
    hexkl_attn_f16_choose_tiling(&s, &t);
    EXPECT_EQ(t.br, c.want_br) << "n_q=" << c.n_q << " to=" << c.cache_to;
    EXPECT_EQ(t.bc, c.want_bc) << "n_q=" << c.n_q << " to=" << c.cache_to;
    ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
    EXPECT_LE(t.g_br, 64u);
    hexkl_attn_f16_layout L{};
    EXPECT_EQ(hexkl_attn_f16_plan(&s, &t, 8u * 1024u * 1024u, 256, &L),
              HEXKL_ATTN_OK);
  }
}

TEST(HexklAttnF16Plan, TileRowMapsToQueryAndGroupAndFlagsPadding) {
  auto s = shape(10, 0, 10, 12, 4, 64); // G=3, br=4 -> 12 real rows, g_br=32
  auto t = tiling(4, 64);
  ASSERT_EQ(hexkl_attn_f16_tiling_init(&s, &t), HEXKL_ATTN_OK);
  EXPECT_EQ(t.g_br, 32u);
  uint32_t q, g;
  ASSERT_TRUE(hexkl_attn_f16_tile_row_to_qg(&s, &t, 0, 0, &q, &g));
  EXPECT_EQ(q, 0u);
  EXPECT_EQ(g, 0u);
  ASSERT_TRUE(hexkl_attn_f16_tile_row_to_qg(&s, &t, 0, 11, &q, &g));
  EXPECT_EQ(q, 3u);
  EXPECT_EQ(g, 2u);
  EXPECT_FALSE(hexkl_attn_f16_tile_row_to_qg(&s, &t, 0, 12, &q, &g)); // pad
  // Last query block holds rows 8, 9 only: tile rows 6..11 are past n_q.
  ASSERT_TRUE(hexkl_attn_f16_tile_row_to_qg(&s, &t, 2, 5, &q, &g));
  EXPECT_EQ(q, 9u);
  EXPECT_EQ(g, 2u);
  EXPECT_FALSE(hexkl_attn_f16_tile_row_to_qg(&s, &t, 2, 6, &q, &g));
}

/**
 * The recurrence the kernel implements per row, block by block:
 *   m' = max(m, rowmax(S_blk));  a = 2^(m - m');
 *   l' = l*a + sum 2^(S_blk - m');  O' = a*O + P.V_blk;  final O/l.
 * With S pre-multiplied by log2(e) this equals softmax(S/log2e).V. Checked
 * against the two-pass definition for a row with several blocks, including
 * a block whose maximum is far below the running max (a ~ 1) and one far
 * above it (a ~ 0), which is where sign or order mistakes show.
 */
TEST(HexklAttnF16Plan, OnlineSoftmaxRecurrenceMatchesTwoPass) {
  const uint32_t kv = 256, bc = 64, hd = 32;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  std::vector<float> s_raw(kv), v(kv * hd);
  for (uint32_t k = 0; k < kv; ++k) {
    // Make block 1 the loudest and block 2 the quietest.
    float bias = (k / bc == 1) ? 12.f : (k / bc == 2) ? -12.f : 0.f;
    s_raw[k] = uni(rng) * 3.f + bias;
    for (uint32_t d = 0; d < hd; ++d)
      v[k * hd + d] = uni(rng);
  }
  // Two-pass reference in natural base.
  float mx = -INFINITY;
  for (float x : s_raw)
    mx = std::max(mx, x);
  std::vector<double> ref(hd, 0.0);
  double den = 0.0;
  for (uint32_t k = 0; k < kv; ++k) {
    double p = std::exp((double)s_raw[k] - mx);
    den += p;
    for (uint32_t d = 0; d < hd; ++d)
      ref[d] += p * v[k * hd + d];
  }
  for (auto &x : ref)
    x /= den;

  // Online, base-2, exactly the kernel's per-block steps.
  const float log2e = 1.4426950408889634f;
  float m = -INFINITY, l = 0.f;
  std::vector<float> o(hd, 0.f);
  for (uint32_t b = 0; b < kv / bc; ++b) {
    float bm = -INFINITY;
    for (uint32_t k = b * bc; k < (b + 1) * bc; ++k)
      bm = std::max(bm, s_raw[k] * log2e);
    const float m_new = std::max(m, bm);
    const float a = std::exp2(m - m_new); // exp2(-inf) = 0 on the first block
    float rowsum = 0.f;
    std::vector<float> pv(hd, 0.f);
    for (uint32_t k = b * bc; k < (b + 1) * bc; ++k) {
      const float p = std::exp2(s_raw[k] * log2e - m_new);
      rowsum += p;
      for (uint32_t d = 0; d < hd; ++d)
        pv[d] += p * v[k * hd + d];
    }
    l = l * a + rowsum;
    for (uint32_t d = 0; d < hd; ++d)
      o[d] = a * o[d] + pv[d];
    m = m_new;
  }
  for (uint32_t d = 0; d < hd; ++d) {
    EXPECT_NEAR(o[d] / l, ref[d], 1e-5) << "d=" << d;
  }
}
