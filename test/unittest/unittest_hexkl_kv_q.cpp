// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hexkl_kv_q.cpp
 * @date   23 Sep 2026
 * @brief  Host tests for the quantized (int8 / int4) KV cache registry
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Everything but the HexKL tile bake is plain C, so the quantizer, the
 * master layouts and the append / dump / stage bookkeeping are pinned down
 * here; the device test then only has to show that the baked tiles multiply
 * to the same integers these masters do.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "hexkl_kv_q.h"

namespace {

/** @brief f32 -> fp16 bits, round to nearest even (test input only). */
uint16_t f32_to_hf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  uint32_t hm = mant >> 13;
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
    ++hm;
    if (hm == 0x400u) {
      hm = 0;
      ++exp;
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | hm);
}

/** @brief Deterministic fp16 rows in [-amp, amp): [rows][n_kv*hd]. */
std::vector<uint16_t> rows_hf(uint32_t rows, uint32_t width, uint32_t seed,
                              float amp) {
  std::vector<uint16_t> v(static_cast<size_t>(rows) * width);
  uint32_t s = seed;
  for (auto &x : v) {
    s = s * 1664525u + 1013904223u;
    x =
      f32_to_hf((static_cast<float>(s >> 8) / 16777216.0f * 2.0f - 1.0f) * amp);
  }
  return v;
}

} // namespace

TEST(HexklKvQ, QuantKRowRangeScaleColsum) {
  const float x[32] = {0.5f,  -1.0f, 0.25f, 0.0f, 2.0f, -0.125f, 1.5f, 0.75f,
                       0.5f,  -1.0f, 0.25f, 0.0f, 2.0f, -0.125f, 1.5f, 0.75f,
                       -2.0f, 0.1f,  0.2f,  0.3f, 0.4f, 0.6f,    0.7f, 0.8f,
                       0.9f,  1.1f,  1.2f,  1.3f, 1.4f, 1.6f,    1.7f, 1.8f};
  for (int32_t qmax : {127, 7}) {
    int8_t q[32];
    float scale;
    int32_t colsum;
    hexkl_kv_q_quant_k_row(x, 32, qmax, q, &scale, &colsum);
    EXPECT_FLOAT_EQ(scale, 2.0f / static_cast<float>(qmax));
    int32_t sum = 0;
    for (int i = 0; i < 32; ++i) {
      EXPECT_LE(q[i], qmax);
      EXPECT_GE(q[i], -qmax);
      // Within half a step of the exact quotient.
      EXPECT_NEAR(static_cast<float>(q[i]), x[i] / scale, 0.5f + 1e-5f);
      sum += q[i];
    }
    EXPECT_EQ(colsum, sum);
    EXPECT_EQ(q[4], qmax);
    EXPECT_EQ(q[16], -qmax);
  }
  // All-zero row: scale 1, zeros, colsum 0.
  const float z[32] = {0.0f};
  int8_t q[32];
  float scale;
  int32_t colsum;
  hexkl_kv_q_quant_k_row(z, 32, 127, q, &scale, &colsum);
  EXPECT_FLOAT_EQ(scale, 1.0f);
  EXPECT_EQ(colsum, 0);
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(q[i], 0);
  }
}

TEST(HexklKvQ, QuantVRowIsPerGroup) {
  float x[64];
  for (int i = 0; i < 32; ++i) {
    x[i] = 0.01f * static_cast<float>(i - 16);             // group 0: amax 0.16
    x[32 + i] = 4.0f * static_cast<float>(i - 16) / 16.0f; // group 1: amax 4
  }
  int8_t q[64];
  float scale = 0.0f;
  hexkl_kv_q_quant_v_row(x, 64, 7, q, &scale);
  // One scale for the row: the larger group sets it, and the smaller one
  // is quantized against the same step rather than its own.
  EXPECT_FLOAT_EQ(scale, 4.0f / 7.0f);
  EXPECT_EQ(q[32], -7);
  EXPECT_EQ(q[0], 0);
  for (int i = 0; i < 64; ++i) {
    EXPECT_NEAR(static_cast<float>(q[i]), x[i] / scale, 0.5f + 1e-5f);
  }
}

TEST(HexklKvQ, RegisterRoundsUpAndReleases) {
  hexkl_kv_q_table tbl{};
  uint32_t h = 99;
  ASSERT_EQ(hexkl_kv_q_register(&tbl, HEXKL_KV_Q4, 100, 2, 64, &h), 0);
  const hexkl_kv_q *kv = hexkl_kv_q_get(&tbl, h);
  ASSERT_NE(kv, nullptr);
  EXPECT_EQ(kv->max_rows, 128u);
  EXPECT_EQ(kv->n_col_tiles, 4u);
  EXPECT_EQ(kv->n_dot_tiles, 2u);
  EXPECT_EQ(kv->qmax, 7);
  EXPECT_EQ(kv->tile_bytes, 512u);
  // Untouched rows: scale 1, zero values.
  EXPECT_FLOAT_EQ(kv->s_k[hexkl_kv_q_sk_index(kv, 1, 127)], 1.0f);
  EXPECT_FLOAT_EQ(kv->s_v[hexkl_kv_q_sv_index(kv, 1, 127)], 1.0f);
  EXPECT_EQ(kv->kt4[hexkl_kv_q_kt4_index(kv, 1, 127, 63)], HEXKL_KV_Q_BIAS);
  EXPECT_EQ(hexkl_kv_q_release(&tbl, h), 0);
  EXPECT_EQ(hexkl_kv_q_get(&tbl, h), nullptr);
  EXPECT_NE(hexkl_kv_q_release(&tbl, h), 0);
  EXPECT_NE(hexkl_kv_q_register(&tbl, HEXKL_KV_Q8, 64, 2, 100, &h), 0);
  EXPECT_NE(
    hexkl_kv_q_register(&tbl, static_cast<hexkl_kv_q_kind>(2), 64, 2, 64, &h),
    0);
}

TEST(HexklKvQ, MasterIndicesAreBijective) {
  hexkl_kv_q_table tbl{};
  uint32_t h = 0;
  ASSERT_EQ(hexkl_kv_q_register(&tbl, HEXKL_KV_Q8, 64, 3, 96, &h), 0);
  const hexkl_kv_q *kv = hexkl_kv_q_get(&tbl, h);
  const size_t total = static_cast<size_t>(3) * 64 * 96;
  std::set<size_t> kt, v;
  for (uint32_t n = 0; n < 3; ++n) {
    for (uint32_t r = 0; r < 64; ++r) {
      for (uint32_t d = 0; d < 96; ++d) {
        const size_t a = hexkl_kv_q_kt4_index(kv, n, r, d);
        const size_t b = hexkl_kv_q_v4_index(kv, n, r, d);
        ASSERT_LT(a, total);
        ASSERT_LT(b, total);
        kt.insert(a);
        v.insert(b);
      }
    }
  }
  EXPECT_EQ(kt.size(), total);
  EXPECT_EQ(v.size(), total);
  // The documented interleaves: 4 consecutive dims of one row are 4
  // consecutive bytes in kt4; 4 consecutive rows of one dim in v4.
  EXPECT_EQ(hexkl_kv_q_kt4_index(kv, 1, 5, 9),
            hexkl_kv_q_kt4_index(kv, 1, 5, 8) + 1);
  EXPECT_EQ(hexkl_kv_q_kt4_index(kv, 1, 6, 8),
            hexkl_kv_q_kt4_index(kv, 1, 5, 8) + 4);
  EXPECT_EQ(hexkl_kv_q_v4_index(kv, 1, 9, 5),
            hexkl_kv_q_v4_index(kv, 1, 8, 5) + 1);
  EXPECT_EQ(hexkl_kv_q_v4_index(kv, 1, 8, 6),
            hexkl_kv_q_v4_index(kv, 1, 8, 5) + 4);
  hexkl_kv_q_release(&tbl, h);
}

TEST(HexklKvQ, AppendDumpRoundTripAndRewrite) {
  for (auto kind : {HEXKL_KV_Q8, HEXKL_KV_Q4}) {
    hexkl_kv_q_table tbl{};
    uint32_t h = 0;
    const uint32_t n_kv = 2, hd = 64, rows = 70, width = n_kv * hd;
    ASSERT_EQ(hexkl_kv_q_register(&tbl, kind, rows, n_kv, hd, &h), 0);
    const hexkl_kv_q *kv = hexkl_kv_q_get(&tbl, h);
    auto k = rows_hf(rows, width, 0x51u + kind, 1.0f);
    auto v = rows_hf(rows, width, 0x52u + kind, 1.0f);
    // Two appends, host build: no bake.
    ASSERT_EQ(
      hexkl_kv_q_append(&tbl, h, 0, 40, k.data(), v.data(), nullptr, nullptr),
      0);
    ASSERT_EQ(hexkl_kv_q_append(&tbl, h, 40, 30, k.data() + 40 * width,
                                v.data() + 40 * width, nullptr, nullptr),
              0);

    std::vector<int8_t> kq(static_cast<size_t>(rows) * width), vq(kq.size());
    std::vector<float> sk(static_cast<size_t>(rows) * n_kv),
      sv(static_cast<size_t>(rows) * n_kv);
    std::vector<int32_t> cs(sk.size());
    ASSERT_EQ(hexkl_kv_q_dump(kv, 0, rows, kq.data(), vq.data(), sk.data(),
                              cs.data(), sv.data()),
              0);

    // Against the quantizer run directly on the same rows.
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t n = 0; n < n_kv; ++n) {
        float x[64];
        int8_t q[64];
        float scale, v_scale;
        int32_t colsum;
        for (uint32_t d = 0; d < hd; ++d) {
          x[d] = hexkl_kv_q_hf_to_f32(
            k[static_cast<size_t>(r) * width + n * hd + d]);
        }
        hexkl_kv_q_quant_k_row(x, hd, kv->qmax, q, &scale, &colsum);
        EXPECT_EQ(
          std::memcmp(q, &kq[static_cast<size_t>(r) * width + n * hd], hd), 0)
          << "K row " << r << " head " << n;
        EXPECT_FLOAT_EQ(sk[r * n_kv + n], scale);
        EXPECT_EQ(cs[r * n_kv + n], colsum);
        for (uint32_t d = 0; d < hd; ++d) {
          x[d] = hexkl_kv_q_hf_to_f32(
            v[static_cast<size_t>(r) * width + n * hd + d]);
        }
        hexkl_kv_q_quant_v_row(x, hd, kv->qmax, q, &v_scale);
        EXPECT_EQ(
          std::memcmp(q, &vq[static_cast<size_t>(r) * width + n * hd], hd), 0)
          << "V row " << r << " head " << n;
        EXPECT_FLOAT_EQ(sv[r * n_kv + n], v_scale);
      }
    }

    // Rewrite rows 10..12 with new data; only those change.
    auto k2 = rows_hf(3, width, 0x77u, 1.0f);
    auto v2 = rows_hf(3, width, 0x78u, 1.0f);
    ASSERT_EQ(
      hexkl_kv_q_append(&tbl, h, 10, 3, k2.data(), v2.data(), nullptr, nullptr),
      0);
    std::vector<int8_t> kq2(kq.size()), vq2(vq.size());
    ASSERT_EQ(hexkl_kv_q_dump(kv, 0, rows, kq2.data(), vq2.data(), nullptr,
                              nullptr, nullptr),
              0);
    for (uint32_t r = 0; r < rows; ++r) {
      const bool same_k =
        std::memcmp(&kq[static_cast<size_t>(r) * width],
                    &kq2[static_cast<size_t>(r) * width], width) == 0;
      const bool same_v =
        std::memcmp(&vq[static_cast<size_t>(r) * width],
                    &vq2[static_cast<size_t>(r) * width], width) == 0;
      if (r >= 10 && r < 13) {
        EXPECT_FALSE(same_k && same_v) << "row " << r << " not rewritten";
      } else {
        EXPECT_TRUE(same_k) << "K row " << r << " changed";
        EXPECT_TRUE(same_v) << "V row " << r << " changed";
      }
    }

    // Staging of column tile 1 (rows 32..63) is the masters, transposed for K.
    hexkl_kv_q_stage(const_cast<hexkl_kv_q *>(kv), 1, 1);
    for (uint32_t rr = 0; rr < 32; ++rr) {
      for (uint32_t d = 0; d < hd; ++d) {
        EXPECT_EQ(kv->stage_kt[d * 32 + rr],
                  kq2[static_cast<size_t>(32 + rr) * width + 1 * hd + d]);
        EXPECT_EQ(kv->stage_v[rr * hd + d],
                  vq2[static_cast<size_t>(32 + rr) * width + 1 * hd + d]);
      }
    }

    // Out of range.
    EXPECT_NE(
      hexkl_kv_q_append(&tbl, h, 120, 9, k.data(), v.data(), nullptr, nullptr),
      0);
    EXPECT_NE(
      hexkl_kv_q_dump(kv, 120, 9, nullptr, nullptr, nullptr, nullptr, nullptr),
      0);
    hexkl_kv_q_release(&tbl, h);
  }
}

/**
 * Fixed scales (plan 23, R4a): K quantized with one scale per head, V with
 * one per (head, dim), at the global layer's head_dim of 512. The dump must
 * be rint(x / scale) clamped, with colsum the sum of what was stored.
 */
TEST(HexklKvQ, FixedScalesHeadDim512) {
  hexkl_kv_q_table tbl{};
  uint32_t h = 0;
  const uint32_t n_kv = 2, hd = 512, rows = 70, width = n_kv * hd;
  ASSERT_EQ(hexkl_kv_q_register(&tbl, HEXKL_KV_Q8, rows, n_kv, hd, &h), 0);
  auto k = rows_hf(rows, width, 0x61u, 3.0f);
  auto v = rows_hf(rows, width, 0x62u, 2.0f);
  std::vector<float> s_k(n_kv), s_v(static_cast<size_t>(n_kv) * hd);
  for (uint32_t n = 0; n < n_kv; ++n) {
    s_k[n] = 0.02f + 0.01f * n; // |k| <= 3 -> some clamping at 127
    for (uint32_t d = 0; d < hd; ++d) {
      s_v[n * hd + d] = 0.01f + 0.00005f * d;
    }
  }
  // Rejects non-positive scales, accepts proper ones.
  std::vector<float> bad(s_k);
  bad[1] = 0.0f;
  EXPECT_NE(hexkl_kv_q_set_fixed_scales(&tbl, h, bad.data(), s_v.data()), 0);
  ASSERT_EQ(hexkl_kv_q_set_fixed_scales(&tbl, h, s_k.data(), s_v.data()), 0);
  ASSERT_EQ(
    hexkl_kv_q_append(&tbl, h, 0, rows, k.data(), v.data(), nullptr, nullptr),
    0);
  const hexkl_kv_q *kv = hexkl_kv_q_get(&tbl, h);
  ASSERT_NE(kv, nullptr);
  EXPECT_EQ(kv->fixed, 1);

  std::vector<int8_t> kq(static_cast<size_t>(rows) * width), vq(kq.size());
  std::vector<float> dsk(static_cast<size_t>(rows) * n_kv);
  std::vector<int32_t> cs(dsk.size());
  ASSERT_EQ(hexkl_kv_q_dump(kv, 0, rows, kq.data(), vq.data(), dsk.data(),
                            cs.data(), nullptr),
            0);
  size_t clamped = 0;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t n = 0; n < n_kv; ++n) {
      int32_t sum = 0;
      for (uint32_t d = 0; d < hd; ++d) {
        const size_t at = static_cast<size_t>(r) * width + n * hd + d;
        // The definition multiplies by the f32 reciprocal, as HVX does.
        float wk = std::rint(hexkl_kv_q_hf_to_f32(k[at]) * (1.0f / s_k[n]));
        wk = std::min(127.0f, std::max(-127.0f, wk));
        clamped += std::fabs(wk) == 127.0f;
        ASSERT_EQ(kq[at], static_cast<int8_t>(wk))
          << "k row " << r << " d " << d;
        sum += kq[at];
        float wv =
          std::rint(hexkl_kv_q_hf_to_f32(v[at]) * (1.0f / s_v[n * hd + d]));
        wv = std::min(127.0f, std::max(-127.0f, wv));
        ASSERT_EQ(vq[at], static_cast<int8_t>(wv))
          << "v row " << r << " d " << d;
      }
      EXPECT_EQ(cs[r * n_kv + n], sum);
      EXPECT_EQ(dsk[r * n_kv + n], s_k[n]);
    }
  }
  EXPECT_GT(clamped, 0u) << "the test data should exercise the clamp";
  EXPECT_EQ(hexkl_kv_q_release(&tbl, h), 0);
}

/**
 * K per (head, dim) (NNTR_HTP_Q2_KDIM): the masters are rint(x / s_kd), the
 * row-blocked kernel keeps fs_k = max_d s_kd in alpha and multiplies Q by
 * fk_ratio = s_kd / fs_k, so fs_k sum(q r k8) must be sum(q s_kd k8). A
 * later per-head call puts the cache back to ratio 1.
 */
TEST(HexklKvQ, FixedScalesKPerDim) {
  hexkl_kv_q_table tbl{};
  uint32_t h = 0;
  const uint32_t n_kv = 2, hd = 256, rows = 70, width = n_kv * hd;
  ASSERT_EQ(hexkl_kv_q_register(&tbl, HEXKL_KV_Q8, rows, n_kv, hd, &h), 0);
  auto k = rows_hf(rows, width, 0x71u, 3.0f);
  auto v = rows_hf(rows, width, 0x72u, 2.0f);
  std::vector<float> s_k(width), s_v(width, 0.02f);
  for (uint32_t i = 0; i < width; ++i)
    s_k[i] = 0.004f + 0.0001f * static_cast<float>((i * 37u) % 251u);
  std::vector<float> bad(s_k);
  bad[hd + 3] = -1.0f;
  EXPECT_NE(hexkl_kv_q_set_fixed_scales_kd(&tbl, h, bad.data(), s_v.data()), 0);
  ASSERT_EQ(hexkl_kv_q_set_fixed_scales_kd(&tbl, h, s_k.data(), s_v.data()), 0);
  const hexkl_kv_q *kv = hexkl_kv_q_get(&tbl, h);
  ASSERT_NE(kv, nullptr);
  EXPECT_EQ(kv->k_per_dim, 1);
  for (uint32_t n = 0; n < n_kv; ++n) {
    const float top =
      *std::max_element(s_k.begin() + n * hd, s_k.begin() + (n + 1) * hd);
    EXPECT_EQ(kv->fs_k[n], top);
    for (uint32_t d = 0; d < hd; ++d) {
      EXPECT_EQ(kv->fs_kd[n * hd + d], s_k[n * hd + d]);
      EXPECT_EQ(kv->fk_ratio[n * hd + d], s_k[n * hd + d] / top);
      EXPECT_LE(kv->fk_ratio[n * hd + d], 1.0f);
    }
  }
  ASSERT_EQ(
    hexkl_kv_q_append(&tbl, h, 0, rows, k.data(), v.data(), nullptr, nullptr),
    0);
  std::vector<int8_t> kq(static_cast<size_t>(rows) * width), vq(kq.size());
  std::vector<float> dsk(static_cast<size_t>(rows) * n_kv);
  std::vector<int32_t> cs(dsk.size());
  ASSERT_EQ(hexkl_kv_q_dump(kv, 0, rows, kq.data(), vq.data(), dsk.data(),
                            cs.data(), nullptr),
            0);
  uint32_t lcg = 0x73u;
  std::vector<float> q(hd);
  double worst = 0.0;
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t n = 0; n < n_kv; ++n) {
      int32_t sum = 0;
      double ref = 0.0, fold = 0.0;
      for (uint32_t d = 0; d < hd; ++d) {
        const size_t at = static_cast<size_t>(r) * width + n * hd + d;
        float wk =
          std::rint(hexkl_kv_q_hf_to_f32(k[at]) * (1.0f / s_k[n * hd + d]));
        wk = std::min(127.0f, std::max(-127.0f, wk));
        ASSERT_EQ(kq[at], static_cast<int8_t>(wk))
          << "k row " << r << " d " << d;
        sum += kq[at];
        lcg = lcg * 1664525u + 1013904223u;
        q[d] = static_cast<float>(lcg >> 8) / 16777216.0f - 0.5f;
        ref += static_cast<double>(q[d]) * s_k[n * hd + d] * kq[at];
        fold += static_cast<double>(q[d] * kv->fk_ratio[n * hd + d]) * kq[at];
      }
      EXPECT_EQ(cs[r * n_kv + n], sum);
      EXPECT_EQ(dsk[r * n_kv + n], kv->fs_k[n]);
      fold *= kv->fs_k[n];
      worst =
        std::max(worst, std::fabs(fold - ref) / std::max(1e-6, std::fabs(ref)));
    }
  }
  EXPECT_LT(worst, 1e-5) << "fs_k sum(q r k8) != sum(q s_kd k8)";

  std::vector<float> s_head = {0.02f, 0.03f};
  ASSERT_EQ(hexkl_kv_q_set_fixed_scales(&tbl, h, s_head.data(), s_v.data()), 0);
  EXPECT_EQ(kv->k_per_dim, 0);
  for (uint32_t i = 0; i < width; ++i) {
    EXPECT_EQ(kv->fk_ratio[i], 1.0f);
    EXPECT_EQ(kv->fs_kd[i], s_head[i / hd]);
  }
  EXPECT_EQ(hexkl_kv_q_release(&tbl, h), 0);

  // int4 caches keep one K scale per head
  ASSERT_EQ(hexkl_kv_q_register(&tbl, HEXKL_KV_Q4, rows, n_kv, hd, &h), 0);
  EXPECT_NE(hexkl_kv_q_set_fixed_scales_kd(&tbl, h, s_k.data(), s_v.data()), 0);
  EXPECT_EQ(hexkl_kv_q_release(&tbl, h), 0);
}

/** [C] hexkl_kv_q_fill_q8 takes rows back from the decode cache's layout
 *  (K [tiles][hd][64] a head, V [seq][hd]): with no ratio the masters are
 *  those bytes (offset by 128), the column sums their sums and the scales
 *  the fixed ones; with a ratio each value is rne(q * ratio) clamped. */
TEST(HexklKvQ, FillFromDecodeLayout) {
  hexkl_kv_q_table tbl{};
  uint32_t h = 0;
  const uint32_t n_kv = 2, hd = 256, rows = 150, seq = 192, width = n_kv * hd;
  ASSERT_EQ(hexkl_kv_q_register(&tbl, HEXKL_KV_Q8, 160, n_kv, hd, &h), 0);
  std::vector<float> s_k(width), s_v(width, 0.02f);
  for (uint32_t i = 0; i < width; ++i)
    s_k[i] = 0.004f + 0.0001f * static_cast<float>((i * 37u) % 251u);
  ASSERT_EQ(hexkl_kv_q_set_fixed_scales_kd(&tbl, h, s_k.data(), s_v.data()), 0);
  const hexkl_kv_q *kv = hexkl_kv_q_get(&tbl, h);
  ASSERT_NE(kv, nullptr);
  const size_t hs = static_cast<size_t>(seq) * hd;
  std::vector<int8_t> kt8(n_kv * hs), v8(n_kv * hs);
  uint32_t lcg = 0x9u;
  auto next = [&lcg]() {
    lcg = lcg * 1664525u + 1013904223u;
    return static_cast<int8_t>(static_cast<int>((lcg >> 24) % 255u) - 127);
  };
  for (auto &x : kt8)
    x = next();
  for (auto &x : v8)
    x = next();
  auto k_at = [&](uint32_t n, uint32_t r, uint32_t d) {
    return kt8[n * hs + (r / 64u) * hd * 64u + d * 64u + r % 64u];
  };
  // rows [10, 150): a later call's history after a first one's 10
  ASSERT_EQ(hexkl_kv_q_fill_q8(&tbl, h, 10, rows - 10, kt8.data(), v8.data(),
                               hs, nullptr, nullptr, nullptr, nullptr),
            0);
  int bad = 0;
  for (uint32_t r = 10; r < rows; ++r)
    for (uint32_t n = 0; n < n_kv; ++n) {
      int32_t cs = 0;
      for (uint32_t d = 0; d < hd; ++d) {
        const int8_t qk = k_at(n, r, d), qv = v8[n * hs + r * hd + d];
        cs += qk;
        bad += kv->kt4[hexkl_kv_q_kt4_index(kv, n, r, d)] !=
               static_cast<uint8_t>(static_cast<uint8_t>(qk) ^ 0x80u);
        bad += kv->v4[hexkl_kv_q_v4_index(kv, n, r, d)] !=
               static_cast<uint8_t>(static_cast<uint8_t>(qv) ^ 0x80u);
      }
      bad += kv->colsum_k[hexkl_kv_q_sk_index(kv, n, r)] != cs;
      bad += kv->s_k[hexkl_kv_q_sk_index(kv, n, r)] != kv->fs_k[n];
      bad += kv->s_v[hexkl_kv_q_sv_index(kv, n, r)] != 1.0f;
    }
  EXPECT_EQ(bad, 0);
  // a scale that grew: the values come back at the new one
  std::vector<float> kr(width), vr(width);
  for (uint32_t i = 0; i < width; ++i) {
    kr[i] = 0.5f + 0.25f * static_cast<float>(i % 3u); // 0.5, 0.75, 1.0
    vr[i] = 1.0f / 1.5f;
  }
  ASSERT_EQ(hexkl_kv_q_fill_q8(&tbl, h, 0, 64, kt8.data(), v8.data(), hs,
                               kr.data(), vr.data(), nullptr, nullptr),
            0);
  // Q15: r as round(r 2^15) (1.0 as 2^15 - 1), (q r + 2^14) >> 15
  auto q15 = [](int q, float r) {
    const float x = r * 32768.0f + 0.5f;
    const int32_t R = x >= 32767.0f ? 32767 : static_cast<int32_t>(x);
    return std::max(-127, std::min(127, (q * R + 16384) >> 15));
  };
  bad = 0;
  int far = 0, moved_at_one = 0;
  for (uint32_t r = 0; r < 64; ++r)
    for (uint32_t n = 0; n < n_kv; ++n)
      for (uint32_t d = 0; d < hd; ++d) {
        const size_t i = n * hd + d;
        const int qk = k_at(n, r, d), qv = v8[n * hs + r * hd + d];
        const int ek = q15(qk, kr[i]), ev = q15(qv, vr[i]);
        bad += static_cast<int8_t>(kv->kt4[hexkl_kv_q_kt4_index(kv, n, r, d)] ^
                                   0x80u) != ek;
        bad += static_cast<int8_t>(kv->v4[hexkl_kv_q_v4_index(kv, n, r, d)] ^
                                   0x80u) != ev;
        far += std::abs(ek - static_cast<int>(std::nearbyint(qk * kr[i]))) > 1;
        moved_at_one += kr[i] == 1.0f && ek != qk;
      }
  EXPECT_EQ(bad, 0);
  EXPECT_EQ(far, 0) << "the Q15 rescale is within 1 of rint(q r)";
  EXPECT_EQ(moved_at_one, 0) << "a ratio of 1 keeps every value";
  EXPECT_NE(hexkl_kv_q_fill_q8(&tbl, h, 150, 20, kt8.data(), v8.data(), hs,
                               nullptr, nullptr, nullptr, nullptr),
            0);
  EXPECT_EQ(hexkl_kv_q_release(&tbl, h), 0);
}
