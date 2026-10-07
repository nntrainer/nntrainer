// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hvx_softmax_q.cpp
 * @date   06 Oct 2026
 * @brief  Host test of the integer softmax definition (R3 of plan 23)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Models the whole chain the kernel will run: a log2-domain logit L is
 * turned into the accumulator the HMX would hold (acc = L / alpha, plus
 * the zero-point term zq * colsum), the accumulator into the int16 the
 * convert unit writes (floor(acc * 2^k / 512) mod 2^16), and the int16
 * through hvx_softmax_q_ref. P' / rowsum is then compared with the exact
 * softmax of L.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hvx_softmax_q.h"

namespace {

constexpr uint32_t kRows = HVX_SOFTMAX_Q_ROWS;
constexpr uint32_t kTile = HVX_SOFTMAX_Q_TILE;

struct Case {
  uint32_t ct, col0, n_cols, row0, n_rows, window;
  float alpha;
  uint32_t F;
  int zq;
};

/** @brief Builds the raw int16 score tiles and corr from logits L. */
struct Built {
  std::vector<int16_t> s, corr;
  std::vector<float> L; /**< [ct*2048], the exact log2 logits */
  int k;
  uint16_t rho_q15;
};

Built build(const Case &c, std::mt19937 &rng) {
  Built b;
  EXPECT_EQ(hvx_softmax_q_scale(c.alpha, c.F, &b.k, &b.rho_q15), 0);
  const double cvt = std::ldexp(1.0, b.k) / 512.0;
  std::normal_distribution<float> logit(0.0f, 4.0f);
  std::uniform_int_distribution<int> colsum(-4000, 4000);
  b.s.assign(static_cast<size_t>(c.ct) * kTile, 0);
  b.corr.assign(static_cast<size_t>(c.ct) * 32, 0);
  b.L.assign(b.s.size(), 0.0f);
  for (uint32_t t = 0; t < c.ct; ++t) {
    for (uint32_t j = 0; j < 32; ++j) {
      const int cs = colsum(rng);
      const double zp = static_cast<double>(c.zq) * cs;
      b.corr[t * 32 + j] = static_cast<int16_t>(static_cast<uint16_t>(
        static_cast<int64_t>(std::lround(zp * cvt)) & 0xffff));
      for (uint32_t r = 0; r < kRows; ++r) {
        float Lv = logit(rng);
        if ((r * 7 + j) % 53 == 0) {
          Lv += 20.0f; // an outlier the row max must track
        }
        const int32_t acc = static_cast<int32_t>(std::lround(Lv / c.alpha));
        // The logit the integer path can represent: acc * alpha exactly.
        b.L[t * kTile + r * 32 + j] =
          static_cast<float>(acc * static_cast<double>(c.alpha));
        const double raw = std::floor((static_cast<double>(acc) + zp) * cvt);
        b.s[t * kTile + r * 32 + j] = static_cast<int16_t>(
          static_cast<uint16_t>(static_cast<int64_t>(raw) & 0xffff));
      }
    }
  }
  return b;
}

bool masked(const Case &c, uint32_t r, uint32_t col) {
  const uint32_t row = c.row0 + r;
  if (r >= c.n_rows || col >= c.col0 + c.n_cols || col > row) {
    return true;
  }
  return c.window != 0 && col + c.window <= row;
}

/** @brief What the test measures for one block. */
struct Snr {
  double p;       /**< P'/rowsum vs exact softmax */
  double p_ideal; /**< float a8 P with the same fixed 1/256 scale vs exact:
                       the bound an 8-bit P can reach */
  double o;       /**< (P'/rowsum).V vs exact softmax(L).V, random V */
};

/** @brief SNR in dB of P'/rowsum against the exact softmax, over all rows. */
Snr run(const Case &c, std::mt19937 &rng, bool check_masks) {
  Built b = build(c, rng);
  hvx_softmax_q_block blk{};
  blk.n_col_tiles = c.ct;
  blk.col0 = c.col0;
  blk.n_cols = c.n_cols;
  blk.row0 = c.row0;
  blk.n_rows = c.n_rows;
  blk.window = c.window;
  blk.rho_q15 = b.rho_q15;
  blk.frac_bits = static_cast<uint8_t>(c.F);
  std::vector<uint8_t> p(b.s.size(), 0xAA);
  std::vector<int32_t> rowsum_v(HVX_SOFTMAX_Q_ROWSUM_WORDS, -1);
  hvx_softmax_q_ref(&blk, b.s.data(), b.corr.data(), p.data(), rowsum_v.data());
  std::vector<int32_t> rowsum(kRows);
  for (uint32_t r = 0; r < kRows; ++r) {
    rowsum[r] = rowsum_v[hvx_softmax_q_rowsum_index(r)];
  }

  // A random V (one value per column, 16 head dims) for the output-level
  // comparison: O = sum_c P[c] V[c].
  constexpr uint32_t kDims = 16;
  std::vector<float> V(static_cast<size_t>(c.ct) * 32 * kDims);
  std::normal_distribution<float> vdist(0.0f, 1.0f);
  for (auto &x : V) {
    x = vdist(rng);
  }
  double sig = 0.0, err = 0.0, err_ideal = 0.0, o_sig = 0.0, o_err = 0.0;
  for (uint32_t r = 0; r < kRows; ++r) {
    // Exact softmax over the unmasked columns of this row.
    double mx = -1e30;
    for (uint32_t t = 0; t < c.ct; ++t) {
      for (uint32_t j = 0; j < 32; ++j) {
        if (!masked(c, r, c.col0 + 32 * t + j)) {
          mx = std::max(mx, static_cast<double>(b.L[t * kTile + r * 32 + j]));
        }
      }
    }
    double sum = 0.0;
    for (uint32_t t = 0; t < c.ct; ++t) {
      for (uint32_t j = 0; j < 32; ++j) {
        if (!masked(c, r, c.col0 + 32 * t + j)) {
          sum += std::exp2(b.L[t * kTile + r * 32 + j] - mx);
        }
      }
    }
    if (r >= c.n_rows) {
      EXPECT_EQ(rowsum[r], 0) << "padding row " << r;
    } else {
      EXPECT_GT(rowsum[r], 0) << "row " << r;
    }
    int32_t sum_p = 0;
    // Ideal a8 P: round(2^(L - max) * 256) in float, normalized by its own
    // sum, so the only error is the 8-bit grid itself.
    double ideal_sum = 0.0;
    for (uint32_t t = 0; t < c.ct; ++t) {
      for (uint32_t j = 0; j < 32; ++j) {
        if (!masked(c, r, c.col0 + 32 * t + j)) {
          ideal_sum += std::min(
            255.0,
            std::round(std::exp2(b.L[t * kTile + r * 32 + j] - mx) * 256.0));
        }
      }
    }
    std::vector<double> o_want(kDims, 0.0), o_got(kDims, 0.0);
    for (uint32_t t = 0; t < c.ct; ++t) {
      for (uint32_t j = 0; j < 32; ++j) {
        const size_t at = t * kTile + r * 32 + j;
        const bool m = masked(c, r, c.col0 + 32 * t + j);
        if (check_masks && m) {
          EXPECT_EQ(p[at], 0)
            << "masked entry row " << r << " col " << c.col0 + 32 * t + j;
        }
        sum_p += p[at];
        if (m || sum <= 0.0) {
          continue;
        }
        const double want = std::exp2(b.L[at] - mx) / sum;
        const double got =
          rowsum[r] > 0 ? p[at] / static_cast<double>(rowsum[r]) : 0.0;
        const double ideal =
          ideal_sum > 0.0
            ? std::min(255.0, std::round(std::exp2(b.L[at] - mx) * 256.0)) /
                ideal_sum
            : 0.0;
        sig += want * want;
        err += (want - got) * (want - got);
        err_ideal += (want - ideal) * (want - ideal);
        for (uint32_t d = 0; d < kDims; ++d) {
          const double v = V[(t * 32 + j) * kDims + d];
          o_want[d] += want * v;
          o_got[d] += got * v;
        }
      }
    }
    if (r < c.n_rows) {
      for (uint32_t d = 0; d < kDims; ++d) {
        o_sig += o_want[d] * o_want[d];
        o_err += (o_want[d] - o_got[d]) * (o_want[d] - o_got[d]);
      }
    }
    EXPECT_EQ(sum_p, rowsum[r]) << "rowsum is the sum of P' in row " << r;
  }
  Snr out;
  out.p = 10.0 * std::log10(sig / std::max(err, 1e-300));
  out.p_ideal = 10.0 * std::log10(sig / std::max(err_ideal, 1e-300));
  out.o = 10.0 * std::log10(o_sig / std::max(o_err, 1e-300));
  return out;
}

} // namespace

TEST(HvxSoftmaxQ, CubicIsExp2Within2e4) {
  // The Q15 cubic against 2^x * 16384: at most 3 Q14 units off (1.5e-4),
  // which is 0.04 of a P' step at the top of the grid.
  for (uint32_t F = 7; F <= 8; ++F) {
    int max_err = 0;
    for (uint32_t f = 0; f < (1u << F); ++f) {
      const double want = std::exp2(f / static_cast<double>(1u << F)) * 16384.0;
      const int got = hvx_softmax_q_exp2_q14(f, F);
      max_err = std::max(max_err, static_cast<int>(std::abs(got - want)));
      EXPECT_GE(got, 16384);
      EXPECT_LE(got, 32767);
    }
    EXPECT_LE(max_err, 3) << "F=" << F;
  }
  EXPECT_EQ(hvx_softmax_q_exp2_q14(0, 8), 16384);
}

TEST(HvxSoftmaxQ, ScalePutsRhoInHalfToOne) {
  const float alphas[] = {1e-4f,   3.3e-4f,          1e-3f,
                          2.5e-3f, 1.0f / 131072.0f, 0.05f};
  for (float a : alphas) {
    for (uint32_t F = 7; F <= 8; ++F) {
      int k = 0;
      uint16_t rho = 0;
      ASSERT_EQ(hvx_softmax_q_scale(a, F, &k, &rho), 0) << a;
      const double r = a * (1u << F) * 512.0 / std::ldexp(1.0, k);
      EXPECT_GT(r, 0.5) << a;
      EXPECT_LE(r, 1.0) << a;
      EXPECT_NEAR(rho / 32768.0, r, 1.0 / 32768.0 + 1e-9) << a;
    }
  }
}

TEST(HvxSoftmaxQ, CausalBlockMatchesExactSoftmax) {
  std::mt19937 rng(0x5a17);
  // 64 rows at positions 1000.., 35 tiles starting at column 0: the
  // diagonal crosses the last two tiles.
  Case c{35, 0, 1120, 1000, 64, 0, 1.0f / 1316.0f, 8, 128};
  const Snr r = run(c, rng, true);
  std::cout << "SOFTMAX_Q_FIELD case=causal F=8 p_db=" << r.p
            << " p_ideal_a8_db=" << r.p_ideal << " o_db=" << r.o << "\n";
  // The integer path must sit on the 8-bit grid's own bound, and the
  // output it feeds must keep a8-grade accuracy.
  EXPECT_GT(r.p, r.p_ideal - 1.0);
  EXPECT_GT(r.o, 30.0);
}

TEST(HvxSoftmaxQ, WindowedBlockMasksBothEdges) {
  std::mt19937 rng(0x5a18);
  // Rows 2048..2111 with window 1024 see columns 1025..2111: tiles from
  // column 992 (31 * 32) cover it, left edge partial, right edge partial.
  Case c{36, 992, 1152, 2048, 64, 1024, 1.0f / 2000.0f, 8, 128};
  const Snr r = run(c, rng, true);
  std::cout << "SOFTMAX_Q_FIELD case=window F=8 p_db=" << r.p
            << " p_ideal_a8_db=" << r.p_ideal << " o_db=" << r.o << "\n";
  EXPECT_GT(r.p, r.p_ideal - 1.0);
  EXPECT_GT(r.o, 30.0);
}

TEST(HvxSoftmaxQ, PaddingRowsAndShortColumns) {
  std::mt19937 rng(0x5a19);
  // Only 40 valid rows, and the cache ends at column 70 inside tile 2.
  Case c{3, 0, 70, 30, 40, 0, 1.0f / 1000.0f, 8, 0};
  const Snr r = run(c, rng, true);
  std::cout << "SOFTMAX_Q_FIELD case=padding F=8 p_db=" << r.p
            << " p_ideal_a8_db=" << r.p_ideal << " o_db=" << r.o << "\n";
  EXPECT_GT(r.p, r.p_ideal - 1.0);
  EXPECT_GT(r.o, 30.0);
}

TEST(HvxSoftmaxQ, SevenFractionBitsStillAccurate) {
  std::mt19937 rng(0x5a1a);
  Case c{35, 0, 1120, 1000, 64, 0, 1.0f / 1316.0f, 7, 128};
  const Snr r = run(c, rng, false);
  std::cout << "SOFTMAX_Q_FIELD case=causal F=7 p_db=" << r.p
            << " p_ideal_a8_db=" << r.p_ideal << " o_db=" << r.o << "\n";
  EXPECT_GT(r.p, r.p_ideal - 1.0);
  EXPECT_GT(r.o, 30.0);
}

TEST(HvxSoftmaxQ, FlatRowKeepsFullRange) {
  // All logits equal: the normalized P is 1/n for every column. With the
  // fixed scale on the unnormalized exponential every P' is 255, so the
  // result is exact; a normalized a8 P would have rounded 1/1024 to 0.
  hvx_softmax_q_block blk{};
  blk.n_col_tiles = 32;
  blk.col0 = 0;
  blk.n_cols = 1024;
  blk.row0 = 1023;
  blk.n_rows = 1;
  blk.window = 0;
  blk.rho_q15 = 32767;
  blk.frac_bits = 8;
  std::vector<int16_t> s(32 * kTile, 1234);
  std::vector<uint8_t> p(s.size());
  std::vector<int32_t> rowsum(HVX_SOFTMAX_Q_ROWSUM_WORDS);
  hvx_softmax_q_ref(&blk, s.data(), nullptr, p.data(), rowsum.data());
  EXPECT_EQ(rowsum[hvx_softmax_q_rowsum_index(0)], 255 * 1024);
  for (uint32_t t = 0; t < 32; ++t) {
    EXPECT_EQ(p[t * kTile], 255) << "tile " << t;
  }
}
