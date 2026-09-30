// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_cpu_kv_q_attention.cpp
 * @date   29 Sep 2026
 * @brief  CPU A8W8 attention over the int8 KV registry against an exact
 *         f32 reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The scheme's own noise on uniform data is ~40 dB (int8 K and V per token,
 * uint8 Q per row, uint8 P per 32-row block: see the plan's Q2 table), so
 * the gate is 36 dB against the exact attention on every shape, plus the
 * incremental contract: appending row by row and attending equals one
 * append of everything.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <cpu_kv_q_attention.h>

namespace {

uint16_t f32_to_hf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) {
    return static_cast<uint16_t>(sign);
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  // round to nearest even on the 13 dropped bits
  const uint32_t rem = mant & 0x1FFFu;
  mant >>= 13;
  if (rem > 0x1000u || (rem == 0x1000u && (mant & 1u))) {
    ++mant;
    if (mant == 0x400u) {
      mant = 0;
      ++exp;
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               mant);
}

float hf_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t u;
  if (exp == 0) {
    if (mant == 0) {
      u = sign;
    } else {
      // subnormal
      int e = -1;
      uint32_t m = mant;
      do {
        ++e;
        m <<= 1;
      } while (!(m & 0x400u));
      u = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) |
          ((m & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    u = sign | 0x7F800000u | (mant << 13);
  } else {
    u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

struct Shape {
  unsigned n_q, cache_from, cache_to, n_head_q, n_head_kv, head_dim, window;
  float softcap = 0.0f;
  bool sink = false;
};

uint32_t lcg(uint32_t &s) {
  s = s * 1664525u + 1013904223u;
  return s;
}

float uni(uint32_t &s, float amp) {
  return (static_cast<float>(lcg(s) >> 8) / 16777216.0f * 2.0f - 1.0f) * amp;
}

void ref_attention(const Shape &s, const std::vector<float> &q,
                   const std::vector<uint16_t> &k,
                   const std::vector<uint16_t> &v,
                   const std::vector<float> &sinks, std::vector<float> &out) {
  const unsigned G = s.n_head_q / s.n_head_kv;
  const unsigned qs = s.n_head_q * s.head_dim, ks = s.n_head_kv * s.head_dim;
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
  out.assign(static_cast<size_t>(s.n_q) * qs, 0.0f);
  std::vector<double> sc;
  for (unsigned qi = 0; qi < s.n_q; ++qi) {
    const unsigned pos = s.cache_from + qi;
    const unsigned hi = std::min(pos + 1, s.cache_to);
    const unsigned lo = (s.window && hi > s.window) ? hi - s.window : 0;
    for (unsigned h = 0; h < s.n_head_q; ++h) {
      const unsigned n = h / G;
      const float *qr = &q[static_cast<size_t>(qi) * qs + h * s.head_dim];
      sc.assign(hi - lo, 0.0);
      double mx = -1e300;
      for (unsigned r = lo; r < hi; ++r) {
        double a = 0.0;
        for (unsigned d = 0; d < s.head_dim; ++d) {
          a += static_cast<double>(qr[d]) *
               hf_to_f32(k[static_cast<size_t>(r) * ks + n * s.head_dim + d]);
        }
        a *= inv_sqrt;
        if (s.softcap > 0.0f) {
          a = std::tanh(a / s.softcap) * s.softcap;
        }
        sc[r - lo] = a;
        mx = std::max(mx, a);
      }
      double den = 0.0;
      if (s.sink) {
        mx = std::max(mx, static_cast<double>(sinks[h]));
        den = std::exp(sinks[h] - mx);
      }
      for (unsigned r = lo; r < hi; ++r) {
        sc[r - lo] = std::exp(sc[r - lo] - mx);
        den += sc[r - lo];
      }
      float *o = &out[static_cast<size_t>(qi) * qs + h * s.head_dim];
      for (unsigned d = 0; d < s.head_dim; ++d) {
        double a = 0.0;
        for (unsigned r = lo; r < hi; ++r) {
          a += sc[r - lo] *
               hf_to_f32(v[static_cast<size_t>(r) * ks + n * s.head_dim + d]);
        }
        o[d] = static_cast<float>(a / den);
      }
    }
  }
}

double snr_db(const std::vector<float> &ref, const std::vector<float> &got) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(ref[i]) - got[i];
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  return 10.0 * std::log10(den / std::max(num, 1e-30));
}

void run(const Shape &s, double min_db) {
  const unsigned qs = s.n_head_q * s.head_dim, ks = s.n_head_kv * s.head_dim;
  uint32_t seed = 0xC0FFEE01u + s.cache_to * 7u + s.n_q;
  std::vector<float> q(static_cast<size_t>(s.n_q) * qs);
  for (auto &x : q) {
    x = uni(seed, 3.0f);
  }
  std::vector<uint16_t> k(static_cast<size_t>(s.cache_to) * ks), v(k.size());
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = f32_to_hf(uni(seed, 1.0f));
    v[i] = f32_to_hf(uni(seed, 1.0f));
  }
  std::vector<float> sinks(s.n_head_q, 0.0f);
  for (auto &x : sinks) {
    x = uni(seed, 2.0f);
  }
  std::vector<float> ref, got(q.size(), 0.0f);
  ref_attention(s, q, k, v, sinks, ref);

  const int h =
    nntrainer::cpu_kv_q_register(0, s.cache_to + 5, s.n_head_kv, s.head_dim);
  ASSERT_GE(h, 0);
  ASSERT_TRUE(nntrainer::cpu_sdpa_q_kvcache(
    h, 0, s.cache_to, ks, k.data(), v.data(), q.data(), qs, s.n_q, s.cache_from,
    s.cache_to, s.n_head_q, s.n_head_kv, s.head_dim, s.window, s.softcap,
    s.sink ? sinks.data() : nullptr, got.data(), qs));
  const double db = snr_db(ref, got);
  std::cout << "CPU_KV_Q n_q=" << s.n_q << " cache=" << s.cache_from << ".."
            << s.cache_to << " heads=" << s.n_head_q << "/" << s.n_head_kv
            << " hd=" << s.head_dim << " snr_db=" << db << "\n";
  EXPECT_GE(db, min_db) << "n_q=" << s.n_q << " cache_to=" << s.cache_to
                        << " hd=" << s.head_dim;

  // Incremental: the rows appended one at a time, attention over the same
  // cache, must agree exactly with the one-shot append.
  const int h2 =
    nntrainer::cpu_kv_q_register(0, s.cache_to + 5, s.n_head_kv, s.head_dim);
  ASSERT_GE(h2, 0);
  for (unsigned r = 0; r < s.cache_to; ++r) {
    ASSERT_TRUE(nntrainer::cpu_kv_q_append(
      h2, r, 1, ks, k.data() + static_cast<size_t>(r) * ks,
      v.data() + static_cast<size_t>(r) * ks));
  }
  std::vector<float> got2(q.size(), 0.0f);
  ASSERT_TRUE(nntrainer::cpu_sdpa_q_kvcache(
    h2, 0, 0, ks, nullptr, nullptr, q.data(), qs, s.n_q, s.cache_from,
    s.cache_to, s.n_head_q, s.n_head_kv, s.head_dim, s.window, s.softcap,
    s.sink ? sinks.data() : nullptr, got2.data(), qs));
  EXPECT_EQ(std::memcmp(got.data(), got2.data(), got.size() * sizeof(float)),
            0);
  nntrainer::cpu_kv_q_release(h);
  nntrainer::cpu_kv_q_release(h2);
}

} // namespace

TEST(CpuKvQAttention, DecodeGqa) { run({1, 1023, 1024, 16, 8, 128, 0}, 36.0); }

TEST(CpuKvQAttention, DecodeRaggedSmallHead) {
  run({3, 70, 73, 8, 2, 64, 0}, 36.0);
}

TEST(CpuKvQAttention, PrefillFromEmpty) {
  run({100, 0, 100, 16, 8, 128, 0}, 36.0);
}

TEST(CpuKvQAttention, PrefillAppendedStraddlingBlocks) {
  run({45, 83, 128, 4, 4, 96, 0}, 36.0);
}

TEST(CpuKvQAttention, WindowSinkSoftcap) {
  Shape s{20, 200, 220, 8, 4, 64, 64};
  s.softcap = 30.0f;
  s.sink = true;
  run(s, 36.0);
}

TEST(CpuKvQAttention, RejectsInt4AndBadShapes) {
  EXPECT_EQ(nntrainer::cpu_kv_q_register(1, 64, 2, 64), -1);
  EXPECT_EQ(nntrainer::cpu_kv_q_register(0, 64, 2, 100), -1);
  const int h = nntrainer::cpu_kv_q_register(0, 64, 2, 64);
  ASSERT_GE(h, 0);
  std::vector<uint16_t> rows(128, 0);
  EXPECT_FALSE(
    nntrainer::cpu_kv_q_append(h, 64, 1, 128, rows.data(), rows.data()));
  EXPECT_FALSE(
    nntrainer::cpu_kv_q_append(h, 0, 1, 100, rows.data(), rows.data()));
  nntrainer::cpu_kv_q_release(h);
}
