// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   unittest_hvx_attn_f16.cpp
 * @date   14 Sep 2026
 * @brief  Device test: fp16 flash attention on HMX matches the CPU reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH. See test/htp/build.sh.
 *
 * The reference is MHACoreLayer's math written out plainly in f32
 * (compute_kcaches -> softmax_row -> compute_fp16vcache in
 * Applications/CausalLM/layers/mha_core.cpp): per query row and head,
 * scores over exactly the cache rows that row may see, scaled by
 * 1/sqrt(head_dim), softmax, weighted sum of V. K and V are generated as
 * exact fp16 values so the only differences left are the DSP's fp16
 * scores, probabilities and partial sums -- which is what the SNR gate
 * measures.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

namespace {

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/**
 * @brief AEE_EBADPARM as the DSP reports it. AEEStdErr.h defines
 *        AEE_EOFFSET as 0x80000400 on Hexagon and 0 on the ARM side, so the
 *        skel's AEE_EBADPARM (0x8000040E) never equals the host macro (14);
 *        compare the low bits only.
 */
bool is_badparm(int err) {
  return (static_cast<unsigned>(err) & 0x3FFu) == 0x00Eu;
}

/** @brief IEEE binary16 -> f32, bit-exact, no compiler fp16 support needed. */
float hf_to_f32(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // subnormal: normalize
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/** @brief f32 -> IEEE binary16 bits, round to nearest even. */
uint16_t f32_to_hf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFFu) == 0xFFu) {
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t hm = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (hm & 1u))) {
      ++hm;
    }
    return static_cast<uint16_t>(sign | hm);
  }
  uint32_t hm = mant >> 13;
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) {
    ++hm;
    if (hm == 0x400u) {
      hm = 0;
      ++exp;
      if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | hm);
}

/** @brief Deterministic pseudo-random fill in [-amp, amp). */
void fill_deterministic(std::vector<float> &v, uint32_t seed, float amp) {
  uint32_t s = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>(static_cast<int32_t>(s >> 8)) /
              static_cast<float>(1 << 23) -
            1.0f) *
           amp;
  }
}

/** @brief fp16-exact fill: values are rounded to fp16 and stored as bits. */
void fill_hf(std::vector<uint16_t> &v, uint32_t seed, float amp) {
  std::vector<float> f(v.size());
  fill_deterministic(f, seed, amp);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = f32_to_hf(f[i]);
  }
}

double snr_db(const std::vector<float> &ref, const std::vector<float> &got) {
  double sig = 0.0, noise = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double r = ref[i];
    const double e = static_cast<double>(got[i]) - r;
    sig += r * r;
    noise += e * e;
  }
  if (noise == 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return 10.0 * std::log10(sig / noise);
}

struct AttnShape {
  uint32_t n_q, cache_from, cache_to, n_head_q, n_head_kv, head_dim, window;
  uint32_t br, bc;
  float softcap = 0.0f;  /**< > 0: tanh(s/softcap)*softcap on the logits */
  bool use_sink = false; /**< per-head sink logit joins the denominator */
};

/**
 * @brief The CPU attention, MHACoreLayer's rules: row q at position
 *        cache_from+q sees cache rows [max(0,pos+1-window), pos], scale is
 *        1/sqrt(head_dim), heads n*G..n*G+G-1 share KV head n.
 */
void ref_attention(const AttnShape &s, const std::vector<float> &q,
                   const std::vector<uint16_t> &k,
                   const std::vector<uint16_t> &v,
                   const std::vector<float> &sinks, std::vector<float> &out) {
  const uint32_t G = s.n_head_q / s.n_head_kv;
  const uint32_t qs = s.n_head_q * s.head_dim;
  const uint32_t ks = s.n_head_kv * s.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
  out.assign(static_cast<size_t>(s.n_q) * qs, 0.0f);
  std::vector<float> sc;
  for (uint32_t qi = 0; qi < s.n_q; ++qi) {
    const uint32_t pos = s.cache_from + qi;
    const uint32_t hi = std::min(pos + 1, s.cache_to);
    const uint32_t lo = (s.window != 0 && hi > s.window) ? hi - s.window : 0;
    for (uint32_t h = 0; h < s.n_head_q; ++h) {
      const uint32_t n = h / G;
      const float *qrow = &q[static_cast<size_t>(qi) * qs + h * s.head_dim];
      sc.assign(hi - lo, 0.0f);
      float mx = -std::numeric_limits<float>::infinity();
      for (uint32_t kk = lo; kk < hi; ++kk) {
        const uint16_t *krow =
          &k[static_cast<size_t>(kk) * ks + n * s.head_dim];
        float acc = 0.0f;
        for (uint32_t d = 0; d < s.head_dim; ++d) {
          acc += qrow[d] * hf_to_f32(krow[d]);
        }
        float sv = acc * scale;
        if (s.softcap > 0.0f) {
          sv = std::tanh(sv / s.softcap) * s.softcap;
        }
        sc[kk - lo] = sv;
        mx = std::max(mx, sv);
      }
      // MHACoreLayer's sink: one more logit per head in the softmax, with
      // no value row behind it -- it only takes probability mass away.
      if (s.use_sink) {
        mx = std::max(mx, sinks[h]);
      }
      double den = s.use_sink ? std::exp(sinks[h] - mx) : 0.0;
      for (auto &x : sc) {
        x = std::exp(x - mx);
        den += x;
      }
      float *orow = &out[static_cast<size_t>(qi) * qs + h * s.head_dim];
      for (uint32_t kk = lo; kk < hi; ++kk) {
        const float p = static_cast<float>(sc[kk - lo] / den);
        const uint16_t *vrow =
          &v[static_cast<size_t>(kk) * ks + n * s.head_dim];
        for (uint32_t d = 0; d < s.head_dim; ++d) {
          orow[d] += p * hf_to_f32(vrow[d]);
        }
      }
    }
  }
}

constexpr int kStatCount = 8;
const char *const kStatNames[kStatCount] = {
  "qprep", "dma", "tile", "qk", "softmax", "pv", "store", "n_blocks"};

/**
 * @brief Opens one unsigned-PD CDSP session per test; a failure is a FAIL,
 *        not a skip, because bringing the DSP up is part of what is tested.
 */
class HmxAttnF16 : public ::testing::Test {
protected:
  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;

  /**
   * @brief Runs one shape on the DSP and checks it against the reference.
   *
   * @param min_snr_db  gate; fp16 scores/probabilities/partial sums bound
   *                    this from above at roughly 55-65 dB on random data,
   *                    so 40 dB catches any structural error (wrong lane,
   *                    wrong row, missed block) without tripping on
   *                    rounding.
   */
  void RunShape(const AttnShape &s, double min_snr_db, bool report = false) {
    SCOPED_TRACE(
      "n_q=" + std::to_string(s.n_q) + " from=" + std::to_string(s.cache_from) +
      " to=" + std::to_string(s.cache_to) + " hq=" +
      std::to_string(s.n_head_q) + " hkv=" + std::to_string(s.n_head_kv) +
      " hd=" + std::to_string(s.head_dim) + " win=" + std::to_string(s.window) +
      " br=" + std::to_string(s.br) + " bc=" + std::to_string(s.bc));
    const size_t q_elems = static_cast<size_t>(s.n_q) * s.n_head_q * s.head_dim;
    const size_t kv_elems =
      static_cast<size_t>(s.cache_to) * s.n_head_kv * s.head_dim;

    // Q at +-3 so the scores spread enough for the running max to move
    // between blocks; a near-uniform softmax would not exercise rescaling.
    std::vector<float> q(q_elems);
    fill_deterministic(q, 0xA77E0001u, 3.0f);
    std::vector<uint16_t> k(kv_elems), v(kv_elems);
    fill_hf(k, 0xA77E0002u, 1.0f);
    fill_hf(v, 0xA77E0003u, 1.0f);

    // Sinks around 0 with the same spread as a typical logit, so they
    // actually compete for probability mass instead of vanishing.
    std::vector<float> sinks;
    if (s.use_sink) {
      sinks.resize(s.n_head_q);
      fill_deterministic(sinks, 0xA77E0004u, 2.0f);
    }

    std::vector<float> want;
    ref_attention(s, q, k, v, sinks, want);

    std::vector<float> got(q_elems, 0.0f);
    std::vector<uint32_t> stats(kStatCount, 0);
    const int err = nntr_hvx_attn_f16_prefill(
      handle_, s.n_q, s.cache_from, s.cache_to, s.n_head_q, s.n_head_kv,
      s.head_dim, s.window, s.br, s.bc, s.softcap, q.data(),
      static_cast<int>(q.size()), k.data(), static_cast<int>(k.size()),
      v.data(), static_cast<int>(v.size()), sinks.data(),
      static_cast<int>(sinks.size()), got.data(), static_cast<int>(got.size()),
      stats.data(), kStatCount);
    ASSERT_EQ(err, AEE_SUCCESS) << "attn_f16_prefill failed: " << hex(err);

    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at " << i;
    }
    const double snr = snr_db(want, got);
    EXPECT_GT(snr, min_snr_db) << "SNR " << snr << " dB";
    if (report) {
      std::cout << "ATTN_F16_FIELD shape=" << s.n_q << "x" << s.cache_to << "x"
                << s.n_head_q << "/" << s.n_head_kv << "x" << s.head_dim
                << " field=snr_db value=" << snr << "\n";
      for (int i = 0; i < kStatCount; ++i) {
        std::cout << "ATTN_F16_FIELD shape=" << s.n_q << "x" << s.cache_to
                  << "x" << s.n_head_q << "/" << s.n_head_kv << "x"
                  << s.head_dim << " field=" << kStatNames[i]
                  << " value=" << stats[i] << "\n";
      }
    }
  }
};

/**
 * Every hand-written tile builder must reproduce HexKL's converter byte
 * for byte, and fp16 AH must equal fp16 WH (the O feed-back trick depends
 * on it). Any zero flag here means the kernel's layouts are wrong on this
 * part and nothing after this test can be trusted.
 */
TEST_F(HmxAttnF16, ProbeLayoutsMatchHexkl) {
  std::vector<uint16_t> x(1024), w(1024), wt(1024);
  fill_hf(x, 0x50B30001u, 1.0f);
  fill_hf(w, 0x50B30002u, 1.0f);
  for (uint32_t r = 0; r < 32; ++r) {
    for (uint32_t c = 0; c < 32; ++c) {
      wt[c * 32 + r] = w[r * 32 + c];
    }
  }
  constexpr int kFlags = 7;
  std::vector<uint32_t> flags(kFlags, 99);
  std::vector<uint16_t> y_hexkl(1024), y_hand(1024);
  const int err = nntr_hvx_probe_f16_layouts(
    handle_, x.data(), 1024, w.data(), 1024, wt.data(), 1024, flags.data(),
    kFlags, y_hexkl.data(), 1024, y_hand.data(), 1024);
  ASSERT_EQ(err, AEE_SUCCESS)
    << "probe failed: " << hex(err) << " (AEE_EUNSUPPORTED = no fp16 HMX)";
  const char *const names[kFlags] = {"ah_hand_eq_hexkl", "wh_eq_ah",
                                     "wt_hand_eq_hexkl", "diag_hand_eq_hexkl",
                                     "rows_roundtrip",   "f32_fused_eq_hexkl",
                                     "widen_even_to_lo"};
  for (int i = 0; i < kFlags; ++i) {
    EXPECT_EQ(flags[i], 1u) << names[i];
    std::cout << "ATTN_F16_FIELD probe=" << names[i] << " value=" << flags[i]
              << "\n";
  }

  // Both matmul read-backs against the host: fp16 inputs, wide accumulate.
  std::vector<float> want(1024, 0.0f), g1(1024), g2(1024);
  for (uint32_t r = 0; r < 32; ++r) {
    for (uint32_t c = 0; c < 32; ++c) {
      double acc = 0.0;
      for (uint32_t kk = 0; kk < 32; ++kk) {
        acc += static_cast<double>(hf_to_f32(x[r * 32 + kk])) *
               hf_to_f32(w[kk * 32 + c]);
      }
      want[r * 32 + c] = static_cast<float>(acc);
    }
  }
  for (size_t i = 0; i < 1024; ++i) {
    g1[i] = hf_to_f32(y_hexkl[i]);
    g2[i] = hf_to_f32(y_hand[i]);
  }
  EXPECT_GT(snr_db(want, g1), 50.0) << "HexKL-tiled matmul off";
  EXPECT_GT(snr_db(want, g2), 50.0) << "hand-tiled matmul off";
}

/**
 * The HVX primitives both kernels share, each against the host: the
 * f32->fp16 natural-order narrowing, exp2, softcap, the 32-lane f32 sum,
 * the per-parity and full fp16 maxima, and the widening dot product.
 * Reported per element so a wrong lane mapping and a wrong value read
 * differently.
 */
TEST_F(HmxAttnF16, ProbeMathPrimitives) {
  std::vector<float> x(64);
  for (int i = 0; i < 64; ++i) {
    // -24..0 across the vector, with a few positives and exact points.
    x[i] = -24.0f + 24.0f * static_cast<float>(i) / 63.0f;
  }
  x[0] = 0.0f;
  x[1] = -1.0f;
  x[2] = -0.5f;
  x[3] = -3.25f;
  x[4] = 2.0f;
  x[5] = 0.75f;
  std::vector<uint16_t> hf_x(64), exp2_x(64), softcap_x(64);
  std::vector<float> misc(9, 0.0f);
  const int err = nntr_hvx_probe_f16_math(handle_, x.data(), 64, hf_x.data(),
                                          64, exp2_x.data(), 64,
                                          softcap_x.data(), 64, misc.data(), 9);
  ASSERT_EQ(err, AEE_SUCCESS) << hex(err);

  int bad_hf = 0, bad_exp = 0, bad_cap = 0;
  const float cap = 30.0f * 1.4426950408889634f;
  for (int i = 0; i < 64; ++i) {
    const float hx = hf_to_f32(f32_to_hf(x[i]));
    if (hf_x[i] != f32_to_hf(x[i])) {
      if (bad_hf++ < 4)
        std::cout << "hf_x[" << i << "]=" << hf_to_f32(hf_x[i]) << " want "
                  << hx << "\n";
    }
    const float e_got = hf_to_f32(exp2_x[i]);
    const float e_want = std::exp2(std::max(hx, -24.0f));
    if (std::abs(e_got - e_want) > 2e-3f * std::max(1.0f, e_want)) {
      if (bad_exp++ < 6)
        std::cout << "exp2[" << i << "] x=" << hx << " got " << e_got
                  << " want " << e_want << "\n";
    }
    const float c_got = hf_to_f32(softcap_x[i]);
    const float c_want = cap * std::tanh(hx / cap);
    if (std::abs(c_got - c_want) > 2e-2f * std::max(1.0f, std::abs(c_want))) {
      if (bad_cap++ < 6)
        std::cout << "softcap[" << i << "] x=" << hx << " got " << c_got
                  << " want " << c_want << "\n";
    }
  }
  EXPECT_EQ(bad_hf, 0) << "f32x64_to_hf lane/values";
  EXPECT_EQ(bad_exp, 0) << "exp2_hf";
  EXPECT_EQ(bad_cap, 0) << "softcap_hf";

  double sum = 0.0, dot = 0.0;
  float mx_even = -1e30f, mx_odd = -1e30f, mx_all = -1e30f;
  for (int i = 0; i < 64; ++i) {
    const float hx = hf_to_f32(f32_to_hf(x[i]));
    if (i < 32)
      sum += x[i];
    dot += static_cast<double>(hx) * hx;
    mx_all = std::max(mx_all, hx);
    if (i & 1)
      mx_odd = std::max(mx_odd, hx);
    else
      mx_even = std::max(mx_even, hx);
  }
  EXPECT_NEAR(misc[0], sum, 1e-3 * std::abs(sum) + 1e-4) << "sum32_sf";
  EXPECT_EQ(misc[1], mx_even) << "pairmax even";
  EXPECT_EQ(misc[2], mx_odd) << "pairmax odd";
  EXPECT_EQ(misc[3], mx_all) << "max64";
  EXPECT_NEAR(misc[4], dot, 2e-3 * dot) << "widening dot";
  EXPECT_NEAR(misc[5], std::exp2(hf_to_f32(f32_to_hf(x[0]))), 2e-3)
    << "exp2 via scalar hf splat";
  EXPECT_NEAR(misc[6], 8.0f, 1e-5f) << "DSP sqrtf(64)";
  EXPECT_NEAR(misc[7], 1.4426950408889634f / std::sqrt(128.0f), 1e-6f)
    << "DSP scale for hd=128";
  EXPECT_NEAR(misc[8], hf_to_f32(f32_to_hf(x[1] * 1.4426950408889634f / 8.0f)),
              2e-4f)
    << "scaled-Q narrowing lane 0";
  std::cout << "ATTN_F16_FIELD probe=scale sqrtf64=" << misc[6]
            << " scale128=" << misc[7] << " qscaled=" << misc[8] << "\n";
  std::cout << "ATTN_F16_FIELD probe=math bad_hf=" << bad_hf
            << " bad_exp2=" << bad_exp << " bad_softcap=" << bad_cap
            << " sum=" << misc[0] << "/" << sum << " dot=" << misc[4] << "/"
            << dot << "\n";
}

TEST_F(HmxAttnF16, ProbeSubtraction) {
  std::vector<float> x(64);
  for (int i = 0; i < 64; ++i) {
    x[i] = -6.0f + 12.0f * static_cast<float>(i) / 63.0f; // -6..6
  }
  const float c = 2.5f;
  std::vector<uint16_t> s_hf(64), s_qf(64), e_hf(64);
  const int err = nntr_hvx_probe_f16_sub(handle_, x.data(), 64, c, s_hf.data(),
                                         64, s_qf.data(), 64, e_hf.data(), 64);
  ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
  int bad_hf = 0, bad_qf = 0, bad_e = 0;
  for (int i = 0; i < 64; ++i) {
    const float hx = hf_to_f32(f32_to_hf(x[i]));
    const float hc = hf_to_f32(f32_to_hf(c));
    const float want = hx - hc;
    const float g1 = hf_to_f32(s_hf[i]), g2 = hf_to_f32(s_qf[i]);
    if (std::abs(g1 - want) > 4e-3f * std::max(1.0f, std::abs(want))) {
      if (bad_hf++ < 5)
        std::cout << "vsub_hf[" << i << "] x=" << hx << " got " << g1
                  << " want " << want << "\n";
    }
    if (std::abs(g2 - want) > 4e-3f * std::max(1.0f, std::abs(want))) {
      if (bad_qf++ < 5)
        std::cout << "vsub_qf16[" << i << "] x=" << hx << " got " << g2
                  << " want " << want << "\n";
    }
    const float e_want = std::exp2(std::max(want, -24.0f));
    const float e_got = hf_to_f32(e_hf[i]);
    if (std::abs(e_got - e_want) > 4e-3f * std::max(1.0f, e_want)) {
      if (bad_e++ < 5)
        std::cout << "exp2(vsub_hf)[" << i << "] d=" << want << " got " << e_got
                  << " want " << e_want << "\n";
    }
  }
  std::cout << "ATTN_F16_FIELD probe=sub bad_vsub_hf=" << bad_hf
            << " bad_vsub_qf16=" << bad_qf << " bad_exp2_sub=" << bad_e << "\n";
  EXPECT_EQ(bad_hf, 0);
  EXPECT_EQ(bad_qf, 0);
  EXPECT_EQ(bad_e, 0);
}

TEST_F(HmxAttnF16, PrefillFromEmptyCacheGqaMultiBlock) {
  // Qwen3-like: 16 query heads over 4 KV heads, hd 128. br=16 -> g_br=64,
  // bc=64 -> query blocks straddle cache blocks, exercising both the
  // diagonal mask and rows that see nothing in their last block.
  RunShape({128, 0, 128, 16, 4, 128, 0, 16, 64}, 40.0);
}

TEST_F(HmxAttnF16, PrefillAppendedToExistingCacheMha) {
  // MHA (G=1), appended after 200 cached rows: fully visible blocks first,
  // then the diagonal region; br=32 fills a tile exactly.
  RunShape({64, 200, 264, 8, 8, 64, 0, 32, 32}, 40.0);
}

TEST_F(HmxAttnF16, DecodeSingleRowRaggedCache) {
  // One query row at position 511 over 512 cached rows with bc=128:
  // exercises pad rows (G*br=4 of 32) and the un-ragged 4 full blocks.
  RunShape({1, 511, 512, 16, 4, 128, 0, 1, 128}, 40.0);
  // And a cache length that is not a block multiple: 500 rows, bc=128 ->
  // the last block lands 116 rows and zero-fills the rest.
  RunShape({1, 499, 500, 16, 4, 128, 0, 1, 128}, 40.0);
}

TEST_F(HmxAttnF16, SlidingWindowMasksTheLowerBound) {
  RunShape({48, 100, 148, 4, 2, 64, 64, 8, 32}, 40.0);
}

TEST_F(HmxAttnF16, GroupOfThreeRoundsRowsUp) {
  // G=3, br=16 -> 48 real rows padded to 64.
  RunShape({40, 0, 40, 24, 8, 64, 0, 16, 32}, 40.0);
}

TEST_F(HmxAttnF16, RejectsBadShapes) {
  std::vector<float> q(16 * 100, 0.0f), out(16 * 100, 0.0f);
  std::vector<uint16_t> k(16 * 100, 0), v(16 * 100, 0);
  std::vector<uint32_t> stats(kStatCount, 0);
  // head_dim 100 is not a tile multiple.
  int err = nntr_hvx_attn_f16_prefill(handle_, 16, 0, 16, 1, 1, 100, 0, 16, 32,
                                      0.0f, q.data(), 1600, k.data(), 1600,
                                      v.data(), 1600, nullptr, 0, out.data(),
                                      1600, stats.data(), kStatCount);
  EXPECT_TRUE(is_badparm(err)) << hex(err);
  // Length mismatch.
  err = nntr_hvx_attn_f16_prefill(handle_, 16, 0, 16, 1, 1, 64, 0, 16, 32, 0.0f,
                                  q.data(), 1024, k.data(), 1600, v.data(),
                                  1600, nullptr, 0, out.data(), 1024,
                                  stats.data(), kStatCount);
  EXPECT_TRUE(is_badparm(err)) << hex(err);
  // Sinks must be empty or exactly n_head_q long.
  std::vector<float> two_sinks(2, 0.0f);
  err = nntr_hvx_attn_f16_prefill(handle_, 16, 0, 16, 1, 1, 64, 0, 16, 32, 0.0f,
                                  q.data(), 1024, k.data(), 1024, v.data(),
                                  1024, two_sinks.data(), 2, out.data(), 1024,
                                  stats.data(), kStatCount);
  EXPECT_TRUE(is_badparm(err)) << hex(err);
}

TEST_F(HmxAttnF16, SinkJoinsTheDenominator) {
  // GPT-OSS style: 8 heads over 8 KV heads, 64-dim, per-head sinks.
  AttnShape s{64, 0, 64, 8, 8, 64, 0, 16, 32};
  s.use_sink = true;
  RunShape(s, 40.0);
}

TEST_F(HmxAttnF16, SoftcapSquashesTheLogits) {
  // Gemma style softcap of 30 on a shape whose Q amplitude pushes raw
  // logits well past the linear region of tanh.
  AttnShape s{64, 0, 64, 16, 4, 128, 0, 16, 32};
  s.softcap = 30.0f;
  RunShape(s, 40.0);
  // Both together.
  s.use_sink = true;
  RunShape(s, 40.0);
}

/**
 * Prints the per-phase DSP times for the shape the plan's worked example
 * uses and for a longer context. Reported, not gated: the numbers exist to
 * be compared against the prior branch's fp16 bench (Q.K^T 553 us/layer,
 * P.V 371 us/layer at prefill-128, kv=1024) and, later, llama.cpp.
 */
/**
 * The pure-HVX decode kernel: same reference, same gates. It accumulates
 * the output in f32, so it should sit a little above the HMX path's SNR on
 * the same shape.
 */
class HvxAttnDecodeF16 : public HmxAttnF16 {
protected:
  void RunDecode(const AttnShape &s, double min_snr_db, bool report = false) {
    SCOPED_TRACE("decode n_q=" + std::to_string(s.n_q) +
                 " from=" + std::to_string(s.cache_from) +
                 " to=" + std::to_string(s.cache_to) +
                 " hq=" + std::to_string(s.n_head_q) +
                 " hkv=" + std::to_string(s.n_head_kv) +
                 " hd=" + std::to_string(s.head_dim) +
                 " win=" + std::to_string(s.window));
    const size_t q_elems = static_cast<size_t>(s.n_q) * s.n_head_q * s.head_dim;
    const size_t kv_elems =
      static_cast<size_t>(s.cache_to) * s.n_head_kv * s.head_dim;
    std::vector<float> q(q_elems);
    fill_deterministic(q, 0xDEC0DE01u, 3.0f);
    std::vector<uint16_t> k(kv_elems), v(kv_elems);
    fill_hf(k, 0xDEC0DE02u, 1.0f);
    fill_hf(v, 0xDEC0DE03u, 1.0f);
    std::vector<float> sinks;
    if (s.use_sink) {
      sinks.resize(s.n_head_q);
      fill_deterministic(sinks, 0xDEC0DE04u, 2.0f);
    }
    std::vector<float> want;
    ref_attention(s, q, k, v, sinks, want);

    std::vector<float> got(q_elems, 0.0f);
    std::vector<uint32_t> stats(1, 0);
    const int err = nntr_hvx_attn_f16_decode(
      handle_, s.n_q, s.cache_from, s.cache_to, s.n_head_q, s.n_head_kv,
      s.head_dim, s.window, s.softcap, q.data(), static_cast<int>(q.size()),
      k.data(), static_cast<int>(k.size()), v.data(),
      static_cast<int>(v.size()), sinks.data(), static_cast<int>(sinks.size()),
      got.data(), static_cast<int>(got.size()), stats.data(), 1);
    ASSERT_EQ(err, AEE_SUCCESS) << "attn_f16_decode failed: " << hex(err);
    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i])) << "non-finite output at " << i;
    }
    const double snr = snr_db(want, got);
    EXPECT_GT(snr, min_snr_db) << "SNR " << snr << " dB";
    if (report) {
      std::cout << "ATTN_F16_FIELD path=decode shape=" << s.n_q << "x"
                << s.cache_to << "x" << s.n_head_q << "/" << s.n_head_kv << "x"
                << s.head_dim << " field=snr_db value=" << snr << "\n";
      std::cout << "ATTN_F16_FIELD path=decode shape=" << s.n_q << "x"
                << s.cache_to << "x" << s.n_head_q << "/" << s.n_head_kv << "x"
                << s.head_dim << " field=total_us value=" << stats[0] << "\n";
    }
  }
};

TEST_F(HvxAttnDecodeF16, SingleTokenGqa) {
  RunDecode({1, 511, 512, 16, 4, 128, 0, 0, 0}, 40.0);
  // Ragged: 500 rows is not a block multiple.
  RunDecode({1, 499, 500, 16, 4, 128, 0, 0, 0}, 40.0);
}

TEST_F(HvxAttnDecodeF16, FewTokensHeadDim64And96) {
  RunDecode({4, 300, 304, 8, 8, 64, 0, 0, 0}, 40.0);
  RunDecode({3, 100, 103, 8, 2, 96, 0, 0, 0}, 40.0);
}

TEST_F(HvxAttnDecodeF16, WindowSinkSoftcap) {
  AttnShape s{2, 700, 702, 8, 4, 128, 256, 0, 0};
  RunDecode(s, 40.0);
  s.use_sink = true;
  RunDecode(s, 40.0);
  s.softcap = 30.0f;
  RunDecode(s, 40.0);
}

/**
 * Two keys: out = p0 v0 + (1 - p0) v1, so p0 can be read back from the
 * output per dimension. Prints, per head, the reference p0, the p0 the
 * DSP output implies (median over dims), and how far the output sits off
 * the v0-v1 line -- which separates "wrong probability" from "wrong
 * value path".
 */
TEST_F(HvxAttnDecodeF16, TwoKeyDiagnostic) {
  const AttnShape s{1, 1, 2, 4, 4, 64, 0, 0, 0};
  const uint32_t hd = 64, nh = 4;
  std::vector<float> q(nh * hd);
  fill_deterministic(q, 0xD1A60001u, 3.0f);
  std::vector<uint16_t> k(2 * nh * hd), v(2 * nh * hd);
  fill_hf(k, 0xD1A60002u, 1.0f);
  fill_hf(v, 0xD1A60003u, 1.0f);
  std::vector<float> sinks, want;
  ref_attention(s, q, k, v, sinks, want);
  std::vector<float> got(nh * hd, 0.0f);
  std::vector<uint32_t> stats(1, 0);
  ASSERT_EQ(nntr_hvx_attn_f16_decode(
              handle_, 1, 1, 2, nh, nh, hd, 0, 0.0f, q.data(),
              static_cast<int>(q.size()), k.data(), static_cast<int>(k.size()),
              v.data(), static_cast<int>(v.size()), nullptr, 0, got.data(),
              static_cast<int>(got.size()), stats.data(), 1),
            AEE_SUCCESS);
  for (uint32_t h = 0; h < nh; ++h) {
    // Reference scores and p0.
    float s0 = 0.f, s1 = 0.f;
    for (uint32_t d = 0; d < hd; ++d) {
      s0 += q[h * hd + d] * hf_to_f32(k[0 * nh * hd + h * hd + d]);
      s1 += q[h * hd + d] * hf_to_f32(k[1 * nh * hd + h * hd + d]);
    }
    s0 /= 8.0f;
    s1 /= 8.0f;
    const float p0_ref = 1.0f / (1.0f + std::exp(s1 - s0));
    std::vector<float> p0_impl;
    double off_line = 0.0;
    for (uint32_t d = 0; d < hd; ++d) {
      const float v0 = hf_to_f32(v[0 * nh * hd + h * hd + d]);
      const float v1 = hf_to_f32(v[1 * nh * hd + h * hd + d]);
      const float o = got[h * hd + d];
      if (std::abs(v0 - v1) > 0.2f) {
        p0_impl.push_back((o - v1) / (v0 - v1));
      }
      // distance from the segment's line in 1-D is just the residual of
      // the best single p0 -- estimate after the loop.
      (void)off_line;
    }
    std::sort(p0_impl.begin(), p0_impl.end());
    const float p0_med = p0_impl[p0_impl.size() / 2];
    double resid = 0.0;
    for (uint32_t d = 0; d < hd; ++d) {
      const float v0 = hf_to_f32(v[0 * nh * hd + h * hd + d]);
      const float v1 = hf_to_f32(v[1 * nh * hd + h * hd + d]);
      const float o = got[h * hd + d];
      const float e = o - (p0_med * v0 + (1.0f - p0_med) * v1);
      resid += e * e;
    }
    std::cout << "ATTN_F16_FIELD diag=kv2 head=" << h << " s0=" << s0
              << " s1=" << s1 << " p0_ref=" << p0_ref
              << " p0_dsp_median=" << p0_med
              << " rms_off_line=" << std::sqrt(resid / hd)
              << " out0=" << got[h * hd] << " ref0=" << want[h * hd]
              << " v0=" << hf_to_f32(v[h * hd])
              << " v1=" << hf_to_f32(v[nh * hd + h * hd]) << "\n";
  }
}

TEST_F(HvxAttnDecodeF16, BlockProbeScoresAndProbabilities) {
  const uint32_t hd = 64, kv = 2, nh = 4;
  std::vector<float> q(nh * hd);
  fill_deterministic(q, 0xD1A60001u, 3.0f);
  std::vector<uint16_t> k(kv * nh * hd);
  fill_hf(k, 0xD1A60002u, 1.0f);
  for (uint32_t h = 0; h < nh; ++h) {
    std::vector<float> qh(q.begin() + h * hd, q.begin() + (h + 1) * hd);
    std::vector<uint16_t> kh(kv * hd);
    for (uint32_t j = 0; j < kv; ++j) {
      std::copy(k.begin() + (j * nh + h) * hd,
                k.begin() + (j * nh + h + 1) * hd, kh.begin() + j * hd);
    }
    std::vector<float> sc(64), misc(3);
    std::vector<uint16_t> sv(64), p(64);
    ASSERT_EQ(nntr_hvx_probe_f16_decode_block(
                handle_, hd, kv, qh.data(), hd, kh.data(), kv * hd, sc.data(),
                64, sv.data(), 64, p.data(), 64, misc.data(), 3),
              AEE_SUCCESS);
    for (uint32_t j = 0; j < kv; ++j) {
      // Host: the same dot with Q rounded to fp16 after scaling.
      double s_exact = 0.0, s_hfq = 0.0;
      for (uint32_t d = 0; d < hd; ++d) {
        const float kd = hf_to_f32(kh[j * hd + d]);
        s_exact += static_cast<double>(qh[d]) * misc[2] * kd;
        s_hfq +=
          static_cast<double>(hf_to_f32(f32_to_hf(qh[d] * misc[2]))) * kd;
      }
      std::cout << "ATTN_F16_FIELD diag=block head=" << h << " j=" << j
                << " sc_dsp=" << sc[j] << " sc_host=" << s_exact
                << " sc_host_hfq=" << s_hfq << " sv_hf=" << hf_to_f32(sv[j])
                << " p_hf=" << hf_to_f32(p[j]) << "\n";
    }
    std::cout << "ATTN_F16_FIELD diag=block head=" << h << " m=" << misc[0]
              << " rowsum=" << misc[1] << " scale=" << misc[2]
              << " sv2=" << hf_to_f32(sv[2]) << " p2=" << hf_to_f32(p[2])
              << "\n";
  }
}

TEST_F(HvxAttnDecodeF16, FirstTokenSeesOnlyItself) {
  RunDecode({1, 0, 1, 16, 4, 128, 0, 0, 0}, 40.0);
}

TEST_F(HvxAttnDecodeF16, TinyCachesShowWhereErrorStarts) {
  // kv = 2 (two keys: softmax of one difference), 3, 8, 63, 64, 65, 200.
  // Reported, not gated: the SNR trend against kv length separates a
  // per-element score error from a block-boundary or normalization one.
  for (uint32_t kv : {2u, 3u, 8u, 63u, 64u, 65u, 200u}) {
    RunDecode({1, kv - 1, kv, 4, 4, 64, 0, 0, 0}, -1000.0, /*report=*/true);
  }
}

TEST_F(HvxAttnDecodeF16, ReportDecodeTime) {
  RunDecode({1, 1023, 1024, 16, 4, 128, 0, 0, 0}, 40.0, /*report=*/true);
  RunDecode({1, 4095, 4096, 16, 4, 128, 0, 0, 0}, 40.0, /*report=*/true);
}

/**
 * Resident tiles: the same rows appended once into a registered tile cache
 * must give the SAME bytes as the raw-row path -- the tiles the registry
 * writes by scalar scatter and the tiles the kernel builds on HVX are one
 * layout, and everything after them is identical arithmetic.
 */
class HmxAttnF16Resident : public HmxAttnF16 {
protected:
  void RunBoth(const AttnShape &s, uint32_t max_rows, bool report = false) {
    SCOPED_TRACE("resident n_q=" + std::to_string(s.n_q) +
                 " to=" + std::to_string(s.cache_to) +
                 " max_rows=" + std::to_string(max_rows) +
                 " hq=" + std::to_string(s.n_head_q) +
                 " hkv=" + std::to_string(s.n_head_kv) + " hd=" +
                 std::to_string(s.head_dim) + " bc=" + std::to_string(s.bc));
    const size_t q_elems = static_cast<size_t>(s.n_q) * s.n_head_q * s.head_dim;
    const uint32_t stride = s.n_head_kv * s.head_dim;
    const size_t kv_elems = static_cast<size_t>(s.cache_to) * stride;
    std::vector<float> q(q_elems);
    fill_deterministic(q, 0x4E574B01u, 3.0f);
    std::vector<uint16_t> k(kv_elems), v(kv_elems);
    fill_hf(k, 0x4E574B02u, 1.0f);
    fill_hf(v, 0x4E574B03u, 1.0f);
    std::vector<float> sinks;
    std::vector<float> want;
    ref_attention(s, q, k, v, sinks, want);

    // Raw path.
    std::vector<float> raw(q_elems, 0.0f), res(q_elems, 0.0f);
    std::vector<uint32_t> stats_raw(kStatCount, 0), stats_res(kStatCount, 0);
    int err = nntr_hvx_attn_f16_prefill(
      handle_, s.n_q, s.cache_from, s.cache_to, s.n_head_q, s.n_head_kv,
      s.head_dim, s.window, s.br, s.bc, s.softcap, q.data(),
      static_cast<int>(q.size()), k.data(), static_cast<int>(k.size()),
      v.data(), static_cast<int>(v.size()), nullptr, 0, raw.data(),
      static_cast<int>(raw.size()), stats_raw.data(), kStatCount);
    ASSERT_EQ(err, AEE_SUCCESS) << "raw prefill failed: " << hex(err);

    // Resident path: register, append in two uneven pieces (exercises
    // row0 != 0 and rows that do not start on a tile boundary), run.
    uint32_t kvh = 0xFFFFFFFFu;
    err = nntr_hvx_kv_register_f16(handle_, max_rows, s.n_head_kv, s.head_dim,
                                   &kvh);
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_register failed: " << hex(err);
    const uint32_t split = s.cache_to > 37 ? 37 : s.cache_to / 2;
    err = nntr_hvx_kv_append_f16(handle_, kvh, 0, k.data(),
                                 static_cast<int>(split * stride), v.data(),
                                 static_cast<int>(split * stride));
    ASSERT_EQ(err, AEE_SUCCESS) << "kv_append #1 failed: " << hex(err);
    if (split < s.cache_to) {
      err = nntr_hvx_kv_append_f16(
        handle_, kvh, split, k.data() + static_cast<size_t>(split) * stride,
        static_cast<int>((s.cache_to - split) * stride),
        v.data() + static_cast<size_t>(split) * stride,
        static_cast<int>((s.cache_to - split) * stride));
      ASSERT_EQ(err, AEE_SUCCESS) << "kv_append #2 failed: " << hex(err);
    }
    err = nntr_hvx_attn_f16_prefill_resident(
      handle_, kvh, s.n_q, s.cache_from, s.cache_to, s.n_head_q, s.window, s.br,
      s.bc, s.softcap, q.data(), static_cast<int>(q.size()), nullptr, 0,
      res.data(), static_cast<int>(res.size()), stats_res.data(), kStatCount);
    ASSERT_EQ(err, AEE_SUCCESS) << "resident prefill failed: " << hex(err);
    EXPECT_EQ(nntr_hvx_kv_release_f16(handle_, kvh), AEE_SUCCESS);

    // Bit-identical to the raw path, and both correct.
    size_t n_diff = 0;
    for (size_t i = 0; i < q_elems; ++i) {
      if (std::memcmp(&raw[i], &res[i], sizeof(float)) != 0) {
        ++n_diff;
      }
    }
    EXPECT_EQ(n_diff, 0u) << "resident differs from raw in " << n_diff << " of "
                          << q_elems << " outputs";
    EXPECT_GT(snr_db(want, res), 40.0);
    if (report) {
      for (int i = 0; i < kStatCount; ++i) {
        std::cout << "ATTN_F16_FIELD path=raw field=" << kStatNames[i]
                  << " value=" << stats_raw[i] << "\n";
        std::cout << "ATTN_F16_FIELD path=resident field=" << kStatNames[i]
                  << " value=" << stats_res[i] << "\n";
      }
    }
  }
};

TEST_F(HmxAttnF16Resident, MatchesRawPathBitForBit) {
  RunBoth({128, 0, 128, 16, 4, 128, 0, 16, 64}, 128);
  RunBoth({64, 200, 264, 8, 8, 64, 0, 32, 32}, 512);
}

TEST_F(HmxAttnF16Resident, BlockPastRegistryEndIsZeroFilled) {
  // max_rows 320 (10 col tiles) with bc 96 (3 col tiles): the last block
  // covers tiles 9..11, of which 10 and 11 do not exist and must read 0.
  RunBoth({40, 280, 320, 8, 8, 64, 0, 8, 96}, 320);
}

TEST_F(HmxAttnF16Resident, RejectsCacheBeyondRegistration) {
  uint32_t kvh = 0;
  ASSERT_EQ(nntr_hvx_kv_register_f16(handle_, 64, 2, 64, &kvh), AEE_SUCCESS);
  std::vector<float> q(2 * 128, 0.0f), out(2 * 128, 0.0f);
  std::vector<uint32_t> stats(kStatCount, 0);
  const int err = nntr_hvx_attn_f16_prefill_resident(
    handle_, kvh, 1, 99, 100, 2, 0, 0, 0, 0.0f, q.data(), 128, nullptr, 0,
    out.data(), 128, stats.data(), kStatCount);
  EXPECT_TRUE(is_badparm(err)) << hex(err);
  EXPECT_EQ(nntr_hvx_kv_release_f16(handle_, kvh), AEE_SUCCESS);
  EXPECT_TRUE(is_badparm(nntr_hvx_kv_release_f16(handle_, kvh)));
}

TEST_F(HmxAttnF16Resident, ReportRawVsResident) {
  RunBoth({128, 896, 1024, 16, 4, 128, 0, 16, 128}, 1024, /*report=*/true);
}

TEST_F(HmxAttnF16, AutoTilingMatchesReference) {
  // br = bc = 0: the DSP chooses. Two shapes the chooser treats
  // differently (bc capped at 256 vs. forced up to 64).
  RunShape({128, 896, 1024, 16, 4, 128, 0, 0, 0}, 40.0);
  RunShape({100, 0, 100, 8, 8, 64, 0, 0, 0}, 40.0);
}

TEST_F(HmxAttnF16, ReportPhaseTimes) {
  RunShape({128, 896, 1024, 16, 4, 128, 0, 16, 128}, 40.0, /*report=*/true);
  RunShape({128, 0, 128, 16, 4, 128, 0, 16, 64}, 40.0, /*report=*/true);
}

} // namespace

/**
 * @brief Main gtest (this tree's googletest_main static library is
 *        gtest-all only; every device test carries its own main).
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
