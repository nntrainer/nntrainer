// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_test_util.h
 * @date   23 Sep 2026
 * @brief  Shared helpers for the device-side HTP attention gtests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The fp16 attention test and the quantized-cache tests are separate
 * executables (test/jni/Android.mk) that share the session fixture, the
 * bit-exact fp16 conversions, the deterministic fills, the SNR gate and
 * the f32 attention reference (MHACoreLayer's math written out plainly).
 * Header-only so each keeps its own main.
 */

#ifndef __NNTRAINER_HVX_ATTN_TEST_UTIL_H__
#define __NNTRAINER_HVX_ATTN_TEST_UTIL_H__

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

namespace hvx_test {

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

/**
 * @brief Opens one unsigned-PD CDSP session per test; a failure is a FAIL,
 *        not a skip, because bringing the DSP up is part of what is tested.
 */
class SessionTest : public ::testing::Test {
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
};

} // namespace hvx_test

#endif /* __NNTRAINER_HVX_ATTN_TEST_UTIL_H__ */
