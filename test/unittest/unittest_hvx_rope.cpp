// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_rope.cpp
 * @brief  Device tests for the HVX uint8 RoPE endpoint.
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "mha_htp_host_model.h"
#include "nntr_hvx.h"

namespace {

constexpr int kDspOffset = 0x80000400;

// Per-tensor quant for cos and sin. Symmetric Q8: value in [-1, 1] maps to
// uint8 [1, 255], with cos(0)=1 -> 255 and sin(0)=0 -> 128 exactly. Mirrors
// what QNN ships for its RoPE lowering (separate scale/zp per cos/sin
// tensor, see node_mul_4/5/6/7 in the op trace); the same constants ride
// through rope_cache_init's key.
constexpr float kCosScale = 1.0f / 127.0f;
constexpr int32_t kCosZp = 128;
constexpr float kSinScale = 1.0f / 127.0f;
constexpr int32_t kSinZp = 128;

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str;
}

/** @brief Default rope thetas, matching mha_core.cpp's
 * _compute_default_parameters: thetas[i] = theta_base^(-2i/dim). */
std::vector<float> DefaultThetas(uint32_t dim, float theta_base) {
  std::vector<float> thetas(dim / 2);
  for (uint32_t i = 0; i < thetas.size(); ++i) {
    thetas[i] = 1.0f / std::pow(theta_base, (2.0f * (float)i) / (float)dim);
  }
  return thetas;
}

/** @brief Matches rope_real_to_u8 in nntr_hvx_rope.c exactly. */
uint8_t RealToU8(float value, float scale, int32_t zp) {
  int32_t v = (int32_t)std::nearbyint(value / scale) + zp;
  return (uint8_t)std::max(0, std::min(255, v));
}

/** @brief Quantize a uint8-encoded cos/sin pattern back from an arbitrary
 * linear ramp; used only by the bit-exact tests where the actual
 * trig value is irrelevant (the harness compares DSP vs host ref,
 * both fed identical cos_u8/sin_u8 bytes). */
uint8_t RampU8(int k, int bias, int step) {
  int32_t v = bias - k * step;
  return (uint8_t)std::max(0, std::min(255, v));
}

class HtpSession : public ::testing::Test {
protected:
  void SetUp override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str, &handle_);
    ASSERT_EQ(err, AEE_SUCCESS) << "nntr_hvx_open failed: " << hex(err);
  }

  void TearDown override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
};

class HvxRope : public HtpSession {};

struct Case {
  uint32_t rows;
  uint32_t width;
  uint32_t dim;
};

TEST_F(HvxRope, RejectsBadLengths) {
  std::vector<uint8_t> x(64, 17), y(64);
  std::vector<float> scale(1, 1.0f);
  std::vector<int32_t> zp(1, 0);
  std::vector<uint8_t> cos(32, 255), sin(32, 128);
  uint32_t saturated = 0;

  int err = nntr_hvx_rope_u8(
    handle_, 1, 64, 64, x.data, 63, scale.data, 1, zp.data, 1, cos.data,
    (int)cos.size(), sin.data, (int)sin.size(), kCosScale, kCosZp, kSinScale,
    kSinZp, 1.0f, 0, y.data, (int)y.size(), &saturated);
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset) << "got " << hex(err);
}

/**
 * @brief Regression test for the tail-copy overrun in the pre-fix kernel.
 *
 * hvx_rope_u8.c used to memcpy the unrotated tail with the destination
 * computed as `out + w + dim` at the LAST w, i.e. one full row past the
 * end of the output row width - dim bytes past the buffer on the last
 * row. Comparing outputs alone can never see this: the overrun target got
 * re-rotated and overwritten by the next iteration in every case except
 * the very last row, so the visible result was always correct.
 *
 * This needs width/dim >= 2 and width % dim == 0. Large FastRPC sequence
 * buffers are typically rpcmem/ION-backed and rounded up to page size, so
 * bytes just past a non-page-aligned declared length often live in the
 * same mapped page the DSP can still reach. This is a best-effort
 * detector; the actual fix is the code change itself. */
TEST_F(HvxRope, DoesNotWritePastTheOutputBuffer) {
  const uint32_t rows = 2, width = 512, dim = 64, half = dim / 2u;
  constexpr uint8_t kGuard = 0xA5;
  const size_t payload = (size_t)rows * width;
  const size_t guard = 4096; // one page

  std::vector<uint8_t> x(payload);
  std::vector<uint8_t> y(payload + guard, kGuard);
  std::vector<float> scale(rows, 0.25f);
  std::vector<int32_t> zp = {17, 233};
  std::vector<uint8_t> cos((size_t)rows * half), sin(cos.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = (uint8_t)((i * 31u + 5u) & 255u);
  }
  for (uint32_t m = 0; m < rows; ++m) {
    for (uint32_t k = 0; k < half; ++k) {
      cos[(size_t)m * half + k] = RampU8((int)k, 240, 2);
      sin[(size_t)m * half + k] = RampU8((int)k, 16, 2);
    }
  }
  uint32_t saturated = 0;
  ASSERT_EQ(nntr_hvx_rope_u8(
              handle_, rows, width, dim, x.data, (int)payload, scale.data,
              (int)scale.size(), zp.data, (int)zp.size(), cos.data,
              (int)cos.size(), sin.data, (int)sin.size(), kCosScale, kCosZp,
              kSinScale, kSinZp, 0.5f, 111, y.data, (int)payload, &saturated),
            AEE_SUCCESS);

  for (size_t i = payload; i < y.size(); ++i) {
    ASSERT_EQ(y[i], kGuard)
      << "wrote " << (i - payload) << " bytes past the declared output length";
  }
}

TEST_F(HvxRope, MatchesReferenceWithinOneLsb) {
  // dim and width/dim each appear >= 2 times; n_rows covers (1, 7, 32,
  // 1024), each twice.
  // dim: 64 x3, 96 x2, 128 x3
  // width/dim: 1 x4, 8 x2, 16 x2
  // n_rows: 1 x2, 7 x2, 32 x2, 1024 x2
  const std::vector<Case> cases = {
    {1, 64, 64},    {1, 1024, 64},   {7, 96, 96},     {7, 768, 96},
    {32, 128, 128}, {32, 2048, 128}, {1024, 512, 64}, {1024, 128, 128},
  };

  constexpr float kSOut = 0.5f;
  constexpr int32_t kZpOut = 123;

  for (const Case &c : cases) {
    const uint32_t half = c.dim / 2u;
    std::vector<uint8_t> x((size_t)c.rows * c.width);
    std::vector<uint8_t> out(x.size()), ref(x.size());
    std::vector<float> scale(c.rows);
    std::vector<int32_t> zp(c.rows);
    std::vector<uint8_t> cos((size_t)c.rows * half);
    std::vector<uint8_t> sin(cos.size());

    // zp cycles through the three boundary values plus a per-row spread.
    constexpr int32_t kZpValues[3] = {0, 128, 255};
    for (uint32_t m = 0; m < c.rows; ++m) {
      scale[m] = 0.25f + 0.125f * (float)(m % 5);
      zp[m] = kZpValues[m % 3];
      for (uint32_t k = 0; k < half; ++k) {
        cos[(size_t)m * half + k] = RampU8((int)k, 240, 3);
        sin[(size_t)m * half + k] = RampU8((int)k, 20, 4);
      }
    }
    for (uint32_t i = 0; i < x.size(); ++i) {
      x[i] = (uint8_t)((i * 29u + 7u) & 255u);
    }

    uint32_t ref_sat = 0;
    rope_u8_ref(x.data, ref.data, c.rows, c.width, c.dim, scale.data, zp.data,
                cos.data, sin.data, kCosScale, kCosZp, kSinScale, kSinZp, kSOut,
                kZpOut, &ref_sat);
    uint32_t dsp_sat = 0;
    int err = nntr_hvx_rope_u8(
      handle_, c.rows, c.width, c.dim, x.data, (int)x.size(), scale.data,
      (int)scale.size(), zp.data, (int)zp.size(), cos.data, (int)cos.size(),
      sin.data, (int)sin.size(), kCosScale, kCosZp, kSinScale, kSinZp, kSOut,
      kZpOut, out.data, (int)out.size(), &dsp_sat);
    ASSERT_EQ(err, AEE_SUCCESS) << "shape " << c.rows << "x" << c.width
                                << " dim=" << c.dim << ": " << hex(err);
    EXPECT_EQ(dsp_sat, ref_sat)
      << "shape " << c.rows << "x" << c.width << " dim=" << c.dim;

    uint32_t mismatch = 0;
    int max_err = 0;
    double sq_err = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
      const int e = std::abs((int)out[i] - (int)ref[i]);
      max_err = std::max(max_err, e);
      mismatch += e != 0;
      sq_err += (double)e * (double)e;
      EXPECT_LE(e, 1) << "shape " << c.rows << "x" << c.width
                      << " dim=" << c.dim << " element " << i;
    }
    const double rms_lsb = std::sqrt(sq_err / (double)out.size());

    // amax_out <= sqrt(2) * amax_in per dim-sized rotation segment:
    // rotation preserves a^2+b^2 per pair, so the segment max can grow
    // by at most sqrt(2). 0.05 slack absorbs quantization rounding.
    double max_amax_ratio = 0.0;
    for (uint32_t m = 0; m < c.rows; ++m) {
      const uint8_t *xr = x.data + (size_t)m * c.width;
      const uint8_t *outr = out.data + (size_t)m * c.width;
      for (uint32_t w = 0; w + c.dim <= c.width; w += c.dim) {
        float amax_in = 0.0f, amax_out = 0.0f;
        for (uint32_t k = 0; k < half; ++k) {
          const float a_in =
            std::fabs(((float)xr[w + k] - (float)zp[m]) * scale[m]);
          const float b_in =
            std::fabs(((float)xr[w + half + k] - (float)zp[m]) * scale[m]);
          const float a_out =
            std::fabs(((float)outr[w + k] - (float)kZpOut) * kSOut);
          const float b_out =
            std::fabs(((float)outr[w + half + k] - (float)kZpOut) * kSOut);
          amax_in = std::max({amax_in, a_in, b_in});
          amax_out = std::max({amax_out, a_out, b_out});
        }
        if (amax_in > 0.0f) {
          max_amax_ratio =
            std::max(max_amax_ratio, (double)(amax_out / amax_in));
        }
      }
    }
    EXPECT_LE(max_amax_ratio, std::sqrt(2.0) + 0.05)
      << "shape " << c.rows << "x" << c.width << " dim=" << c.dim;

    std::cout << "ROPE_FIELD rows=" << c.rows << " width=" << c.width
              << " dim=" << c.dim << " max_lsb=" << max_err
              << " rms_lsb=" << rms_lsb
              << " mismatch_ratio=" << (double)mismatch / (double)out.size()
              << " saturated=" << dsp_sat
              << " max_amax_ratio=" << max_amax_ratio << std::endl;
  }
}

/**
 * @brief s_in/zp_in given as a length-1 broadcast must equal the same
 * value duplicated across every row. */
TEST_F(HvxRope, BroadcastEqualsPerRowSameValue) {
  const uint32_t rows = 5, width = 128, dim = 64, half = dim / 2u;
  std::vector<uint8_t> x((size_t)rows * width);
  std::vector<uint8_t> broadcast_out(x.size()), per_row_out(x.size());
  std::vector<uint8_t> cos((size_t)rows * half), sin(cos.size());
  for (uint32_t i = 0; i < x.size(); ++i) {
    x[i] = (uint8_t)((i * 43u + 13u) & 255u);
  }
  for (uint32_t m = 0; m < rows; ++m) {
    for (uint32_t k = 0; k < half; ++k) {
      cos[(size_t)m * half + k] = RampU8((int)k, 230, 2);
      sin[(size_t)m * half + k] = RampU8((int)k, 30, 2);
    }
  }
  std::vector<float> broadcast_scale = {0.375f};
  std::vector<int32_t> broadcast_zp = {77};
  std::vector<float> per_row_scale(rows, 0.375f);
  std::vector<int32_t> per_row_zp(rows, 77);

  uint32_t bsat = 0, psat = 0;
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data, (int)x.size(),
                             broadcast_scale.data, 1, broadcast_zp.data, 1,
                             cos.data, (int)cos.size(), sin.data,
                             (int)sin.size(), kCosScale, kCosZp, kSinScale,
                             kSinZp, 0.5f, 90, broadcast_out.data,
                             (int)broadcast_out.size(), &bsat),
            AEE_SUCCESS);
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data, (int)x.size(),
                             per_row_scale.data, (int)per_row_scale.size(),
                             per_row_zp.data, (int)per_row_zp.size(), cos.data,
                             (int)cos.size(), sin.data, (int)sin.size(),
                             kCosScale, kCosZp, kSinScale, kSinZp, 0.5f, 90,
                             per_row_out.data, (int)per_row_out.size(), &psat),
            AEE_SUCCESS);

  EXPECT_EQ(bsat, psat);
  EXPECT_EQ(broadcast_out, per_row_out);
}

TEST_F(HvxRope, LaneOrderAndSaturationAreObservable) {
  const uint32_t rows = 1, width = 64, dim = 64, half = dim / 2u;
  std::vector<uint8_t> x(width, 200), out(width), ref(width);
  std::vector<float> scale(1, 1.0f);
  std::vector<int32_t> zp(1, 0);
  std::vector<uint8_t> cos(half), sin(half, kSinZp);
  for (uint32_t k = 0; k < half; ++k) {
    // cos goes 1.0 (lane 0) down through the lane range so the first
    // pair dominates; sin is held at 0 (encoded as zp).
    cos[k] = RealToU8(1.0f - (float)k * (1.0f / 127.0f), kCosScale, kCosZp);
  }
  uint32_t ref_sat = 0;
  rope_u8_ref(x.data, ref.data, rows, width, dim, scale.data, zp.data, cos.data,
              sin.data, kCosScale, kCosZp, kSinScale, kSinZp, 1.0f, 0,
              &ref_sat);
  uint32_t dsp_sat = 0;
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data, width,
                             scale.data, 1, zp.data, 1, cos.data,
                             (int)cos.size(), sin.data, (int)sin.size(),
                             kCosScale, kCosZp, kSinScale, kSinZp, 1.0f, 0,
                             out.data, width, &dsp_sat),
            AEE_SUCCESS);
  EXPECT_EQ(dsp_sat, ref_sat);
  EXPECT_EQ(out, ref);

  // Drive cos to its maximum representable real (1.0 -> 255) and x to 255
  // with a sub-unity output scale to force the requant to clip.
  std::fill(cos.begin, cos.end, 255);
  std::fill(x.begin, x.end, 255);
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data, width,
                             scale.data, 1, zp.data, 1, cos.data,
                             (int)cos.size(), sin.data, (int)sin.size(),
                             kCosScale, kCosZp, kSinScale, kSinZp, 0.125f, 0,
                             out.data, width, &dsp_sat),
            AEE_SUCCESS);
  EXPECT_GT(dsp_sat, 0u);
}

TEST_F(HvxRope, GeneratesCacheAndReusesIt) {
  constexpr uint32_t positions = 16;
  constexpr uint32_t dim = 64;
  constexpr uint32_t rows = 3;
  constexpr uint32_t width = 64;
  constexpr uint32_t half = dim / 2;
  constexpr float theta_base = 1000000.0f;
  constexpr float attention_scaling = 0.87f; // deliberately != 1.0

  const std::vector<float> thetas = DefaultThetas(dim, theta_base);
  uint32_t generation = 0;
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, positions, thetas.data,
                                     (int)thetas.size(), attention_scaling,
                                     kCosScale, kCosZp, kSinScale, kSinZp,
                                     &generation),
            AEE_SUCCESS);
  ASSERT_NE(generation, 0u);
  uint32_t same_generation = 0;
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, positions, thetas.data,
                                     (int)thetas.size(), attention_scaling,
                                     kCosScale, kCosZp, kSinScale, kSinZp,
                                     &same_generation),
            AEE_SUCCESS);
  EXPECT_EQ(same_generation, generation);

  // A different attention_scaling must bump the generation.
  uint32_t different_generation = 0;
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, positions, thetas.data,
                                     (int)thetas.size(), attention_scaling * 2,
                                     kCosScale, kCosZp, kSinScale, kSinZp,
                                     &different_generation),
            AEE_SUCCESS);
  EXPECT_NE(different_generation, generation);
  // A different cos_scale must also bump the generation uint8 cos/sin
  // are now load-bearing, not just a key fudge.
  uint32_t cos_scale_gen = 0;
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, positions, thetas.data,
                                     (int)thetas.size(), attention_scaling,
                                     kCosScale * 2.0f, kCosZp, kSinScale,
                                     kSinZp, &cos_scale_gen),
            AEE_SUCCESS);
  EXPECT_NE(cos_scale_gen, generation);
  // Restore the cache the rest of this test relies on.
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, positions, thetas.data,
                                     (int)thetas.size(), attention_scaling,
                                     kCosScale, kCosZp, kSinScale, kSinZp,
                                     &generation),
            AEE_SUCCESS);

  std::vector<uint8_t> x(rows * width), cached(x.size()), direct(x.size());
  std::vector<float> scale(rows, 0.25f);
  std::vector<int32_t> zp(rows, 127);
  const uint32_t position_start = 2;
  std::vector<uint8_t> cos(rows * half), sin(rows * half);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<uint8_t>((i * 37u + 11u) & 255u);
  }
  for (uint32_t m = 0; m < rows; ++m) {
    const uint32_t pos = position_start + m;
    for (uint32_t k = 0; k < half; ++k) {
      const float angle = static_cast<float>(pos) * thetas[k];
      cos[m * half + k] =
        RealToU8(std::cos(angle) * attention_scaling, kCosScale, kCosZp);
      sin[m * half + k] =
        RealToU8(std::sin(angle) * attention_scaling, kSinScale, kSinZp);
    }
  }
  uint32_t cached_sat = 0;
  ASSERT_EQ(nntr_hvx_rope_u8_cached(handle_, rows, width, dim, position_start,
                                    x.data, x.size(), scale.data, scale.size(),
                                    zp.data, zp.size(), 0.5f, 123, cached.data,
                                    cached.size(), &cached_sat),
            AEE_SUCCESS);
  uint32_t direct_sat = 0;
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data, x.size(),
                             scale.data, scale.size(), zp.data, zp.size(),
                             cos.data, cos.size(), sin.data, sin.size(),
                             kCosScale, kCosZp, kSinScale, kSinZp, 0.5f, 123,
                             direct.data, direct.size(), &direct_sat),
            AEE_SUCCESS);
  EXPECT_EQ(cached_sat, direct_sat);
  EXPECT_EQ(cached, direct);
  ASSERT_EQ(nntr_hvx_rope_cache_clear(handle_), AEE_SUCCESS);
}

/**
 * @brief A zeroed theta produces a uint8-exact identity rotation the
 * mechanism partial rotary relies on: the caller does not need a
 * "skip this channel" code path, just a zero in thetas.
 *
 * Unlike the old Q15 path (where 1.0 was not representable and the test
 * allowed one LSB), symmetric uint8 quant hits cos(0)=1.0 -> 255 and
 * sin(0)=0 -> 128 exactly, so the identity lanes are bit-exact. */
TEST_F(HvxRope, ZeroedThetaGivesIdentityRotation) {
  constexpr uint32_t dim = 64;
  constexpr uint32_t half = dim / 2;
  constexpr uint32_t width = dim;
  constexpr uint32_t rows = 1;
  constexpr uint32_t position = 5;

  std::vector<float> thetas = DefaultThetas(dim, 1000000.0f);
  // Emulate a partial-rotary factor of 0.5: the upper half of the rotated
  // dimension gets angle 0 at every position.
  for (uint32_t k = half / 2; k < half; ++k) {
    thetas[k] = 0.0f;
  }

  uint32_t generation = 0;
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, position + 1, thetas.data,
                                     (int)thetas.size(), 1.0f, kCosScale,
                                     kCosZp, kSinScale, kSinZp, &generation),
            AEE_SUCCESS);

  std::vector<uint8_t> x(width), out(width);
  for (uint32_t i = 0; i < width; ++i) {
    x[i] = (uint8_t)(20 + i * 3);
  }
  std::vector<float> scale(1, 0.1f);
  std::vector<int32_t> zp(1, 128);
  uint32_t saturated = 0;
  ASSERT_EQ(nntr_hvx_rope_u8_cached(handle_, rows, width, dim, position, x.data,
                                    (int)x.size(), scale.data, 1, zp.data, 1,
                                    0.1f, 128, out.data, (int)out.size(),
                                    &saturated),
            AEE_SUCCESS);

  // theta == 0 -> cos(0) == 1 (encodes exactly to 255 under symmetric
  // uint8) and sin(0) == 0 (encodes exactly to 128, zp), so the rotated
  // lanes are bit-exact.
  for (uint32_t k = half / 2; k < half; ++k) {
    EXPECT_EQ(out[k], x[k]) << "lane " << k;
    EXPECT_EQ(out[half + k], x[half + k]) << "lane " << (half + k);
  }
}

TEST_F(HvxRope, RejectsCacheRangeMismatch) {
  const std::vector<float> thetas = DefaultThetas(64, 1000000.0f);
  uint32_t generation = 0;
  ASSERT_EQ(nntr_hvx_rope_cache_init(handle_, 4, thetas.data,
                                     (int)thetas.size(), 1.0f, kCosScale,
                                     kCosZp, kSinScale, kSinZp, &generation),
            AEE_SUCCESS);
  std::vector<uint8_t> x(64), y(64);
  std::vector<float> scale(1, 1.0f);
  std::vector<int32_t> zp(1, 0);
  uint32_t saturated = 0;
  EXPECT_EQ(nntr_hvx_rope_u8_cached(handle_, 1, 64, 64, 4, x.data, x.size(),
                                    scale.data, 1, zp.data, 1, 1.0f, 0, y.data,
                                    y.size(), &saturated),
            AEE_EBADPARM + kDspOffset);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS;
}
