// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_mm_u8i4.cpp
 * @date   03 Aug 2026
 * @brief  Device test: A8W4 matmul on HMX matches the CPU reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH. See test/htp/build.sh.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

namespace {

/** @brief Render a FastRPC error as hex so the code is searchable. */
std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/** @brief Rounds @a v up to a multiple of @a a. */
constexpr uint32_t round_up(uint32_t v, uint32_t a) {
  return ((v + a - 1) / a) * a;
}

/** @brief HMX int8 tile geometry, mirrored from hexkl_micro.h. */
constexpr uint32_t kTileRow = 64;   // HEXKL_HMX_INT8_BLOCK_N_ROW
constexpr uint32_t kTileInner = 32; // HEXKL_HMX_INT8_BLOCK_N_INNER
constexpr uint32_t kTileCol = 32;   // HEXKL_HMX_INT8_BLOCK_N_COL
constexpr uint32_t kActTileBytes = 2048;

/**
 * @brief Byte offset of activation element (m, k) in AH layout.
 *
 * AH is a tiling, not a shuffle: each 64x32 tile is flat row-major, and
 * the tiles run in (row_block, inner_tile) order at a 2048-byte stride.
 */
inline size_t ah_offset(uint32_t m, uint32_t k, uint32_t K) {
  const uint32_t n_ktiles = K / kTileInner;
  const uint32_t rb = m / kTileRow;
  const uint32_t r = m % kTileRow;
  const uint32_t kt = k / kTileInner;
  const uint32_t c = k % kTileInner;
  return static_cast<size_t>(rb * n_ktiles + kt) * kActTileBytes +
         r * kTileInner + c;
}

/** @brief Scatters a row-major uint8 activation into AH layout. */
void pack_ah_from_rowmajor(const std::vector<uint8_t> &u_rm, uint32_t m_pad,
                           uint32_t K, std::vector<uint8_t> &out_ah) {
  out_ah.assign(static_cast<size_t>(m_pad) * K, 0);
  for (uint32_t m = 0; m < m_pad; ++m) {
    for (uint32_t k = 0; k < K; ++k) {
      out_ah[ah_offset(m, k, K)] = u_rm[static_cast<size_t>(m) * K + k];
    }
  }
}

/**
 * @brief Per-channel symmetric int4 weight quantization.
 *
 * Deliberately replicates __fallback_quant_nxk_qs4cx_f32 in
 * nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp:713 so a
 * later cross-check against nntrainer's CPU QS4CX path does not need a
 * second quantizer. Note it derives the scale from the min/max span but
 * quantizes symmetrically with no zero point, so skewed channels clamp.
 *
 * std::round (half away from zero) is correct here: this runs on the host
 * only and the DSP consumes the result, so no rounding mismatch is
 * possible. Activation quantization uses RNE instead -- see quantize_act.
 *
 * @param[in]  w_f32 weight matrix, K rows by N columns, row-major
 * @param[out] q_w   quantized values in [-8, 7], K by N, row-major
 * @param[out] d     per-channel dequantization multiplier, N entries
 * @param[out] colsum per-channel sum of q_w over K, N entries
 */
void quantize_weights_qs4cx(const std::vector<float> &w_f32, uint32_t K,
                            uint32_t N, std::vector<int8_t> &q_w,
                            std::vector<float> &d,
                            std::vector<int32_t> &colsum) {
  q_w.assign(static_cast<size_t>(K) * N, 0);
  d.assign(N, 0.0f);
  colsum.assign(N, 0);

  for (uint32_t n = 0; n < N; ++n) {
    float min0 = w_f32[n];
    float max0 = min0;
    for (uint32_t k = 0; k < K; ++k) {
      const float v = w_f32[static_cast<size_t>(k) * N + n];
      min0 = std::min(min0, v);
      max0 = std::max(max0, v);
    }
    const float rmin = std::min(0.0f, min0);
    const float rmax = std::max(0.0f, max0);
    const float scale = (rmin == rmax) ? 1.0f : 15.0f / (rmax - rmin);

    int32_t sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      int32_t q = static_cast<int32_t>(
        std::round(w_f32[static_cast<size_t>(k) * N + n] * scale));
      q = std::max(-8, std::min(7, q));
      q_w[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q);
      sum += q;
    }
    d[n] = 1.0f / scale;
    colsum[n] = sum;
  }
}

/**
 * @brief Per-channel symmetric int8 weight quantization -- same shape as
 *        quantize_weights_qs4cx, at the full int8 range [-128, 127]
 *        instead of int4's [-8, 7]. Kept separate rather than
 *        parametrizing qs4cx over the range: that function's name and
 *        callers are already tied to QS4CX specifically.
 */
void quantize_weights_symmetric_i8(const std::vector<float> &w_f32, uint32_t K,
                                   uint32_t N, std::vector<int8_t> &q_w,
                                   std::vector<float> &d,
                                   std::vector<int32_t> &colsum) {
  q_w.assign(static_cast<size_t>(K) * N, 0);
  d.assign(N, 0.0f);
  colsum.assign(N, 0);

  for (uint32_t n = 0; n < N; ++n) {
    float min0 = w_f32[n];
    float max0 = min0;
    for (uint32_t k = 0; k < K; ++k) {
      const float v = w_f32[static_cast<size_t>(k) * N + n];
      min0 = std::min(min0, v);
      max0 = std::max(max0, v);
    }
    const float rmin = std::min(0.0f, min0);
    const float rmax = std::max(0.0f, max0);
    const float scale = (rmin == rmax) ? 1.0f : 255.0f / (rmax - rmin);

    int32_t sum = 0;
    for (uint32_t k = 0; k < K; ++k) {
      int32_t q = static_cast<int32_t>(
        std::round(w_f32[static_cast<size_t>(k) * N + n] * scale));
      q = std::max(-128, std::min(127, q));
      q_w[static_cast<size_t>(k) * N + n] = static_cast<int8_t>(q);
      sum += q;
    }
    d[n] = 1.0f / scale;
    colsum[n] = sum;
  }
}

/**
 * @brief Integer reference matmul: uint8 activation by int4 weight.
 *
 * Deterministic integer arithmetic, so the HMX result must match this bit
 * for bit. |acc| <= 255 * 8 * K, which is 2,088,960 at K=1024 -- three
 * orders of magnitude inside int32.
 *
 * @param[in] u_rm activation, m_pad by K, row-major (NOT AH)
 * @param[in] q_w  weights, K by N, row-major
 */
void ref_int_matmul(const std::vector<uint8_t> &u_rm,
                    const std::vector<int8_t> &q_w, uint32_t m_pad, uint32_t K,
                    uint32_t N, std::vector<int32_t> &acc) {
  acc.assign(static_cast<size_t>(m_pad) * N, 0);
  for (uint32_t m = 0; m < m_pad; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      int32_t sum = 0;
      for (uint32_t k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(u_rm[static_cast<size_t>(m) * K + k]) *
               static_cast<int32_t>(q_w[static_cast<size_t>(k) * N + n]);
      }
      acc[static_cast<size_t>(m) * N + n] = sum;
    }
  }
}

/**
 * @brief Dequantization reference.
 *
 * out[m][n] = (acc[m][n] - zp[m]*colsum[n]) * scale[m] * d[n] + bias[n]
 *
 * The correction term comes from feeding HMX unsigned activations:
 *   sum_k x*w = sum_k s*(u - zp) * d*q_w
 *             = s*d * (sum_k u*q_w  -  zp * sum_k q_w)
 * and sum_k q_w is colsum, which the weights alone determine.
 *
 * |acc| <= 255*8*K and |zp*colsum| <= 255*8*K, so the difference stays
 * inside int32 and both operands are exactly representable in f32 (below
 * 2^24), which is why the DSP may do the correction in either domain.
 *
 * @param[in] acc m_pad by n, row-major
 * @param[out] out m_valid by n, row-major
 */
void ref_dequant(const std::vector<int32_t> &acc, uint32_t m_valid, uint32_t N,
                 const std::vector<float> &act_scale,
                 const std::vector<int32_t> &act_zp,
                 const std::vector<int32_t> &colsum,
                 const std::vector<float> &d, const std::vector<float> &bias,
                 std::vector<float> &out) {
  out.assign(static_cast<size_t>(m_valid) * N, 0.0f);
  for (uint32_t m = 0; m < m_valid; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      const int32_t corrected =
        acc[static_cast<size_t>(m) * N + n] - act_zp[m] * colsum[n];
      out[static_cast<size_t>(m) * N + n] =
        static_cast<float>(corrected) * act_scale[m] * d[n] + bias[n];
    }
  }
}

/**
 * @brief Unquantized fp32 matmul. The yardstick S4 measures against.
 */
void ref_fp32_matmul(const std::vector<float> &x, const std::vector<float> &w,
                     uint32_t M, uint32_t K, uint32_t N,
                     const std::vector<float> &bias, std::vector<float> &out) {
  out.assign(static_cast<size_t>(M) * N, 0.0f);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        sum +=
          x[static_cast<size_t>(m) * K + k] * w[static_cast<size_t>(k) * N + n];
      }
      out[static_cast<size_t>(m) * N + n] = sum + bias[n];
    }
  }
}

/**
 * @brief Signal-to-noise ratio in dB between a reference and a result.
 *
 * Reported rather than asserted tightly: no measurement exists yet for
 * what per-channel int4 costs on this workload, and inventing a threshold
 * before measuring would just encode a guess.
 */
double snr_db(const std::vector<float> &ref, const std::vector<float> &got) {
  double sig = 0.0;
  double noise = 0.0;
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

/** @brief Deterministic pseudo-random fill in [-1, 1). */
void fill_deterministic(std::vector<float> &v, uint32_t seed) {
  uint32_t s = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = static_cast<float>(static_cast<int32_t>(s >> 8)) /
             static_cast<float>(1 << 23) -
           1.0f;
  }
}

/**
 * @brief Per-row asymmetric uint8 activation quantization: scale and zp.
 *
 * x[m][k] is recovered as scale[m] * (u[m][k] - zp[m]).
 *
 * Padded rows (m >= m_valid) get scale 1 and zp 0. Their uint8 values are
 * zero, which does not decode to zero -- s*(0-zp) is nonzero whenever zp
 * is -- so their outputs are meaningless. Rows are independent in a
 * matmul so valid rows are unaffected; fixing scale and zp for pad rows
 * just makes the host reference easy to keep identical to the DSP.
 */
void quantize_act_rows(const std::vector<float> &x, uint32_t m_valid,
                       uint32_t m_pad, uint32_t K, std::vector<float> &scale,
                       std::vector<int32_t> &zp) {
  scale.assign(m_pad, 1.0f);
  zp.assign(m_pad, 0);

  for (uint32_t m = 0; m < m_valid; ++m) {
    float min0 = x[static_cast<size_t>(m) * K];
    float max0 = min0;
    for (uint32_t k = 0; k < K; ++k) {
      const float v = x[static_cast<size_t>(m) * K + k];
      min0 = std::min(min0, v);
      max0 = std::max(max0, v);
    }
    const float rmin = std::min(0.0f, min0);
    const float rmax = std::max(0.0f, max0);
    if (rmin == rmax) {
      scale[m] = 1.0f;
      zp[m] = 0;
      continue;
    }
    scale[m] = (rmax - rmin) / 255.0f;
    int32_t z = static_cast<int32_t>(std::nearbyint(-rmin / scale[m]));
    zp[m] = std::max(0, std::min(255, z));
  }
}

/**
 * @brief Per-row asymmetric uint8 activation quantization: the values.
 *
 * std::nearbyint, not std::round: this value is computed independently on
 * the DSP and compared byte for byte, and HVX's float add rounds to
 * nearest-even. std::round (half away from zero) would disagree at exact
 * .5. Weight quantization uses std::round because it runs on the host
 * only -- see quantize_weights_qs4cx.
 *
 * @param[out] u_rm row-major uint8, m_pad by K
 */
void quantize_act_values(const std::vector<float> &x, uint32_t m_valid,
                         uint32_t m_pad, uint32_t K,
                         const std::vector<float> &scale,
                         const std::vector<int32_t> &zp,
                         std::vector<uint8_t> &u_rm) {
  u_rm.assign(static_cast<size_t>(m_pad) * K, 0);
  for (uint32_t m = 0; m < m_valid; ++m) {
    const float inv_s = 1.0f / scale[m];
    for (uint32_t k = 0; k < K; ++k) {
      // Reciprocal multiply, matching the DSP kernel: f32 a/b and a*(1/b)
      // can differ by 1 ULP, which breaks the S1 byte-exact check.
      const float q = std::nearbyint(x[static_cast<size_t>(m) * K + k] * inv_s);
      int32_t v = static_cast<int32_t>(q) + zp[m];
      v = std::max(0, std::min(255, v));
      u_rm[static_cast<size_t>(m) * K + k] = static_cast<uint8_t>(v);
    }
  }
}

/**
 * @brief Opens one unsigned-PD CDSP session for the whole test case.
 *
 * A failure here is a hard FAIL rather than a skip: proving the DSP comes
 * up on the device is the point of this test, so a quiet skip would
 * report success for the thing being measured.
 */
class HmxMmU8I4 : public ::testing::Test {
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
   * @brief Runs S1 through S4 for one shape.
   *
   * S1 and S2 are bit-exact: integer arithmetic is deterministic, so any
   * layout or wiring error shows up as an exact mismatch rather than
   * being absorbed into a tolerance. S3 allows f32 ordering slack. S4 is
   * reported, not gated.
   */
  void CheckShape(uint32_t M, uint32_t K, uint32_t N) {
    SCOPED_TRACE("M=" + std::to_string(M) + " K=" + std::to_string(K) +
                 " N=" + std::to_string(N));
    const uint32_t m_pad = round_up(M, kTileRow);

    std::vector<float> w_f32(static_cast<size_t>(K) * N);
    fill_deterministic(w_f32, 0x5EED0001u);
    std::vector<int8_t> q_w;
    std::vector<float> d;
    std::vector<int32_t> colsum;
    quantize_weights_qs4cx(w_f32, K, N, q_w, d, colsum);

    std::vector<float> x(static_cast<size_t>(M) * K);
    fill_deterministic(x, 0x5EED0002u);
    std::vector<float> bias(N);
    fill_deterministic(bias, 0x5EED0003u);

    std::vector<float> exp_scale;
    std::vector<int32_t> exp_zp;
    quantize_act_rows(x, M, m_pad, K, exp_scale, exp_zp);
    std::vector<uint8_t> exp_u_rm;
    quantize_act_values(x, M, m_pad, K, exp_scale, exp_zp, exp_u_rm);
    std::vector<uint8_t> exp_ah;
    pack_ah_from_rowmajor(exp_u_rm, m_pad, K, exp_ah);
    std::vector<int32_t> exp_acc;
    ref_int_matmul(exp_u_rm, q_w, m_pad, K, N, exp_acc);
    std::vector<float> exp_out;
    ref_dequant(exp_acc, M, N, exp_scale, exp_zp, colsum, d, bias, exp_out);

    std::vector<uint8_t> got_ah(static_cast<size_t>(m_pad) * K, 0);
    std::vector<float> got_scale(m_pad, 0.0f);
    std::vector<int32_t> got_zp(m_pad, -1);
    std::vector<int32_t> got_acc(static_cast<size_t>(m_pad) * N, 0);
    std::vector<float> got_out(static_cast<size_t>(M) * N, 0.0f);

    int err = nntr_hvx_mm_u8i4_from_f32(
      handle_, M, K, N, x.data(), static_cast<int>(x.size()), q_w.data(),
      static_cast<int>(q_w.size()), d.data(), static_cast<int>(d.size()),
      colsum.data(), static_cast<int>(colsum.size()), bias.data(),
      static_cast<int>(bias.size()), got_ah.data(),
      static_cast<int>(got_ah.size()), got_scale.data(),
      static_cast<int>(got_scale.size()), got_zp.data(),
      static_cast<int>(got_zp.size()), got_acc.data(),
      static_cast<int>(got_acc.size()), got_out.data(),
      static_cast<int>(got_out.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_from_f32 failed: " << hex(err);

    // S1
    for (uint32_t m = 0; m < m_pad; ++m) {
      EXPECT_NEAR(got_scale[m], exp_scale[m], std::abs(exp_scale[m]) * 1e-6f)
        << "scale[" << m << "]";
      EXPECT_EQ(got_zp[m], exp_zp[m]) << "zp[" << m << "]";
    }
    EXPECT_EQ(got_ah, exp_ah);

    // S2
    EXPECT_EQ(got_acc, exp_acc);

    // S3
    for (size_t i = 0; i < exp_out.size(); ++i) {
      EXPECT_NEAR(got_out[i], exp_out[i], std::abs(exp_out[i]) * 1e-5f + 1e-6f);
    }

    // S4 -- measured and printed, deliberately not gated tightly.
    std::vector<float> fp32_ref;
    ref_fp32_matmul(x, w_f32, M, K, N, bias, fp32_ref);
    double max_rel = 0.0;
    for (size_t i = 0; i < fp32_ref.size(); ++i) {
      const double denom = std::abs(static_cast<double>(fp32_ref[i]));
      if (denom > 1e-6) {
        max_rel = std::max(
          max_rel,
          std::abs(static_cast<double>(got_out[i]) - fp32_ref[i]) / denom);
      }
    }
    const double snr = snr_db(fp32_ref, got_out);
    std::cout << "[S4] M=" << M << " K=" << K << " N=" << N << " SNR=" << snr
              << " dB  max_rel=" << max_rel << std::endl;
    EXPECT_GT(snr, 0.0) << "quantized output carries no signal at all";
  }
};

TEST_F(HmxMmU8I4, Shape1_Minimal) {
  // Same dimensions as HexKL's own example, so a failure here is ours.
  CheckShape(64, 128, 128);
}

TEST_F(HmxMmU8I4, Shape2_DecodeSingleToken) {
  // One token padded out to a 64-row tile: the decode case, and the one
  // that exercises the zero-pad path for 63 of 64 rows.
  CheckShape(1, 1024, 1024);
}

TEST_F(HmxMmU8I4, Shape3_PrefillQwen3Scale) {
  // Weights alone need (1024/32)*(1024/32)*512 = 512 KiB of VTCM here.
  CheckShape(64, 1024, 1024);
}

TEST_F(HmxMmU8I4, Shape4_MultipleRowBlocks) {
  // Shapes 1 through 3 all pad to 64 rows, so the row-block loop only
  // ever runs with rb=0. This is the one that runs it twice.
  CheckShape(128, 128, 128);
}

/**
 * @brief The performance path: weights registered once, several matmuls
 *        per call sharing one activation.
 *
 * Shares HmxMmU8I4's helpers rather than duplicating a reference: the
 * layer endpoint must produce exactly what calling the accuracy endpoint
 * once per weight would, so that is what these compare against.
 */
class HmxMmU8I4Layer : public HmxMmU8I4 {
protected:
  /** @brief One weight plus everything needed to check its output. */
  struct Weight {
    uint32_t handle;
    uint32_t N;
    std::vector<int8_t> q_w;
    std::vector<float> d;
    std::vector<int32_t> colsum;
    std::vector<float> bias;
    std::vector<float> w_f32;
  };

  /** @brief Quantizes a deterministic K x N weight and registers it.
   *
   *  @param bias_overrides (column, value) pairs written over the generated
   *  bias before registration. The bias is the only term of the dequantized
   *  output a test can set directly -- everything else is the matmul of two
   *  deterministic fills -- so it is how a test drives one output column to
   *  a chosen magnitude. Used by SwigluSurvivesExtremeNegativeGate to place
   *  a gate below the SwiGLU clamp; empty (the default) leaves every
   *  existing caller's weight byte-for-byte what it was. */
  void MakeAndRegister(
    uint32_t K, uint32_t N, uint32_t seed, Weight &w,
    const std::vector<std::pair<uint32_t, float>> &bias_overrides = {}) {
    w.N = N;
    w.w_f32.resize(static_cast<size_t>(K) * N);
    fill_deterministic(w.w_f32, seed);
    quantize_weights_qs4cx(w.w_f32, K, N, w.q_w, w.d, w.colsum);
    w.bias.resize(N);
    fill_deterministic(w.bias, seed ^ 0xA5A5A5A5u);
    for (const auto &ov : bias_overrides) {
      ASSERT_LT(ov.first, N) << "bias override column out of range";
      w.bias[ov.first] = ov.second;
    }

    w.handle = 0xFFFFFFFFu;
    int err = nntr_hvx_weight_register_u8i4(
      handle_, K, N, w.q_w.data(), static_cast<int>(w.q_w.size()), w.d.data(),
      static_cast<int>(w.d.size()), w.colsum.data(),
      static_cast<int>(w.colsum.size()), w.bias.data(),
      static_cast<int>(w.bias.size()), &w.handle);
    ASSERT_EQ(err, AEE_SUCCESS) << "weight_register_u8i4 failed: " << hex(err);
    ASSERT_NE(w.handle, 0xFFFFFFFFu) << "handle not written";
  }

  /** @brief Host reference for one weight's slice of the concatenated
   *         output, via the same path the accuracy harness checks. */
  void ExpectedFor(const Weight &w, const std::vector<float> &x, uint32_t M,
                   uint32_t K, std::vector<float> &out) {
    const uint32_t m_pad = round_up(M, kTileRow);
    std::vector<float> scale;
    std::vector<int32_t> zp;
    quantize_act_rows(x, M, m_pad, K, scale, zp);
    std::vector<uint8_t> u_rm;
    quantize_act_values(x, M, m_pad, K, scale, zp, u_rm);
    std::vector<int32_t> acc;
    ref_int_matmul(u_rm, w.q_w, m_pad, K, w.N, acc);
    ref_dequant(acc, M, w.N, scale, zp, w.colsum, w.d, w.bias, out);
  }
};

TEST_F(HmxMmU8I4Layer, ThreeWeightsMatchPerWeightReference) {
  // A Q/K/V set: three weights, one shared activation, one call. The
  // shapes differ in N so a bug that assumes a uniform stride into
  // out_cat shows up as a mismatch rather than passing by luck.
  const uint32_t M = 64, K = 256;
  const uint32_t Ns[3] = {128, 256, 64};

  std::vector<Weight> ws(3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(
      MakeAndRegister(K, Ns[i], 0xB0B00001u + i * 0x1000u, ws[i]));
  }

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);

  uint32_t n_total = 0;
  std::vector<uint32_t> handles;
  for (const auto &w : ws) {
    handles.push_back(w.handle);
    n_total += w.N;
  }
  std::vector<float> got(static_cast<size_t>(M) * n_total, 0.0f);

  int err = nntr_hvx_mm_u8i4_layer(
    handle_, M, K, handles.data(), static_cast<int>(handles.size()), x.data(),
    static_cast<int>(x.size()), got.data(), static_cast<int>(got.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_layer failed: " << hex(err);

  size_t off = 0;
  for (int i = 0; i < 3; ++i) {
    SCOPED_TRACE("weight " + std::to_string(i) + " N=" + std::to_string(Ns[i]));
    std::vector<float> want;
    ExpectedFor(ws[i], x, M, K, want);
    for (size_t j = 0; j < want.size(); ++j) {
      EXPECT_NEAR(got[off + j], want[j], std::abs(want[j]) * 1e-5f + 1e-6f)
        << "element " << j;
    }
    off += want.size();
  }

  for (const auto &w : ws) {
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w.handle), AEE_SUCCESS);
  }
}

TEST_F(HmxMmU8I4Layer, RegisteredWeightSurvivesRepeatedCalls) {
  // The whole point of registering is that the bake is not repaid per
  // call, so the second call must return exactly what the first did --
  // bitwise, since nothing between them is supposed to differ.
  const uint32_t M = 1, K = 512, N = 128;
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xC0FFEE01u, w));

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);
  const uint32_t handles[1] = {w.handle};

  std::vector<float> first(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> second(static_cast<size_t>(M) * N, 1.0f);
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<float> &dst = pass == 0 ? first : second;
    int err = nntr_hvx_mm_u8i4_layer(handle_, M, K, handles, 1, x.data(),
                                     static_cast<int>(x.size()), dst.data(),
                                     static_cast<int>(dst.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "pass " << pass << ": " << hex(err);
  }
  EXPECT_EQ(first, second);

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w.handle), AEE_SUCCESS);
}

TEST_F(HmxMmU8I4Layer, ReleasedHandleIsRejectedAndSlotIsReused) {
  const uint32_t K = 128, N = 128;
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xD00D0001u, w));
  const uint32_t released = w.handle;
  ASSERT_EQ(nntr_hvx_weight_release_u8i4(handle_, released), AEE_SUCCESS);

  // Using a released handle must fail rather than read freed memory.
  std::vector<float> x(K, 0.0f);
  std::vector<float> out(N, 0.0f);
  const uint32_t handles[1] = {released};
  EXPECT_NE(nntr_hvx_mm_u8i4_layer(handle_, 1, K, handles, 1, x.data(),
                                   static_cast<int>(x.size()), out.data(),
                                   static_cast<int>(out.size())),
            AEE_SUCCESS);
  EXPECT_NE(nntr_hvx_weight_release_u8i4(handle_, released), AEE_SUCCESS)
    << "double release accepted";

  // The freed slot must come back, or a long-running process leaks handles.
  Weight w2;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xD00D0002u, w2));
  EXPECT_EQ(w2.handle, released);
  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w2.handle), AEE_SUCCESS);
}

TEST_F(HmxMmU8I4Layer, MismatchedKIsRejected) {
  // Every handle in one call shares the activation, so a handle baked for
  // a different K is a caller bug that must not read out of bounds.
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(256, 128, 0xE0E00001u, w));
  const uint32_t handles[1] = {w.handle};
  std::vector<float> x(128, 0.0f); // K=128, but the weight was baked at 256
  std::vector<float> out(128, 0.0f);
  EXPECT_NE(nntr_hvx_mm_u8i4_layer(handle_, 1, 128, handles, 1, x.data(),
                                   static_cast<int>(x.size()), out.data(),
                                   static_cast<int>(out.size())),
            AEE_SUCCESS);
  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w.handle), AEE_SUCCESS);
}

TEST_F(HmxMmU8I4Layer, BakeBitsDump) {
  // Print-only: one registered weight at the MoE gate_up prefill shape, its
  // layer output reduced to an FNV-1a hash of the raw f32 bytes. Comparing
  // the printed value across two skel builds is the bit-identity gate doc
  // 43 §5 L1 asks for ("compare a registered weight's matmul output against
  // the current build") -- any WH byte a parallelised bake drops or
  // transposes moves some accumulator, hence some output bit. The expected
  // hash lives in the measurement log, not in an assertion: there is no
  // independent oracle for it, only the previous build.
  const uint32_t M = 64, K = 2048, N = 3584;

  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xBA5E0001u, w));

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);
  const uint32_t handles[1] = {w.handle};
  std::vector<float> got(static_cast<size_t>(M) * N, 0.0f);
  int err = nntr_hvx_mm_u8i4_layer(handle_, M, K, handles, 1, x.data(),
                                   static_cast<int>(x.size()), got.data(),
                                   static_cast<int>(got.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_layer failed: " << hex(err);

  uint64_t h = 1469598103934665603ull; // FNV-1a 64-bit offset basis
  const auto *bytes = reinterpret_cast<const uint8_t *>(got.data());
  for (size_t i = 0; i < got.size() * sizeof(float); ++i) {
    h ^= bytes[i];
    h *= 1099511628211ull;
  }
  std::cout << "U8I4_FIELD path=bake_bits field=out_f32_fnv1a value=0x"
            << std::hex << h << std::dec << " (M=" << M << " K=" << K
            << " N=" << N << ")" << std::endl;

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w.handle), AEE_SUCCESS);
}

/**
 * @brief Per-call cost of the harness endpoint against the layer endpoint.
 *
 * Printed, never asserted. doc13 §3a measured 1.7-2x for cross-matmul
 * prefetch on a V79, but that was inside a standalone DSP program with no
 * FastRPC in the timed region, and thermal state moves these numbers
 * between runs -- a threshold here would encode the lab conditions rather
 * than a property of the code. Read the ratio, do not gate on it.
 */
TEST_F(HmxMmU8I4Layer, ReportPerCallCost) {
  const uint32_t M = 64, K = 1024, N = 1024;
  const int kReps = 20;

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);

  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xF00D0001u, w));
  const uint32_t handles[1] = {w.handle};

  std::vector<float> out(static_cast<size_t>(M) * N, 0.0f);
  auto time_us = [&](const std::function<void()> &fn) {
    fn(); // warm-up, discarded
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
      fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / kReps;
  };

  const uint32_t m_pad = round_up(M, kTileRow);
  std::vector<uint8_t> ah(static_cast<size_t>(m_pad) * K, 0);
  std::vector<float> sc(m_pad, 0.0f);
  std::vector<int32_t> zp(m_pad, 0);
  std::vector<int32_t> acc(static_cast<size_t>(m_pad) * N, 0);
  std::vector<float> harness_out(static_cast<size_t>(M) * N, 0.0f);

  const double harness_us = time_us([&] {
    nntr_hvx_mm_u8i4_from_f32(
      handle_, M, K, N, x.data(), static_cast<int>(x.size()), w.q_w.data(),
      static_cast<int>(w.q_w.size()), w.d.data(), static_cast<int>(w.d.size()),
      w.colsum.data(), static_cast<int>(w.colsum.size()), w.bias.data(),
      static_cast<int>(w.bias.size()), ah.data(), static_cast<int>(ah.size()),
      sc.data(), static_cast<int>(sc.size()), zp.data(),
      static_cast<int>(zp.size()), acc.data(), static_cast<int>(acc.size()),
      harness_out.data(), static_cast<int>(harness_out.size()));
  });

  const double layer_us = time_us([&] {
    nntr_hvx_mm_u8i4_layer(handle_, M, K, handles, 1, x.data(),
                           static_cast<int>(x.size()), out.data(),
                           static_cast<int>(out.size()));
  });

  std::cout << "U8I4_FIELD path=harness  field=us_per_matmul value="
            << harness_us << std::endl;
  std::cout << "U8I4_FIELD path=layer_x1 field=us_per_matmul value=" << layer_us
            << std::endl;

  // Several weights per call is where the prefetch has something to hide
  // behind; x1 above cannot show it by construction (doc13 §3a: a single
  // matmul is parity).
  std::vector<Weight> ws(4);
  std::vector<uint32_t> hs;
  uint32_t n_total = 0;
  for (int i = 0; i < 4; ++i) {
    ASSERT_NO_FATAL_FAILURE(
      MakeAndRegister(K, N, 0xF00D1000u + i * 0x100u, ws[i]));
    hs.push_back(ws[i].handle);
    n_total += ws[i].N;
  }
  std::vector<float> out4(static_cast<size_t>(M) * n_total, 0.0f);
  const double layer4_us = time_us([&] {
    nntr_hvx_mm_u8i4_layer(
      handle_, M, K, hs.data(), static_cast<int>(hs.size()), x.data(),
      static_cast<int>(x.size()), out4.data(), static_cast<int>(out4.size()));
  });
  std::cout << "U8I4_FIELD path=layer_x4 field=us_per_matmul value="
            << (layer4_us / 4.0) << std::endl;
  std::cout << "U8I4_FIELD path=layer_x4 field=speedup_vs_harness value="
            << (harness_us / (layer4_us / 4.0)) << std::endl;

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w.handle), AEE_SUCCESS);
  for (const auto &ww : ws) {
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, ww.handle), AEE_SUCCESS);
  }
}

/**
 * @brief [L2] fused gate_up -> SwiGLU -> down against the unfused two-call
 *        path.
 *
 * Reference: one mm_u8i4_layer call for gate_up, host SwiGLU (plain expf),
 * one mm_u8i4_layer call for down. The fused call requantizes ITS OWN HVX
 * SwiGLU output, so the intermediate's per-row scale comes from slightly
 * different values and bit equality is not expected even in principle --
 * the gate is SNR (doc 43 §[L2] accuracy rule), thresholded at 40 dB, the
 * u8 requantization floor this seam sits on. This is also the only check
 * that pins the gate/up split semantics: swapping gate and up changes
 * every output element, not a few.
 */
TEST_F(HmxMmU8I4Layer, FusedSwigluMatchesTwoCallReference) {
  // The real seam's shape: LFM2-A1B expert FFN at prefill (gate_up
  // 2048x3584, down 1792x2048, 64 tokens) -- exercises the gate/up split
  // at the real tile boundary (I = 1792 = 56 tiles) and the ~6.65 MB
  // fused VTCM layout.
  const uint32_t K = 2048, I = 1792, N = 2048;

  Weight gu, dn;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, 2 * I, 0xBA5E0002u, gu));
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(I, N, 0xBA5E0003u, dn));

  // M sweep, not just 64: the fused kernel walks the rows in 64-row blocks,
  // and M == 64 is the ONE value that exercises neither a partial block nor
  // a second block. The real model never sees it -- 444 tokens x topk 4 over
  // 32 experts averages M = 55.5, spread either side -- so 64-only coverage
  // left the whole block loop untested (doc 43 §7's fused row).
  for (const uint32_t M : {55u, 64u, 100u, 128u, 138u, 200u, 11u}) {
    std::vector<float> x(static_cast<size_t>(M) * K);
    fill_deterministic(x, 0x5EED0002u);

    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * I, 0.0f);
    {
      const uint32_t handles[1] = {gu.handle};
      int err = nntr_hvx_mm_u8i4_layer(
        handle_, M, K, handles, 1, x.data(), static_cast<int>(x.size()),
        gu_out.data(), static_cast<int>(gu_out.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "gate_up layer call failed: " << hex(err);
    }

    std::vector<float> inter(static_cast<size_t>(M) * I);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t j = 0; j < I; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * I + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * I + I + j];
        inter[static_cast<size_t>(m) * I + j] = g / (1.0f + std::exp(-g)) * u;
      }
    }

    std::vector<float> dn_ref(static_cast<size_t>(M) * N, 0.0f);
    {
      const uint32_t handles[1] = {dn.handle};
      int err = nntr_hvx_mm_u8i4_layer(
        handle_, M, I, handles, 1, inter.data(), static_cast<int>(inter.size()),
        dn_ref.data(), static_cast<int>(dn_ref.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "down layer call failed: " << hex(err);
    }

    const uint32_t handles[2] = {gu.handle, dn.handle};
    std::vector<float> got(static_cast<size_t>(M) * N, 0.0f);
    int err = nntr_hvx_mm_u8i4_layer_fused(
      handle_, M, K, handles, 2, x.data(), static_cast<int>(x.size()),
      got.data(), static_cast<int>(got.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_layer_fused failed: " << hex(err);

    const double snr = snr_db(dn_ref, got);
    std::cout << "U8I4_FIELD path=fused_swiglu field=snr_db value=" << snr
              << " (M=" << M << " K=" << K << " I=" << I << " N=" << N << ")"
              << std::endl;
    EXPECT_GT(snr, 40.0)
      << "fused SwiGLU output below the u8 requantization floor, M=" << M;
  } // M sweep

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, gu.handle), AEE_SUCCESS);
  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, dn.handle), AEE_SUCCESS);
}

/**
 * @brief [L2, split-call variant] gate_up+SwiGLU+requant, checked in
 *        ISOLATION against the two-call reference's OWN intermediate --
 *        not just the final output.
 *
 * This is the gap the fused kernel's unit test had: SNR against a
 * two-call reference only ever checks the END of the pipeline, so a bug
 * anywhere upstream of the last matmul is invisible as long as it is
 * consistent between the path under test and the reference (which it is
 * not here -- the reference's intermediate is plain host f32, computed
 * independently of any DSP call). Comparing the SPLIT call's own
 * requantized intermediate against that host intermediate, before ever
 * touching the down matmul, localizes a divergence to stage 1 instead of
 * leaving it to be inferred from the end-to-end number.
 */
TEST_F(HmxMmU8I4Layer, GateUpSwigluMatchesHostIntermediate) {
  for (const uint32_t M : {55u, 64u, 100u, 128u, 138u, 200u, 11u}) {
    const uint32_t K = 2048, I = 1792;

    Weight gu;
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, 2 * I, 0xBA5E0004u, gu));

    std::vector<float> x(static_cast<size_t>(M) * K);
    fill_deterministic(x, 0x5EED0002u);

    // Host reference: gate_up via the already-proven mm_u8i4_layer, SwiGLU
    // on the host with expf (same reference formula as
    // FusedSwigluMatchesTwoCallReference above).
    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * I, 0.0f);
    {
      const uint32_t handles[1] = {gu.handle};
      int err = nntr_hvx_mm_u8i4_layer(
        handle_, M, K, handles, 1, x.data(), static_cast<int>(x.size()),
        gu_out.data(), static_cast<int>(gu_out.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "gate_up layer call failed: " << hex(err);
    }
    std::vector<float> inter_ref(static_cast<size_t>(M) * I);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t j = 0; j < I; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * I + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * I + I + j];
        inter_ref[static_cast<size_t>(m) * I + j] =
          g / (1.0f + std::exp(-g)) * u;
      }
    }

    // Split call under test: gate_up + SwiGLU + requant, ONE weight.
    const uint32_t m_pad = (M + 63) / 64 * 64;
    const uint32_t n_ktiles = I / 32;
    std::vector<uint8_t> out_ah(static_cast<size_t>(m_pad) * I, 0);
    std::vector<float> out_scale(m_pad, 1.0f);
    std::vector<int32_t> out_zp(m_pad, 0);
    int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
      handle_, M, K, gu.handle, x.data(), static_cast<int>(x.size()),
      out_ah.data(), static_cast<int>(out_ah.size()), out_scale.data(),
      static_cast<int>(out_scale.size()), out_zp.data(),
      static_cast<int>(out_zp.size()));
    ASSERT_EQ(err, AEE_SUCCESS)
      << "mm_u8i4_gate_up_swiglu failed: " << hex(err) << " M=" << M;

    // Dequantize out_ah with the SAME AH-tile addressing
    // htp_act_quant.h/hvx_quant_pack_u8_ah use, and compare against
    // inter_ref -- this is the stage-1-only check.
    std::vector<float> inter_got(static_cast<size_t>(M) * I);
    for (uint32_t m = 0; m < M; ++m) {
      const uint32_t rb = m / 64, r = m % 64;
      for (uint32_t k = 0; k < I; ++k) {
        const uint32_t kt = k / 32, c = k % 32;
        const size_t idx =
          (static_cast<size_t>(rb) * n_ktiles + kt) * 2048 + r * 32 + c;
        const uint8_t q = out_ah[idx];
        inter_got[static_cast<size_t>(m) * I + k] =
          out_scale[m] * (static_cast<float>(q) - out_zp[m]);
      }
    }
    const double snr_stage1 = snr_db(inter_ref, inter_got);
    std::cout << "U8I4_FIELD path=gate_up_swiglu field=snr_db_stage1 value="
              << snr_stage1 << " (M=" << M << " K=" << K << " I=" << I << ")"
              << std::endl;
    // 30, not 40: this compares against a NEVER-quantized host f32
    // intermediate, unlike the end-to-end test below (which compares two
    // paths that both quantize down's activation, so a lot of noise cancels
    // between them -- see this test's own doc comment). Measured stable at
    // 37.5-37.7 dB across M in {55,64,100,128}: two u8 quantization hops
    // (gate_up's activation, then this requant) with SwiGLU in between,
    // still comfortably above the project's own single-hop floor (S4's
    // 23.5 dB for one bare u8i4 matmul) -- 30 leaves headroom below the
    // measured band without diluting the gate into meaninglessness.
    EXPECT_GT(snr_stage1, 30.0)
      << "gate_up+SwiGLU+requant intermediate below the u8 requantization "
         "floor, M="
      << M;

    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, gu.handle), AEE_SUCCESS);
  } // M sweep
}

/**
 * @brief [L2, split-call variant] end to end: gate_up_swiglu's output fed
 *        straight into mm_u8i4_layer_u8in for the down matmul, compared
 *        against the same two-call reference FusedSwigluMatchesTwoCall
 *        Reference uses. Complements that test's stage-1-only sibling
 *        above: this one exercises the ACTUAL production call sequence
 *        (what ARM dispatch will do), not just stage 1 in isolation.
 */
TEST_F(HmxMmU8I4Layer, GateUpSwigluPlusU8InMatchesTwoCallReference) {
  for (const uint32_t M : {55u, 64u, 100u, 128u, 138u, 200u, 11u}) {
    const uint32_t K = 2048, I = 1792, N = 2048;

    Weight gu, dn;
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, 2 * I, 0xBA5E0005u, gu));
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(I, N, 0xBA5E0006u, dn));

    std::vector<float> x(static_cast<size_t>(M) * K);
    fill_deterministic(x, 0x5EED0002u);

    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * I, 0.0f);
    {
      const uint32_t handles[1] = {gu.handle};
      int err = nntr_hvx_mm_u8i4_layer(
        handle_, M, K, handles, 1, x.data(), static_cast<int>(x.size()),
        gu_out.data(), static_cast<int>(gu_out.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "gate_up layer call failed: " << hex(err);
    }
    std::vector<float> inter(static_cast<size_t>(M) * I);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t j = 0; j < I; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * I + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * I + I + j];
        inter[static_cast<size_t>(m) * I + j] = g / (1.0f + std::exp(-g)) * u;
      }
    }
    std::vector<float> dn_ref(static_cast<size_t>(M) * N, 0.0f);
    {
      const uint32_t handles[1] = {dn.handle};
      int err = nntr_hvx_mm_u8i4_layer(
        handle_, M, I, handles, 1, inter.data(), static_cast<int>(inter.size()),
        dn_ref.data(), static_cast<int>(dn_ref.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "down layer call failed: " << hex(err);
    }

    // Split-call path: gate_up_swiglu, then feed its output straight into
    // mm_u8i4_layer_u8in for down -- the actual sequence ARM dispatch uses.
    const uint32_t m_pad = (M + 63) / 64 * 64;
    std::vector<uint8_t> out_ah(static_cast<size_t>(m_pad) * I, 0);
    std::vector<float> out_scale(m_pad, 1.0f);
    std::vector<int32_t> out_zp(m_pad, 0);
    int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
      handle_, M, K, gu.handle, x.data(), static_cast<int>(x.size()),
      out_ah.data(), static_cast<int>(out_ah.size()), out_scale.data(),
      static_cast<int>(out_scale.size()), out_zp.data(),
      static_cast<int>(out_zp.size()));
    ASSERT_EQ(err, AEE_SUCCESS)
      << "mm_u8i4_gate_up_swiglu failed: " << hex(err);

    std::vector<float> got(static_cast<size_t>(M) * N, 0.0f);
    const uint32_t dn_handles[1] = {dn.handle};
    err = nntr_hvx_mm_u8i4_layer_u8in(
      handle_, M, I, dn_handles, 1, out_ah.data(),
      static_cast<int>(out_ah.size()), out_scale.data(),
      static_cast<int>(out_scale.size()), out_zp.data(),
      static_cast<int>(out_zp.size()), got.data(),
      static_cast<int>(got.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_layer_u8in failed: " << hex(err);

    const double snr = snr_db(dn_ref, got);
    std::cout << "U8I4_FIELD path=gate_up_swiglu_plus_u8in field=snr_db value="
              << snr << " (M=" << M << " K=" << K << " I=" << I << " N=" << N
              << ")" << std::endl;
    EXPECT_GT(snr, 40.0)
      << "split-call fused path below the u8 requantization floor, M=" << M;

    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, gu.handle), AEE_SUCCESS);
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, dn.handle), AEE_SUCCESS);
  } // M sweep
}

/**
 * @brief [L2] Both SwiGLU kernels, with a gate driven below the exp clamp.
 *
 * The coverage hole that let doc 43 section 7's two L2 attempts ship: every
 * other SwiGLU test builds its gate from fill_deterministic x
 * fill_deterministic, whose |gate| never approaches the clamp, so both
 * kernels passed every gate while failing the real model. hvx_swiglu_row_f32
 * clamps -gate to exp_top and feeds a = 1 + exp(t) to hvx_recip_qf32, whose
 * magic seed is a word subtract on bits(a) and goes negative -- NR then
 * diverges to NaN -- once bits(a) > 0x7EF311C2, i.e. t > 87.977861. At the
 * old exp_top = 88.0f EVERY gate at or below the clamp landed there, and one
 * NaN lane poisons hvx_quant_rows_u8_params' whole-row min/max scan, taking
 * that row's scale/zp and everything downstream of it with it.
 *
 * Bias is what makes the gate reachable: it is the one term of the
 * dequantized output a test sets directly. Three gate columns are pinned far
 * below the clamp, spread across separate 32-lane HVX chunks (I = 1792 = 56
 * chunks exactly, so there is no scalar tail to hide in -- every column runs
 * the vector path). The test asserts the gates it built are actually past
 * the clamp before it judges anything, so it cannot quietly stop testing
 * what it claims to if the magnitudes ever drift.
 *
 * Both kernels are exercised because both call hvx_swiglu_inplace_f32 --
 * that shared call, not either kernel's own arithmetic, is what doc 43
 * section 7 identified as the common factor in two independent failures.
 */
TEST_F(HmxMmU8I4Layer, SwigluSurvivesExtremeNegativeGate) {
  const uint32_t K = 2048, I = 1792, N = 2048;

  // Well past the clamp, not at it: gate = matmul + bias, and the matmul
  // term is O(10) for these fills, so -200 and beyond is past -88 with room
  // to spare. The ASSERT_LE below is what actually holds that claim.
  const std::vector<std::pair<uint32_t, float>> gate_overrides = {
    {37u, -200.0f}, {1000u, -300.0f}, {1791u, -400.0f}};

  Weight gu, dn;
  ASSERT_NO_FATAL_FAILURE(
    MakeAndRegister(K, 2 * I, 0xBA5E0007u, gu, gate_overrides));
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(I, N, 0xBA5E0008u, dn));

  for (const uint32_t M : {55u, 64u}) {
    std::vector<float> x(static_cast<size_t>(M) * K);
    fill_deterministic(x, 0x5EED0002u);

    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * I, 0.0f);
    {
      const uint32_t handles[1] = {gu.handle};
      int err = nntr_hvx_mm_u8i4_layer(
        handle_, M, K, handles, 1, x.data(), static_cast<int>(x.size()),
        gu_out.data(), static_cast<int>(gu_out.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "gate_up layer call failed: " << hex(err);
    }

    // The test's own premise, checked: these columns really are past the
    // clamp. Without this the test could pass by never testing anything.
    for (const auto &ov : gate_overrides) {
      for (uint32_t m = 0; m < M; ++m) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * I + ov.first];
        ASSERT_LE(g, -88.0f)
          << "premise broken: gate column " << ov.first << " row " << m
          << " is " << g
          << ", not below the SwiGLU clamp -- this test is no "
             "longer exercising hvx_recip_qf32's failing range";
      }
    }

    std::vector<float> inter_ref(static_cast<size_t>(M) * I);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t j = 0; j < I; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * I + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * I + I + j];
        inter_ref[static_cast<size_t>(m) * I + j] =
          g / (1.0f + std::exp(-g)) * u;
      }
    }

    // Kernel 1: the split gate_up + SwiGLU + requant call.
    const uint32_t m_pad = round_up(M, kTileRow);
    const uint32_t n_ktiles = I / 32;
    std::vector<uint8_t> out_ah(static_cast<size_t>(m_pad) * I, 0);
    std::vector<float> out_scale(m_pad, 1.0f);
    std::vector<int32_t> out_zp(m_pad, 0);
    int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
      handle_, M, K, gu.handle, x.data(), static_cast<int>(x.size()),
      out_ah.data(), static_cast<int>(out_ah.size()), out_scale.data(),
      static_cast<int>(out_scale.size()), out_zp.data(),
      static_cast<int>(out_zp.size()));
    ASSERT_EQ(err, AEE_SUCCESS)
      << "mm_u8i4_gate_up_swiglu failed: " << hex(err) << " M=" << M;

    // The requantization parameters are where a NaN gate lane actually
    // lands: hvx_quant_rows_u8_params scans the whole row for min/max, so
    // one poisoned lane takes the row's scale and zp with it. Checking them
    // directly names the failure instead of leaving it as "SNR was nan".
    for (uint32_t m = 0; m < M; ++m) {
      ASSERT_TRUE(std::isfinite(out_scale[m]))
        << "row " << m << " requant scale is " << out_scale[m]
        << " -- a NaN SwiGLU lane poisoned the row min/max scan (M=" << M
        << ")";
    }

    std::vector<float> inter_got(static_cast<size_t>(M) * I);
    for (uint32_t m = 0; m < M; ++m) {
      const uint32_t rb = m / 64, r = m % 64;
      for (uint32_t k = 0; k < I; ++k) {
        const uint32_t kt = k / 32, c = k % 32;
        const size_t idx =
          (static_cast<size_t>(rb) * n_ktiles + kt) * 2048 + r * 32 + c;
        inter_got[static_cast<size_t>(m) * I + k] =
          out_scale[m] * (static_cast<float>(out_ah[idx]) - out_zp[m]);
      }
    }
    const double snr_stage1 = snr_db(inter_ref, inter_got);
    std::cout << "U8I4_FIELD path=gate_up_swiglu_extreme field=snr_db_stage1 "
                 "value="
              << snr_stage1 << " (M=" << M << ")" << std::endl;
    EXPECT_GT(snr_stage1, 30.0)
      << "gate_up+SwiGLU+requant intermediate degraded by an out-of-clamp "
         "gate, M="
      << M;

    // Kernel 2: the one-call fused path, same gates, end to end.
    std::vector<float> dn_ref(static_cast<size_t>(M) * N, 0.0f);
    {
      const uint32_t handles[1] = {dn.handle};
      int e2 =
        nntr_hvx_mm_u8i4_layer(handle_, M, I, handles, 1, inter_ref.data(),
                               static_cast<int>(inter_ref.size()),
                               dn_ref.data(), static_cast<int>(dn_ref.size()));
      ASSERT_EQ(e2, AEE_SUCCESS) << "down layer call failed: " << hex(e2);
    }
    const uint32_t fused_handles[2] = {gu.handle, dn.handle};
    std::vector<float> got(static_cast<size_t>(M) * N, 0.0f);
    err = nntr_hvx_mm_u8i4_layer_fused(
      handle_, M, K, fused_handles, 2, x.data(), static_cast<int>(x.size()),
      got.data(), static_cast<int>(got.size()));
    ASSERT_EQ(err, AEE_SUCCESS)
      << "mm_u8i4_layer_fused failed: " << hex(err) << " M=" << M;

    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i]))
        << "fused output element " << i << " is " << got[i]
        << " -- NaN reached the down matmul's output (M=" << M << ")";
    }
    const double snr_fused = snr_db(dn_ref, got);
    std::cout << "U8I4_FIELD path=fused_swiglu_extreme field=snr_db value="
              << snr_fused << " (M=" << M << ")" << std::endl;
    EXPECT_GT(snr_fused, 40.0)
      << "fused SwiGLU output degraded by an out-of-clamp gate, M=" << M;
  } // M sweep

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, gu.handle), AEE_SUCCESS);
  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, dn.handle), AEE_SUCCESS);
}

/**
 * @brief u8i8's counterpart to HmxMmU8I4Layer. Same tests, same shapes,
 *        the wider weight quantizer, and the _u8i8 entry points --
 *        mirrored rather than templated on width for the same reason
 *        hexkl_mm_u8i8_dma.c mirrors hexkl_mm_u8i4_dma.c: the u8i4 side
 *        is proven, and a shared test fixture parametrised on width would
 *        need re-verifying that it still exercises u8i4 correctly to
 *        trust either.
 */
/**
 * @brief [doc 45 Gate 0] How much weight the DSP can hold resident.
 *
 * The whole-model plan needs every weight of LFM2.5-8B-A1B -- about 4.3 GB
 * at 4 bits -- resident on the DSP at once, because streaming a layer's
 * weights over FastRPC per forward (~180 MB, ~90 ms) would cost more than
 * the layer saves. Registered weights live in the DSP PD's own heap
 * (hexkl_mm_u8i4_dma.c bakes into VTCM and mallocs the resident copy), so
 * the question is whether that heap, and the PD's address space, reach
 * 4.3 GB. Nothing else in the plan matters if this does not, which is why
 * it is a gate and runs before any of it is built.
 *
 * Registers one gate_up-sized weight's bytes over and over -- the DSP
 * copies each into fresh heap, so the host needs only one buffer -- until
 * registration fails, and reports how far it got and why. The cap is
 * comfortably above the target so a device that can hold more still
 * terminates.
 */
TEST_F(HmxMmU8I4Layer, RegistryCapacity) {
  const uint32_t K = 2048, N = 3584; // gate_up: the model's largest weight
  Weight w;
  MakeAndRegister(K, N, 0xC0FFEEu, w);
  std::vector<uint32_t> handles = {w.handle};

  const uint32_t k_tiles = K / 32u, n_tiles = N / 32u;
  const double gb_per = k_tiles * n_tiles * 512.0 / 1e9; // WH bytes
  const double target_gb = 4.3;
  const size_t cap = static_cast<size_t>(6.0 / gb_per); // stop past 6 GB
  int stop_err = AEE_SUCCESS;

  const auto t0 = std::chrono::steady_clock::now();
  while (handles.size() < cap) {
    uint32_t h = 0xFFFFFFFFu;
    const int err = nntr_hvx_weight_register_u8i4(
      handle_, K, N, w.q_w.data(), static_cast<int>(w.q_w.size()), w.d.data(),
      static_cast<int>(w.d.size()), w.colsum.data(),
      static_cast<int>(w.colsum.size()), w.bias.data(),
      static_cast<int>(w.bias.size()), &h);
    if (err != AEE_SUCCESS) {
      stop_err = err;
      break;
    }
    handles.push_back(h);
  }
  const double secs =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();

  const double gb = handles.size() * gb_per;
  std::cout << "U8I4_FIELD path=registry field=weights_resident value="
            << handles.size() << "\n"
            << "U8I4_FIELD path=registry field=gb_resident value=" << gb << "\n"
            << "U8I4_FIELD path=registry field=stop_reason value="
            << (stop_err == AEE_SUCCESS ? std::string("cap") : hex(stop_err))
            << "\n"
            << "U8I4_FIELD path=registry field=register_ms_per_weight value="
            << (handles.size() > 1 ? secs * 1e3 / (handles.size() - 1) : 0.0)
            << std::endl;

  for (uint32_t h : handles) {
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, h), AEE_SUCCESS);
  }

  // The gate. A FAILED here is the finding, not a broken test: it means the
  // whole-model plan cannot keep its weights resident on this device.
  EXPECT_GE(gb, target_gb) << "the DSP holds " << gb
                           << " GB of registered weight; the whole-model "
                              "plan (doc 45) needs "
                           << target_gb
                           << " GB resident. Registration stopped with "
                           << (stop_err == AEE_SUCCESS ? std::string("the cap")
                                                       : hex(stop_err));
}

class HmxMmU8I8Layer : public HmxMmU8I4 {
protected:
  struct Weight {
    uint32_t handle;
    uint32_t N;
    std::vector<int8_t> q_w;
    std::vector<float> d;
    std::vector<int32_t> colsum;
    std::vector<float> bias;
    std::vector<float> w_f32;
  };

  void MakeAndRegister(uint32_t K, uint32_t N, uint32_t seed, Weight &w) {
    w.N = N;
    w.w_f32.resize(static_cast<size_t>(K) * N);
    fill_deterministic(w.w_f32, seed);
    quantize_weights_symmetric_i8(w.w_f32, K, N, w.q_w, w.d, w.colsum);
    w.bias.resize(N);
    fill_deterministic(w.bias, seed ^ 0xA5A5A5A5u);

    w.handle = 0xFFFFFFFFu;
    int err = nntr_hvx_weight_register_u8i8(
      handle_, K, N, w.q_w.data(), static_cast<int>(w.q_w.size()), w.d.data(),
      static_cast<int>(w.d.size()), w.colsum.data(),
      static_cast<int>(w.colsum.size()), w.bias.data(),
      static_cast<int>(w.bias.size()), &w.handle);
    ASSERT_EQ(err, AEE_SUCCESS) << "weight_register_u8i8 failed: " << hex(err);
    ASSERT_NE(w.handle, 0xFFFFFFFFu) << "handle not written";
  }

  void ExpectedFor(const Weight &w, const std::vector<float> &x, uint32_t M,
                   uint32_t K, std::vector<float> &out) {
    const uint32_t m_pad = round_up(M, kTileRow);
    std::vector<float> scale;
    std::vector<int32_t> zp;
    quantize_act_rows(x, M, m_pad, K, scale, zp);
    std::vector<uint8_t> u_rm;
    quantize_act_values(x, M, m_pad, K, scale, zp, u_rm);
    std::vector<int32_t> acc;
    ref_int_matmul(u_rm, w.q_w, m_pad, K, w.N, acc);
    ref_dequant(acc, M, w.N, scale, zp, w.colsum, w.d, w.bias, out);
  }
};

TEST_F(HmxMmU8I8Layer, ThreeWeightsMatchPerWeightReference) {
  const uint32_t M = 64, K = 256;
  const uint32_t Ns[3] = {128, 256, 64};

  std::vector<Weight> ws(3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(
      MakeAndRegister(K, Ns[i], 0xB0B08001u + i * 0x1000u, ws[i]));
  }

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);

  uint32_t n_total = 0;
  std::vector<uint32_t> handles;
  for (const auto &w : ws) {
    handles.push_back(w.handle);
    n_total += w.N;
  }
  std::vector<float> got(static_cast<size_t>(M) * n_total, 0.0f);

  int err = nntr_hvx_mm_u8i8_layer(
    handle_, M, K, handles.data(), static_cast<int>(handles.size()), x.data(),
    static_cast<int>(x.size()), got.data(), static_cast<int>(got.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i8_layer failed: " << hex(err);

  size_t off = 0;
  for (int i = 0; i < 3; ++i) {
    SCOPED_TRACE("weight " + std::to_string(i) + " N=" + std::to_string(Ns[i]));
    std::vector<float> want;
    ExpectedFor(ws[i], x, M, K, want);
    for (size_t j = 0; j < want.size(); ++j) {
      EXPECT_NEAR(got[off + j], want[j], std::abs(want[j]) * 1e-5f + 1e-6f)
        << "element " << j;
    }
    off += want.size();
  }

  for (const auto &w : ws) {
    EXPECT_EQ(nntr_hvx_weight_release_u8i8(handle_, w.handle), AEE_SUCCESS);
  }
}

TEST_F(HmxMmU8I8Layer, RegisteredWeightSurvivesRepeatedCalls) {
  const uint32_t M = 1, K = 512, N = 128;
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xC0FFEE81u, w));

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);
  const uint32_t handles[1] = {w.handle};

  std::vector<float> first(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> second(static_cast<size_t>(M) * N, 1.0f);
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<float> &dst = pass == 0 ? first : second;
    int err = nntr_hvx_mm_u8i8_layer(handle_, M, K, handles, 1, x.data(),
                                     static_cast<int>(x.size()), dst.data(),
                                     static_cast<int>(dst.size()));
    ASSERT_EQ(err, AEE_SUCCESS) << "pass " << pass << ": " << hex(err);
  }
  EXPECT_EQ(first, second);

  EXPECT_EQ(nntr_hvx_weight_release_u8i8(handle_, w.handle), AEE_SUCCESS);
}

TEST_F(HmxMmU8I8Layer, ReleasedHandleIsRejectedAndSlotIsReused) {
  const uint32_t K = 128, N = 128;
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xD00D8001u, w));
  const uint32_t released = w.handle;
  ASSERT_EQ(nntr_hvx_weight_release_u8i8(handle_, released), AEE_SUCCESS);

  std::vector<float> x(K, 0.0f);
  std::vector<float> out(N, 0.0f);
  const uint32_t handles[1] = {released};
  EXPECT_NE(nntr_hvx_mm_u8i8_layer(handle_, 1, K, handles, 1, x.data(),
                                   static_cast<int>(x.size()), out.data(),
                                   static_cast<int>(out.size())),
            AEE_SUCCESS);
  EXPECT_NE(nntr_hvx_weight_release_u8i8(handle_, released), AEE_SUCCESS)
    << "double release accepted";

  Weight w2;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0xD00D8002u, w2));
  EXPECT_EQ(w2.handle, released);
  EXPECT_EQ(nntr_hvx_weight_release_u8i8(handle_, w2.handle), AEE_SUCCESS);
}

TEST_F(HmxMmU8I8Layer, MismatchedKIsRejected) {
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(256, 128, 0xE0E08001u, w));
  const uint32_t handles[1] = {w.handle};
  std::vector<float> x(128, 0.0f);
  std::vector<float> out(128, 0.0f);
  EXPECT_NE(nntr_hvx_mm_u8i8_layer(handle_, 1, 128, handles, 1, x.data(),
                                   static_cast<int>(x.size()), out.data(),
                                   static_cast<int>(out.size())),
            AEE_SUCCESS);
  EXPECT_EQ(nntr_hvx_weight_release_u8i8(handle_, w.handle), AEE_SUCCESS);
}

/**
 * @brief Per-call cost, u8i4 vs u8i8, both through the layer endpoint --
 *        not against the (u8i4-only) accuracy harness this time, since
 *        there is no u8i8 harness to compare against. Printed, not
 *        asserted, for the same reason as HmxMmU8I4Layer.ReportPerCallCost.
 */
TEST_F(HmxMmU8I8Layer, ReportPerCallCostVsU8I4) {
  const uint32_t M = 64, K = 1024, N = 1024;
  const int kReps = 20;

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0002u);

  auto time_us = [&](const std::function<void()> &fn) {
    fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kReps; ++i) {
      fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / kReps;
  };

  std::vector<Weight> ws8(4);
  std::vector<uint32_t> hs8;
  uint32_t n_total8 = 0;
  for (int i = 0; i < 4; ++i) {
    ASSERT_NO_FATAL_FAILURE(
      MakeAndRegister(K, N, 0xF00D9000u + i * 0x100u, ws8[i]));
    hs8.push_back(ws8[i].handle);
    n_total8 += ws8[i].N;
  }
  std::vector<float> out8(static_cast<size_t>(M) * n_total8, 0.0f);
  const double u8i8_x4_us = time_us([&] {
    nntr_hvx_mm_u8i8_layer(
      handle_, M, K, hs8.data(), static_cast<int>(hs8.size()), x.data(),
      static_cast<int>(x.size()), out8.data(), static_cast<int>(out8.size()));
  });

  std::vector<int8_t> q_w4;
  std::vector<float> d4;
  std::vector<int32_t> colsum4;
  std::vector<float> w4_f32(static_cast<size_t>(K) * N);
  fill_deterministic(w4_f32, 0xF00D9101u);
  quantize_weights_qs4cx(w4_f32, K, N, q_w4, d4, colsum4);
  std::vector<float> bias4(N);
  fill_deterministic(bias4, 0xF00D9102u);
  uint32_t handle4 = 0;
  ASSERT_EQ(nntr_hvx_weight_register_u8i4(
              handle_, K, N, q_w4.data(), static_cast<int>(q_w4.size()),
              d4.data(), static_cast<int>(d4.size()), colsum4.data(),
              static_cast<int>(colsum4.size()), bias4.data(),
              static_cast<int>(bias4.size()), &handle4),
            AEE_SUCCESS);
  const uint32_t handles4[1] = {handle4};
  std::vector<float> out4(static_cast<size_t>(M) * N, 0.0f);
  const double u8i4_x1_us = time_us([&] {
    nntr_hvx_mm_u8i4_layer(handle_, M, K, handles4, 1, x.data(),
                           static_cast<int>(x.size()), out4.data(),
                           static_cast<int>(out4.size()));
  });

  std::cout << "U8I8_FIELD path=layer_x4 field=us_per_matmul value="
            << (u8i8_x4_us / 4.0) << std::endl;
  std::cout << "U8I8_FIELD path=layer_x4_vs_u8i4_x1 field=ratio value="
            << (u8i4_x1_us / (u8i8_x4_us / 4.0)) << std::endl;

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, handle4), AEE_SUCCESS);
  for (const auto &w : ws8) {
    EXPECT_EQ(nntr_hvx_weight_release_u8i8(handle_, w.handle), AEE_SUCCESS);
  }
}

} // namespace

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
