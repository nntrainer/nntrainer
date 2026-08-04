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
#include <cmath>
#include <cstdint>
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
  void CheckShape(uint32_t M, uint32_t K, uint32_t N, bool want_wh_dump) {
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
  CheckShape(64, 128, 128, /*want_wh_dump=*/true);
}

TEST_F(HmxMmU8I4, Shape2_DecodeSingleToken) {
  // One token padded out to a 64-row tile: the decode case, and the one
  // that exercises the zero-pad path for 63 of 64 rows.
  CheckShape(1, 1024, 1024, /*want_wh_dump=*/false);
}

TEST_F(HmxMmU8I4, Shape3_PrefillQwen3Scale) {
  // Weights alone need (1024/32)*(1024/32)*512 = 512 KiB of VTCM here.
  CheckShape(64, 1024, 1024, /*want_wh_dump=*/false);
}

TEST_F(HmxMmU8I4, Shape4_MultipleRowBlocks) {
  // Shapes 1 through 3 all pad to 64 rows, so the row-block loop only
  // ever runs with rb=0. This is the one that runs it twice.
  CheckShape(128, 128, 128, /*want_wh_dump=*/false);
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
