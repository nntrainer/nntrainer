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
};

TEST_F(HmxMmU8I4, HmxBringsUp) {
  // Task 1 gate: the entry point links against libhexkl_micro.a, hw_init
  // succeeds and the HMX lock round-trips. Buffers are unused here.
  std::vector<uint8_t> act(64 * 128, 0);
  std::vector<int8_t> w(128 * 128, 0);
  std::vector<int32_t> acc(64 * 128, 0);
  std::vector<uint8_t> dump(0);

  int err = nntr_hvx_mm_u8i4_from_u8(
    handle_, 64, 128, 128, act.data(), static_cast<int>(act.size()), w.data(),
    static_cast<int>(w.size()), acc.data(), static_cast<int>(acc.size()),
    dump.data(), 0);
  ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_from_u8 failed: " << hex(err);
}

TEST_F(HmxMmU8I4, S2_IntegerAccumulatorIsBitExact) {
  const uint32_t M = 64, K = 128, N = 128;
  const uint32_t m_pad = round_up(M, kTileRow);

  // Host-side weight quantization: fp32 -> per-channel int4.
  std::vector<float> w_f32(static_cast<size_t>(K) * N);
  fill_deterministic(w_f32, 0x5EED0001u);
  std::vector<int8_t> q_w;
  std::vector<float> d;
  std::vector<int32_t> colsum;
  quantize_weights_qs4cx(w_f32, K, N, q_w, d, colsum);

  // Host-side activation: an arbitrary but deterministic uint8 pattern.
  // Activation quantization from fp32 is Task 3; this task isolates the
  // weight bake, the WH layout and the HMX matmul.
  std::vector<uint8_t> u_rm(static_cast<size_t>(m_pad) * K);
  for (size_t i = 0; i < u_rm.size(); ++i) {
    u_rm[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xFFu);
  }
  std::vector<uint8_t> act_ah;
  pack_ah_from_rowmajor(u_rm, m_pad, K, act_ah);

  std::vector<int32_t> expected;
  ref_int_matmul(u_rm, q_w, m_pad, K, N, expected);

  std::vector<int32_t> acc(static_cast<size_t>(m_pad) * N, 0);
  std::vector<uint8_t> dump(static_cast<size_t>(K / kTileInner) *
                            (N / kTileCol) * 512);

  int err = nntr_hvx_mm_u8i4_from_u8(
    handle_, M, K, N, act_ah.data(), static_cast<int>(act_ah.size()),
    q_w.data(), static_cast<int>(q_w.size()), acc.data(),
    static_cast<int>(acc.size()), dump.data(), static_cast<int>(dump.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_from_u8 failed: " << hex(err);

  size_t mismatches = 0;
  for (uint32_t m = 0; m < m_pad && mismatches < 10; ++m) {
    for (uint32_t n = 0; n < N && mismatches < 10; ++n) {
      const size_t i = static_cast<size_t>(m) * N + n;
      if (acc[i] != expected[i]) {
        ++mismatches;
        ADD_FAILURE() << "acc[" << m << "][" << n << "] = " << acc[i]
                      << ", expected " << expected[i];
      }
    }
  }
  EXPECT_EQ(acc, expected);

  // The baked buffer is the WH i4 layout, which HexKL does not document.
  // Print the first tile so the offline converter has a reference.
  std::cout << "WH tile 0 (512B, hex):" << std::endl;
  for (size_t i = 0; i < 512; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(dump[i])
              << ((i % 32 == 31) ? "\n" : " ");
  }
  std::cout << std::dec << std::flush;
}

TEST_F(HmxMmU8I4, S1_ActivationQuantMatchesHost) {
  const uint32_t M = 64, K = 128, N = 128;
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

  for (uint32_t m = 0; m < m_pad; ++m) {
    EXPECT_NEAR(got_scale[m], exp_scale[m], std::abs(exp_scale[m]) * 1e-6f)
      << "scale[" << m << "]";
    EXPECT_EQ(got_zp[m], exp_zp[m]) << "zp[" << m << "]";
  }
  EXPECT_EQ(got_ah, exp_ah);

  // The activation the HMX actually consumed now comes from the HVX
  // kernel, so this re-checks S2 end to end rather than with a fixed
  // pattern.
  EXPECT_EQ(got_acc, exp_acc);
}

TEST_F(HmxMmU8I4, S3_DequantMatchesHost) {
  const uint32_t M = 64, K = 128, N = 128;
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

  ASSERT_EQ(got_acc, exp_acc) << "accumulator diverged; S3 is meaningless";

  size_t reported = 0;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      const size_t i = static_cast<size_t>(m) * N + n;
      const float tol = std::abs(exp_out[i]) * 1e-5f + 1e-6f;
      if (std::abs(got_out[i] - exp_out[i]) > tol && reported < 10) {
        ++reported;
        ADD_FAILURE() << "out[" << m << "][" << n << "] = " << got_out[i]
                      << ", expected " << exp_out[i];
      }
      EXPECT_NEAR(got_out[i], exp_out[i], tol);
    }
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
