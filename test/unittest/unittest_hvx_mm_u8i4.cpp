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
#include <cstring>
#include <dlfcn.h>
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

#include <htp_wh_layout.h>

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
/**
 * @brief Offset AEEStdErr.h adds to every AEE_* code on the DSP side.
 *
 * Skel code is compiled with __hexagon__ defined, where AEEStdErr.h offsets
 * every AEE_* code; this host binary is not, so AEE_EBADSTATE here is plain
 * 13 and the same error arrives from the DSP as 0x8000040d. Only
 * AEE_SUCCESS is 0 on both sides, which is why every other check in this
 * file compares against that one and nothing caught this until a test
 * expected a specific failure. Same constant as unittest_hvx_softmax.cpp
 * and unittest_hvx_attn.cpp.
 */
constexpr int kDspOffset = 0x80000400;

/**
 * @brief HexKL's ARM-side RM->WH conversion, if this device has libsdkl.so.
 *
 * sdkl_cpu_* takes no domain argument where sdkl_npu_* does, so this runs on
 * the application processor: the same layout the DSP builds at registration,
 * without a FastRPC round trip. dlopen rather than linked, matching how this
 * file already reaches rpcmem and fastrpc_mmap -- a HexKL drop with no armv8
 * build should skip these tests, not fail to link the binary.
 *
 * Signature from sdkl.h: the output is
 * ((rows+31)/32)*((cols+31)/32)*512 bytes, which is whBytes(),
 * and the input is one sign-extended int8 per i4 value in row-major order,
 * which is what htp_qs4cx_from_packed already produces.
 */
using SdklRmToWh = int (*)(uint8_t *, int8_t *, size_t, size_t);

/** Shared with the offline quantizer, so this checks the code the model file
 *  is built with rather than a copy of it. */
using nntrainer::whPack;
using nntrainer::whSlot;

SdklRmToWh loadSdklRmToWh() {
  static void *lib = dlopen("libsdkl.so", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<SdklRmToWh>(dlsym(lib, "sdkl_cpu_i4_rm_to_i4_wh"));
}

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
/**
 * @brief [doc 46 V1] The batched MoE layer against the 64-call path it
 *        replaces, bit for bit.
 *
 * Not SNR. The two paths run the same quantizer, the same swiglu_det and
 * the same HMX kernel on the same bytes; only the call structure differs,
 * so anything but identical output is a bug in the batching, and an SNR
 * gate would hide exactly the class of one-level differences doc 44
 * section 3 spent a week tracking down.
 *
 * The routing here is shaped like real routing rather than uniform: one
 * expert spanning several 64-row blocks, one with no rows at all, one with
 * exactly a block, and token rows repeating across experts the way top-k
 * produces them. Those are the cases the kernel's expert compaction and
 * its cross-expert weight prefetch actually turn on.
 */
TEST_F(HmxMmU8I4Layer, MoeLayerMatchesTwoCallReference) {
  const uint32_t K = 2048, I = 1792, N = 2048, M = 200, NE = 4;

  std::vector<Weight> gu(NE), dn(NE);
  std::vector<uint32_t> h_gu(NE), h_dn(NE);
  for (uint32_t e = 0; e < NE; ++e) {
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, 2 * I, 0xB0E00000u + e, gu[e]));
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(I, N, 0xD0000000u + e, dn[e]));
    h_gu[e] = gu[e].handle;
    h_dn[e] = dn[e].handle;
  }

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0007u);

  // 70 spans two blocks, 0 exercises the compaction, 64 is the boundary.
  const std::vector<uint32_t> row_count = {70u, 0u, 64u, 33u};
  std::vector<uint32_t> row_index;
  std::vector<float> row_weight;
  {
    uint32_t st = 0xC0FFEEu;
    for (uint32_t e = 0; e < NE; ++e) {
      // Top-k routing gives a token k DISTINCT experts, so one expert sees a
      // given row at most once. moe_scatter_worker splits a block by row and
      // relies on that: two entries in one block with the same row_index are
      // two workers doing read-modify-write on one output row. Drawing with
      // replacement here put 13 such pairs in expert 0 alone, which made this
      // comparison a race against an input real routing never produces.
      std::vector<bool> taken(M, false);
      for (uint32_t i = 0; i < row_count[e]; ++i) {
        uint32_t r;
        do {
          st = st * 1664525u + 1013904223u;
          r = (st >> 8) % M;
        } while (taken[r]);
        taken[r] = true;
        row_index.push_back(r);
        st = st * 1664525u + 1013904223u;
        row_weight.push_back(0.1f + 0.9f * ((st >> 8) % 1000u) / 1000.0f);
      }
    }
  }

  // Rows several experts land on are the ones where a reordered scatter shows
  // up, since float addition is associative only for two terms. Counting them
  // turns a bad_elems number into a place to look: confined to these rows means
  // accumulation order, spread beyond them means the matmul or the quantizer.
  std::vector<uint32_t> hits(M, 0);
  for (uint32_t r : row_index) {
    ++hits[r];
  }
  {
    std::vector<uint32_t> hist(NE + 1, 0);
    for (uint32_t h : hits) {
      ++hist[h];
    }
    for (uint32_t k = 2; k <= NE; ++k) {
      std::cout << "U8I4_FIELD path=moe_layer field=rows_with_" << k
                << "_experts value=" << hist[k] << std::endl;
    }
  }

  // --- reference: the two calls per expert, and the routing multiply and
  // scatter-add on the host, which is exactly what Lfm2MoELayer does today.
  std::vector<float> want(static_cast<size_t>(M) * N, 0.0f);
  {
    uint32_t base = 0;
    for (uint32_t e = 0; e < NE; ++e) {
      const uint32_t n_e = row_count[e];
      if (n_e == 0) {
        continue;
      }
      std::vector<float> xe(static_cast<size_t>(n_e) * K);
      for (uint32_t i = 0; i < n_e; ++i) {
        std::memcpy(&xe[static_cast<size_t>(i) * K],
                    &x[static_cast<size_t>(row_index[base + i]) * K],
                    sizeof(float) * K);
      }
      const uint32_t m_pad = (n_e + 63) / 64 * 64;
      std::vector<uint8_t> mid_ah(static_cast<size_t>(m_pad) * I, 0);
      std::vector<float> mid_scale(m_pad, 1.0f);
      std::vector<int32_t> mid_zp(m_pad, 0);
      int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
        handle_, n_e, K, h_gu[e], xe.data(), static_cast<int>(xe.size()),
        mid_ah.data(), static_cast<int>(mid_ah.size()), mid_scale.data(),
        static_cast<int>(mid_scale.size()), mid_zp.data(),
        static_cast<int>(mid_zp.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "reference gate_up_swiglu: " << hex(err);

      std::vector<float> ye(static_cast<size_t>(n_e) * N, 0.0f);
      const uint32_t dh[1] = {h_dn[e]};
      err = nntr_hvx_mm_u8i4_layer_u8in(
        handle_, n_e, I, dh, 1, mid_ah.data(), static_cast<int>(mid_ah.size()),
        mid_scale.data(), static_cast<int>(mid_scale.size()), mid_zp.data(),
        static_cast<int>(mid_zp.size()), ye.data(),
        static_cast<int>(ye.size()));
      ASSERT_EQ(err, AEE_SUCCESS) << "reference layer_u8in: " << hex(err);

      for (uint32_t i = 0; i < n_e; ++i) {
        float *dst = &want[static_cast<size_t>(row_index[base + i]) * N];
        const float *src = &ye[static_cast<size_t>(i) * N];
        const float w = row_weight[base + i];
        for (uint32_t c = 0; c < N; ++c) {
          // Two statements, so the product is rounded before it is added.
          // Written as one, clang contracts this to FMLA at the default
          // -ffp-contract=on and rounds once, while hvx_scale_add_rows_f32
          // on the DSP multiplies and adds separately. A row one expert
          // wrote still matched -- dst is 0 there and fma(s, w, 0) rounds
          // the same -- so the gap only opened on the second contribution,
          // which is what made it look like the batching.
          const float p = src[c] * w;
          dst[c] = dst[c] + p;
        }
      }
      base += n_e;
    }
  }

  // --- the batched call
  std::vector<float> got(static_cast<size_t>(M) * N, 1.0f); // not pre-zeroed
  int err = nntr_hvx_mm_u8i4_moe_layer(
    handle_, M, K, I, N, h_gu.data(), static_cast<int>(h_gu.size()),
    h_dn.data(), static_cast<int>(h_dn.size()), row_index.data(),
    static_cast<int>(row_index.size()), row_count.data(),
    static_cast<int>(row_count.size()), row_weight.data(),
    static_cast<int>(row_weight.size()), x.data(), static_cast<int>(x.size()),
    got.data(), static_cast<int>(got.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "mm_u8i4_moe_layer failed: " << hex(err);

  size_t bad = 0;
  size_t first = got.size();
  size_t bad_single = 0; // on a row exactly one expert wrote
  uint32_t max_ulp = 0;
  std::vector<uint32_t> bad_per_row(M, 0);
  for (size_t i = 0; i < got.size(); ++i) {
    if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
      if (bad == 0) {
        first = i;
      }
      ++bad;
      ++bad_per_row[i / N];
      if (hits[i / N] <= 1u) {
        ++bad_single;
      }
      // Distance in representable steps. Same sign and both finite here, so
      // the bit patterns as integers are monotone and subtracting them counts
      // the floats in between: 1 is adjacent, which only reassociation does.
      uint32_t a, b;
      std::memcpy(&a, &got[i], sizeof a);
      std::memcpy(&b, &want[i], sizeof b);
      const uint32_t d = (a > b) ? (a - b) : (b - a);
      if (d > max_ulp) {
        max_ulp = d;
      }
    }
  }
  std::cout << "U8I4_FIELD path=moe_layer field=bad_elems value=" << bad
            << " of " << got.size() << std::endl;
  std::cout << "U8I4_FIELD path=moe_layer field=bad_on_single_expert_rows"
               " value="
            << bad_single << std::endl;
  std::cout << "U8I4_FIELD path=moe_layer field=max_ulp value=" << max_ulp
            << std::endl;
  // Spread decides where to look next. A mismatch sitting on a few rows is
  // about those rows -- how their contributions were combined. One that
  // touches most routed rows a little is the matmul or the quantizer, which
  // every row goes through.
  {
    uint32_t bad_rows = 0, routed_rows = 0, worst = 0;
    for (uint32_t r = 0; r < M; ++r) {
      if (hits[r] != 0u) {
        ++routed_rows;
      }
      if (bad_per_row[r] != 0u) {
        ++bad_rows;
        if (bad_per_row[r] > worst) {
          worst = bad_per_row[r];
        }
      }
    }
    std::cout << "U8I4_FIELD path=moe_layer field=bad_rows value=" << bad_rows
              << " of " << routed_rows << " routed" << std::endl;
    std::cout << "U8I4_FIELD path=moe_layer field=worst_row_bad_cols value="
              << worst << " of " << N << std::endl;
    std::cout << "U8I4_FIELD path=moe_layer field=first_bad_row_experts value="
              << (first < got.size() ? hits[first / N] : 0u) << std::endl;
  }
  if (bad != 0) {
    std::cout << "  first at " << first << " (row " << first / N << " col "
              << first % N << "): got " << std::hexfloat << got[first]
              << " want " << want[first] << std::defaultfloat << std::endl;
  }
  EXPECT_EQ(bad, 0u)
    << "the batched MoE layer differs from the 64-call path it replaces; "
       "same quantizer, same SwiGLU, same HMX kernel, so this is the "
       "batching";

  for (uint32_t e = 0; e < NE; ++e) {
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, h_gu[e]), AEE_SUCCESS);
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, h_dn[e]), AEE_SUCCESS);
  }
}

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

  // Releasing is only a fair check while the PD can still service a call.
  // Registration stops here by running the PD out of memory (0x8000040d is
  // raised before the registration function runs), and a PD in that state
  // fails every call including this one -- which printed 516 identical
  // failures and turned the suite red for the ceiling working as measured.
  // The code they fail with says as much: 0x27, which is neither success nor
  // the AEE_EBADPARM hexkl_weight_u8i4_release can return, so the call never
  // reached it. So count them, and assert only in the case where the PD is
  // healthy: stopping at our own cap means releases have no excuse.
  size_t released = 0;
  int first_release_err = AEE_SUCCESS;
  for (uint32_t h : handles) {
    const int rerr = nntr_hvx_weight_release_u8i4(handle_, h);
    if (rerr == AEE_SUCCESS) {
      ++released;
    } else if (first_release_err == AEE_SUCCESS) {
      first_release_err = rerr;
    }
  }
  std::cout << "U8I4_FIELD path=registry field=released value=" << released
            << " of " << handles.size() << "\n"
            << "U8I4_FIELD path=registry field=first_release_err value="
            << (first_release_err == AEE_SUCCESS ? std::string("none")
                                                 : hex(first_release_err))
            << std::endl;
  if (stop_err == AEE_SUCCESS) {
    EXPECT_EQ(released, handles.size())
      << "registration stopped at the cap, so the PD was healthy and every "
         "handle should have released; first error "
      << hex(first_release_err);
  }

  // Reports; does not assert a target. It did assert >= 4.3 GB, which was
  // right while that was an open question and wrong once it was answered:
  // this path tops out at 1.89 GB on device (doc 45 section 8), the DSP heap
  // alone reaches 3.75 and ION at least 6 (section 9), and the design moved
  // the weights into an ION arena because of it. Asserting a target against
  // a path that is being replaced turns every run red and teaches people to
  // skim past failures.
  //
  // What would be a real regression is this number collapsing -- so that is
  // what is checked, along with every handle releasing cleanly above.
  (void)target_gb;
  EXPECT_GT(handles.size(), 1u)
    << "no weight registered at all; registration is broken, not merely "
       "bounded. Stopped with "
    << (stop_err == AEE_SUCCESS ? std::string("the cap") : hex(stop_err));
}

/**
 * @brief [doc 45 Gate 0b] Which of the two memory limits stopped Gate 0.
 *
 * Gate 0 registered 1.89 GB of the 4.10 the whole-model plan needs and
 * failed with 0x8000040d -- raised before the registration function ran, so
 * the PD could not service the call rather than running out of the malloc
 * heap the baked weights sit in. Section 8.4's redesign moves those weights
 * into one ION arena the DSP maps and reads directly, which gets past a PD
 * heap limit but not past the DSP's virtual address space. This measures
 * three numbers that separate them:
 *
 *   dsp_heap_gb   what the DSP can malloc and touch with no payload in
 *                 flight -- the heap ceiling on its own
 *   ion_single_gb the largest single rpcmem buffer the DSP can touch every
 *                 page of; what the one-arena design needs to be >= 4.10
 *   ion_total_gb  how much rpcmem the host can allocate and have the DSP
 *                 touch across several buffers -- the fallback if one
 *                 arena is capped
 *
 * Not an assertion. Three branches follow from the numbers (doc 45 section
 * 8.5) and only one of them is a failure, so this reports and passes; the
 * decision is a design one, not a regression.
 */
/**
 * @brief [doc 46 section 32.4] Gate 0c: can the DSP map a host ION buffer
 *        and DMA out of it?
 *
 * The converted-model plan rests on one mechanism this tree has never
 * exercised. Every weight path so far copies bytes to the DSP, which then
 * owns them in its own heap; the arena instead hands the DSP a file
 * descriptor, maps it once, and leaves the bytes where they are. Gate 0b
 * measured how much ION the host could allocate -- it said nothing about
 * whether the DSP can reach it.
 *
 * Reports rather than asserts, like the other two probes: a device that
 * cannot do this is a fact about the plan, not a broken build. The one
 * thing it does check is the checksum, because a mapping that succeeds and
 * reads as zeros would otherwise look like a pass with a very good
 * bandwidth number.
 */
TEST_F(HmxMmU8I4Layer, ArenaMapAndDma) {
  auto field = [](const char *k, const std::string &v) {
    std::cout << "U8I4_FIELD path=arena field=" << k << " value=" << v << "\n";
  };

  auto init = (void (*)(void))dlsym(RTLD_DEFAULT, "rpcmem_init");
  auto alloc =
    (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto rfree = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
  auto to_fd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
  field("rpcmem_to_fd", to_fd ? "yes" : "no");
  if (!alloc || !rfree || !to_fd) {
    GTEST_SKIP() << "no rpcmem_to_fd -- the arena cannot be handed to the DSP";
  }
  if (init) {
    init();
  }

  // One gate_up weight's worth, so the rate is directly comparable to the
  // 27-33 GB/s the profile reports for the same transfer out of DSP heap.
  const uint32_t kBytes = 3670016u;
  const uint32_t kDma = 1048576u; // fits VTCM with room to spare
  void *buf = alloc(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 1, (int)kBytes);
  if (buf == nullptr) {
    field("alloc", "failed");
    GTEST_SKIP() << "rpcmem_alloc failed";
  }
  // A pattern, not zeros: the checksum below has to be able to tell a live
  // mapping from one that reads back empty.
  auto *p = static_cast<uint8_t *>(buf);
  uint32_t want = 0;
  for (uint32_t i = 0; i < kBytes; ++i) {
    p[i] = static_cast<uint8_t>(i * 31u + 7u);
  }
  for (uint32_t i = 0; i < kDma; i += 64u) {
    want += p[i];
  }

  const int fd = to_fd(buf);
  field("fd", std::to_string(fd));

  // An fd number means nothing to the DSP until FastRPC has attached the
  // buffer to the session -- which is what the first attempt at this probe
  // missed: HAP_mem.h was present and HAP_mmap linked, and it still
  // returned null. fastrpc_mmap is the documented call that does the
  // attaching. Resolved rather than linked because a libcdsprpc.so without
  // it is a fact worth reporting, not a link error.
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  using FastrpcMunmap = int (*)(int, int, void *, size_t);
  auto fmmap = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
  auto fmunmap = (FastrpcMunmap)dlsym(RTLD_DEFAULT, "fastrpc_munmap");
  field("fastrpc_mmap", fmmap ? "yes" : "no");

  // Taken from remote.h rather than written as numbers: attempt 3 passed 0
  // with a comment calling it FASTRPC_MAP_FD, and 0 is FASTRPC_MAP_STATIC --
  // the mapping the driver makes for a buffer passed as a call argument,
  // pinned to one remote address and not tagged with the fd. FASTRPC_MAP_FD
  // is the one whose documentation says the DSP fetches the address with
  // HAP_mmap_get / HAP_mmap_put, which is what the probe calls. The values
  // are printed so the log carries what this build actually sent.
  const int kCdspDomain = CDSP_DOMAIN_ID;
  const int kMapFd = static_cast<int>(FASTRPC_MAP_FD);
  field("map_domain", std::to_string(kCdspDomain));
  field("map_flag", std::to_string(kMapFd));
  bool attached = false;
  if (fmmap != nullptr) {
    const int rc = fmmap(kCdspDomain, fd, buf, 0, kBytes, kMapFd);
    field("fastrpc_mmap_rc", hex(rc));
    attached = (rc == 0);
  }

  std::vector<uint32_t> res(10, 0);
  const int err = nntr_hvx_arena_probe(handle_, fd, kBytes, kDma, res.data(),
                                       (int)res.size());
  field("err", hex(err));
  field("attached", attached ? "yes" : "no");
  // Which of the two mapping calls got there, if either -- the DSP reports
  // them apart so a failure names the API rather than just the outcome.
  // Which prot/flags pair the device accepted, and what the rejected ones
  // said -- two bits each, 1 null and 2 MAP_FAILED, in the order the DSP
  // tries them.
  static const char *kTryName[] = {"none",      "rw|shared", "r|shared",
                                   "rw|private", "r|private", "rw|0",
                                   "mmap_get"};
  field("hap_mmap_accepted", res[6] < (sizeof(kTryName) / sizeof(kTryName[0]))
                               ? kTryName[res[6]]
                               : "?");
  field("hap_mmap_fail_mask", hex((int)res[7]));
  // Whether the DSP had HAP_mmap_get at all, and what it said. An fd the
  // host attached with fastrpc_mmap is already mapped on the DSP, so this
  // asks for that address instead of making a second mapping -- which is
  // what every prot/flags pair refused to do.
  field("hap_mmap_get_rc", hex((int)res[8]));
  // Low half of the physical address it reported. Zero with rc 0 would mean
  // the call answered without actually pointing anywhere.
  field("hap_mmap_get_paddr_lo", hex((int)res[9]));
  if (err == AEE_SUCCESS) {
    const double gbs = res[2] > 0 ? (double)res[3] / res[2] / 1000.0 : 0.0;
    field("mapped", res[0] ? "yes" : "no");
    field("map_us", std::to_string(res[1]));
    field("dma_us", std::to_string(res[2]));
    field("dma_gbs", std::to_string(gbs));
    field("checksum_ok", res[4] == want ? "yes" : "no");
    field("unmapped", res[5] ? "yes" : "no");
    EXPECT_EQ(res[4], want)
      << "the mapping was readable but did not carry the host's bytes";
  }
  if (attached && fmunmap != nullptr) {
    field("fastrpc_munmap_rc", hex(fmunmap(kCdspDomain, fd, buf, kBytes)));
  }
  rfree(buf);
}

/**
 * @brief [doc 46 section 35] The closed form reproduces the DSP's own WH
 *        bytes, at the shape that matters.
 *
 * This is the question the offline quantizer turns on: can a host build the
 * bytes the matmul reads, without the DSP. whPack is the candidate
 * and hexkl_micro_hmx_rm_to_wh_i4's output, fetched with weight_bake_export,
 * is the truth.
 *
 * memcmp, not a tolerance: a permutation of nibbles either matches or the
 * two are different layouts.
 *
 * sdkl_cpu_i4_rm_to_i4_wh is measured alongside but is no longer the
 * reference. Its first run disagreed with the DSP on 99.5% of a 2048x3584
 * weight while agreeing on a 32x32 one, which is what a swapped
 * rows/cols argument looks like -- square inputs cannot tell the two apart,
 * and hexkl_macro_i4_rm_to_i4_wh takes (n_col, n_inner) where this one is
 * documented (wt_rows, wt_cols). So both orders are tried and reported.
 */
TEST_F(HmxMmU8I4Layer, WhPackReferenceMatchesDspBake) {
  // The model's largest weight, and deliberately not square.
  const uint32_t K = 2048, N = 3584;
  Weight w;
  ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, N, 0x5DC10000u, w));

  const uint32_t wh_len = (K / 32u) * (N / 32u) * 512u;
  std::vector<uint8_t> from_dsp(wh_len, 0);
  ASSERT_EQ(nntr_hvx_weight_bake_export(handle_, w.handle, from_dsp.data(),
                                        (int)wh_len),
            AEE_SUCCESS);

  auto count_diff = [&](const std::vector<uint8_t> &v, size_t *first) {
    size_t bad = 0;
    *first = wh_len;
    for (size_t i = 0; i < wh_len; ++i) {
      if (v[i] != from_dsp[i]) {
        if (bad == 0) {
          *first = i;
        }
        ++bad;
      }
    }
    return bad;
  };

  std::vector<uint8_t> host(wh_len, 0);
  const auto t0 = std::chrono::steady_clock::now();
  whPack(w.q_w.data(), K, N, host.data());
  const auto host_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
  size_t first = 0;
  const size_t bad = count_diff(host, &first);
  std::cout << "U8I4_FIELD path=wh_layout field=closed_form_bad_bytes value="
            << bad << " of " << wh_len << "\n"
            << "U8I4_FIELD path=wh_layout field=host_pack_us value=" << host_us
            << "\n"
            << "U8I4_FIELD path=wh_layout field=host_pack_mbps value="
            << (host_us > 0 ? (double)wh_len / (double)host_us : 0.0)
            << std::endl;
  if (bad != 0) {
    std::cout << "  first at " << first << " (tile " << first / 512 << " of "
              << wh_len / 512 << ", byte " << first % 512 << "): host "
              << hex(host[first]) << " dsp " << hex(from_dsp[first])
              << std::endl;
  }
  EXPECT_EQ(bad, 0u)
    << "whSlot does not describe what hexkl_micro_hmx_rm_to_wh_i4 produces "
       "at this shape; the offline packer cannot be written from it yet";

  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, w.handle), AEE_SUCCESS);
}

/**
 * @brief [doc 46 section 35] Reads the permutation out of the DSP itself.
 *
 * The first version of whSlot came from sdkl_cpu_i4_rm_to_i4_wh and matched
 * it exactly, square and non-square alike, and still disagreed with the DSP's
 * own bake on 3639322 of 3670016 bytes. Tile order was not the difference:
 * hexkl_weight_u8i4_register sets n_tiles_row = N/32 and takes
 * kt = t/n_tiles_row, which is what whPack does. The two APIs simply
 * build transposed tiles -- this table is the library's with r and c swapped
 * -- and the matmul here is hexkl_micro_hmx_mm_u8i4, so this is the one that
 * counts. Deriving from the library rather than from the thing under test
 * cost two device rounds.
 *
 * Four probes rather than the 1024 single-element calls the library version
 * uses, because each of these is a FastRPC round trip: source index p is 10
 * bits and an i4 safely holds 0..7, so p's bits arrive three at a time and
 * are reassembled per output slot.
 */
TEST_F(HmxMmU8I4Layer, WhLayoutTableFromDsp) {
  const uint32_t K = 32, N = 32;
  const std::vector<float> scale(N, 1.0f), bias(N, 0.0f);
  const std::vector<int32_t> colsum(N, 0);

  std::vector<uint32_t> src_of(1024, 0); // output slot -> source index
  for (int j = 0; j < 4; ++j) {
    std::vector<int8_t> rm(1024);
    for (uint32_t p = 0; p < 1024u; ++p) {
      rm[p] = static_cast<int8_t>((p >> (3 * j)) & 7u);
    }
    uint32_t h = 0xFFFFFFFFu;
    ASSERT_EQ(nntr_hvx_weight_register_u8i4(
                handle_, K, N, rm.data(), (int)rm.size(), scale.data(), (int)N,
                colsum.data(), (int)N, bias.data(), (int)N, &h),
              AEE_SUCCESS);
    std::vector<uint8_t> wh(512, 0);
    ASSERT_EQ(nntr_hvx_weight_bake_export(handle_, h, wh.data(), 512),
              AEE_SUCCESS);
    EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, h), AEE_SUCCESS);
    for (uint32_t sl = 0; sl < 1024u; ++sl) {
      const uint32_t v = (wh[sl / 2] >> (4 * (sl % 2))) & 0x0Fu;
      src_of[sl] |= (v & 7u) << (3 * j);
    }
  }

  std::vector<int> slot_of(1024, -1);
  std::vector<int> hits(1024, 0);
  for (uint32_t sl = 0; sl < 1024u; ++sl) {
    ASSERT_LT(src_of[sl], 1024u)
      << "slot " << sl << " reassembled to source " << src_of[sl]
      << "; the bake does not pass i4 values through unchanged";
    slot_of[src_of[sl]] = (int)sl;
    ++hits[src_of[sl]];
  }
  for (uint32_t p = 0; p < 1024u; ++p) {
    ASSERT_EQ(hits[p], 1) << "source " << p << " landed in " << hits[p]
                          << " slots; not a bijection";
  }

  size_t form_bad = 0;
  for (uint32_t r = 0; r < 32u; ++r) {
    for (uint32_t c = 0; c < 32u; ++c) {
      if (slot_of[r * 32u + c] != (int)whSlot(r, c)) {
        ++form_bad;
      }
    }
  }
  std::cout << "U8I4_FIELD path=wh_layout field=dsp_closed_form_bad_slots "
               "value="
            << form_bad << " of 1024" << std::endl;
  if (form_bad != 0) {
    std::cout << "WH_DSP_SLOT_TABLE begin (row r, col c) -> nibble slot\n";
    for (uint32_t r = 0; r < 32u; ++r) {
      std::cout << "WH_DSP_SLOT_ROW " << std::setw(2) << r << ":";
      for (uint32_t c = 0; c < 32u; ++c) {
        std::cout << " " << slot_of[r * 32u + c];
      }
      std::cout << "\n";
    }
    std::cout << "WH_DSP_SLOT_TABLE end" << std::endl;
  }
  EXPECT_EQ(form_bad, 0u)
    << "hexkl_micro_hmx_rm_to_wh_i4 uses a different tile layout than "
       "sdkl_cpu_i4_rm_to_i4_wh, which whSlot was derived from";

  // Is the tile ORDER also what whPack assumes? Checked separately
  // and on a non-square shape, because a wrong order and a wrong intra-tile
  // layout look the same in a single byte count.
  const uint32_t K2 = 64, N2 = 128;
  std::vector<int8_t> big(static_cast<size_t>(K2) * N2);
  for (size_t i = 0; i < big.size(); ++i) {
    big[i] = static_cast<int8_t>((int)(i % 15u) - 7);
  }
  const std::vector<float> s2(N2, 1.0f), b2(N2, 0.0f);
  const std::vector<int32_t> c2(N2, 0);
  uint32_t h2 = 0xFFFFFFFFu;
  ASSERT_EQ(nntr_hvx_weight_register_u8i4(
              handle_, K2, N2, big.data(), (int)big.size(), s2.data(), (int)N2,
              c2.data(), (int)N2, b2.data(), (int)N2, &h2),
            AEE_SUCCESS);
  const uint32_t len2 = (K2 / 32u) * (N2 / 32u) * 512u;
  std::vector<uint8_t> dsp2(len2, 0), host2(len2, 0);
  ASSERT_EQ(nntr_hvx_weight_bake_export(handle_, h2, dsp2.data(), (int)len2),
            AEE_SUCCESS);
  EXPECT_EQ(nntr_hvx_weight_release_u8i4(handle_, h2), AEE_SUCCESS);
  whPack(big.data(), K2, N2, host2.data());
  size_t d2 = 0, tiles_bad = 0;
  for (uint32_t t = 0; t < len2 / 512u; ++t) {
    if (std::memcmp(dsp2.data() + t * 512u, host2.data() + t * 512u, 512) != 0)
      ++tiles_bad;
  }
  for (uint32_t i = 0; i < len2; ++i) {
    if (dsp2[i] != host2[i])
      ++d2;
  }
  std::cout << "U8I4_FIELD path=wh_layout field=dsp_nonsquare_bad value=" << d2
            << " of " << len2 << "\n"
            << "U8I4_FIELD path=wh_layout field=dsp_nonsquare_tiles_bad value="
            << tiles_bad << " of " << (len2 / 512u) << std::endl;
}

/**
 * @brief [doc 46 section 35] Reads the WH nibble permutation out of the
 *        library, so the offline quantizer can reproduce it.
 *
 * One tile is 32x32 i4 = 1024 nibbles = 512 bytes = WEIGHT_TILE_BYTES_U8I4
 * exactly, so there is no room in the output for anything but the values:
 * the transform has to be a pure bijection of nibbles. That is what makes it
 * derivable at all, and this checks it rather than assuming it.
 *
 * Derived the boring way -- one source element set to 7, everything else 0,
 * 1024 times -- because the alternative (packing position bits into 4-bit
 * values) is cleverness this does not need at 512 bytes a call. Every slot
 * landing exactly once is the proof that it is a bijection.
 */
TEST_F(HmxMmU8I4Layer, WhLayoutTable) {
  auto rm_to_wh = loadSdklRmToWh();
  if (rm_to_wh == nullptr) {
    GTEST_SKIP() << "libsdkl.so / sdkl_cpu_i4_rm_to_i4_wh not on this device";
  }

  // Which output nibble slot each source element lands in. Slot s is the low
  // half of byte s/2 when s is even, the high half when odd -- recorded as
  // one number so the packer can be written as out_nibble[slot_of[p]] = v.
  std::vector<int> slot_of(1024, -1);
  std::vector<int> hits(1024, 0);
  std::vector<int8_t> rm(1024, 0);
  std::vector<uint8_t> wh(512, 0);

  for (uint32_t p = 0; p < 1024u; ++p) {
    std::fill(rm.begin(), rm.end(), int8_t(0));
    rm[p] = 7;
    std::fill(wh.begin(), wh.end(), uint8_t(0));
    ASSERT_EQ(rm_to_wh(wh.data(), rm.data(), 32, 32), AEE_SUCCESS);
    int found = -1, n_found = 0;
    for (uint32_t b = 0; b < 512u; ++b) {
      if ((wh[b] & 0x0Fu) == 7u) {
        found = (int)(2 * b);
        ++n_found;
      }
      if ((wh[b] >> 4) == 7u) {
        found = (int)(2 * b + 1);
        ++n_found;
      }
    }
    ASSERT_EQ(n_found, 1) << "source " << p << " produced " << n_found
                          << " nibbles of value 7; the transform is not a "
                             "plain permutation of values";
    slot_of[p] = found;
    ++hits[found];
  }
  for (uint32_t sl = 0; sl < 1024u; ++sl) {
    ASSERT_EQ(hits[sl], 1) << "slot " << sl << " was written by " << hits[sl]
                           << " sources; not a bijection";
  }
  std::cout << "U8I4_FIELD path=wh_layout field=bijection value=yes"
            << std::endl;

  // This library's tile is the transpose of the one the DSP bakes, so the
  // check is whSlot(c, r). Recorded rather than dropped: it is what makes
  // "the CPU converter and the micro API differ" a fact instead of a story,
  // and it is the relationship anyone reaching for sdkl_cpu_i4_rm_to_i4_wh
  // again will need.
  size_t form_bad = 0;
  for (uint32_t r = 0; r < 32u; ++r) {
    for (uint32_t c = 0; c < 32u; ++c) {
      if (slot_of[r * 32u + c] != (int)whSlot(c, r)) {
        ++form_bad;
      }
    }
  }
  std::cout << "U8I4_FIELD path=wh_layout field=closed_form_bad_slots value="
            << form_bad << " of 1024" << std::endl;
  EXPECT_EQ(form_bad, 0u)
    << "whSlot(c, r) no longer describes this library's layout";
  // 32 lines, one per source row, each 32 slot numbers. This is the whole
  // answer -- the offline packer needs nothing else -- and it is what the
  // closed form was read off the first time.
  if (form_bad != 0) {
    std::cout << "WH_SLOT_TABLE begin (row r, col c) -> nibble slot\n";
    for (uint32_t r = 0; r < 32u; ++r) {
      std::cout << "WH_SLOT_ROW " << std::setw(2) << r << ":";
      for (uint32_t c = 0; c < 32u; ++c) {
        std::cout << " " << slot_of[r * 32u + c];
      }
      std::cout << "\n";
    }
    std::cout << "WH_SLOT_TABLE end" << std::endl;
  }
}

/**
 * @brief [doc 46 section 34.6 item 1] The ordering the arena actually uses.
 *
 * ArenaMapAndDma proved a cached buffer written BEFORE fastrpc_mmap. The
 * arena does the opposite: it allocates uncached, attaches once, and then
 * keeps writing weights into it for the rest of the run, with no flush --
 * which is the whole reason it is uncached. That ordering is the one thing
 * section 34 rests on that nothing has shown yet, so this shows it or says
 * it does not hold.
 */
TEST_F(HmxMmU8I4Layer, ArenaUncachedWriteAfterMap) {
  auto field = [](const char *k, const std::string &v) {
    std::cout << "U8I4_FIELD path=arena_uncached field=" << k << " value=" << v
              << "\n";
  };

  auto alloc =
    (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto rfree = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
  auto to_fd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  using FastrpcMunmap = int (*)(int, int, void *, size_t);
  auto fmmap = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
  auto fmunmap = (FastrpcMunmap)dlsym(RTLD_DEFAULT, "fastrpc_munmap");
  if (!alloc || !rfree || !to_fd || !fmmap) {
    GTEST_SKIP() << "rpcmem/fastrpc_mmap not available";
  }

  const uint32_t kBytes = 3670016u;
  const uint32_t kDma = 1048576u;
  // RPCMEM_FLAG_UNCACHED. The flag, not the heap, is what decides whether
  // the CPU's writes need a flush before the DSP can see them.
  void *buf = alloc(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 0 /*UNCACHED*/, (int)kBytes);
  field("alloc", buf ? "ok" : "failed");
  if (buf == nullptr) {
    GTEST_SKIP() << "uncached rpcmem_alloc failed";
  }
  const int fd = to_fd(buf);
  field("fd", std::to_string(fd));

  // Attach FIRST. Everything after this point is the steady state the
  // arena runs in.
  const int rc =
    fmmap(CDSP_DOMAIN_ID, fd, buf, 0, kBytes, static_cast<int>(FASTRPC_MAP_FD));
  field("fastrpc_mmap_rc", hex(rc));

  auto *p = static_cast<uint8_t *>(buf);
  uint32_t want = 0;
  for (uint32_t i = 0; i < kBytes; ++i) {
    p[i] = static_cast<uint8_t>(i * 31u + 7u);
  }
  for (uint32_t i = 0; i < kDma; i += 64u) {
    want += p[i];
  }

  std::vector<uint32_t> res(10, 0);
  const int err = nntr_hvx_arena_probe(handle_, fd, kBytes, kDma, res.data(),
                                       (int)res.size());
  field("err", hex(err));
  field("hap_mmap_get_rc", hex((int)res[8]));
  if (err == AEE_SUCCESS) {
    const double gbs = res[2] > 0 ? (double)res[3] / res[2] / 1000.0 : 0.0;
    field("dma_gbs", std::to_string(gbs));
    field("checksum_ok", res[4] == want ? "yes" : "no");
    EXPECT_EQ(res[4], want)
      << "uncached writes made AFTER fastrpc_mmap did not reach the DSP. "
         "doc 46 section 34 assumes they do; the fallback is a cached arena "
         "filled before attach, with misses left on the DSP heap.";
  }
  if (rc == 0 && fmunmap != nullptr) {
    field("fastrpc_munmap_rc", hex(fmunmap(CDSP_DOMAIN_ID, fd, buf, kBytes)));
  }
  rfree(buf);
}

/**
 * @brief [doc 46 section 34.5] A weight borrowed from the arena multiplies
 *        exactly like the same weight copied onto the DSP heap.
 *
 * Same bytes through the same kernel, so this is bit-for-bit or it is a
 * bug: the only difference is where wh_bytes points. Also checks the one
 * rule that keeps a borrowed slot safe -- an arena cannot be detached while
 * a weight still points into it.
 */
/**
 * [doc 52] The sequence the expert LRU makes and no test above covers: the
 * DSP reads an arena slot, the host overwrites the slot with another
 * expert's bytes, the DSP reads it again. The DMA descriptors read DDR
 * through L2 (src_bypass = 0), so if the DSP's mapping of the arena is
 * cacheable, the second read could return the first expert's stale lines.
 * Gate for slot reuse on device: bit-identical to the heap-registered
 * weight, both before and after the overwrite.
 */
TEST_F(HmxMmU8I4Layer, ArenaSlotReuseMatchesHeap) {
  const uint32_t K = 2048, I = 1792, N = 2048, M = 64, NE = 2;

  auto alloc =
    (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto rfree = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
  auto to_fd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  auto fmmap = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
  if (!alloc || !rfree || !to_fd || !fmmap) {
    GTEST_SKIP() << "rpcmem/fastrpc_mmap not available";
  }

  // Two experts on the heap: the references, and the bytes the slot gets.
  std::vector<Weight> gu(NE), dn(NE);
  std::vector<uint32_t> h_gu(NE), h_dn(NE);
  for (uint32_t e = 0; e < NE; ++e) {
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, 2 * I, 0x5107A000u + e, gu[e]));
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(I, N, 0x5107B000u + e, dn[e]));
    h_gu[e] = gu[e].handle;
    h_dn[e] = dn[e].handle;
  }

  auto wh_bytes = [](uint32_t k, uint32_t n) {
    return (k / 32u) * (n / 32u) * 512u;
  };
  const uint32_t gu_len = wh_bytes(K, 2 * I), dn_len = wh_bytes(I, N);
  const uint32_t stride_gu = (gu_len + 4095u) & ~4095u;
  // ONE expert slot, laid out as register_qs4cx_wh_expert_file lays it:
  // gate_up at 0, down at stride_gu.
  const uint32_t arena_bytes = stride_gu + ((dn_len + 4095u) & ~4095u);

  void *buf = alloc(25, 0 /*UNCACHED*/, (int)arena_bytes);
  ASSERT_NE(buf, nullptr) << "uncached rpcmem_alloc failed";
  const int fd = to_fd(buf);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(fmmap(CDSP_DOMAIN_ID, fd, buf, 0, arena_bytes,
                  static_cast<int>(FASTRPC_MAP_FD)),
            0);
  uint32_t arena = 0xFFFFFFFFu;
  ASSERT_EQ(nntr_hvx_arena_attach(handle_, fd, arena_bytes, &arena),
            AEE_SUCCESS);
  auto *base = static_cast<uint8_t *>(buf);

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0052u);
  // One expert takes every row: the call is [e] alone, so the slot's
  // contents are the only thing that can differ between runs.
  std::vector<uint32_t> row_index(M);
  std::vector<float> row_weight(M);
  for (uint32_t r = 0; r < M; ++r) {
    row_index[r] = r;
    row_weight[r] = 0.25f + 0.5f * (r % 7u) / 7.0f;
  }
  const std::vector<uint32_t> row_count = {M};

  auto run = [&](uint32_t hg, uint32_t hd, std::vector<float> &out) {
    out.assign(static_cast<size_t>(M) * N, 1.0f);
    return nntr_hvx_mm_u8i4_moe_layer(
      handle_, M, K, I, N, &hg, 1, &hd, 1, row_index.data(),
      (int)row_index.size(), row_count.data(), (int)row_count.size(),
      row_weight.data(), (int)row_weight.size(), x.data(), (int)x.size(),
      out.data(), (int)out.size());
  };
  auto count_bad = [](const std::vector<float> &a,
                      const std::vector<float> &b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
      if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0)
        ++bad;
    return bad;
  };

  // Fill the slot with expert e, register, run, release -- twice, so the
  // second pass is a reuse of a slot the DSP has already read.
  for (uint32_t e = 0; e < NE; ++e) {
    ASSERT_EQ(nntr_hvx_weight_bake_export(handle_, h_gu[e], base, (int)gu_len),
              AEE_SUCCESS);
    ASSERT_EQ(nntr_hvx_weight_bake_export(handle_, h_dn[e], base + stride_gu,
                                          (int)dn_len),
              AEE_SUCCESS);
    uint32_t a_gu = 0xFFFFFFFFu, a_dn = 0xFFFFFFFFu;
    ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                handle_, K, 2 * I, arena, 0, gu[e].d.data(), (int)(2 * I),
                gu[e].colsum.data(), (int)(2 * I), gu[e].bias.data(),
                (int)(2 * I), &a_gu),
              AEE_SUCCESS);
    ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                handle_, I, N, arena, stride_gu, dn[e].d.data(), (int)N,
                dn[e].colsum.data(), (int)N, dn[e].bias.data(), (int)N, &a_dn),
              AEE_SUCCESS);

    std::vector<float> from_heap, from_slot;
    ASSERT_EQ(run(h_gu[e], h_dn[e], from_heap), AEE_SUCCESS);
    ASSERT_EQ(run(a_gu, a_dn, from_slot), AEE_SUCCESS);
    const size_t bad = count_bad(from_heap, from_slot);
    std::cout << "U8I4_FIELD path=arena_slot_reuse pass=" << e
              << " field=bad_elems value=" << bad << " of " << from_heap.size()
              << std::endl;
    EXPECT_EQ(bad, 0u) << (e == 0 ? "first fill of the slot differs from the "
                                    "heap weight: not a reuse problem"
                                  : "REUSED slot differs from the heap "
                                    "weight: the DSP read stale bytes -- the "
                                    "arena mapping is cacheable on the DSP "
                                    "and register_arena needs an L2 "
                                    "invalidate (doc 52 section 10)");

    ASSERT_EQ(nntr_hvx_weight_release_u8i4(handle_, a_gu), AEE_SUCCESS);
    ASSERT_EQ(nntr_hvx_weight_release_u8i4(handle_, a_dn), AEE_SUCCESS);
  }

  ASSERT_EQ(nntr_hvx_arena_detach(handle_, arena), AEE_SUCCESS);
  for (uint32_t e = 0; e < NE; ++e) {
    nntr_hvx_weight_release_u8i4(handle_, h_gu[e]);
    nntr_hvx_weight_release_u8i4(handle_, h_dn[e]);
  }
  rfree(buf);
}

TEST_F(HmxMmU8I4Layer, MoeLayerFromArenaMatchesHeap) {
  const uint32_t K = 2048, I = 1792, N = 2048, M = 64, NE = 2;

  auto alloc =
    (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto rfree = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
  auto to_fd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  auto fmmap = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
  if (!alloc || !rfree || !to_fd || !fmmap) {
    GTEST_SKIP() << "rpcmem/fastrpc_mmap not available";
  }

  std::vector<Weight> gu(NE), dn(NE);
  std::vector<uint32_t> h_gu(NE), h_dn(NE);
  for (uint32_t e = 0; e < NE; ++e) {
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(K, 2 * I, 0xA2E00000u + e, gu[e]));
    ASSERT_NO_FATAL_FAILURE(MakeAndRegister(I, N, 0xC1000000u + e, dn[e]));
    h_gu[e] = gu[e].handle;
    h_dn[e] = dn[e].handle;
  }

  auto wh_bytes = [](uint32_t k, uint32_t n) {
    return (k / 32u) * (n / 32u) * 512u;
  };
  const uint32_t gu_len = wh_bytes(K, 2 * I), dn_len = wh_bytes(I, N);
  // 4 KB apart, the same spacing HtpComputeOps::place uses.
  const uint32_t stride_gu = (gu_len + 4095u) & ~4095u;
  const uint32_t stride_dn = (dn_len + 4095u) & ~4095u;
  const uint32_t arena_bytes = NE * (stride_gu + stride_dn);

  void *buf = alloc(25, 0 /*UNCACHED*/, (int)arena_bytes);
  ASSERT_NE(buf, nullptr) << "uncached rpcmem_alloc failed";
  const int fd = to_fd(buf);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(fmmap(CDSP_DOMAIN_ID, fd, buf, 0, arena_bytes,
                  static_cast<int>(FASTRPC_MAP_FD)),
            0);

  uint32_t arena = 0xFFFFFFFFu;
  ASSERT_EQ(nntr_hvx_arena_attach(handle_, fd, arena_bytes, &arena),
            AEE_SUCCESS);

  // Export each baked weight straight into the arena, then register it
  // there -- the miss path of get_or_register_qs4cx, in miniature.
  auto *base = static_cast<uint8_t *>(buf);
  std::vector<uint32_t> a_gu(NE), a_dn(NE);
  uint32_t off = 0;
  for (uint32_t e = 0; e < NE; ++e) {
    ASSERT_EQ(
      nntr_hvx_weight_bake_export(handle_, h_gu[e], base + off, (int)gu_len),
      AEE_SUCCESS);
    ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                handle_, K, 2 * I, arena, off, gu[e].d.data(), (int)(2 * I),
                gu[e].colsum.data(), (int)(2 * I), gu[e].bias.data(),
                (int)(2 * I), &a_gu[e]),
              AEE_SUCCESS);
    off += stride_gu;

    ASSERT_EQ(
      nntr_hvx_weight_bake_export(handle_, h_dn[e], base + off, (int)dn_len),
      AEE_SUCCESS);
    ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                handle_, I, N, arena, off, dn[e].d.data(), (int)N,
                dn[e].colsum.data(), (int)N, dn[e].bias.data(), (int)N,
                &a_dn[e]),
              AEE_SUCCESS);
    off += stride_dn;
  }

  std::vector<float> x(static_cast<size_t>(M) * K);
  fill_deterministic(x, 0x5EED0011u);
  const std::vector<uint32_t> row_count = {40u, 24u};
  std::vector<uint32_t> row_index;
  std::vector<float> row_weight;
  {
    uint32_t st = 0xBEEF01u;
    for (uint32_t e = 0; e < NE; ++e) {
      std::vector<bool> taken(M, false);
      for (uint32_t i = 0; i < row_count[e]; ++i) {
        uint32_t r;
        do {
          st = st * 1664525u + 1013904223u;
          r = (st >> 8) % M;
        } while (taken[r]);
        taken[r] = true;
        row_index.push_back(r);
        st = st * 1664525u + 1013904223u;
        row_weight.push_back(0.1f + 0.9f * ((st >> 8) % 1000u) / 1000.0f);
      }
    }
  }

  auto run = [&](const std::vector<uint32_t> &hg,
                 const std::vector<uint32_t> &hd, std::vector<float> &out) {
    out.assign(static_cast<size_t>(M) * N, 1.0f);
    return nntr_hvx_mm_u8i4_moe_layer(
      handle_, M, K, I, N, hg.data(), (int)hg.size(), hd.data(), (int)hd.size(),
      row_index.data(), (int)row_index.size(), row_count.data(),
      (int)row_count.size(), row_weight.data(), (int)row_weight.size(),
      x.data(), (int)x.size(), out.data(), (int)out.size());
  };

  std::vector<float> from_heap, from_arena;
  ASSERT_EQ(run(h_gu, h_dn, from_heap), AEE_SUCCESS);
  ASSERT_EQ(run(a_gu, a_dn, from_arena), AEE_SUCCESS);

  size_t bad = 0;
  for (size_t i = 0; i < from_heap.size(); ++i) {
    if (std::memcmp(&from_arena[i], &from_heap[i], sizeof(float)) != 0)
      ++bad;
  }
  std::cout << "U8I4_FIELD path=arena_moe field=bad_elems value=" << bad
            << " of " << from_heap.size() << std::endl;
  EXPECT_EQ(bad, 0u) << "a weight borrowed from the arena and the same weight "
                        "copied to DSP heap are the same bytes through the "
                        "same kernel; any difference is the borrowing";

  // Detaching under a live borrow would leave the next matmul reading
  // unmapped memory, so the DSP refuses it. This is the check that lets
  // register_arena skip copying at all. Reported as well as asserted: which
  // code came back is the difference between "the borrow scan missed the
  // slots" and "the call never got that far".
  const int busy_rc = nntr_hvx_arena_detach(handle_, arena);
  std::cout << "U8I4_FIELD path=arena_moe field=detach_while_borrowed value="
            << hex(busy_rc) << " (want " << hex(AEE_EBADSTATE + kDspOffset)
            << ")" << std::endl;
  EXPECT_EQ(busy_rc, AEE_EBADSTATE + kDspOffset)
    << "arena detached while weights still borrow from it";

  uint32_t released = 0, first_rel_err = AEE_SUCCESS;
  for (uint32_t e = 0; e < NE; ++e) {
    for (uint32_t h : {a_gu[e], a_dn[e], h_gu[e], h_dn[e]}) {
      const int rerr = nntr_hvx_weight_release_u8i4(handle_, h);
      if (rerr == AEE_SUCCESS) {
        ++released;
      } else if (first_rel_err == AEE_SUCCESS) {
        first_rel_err = rerr;
      }
    }
  }
  const int free_rc = nntr_hvx_arena_detach(handle_, arena);
  std::cout << "U8I4_FIELD path=arena_moe field=released value=" << released
            << " of " << (4u * NE) << "\n"
            << "U8I4_FIELD path=arena_moe field=first_release_err value="
            << (first_rel_err == AEE_SUCCESS ? std::string("none")
                                             : hex(first_rel_err))
            << "\n"
            << "U8I4_FIELD path=arena_moe field=detach_after_release value="
            << hex(free_rc) << std::endl;
  EXPECT_EQ(released, 4u * NE) << "first error " << hex(first_rel_err);
  EXPECT_EQ(free_rc, AEE_SUCCESS)
    << "every borrower released, so the arena should have come back";
  rfree(buf);
}

TEST_F(HmxMmU8I4Layer, MemoryCeilings) {
  const double need_gb = 4.10;

  // --- 1. the DSP's own heap, no payload -------------------------------
  {
    const uint32_t chunk_mb = 64, max_chunks = 128; // stop past 8 GB
    std::vector<uint32_t> chunks_ok(1, 0);
    // unsigned long long, matching what the IDL's uint64 generates. The
    // fixed-width alias is `unsigned long` on aarch64 -- same width, and
    // C++ still refuses the pointer conversion.
    std::vector<unsigned long long> touched(1, 0);
    const int err = nntr_hvx_mem_probe_dsp_heap(
      handle_, chunk_mb, max_chunks, chunks_ok.data(), 1, touched.data(), 1);
    const double gb =
      err == AEE_SUCCESS ? chunks_ok[0] * chunk_mb / 1024.0 : 0.0;
    std::cout << "U8I4_FIELD path=mem field=dsp_heap_gb value=" << gb << "\n"
              << "U8I4_FIELD path=mem field=dsp_heap_err value=" << hex(err)
              << std::endl;
  }

  // --- 2. rpcmem/ION: one buffer, as large as it will go ---------------
  // rpcmem_alloc's size argument is int, so 2 GB - 1 is the API's own
  // ceiling regardless of the device; rpcmem_alloc2 takes size_t where it
  // exists. Which one is available is itself part of the answer.
  using AllocFn = void *(*)(int, uint32_t, int);
  using Alloc2Fn = void *(*)(int, uint32_t, size_t);
  using FreeFn = void (*)(void *);
  auto init = (void (*)(void))dlsym(RTLD_DEFAULT, "rpcmem_init");
  auto alloc = (AllocFn)dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto alloc2 = (Alloc2Fn)dlsym(RTLD_DEFAULT, "rpcmem_alloc2");
  auto rfree = (FreeFn)dlsym(RTLD_DEFAULT, "rpcmem_free");
  if (!alloc || !rfree) {
    std::cout << "U8I4_FIELD path=mem field=rpcmem value=absent" << std::endl;
    GTEST_SKIP() << "no rpcmem in this process -- ION cannot be measured";
  }
  if (init) {
    init();
  }
  const int kHeapIdContig = 25; // RPCMEM_HEAP_ID_SYSTEM
  const uint32_t kFlags = 1;    // RPCMEM_DEFAULT_FLAGS
  std::cout << "U8I4_FIELD path=mem field=rpcmem_alloc2 value="
            << (alloc2 ? "yes" : "no") << std::endl;

  auto try_alloc = [&](size_t bytes) -> void * {
    if (alloc2) {
      return alloc2(kHeapIdContig, kFlags, bytes);
    }
    if (bytes > 0x7FFFFFFFull) {
      return nullptr; // int would overflow; not a device limit, an API one
    }
    return alloc(kHeapIdContig, kFlags, static_cast<int>(bytes));
  };
  // The DSP side takes bufLen as int, so a single call cannot describe more
  // than 2 GB - 1 no matter how the allocation went.
  auto dsp_can_touch = [&](void *p, size_t bytes) -> bool {
    if (bytes > 0x7FFFFFFFull) {
      return false;
    }
    std::vector<unsigned long long> touched(1, 0);
    return nntr_hvx_mem_probe_touch(handle_, static_cast<const uint8_t *>(p),
                                    static_cast<int>(bytes), touched.data(),
                                    1) == AEE_SUCCESS;
  };

  double single_gb = 0.0;
  for (size_t mb : {256u, 512u, 1024u, 1536u, 2047u}) {
    const size_t bytes = mb * 1024u * 1024u;
    void *p = try_alloc(bytes);
    if (!p) {
      break;
    }
    const bool ok = dsp_can_touch(p, bytes);
    rfree(p);
    if (!ok) {
      break;
    }
    single_gb = mb / 1024.0;
  }
  std::cout << "U8I4_FIELD path=mem field=ion_single_gb value=" << single_gb
            << std::endl;

  // --- 3. how much ION in total, across buffers, the DSP can reach -----
  // FastRPC keeps an ION buffer's SMMU mapping across calls (htp_rpcmem.h),
  // so buffers touched in earlier calls should still be mapped; if the
  // ceiling here is the DSP's address space rather than the host's memory,
  // this is where it shows.
  {
    // 24 x 256 MB = 6 GB, not 10: the first run of this test produced no
    // ion_total_gb line at all, which is what an OOM kill looks like from
    // the log's side. 6 GB is comfortably past the 4.10 the plan needs and
    // leaves the phone room to keep the process alive.
    const size_t chunk = 256u * 1024u * 1024u;
    const size_t max_chunks = 24;
    std::vector<void *> bufs;
    while (bufs.size() < max_chunks) {
      void *p = try_alloc(chunk);
      if (!p) {
        break;
      }
      if (!dsp_can_touch(p, chunk)) {
        rfree(p);
        break;
      }
      bufs.push_back(p);
      // Printed as it goes, so a run that dies partway still leaves its
      // last good value in the log rather than nothing.
      std::cout << "U8I4_FIELD path=mem field=ion_total_gb value="
                << bufs.size() * chunk / 1073741824.0 << std::endl;
    }
    const double total_gb = bufs.size() * chunk / 1073741824.0;
    for (void *p : bufs) {
      rfree(p);
    }
    std::cout << "U8I4_FIELD path=mem field=ion_total_final_gb value="
              << total_gb << "\n"
              << "U8I4_FIELD path=mem field=need_gb value=" << need_gb
              << std::endl;
  }
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
/**
 * @brief Where the DSP's address space ends, and whether its heap is a
 *        separate budget from its mappings.
 *
 * The model needs 3.61 GiB of weights DSP-resident. A device run mapped
 * exactly 3.00 GiB with fastrpc_mmap and was then refused at every size down
 * to 64 MiB (doc 46 section 40), so the mapping budget is spent. The DSP heap
 * was untouched in that run -- every weight went to the arena -- and Gate 0
 * measured that heap reaching 1.89 GB, which is three times the 0.61 GiB
 * still needed.
 *
 * So: are the two the same exhausted address space, or two regions of it?
 * The answer decides between a small change (overflow the last weights onto
 * the heap) and a large one (a second PD, or a smaller model), and nothing
 * on the device reports it. This fills the mapping budget and then asks the
 * heap for weights until it refuses too.
 *
 * Not gated -- there is no right answer to assert, only a number to read.
 * Allocates and frees every buffer it takes; at 256 MiB a step it holds the
 * whole mapping budget at once, which is the point.
 */
TEST_F(HmxMmU8I4Layer, ArenaCeilingThenHeapHeadroom) {
  auto field = [](const char *k, const std::string &v) {
    std::cout << "U8I4_FIELD path=ceiling field=" << k << " value=" << v
              << "\n";
  };

  auto alloc =
    (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
  auto rfree = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
  auto to_fd = (int (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  using FastrpcMunmap = int (*)(int, int, void *, size_t);
  auto fmmap = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
  auto fmunmap = (FastrpcMunmap)dlsym(RTLD_DEFAULT, "fastrpc_munmap");
  if (!alloc || !rfree || !to_fd || !fmmap) {
    GTEST_SKIP() << "rpcmem/fastrpc_mmap not available";
  }

  // 256 MiB a step: fine enough to land within a quarter GiB of the real
  // ceiling, coarse enough that filling ~3 GiB is a dozen calls.
  const size_t kStep = size_t(256) << 20;
  struct Mapped {
    void *buf;
    int fd;
    size_t bytes;
  };
  std::vector<Mapped> mapped;
  size_t total = 0;
  int last_rc = 0;
  for (int i = 0; i < 32; ++i) {
    void *buf = alloc(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 0 /*UNCACHED*/,
                      static_cast<int>(kStep));
    if (buf == nullptr) {
      field("stopped_by", "rpcmem_alloc"); // host ION, not the DSP
      break;
    }
    const int fd = to_fd(buf);
    last_rc = fmmap(CDSP_DOMAIN_ID, fd, buf, 0, kStep,
                    static_cast<int>(FASTRPC_MAP_FD));
    if (last_rc != 0) {
      rfree(buf);
      field("stopped_by", "fastrpc_mmap");
      field("last_mmap_rc", hex(last_rc));
      break;
    }
    mapped.push_back(Mapped{buf, fd, kStep});
    total += kStep;
  }
  field("mmap_ceiling_mib", std::to_string(total >> 20));

  // Now the heap, with the mapping budget held. weight_register_u8i4 bakes,
  // so each call leaves (K/32)*(N/32)*512 bytes on the DSP heap -- the
  // model's largest weight, so the count converts straight to what the
  // overflow would need (238 weights short, doc 46 section 40).
  const uint32_t K = 2048, N = 3584;
  const size_t wh_bytes = nntrainer::whBytes(K, N);
  std::vector<int8_t> q_w(static_cast<size_t>(K) * N);
  for (size_t i = 0; i < q_w.size(); ++i)
    q_w[i] = static_cast<int8_t>((i % 15u) - 7);
  std::vector<float> d(N, 0.01f), bias(N, 0.0f);
  std::vector<int32_t> colsum(N, 0);

  std::vector<uint32_t> heap_handles;
  int heap_rc = AEE_SUCCESS;
  for (int i = 0; i < 300; ++i) {
    uint32_t h = 0xFFFFFFFFu;
    heap_rc = nntr_hvx_weight_register_u8i4(
      handle_, K, N, q_w.data(), static_cast<int>(q_w.size()), d.data(),
      static_cast<int>(d.size()), colsum.data(),
      static_cast<int>(colsum.size()), bias.data(),
      static_cast<int>(bias.size()), &h);
    if (heap_rc != AEE_SUCCESS)
      break;
    heap_handles.push_back(h);
  }
  field("heap_weights", std::to_string(heap_handles.size()));
  field("heap_mib", std::to_string((heap_handles.size() * wh_bytes) >> 20));
  field("heap_stop_rc", hex(heap_rc));
  // 238 is what the model is short by; anything at or above it means the
  // overflow fits and the fix is small.
  field("covers_overflow", heap_handles.size() >= 238 ? "yes" : "no");

  for (uint32_t h : heap_handles)
    nntr_hvx_weight_release_u8i4(handle_, h);
  for (const Mapped &m : mapped) {
    if (fmunmap != nullptr)
      fmunmap(CDSP_DOMAIN_ID, m.fd, m.buf, m.bytes);
    rfree(m.buf);
  }
}

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
