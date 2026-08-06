// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_softmax.cpp
 * @date   05 Aug 2026
 * @brief  Device test: HVX exp and softmax match a double reference
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
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

namespace {

/**
 * @brief Offset AEEStdErr.h adds to every AEE_* code on the DSP side.
 *
 * DSP-side skel code is compiled with __hexagon__ defined, where
 * AEEStdErr.h adds this offset to every AEE_* code before it crosses the
 * FastRPC boundary. This host binary is not compiled with __hexagon__, so
 * AEE_EBADPARM here is plain 14 -- the offset has to be added back when
 * checking an error the DSP returned.
 */
constexpr int kDspOffset = 0x80000400;

/** @brief Render a FastRPC error as hex so the code is searchable. */
std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/**
 * @brief Softmax reference in double, row by row.
 *
 * double rather than the nntrainer CPU softmax: comparing two f32
 * approximations would blur where the error comes from.
 */
std::vector<float> ref_softmax(const std::vector<float> &x, uint32_t m,
                               uint32_t k, float scale) {
  std::vector<float> y(x.size());
  std::vector<double> e(k);

  for (uint32_t r = 0; r < m; ++r) {
    const float *xr = x.data() + static_cast<size_t>(r) * k;
    double mx = -std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < k; ++i) {
      mx = std::max(mx, static_cast<double>(xr[i]) * scale);
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
      e[i] = std::exp(static_cast<double>(xr[i]) * scale - mx);
      sum += e[i];
    }
    for (uint32_t i = 0; i < k; ++i) {
      y[static_cast<size_t>(r) * k + i] = static_cast<float>(e[i] / sum);
    }
  }
  return y;
}

/**
 * @brief Opens an unsigned-PD CDSP session for each test.
 *
 * A failure here is a hard FAIL rather than a skip: proving the DSP comes
 * up on the device is part of what this test measures, so a quiet skip
 * would report success for it.
 */
class HtpSession : public ::testing::Test {
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

class HvxExp : public HtpSession {};
class HvxSoftmax : public HtpSession {};

TEST_F(HvxExp, RejectsNonVectorLength) {
  const int n = 33;
  std::vector<float> in(n, 1.0f), out(n, 0.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset)
    << "expected EBADPARM, got " << hex(err);
}

TEST_F(HvxExp, MatchesDoubleOverTheNormalRange) {
  // 8192 is a multiple of 32. Sweep stops at -87: below that the true
  // value is subnormal and the kernel flushes it to zero by contract.
  const int n = 8192;
  std::vector<float> in(n), out(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    in[i] = -87.0f + 175.0f * static_cast<float>(i) / static_cast<float>(n - 1);
  }

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);

  double worst = 0.0;
  int worst_i = 0;
  for (int i = 0; i < n; ++i) {
    const double ref = std::exp(static_cast<double>(in[i]));
    const double rel = std::abs(static_cast<double>(out[i]) - ref) / ref;
    if (rel > worst) {
      worst = rel;
      worst_i = i;
    }
  }
  EXPECT_LT(worst, 1e-6) << "worst relative error at x=" << in[worst_i]
                         << ": got " << out[worst_i] << ", want "
                         << std::exp(static_cast<double>(in[worst_i]));
}

TEST_F(HvxExp, ExactlyOneAtZero) {
  const int n = 32;
  const std::vector<float> in(n, 0.0f);
  std::vector<float> out(n, -1.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], 1.0f) << "lane " << i;
  }
}

TEST_F(HvxExp, FlushesFarNegativeToZero) {
  // softmax feeds x - max, which can be arbitrarily negative. The low
  // clamp inside hvx_exp_sf is what keeps the range reduction from
  // overflowing its own valid domain on these.
  const int n = 32;
  std::vector<float> in(n, -90.0f);
  in[1] = -200.0f;
  in[2] = -1e30f;
  in[3] = -3.4e38f;
  std::vector<float> out(n, 1.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], 0.0f) << "lane " << i << " x=" << in[i];
  }
}

TEST_F(HvxSoftmax, RejectsLengthMismatch) {
  const uint32_t m = 2, k = 32;
  std::vector<float> in(m * k, 1.0f), out(k, 0.0f);

  int err = nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(),
                                 static_cast<int>(in.size()), out.data(),
                                 static_cast<int>(out.size()));
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset)
    << "expected EBADPARM, got " << hex(err);
}

TEST_F(HvxSoftmax, MatchesDoubleForOneFullRow) {
  const uint32_t m = 1, k = 1024;
  std::vector<float> in(m * k), out(m * k, 0.0f);

  std::mt19937 rng(20260805u);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  for (auto &v : in) {
    v = dist(rng);
  }

  int err = nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(),
                                 static_cast<int>(in.size()), out.data(),
                                 static_cast<int>(out.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "softmax_f32 failed: " << hex(err);

  const std::vector<float> ref = ref_softmax(in, m, k, 1.0f);
  double worst = 0.0;
  double sum = 0.0;
  for (uint32_t i = 0; i < k; ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(out[i]) - ref[i]));
    sum += out[i];
  }
  EXPECT_LT(worst, 1e-6) << "worst absolute error";
  EXPECT_NEAR(sum, 1.0, 1e-6) << "row does not sum to 1";
}

TEST_F(HvxSoftmax, IsInvariantToAConstantShift) {
  // Adding a constant to every element must not change the result. This is
  // what the max subtraction buys. exp(100) overflows f32, so 1e2 still
  // forces max subtraction; going larger (e.g. 1e4) rounds f32 inputs to
  // ULP ~0.001 on the host before the DSP sees them, making 1e-6
  // unachievable regardless of kernel precision.
  const uint32_t m = 1, k = 256;
  std::vector<float> base(m * k), shifted(m * k);
  std::vector<float> out_base(m * k, 0.0f), out_shift(m * k, 0.0f);

  std::mt19937 rng(7u);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (uint32_t i = 0; i < k; ++i) {
    base[i] = dist(rng);
    shifted[i] = base[i] + 1e2f;
  }

  const int n = static_cast<int>(m * k);
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 1.0f, base.data(), n,
                                 out_base.data(), n),
            AEE_SUCCESS);
  ASSERT_EQ(nntr_hvx_softmax_f32(handle_, m, k, 1.0f, shifted.data(), n,
                                 out_shift.data(), n),
            AEE_SUCCESS);

  for (uint32_t i = 0; i < k; ++i) {
    EXPECT_NEAR(out_base[i], out_shift[i], 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, SpreadsUniformlyForEqualInputs) {
  const uint32_t m = 1, k = 64;
  const std::vector<float> in(m * k, 5.0f);
  std::vector<float> out(m * k, 0.0f);

  const int n = static_cast<int>(m * k);
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(), n, out.data(), n),
    AEE_SUCCESS);

  for (uint32_t i = 0; i < k; ++i) {
    EXPECT_NEAR(out[i], 1.0f / 64.0f, 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, CollapsesOntoADominantElement) {
  // Every other term underflows exp. Without the guard in hvx_exp_sf these
  // come back as garbage rather than zero.
  const uint32_t m = 1, k = 32;
  std::vector<float> in(m * k, 0.0f);
  in[7] = 100.0f;
  std::vector<float> out(m * k, -1.0f);

  const int n = static_cast<int>(m * k);
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(), n, out.data(), n),
    AEE_SUCCESS);

  for (uint32_t i = 0; i < k; ++i) {
    EXPECT_NEAR(out[i], i == 7 ? 1.0f : 0.0f, 1e-6) << "lane " << i;
  }
}

TEST_F(HvxSoftmax, HandlesRowLengthsThatAreNotWholeVectors) {
  // 1 is below one vector; 31 is one short; 33 is one over; 100 is three
  // vectors plus four.
  for (const uint32_t k : {1u, 31u, 33u, 100u}) {
    const uint32_t m = 1;
    std::vector<float> in(k), out(k, -1.0f);

    std::mt19937 rng(k);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    for (auto &v : in) {
      v = dist(rng);
    }

    const int n = static_cast<int>(k);
    ASSERT_EQ(
      nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(), n, out.data(), n),
      AEE_SUCCESS)
      << "k=" << k;

    const std::vector<float> ref = ref_softmax(in, m, k, 1.0f);
    double sum = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
      EXPECT_NEAR(out[i], ref[i], 1e-6) << "k=" << k << " lane " << i;
      sum += out[i];
    }
    EXPECT_NEAR(sum, 1.0, 1e-6) << "k=" << k << " does not sum to 1";
  }
}

TEST_F(HvxSoftmax, DoesNotWritePastTheEndOfTheBuffer) {
  // k=33 means the tail vector covers 32 lanes but only 1 is real. A
  // full-vector store would clobber 31 floats of whatever follows.
  const uint32_t m = 1, k = 33;
  const uint32_t guard = 64;
  std::vector<float> buf(k + guard, 12345.0f);
  std::vector<float> in(k, 1.0f);

  const int n = static_cast<int>(k);
  ASSERT_EQ(
    nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(), n, buf.data(), n),
    AEE_SUCCESS);

  for (uint32_t i = 0; i < guard; ++i) {
    EXPECT_EQ(buf[k + i], 12345.0f) << "clobbered guard word " << i;
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
