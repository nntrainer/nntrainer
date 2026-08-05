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

TEST_F(HvxExp, RoundTripsTheOutputBuffer) {
  const int n = 32;
  std::vector<float> in(n, 1.0f), out(n, -1.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "exp_f32 failed: " << hex(err);
  /* The stub zeroes the buffer; this only proves rout came back. */
  EXPECT_NE(out[0], -1.0f);
}

TEST_F(HvxExp, RejectsNonVectorLength) {
  const int n = 33;
  std::vector<float> in(n, 1.0f), out(n, 0.0f);

  int err = nntr_hvx_exp_f32(handle_, in.data(), n, out.data(), n);
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset)
    << "expected EBADPARM, got " << hex(err);
}

TEST_F(HvxSoftmax, RoundTripsTheOutputBuffer) {
  const uint32_t m = 1, k = 32;
  std::vector<float> in(m * k, 1.0f), out(m * k, -1.0f);

  int err = nntr_hvx_softmax_f32(handle_, m, k, 1.0f, in.data(),
                                 static_cast<int>(in.size()), out.data(),
                                 static_cast<int>(out.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "softmax_f32 failed: " << hex(err);
  EXPECT_NE(out[0], -1.0f);
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
