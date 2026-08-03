// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_kernels.cpp
 * @date   03 Aug 2026
 * @brief  Device test: an HVX kernel on the CDSP returns what the CPU does
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH. See test/htp/build.sh.
 */

#include <gtest/gtest.h>

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

/**
 * @brief Opens one unsigned-PD CDSP session for the whole test case.
 *
 * A failure here is a hard FAIL rather than a skip: proving the DSP comes
 * up on the device is the point of this test, so a quiet skip would
 * report success for the thing being measured.
 */
class HvxAdd : public ::testing::Test {
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

TEST_F(HvxAdd, MatchesCpuForFullVectorsPlusRemainder) {
  // 100 floats = 3 full 128-byte vectors (32 floats each) + 4 left over.
  const int n = 100;
  std::vector<float> a(n), b(n), c(n, 0.0f), expected(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i) * 0.5f;
    b[i] = static_cast<float>(n - i) * 0.25f;
    expected[i] = a[i] + b[i];
  }

  int err = nntr_hvx_add_f32(handle_, a.data(), n, b.data(), n, c.data(), n);
  ASSERT_EQ(err, AEE_SUCCESS) << "add_f32 failed: " << hex(err);
  EXPECT_EQ(c, expected);
}

TEST_F(HvxAdd, MatchesCpuBelowOneVector) {
  // Fewer floats than a single vector holds: n_vec is 0 and only the
  // scalar tail runs.
  const std::vector<float> a = {1.5f};
  const std::vector<float> b = {2.25f};
  std::vector<float> c = {0.0f};

  int err = nntr_hvx_add_f32(handle_, a.data(), 1, b.data(), 1, c.data(), 1);
  ASSERT_EQ(err, AEE_SUCCESS) << "add_f32 failed: " << hex(err);
  EXPECT_EQ(c[0], 3.75f);
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
