// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics
 *
 * @file        unittest_rms_norm_fp16.cpp
 * @date        March 24, 2026
 * @brief       Unit test for fallback rms_norm_wrt_width_fp16_intrinsic (_FP16)
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Samsung Electronics
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>
#if defined(__ARM_NEON) && defined(ENABLE_FP16)
#include <arm_compute_backend.h>
#include <fallback_internal.h>
#endif

static void rms_norm_fp32_scalar(const float *X, float *Y, size_t H, size_t W,
                                 float epsilon) {
  for (size_t h = 0; h < H; ++h) {
    const float *rowX = X + h * W;
    float *rowY = Y + h * W;
    float sum = 0.0f;
    for (size_t i = 0; i < W; ++i)
      sum += rowX[i] * rowX[i];
    float scale = 1.0f / std::sqrt(sum / W + epsilon);
    for (size_t i = 0; i < W; ++i)
      rowY[i] = rowX[i] * scale;
  }
}

#if defined(__ARM_NEON) && defined(ENABLE_FP16)

TEST(rms_norm_fallback_fp16, small_dimensions_4_8) {
  size_t H = 4, W = 8;
  float epsilon = 1e-6f;
  std::vector<float> Xf(H * W);
  for (size_t i = 0; i < H * W; ++i)
    Xf[i] = (float)(rand() % 100) / 10.0f;
  std::vector<_FP16> X(H * W), Y(H * W);
  for (size_t i = 0; i < H * W; ++i)
    X[i] = (_FP16)Xf[i];
  nntrainer::rms_norm_wrt_width_fp16_intrinsic<_FP16>(X.data(), Y.data(), H,
                                                      W, epsilon);
  std::vector<float> Yref(H * W);
  rms_norm_fp32_scalar(Xf.data(), Yref.data(), H, W, epsilon);
  for (size_t i = 0; i < H * W; ++i)
    EXPECT_NEAR((float)Y[i], Yref[i], 1e-2f) << "index " << i;
}

TEST(rms_norm_fallback_fp16, non_divisible_dimensions_5_13) {
  size_t H = 5, W = 13;
  float epsilon = 1e-6f;
  std::vector<float> Xf(H * W);
  for (size_t i = 0; i < H * W; ++i)
    Xf[i] = (float)(rand() % 100) / 10.0f;
  std::vector<_FP16> X(H * W), Y(H * W);
  for (size_t i = 0; i < H * W; ++i)
    X[i] = (_FP16)Xf[i];
  nntrainer::rms_norm_wrt_width_fp16_intrinsic<_FP16>(X.data(), Y.data(), H,
                                                      W, epsilon);
  std::vector<float> Yref(H * W);
  rms_norm_fp32_scalar(Xf.data(), Yref.data(), H, W, epsilon);
  for (size_t i = 0; i < H * W; ++i)
    EXPECT_NEAR((float)Y[i], Yref[i], 1e-2f) << "index " << i;
}

TEST(rms_norm_fallback_fp16, negative_values) {
  size_t H = 4, W = 8;
  float epsilon = 1e-6f;
  std::vector<float> Xf(H * W);
  for (size_t i = 0; i < H * W; ++i)
    Xf[i] = (float)((rand() % 200) - 100) / 10.0f;
  std::vector<_FP16> X(H * W), Y(H * W);
  for (size_t i = 0; i < H * W; ++i)
    X[i] = (_FP16)Xf[i];
  nntrainer::rms_norm_wrt_width_fp16_intrinsic<_FP16>(X.data(), Y.data(), H,
                                                      W, epsilon);
  std::vector<float> Yref(H * W);
  rms_norm_fp32_scalar(Xf.data(), Yref.data(), H, W, epsilon);
  for (size_t i = 0; i < H * W; ++i)
    EXPECT_NEAR((float)Y[i], Yref[i], 1e-2f) << "index " << i;
}

TEST(rms_norm_fallback_fp16, embedding_dimension_768) {
  size_t H = 10, W = 768;
  float epsilon = 1e-6f;
  std::vector<float> Xf(H * W);
  for (size_t i = 0; i < H * W; ++i)
    Xf[i] = (float)(rand() % 100) / 10.0f;
  std::vector<_FP16> X(H * W), Y(H * W);
  for (size_t i = 0; i < H * W; ++i)
    X[i] = (_FP16)Xf[i];
  nntrainer::rms_norm_wrt_width_fp16_intrinsic<_FP16>(X.data(), Y.data(), H,
                                                      W, epsilon);
  std::vector<float> Yref(H * W);
  rms_norm_fp32_scalar(Xf.data(), Yref.data(), H, W, epsilon);
  float max_err = 0.0f;
  for (size_t i = 0; i < H * W; ++i) {
    float err = std::fabs((float)Y[i] - Yref[i]);
    if (err > max_err)
      max_err = err;
  }
  EXPECT_LE(max_err, 1e-2f);
}

#endif // __ARM_NEON && ENABLE_FP16

GTEST_API_ int main(int argc, char **argv) {
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
