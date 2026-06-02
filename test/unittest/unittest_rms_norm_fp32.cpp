// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics
 *
 * @file        unittest_rms_norm_fp32.cpp
 * @date        March 24, 2026
 * @brief       Unit test for rms_norm_wrt_width_fp32_intrinsic function
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Samsung Electronics
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#if defined(__ARM_NEON)
#include <neon_impl.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

// Reference implementation of RMS normalization in FP32
void rms_norm_fp32_reference(const float *__restrict X, float *__restrict Y,
                             size_t H, size_t W, float epsilon) {
  for (size_t h = 0; h < H; ++h) {
    const float *rowX = X + h * W;
    float *rowY = Y + h * W;

    // Calculate mean of squares
    float sum_squares = 0.0f;
    for (size_t i = 0; i < W; ++i) {
      sum_squares += rowX[i] * rowX[i];
    }
    float mean = sum_squares / W;

    // Calculate scale
    float scale = 1.0f / std::sqrt(mean + epsilon);

    // Apply normalization
    for (size_t i = 0; i < W; ++i) {
      rowY[i] = rowX[i] * scale;
    }
  }
}

// Test with small dimensions
TEST(rms_norm_fp32_intrinsic, small_dimensions_4_8) {
  size_t H = 4;
  size_t W = 8;
  float epsilon = 1e-6f;

  // Create test data
  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  // Initialize with random values
  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f; // Values between 0.0 and 9.9
  }

  // Run FP32 intrinsic version
  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);

  // Run reference FP32 version
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  // Compare results
  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with dimensions not divisible by 8
TEST(rms_norm_fp32_intrinsic, non_divisible_dimensions_5_13) {
  size_t H = 5;
  size_t W = 13;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with dimensions that require remainder handling
TEST(rms_norm_fp32_intrinsic, remainder_handling_3_11) {
  size_t H = 3;
  size_t W = 11;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with typical embedding dimension (768)
TEST(rms_norm_fp32_intrinsic, embedding_dimension_768) {
  size_t H = 10;
  size_t W = 768;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with different epsilon values
TEST(rms_norm_fp32_intrinsic, different_epsilon_values) {
  size_t H = 4;
  size_t W = 16;
  std::vector<float> epsilon_values = {1e-8f, 1e-6f, 1e-4f, 1e-2f};

  std::vector<float> X(H * W);
  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  for (float epsilon : epsilon_values) {
    std::vector<float> Y_intrinsic(H * W);
    std::vector<float> Y_reference(H * W);

    nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
      X.data(), Y_intrinsic.data(), H, W, epsilon);
    rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

    float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
    double cos_sim =
      cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

    const float epsilon_tolerance = 1e-6f;
    EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
    EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
  }
}

// Test with single row
TEST(rms_norm_fp32_intrinsic, single_row) {
  size_t H = 1;
  size_t W = 32;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with single column
TEST(rms_norm_fp32_intrinsic, single_column) {
  size_t H = 10;
  size_t W = 1;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with negative values
TEST(rms_norm_fp32_intrinsic, negative_values) {
  size_t H = 4;
  size_t W = 8;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  // Initialize with both positive and negative values
  for (size_t i = 0; i < H * W; ++i) {
    X[i] =
      (float)((rand() % 200) - 100) / 10.0f; // Values between -10.0 and 9.9
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with zero values
TEST(rms_norm_fp32_intrinsic, zero_values) {
  size_t H = 4;
  size_t W = 8;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W, 0.0f);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
}

// Test with very large dimensions
TEST(rms_norm_fp32_intrinsic, large_dimensions_100_1024) {
  size_t H = 100;
  size_t W = 1024;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with width exactly divisible by 8
TEST(rms_norm_fp32_intrinsic, width_divisible_by_8) {
  size_t H = 4;
  size_t W = 16;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test with width leaving 4 elements remainder
TEST(rms_norm_fp32_intrinsic, width_remainder_4) {
  size_t H = 4;
  size_t W = 12;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y_intrinsic(H * W);
  std::vector<float> Y_reference(H * W);

  for (size_t i = 0; i < H * W; ++i) {
    X[i] = (float)(rand() % 100) / 10.0f;
  }

  nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic(
    X.data(), Y_intrinsic.data(), H, W, epsilon);
  rms_norm_fp32_reference(X.data(), Y_reference.data(), H, W, epsilon);

  float mse_error = mse<float>(Y_intrinsic.data(), Y_reference.data(), H * W);
  double cos_sim =
    cosine_similarity<float>(Y_intrinsic.data(), Y_reference.data(), H * W);

  const float epsilon_tolerance = 1e-6f;
  EXPECT_IN_RANGE(mse_error, 0.0f, epsilon_tolerance);
  EXPECT_IN_RANGE((float)cos_sim, 0.999999f, 1.0f);
}

// Test to verify the function is being called - using function pointer
TEST(rms_norm_fp32_intrinsic, verify_function_called) {
  // Get function pointer
  auto func_ptr = &nntrainer::neon::rms_norm_wrt_width_fp32_intrinsic;

  size_t H = 2;
  size_t W = 8;
  float epsilon = 1e-6f;

  std::vector<float> X(H * W);
  std::vector<float> Y(H * W);
  std::vector<float> Y_ref(H * W);

  // Fill with known values
  for (size_t i = 0; i < H * W; ++i) {
    X[i] = static_cast<float>(i + 1);
  }

  // Call via function pointer
  func_ptr(X.data(), Y.data(), H, W, epsilon);

  // Call reference implementation
  rms_norm_fp32_reference(X.data(), Y_ref.data(), H, W, epsilon);

  // Verify results match
  float mse_error = mse<float>(Y.data(), Y_ref.data(), H * W);
  EXPECT_LE(mse_error, 1e-6f)
    << "Function should be called and produce correct results";
}

#endif // __ARM_NEON

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
