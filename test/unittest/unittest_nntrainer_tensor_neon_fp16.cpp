// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file        unittest_nntrainer_tensor_neon_fp16.cpp
 * @date        03 August 2023
 * @brief       Unit test utility for tensor with NEON __fp16 support for ARM.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Debadri Samaddar <s.debadri@samsung.com>
 * @author      Sungsik Kong <ss.kong@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <cpu_backend.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

TEST(nntrainer_Tensor, add_i) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // NEON fp16
  int result = input.add_i(input);

  // fp32
  result = input_fp32.add_i(input_fp32);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  // conditions for fp16 sdot call:
  // this->(batch * channel * height) = arg->(width) = 1;

  size_t width = 23;

  __fp16 a_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                     12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);
  __fp16 b_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                     12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp16), b_data);

  float a_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                         12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);
  float b_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                         12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_fp32_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp32), b_data_fp32);

  nntrainer::Tensor result_neon;
  nntrainer::Tensor result_fp32;

  // NEON fp16
  result_neon = input.dot(input_2, false, false);

  // fp32
  result_fp32 = input_fp32.dot(input_fp32_2, false, false);

  float mseErrorNeon =
    mse<__fp16>(result_neon.getData<__fp16>(), result_fp32.getData<float>(),
                result_neon.size());

  double cosSimNeon =
    cosine_similarity<__fp16>(result_neon.getData<__fp16>(),
                              result_fp32.getData<float>(), result_neon.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, hdot_768) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  // conditions for fp16 hdot call:
  // this->(batch * channel * height) = arg->(width) = 1;
  size_t batch = 1;
  size_t channel = 1;
  size_t height = 1;
  size_t width = 768;

  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16));

  nntrainer::Tensor input_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp16));

  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32));

  nntrainer::Tensor input_fp32_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp32));

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((i * j * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT(input_fp32, ((i * j * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT(input_2, ((i * k * (batch * height * channel) +
                            j * (batch * height) + k * (width) + l + 1) %
                           MOD) *
                            alpha);
  GEN_TEST_INPUT(input_fp32_2, ((i * k * (batch * height * channel) +
                                 j * (batch * height) + k * (width) + l + 1) %
                                MOD) *
                                 alpha);

  nntrainer::Tensor result_neon = input.dot(input_2, false, false);
  nntrainer::Tensor result_fp32 = input_fp32.dot(input_fp32_2, false, false);

  float mseErrorNeon =
    mse<__fp16>(result_neon.getData<__fp16>(), result_fp32.getData<float>(),
                result_neon.size());

  double cosSimNeon =
    cosine_similarity<__fp16>(result_neon.getData<__fp16>(),
                              result_fp32.getData<float>(), result_neon.size());

  const float epsilon = 1e-3;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, l2norm) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t width = 23;

  __fp16 a_data[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
                     2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

  float a_data_fp32[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
                         2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

  __fp16 result_neon;
  float result_fp32;

  // NEON fp16
  result_neon = input.l2norm();

  // fp32
  result_fp32 = input_fp32.l2norm();

  // absolute error
  const float epsilon = 1e-3 * width;

  EXPECT_NEAR(result_neon, result_fp32, epsilon);
}

TEST(nntrainer_Tensor, l2norm_big768) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t batch = 1;
  size_t channel = 1;
  size_t height = 768;
  size_t width = 768;

  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, height, width, t_type_nchw_fp16));

  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, height, width, t_type_nchw_fp32));

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((i * j * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT(input_fp32, ((i * j * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);

  __fp16 result_neon;
  float result_fp32;

  result_neon = input.l2norm();
  result_fp32 = input_fp32.l2norm();

  float ErrorNeon = abs(result_neon - result_fp32);

  const float epsilon = 1e-3 * width;
  EXPECT_IN_RANGE(ErrorNeon, 0, epsilon);
}

TEST(nntrainer_Tensor, multiply_i) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // NEON fp16
  int result = input.multiply_i(0.1);

  // fp32
  result = input_fp32.multiply_i(0.1);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, copy) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor output;
  nntrainer::Tensor output_fp32;

  // NEON fp16
  output.copy(input);

  // fp32
  output_fp32.copy(input_fp32);

  float mseErrorNeon = mse<__fp16>(output.getData<__fp16>(),
                                   output_fp32.getData<float>(), output.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    output.getData<__fp16>(), output_fp32.getData<float>(), output.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, max_abs) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t width = 25;

  __fp16 a_data[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
                     1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
                     2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

  float a_data_fp32[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
                         1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
                         2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

  __fp16 result_neon;
  float result_fp32;

  // NEON fp16
  result_neon = input.max_abs();

  // fp32
  result_fp32 = input_fp32.max_abs();

  // absolute error
  const float epsilon = 1e-2;

  EXPECT_NEAR(result_neon, result_fp32, epsilon);
}

TEST(nntrainer_Tensor, max_abs_768_768) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t batch = 1;
  size_t channel = 1;
  size_t height = 768;
  size_t width = 768;

  nntrainer::Tensor input(
    nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp16));

  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp32));

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((k * l * (batch * height * channel) +
                          l * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT(input_fp32, ((k * l * (batch * height * channel) +
                               l * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);

  __fp16 result_neon = input.max_abs();
  float result_fp32 = input_fp32.max_abs();

  float absErrorNeon = std::abs(result_neon - result_fp32);

  const float epsilon = 1e-3;

  EXPECT_IN_RANGE(absErrorNeon, 0, epsilon);
}

TEST(nntrainer_Tensor, max_abs_65535) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t batch = 1;
  size_t channel = 1;
  size_t height = 1;
  size_t width = UINT16_MAX;

  nntrainer::Tensor input(
    nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp16));

  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp32));

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((k * l * (batch * height * channel) +
                          l * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT(input_fp32, ((k * l * (batch * height * channel) +
                               l * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);

  __fp16 result_neon = input.max_abs();
  float result_fp32 = input_fp32.max_abs();

  float absErrorNeon = std::abs(result_neon - result_fp32);

  const float epsilon = 1e-3;

  EXPECT_IN_RANGE(absErrorNeon, 0, epsilon);
}

TEST(nntrainer_Tensor, max_abs_65534) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t batch = 1;
  size_t channel = 1;
  size_t height = 1;
  size_t width = UINT16_MAX - 1;

  nntrainer::Tensor input(
    nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp16));

  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(batch, channel, height, width, t_type_nchw_fp32));

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((k * l * (batch * height * channel) +
                          l * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT(input_fp32, ((k * l * (batch * height * channel) +
                               l * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);

  __fp16 result_neon = input.max_abs();
  float result_fp32 = input_fp32.max_abs();

  float absErrorNeon = std::abs(result_neon - result_fp32);

  const float epsilon = 1e-3;

  EXPECT_IN_RANGE(absErrorNeon, 0, epsilon);
}

TEST(nntrainer_Tensor, sum_gemv_transpose_2_10) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result0_fp32 = input_fp32.sum(0);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, sum_gemv_2_10) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);

  nntrainer::Tensor result0 = input.sum_by_batch();
  nntrainer::Tensor result0_fp32 = input_fp32.sum_by_batch();

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_16_16_16) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 16;
  int width = 16;

  int height_b = 16;
  int width_b = 16;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_1024_1024_1024) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1024;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_768) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 768;

  int height_b = 768;
  int width_b = 768;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_padding_M_transB) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 70;
  int width = 768;

  int height_b = 960;
  int width_b = 768;

  bool transA = false;
  bool transB = true;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_padding_K) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 88;
  int width = 769;

  int height_b = 769;
  int width_b = 768;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_padding_N) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 160;
  int width = 768;

  int height_b = 768;
  int width_b = 799;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_padding_MK) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 768;
  int width_b = 123;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_padding_KN) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 321;

  int height_b = 321;
  int width_b = 123;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_padding_MKN) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 77;

  int height_b = 77;
  int width_b = 991;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_48000) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 768;
  int width_b = 48000;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_20000) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 768;
  int width_b = 20000;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);

  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_512_520_1032) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 512;
  int width = 520;

  int height_b = 520;
  int width_b = 1032;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);

  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_1001_1024_20000) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1001;
  int width = 1024;

  int height_b = 1024;
  int width_b = 20000;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);

  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_50_768_516) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 50;
  int width = 768;

  int height_b = 768;
  int width_b = 516;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);

  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_K1) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 56;
  int width = 1;

  int height_b = 1;
  int width_b = 516;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);

  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

/**
 * @brief Run fp16 sgemm on exactly representable inputs and compare with a
 * scalar reference. Every partial sum stays a small multiple of 1/64, so the
 * result is exact in fp16 regardless of the accumulation order.
 */
static void check_small_hgemm(unsigned int M, unsigned int N, unsigned int K,
                              bool transA, bool transB, float beta) {
  std::vector<__fp16> A(M * K), B(K * N), C(M * N);
  std::vector<float> ref(M * N);
  for (unsigned int i = 0; i < A.size(); ++i)
    A[i] = static_cast<__fp16>(static_cast<int>(i * 7 % 13) - 6) / 8;
  for (unsigned int i = 0; i < B.size(); ++i)
    B[i] = static_cast<__fp16>(static_cast<int>((i * 5 + 3) % 11) - 5) / 8;
  for (unsigned int i = 0; i < C.size(); ++i)
    C[i] = static_cast<__fp16>(static_cast<int>(i % 5) - 2) / 4;

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      float sum = 0.F;
      for (unsigned int k = 0; k < K; ++k) {
        float a = transA ? A[k * M + m] : A[m * K + k];
        float b = transB ? B[n * K + k] : B[k * N + n];
        sum += a * b;
      }
      ref[m * N + n] = sum + beta * static_cast<float>(C[m * N + n]);
    }
  }

  nntrainer::sgemm(0, transA, transB, M, N, K, 1.F, A.data(), transA ? M : K,
                   B.data(), transB ? K : N, beta, C.data(), N);

  for (unsigned int i = 0; i < C.size(); ++i) {
    EXPECT_EQ(static_cast<float>(C[i]), ref[i])
      << "M=" << M << " N=" << N << " K=" << K << " transA=" << transA
      << " transB=" << transB << " beta=" << beta << " idx=" << i;
  }
}

/**
 * @brief hgemm_small routes M < 8, K % 8 == 0 and N % 4 == 0 to the 1x8 / 1x4
 * kernels, which pack A in blocks of 1-3 rows. Covers every M hgemm_small can
 * see, both kernels, the neighbouring fallback path and all transpose modes.
 */
TEST(nntrainer_Tensor, hgemm_small_packing_A1_any_M) {
  nntrainer::init_backend();
  for (bool transA : {false, true})
    for (bool transB : {false, true})
      for (float beta : {0.F, 1.F})
        for (unsigned int M = 1; M < 8; ++M)
          for (unsigned int N : {4u, 8u, 12u})
            for (unsigned int K : {4u, 8u, 12u})
              check_small_hgemm(M, N, K, transA, transB, beta);
}

/**
 * @brief (3x8) x (8x8) fp16 dot, the shape that used to abort in packing_A1
 */
TEST(nntrainer_Tensor, dot_gemm_small_3_8_8) {
  int batch = 1;
  int channel = 1;
  int height = 3;
  int width = 8;

  int height_b = 8;
  int width_b = 8;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};
  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);
  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);
  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32);

  ASSERT_EQ(C.height(), (size_t)height);
  ASSERT_EQ(C.width(), (size_t)width_b);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());
  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, 1e-3 * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

/**
 * @brief a small fp16 dot with mismatched inner dimensions must be rejected
 * before it reaches hgemm
 */
TEST(nntrainer_Tensor, dot_gemm_small_incompatible_K_n) {
  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A(1, 1, 3, 8, t_type_nchw_fp16);
  nntrainer::Tensor B(1, 1, 4, 8, t_type_nchw_fp16);
  A.setValue(1.F);
  B.setValue(1.F);

  EXPECT_THROW(A.dot(B), std::runtime_error);
}

TEST(nntrainer_Tensor, dot_gemv_768_96000) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 768;
  int width_b = 96000;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemv_768_48000) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 768;
  int width_b = 48000;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemv_768_20000) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 768;

  int height_b = 768;
  int width_b = 20000;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, inv_sqrt_i_p) {
  int batch = 4;
  int channel = 10;
  int height = 10;
  int width = 10;

  const int MOD = 10;
  const float eps = 1e-3;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, (i * (batch * height) + j * (width) + k) % MOD + 1);

  nntrainer::Tensor ground_truth(batch, channel, height, width,
                                 nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP32);
  GEN_TEST_INPUT(ground_truth,
                 (i * (batch * height) + j * (width) + k) % MOD + 1);

  input.inv_sqrt_i();

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          ground_truth.setValue(
            b, c, h, w, 1 / std::sqrt(ground_truth.getValue(b, c, h, w)));
        }
      }
    }
  }

  bool flag = true;

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          double err = std::abs(input.getValue<__fp16>(b, c, h, w) -
                                ground_truth.getValue(b, c, h, w));

          if (err > eps) {
            flag = false;
            std::cout << input.getValue<__fp16>(b, c, h, w) << " VS "
                      << ground_truth.getValue(b, c, h, w) << std::endl;
          }
        }
      }
    }
  }

  EXPECT_EQ(flag, true);
}

/**
 * @brief (A*B).T = (B.T * A.T)
 *
 */
TEST(nntrainer_Tensor, transpose_check_with_gemm_fp16) {
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 512;

  int height_b = 512;
  int width_b = 768;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  GEN_TEST_INPUT_RAND(A, 0, 1);
  GEN_TEST_INPUT_RAND_B(B, 0, 1);

  nntrainer::Tensor A_T = A.transpose("0:2:1");
  nntrainer::Tensor B_T = B.transpose("0:2:1");
  nntrainer::Tensor C_T = B_T.dot(A_T);

  nntrainer::Tensor C = A.dot(B);
  nntrainer::Tensor C_transposed = C.transpose("0:2:1");

  EXPECT_EQ(C_T, C_transposed);
}

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
