// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file        unittest_nntrainer_quantizer.cpp
 * @date        16 December 2024
 * @brief       Unit test utility for quantizer.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <array>
#include <fstream>
#include <nntrainer_error.h>
#include <quantizer.h>
#include <tensor.h>

#include <cpu_backend.h>

/**
 * @brief Quantize to FP32 Tensor (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_01_n) {
  nntrainer::Tensor input(3, 2, 4, 5);
  input.setRandNormal(1.235f, 0.04f);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, nntrainer::Tdatatype::FP32),
               std::invalid_argument);
}

/**
 * @brief Dequantize to Quantized Tensor (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_02_n) {
  nntrainer::Tensor input(3, 3, 24, 24);
  input.setRandNormal(3.812f, 0.15f);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  nntrainer::Tensor quantized_tensor =
    quantizer->quantize(input, nntrainer::Tdatatype::QINT8);

  EXPECT_THROW(quantizer->dequantize(input, nntrainer::Tdatatype::QINT8),
               std::invalid_argument);
}

/**
 * @brief Quantize Quantized Tensor (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_03_n) {
  nntrainer::Tensor input(3, 3, 24, 24, nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::QINT8);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, nntrainer::Tdatatype::QINT4),
               std::invalid_argument);
}

/**
 * @brief Output tensor size not matching (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_04_n) {
  nntrainer::Tensor input(3, 2, 4, 5);
  input.setRandNormal(1.235f, 0.04f);

  nntrainer::Tensor output(3, 1, 1, 1, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::QINT8);

  float scale = 0.00235f;

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, output, &scale),
               std::invalid_argument);
}

/**
 * @brief Zero point not given for unsigned int (negative test)
 */
TEST(nntrainer_Quantizer, per_tensor_affine_05_n) {
  nntrainer::Tensor input(3, 2, 4, 5);
  input.setRandNormal(1.235f, 0.04f);

  nntrainer::Tensor uint8_output(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::UINT8);

  nntrainer::Tensor uint16_output(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                  nntrainer::Tdatatype::UINT16);

  float scale = 0.00235f;

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_THROW(quantizer->quantize(input, uint8_output, &scale),
               std::invalid_argument);

  EXPECT_THROW(quantizer->quantize(input, uint16_output, &scale),
               std::invalid_argument);
}

TEST(nntrainer_Quantizer, per_tensor_affine_01_p) {
  float input_data[] = {-0.16924214f, -0.10338581f, 0.31561565f,  -0.00533330f,
                        0.44809300f,  -0.15348488f, 0.14003623f,  -0.07908171f,
                        -0.21415669f, -0.35267806f, 0.46354777f,  -0.35009885f,
                        -0.07760239f, -0.28348053f, -0.37242615f, 0.30941701f};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  std::vector<int8_t> qdata = {-47, -28, 87,  -1,  123, -42, 39,   -22,
                               -59, -97, 127, -96, -21, -78, -102, 85};
  float qscale = 0.00363567f;
  int8_t *scale_array = reinterpret_cast<int8_t *>(&qscale);
  for (unsigned int i = 0; i < 4; ++i) {
    qdata.push_back(scale_array[i]);
  }
  nntrainer::Tensor quant_answer(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8},
    qdata.data());

  float output_data[] = {-0.17087643f, -0.10179872f, 0.31630316f,  -0.00363567f,
                         0.44718724f,  -0.15269808f, 0.14179108f,  -0.07998471f,
                         -0.21450445f, -0.35265985f, 0.46172991f,  -0.34902418f,
                         -0.07634904f, -0.28358215f, -0.37083820f, 0.30903184f};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  nntrainer::Tensor quantized_tensor =
    quantizer->quantize(input, nntrainer::Tdatatype::QINT8);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

TEST(nntrainer_Quantizer, per_tensor_affine_02_p) {
  float input_data[] = {
    -0.29562217f, 0.02348283f,  0.04334664f,  0.03752254f,  0.17764580f,
    0.04449826f,  0.15144463f,  -0.15716791f, -0.07842141f, 0.34517670f,
    0.16458672f,  -0.09487095f, -0.28020513f, 0.32698259f,  -0.24903688f,
    -0.33132783f, 0.13940062f,  0.18400775f,  -0.26359966f, 0.30900121f,
    0.08309542f,  -0.09066082f, 0.08950174f,  -0.29709017f, -0.26397359f,
    -0.16240828f, -0.18758762f, -0.31878781f, 0.06728745f,  -0.04749811f,
    0.16789703f,  0.02212419f,  0.10671097f,  -0.28938687f, 0.16250020f,
    -0.09017495f, 0.24699482f,  -0.26789218f, 0.16414545f,  0.22879964f,
    -0.15821624f, -0.23149055f, 0.26526868f,  -0.11006282f, -0.20480227f,
    0.29863110f,  0.24005184f,  -0.09062263f, 0.22294718f,  0.32583672f,
    -0.10362835f, 0.03243832f,  0.24707781f,  0.27685603f,  0.03360258f,
    -0.00209959f, 0.27976128f,  -0.24468939f, -0.19273037f, -0.25921509f,
    -0.20489319f, 0.33036807f,  0.27226517f,  -0.25207010f};
  nntrainer::Tensor input({1, 1, 8, 8}, input_data);

  std::vector<int8_t> qdata = {
    -109, 9,    16,   14,  66,  16,  56,  -58,  -29, 127, 61,   -35, -104,
    121,  -92,  -122, 51,  68,  -97, 114, 31,   -33, 33,  -110, -98, -60,
    -69,  -118, 25,   -18, 62,  8,   39,  -107, 60,  -33, 91,   -99, 61,
    85,   -58,  -86,  98,  -41, -76, 110, 89,   -33, 82,  120,  -38, 12,
    91,   102,  12,   -1,  103, -90, -71, -96,  -76, 122, 101,  -93};
  float qscale = 0.00270727f;
  int8_t *scale_array = reinterpret_cast<int8_t *>(&qscale);
  for (unsigned int i = 0; i < 4; ++i) {
    qdata.push_back(scale_array[i]);
  }
  nntrainer::Tensor quant_answer(
    {1, 1, 8, 8, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8},
    qdata.data());

  float output_data[] = {
    -0.29509223f, 0.02436541f,  0.04331629f,  0.03790175f,  0.17867969f,
    0.04331629f,  0.15160701f,  -0.15702155f, -0.07851078f, 0.34382305f,
    0.16514336f,  -0.09475438f, -0.28155589f, 0.32757944f,  -0.24906866f,
    -0.33028671f, 0.13807067f,  0.18409424f,  -0.26260501f, 0.30862856f,
    0.08392531f,  -0.08933984f, 0.08933984f,  -0.29779950f, -0.26531228f,
    -0.16243608f, -0.18680149f, -0.31945765f, 0.06768170f,  -0.04873083f,
    0.16785063f,  0.02165814f,  0.10558346f,  -0.28967768f, 0.16243608f,
    -0.08933984f, 0.24636140f,  -0.26801956f, 0.16514336f,  0.23011778f,
    -0.15702155f, -0.23282506f, 0.26531228f,  -0.11099799f, -0.20575237f,
    0.29779950f,  0.24094686f,  -0.08933984f, 0.22199598f,  0.32487217f,
    -0.10287619f, 0.03248722f,  0.24636140f,  0.27614135f,  0.03248722f,
    -0.00270727f, 0.27884862f,  -0.24365413f, -0.19221604f, -0.25989774f,
    -0.20575237f, 0.33028671f,  0.27343407f,  -0.25177592f};
  nntrainer::Tensor float_answer({1, 1, 8, 8}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  nntrainer::Tensor quantized_tensor =
    quantizer->quantize(input, nntrainer::Tdatatype::QINT8);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

TEST(nntrainer_Quantizer, per_tensor_affine_03_p) {
  float input_data[] = {-0.16924214f, -0.10338581f, 0.31561565f,  -0.00533330f,
                        0.44809300f,  -0.15348488f, 0.14003623f,  -0.07908171f,
                        -0.21415669f, -0.35267806f, 0.46354777f,  -0.35009885f,
                        -0.07760239f, -0.28348053f, -0.37242615f, 0.30941701f};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  std::vector<int16_t> qdata = {-47, -28, 87,  -1,  123, -42, 39,   -22,
                                -59, -97, 127, -96, -21, -78, -102, 85};
  float qscale = 0.00363567f;
  int16_t *scale_array = reinterpret_cast<int16_t *>(&qscale);
  for (unsigned int i = 0; i < 2; ++i) {
    qdata.push_back(scale_array[i]);
  }
  nntrainer::Tensor quant_answer(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT16},
    qdata.data());

  nntrainer::Tensor quantized_tensor(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::QINT16);

  float output_data[] = {-0.17087643f, -0.10179872f, 0.31630316f,  -0.00363567f,
                         0.44718724f,  -0.15269808f, 0.14179108f,  -0.07998471f,
                         -0.21450445f, -0.35265985f, 0.46172991f,  -0.34902418f,
                         -0.07634904f, -0.28358215f, -0.37083820f, 0.30903184f};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  quantizer->quantize(input, quantized_tensor, &qscale);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

/**
 * @brief Quantize / Dequantize UInt8Tensor
 */
TEST(nntrainer_Quantizer, per_tensor_affine_04_p) {
  float input_data[] = {2.31893802f,  4.46305752f,  -0.75207627f, -2.51219273f,
                        -0.59212941f, -3.74816823f, 0.58360142f,  0.86855388f,
                        2.07299328f,  2.69872355f,  -1.41879117f, 2.31787777f,
                        -0.29471058f, -0.72146493f, 1.81435537f,  -1.59683037f};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  std::vector<uint8_t> qdata = {194, 255, 107, 56,  111, 21,  145, 153,
                                187, 205, 87,  194, 120, 107, 180, 82};
  float qscale = 0.03500437f;
  unsigned int zero_point = 128;

  uint8_t *scale_array = reinterpret_cast<uint8_t *>(&qscale);
  for (unsigned int i = 0; i < sizeof(float) / sizeof(uint8_t); ++i) {
    qdata.push_back(scale_array[i]);
  }

  uint8_t *zp_array = reinterpret_cast<uint8_t *>(&zero_point);
  for (unsigned int i = 0; i < sizeof(unsigned int) / sizeof(uint8_t); ++i) {
    qdata.push_back(zp_array[i]);
  }

  nntrainer::Tensor quant_answer(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8},
    qdata.data());

  nntrainer::Tensor quantized_tensor(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::UINT8);

  float output_data[] = {
    2.31028867f,  4.44555569f,  -0.73509187f, -2.52031493f,
    -0.59507436f, -3.74546790f, 0.59507436f,  0.87510931f,
    2.06525803f,  2.69533682f,  -1.43517935f, 2.31028867f,
    -0.28003499f, -0.73509187f, 1.82022738f,  -1.61020124f};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  quantizer->quantize(input, quantized_tensor, &qscale, &zero_point);
  ASSERT_EQ(quantized_tensor, quant_answer);

  // Perform Dequantization
  nntrainer::Tensor output =
    quantizer->dequantize(quantized_tensor, nntrainer::Tdatatype::FP32);
  ASSERT_EQ(output, float_answer);
}

/**
 * @brief Tensor quantization to QINT8 and QUINT8
 *
 * @note This test quantizes a float tensor to int8 and uint8, then compares the
 * dequantized output to check if they are the same.
 */
TEST(nntrainer_Quantizer, per_tensor_affine_05_p) {
  // float input tensor
  float input_data[] = {-0.69094253f, 3.77131414f,  -4.77607393f, 1.86816788f,
                        -2.97529221f, 3.99959946f,  -1.44690418f, 2.54158640f,
                        -0.79941863f, -3.75069141f, -1.38934612f, -0.23342809f,
                        -4.05783129f, 1.41701365f,  -0.84545374f, 2.20419312f};
  nntrainer::Tensor input({1, 1, 4, 4}, input_data);

  // quantization params
  float scale = 0.03136941f;
  unsigned int zero_point = 128;

  // Construct unsigned 8-int tensor (answer)
  std::vector<uint8_t> data_u8 = {106, 248, 0,  188, 33, 255, 82,  209,
                                  103, 8,   84, 121, 0,  173, 101, 198};

  uint8_t *scales_u8 = reinterpret_cast<uint8_t *>(&scale);
  for (unsigned int i = 0; i < sizeof(float) / sizeof(uint8_t); ++i) {
    data_u8.push_back(scales_u8[i]);
  }

  uint8_t *zps_u8 = reinterpret_cast<uint8_t *>(&zero_point);
  for (unsigned int i = 0; i < sizeof(unsigned int) / sizeof(uint8_t); ++i) {
    data_u8.push_back(zps_u8[i]);
  }

  nntrainer::Tensor q_answer_u8(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT8},
    data_u8.data());

  // Construct signed 8-int tensor (answer)
  std::vector<int8_t> data_s8 = {-22, 120,  -128, 60, -95,  127, -46, 81,
                                 -25, -120, -44,  -7, -128, 45,  -27, 70};

  int8_t *scales_s8 = reinterpret_cast<int8_t *>(&scale);
  for (unsigned int i = 0; i < sizeof(float) / sizeof(int8_t); ++i) {
    data_s8.push_back(scales_s8[i]);
  }

  int8_t *zps_s8 = reinterpret_cast<int8_t *>(&zero_point);
  for (unsigned int i = 0; i < sizeof(unsigned int) / sizeof(int8_t); ++i) {
    data_s8.push_back(zps_s8[i]);
  }

  nntrainer::Tensor q_answer_s8(
    {1, 1, 4, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8},
    data_s8.data());

  // Output quantized tensors
  nntrainer::Tensor q_tensor_u8(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::UINT8);

  nntrainer::Tensor q_tensor_s8(1, 1, 4, 4, nntrainer::Tformat::NCHW,
                                nntrainer::Tdatatype::QINT8);

  // Construct output float tensor (answer)
  float output_data[] = {-0.69012696f, 3.76432872f,  -4.01528406f, 1.88216436f,
                         -2.98009372f, 3.98391461f,  -1.44299269f, 2.54092193f,
                         -0.78423518f, -3.76432872f, -1.38025391f, -0.21958585f,
                         -4.01528406f, 1.41162336f,  -0.84697396f, 2.19585848f};
  nntrainer::Tensor float_answer({1, 1, 4, 4}, output_data);

  // Per tensor affine quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  // Perform Quantization
  quantizer->quantize(input, q_tensor_u8, &scale, &zero_point);
  ASSERT_EQ(q_tensor_u8, q_answer_u8);

  quantizer->quantize(input, q_tensor_s8, &scale, &zero_point);
  ASSERT_EQ(q_tensor_s8, q_answer_s8);

  // Perform Dequantization
  nntrainer::Tensor output_s8 =
    quantizer->dequantize(q_tensor_s8, nntrainer::Tdatatype::FP32);

  nntrainer::Tensor output_u8 =
    quantizer->dequantize(q_tensor_u8, nntrainer::Tdatatype::FP32);

  ASSERT_EQ(output_s8, float_answer);
  ASSERT_EQ(output_u8, float_answer);
}

/**
 * @brief Automatically calculate UINT8 affine parameters and round-trip data.
 */
TEST(nntrainer_Quantizer, per_tensor_affine_uint8_auto_qparams) {
  const std::array<std::array<float, 3>, 3> inputs = {{
    {-127.5f, 0.0f, 127.5f},
    {0.0f, 127.5f, 255.0f},
    {-255.0f, -127.5f, 0.0f},
  }};
  const std::array<unsigned int, 3> expected_zero_points = {128, 0, 255};

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  for (unsigned int i = 0; i < inputs.size(); ++i) {
    nntrainer::Tensor input({1, 1, 1, 3}, inputs[i].data());
    nntrainer::Tensor quantized =
      quantizer->quantize(input, nntrainer::Tdatatype::UINT8);

    EXPECT_EQ(*quantized.getZeroPoint(), expected_zero_points[i]);
    EXPECT_FLOAT_EQ(*quantized.getScale<float>(), 1.0f);

    nntrainer::Tensor output =
      quantizer->dequantize(quantized, nntrainer::Tdatatype::FP32);
    for (unsigned int w = 0; w < 3; ++w)
      EXPECT_NEAR(output.getValue(0, 0, 0, w), inputs[i][w], 0.50001f);
  }
}

/**
 * @brief Automatically calculate UINT16 and UINT4 affine parameters.
 */
TEST(nntrainer_Quantizer, per_tensor_affine_other_unsigned_auto_qparams) {
  float uint16_data[] = {-32767.5f, 0.0f, 32767.5f};
  nntrainer::Tensor uint16_input({1, 1, 1, 3}, uint16_data);
  float uint4_data[] = {-7.5f, 0.0f, 7.5f};
  nntrainer::Tensor uint4_input({1, 1, 1, 3}, uint4_data);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);

  nntrainer::Tensor uint16_quantized =
    quantizer->quantize(uint16_input, nntrainer::Tdatatype::UINT16);
  EXPECT_EQ(*uint16_quantized.getZeroPoint(), 32768u);
  EXPECT_FLOAT_EQ(*uint16_quantized.getScale<float>(), 1.0f);
  EXPECT_EQ(uint16_quantized.getValue<uint16_t>(0), 0u);
  EXPECT_EQ(uint16_quantized.getValue<uint16_t>(1), 32768u);
  EXPECT_EQ(uint16_quantized.getValue<uint16_t>(2), 65535u);

  nntrainer::Tensor uint4_quantized =
    quantizer->quantize(uint4_input, nntrainer::Tdatatype::UINT4);
  EXPECT_EQ(*uint4_quantized.getZeroPoint(), 8u);
  EXPECT_FLOAT_EQ(*uint4_quantized.getScale<float>(), 1.0f);

  nntrainer::Tensor uint4_output =
    quantizer->dequantize(uint4_quantized, nntrainer::Tdatatype::FP32);
  for (unsigned int w = 0; w < 3; ++w)
    EXPECT_NEAR(uint4_output.getValue(0, 0, 0, w), uint4_data[w], 0.50001f);
}

/**
 * @brief UINT8 automatic affine codes are invariant under unit scaling.
 */
TEST(nntrainer_Quantizer, per_tensor_affine_uint8_unit_invariance) {
  float input_data[] = {-128.0f, 0.0f, 127.0f};
  float scaled_input_data[] = {-256.0f, 0.0f, 254.0f};
  nntrainer::Tensor input({1, 1, 1, 3}, input_data);
  nntrainer::Tensor scaled_input({1, 1, 1, 3}, scaled_input_data);

  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(
      nntrainer::QScheme::PER_TENSOR_AFFINE);
  nntrainer::Tensor quantized =
    quantizer->quantize(input, nntrainer::Tdatatype::UINT8);
  nntrainer::Tensor scaled_quantized =
    quantizer->quantize(scaled_input, nntrainer::Tdatatype::UINT8);

  EXPECT_EQ(*quantized.getZeroPoint(), *scaled_quantized.getZeroPoint());
  for (unsigned int w = 0; w < 3; ++w)
    EXPECT_EQ(quantized.getValue<uint8_t>(w),
              scaled_quantized.getValue<uint8_t>(w));
}

/**
 * @brief GGMLQuantizer Q6_K quantize and dequantize
 */
TEST(nntrainer_Quantizer, ggml_q6k_01_p) {
  nntrainer::init_backend();

  uint32_t K = 512;
  uint32_t N = 256;

  std::vector<float> weight =
    generate_random_vector<float>(K * N, -0.05f, 0.05f);

  nntrainer::Tensor W_fp32(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      W_fp32.setValue(0, 0, k, n, weight[k * N + n]);

  // Create GGML Q6_K quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q6_K);

  EXPECT_EQ(quantizer->qscheme(), nntrainer::QScheme::Q6_K);

  // Quantize
  nntrainer::Tensor W_q6k =
    quantizer->quantize(W_fp32, nntrainer::Tdatatype::Q6_K);

  EXPECT_EQ(W_q6k.getDataType(), nntrainer::Tdatatype::Q6_K);

  // Dequantize
  nntrainer::Tensor W_deq =
    quantizer->dequantize(W_q6k, nntrainer::Tdatatype::FP32);

  EXPECT_EQ(W_deq.getDataType(), nntrainer::Tdatatype::FP32);

  // Verify quantization quality: MSE should be small
  auto mean_squared_error =
    mse<float, float>(W_fp32.getData(), W_deq.getData(), K * N);

  const float eps = 1e-5;
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
}

/**
 * @brief GGMLQuantizer Q4_0 quantize and dequantize
 */
TEST(nntrainer_Quantizer, ggml_q4_0_01_p) {
  nntrainer::init_backend();

  uint32_t K = 768;
  uint32_t N = 512;

  std::vector<float> weight =
    generate_random_vector<float>(K * N, -0.05f, 0.05f);

  nntrainer::Tensor W_fp32(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n)
      W_fp32.setValue(0, 0, k, n, weight[k * N + n]);

  // Create GGML Q4_0 quantizer
  std::unique_ptr<nntrainer::Quantizer> quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q4_0);

  EXPECT_EQ(quantizer->qscheme(), nntrainer::QScheme::Q4_0);

  // Quantize
  nntrainer::Tensor W_q40 =
    quantizer->quantize(W_fp32, nntrainer::Tdatatype::Q4_0);

  EXPECT_EQ(W_q40.getDataType(), nntrainer::Tdatatype::Q4_0);

  // Dequantize
  nntrainer::Tensor W_deq =
    quantizer->dequantize(W_q40, nntrainer::Tdatatype::FP32);

  EXPECT_EQ(W_deq.getDataType(), nntrainer::Tdatatype::FP32);

  // Verify quantization quality: MSE should be small
  auto mean_squared_error =
    mse<float, float>(W_fp32.getData(), W_deq.getData(), K * N);

  const float eps = 1e-5;
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
}

int main(int argc, char **argv) {
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
