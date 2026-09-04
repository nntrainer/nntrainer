// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_nntrainer_tensor.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for tensor.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <nntrainer_error.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

TEST(nntrainer_Tensor, Tensor_01_fp16_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(
    1, 2, 3, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData<_FP16>());
  if (tensor.getValue<_FP16>(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_01_nhwc_fp16_p) {
  int status = ML_ERROR_NONE;
  nntrainer::Tensor tensor = nntrainer::Tensor(
    1, 2, 3, nntrainer::Tformat::NHWC, nntrainer::Tdatatype::FP16);
  tensor.setZero();
  ASSERT_NE(nullptr, tensor.getData<_FP16>());
  if (tensor.getValue<_FP16>(0, 0, 0, 0) != 0.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_02_fp16_p) {
  int status = ML_ERROR_NONE;
  int height = 3;
  int width = 10;
  std::vector<std::vector<_FP16>> in;
  for (int i = 0; i < height; ++i) {
    std::vector<_FP16> tv;
    for (int j = 0; j < width; ++j) {
      tv.push_back(static_cast<_FP16>(i * 2.0 + j));
    }
    in.push_back(tv);
  }

  nntrainer::Tensor tensor =
    nntrainer::Tensor(in, {ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP16});
  ASSERT_NE(nullptr, tensor.getData<_FP16>());

  if (tensor.getValue<_FP16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, Tensor_03_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  std::vector<std::vector<std::vector<_FP16>>> in;
  for (int k = 0; k < batch; ++k) {
    std::vector<std::vector<_FP16>> ttv;
    for (int i = 0; i < height; ++i) {
      std::vector<_FP16> tv;
      for (int j = 0; j < width; ++j) {
        tv.push_back(k * height * width + i * width + j);
      }
      ttv.push_back(tv);
    }
    in.push_back(ttv);
  }

  nntrainer::Tensor tensor =
    nntrainer::Tensor(in, {ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP16});
  ASSERT_NE(nullptr, tensor.getData<_FP16>());

  if (tensor.getValue<_FP16>(0, 0, 0, 1) != 1.0)
    status = ML_ERROR_INVALID_PARAMETER;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, default_constructor_with_tensor_type_fp16_p) {
  unsigned int b = 3;
  unsigned int c = 2;
  unsigned int h = 4;
  unsigned int w = 5;

  nntrainer::TensorDim::TensorType tensor_type = {
    nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP16};

  nntrainer::TensorDim t = {w, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, 1, w, tensor_type), t);
  EXPECT_EQ(nntrainer::TensorDim(w, tensor_type), t);

  t = {h, w, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(1, 1, h, w, tensor_type), t);
  EXPECT_EQ(nntrainer::TensorDim(h, w, tensor_type), t);

  t = {c, h, w, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(1, c, h, w, tensor_type), t);
  EXPECT_EQ(nntrainer::TensorDim(c, h, w, tensor_type), t);

  t = {b, h, w, c, tensor_type};
  EXPECT_EQ(nntrainer::TensorDim(b, h, w, c, tensor_type), t);
}

TEST(nntrainer_Tensor, multiply_i_01_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);

  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.multiply_i(2.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = original.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i] + data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_02_fp16_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.multiply_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = original.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i] * data[i], indata[i]);
  }
}

TEST(nntrainer_Tensor, multiply_i_03_fp16_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor target2(batch, channel, height - 2, width - 1,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);
  status = input.multiply_i(target2);

  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_01_fp16_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(25),
                           static_cast<_FP16>(36),   static_cast<_FP16>(49),
                           static_cast<_FP16>(64),   static_cast<_FP16>(81),
                           static_cast<_FP16>(100),  static_cast<_FP16>(121),
                           static_cast<_FP16>(144),  static_cast<_FP16>(169),
                           static_cast<_FP16>(196),  static_cast<_FP16>(225),
                           static_cast<_FP16>(256),  static_cast<_FP16>(289),
                           static_cast<_FP16>(324),  static_cast<_FP16>(361),
                           static_cast<_FP16>(400),  static_cast<_FP16>(441),
                           static_cast<_FP16>(484),  static_cast<_FP16>(529),
                           static_cast<_FP16>(576),  static_cast<_FP16>(625),
                           static_cast<_FP16>(676),  static_cast<_FP16>(729),
                           static_cast<_FP16>(784),  static_cast<_FP16>(841),
                           static_cast<_FP16>(900),  static_cast<_FP16>(961),
                           static_cast<_FP16>(1024), static_cast<_FP16>(1089),
                           static_cast<_FP16>(1156), static_cast<_FP16>(1225),
                           static_cast<_FP16>(1296), static_cast<_FP16>(1369),
                           static_cast<_FP16>(1444), static_cast<_FP16>(1521),
                           static_cast<_FP16>(0),    static_cast<_FP16>(41),
                           static_cast<_FP16>(84),   static_cast<_FP16>(129),
                           static_cast<_FP16>(176),  static_cast<_FP16>(225),
                           static_cast<_FP16>(276),  static_cast<_FP16>(329),
                           static_cast<_FP16>(384),  static_cast<_FP16>(441),
                           static_cast<_FP16>(500),  static_cast<_FP16>(561),
                           static_cast<_FP16>(624),  static_cast<_FP16>(689),
                           static_cast<_FP16>(756),  static_cast<_FP16>(825),
                           static_cast<_FP16>(896),  static_cast<_FP16>(969),
                           static_cast<_FP16>(1044), static_cast<_FP16>(1121),
                           static_cast<_FP16>(1200), static_cast<_FP16>(1281),
                           static_cast<_FP16>(1364), static_cast<_FP16>(1449),
                           static_cast<_FP16>(1536), static_cast<_FP16>(1625),
                           static_cast<_FP16>(1716), static_cast<_FP16>(1809),
                           static_cast<_FP16>(1904), static_cast<_FP16>(2001),
                           static_cast<_FP16>(2100), static_cast<_FP16>(2201),
                           static_cast<_FP16>(2304), static_cast<_FP16>(2409),
                           static_cast<_FP16>(2516), static_cast<_FP16>(2625),
                           static_cast<_FP16>(2736), static_cast<_FP16>(2849),
                           static_cast<_FP16>(2964), static_cast<_FP16>(3081),
                           static_cast<_FP16>(0),    static_cast<_FP16>(81),
                           static_cast<_FP16>(164),  static_cast<_FP16>(249),
                           static_cast<_FP16>(336),  static_cast<_FP16>(425),
                           static_cast<_FP16>(516),  static_cast<_FP16>(609),
                           static_cast<_FP16>(704),  static_cast<_FP16>(801),
                           static_cast<_FP16>(900),  static_cast<_FP16>(1001),
                           static_cast<_FP16>(1104), static_cast<_FP16>(1209),
                           static_cast<_FP16>(1316), static_cast<_FP16>(1425),
                           static_cast<_FP16>(1536), static_cast<_FP16>(1649),
                           static_cast<_FP16>(1764), static_cast<_FP16>(1881),
                           static_cast<_FP16>(2000), static_cast<_FP16>(2121),
                           static_cast<_FP16>(2244), static_cast<_FP16>(2369),
                           static_cast<_FP16>(2496), static_cast<_FP16>(2625),
                           static_cast<_FP16>(2756), static_cast<_FP16>(2889),
                           static_cast<_FP16>(3024), static_cast<_FP16>(3161),
                           static_cast<_FP16>(3300), static_cast<_FP16>(3441),
                           static_cast<_FP16>(3584), static_cast<_FP16>(3729),
                           static_cast<_FP16>(3876), static_cast<_FP16>(4025),
                           static_cast<_FP16>(4176), static_cast<_FP16>(4329),
                           static_cast<_FP16>(4484), static_cast<_FP16>(4641)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(25),
                           static_cast<_FP16>(36),   static_cast<_FP16>(49),
                           static_cast<_FP16>(64),   static_cast<_FP16>(81),
                           static_cast<_FP16>(100),  static_cast<_FP16>(121),
                           static_cast<_FP16>(144),  static_cast<_FP16>(169),
                           static_cast<_FP16>(196),  static_cast<_FP16>(225),
                           static_cast<_FP16>(256),  static_cast<_FP16>(289),
                           static_cast<_FP16>(324),  static_cast<_FP16>(361),
                           static_cast<_FP16>(0),    static_cast<_FP16>(21),
                           static_cast<_FP16>(44),   static_cast<_FP16>(69),
                           static_cast<_FP16>(96),   static_cast<_FP16>(125),
                           static_cast<_FP16>(156),  static_cast<_FP16>(189),
                           static_cast<_FP16>(224),  static_cast<_FP16>(261),
                           static_cast<_FP16>(300),  static_cast<_FP16>(341),
                           static_cast<_FP16>(384),  static_cast<_FP16>(429),
                           static_cast<_FP16>(476),  static_cast<_FP16>(525),
                           static_cast<_FP16>(576),  static_cast<_FP16>(629),
                           static_cast<_FP16>(684),  static_cast<_FP16>(741),
                           static_cast<_FP16>(800),  static_cast<_FP16>(861),
                           static_cast<_FP16>(924),  static_cast<_FP16>(989),
                           static_cast<_FP16>(1056), static_cast<_FP16>(1125),
                           static_cast<_FP16>(1196), static_cast<_FP16>(1269),
                           static_cast<_FP16>(1344), static_cast<_FP16>(1421),
                           static_cast<_FP16>(1500), static_cast<_FP16>(1581),
                           static_cast<_FP16>(1664), static_cast<_FP16>(1749),
                           static_cast<_FP16>(1836), static_cast<_FP16>(1925),
                           static_cast<_FP16>(2016), static_cast<_FP16>(2109),
                           static_cast<_FP16>(2204), static_cast<_FP16>(2301),
                           static_cast<_FP16>(1200), static_cast<_FP16>(1281),
                           static_cast<_FP16>(1364), static_cast<_FP16>(1449),
                           static_cast<_FP16>(1536), static_cast<_FP16>(1625),
                           static_cast<_FP16>(1716), static_cast<_FP16>(1809),
                           static_cast<_FP16>(1904), static_cast<_FP16>(2001),
                           static_cast<_FP16>(2100), static_cast<_FP16>(2201),
                           static_cast<_FP16>(2304), static_cast<_FP16>(2409),
                           static_cast<_FP16>(2516), static_cast<_FP16>(2625),
                           static_cast<_FP16>(2736), static_cast<_FP16>(2849),
                           static_cast<_FP16>(2964), static_cast<_FP16>(3081),
                           static_cast<_FP16>(3200), static_cast<_FP16>(3321),
                           static_cast<_FP16>(3444), static_cast<_FP16>(3569),
                           static_cast<_FP16>(3696), static_cast<_FP16>(3825),
                           static_cast<_FP16>(3956), static_cast<_FP16>(4089),
                           static_cast<_FP16>(4224), static_cast<_FP16>(4361),
                           static_cast<_FP16>(4500), static_cast<_FP16>(4641),
                           static_cast<_FP16>(4784), static_cast<_FP16>(4929),
                           static_cast<_FP16>(5076), static_cast<_FP16>(5225),
                           static_cast<_FP16>(5376), static_cast<_FP16>(5529),
                           static_cast<_FP16>(5684), static_cast<_FP16>(5841),
                           static_cast<_FP16>(4000), static_cast<_FP16>(4141),
                           static_cast<_FP16>(4284), static_cast<_FP16>(4429),
                           static_cast<_FP16>(4576), static_cast<_FP16>(4725),
                           static_cast<_FP16>(4876), static_cast<_FP16>(5029),
                           static_cast<_FP16>(5184), static_cast<_FP16>(5341),
                           static_cast<_FP16>(5500), static_cast<_FP16>(5661),
                           static_cast<_FP16>(5824), static_cast<_FP16>(5989),
                           static_cast<_FP16>(6156), static_cast<_FP16>(6325),
                           static_cast<_FP16>(6496), static_cast<_FP16>(6669),
                           static_cast<_FP16>(6844), static_cast<_FP16>(7021)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 2, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(5),
                           static_cast<_FP16>(6),    static_cast<_FP16>(7),
                           static_cast<_FP16>(8),    static_cast<_FP16>(9),
                           static_cast<_FP16>(20),   static_cast<_FP16>(22),
                           static_cast<_FP16>(24),   static_cast<_FP16>(26),
                           static_cast<_FP16>(28),   static_cast<_FP16>(45),
                           static_cast<_FP16>(48),   static_cast<_FP16>(51),
                           static_cast<_FP16>(54),   static_cast<_FP16>(57),
                           static_cast<_FP16>(80),   static_cast<_FP16>(84),
                           static_cast<_FP16>(88),   static_cast<_FP16>(92),
                           static_cast<_FP16>(96),   static_cast<_FP16>(125),
                           static_cast<_FP16>(130),  static_cast<_FP16>(135),
                           static_cast<_FP16>(140),  static_cast<_FP16>(145),
                           static_cast<_FP16>(180),  static_cast<_FP16>(186),
                           static_cast<_FP16>(192),  static_cast<_FP16>(198),
                           static_cast<_FP16>(204),  static_cast<_FP16>(245),
                           static_cast<_FP16>(252),  static_cast<_FP16>(259),
                           static_cast<_FP16>(266),  static_cast<_FP16>(273),
                           static_cast<_FP16>(320),  static_cast<_FP16>(328),
                           static_cast<_FP16>(336),  static_cast<_FP16>(344),
                           static_cast<_FP16>(352),  static_cast<_FP16>(405),
                           static_cast<_FP16>(414),  static_cast<_FP16>(423),
                           static_cast<_FP16>(432),  static_cast<_FP16>(441),
                           static_cast<_FP16>(500),  static_cast<_FP16>(510),
                           static_cast<_FP16>(520),  static_cast<_FP16>(530),
                           static_cast<_FP16>(540),  static_cast<_FP16>(605),
                           static_cast<_FP16>(616),  static_cast<_FP16>(627),
                           static_cast<_FP16>(638),  static_cast<_FP16>(649),
                           static_cast<_FP16>(720),  static_cast<_FP16>(732),
                           static_cast<_FP16>(744),  static_cast<_FP16>(756),
                           static_cast<_FP16>(768),  static_cast<_FP16>(845),
                           static_cast<_FP16>(858),  static_cast<_FP16>(871),
                           static_cast<_FP16>(884),  static_cast<_FP16>(897),
                           static_cast<_FP16>(980),  static_cast<_FP16>(994),
                           static_cast<_FP16>(1008), static_cast<_FP16>(1022),
                           static_cast<_FP16>(1036), static_cast<_FP16>(1125),
                           static_cast<_FP16>(1140), static_cast<_FP16>(1155),
                           static_cast<_FP16>(1170), static_cast<_FP16>(1185),
                           static_cast<_FP16>(1280), static_cast<_FP16>(1296),
                           static_cast<_FP16>(1312), static_cast<_FP16>(1328),
                           static_cast<_FP16>(1344), static_cast<_FP16>(1445),
                           static_cast<_FP16>(1462), static_cast<_FP16>(1479),
                           static_cast<_FP16>(1496), static_cast<_FP16>(1513),
                           static_cast<_FP16>(1620), static_cast<_FP16>(1638),
                           static_cast<_FP16>(1656), static_cast<_FP16>(1674),
                           static_cast<_FP16>(1692), static_cast<_FP16>(1805),
                           static_cast<_FP16>(1824), static_cast<_FP16>(1843),
                           static_cast<_FP16>(1862), static_cast<_FP16>(1881),
                           static_cast<_FP16>(2000), static_cast<_FP16>(2020),
                           static_cast<_FP16>(2040), static_cast<_FP16>(2060),
                           static_cast<_FP16>(2080), static_cast<_FP16>(2205),
                           static_cast<_FP16>(2226), static_cast<_FP16>(2247),
                           static_cast<_FP16>(2268), static_cast<_FP16>(2289),
                           static_cast<_FP16>(2420), static_cast<_FP16>(2442),
                           static_cast<_FP16>(2464), static_cast<_FP16>(2486),
                           static_cast<_FP16>(2508), static_cast<_FP16>(2645),
                           static_cast<_FP16>(2668), static_cast<_FP16>(2691),
                           static_cast<_FP16>(2714), static_cast<_FP16>(2737)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(0),
                           static_cast<_FP16>(6),    static_cast<_FP16>(14),
                           static_cast<_FP16>(24),   static_cast<_FP16>(36),
                           static_cast<_FP16>(0),    static_cast<_FP16>(11),
                           static_cast<_FP16>(24),   static_cast<_FP16>(39),
                           static_cast<_FP16>(56),   static_cast<_FP16>(0),
                           static_cast<_FP16>(16),   static_cast<_FP16>(34),
                           static_cast<_FP16>(54),   static_cast<_FP16>(76),
                           static_cast<_FP16>(0),    static_cast<_FP16>(21),
                           static_cast<_FP16>(44),   static_cast<_FP16>(69),
                           static_cast<_FP16>(96),   static_cast<_FP16>(0),
                           static_cast<_FP16>(26),   static_cast<_FP16>(54),
                           static_cast<_FP16>(84),   static_cast<_FP16>(116),
                           static_cast<_FP16>(0),    static_cast<_FP16>(31),
                           static_cast<_FP16>(64),   static_cast<_FP16>(99),
                           static_cast<_FP16>(136),  static_cast<_FP16>(0),
                           static_cast<_FP16>(36),   static_cast<_FP16>(74),
                           static_cast<_FP16>(114),  static_cast<_FP16>(156),
                           static_cast<_FP16>(200),  static_cast<_FP16>(246),
                           static_cast<_FP16>(294),  static_cast<_FP16>(344),
                           static_cast<_FP16>(396),  static_cast<_FP16>(225),
                           static_cast<_FP16>(276),  static_cast<_FP16>(329),
                           static_cast<_FP16>(384),  static_cast<_FP16>(441),
                           static_cast<_FP16>(250),  static_cast<_FP16>(306),
                           static_cast<_FP16>(364),  static_cast<_FP16>(424),
                           static_cast<_FP16>(486),  static_cast<_FP16>(275),
                           static_cast<_FP16>(336),  static_cast<_FP16>(399),
                           static_cast<_FP16>(464),  static_cast<_FP16>(531),
                           static_cast<_FP16>(300),  static_cast<_FP16>(366),
                           static_cast<_FP16>(434),  static_cast<_FP16>(504),
                           static_cast<_FP16>(576),  static_cast<_FP16>(325),
                           static_cast<_FP16>(396),  static_cast<_FP16>(469),
                           static_cast<_FP16>(544),  static_cast<_FP16>(621),
                           static_cast<_FP16>(350),  static_cast<_FP16>(426),
                           static_cast<_FP16>(504),  static_cast<_FP16>(584),
                           static_cast<_FP16>(666),  static_cast<_FP16>(375),
                           static_cast<_FP16>(456),  static_cast<_FP16>(539),
                           static_cast<_FP16>(624),  static_cast<_FP16>(711),
                           static_cast<_FP16>(800),  static_cast<_FP16>(891),
                           static_cast<_FP16>(984),  static_cast<_FP16>(1079),
                           static_cast<_FP16>(1176), static_cast<_FP16>(850),
                           static_cast<_FP16>(946),  static_cast<_FP16>(1044),
                           static_cast<_FP16>(1144), static_cast<_FP16>(1246),
                           static_cast<_FP16>(900),  static_cast<_FP16>(1001),
                           static_cast<_FP16>(1104), static_cast<_FP16>(1209),
                           static_cast<_FP16>(1316), static_cast<_FP16>(950),
                           static_cast<_FP16>(1056), static_cast<_FP16>(1164),
                           static_cast<_FP16>(1274), static_cast<_FP16>(1386),
                           static_cast<_FP16>(1000), static_cast<_FP16>(1111),
                           static_cast<_FP16>(1224), static_cast<_FP16>(1339),
                           static_cast<_FP16>(1456), static_cast<_FP16>(1050),
                           static_cast<_FP16>(1166), static_cast<_FP16>(1284),
                           static_cast<_FP16>(1404), static_cast<_FP16>(1526),
                           static_cast<_FP16>(1100), static_cast<_FP16>(1221),
                           static_cast<_FP16>(1344), static_cast<_FP16>(1469),
                           static_cast<_FP16>(1596), static_cast<_FP16>(1150),
                           static_cast<_FP16>(1276), static_cast<_FP16>(1404),
                           static_cast<_FP16>(1534), static_cast<_FP16>(1666)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 2, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(1),
                           static_cast<_FP16>(4),    static_cast<_FP16>(9),
                           static_cast<_FP16>(16),   static_cast<_FP16>(0),
                           static_cast<_FP16>(6),    static_cast<_FP16>(14),
                           static_cast<_FP16>(24),   static_cast<_FP16>(36),
                           static_cast<_FP16>(0),    static_cast<_FP16>(11),
                           static_cast<_FP16>(24),   static_cast<_FP16>(39),
                           static_cast<_FP16>(56),   static_cast<_FP16>(0),
                           static_cast<_FP16>(16),   static_cast<_FP16>(34),
                           static_cast<_FP16>(54),   static_cast<_FP16>(76),
                           static_cast<_FP16>(100),  static_cast<_FP16>(126),
                           static_cast<_FP16>(154),  static_cast<_FP16>(184),
                           static_cast<_FP16>(216),  static_cast<_FP16>(125),
                           static_cast<_FP16>(156),  static_cast<_FP16>(189),
                           static_cast<_FP16>(224),  static_cast<_FP16>(261),
                           static_cast<_FP16>(150),  static_cast<_FP16>(186),
                           static_cast<_FP16>(224),  static_cast<_FP16>(264),
                           static_cast<_FP16>(306),  static_cast<_FP16>(175),
                           static_cast<_FP16>(216),  static_cast<_FP16>(259),
                           static_cast<_FP16>(304),  static_cast<_FP16>(351),
                           static_cast<_FP16>(0),    static_cast<_FP16>(41),
                           static_cast<_FP16>(84),   static_cast<_FP16>(129),
                           static_cast<_FP16>(176),  static_cast<_FP16>(0),
                           static_cast<_FP16>(46),   static_cast<_FP16>(94),
                           static_cast<_FP16>(144),  static_cast<_FP16>(196),
                           static_cast<_FP16>(0),    static_cast<_FP16>(51),
                           static_cast<_FP16>(104),  static_cast<_FP16>(159),
                           static_cast<_FP16>(216),  static_cast<_FP16>(0),
                           static_cast<_FP16>(56),   static_cast<_FP16>(114),
                           static_cast<_FP16>(174),  static_cast<_FP16>(236),
                           static_cast<_FP16>(300),  static_cast<_FP16>(366),
                           static_cast<_FP16>(434),  static_cast<_FP16>(504),
                           static_cast<_FP16>(576),  static_cast<_FP16>(325),
                           static_cast<_FP16>(396),  static_cast<_FP16>(469),
                           static_cast<_FP16>(544),  static_cast<_FP16>(621),
                           static_cast<_FP16>(350),  static_cast<_FP16>(426),
                           static_cast<_FP16>(504),  static_cast<_FP16>(584),
                           static_cast<_FP16>(666),  static_cast<_FP16>(375),
                           static_cast<_FP16>(456),  static_cast<_FP16>(539),
                           static_cast<_FP16>(624),  static_cast<_FP16>(711),
                           static_cast<_FP16>(0),    static_cast<_FP16>(81),
                           static_cast<_FP16>(164),  static_cast<_FP16>(249),
                           static_cast<_FP16>(336),  static_cast<_FP16>(0),
                           static_cast<_FP16>(86),   static_cast<_FP16>(174),
                           static_cast<_FP16>(264),  static_cast<_FP16>(356),
                           static_cast<_FP16>(0),    static_cast<_FP16>(91),
                           static_cast<_FP16>(184),  static_cast<_FP16>(279),
                           static_cast<_FP16>(376),  static_cast<_FP16>(0),
                           static_cast<_FP16>(96),   static_cast<_FP16>(194),
                           static_cast<_FP16>(294),  static_cast<_FP16>(396),
                           static_cast<_FP16>(500),  static_cast<_FP16>(606),
                           static_cast<_FP16>(714),  static_cast<_FP16>(824),
                           static_cast<_FP16>(936),  static_cast<_FP16>(525),
                           static_cast<_FP16>(636),  static_cast<_FP16>(749),
                           static_cast<_FP16>(864),  static_cast<_FP16>(981),
                           static_cast<_FP16>(550),  static_cast<_FP16>(666),
                           static_cast<_FP16>(784),  static_cast<_FP16>(904),
                           static_cast<_FP16>(1026), static_cast<_FP16>(575),
                           static_cast<_FP16>(696),  static_cast<_FP16>(819),
                           static_cast<_FP16>(944),  static_cast<_FP16>(1071)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(5),
                           static_cast<_FP16>(6),    static_cast<_FP16>(7),
                           static_cast<_FP16>(8),    static_cast<_FP16>(9),
                           static_cast<_FP16>(20),   static_cast<_FP16>(22),
                           static_cast<_FP16>(24),   static_cast<_FP16>(26),
                           static_cast<_FP16>(28),   static_cast<_FP16>(45),
                           static_cast<_FP16>(48),   static_cast<_FP16>(51),
                           static_cast<_FP16>(54),   static_cast<_FP16>(57),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(0),
                           static_cast<_FP16>(0),    static_cast<_FP16>(25),
                           static_cast<_FP16>(26),   static_cast<_FP16>(27),
                           static_cast<_FP16>(28),   static_cast<_FP16>(29),
                           static_cast<_FP16>(60),   static_cast<_FP16>(62),
                           static_cast<_FP16>(64),   static_cast<_FP16>(66),
                           static_cast<_FP16>(68),   static_cast<_FP16>(105),
                           static_cast<_FP16>(108),  static_cast<_FP16>(111),
                           static_cast<_FP16>(114),  static_cast<_FP16>(117),
                           static_cast<_FP16>(160),  static_cast<_FP16>(164),
                           static_cast<_FP16>(168),  static_cast<_FP16>(172),
                           static_cast<_FP16>(176),  static_cast<_FP16>(225),
                           static_cast<_FP16>(230),  static_cast<_FP16>(235),
                           static_cast<_FP16>(240),  static_cast<_FP16>(245),
                           static_cast<_FP16>(300),  static_cast<_FP16>(306),
                           static_cast<_FP16>(312),  static_cast<_FP16>(318),
                           static_cast<_FP16>(324),  static_cast<_FP16>(385),
                           static_cast<_FP16>(392),  static_cast<_FP16>(399),
                           static_cast<_FP16>(406),  static_cast<_FP16>(413),
                           static_cast<_FP16>(240),  static_cast<_FP16>(244),
                           static_cast<_FP16>(248),  static_cast<_FP16>(252),
                           static_cast<_FP16>(256),  static_cast<_FP16>(325),
                           static_cast<_FP16>(330),  static_cast<_FP16>(335),
                           static_cast<_FP16>(340),  static_cast<_FP16>(345),
                           static_cast<_FP16>(420),  static_cast<_FP16>(426),
                           static_cast<_FP16>(432),  static_cast<_FP16>(438),
                           static_cast<_FP16>(444),  static_cast<_FP16>(525),
                           static_cast<_FP16>(532),  static_cast<_FP16>(539),
                           static_cast<_FP16>(546),  static_cast<_FP16>(553),
                           static_cast<_FP16>(640),  static_cast<_FP16>(648),
                           static_cast<_FP16>(656),  static_cast<_FP16>(664),
                           static_cast<_FP16>(672),  static_cast<_FP16>(765),
                           static_cast<_FP16>(774),  static_cast<_FP16>(783),
                           static_cast<_FP16>(792),  static_cast<_FP16>(801),
                           static_cast<_FP16>(900),  static_cast<_FP16>(910),
                           static_cast<_FP16>(920),  static_cast<_FP16>(930),
                           static_cast<_FP16>(940),  static_cast<_FP16>(1045),
                           static_cast<_FP16>(1056), static_cast<_FP16>(1067),
                           static_cast<_FP16>(1078), static_cast<_FP16>(1089),
                           static_cast<_FP16>(800),  static_cast<_FP16>(808),
                           static_cast<_FP16>(816),  static_cast<_FP16>(824),
                           static_cast<_FP16>(832),  static_cast<_FP16>(945),
                           static_cast<_FP16>(954),  static_cast<_FP16>(963),
                           static_cast<_FP16>(972),  static_cast<_FP16>(981),
                           static_cast<_FP16>(1100), static_cast<_FP16>(1110),
                           static_cast<_FP16>(1120), static_cast<_FP16>(1130),
                           static_cast<_FP16>(1140), static_cast<_FP16>(1265),
                           static_cast<_FP16>(1276), static_cast<_FP16>(1287),
                           static_cast<_FP16>(1298), static_cast<_FP16>(1309)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 1, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(4),   static_cast<_FP16>(9),
                           static_cast<_FP16>(16),  static_cast<_FP16>(0),
                           static_cast<_FP16>(6),   static_cast<_FP16>(14),
                           static_cast<_FP16>(24),  static_cast<_FP16>(36),
                           static_cast<_FP16>(0),   static_cast<_FP16>(11),
                           static_cast<_FP16>(24),  static_cast<_FP16>(39),
                           static_cast<_FP16>(56),  static_cast<_FP16>(0),
                           static_cast<_FP16>(16),  static_cast<_FP16>(34),
                           static_cast<_FP16>(54),  static_cast<_FP16>(76),
                           static_cast<_FP16>(0),   static_cast<_FP16>(21),
                           static_cast<_FP16>(44),  static_cast<_FP16>(69),
                           static_cast<_FP16>(96),  static_cast<_FP16>(0),
                           static_cast<_FP16>(26),  static_cast<_FP16>(54),
                           static_cast<_FP16>(84),  static_cast<_FP16>(116),
                           static_cast<_FP16>(0),   static_cast<_FP16>(31),
                           static_cast<_FP16>(64),  static_cast<_FP16>(99),
                           static_cast<_FP16>(136), static_cast<_FP16>(0),
                           static_cast<_FP16>(36),  static_cast<_FP16>(74),
                           static_cast<_FP16>(114), static_cast<_FP16>(156),
                           static_cast<_FP16>(0),   static_cast<_FP16>(41),
                           static_cast<_FP16>(84),  static_cast<_FP16>(129),
                           static_cast<_FP16>(176), static_cast<_FP16>(0),
                           static_cast<_FP16>(46),  static_cast<_FP16>(94),
                           static_cast<_FP16>(144), static_cast<_FP16>(196),
                           static_cast<_FP16>(0),   static_cast<_FP16>(51),
                           static_cast<_FP16>(104), static_cast<_FP16>(159),
                           static_cast<_FP16>(216), static_cast<_FP16>(0),
                           static_cast<_FP16>(56),  static_cast<_FP16>(114),
                           static_cast<_FP16>(174), static_cast<_FP16>(236),
                           static_cast<_FP16>(0),   static_cast<_FP16>(61),
                           static_cast<_FP16>(124), static_cast<_FP16>(189),
                           static_cast<_FP16>(256), static_cast<_FP16>(0),
                           static_cast<_FP16>(66),  static_cast<_FP16>(134),
                           static_cast<_FP16>(204), static_cast<_FP16>(276),
                           static_cast<_FP16>(0),   static_cast<_FP16>(71),
                           static_cast<_FP16>(144), static_cast<_FP16>(219),
                           static_cast<_FP16>(296), static_cast<_FP16>(0),
                           static_cast<_FP16>(76),  static_cast<_FP16>(154),
                           static_cast<_FP16>(234), static_cast<_FP16>(316),
                           static_cast<_FP16>(0),   static_cast<_FP16>(81),
                           static_cast<_FP16>(164), static_cast<_FP16>(249),
                           static_cast<_FP16>(336), static_cast<_FP16>(0),
                           static_cast<_FP16>(86),  static_cast<_FP16>(174),
                           static_cast<_FP16>(264), static_cast<_FP16>(356),
                           static_cast<_FP16>(0),   static_cast<_FP16>(91),
                           static_cast<_FP16>(184), static_cast<_FP16>(279),
                           static_cast<_FP16>(376), static_cast<_FP16>(0),
                           static_cast<_FP16>(96),  static_cast<_FP16>(194),
                           static_cast<_FP16>(294), static_cast<_FP16>(396),
                           static_cast<_FP16>(0),   static_cast<_FP16>(101),
                           static_cast<_FP16>(204), static_cast<_FP16>(309),
                           static_cast<_FP16>(416), static_cast<_FP16>(0),
                           static_cast<_FP16>(106), static_cast<_FP16>(214),
                           static_cast<_FP16>(324), static_cast<_FP16>(436),
                           static_cast<_FP16>(0),   static_cast<_FP16>(111),
                           static_cast<_FP16>(224), static_cast<_FP16>(339),
                           static_cast<_FP16>(456), static_cast<_FP16>(0),
                           static_cast<_FP16>(116), static_cast<_FP16>(234),
                           static_cast<_FP16>(354), static_cast<_FP16>(476)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 2, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(20),  static_cast<_FP16>(21),
                           static_cast<_FP16>(22),  static_cast<_FP16>(23),
                           static_cast<_FP16>(24),  static_cast<_FP16>(25),
                           static_cast<_FP16>(26),  static_cast<_FP16>(27),
                           static_cast<_FP16>(28),  static_cast<_FP16>(29),
                           static_cast<_FP16>(30),  static_cast<_FP16>(31),
                           static_cast<_FP16>(32),  static_cast<_FP16>(33),
                           static_cast<_FP16>(34),  static_cast<_FP16>(35),
                           static_cast<_FP16>(36),  static_cast<_FP16>(37),
                           static_cast<_FP16>(38),  static_cast<_FP16>(39),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(60),  static_cast<_FP16>(61),
                           static_cast<_FP16>(62),  static_cast<_FP16>(63),
                           static_cast<_FP16>(64),  static_cast<_FP16>(65),
                           static_cast<_FP16>(66),  static_cast<_FP16>(67),
                           static_cast<_FP16>(68),  static_cast<_FP16>(69),
                           static_cast<_FP16>(70),  static_cast<_FP16>(71),
                           static_cast<_FP16>(72),  static_cast<_FP16>(73),
                           static_cast<_FP16>(74),  static_cast<_FP16>(75),
                           static_cast<_FP16>(76),  static_cast<_FP16>(77),
                           static_cast<_FP16>(78),  static_cast<_FP16>(79),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(100), static_cast<_FP16>(101),
                           static_cast<_FP16>(102), static_cast<_FP16>(103),
                           static_cast<_FP16>(104), static_cast<_FP16>(105),
                           static_cast<_FP16>(106), static_cast<_FP16>(107),
                           static_cast<_FP16>(108), static_cast<_FP16>(109),
                           static_cast<_FP16>(110), static_cast<_FP16>(111),
                           static_cast<_FP16>(112), static_cast<_FP16>(113),
                           static_cast<_FP16>(114), static_cast<_FP16>(115),
                           static_cast<_FP16>(116), static_cast<_FP16>(117),
                           static_cast<_FP16>(118), static_cast<_FP16>(119)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(0),   static_cast<_FP16>(0),
                           static_cast<_FP16>(40),  static_cast<_FP16>(41),
                           static_cast<_FP16>(42),  static_cast<_FP16>(43),
                           static_cast<_FP16>(44),  static_cast<_FP16>(45),
                           static_cast<_FP16>(46),  static_cast<_FP16>(47),
                           static_cast<_FP16>(48),  static_cast<_FP16>(49),
                           static_cast<_FP16>(50),  static_cast<_FP16>(51),
                           static_cast<_FP16>(52),  static_cast<_FP16>(53),
                           static_cast<_FP16>(54),  static_cast<_FP16>(55),
                           static_cast<_FP16>(56),  static_cast<_FP16>(57),
                           static_cast<_FP16>(58),  static_cast<_FP16>(59),
                           static_cast<_FP16>(60),  static_cast<_FP16>(61),
                           static_cast<_FP16>(62),  static_cast<_FP16>(63),
                           static_cast<_FP16>(64),  static_cast<_FP16>(65),
                           static_cast<_FP16>(66),  static_cast<_FP16>(67),
                           static_cast<_FP16>(68),  static_cast<_FP16>(69),
                           static_cast<_FP16>(70),  static_cast<_FP16>(71),
                           static_cast<_FP16>(72),  static_cast<_FP16>(73),
                           static_cast<_FP16>(74),  static_cast<_FP16>(75),
                           static_cast<_FP16>(76),  static_cast<_FP16>(77),
                           static_cast<_FP16>(78),  static_cast<_FP16>(79),
                           static_cast<_FP16>(160), static_cast<_FP16>(162),
                           static_cast<_FP16>(164), static_cast<_FP16>(166),
                           static_cast<_FP16>(168), static_cast<_FP16>(170),
                           static_cast<_FP16>(172), static_cast<_FP16>(174),
                           static_cast<_FP16>(176), static_cast<_FP16>(178),
                           static_cast<_FP16>(180), static_cast<_FP16>(182),
                           static_cast<_FP16>(184), static_cast<_FP16>(186),
                           static_cast<_FP16>(188), static_cast<_FP16>(190),
                           static_cast<_FP16>(192), static_cast<_FP16>(194),
                           static_cast<_FP16>(196), static_cast<_FP16>(198),
                           static_cast<_FP16>(200), static_cast<_FP16>(202),
                           static_cast<_FP16>(204), static_cast<_FP16>(206),
                           static_cast<_FP16>(208), static_cast<_FP16>(210),
                           static_cast<_FP16>(212), static_cast<_FP16>(214),
                           static_cast<_FP16>(216), static_cast<_FP16>(218),
                           static_cast<_FP16>(220), static_cast<_FP16>(222),
                           static_cast<_FP16>(224), static_cast<_FP16>(226),
                           static_cast<_FP16>(228), static_cast<_FP16>(230),
                           static_cast<_FP16>(232), static_cast<_FP16>(234),
                           static_cast<_FP16>(236), static_cast<_FP16>(238)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 5, 1, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(4),   static_cast<_FP16>(9),
                           static_cast<_FP16>(0),   static_cast<_FP16>(5),
                           static_cast<_FP16>(12),  static_cast<_FP16>(21),
                           static_cast<_FP16>(0),   static_cast<_FP16>(9),
                           static_cast<_FP16>(20),  static_cast<_FP16>(33),
                           static_cast<_FP16>(0),   static_cast<_FP16>(13),
                           static_cast<_FP16>(28),  static_cast<_FP16>(45),
                           static_cast<_FP16>(0),   static_cast<_FP16>(17),
                           static_cast<_FP16>(36),  static_cast<_FP16>(57),
                           static_cast<_FP16>(80),  static_cast<_FP16>(105),
                           static_cast<_FP16>(132), static_cast<_FP16>(161),
                           static_cast<_FP16>(96),  static_cast<_FP16>(125),
                           static_cast<_FP16>(156), static_cast<_FP16>(189),
                           static_cast<_FP16>(112), static_cast<_FP16>(145),
                           static_cast<_FP16>(180), static_cast<_FP16>(217),
                           static_cast<_FP16>(128), static_cast<_FP16>(165),
                           static_cast<_FP16>(204), static_cast<_FP16>(245),
                           static_cast<_FP16>(144), static_cast<_FP16>(185),
                           static_cast<_FP16>(228), static_cast<_FP16>(273),
                           static_cast<_FP16>(320), static_cast<_FP16>(369),
                           static_cast<_FP16>(420), static_cast<_FP16>(473),
                           static_cast<_FP16>(352), static_cast<_FP16>(405),
                           static_cast<_FP16>(460), static_cast<_FP16>(517),
                           static_cast<_FP16>(384), static_cast<_FP16>(441),
                           static_cast<_FP16>(500), static_cast<_FP16>(561),
                           static_cast<_FP16>(416), static_cast<_FP16>(477),
                           static_cast<_FP16>(540), static_cast<_FP16>(605),
                           static_cast<_FP16>(448), static_cast<_FP16>(513),
                           static_cast<_FP16>(580), static_cast<_FP16>(649)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.multiply_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_supported_01_n) {

  nntrainer::Tensor target(3, 1, 3, 1, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::Tensor target2(3, 1, 3, 3, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_i_broadcast_not_broadcastable_02_n) {
  nntrainer::Tensor target(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::Tensor target2(3, 2, 3, 1, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.multiply_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, multiply_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor result = input.multiply(0.0);
  if (result.getValue<_FP16>(0, 0, 1, 1) != 0.0)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.multiply(input);

  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] * indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiply_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, height - 1, width - 1,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);

  EXPECT_THROW({ input.multiply(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(batch, channel, height, 2 * width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.multiply(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.multiply(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.multiply(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiply__Float16_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor expected(batch, channel, height, width,
                             nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(expected, (i * (batch * height) + j * (width) + k + 1) * 2);

  nntrainer::Tensor result = input.multiply(2.0);

  EXPECT_EQ(result, expected);
}

TEST(nntrainer_Tensor, divide_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original;
  original.copy(input);

  status = input.divide_i(2.0f);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = original.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], indata[i] + indata[i]);
  }
}

TEST(nntrainer_Tensor, divide_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  status = input.divide_i(input);
  EXPECT_EQ(status, ML_ERROR_NONE);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(indata[i], _FP16(1.0));
  }
}

TEST(nntrainer_Tensor, divide_i_01_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  status = input.divide_i((_FP16)0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_02_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor original(batch, channel, height - 2, width - 1,
                             nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);

  status = input.divide_i(original);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.divide(1.0);

  _FP16 *previous = input.getData<_FP16>();
  ASSERT_NE(nullptr, previous);
  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width * channel; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i]);
  }
}

TEST(nntrainer_Tensor, divide_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW({ input.divide(0.0); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);

  EXPECT_THROW({ input.divide(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(batch, channel, height, 2 * width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.divide(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.divide(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.divide(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, divide_i_broadcast_01_p) {
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(1, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(41.0),      static_cast<_FP16>(21.0),
      static_cast<_FP16>(14.333333), static_cast<_FP16>(11.0),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(7.6666665),
      static_cast<_FP16>(6.714286),  static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.4444447), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.6363635), static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(4.076923),  static_cast<_FP16>(3.857143),
      static_cast<_FP16>(3.6666667), static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.3529413), static_cast<_FP16>(3.2222223),
      static_cast<_FP16>(3.1052632), static_cast<_FP16>(3.0),
      static_cast<_FP16>(2.9047618), static_cast<_FP16>(2.8181818),
      static_cast<_FP16>(2.7391305), static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.6),       static_cast<_FP16>(2.5384614),
      static_cast<_FP16>(2.4814816), static_cast<_FP16>(2.4285715),
      static_cast<_FP16>(2.3793104), static_cast<_FP16>(2.3333333),
      static_cast<_FP16>(2.2903225), static_cast<_FP16>(2.25),
      static_cast<_FP16>(2.2121212), static_cast<_FP16>(2.1764705),
      static_cast<_FP16>(2.142857),  static_cast<_FP16>(2.1111112),
      static_cast<_FP16>(2.0810812), static_cast<_FP16>(2.0526316),
      static_cast<_FP16>(2.025641),  static_cast<_FP16>(2.0),
      static_cast<_FP16>(81.0),      static_cast<_FP16>(41.0),
      static_cast<_FP16>(27.666666), static_cast<_FP16>(21.0),
      static_cast<_FP16>(17.0),      static_cast<_FP16>(14.333333),
      static_cast<_FP16>(12.428572), static_cast<_FP16>(11.0),
      static_cast<_FP16>(9.888889),  static_cast<_FP16>(9.0),
      static_cast<_FP16>(8.272727),  static_cast<_FP16>(7.6666665),
      static_cast<_FP16>(7.1538463), static_cast<_FP16>(6.714286),
      static_cast<_FP16>(6.3333335), static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.7058825), static_cast<_FP16>(5.4444447),
      static_cast<_FP16>(5.2105265), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.8095236), static_cast<_FP16>(4.6363635),
      static_cast<_FP16>(4.478261),  static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(4.2),       static_cast<_FP16>(4.076923),
      static_cast<_FP16>(3.9629629), static_cast<_FP16>(3.857143),
      static_cast<_FP16>(3.7586207), static_cast<_FP16>(3.6666667),
      static_cast<_FP16>(3.580645),  static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.4242425), static_cast<_FP16>(3.3529413),
      static_cast<_FP16>(3.2857144), static_cast<_FP16>(3.2222223),
      static_cast<_FP16>(3.162162),  static_cast<_FP16>(3.1052632),
      static_cast<_FP16>(3.0512822), static_cast<_FP16>(3.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(3, 1, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(11.0),
      static_cast<_FP16>(7.6666665), static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(3.857143),  static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.2222223), static_cast<_FP16>(3.0),
      static_cast<_FP16>(2.8181818), static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.5384614), static_cast<_FP16>(2.4285715),
      static_cast<_FP16>(2.3333333), static_cast<_FP16>(2.25),
      static_cast<_FP16>(2.1764705), static_cast<_FP16>(2.1111112),
      static_cast<_FP16>(2.0526316), static_cast<_FP16>(2.0),
      static_cast<_FP16>(1.9523809), static_cast<_FP16>(1.9090909),
      static_cast<_FP16>(1.8695652), static_cast<_FP16>(1.8333334),
      static_cast<_FP16>(1.8),       static_cast<_FP16>(1.7692307),
      static_cast<_FP16>(1.7407408), static_cast<_FP16>(1.7142857),
      static_cast<_FP16>(1.6896552), static_cast<_FP16>(1.6666666),
      static_cast<_FP16>(1.6451613), static_cast<_FP16>(1.625),
      static_cast<_FP16>(1.6060606), static_cast<_FP16>(1.5882353),
      static_cast<_FP16>(1.5714285), static_cast<_FP16>(1.5555556),
      static_cast<_FP16>(1.5405406), static_cast<_FP16>(1.5263158),
      static_cast<_FP16>(1.5128205), static_cast<_FP16>(1.5),
      static_cast<_FP16>(2.9047618), static_cast<_FP16>(2.8181818),
      static_cast<_FP16>(2.7391305), static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.6),       static_cast<_FP16>(2.5384614),
      static_cast<_FP16>(2.4814816), static_cast<_FP16>(2.4285715),
      static_cast<_FP16>(2.3793104), static_cast<_FP16>(2.3333333),
      static_cast<_FP16>(2.2903225), static_cast<_FP16>(2.25),
      static_cast<_FP16>(2.2121212), static_cast<_FP16>(2.1764705),
      static_cast<_FP16>(2.142857),  static_cast<_FP16>(2.1111112),
      static_cast<_FP16>(2.0810812), static_cast<_FP16>(2.0526316),
      static_cast<_FP16>(2.025641),  static_cast<_FP16>(2.0),
      static_cast<_FP16>(1.9756098), static_cast<_FP16>(1.9523809),
      static_cast<_FP16>(1.9302325), static_cast<_FP16>(1.9090909),
      static_cast<_FP16>(1.8888888), static_cast<_FP16>(1.8695652),
      static_cast<_FP16>(1.8510638), static_cast<_FP16>(1.8333334),
      static_cast<_FP16>(1.8163265), static_cast<_FP16>(1.8),
      static_cast<_FP16>(1.7843137), static_cast<_FP16>(1.7692307),
      static_cast<_FP16>(1.754717),  static_cast<_FP16>(1.7407408),
      static_cast<_FP16>(1.7272727), static_cast<_FP16>(1.7142857),
      static_cast<_FP16>(1.7017543), static_cast<_FP16>(1.6896552),
      static_cast<_FP16>(1.6779661), static_cast<_FP16>(1.6666666),
      static_cast<_FP16>(2.4634147), static_cast<_FP16>(2.4285715),
      static_cast<_FP16>(2.3953488), static_cast<_FP16>(2.3636363),
      static_cast<_FP16>(2.3333333), static_cast<_FP16>(2.3043478),
      static_cast<_FP16>(2.2765958), static_cast<_FP16>(2.25),
      static_cast<_FP16>(2.2244897), static_cast<_FP16>(2.2),
      static_cast<_FP16>(2.1764705), static_cast<_FP16>(2.1538463),
      static_cast<_FP16>(2.1320755), static_cast<_FP16>(2.1111112),
      static_cast<_FP16>(2.090909),  static_cast<_FP16>(2.0714285),
      static_cast<_FP16>(2.0526316), static_cast<_FP16>(2.0344827),
      static_cast<_FP16>(2.0169492), static_cast<_FP16>(2.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(3, 2, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(2.0),
      static_cast<_FP16>(3.0),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(3.0),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(4.5),       static_cast<_FP16>(5.0),
      static_cast<_FP16>(3.6666667), static_cast<_FP16>(4.0),
      static_cast<_FP16>(4.3333335), static_cast<_FP16>(4.6666665),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(4.25),      static_cast<_FP16>(4.5),
      static_cast<_FP16>(4.75),      static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.2),       static_cast<_FP16>(4.4),
      static_cast<_FP16>(4.6),       static_cast<_FP16>(4.8),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(4.5),       static_cast<_FP16>(4.6666665),
      static_cast<_FP16>(4.8333335), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.428571),  static_cast<_FP16>(4.571429),
      static_cast<_FP16>(4.714286),  static_cast<_FP16>(4.857143),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.5),
      static_cast<_FP16>(4.625),     static_cast<_FP16>(4.75),
      static_cast<_FP16>(4.875),     static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.5555553), static_cast<_FP16>(4.6666665),
      static_cast<_FP16>(4.7777777), static_cast<_FP16>(4.888889),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.6),
      static_cast<_FP16>(4.7),       static_cast<_FP16>(4.8),
      static_cast<_FP16>(4.9),       static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.6363635), static_cast<_FP16>(4.7272725),
      static_cast<_FP16>(4.818182),  static_cast<_FP16>(4.909091),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.6666665),
      static_cast<_FP16>(4.75),      static_cast<_FP16>(4.8333335),
      static_cast<_FP16>(4.9166665), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.6923075), static_cast<_FP16>(4.769231),
      static_cast<_FP16>(4.8461537), static_cast<_FP16>(4.923077),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.714286),
      static_cast<_FP16>(4.785714),  static_cast<_FP16>(4.857143),
      static_cast<_FP16>(4.928571),  static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.733333),  static_cast<_FP16>(4.8),
      static_cast<_FP16>(4.866667),  static_cast<_FP16>(4.9333334),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.75),
      static_cast<_FP16>(4.8125),    static_cast<_FP16>(4.875),
      static_cast<_FP16>(4.9375),    static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.7647057), static_cast<_FP16>(4.8235292),
      static_cast<_FP16>(4.882353),  static_cast<_FP16>(4.9411764),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.7777777),
      static_cast<_FP16>(4.8333335), static_cast<_FP16>(4.888889),
      static_cast<_FP16>(4.9444447), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.7894735), static_cast<_FP16>(4.8421054),
      static_cast<_FP16>(4.894737),  static_cast<_FP16>(4.9473686),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.8),
      static_cast<_FP16>(4.85),      static_cast<_FP16>(4.9),
      static_cast<_FP16>(4.95),      static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.8095236), static_cast<_FP16>(4.857143),
      static_cast<_FP16>(4.904762),  static_cast<_FP16>(4.952381),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.818182),
      static_cast<_FP16>(4.8636365), static_cast<_FP16>(4.909091),
      static_cast<_FP16>(4.9545455), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.826087),  static_cast<_FP16>(4.869565),
      static_cast<_FP16>(4.9130435), static_cast<_FP16>(4.9565215),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.8333335),
      static_cast<_FP16>(4.875),     static_cast<_FP16>(4.9166665),
      static_cast<_FP16>(4.9583335), static_cast<_FP16>(5.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.25),      static_cast<_FP16>(2.0),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(6.0),
      static_cast<_FP16>(4.3333335), static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.0),       static_cast<_FP16>(16.0),
      static_cast<_FP16>(8.5),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(4.75),      static_cast<_FP16>(4.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(11.0),
      static_cast<_FP16>(7.6666665), static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(26.0),
      static_cast<_FP16>(13.5),      static_cast<_FP16>(9.333333),
      static_cast<_FP16>(7.25),      static_cast<_FP16>(6.0),
      static_cast<_FP16>(31.0),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(8.5),
      static_cast<_FP16>(7.0),       static_cast<_FP16>(36.0),
      static_cast<_FP16>(18.5),      static_cast<_FP16>(12.666667),
      static_cast<_FP16>(9.75),      static_cast<_FP16>(8.0),
      static_cast<_FP16>(6.8333335), static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.375),     static_cast<_FP16>(4.888889),
      static_cast<_FP16>(4.5),       static_cast<_FP16>(7.6666665),
      static_cast<_FP16>(6.714286),  static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.4444447), static_cast<_FP16>(5.0),
      static_cast<_FP16>(8.5),       static_cast<_FP16>(7.428571),
      static_cast<_FP16>(6.625),     static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.5),       static_cast<_FP16>(9.333333),
      static_cast<_FP16>(8.142858),  static_cast<_FP16>(7.25),
      static_cast<_FP16>(6.5555553), static_cast<_FP16>(6.0),
      static_cast<_FP16>(10.166667), static_cast<_FP16>(8.857142),
      static_cast<_FP16>(7.875),     static_cast<_FP16>(7.111111),
      static_cast<_FP16>(6.5),       static_cast<_FP16>(11.0),
      static_cast<_FP16>(9.571428),  static_cast<_FP16>(8.5),
      static_cast<_FP16>(7.6666665), static_cast<_FP16>(7.0),
      static_cast<_FP16>(11.833333), static_cast<_FP16>(10.285714),
      static_cast<_FP16>(9.125),     static_cast<_FP16>(8.222222),
      static_cast<_FP16>(7.5),       static_cast<_FP16>(12.666667),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(9.75),
      static_cast<_FP16>(8.777778),  static_cast<_FP16>(8.0),
      static_cast<_FP16>(7.3636365), static_cast<_FP16>(6.8333335),
      static_cast<_FP16>(6.3846154), static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.6666665), static_cast<_FP16>(7.818182),
      static_cast<_FP16>(7.25),      static_cast<_FP16>(6.769231),
      static_cast<_FP16>(6.357143),  static_cast<_FP16>(6.0),
      static_cast<_FP16>(8.272727),  static_cast<_FP16>(7.6666665),
      static_cast<_FP16>(7.1538463), static_cast<_FP16>(6.714286),
      static_cast<_FP16>(6.3333335), static_cast<_FP16>(8.727273),
      static_cast<_FP16>(8.083333),  static_cast<_FP16>(7.5384617),
      static_cast<_FP16>(7.071429),  static_cast<_FP16>(6.6666665),
      static_cast<_FP16>(9.181818),  static_cast<_FP16>(8.5),
      static_cast<_FP16>(7.923077),  static_cast<_FP16>(7.428571),
      static_cast<_FP16>(7.0),       static_cast<_FP16>(9.636364),
      static_cast<_FP16>(8.916667),  static_cast<_FP16>(8.307693),
      static_cast<_FP16>(7.785714),  static_cast<_FP16>(7.3333335),
      static_cast<_FP16>(10.090909), static_cast<_FP16>(9.333333),
      static_cast<_FP16>(8.692307),  static_cast<_FP16>(8.142858),
      static_cast<_FP16>(7.6666665), static_cast<_FP16>(10.545455),
      static_cast<_FP16>(9.75),      static_cast<_FP16>(9.076923),
      static_cast<_FP16>(8.5),       static_cast<_FP16>(8.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(1, 2, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.25),      static_cast<_FP16>(2.0),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(6.0),
      static_cast<_FP16>(4.3333335), static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.0),       static_cast<_FP16>(16.0),
      static_cast<_FP16>(8.5),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(4.75),      static_cast<_FP16>(4.0),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(3.142857),
      static_cast<_FP16>(2.875),     static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.5),       static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(3.857143),  static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.2222223), static_cast<_FP16>(3.0),
      static_cast<_FP16>(5.1666665), static_cast<_FP16>(4.571429),
      static_cast<_FP16>(4.125),     static_cast<_FP16>(3.7777777),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.285714),  static_cast<_FP16>(4.75),
      static_cast<_FP16>(4.3333335), static_cast<_FP16>(4.0),
      static_cast<_FP16>(41.0),      static_cast<_FP16>(21.0),
      static_cast<_FP16>(14.333333), static_cast<_FP16>(11.0),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(46.0),
      static_cast<_FP16>(23.5),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(12.25),     static_cast<_FP16>(10.0),
      static_cast<_FP16>(51.0),      static_cast<_FP16>(26.0),
      static_cast<_FP16>(17.666666), static_cast<_FP16>(13.5),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(56.0),
      static_cast<_FP16>(28.5),      static_cast<_FP16>(19.333334),
      static_cast<_FP16>(14.75),     static_cast<_FP16>(12.0),
      static_cast<_FP16>(10.166667), static_cast<_FP16>(8.857142),
      static_cast<_FP16>(7.875),     static_cast<_FP16>(7.111111),
      static_cast<_FP16>(6.5),       static_cast<_FP16>(11.0),
      static_cast<_FP16>(9.571428),  static_cast<_FP16>(8.5),
      static_cast<_FP16>(7.6666665), static_cast<_FP16>(7.0),
      static_cast<_FP16>(11.833333), static_cast<_FP16>(10.285714),
      static_cast<_FP16>(9.125),     static_cast<_FP16>(8.222222),
      static_cast<_FP16>(7.5),       static_cast<_FP16>(12.666667),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(9.75),
      static_cast<_FP16>(8.777778),  static_cast<_FP16>(8.0),
      static_cast<_FP16>(81.0),      static_cast<_FP16>(41.0),
      static_cast<_FP16>(27.666666), static_cast<_FP16>(21.0),
      static_cast<_FP16>(17.0),      static_cast<_FP16>(86.0),
      static_cast<_FP16>(43.5),      static_cast<_FP16>(29.333334),
      static_cast<_FP16>(22.25),     static_cast<_FP16>(18.0),
      static_cast<_FP16>(91.0),      static_cast<_FP16>(46.0),
      static_cast<_FP16>(31.0),      static_cast<_FP16>(23.5),
      static_cast<_FP16>(19.0),      static_cast<_FP16>(96.0),
      static_cast<_FP16>(48.5),      static_cast<_FP16>(32.666668),
      static_cast<_FP16>(24.75),     static_cast<_FP16>(20.0),
      static_cast<_FP16>(16.833334), static_cast<_FP16>(14.571428),
      static_cast<_FP16>(12.875),    static_cast<_FP16>(11.555555),
      static_cast<_FP16>(10.5),      static_cast<_FP16>(17.666666),
      static_cast<_FP16>(15.285714), static_cast<_FP16>(13.5),
      static_cast<_FP16>(12.111111), static_cast<_FP16>(11.0),
      static_cast<_FP16>(18.5),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(14.125),    static_cast<_FP16>(12.666667),
      static_cast<_FP16>(11.5),      static_cast<_FP16>(19.333334),
      static_cast<_FP16>(16.714285), static_cast<_FP16>(14.75),
      static_cast<_FP16>(13.222222), static_cast<_FP16>(12.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(3, 1, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(2.0),
      static_cast<_FP16>(3.0),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(3.0),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(4.5),       static_cast<_FP16>(5.0),
      static_cast<_FP16>(3.6666667), static_cast<_FP16>(4.0),
      static_cast<_FP16>(4.3333335), static_cast<_FP16>(4.6666665),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(4.25),      static_cast<_FP16>(4.5),
      static_cast<_FP16>(4.75),      static_cast<_FP16>(5.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(22.0),
      static_cast<_FP16>(23.0),      static_cast<_FP16>(24.0),
      static_cast<_FP16>(25.0),      static_cast<_FP16>(13.0),
      static_cast<_FP16>(13.5),      static_cast<_FP16>(14.0),
      static_cast<_FP16>(14.5),      static_cast<_FP16>(15.0),
      static_cast<_FP16>(10.333333), static_cast<_FP16>(10.666667),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(11.333333),
      static_cast<_FP16>(11.666667), static_cast<_FP16>(9.0),
      static_cast<_FP16>(9.25),      static_cast<_FP16>(9.5),
      static_cast<_FP16>(9.75),      static_cast<_FP16>(10.0),
      static_cast<_FP16>(8.2),       static_cast<_FP16>(8.4),
      static_cast<_FP16>(8.6),       static_cast<_FP16>(8.8),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(7.6666665),
      static_cast<_FP16>(7.8333335), static_cast<_FP16>(8.0),
      static_cast<_FP16>(8.166667),  static_cast<_FP16>(8.333333),
      static_cast<_FP16>(7.285714),  static_cast<_FP16>(7.428571),
      static_cast<_FP16>(7.571429),  static_cast<_FP16>(7.714286),
      static_cast<_FP16>(7.857143),  static_cast<_FP16>(7.0),
      static_cast<_FP16>(7.125),     static_cast<_FP16>(7.25),
      static_cast<_FP16>(7.375),     static_cast<_FP16>(7.5),
      static_cast<_FP16>(12.2),      static_cast<_FP16>(12.4),
      static_cast<_FP16>(12.6),      static_cast<_FP16>(12.8),
      static_cast<_FP16>(13.0),      static_cast<_FP16>(11.0),
      static_cast<_FP16>(11.166667), static_cast<_FP16>(11.333333),
      static_cast<_FP16>(11.5),      static_cast<_FP16>(11.666667),
      static_cast<_FP16>(10.142858), static_cast<_FP16>(10.285714),
      static_cast<_FP16>(10.428572), static_cast<_FP16>(10.571428),
      static_cast<_FP16>(10.714286), static_cast<_FP16>(9.5),
      static_cast<_FP16>(9.625),     static_cast<_FP16>(9.75),
      static_cast<_FP16>(9.875),     static_cast<_FP16>(10.0),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(9.111111),
      static_cast<_FP16>(9.222222),  static_cast<_FP16>(9.333333),
      static_cast<_FP16>(9.444445),  static_cast<_FP16>(8.6),
      static_cast<_FP16>(8.7),       static_cast<_FP16>(8.8),
      static_cast<_FP16>(8.9),       static_cast<_FP16>(9.0),
      static_cast<_FP16>(8.272727),  static_cast<_FP16>(8.363636),
      static_cast<_FP16>(8.454545),  static_cast<_FP16>(8.545455),
      static_cast<_FP16>(8.636364),  static_cast<_FP16>(8.0),
      static_cast<_FP16>(8.083333),  static_cast<_FP16>(8.166667),
      static_cast<_FP16>(8.25),      static_cast<_FP16>(8.333333),
      static_cast<_FP16>(11.222222), static_cast<_FP16>(11.333333),
      static_cast<_FP16>(11.444445), static_cast<_FP16>(11.555555),
      static_cast<_FP16>(11.666667), static_cast<_FP16>(10.6),
      static_cast<_FP16>(10.7),      static_cast<_FP16>(10.8),
      static_cast<_FP16>(10.9),      static_cast<_FP16>(11.0),
      static_cast<_FP16>(10.090909), static_cast<_FP16>(10.181818),
      static_cast<_FP16>(10.272727), static_cast<_FP16>(10.363636),
      static_cast<_FP16>(10.454545), static_cast<_FP16>(9.666667),
      static_cast<_FP16>(9.75),      static_cast<_FP16>(9.833333),
      static_cast<_FP16>(9.916667),  static_cast<_FP16>(10.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(1, 1, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(3.5),       static_cast<_FP16>(2.6666667),
      static_cast<_FP16>(2.25),      static_cast<_FP16>(2.0),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(6.0),
      static_cast<_FP16>(4.3333335), static_cast<_FP16>(3.5),
      static_cast<_FP16>(3.0),       static_cast<_FP16>(16.0),
      static_cast<_FP16>(8.5),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(4.75),      static_cast<_FP16>(4.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(11.0),
      static_cast<_FP16>(7.6666665), static_cast<_FP16>(6.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(26.0),
      static_cast<_FP16>(13.5),      static_cast<_FP16>(9.333333),
      static_cast<_FP16>(7.25),      static_cast<_FP16>(6.0),
      static_cast<_FP16>(31.0),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(8.5),
      static_cast<_FP16>(7.0),       static_cast<_FP16>(36.0),
      static_cast<_FP16>(18.5),      static_cast<_FP16>(12.666667),
      static_cast<_FP16>(9.75),      static_cast<_FP16>(8.0),
      static_cast<_FP16>(41.0),      static_cast<_FP16>(21.0),
      static_cast<_FP16>(14.333333), static_cast<_FP16>(11.0),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(46.0),
      static_cast<_FP16>(23.5),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(12.25),     static_cast<_FP16>(10.0),
      static_cast<_FP16>(51.0),      static_cast<_FP16>(26.0),
      static_cast<_FP16>(17.666666), static_cast<_FP16>(13.5),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(56.0),
      static_cast<_FP16>(28.5),      static_cast<_FP16>(19.333334),
      static_cast<_FP16>(14.75),     static_cast<_FP16>(12.0),
      static_cast<_FP16>(61.0),      static_cast<_FP16>(31.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(13.0),      static_cast<_FP16>(66.0),
      static_cast<_FP16>(33.5),      static_cast<_FP16>(22.666666),
      static_cast<_FP16>(17.25),     static_cast<_FP16>(14.0),
      static_cast<_FP16>(71.0),      static_cast<_FP16>(36.0),
      static_cast<_FP16>(24.333334), static_cast<_FP16>(18.5),
      static_cast<_FP16>(15.0),      static_cast<_FP16>(76.0),
      static_cast<_FP16>(38.5),      static_cast<_FP16>(26.0),
      static_cast<_FP16>(19.75),     static_cast<_FP16>(16.0),
      static_cast<_FP16>(81.0),      static_cast<_FP16>(41.0),
      static_cast<_FP16>(27.666666), static_cast<_FP16>(21.0),
      static_cast<_FP16>(17.0),      static_cast<_FP16>(86.0),
      static_cast<_FP16>(43.5),      static_cast<_FP16>(29.333334),
      static_cast<_FP16>(22.25),     static_cast<_FP16>(18.0),
      static_cast<_FP16>(91.0),      static_cast<_FP16>(46.0),
      static_cast<_FP16>(31.0),      static_cast<_FP16>(23.5),
      static_cast<_FP16>(19.0),      static_cast<_FP16>(96.0),
      static_cast<_FP16>(48.5),      static_cast<_FP16>(32.666668),
      static_cast<_FP16>(24.75),     static_cast<_FP16>(20.0),
      static_cast<_FP16>(101.0),     static_cast<_FP16>(51.0),
      static_cast<_FP16>(34.333332), static_cast<_FP16>(26.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(106.0),
      static_cast<_FP16>(53.5),      static_cast<_FP16>(36.0),
      static_cast<_FP16>(27.25),     static_cast<_FP16>(22.0),
      static_cast<_FP16>(111.0),     static_cast<_FP16>(56.0),
      static_cast<_FP16>(37.666668), static_cast<_FP16>(28.5),
      static_cast<_FP16>(23.0),      static_cast<_FP16>(116.0),
      static_cast<_FP16>(58.5),      static_cast<_FP16>(39.333332),
      static_cast<_FP16>(29.75),     static_cast<_FP16>(24.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(1, 2, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {static_cast<_FP16>(1.0),  static_cast<_FP16>(2.0),
                           static_cast<_FP16>(3.0),  static_cast<_FP16>(4.0),
                           static_cast<_FP16>(5.0),  static_cast<_FP16>(6.0),
                           static_cast<_FP16>(7.0),  static_cast<_FP16>(8.0),
                           static_cast<_FP16>(9.0),  static_cast<_FP16>(10.0),
                           static_cast<_FP16>(11.0), static_cast<_FP16>(12.0),
                           static_cast<_FP16>(13.0), static_cast<_FP16>(14.0),
                           static_cast<_FP16>(15.0), static_cast<_FP16>(16.0),
                           static_cast<_FP16>(17.0), static_cast<_FP16>(18.0),
                           static_cast<_FP16>(19.0), static_cast<_FP16>(20.0),
                           static_cast<_FP16>(10.5), static_cast<_FP16>(11.0),
                           static_cast<_FP16>(11.5), static_cast<_FP16>(12.0),
                           static_cast<_FP16>(12.5), static_cast<_FP16>(13.0),
                           static_cast<_FP16>(13.5), static_cast<_FP16>(14.0),
                           static_cast<_FP16>(14.5), static_cast<_FP16>(15.0),
                           static_cast<_FP16>(15.5), static_cast<_FP16>(16.0),
                           static_cast<_FP16>(16.5), static_cast<_FP16>(17.0),
                           static_cast<_FP16>(17.5), static_cast<_FP16>(18.0),
                           static_cast<_FP16>(18.5), static_cast<_FP16>(19.0),
                           static_cast<_FP16>(19.5), static_cast<_FP16>(20.0),
                           static_cast<_FP16>(41.0), static_cast<_FP16>(42.0),
                           static_cast<_FP16>(43.0), static_cast<_FP16>(44.0),
                           static_cast<_FP16>(45.0), static_cast<_FP16>(46.0),
                           static_cast<_FP16>(47.0), static_cast<_FP16>(48.0),
                           static_cast<_FP16>(49.0), static_cast<_FP16>(50.0),
                           static_cast<_FP16>(51.0), static_cast<_FP16>(52.0),
                           static_cast<_FP16>(53.0), static_cast<_FP16>(54.0),
                           static_cast<_FP16>(55.0), static_cast<_FP16>(56.0),
                           static_cast<_FP16>(57.0), static_cast<_FP16>(58.0),
                           static_cast<_FP16>(59.0), static_cast<_FP16>(60.0),
                           static_cast<_FP16>(30.5), static_cast<_FP16>(31.0),
                           static_cast<_FP16>(31.5), static_cast<_FP16>(32.0),
                           static_cast<_FP16>(32.5), static_cast<_FP16>(33.0),
                           static_cast<_FP16>(33.5), static_cast<_FP16>(34.0),
                           static_cast<_FP16>(34.5), static_cast<_FP16>(35.0),
                           static_cast<_FP16>(35.5), static_cast<_FP16>(36.0),
                           static_cast<_FP16>(36.5), static_cast<_FP16>(37.0),
                           static_cast<_FP16>(37.5), static_cast<_FP16>(38.0),
                           static_cast<_FP16>(38.5), static_cast<_FP16>(39.0),
                           static_cast<_FP16>(39.5), static_cast<_FP16>(40.0),
                           static_cast<_FP16>(81.0), static_cast<_FP16>(82.0),
                           static_cast<_FP16>(83.0), static_cast<_FP16>(84.0),
                           static_cast<_FP16>(85.0), static_cast<_FP16>(86.0),
                           static_cast<_FP16>(87.0), static_cast<_FP16>(88.0),
                           static_cast<_FP16>(89.0), static_cast<_FP16>(90.0),
                           static_cast<_FP16>(91.0), static_cast<_FP16>(92.0),
                           static_cast<_FP16>(93.0), static_cast<_FP16>(94.0),
                           static_cast<_FP16>(95.0), static_cast<_FP16>(96.0),
                           static_cast<_FP16>(97.0), static_cast<_FP16>(98.0),
                           static_cast<_FP16>(99.0), static_cast<_FP16>(100.0),
                           static_cast<_FP16>(50.5), static_cast<_FP16>(51.0),
                           static_cast<_FP16>(51.5), static_cast<_FP16>(52.0),
                           static_cast<_FP16>(52.5), static_cast<_FP16>(53.0),
                           static_cast<_FP16>(53.5), static_cast<_FP16>(54.0),
                           static_cast<_FP16>(54.5), static_cast<_FP16>(55.0),
                           static_cast<_FP16>(55.5), static_cast<_FP16>(56.0),
                           static_cast<_FP16>(56.5), static_cast<_FP16>(57.0),
                           static_cast<_FP16>(57.5), static_cast<_FP16>(58.0),
                           static_cast<_FP16>(58.5), static_cast<_FP16>(59.0),
                           static_cast<_FP16>(59.5), static_cast<_FP16>(60.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(2.0),
      static_cast<_FP16>(3.0),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(6.0),
      static_cast<_FP16>(7.0),       static_cast<_FP16>(8.0),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(10.0),
      static_cast<_FP16>(11.0),      static_cast<_FP16>(12.0),
      static_cast<_FP16>(13.0),      static_cast<_FP16>(14.0),
      static_cast<_FP16>(15.0),      static_cast<_FP16>(16.0),
      static_cast<_FP16>(17.0),      static_cast<_FP16>(18.0),
      static_cast<_FP16>(19.0),      static_cast<_FP16>(20.0),
      static_cast<_FP16>(21.0),      static_cast<_FP16>(22.0),
      static_cast<_FP16>(23.0),      static_cast<_FP16>(24.0),
      static_cast<_FP16>(25.0),      static_cast<_FP16>(26.0),
      static_cast<_FP16>(27.0),      static_cast<_FP16>(28.0),
      static_cast<_FP16>(29.0),      static_cast<_FP16>(30.0),
      static_cast<_FP16>(31.0),      static_cast<_FP16>(32.0),
      static_cast<_FP16>(33.0),      static_cast<_FP16>(34.0),
      static_cast<_FP16>(35.0),      static_cast<_FP16>(36.0),
      static_cast<_FP16>(37.0),      static_cast<_FP16>(38.0),
      static_cast<_FP16>(39.0),      static_cast<_FP16>(40.0),
      static_cast<_FP16>(20.5),      static_cast<_FP16>(21.0),
      static_cast<_FP16>(21.5),      static_cast<_FP16>(22.0),
      static_cast<_FP16>(22.5),      static_cast<_FP16>(23.0),
      static_cast<_FP16>(23.5),      static_cast<_FP16>(24.0),
      static_cast<_FP16>(24.5),      static_cast<_FP16>(25.0),
      static_cast<_FP16>(25.5),      static_cast<_FP16>(26.0),
      static_cast<_FP16>(26.5),      static_cast<_FP16>(27.0),
      static_cast<_FP16>(27.5),      static_cast<_FP16>(28.0),
      static_cast<_FP16>(28.5),      static_cast<_FP16>(29.0),
      static_cast<_FP16>(29.5),      static_cast<_FP16>(30.0),
      static_cast<_FP16>(30.5),      static_cast<_FP16>(31.0),
      static_cast<_FP16>(31.5),      static_cast<_FP16>(32.0),
      static_cast<_FP16>(32.5),      static_cast<_FP16>(33.0),
      static_cast<_FP16>(33.5),      static_cast<_FP16>(34.0),
      static_cast<_FP16>(34.5),      static_cast<_FP16>(35.0),
      static_cast<_FP16>(35.5),      static_cast<_FP16>(36.0),
      static_cast<_FP16>(36.5),      static_cast<_FP16>(37.0),
      static_cast<_FP16>(37.5),      static_cast<_FP16>(38.0),
      static_cast<_FP16>(38.5),      static_cast<_FP16>(39.0),
      static_cast<_FP16>(39.5),      static_cast<_FP16>(40.0),
      static_cast<_FP16>(27.0),      static_cast<_FP16>(27.333334),
      static_cast<_FP16>(27.666666), static_cast<_FP16>(28.0),
      static_cast<_FP16>(28.333334), static_cast<_FP16>(28.666666),
      static_cast<_FP16>(29.0),      static_cast<_FP16>(29.333334),
      static_cast<_FP16>(29.666666), static_cast<_FP16>(30.0),
      static_cast<_FP16>(30.333334), static_cast<_FP16>(30.666666),
      static_cast<_FP16>(31.0),      static_cast<_FP16>(31.333334),
      static_cast<_FP16>(31.666666), static_cast<_FP16>(32.0),
      static_cast<_FP16>(32.333332), static_cast<_FP16>(32.666668),
      static_cast<_FP16>(33.0),      static_cast<_FP16>(33.333332),
      static_cast<_FP16>(33.666668), static_cast<_FP16>(34.0),
      static_cast<_FP16>(34.333332), static_cast<_FP16>(34.666668),
      static_cast<_FP16>(35.0),      static_cast<_FP16>(35.333332),
      static_cast<_FP16>(35.666668), static_cast<_FP16>(36.0),
      static_cast<_FP16>(36.333332), static_cast<_FP16>(36.666668),
      static_cast<_FP16>(37.0),      static_cast<_FP16>(37.333332),
      static_cast<_FP16>(37.666668), static_cast<_FP16>(38.0),
      static_cast<_FP16>(38.333332), static_cast<_FP16>(38.666668),
      static_cast<_FP16>(39.0),      static_cast<_FP16>(39.333332),
      static_cast<_FP16>(39.666668), static_cast<_FP16>(40.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 5, 1, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    t.add_i(1);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1);
    _FP16 answer_data[] = {
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(1.0),       static_cast<_FP16>(1.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(3.0),
      static_cast<_FP16>(2.3333333), static_cast<_FP16>(2.0),
      static_cast<_FP16>(9.0),       static_cast<_FP16>(5.0),
      static_cast<_FP16>(3.6666667), static_cast<_FP16>(3.0),
      static_cast<_FP16>(13.0),      static_cast<_FP16>(7.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.0),
      static_cast<_FP16>(17.0),      static_cast<_FP16>(9.0),
      static_cast<_FP16>(6.3333335), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.2),       static_cast<_FP16>(3.6666667),
      static_cast<_FP16>(3.2857144), static_cast<_FP16>(3.0),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(3.857143),  static_cast<_FP16>(3.5),
      static_cast<_FP16>(5.8),       static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.428571),  static_cast<_FP16>(4.0),
      static_cast<_FP16>(6.6),       static_cast<_FP16>(5.6666665),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.5),
      static_cast<_FP16>(7.4),       static_cast<_FP16>(6.3333335),
      static_cast<_FP16>(5.571429),  static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.5555553), static_cast<_FP16>(4.2),
      static_cast<_FP16>(3.909091),  static_cast<_FP16>(3.6666667),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.6),
      static_cast<_FP16>(4.2727275), static_cast<_FP16>(4.0),
      static_cast<_FP16>(5.4444447), static_cast<_FP16>(5.0),
      static_cast<_FP16>(4.6363635), static_cast<_FP16>(4.3333335),
      static_cast<_FP16>(5.888889),  static_cast<_FP16>(5.4),
      static_cast<_FP16>(5.0),       static_cast<_FP16>(4.6666665),
      static_cast<_FP16>(6.3333335), static_cast<_FP16>(5.8),
      static_cast<_FP16>(5.3636365), static_cast<_FP16>(5.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.divide_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_supported_01_n) {
  nntrainer::Tensor target(3, 1, 3, 1, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::Tensor target2(3, 1, 3, 3, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, divide_i_broadcast_not_broadcastable_02_n) {
  nntrainer::Tensor target(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::Tensor target2(3, 2, 3, 1, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.divide_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  nntrainer::Tensor original(batch, channel, height, width,
                             nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  original.copy(target);

  status = target.add_i((_FP16)2.1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *previous = original.getData<_FP16>();
  ASSERT_NE(nullptr, previous);
  _FP16 *data = target.getData<_FP16>();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], (_FP16)(previous[i] + (_FP16)2.1));
  }
}

TEST(nntrainer_Tensor, add_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor original(batch, height, width, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  original.copy(target);

  status = target.add_i(target, 3.0);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *previous = original.getData<_FP16>();
  ASSERT_NE(nullptr, previous);
  _FP16 *data = target.getData<_FP16>();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], previous[i] * 4.0);
  }
}

// /**
//  * @brief operand dimension is not right
//  */
TEST(nntrainer_Tensor, add_i_01_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor target2(batch, height - 2, width - 3,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  status = target.add_i(target2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_i_broadcast_01_p) {
  nntrainer::TensorDim ref_dim(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                               nntrainer::Tdatatype::FP16);
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(2),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(8),   static_cast<_FP16>(10),
                           static_cast<_FP16>(12),  static_cast<_FP16>(14),
                           static_cast<_FP16>(16),  static_cast<_FP16>(18),
                           static_cast<_FP16>(20),  static_cast<_FP16>(22),
                           static_cast<_FP16>(24),  static_cast<_FP16>(26),
                           static_cast<_FP16>(28),  static_cast<_FP16>(30),
                           static_cast<_FP16>(32),  static_cast<_FP16>(34),
                           static_cast<_FP16>(36),  static_cast<_FP16>(38),
                           static_cast<_FP16>(40),  static_cast<_FP16>(42),
                           static_cast<_FP16>(44),  static_cast<_FP16>(46),
                           static_cast<_FP16>(48),  static_cast<_FP16>(50),
                           static_cast<_FP16>(52),  static_cast<_FP16>(54),
                           static_cast<_FP16>(56),  static_cast<_FP16>(58),
                           static_cast<_FP16>(60),  static_cast<_FP16>(62),
                           static_cast<_FP16>(64),  static_cast<_FP16>(66),
                           static_cast<_FP16>(68),  static_cast<_FP16>(70),
                           static_cast<_FP16>(72),  static_cast<_FP16>(74),
                           static_cast<_FP16>(76),  static_cast<_FP16>(78),
                           static_cast<_FP16>(40),  static_cast<_FP16>(42),
                           static_cast<_FP16>(44),  static_cast<_FP16>(46),
                           static_cast<_FP16>(48),  static_cast<_FP16>(50),
                           static_cast<_FP16>(52),  static_cast<_FP16>(54),
                           static_cast<_FP16>(56),  static_cast<_FP16>(58),
                           static_cast<_FP16>(60),  static_cast<_FP16>(62),
                           static_cast<_FP16>(64),  static_cast<_FP16>(66),
                           static_cast<_FP16>(68),  static_cast<_FP16>(70),
                           static_cast<_FP16>(72),  static_cast<_FP16>(74),
                           static_cast<_FP16>(76),  static_cast<_FP16>(78),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(90),
                           static_cast<_FP16>(92),  static_cast<_FP16>(94),
                           static_cast<_FP16>(96),  static_cast<_FP16>(98),
                           static_cast<_FP16>(100), static_cast<_FP16>(102),
                           static_cast<_FP16>(104), static_cast<_FP16>(106),
                           static_cast<_FP16>(108), static_cast<_FP16>(110),
                           static_cast<_FP16>(112), static_cast<_FP16>(114),
                           static_cast<_FP16>(116), static_cast<_FP16>(118),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(90),
                           static_cast<_FP16>(92),  static_cast<_FP16>(94),
                           static_cast<_FP16>(96),  static_cast<_FP16>(98),
                           static_cast<_FP16>(100), static_cast<_FP16>(102),
                           static_cast<_FP16>(104), static_cast<_FP16>(106),
                           static_cast<_FP16>(108), static_cast<_FP16>(110),
                           static_cast<_FP16>(112), static_cast<_FP16>(114),
                           static_cast<_FP16>(116), static_cast<_FP16>(118),
                           static_cast<_FP16>(120), static_cast<_FP16>(122),
                           static_cast<_FP16>(124), static_cast<_FP16>(126),
                           static_cast<_FP16>(128), static_cast<_FP16>(130),
                           static_cast<_FP16>(132), static_cast<_FP16>(134),
                           static_cast<_FP16>(136), static_cast<_FP16>(138),
                           static_cast<_FP16>(140), static_cast<_FP16>(142),
                           static_cast<_FP16>(144), static_cast<_FP16>(146),
                           static_cast<_FP16>(148), static_cast<_FP16>(150),
                           static_cast<_FP16>(152), static_cast<_FP16>(154),
                           static_cast<_FP16>(156), static_cast<_FP16>(158)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(2),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(8),   static_cast<_FP16>(10),
                           static_cast<_FP16>(12),  static_cast<_FP16>(14),
                           static_cast<_FP16>(16),  static_cast<_FP16>(18),
                           static_cast<_FP16>(20),  static_cast<_FP16>(22),
                           static_cast<_FP16>(24),  static_cast<_FP16>(26),
                           static_cast<_FP16>(28),  static_cast<_FP16>(30),
                           static_cast<_FP16>(32),  static_cast<_FP16>(34),
                           static_cast<_FP16>(36),  static_cast<_FP16>(38),
                           static_cast<_FP16>(20),  static_cast<_FP16>(22),
                           static_cast<_FP16>(24),  static_cast<_FP16>(26),
                           static_cast<_FP16>(28),  static_cast<_FP16>(30),
                           static_cast<_FP16>(32),  static_cast<_FP16>(34),
                           static_cast<_FP16>(36),  static_cast<_FP16>(38),
                           static_cast<_FP16>(40),  static_cast<_FP16>(42),
                           static_cast<_FP16>(44),  static_cast<_FP16>(46),
                           static_cast<_FP16>(48),  static_cast<_FP16>(50),
                           static_cast<_FP16>(52),  static_cast<_FP16>(54),
                           static_cast<_FP16>(56),  static_cast<_FP16>(58),
                           static_cast<_FP16>(60),  static_cast<_FP16>(62),
                           static_cast<_FP16>(64),  static_cast<_FP16>(66),
                           static_cast<_FP16>(68),  static_cast<_FP16>(70),
                           static_cast<_FP16>(72),  static_cast<_FP16>(74),
                           static_cast<_FP16>(76),  static_cast<_FP16>(78),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(90),
                           static_cast<_FP16>(92),  static_cast<_FP16>(94),
                           static_cast<_FP16>(96),  static_cast<_FP16>(98),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(90),
                           static_cast<_FP16>(92),  static_cast<_FP16>(94),
                           static_cast<_FP16>(96),  static_cast<_FP16>(98),
                           static_cast<_FP16>(100), static_cast<_FP16>(102),
                           static_cast<_FP16>(104), static_cast<_FP16>(106),
                           static_cast<_FP16>(108), static_cast<_FP16>(110),
                           static_cast<_FP16>(112), static_cast<_FP16>(114),
                           static_cast<_FP16>(116), static_cast<_FP16>(118),
                           static_cast<_FP16>(120), static_cast<_FP16>(122),
                           static_cast<_FP16>(124), static_cast<_FP16>(126),
                           static_cast<_FP16>(128), static_cast<_FP16>(130),
                           static_cast<_FP16>(132), static_cast<_FP16>(134),
                           static_cast<_FP16>(136), static_cast<_FP16>(138),
                           static_cast<_FP16>(140), static_cast<_FP16>(142),
                           static_cast<_FP16>(144), static_cast<_FP16>(146),
                           static_cast<_FP16>(148), static_cast<_FP16>(150),
                           static_cast<_FP16>(152), static_cast<_FP16>(154),
                           static_cast<_FP16>(156), static_cast<_FP16>(158),
                           static_cast<_FP16>(140), static_cast<_FP16>(142),
                           static_cast<_FP16>(144), static_cast<_FP16>(146),
                           static_cast<_FP16>(148), static_cast<_FP16>(150),
                           static_cast<_FP16>(152), static_cast<_FP16>(154),
                           static_cast<_FP16>(156), static_cast<_FP16>(158),
                           static_cast<_FP16>(160), static_cast<_FP16>(162),
                           static_cast<_FP16>(164), static_cast<_FP16>(166),
                           static_cast<_FP16>(168), static_cast<_FP16>(170),
                           static_cast<_FP16>(172), static_cast<_FP16>(174),
                           static_cast<_FP16>(176), static_cast<_FP16>(178)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 2, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(2),   static_cast<_FP16>(3),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(7),   static_cast<_FP16>(8),
                           static_cast<_FP16>(9),   static_cast<_FP16>(10),
                           static_cast<_FP16>(12),  static_cast<_FP16>(13),
                           static_cast<_FP16>(14),  static_cast<_FP16>(15),
                           static_cast<_FP16>(16),  static_cast<_FP16>(18),
                           static_cast<_FP16>(19),  static_cast<_FP16>(20),
                           static_cast<_FP16>(21),  static_cast<_FP16>(22),
                           static_cast<_FP16>(24),  static_cast<_FP16>(25),
                           static_cast<_FP16>(26),  static_cast<_FP16>(27),
                           static_cast<_FP16>(28),  static_cast<_FP16>(30),
                           static_cast<_FP16>(31),  static_cast<_FP16>(32),
                           static_cast<_FP16>(33),  static_cast<_FP16>(34),
                           static_cast<_FP16>(36),  static_cast<_FP16>(37),
                           static_cast<_FP16>(38),  static_cast<_FP16>(39),
                           static_cast<_FP16>(40),  static_cast<_FP16>(42),
                           static_cast<_FP16>(43),  static_cast<_FP16>(44),
                           static_cast<_FP16>(45),  static_cast<_FP16>(46),
                           static_cast<_FP16>(48),  static_cast<_FP16>(49),
                           static_cast<_FP16>(50),  static_cast<_FP16>(51),
                           static_cast<_FP16>(52),  static_cast<_FP16>(54),
                           static_cast<_FP16>(55),  static_cast<_FP16>(56),
                           static_cast<_FP16>(57),  static_cast<_FP16>(58),
                           static_cast<_FP16>(60),  static_cast<_FP16>(61),
                           static_cast<_FP16>(62),  static_cast<_FP16>(63),
                           static_cast<_FP16>(64),  static_cast<_FP16>(66),
                           static_cast<_FP16>(67),  static_cast<_FP16>(68),
                           static_cast<_FP16>(69),  static_cast<_FP16>(70),
                           static_cast<_FP16>(72),  static_cast<_FP16>(73),
                           static_cast<_FP16>(74),  static_cast<_FP16>(75),
                           static_cast<_FP16>(76),  static_cast<_FP16>(78),
                           static_cast<_FP16>(79),  static_cast<_FP16>(80),
                           static_cast<_FP16>(81),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(85),
                           static_cast<_FP16>(86),  static_cast<_FP16>(87),
                           static_cast<_FP16>(88),  static_cast<_FP16>(90),
                           static_cast<_FP16>(91),  static_cast<_FP16>(92),
                           static_cast<_FP16>(93),  static_cast<_FP16>(94),
                           static_cast<_FP16>(96),  static_cast<_FP16>(97),
                           static_cast<_FP16>(98),  static_cast<_FP16>(99),
                           static_cast<_FP16>(100), static_cast<_FP16>(102),
                           static_cast<_FP16>(103), static_cast<_FP16>(104),
                           static_cast<_FP16>(105), static_cast<_FP16>(106),
                           static_cast<_FP16>(108), static_cast<_FP16>(109),
                           static_cast<_FP16>(110), static_cast<_FP16>(111),
                           static_cast<_FP16>(112), static_cast<_FP16>(114),
                           static_cast<_FP16>(115), static_cast<_FP16>(116),
                           static_cast<_FP16>(117), static_cast<_FP16>(118),
                           static_cast<_FP16>(120), static_cast<_FP16>(121),
                           static_cast<_FP16>(122), static_cast<_FP16>(123),
                           static_cast<_FP16>(124), static_cast<_FP16>(126),
                           static_cast<_FP16>(127), static_cast<_FP16>(128),
                           static_cast<_FP16>(129), static_cast<_FP16>(130),
                           static_cast<_FP16>(132), static_cast<_FP16>(133),
                           static_cast<_FP16>(134), static_cast<_FP16>(135),
                           static_cast<_FP16>(136), static_cast<_FP16>(138),
                           static_cast<_FP16>(139), static_cast<_FP16>(140),
                           static_cast<_FP16>(141), static_cast<_FP16>(142)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(2),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(8),   static_cast<_FP16>(5),
                           static_cast<_FP16>(7),   static_cast<_FP16>(9),
                           static_cast<_FP16>(11),  static_cast<_FP16>(13),
                           static_cast<_FP16>(10),  static_cast<_FP16>(12),
                           static_cast<_FP16>(14),  static_cast<_FP16>(16),
                           static_cast<_FP16>(18),  static_cast<_FP16>(15),
                           static_cast<_FP16>(17),  static_cast<_FP16>(19),
                           static_cast<_FP16>(21),  static_cast<_FP16>(23),
                           static_cast<_FP16>(20),  static_cast<_FP16>(22),
                           static_cast<_FP16>(24),  static_cast<_FP16>(26),
                           static_cast<_FP16>(28),  static_cast<_FP16>(25),
                           static_cast<_FP16>(27),  static_cast<_FP16>(29),
                           static_cast<_FP16>(31),  static_cast<_FP16>(33),
                           static_cast<_FP16>(30),  static_cast<_FP16>(32),
                           static_cast<_FP16>(34),  static_cast<_FP16>(36),
                           static_cast<_FP16>(38),  static_cast<_FP16>(35),
                           static_cast<_FP16>(37),  static_cast<_FP16>(39),
                           static_cast<_FP16>(41),  static_cast<_FP16>(43),
                           static_cast<_FP16>(45),  static_cast<_FP16>(47),
                           static_cast<_FP16>(49),  static_cast<_FP16>(51),
                           static_cast<_FP16>(53),  static_cast<_FP16>(50),
                           static_cast<_FP16>(52),  static_cast<_FP16>(54),
                           static_cast<_FP16>(56),  static_cast<_FP16>(58),
                           static_cast<_FP16>(55),  static_cast<_FP16>(57),
                           static_cast<_FP16>(59),  static_cast<_FP16>(61),
                           static_cast<_FP16>(63),  static_cast<_FP16>(60),
                           static_cast<_FP16>(62),  static_cast<_FP16>(64),
                           static_cast<_FP16>(66),  static_cast<_FP16>(68),
                           static_cast<_FP16>(65),  static_cast<_FP16>(67),
                           static_cast<_FP16>(69),  static_cast<_FP16>(71),
                           static_cast<_FP16>(73),  static_cast<_FP16>(70),
                           static_cast<_FP16>(72),  static_cast<_FP16>(74),
                           static_cast<_FP16>(76),  static_cast<_FP16>(78),
                           static_cast<_FP16>(75),  static_cast<_FP16>(77),
                           static_cast<_FP16>(79),  static_cast<_FP16>(81),
                           static_cast<_FP16>(83),  static_cast<_FP16>(80),
                           static_cast<_FP16>(82),  static_cast<_FP16>(84),
                           static_cast<_FP16>(86),  static_cast<_FP16>(88),
                           static_cast<_FP16>(90),  static_cast<_FP16>(92),
                           static_cast<_FP16>(94),  static_cast<_FP16>(96),
                           static_cast<_FP16>(98),  static_cast<_FP16>(95),
                           static_cast<_FP16>(97),  static_cast<_FP16>(99),
                           static_cast<_FP16>(101), static_cast<_FP16>(103),
                           static_cast<_FP16>(100), static_cast<_FP16>(102),
                           static_cast<_FP16>(104), static_cast<_FP16>(106),
                           static_cast<_FP16>(108), static_cast<_FP16>(105),
                           static_cast<_FP16>(107), static_cast<_FP16>(109),
                           static_cast<_FP16>(111), static_cast<_FP16>(113),
                           static_cast<_FP16>(110), static_cast<_FP16>(112),
                           static_cast<_FP16>(114), static_cast<_FP16>(116),
                           static_cast<_FP16>(118), static_cast<_FP16>(115),
                           static_cast<_FP16>(117), static_cast<_FP16>(119),
                           static_cast<_FP16>(121), static_cast<_FP16>(123),
                           static_cast<_FP16>(120), static_cast<_FP16>(122),
                           static_cast<_FP16>(124), static_cast<_FP16>(126),
                           static_cast<_FP16>(128), static_cast<_FP16>(125),
                           static_cast<_FP16>(127), static_cast<_FP16>(129),
                           static_cast<_FP16>(131), static_cast<_FP16>(133)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 2, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(2),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(8),   static_cast<_FP16>(5),
                           static_cast<_FP16>(7),   static_cast<_FP16>(9),
                           static_cast<_FP16>(11),  static_cast<_FP16>(13),
                           static_cast<_FP16>(10),  static_cast<_FP16>(12),
                           static_cast<_FP16>(14),  static_cast<_FP16>(16),
                           static_cast<_FP16>(18),  static_cast<_FP16>(15),
                           static_cast<_FP16>(17),  static_cast<_FP16>(19),
                           static_cast<_FP16>(21),  static_cast<_FP16>(23),
                           static_cast<_FP16>(25),  static_cast<_FP16>(27),
                           static_cast<_FP16>(29),  static_cast<_FP16>(31),
                           static_cast<_FP16>(33),  static_cast<_FP16>(30),
                           static_cast<_FP16>(32),  static_cast<_FP16>(34),
                           static_cast<_FP16>(36),  static_cast<_FP16>(38),
                           static_cast<_FP16>(35),  static_cast<_FP16>(37),
                           static_cast<_FP16>(39),  static_cast<_FP16>(41),
                           static_cast<_FP16>(43),  static_cast<_FP16>(40),
                           static_cast<_FP16>(42),  static_cast<_FP16>(44),
                           static_cast<_FP16>(46),  static_cast<_FP16>(48),
                           static_cast<_FP16>(40),  static_cast<_FP16>(42),
                           static_cast<_FP16>(44),  static_cast<_FP16>(46),
                           static_cast<_FP16>(48),  static_cast<_FP16>(45),
                           static_cast<_FP16>(47),  static_cast<_FP16>(49),
                           static_cast<_FP16>(51),  static_cast<_FP16>(53),
                           static_cast<_FP16>(50),  static_cast<_FP16>(52),
                           static_cast<_FP16>(54),  static_cast<_FP16>(56),
                           static_cast<_FP16>(58),  static_cast<_FP16>(55),
                           static_cast<_FP16>(57),  static_cast<_FP16>(59),
                           static_cast<_FP16>(61),  static_cast<_FP16>(63),
                           static_cast<_FP16>(65),  static_cast<_FP16>(67),
                           static_cast<_FP16>(69),  static_cast<_FP16>(71),
                           static_cast<_FP16>(73),  static_cast<_FP16>(70),
                           static_cast<_FP16>(72),  static_cast<_FP16>(74),
                           static_cast<_FP16>(76),  static_cast<_FP16>(78),
                           static_cast<_FP16>(75),  static_cast<_FP16>(77),
                           static_cast<_FP16>(79),  static_cast<_FP16>(81),
                           static_cast<_FP16>(83),  static_cast<_FP16>(80),
                           static_cast<_FP16>(82),  static_cast<_FP16>(84),
                           static_cast<_FP16>(86),  static_cast<_FP16>(88),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(85),
                           static_cast<_FP16>(87),  static_cast<_FP16>(89),
                           static_cast<_FP16>(91),  static_cast<_FP16>(93),
                           static_cast<_FP16>(90),  static_cast<_FP16>(92),
                           static_cast<_FP16>(94),  static_cast<_FP16>(96),
                           static_cast<_FP16>(98),  static_cast<_FP16>(95),
                           static_cast<_FP16>(97),  static_cast<_FP16>(99),
                           static_cast<_FP16>(101), static_cast<_FP16>(103),
                           static_cast<_FP16>(105), static_cast<_FP16>(107),
                           static_cast<_FP16>(109), static_cast<_FP16>(111),
                           static_cast<_FP16>(113), static_cast<_FP16>(110),
                           static_cast<_FP16>(112), static_cast<_FP16>(114),
                           static_cast<_FP16>(116), static_cast<_FP16>(118),
                           static_cast<_FP16>(115), static_cast<_FP16>(117),
                           static_cast<_FP16>(119), static_cast<_FP16>(121),
                           static_cast<_FP16>(123), static_cast<_FP16>(120),
                           static_cast<_FP16>(122), static_cast<_FP16>(124),
                           static_cast<_FP16>(126), static_cast<_FP16>(128)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(2),   static_cast<_FP16>(3),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(7),   static_cast<_FP16>(8),
                           static_cast<_FP16>(9),   static_cast<_FP16>(10),
                           static_cast<_FP16>(12),  static_cast<_FP16>(13),
                           static_cast<_FP16>(14),  static_cast<_FP16>(15),
                           static_cast<_FP16>(16),  static_cast<_FP16>(18),
                           static_cast<_FP16>(19),  static_cast<_FP16>(20),
                           static_cast<_FP16>(21),  static_cast<_FP16>(22),
                           static_cast<_FP16>(20),  static_cast<_FP16>(21),
                           static_cast<_FP16>(22),  static_cast<_FP16>(23),
                           static_cast<_FP16>(24),  static_cast<_FP16>(26),
                           static_cast<_FP16>(27),  static_cast<_FP16>(28),
                           static_cast<_FP16>(29),  static_cast<_FP16>(30),
                           static_cast<_FP16>(32),  static_cast<_FP16>(33),
                           static_cast<_FP16>(34),  static_cast<_FP16>(35),
                           static_cast<_FP16>(36),  static_cast<_FP16>(38),
                           static_cast<_FP16>(39),  static_cast<_FP16>(40),
                           static_cast<_FP16>(41),  static_cast<_FP16>(42),
                           static_cast<_FP16>(44),  static_cast<_FP16>(45),
                           static_cast<_FP16>(46),  static_cast<_FP16>(47),
                           static_cast<_FP16>(48),  static_cast<_FP16>(50),
                           static_cast<_FP16>(51),  static_cast<_FP16>(52),
                           static_cast<_FP16>(53),  static_cast<_FP16>(54),
                           static_cast<_FP16>(56),  static_cast<_FP16>(57),
                           static_cast<_FP16>(58),  static_cast<_FP16>(59),
                           static_cast<_FP16>(60),  static_cast<_FP16>(62),
                           static_cast<_FP16>(63),  static_cast<_FP16>(64),
                           static_cast<_FP16>(65),  static_cast<_FP16>(66),
                           static_cast<_FP16>(64),  static_cast<_FP16>(65),
                           static_cast<_FP16>(66),  static_cast<_FP16>(67),
                           static_cast<_FP16>(68),  static_cast<_FP16>(70),
                           static_cast<_FP16>(71),  static_cast<_FP16>(72),
                           static_cast<_FP16>(73),  static_cast<_FP16>(74),
                           static_cast<_FP16>(76),  static_cast<_FP16>(77),
                           static_cast<_FP16>(78),  static_cast<_FP16>(79),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(83),  static_cast<_FP16>(84),
                           static_cast<_FP16>(85),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(89),
                           static_cast<_FP16>(90),  static_cast<_FP16>(91),
                           static_cast<_FP16>(92),  static_cast<_FP16>(94),
                           static_cast<_FP16>(95),  static_cast<_FP16>(96),
                           static_cast<_FP16>(97),  static_cast<_FP16>(98),
                           static_cast<_FP16>(100), static_cast<_FP16>(101),
                           static_cast<_FP16>(102), static_cast<_FP16>(103),
                           static_cast<_FP16>(104), static_cast<_FP16>(106),
                           static_cast<_FP16>(107), static_cast<_FP16>(108),
                           static_cast<_FP16>(109), static_cast<_FP16>(110),
                           static_cast<_FP16>(108), static_cast<_FP16>(109),
                           static_cast<_FP16>(110), static_cast<_FP16>(111),
                           static_cast<_FP16>(112), static_cast<_FP16>(114),
                           static_cast<_FP16>(115), static_cast<_FP16>(116),
                           static_cast<_FP16>(117), static_cast<_FP16>(118),
                           static_cast<_FP16>(120), static_cast<_FP16>(121),
                           static_cast<_FP16>(122), static_cast<_FP16>(123),
                           static_cast<_FP16>(124), static_cast<_FP16>(126),
                           static_cast<_FP16>(127), static_cast<_FP16>(128),
                           static_cast<_FP16>(129), static_cast<_FP16>(130)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 1, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(2),
                           static_cast<_FP16>(4),   static_cast<_FP16>(6),
                           static_cast<_FP16>(8),   static_cast<_FP16>(5),
                           static_cast<_FP16>(7),   static_cast<_FP16>(9),
                           static_cast<_FP16>(11),  static_cast<_FP16>(13),
                           static_cast<_FP16>(10),  static_cast<_FP16>(12),
                           static_cast<_FP16>(14),  static_cast<_FP16>(16),
                           static_cast<_FP16>(18),  static_cast<_FP16>(15),
                           static_cast<_FP16>(17),  static_cast<_FP16>(19),
                           static_cast<_FP16>(21),  static_cast<_FP16>(23),
                           static_cast<_FP16>(20),  static_cast<_FP16>(22),
                           static_cast<_FP16>(24),  static_cast<_FP16>(26),
                           static_cast<_FP16>(28),  static_cast<_FP16>(25),
                           static_cast<_FP16>(27),  static_cast<_FP16>(29),
                           static_cast<_FP16>(31),  static_cast<_FP16>(33),
                           static_cast<_FP16>(30),  static_cast<_FP16>(32),
                           static_cast<_FP16>(34),  static_cast<_FP16>(36),
                           static_cast<_FP16>(38),  static_cast<_FP16>(35),
                           static_cast<_FP16>(37),  static_cast<_FP16>(39),
                           static_cast<_FP16>(41),  static_cast<_FP16>(43),
                           static_cast<_FP16>(40),  static_cast<_FP16>(42),
                           static_cast<_FP16>(44),  static_cast<_FP16>(46),
                           static_cast<_FP16>(48),  static_cast<_FP16>(45),
                           static_cast<_FP16>(47),  static_cast<_FP16>(49),
                           static_cast<_FP16>(51),  static_cast<_FP16>(53),
                           static_cast<_FP16>(50),  static_cast<_FP16>(52),
                           static_cast<_FP16>(54),  static_cast<_FP16>(56),
                           static_cast<_FP16>(58),  static_cast<_FP16>(55),
                           static_cast<_FP16>(57),  static_cast<_FP16>(59),
                           static_cast<_FP16>(61),  static_cast<_FP16>(63),
                           static_cast<_FP16>(60),  static_cast<_FP16>(62),
                           static_cast<_FP16>(64),  static_cast<_FP16>(66),
                           static_cast<_FP16>(68),  static_cast<_FP16>(65),
                           static_cast<_FP16>(67),  static_cast<_FP16>(69),
                           static_cast<_FP16>(71),  static_cast<_FP16>(73),
                           static_cast<_FP16>(70),  static_cast<_FP16>(72),
                           static_cast<_FP16>(74),  static_cast<_FP16>(76),
                           static_cast<_FP16>(78),  static_cast<_FP16>(75),
                           static_cast<_FP16>(77),  static_cast<_FP16>(79),
                           static_cast<_FP16>(81),  static_cast<_FP16>(83),
                           static_cast<_FP16>(80),  static_cast<_FP16>(82),
                           static_cast<_FP16>(84),  static_cast<_FP16>(86),
                           static_cast<_FP16>(88),  static_cast<_FP16>(85),
                           static_cast<_FP16>(87),  static_cast<_FP16>(89),
                           static_cast<_FP16>(91),  static_cast<_FP16>(93),
                           static_cast<_FP16>(90),  static_cast<_FP16>(92),
                           static_cast<_FP16>(94),  static_cast<_FP16>(96),
                           static_cast<_FP16>(98),  static_cast<_FP16>(95),
                           static_cast<_FP16>(97),  static_cast<_FP16>(99),
                           static_cast<_FP16>(101), static_cast<_FP16>(103),
                           static_cast<_FP16>(100), static_cast<_FP16>(102),
                           static_cast<_FP16>(104), static_cast<_FP16>(106),
                           static_cast<_FP16>(108), static_cast<_FP16>(105),
                           static_cast<_FP16>(107), static_cast<_FP16>(109),
                           static_cast<_FP16>(111), static_cast<_FP16>(113),
                           static_cast<_FP16>(110), static_cast<_FP16>(112),
                           static_cast<_FP16>(114), static_cast<_FP16>(116),
                           static_cast<_FP16>(118), static_cast<_FP16>(115),
                           static_cast<_FP16>(117), static_cast<_FP16>(119),
                           static_cast<_FP16>(121), static_cast<_FP16>(123)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 2, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(2),   static_cast<_FP16>(3),
                           static_cast<_FP16>(4),   static_cast<_FP16>(5),
                           static_cast<_FP16>(6),   static_cast<_FP16>(7),
                           static_cast<_FP16>(8),   static_cast<_FP16>(9),
                           static_cast<_FP16>(10),  static_cast<_FP16>(11),
                           static_cast<_FP16>(12),  static_cast<_FP16>(13),
                           static_cast<_FP16>(14),  static_cast<_FP16>(15),
                           static_cast<_FP16>(16),  static_cast<_FP16>(17),
                           static_cast<_FP16>(18),  static_cast<_FP16>(19),
                           static_cast<_FP16>(21),  static_cast<_FP16>(22),
                           static_cast<_FP16>(23),  static_cast<_FP16>(24),
                           static_cast<_FP16>(25),  static_cast<_FP16>(26),
                           static_cast<_FP16>(27),  static_cast<_FP16>(28),
                           static_cast<_FP16>(29),  static_cast<_FP16>(30),
                           static_cast<_FP16>(31),  static_cast<_FP16>(32),
                           static_cast<_FP16>(33),  static_cast<_FP16>(34),
                           static_cast<_FP16>(35),  static_cast<_FP16>(36),
                           static_cast<_FP16>(37),  static_cast<_FP16>(38),
                           static_cast<_FP16>(39),  static_cast<_FP16>(40),
                           static_cast<_FP16>(40),  static_cast<_FP16>(41),
                           static_cast<_FP16>(42),  static_cast<_FP16>(43),
                           static_cast<_FP16>(44),  static_cast<_FP16>(45),
                           static_cast<_FP16>(46),  static_cast<_FP16>(47),
                           static_cast<_FP16>(48),  static_cast<_FP16>(49),
                           static_cast<_FP16>(50),  static_cast<_FP16>(51),
                           static_cast<_FP16>(52),  static_cast<_FP16>(53),
                           static_cast<_FP16>(54),  static_cast<_FP16>(55),
                           static_cast<_FP16>(56),  static_cast<_FP16>(57),
                           static_cast<_FP16>(58),  static_cast<_FP16>(59),
                           static_cast<_FP16>(61),  static_cast<_FP16>(62),
                           static_cast<_FP16>(63),  static_cast<_FP16>(64),
                           static_cast<_FP16>(65),  static_cast<_FP16>(66),
                           static_cast<_FP16>(67),  static_cast<_FP16>(68),
                           static_cast<_FP16>(69),  static_cast<_FP16>(70),
                           static_cast<_FP16>(71),  static_cast<_FP16>(72),
                           static_cast<_FP16>(73),  static_cast<_FP16>(74),
                           static_cast<_FP16>(75),  static_cast<_FP16>(76),
                           static_cast<_FP16>(77),  static_cast<_FP16>(78),
                           static_cast<_FP16>(79),  static_cast<_FP16>(80),
                           static_cast<_FP16>(80),  static_cast<_FP16>(81),
                           static_cast<_FP16>(82),  static_cast<_FP16>(83),
                           static_cast<_FP16>(84),  static_cast<_FP16>(85),
                           static_cast<_FP16>(86),  static_cast<_FP16>(87),
                           static_cast<_FP16>(88),  static_cast<_FP16>(89),
                           static_cast<_FP16>(90),  static_cast<_FP16>(91),
                           static_cast<_FP16>(92),  static_cast<_FP16>(93),
                           static_cast<_FP16>(94),  static_cast<_FP16>(95),
                           static_cast<_FP16>(96),  static_cast<_FP16>(97),
                           static_cast<_FP16>(98),  static_cast<_FP16>(99),
                           static_cast<_FP16>(101), static_cast<_FP16>(102),
                           static_cast<_FP16>(103), static_cast<_FP16>(104),
                           static_cast<_FP16>(105), static_cast<_FP16>(106),
                           static_cast<_FP16>(107), static_cast<_FP16>(108),
                           static_cast<_FP16>(109), static_cast<_FP16>(110),
                           static_cast<_FP16>(111), static_cast<_FP16>(112),
                           static_cast<_FP16>(113), static_cast<_FP16>(114),
                           static_cast<_FP16>(115), static_cast<_FP16>(116),
                           static_cast<_FP16>(117), static_cast<_FP16>(118),
                           static_cast<_FP16>(119), static_cast<_FP16>(120)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0),   static_cast<_FP16>(1),
                           static_cast<_FP16>(2),   static_cast<_FP16>(3),
                           static_cast<_FP16>(4),   static_cast<_FP16>(5),
                           static_cast<_FP16>(6),   static_cast<_FP16>(7),
                           static_cast<_FP16>(8),   static_cast<_FP16>(9),
                           static_cast<_FP16>(10),  static_cast<_FP16>(11),
                           static_cast<_FP16>(12),  static_cast<_FP16>(13),
                           static_cast<_FP16>(14),  static_cast<_FP16>(15),
                           static_cast<_FP16>(16),  static_cast<_FP16>(17),
                           static_cast<_FP16>(18),  static_cast<_FP16>(19),
                           static_cast<_FP16>(20),  static_cast<_FP16>(21),
                           static_cast<_FP16>(22),  static_cast<_FP16>(23),
                           static_cast<_FP16>(24),  static_cast<_FP16>(25),
                           static_cast<_FP16>(26),  static_cast<_FP16>(27),
                           static_cast<_FP16>(28),  static_cast<_FP16>(29),
                           static_cast<_FP16>(30),  static_cast<_FP16>(31),
                           static_cast<_FP16>(32),  static_cast<_FP16>(33),
                           static_cast<_FP16>(34),  static_cast<_FP16>(35),
                           static_cast<_FP16>(36),  static_cast<_FP16>(37),
                           static_cast<_FP16>(38),  static_cast<_FP16>(39),
                           static_cast<_FP16>(41),  static_cast<_FP16>(42),
                           static_cast<_FP16>(43),  static_cast<_FP16>(44),
                           static_cast<_FP16>(45),  static_cast<_FP16>(46),
                           static_cast<_FP16>(47),  static_cast<_FP16>(48),
                           static_cast<_FP16>(49),  static_cast<_FP16>(50),
                           static_cast<_FP16>(51),  static_cast<_FP16>(52),
                           static_cast<_FP16>(53),  static_cast<_FP16>(54),
                           static_cast<_FP16>(55),  static_cast<_FP16>(56),
                           static_cast<_FP16>(57),  static_cast<_FP16>(58),
                           static_cast<_FP16>(59),  static_cast<_FP16>(60),
                           static_cast<_FP16>(61),  static_cast<_FP16>(62),
                           static_cast<_FP16>(63),  static_cast<_FP16>(64),
                           static_cast<_FP16>(65),  static_cast<_FP16>(66),
                           static_cast<_FP16>(67),  static_cast<_FP16>(68),
                           static_cast<_FP16>(69),  static_cast<_FP16>(70),
                           static_cast<_FP16>(71),  static_cast<_FP16>(72),
                           static_cast<_FP16>(73),  static_cast<_FP16>(74),
                           static_cast<_FP16>(75),  static_cast<_FP16>(76),
                           static_cast<_FP16>(77),  static_cast<_FP16>(78),
                           static_cast<_FP16>(79),  static_cast<_FP16>(80),
                           static_cast<_FP16>(82),  static_cast<_FP16>(83),
                           static_cast<_FP16>(84),  static_cast<_FP16>(85),
                           static_cast<_FP16>(86),  static_cast<_FP16>(87),
                           static_cast<_FP16>(88),  static_cast<_FP16>(89),
                           static_cast<_FP16>(90),  static_cast<_FP16>(91),
                           static_cast<_FP16>(92),  static_cast<_FP16>(93),
                           static_cast<_FP16>(94),  static_cast<_FP16>(95),
                           static_cast<_FP16>(96),  static_cast<_FP16>(97),
                           static_cast<_FP16>(98),  static_cast<_FP16>(99),
                           static_cast<_FP16>(100), static_cast<_FP16>(101),
                           static_cast<_FP16>(102), static_cast<_FP16>(103),
                           static_cast<_FP16>(104), static_cast<_FP16>(105),
                           static_cast<_FP16>(106), static_cast<_FP16>(107),
                           static_cast<_FP16>(108), static_cast<_FP16>(109),
                           static_cast<_FP16>(110), static_cast<_FP16>(111),
                           static_cast<_FP16>(112), static_cast<_FP16>(113),
                           static_cast<_FP16>(114), static_cast<_FP16>(115),
                           static_cast<_FP16>(116), static_cast<_FP16>(117),
                           static_cast<_FP16>(118), static_cast<_FP16>(119),
                           static_cast<_FP16>(120), static_cast<_FP16>(121)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    m.add_i(1.0);
    _FP16 answer_data[] = {static_cast<_FP16>(1),   static_cast<_FP16>(2),
                           static_cast<_FP16>(3),   static_cast<_FP16>(4),
                           static_cast<_FP16>(5),   static_cast<_FP16>(6),
                           static_cast<_FP16>(7),   static_cast<_FP16>(8),
                           static_cast<_FP16>(9),   static_cast<_FP16>(10),
                           static_cast<_FP16>(11),  static_cast<_FP16>(12),
                           static_cast<_FP16>(13),  static_cast<_FP16>(14),
                           static_cast<_FP16>(15),  static_cast<_FP16>(16),
                           static_cast<_FP16>(17),  static_cast<_FP16>(18),
                           static_cast<_FP16>(19),  static_cast<_FP16>(20),
                           static_cast<_FP16>(21),  static_cast<_FP16>(22),
                           static_cast<_FP16>(23),  static_cast<_FP16>(24),
                           static_cast<_FP16>(25),  static_cast<_FP16>(26),
                           static_cast<_FP16>(27),  static_cast<_FP16>(28),
                           static_cast<_FP16>(29),  static_cast<_FP16>(30),
                           static_cast<_FP16>(31),  static_cast<_FP16>(32),
                           static_cast<_FP16>(33),  static_cast<_FP16>(34),
                           static_cast<_FP16>(35),  static_cast<_FP16>(36),
                           static_cast<_FP16>(37),  static_cast<_FP16>(38),
                           static_cast<_FP16>(39),  static_cast<_FP16>(40),
                           static_cast<_FP16>(41),  static_cast<_FP16>(42),
                           static_cast<_FP16>(43),  static_cast<_FP16>(44),
                           static_cast<_FP16>(45),  static_cast<_FP16>(46),
                           static_cast<_FP16>(47),  static_cast<_FP16>(48),
                           static_cast<_FP16>(49),  static_cast<_FP16>(50),
                           static_cast<_FP16>(51),  static_cast<_FP16>(52),
                           static_cast<_FP16>(53),  static_cast<_FP16>(54),
                           static_cast<_FP16>(55),  static_cast<_FP16>(56),
                           static_cast<_FP16>(57),  static_cast<_FP16>(58),
                           static_cast<_FP16>(59),  static_cast<_FP16>(60),
                           static_cast<_FP16>(61),  static_cast<_FP16>(62),
                           static_cast<_FP16>(63),  static_cast<_FP16>(64),
                           static_cast<_FP16>(65),  static_cast<_FP16>(66),
                           static_cast<_FP16>(67),  static_cast<_FP16>(68),
                           static_cast<_FP16>(69),  static_cast<_FP16>(70),
                           static_cast<_FP16>(71),  static_cast<_FP16>(72),
                           static_cast<_FP16>(73),  static_cast<_FP16>(74),
                           static_cast<_FP16>(75),  static_cast<_FP16>(76),
                           static_cast<_FP16>(77),  static_cast<_FP16>(78),
                           static_cast<_FP16>(79),  static_cast<_FP16>(80),
                           static_cast<_FP16>(81),  static_cast<_FP16>(82),
                           static_cast<_FP16>(83),  static_cast<_FP16>(84),
                           static_cast<_FP16>(85),  static_cast<_FP16>(86),
                           static_cast<_FP16>(87),  static_cast<_FP16>(88),
                           static_cast<_FP16>(89),  static_cast<_FP16>(90),
                           static_cast<_FP16>(91),  static_cast<_FP16>(92),
                           static_cast<_FP16>(93),  static_cast<_FP16>(94),
                           static_cast<_FP16>(95),  static_cast<_FP16>(96),
                           static_cast<_FP16>(97),  static_cast<_FP16>(98),
                           static_cast<_FP16>(99),  static_cast<_FP16>(100),
                           static_cast<_FP16>(101), static_cast<_FP16>(102),
                           static_cast<_FP16>(103), static_cast<_FP16>(104),
                           static_cast<_FP16>(105), static_cast<_FP16>(106),
                           static_cast<_FP16>(107), static_cast<_FP16>(108),
                           static_cast<_FP16>(109), static_cast<_FP16>(110),
                           static_cast<_FP16>(111), static_cast<_FP16>(112),
                           static_cast<_FP16>(113), static_cast<_FP16>(114),
                           static_cast<_FP16>(115), static_cast<_FP16>(116),
                           static_cast<_FP16>(117), static_cast<_FP16>(118),
                           static_cast<_FP16>(119), static_cast<_FP16>(120)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 5, 1, 4, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(3, 5, 1, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 1, 1, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      static_cast<_FP16>(0),  static_cast<_FP16>(2),  static_cast<_FP16>(4),
      static_cast<_FP16>(6),  static_cast<_FP16>(4),  static_cast<_FP16>(6),
      static_cast<_FP16>(8),  static_cast<_FP16>(10), static_cast<_FP16>(8),
      static_cast<_FP16>(10), static_cast<_FP16>(12), static_cast<_FP16>(14),
      static_cast<_FP16>(12), static_cast<_FP16>(14), static_cast<_FP16>(16),
      static_cast<_FP16>(18), static_cast<_FP16>(16), static_cast<_FP16>(18),
      static_cast<_FP16>(20), static_cast<_FP16>(22), static_cast<_FP16>(24),
      static_cast<_FP16>(26), static_cast<_FP16>(28), static_cast<_FP16>(30),
      static_cast<_FP16>(28), static_cast<_FP16>(30), static_cast<_FP16>(32),
      static_cast<_FP16>(34), static_cast<_FP16>(32), static_cast<_FP16>(34),
      static_cast<_FP16>(36), static_cast<_FP16>(38), static_cast<_FP16>(36),
      static_cast<_FP16>(38), static_cast<_FP16>(40), static_cast<_FP16>(42),
      static_cast<_FP16>(40), static_cast<_FP16>(42), static_cast<_FP16>(44),
      static_cast<_FP16>(46), static_cast<_FP16>(48), static_cast<_FP16>(50),
      static_cast<_FP16>(52), static_cast<_FP16>(54), static_cast<_FP16>(52),
      static_cast<_FP16>(54), static_cast<_FP16>(56), static_cast<_FP16>(58),
      static_cast<_FP16>(56), static_cast<_FP16>(58), static_cast<_FP16>(60),
      static_cast<_FP16>(62), static_cast<_FP16>(60), static_cast<_FP16>(62),
      static_cast<_FP16>(64), static_cast<_FP16>(66), static_cast<_FP16>(64),
      static_cast<_FP16>(66), static_cast<_FP16>(68), static_cast<_FP16>(70)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 1, 2, 1, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(1, 1, 2, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 1, 2, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0.0), static_cast<_FP16>(2.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
  {
    nntrainer::TensorDim ref_dim(16, 1, 1, 1, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
    nntrainer::Tensor t =
      ranged(16, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(1, 1, 1, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {static_cast<_FP16>(0.0),  static_cast<_FP16>(1.0),
                           static_cast<_FP16>(2.0),  static_cast<_FP16>(3.0),
                           static_cast<_FP16>(4.0),  static_cast<_FP16>(5.0),
                           static_cast<_FP16>(6.0),  static_cast<_FP16>(7.0),
                           static_cast<_FP16>(8.0),  static_cast<_FP16>(9.0),
                           static_cast<_FP16>(10.0), static_cast<_FP16>(11.0),
                           static_cast<_FP16>(12.0), static_cast<_FP16>(13.0),
                           static_cast<_FP16>(14.0), static_cast<_FP16>(15.0)};
    nntrainer::Tensor answer(ref_dim, answer_data);
    int status = t.add_i(m);
    EXPECT_EQ(status, ML_ERROR_NONE);
    EXPECT_EQ(t, answer);
  }
}

TEST(nntrainer_Tensor, add_i_broadcast_not_supported_01_n) {
  nntrainer::Tensor target(3, 1, 3, 1, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::Tensor target2(3, 1, 3, 3, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_i_broadcast_not_broadcastable_02_n) {
  nntrainer::Tensor target(3, 2, 4, 5, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  nntrainer::Tensor target2(3, 2, 3, 1, nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  EXPECT_EQ(target.add_i(target2), ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, add_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(1.0);

  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != (_FP16)(indata[i] + (_FP16)1.0)) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.add(input);

  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] + indata[i]) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, add_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);

  EXPECT_THROW({ input.add(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, add_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(batch, channel, height, 2 * width);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.add(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.add(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, add_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.add(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, pow_01_p) {

  nntrainer::Tensor input = constant(4.0, 3, 2, 4, 5, nntrainer::Tformat::NCHW,
                                     nntrainer::Tdatatype::FP16);

  nntrainer::Tensor actual, expected;

  actual = input.pow(0.5f);
  expected = constant(2.0, 3, 2, 4, 5, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = input.pow(2.0f);
  expected = constant(16.0, 3, 2, 4, 5, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = input.pow(-0.5f);
  expected = constant(0.5, 3, 2, 4, 5, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);
}

// TEST(nntrainer_Tensor, erf_01_p) {
//   int batch = 1;
//   int channel = 1;
//   int height = 2;
//   int width = 2;

//   nntrainer::TensorDim dim(batch, channel, height, width,
//                            nntrainer::Tformat::NCHW,
//                            nntrainer::Tdatatype::FP16);

//   nntrainer::Tensor input(dim);
//   GEN_TEST_INPUT(input, k + l * 0.5 + 0.5);
//   nntrainer::Tensor actual = input.erf();
//   nntrainer::Tensor expected(
//     std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
//       {{{{0.5205, 0.8427}, {0.966105, 0.995322}}}}),
//     dim.getTensorType());

//   EXPECT_EQ(actual, expected);
// }

TEST(nntrainer_Tensor, subtract_i_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  nntrainer::Tensor original(batch, height, width, nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  original.copy(target);

  status = target.subtract_i(2.1);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *previous = original.getData<_FP16>();
  ASSERT_NE(nullptr, previous);
  _FP16 *data = target.getData<_FP16>();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], ((_FP16)(previous[i] - (_FP16)2.1)));
  }
}

TEST(nntrainer_Tensor, subtract_i_02_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  status = target.subtract_i(target);
  EXPECT_EQ(status, ML_ERROR_NONE);

  _FP16 *data = target.getData<_FP16>();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_FLOAT_EQ(data[i], 0);
  }
}

TEST(nntrainer_Tensor, subtract_i_03_n) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int height = 3;
  int width = 10;
  int channel = 1;

  nntrainer::Tensor target(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(target, i * (batch * height) + j * (width) + k + 1 + channel);

  nntrainer::Tensor target2(batch, channel, height - 1, width - 3,
                            nntrainer::Tformat::NCHW,
                            nntrainer::Tdatatype::FP16);

  status = target.subtract_i(target2);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_Tensor, subtract_01_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(1.0);

  _FP16 *data = result.getData<_FP16>();
  ASSERT_NE(nullptr, data);
  _FP16 *indata = input.getData<_FP16>();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    if (data[i] != indata[i] - 1.0) {
      status = ML_ERROR_RESULT_OUT_OF_RANGE;
      break;
    }
  }

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, subtract_02_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor result = input.subtract(input);

  EXPECT_EQ(constant(0.0, batch, channel, height, width), result);
}

TEST(nntrainer_Tensor, subtract_03_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor test(batch - 1, channel, height - 1, width - 1,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);

  EXPECT_THROW({ input.subtract(test); }, std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_04_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(batch, channel, height, 2 * width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_input = input.getSharedDataTensor(dim, 0, false);
  nntrainer::Tensor test(dim);

  EXPECT_THROW(shared_input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_05_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  nntrainer::Tensor test(batch, channel, height, 2 * width,
                         nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor shared_test = test.getSharedDataTensor(dim, 0, false);

  EXPECT_THROW(input.subtract(shared_test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_06_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim, false);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 1);

  EXPECT_THROW(input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_07_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim, false);

  EXPECT_THROW(input.subtract(test), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract_08_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::TensorDim dim(batch, channel, height, width,
                           nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP16);

  nntrainer::Tensor input(dim);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);
  nntrainer::Tensor test(dim);
  GEN_TEST_INPUT(test, i * (batch * height) + j * (width) + k + 2);
  nntrainer::Tensor output(dim, false);

  EXPECT_THROW(input.subtract(test, output), std::invalid_argument);
}

TEST(nntrainer_Tensor, subtract__Float16_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

  nntrainer::Tensor expected(batch, channel, height, width,
                             nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(expected, i * (batch * height) + j * (width) + k);

  nntrainer::Tensor result = input.subtract(1.0);

  EXPECT_EQ(result, expected);
}

TEST(nntrainer_Tensor, sum_01_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  EXPECT_THROW({ input.sum(4); }, std::out_of_range);
}

TEST(nntrainer_Tensor, sum_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 3;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

  EXPECT_THROW({ input.sum(-1); }, std::out_of_range);
}

TEST(nntrainer_Tensor, sum_02_p) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor ans0(
    std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
      {{{{39, 42, 45, 48, 51, 54, 57, 60, 63, 66},
         {69, 72, 75, 78, 81, 84, 87, 90, 93, 96}},
        {{57, 60, 63, 66, 69, 72, 75, 78, 81, 84},
         {87, 90, 93, 96, 99, 102, 105, 108, 111, 114}}}}),
    t_type);

  nntrainer::Tensor ans1(
    std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
      {{{{8, 10, 12, 14, 16, 18, 20, 22, 24, 26},
         {28, 30, 32, 34, 36, 38, 40, 42, 44, 46}}},
       {{{32, 34, 36, 38, 40, 42, 44, 46, 48, 50},
         {52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
       {{{56, 58, 60, 62, 64, 66, 68, 70, 72, 74},
         {76, 78, 80, 82, 84, 86, 88, 90, 92, 94}}}}),
    t_type);

  nntrainer::Tensor ans2(
    std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
      {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}},
        {{24, 26, 28, 30, 32, 34, 36, 38, 40, 42}}},
       {{{36, 38, 40, 42, 44, 46, 48, 50, 52, 54}},
        {{48, 50, 52, 54, 56, 58, 60, 62, 64, 66}}},
       {{{60, 62, 64, 66, 68, 70, 72, 74, 76, 78}},
        {{72, 74, 76, 78, 80, 82, 84, 86, 88, 90}}}}),
    t_type);

  nntrainer::Tensor ans3(
    std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
      {{{{55}, {155}}, {{115}, {215}}},
       {{{175}, {275}}, {{235}, {335}}},
       {{{295}, {395}}, {{355}, {455}}}}),
    t_type);

  nntrainer::Tensor input(batch, channel, height, width, t_type);
  GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (batch * height) +
                          k * (width) + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result1 = input.sum(1);
  nntrainer::Tensor result2 = input.sum(2);
  nntrainer::Tensor result3 = input.sum(3);

  EXPECT_EQ(ans0, result0);
  EXPECT_EQ(ans1, result1);
  EXPECT_EQ(ans2, result2);
  EXPECT_EQ(ans3, result3);
}

TEST(nntrainer_Tensor, sum_03_p) {
  const int batch = 3;
  const int channel = 2;
  const int height = 1;
  const int width = 10;

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(batch, channel, height, width, t_type);
  GEN_TEST_INPUT(input, i * (height * channel * width) + j * (height * width) +
                          k * (width) + l + 1);
  // Test for alpha == 1 and beta == 0 and dimension of reduced axis == 1
  {
    nntrainer::Tensor ans_0_1_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{63, 66, 69, 72, 75, 78, 81, 84, 87, 90}},
          {{93, 96, 99, 102, 105, 108, 111, 114, 117, 120}}}}),
      t_type);

    nntrainer::Tensor ans_1_1_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{12, 14, 16, 18, 20, 22, 24, 26, 28, 30}}},
         {{{52, 54, 56, 58, 60, 62, 64, 66, 68, 70}}},
         {{{92, 94, 96, 98, 100, 102, 104, 106, 108, 110}}}}),
      t_type);

    nntrainer::Tensor ans_2_1_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
          {{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}}},
         {{{21, 22, 23, 24, 25, 26, 27, 28, 29, 30}},
          {{31, 32, 33, 34, 35, 36, 37, 38, 39, 40}}},
         {{{41, 42, 43, 44, 45, 46, 47, 48, 49, 50}},
          {{51, 52, 53, 54, 55, 56, 57, 58, 59, 60}}}}),
      t_type);

    nntrainer::Tensor ans_3_1_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{55}}, {{155}}}, {{{255}}, {{355}}}, {{{455}}, {{555}}}}),
      t_type);

    nntrainer::Tensor result_0_1_0 = input.sum(0, 1);
    nntrainer::Tensor result_1_1_0 = input.sum(1, 1);
    nntrainer::Tensor result_2_1_0 = input.sum(2, 1);
    nntrainer::Tensor result_3_1_0 = input.sum(3, 1);

    EXPECT_EQ(ans_0_1_0, result_0_1_0);
    EXPECT_EQ(ans_1_1_0, result_1_1_0);
    EXPECT_EQ(ans_2_1_0, result_2_1_0);
    EXPECT_EQ(ans_3_1_0, result_3_1_0);
  }

  // Test for alpha == 1 and beta == 2 and dimension of reduced axis == 1
  {
    nntrainer::Tensor ans_0_1_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{65, 70, 75, 80, 85, 90, 95, 100, 105, 110}},
          {{115, 120, 125, 130, 135, 140, 145, 150, 155, 160}}}}),
      t_type);

    nntrainer::Tensor ans_1_1_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{14, 18, 22, 26, 30, 34, 38, 42, 46, 50}}},
         {{{74, 78, 82, 86, 90, 94, 98, 102, 106, 110}}},
         {{{134, 138, 142, 146, 150, 154, 158, 162, 166, 170}}}}),
      t_type);

    nntrainer::Tensor ans_2_1_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{3, 6, 9, 12, 15, 18, 21, 24, 27, 30}},
          {{33, 36, 39, 42, 45, 48, 51, 54, 57, 60}}},
         {{{63, 66, 69, 72, 75, 78, 81, 84, 87, 90}},
          {{93, 96, 99, 102, 105, 108, 111, 114, 117, 120}}},
         {{{123, 126, 129, 132, 135, 138, 141, 144, 147, 150}},
          {{153, 156, 159, 162, 165, 168, 171, 174, 177, 180}}}}),
      t_type);

    nntrainer::Tensor ans_3_1_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{57}}, {{159}}}, {{{261}}, {{363}}}, {{{465}}, {{567}}}}),
      t_type);

    nntrainer::Tensor output_0_1_2(1, channel, height, width, t_type);
    {
      const int batch = 1;
      GEN_TEST_INPUT(output_0_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_1_1_2(batch, 1, height, width, t_type);
    {
      const int channel = 1;
      GEN_TEST_INPUT(output_1_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_2_1_2(batch, channel, 1, width, t_type);
    {
      const int height = 1;
      GEN_TEST_INPUT(output_2_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_3_1_2(batch, channel, height, 1, t_type);
    {
      const int width = 1;
      GEN_TEST_INPUT(output_3_1_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor result_0_1_2 = input.sum(0, output_0_1_2, 1, 2);
    nntrainer::Tensor result_1_1_2 = input.sum(1, output_1_1_2, 1, 2);
    nntrainer::Tensor result_2_1_2 = input.sum(2, output_2_1_2, 1, 2);
    nntrainer::Tensor result_3_1_2 = input.sum(3, output_3_1_2, 1, 2);

    EXPECT_EQ(ans_0_1_2, result_0_1_2);
    EXPECT_EQ(ans_1_1_2, result_1_1_2);
    EXPECT_EQ(ans_2_1_2, result_2_1_2);
    EXPECT_EQ(ans_3_1_2, result_3_1_2);
  }

  // Test for alpha == 2 and beta == 0
  {
    nntrainer::Tensor ans_0_2_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{126, 132, 138, 144, 150, 156, 162, 168, 174, 180}},
          {{186, 192, 198, 204, 210, 216, 222, 228, 234, 240}}}}),
      t_type);

    nntrainer::Tensor ans_1_2_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{24, 28, 32, 36, 40, 44, 48, 52, 56, 60}}},
         {{{104, 108, 112, 116, 120, 124, 128, 132, 136, 140}}},
         {{{184, 188, 192, 196, 200, 204, 208, 212, 216, 220}}}}),
      t_type);

    nntrainer::Tensor ans_2_2_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}},
          {{22, 24, 26, 28, 30, 32, 34, 36, 38, 40}}},
         {{{42, 44, 46, 48, 50, 52, 54, 56, 58, 60}},
          {{62, 64, 66, 68, 70, 72, 74, 76, 78, 80}}},
         {{{82, 84, 86, 88, 90, 92, 94, 96, 98, 100}},
          {{102, 104, 106, 108, 110, 112, 114, 116, 118, 120}}}}),
      t_type);

    nntrainer::Tensor ans_3_2_0(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{110}}, {{310}}}, {{{510}}, {{710}}}, {{{910}}, {{1110}}}}),
      t_type);

    nntrainer::Tensor result_0_2_0 = input.sum(0, 2);
    nntrainer::Tensor result_1_2_0 = input.sum(1, 2);
    nntrainer::Tensor result_2_2_0 = input.sum(2, 2);
    nntrainer::Tensor result_3_2_0 = input.sum(3, 2);

    EXPECT_EQ(ans_0_2_0, result_0_2_0);
    EXPECT_EQ(ans_1_2_0, result_1_2_0);
    EXPECT_EQ(ans_2_2_0, result_2_2_0);
    EXPECT_EQ(ans_3_2_0, result_3_2_0);
  }

  // Test for alpha == 2 and beta == 2
  {
    nntrainer::Tensor ans_0_2_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{128, 136, 144, 152, 160, 168, 176, 184, 192, 200}},
          {{208, 216, 224, 232, 240, 248, 256, 264, 272, 280}}}}),
      t_type);

    nntrainer::Tensor ans_1_2_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{26, 32, 38, 44, 50, 56, 62, 68, 74, 80}}},
         {{{126, 132, 138, 144, 150, 156, 162, 168, 174, 180}}},
         {{{226, 232, 238, 244, 250, 256, 262, 268, 274, 280}}}}),
      t_type);

    nntrainer::Tensor ans_2_2_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{4, 8, 12, 16, 20, 24, 28, 32, 36, 40}},
          {{44, 48, 52, 56, 60, 64, 68, 72, 76, 80}}},
         {{{84, 88, 92, 96, 100, 104, 108, 112, 116, 120}},
          {{124, 128, 132, 136, 140, 144, 148, 152, 156, 160}}},
         {{{164, 168, 172, 176, 180, 184, 188, 192, 196, 200}},
          {{204, 208, 212, 216, 220, 224, 228, 232, 236, 240}}}}),
      t_type);

    nntrainer::Tensor ans_3_2_2(
      std::vector<std::vector<std::vector<std::vector<_FP16>>>>(
        {{{{112}}, {{314}}}, {{{516}}, {{718}}}, {{{920}}, {{1122}}}}),
      t_type);

    nntrainer::Tensor output_0_2_2(1, channel, height, width, t_type);
    {
      const int batch = 1;
      GEN_TEST_INPUT(output_0_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_1_2_2(batch, 1, height, width, t_type);
    {
      const int channel = 1;
      GEN_TEST_INPUT(output_1_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_2_2_2(batch, channel, 1, width, t_type);
    {
      const int height = 1;
      GEN_TEST_INPUT(output_2_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor output_3_2_2(batch, channel, height, 1, t_type);
    {
      const int width = 1;
      GEN_TEST_INPUT(output_3_2_2, i * (channel * height * width) +
                                     j * (height * width) + k * (width) + l +
                                     1);
    }
    nntrainer::Tensor result_0_2_2 = input.sum(0, output_0_2_2, 2, 2);
    nntrainer::Tensor result_1_2_2 = input.sum(1, output_1_2_2, 2, 2);
    nntrainer::Tensor result_2_2_2 = input.sum(2, output_2_2_2, 2, 2);
    nntrainer::Tensor result_3_2_2 = input.sum(3, output_3_2_2, 2, 2);

    EXPECT_EQ(ans_0_2_2, result_0_2_2);
    EXPECT_EQ(ans_1_2_2, result_1_2_2);
    EXPECT_EQ(ans_2_2_2, result_2_2_2);
    EXPECT_EQ(ans_3_2_2, result_3_2_2);
  }
}

TEST(nntrainer_Tensor, sum_04_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(batch, channel, height, width, t_type);
  GEN_TEST_INPUT(input, i * (batch * height * channel) + j * (height * width) +
                          k * width + l + 1);

  nntrainer::Tensor result = input.sum_by_batch();
  if (result.getValue<_FP16>(0, 0, 0, 0) != 820 ||
      result.getValue<_FP16>(1, 0, 0, 0) != 1300 ||
      result.getValue<_FP16>(2, 0, 0, 0) != 1780)
    status = ML_ERROR_RESULT_OUT_OF_RANGE;

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, multiple_sum_invalid_args_01_n) {

  nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  EXPECT_THROW(t.sum(std::vector<unsigned int>()), std::invalid_argument);
}

TEST(nntrainer_Tensor, multiple_sum_out_of_range_n) {
  nntrainer::Tensor t = constant(1.0, 1, 1, 1, 1, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  EXPECT_THROW(t.sum(7), std::out_of_range);
}

TEST(nntrainer_Tensor, multiple_sum_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  nntrainer::Tensor actual, expected;

  actual = t.sum({0, 1});
  expected = constant(2 * 3, 1, 1, 5, 7, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = t.sum({1, 2, 3});
  expected = constant(3 * 5 * 7, 2, 1, 1, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = t.sum({3, 1});
  expected = constant(7 * 3, 2, 1, 5, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = t.sum({3, 1}, 0.5);
  expected = constant(7 * 3 * 0.5, 2, 1, 5, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);
}

// @todo check later
// TEST(nntrainer_Tensor, average_p) {
//   nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, nntrainer::Tformat::NCHW,
//                                  nntrainer::Tdatatype::FP16);

//   nntrainer::Tensor actual, expected;
//   actual.setTensorType({nntrainer::Tformat::NCHW,
//   nntrainer::Tdatatype::FP16});

//   actual = t.average();
//   expected = constant(1.0, 1, 1, 1, 1, nntrainer::Tformat::NCHW,
//                       nntrainer::Tdatatype::FP16);
//   EXPECT_EQ(actual, expected);

//   int idx = 0;
//   t = t.apply(
//     (std::function<_FP16(_FP16)>)[&](_FP16 in) { return idx++ % 2; });

//   actual = t.average();
//   expected = constant(0.5, 1, 1, 1, 1, nntrainer::Tformat::NCHW,
//                       nntrainer::Tdatatype::FP16);
//   EXPECT_EQ(actual, expected);
// }

TEST(nntrainer_Tensor, average_axis_p) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  int idx = 0;
  std::function<_FP16(_FP16)> f = [&](_FP16 in) {
    return static_cast<_FP16>(idx++ % 2);
  };
  t = t.apply(f);

  nntrainer::Tensor actual, expected;

  actual = t.average(0);
  expected = constant(0, 1, 2, 2, 2, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16)
               .apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(1);
  expected = constant(0, 2, 1, 2, 2, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16)
               .apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(2);
  expected = constant(0, 2, 2, 1, 2, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16)
               .apply(f);
  EXPECT_EQ(actual, expected);

  actual = t.average(3);
  expected = constant(0.5, 2, 2, 2, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_axis_out_of_range_01_n) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  EXPECT_THROW(t.average(-1), std::out_of_range);
}

TEST(nntrainer_Tensor, average_axis_out_of_range_02_n) {
  nntrainer::Tensor t = constant(1.0, 2, 2, 2, 2, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  EXPECT_THROW(t.average(7), std::out_of_range);
}

TEST(nntrainer_Tensor, average_multiple_axes_p) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  nntrainer::Tensor actual, expected;

  actual = t.average({0, 1, 2});
  expected = constant(1.0, 1, 1, 1, 7, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = t.average({0, 1, 2, 3});
  expected = constant(1.0, 1, 1, 1, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = t.average({3, 1});
  expected = constant(1.0, 2, 1, 5, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);

  actual = t.average({3, 1, 1, 1, 3});
  expected = constant(1.0, 2, 1, 5, 1, nntrainer::Tformat::NCHW,
                      nntrainer::Tdatatype::FP16);
  EXPECT_EQ(actual, expected);
}

TEST(nntrainer_Tensor, average_multiple_axes_01_n) {
  nntrainer::Tensor t = constant(1.0, 2, 3, 5, 7, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  EXPECT_THROW(t.average({5, 7}), std::out_of_range);
}

TEST(nntrainer_Tensor, dot_01_n) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(2, 3, 4, 5, t_type);
  nntrainer::Tensor m(1, 3, 4, 5, t_type);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
}

TEST(nntrainer_Tensor, dot_02_n) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(2, 3, 4, 5, t_type);
  nntrainer::Tensor m(1, 3, 4, 5, t_type);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m, true),
               std::runtime_error);
}

TEST(nntrainer_Tensor, dot_02_p) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(2, 3, 4, 5, t_type);
  nntrainer::Tensor m(1, 3, 4, 5, t_type);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
}

TEST(nntrainer_Tensor, dot_03_p) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(1, 3, 4, 5, t_type);
  nntrainer::Tensor m(1, 3, 4, 5, t_type);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, true));
}

TEST(nntrainer_Tensor, dot_04_n) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor input(2, 3, 4, 5, t_type);
  nntrainer::Tensor m(1, 1, 4, 5, t_type);
  EXPECT_THROW(nntrainer::Tensor result = input.dot(m), std::runtime_error);
  EXPECT_NO_THROW(nntrainer::Tensor result = input.dot(m, false, true));
}

TEST(nntrainer_Tensor, dot_06_p) {
  int status = ML_ERROR_NONE;
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 3;

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  _FP16 ans[3][1][1][3] = {
    {{{30, 36, 42}}}, {{{66, 81, 96}}}, {{{102, 126, 150}}}};

  nntrainer::Tensor input(batch, channel, height, width, t_type);
  GEN_TEST_INPUT(input, i * (channel * width * height) + j * (height * width) +
                          k * (width) + l + 1);

  nntrainer::Tensor result = input.dot(input);

  for (unsigned int i = 0; i < result.batch(); ++i) {
    for (unsigned int j = 0; j < result.height(); ++j) {
      for (unsigned int k = 0; k < result.width(); ++k) {
        if (ans[i][0][j][k] != result.getValue<_FP16>(i, 0, j, k)) {
          status = ML_ERROR_RESULT_OUT_OF_RANGE;
          goto end_dot_01_p;
        }
      }
    }
  }
end_dot_01_p:
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_Tensor, dot_transpose_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;
  {
    _FP16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23,  26,  29,  56,  68,  80,  92,
                           92, 113, 134, 155, 128, 158, 188, 218};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29, 56, 68, 80, 92};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4, t_type), a_data);
    _FP16 b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3, t_type), a_data);
    _FP16 b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40, 46, 67, 64, 94};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2, t_type), a_data);
    _FP16 b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3, t_type), a_data);
    _FP16 b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2, t_type), b_data);
    _FP16 answer_data[] = {10, 13, 28, 40};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
}

TEST(nntrainer_Tensor, dot_shortcuts_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, t_type), b_data);
    _FP16 answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, t_type), b_data);
    _FP16 answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3, t_type), b_data);
    _FP16 answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3, t_type), b_data);
    _FP16 answer_data[] = {5};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, t_type), b_data);
    _FP16 answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, t_type), b_data);
    _FP16 answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 2, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3, t_type), b_data);
    _FP16 answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 1, 4, 2, 5};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 2, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3, t_type), b_data);
    _FP16 answer_data[] = {5, 14};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 2, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, t_type), b_data);
    _FP16 answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 1, t_type), b_data);
    _FP16 answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 4, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3, t_type), b_data);
    _FP16 answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 4, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 1, 3, t_type), b_data);
    _FP16 answer_data[] = {5, 14, 23, 32};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 4, 1, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 4, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 4, 3, t_type), b_data);
    _FP16 answer_data[] = {20, 23, 26, 29};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 4, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2, t_type), b_data);
    _FP16 answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 1, 2, 3, 4, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 3, 2, t_type), b_data);
    _FP16 answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, false);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 1, 3, t_type), a_data);
    _FP16 b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3, t_type), b_data);
    _FP16 answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, false, true);
    EXPECT_EQ(ret, answer);
  }
  {
    _FP16 a_data[] = {0, 1, 2};
    nntrainer::Tensor a(nntrainer::TensorDim(1, 1, 3, 1, t_type), a_data);
    _FP16 b_data[] = {0, 2, 4, 1, 3, 5};
    nntrainer::Tensor b(nntrainer::TensorDim(1, 1, 2, 3, t_type), b_data);
    _FP16 answer_data[] = {10, 13};
    nntrainer::Tensor answer(nntrainer::TensorDim(1, 1, 1, 2, t_type),
                             answer_data);
    nntrainer::Tensor ret = a.dot(b, true, true);
    EXPECT_EQ(ret, answer);
  }
}

TEST(nntrainer_Tensor, transpose_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);

  /// plain transpose
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
      98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 2, 4, 5, t_type}, answer_data);
    nntrainer::Tensor m = t.transpose("0:1:2");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,   5,   10,  15,  1,   6,   11,  16,  2,   7,   12,  17,  3,   8,
      13,  18,  4,   9,   14,  19,  20,  25,  30,  35,  21,  26,  31,  36,
      22,  27,  32,  37,  23,  28,  33,  38,  24,  29,  34,  39,  40,  45,
      50,  55,  41,  46,  51,  56,  42,  47,  52,  57,  43,  48,  53,  58,
      44,  49,  54,  59,  60,  65,  70,  75,  61,  66,  71,  76,  62,  67,
      72,  77,  63,  68,  73,  78,  64,  69,  74,  79,  80,  85,  90,  95,
      81,  86,  91,  96,  82,  87,  92,  97,  83,  88,  93,  98,  84,  89,
      94,  99,  100, 105, 110, 115, 101, 106, 111, 116, 102, 107, 112, 117,
      103, 108, 113, 118, 104, 109, 114, 119};
    nntrainer::Tensor answer({3, 2, 5, 4, t_type}, answer_data);
    nntrainer::Tensor m = t.transpose("0:2:1");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,   1,   2,   3,   4,   20,  21,  22,  23,  24,  5,   6,   7,   8,
      9,   25,  26,  27,  28,  29,  10,  11,  12,  13,  14,  30,  31,  32,
      33,  34,  15,  16,  17,  18,  19,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  60,  61,  62,  63,  64,  45,  46,  47,  48,  49,  65,
      66,  67,  68,  69,  50,  51,  52,  53,  54,  70,  71,  72,  73,  74,
      55,  56,  57,  58,  59,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  100, 101, 102, 103, 104, 85,  86,  87,  88,  89,  105, 106, 107,
      108, 109, 90,  91,  92,  93,  94,  110, 111, 112, 113, 114, 95,  96,
      97,  98,  99,  115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 4, 2, 5, t_type}, answer_data);
    nntrainer::Tensor m = t.transpose("1:0:2");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
      8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
      16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
      44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
      52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
      80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
      88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
      96, 116, 97, 117, 98, 118, 99, 119};
    nntrainer::Tensor answer({3, 4, 5, 2, t_type}, answer_data);
    nntrainer::Tensor m = t.transpose("1:2:0");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
      36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
      33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
      65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
      62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
      59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
      91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
      88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
    nntrainer::Tensor answer({3, 5, 2, 4, t_type}, answer_data);
    nntrainer::Tensor m = t.transpose("2:0:1");
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
      2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
      4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
      41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
      43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
      80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
      82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
      84, 104, 89, 109, 94, 114, 99, 119};
    nntrainer::Tensor answer({3, 5, 4, 2, t_type}, answer_data);
    nntrainer::Tensor m = t.transpose("2:1:0");
    EXPECT_EQ(answer, m);
  }

  /// outplace transpose
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
      98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 2, 4, 5, t_type}, answer_data);
    t.transpose("0:1:2", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 2, 5, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,   5,   10,  15,  1,   6,   11,  16,  2,   7,   12,  17,  3,   8,
      13,  18,  4,   9,   14,  19,  20,  25,  30,  35,  21,  26,  31,  36,
      22,  27,  32,  37,  23,  28,  33,  38,  24,  29,  34,  39,  40,  45,
      50,  55,  41,  46,  51,  56,  42,  47,  52,  57,  43,  48,  53,  58,
      44,  49,  54,  59,  60,  65,  70,  75,  61,  66,  71,  76,  62,  67,
      72,  77,  63,  68,  73,  78,  64,  69,  74,  79,  80,  85,  90,  95,
      81,  86,  91,  96,  82,  87,  92,  97,  83,  88,  93,  98,  84,  89,
      94,  99,  100, 105, 110, 115, 101, 106, 111, 116, 102, 107, 112, 117,
      103, 108, 113, 118, 104, 109, 114, 119};
    nntrainer::Tensor answer({3, 2, 5, 4, t_type}, answer_data);
    t.transpose("0:2:1", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 4, 2, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,   1,   2,   3,   4,   20,  21,  22,  23,  24,  5,   6,   7,   8,
      9,   25,  26,  27,  28,  29,  10,  11,  12,  13,  14,  30,  31,  32,
      33,  34,  15,  16,  17,  18,  19,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  60,  61,  62,  63,  64,  45,  46,  47,  48,  49,  65,
      66,  67,  68,  69,  50,  51,  52,  53,  54,  70,  71,  72,  73,  74,
      55,  56,  57,  58,  59,  75,  76,  77,  78,  79,  80,  81,  82,  83,
      84,  100, 101, 102, 103, 104, 85,  86,  87,  88,  89,  105, 106, 107,
      108, 109, 90,  91,  92,  93,  94,  110, 111, 112, 113, 114, 95,  96,
      97,  98,  99,  115, 116, 117, 118, 119};
    nntrainer::Tensor answer({3, 4, 2, 5, t_type}, answer_data);
    t.transpose("1:0:2", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 4, 5, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,  20,  1,  21,  2,  22,  3,  23,  4,  24,  5,  25,  6,  26,  7,  27,
      8,  28,  9,  29,  10, 30,  11, 31,  12, 32,  13, 33,  14, 34,  15, 35,
      16, 36,  17, 37,  18, 38,  19, 39,  40, 60,  41, 61,  42, 62,  43, 63,
      44, 64,  45, 65,  46, 66,  47, 67,  48, 68,  49, 69,  50, 70,  51, 71,
      52, 72,  53, 73,  54, 74,  55, 75,  56, 76,  57, 77,  58, 78,  59, 79,
      80, 100, 81, 101, 82, 102, 83, 103, 84, 104, 85, 105, 86, 106, 87, 107,
      88, 108, 89, 109, 90, 110, 91, 111, 92, 112, 93, 113, 94, 114, 95, 115,
      96, 116, 97, 117, 98, 118, 99, 119};
    nntrainer::Tensor answer({3, 4, 5, 2, t_type}, answer_data);
    t.transpose("1:2:0", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 5, 2, 4, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,  5,  10,  15,  20,  25,  30,  35, 1,  6,   11,  16,  21,  26,  31,
      36, 2,  7,   12,  17,  22,  27,  32, 37, 3,   8,   13,  18,  23,  28,
      33, 38, 4,   9,   14,  19,  24,  29, 34, 39,  40,  45,  50,  55,  60,
      65, 70, 75,  41,  46,  51,  56,  61, 66, 71,  76,  42,  47,  52,  57,
      62, 67, 72,  77,  43,  48,  53,  58, 63, 68,  73,  78,  44,  49,  54,
      59, 64, 69,  74,  79,  80,  85,  90, 95, 100, 105, 110, 115, 81,  86,
      91, 96, 101, 106, 111, 116, 82,  87, 92, 97,  102, 107, 112, 117, 83,
      88, 93, 98,  103, 108, 113, 118, 84, 89, 94,  99,  104, 109, 114, 119};
    nntrainer::Tensor answer({3, 5, 2, 4, t_type}, answer_data);
    t.transpose("2:0:1", m);
    EXPECT_EQ(answer, m);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    nntrainer::Tensor m =
      ranged(3, 5, 4, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    _FP16 answer_data[] = {
      0,  20,  5,  25,  10, 30,  15, 35,  1,  21,  6,  26,  11, 31,  16, 36,
      2,  22,  7,  27,  12, 32,  17, 37,  3,  23,  8,  28,  13, 33,  18, 38,
      4,  24,  9,  29,  14, 34,  19, 39,  40, 60,  45, 65,  50, 70,  55, 75,
      41, 61,  46, 66,  51, 71,  56, 76,  42, 62,  47, 67,  52, 72,  57, 77,
      43, 63,  48, 68,  53, 73,  58, 78,  44, 64,  49, 69,  54, 74,  59, 79,
      80, 100, 85, 105, 90, 110, 95, 115, 81, 101, 86, 106, 91, 111, 96, 116,
      82, 102, 87, 107, 92, 112, 97, 117, 83, 103, 88, 108, 93, 113, 98, 118,
      84, 104, 89, 109, 94, 114, 99, 119};
    nntrainer::Tensor answer({3, 5, 4, 2, t_type}, answer_data);
    t.transpose("2:1:0", m);
    EXPECT_EQ(answer, m);
  }
}

TEST(nntrainer_Tensor, tranpose_dimension_not_match_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor a(3, 2, 4, 5, t_type);
  nntrainer::Tensor b(3, 1, 2, 3, t_type);

  EXPECT_THROW(a.transpose("0:1:2", b), std::invalid_argument);
}

// TEST(nntrainer_Tensor, DISABLED_set_01_p) {
//   nntrainer::TensorDim::TensorType t_type;
//   t_type.format = nntrainer::Tformat::NCHW;
//   t_type.data_type = nntrainer::Tdatatype::FP16;

//   nntrainer::Tensor tensor = nntrainer::Tensor(1, 1, 1, 1, t_type);

//   tensor.setZero();
//   EXPECT_EQ(tensor.getValue<_FP16>(0, 0, 0, 0), 0.0);

//   tensor.setRandUniform(-0.5, 0.0);
//   std::cout << "val : " << (float)tensor.getValue<_FP16>(0, 0, 0, 0)
//             << std::endl;

//   _FP16 val = tensor.getValue<_FP16>(0, 0, 0, 0);
//   EXPECT_TRUE(val >= -0.5 && val < 0);
// }

TEST(nntrainer_Tensor, save_read_01_p) {
  int batch = 3;
  int channel = 4;
  int height = 5;
  int width = 6;

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor target(3, 4, 5, 6, t_type);
  nntrainer::Tensor readed(3, 4, 5, 6, t_type);

  GEN_TEST_INPUT(target, i * (channel * width * height) + j * (height * width) +
                           k * (width) + l + 1);

  std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
  target.save(save_file);
  save_file.close();

  std::ifstream read_file("save.bin");
  readed.read(read_file);
  read_file.close();

  EXPECT_EQ(target, readed);

  int status = std::remove("save.bin");

  ASSERT_EQ(status, 0);
}

TEST(nntrainer_Tensor, save_read_01_n) {
  int batch = 3;
  int channel = 4;
  int height = 5;
  int width = 6;

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor target(3, 4, 5, 6, t_type);
  nntrainer::Tensor readed(3, 4, 1, 1, t_type);

  GEN_TEST_INPUT(target, i * (channel * width * height) + j * (height * width) +
                           k * (width) + l + 1);

  std::ofstream save_file("save.bin", std::ios::out | std::ios::binary);
  target.save(save_file);
  save_file.close();

  std::ifstream read_file("save.bin");
  readed.read(read_file);
  read_file.close();

  EXPECT_NE(target, readed);

  int status = std::remove("save.bin");

  ASSERT_EQ(status, 0);
}

TEST(nntrainer_Tensor, copy_and_shares_variable_p) {

  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  nntrainer::Tensor B = A.clone();
  nntrainer::Tensor C = A;

  C.setValue(1, 1, 1, 1, 2.0f);

  EXPECT_EQ(A, C);
  EXPECT_NE(B, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5, t_type));
  EXPECT_EQ(A.getDim(), B.getDim());
  EXPECT_NE(A.getDim(), C.getDim());
}

TEST(nntrainer_Tensor, reshape_n_01) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);

  EXPECT_THROW(A.reshape(nntrainer::TensorDim(9, 9, 9, 9, t_type)),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, reshape_n_02) {
  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  nntrainer::TensorDim A_dim = A.getDim();

  /** Changing the dim of a tensor only affects local copy of the dim */
  A_dim.setTensorDim(1, 100);
  EXPECT_EQ(A_dim.getTensorDim(1), 100u);

  nntrainer::TensorDim A_dim_2 = A.getDim();
  EXPECT_EQ(A_dim_2.getTensorDim(1), 4u);
}

TEST(nntrainer_Tensor, copy_and_reshape_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor A = constant(1.0f, 3, 4, 5, 6, nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP16);
  nntrainer::Tensor B = A;
  nntrainer::Tensor C = A.clone();

  EXPECT_THROW(B.reshape(nntrainer::TensorDim(9, 9, 9, 9, t_type)),
               std::invalid_argument);
}

/// @note this test case demonstrates it is dangerous to use sharedConstTensor
/// to const correct the inner data.
TEST(nntrainer_Tensor, constructor_from_shared_const_ptr_shares_variable_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::sharedConstTensor A = MAKE_SHARED_TENSOR(constant(
    1.0f, 3, 4, 5, 6, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));

  nntrainer::Tensor B = *A;
  nntrainer::Tensor C = A->clone();

  B.setValue(2, 3, 4, 5, 2.0f);
  EXPECT_EQ(*A, B);
  EXPECT_NE(*A, C);

  C.reshape(nntrainer::TensorDim(3, 4, 6, 5, t_type));
  EXPECT_EQ(A->getDim(), B.getDim());
  EXPECT_NE(A->getDim(), C.getDim());
}

TEST(nntrainer_Tensor, print_small_size) {
  nntrainer::Tensor target = constant(1.0, 3, 1, 2, 3, nntrainer::Tformat::NCHW,
                                      nntrainer::Tdatatype::FP16);

  std::stringstream ss, expected;
  ss << target;

  expected << '<' << typeid(target).name() << " at " << &target << ">\n"
           << "data addr: " << target.getData<_FP16>() << '\n'
           << "Shape: 3:1:2:3 [ FP16 : NCHW ]\n"
           << "         1          1          1 \n"
           << "         1          1          1 \n"
           << "\n"
           << "-------\n"
           << "         1          1          1 \n"
           << "         1          1          1 \n"
           << "\n"
           << "-------\n"
           << "         1          1          1 \n"
           << "         1          1          1 \n"
           << "\n"
           << "-------\n";

  EXPECT_EQ(ss.str(), expected.str());
}

TEST(nntrainer_Tensor, print_large_size) {
  nntrainer::Tensor target = constant(
    1.2, 3, 10, 10, 10, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);

  std::stringstream ss, expected;

  expected << '<' << typeid(target).name() << " at " << &target << ">\n"
           << "data addr: " << target.getData<_FP16>() << '\n'
           << "Shape: 3:10:10:10 [ FP16 : NCHW ]\n"
           << "[1.2002 1.2002 1.2002 ... 1.2002 1.2002 1.2002]\n";
  ss << target;

  EXPECT_EQ(ss.str(), expected.str());
}

// TEST(nntrainer_Tensor, DISABLED_equation_test_01_p) {
//   nntrainer::Tensor a, b, c;
//   nntrainer::Tensor ret1, ret2;

//   a = randUniform(4, 6, 7, 3, -100, 100);
//   b = randUniform(4, 6, 7, 3, -100, 100);
//   c = randUniform(4, 6, 7, 3, -100, 100);

//   ret1 = a.subtract(b).multiply(c);
//   ret2 = a.multiply(c).subtract(b.multiply(c));

//   _FP16 *data1 = ret1.getData<_FP16>();
//   _FP16 *data2 = ret2.getData<_FP16>();
//   EXPECT_EQ(ret1, ret2);

//   for (unsigned int i = 0; i < ret1.size(); ++i) {
//     EXPECT_FLOAT_EQ(data1[i], data2[i]);
//   }
// }

TEST(nntrainer_Tensor, fill_p) {
  /// same dimension, buffer size
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;
  {
    nntrainer::Tensor target(3, 2, 4, 5, t_type);
    // nntrainer::Tensor original = randUniform(3, 2, 4, 5, -1.0f, 1.0f);
    nntrainer::Tensor original =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    target.fill(original, false);

    EXPECT_EQ(target, original);
  }

  /// uninitialized with initialized flag is true
  {
    nntrainer::Tensor target;
    // nntrainer::Tensor original = randUniform(3, 2, 4, 5, -1.0f, 1.0f);
    nntrainer::Tensor original =
      ranged(3, 5, 4, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    target.fill(original, true);

    EXPECT_EQ(target, original);
  }
}

TEST(nntrainer_Tensor, fill_uninitialized_n) {
  nntrainer::Tensor target;
  // nntrainer::Tensor original = randUniform(3, 1, 2, 3, -1.0f, 1.0f);
  nntrainer::Tensor original =
    ranged(3, 5, 4, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  EXPECT_THROW(target.fill(original, false), std::invalid_argument);
}

TEST(nntrainer_Tensor, fill_different_dimension_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor target(3, 1, 3, 2, t_type);
  // nntrainer::Tensor original = randUniform(3, 1, 2, 3, -1.0f, 1.0f);
  nntrainer::Tensor original =
    ranged(3, 1, 2, 3, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  EXPECT_THROW(target.fill(original, false), std::invalid_argument);
}

// TEST(nntrainer_Tensor, DISABLED_fill_non_contiguous_n) {
//   /// there is no way to make non contiguous tensor publicily yet
//   EXPECT_TRUE(false);
// }

// TEST(nntrainer_Tensor, DISABLED_fill_different_buffer_size_n) {
//   /// there is no way to make same dimension, diffrent buffersized tensor
//   /// publicily yet
//   EXPECT_TRUE(false);
// }

TEST(nntrainer_Tensor, empty_01) {
  nntrainer::Tensor t;

  EXPECT_TRUE(t.empty());
}

TEST(nntrainer_Tensor, empty_02) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, false);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, empty_03) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true);

  EXPECT_FALSE(t.empty());
}

TEST(nntrainer_Tensor, allocate_01_n) {
  nntrainer::Tensor t;
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_FALSE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_02_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, false);
  EXPECT_FALSE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, allocate_03_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true);
  EXPECT_TRUE(t.isAllocated());

  t.allocate();
  EXPECT_TRUE(t.isAllocated());
}

TEST(nntrainer_Tensor, initialize_01_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true, nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_02_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true);

  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1);

  EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_03_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, false,
                      nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_04_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, false);
  t.initialize(nntrainer::Initializer::ONES);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1);

  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_05_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, false);
  t.allocate();

  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1.f);

  /**
   * Ideally, it should be NE, but it can be equal due to no initialization
   * EXPECT_NE(golden, t);
   */

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_06_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true, nntrainer::Initializer::ONES);
  nntrainer::Tensor golden({1, 2, 3, 4, t_type}, true,
                           nntrainer::Initializer::ZEROS);

  EXPECT_NE(golden, t);

  golden.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_07_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true, nntrainer::Initializer::ONES);
  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1);

  EXPECT_EQ(golden, t);

  t.setValue(0, 0, 0, 0, 0);
  t.setValue(0, 0, 0, t.size() - 1, 0);
  EXPECT_NE(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, initialize_08_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t({1, 2, 3, 4, t_type}, true, nntrainer::Initializer::ONES);

  nntrainer::Tensor golden(1, 2, 3, 4, t_type);
  golden.setValue(1.f);
  EXPECT_EQ(golden, t);

  /// @todo this test case is not valid anymore, since
  /// std::uniform_real_distribution does not support _FP16
  // t.initialize(nntrainer::Initializer::HE_NORMAL);
  // EXPECT_NE(golden, t);
  // t.initialize();
  // EXPECT_NE(golden, t);

  t.initialize(nntrainer::Initializer::ONES);
  EXPECT_EQ(golden, t);

  t.initialize();
  EXPECT_EQ(golden, t);
}

TEST(nntrainer_Tensor, split_01_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      _FP16 answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 2, 4, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                             50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                             70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 2, 4, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                             90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 2, 4, 5, t_type}, answer_data));
    }
    EXPECT_EQ(t.split(3, 0), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
                             88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 1, 4, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                             30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                             60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                             70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 1, 4, 5, t_type}, answer_data));
    }
    EXPECT_EQ(t.split(2, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
        25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
        60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
        85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 2, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {
        10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
        35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
        95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 2, 5, t_type}, answer_data));
    }
    EXPECT_EQ(t.split(2, 2), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(5);
    {
      _FP16 answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
                             40, 45, 50, 55, 60,  65,  70,  75,
                             80, 85, 90, 95, 100, 105, 110, 115};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {1,  6,  11, 16, 21,  26,  31,  36,
                             41, 46, 51, 56, 61,  66,  71,  76,
                             81, 86, 91, 96, 101, 106, 111, 116};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {2,  7,  12, 17, 22,  27,  32,  37,
                             42, 47, 52, 57, 62,  67,  72,  77,
                             82, 87, 92, 97, 102, 107, 112, 117};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {3,  8,  13, 18, 23,  28,  33,  38,
                             43, 48, 53, 58, 63,  68,  73,  78,
                             83, 88, 93, 98, 103, 108, 113, 118};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    EXPECT_EQ(t.split(5, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 1, 4, 6, t_type);
    nntrainer::Tensor t =
      ranged(1, 1, 4, 6, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 1, 4, 3, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 1, 4, 3, t_type}, answer_data));
    }
    EXPECT_EQ(t.split(2, 3), answer);
  }
}

TEST(nntrainer_Tensor, split_02_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t(1, 1, 1, 1, t_type);
  EXPECT_THROW(t.split(0, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_03_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t(3, 1, 1, 1, t_type);
  EXPECT_THROW(t.split(2, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_04_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{2, 2, 4, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
                             90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 2, 4, 5, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({2, 1}, 0), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 40, 41, 42, 43,
                             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                             56, 57, 58, 59, 80, 81, 82, 83, 84, 85, 86, 87,
                             88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 1, 4, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
                             30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                             60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                             70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                             100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                             110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 1, 4, 5, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({1, 1}, 1), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {
        0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   20,  21,  22,  23,  24,
        25, 26, 27, 28, 29, 40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
        60, 61, 62, 63, 64, 65,  66,  67,  68,  69,  80,  81,  82,  83,  84,
        85, 86, 87, 88, 89, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 2, 5, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {
        10, 11, 12, 13, 14, 15,  16,  17,  18,  19,  30,  31,  32,  33,  34,
        35, 36, 37, 38, 39, 50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
        70, 71, 72, 73, 74, 75,  76,  77,  78,  79,  90,  91,  92,  93,  94,
        95, 96, 97, 98, 99, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 2, 5, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({2, 2}, 2), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      _FP16 answer_data[] = {0,  5,  10, 15, 20,  25,  30,  35,
                             40, 45, 50, 55, 60,  65,  70,  75,
                             80, 85, 90, 95, 100, 105, 110, 115};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {
        1,   2,   3,   6,   7,   8,   11,  12,  13,  16,  17,  18, 21, 22, 23,
        26,  27,  28,  31,  32,  33,  36,  37,  38,  41,  42,  43, 46, 47, 48,
        51,  52,  53,  56,  57,  58,  61,  62,  63,  66,  67,  68, 71, 72, 73,
        76,  77,  78,  81,  82,  83,  86,  87,  88,  91,  92,  93, 96, 97, 98,
        101, 102, 103, 106, 107, 108, 111, 112, 113, 116, 117, 118};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 3, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({1, 3, 1}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      _FP16 answer_data[] = {
        0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
        40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
        80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 2, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {
        2,  3,  7,  8,  12, 13, 17, 18, 22,  23,  27,  28,  32,  33,  37,  38,
        42, 43, 47, 48, 52, 53, 57, 58, 62,  63,  67,  68,  72,  73,  77,  78,
        82, 83, 87, 88, 92, 93, 97, 98, 102, 103, 107, 108, 112, 113, 117, 118};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 2, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {4,  9,  14, 19, 24,  29,  34,  39,
                             44, 49, 54, 59, 64,  69,  74,  79,
                             84, 89, 94, 99, 104, 109, 114, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 1, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({2, 2, 1}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(3, 2, 4, 5, t_type);
    nntrainer::Tensor t =
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(2);
    {
      _FP16 answer_data[] = {
        0,  1,  5,  6,  10, 11, 15, 16, 20,  21,  25,  26,  30,  31,  35,  36,
        40, 41, 45, 46, 50, 51, 55, 56, 60,  61,  65,  66,  70,  71,  75,  76,
        80, 81, 85, 86, 90, 91, 95, 96, 100, 101, 105, 106, 110, 111, 115, 116};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 2, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {
        2,   3,   4,   7,   8,   9,   12,  13,  14,  17,  18,  19, 22, 23, 24,
        27,  28,  29,  32,  33,  34,  37,  38,  39,  42,  43,  44, 47, 48, 49,
        52,  53,  54,  57,  58,  59,  62,  63,  64,  67,  68,  69, 72, 73, 74,
        77,  78,  79,  82,  83,  84,  87,  88,  89,  92,  93,  94, 97, 98, 99,
        102, 103, 104, 107, 108, 109, 112, 113, 114, 117, 118, 119};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{3, 2, 4, 3, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({2, 3}, 3), answer);
  }
  {
    nntrainer::TensorDim ref_dim(1, 1, 4, 6, t_type);
    nntrainer::Tensor t =
      ranged(1, 1, 4, 6, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
    std::vector<nntrainer::Tensor> answer;
    answer.reserve(3);
    {
      _FP16 answer_data[] = {0, 6, 12, 18};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 1, 4, 1, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 1, 4, 3, t_type}, answer_data));
    }
    {
      _FP16 answer_data[] = {4, 5, 10, 11, 16, 17, 22, 23};
      answer.emplace_back(nntrainer::Tensor(
        ml::train::TensorDim{1, 1, 4, 2, t_type}, answer_data));
    }
    EXPECT_EQ(t.split({1, 3, 2}, 3), answer);
  }
}

TEST(nntrainer_Tensor, split_05_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t(3, 1, 1, 1, t_type);
  EXPECT_THROW(t.split({1, 1}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_06_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t(3, 1, 1, 1, t_type);
  EXPECT_THROW(t.split({2, 0, 1}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, split_07_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  nntrainer::Tensor t(3, 1, 1, 1, t_type);
  EXPECT_THROW(t.split({}, 0), std::invalid_argument);
}

TEST(nntrainer_Tensor, cat_01_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(
      ranged(2, 1, 1, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    inputs.emplace_back(
      ranged(2, 2, 1, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    _FP16 answer_data[] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7};
    nntrainer::Tensor answer(ml::train::TensorDim{2, 3, 1, 2, t_type},
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    inputs.emplace_back(
      ranged(2, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    _FP16 answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
      90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104,
      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
      30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
      45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
      60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
      75,  76,  77,  78,  79};
    nntrainer::Tensor answer(ml::train::TensorDim{5, 2, 4, 5, t_type},
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 0), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(
      ranged(3, 3, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    inputs.emplace_back(
      ranged(3, 2, 4, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    _FP16 answer_data[] = {
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,
      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
      56,  57,  58,  59,  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
      10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
      24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
      38,  39,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
      72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,
      86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
      114, 115, 116, 117, 118, 119, 40,  41,  42,  43,  44,  45,  46,  47,
      48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
      62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
      76,  77,  78,  79,  120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
      130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
      144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
      158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
      172, 173, 174, 175, 176, 177, 178, 179, 80,  81,  82,  83,  84,  85,
      86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
      114, 115, 116, 117, 118, 119};
    nntrainer::Tensor answer(ml::train::TensorDim{3, 5, 4, 5, t_type},
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 1), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(
      ranged(3, 2, 1, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    inputs.emplace_back(
      ranged(3, 2, 2, 5, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    _FP16 answer_data[] = {
      0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  5,  6,  7,
      8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 15, 16, 17, 18, 19, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44, 45, 46,
      47, 48, 49, 25, 26, 27, 28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
    nntrainer::Tensor answer(ml::train::TensorDim{3, 2, 3, 5, t_type},
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 2), answer);
  }
  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(3);
    inputs.emplace_back(
      ranged(3, 2, 4, 1, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    inputs.emplace_back(
      ranged(3, 2, 4, 3, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    inputs.emplace_back(
      ranged(3, 2, 4, 2, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16));
    _FP16 answer_data[] = {
      0,  0,  1,  2,  0,  1,  1,  3,  4,  5,  2,  3,  2,  6,  7,  8,  4,  5,
      3,  9,  10, 11, 6,  7,  4,  12, 13, 14, 8,  9,  5,  15, 16, 17, 10, 11,
      6,  18, 19, 20, 12, 13, 7,  21, 22, 23, 14, 15, 8,  24, 25, 26, 16, 17,
      9,  27, 28, 29, 18, 19, 10, 30, 31, 32, 20, 21, 11, 33, 34, 35, 22, 23,
      12, 36, 37, 38, 24, 25, 13, 39, 40, 41, 26, 27, 14, 42, 43, 44, 28, 29,
      15, 45, 46, 47, 30, 31, 16, 48, 49, 50, 32, 33, 17, 51, 52, 53, 34, 35,
      18, 54, 55, 56, 36, 37, 19, 57, 58, 59, 38, 39, 20, 60, 61, 62, 40, 41,
      21, 63, 64, 65, 42, 43, 22, 66, 67, 68, 44, 45, 23, 69, 70, 71, 46, 47};
    nntrainer::Tensor answer(ml::train::TensorDim{3, 2, 4, 6, t_type},
                             answer_data);
    EXPECT_EQ(nntrainer::Tensor::cat(inputs, 3), answer);
  }
}

TEST(nntrainer_Tensor, cat_02_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  {
    std::vector<nntrainer::Tensor> inputs;
    inputs.reserve(2);
    inputs.emplace_back(nntrainer::Tensor(2, 1, 1, 2, t_type));
    inputs.emplace_back(nntrainer::Tensor(2, 2, 1, 2, t_type));
    EXPECT_THROW(nntrainer::Tensor::cat(inputs, 2), std::invalid_argument);
  }
}

TEST(nntrainer_Tensor, zoneout_mask_01_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 10, 10, t_type);
  nntrainer::Tensor opposite(20, 20, 20, 20, t_type);
  EXPECT_THROW(t.zoneout_mask(opposite, zoneout_rate), std::invalid_argument);
}

TEST(nntrainer_Tensor, zoneout_mask_02_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 10, 10, t_type);
  nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
  constexpr float epsilon = 1e-3;

  EXPECT_EQ(t.size(), opposite.size());

  auto is_near = [epsilon](_FP16 val1, _FP16 val2) {
    return val2 - epsilon < val1 && val1 < val2 + epsilon;
  };

  for (unsigned int i = 0; i < opposite.size(); ++i) {
    if (is_near(opposite.getValue<_FP16>(i), static_cast<_FP16>(0.0f))) {
      EXPECT_NEAR(t.getValue<_FP16>(i), 1.0f, epsilon);
    } else if (is_near(opposite.getValue<_FP16>(i), static_cast<_FP16>(1.0f))) {
      EXPECT_NEAR(t.getValue<_FP16>(i), 0.0f, epsilon);
    } else {
      FAIL() << "This should not be happen";
    }
  }
}

TEST(nntrainer_Tensor, zoneout_mask_03_p) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 100, 100, t_type);
  nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
  constexpr float epsilon = 1e-3;

  auto is_near = [epsilon](_FP16 val1, _FP16 val2) {
    return val2 - epsilon < val1 && val1 < val2 + epsilon;
  };
  auto percentage = [](unsigned int dividend, unsigned int divisor) {
    return (float)dividend / (float)divisor;
  };

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < opposite.size(); ++i) {
      if (is_near(opposite.getValue<_FP16>(i), static_cast<_FP16>(0.0))) {
        ++zeros;
      } else if (is_near(opposite.getValue<_FP16>(i),
                         static_cast<_FP16>(1.0))) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_NEAR(percentage(zeros, opposite.size()), (1.0 - zoneout_rate),
                epsilon);

    // main test
    EXPECT_NEAR(percentage(ones, opposite.size()), zoneout_rate, epsilon);
  }

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < t.size(); ++i) {
      if (is_near(t.getValue<_FP16>(i), (_FP16)0.0)) {
        ++zeros;
      } else if (is_near(t.getValue<_FP16>(i), (_FP16)1.0)) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_NEAR(percentage(zeros, t.size()), zoneout_rate, epsilon);

    // main test
    EXPECT_NEAR(percentage(ones, t.size()), (1.0f - zoneout_rate), epsilon);
  }
}

TEST(nntrainer_Tensor, zoneout_mask_04_n) {
  nntrainer::TensorDim::TensorType t_type;
  t_type.format = nntrainer::Tformat::NCHW;
  t_type.data_type = nntrainer::Tdatatype::FP16;

  const float zoneout_rate = 0.3f;
  nntrainer::Tensor t(10, 10, 100, 100, t_type);
  nntrainer::Tensor opposite = t.zoneout_mask(zoneout_rate);
  constexpr float epsilon = 1e-3;

  auto is_near = [epsilon](_FP16 val1, _FP16 val2) {
    return val2 - epsilon < val1 && val1 < val2 + epsilon;
  };
  auto percentage = [](unsigned int dividend, unsigned int divisor) {
    return (float)dividend / (float)divisor;
  };

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < opposite.size(); ++i) {
      if (is_near(opposite.getValue<_FP16>(i), static_cast<_FP16>(0.0f))) {
        ++zeros;
      } else if (is_near(opposite.getValue<_FP16>(i),
                         static_cast<_FP16>(1.0f))) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_FALSE(is_near(static_cast<_FP16>(percentage(ones, opposite.size())),
                         static_cast<_FP16>(1.0f - zoneout_rate)));
  }

  {
    unsigned int zeros = 0;
    unsigned int ones = 0;
    for (unsigned int i = 0; i < t.size(); ++i) {
      if (is_near(t.getValue<_FP16>(i), static_cast<_FP16>(0.0f))) {
        ++zeros;
      } else if (is_near(t.getValue<_FP16>(i), static_cast<_FP16>(1.0f))) {
        ++ones;
      } else {
        FAIL() << "This should not be happen";
      }
    }
    EXPECT_FALSE(is_near(static_cast<_FP16>(percentage(ones, t.size())),
                         static_cast<_FP16>(zoneout_rate)));
  }
}

TEST(nntrainer_Tensor, TensorMap_p) {
  _FP16 dat[] = {1, 2, 3};

  {
    nntrainer::Tensor a = nntrainer::Tensor::Map(dat, 3 * sizeof(_FP16), {3});
    /// check if a.getData<_FP16>() has same address with dat
    EXPECT_EQ(dat, a.getData<_FP16>());
    {
      /// check if b.getData<_FP16>() has same address with data
      nntrainer::Tensor b = a;
      EXPECT_EQ(dat, b.getData<_FP16>());
    }
  }
  /// check if dat is accessible after destruction of all the tensor
  EXPECT_FLOAT_EQ(dat[2], 3);
}

TEST(nntrainer_Tensor, TensorWrap_01_n) {
  _FP16 dat[] = {1, 2, 3};
  EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, nntrainer::TensorDim({})),
               std::invalid_argument);
}

TEST(nntrainer_Tensor, TensorWrap_02_n) {
  _FP16 dat[] = {1, 2, 3};
  EXPECT_THROW(nntrainer::Tensor::Map(dat, 3, {4}), std::invalid_argument);
}

// TEST(nntrainer_Tensor, TensorPaddedValue_p) {
//   nntrainer::Tensor a =
//     ranged(1, 1, 3, 3, nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
//   _FP16 default_padded = -1;

//   for (int i = 0; i < 5; ++i) {
//     for (int j = 0; j < 5; ++j) {
//       _FP16 expected = default_padded;
//       if (1 <= i && i <= 3 && 1 <= j && j <= 3) {
//         expected = (i - 1) * 3 + (j - 1);
//       }
//       _FP16 actual =
//         a.getValuePaddedVirtual<_FP16>(0, 0, i, j, 1, 1, default_padded);
//       EXPECT_FLOAT_EQ(actual, expected);
//     }
//   }
// }

// /**
//  * @brief dequantize FP16 tensor
//  */
// TEST(nntrainer_Tensor, dequantize_01_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(batch, channel, height, width,
//                           nntrainer::Tformat::NCHW,
//                           nntrainer::Tdatatype::FP16);
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   input.setScaleFactorsFP16({static_cast<_FP16>(1.5),
//   static_cast<_FP16>(1.0),
//                              static_cast<_FP16>(0.5)});
//   input.setZeroPoints({1, 4, 7});

//   nntrainer::Tensor output(batch, channel, height, width,
//                            nntrainer::Tformat::NCHW,
//                            nntrainer::Tdatatype::FP16);

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize tensor with different dimension
//  */
// TEST(nntrainer_Tensor, dequantize_02_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch + 1, channel, height + 1, width + 1,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   input.setScaleFactorsFP16({static_cast<_FP16>(1.5),
//   static_cast<_FP16>(1.0),
//                              static_cast<_FP16>(0.5)});
//   input.setZeroPoints({1, 4, 7});

//   nntrainer::Tensor output(batch, channel, height, width,
//                            nntrainer::Tformat::NCHW,
//                            nntrainer::Tdatatype::FP16);

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize tensor with no scale factors
//  */
// TEST(nntrainer_Tensor, dequantize_03_n) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k);

//   nntrainer::Tensor output(batch, channel, height, width,
//                            nntrainer::Tformat::NCHW,
//                            nntrainer::Tdatatype::FP16);

//   EXPECT_THROW({ input.dequantize(output, 1); }, std::invalid_argument);
// }

// /**
//  * @brief dequantize qint8 tensor to fp16
//  */
// TEST(nntrainer_Tensor, dequantize_04_p) {
//   int batch = 1;
//   int channel = 3;
//   int height = 4;
//   int width = 5;

//   nntrainer::Tensor input(
//     batch, channel, height, width,
//     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8});
//   GEN_TEST_INPUT(input, i * (batch * height) + j * (width) + k + 1);

//   input.setScaleFactorsFP16({static_cast<_FP16>(1.5),
//   static_cast<_FP16>(1.0),
//                              static_cast<_FP16>(0.5)});
//   input.setZeroPoints({0, 0, 0});

//   nntrainer::Tensor output(
//     {1, 3, 4, 5, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
//     true);

//   EXPECT_NO_THROW({ input.dequantize(output, 1); });

//   _FP16 answer_data[] = {
//     static_cast<_FP16>(1.5), static_cast<_FP16>(1.5),
//     static_cast<_FP16>(1.5), static_cast<_FP16>(1.5),
//     static_cast<_FP16>(1.5), static_cast<_FP16>(3), static_cast<_FP16>(3),
//     static_cast<_FP16>(3),   static_cast<_FP16>(3), static_cast<_FP16>(3),
//     static_cast<_FP16>(4.5), static_cast<_FP16>(4.5),
//     static_cast<_FP16>(4.5), static_cast<_FP16>(4.5),
//     static_cast<_FP16>(4.5), static_cast<_FP16>(6),   static_cast<_FP16>(6),
//     static_cast<_FP16>(6), static_cast<_FP16>(6),   static_cast<_FP16>(6),
//     static_cast<_FP16>(6), static_cast<_FP16>(6),   static_cast<_FP16>(6),
//     static_cast<_FP16>(6), static_cast<_FP16>(6),   static_cast<_FP16>(7),
//     static_cast<_FP16>(7), static_cast<_FP16>(7),   static_cast<_FP16>(7),
//     static_cast<_FP16>(7), static_cast<_FP16>(8),   static_cast<_FP16>(8),
//     static_cast<_FP16>(8), static_cast<_FP16>(8),   static_cast<_FP16>(8),
//     static_cast<_FP16>(9), static_cast<_FP16>(9),   static_cast<_FP16>(9),
//     static_cast<_FP16>(9), static_cast<_FP16>(9),   static_cast<_FP16>(5.5),
//     static_cast<_FP16>(5.5), static_cast<_FP16>(5.5),
//     static_cast<_FP16>(5.5), static_cast<_FP16>(5.5), static_cast<_FP16>(6),
//     static_cast<_FP16>(6),   static_cast<_FP16>(6), static_cast<_FP16>(6),
//     static_cast<_FP16>(6),   static_cast<_FP16>(6.5),
//     static_cast<_FP16>(6.5), static_cast<_FP16>(6.5),
//     static_cast<_FP16>(6.5), static_cast<_FP16>(6.5), static_cast<_FP16>(7),
//     static_cast<_FP16>(7), static_cast<_FP16>(7),   static_cast<_FP16>(7),
//     static_cast<_FP16>(7)};

//   nntrainer::Tensor answer(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                 {nntrainer::Tformat::NCHW,
//                                                  nntrainer::Tdatatype::FP16}),
//                            answer_data);

//   EXPECT_EQ(output, answer);
// }

// /**
//  * @brief dequantize qint8 tensor to fp16
//  */
// TEST(nntrainer_Tensor, dequantize_05_p) {
//   size_t batch = 1;
//   size_t channel = 3;
//   size_t height = 4;
//   size_t width = 5;

//   nntrainer::Tensor input(
//     {batch,
//      channel,
//      height,
//      width,
//      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT8}},
//     true, nntrainer::Initializer::ZEROS);
//   nntrainer::Tensor output(batch, channel, height, width,
//                            nntrainer::Tformat::NCHW,
//                            nntrainer::Tdatatype::FP16);

//   // Dequantize by channel
//   EXPECT_NO_THROW(input.setScaleFactorsFP16(
//     {static_cast<_FP16>(2), static_cast<_FP16>(-2),
//     static_cast<_FP16>(-4)}));
//   EXPECT_NO_THROW(input.setZeroPoints({1, 1, 1}));
//   EXPECT_NO_THROW({ input.dequantize(output, 1); });

//   _FP16 answer_data_1[] = {-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
//                            -2, -2, -2, -2, -2, -2, -2, -2, 2,  2,  2,  2,
//                            2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
//                            2,  2,  2,  2,  4,  4,  4,  4,  4,  4,  4,  4,
//                            4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4};

//   nntrainer::Tensor answer1(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                  {nntrainer::Tformat::NCHW,
//                                                   nntrainer::Tdatatype::FP16}),
//                             answer_data_1);

//   EXPECT_EQ(output, answer1);

//   // Dequantize by height

//   EXPECT_NO_THROW(input.setScaleFactorsFP16(
//     {static_cast<_FP16>(4.2), static_cast<_FP16>(2),
//     static_cast<_FP16>(-2),
//      static_cast<_FP16>(-4.8)}));
//   EXPECT_NO_THROW(input.setZeroPoints({1, 1, 1, 1}));
//   EXPECT_NO_THROW({ input.dequantize(output, 2); });

//   _FP16 answer_data_2[] = {static_cast<_FP16>(-4.2),
//   static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8),
//                            static_cast<_FP16>(4.8)};
//   nntrainer::Tensor answer2(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                  {nntrainer::Tformat::NCHW,
//                                                   nntrainer::Tdatatype::FP16}),
//                             answer_data_2);

//   EXPECT_EQ(output, answer2);

//   // Dequantize by width
//   EXPECT_NO_THROW(input.setScaleFactorsFP16(
//     {static_cast<_FP16>(4.2), static_cast<_FP16>(2),
//     static_cast<_FP16>(-2),
//      static_cast<_FP16>(-4), static_cast<_FP16>(8)}));
//   EXPECT_NO_THROW(input.setZeroPoints({1, 1, 1, 1, 1}));
//   EXPECT_NO_THROW({ input.dequantize(output, 3); });

//   _FP16 answer_data_3[] = {static_cast<_FP16>(-4.2),
//   static_cast<_FP16>(-2),
//                            static_cast<_FP16>(2),    static_cast<_FP16>(4),
//                            static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),   static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),   static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),   static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),   static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),   static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),   static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8)};

//   nntrainer::Tensor answer3(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                  {nntrainer::Tformat::NCHW,
//                                                   nntrainer::Tdatatype::FP16}),
//                             answer_data_3);

//   EXPECT_EQ(output, answer3);
// }

// /**
//  * @brief dequantize qint4 tensor
//  */
// TEST(nntrainer_Tensor, dequantize_06_p) {
//   size_t batch = 1;
//   size_t channel = 3;
//   size_t height = 4;
//   size_t width = 5;

//   nntrainer::Tensor input(
//     {batch,
//      channel,
//      height,
//      width,
//      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4}},
//     true, nntrainer::Initializer::ZEROS);
//   nntrainer::Tensor output(batch, channel, height, width,
//                            nntrainer::Tformat::NCHW,
//                            nntrainer::Tdatatype::FP16);
//   // Dequantize by channel
//   EXPECT_NO_THROW(input.setScaleFactorsFP16(
//     {static_cast<_FP16>(2), static_cast<_FP16>(-2),
//     static_cast<_FP16>(-4)}));
//   EXPECT_NO_THROW(input.setZeroPoints({1, 1, 1}));
//   EXPECT_NO_THROW({ input.dequantize(output, 1); });

//   _FP16 answer_data_1[] = {-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
//   -2,
//                            -2, -2, -2, -2, -2, -2, -2, -2, 2,  2,  2,  2,
//                            2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
//                            2,  2,  2,  2,  4,  4,  4,  4,  4,  4,  4,  4,
//                            4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 4};

//   nntrainer::Tensor answer1(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                  {nntrainer::Tformat::NCHW,
//                                                   nntrainer::Tdatatype::FP16}),
//                             answer_data_1);

//   EXPECT_EQ(output, answer1);

//   // Dequantize by height
//   EXPECT_NO_THROW(input.setScaleFactorsFP16(
//     {static_cast<_FP16>(4.2), static_cast<_FP16>(2),
//     static_cast<_FP16>(-2),
//      static_cast<_FP16>(-4)}));
//   EXPECT_NO_THROW(input.setZeroPoints({1, 1, 1, 1}));
//   EXPECT_NO_THROW({ input.dequantize(output, 2); });

//   _FP16 answer_data_2[] = {static_cast<_FP16>(-4.2),
//   static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(4),
//                            static_cast<_FP16>(4), static_cast<_FP16>(4),
//                            static_cast<_FP16>(4),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(4),
//                            static_cast<_FP16>(4), static_cast<_FP16>(4),
//                            static_cast<_FP16>(4),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(4),
//                            static_cast<_FP16>(4), static_cast<_FP16>(4),
//                            static_cast<_FP16>(4)};
//   nntrainer::Tensor answer2(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                  {nntrainer::Tformat::NCHW,
//                                                   nntrainer::Tdatatype::FP16}),
//                             answer_data_2);

//   EXPECT_EQ(output, answer2);

//   // Dequantize by width
//   EXPECT_NO_THROW(input.setScaleFactorsFP16(
//     {static_cast<_FP16>(4.2), static_cast<_FP16>(2),
//     static_cast<_FP16>(-2),
//      static_cast<_FP16>(-4), static_cast<_FP16>(8)}));
//   EXPECT_NO_THROW(input.setZeroPoints({1, 1, 1, 1, 1}));
//   EXPECT_NO_THROW({ input.dequantize(output, 3); });

//   _FP16 answer_data_3[] = {static_cast<_FP16>(-4.2),
//   static_cast<_FP16>(-2),
//                            static_cast<_FP16>(2), static_cast<_FP16>(4),
//                            static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4), static_cast<_FP16>(-8),
//                            static_cast<_FP16>(-4.2),
//                            static_cast<_FP16>(-2), static_cast<_FP16>(2),
//                            static_cast<_FP16>(4),
//                            static_cast<_FP16>(-8)};

//   nntrainer::Tensor answer3(ml::train::TensorDim(batch, channel, height,
//   width,
//                                                  {nntrainer::Tformat::NCHW,
//                                                   nntrainer::Tdatatype::FP16}),
//                             answer_data_3);

//   EXPECT_EQ(output, answer3);
// }

/**
 * @brief fp16 tensor has NaN (16 float vec)
 */
TEST(nntrainer_Tensor, is_valid_01) {
  size_t batch = 1;
  size_t channel = 3;
  size_t height = 4;
  size_t width = 5;

  nntrainer::Tensor input(
    {batch,
     channel,
     height,
     width,
     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ZEROS);

  EXPECT_EQ(input.isValid(), true);

  input.setValue(0, 0, 0, 0, std::nan("1"));

  EXPECT_EQ(input.isValid(), false);

  input.setValue(0, 0, 0, 0, std::numeric_limits<float>::infinity());

  EXPECT_EQ(input.isValid(), false);

  input.setValue(0, 0, 0, 0, -std::numeric_limits<float>::infinity());

  EXPECT_EQ(input.isValid(), false);
}

/**
 * @brief fp16 tensor has NaN (8 float vec)
 */
TEST(nntrainer_Tensor, is_valid_02) {
  size_t batch = 1;
  size_t channel = 2;
  size_t height = 2;
  size_t width = 2;

  nntrainer::Tensor input(
    {batch,
     channel,
     height,
     width,
     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ZEROS);

  EXPECT_EQ(input.isValid(), true);

  input.setValue(0, 0, 0, 0, std::nan("1"));

  EXPECT_EQ(input.isValid(), false);

  input.setValue(0, 0, 0, 0, std::numeric_limits<float>::infinity());

  EXPECT_EQ(input.isValid(), false);

  input.setValue(0, 0, 0, 0, -std::numeric_limits<float>::infinity());

  EXPECT_EQ(input.isValid(), false);
}

/**
 * @brief fp16 tensor has NaN (1 float)
 */
TEST(nntrainer_Tensor, is_valid_03) {
  size_t batch = 1;
  size_t channel = 1;
  size_t height = 1;
  size_t width = 1;

  nntrainer::Tensor input(
    {batch,
     channel,
     height,
     width,
     {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}},
    true, nntrainer::Initializer::ZEROS);

  EXPECT_EQ(input.isValid(), true);

  input.setValue(0, 0, 0, 0, std::nan("1"));

  EXPECT_EQ(input.isValid(), false);

  input.setValue(0, 0, 0, 0, std::numeric_limits<float>::infinity());

  EXPECT_EQ(input.isValid(), false);

  input.setValue(0, 0, 0, 0, -std::numeric_limits<float>::infinity());

  EXPECT_EQ(input.isValid(), false);
}

/**
 * @brief An FP32 weight whose output channels differ in magnitude, so a
 *        per-channel scale read at the wrong channel shows up as an error.
 */
static nntrainer::Tensor qs4cxDotWeight(unsigned int K, unsigned int N) {
  nntrainer::Tensor w(1, 1, K, N);
  for (unsigned int n = 0; n < N; ++n) {
    const float amp = 0.25f * static_cast<float>(n + 1);
    for (unsigned int k = 0; k < K; ++k)
      w.setValue(
        0, 0, k, n,
        amp * std::sin(1.7f * static_cast<float>(k) + static_cast<float>(n)));
  }
  return w;
}

/**
 * @brief An FP16 activation times a QS4CX weight agrees with the same product
 *        taken in FP32 from the same record.
 *
 * The QS4CX record has exactly one plain nibble layout in this tree -- the one
 * quant_qs4cx_f32() writes: N rows of ceil(K/2) bytes, row-major [N, K], even
 * k in the low nibble, each stored uint4 being int4 + 8. Nothing in the record
 * names it, so the only thing keeping a consumer honest is a round trip
 * through a second, independent reader. Here the reference is the dequantizer
 * reading the very bytes the fp16 GEMM reads; a consumer that walked the
 * record on any other axis, nibble order, or sign encoding would come back
 * with an unrelated product rather than a slightly rounded one.
 */
TEST(nntrainer_Tensor, dot_qs4cx_fp16_activation_p) {
  const unsigned int M = 3;
  const unsigned int K = 16;
  const unsigned int N = 8;

  nntrainer::Tensor w = qs4cxDotWeight(K, N);

  auto quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::QS4CX);
  nntrainer::Tensor q = quantizer->quantize(w, nntrainer::Tdatatype::QS4CX);
  ASSERT_EQ(q.getDataType(), nntrainer::Tdatatype::QS4CX);
  nntrainer::Tensor w_deq =
    quantizer->dequantize(q, nntrainer::Tdatatype::FP32);

  // activations in [0, 1]: an fp16 lhs and the int8 lhs the Arm micro-kernel
  // quantizes it to are then both a fraction of a step away from the value
  // this reference uses
  nntrainer::Tensor a_fp32(1, 1, M, K);
  nntrainer::Tensor a_fp16(
    {1, 1, M, K, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; ++k) {
      const float v = 0.5f + 0.5f * std::sin(0.9f * static_cast<float>(k) +
                                             2.3f * static_cast<float>(m));
      a_fp32.setValue(0, 0, m, k, v);
      a_fp16.setValue(0, 0, m, k, v);
    }
  }

  nntrainer::Tensor ref = a_fp32.dot(w_deq);
  ASSERT_EQ(ref.height(), M);
  ASSERT_EQ(ref.width(), N);

  nntrainer::Tensor out(
    {1, 1, M, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  out.setZero();
  a_fp16.dot(q, out, false, false, 0.0f);

  float peak = 0.0f;
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n)
      peak = std::max(peak, std::fabs(ref.getValue(0, 0, m, n)));
  // a product that came out all zeros would pass every bound below
  ASSERT_GT(peak, 0.5f);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const float want = ref.getValue(0, 0, m, n);
      const float got = static_cast<float>(out.getValue<_FP16>(0, 0, m, n));
      EXPECT_NEAR(got, want, 0.02f + 0.05f * std::fabs(want))
        << " at m=" << m << ", n=" << n;
    }
  }
}

/**
 * @brief beta scales what the output already holds instead of being dropped
 */
TEST(nntrainer_Tensor, dot_qs4cx_fp16_activation_beta_p) {
  const unsigned int M = 2;
  const unsigned int K = 16;
  const unsigned int N = 8;

  auto quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::QS4CX);
  nntrainer::Tensor q =
    quantizer->quantize(qs4cxDotWeight(K, N), nntrainer::Tdatatype::QS4CX);

  nntrainer::Tensor a(
    {1, 1, M, K, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int k = 0; k < K; ++k)
      a.setValue(0, 0, m, k,
                 0.5f + 0.5f * std::sin(0.9f * static_cast<float>(k) +
                                        2.3f * static_cast<float>(m)));

  nntrainer::Tensor plain(
    {1, 1, M, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  plain.setZero();
  a.dot(q, plain, false, false, 0.0f);

  nntrainer::Tensor accumulated(
    {1, 1, M, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n)
      accumulated.setValue(0, 0, m, n, 1.0f);
  a.dot(q, accumulated, false, false, 1.0f);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const float base = static_cast<float>(plain.getValue<_FP16>(0, 0, m, n));
      const float got =
        static_cast<float>(accumulated.getValue<_FP16>(0, 0, m, n));
      EXPECT_NEAR(got, base + 1.0f, 0.02f + 0.01f * std::fabs(base))
        << " at m=" << m << ", n=" << n;
    }
  }
}

/**
 * @brief a QS4CX weight has no transposed form to ask for
 */
TEST(nntrainer_Tensor, dot_qs4cx_fp16_activation_trans_n) {
  const unsigned int M = 2;
  const unsigned int K = 16;
  const unsigned int N = 8;

  auto quantizer =
    nntrainer::Quantization::createQuantizer(nntrainer::QScheme::QS4CX);
  nntrainer::Tensor q =
    quantizer->quantize(qs4cxDotWeight(K, N), nntrainer::Tdatatype::QS4CX);

  nntrainer::Tensor a(
    {1, 1, M, K, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  a.setZero();
  nntrainer::Tensor out(
    {1, 1, M, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16}}, true);
  out.setZero();

  EXPECT_THROW(a.dot(q, out, false, true, 0.0f), std::invalid_argument);
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
