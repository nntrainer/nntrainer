/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_nntrainer_activations.cpp
 * @date        21 July 2020
 * @brief       Unit test for static functions in activation_layer.cpp.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @author      Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <activation_layer.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <nntrainer_test_util.h>

TEST(nntrainer_activation, softmax_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float results[10] = {7.80134161e-05f, 2.12062451e-04f, 5.76445508e-04f,
                       1.56694135e-03f, 4.25938820e-03f, 1.15782175e-02f,
                       3.14728583e-02f, 8.55520989e-02f, 2.32554716e-01f,
                       6.32149258e-01f};

  nntrainer::Tensor T(batch, channel, height, width);
  nntrainer::Tensor Results(batch, channel, height, width);

  GEN_TEST_INPUT(T, (i * (width) + l + 1));

  Results = T.apply(nntrainer::ActiFunc::softmax<float>, Results);
  float *data = Results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], results[i % width], tolerance);
  }
}

TEST(nntrainer_activation, softmax_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor T(batch, channel, height, width);
  nntrainer::Tensor Results(batch, channel, height, 2 * width);
  nntrainer::Tensor Result = Results.getSharedDataTensor(
    nntrainer::TensorDim(batch, channel, height, width), 0, false);

  EXPECT_THROW(T.apply(nntrainer::ActiFunc::softmax<float>, Result),
               std::invalid_argument);
}

TEST(nntrainer_activation, softmax_prime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float results[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (i * (width) + k + 1));

  nntrainer::Tensor softmax_result;
  softmax_result =
    input.apply(nntrainer::ActiFunc::softmax<float>, softmax_result);

  float *data = softmax_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor softmax_prime_result;
  softmax_prime_result =
    nntrainer::ActiFunc::softmaxPrime(softmax_result, softmax_prime_result);

  data = softmax_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], results[i % width], tolerance);
  }
}

TEST(nntrainer_activation, softmax_prime_02_n) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (i * (width) + k + 1));

  nntrainer::Tensor softmax_result;
  softmax_result =
    input.apply(nntrainer::ActiFunc::softmax<float>, softmax_result);

  float *data = softmax_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor softmax_prime_result(batch, channel, height, 2 * width);
  nntrainer::Tensor softmax_prime_result_shared =
    softmax_prime_result.getSharedDataTensor(
      nntrainer::TensorDim(batch, channel, height, width), 0, false);
  EXPECT_THROW(nntrainer::ActiFunc::softmaxPrime(softmax_result,
                                                 softmax_prime_result_shared),
               std::invalid_argument);
}

TEST(nntrainer_activation, softplus_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.51301527f, 0.55435520f, 0.59813887f, 0.64439666f, 0.69314718f,
    0.74439669f, 0.79813886f, 0.85435522f, 0.91301525f, 0.97407699f,
    0.37110066f, 0.43748793f, 0.51301527f, 0.59813887f, 0.69314718f,
    0.79813886f, 0.91301525f, 1.03748798f, 1.17110062f, 1.31326163f,
    0.26328245f, 0.34115386f, 0.43748793f, 0.55435526f, 0.69314718f,
    0.85435528f, 1.03748798f, 1.24115384f, 1.46328247f, 1.70141327f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor softplus_result =
    input.apply<float>(nntrainer::ActiFunc::softplus<float>);

  float *data = softplus_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, softplusPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.40131232f, 0.42555746f, 0.45016599f, 0.47502083f, 0.50000000f,
    0.52497917f, 0.54983401f, 0.57444251f, 0.59868765f, 0.62245935f,
    0.31002554f, 0.35434368f, 0.40131232f, 0.45016599f, 0.50000000f,
    0.54983401f, 0.59868771f, 0.64565635f, 0.68997449f, 0.73105860f,
    0.23147520f, 0.28905049f, 0.35434368f, 0.42555746f, 0.50000000f,
    0.57444257f, 0.64565635f, 0.71094948f, 0.76852477f, 0.81757450f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor softplus_prime_result =
    input.apply<float>(nntrainer::ActiFunc::softplusPrime<float>);

  float *data = softplus_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, sigmoid_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.4013123f, 0.4255575f, 0.450166f,  0.4750208f, 0.5f,       0.5249792f,
    0.549834f,  0.5744425f, 0.5986877f, 0.6224593f, 0.3100255f, 0.3543437f,
    0.4013123f, 0.450166f,  0.5f,       0.549834f,  0.5986877f, 0.6456563f,
    0.6899745f, 0.7310586f, 0.2314752f, 0.2890505f, 0.3543437f, 0.4255575f,
    0.5f,       0.5744425f, 0.6456563f, 0.7109495f, 0.7685248f, 0.8175745f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results =
    input.apply<float>(nntrainer::ActiFunc::sigmoid<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, DISABLED_sigmoidPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    2.40198421e-01f, 2.39014374e-01f, 2.37750225e-01f, 2.36411103e-01f,
    2.35003712e-01f, 2.33536203e-01f, 2.32017974e-01f, 2.30459419e-01f,
    2.28871631e-01f, 2.27266092e-01f, 2.44087699e-01f, 2.42313881e-01f,
    2.40198421e-01f, 2.37750225e-01f, 2.35003712e-01f, 2.32017974e-01f,
    2.28871631e-01f, 2.25654341e-01f, 2.22456864e-01f, 2.19361864e-01f,
    2.46680881e-01f, 2.44849977e-01f, 2.42313881e-01f, 2.39014374e-01f,
    2.35003712e-01f, 2.30459419e-01f, 2.25654341e-01f, 2.2089191e-01f,
    2.16437141e-01f, 2.12472086e-01f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor sigmoid_result =
    input.apply<float>(nntrainer::ActiFunc::sigmoid<float>);
  float *data = sigmoid_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    sigmoid_result.apply<float>(nntrainer::ActiFunc::sigmoidPrime<float>);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, tanhFloat_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    -3.79948962e-01f, -2.91312612e-01f, -1.9737532e-01f,  -9.96679946e-02f,
    0e+00f,           9.96679946e-02f,  1.9737532e-01f,   2.91312612e-01f,
    3.79948962e-01f,  4.62117157e-01f,  -6.6403677e-01f,  -5.37049567e-01f,
    -3.79948962e-01f, -1.9737532e-01f,  0e+00f,           1.9737532e-01f,
    3.79948962e-01f,  5.37049567e-01f,  6.6403677e-01f,   7.61594156e-01f,
    -8.33654607e-01f, -7.1629787e-01f,  -5.37049567e-01f, -2.91312612e-01f,
    0e+00f,           2.91312612e-01f,  5.37049567e-01f,  7.1629787e-01f,
    8.33654607e-01f,  9.05148254e-01f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results =
    input.apply<float>(nntrainer::ActiFunc::tanhFloat<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, DISABLED_tanhFloatPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    0.8684754f, 0.919717f,  0.9620329f, 0.9901317f, 1.0f,       0.9901317f,
    0.9620329f, 0.919717f,  0.8684754f, 0.8135417f, 0.6623883f, 0.7591631f,
    0.8684754f, 0.9620329f, 1.0f,       0.9620329f, 0.8684754f, 0.7591631f,
    0.6623883f, 0.5878168f, 0.5342845f, 0.6222535f, 0.7591631f, 0.919717f,
    1.0f,       0.919717f,  0.7591631f, 0.6222535f, 0.5342845f, 0.4833332f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor tanh_result =
    input.apply<float>(nntrainer::ActiFunc::tanhFloat<float>);
  float *data = tanh_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    tanh_result.apply<float>(nntrainer::ActiFunc::tanhPrime<float>);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, relu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.3f,
                      0.4f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f,
                      0.4f, 0.6f, 0.8f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.3f, 0.6f, 0.9f, 1.2f, 1.5f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor Results =
    input.apply<float>(nntrainer::ActiFunc::relu<float>);

  float *data = Results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, reluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                      1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                      1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor relu_result =
    input.apply<float>(nntrainer::ActiFunc::relu<float>);
  float *data = relu_result.getData();
  ASSERT_NE(nullptr, data);

  nntrainer::Tensor prime_result =
    relu_result.apply<float>(nntrainer::ActiFunc::reluPrime<float>);
  data = prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, swish_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    -0.16052495f, -0.12766725f, -0.0900332f,  -0.04750208f, 0.0f,
    0.05249792f,  0.10996679f,  0.17233276f,  0.23947506f,  0.31122968f,
    -0.24802041f, -0.21260624f, -0.16052495f, -0.0900332f,  0.0f,
    0.10996679f,  0.23947506f,  0.3873938f,   0.5519796f,   0.73105854f,
    -0.27777028f, -0.26014543f, -0.21260624f, -0.12766725f, 0.0f,
    0.17233276f,  0.3873938f,   0.6398545f,   0.9222298f,   1.2263616f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  results = nntrainer::ActiFunc::swish(input, results);

  float *data = results.getData();
  ASSERT_NE(nullptr, data);
  float *indata = input.getData();
  ASSERT_NE(nullptr, indata);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, swishPrime_01_p) {

  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    0.30520803f, 0.35221997f, 0.40066269f, 0.45008320f, 0.50000000f,
    0.54991674f, 0.59933728f, 0.64778000f, 0.69479191f, 0.73996115f,
    0.13889773f, 0.21707317f, 0.30520803f, 0.40066269f, 0.50000000f,
    0.59933728f, 0.69479191f, 0.78292680f, 0.86110222f, 0.92767054f,
    0.01800188f, 0.10410020f, 0.21707317f, 0.35221997f, 0.50000000f,
    0.64778000f, 0.78292680f, 0.89589977f, 0.98199815f, 1.04129410f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  nntrainer::ActiFunc::swish(input, results);

  nntrainer::Tensor prime_results(batch, channel, height, width);
  nntrainer::ActiFunc::swishPrime(input, results, prime_results);

  float *data = prime_results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, gelu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    -0.13783135f, -0.11462659f, -0.08414805f, -0.04601721f, 0.0f,
    0.05398279f,  0.11585195f,  0.18537343f,  0.26216868f,  0.34573120f,
    -0.16948429f, -0.16455182f, -0.13783135f, -0.08414805f, 0.0f,
    0.11585195f,  0.26216868f,  0.43544820f,  0.63051575f,  0.84134471f,
    -0.13808367f, -0.16565408f, -0.16455182f, -0.11462659f, 0.0f,
    0.18537343f,  0.43544820f,  0.73434591f,  1.06191635f,  1.39978909f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  results = nntrainer::ActiFunc::gelu(input, results);

  float *data = results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, gelu_large_tensor_p) {
  constexpr size_t size = 4097;
  nntrainer::Tensor input(1, 1, 1, size);
  nntrainer::Tensor results(1, 1, 1, size);
  std::vector<float> expected(size);

  float *input_data = input.getData<float>();
  for (size_t i = 0; i < size; ++i)
    input_data[i] = static_cast<float>(i % 101) / 10.0f - 5.0f;

  nntrainer::gelu_v2(size, input_data, expected.data());
  nntrainer::ActiFunc::gelu<float>(input, results);

  const float *result_data = results.getData<float>();
  for (size_t i = 0; i < size; ++i)
    EXPECT_FLOAT_EQ(result_data[i], expected[i]);
}

#ifdef ENABLE_FP16
TEST(nntrainer_activation, gelu_fp16_p) {
  constexpr size_t size = 4097;
  const nntrainer::TensorDim::TensorType fp16_type = {
    nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP16};
  nntrainer::Tensor input(nntrainer::TensorDim(1, 1, 1, size, fp16_type));
  nntrainer::Tensor results(nntrainer::TensorDim(1, 1, 1, size, fp16_type));

  _FP16 *input_data = input.getData<_FP16>();
  for (size_t i = 0; i < size; ++i)
    input_data[i] =
      static_cast<_FP16>(static_cast<float>(i % 101) / 10.0f - 5.0f);

  nntrainer::ActiFunc::gelu<_FP16>(input, results);

  constexpr float inv_sqrt_two = 0.70710678118654752440f;
  const _FP16 *result_data = results.getData<_FP16>();
  for (size_t i = 0; i < size; ++i) {
    const float value = static_cast<float>(input_data[i]);
    const float expected =
      0.5f * value * (1.0f + std::erf(value * inv_sqrt_two));
    EXPECT_NEAR(static_cast<float>(result_data[i]), expected, 0.004f);
  }
}
#endif

TEST(nntrainer_activation, geluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.19727029f,  0.26767227f,  0.34253171f, 0.42047682f, 0.5f,
    0.57952315f,  0.65746832f,  0.73232776f, 0.80272973f, 0.86749506f,
    -0.01989788f, 0.07431830f,  0.19727029f, 0.34253171f, 0.5f,
    0.65746832f,  0.80272973f,  0.92568171f, 1.01989794f, 1.08331537f,
    -0.11795351f, -0.05541664f, 0.07431830f, 0.26767227f, 0.5f,
    0.73232776f,  0.92568171f,  1.05541658f, 1.11795354f, 1.12746918f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor results(batch, channel, height, width);
  nntrainer::ActiFunc::gelu(input, results);

  nntrainer::Tensor prime_results(batch, channel, height, width);
  nntrainer::ActiFunc::geluPrime(input, results, prime_results);

  float *data = prime_results.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, elu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    -0.32967997f, -0.25918180f, -0.18126923f, -0.09516257f, 0.0f,
    0.1f,         0.2f,         0.3f,         0.4f,         0.5f,
    -0.55067104f, -0.45118839f, -0.32967997f, -0.18126923f, 0.0f,
    0.2f,         0.4f,         0.6f,         0.8f,         1.0f,
    -0.69880581f, -0.59343034f, -0.45118839f, -0.25918180f, 0.0f,
    0.3f,         0.6f,         0.9f,         1.2f,         1.5f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor elu_result =
    input.apply<float>(nntrainer::ActiFunc::elu<float>);
  float *data = elu_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, eluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {0.67032003f, 0.74081820f, 0.81873077f, 0.90483743f, 1.0f,
                      1.0f,        1.0f,        1.0f,        1.0f,        1.0f,
                      0.44932896f, 0.54881161f, 0.67032003f, 0.81873077f, 1.0f,
                      1.0f,        1.0f,        1.0f,        1.0f,        1.0f,
                      0.30119419f, 0.40656966f, 0.54881161f, 0.74081820f, 1.0f,
                      1.0f,        1.0f,        1.0f,        1.0f,        1.0f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor elu_prime_result =
    input.apply<float>(nntrainer::ActiFunc::eluPrime<float>);
  float *data = elu_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, selu_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    -0.57961011f, -0.45566735f, -0.31868932f, -0.16730525f, 0.00000000f,
    0.10507011f,  0.21014021f,  0.31521031f,  0.42028043f,  0.52535051f,
    -0.96813440f, -0.79323399f, -0.57961011f, -0.31868932f, 0.00000000f,
    0.21014021f,  0.42028043f,  0.63042063f,  0.84056085f,  1.05070102f,
    -1.22856998f, -1.04330945f, -0.79323399f, -0.45566735f, 0.00000000f,
    0.31521031f,  0.63042063f,  0.94563091f,  1.26084125f,  1.57605147f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor selu_result =
    input.apply<float>(nntrainer::ActiFunc::selu<float>);
  float *data = selu_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, seluPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;
  float answer[30] = {
    1.17848921f, 1.30243194f, 1.43940997f, 1.59079409f, 1.75809932f,
    1.05070102f, 1.05070102f, 1.05070102f, 1.05070102f, 1.05070102f,
    0.78996491f, 0.96486533f, 1.17848921f, 1.43940997f, 1.75809932f,
    1.05070102f, 1.05070102f, 1.05070102f, 1.05070102f, 1.05070102f,
    0.52952927f, 0.71478987f, 0.96486533f, 1.30243194f, 1.75809932f,
    1.05070102f, 1.05070102f, 1.05070102f, 1.05070102f, 1.05070102f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor selu_prime_result =
    input.apply<float>(nntrainer::ActiFunc::seluPrime<float>);
  float *data = selu_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, mish_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    -0.18891649f, -0.15113318f, -0.10714479f, -0.05678857f, 0.00000000f,
    0.06317942f,  0.13259898f,  0.20800139f,  0.28903052f,  0.37524524f,
    -0.28396326f, -0.24693604f, -0.18891649f, -0.10714479f, 0.00000000f,
    0.13259898f,  0.28903052f,  0.46613652f,  0.65969974f,  0.86509836f,
    -0.30883580f, -0.29565641f, -0.24693604f, -0.15113318f, 0.00000000f,
    0.20800139f,  0.46613652f,  0.76120591f,  1.07794595f,  1.40337825f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor mish_result =
    input.apply<float>(nntrainer::ActiFunc::mish<float>);
  float *data = mish_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

TEST(nntrainer_activation, mishPrime_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  float answer[30] = {
    0.34757277f,  0.40851089f, 0.47153026f, 0.53570276f, 0.60000002f,
    0.66333681f,  0.72462445f, 0.78282732f, 0.83701748f, 0.88642442f,
    0.13818233f,  0.23496543f, 0.34757277f, 0.47153026f, 0.60000002f,
    0.72462445f,  0.83701748f, 0.93047082f, 1.00125492f, 1.04903626f,
    -0.00200878f, 0.09643580f, 0.23496543f, 0.40851089f, 0.60000002f,
    0.78282732f,  0.93047082f, 1.02791822f, 1.07635069f, 1.08848798f};

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));

  nntrainer::Tensor mish_prime_result =
    input.apply<float>(nntrainer::ActiFunc::mishPrime<float>);
  float *data = mish_prime_result.getData();
  ASSERT_NE(nullptr, data);

  for (int i = 0; i < batch * height * width; ++i) {
    EXPECT_NEAR(data[i], answer[i], tolerance);
  }
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    ml_loge("Failed to init gtest\n");
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    ml_loge("Failed to run test.\n");
  }

  return result;
}
