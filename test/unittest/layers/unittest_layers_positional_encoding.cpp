// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_layers_positional_encoding.cpp
 * @date 24 August 2022
 * @brief PositionalEncodingLayer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <layers_common_tests.h>
#include <positional_encoding_layer.h>
#include <var_grad.h>

auto semantic_positional_encoding = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  nntrainer::PositionalEncodingLayer::type, {"max_timestep=10"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

INSTANTIATE_TEST_SUITE_P(PositionalEncoding, LayerSemantics,
                         ::testing::Values(semantic_positional_encoding));

auto positional_encoding_partial = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  {"max_timestep=10"}, "3:1:7:6", "positional_encoding_partial.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto positional_encoding = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  {"max_timestep=10"}, "3:1:10:6", "positional_encoding.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

INSTANTIATE_TEST_SUITE_P(PositionalEncoding, LayerGoldenTest,
                         ::testing::Values(positional_encoding_partial,
                                           positional_encoding));

/**
 * @brief the model reallocates the tensor pool before every inference(), so
 * forwarding must not rely on context tensors keeping their contents
 * @param format tensor format
 * @param format_str format as a tensor_type string
 * @param dtype weight and activation data type
 * @param dtype_str dtype as a tensor_type string
 * @param max_abs_err maximum allowed absolute error
 */
static void expectForwardingAfterTensorReallocation(
  ml::train::TensorDim::Format format, const std::string &format_str,
  ml::train::TensorDim::DataType dtype, const std::string &dtype_str,
  float max_abs_err) {
  constexpr unsigned int seq_len = 5, model_dim = 6;
  nntrainer::PositionalEncodingLayer layer;
  layer.setProperty({"max_timestep=7"});

  nntrainer::TensorDim in_dim(1, 1, seq_len, model_dim);
  in_dim.setFormat(format);
  in_dim.setDataType(dtype);
  nntrainer::InitLayerContext init_context(
    {in_dim}, {true}, false, "pe", "", 0.0, {format_str, dtype_str, dtype_str});
  layer.finalize(init_context);

  nntrainer::Var_Grad in(init_context.getInputDimensions()[0],
                         nntrainer::Initializer::ZEROS, false, true, "in");
  nntrainer::Var_Grad out(init_context.getOutSpecs()[0].variable_spec.dim,
                          nntrainer::Initializer::NONE, false, true, "out");
  std::vector<nntrainer::Var_Grad> tensors;
  for (auto &spec : init_context.getTensorsSpec())
    tensors.emplace_back(spec, true);
  std::vector<nntrainer::Var_Grad *> tensor_views;
  for (auto &t : tensors)
    tensor_views.push_back(&t);

  nntrainer::RunLayerContext rc("pe", false, 0.0f, false, 1.0f, nullptr, false,
                                {}, {&in}, {&out}, tensor_views);

  for (int run = 0; run < 2; ++run) {
    SCOPED_TRACE("forwarding #" + std::to_string(run));
    /** simulate a fresh tensor pool whose memory holds garbage */
    for (unsigned int i = 0; i < rc.getNumTensors(); ++i)
      rc.getTensor(i).setValue(std::numeric_limits<float>::quiet_NaN());

    layer.forwarding(rc, false);
    nntrainer::Tensor output =
      rc.getOutput(0).clone(ml::train::TensorDim::DataType::FP32);

    float max_err = 0.0f;
    for (unsigned int pos = 0; pos < seq_len; ++pos) {
      for (unsigned int i = 0; i < model_dim; ++i) {
        float angle =
          pos / std::pow(10000.0f, ((i >> 1) << 1) / (float)model_dim);
        float expected = (i & 1) ? std::cos(angle) : std::sin(angle);
        float err = std::abs(output.getValue(0, 0, pos, i) - expected);
        max_err = std::isnan(err) ? INFINITY : std::max(max_err, err);
      }
    }
    EXPECT_LE(max_err, max_abs_err);
  }
}

/**
 * @brief FP32 table refilled after the tensor pool is reallocated
 */
TEST(PositionalEncoding, forwarding_after_tensor_reallocation) {
  expectForwardingAfterTensorReallocation(
    ml::train::TensorDim::Format::NCHW, "NCHW",
    ml::train::TensorDim::DataType::FP32, "FP32", 1e-6f);
}

/**
 * @brief NHWC table refilled after the tensor pool is reallocated
 */
TEST(PositionalEncoding, forwarding_after_tensor_reallocation_nhwc) {
  expectForwardingAfterTensorReallocation(
    ml::train::TensorDim::Format::NHWC, "NHWC",
    ml::train::TensorDim::DataType::FP32, "FP32", 1e-6f);
}

#ifdef ENABLE_FP16
/**
 * @brief FP16 table refilled after the tensor pool is reallocated
 */
TEST(PositionalEncoding, forwarding_after_tensor_reallocation_fp16) {
  expectForwardingAfterTensorReallocation(
    ml::train::TensorDim::Format::NCHW, "NCHW",
    ml::train::TensorDim::DataType::FP16, "FP16", 1e-3f);
}
#endif

/**
 * @brief an input longer than max_timestep must be rejected at finalize
 */
TEST(PositionalEncoding, finalize_seq_longer_than_max_timestep_n) {
  nntrainer::PositionalEncodingLayer layer;
  layer.setProperty({"max_timestep=7"});

  nntrainer::InitLayerContext init_context({nntrainer::TensorDim(1, 1, 8, 6)},
                                           {true}, false, "pe");
  EXPECT_THROW(layer.finalize(init_context), std::invalid_argument);
}

#ifdef ENABLE_FP16
auto positional_encoding_partial_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  {"max_timestep=10"}, "3:1:7:6",
  "positional_encoding_partial_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto positional_encoding_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::PositionalEncodingLayer>,
  {"max_timestep=10"}, "3:1:10:6", "positional_encoding_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(PositionalEncoding16, LayerGoldenTest,
                     ::testing::Values(positional_encoding_partial_w16a16,
                                       positional_encoding_w16a16));
#endif
