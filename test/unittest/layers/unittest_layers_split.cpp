// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_split.cpp
 * @date 12 June 2021
 * @brief Split Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <layers_common_tests.h>
#include <split_layer.h>
#include <var_grad.h>
#include <weight.h>

auto semantic_split = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SplitLayer>, nntrainer::SplitLayer::type,
  {"axis=3"}, LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false,
  1);

GTEST_PARAMETER_TEST(Split, LayerSemantics, ::testing::Values(semantic_split));

namespace {

constexpr unsigned int BATCH = 2;
constexpr unsigned int CHANNEL = 2;
constexpr unsigned int HEIGHT = 4;
constexpr unsigned int WIDTH = 6;
constexpr float SENTINEL = -1.0f;

/**
 * @brief flat index of an element of the input tensor, used as its value so
 * that every element of every chunk is uniquely identifiable
 */
float inputValue(const nntrainer::TensorDim &dim, unsigned int b,
                 unsigned int c, unsigned int h, unsigned int w) {
  return static_cast<float>(
    ((b * dim.channel() + c) * dim.height() + h) * dim.width() + w);
}

/**
 * @brief A split layer together with the tensors it runs on.
 *
 * The Var_Grad storage has to outlive the run context that holds pointers into
 * it, so both are owned here. The input variable is filled with inputValue().
 */
class SplitFixture {
public:
  SplitFixture(const nntrainer::TensorDim &in_dim, unsigned int axis,
               unsigned int split_number) {
    layer.setProperty({"axis=" + std::to_string(axis),
                       "split_number=" + std::to_string(split_number)});

    nntrainer::InitLayerContext init_ctx(
      {in_dim}, std::vector<bool>(split_number, true), false, "split");
    layer.finalize(init_ctx);

    input = std::make_unique<nntrainer::Var_Grad>(
      in_dim, nntrainer::Initializer::NONE, true, true, "split_input");

    outputs.reserve(split_number);
    for (auto &spec : init_ctx.getOutSpecs()) {
      outputs.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                           true, true,
                           "split_output" + std::to_string(outputs.size()));
    }

    in_view.push_back(input.get());
    out_view.reserve(outputs.size());
    for (auto &output : outputs) {
      out_view.push_back(&output);
    }

    run_ctx = std::make_unique<nntrainer::RunLayerContext>(
      "split", true, 0.0f, false, 1.0f, nullptr, false,
      std::vector<nntrainer::Weight *>{}, in_view, out_view,
      std::vector<nntrainer::Var_Grad *>{});

    nntrainer::Tensor &in_var = input->getVariableRef();
    for (unsigned int b = 0; b < in_dim.batch(); ++b)
      for (unsigned int c = 0; c < in_dim.channel(); ++c)
        for (unsigned int h = 0; h < in_dim.height(); ++h)
          for (unsigned int w = 0; w < in_dim.width(); ++w)
            in_var.setValue(b, c, h, w, inputValue(in_dim, b, c, h, w));
  }

  void fillOutputs(float value) {
    for (auto &output : outputs) {
      output.getVariableRef().setValue(value);
    }
  }

  nntrainer::SplitLayer layer;
  std::unique_ptr<nntrainer::Var_Grad> input;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad *> in_view;
  std::vector<nntrainer::Var_Grad *> out_view;
  std::unique_ptr<nntrainer::RunLayerContext> run_ctx;
};

nntrainer::TensorDim makeDim(ml::train::TensorDim::DataType data_type,
                             unsigned int channel) {
  return nntrainer::TensorDim(BATCH, channel, HEIGHT, WIDTH,
                              ml::train::TensorDim::TensorType(
                                ml::train::TensorDim::Format::NCHW, data_type));
}

/**
 * @brief check that every element of every output chunk holds its own slice of
 * the input
 */
void expectChunkValues(nntrainer::RunLayerContext &run_ctx,
                       const nntrainer::TensorDim &in_dim, unsigned int axis,
                       unsigned int split_number) {
  const unsigned int split_size = in_dim.getTensorDim(axis) / split_number;
  const nntrainer::TensorDim out_dim = run_ctx.getOutput(0).getDim();

  for (unsigned int idx = 0; idx < split_number; ++idx) {
    const nntrainer::Tensor out =
      run_ctx.getOutput(idx).clone(ml::train::TensorDim::DataType::FP32);

    for (unsigned int b = 0; b < out_dim.batch(); ++b)
      for (unsigned int c = 0; c < out_dim.channel(); ++c)
        for (unsigned int h = 0; h < out_dim.height(); ++h)
          for (unsigned int w = 0; w < out_dim.width(); ++w) {
            const unsigned int in_c = (axis == 1) ? idx * split_size + c : c;
            const unsigned int in_h = (axis == 2) ? idx * split_size + h : h;
            const unsigned int in_w = (axis == 3) ? idx * split_size + w : w;

            EXPECT_FLOAT_EQ(out.getValue<float>(b, c, h, w),
                            inputValue(in_dim, b, in_c, in_h, in_w))
              << "chunk " << idx << " at " << b << "," << c << "," << h << ","
              << w;
          }
  }
}

/**
 * @brief Run forwarding() and calcDerivative() and check every element of
 * every chunk in both directions.
 *
 * Buffers are sized in elements, so the layer must derive its copy sizes from
 * the tensor data type. Hard-coding sizeof(float) makes the FP16 case copy
 * twice the number of bytes each chunk owns, which both corrupts the values
 * and runs off the end of the input and the output buffers.
 */
void checkSplitRoundTrip(ml::train::TensorDim::DataType data_type,
                         unsigned int axis, unsigned int split_number) {
  const nntrainer::TensorDim in_dim = makeDim(data_type, CHANNEL);
  const unsigned int split_size = in_dim.getTensorDim(axis) / split_number;

  SplitFixture f(in_dim, axis, split_number);
  f.layer.forwarding(*f.run_ctx, false);
  expectChunkValues(*f.run_ctx, in_dim, axis, split_number);

  /** the incoming derivative of a chunk carries the value of the chunk */
  const nntrainer::TensorDim out_dim = f.run_ctx->getOutput(0).getDim();
  for (unsigned int idx = 0; idx < split_number; ++idx) {
    nntrainer::Tensor &out_grad = f.outputs[idx].getGradientRef();

    for (unsigned int b = 0; b < out_dim.batch(); ++b)
      for (unsigned int c = 0; c < out_dim.channel(); ++c)
        for (unsigned int h = 0; h < out_dim.height(); ++h)
          for (unsigned int w = 0; w < out_dim.width(); ++w) {
            const unsigned int in_c = (axis == 1) ? idx * split_size + c : c;
            const unsigned int in_h = (axis == 2) ? idx * split_size + h : h;
            const unsigned int in_w = (axis == 3) ? idx * split_size + w : w;
            out_grad.setValue(b, c, h, w,
                              inputValue(in_dim, b, in_c, in_h, in_w));
          }
  }

  f.layer.calcDerivative(*f.run_ctx);

  const nntrainer::Tensor in_grad = f.run_ctx->getOutgoingDerivative(0).clone(
    ml::train::TensorDim::DataType::FP32);

  for (unsigned int b = 0; b < BATCH; ++b)
    for (unsigned int c = 0; c < CHANNEL; ++c)
      for (unsigned int h = 0; h < HEIGHT; ++h)
        for (unsigned int w = 0; w < WIDTH; ++w)
          EXPECT_FLOAT_EQ(in_grad.getValue<float>(b, c, h, w),
                          inputValue(in_dim, b, c, h, w))
            << "derivative at " << b << "," << c << "," << h << "," << w;
}

/**
 * @brief Run incremental_forwarding() over its axis=3 memcpy fast path and
 * check every copied element, then check that a partial step leaves the rows
 * it does not cover untouched.
 *
 * @note the fast path addresses channel 0 only, so it is driven with the
 * single-channel [b, 1, seq, w] shape it is written for.
 */
void checkSplitIncrementalForwarding(ml::train::TensorDim::DataType data_type,
                                     unsigned int split_number) {
  const nntrainer::TensorDim in_dim = makeDim(data_type, 1);
  const unsigned int split_w = WIDTH / split_number;

  SplitFixture f(in_dim, 3, split_number);

  f.fillOutputs(SENTINEL);
  f.layer.incremental_forwarding(*f.run_ctx, 0, HEIGHT, false);
  expectChunkValues(*f.run_ctx, in_dim, 3, split_number);

  f.fillOutputs(SENTINEL);
  f.layer.incremental_forwarding(*f.run_ctx, 0, 1, false);

  for (unsigned int idx = 0; idx < split_number; ++idx) {
    const nntrainer::Tensor out =
      f.run_ctx->getOutput(idx).clone(ml::train::TensorDim::DataType::FP32);

    for (unsigned int b = 0; b < BATCH; ++b)
      for (unsigned int h = 0; h < HEIGHT; ++h)
        for (unsigned int w = 0; w < split_w; ++w) {
          const float expected =
            (h == 0) ? inputValue(in_dim, b, 0, 0, idx * split_w + w)
                     : SENTINEL;
          EXPECT_FLOAT_EQ(out.getValue<float>(b, 0, h, w), expected)
            << "chunk " << idx << " at " << b << ",0," << h << "," << w;
        }
  }
}

/**
 * @brief incremental_forwarding() delegates to forwarding() for any axis but
 * the width, so the result must be the full split
 */
void checkSplitIncrementalFallback(ml::train::TensorDim::DataType data_type,
                                   unsigned int axis,
                                   unsigned int split_number) {
  const nntrainer::TensorDim in_dim = makeDim(data_type, CHANNEL);

  SplitFixture f(in_dim, axis, split_number);

  f.fillOutputs(SENTINEL);
  f.layer.incremental_forwarding(*f.run_ctx, 0, 1, false);
  expectChunkValues(*f.run_ctx, in_dim, axis, split_number);
}

} // namespace

TEST(SplitLayer, forwardingAxis1Fp32) {
  checkSplitRoundTrip(ml::train::TensorDim::DataType::FP32, 1, CHANNEL);
}

TEST(SplitLayer, forwardingAxis2Fp32) {
  checkSplitRoundTrip(ml::train::TensorDim::DataType::FP32, 2, 2);
}

TEST(SplitLayer, forwardingAxis3Fp32) {
  checkSplitRoundTrip(ml::train::TensorDim::DataType::FP32, 3, 3);
}

TEST(SplitLayer, incrementalForwardingAxis3Fp32) {
  checkSplitIncrementalForwarding(ml::train::TensorDim::DataType::FP32, 3);
}

TEST(SplitLayer, incrementalForwardingFallbackAxis2Fp32) {
  checkSplitIncrementalFallback(ml::train::TensorDim::DataType::FP32, 2, 2);
}

#ifdef ENABLE_FP16
TEST(SplitLayer, forwardingAxis1Fp16) {
  checkSplitRoundTrip(ml::train::TensorDim::DataType::FP16, 1, CHANNEL);
}

TEST(SplitLayer, forwardingAxis2Fp16) {
  checkSplitRoundTrip(ml::train::TensorDim::DataType::FP16, 2, 2);
}

TEST(SplitLayer, forwardingAxis3Fp16) {
  checkSplitRoundTrip(ml::train::TensorDim::DataType::FP16, 3, 3);
}

TEST(SplitLayer, incrementalForwardingAxis3Fp16) {
  checkSplitIncrementalForwarding(ml::train::TensorDim::DataType::FP16, 3);
}

TEST(SplitLayer, incrementalForwardingFallbackAxis2Fp16) {
  checkSplitIncrementalFallback(ml::train::TensorDim::DataType::FP16, 2, 2);
}
#endif

TEST(SplitLayer, notDivisibleSplitNumber_n) {
  const nntrainer::TensorDim in_dim(BATCH, CHANNEL, HEIGHT, WIDTH);

  nntrainer::SplitLayer layer;
  layer.setProperty({"axis=3", "split_number=4"});

  nntrainer::InitLayerContext init_ctx({in_dim}, std::vector<bool>(4, true),
                                       false, "split");
  EXPECT_THROW(layer.finalize(init_ctx), std::invalid_argument);
}

TEST(SplitLayer, outputCountMismatch_n) {
  const nntrainer::TensorDim in_dim(BATCH, CHANNEL, HEIGHT, WIDTH);

  nntrainer::SplitLayer layer;
  layer.setProperty({"axis=3", "split_number=3"});

  nntrainer::InitLayerContext init_ctx({in_dim}, std::vector<bool>(2, true),
                                       false, "split");
  EXPECT_THROW(layer.finalize(init_ctx), std::invalid_argument);
}

TEST(SplitLayer, multipleInputs_n) {
  const nntrainer::TensorDim in_dim(BATCH, CHANNEL, HEIGHT, WIDTH);

  nntrainer::SplitLayer layer;
  layer.setProperty({"axis=3", "split_number=3"});

  nntrainer::InitLayerContext init_ctx(
    {in_dim, in_dim}, std::vector<bool>(3, true), false, "split");
  EXPECT_THROW(layer.finalize(init_ctx), std::invalid_argument);
}
