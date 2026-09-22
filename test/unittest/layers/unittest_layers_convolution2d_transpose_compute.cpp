// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * @file unittest_layers_convolution2d_transpose_compute.cpp
 * @date 22 September 2026
 * @brief Numeric tests for the Conv2DTranspose layer im2col/col2im index math
 * @see	https://github.com/nntrainer/nntrainer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <conv2d_transpose_layer.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <tensor.h>
#include <var_grad.h>
#include <weight.h>

namespace {

/**
 * @brief Owns every tensor a Conv2DTranspose layer needs and drives it
 *
 * The golden test harness reads all tensors from a golden file, which does not
 * exist for this layer. This helper instead lets a test set the input and the
 * kernel explicitly and check the result against a hand computed value.
 */
class Conv2DTransposeRunner {
public:
  Conv2DTransposeRunner(const std::vector<std::string> &props,
                        const nntrainer::TensorDim &input_dim) :
    layer(nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>(props)) {
    nntrainer::InitLayerContext init({input_dim}, {true}, false,
                                     "conv2d_transpose_compute");
    layer->finalize(init);

    weights.reserve(init.getWeightsSpec().size());
    for (auto &spec : init.getWeightsSpec()) {
      weights.emplace_back(spec, true);
      weights.back().getVariableRef().setZero();
      weights.back().getGradientRef().setZero();
    }

    inputs.reserve(1);
    inputs.emplace_back(input_dim, nntrainer::Initializer::ZEROS, true, true,
                        "input");

    outputs.reserve(init.getOutSpecs().size());
    for (auto &spec : init.getOutSpecs()) {
      outputs.emplace_back(spec.variable_spec.dim,
                           nntrainer::Initializer::ZEROS, true, true, "output");
    }

    tensors.reserve(init.getTensorsSpec().size());
    for (auto &spec : init.getTensorsSpec()) {
      tensors.emplace_back(spec, true);
    }

    auto view = [](auto &owned) {
      using ptr_type = typename std::decay_t<decltype(owned)>::value_type *;
      std::vector<ptr_type> ret;
      ret.reserve(owned.size());
      for (auto &e : owned) {
        ret.push_back(&e);
      }
      return ret;
    };

    context = std::make_unique<nntrainer::RunLayerContext>(
      "conv2d_transpose_compute", true, 0.0f, false, 1.0, nullptr, false,
      view(weights), view(inputs), view(outputs), view(tensors));
  }

  nntrainer::Tensor &getInput() { return inputs[0].getVariableRef(); }
  nntrainer::Tensor &getInputGrad() { return inputs[0].getGradientRef(); }
  nntrainer::Tensor &getOutput() { return outputs[0].getVariableRef(); }
  nntrainer::Tensor &getOutputGrad() { return outputs[0].getGradientRef(); }
  nntrainer::Tensor &getKernel() { return weights[0].getVariableRef(); }
  nntrainer::Tensor &getKernelGrad() { return weights[0].getGradientRef(); }
  nntrainer::Tensor &getBiasGrad() { return weights[1].getGradientRef(); }

  void forwarding() { layer->forwarding(*context, true); }
  void calcDerivative() { layer->calcDerivative(*context); }
  void calcGradient() { layer->calcGradient(*context); }

private:
  std::unique_ptr<nntrainer::Layer> layer;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs;
  std::vector<nntrainer::Var_Grad> outputs;
  std::vector<nntrainer::Var_Grad> tensors;
  std::unique_ptr<nntrainer::RunLayerContext> context;
};

void fillTensor(nntrainer::Tensor &t, const std::vector<float> &values) {
  ASSERT_EQ(t.size(), values.size());
  unsigned int i = 0;
  for (unsigned int b = 0; b < t.batch(); ++b) {
    for (unsigned int c = 0; c < t.channel(); ++c) {
      for (unsigned int h = 0; h < t.height(); ++h) {
        for (unsigned int w = 0; w < t.width(); ++w) {
          t.setValue(b, c, h, w, values[i++]);
        }
      }
    }
  }
}

void expectTensorEq(const nntrainer::Tensor &actual,
                    const std::vector<float> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (unsigned int i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(actual.getValue<float>(i), expected[i]) << "index " << i;
  }
}

} // namespace

/**
 * @brief kernel larger than the input is the usual upsampling configuration
 *
 * im2col_transpose used to bound the reconstructed input coordinate by the
 * kernel size, so every coordinate in [in_height, k_height) was read past the
 * input slice.
 */
TEST(Convolution2DTranspose, forwarding_kernel_larger_than_input) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=3,3"},
                               nntrainer::TensorDim({1, 1, 1, 1}));

  runner.getInput().setValue(2.0f);
  fillTensor(runner.getKernel(), {1, 2, 3, 4, 5, 6, 7, 8, 9});

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().height(), 3u);
  EXPECT_EQ(runner.getOutput().width(), 3u);
  expectTensorEq(runner.getOutput(), {2, 4, 6, 8, 10, 12, 14, 16, 18});
}

/**
 * @brief input larger than the kernel must not drop the trailing input rows
 */
TEST(Convolution2DTranspose, forwarding_input_larger_than_kernel) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 1, 3, 3}));

  runner.getInput().setValue(1.0f);
  runner.getKernel().setValue(1.0f);

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().height(), 4u);
  EXPECT_EQ(runner.getOutput().width(), 4u);
  expectTensorEq(runner.getOutput(),
                 {1, 2, 2, 1, 2, 4, 4, 2, 2, 4, 4, 2, 1, 2, 2, 1});
}

/**
 * @brief a non square kernel must size the output width by the kernel width
 */
TEST(Convolution2DTranspose, forwarding_non_square_kernel) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,3"},
                               nntrainer::TensorDim({1, 1, 2, 2}));

  runner.getInput().setValue(1.0f);
  runner.getKernel().setValue(1.0f);

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().height(), 3u);
  EXPECT_EQ(runner.getOutput().width(), 4u);
  expectTensorEq(runner.getOutput(), {1, 2, 2, 1, 2, 4, 4, 2, 1, 2, 2, 1});
}

/**
 * @brief strided upsampling keeps every input pixel in its own output block
 */
TEST(Convolution2DTranspose, forwarding_stride) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,2", "stride=2,2"},
                               nntrainer::TensorDim({1, 1, 2, 2}));

  fillTensor(runner.getInput(), {1, 2, 3, 4});
  runner.getKernel().setValue(1.0f);

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().height(), 4u);
  EXPECT_EQ(runner.getOutput().width(), 4u);
  expectTensorEq(runner.getOutput(),
                 {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4});
}

/**
 * @brief dilation spreads the kernel taps over the input coordinates
 */
TEST(Convolution2DTranspose, forwarding_dilation) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,2", "dilation=2,2"},
                               nntrainer::TensorDim({1, 1, 3, 3}));

  runner.getInput().setValue(1.0f);
  runner.getKernel().setValue(1.0f);

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().height(), 5u);
  EXPECT_EQ(runner.getOutput().width(), 5u);
  expectTensorEq(runner.getOutput(), {1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 4,
                                      2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1});
}

/**
 * @brief padding crops the reconstructed output on every side
 */
TEST(Convolution2DTranspose, forwarding_padding) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=3,3", "padding=1,1"},
                               nntrainer::TensorDim({1, 1, 3, 3}));

  runner.getInput().setValue(1.0f);
  runner.getKernel().setValue(1.0f);

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().height(), 3u);
  EXPECT_EQ(runner.getOutput().width(), 3u);
  expectTensorEq(runner.getOutput(), {4, 6, 4, 6, 9, 6, 4, 6, 4});
}

/**
 * @brief col2im_transpose used to accumulate past the outgoing derivative
 */
TEST(Convolution2DTranspose, calcDerivative_kernel_larger_than_input) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=3,3"},
                               nntrainer::TensorDim({1, 1, 1, 1}));

  runner.getInput().setValue(1.0f);
  fillTensor(runner.getKernel(), {1, 2, 3, 4, 5, 6, 7, 8, 9});
  runner.getOutputGrad().setValue(1.0f);

  runner.calcDerivative();

  EXPECT_EQ(runner.getInputGrad().height(), 1u);
  EXPECT_EQ(runner.getInputGrad().width(), 1u);
  expectTensorEq(runner.getInputGrad(), {45});
}

/**
 * @brief the outgoing derivative of a larger input keeps its trailing rows
 */
TEST(Convolution2DTranspose, calcDerivative_input_larger_than_kernel) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 1, 3, 3}));

  runner.getInput().setValue(1.0f);
  runner.getKernel().setValue(1.0f);
  runner.getOutputGrad().setValue(1.0f);

  runner.calcDerivative();

  expectTensorEq(runner.getInputGrad(), {4, 4, 4, 4, 4, 4, 4, 4, 4});
}

/**
 * @brief calcGradient shares im2col_transpose with forwarding
 */
TEST(Convolution2DTranspose, calcGradient_kernel_larger_than_input) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 1, 1, 1}));

  runner.getInput().setValue(3.0f);
  fillTensor(runner.getOutputGrad(), {1, 2, 3, 4});

  runner.calcGradient();

  expectTensorEq(runner.getKernelGrad(), {3, 6, 9, 12});
  expectTensorEq(runner.getBiasGrad(), {10});
}

/**
 * @brief calcGradient keeps the trailing input rows of a larger input
 */
TEST(Convolution2DTranspose, calcGradient_input_larger_than_kernel) {
  Conv2DTransposeRunner runner({"filters=1", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 1, 3, 3}));

  runner.getInput().setValue(1.0f);
  runner.getOutputGrad().setValue(1.0f);

  runner.calcGradient();

  expectTensorEq(runner.getKernelGrad(), {9, 9, 9, 9});
  expectTensorEq(runner.getBiasGrad(), {16});
}

/**
 * @brief every input channel and every filter keeps its own kernel slice
 */
TEST(Convolution2DTranspose, forwarding_multiple_channels_and_filters) {
  Conv2DTransposeRunner runner({"filters=2", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 2, 1, 1}));

  fillTensor(runner.getInput(), {1, 10});
  fillTensor(runner.getKernel(),
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  runner.forwarding();

  EXPECT_EQ(runner.getOutput().channel(), 2u);
  EXPECT_EQ(runner.getOutput().height(), 2u);
  EXPECT_EQ(runner.getOutput().width(), 2u);
  expectTensorEq(runner.getOutput(), {51, 62, 73, 84, 139, 150, 161, 172});
}

/**
 * @brief the outgoing derivative sums every filter back onto its own channel
 */
TEST(Convolution2DTranspose, calcDerivative_multiple_channels_and_filters) {
  Conv2DTransposeRunner runner({"filters=2", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 2, 1, 1}));

  fillTensor(runner.getInput(), {1, 10});
  fillTensor(runner.getKernel(),
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  runner.getOutputGrad().setValue(1.0f);

  runner.calcDerivative();

  EXPECT_EQ(runner.getInputGrad().channel(), 2u);
  expectTensorEq(runner.getInputGrad(), {52, 84});
}

/**
 * @brief the weight gradient keeps every filter and channel slice apart
 */
TEST(Convolution2DTranspose, calcGradient_multiple_channels_and_filters) {
  Conv2DTransposeRunner runner({"filters=2", "kernel_size=2,2"},
                               nntrainer::TensorDim({1, 2, 1, 1}));

  fillTensor(runner.getInput(), {1, 10});
  fillTensor(runner.getOutputGrad(), {1, 2, 3, 4, 5, 6, 7, 8});

  runner.calcGradient();

  expectTensorEq(runner.getKernelGrad(),
                 {1, 2, 3, 4, 10, 20, 30, 40, 5, 6, 7, 8, 50, 60, 70, 80});
  expectTensorEq(runner.getBiasGrad(), {10, 26});
}

/**
 * @brief the layer accepts exactly one input
 */
TEST(Convolution2DTranspose, finalize_multiple_inputs_n) {
  auto layer = nntrainer::createLayer<nntrainer::Conv2DTransposeLayer>(
    {"filters=1", "kernel_size=2,2"});

  nntrainer::TensorDim dim({1, 1, 3, 3});
  nntrainer::InitLayerContext context({dim, dim}, {true}, false,
                                      "conv2d_transpose_compute_n");

  EXPECT_THROW(layer->finalize(context), std::invalid_argument);
}
