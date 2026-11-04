// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_layers_unary_derivative.cpp
 * @brief Check unary layer derivatives with non-unit incoming gradients.
 * @author Minsoo Kim <61856979+33modeling@users.noreply.github.com>
 * @bug No known bugs.
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <gtest/gtest.h>

#include <cosine_layer.h>
#include <negative_layer.h>
#include <sine_layer.h>
#include <sqrt_layer.h>
#include <tangent_layer.h>
#include <var_grad.h>

namespace {

enum class UnaryKind { Sqrt, Negative, Sine, Cosine, Tangent };
using UnaryParam = std::tuple<UnaryKind, unsigned int>;

/** @brief Create the actual unary layer used by each regression case. */
std::unique_ptr<nntrainer::UnaryOperationLayer> makeUnary(UnaryKind kind) {
  switch (kind) {
  case UnaryKind::Sqrt:
    return std::make_unique<nntrainer::SQRTLayer>();
  case UnaryKind::Negative:
    return std::make_unique<nntrainer::NegativeLayer>();
  case UnaryKind::Sine:
    return std::make_unique<nntrainer::SineLayer>();
  case UnaryKind::Cosine:
    return std::make_unique<nntrainer::CosineLayer>();
  case UnaryKind::Tangent:
    return std::make_unique<nntrainer::TangentLayer>();
  }
  throw std::invalid_argument("Unknown unary layer");
}

/** @brief Independent scalar reference for the forward operation. */
double scalarForward(UnaryKind kind, double x) {
  switch (kind) {
  case UnaryKind::Sqrt:
    return std::sqrt(x);
  case UnaryKind::Negative:
    return -x;
  case UnaryKind::Sine:
    return std::sin(x);
  case UnaryKind::Cosine:
    return std::cos(x);
  case UnaryKind::Tangent:
    return std::tan(x);
  }
  throw std::invalid_argument("Unknown unary layer");
}

class UnaryDerivativeTest : public ::testing::TestWithParam<UnaryParam> {};

/** @brief Own small input/output tensors without external model fixtures. */
struct UnaryState {
  explicit UnaryState(const UnaryParam &param) :
    layer(makeUnary(std::get<0>(param))),
    input(nntrainer::TensorDim(std::get<1>(param), 1, 1, 6),
          nntrainer::Initializer::ZEROS, true, true, "input"),
    output(input.getDim(), nntrainer::Initializer::ZEROS, true, true, "output"),
    context("unary", false, 0.0f, false, 1.0f, nullptr, false, {}, {&input},
            {&output}, {}) {
    const float values[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const float gradients[] = {3.0f, -2.0f, 0.25f, -0.5f, 4.0f, -3.0f};
    for (unsigned int i = 0; i < input.getDim().getDataLen(); ++i) {
      const auto offset = (i + i / 6) % 6;
      const float x = values[offset];
      input.getVariableRef().getData<float>()[i] =
        std::get<0>(param) == UnaryKind::Sqrt ? std::abs(x) + 0.25f : x;
      output.getGradientRef().getData<float>()[i] = gradients[offset];
    }
  }

  std::unique_ptr<nntrainer::UnaryOperationLayer> layer;
  nntrainer::Var_Grad input;
  nntrainer::Var_Grad output;
  nntrainer::RunLayerContext context;
};

TEST_P(UnaryDerivativeTest, nonUnitIncomingMatchesFiniteDifference_n) {
  UnaryState state(GetParam());
  auto input = state.input.getVariableRef().clone();
  auto incoming = state.output.getGradientRef().clone();
  auto plus = input.clone();
  auto minus = input.clone();
  constexpr float epsilon = 1.0e-3f;
  plus.add_i(epsilon);
  minus.add_i(-epsilon);
  auto upper = state.output.getVariableRef().clone();
  auto lower = upper.clone();
  state.layer->forwarding_operation(plus, upper);
  state.layer->forwarding_operation(minus, lower);
  state.layer->forwarding(state.context, true);

  for (unsigned int call = 0; call < 2; ++call) {
    state.layer->calcDerivative(state.context);
    const auto *actual = state.input.getGradientRef().getData<float>();
    for (unsigned int i = 0; i < input.getDim().getDataLen(); ++i) {
      const double expected =
        (upper.getData<float>()[i] - lower.getData<float>()[i]) /
        (2.0 * epsilon) * incoming.getData<float>()[i];
      EXPECT_NEAR(actual[i], expected,
                  3.0e-3 * std::max(1.0, std::abs(expected)))
        << "element " << i << ", backward call " << call;
    }
    EXPECT_EQ(state.input.getVariableRef(), input);
    EXPECT_EQ(state.output.getGradientRef(), incoming);
  }
}

TEST_P(UnaryDerivativeTest, zeroIncomingClearsPreviousDerivative_n) {
  UnaryState state(GetParam());
  auto input = state.input.getVariableRef().clone();
  state.layer->calcDerivative(state.context);
  state.output.getGradientRef().setValue(0.0f);
  state.layer->calcDerivative(state.context);

  for (unsigned int i = 0; i < input.getDim().getDataLen(); ++i) {
    EXPECT_FLOAT_EQ(state.input.getGradientRef().getData<float>()[i], 0.0f);
    EXPECT_FLOAT_EQ(state.output.getGradientRef().getData<float>()[i], 0.0f);
  }
  EXPECT_EQ(state.input.getVariableRef(), input);
}

TEST_P(UnaryDerivativeTest, forwardMatchesScalarFunction_p) {
  UnaryState state(GetParam());
  state.layer->forwarding(state.context, true);

  for (unsigned int i = 0; i < state.input.getDim().getDataLen(); ++i) {
    const auto x = state.input.getVariableRef().getData<float>()[i];
    const auto expected = scalarForward(std::get<0>(GetParam()), x);
    EXPECT_NEAR(state.output.getVariableRef().getData<float>()[i], expected,
                1.0e-6 * std::max(1.0, std::abs(expected)));
  }
}

INSTANTIATE_TEST_SUITE_P(
  UnaryLayers, UnaryDerivativeTest,
  ::testing::Combine(::testing::Values(UnaryKind::Sqrt, UnaryKind::Negative,
                                       UnaryKind::Sine, UnaryKind::Cosine,
                                       UnaryKind::Tangent),
                     ::testing::Values(1U, 2U)));

} // namespace
