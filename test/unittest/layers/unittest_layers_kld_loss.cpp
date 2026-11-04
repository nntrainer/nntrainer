// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_layers_kld_loss.cpp
 * @brief Check KLD derivatives and preservation of the label tensor.
 * @author Minsoo Kim <61856979+33modeling@users.noreply.github.com>
 * @bug No known bugs.
 */

#include <gtest/gtest.h>

#include <kld_loss_layer.h>
#include <layer_context.h>
#include <var_grad.h>

namespace {

class KLDLossDerivativeTest : public ::testing::TestWithParam<unsigned int> {};

/** @brief Construct two distributions per batch without external fixtures. */
struct KLDState {
  explicit KLDState(unsigned int batch) :
    input(nntrainer::TensorDim(batch, 1, 1, 3), nntrainer::Initializer::ZEROS,
          true, true, "prediction"),
    output(nntrainer::TensorDim(batch, 1, 1, 3), nntrainer::Initializer::ZEROS,
           true, true, "loss"),
    context("kld", false, 0.0f, false, 1.0f, nullptr, false, {}, {&input},
            {&output}, {}) {
    const float predictions[] = {0.5f, 0.25f, 0.25f, 0.25f, 0.5f, 0.25f};
    const float labels[] = {0.25f, 0.5f, 0.25f, 0.5f, 0.25f, 0.25f};
    for (unsigned int i = 0; i < batch * 3; ++i) {
      input.getVariableRef().getData<float>()[i] = predictions[i];
      output.getGradientRef().getData<float>()[i] = labels[i];
    }
  }

  nntrainer::KLDLossLayer layer;
  nntrainer::Var_Grad input;
  nntrainer::Var_Grad output;
  nntrainer::RunLayerContext context;
};

TEST_P(KLDLossDerivativeTest, preservesPredictionAndLabel_n) {
  KLDState state(GetParam());
  auto prediction = state.input.getVariableRef().clone();
  auto label = state.output.getGradientRef().clone();

  state.layer.calcDerivative(state.context);

  EXPECT_EQ(state.input.getVariableRef(), prediction);
  EXPECT_EQ(state.output.getGradientRef(), label);
}

TEST_P(KLDLossDerivativeTest, repeatedBackwardPreservesDerivative_n) {
  KLDState state(GetParam());
  state.layer.calcDerivative(state.context);
  auto derivative = state.input.getGradientRef().clone();

  for (unsigned int call = 0; call < 2; ++call) {
    state.layer.calcDerivative(state.context);
    EXPECT_EQ(state.input.getGradientRef(), derivative);
  }
}

TEST_P(KLDLossDerivativeTest, firstBackwardMatchesAnalyticDerivative_p) {
  KLDState state(GetParam());
  state.layer.calcDerivative(state.context);

  const float expected[] = {-0.5f, -2.0f, -1.0f, -2.0f, -0.5f, -1.0f};
  const auto *actual = state.input.getGradientRef().getData<float>();
  for (unsigned int i = 0; i < GetParam() * 3; ++i)
    EXPECT_FLOAT_EQ(actual[i], expected[i]) << "element " << i;
}

INSTANTIATE_TEST_SUITE_P(BatchSizes, KLDLossDerivativeTest,
                         ::testing::Values(1U, 2U));

} // namespace
