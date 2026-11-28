// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_layers_reduce_mean_compute.cpp
 * @brief Check default and explicit ReduceMean axes through backward.
 * @author Minsoo Kim <61856979+33modeling@users.noreply.github.com>
 * @bug No known bugs.
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <layer_context.h>
#include <reduce_mean_layer.h>
#include <var_grad.h>

namespace {

class ReduceMeanComputeTest : public ::testing::TestWithParam<unsigned int> {};

/** @brief Finalize the real layer and obtain its output allocation shape. */
nntrainer::TensorDim finalizeMean(nntrainer::ReduceMeanLayer &layer,
                                  const nntrainer::TensorDim &dim,
                                  const std::vector<std::string> &properties) {
  layer.setProperty(properties);
  nntrainer::InitLayerContext init({dim}, {true}, false, "mean");
  layer.finalize(init);
  return init.getOutSpecs().at(0).variable_spec.dim;
}

/** @brief Own a nontrivial tensor shape and the real forward/backward context.
 */
struct MeanState {
  MeanState(unsigned int batch, const std::vector<std::string> &properties) :
    input(nntrainer::TensorDim(batch, 2, 3, 4), nntrainer::Initializer::ZEROS,
          true, true, "input"),
    output(finalizeMean(layer, input.getDim(), properties),
           nntrainer::Initializer::ZEROS, true, true, "output"),
    context("mean", false, 0.0f, false, 1.0f, nullptr, false, {}, {&input},
            {&output}, {}) {
    for (unsigned int i = 0; i < input.getDim().getDataLen(); ++i)
      input.getVariableRef().getData<float>()[i] =
        static_cast<float>(i) - 13.0f;
    for (unsigned int i = 0; i < output.getDim().getDataLen(); ++i)
      output.getGradientRef().getData<float>()[i] = i % 2 ? -2.0f : 3.0f;
  }

  nntrainer::ReduceMeanLayer layer;
  nntrainer::Var_Grad input;
  nntrainer::Var_Grad output;
  nntrainer::RunLayerContext context;
};

/** @brief Check reduction and broadcast gradients using scalar accumulation. */
void expectMeanAndDerivative(MeanState &state) {
  const auto in_dim = state.input.getDim();
  const auto out_dim = state.output.getDim();
  const auto reduction = in_dim.getDataLen() / out_dim.getDataLen();
  std::vector<float> expected(out_dim.getDataLen(), 0.0f);
  auto input = state.input.getVariableRef().clone();
  auto incoming = state.output.getGradientRef().clone();

  for (unsigned int i = 0; i < in_dim.getDataLen(); ++i) {
    unsigned int remaining = i, output_index = 0, stride = 1;
    for (int axis = 3; axis >= 0; --axis) {
      const auto coordinate = remaining % in_dim[axis];
      remaining /= in_dim[axis];
      output_index += (out_dim[axis] == 1 ? 0 : coordinate) * stride;
      stride *= out_dim[axis];
    }
    expected[output_index] += input.getData<float>()[i] / reduction;
  }

  state.layer.forwarding(state.context, true);
  state.layer.calcDerivative(state.context);
  for (unsigned int i = 0; i < out_dim.getDataLen(); ++i)
    EXPECT_NEAR(state.output.getVariableRef().getData<float>()[i], expected[i],
                1.0e-5f);

  for (unsigned int i = 0; i < in_dim.getDataLen(); ++i) {
    unsigned int remaining = i, output_index = 0, stride = 1;
    for (int axis = 3; axis >= 0; --axis) {
      const auto coordinate = remaining % in_dim[axis];
      remaining /= in_dim[axis];
      output_index += (out_dim[axis] == 1 ? 0 : coordinate) * stride;
      stride *= out_dim[axis];
    }
    EXPECT_FLOAT_EQ(state.input.getGradientRef().getData<float>()[i],
                    incoming.getData<float>()[output_index] / reduction);
  }
  EXPECT_EQ(state.input.getVariableRef(), input);
  EXPECT_EQ(state.output.getGradientRef(), incoming);
}

TEST_P(ReduceMeanComputeTest, omittedAxisPreservesBatch_n) {
  nntrainer::ReduceMeanLayer layer;
  const auto out_dim =
    finalizeMean(layer, nntrainer::TensorDim(GetParam(), 2, 3, 4), {});
  EXPECT_EQ(out_dim, nntrainer::TensorDim(GetParam(), 1, 1, 1));
}

TEST_P(ReduceMeanComputeTest, omittedAxisRunsForwardAndBackward_n) {
  MeanState state(GetParam(), {});
  expectMeanAndDerivative(state);
}

TEST_P(ReduceMeanComputeTest, explicitAxesKeepTheirReduction_p) {
  for (unsigned int axis = 1; axis < 4; ++axis) {
    SCOPED_TRACE(axis);
    MeanState state(GetParam(), {"axis=" + std::to_string(axis)});
    auto expected_dim = state.input.getDim();
    expected_dim.setTensorDim(axis, 1);
    EXPECT_EQ(state.output.getDim(), expected_dim);
    expectMeanAndDerivative(state);
  }
}

INSTANTIATE_TEST_SUITE_P(BatchSizes, ReduceMeanComputeTest,
                         ::testing::Values(1U, 2U));

} // namespace
