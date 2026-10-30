// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_layers_recurrent_cell_lifespan.cpp
 * @date   28 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs
 * @brief  Tests that the recurrent cell layers declare the lifespan of their
 *         outputs in finalize(), rather than having NetworkGraph pick it by
 *         their type name.
 */
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <grucell.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <lstmcell.h>
#include <rnncell.h>
#include <tensor_dim.h>

namespace {

using LS = nntrainer::TensorLifespan;

/**
 * @brief a recurrent cell and the outputs its finalize() must declare
 */
struct CellOutputCase {
  std::string name;                                        /**< test name */
  std::function<std::unique_ptr<nntrainer::Layer>()> make; /**< the layer */
  std::vector<std::string> props; /**< properties needed to finalize */
  unsigned int num_inputs;        /**< inputs the cell expects */
  size_t num_outputs;             /**< outputs the cell declares */
};

/**
 * @brief print a case by its name in gtest output
 */
std::string caseName(const ::testing::TestParamInfo<CellOutputCase> &info) {
  return info.param.name;
}

} // namespace

/**
 * @brief parameterized fixture for the output lifespans of recurrent cells
 */
class OutputLifespan : public ::testing::TestWithParam<CellOutputCase> {};

/**
 * @brief every output of the cell lives until its gradient is computed
 */
TEST_P(OutputLifespan, declared_output_lifespans_p) {
  const CellOutputCase &c = GetParam();
  std::unique_ptr<nntrainer::Layer> layer = c.make();
  ASSERT_NO_THROW(layer->setProperty(c.props));

  std::vector<ml::train::TensorDim> input_dims(
    c.num_inputs, ml::train::TensorDim({1, 1, 1, 1}));
  /** every output is connected, as in a graph that reads all of them */
  nntrainer::InitLayerContext context(
    input_dims, std::vector<bool>(c.num_outputs, true), false, "layer");
  ASSERT_NO_THROW(layer->finalize(context));

  const std::vector<nntrainer::VarGradSpecV2> &specs = context.getOutSpecs();
  ASSERT_EQ(specs.size(), c.num_outputs);
  for (size_t i = 0; i < specs.size(); ++i) {
    SCOPED_TRACE("output " + std::to_string(i));
    EXPECT_EQ(specs[i].variable_spec.ls, LS::FORWARD_GRAD_LIFESPAN);
    ASSERT_NE(specs[i].gradient_spec, nullptr);
    EXPECT_EQ(specs[i].gradient_spec->ls, LS::CALC_GRAD_DERIV_LIFESPAN);
  }
}

INSTANTIATE_TEST_SUITE_P(
  nntrainer_RecurrentCell, OutputLifespan,
  ::testing::Values(
    CellOutputCase{
      "rnncell",
      [] { return nntrainer::createLayer<nntrainer::RNNCellLayer>(); },
      {"unit=1"},
      2,
      1},
    CellOutputCase{
      "grucell",
      [] { return nntrainer::createLayer<nntrainer::GRUCellLayer>(); },
      {"unit=1", "integrate_bias=false", "reset_after=true"},
      2,
      1},
    CellOutputCase{
      "lstmcell",
      [] { return nntrainer::createLayer<nntrainer::LSTMCellLayer>(); },
      {"unit=1"},
      3,
      2}),
  caseName);
