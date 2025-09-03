// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_models_mixed_precision.cpp
 * @date 3 May 2024
 * @brief unittest models to cover mixed precision
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static std::unique_ptr<NeuralNetwork> fc_mixed_training() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3", "input_dtype=FP32"}},
    {"Fully_connected", {"name=fc", "input_layers=in", "unit=10"}},
    {"mse", {"name=loss", "input_layers=fc"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer(
    "adam", {"learning_rate = 0.1", "torch_ref=true"}));

  return nn;
}

static std::unique_ptr<NeuralNetwork> fc_mixed_training_nan_sgd() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:1", "input_dtype=FP32"}},
    {"Fully_connected", {"name=fc0", "input_layers=in", "unit=1"}},
    {"Fully_connected", {"name=fc1", "input_layers=fc0", "unit=1"}},
    {"mse", {"name=loss", "input_layers=fc1"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));

  return nn;
}

GTEST_PARAMETER_TEST(
  MixedPrecision, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelTc_V2(fc_mixed_training, "fc_mixed_training",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(fc_mixed_training_nan_sgd, "fc_mixed_training_nan_sgd",
                 ModelTestOption::ALL_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info)
    -> const auto & { return std::get<1>(info.param); });
