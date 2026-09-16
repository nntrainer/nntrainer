// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file unittest_weight_sharing.cpp
 * @date 04 Mar 2026
 * @brief This file contains test and specification of network weight sharing
 * @see	https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>
#include <neuralnet.h>

TEST(WeightSharing, sharing_p) {
  std::shared_ptr<ml::train::Model> model1, model2;
  model1 = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model2 = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::shared_ptr<ml::train::Layer> input1 =
    ml::train::layer::Input({"name=input1", "input_shape=1:1:1"});
  std::shared_ptr<ml::train::Layer> input2 =
    ml::train::layer::Input({"name=input2", "input_shape=1:1:1"});

  model1->addLayer(input1);
  model2->addLayer(input2);

  std::shared_ptr<ml::train::Layer> fc1 =
    ml::train::layer::FullyConnected({"name=fc1", "unit=2"});
  std::shared_ptr<ml::train::Layer> fc2 =
    ml::train::layer::FullyConnected({"name=fc2", "unit=2"});

  model1->addLayer(fc1);
  model2->addLayer(fc2);

  model1->compile(ml::train::ExecutionMode::INFERENCE);
  model2->compile(ml::train::ExecutionMode::INFERENCE);

  model1->initialize(ml::train::ExecutionMode::INFERENCE);
  model2->initialize(ml::train::ExecutionMode::INFERENCE, model1);

  std::shared_ptr<ml::train::Layer> layer1;
  std::shared_ptr<ml::train::Layer> layer2;

  model1->getLayer("fc1", &layer1);
  model2->getLayer("fc2", &layer2);

  EXPECT_EQ(layer1->getWeights()[0], layer2->getWeights()[0]);
}

TEST(WeightSharing, sharing_initialize_n) {
  std::shared_ptr<ml::train::Model> model1, model2;
  model1 = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model2 = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::shared_ptr<ml::train::Layer> input1 =
    ml::train::layer::Input({"name=input1", "input_shape=1:1:1"});
  std::shared_ptr<ml::train::Layer> input2 =
    ml::train::layer::Input({"name=input2", "input_shape=1:1:1"});

  model1->addLayer(input1);
  model2->addLayer(input2);

  std::shared_ptr<ml::train::Layer> fc1 =
    ml::train::layer::FullyConnected({"name=fc1", "unit=2"});
  std::shared_ptr<ml::train::Layer> fc2 =
    ml::train::layer::FullyConnected({"name=fc2", "unit=2"});

  model1->addLayer(fc1);
  model2->addLayer(fc2);

  model1->compile(ml::train::ExecutionMode::INFERENCE);
  model2->compile(ml::train::ExecutionMode::INFERENCE);

  EXPECT_THROW(model2->initialize(ml::train::ExecutionMode::INFERENCE, model1),
               std::invalid_argument);
}

TEST(WeightSharing, sharing_unmatched_ref_n) {
  std::shared_ptr<ml::train::Model> model1, model2;
  model1 = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model2 = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::shared_ptr<ml::train::Layer> input1 =
    ml::train::layer::Input({"name=input1", "input_shape=1:1:1"});
  std::shared_ptr<ml::train::Layer> input2 =
    ml::train::layer::Input({"name=input2", "input_shape=1:1:1"});

  model1->addLayer(input1);
  model2->addLayer(input2);

  std::shared_ptr<ml::train::Layer> fc1 =
    ml::train::layer::FullyConnected({"name=fc1", "unit=2"});
  std::shared_ptr<ml::train::Layer> fc2 =
    ml::train::layer::FullyConnected({"name=fc2", "unit=3"});

  model1->addLayer(fc1);
  model2->addLayer(fc2);

  model1->compile(ml::train::ExecutionMode::INFERENCE);
  model2->compile(ml::train::ExecutionMode::INFERENCE);

  model1->initialize(ml::train::ExecutionMode::INFERENCE);
  EXPECT_THROW(model2->initialize(ml::train::ExecutionMode::INFERENCE, model1),
               std::invalid_argument);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
