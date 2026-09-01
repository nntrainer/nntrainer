// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file integration_test_mixed_precision.cpp
 * @date 26 Nov 2024
 * @brief Mixed Precision integration test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <tuple>

#include <gtest/gtest.h>

#include <app_context.h>
#include <layer.h>
#include <lite/core/c/common.h>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>
using namespace nntrainer;

/**
 *  Supported Layer list
 *  [InputLayer, BatchNormalizationLayer, FullyConnectedLayer, Conv1DLayer]
 */

TEST(mixed_precision, input_only_model_test) {

  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:1:3"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate = 0.1"}));

  EXPECT_EQ(nn->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(), ML_ERROR_NONE);
  EXPECT_EQ(nn->reinitialize(), ML_ERROR_NONE);
}

TEST(mixed_precision, conv1d_model_test) {

  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=3:1:3"}},
    {"conv1d", {"name=conv1d_0", "filters=1", "kernel_size=3"}},
    {"conv1d", {"name=conv1d_1", "filters=1", "kernel_size=1"}},
    {"batch_normalization", {"name=bn_0", "epsilon=0.00001", "momentum=0.9"}},
    {"batch_normalization", {"name=bn_1", "epsilon=0.00001", "momentum=0.9"}},
  });

  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate = 0.1"}));

  EXPECT_EQ(nn->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(), ML_ERROR_NONE);
  EXPECT_EQ(nn->reinitialize(), ML_ERROR_NONE);
}

TEST(mixed_precision, lstm_test) {

  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"});

  auto graph = makeGraph({
    {"input", {"name=in", "input_shape=1:3:3"}},
    {"lstm", {"name=lstm_0", "unit=1"}},
    {"lstm", {"name=lstm_1", "unit=1"}},
    {"lstm", {"name=lstm_2", "unit=1"}},
    {"batch_normalization", {"name=bn_0", "epsilon=0.00001", "momentum=0.9"}},
    {"batch_normalization", {"name=bn_1", "epsilon=0.00001", "momentum=0.9"}},
  });

  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("adam", {"learning_rate = 0.1"}));

  EXPECT_EQ(nn->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn->initialize(), ML_ERROR_NONE);
  EXPECT_EQ(nn->reinitialize(), ML_ERROR_NONE);
}

TEST(mixed_precision, loss_scale_test) {
  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

  EXPECT_THROW(nn->setProperty({"batch_size=1", "model_tensor_type=FP16-FP16",
                                "loss_scale=0"}),
               std::invalid_argument);

  EXPECT_NO_THROW(nn->setProperty(
    {"batch_size=1", "model_tensor_type=FP16-FP16", "loss_scale=65536"}));
}

TEST(mixed_precision, model_tensor_type_test) {
  std::unique_ptr<ml::train::Model> nn =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});

  std::string positive_type_list[] = {
    "QINT8-FP16",   "QINT8-FP32",  "FP16-FP16",     "FP16-FP32",
    "FP32-FP16",    "FP32-FP32",   "QINT16-QINT16", "UINT16-UINT16",
    "QINT8-UINT16", "UINT4-UINT8", "UINT4-UINT16",  "UINT8-UINT8",
    "UINT8-UINT16", "QS4CX-FP32",  "QS4CX-FP16"};

  /// @note QINT4-FP16 and QINT4-FP32 are rejected at property-parse time.
  /// Channel-wise int4 is QS4CX and QINT4 is deprecated: it is an on-disk
  /// format only, so a model that asked for one could not allocate a single
  /// weight. QS4CX-FP16 / QS4CX-FP32 replace them.
  std::string negative_type_list[] = {"FP16-XXX", "XXX-XXX",    "",
                                      "ttkt",     "QINT4-FP16", "QINT4-FP32"};

  for (auto type_item : positive_type_list) {
    EXPECT_NO_THROW(nn->setProperty(
      {"batch_size=1", "model_tensor_type=" + type_item, "loss_scale=65536"}));
  }

  for (auto type_item : negative_type_list) {
    EXPECT_THROW(
      nn->setProperty(
        {"batch_size=1", "model_tensor_type=" + type_item, "loss_scale=65536"}),
      std::invalid_argument);
  }
}
