// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   16 November 2020
 * @brief  This file contains the execution part of LayerClient example
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <iostream>
#include <memory>
#include <string>

#include <dataset.h>
#include <layer.h>
#include <model.h>

/// @todo Migrate this to api
#include <app_context.h>
#include <engine.h>

#include <mae_loss.h>
#include <pow.h>

#define BATCH_SIZE 10
#define FEATURE_SIZE 100
#define NUM_CLASS 10

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int constant_generator_cb(float **outVec, float **outLabel, bool *last,
                          void *user_data) {
  static int count = 0;
  unsigned int i;
  unsigned int data_size = FEATURE_SIZE;

  for (i = 0; i < data_size; ++i) {
    outVec[0][i] = 2.0f;
  }

  for (i = 0; i < NUM_CLASS - 1; ++i) {
    outLabel[0][i] = 0.0f;
  }
  outLabel[0][0] = 1.0f;

  count++;
  if (count == 10) {
    *last = true;
    count = 0;
  } else {
    *last = false;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief run from a ini model
 *
 * @param ini_path ini_path to load configuration
 * @return int 0 if successfully ran
 */
static int ini_model_run(const std::string &ini_path) {
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  if (model->loadFromConfig(ini_path) != 0) {
    std::cerr << "failed to load configuration";
    return 1;
  }

  std::shared_ptr<ml::train::Dataset> dataset;
  try {
    dataset = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                       constant_generator_cb);
  } catch (...) {
    std::cerr << "creating dataset failed";
    return 1;
  }

  if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset) != 0) {
    std::cerr << "failed to set datatset";
    return 1;
  }

  if (model->compile() != 0) {
    std::cerr << "model compilation failed";
    return 1;
  }

  if (model->initialize() != 0) {
    std::cerr << "model initiation failed";
    return 1;
  }

  if (model->train() != 0) {
    std::cerr << "train failed";
    return 1;
  }

  std::cout << "successfully ran";
  return 0;
}

/**
 * @brief run from an api
 *
 * @return int 0 if successfully ran
 */
int api_model_run() {
  auto model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"batch_size=10"});

  std::shared_ptr<ml::train::Dataset> dataset;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  std::vector<std::shared_ptr<ml::train::Layer>> layers;

  try {
    dataset = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                       constant_generator_cb);
  } catch (...) {
    std::cerr << "creating dataset failed";
    return 1;
  }

  if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset) != 0) {
    std::cerr << "failed to set datatset";
    return 1;
  }

  try {
    optimizer = ml::train::optimizer::SGD({"learning_rate=0.1"});
  } catch (...) {
    std::cerr << "creating optimizer failed";
    return 1;
  }

  if (model->setOptimizer(optimizer) != 0) {
    std::cerr << "failed to set optimizer";
    return 1;
  }

  try {
    /// creating array of layers same as in `custom_layer_client.ini`
    layers = std::vector<std::shared_ptr<ml::train::Layer>>{
      ml::train::layer::Input({"name=inputlayer", "input_shape=1:1:100"}),
      ml::train::createLayer("custom_pow", {"name=powlayer", "exponent=3"}),
      ml::train::layer::FullyConnected(
        {"name=outputlayer", "input_layers=powlayer", "unit=10",
         "bias_initializer=zeros", "activation=softmax"}),
      ml::train::createLayer("mae_loss", {"name=mae_loss"})};
  } catch (nntrainer::exception::not_supported &e) {
    std::cerr << "creating model failed " << e.what();
    return 1;
  }

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  if (model->compile() != 0) {
    std::cerr << "model compilation failed";
    return 1;
  }

  if (model->initialize() != 0) {
    std::cerr << "model initiation failed";
    return 1;
  }

  if (model->train() != 0) {
    std::cerr << "train failed";
    return 1;
  }

  std::cout << "successfully ran";
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "model mode... \n"
              << "usage: ./main ./foo/model.ini\n"
              << "api mode... \n"
              << "usage: ./main model\n";

    return 1;
  }

  try {
    auto &ct_engine = nntrainer::Engine::Global();
    /// registering custom layer here
    /// registerFactory excepts a function that returns unique_ptr<Layer> from
    /// std::vector<std::string> ml::train::createLayer<T> is a templated
    /// function for generic usage
    ct_engine.registerLayerFactory("cpu",
                                   nntrainer::createLayer<custom::PowLayer>);
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<custom::MaeLossLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  const std::string arg(argv[1]);

  try {
    if (arg == "model") {
      return api_model_run();
    } else {
      return ini_model_run(arg);
    }
  } catch (std::exception &e) {
    std::cerr << "failed to run the model, reason: " << e.what() << std::endl;
    return 1;
  }

  /// should not reach here
  return 1;
}
