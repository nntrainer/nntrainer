// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   5 July 2023
 * @brief  Example of multi-input dataloader
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <util_func.h>

#include <multi_loader.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

ModelHandle createMultiInoutModel() {
  using ml::train::createLayer;
  ModelHandle model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "mse")});
  std::vector<LayerHandle> layers;

  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input_0"),
                          nntrainer::withKey("input_shape", "1:1:2")}));
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input_1"),
                          nntrainer::withKey("input_shape", "1:4:2")}));

  layers.push_back(
    createLayer("fully_connected",
                {
                  nntrainer::withKey("name", "shared_fc"),
                  nntrainer::withKey("input_layers", "input_0"),
                  nntrainer::withKey("unit", 2),
                  nntrainer::withKey("activation", "none"),
                  nntrainer::withKey("weight_initializer", "he_normal"),
                  nntrainer::withKey("weight_regularizer", "l2norm"),
                  nntrainer::withKey("weight_regularizer_constant", "1e-5"),
                }));

  layers.push_back(createLayer(
    "lstm", {
              nntrainer::withKey("name", "shared_lstm"),
              nntrainer::withKey("unit", 2),
              nntrainer::withKey("input_layers", "input_1"),
              nntrainer::withKey("weight_initializer", "he_normal"),
              nntrainer::withKey("weight_regularizer", "l2norm"),
              nntrainer::withKey("weight_regularizer_constant", "1e-5"),
              nntrainer::withKey("trainable", "true"),
            }));

  layers.push_back(createLayer(
    "concat",
    {nntrainer::withKey("name", "concat0"), nntrainer::withKey("axis", "2"),
     nntrainer::withKey("input_layers", "shared_fc, shared_lstm")}));

  layers.push_back(
    createLayer("flatten", {nntrainer::withKey("name", "flatten_0"),
                            nntrainer::withKey("input_layers", "concat0")}));

  layers.push_back(
    createLayer("fully_connected",
                {
                  nntrainer::withKey("name", "output_0"),
                  nntrainer::withKey("input_layers", "flatten_0"),
                  nntrainer::withKey("unit", 1),
                  nntrainer::withKey("activation", "none"),
                  nntrainer::withKey("weight_initializer", "he_normal"),
                  nntrainer::withKey("weight_regularizer", "l2norm"),
                  nntrainer::withKey("weight_regularizer_constant", "1e-5"),
                }));

  layers.push_back(
    createLayer("fully_connected",
                {
                  nntrainer::withKey("name", "output_1"),
                  nntrainer::withKey("input_layers", "flatten_0"),
                  nntrainer::withKey("unit", 1),
                  nntrainer::withKey("activation", "none"),
                  nntrainer::withKey("weight_initializer", "he_normal"),
                  nntrainer::withKey("weight_regularizer", "l2norm"),
                  nntrainer::withKey("weight_regularizer_constant", "1e-5"),
                }));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);
  data->next(input, label, last);
  return 0;
}

ModelHandle createAndRun(unsigned int epochs, unsigned int batch_size,
                         UserDataType &train_user_data) {

  ModelHandle model = createMultiInoutModel();

  model->setProperty({nntrainer::withKey("batch_size", batch_size),
                      nntrainer::withKey("epochs", epochs),
                      nntrainer::withKey("save_path", "resnet_full.bin")});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  int status = model->setOptimizer(std::move(optimizer));
  if (status) {
    throw std::invalid_argument("failed to set optimizer!");
  };

  status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));

  model->summarize(std::cout, ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);
  model->train();

  return model;
}

std::array<UserDataType, 1>
createFakeMultiInoutDataGenerator(unsigned int batch_size,
                                  unsigned int simulated_data_size) {
  UserDataType train_data(new nntrainer::util::MultiInoutDataLoader(
    {{batch_size, 1, 4, 2}, {batch_size, 1, 1, 2}},
    {{batch_size, 1, 1, 1}, {batch_size, 1, 1, 1}}, simulated_data_size));

  return {std::move(train_data)};
}

void runInferenceAndPrintSamples(const ModelHandle &model,
                                 unsigned int batch_size,
                                 UserDataType &train_user_data) {
  std::cout << "\n=== Running Inference and Printing Samples ===" << std::endl;
  std::cout << "Using trained model for inference" << std::endl;

  // Prepare input and output tensors for inference using std::vector for
  // automatic memory management Note: Input order follows compilation order:
  // input1 (1:4:2) comes before input0 (1:1:2)
  std::vector<float> input1_data(batch_size * 1 * 4 *
                                 2); // input1: batch_size:1:4:2
  std::vector<float> input0_data(batch_size * 1 * 1 *
                                 2); // input0: batch_size:1:1:2
  std::vector<float> label0_data(batch_size * 1 * 1 *
                                 1); // output_1: batch_size:1:1:1
  std::vector<float> label1_data(batch_size * 1 * 1 *
                                 1); // output_2: batch_size:1:1:1

  std::vector<float *> input_tensors(2);
  std::vector<float *> label_tensors(2);

  // Set up input tensors in compilation order: input1, input0
  input_tensors[0] =
    input1_data.data(); // input1 (1:4:2) - first in compilation order
  input_tensors[1] =
    input0_data.data(); // input0 (1:1:2) - second in compilation order

  // Set up label tensors
  label_tensors[0] = label0_data.data(); // output_1
  label_tensors[1] = label1_data.data(); // output_2

  // Run inference on a few samples
  const int num_samples_to_test = std::min(2, (int)batch_size);
  bool last = false;

  std::cout << "\nTesting " << num_samples_to_test << " samples:" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (int sample_idx = 0; sample_idx < num_samples_to_test && !last;
       ++sample_idx) {
    // Hardcode inputs for deterministic comparison with Python
    train_user_data->next(input_tensors.data(), label_tensors.data(), &last);

    if (sample_idx == 0) {
      input_tensors[1][0] = -0.5f;
      input_tensors[1][1] = 0.2f;

      float vals[] = {-0.9f, 0.1f, 0.8f, -0.2f, 0.0f, 0.5f, -0.4f, 0.7f};
      for (int i = 0; i < 8; ++i)
        input_tensors[0][i] = vals[i];
    } else {
      input_tensors[1][0] = 0.8f;
      input_tensors[1][1] = -0.9f;

      float vals[] = {0.3f, -0.6f, -0.1f, 0.4f, 0.9f, -0.8f, 0.2f, -0.5f};
      for (int i = 0; i < 8; ++i)
        input_tensors[0][i] = vals[i];
    }

    float input_val = input_tensors[1][0];
    label_tensors[0][0] = input_val * 2.0f;
    label_tensors[1][0] = input_val * input_val;

    std::vector<float *> outputs = model->inference(batch_size, input_tensors);

    std::cout << "\nSample " << (sample_idx + 1) << ":" << std::endl;

    std::cout << "Input0: [";
    for (int i = 0; i < 2; ++i) {
      std::cout << input_tensors[1][i];
      if (i < 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Input1: [";
    for (int i = 0; i < 8; ++i) {
      std::cout << input_tensors[0][i];
      if (i < 7)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    if (outputs.size() >= 2) {
      std::cout << "Label0: " << label_tensors[0][0]
                << ", Predicted0: " << outputs[1][0] << std::endl;
      std::cout << "Label1: " << label_tensors[1][0]
                << ", Predicted1: " << outputs[0][0] << std::endl;
    } else {
      std::cout << "Error: Expected 2 outputs but got " << outputs.size()
                << std::endl;
    }
  }

  std::cout << "\n----------------------------------------" << std::endl;
  std::cout << "Inference completed!" << std::endl;
}

int main(int argc, char *argv[]) {
  unsigned int total_data_size = 1024;
  unsigned int batch_size = 32;
  unsigned int epoch = 100;

  std::array<UserDataType, 1> user_datas;

  try {
    user_datas = createFakeMultiInoutDataGenerator(batch_size, total_data_size);
    auto &[train_user_data] = user_datas;
    ModelHandle model = createAndRun(epoch, batch_size, train_user_data);

    // Perform inference and print samples
    runInferenceAndPrintSamples(model, batch_size, train_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
