// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   01 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  MNIST CNN example using symbolic tensor graph construction.
 *
 * Input 28x28
 * Conv2D 5x5 : 6 filters, stride 1x1, padding=0,0, sigmoid
 * Pooling2D : 2x2, Average pooling, stride 2x2
 * Conv2D 5x5 : 12 filters, stride 1x1, padding=0,0, sigmoid
 * Pooling2D : 2x2, Average Pooling, stride 2x2
 * Flatten
 * Fully Connected Layer with 10 units, softmax
 */

#if defined(ENABLE_TEST)
#define APP_VALIDATE
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#if defined(APP_VALIDATE)
#include <gtest/gtest.h>
#endif

#include <dataset.h>
#include <model.h>
#include <optimizer.h>
#include <tensor_api.h>

#ifdef PROFILE
#include <profiler.h>
#endif

#define TRAINING true

#define VALIDATION false

constexpr unsigned int SEED = 0;

#if VALIDATION
const unsigned int total_train_data_size = 32;
const unsigned int total_val_data_size = 32;
const unsigned int total_test_data_size = 32;
const unsigned int batch_size = 32;
#else
const unsigned int total_train_data_size = 100;
const unsigned int total_val_data_size = 100;
const unsigned int total_test_data_size = 100;
const unsigned int batch_size = 32;
#endif

const unsigned int total_label_size = 10;

unsigned int train_count = 0;
unsigned int val_count = 0;

const unsigned int feature_size = 784;

const float tolerance = 0.1f;

std::string data_path;

float training_loss = 0.0f;
float validation_loss = 0.0f;

std::string filename = "mnist_trainingSet.dat";

float stepFunction(float x) {
  if (x + tolerance > 1.0)
    return 1.0;
  if (x - tolerance < 0.0)
    return 0.0;
  return x;
}

bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((feature_size + total_label_size) *
                                 (uint64_t)id * sizeof(float));

  if (position > file_length)
    return false;

  F.seekg(position, std::ios::beg);
  F.read((char *)input, sizeof(float) * feature_size);
  F.read((char *)label, sizeof(float) * total_label_size);
  return true;
}

/**
 * @brief Per-dataset bookkeeping for random sample ordering.
 */
class DataInformation {
public:
  DataInformation(unsigned int num_samples, const std::string &filename);
  unsigned int count;
  unsigned int num_samples;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

DataInformation::DataInformation(unsigned int num_samples,
                                 const std::string &filename) :
  count(0),
  num_samples(num_samples),
  file(filename, std::ios::in | std::ios::binary),
  idxes(num_samples) {
  std::iota(idxes.begin(), idxes.end(), 0);
  rng.seed(SEED);
  std::shuffle(idxes.begin(), idxes.end(), rng);
  if (!file.good()) {
    throw std::invalid_argument("given file is not good, filename: " +
                                filename);
  }
}

int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);

  getData(data->file, *outVec, *outLabel, data->idxes.at(data->count));
  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
  }

  return 0;
}

#if defined(APP_VALIDATE)
TEST(MNIST_training, verify_accuracy) {
  EXPECT_FLOAT_EQ(training_loss, 2.586082f);
  EXPECT_FLOAT_EQ(validation_loss, 2.5753405f);
}
#endif

/**
 * @brief Build the MNIST CNN model using symbolic tensor graph.
 *
 * Returns {input_tensor, output_tensor} for compile(Tensor, Tensor).
 */
static std::pair<ml::train::Tensor, ml::train::Tensor> buildGraph() {
  using ml::train::createLayer;
  using ml::train::LayerHandle;
  using ml::train::Tensor;

  auto x = Tensor({1, 1, 28, 28}, "inputlayer");

  LayerHandle conv1(
    createLayer("conv2d", {"name=conv2d_c1_layer", "kernel_size=5,5",
                           "bias_initializer=zeros", "activation=sigmoid",
                           "weight_initializer=xavier_uniform", "filters=6",
                           "stride=1,1", "padding=0,0"}));
  LayerHandle pool1(
    createLayer("pooling2d", {"name=pooling2d_p1", "pool_size=2,2",
                              "stride=2,2", "padding=0,0", "pooling=average"}));
  LayerHandle conv2(
    createLayer("conv2d", {"name=conv2d_c2_layer", "kernel_size=5,5",
                           "bias_initializer=zeros", "activation=sigmoid",
                           "weight_initializer=xavier_uniform", "filters=12",
                           "stride=1,1", "padding=0,0"}));
  LayerHandle pool2(
    createLayer("pooling2d", {"name=pooling2d_p2", "pool_size=2,2",
                              "stride=2,2", "padding=0,0", "pooling=average"}));
  LayerHandle flat(createLayer("flatten", {"name=flatten"}));
  LayerHandle fc(createLayer("fully_connected",
                             {"name=outputlayer", "unit=10",
                              "weight_initializer=xavier_uniform",
                              "bias_initializer=zeros", "activation=softmax"}));

  auto h = conv1(x);
  h = pool1(h);
  h = conv2(h);
  h = pool2(h);
  h = flat(h);
  auto y = fc(h);

  return {x, y};
}

/**
 * @brief     create model and train on MNIST
 * @param[in]  arg 1 : resource path (dataset.dat)
 */
int main(int argc, char *argv[]) {
  int status = 0;
#ifdef APP_VALIDATE
  status = remove("mnist_model.bin");
  if (status != 0) {
    std::cout << "Pre-existing model file doesn't exist.\n";
  }
#endif
  if (argc < 2) {
    std::cout << "./nntrainer_mnist dataset.dat\n";
    exit(0);
  }

#ifdef PROFILE
  auto listener =
    std::make_shared<nntrainer::profile::GenericProfileListener>();
  nntrainer::profile::Profiler::Global().subscribe(listener);
#endif

  const std::vector<std::string> args(argv + 1, argv + argc);
  filename = args[0];

  std::unique_ptr<DataInformation> train_user_data;
  std::unique_ptr<DataInformation> valid_user_data;
  try {
    train_user_data =
      std::make_unique<DataInformation>(total_train_data_size, filename);
    valid_user_data =
      std::make_unique<DataInformation>(total_val_data_size, filename);
  } catch (std::invalid_argument &e) {
    std::cerr << "Error creating userdata for the data callback " << e.what()
              << std::endl;
    return 1;
  }

  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;
  try {
    dataset_train = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             getSample, train_user_data.get());
    dataset_val = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                           getSample, valid_user_data.get());
  } catch (const std::exception &e) {
    std::cerr << "Error creating dataset" << e.what() << std::endl;
    return 1;
  }

  // Build symbolic graph
  auto [x, y] = buildGraph();

  auto model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                           {"epochs=1500", "loss=cross", "batch_size=32"});

  try {
    auto optimizer = ml::train::createOptimizer("adam");
    auto lr_scheduler = ml::train::createLearningRateScheduler("step");

    lr_scheduler->setProperty(
      {"learning_rate=0.0001, 0.00009, 0.00007, 0.00005"});
    lr_scheduler->setProperty({"iteration=4, 6, 15"});
    optimizer->setLearningRateScheduler(std::move(lr_scheduler));
    model->setOptimizer(std::move(optimizer));
  } catch (const std::exception &e) {
    std::cerr << "Error during set optimizer " << e.what() << std::endl;
    return 1;
  }

  try {
    // compile(Tensor, Tensor) internally calls compile + initialize + allocate
    model->compile(x, y, ml::train::ExecutionMode::TRAIN);
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);
  } catch (const std::exception &e) {
    std::cerr << "Error during init " << e.what() << std::endl;
    return 1;
  }

#if defined(APP_VALIDATE)
  try {
    model->setProperty({"epochs=5"});
  } catch (...) {
    std::cerr << "Error during setting epochs\n";
    return -1;
  }
#endif

  try {
    model->train();
    training_loss = model->getTrainingLoss();
    validation_loss = model->getValidationLoss();
  } catch (const std::exception &e) {
    std::cerr << "Error during train " << e.what() << std::endl;
    return 0;
  }

#ifdef PROFILE
  std::cout << *listener;
#endif

#if defined(APP_VALIDATE)
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    status = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
#endif

  return status;
}
