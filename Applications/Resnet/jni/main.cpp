// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet using symbolic graph construction
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#if defined(ENABLE_TEST)
#include <gtest/gtest.h>
#endif

#include <model.h>
#include <optimizer.h>
#include <tensor_api.h>

#include <cifar_dataloader.h>

#ifdef PROFILE
#include <profiler.h>
#endif

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

/**
 * @brief resnet block using symbolic tensor graph
 *
 * @param block_name name of the block
 * @param input symbolic tensor input
 * @param filters number of filters
 * @param kernel_size kernel size
 * @param downsample downsample to halve spatial dims
 * @param pre_trained whether layers are trainable
 * @return Tensor symbolic output tensor
 */
Tensor resnetBlock(const std::string &block_name, Tensor input, int filters,
                   int kernel_size, bool downsample, bool pre_trained) {
  auto scoped = [&block_name](const std::string &s) {
    return block_name + "/" + s;
  };
  std::string trainable = pre_trained ? "true" : "false";
  std::string f = std::to_string(filters);

  auto make_conv = [&](const std::string &name, int ks, int stride,
                       const std::string &padding) -> LayerHandle {
    std::string k = std::to_string(ks) + "," + std::to_string(ks);
    std::string s = std::to_string(stride) + "," + std::to_string(stride);
    return LayerHandle(
      createLayer("conv2d", {"name=" + scoped(name), "filters=" + f,
                             "kernel_size=" + k, "stride=" + s,
                             "padding=" + padding, "trainable=" + trainable}));
  };

  /** residual path */
#if defined(ENABLE_TFLITE_INTERPRETER)
  auto a1 = make_conv("a1", kernel_size, downsample ? 2 : 1, "same");
#else
  auto a1 = make_conv("a1", kernel_size, downsample ? 2 : 1,
                      downsample ? "1,1" : "same");
#endif
  LayerHandle a2(
    createLayer("batch_normalization",
                {"name=" + scoped("a2"), "activation=relu", "momentum=0.9",
                 "epsilon=0.00001", "trainable=" + trainable}));
  auto a3 = make_conv("a3", kernel_size, 1, "same");

  auto h = a1(input);
  h = a2(h);
  h = a3(h);

  /** skip path */
  Tensor skip = input;
  if (downsample) {
#if defined(ENABLE_TFLITE_INTERPRETER)
    auto b1 = make_conv("b1", 1, 2, "same");
#else
    auto b1 = make_conv("b1", 1, 2, "0,0");
#endif
    skip = b1(input);
  }

  /** addition + final bn */
  LayerHandle add_layer(createLayer("Addition", {"name=" + scoped("c1")}));
  auto merged = add_layer({h, skip});

  LayerHandle bn(
    createLayer("batch_normalization",
                {"name=" + block_name, "activation=relu", "momentum=0.9",
                 "epsilon=0.00001", "trainable=false"}));

  return bn(merged);
}

/**
 * @brief Build resnet18 as a symbolic tensor graph.
 *
 * @param input symbolic input tensor
 * @param pre_trained whether layers are trainable
 * @return Tensor symbolic output tensor
 */
Tensor buildResnet18Graph(Tensor input, bool pre_trained) {
  std::string trainable = pre_trained ? "true" : "false";

  LayerHandle conv0(createLayer(
    "conv2d", {"name=conv0", "filters=64", "kernel_size=3,3", "stride=1,1",
               "padding=same", "bias_initializer=zeros",
               "weight_initializer=xavier_uniform", "trainable=" + trainable}));

  LayerHandle first_bn(
    createLayer("batch_normalization",
                {"name=first_bn_relu", "activation=relu", "momentum=0.9",
                 "epsilon=0.00001", "trainable=" + trainable}));

  auto h = conv0(input);
  h = first_bn(h);

  h = resnetBlock("conv1_0", h, 64, 3, false, pre_trained);
  h = resnetBlock("conv1_1", h, 64, 3, false, pre_trained);
  h = resnetBlock("conv2_0", h, 128, 3, true, pre_trained);
  h = resnetBlock("conv2_1", h, 128, 3, false, pre_trained);
  h = resnetBlock("conv3_0", h, 256, 3, true, pre_trained);
  h = resnetBlock("conv3_1", h, 256, 3, false, pre_trained);
  h = resnetBlock("conv4_0", h, 512, 3, true, pre_trained);
  h = resnetBlock("conv4_1", h, 512, 3, false, pre_trained);

  LayerHandle pool(createLayer("pooling2d", {"name=last_p1", "pooling=average",
                                             "pool_size=4,4", "stride=4,4"}));
  LayerHandle flat(createLayer("flatten", {"name=last_f1"}));
  LayerHandle fc(
    createLayer("fully_connected", {"unit=100", "activation=softmax"}));

  h = pool(h);
  h = flat(h);
  h = fc(h);

  return h;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

#if defined(ENABLE_TEST)
TEST(Resnet_Training, verify_accuracy) {
  EXPECT_FLOAT_EQ(training_loss, 4.5145545f);
  EXPECT_FLOAT_EQ(validation_loss, 3.9630103f);
}
#endif

/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data,
                  UserDataType &valid_user_data) {
  const bool transfer_learning = false;
  std::string pretrained_bin_path = "./pretrained_resnet18.bin";

  // Build symbolic graph
  auto x = Tensor({1, 3, 32, 32}, "input0");
  auto y = buildResnet18Graph(x, transfer_learning);

  // Create model and compile from symbolic graph
/// @todo support "LOSS : cross" for TF_Lite Exporter
#if (defined(ENABLE_TFLITE_INTERPRETER) && !defined(ENABLE_TEST))
  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
#else
  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=cross"});
#endif

  model->setProperty({"batch_size=" + std::to_string(batch_size),
                      "epochs=" + std::to_string(epochs),
                      "save_path=resnet_full.bin"});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  int status = model->setOptimizer(std::move(optimizer));
  if (status) {
    throw std::invalid_argument("failed to set optimizer!");
  }

  // compile(Tensor, Tensor) internally calls compile + initialize + allocate
  status = model->compile(x, y, ml::train::ExecutionMode::TRAIN);
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                    std::move(dataset_valid));

  if (transfer_learning)
    model->load(pretrained_bin_path);
  model->train();

#if defined(ENABLE_TEST)
  training_loss = model->getTrainingLoss();
  validation_loss = model->getValidationLoss();
#elif defined(ENABLE_TFLITE_INTERPRETER)
  model->exports(ml::train::ExportMethods::METHOD_TFLITE, "resnet_test.tflite");
#endif
}

std::array<UserDataType, 2>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int simulated_data_size,
                        unsigned int data_split) {
  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));
  UserDataType valid_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));

  return {std::move(train_data), std::move(valid_data)};
}

std::array<UserDataType, 2>
createRealDataGenerator(const std::string &directory, unsigned int batch_size,
                        unsigned int data_split) {

  UserDataType train_data(new nntrainer::util::Cifar100DataLoader(
    directory + "/train.bin", batch_size, data_split));
  UserDataType valid_data(new nntrainer::util::Cifar100DataLoader(
    directory + "/test.bin", batch_size, data_split));

  return {std::move(train_data), std::move(valid_data)};
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr
      << "usage: ./main [{data_directory}|\"fake\"] [batchsize] [data_split] "
         "[epoch] \n"
      << "when \"fake\" is given, original data size is assumed 512 for both "
         "train and validation\n";
    return EXIT_FAILURE;
  }

  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

#ifdef PROFILE
  auto listener =
    std::make_shared<nntrainer::profile::GenericProfileListener>();
  nntrainer::profile::Profiler::Global().subscribe(listener);
#endif

  std::string data_dir = argv[1];
  unsigned int batch_size = std::stoul(argv[2]);
  unsigned int data_split = std::stoul(argv[3]);
  unsigned int epoch = std::stoul(argv[4]);

  std::cout << "data_dir: " << data_dir << ' ' << "batch_size: " << batch_size
            << " data_split: " << data_split << " epoch: " << epoch
            << std::endl;

  std::array<UserDataType, 2> user_datas;

  try {
    if (data_dir == "fake") {
      user_datas = createFakeDataGenerator(batch_size, 512, data_split);
    } else {
      user_datas = createRealDataGenerator(data_dir, batch_size, data_split);
    }
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto &[train_user_data, valid_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data, valid_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

#ifdef PROFILE
  std::cout << *listener;
#endif

  int status = EXIT_SUCCESS;
#if defined(ENABLE_TEST)
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    status = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
#endif

  return status;
}
