// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   01 June 2023
 * @brief  application example for YOLO v3
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <app_context.h>
#include <det_dataloader.h>
#include <engine.h>
#include <model.h>
#include <optimizer.h>
#include <tensor_api.h>
#include <util_func.h>

#include <upsample_layer.h>
#include <yolo_v3_loss.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DirDataLoader>;

const unsigned int ANCHOR_NUMBER = 5;

const unsigned int MAX_OBJECT_NUMBER = 4;
const unsigned int CLASS_NUMBER = 4;
const unsigned int GRID_HEIGHT_NUMBER = 13;
const unsigned int GRID_WIDTH_NUMBER = 13;
const unsigned int IMAGE_HEIGHT_SIZE = 416;
const unsigned int IMAGE_WIDTH_SIZE = 416;
const unsigned int BATCH_SIZE = 4;
const unsigned int EPOCHS = 2;
const char *TRAIN_DIR_PATH = "/home/user/train_dir/";

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DirDataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DirDataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

std::array<UserDataType, 1> createDetDataGenerator(const char *train_dir,
                                                   int max_num_label, int c,
                                                   int h, int w) {
  UserDataType train_data(
    new nntrainer::util::DirDataLoader(train_dir, max_num_label, c, h, w, true,
                                       {
                                         {BATCH_SIZE, 1, 4, 5},
                                         {BATCH_SIZE, 1, 4, 5},
                                         {BATCH_SIZE, 1, 4, 5},
                                       }));

  return {std::move(train_data)};
}

/**
 * @brief Convolution block using symbolic tensor graph
 */
Tensor convBlock(const std::string &block_name, Tensor input, int kernel_size,
                 int num_filters, int stride, int padding) {
  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return nntrainer::withKey("name", scoped_name(layer_name));
  };

  LayerHandle conv(createLayer(
    "conv2d", {with_name("conv"),
               nntrainer::withKey("kernel_size", {kernel_size, kernel_size}),
               nntrainer::withKey("filters", num_filters),
               nntrainer::withKey("stride", {stride, stride}),
               nntrainer::withKey("padding", padding),
               nntrainer::withKey("disable_bias", "true")}));
  auto h = conv(input);

  LayerHandle bn_act(createLayer(
    "batch_normalization", {nntrainer::withKey("name", block_name),
                            nntrainer::withKey("momentum", "0.9"),
                            nntrainer::withKey("activation", "leaky_relu")}));
  return bn_act(h);
}

/**
 * @brief Darknet block using symbolic tensor graph
 */
Tensor darknetBlock(const std::string &block_name, Tensor input,
                    int num_filters, int repeat) {
  auto scoped_name = [&block_name](const std::string &layer_name, int uid) {
    return block_name + "/" + layer_name + "_" + std::to_string(uid);
  };

  Tensor h = input;
  for (int i = 0; i < repeat; i++) {
    auto c1 = convBlock(scoped_name("c1", i), h, 1, num_filters / 2, 1, 0);
    auto c2 = convBlock(scoped_name("c2", i), c1, 3, num_filters, 1, 1);

    std::string add_name =
      (repeat - 1 != i) ? scoped_name("res", i) : block_name;
    LayerHandle add(createLayer(
      "addition", {nntrainer::withKey("name", add_name)}));
    h = add({h, c2});
  }

  return h;
}

/**
 * @brief Create a detection head (conv → conv → permute → reshape → loss)
 */
Tensor createHead(const std::string &prefix, Tensor fp, int conv_filters,
                  int grid_size, int scale) {
  auto h = convBlock(prefix + "_1", fp, 3, conv_filters, 1, 1);
  h = convBlock(prefix, h, 1, 3 * (5 + CLASS_NUMBER), 1, 0);

  LayerHandle permute(createLayer(
    "permute", {nntrainer::withKey("name", prefix.substr(4) + "_permute"),
                nntrainer::withKey("direction", {2, 3, 1})}));
  h = permute(h);

  LayerHandle reshape(createLayer(
    "reshape",
    {nntrainer::withKey("name", prefix.substr(4) + "_reshape"),
     nntrainer::withKey(
       "target_shape",
       std::to_string(grid_size * grid_size) + ":" + std::to_string(3) + ":" +
         std::to_string(5 + CLASS_NUMBER))}));
  h = reshape(h);

  std::string loss_name;
  if (scale == 1)
    loss_name = "loss_for_large";
  else if (scale == 2)
    loss_name = "loss_for_medium";
  else
    loss_name = "loss_for_small";

  LayerHandle loss(createLayer(
    "yolo_v3_loss",
    {nntrainer::withKey("name", loss_name),
     nntrainer::withKey("max_object_number", MAX_OBJECT_NUMBER),
     nntrainer::withKey("class_number", CLASS_NUMBER),
     nntrainer::withKey("grid_height_number", grid_size),
     nntrainer::withKey("grid_width_number", grid_size),
     nntrainer::withKey("scale", scale)}));
  return loss(h);
}

/**
 * @brief Build YOLOv3 symbolic tensor graph
 * @return {input, {output_large, output_medium, output_small}}
 */
std::pair<Tensor, std::vector<Tensor>> buildYOLOv3Graph() {
  auto x = Tensor({1, 3, IMAGE_HEIGHT_SIZE, IMAGE_WIDTH_SIZE}, "input0");

  // DarkNet53 backbone
  auto h = convBlock("conv1", x, 3, 32, 1, 1);
  h = convBlock("conv2", h, 3, 64, 2, 1);
  h = darknetBlock("block1", h, 64, 1);
  h = convBlock("conv3", h, 3, 128, 2, 1);
  h = darknetBlock("block2", h, 128, 2);
  h = convBlock("conv4", h, 3, 256, 2, 1);
  auto block3 = darknetBlock("block3", h, 256, 8);
  h = convBlock("conv5", block3, 3, 512, 2, 1);
  auto block4 = darknetBlock("block4", h, 512, 8);
  h = convBlock("conv6", block4, 3, 1024, 2, 1);
  auto block5 = darknetBlock("block5", h, 1024, 4);

  // Feature pyramid for large object
  auto fp3 = convBlock("fp3_1", block5, 1, 512, 1, 0);
  fp3 = convBlock("fp3_2", fp3, 3, 1024, 1, 1);
  fp3 = convBlock("fp3_3", fp3, 1, 512, 1, 0);
  fp3 = convBlock("fp3_4", fp3, 3, 1024, 1, 1);
  fp3 = convBlock("fp3", fp3, 1, 512, 1, 0);

  // Neck: fp3 → upsample → concat with block4
  auto neck3_2 = convBlock("neck3_2_1", fp3, 1, 256, 1, 0);
  LayerHandle upsample1(
    createLayer("upsample", {nntrainer::withKey("name", "neck3_2")}));
  neck3_2 = upsample1(neck3_2);

  LayerHandle concat1(
    createLayer("concat", {nntrainer::withKey("name", "fp2_1"),
                           nntrainer::withKey("axis", "1")}));
  auto fp2 = concat1({neck3_2, block4});

  // Feature pyramid for medium object
  fp2 = convBlock("fp2_2", fp2, 1, 256, 1, 0);
  fp2 = convBlock("fp2_3", fp2, 3, 512, 1, 1);
  fp2 = convBlock("fp2_4", fp2, 1, 256, 1, 0);
  fp2 = convBlock("fp2_5", fp2, 3, 512, 1, 1);
  fp2 = convBlock("fp2", fp2, 1, 256, 1, 0);

  // Neck: fp2 → upsample → concat with block3
  auto neck2_1 = convBlock("neck2_1_1", fp2, 1, 128, 1, 0);
  LayerHandle upsample2(
    createLayer("upsample", {nntrainer::withKey("name", "neck2_1")}));
  neck2_1 = upsample2(neck2_1);

  LayerHandle concat2(
    createLayer("concat", {nntrainer::withKey("name", "fp1_1"),
                           nntrainer::withKey("axis", "1")}));
  auto fp1 = concat2({neck2_1, block3});

  // Feature pyramid for small object
  fp1 = convBlock("fp1_2", fp1, 1, 128, 1, 0);
  fp1 = convBlock("fp1_3", fp1, 3, 256, 1, 1);
  fp1 = convBlock("fp1_4", fp1, 1, 128, 1, 0);
  fp1 = convBlock("fp1_5", fp1, 3, 256, 1, 1);
  fp1 = convBlock("fp1", fp1, 1, 128, 1, 0);

  // Detection heads
  auto y_large = createHead("head3", fp3, 1024, 13, 1);
  auto y_medium = createHead("head2", fp2, 512, 26, 2);
  auto y_small = createHead("head1", fp1, 256, 52, 3);

  return {x, {y_large, y_medium, y_small}};
}

int main(int argc, char *argv[]) {
  // print start time
  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

  // set training config and print it
  std::cout << "batch_size: " << BATCH_SIZE << " epochs: " << EPOCHS
            << std::endl;

  try {
    auto &ct_engine = nntrainer::Engine::Global();
    auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));

    app_context->registerFactory(nntrainer::createLayer<custom::UpsampleLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    auto &ct_engine = nntrainer::Engine::Global();
    auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    app_context->registerFactory(
      nntrainer::createLayer<custom::YoloV3LossLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register yolov3 loss, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    // build YOLOv3 symbolic graph
    auto [input_t, output_ts] = buildYOLOv3Graph();

    ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", BATCH_SIZE),
                        nntrainer::withKey("epochs", EPOCHS),
                        nntrainer::withKey("save_path", "darknet53.bin")});

    // create optimizer
    auto optimizer = ml::train::createOptimizer(
      "adam", {"learning_rate=0.000001", "epsilon=1e-8", "torch_ref=true"});
    int status = model->setOptimizer(std::move(optimizer));
    if (status) {
      throw std::invalid_argument("failed to set optimizer");
    }

    // compile with symbolic tensors (multi-output)
    status = model->compile(input_t, output_ts);
    if (status) {
      throw std::invalid_argument("model compilation failed!");
    }

    model->summarize(std::cout,
                     ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);

    // create train and validation data
    std::array<UserDataType, 1> user_datas;
    user_datas = createDetDataGenerator(TRAIN_DIR_PATH, MAX_OBJECT_NUMBER, 3,
                                        IMAGE_HEIGHT_SIZE, IMAGE_WIDTH_SIZE);
    auto &[train_user_data] = user_datas;

    auto dataset_train = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                      std::move(dataset_train));

    model->train();
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  // print end time and duration
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
