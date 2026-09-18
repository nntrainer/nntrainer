// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   task_runner.cpp
 * @date   08 Jan 2021
 * @brief  task runner for the simpleshot demonstration
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

#include <app_context.h>
#include <engine.h>
#include <model.h>
#include <nntrainer-api-common.h>

#include "layers/centering.h"

namespace simpleshot {

namespace {

/**
 * @brief get backbone path from a model name
 *
 * @param model resnet50 or conv4  is supported
 *
 */
const std::string getModelFilePath(const std::string &model,
                                   const std::string &app_path) {
  const std::string resnet_model_path =
    app_path + "/backbones/resnet50_60classes.tflite";
  const std::string conv4_model_path =
    app_path + "/backbones/conv4_60classes.tflite";

  std::string model_path;

  if (model == "resnet50") {
    model_path = resnet_model_path;
  } else if (model == "conv4") {
    model_path = conv4_model_path;
  }

  std::ifstream infile(model_path);
  if (!infile.good()) {
    std::stringstream ss;
    ss << model_path << " as backbone does not exist!";
    throw std::invalid_argument(ss.str().c_str());
  }

  if (model_path.empty()) {
    std::stringstream ss;
    ss << "not supported model type given, model type: " << model;
    throw std::invalid_argument(ss.str().c_str());
  }

  return model_path;
}

const std::string getFeatureFilePath(const std::string &model,
                                     const std::string &app_path) {
  const std::string resnet_model_path =
    app_path + "/backbones/resnet50_60classes_feature_vector.bin";
  const std::string conv4_model_path =
    app_path + "/backbones/conv4_60classes_feature_vector.bin";

  std::string model_path;

  if (model == "resnet50") {
    model_path = resnet_model_path;
  } else if (model == "conv4") {
    model_path = conv4_model_path;
  }

  std::ifstream infile(model_path);
  if (!infile.good()) {
    std::stringstream ss;
    ss << model_path << " as backbone does not exist!";
    throw std::invalid_argument(ss.str().c_str());
  }

  if (model_path.empty()) {
    std::stringstream ss;
    ss << "not supported model type given, model type: " << model;
    throw std::invalid_argument(ss.str().c_str());
  }

  return model_path;
}

/**
 * @brief get current working directory by cpp string
 *
 * @return const std::string current working directory
 */
const std::string getcwd_() {
  const size_t bufsize = 4096;
  char buffer[bufsize];

  char *cwd = getcwd(buffer, bufsize);
  std::string ret = (cwd == NULL) ? "" : std::string(cwd);
  return ret;
}
} // namespace

using LayerHandle = std::shared_ptr<ml::train::Layer>;

/**
 * @brief Create a Model with given backbone and variant setup
 *
 * @param backbone either conv4 or resnet50, hardcoded tflite path will be
 * selected
 * @param app_path designated app path to search the backbone file
 * @param variant "one of UN, L2N, CL2N"
 * @return std::unique_ptr<ml::train::Model>
 */
std::unique_ptr<ml::train::Model> createModel(const std::string &backbone,
                                              const std::string &app_path,
                                              const std::string &variant = "UN",
                                              const int num_classes = 5) {
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                      {"batch_size=1", "epochs=1"});

  LayerHandle backbone_layer = ml::train::layer::BackboneTFLite(
    {"name=backbone", "model_path=" + getModelFilePath(backbone, app_path),
     "input_shape=32:32:3", "trainable=false"});
  model->addLayer(backbone_layer);

  auto generate_knn_part = [&backbone, &app_path,
                            num_classes](const std::string &variant_) {
    std::vector<LayerHandle> v;

    const std::string num_class_prop =
      "num_class=" + std::to_string(num_classes);

    if (variant_ == "UN") {
      /// left empty intended
    } else if (variant_ == "L2N") {
      LayerHandle l2 = ml::train::createLayer(
        "preprocess_l2norm", {"name=l2norm", "trainable=false"});
      v.push_back(l2);
    } else if (variant_ == "CL2N") {
      LayerHandle centering = ml::train::createLayer(
        "centering", {"name=center",
                      "feature_path=" + getFeatureFilePath(backbone, app_path),
                      "trainable=false"});
      LayerHandle l2 = ml::train::createLayer(
        "preprocess_l2norm", {"name=l2norm", "trainable=false"});
      v.push_back(centering);
      v.push_back(l2);
    } else {
      std::stringstream ss;
      ss << "unsupported variant type: " << variant_;
      throw std::invalid_argument(ss.str().c_str());
    }

    LayerHandle knn = ml::train::createLayer(
      "centroid_knn", {"name=knn", num_class_prop, "trainable=false"});
    v.push_back(knn);

    return v;
  };

  auto knn_part = generate_knn_part(variant);
  for (auto &layer : knn_part) {
    model->addLayer(layer);
  }

  return model;
}
} // namespace simpleshot

/**
 * @brief main runner
 *
 * @return int
 */
int main(int argc, char **argv) {
  auto &ct_engine = nntrainer::Engine::Global();

  if (argc != 6 && argc != 5) {
    std::cout
      << "usage: model method train_file validation_file app_path\n"
      << "model: are [resnet50, conv4]\n"
      << "methods: are [UN, L2N, CL2N]\n"
      << "train file: [app_path]/tasks/[train_file] is used for training\n"
      << "validation file: [app_path]/tasks/[validation_file] is used for "
         "validation\n"
      << "app_path: root path to refer to resources, if not given"
         "path is set current working directory\n";
    return 1;
  }

  for (int i = 0; i < argc; ++i) {
    if (argv[i] == nullptr) {
      std::cout
        << "usage: model method train_file_path validation_file_path app_path\n"
        << "Supported model types are [resnet50, conv4]\n"
        << "Supported methods are [UN, L2N, CL2N]\n"
        << "train file: [app_path]/tasks/[train_file] is used for training\n"
        << "validation file: [app_path]/tasks/[validation_file] is used for "
           "validation\n"
        << "app_path: root path to refer to resources, if not given"
           "path is set current working directory\n";
      return 1;
    }
  }

  std::string model_str(argv[1]);
  std::string app_path =
    argc == 6 ? std::string(argv[5]) : simpleshot::getcwd_();
  std::string method = argv[2];
  std::string train_path = app_path + "/tasks/" + argv[3];
  std::string val_path = app_path + "/tasks/" + argv[4];

  try {
    ct_engine.registerLayerFactory(
      "cpu", nntrainer::createLayer<simpleshot::layers::CenteringLayer>);
  } catch (std::system_error &e) {
    std::cerr << "registering factory failed: " << e.what();
    return 1;
  } catch (std::exception &e) {
    std::cerr << "registering factory failed: " << e.what();
    return 1;
  }

  std::unique_ptr<ml::train::Model> model;
  try {
    model = simpleshot::createModel(model_str, app_path, method);
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  } catch (std::exception &e) {
    std::cerr << "creating Model failed: " << e.what();
    return 1;
  }

  std::shared_ptr<ml::train::Dataset> train_dataset, valid_dataset;
  try {
    train_dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                             train_path.c_str());
    valid_dataset =
      ml::train::createDataset(ml::train::DatasetType::FILE, val_path.c_str());

  } catch (...) {
    std::cerr << "creating dataset failed";
    return 1;
  }

  if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                        train_dataset)) {
    std::cerr << "failed to set train dataset" << std::endl;
    return 1;
  };

  if (model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                        valid_dataset)) {
    std::cerr << "failed to set valid dataset" << std::endl;
    return 1;
  };

  std::shared_ptr<ml::train::Optimizer> optimizer;
  try {
    optimizer = ml::train::optimizer::SGD({"learning_rate=0.1"});
  } catch (...) {
    std::cerr << "creating optimizer failed";
    return 1;
  }

  if (model->setOptimizer(optimizer) != 0) {
    std::cerr << "failed to set optimizer" << std::endl;
    return 1;
  }

  if (model->compile() != 0) {
    std::cerr << "model compilation failed" << std::endl;
    return 1;
  }

  if (model->initialize() != 0) {
    std::cerr << "model initiation failed" << std::endl;
    return 1;
  }

  if (model->train() != 0) {
    std::cerr << "train failed" << std::endl;
    return 1;
  }

  std::cout << "successfully ran" << std::endl;
  return 0;
}
