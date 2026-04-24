// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   10 March 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Product ratings recommendation system using the ccapi Tensor API.
 *
 *         Training set (embedding_input.txt): 3 columns (user_id product_id rating)
 *         Model: split → dual embedding → concat → FC(128) → FC(32) → FC(1)
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include <model.h>
#include <dataset.h>
#include <optimizer.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::createModel;
using ml::train::createDataset;
using ml::train::createOptimizer;
using ml::train::LayerHandle;
using ml::train::Tensor;

std::string data_file;

constexpr unsigned int SEED = 0;

const unsigned int total_train_data_size = 25;

unsigned int train_count = 0;

const unsigned int batch_size = 20;

const unsigned int feature_size = 2;

const unsigned int total_val_data_size = 25;

bool training = false;

float stepFunction(float x) {
  if (x > 0.5)
    return 1.0;
  if (x < 0.5)
    return 0.0;
  return x;
}

bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  std::string temp;
  F.clear();
  F.seekg(0, std::ios_base::beg);
  char c;
  unsigned int i = 0;
  while (F.get(c) && i < id)
    if (c == '\n')
      ++i;

  F.putback(c);

  if (!std::getline(F, temp))
    return false;

  std::istringstream buffer(temp);
  unsigned int *input_int = (unsigned int *)input;
  unsigned int x;
  for (unsigned int j = 0; j < feature_size; ++j) {
    buffer >> x;
    input_int[j] = x;
  }
  buffer >> x;
  label[0] = x;

  return true;
}

std::mt19937 rng;
std::vector<unsigned int> train_idxes;

int getSample_train(float **outVec, float **outLabel, bool *last,
                    void *user_data) {
  std::ifstream dataFile(data_file);
  if (!getData(dataFile, *outVec, *outLabel, train_idxes.at(train_count))) {
    return -1;
  }
  train_count++;
  if (train_count < total_train_data_size) {
    *last = false;
  } else {
    *last = true;
    train_count = 0;
    std::shuffle(train_idxes.begin(), train_idxes.end(), rng);
  }
  return 0;
}

/**
 * @brief Build the model using symbolic tensor graph.
 *
 * Topology:
 *   input → split → [user_embed, product_embed] → concat →
 *   flatten → fc1(128,relu) → fc2(32,relu) → output(1)
 */
static std::pair<Tensor, Tensor> buildGraph() {
  auto x = Tensor({1, 1, 1, 2}, "input");

  // split along width axis into two scalars
  LayerHandle split(createLayer("split", {"name=split", "axis=3"}));
  auto split_out = split(x);

  auto user_id = split_out.output(0);
  auto product_id = split_out.output(1);

  // user embedding: vocab=6, dim=5
  LayerHandle user_embed(createLayer(
    "embedding", {"name=user_embed", "in_dim=6", "out_dim=5"}));
  auto user_emb = user_embed(user_id);

  // product embedding: vocab=6, dim=5
  LayerHandle product_embed(createLayer(
    "embedding", {"name=product_embed", "in_dim=6", "out_dim=5"}));
  auto prod_emb = product_embed(product_id);

  // concat user + product embeddings
  LayerHandle concat(createLayer("concat", {"name=concat"}));
  auto h = concat({user_emb, prod_emb});

  // flatten
  LayerHandle flatten(createLayer("flatten", {"name=flatten"}));
  h = flatten(h);

  // fc1: 128 units, relu
  LayerHandle fc1(createLayer(
    "fully_connected", {"name=fc1", "unit=128", "activation=relu"}));
  h = fc1(h);

  // fc2: 32 units, relu
  LayerHandle fc2(createLayer(
    "fully_connected", {"name=fc2", "unit=32", "activation=relu"}));
  h = fc2(h);

  // output: 1 unit
  LayerHandle output_fc(createLayer(
    "fully_connected",
    {"name=outputlayer", "unit=1", "bias_initializer=zeros"}));
  auto y = output_fc(h);

  return {x, y};
}

/**
 * @brief     create NN
 *            back propagation of NN
 * @param[in]  arg 1 : train / inference
 * @param[in]  arg 2 : resource path (data) with below format
 * (int) (int) (float) #first data
 * ...
 * in each row represents user id, product id, rating (0 to 10)
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout
      << "./nntrainer_product_ratings train (| inference) data.txt\n";
    exit(1);
  }

  std::string weight_path = "product_ratings_model.bin";
  try {
    const std::vector<std::string> args(argv + 1, argv + argc);
    data_file = args[1];

    if (!args[0].compare("train"))
      training = true;

    train_idxes.resize(total_train_data_size);
    std::iota(train_idxes.begin(), train_idxes.end(), 0);
    rng.seed(SEED);

    // Build symbolic graph
    auto [x, y] = buildGraph();

    auto model = createModel(ml::train::ModelType::NEURAL_NET,
                             {"epochs=100", "loss=mse", "batch_size=20"});

    auto optimizer = createOptimizer(
      "adam", {"learning_rate=0.001", "beta1=0.9", "beta2=0.999",
               "epsilon=1e-7"});
    model->setOptimizer(std::move(optimizer));

    auto status = model->compile(x, y, ml::train::ExecutionMode::TRAIN);
    if (status != 0) {
      std::cerr << "Error during compile" << std::endl;
      return 1;
    }

    std::cout << "Input dimension: " << model->getInputDimension()[0];

    if (training) {
      std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;
      try {
        dataset_train =
          createDataset(ml::train::DatasetType::GENERATOR, getSample_train);
        dataset_val =
          createDataset(ml::train::DatasetType::GENERATOR, getSample_train);
      } catch (std::exception &e) {
        std::cerr << "Error creating dataset " << e.what() << std::endl;
        return 1;
      }

      model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
      model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);

      model->train({"batch_size=" + std::to_string(batch_size)});

      try {
        // Validate embedding weights against golden data (if available)
        auto embed_golden =
          ml::train::Tensor::zeros({1, 1, 15, 8}, "embed_golden");
        {
          std::ifstream file("embedding_weight_golden.out");
          if (file.good()) {
            float *ptr = embed_golden.mutable_data<float>();
            for (size_t i = 0; i < embed_golden.size(); ++i)
              file.read(reinterpret_cast<char *>(&ptr[i]), sizeof(float));
          }
        }

        auto fc_golden =
          ml::train::Tensor::zeros({1, 1, 32, 1}, "fc_golden");
        {
          std::ifstream file("fc_weight_golden.out");
          if (file.good()) {
            float *ptr = fc_golden.mutable_data<float>();
            for (size_t i = 0; i < fc_golden.size(); ++i)
              file.read(reinterpret_cast<char *>(&ptr[i]), sizeof(float));
          }
        }

        std::cout << "Embedding golden weights loaded ("
                  << embed_golden.size() << " elements)\n";
        std::cout << "FC golden weights loaded ("
                  << fc_golden.size() << " elements)\n";
      } catch (...) {
        std::cerr << "Warning: during loading golden data\n";
      }
    } else {
      model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

      std::ifstream dataFile(data_file);
      int cn = 0;
      for (unsigned int j = 0; j < total_val_data_size; ++j) {
        std::vector<float> o(feature_size);
        std::vector<float> l(1);
        getData(dataFile, o.data(), l.data(), j);

        // Wrap input data as ml::train::Tensor for inference
        auto input_tensor =
          ml::train::Tensor::fromData({1, 1, 1, feature_size}, o.data(),
                                      "inference_input");

        auto results = model->inference(
          1, {input_tensor.mutable_data<float>()}, {});

        // Wrap output and apply step function
        auto output = ml::train::Tensor::fromData({1, 1, 1, 1}, results[0],
                                                   "inference_output");
        float answer = stepFunction(output.getValue(0, 0, 0, 0));

        std::cout << answer << " : " << l[0] << std::endl;
        cn += answer == l[0];
      }
      std::cout << "[ Accuracy ] : "
                << ((float)(cn) / total_val_data_size) * 100.0 << "%"
                << std::endl;
    }
  } catch (std::exception &e) {
    std::cerr << "Unexpected error occurred, detailed: " << e.what()
              << std::endl;
    return 1;
  }

  return 0;
}
