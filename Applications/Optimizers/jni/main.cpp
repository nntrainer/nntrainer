// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jeonghun Park <top231902@naver.com>
 *
 * @file   main.cpp
 * @date   1 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jeonghun Park <top231902@naver.com>
 * @author Minseo Kim <ms05251@naver.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is an application for Optimizer validation.
 */

#include <algorithm>
#include <cifar_dataloader.h>
#include <dataset.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <layer.h>
#include <memory>
#include <model.h>
#include <numeric>
#include <optimizer.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Configuration & Globals
// -----------------------------------------------------------------------------

/**
 * @brief Application Dataset Type
 *
 */
enum class AppDataset { RANDOM, MNIST };

/**
 * @brief Application Configuration
 *
 */
struct AppConfig {
  AppDataset dataset = AppDataset::RANDOM;

  // Common Settings
  unsigned int batch_size = 32;
  unsigned int epochs = 5;
  float learning_rate = 1e-3f;
  float weight_decay = 0.0f;
  std::string optimizer_type = "lion";

  // Random Dataset Specific
  int number_of_db = 16; // Iterations per epoch for random

  // MNIST Dataset Specific
  std::string config_path;
  std::string data_path;
  unsigned int train_data_size = 100;
  unsigned int val_data_size = 100;
};

static AppConfig g_conf;

// MNIST Constants
constexpr unsigned int MNIST_SEED = 0;
constexpr unsigned int MNIST_FEATURE_SIZE = 784;
constexpr unsigned int MNIST_LABEL_SIZE = 10;

// -----------------------------------------------------------------------------
// Helpers: Weight Measurement (Common)
// -----------------------------------------------------------------------------

/**
 * @brief Weight Statistics
 *
 */
struct WeightStats {
  double l2 = 0.0;
  size_t num_elems = 0;
};

static WeightStats measure_weight_l2(ml::train::Model &model) {
  WeightStats ws{};
  model.forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      std::vector<float *> weights;
      std::vector<ml::train::TensorDim> dims;
      layer.getWeights(weights, dims);
      for (size_t wi = 0; wi < weights.size(); ++wi) {
        float *p = weights[wi];
        size_t n = dims[wi].getDataLen();
        ws.num_elems += n;
        double acc = 0.0;
        for (size_t k = 0; k < n; ++k) {
          double v = static_cast<double>(p[k]);
          acc += v * v;
        }
        ws.l2 += acc;
      }
    });
  return ws;
}

static std::vector<std::vector<float>>
snapshot_weights(ml::train::Model &model) {
  std::vector<std::vector<float>> snap;
  model.forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      std::vector<float *> weights;
      std::vector<ml::train::TensorDim> dims;
      layer.getWeights(weights, dims);
      for (size_t wi = 0; wi < weights.size(); ++wi) {
        float *p = weights[wi];
        size_t n = dims[wi].getDataLen();
        std::vector<float> buf(n);
        std::copy(p, p + n, buf.begin());
        snap.emplace_back(std::move(buf));
      }
    });
  return snap;
}

static WeightStats
measure_delta_l2(ml::train::Model &model,
                 const std::vector<std::vector<float>> &snapshot) {
  WeightStats ws{};
  size_t idx = 0;
  model.forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      std::vector<float *> weights;
      std::vector<ml::train::TensorDim> dims;
      layer.getWeights(weights, dims);
      for (size_t wi = 0; wi < weights.size(); ++wi) {
        float *p = weights[wi];
        size_t n = dims[wi].getDataLen();
        ws.num_elems += n;
        double acc = 0.0;
        for (size_t k = 0; k < n; ++k) {
          double d =
            static_cast<double>(p[k]) - static_cast<double>(snapshot[idx][k]);
          acc += d * d;
        }
        ws.l2 += acc;
        ++idx;
      }
    });
  return ws;
}

// -----------------------------------------------------------------------------
// Logic: Random Data Mode
// -----------------------------------------------------------------------------

std::unique_ptr<ml::train::Model> create_random_model() {
  std::vector<std::string> model_props = {"loss=mse",
                                          "model_tensor_type=FP32-FP32"};
  auto model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, model_props);

  // Simple Fully Connected Model
  model->addLayer(ml::train::createLayer("input", {"input_shape=1:1:10"}));
  model->addLayer(ml::train::createLayer("fully_connected",
                                         {"unit=5", "activation=softmax"}));

  return model;
}

std::unique_ptr<nntrainer::util::DataLoader> get_random_generator() {
  // Input: batch x 1 x 1 x 10, Output: batch x 1 x 1 x 5
  return std::unique_ptr<nntrainer::util::DataLoader>(
    new nntrainer::util::RandomDataLoader(
      {{static_cast<unsigned int>(g_conf.batch_size), 1u, 1u, 10u}},
      {{static_cast<unsigned int>(g_conf.batch_size), 1u, 1u, 5u}},
      static_cast<unsigned int>(g_conf.number_of_db)));
}

int random_dataset_cb(float **input, float **label, bool *last,
                      void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);
  data->next(input, label, last);
  return 0;
}

// -----------------------------------------------------------------------------
// Logic: MNIST Mode
// -----------------------------------------------------------------------------

/**
 * @brief Mnist Data Info for callback
 *
 */
class MnistDataInfo {
public:
  MnistDataInfo(unsigned int num_samples, const std::string &filename) :
    count(0),
    num_samples(num_samples),
    file(filename, std::ios::in | std::ios::binary),
    idxes(num_samples) {
    std::iota(idxes.begin(), idxes.end(), 0);
    rng.seed(MNIST_SEED);
    std::shuffle(idxes.begin(), idxes.end(), rng);
    if (!file.good()) {
      throw std::invalid_argument("Data file is not readable: " + filename);
    }
  }
  unsigned int count;
  unsigned int num_samples;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

static bool get_mnist_data_from_file(std::ifstream &F, float *input,
                                     float *label, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position =
    static_cast<uint64_t>((MNIST_FEATURE_SIZE + MNIST_LABEL_SIZE) *
                          static_cast<uint64_t>(id) * sizeof(float));

  if (position > file_length)
    return false;

  F.seekg(position, std::ios::beg);
  F.read(reinterpret_cast<char *>(input), sizeof(float) * MNIST_FEATURE_SIZE);
  F.read(reinterpret_cast<char *>(label), sizeof(float) * MNIST_LABEL_SIZE);
  return true;
}

static int mnist_dataset_cb(float **outVec, float **outLabel, bool *last,
                            void *user_data) {
  auto data = reinterpret_cast<MnistDataInfo *>(user_data);
  get_mnist_data_from_file(data->file, *outVec, *outLabel,
                           data->idxes.at(data->count));
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

static std::string try_find_path(const std::filesystem::path &start_base,
                                 const std::filesystem::path &relative_subpath,
                                 int max_up = 6) {
  std::filesystem::path base = start_base;
  for (int i = 0; i <= max_up; ++i) {
    std::filesystem::path candidate = base / relative_subpath;
    if (std::filesystem::exists(candidate))
      return candidate.string();
    base = base.parent_path();
  }
  return {};
}

static void resolve_mnist_paths() {
  std::filesystem::path cwd = std::filesystem::current_path();

  if (g_conf.config_path.empty()) {
    std::filesystem::path local = cwd / "mnist.ini";
    if (std::filesystem::exists(local))
      g_conf.config_path = local.string();
    else
      g_conf.config_path =
        try_find_path(cwd, "Applications/MNIST/res/mnist.ini");
  }

  if (g_conf.data_path.empty()) {
    std::filesystem::path local = cwd / "mnist_trainingSet.dat";
    if (std::filesystem::exists(local))
      g_conf.data_path = local.string();
    else
      g_conf.data_path =
        try_find_path(cwd, "Applications/MNIST/res/mnist_trainingSet.dat");
  }

  if (g_conf.config_path.empty() || g_conf.data_path.empty()) {
    throw std::runtime_error("MNIST config or data file not found. Please "
                             "provide --config and --data.");
  }
}

// -----------------------------------------------------------------------------
// Main CLI & Orchestration
// -----------------------------------------------------------------------------

static void parse_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto starts_with = [&](const char *prefix) {
      return arg.rfind(prefix, 0) == 0;
    };

    // Core Mode
    if (starts_with("--dataset=")) {
      std::string m = arg.substr(10);
      if (m == "mnist")
        g_conf.dataset = AppDataset::MNIST;
      else if (m == "random")
        g_conf.dataset = AppDataset::RANDOM;
    }
    // Common
    else if (starts_with("--opt="))
      g_conf.optimizer_type = arg.substr(6);
    else if (starts_with("--lr="))
      g_conf.learning_rate = std::stof(arg.substr(5));
    else if (starts_with("--wd="))
      g_conf.weight_decay = std::stof(arg.substr(5));
    else if (starts_with("--epochs="))
      g_conf.epochs = std::stoi(arg.substr(9));
    else if (starts_with("--bs="))
      g_conf.batch_size = std::stoi(arg.substr(5));
    // Random only
    else if (starts_with("--db="))
      g_conf.number_of_db = std::stoi(arg.substr(5));
    // MNIST only
    else if (starts_with("--config="))
      g_conf.config_path = arg.substr(9);
    else if (starts_with("--data="))
      g_conf.data_path = arg.substr(7);
    else if (starts_with("--train_size="))
      g_conf.train_data_size = std::stoi(arg.substr(13));
    else if (starts_with("--val_size="))
      g_conf.val_data_size = std::stoi(arg.substr(11));
    // Help
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: nntrainer_optimizers [options]\n\n"
                << "General Options:\n"
                << "  --dataset=random|mnist       (default: random)\n"
                << "  --opt=lion|adam|adamw|sgd (default: lion)\n"
                << "  --lr=<float>              Learning rate\n"
                << "  --wd=<float>              Weight decay\n"
                << "  --bs=<int>                Batch size\n"
                << "  --epochs=<int>            Epochs\n\n"
                << "Random Dataset Options:\n"
                << "  --db=<int>                Number of batches per epoch\n\n"
                << "MNIST Dataset Options:\n"
                << "  --config=<path>           Path to mnist.ini\n"
                << "  --data=<path>             Path to mnist_trainingSet.dat\n"
                << "  --train_size=<int>        Training samples count\n"
                << "  --val_size=<int>          Validation samples count\n";
      std::exit(0);
    }
  }
}

int main(int argc, char *argv[]) {
  try {
    parse_args(argc, argv);

    std::unique_ptr<ml::train::Model> model;
    std::unique_ptr<nntrainer::util::DataLoader> random_loader;
    std::unique_ptr<MnistDataInfo> mnist_train_info, mnist_val_info;
    std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;

    // 1. Initialize Model and Data based on Mode
    if (g_conf.dataset == AppDataset::RANDOM) {
      model = create_random_model();

      random_loader = get_random_generator();
      dataset_train =
        ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                 random_dataset_cb, random_loader.get());

      // Random Dataset usually just runs training, no separate validation set
      // in original logic
      model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);

    } else { // MNIST
      resolve_mnist_paths();

      // Load Model from INI
      model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
      model->load(g_conf.config_path,
                  ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);

      // Prepare Data
      mnist_train_info = std::make_unique<MnistDataInfo>(g_conf.train_data_size,
                                                         g_conf.data_path);
      mnist_val_info =
        std::make_unique<MnistDataInfo>(g_conf.val_data_size, g_conf.data_path);

      dataset_train =
        ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                 mnist_dataset_cb, mnist_train_info.get());
      dataset_val =
        ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                 mnist_dataset_cb, mnist_val_info.get());

      model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
      model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);
    }

    // 2. Configure Optimizer
    std::vector<std::string> opt_props = {"learning_rate=" +
                                          std::to_string(g_conf.learning_rate)};
    if (g_conf.weight_decay > 0.0f) {
      // Typically used by Lion, AdamW
      opt_props.emplace_back("weight_decay=" +
                             std::to_string(g_conf.weight_decay));
    }
    auto optimizer =
      ml::train::createOptimizer(g_conf.optimizer_type, opt_props);
    model->setOptimizer(std::move(optimizer));

    // 3. Set Common Training Properties
    model->setProperty({"batch_size=" + std::to_string(g_conf.batch_size),
                        "epochs=" + std::to_string(g_conf.epochs)});

    // 4. Compile and Initialize
    model->compile();
    model->initialize();

    // 5. Baseline Measurement
    auto snap_before = snapshot_weights(*model);
    auto ws_before = measure_weight_l2(*model);

    std::cout << "["
              << (g_conf.dataset == AppDataset::RANDOM ? "RANDOM" : "MNIST")
              << "] " << "Start Training. " << "opt=" << g_conf.optimizer_type
              << " " << "lr=" << g_conf.learning_rate << " "
              << "wd=" << g_conf.weight_decay << " "
              << "bs=" << g_conf.batch_size << " "
              << "w_l2_before=" << ws_before.l2 << std::endl;

    // 6. Train
    model->train();

    // 7. Post-Training Measurement
    auto ws_after = measure_weight_l2(*model);
    auto delta = measure_delta_l2(*model, snap_before);

    std::cout << "["
              << (g_conf.dataset == AppDataset::RANDOM ? "RANDOM" : "MNIST")
              << "] " << "Finished. " << "w_l2_after=" << ws_after.l2 << " "
              << "delta_l2=" << delta.l2 << " " << "elems=" << delta.num_elems
              << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Uncaught Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
