// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mnist_npu_train.cpp
 * @date   6 August 2026
 * @brief  MNIST 3-layer FC training driver for NPU (Hexagon cDSP) validation.
 * @see    https://github.com/nntrainer/nntrainer
 *
 * This program trains a 3-layer fully-connected MNIST model with all FC
 * layers tagged engine=cdsp. When libggml-hexagon.so contains the
 * nntr_htp_bridge_sgemm_fp32 symbol, every forward and backward GEMM is
 * dispatched to the DSP. When the symbol is absent (or the bridge fails),
 * sgemm_fp32 transparently falls back to CPU, so this binary also works as a
 * plain CPU MNIST trainer.
 *
 * Usage:
 *   mnist_npu_train <config.ini> <train_images.idx3> <train_labels.idx1> \
 *                   [val_images.idx3] [val_labels.idx1]
 *
 * If val files are omitted, the training set is also used for validation.
 * If only the config is given, the default trainingSet.dat format is used.
 *
 * Build (on device with Hexagon cDSP support):
 *   meson setup build -Denable-hexagon-cdsp=true -Denable-transformer=true
 *   ninja -C build
 *   # The binary is at build/test/mnist_npu_train
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <dataset.h>
#include <model.h>

/**
 * @brief Read a 32-bit big-endian integer from a stream.
 */
static uint32_t read_be32(std::ifstream &f) {
  unsigned char buf[4];
  f.read(reinterpret_cast<char *>(buf), 4);
  return (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) |
         (uint32_t(buf[2]) << 8) | uint32_t(buf[3]);
}

/**
 * @brief MNIST IDX image file loader.
 *
 * Loads raw MNIST IDX3 images into a flat float vector (normalised to [0, 1]).
 * Each image is 28×28 = 784 floats.
 */
struct MnistImages {
  std::vector<float> data;
  uint32_t count = 0;
  uint32_t rows = 0;
  uint32_t cols = 0;

  bool load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      std::cerr << "Cannot open " << path << std::endl;
      return false;
    }
    uint32_t magic = read_be32(f);
    if (magic != 0x00000803) {
      std::cerr << "Bad magic " << magic << " in " << path
                << " (expected 0x803 for IDX3)" << std::endl;
      return false;
    }
    count = read_be32(f);
    rows = read_be32(f);
    cols = read_be32(f);
    size_t pixels = (size_t)count * rows * cols;
    std::vector<unsigned char> raw(pixels);
    f.read(reinterpret_cast<char *>(raw.data()), pixels);
    data.resize(pixels);
    for (size_t i = 0; i < pixels; ++i) {
      data[i] = static_cast<float>(raw[i]) / 255.0f;
    }
    std::cout << "Loaded " << count << " images (" << rows << "×" << cols
              << ") from " << path << std::endl;
    return true;
  }
};

/**
 * @brief MNIST IDX label file loader.
 *
 * Loads raw MNIST IDX1 labels into one-hot float vectors (10 classes).
 */
struct MnistLabels {
  std::vector<float> data;
  uint32_t count = 0;

  bool load(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      std::cerr << "Cannot open " << path << std::endl;
      return false;
    }
    uint32_t magic = read_be32(f);
    if (magic != 0x00000801) {
      std::cerr << "Bad magic " << magic << " in " << path
                << " (expected 0x801 for IDX1)" << std::endl;
      return false;
    }
    count = read_be32(f);
    std::vector<unsigned char> raw(count);
    f.read(reinterpret_cast<char *>(raw.data()), count);
    data.resize((size_t)count * 10, 0.0f);
    for (uint32_t i = 0; i < count; ++i) {
      data[(size_t)i * 10 + raw[i]] = 1.0f;
    }
    std::cout << "Loaded " << count << " labels from " << path << std::endl;
    return true;
  }
};

/**
 * @brief Holds MNIST data for the generator callback.
 */
struct MnistData {
  const MnistImages *images;
  const MnistLabels *labels;
  std::vector<uint32_t> indices;
  uint32_t pos = 0;
  std::mt19937 rng{42};

  MnistData(const MnistImages &img, const MnistLabels &lbl) :
    images(&img), labels(&lbl) {
    indices.resize(img.count);
    for (uint32_t i = 0; i < img.count; ++i) {
      indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), rng);
  }

  void reset() {
    pos = 0;
    std::shuffle(indices.begin(), indices.end(), rng);
  }
};

/**
 * @brief Generator callback for nntrainer's dataset API.
 *
 * Called by the DataBuffer to fetch one sample at a time. The framework
 * pre-allocates outVec[0] and outLabel[0] with sufficient space; we must
 * *copy* data into those buffers (not redirect the pointers). Sets *last =
 * true when the epoch is complete.
 */
static int getSample(float **outVec, float **outLabel, bool *last,
                     void *user_data) {
  auto *d = reinterpret_cast<MnistData *>(user_data);
  uint32_t idx = d->indices[d->pos];
  uint32_t feature_size = d->images->rows * d->images->cols;


  // Copy image data into the framework-provided buffer
  std::copy_n(d->images->data.data() + (size_t)idx * feature_size, feature_size,
              outVec[0]);
  // Copy one-hot label into the framework-provided buffer
  std::copy_n(d->labels->data.data() + (size_t)idx * 10, 10, outLabel[0]);

  d->pos++;
  if (d->pos >= d->images->count) {
    *last = true;
    d->reset();
  } else {
    *last = false;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <config.ini> [train_images.idx3 train_labels.idx1"
              << " val_images.idx3 val_labels.idx1]" << std::endl;
    return 1;
  }

  std::string config_path = argv[1];

  // --- Load MNIST data ---
  std::unique_ptr<MnistImages> train_imgs, val_imgs;
  std::unique_ptr<MnistLabels> train_lbls, val_lbls;
  std::unique_ptr<MnistData> train_data, val_data;

  if (argc >= 4) {
    // Raw MNIST IDX files provided.
    train_imgs = std::make_unique<MnistImages>();
    train_lbls = std::make_unique<MnistLabels>();
    if (!train_imgs->load(argv[2]) || !train_lbls->load(argv[3])) {
      return 1;
    }
    train_data = std::make_unique<MnistData>(*train_imgs, *train_lbls);

    if (argc >= 6) {
      val_imgs = std::make_unique<MnistImages>();
      val_lbls = std::make_unique<MnistLabels>();
      if (!val_imgs->load(argv[4]) || !val_lbls->load(argv[5])) {
        return 1;
      }
      val_data = std::make_unique<MnistData>(*val_imgs, *val_lbls);
    } else {
      // Use training data for validation.
      val_data = std::make_unique<MnistData>(*train_imgs, *train_lbls);
    }
  } else {
    // No MNIST files — try the existing trainingSet.dat format via FILE
    // dataset. This path does not use the generator callback.
    std::cout << "No MNIST IDX files provided; will use FILE dataset mode."
              << std::endl;
    std::cout << "Set NNTR_RES_PATH to the directory containing trainingSet.dat"
              << std::endl;
  }

  // --- Create and configure the model ---
  std::unique_ptr<ml::train::Model> model;
  try {
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  } catch (const std::exception &e) {
    std::cerr << "Failed to create model: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->load(config_path, ml::train::ModelFormat::MODEL_FORMAT_INI);
    model->compile();
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Model setup failed: " << e.what() << std::endl;
    return 1;
  }

  // --- Attach datasets ---
  if (train_data) {
    // Generator-based dataset (raw MNIST IDX files).
    std::shared_ptr<ml::train::Dataset> train_dataset =
      ml::train::createDataset(ml::train::DatasetType::GENERATOR, getSample,
                               train_data.get());
    train_dataset->setProperty({"buffer_size=100"});
    if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                          train_dataset) != 0) {
      std::cerr << "Failed to set training dataset" << std::endl;
      return 1;
    }

    std::shared_ptr<ml::train::Dataset> val_dataset = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, getSample, val_data.get());
    val_dataset->setProperty({"buffer_size=100"});
    if (model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                          val_dataset) != 0) {
      std::cerr << "Failed to set validation dataset" << std::endl;
      return 1;
    }
  } else {
    // FILE-based dataset (trainingSet.dat format).
    // The file must be in NNTR_RES_PATH.
    std::shared_ptr<ml::train::Dataset> train_dataset =
      ml::train::createDataset(ml::train::DatasetType::FILE, "trainingSet.dat");
    train_dataset->setProperty({"buffer_size=100"});
    if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                          train_dataset) != 0) {
      std::cerr << "Failed to set training dataset" << std::endl;
      return 1;
    }

    std::shared_ptr<ml::train::Dataset> val_dataset =
      ml::train::createDataset(ml::train::DatasetType::FILE, "trainingSet.dat");
    val_dataset->setProperty({"buffer_size=100"});
    if (model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                          val_dataset) != 0) {
      std::cerr << "Failed to set validation dataset" << std::endl;
      return 1;
    }
  }

  // --- Train ---
  std::cout << "Starting training..." << std::endl;
  std::cout << "  Config: " << config_path << std::endl;
  if (train_imgs) {
    std::cout << "  Train samples: " << train_imgs->count << std::endl;
  }
  std::cout << "  (GEMMs dispatch to Hexagon cDSP when bridge is available)"
            << std::endl;

  try {
    int rc = model->train();
    if (rc != 0) {

      std::cerr << "Training failed with error code: " << rc << std::endl;
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Training failed: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Training complete." << std::endl;
  return 0;
}
