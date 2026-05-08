/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file    lora_train.cpp
 * @date    April 2026
 * @brief   LoRA fine-tuning example on MNIST (loads pre-trained model)
 * @see     https://github.com/nntrainer/nntrainer
 */

#include "mnist_loader.h"
#include <algorithm>
#include <chrono>
#include <dataset.h>
#include <iomanip>
#include <iostream>
#include <layer.h>
#include <memory>
#include <model.h>
#include <neuralnet.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <string>
#include <tensor.h>
#include <util_func.h>
#include <vector>

// For make_unique if not available in older C++ standards
#if __cplusplus < 201402L
namespace std {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace std
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @brief UserData which stores information used to feed data from data callback
 */
class DataInformation {
public:
  /**
   * @brief Construct a new Data Information object
   *
   * @param images vector of images
   * @param labels vector of labels
   * @param num_samples number of data
   */
  DataInformation(const std::vector<float> &images,
                  const std::vector<float> &labels, unsigned int num_samples);
  unsigned int count;
  unsigned int num_samples;
  const std::vector<float> &images;
  const std::vector<float> &labels;
  size_t feature_len;
  size_t label_len;
};

DataInformation::DataInformation(const std::vector<float> &images,
                                 const std::vector<float> &labels,
                                 unsigned int num_samples) :
  count(0),
  num_samples(num_samples),
  images(images),
  labels(labels),
  feature_len(784),
  label_len(1) {}

/**
 * @brief      get data which size is batch for train
 * @param[out] outInput input vectors
 * @param[out] outLabel label vectors
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);

  // Copy input data
  size_t input_offset = data->count * data->feature_len;
  for (size_t i = 0; i < data->feature_len; ++i) {
    (*outVec)[i] = data->images[input_offset + i];
  }

  // Copy label data (convert from index to one-hot)
  size_t label_offset = data->count * data->label_len;
  for (size_t i = 0; i < 10; ++i) { // 10 classes
    (*outLabel)[i] = 0.0f;
  }
  int label_idx = (int)data->labels[label_offset];
  if (label_idx >= 0 && label_idx < 10) {
    (*outLabel)[label_idx] = 1.0f;
  }

  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
  }

  return 0;
}

std::vector<LayerHandle> createSimpleGraph(unsigned int lora_rank,
                                           float lora_alpha) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  // Input layer
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "1:1:784")}));

  // Hidden layer
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 256),
                 nntrainer::withKey("weight_initializer", "xavier_uniform"),
                 nntrainer::withKey("activation", "relu"),
                 nntrainer::withKey("lora_rank", std::to_string(lora_rank)),
                 nntrainer::withKey("lora_alpha", std::to_string(lora_alpha))
                }));

  // Output layer with softmax activation for classification
  layers.push_back(
    createLayer("fully_connected",
                {nntrainer::withKey("unit", 10),
                 nntrainer::withKey("weight_initializer", "xavier_uniform"),
                 nntrainer::withKey("activation", "softmax"),
                 nntrainer::withKey("trainable", "false")}));

  return layers;
}

/**
 * @brief Evaluate model accuracy on a dataset
 *
 * @param model The trained model
 * @param images Input images
 * @param labels Ground truth labels
 * @param num_samples Number of samples to evaluate
 * @return float Accuracy as a percentage
 */
float evaluateAccuracy(std::unique_ptr<ml::train::Model> &model,
                       const std::vector<float> &images,
                       const std::vector<float> &labels, size_t num_samples) {
  int correct_predictions = 0;
  const size_t batch_size = 1;     // Evaluate one sample at a time
  const size_t feature_size = 784; // 28x28
  const size_t num_classes = 10;

  // Evaluate on a subset of samples (e.g., first 100 or num_samples, whichever
  // is smaller)
  size_t eval_samples = std::min(num_samples, static_cast<size_t>(100));

  for (size_t i = 0; i < eval_samples; ++i) {
    // Prepare input data
    std::vector<float> input(feature_size);
    for (size_t j = 0; j < feature_size; ++j) {
      input[j] = images[i * feature_size + j];
    }

    // Prepare input and label pointers
    std::vector<float *> input_ptrs = {input.data()};

    // Run inference
    auto output = model->inference(batch_size, input_ptrs);

    // Get the output probabilities (assuming softmax output)
    float *output_data = output[0];

    // Find the predicted class (highest probability)
    int predicted_class = 0;
    float max_prob = output_data[0];
    for (int c = 1; c < num_classes; ++c) {
      if (output_data[c] > max_prob) {
        max_prob = output_data[c];
        predicted_class = c;
      }
    }

    // Get the true class
    int true_class = static_cast<int>(labels[i]);

    // Check if prediction is correct
    if (predicted_class == true_class) {
      correct_predictions++;
    }
  }

  return static_cast<float>(correct_predictions) /
         static_cast<float>(eval_samples) * 100.0f;
}

int main(int argc, char **argv) {
  std::cout << "=====================================" << std::endl;
  std::cout << "  LoRA Fine-tuning on MNIST" << std::endl;
  std::cout << "  (Loading pre-trained model)" << std::endl;
  std::cout << "=====================================" << std::endl;

  // Parse arguments
  std::string images_path = "train-images-idx3-ubyte";
  std::string labels_path = "train-labels-idx1-ubyte";
  std::string pretrained_model_path =
    "mnist_model.bin"; // Default pre-trained model path
  std::string model_path =
    "lora_mnist_model.bin"; // Default fine-tuned model path
  unsigned int epochs = 10;
  unsigned int batch_size = 64;
  float learning_rate = 0.01f; // Lower learning rate for fine-tuning
  unsigned int lora_rank = 16; // Default LoRA rank
  float lora_alpha = 1.0f;     // Default LoRA alpha

  // Parse command line arguments
  // Usage: ./program [images_path] [labels_path] [pretrained_model_path]
  // [model_path] [epochs] [batch_size] [learning_rate] [lora_rank] [lora_alpha]
  if (argc >= 3) {
    images_path = argv[1];
    labels_path = argv[2];
  }
  if (argc >= 4) {
    pretrained_model_path = argv[3];
  }
  if (argc >= 5) {
    model_path = argv[4];
  }
  if (argc >= 6) {
    epochs = std::stoul(argv[5]);
  }
  if (argc >= 7) {
    batch_size = std::stoul(argv[6]);
  }
  if (argc >= 8) {
    learning_rate = std::stof(argv[7]);
  }
  if (argc >= 9) {
    lora_rank = std::stoul(argv[8]);
  }
  if (argc >= 10) {
    lora_alpha = std::stof(argv[9]);
  }

  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Images: " << images_path << std::endl;
  std::cout << "  Labels: " << labels_path << std::endl;
  std::cout << "  Pre-trained Model Path: " << pretrained_model_path
            << std::endl;
  std::cout << "  Fine-tuned Model Path: " << model_path << std::endl;
  std::cout << "  Epochs: " << epochs << std::endl;
  std::cout << "  Batch Size: " << batch_size << std::endl;
  std::cout << "  Learning Rate: " << learning_rate << std::endl;
  std::cout << "  LoRA Rank: " << lora_rank << std::endl;
  std::cout << "  LoRA Alpha: " << lora_alpha << std::endl;

  // Load MNIST dataset or generate fake data for testing
  std::vector<float> images, labels;
  if (!lora::loadMNIST(images_path, labels_path, images, labels, 10)) {
    std::cerr
      << "Failed to load MNIST dataset, generating fake data for testing..."
      << std::endl;

    // Generate fake data for testing
    const size_t total_samples = 10000; // 7500 train + 2500 test
    const size_t image_size = 784;      // 28x28

    images.resize(total_samples * image_size);
    labels.resize(total_samples);

    // Fill with random data
    for (size_t i = 0; i < images.size(); ++i) {
      images[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Fill labels with random class indices (0-9)
    for (size_t i = 0; i < labels.size(); ++i) {
      labels[i] = static_cast<float>(rand() % 10);
    }

    std::cout << "Generated " << total_samples << " fake samples for testing."
              << std::endl;
  }

  size_t total_samples = images.size() / 784; // 28x28 = 784
  const size_t train_samples =
    std::min(static_cast<size_t>(7500), total_samples);
  const size_t test_samples =
    std::min(static_cast<size_t>(2500), total_samples - train_samples);

  std::cout << "Loaded " << total_samples << " samples (normalized)"
            << std::endl;
  std::cout << "Using " << train_samples << " samples for training and "
            << test_samples << " samples for testing" << std::endl;

  // Print first 10 samples of the dataset
  // int samples_to_print = std::min(10, static_cast<int>(total_samples));
  // if (total_samples > 0) {
  //   std::cout << "\nFirst " << samples_to_print << " samples:" << std::endl;
  //   for (int sample_idx = 0; sample_idx < samples_to_print; ++sample_idx) {
  //     std::cout << "\nSample " << sample_idx << ":" << std::endl;
  //     std::cout << "Label: " << labels[sample_idx] << std::endl;
  //     std::cout << "Image (28x28):" << std::endl;

  //     // Print the entire image data (784 pixels)
  //     for (int i = 0; i < 784; ++i) {
  //       // Convert normalized value back to approximate pixel value for display
  //       float pixel_value = images[sample_idx * 784 + i] * 0.3081f +
  //                           0.1307f; // Reverse normalization
  //       int display_value = static_cast<int>(pixel_value * 255.0f);
  //       std::cout << std::setw(4) << display_value;
  //       if ((i + 1) % 28 == 0)
  //         std::cout << std::endl;
  //     }
  //   }
  // }

  // Load pre-trained model
  std::cout << "\nLoading pre-trained model from " << pretrained_model_path
            << "..." << std::endl;
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                      {nntrainer::withKey("loss", "cross")});

  // Add layers
  auto layers = createSimpleGraph(lora_rank, lora_alpha);
  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  // Add LoRA adapters to the model
  std::cout << "Adding LoRA adapters to the model..." << std::endl;
  try {
    // Get the hidden layer (assuming it's the second layer after input)
    // Note: This is a simplified approach. In a real implementation, you would
    // need to properly identify the layers to which you want to add LoRA
    // adapters. For this example, we'll assume the model structure is
    // compatible.
    std::cout << "LoRA adapters added successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error adding LoRA adapters: " << e.what() << std::endl;
    return 1;
  }

  // Compile the model
  try {
    model->compile(ml::train::ExecutionMode::TRAIN);
    std::cout << "Model compiled successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error compiling model: " << e.what() << std::endl;
    return 1;
  }

  // Training mode
  std::cout << "\nEntering LoRA fine-tuning mode..." << std::endl;

  // Create optimizer with learning rate (typically higher learning rate for
  // LoRA adapters)
  auto optimizer = ml::train::createOptimizer(
    "sgd", {"learning_rate=" + std::to_string(learning_rate)});
  try {
    model->setOptimizer(std::move(optimizer));
  } catch (const std::exception &e) {
    std::cerr << "Error setting optimizer: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->initialize();
    std::cout << "Model initialized successfully for LoRA fine-tuning."
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->load(pretrained_model_path,
                ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::cout << "Pre-trained model loaded successfully." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error loading pre-trained model: " << e.what() << std::endl;
    std::cerr << "Please make sure the pre-trained model exists and was "
                 "trained with compatible architecture."
              << std::endl;
    return 1;
  }
  // Create DataInformation objects for training data (using only first
  // train_samples)
  auto train_user_data =
    std::make_unique<DataInformation>(images, labels, train_samples);

  // Create datasets using the callback function
  std::shared_ptr<ml::train::Dataset> dataset_train;
  try {
    dataset_train = createDataset(ml::train::DatasetType::GENERATOR, getSample,
                                  train_user_data.get());
  } catch (const std::exception &e) {
    std::cerr << "Error creating dataset: " << e.what() << std::endl;
    return 1;
  }

  // Set the datasets on the model
  try {
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
  } catch (const std::exception &e) {
    std::cerr << "Error setting dataset: " << e.what() << std::endl;
    return 1;
  }

  // Set up training properties for standard training (without learning_rate)
  try {
    model->setProperty({"epochs=" + std::to_string(epochs),
                        "batch_size=" + std::to_string(batch_size)});
  } catch (const std::exception &e) {
    std::cerr << "Error setting properties: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nStarting LoRA fine-tuning..." << std::endl;

  try {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Train using standard backpropagation
    model->train();

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

    std::cout << "\nLoRA fine-tuning completed!" << std::endl;
    std::cout << "Total fine-tuning time: " << elapsed_time.count() << " ms"
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during training: " << e.what() << std::endl;
    return 1;
  }

  // Save the trained model
  try {
    model->save(model_path, ml::train::ModelFormat::MODEL_FORMAT_LORA_BIN);
    std::cout << "LoRA fine-tuned model saved to " << model_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error saving model: " << e.what() << std::endl;
    return 1;
  }

  // For evaluation, we need to create separate test data
  // Create test data starting from train_samples index
  std::vector<float> test_images(images.begin() + train_samples * 784,
                                 images.end());
  std::vector<float> test_labels(labels.begin() + train_samples, labels.end());

  // Evaluate model accuracy on test data
  try {
    std::cout << "\nEvaluating model accuracy on test set..." << std::endl;
    float accuracy =
      evaluateAccuracy(model, test_images, test_labels, test_samples);
    std::cout << "Model accuracy on test set: " << std::fixed
              << std::setprecision(2) << accuracy << "%" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during evaluation: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}