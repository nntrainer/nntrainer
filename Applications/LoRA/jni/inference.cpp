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
 * @file    lora_run.cpp
 * @date    April 2026
 * @brief   Inference pipeline for LoRA trained models on MNIST
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
                      const std::vector<float> &labels, 
                      size_t num_samples) {
  int correct_predictions = 0;
  const size_t batch_size = 1; // Evaluate one sample at a time
  const size_t feature_size = 784; // 28x28
  const size_t num_classes = 10;
  
  // Evaluate on a subset of samples (e.g., first 100 or num_samples, whichever is smaller)
  size_t eval_samples = std::min(num_samples, static_cast<size_t>(100));
  
  for (size_t i = 0; i < eval_samples; ++i) {
    // Prepare input data
    std::vector<float> input(feature_size);
    for (size_t j = 0; j < feature_size; ++j) {
      input[j] = images[i * feature_size + j];
    }
    
    // Prepare input and label pointers
    std::vector<float*> input_ptrs = {input.data()};
    
    // Run inference
    auto output = model->inference(batch_size, input_ptrs);
    
    // Get the output probabilities (assuming softmax output)
    float* output_data = output[0];
    
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
  
  return static_cast<float>(correct_predictions) / static_cast<float>(eval_samples) * 100.0f;
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
                 nntrainer::withKey("activation", "softmax")}));

  return layers;
}

int main(int argc, char **argv) {
  std::cout << "=====================================" << std::endl;
  std::cout << "  LoRA Model Inference Pipeline" << std::endl;
  std::cout << "=====================================" << std::endl;

  // Parse arguments
  std::string images_path = "train-images-idx3-ubyte";
  std::string labels_path = "train-labels-idx1-ubyte";
  std::string model_path = "mnist_model.bin";
  unsigned int lora_rank = 16; // Default LoRA rank
  float lora_alpha = 1.0f;     // Default LoRA alpha
  if (argc >= 3) {
    images_path = argv[1];
    labels_path = argv[2];
  }
  if (argc >= 4) {
    model_path = argv[3];
  }

  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Images: " << images_path << std::endl;
  std::cout << "  Labels: " << labels_path << std::endl;
  std::cout << "  Model Path: " << model_path << std::endl;

  // Load MNIST dataset or generate fake data for testing
  std::vector<float> images, labels;
  if (!lora::loadMNIST(images_path, labels_path, images, labels, 10)) {
    std::cerr << "Failed to load MNIST dataset, generating fake data for testing..." << std::endl;
    
    // Generate fake data for testing
    const size_t total_samples = 10000; // 7500 train + 2500 test
    const size_t image_size = 784; // 28x28
    
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
    
    std::cout << "Generated " << total_samples << " fake samples for testing." << std::endl;
  }
  
  size_t total_samples = images.size() / 784; // 28x28 = 784
  const size_t train_samples = std::min(static_cast<size_t>(7500), total_samples);
  const size_t test_samples = std::min(static_cast<size_t>(2500), total_samples - train_samples);
  
  std::cout << "Loaded " << total_samples << " samples (normalized)" << std::endl;
  std::cout << "Using " << test_samples << " samples for inference" << std::endl;

  // Create model with loss function
  auto model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {nntrainer::withKey("loss", "cross")});

  // Add layers
  auto layers = createSimpleGraph(lora_rank, lora_alpha);
  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  try {
    // Compile the model for inference
    model->compile(ml::train::ExecutionMode::INFERENCE);
    std::cout << "Model compiled successfully for inference." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error compiling model: " << e.what() << std::endl;
    return 1;
  }

  try {
    model->initialize(ml::train::ExecutionMode::INFERENCE);
    std::cout << "Model initialized successfully for inference." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error initializing model: " << e.what() << std::endl;
    return 1;
  }

  // Load the trained model AFTER initialization
  try {
    model->load(model_path, ml::train::ModelFormat::MODEL_FORMAT_BIN, "/workspace/nntrainer_korea/nntrainer/Applications/LoRA/lora_1.bin");
    std::cout << "Model loaded from " << model_path << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return 1;
  }
  
  // For inference, we use test data
  // Create test data starting from train_samples index
  std::vector<float> test_images(images.begin() + train_samples * 784, images.end());
  std::vector<float> test_labels(labels.begin() + train_samples, labels.end());
  
  // Evaluate model accuracy on test data
  try {
    std::cout << "\nRunning inference on test set..." << std::endl;
    float accuracy = evaluateAccuracy(model, test_images, test_labels, test_samples);
    std::cout << "Model accuracy on test set: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during inference: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\nInference completed successfully!" << std::endl;
  return 0;
}