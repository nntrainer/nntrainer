// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 * @file   main.cpp
 * @date   14 October 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fstream>
#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

void saveToRaw(const float *data, size_t size, const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Cannot open file " << filename << " for writing.\n";
    return;
  }

  out.write(reinterpret_cast<const char *>(data), size * sizeof(float));
  out.close();

  std::cout << std::endl << ".bin generated successfully !";
}

int main() {
  auto model = ml::train::createModel();

  std::cout << "--------------------------------------Create Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    std::string path = "../../../../Applications/ONNX/python/qwen3/"
                       "single-token-fp16/qwen3_model.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Load Model "
               "Done--------------------------------------"
            << std::endl;

// Set FP16 tensor type BEFORE compilation for mixed precision
#ifdef ENABLE_FP16
  std::cout << "--------------------------------------Converting model to "
               "FP16--------------------------------------"
            << std::endl;
  model->setProperty({"model_tensor_type=FP16-FP16"});
  model->setProperty({"loss_scale=17768"});
#endif

  try {
    model->compile(ml::train::ExecutionMode::INFERENCE);
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Compile Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Initialize Model "
               "Done--------------------------------------"
            << std::endl;
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::cout << "--------------------------------------Summarize Model "
               "Done--------------------------------------"
            << std::endl;

  std::string weight_path =
    "../../../../Applications/ONNX/python/qwen3/single-token-fp16/bins/";
  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (std::exception &e) {
    std::cerr << "Error during loading weights: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Loading weights "
               "Done--------------------------------------"
            << std::endl;

  float *input = new float[1];
  float *sin = new float[128];
  float *cos = new float[128];
  float *epsilon = new float[1];
  const int num_vocab = 151936;

  input[0] = 52;

  for (int i = 0; i < 128; i++) {
    sin[i] = 0;
    cos[i] = 1;
  }
  epsilon[0] = 1e-6;

  std::vector<float *> in;

  in.push_back(epsilon);
  in.push_back(sin);
  in.push_back(cos);
  in.push_back(input);

  auto ans = model->inference(1, in);

  std::cout << "-------------------------------------------Inference "
               "Done--------------------------------------------"
            << std::endl;

  for (auto it : ans) {
    saveToRaw(it, num_vocab,
              "../../../../Applications/ONNX/python/qwen3/single-token-fp16/"
              "nntrainer_logits.bin");
  }

  return 0;
}
