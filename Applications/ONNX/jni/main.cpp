// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   26 Feb 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.honge@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

int main() {
  auto model = ml::train::createModel();

  try {
    std::string path = "../../../../Applications/ONNX/jni/llama_example.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  return 0;
}
