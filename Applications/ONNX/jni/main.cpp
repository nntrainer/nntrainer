// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   26 Feb 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nntrainer/nntrainer
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

  std::cout << "--------------------------------------Create Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    std::string path =
      "../../../../Applications/ONNX/python/simple_model/model.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Load Model "
               "Done--------------------------------------"
            << std::endl;
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
    "../../../../Applications/ONNX/python/simple_model/";
  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (std::exception &e) {
    std::cerr << "Error during loading weights: " << e.what() << "\n";
    return 1;
  }

  float *input1 = new float[4];
  for (int i = 0; i < 4; i++) {
    input1[i] = (i + 1);
  }
  std::vector<float *> in;
  in.push_back(input1);
  auto ans = model->inference(1, in);

  std::cout << "-------------------------------------------Inference "
               "Done--------------------------------------------"
            << std::endl;
  for (auto it : ans) {
    std::cout << "Ans: " << it[0] << std::endl;
  }
  return 0;
}
