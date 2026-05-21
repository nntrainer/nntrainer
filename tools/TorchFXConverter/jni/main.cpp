// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main.cpp
 * @date   15 Mar 2026
 * @brief  Test driver for TorchFXConverter-generated NNTrainer models.
 *
 * Instantiates the auto-generated model class, registers custom layers,
 * constructs the model graph, compiles, and initializes it.
 * Exit code 0 = success.
 *
 * Usage:
 *   ./converter_model_test          # uses compiled-in defaults
 */

#include <iostream>
#include <stdexcept>

#include GENERATED_MODEL_HEADER

int main(int argc, char *argv[]) {
  try {
    GENERATED_MODEL_CLASS model_builder;

    std::cout << "[converter_test] Initializing model..." << std::endl;
    model_builder.initialize();

    auto &model = model_builder.getModel();
    std::cout << "[converter_test] Summarizing model..." << std::endl;
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    std::cout << "[converter_test] Model initialized successfully."
              << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[converter_test] FAILED: " << e.what() << std::endl;
    return 1;
  }
}
