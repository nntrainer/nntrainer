// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   main_lora_train.cpp
 * @date   01 Apr 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Full Model training entry point for CausalLM
 */

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <causal_lm.h>
#include <factory.h>
#include <lora_train.h>
#include <transformer.h>

#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

#include <dataset.h>
#include <model.h>

using json = nlohmann::json;

std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  return architecture;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <model_dir> <train_data.txt> [--lr <float>] [--epochs <int>]"
                 " [--output <path>] [--max_samples <int>] [--skip_weights]"
              << std::endl;
    return 1;
  }

  std::string model_dir = argv[1];
  std::string train_data_path = argv[2];
  float lr = 1e-4f;
  unsigned int epochs = 1;
  std::string output_path = "model_weights.bin";
  int max_samples = -1;      // -1 = use all samples
  bool skip_weights = false; // skip loading pre-trained weights

  for (int i = 3; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--lr" && i + 1 < argc) {
      lr = std::atof(argv[++i]);
    } else if (arg == "--epochs" && i + 1 < argc) {
      epochs = std::atoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--max_samples" && i + 1 < argc) {
      max_samples = std::atoi(argv[++i]);
    } else if (arg == "--skip_weights") {
      skip_weights = true;
    }
  }

  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3SlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                        nntr_cfg);
    });

  try {
    std::string config_path = model_dir + "/config.json";
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::string nntr_config_path = model_dir + "/nntr_config.json";

    auto cfg = causallm::LoadJsonFile(config_path);
    auto gen_cfg = causallm::LoadJsonFile(gen_config_path);
    auto nntr_cfg = causallm::LoadJsonFile(nntr_config_path);

    std::cout << "=== CausalLM Full Model Training ===" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Train data: " << train_data_path << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    std::cout << "Max samples: "
              << (max_samples > 0 ? std::to_string(max_samples) : "all")
              << std::endl;
    std::cout << "Skip weights: " << (skip_weights ? "yes" : "no") << std::endl;

    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      gen_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Unknown architecture: " << architecture << std::endl;
      return 1;
    }

    model->initializeForTraining(lr, epochs);

    if (!skip_weights) {
      std::string weight_path =
        model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();
      std::cout << "Loading initial weights from: " << weight_path << std::endl;
      model->load_weight(weight_path);
    } else {
      std::cout << "Skipping weight loading (using random initialization)."
                << std::endl;
    }

    std::string tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();
    if (tokenizer_path.empty()) {
      // Auto-detect: look for tokenizer.json in the model directory
      tokenizer_path = model_dir + "/tokenizer.json";
      std::cout << "tokenizer_file not set, using: " << tokenizer_path
                << std::endl;
    }
    auto tokenizer_blob = causallm::LoadBytesFromFile(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);

    unsigned int seq_len = nntr_cfg["init_seq_len"].get<unsigned int>();
    unsigned int vocab_size = cfg["vocab_size"].get<unsigned int>();

    causallm::TrainingDataGenerator data_gen(tokenizer.get(), seq_len,
                                             vocab_size);
    data_gen.loadTextFile(train_data_path);

    // Optionally cap the number of training samples
    if (max_samples > 0 &&
        (unsigned int)max_samples < data_gen.getNumSamples()) {
      std::cout << "Limiting training samples from " << data_gen.getNumSamples()
                << " to " << max_samples << std::endl;
      data_gen.limitSamples(max_samples);
    }

    std::cout << "Training samples: " << data_gen.getNumSamples() << std::endl;
    std::cout << "Label buffer size per sample: " << vocab_size << " floats ("
              << (vocab_size * sizeof(float) / 1024) << " KB)" << std::endl;

    if (data_gen.getNumSamples() == 0) {
      std::cerr << "Error: Not enough training data (need > 0 lines)"
                << std::endl;
      return 1;
    }

    auto dataset_train =
      std::shared_ptr<ml::train::Dataset>(ml::train::createDataset(
        ml::train::DatasetType::GENERATOR,
        causallm::TrainingDataGenerator::dataCb, &data_gen));

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);

    std::cout << "\n=== Starting full model training ===" << std::endl;
    auto train_start = std::chrono::steady_clock::now();
    model->train();
    auto train_end = std::chrono::steady_clock::now();
    double elapsed_sec =
      std::chrono::duration<double>(train_end - train_start).count();

    std::cout << "\nTraining completed in " << elapsed_sec << " seconds."
              << std::endl;

    model->save_weight(output_path);
    std::cout << "Weights saved to: " << output_path << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
