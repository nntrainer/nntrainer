// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file sentence_transformer.cpp
 * @date 02 Jan 2026
 * @see https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 * @brief This file defines SentenceTransformer's basic actions
 */

#include <app_context.h>
#include <embedding_normalize_layer.h>
#include <embedding_pooling_layer.h>
#include <engine.h>
#include <sentence_transformer.h>

#include <filesystem>
#include <iostream>

namespace causallm {

SentenceTransformer::SentenceTransformer(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

std::map<std::string, std::string> SentenceTransformer::layer_map = {
  {"Pooling", "embedding_pooling"},
  {"Normalize", "embedding_normalize"},
  {"Dense", "fully_connected"}};

void SentenceTransformer::setupParameters(json &cfg, json &generation_cfg,
                                          json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  std::string modules_config_path = "modules.json";
  if (nntr_cfg.contains("module_config_path")) {
    modules_config_path = nntr_cfg["module_config_path"].get<std::string>();
  } else {
    std::cout << "module_config_path is not set. Using default: "
              << modules_config_path << std::endl;
  }

  // Get the directory containing modules.json to resolve relative paths
  std::filesystem::path modules_json_path(modules_config_path);
  std::filesystem::path base_dir = modules_json_path.parent_path();

  try {
    // 1. Load modules.json to get the structure and order of layers
    json modules_json = LoadJsonFile(modules_config_path);
    modules = modules_json.get<std::vector<json>>();

    for (auto &module : modules) {
      if (module.contains("path")) {
        std::string module_path_str = module["path"].get<std::string>();
        if (module_path_str.empty()) {
          // For the first module (Transformer), the path might be empty or ".""
          // We generally skip it or handle it if it points to a separate
          // config.
          continue;
        }

        // 2. Resolve config.json path for each module
        std::filesystem::path module_dir = base_dir / module_path_str;

        if (std::filesystem::exists(module_dir) &&
            std::filesystem::is_directory(module_dir)) {
          std::filesystem::path config_path = module_dir / "config.json";
          if (std::filesystem::exists(config_path)) {
            try {
              // 3. Load config.json and store it in module_configs map using
              // idx as key
              json module_config = LoadJsonFile(config_path.string());
              if (module.contains("idx")) {
                int idx = module["idx"].get<int>();
                module_configs[idx] = module_config;
              } else {
                std::cerr << "Warning: Module does not have idx field"
                          << std::endl;
              }
            } catch (const std::exception &e) {
              std::cerr << "Failed to load config for module: "
                        << module_path_str << " Reason: " << e.what()
                        << std::endl;
            }
          } else {
            // It's possible some modules don't have a config.json
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Failed to load modules config from: " << modules_config_path
              << " Reason: " << e.what() << std::endl;
  }
}

std::pair<Tensor, Tensor> SentenceTransformer::constructModel() {

  Tensor x;
  Tensor h;

  for (auto &module : modules) {
    if (!module.contains("type")) {
      continue;
    }
    std::string type = module["type"].get<std::string>();
    std::string component = getLastComponent(type);

    if (component == "Transformer") {
      auto result = Transformer::constructModel();
      x = result.first;
      h = result.second;
    } else {
      if (module.contains("idx")) {
        if (!h.isValid()) {
          throw std::runtime_error(
            "SentenceTransformer: encountered module '" + type +
            "' before any Transformer module — input tensor is undefined.");
        }
        int idx = module["idx"].get<int>();
        h = addModule(type, idx, h);
      } else {
        std::cerr << "Warning: Module does not have idx field, skipping: "
                  << type << std::endl;
      }
    }
  }

  return {x, h};
}

Tensor SentenceTransformer::addModule(const std::string &type, int idx,
                                      Tensor input) {
  json config;
  if (module_configs.find(idx) != module_configs.end()) {
    config = module_configs[idx];
  } else {
    // Config might be empty if no config.json was found.
    // This is valid for layers that don't satisfy specific configurations
    // (e.g., default behavior)
  }

  // Determine the layer type component (e.g., "Pooling" from
  // "sentence_transformers.models.Pooling")
  std::string component = getLastComponent(type);
  std::string layer_name;
  auto it = layer_map.find(component);
  if (it != layer_map.end()) {
    layer_name = it->second;
  }

  if (layer_name.empty()) {
    std::cerr << "Warning: No layer mapping found for module type: " << type
              << " (component: " << component
              << "). Skipping (passing input through)." << std::endl;
    return input;
  }

  // Convert JSON config to nntrainer property format (key=value strings)
  std::vector<std::string> props;
  for (auto &el : config.items()) {
    std::string val_str;
    if (el.value().is_string())
      val_str = el.value().get<std::string>();
    else
      val_str = el.value().dump(); // convert to string

    if (el.key() == "out_features") {
      props.push_back("unit=" + val_str);
    } else if (el.key() == "bias") {
      if (val_str == "false") {
        props.push_back("disable_bias=true");
      }
    } else if (el.key() == "activation_function") {
      if (val_str.find("Identity") == std::string::npos) {
        props.push_back("activation=" + val_str);
      } else {
        // need to support other activations later on
      }
    } else if (el.key() == "in_features") {
      // Ignore in_features as nntrainer infers it
    } else {
      props.push_back(el.key() + "=" + val_str);
    }
  }

  LayerHandle layer(ml::train::createLayer(layer_name, props));
  return layer(input);
}

void SentenceTransformer::run(const WSTR prompt, bool do_sample,
                              const WSTR system_prompt, const WSTR tail_prompt,
                              bool log_output) {

  try {
    std::vector<float *> results = encode(prompt, system_prompt, tail_prompt);

    if (log_output) {

      std::cout << "Embedding Result (" << BATCH_SIZE
                << " batch(es)):" << std::endl;
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        std::cout << "Batch " << b << ": [";
        // Print first few elements as sample
        int print_dim = (DIM > 10) ? 10 : DIM;
        for (int i = 0; i < print_dim; ++i) {
          std::cout << results[0][b * DIM + i]
                    << (i == print_dim - 1 ? "" : ", ");
        }
        if (DIM > 10)
          std::cout << ", ...";
        std::cout << "] (Total DIM: " << DIM << ")" << std::endl;
      }
    }

    // output should be deallocated after use.
    for (auto out : results) {
      delete[] out;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during embedding run: " << e.what() << std::endl;
  }
}

std::vector<float *> SentenceTransformer::encode(const WSTR prompt,
                                                 const WSTR system_prompt,
                                                 const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error(
      "SentenceTransformer model is not initialized. Please call "
      "initialize() before encode().");
  }

#if defined(_WIN32)
  std::wstring prompt_ = system_prompt + prompt + tail_prompt;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto _input = tokenizer->Encode(converter.to_bytes(prompt_), true);
#else
  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto _input = tokenizer->Encode(prompt_, true);
#endif

  std::vector<int64_t> init_input;
  unsigned int input_len =
    std::min((unsigned int)_input.size(), (unsigned int)MAX_SEQ_LEN);

  // feed only available length
  for (unsigned int i = 0; i < input_len; ++i)
    init_input.push_back(_input[i]);

  float *input_sample =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(init_input[i]);
    }
  }

  std::vector<float *> input;
  input.push_back(input_sample);

  std::vector<float *> label; // Empty label for inference

  // Run incremental inference for the prefill stage
  // start: 0, end: input_len (process all tokens at once)
  // This performs a single forward pass for the entire prompt sequence to get
  // embeddings.
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, input_len, 0, input_len, false);

  free(input_sample);

  return output;
}

std::string SentenceTransformer::getLastComponent(const std::string &type) {
  std::string last_component = type;
  size_t last_dot_pos = type.find_last_of('.');
  if (last_dot_pos != std::string::npos) {
    last_component = type.substr(last_dot_pos + 1);
  }
  return last_component;
}

void SentenceTransformer::registerCustomLayers() {
  Transformer::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingPoolingLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingNormalizeLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
