/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "chat_template.h"
#include "model_registry.h"
#include "runtime_config.h"
#ifdef ENABLE_CAUSALLM_SERVER
#include "serving/causal_backend.h"
#include "serving/http_server.h"
#endif
#include <models/gemma3/function.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

namespace {

void resolveNntrConfigPath(json &nntr_cfg, const std::string &key,
                           const std::string &model_path) {
  if (!nntr_cfg.contains(key) || !nntr_cfg[key].is_string())
    return;

  std::filesystem::path path = nntr_cfg[key].get<std::string>();
  if (path.empty() || path.is_absolute())
    return;

  nntr_cfg[key] = (std::filesystem::path(model_path) / path).string();
}

} // namespace

/**
 * @brief Print the maximum resident set size for the current process.
 */
void printMemoryUsage() {
#if defined(_WIN32)
  std::cout << "Max Resident Set Size: unavailable on Windows" << std::endl;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
#endif
}

/**
 * @brief Read the current process resident set size on Linux.
 */
size_t read_vm_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.find("VmRSS:") == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  return 0;
#endif
}

/**
 * @brief Read private resident memory from smaps_rollup on Linux.
 */
size_t read_private_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream smaps("/proc/self/smaps_rollup");
  std::string line;
  size_t total = 0;
  while (std::getline(smaps, line)) {
    if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0) {
      size_t kb;
      sscanf(line.c_str(), "%*s %zu", &kb);
      total += kb;
    }
  }
  return total;
#endif
}

/**
 * @brief Start a background sampler for peak private RSS.
 */
void start_peak_tracker() {
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      size_t prev = peak_rss_kb.load();
      if (current > prev) {
        peak_rss_kb.store(current);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

/**
 * @brief Stop the memory sampler and print the observed peak.
 */
void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

/**
 * @brief Resolve config architecture names to registered model factory names.
 */
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
    } else if (architecture == "BertForMaskedLM") {
      return "MultilingualTinyBert";
    } else if (architecture == "XLMRobertaForMaskedLM" ||
               architecture == "XLMRobertaModel") {
      return "XLMRobertaForMaskedLM";
    } else if (architecture == "TimmViT" ||
               architecture == "vit_base_patch16_siglip_224") {
      return "TimmViT";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  if (architecture == "TimmViT" ||
      architecture == "vit_base_patch16_siglip_224") {
    return "TimmViT";
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  return architecture;
}

/**
 * @brief Entry point for loading, initializing, and running a CausalLM model.
 */
int main(int argc, char *argv[]) {

  auto start_time = std::chrono::high_resolution_clock::now();

  /* INITBISECT_MARKS: env-gated init-phase trace (NNTR_INIT_TRACE). */
  static const bool _itrace = std::getenv("NNTR_INIT_TRACE") != nullptr;
  auto _ilap = [&](const char *what) {
    if (!_itrace)
      return;
    std::fprintf(stderr, "[init-trace] %8.1f ms  rss=%zu kB  %s\n",
                 std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - start_time)
                   .count(),
                 read_vm_rss_kb(), what);
    std::fflush(stderr);
  };

  /** Register all runnable causallm models to factory */
  causallm::registerAllModels();

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input or "
                 "chat_input if omitted)\n"
              << "  --serve [--host HOST] [--port PORT] [--model-id ID] "
                 "[--max-sessions N] [--queue-capacity N]\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  const bool serve = argc >= 3 && std::string(argv[2]) == "--serve";
#ifndef ENABLE_CAUSALLM_SERVER
  if (serve) {
    std::cerr << "Reconfigure with -Denable-causallm-server=true\n";
    return EXIT_FAILURE;
  }
#endif
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
#ifdef ENABLE_CAUSALLM_SERVER
    std::string host = "127.0.0.1";
    std::string model_id =
      std::filesystem::path(model_path).filename().string();
    int port = 8080;
    size_t max_sessions = 32, queue_capacity = 32;
    if (serve) {
      for (int i = 3; i < argc; i += 2) {
        if (i + 1 == argc)
          throw std::invalid_argument("Missing server option value");
        std::string key = argv[i], value = argv[i + 1];
        if (key == "--host")
          host = value;
        else if (key == "--model-id")
          model_id = value;
        else {
          size_t end = 0;
          auto number = std::stoul(value, &end);
          if (end != value.size() || value[0] == '-' || number == 0 ||
              number > 65535)
            throw std::invalid_argument("Invalid server numeric option");
          if (key == "--port")
            port = static_cast<int>(number);
          else if (key == "--max-sessions")
            max_sessions = number;
          else if (key == "--queue-capacity")
            queue_capacity = number;
          else
            throw std::invalid_argument("Unknown server option: " + key);
        }
      }
      if (model_id.empty())
        throw std::invalid_argument("Model id must not be empty");
      // Defaults apply only when not explicitly configured by the caller.
      setenv("NNTR_ENGINE", "cpu", 0);
      setenv("NNTR_KV_PAGED", "1", 0);
      setenv("NNTR_KV_PREFIX_CACHE", "1", 0);
      const auto runtime = causallm::RuntimeConfig::capture();
      if (!runtime.cpu_kv_paged)
        throw std::invalid_argument("Serving requires paged CPU KV storage");
    }
#endif
    // Load configuration files
    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg = json::object();
    std::string generation_config_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(generation_config_path)) {
      generation_cfg = causallm::LoadJsonFile(generation_config_path);
    }
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");
    if (serve && nntr_cfg.value("batch_size", 1) != 1)
      throw std::invalid_argument("Serving requires batch_size=1");
    resolveNntrConfigPath(nntr_cfg, "tokenizer_file", model_path);
    resolveNntrConfigPath(nntr_cfg, "embedding_file_name", model_path);
    resolveNntrConfigPath(nntr_cfg, "ple_file_name", model_path);
    resolveNntrConfigPath(nntr_cfg, "model_file_name", model_path);
    resolveNntrConfigPath(nntr_cfg, "binary_config_path", model_path);
    resolveNntrConfigPath(nntr_cfg, "image_newline_path", model_path);

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path (already rebased onto model_path above)
    const std::string weight_file =
      nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;
    _ilap("config json parsed");

    // Initialize and run model
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else if (cfg.contains("architecture") &&
               cfg["architecture"].is_string()) {
      architecture = cfg["architecture"].get<std::string>();
    } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
      architecture = cfg["model_type"].get<std::string>();
    } else {
      throw std::invalid_argument(
        "config.json must contain 'architectures', 'architecture', or "
        "'model_type'.");
    }

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Load chat template from tokenizer_config.json or jinja (if available)
    std::optional<causallm::ChatTemplate> chat_template;
    if (causallm::ChatTemplate::Exists(model_path)) {
      chat_template.emplace(causallm::ChatTemplate::Load(model_path));
    }

    // Determine input text
    if (serve) {
      // HTTP requests supply their own prompts and complete chat histories.
    } else if (argc >= 3) {
      input_text = argv[2];
    } else {
      if (nntr_cfg.contains("chat_input")) {
        if (chat_template.has_value()) {
          input_text = chat_template->apply(nntr_cfg["chat_input"]);
          system_head_prompt.clear();
          system_tail_prompt.clear();
        } else {
          std::cerr << "[Warning] 'chat_input' is set but support for model "
                       "architecture '"
                    << architecture
                    << "' is not implemented. Falling back to 'sample_input'."
                    << std::endl;
          input_text = nntr_cfg["sample_input"].get<std::string>();
        }
      } else {
        input_text = nntr_cfg["sample_input"].get<std::string>();
      }
    }

    _ilap("chat template + prompt");
    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Unknown architecture: " << architecture << std::endl;
      std::cerr << "Registered architectures:";
      causallm::Factory::Instance().printRegistered(std::cerr);
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
    _ilap("Factory::create");
    model->initialize();
    _ilap("model->initialize");
    model->load_weight(weight_file);
    _ilap("model->load_weight");
    model->repack_weight();
    _ilap("model->repack_weight");

    bool do_sample = generation_cfg.value("do_sample", false);

#ifdef ENABLE_CAUSALLM_SERVER
    if (serve) {
      auto *causal = dynamic_cast<causallm::CausalLM *>(model.get());
      if (!causal)
        throw std::invalid_argument("Serving requires a CausalLM model");
      causallm::serving::CausalBackend backend(
        *causal, chat_template ? &*chat_template : nullptr);
      causallm::serving::Service service(backend, model_id, max_sessions,
                                         queue_capacity);
      causallm::serving::HttpServer server(service);
      if (server.bind(host, port) < 0)
        throw std::runtime_error("Cannot bind server socket");
      std::cerr << "nntrainer serving " << model_id << " at http://" << host
                << ':' << port << std::endl;
      return server.listen() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
#endif

#ifdef PROFILE
    start_peak_tracker();
#endif
#if defined(_WIN32)
    model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
               system_tail_prompt.c_str());
#else
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
#endif
#ifdef PROFILE
    stop_and_print_peak();
#endif
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms \n";
    printMemoryUsage();

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
