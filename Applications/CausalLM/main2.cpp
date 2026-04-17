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
 * @file	main2.cpp
 * @date	17 April 2026
 * @brief	Main file for running Qwen3 model using Qwen3CausalLM class
 * @see		https://github.com/nnstreamer/
 * @author	Auto-generated
 * @bug		No known bugs except for NYI items
 *
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <engine.h>

// Include the Qwen3CausalLM model header
// Note: The model implementation is in models/qwen3/qwen_qwen3_0_6b.cpp
// This header defines the Qwen3CausalLM class
#include "qwen_qwen3_0_6b.h"

#include <sys/resource.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <tokenizers_cpp.h>

/**
 * @brief Load bytes from a file
 */
std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

void printMemoryUsage() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
}

size_t read_vm_rss_kb() {
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
}

size_t read_private_rss_kb() {
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
}

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

void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

json LoadJsonFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path +
                             " | Reason: " + std::strerror(errno));
  }

  try {
    json data;
    file >> data;
    return data;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("JSON parse error in " + file_path +
                             " | Details: " + e.what());
  }
}

void printUsage(const char *program_name) {
  std::cerr << "Usage: " << program_name << " <model_path> [weight_file]\n"
            << "  <model_path>   : Path to model directory containing config files\n"
            << "  [weight_file]  : Optional path to weight file (default: <model_path>/nntr_qwen3_0_6b_fp32.bin)\n"
            << "\nExample:\n"
            << "  " << program_name << " /path/to/qwen3-0.6b\n"
            << "  " << program_name << " /path/to/qwen3-0.6b /path/to/weights.bin\n";
}

int main(int argc, char *argv[]) {
  auto start_time = std::chrono::high_resolution_clock::now();

  // Validate arguments
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  std::string weight_file;

  std::cout << "========================================" << std::endl;
  std::cout << "Qwen3-0.6B Model Runner" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Model path: " << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = LoadJsonFile(model_path + "/config.json");
    json nntr_cfg = LoadJsonFile(model_path + "/nntr_config.json");

    // Determine weight file path
    if (argc >= 3) {
      weight_file = argv[2];
    } else {
      weight_file = model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();
    }
    std::cout << "Weight file: " << weight_file << std::endl;

    // Update model configuration from config files
    // These override the defaults in the header
    if (cfg.contains("vocab_size")) {
      NUM_VOCAB = cfg["vocab_size"].get<int>();
    }
    if (cfg.contains("hidden_size")) {
      DIM = cfg["hidden_size"].get<int>();
    }
    if (cfg.contains("num_hidden_layers")) {
      NUM_LAYERS = cfg["num_hidden_layers"].get<int>();
    }
    if (cfg.contains("num_attention_heads")) {
      NUM_HEADS = cfg["num_attention_heads"].get<int>();
    }
    if (cfg.contains("head_dim")) {
      HEAD_DIM = cfg["head_dim"].get<int>();
    } else if (cfg.contains("hidden_size")) {
      HEAD_DIM = DIM / NUM_HEADS;
    }
    if (cfg.contains("intermediate_size")) {
      INTERMEDIATE_SIZE = cfg["intermediate_size"].get<int>();
    }
    if (cfg.contains("rms_norm_eps")) {
      NORM_EPS = cfg["rms_norm_eps"].get<float>();
    }
    if (cfg.contains("num_key_value_heads")) {
      NUM_KEY_VALUE_HEADS = cfg["num_key_value_heads"].get<int>();
      GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;
    }
    if (cfg.contains("rope_theta")) {
      ROPE_THETA = cfg["rope_theta"].get<unsigned int>();
    }
    if (cfg.contains("max_position_embeddings")) {
      MAX_POSITION_EMBEDDINGS = cfg["max_position_embeddings"].get<int>();
    }
    if (cfg.contains("tie_word_embeddings")) {
      TIE_WORD_EMBEDDINGS = cfg["tie_word_embeddings"].get<bool>();
    }

    // Update from nntr_config
    if (nntr_cfg.contains("init_seq_len")) {
      INIT_SEQ_LEN = nntr_cfg["init_seq_len"].get<int>();
    }
    if (nntr_cfg.contains("num_to_generate")) {
      NUM_TO_GENERATE = nntr_cfg["num_to_generate"].get<int>();
    }

    // Print model configuration
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Model Configuration:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Vocab size: " << NUM_VOCAB << std::endl;
    std::cout << "  Hidden dim: " << DIM << std::endl;
    std::cout << "  Num layers: " << NUM_LAYERS << std::endl;
    std::cout << "  Num heads: " << NUM_HEADS << std::endl;
    std::cout << "  Head dim: " << HEAD_DIM << std::endl;
    std::cout << "  Intermediate size: " << INTERMEDIATE_SIZE << std::endl;
    std::cout << "  GQA size: " << GQA_SIZE << std::endl;
    std::cout << "  Norm epsilon: " << NORM_EPS << std::endl;
    std::cout << "  RoPE theta: " << ROPE_THETA << std::endl;
    std::cout << "  Max position embeddings: " << MAX_POSITION_EMBEDDINGS << std::endl;
    std::cout << "  Tie word embeddings: " << (TIE_WORD_EMBEDDINGS ? "true" : "false") << std::endl;
    std::cout << "  Init sequence length: " << INIT_SEQ_LEN << std::endl;
    std::cout << "  Num to generate: " << NUM_TO_GENERATE << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Create and initialize the model
    std::cout << "\nInitializing Qwen3CausalLM model..." << std::endl;
    
    Qwen3CausalLM model;
    model.initialize();

    std::cout << "Model initialized successfully." << std::endl;

    // Load weights
    std::cout << "\nLoading weights from: " << weight_file << std::endl;
    model.load_weight(weight_file);
    std::cout << "Weights loaded successfully." << std::endl;

    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Initialization completed successfully!" << std::endl;
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms" << std::endl;
    printMemoryUsage();
    std::cout << "========================================" << std::endl;

    // Load and set tokenizer
    std::cout << "\nLoading tokenizer..." << std::endl;
    std::string tokenizer_path = model_path + "/tokenizer.json";
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(tokenizer_path));
    model.setTokenizer(std::move(tokenizer));
    std::cout << "Tokenizer loaded successfully." << std::endl;

    // Run the model with sample input
    std::string input_text = "Hello, how are you?";
    bool do_sample = true;
    std::string system_head_prompt = "";
    std::string system_tail_prompt = "";
    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }
    if (argc >= 3) {
      input_text = argv[2];
    } 
    else {
      input_text = nntr_cfg["sample_input"].get<std::string>();
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running inference..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    model.run(input_text, do_sample, system_head_prompt, system_tail_prompt);

    std::cout << "\nrun completed successfully!" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}