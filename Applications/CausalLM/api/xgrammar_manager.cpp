// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_manager.cpp
 * @date   14 April 2026
 * @brief  Implementation of XGrammarManager for grammar-guided generation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "xgrammar_manager.h"
#include "xgrammar_wrapper.h"

#include <fstream>
#include <iostream>
#include <tokenizers_cpp.h>

#include "json.hpp"

using json = nlohmann::json;

namespace causallm {

XGrammarManager &XGrammarManager::Instance() {
  static XGrammarManager instance;
  return instance;
}

bool XGrammarManager::initialize(tokenizers::Tokenizer *tokenizer,
                                 unsigned int vocab_size) {
  if (tokenizer == nullptr) {
    std::cerr << "[XGrammarManager] Error: Tokenizer is null" << std::endl;
    return false;
  }
  // Clear existing grammars
  compiled_grammars_.clear();
  tokenizer_info_.reset();
  grammar_compiler_.reset();

  // Step 1: Extract vocabulary from tokenizer
  std::cout << "[XGrammarManager] Extracting vocabulary from tokenizer..."
            << std::endl;
  std::vector<std::string> encoded_vocab;
  encoded_vocab.reserve(vocab_size);

  for (size_t i = 0; i < vocab_size; ++i) {
    std::string token = tokenizer->IdToToken(static_cast<int32_t>(i));
    encoded_vocab.push_back(token);
  }
  std::cout << "[XGrammarManager] Vocabulary size: " << vocab_size << std::endl;

  // Step 2: Create TokenizerInfo (shared across all grammars)
  std::cout << "[XGrammarManager] Creating TokenizerInfo..." << std::endl;
  tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(
    encoded_vocab, xgrammar::VocabType::BYTE_LEVEL, encoded_vocab.size());

  // Step 3: Create GrammarCompiler
  std::cout << "[XGrammarManager] Creating GrammarCompiler..." << std::endl;
  grammar_compiler_ =
    std::make_unique<xgrammar::GrammarCompiler>(*tokenizer_info_);

  initialized_ = true;
  return true;
}

bool XGrammarManager::loadToolset(const std::string &toolset_path,
                                  tokenizers::Tokenizer *tokenizer,
                                  unsigned int vocab_size) {
  // auto start_time = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  if (tokenizer == nullptr) {
    std::cerr << "[XGrammarManager] Error: Tokenizer is null" << std::endl;
    return false;
  }

  std::cout << "[XGrammarManager] Loading toolset from: " << toolset_path
            << std::endl;

  // Determine cache file path (same directory as toolset, with .cache
  // extension)
  std::string cache_path = toolset_path + ".cache";

  // Step 1: Try to load from cache first
  std::ifstream cache_file(cache_path);
  if (cache_file.is_open()) {
    std::cout << "[XGrammarManager] Found cache file: " << cache_path
              << std::endl;

    json cache_data;
    try {
      cache_file >> cache_data;
      cache_file.close();

      // Load each grammar from cache
      bool all_loaded = true;
      for (auto it = cache_data.begin(); it != cache_data.end(); ++it) {
        const std::string &tool_name = it.key();
        const std::string &serialized_grammar = it.value();

        auto grammar = std::make_unique<XGrammar>();
        if (grammar->loadFromCache(serialized_grammar, tokenizer_info_.get(),
                                   vocab_size)) {
          compiled_grammars_[tool_name] = std::move(grammar);
        } else {
          std::cerr
            << "[XGrammarManager] Warning: Failed to load cached grammar for: "
            << tool_name << std::endl;
          all_loaded = false;
        }
      }

      if (all_loaded && !compiled_grammars_.empty()) {
        current_toolset_path_ = toolset_path;
        initialized_ = true;
        std::cout << "[XGrammarManager] Successfully loaded "
                  << compiled_grammars_.size() << " tool grammars from cache"
                  << std::endl;
        // auto finish_time = std::chrono::high_resolution_clock::now();
        // auto e2e_duration =
        // std::chrono::duration_cast<std::chrono::milliseconds>(
        //     finish_time - start_time);
        // std::cout << "[Toolset Load time]: " << e2e_duration.count() << " ms
        // \n";
        return true;
      } else {
        std::cout << "[XGrammarManager] Cache incomplete, falling back to "
                     "compilation..."
                  << std::endl;
        compiled_grammars_.clear();
      }
    } catch (const json::parse_error &e) {
      std::cerr << "[XGrammarManager] Warning: Failed to parse cache file: "
                << e.what() << std::endl;
      cache_file.close();
      compiled_grammars_.clear();
    }
  }

  // Step 2: Load from Toolset.json and compile
  std::ifstream file(toolset_path);
  if (!file.is_open()) {
    std::cerr << "[XGrammarManager] Error: Failed to open toolset file: "
              << toolset_path << std::endl;
    return false;
  }

  json toolset;
  try {
    file >> toolset;
  } catch (const json::parse_error &e) {
    std::cerr << "[XGrammarManager] Error: Failed to parse toolset JSON: "
              << e.what() << std::endl;
    return false;
  }

  std::cout << "[XGrammarManager] Pre-compiling all tool grammars..."
            << std::endl;

  // Cache data to save
  json cache_data;

  for (auto it = toolset.begin(); it != toolset.end(); ++it) {
    const std::string &tool_name = it.key();
    const json &tool_schema = it.value();

    // Convert JSON schema to string
    std::string json_schema_str = tool_schema.dump();

    std::cout << "[XGrammarManager] Compiling grammar for tool: " << tool_name
              << std::endl;

    // Create XGrammar instance
    auto grammar = std::make_unique<XGrammar>();

    // Initialize with JSON schema using shared GrammarCompiler and
    // TokenizerInfo (optimized)
    grammar->initializeGrammar("json", json_schema_str, grammar_compiler_.get(),
                               vocab_size);

    // Save to cache data
    cache_data[tool_name] = grammar->serialize();

    // Store in map
    compiled_grammars_[tool_name] = std::move(grammar);
  }

  // Step 6: Save cache file
  std::ofstream out_cache(cache_path);
  if (out_cache.is_open()) {
    out_cache << cache_data.dump();
    out_cache.close();
    std::cout << "[XGrammarManager] Saved grammar cache to: " << cache_path
              << std::endl;
  } else {
    std::cerr << "[XGrammarManager] Warning: Failed to save cache file to: "
              << cache_path << std::endl;
  }

  current_toolset_path_ = toolset_path;
  initialized_ = true;

  std::cout << "[XGrammarManager] Successfully compiled "
            << compiled_grammars_.size() << " tool grammars" << std::endl;

  // auto finish_time = std::chrono::high_resolution_clock::now();
  // auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
  //     finish_time - start_time);
  // std::cout << "[Toolset Load time]: " << e2e_duration.count() << " ms \n";

  return true;
}

XGrammar *XGrammarManager::getGrammar(const std::string &tool_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = compiled_grammars_.find(tool_name);
  if (it != compiled_grammars_.end()) {
    return it->second.get();
  }

  std::cerr << "[XGrammarManager] Warning: Tool not found: " << tool_name
            << std::endl;
  return nullptr;
}

void XGrammarManager::resetGrammar(const std::string &tool_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = compiled_grammars_.find(tool_name);
  if (it != compiled_grammars_.end()) {
    it->second->resetGrammar();
  }
}

bool XGrammarManager::registerTool(const std::string &tool_name,
                                   const std::string &json_schema) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if tokenizer_info_ and grammar_compiler_ are initialized
  if (!tokenizer_info_ || !grammar_compiler_) {
    std::cerr << "[XGrammarManager] Error: Manager not initialized. Call "
                 "loadToolset() first."
              << std::endl;
    return false;
  }

  // Get vocab_size from tokenizer_info_
  unsigned int vocab_size = tokenizer_info_->GetVocabSize();

  std::cout << "[XGrammarManager] Registering tool: " << tool_name << std::endl;

  // Create XGrammar instance
  auto grammar = std::make_unique<XGrammar>();

  // Initialize with JSON schema using shared GrammarCompiler and TokenizerInfo
  grammar->initializeGrammar("json", json_schema, grammar_compiler_.get(),
                             vocab_size);

  // Store in map (will replace if already exists)
  compiled_grammars_[tool_name] = std::move(grammar);

  std::cout << "[XGrammarManager] Successfully registered tool: " << tool_name
            << std::endl;

  return true;
}

bool XGrammarManager::hasTool(const std::string &tool_name) const {
  return compiled_grammars_.find(tool_name) != compiled_grammars_.end();
}

std::vector<std::string> XGrammarManager::getToolNames() const {
  std::vector<std::string> names;
  names.reserve(compiled_grammars_.size());
  for (const auto &pair : compiled_grammars_) {
    names.push_back(pair.first);
  }
  return names;
}

void XGrammarManager::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  compiled_grammars_.clear();
  tokenizer_info_.reset();
  grammar_compiler_.reset();
  current_toolset_path_.clear();
  initialized_ = false;

  std::cout << "[XGrammarManager] Cleared all compiled grammars" << std::endl;
}

} // namespace causallm
