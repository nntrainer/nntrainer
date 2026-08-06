// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_wrapper.cpp
 * @date   14 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file implements XGrammar class for grammar-guided generation
 */

#include "xgrammar_wrapper.h"
#include <iostream>
#include <variant>

#include <dlpack/dlpack.h>

namespace causallm {

XGrammar::XGrammar() {
  // Store configurations if needed
  grammar_enabled = false;
}

void XGrammar::initializeGrammar(const std::string &grammar_type,
                                 const std::string &json_schema,
                                 tokenizers::Tokenizer *tokenizer,
                                 unsigned int vocab_size) {
  if (tokenizer == nullptr) {
    throw std::runtime_error("Tokenizer is null. Cannot initialize grammar.");
  }
  // std::cout << "[xgrammar] Initializing grammar constraints...\n";

  // Step 1: Extract vocabulary from tokenizer
  // std::cout << "[xgrammar] Extracting vocabulary from tokenizer...\n";
  std::vector<std::string> encoded_vocab;
  encoded_vocab.reserve(vocab_size);

  for (size_t i = 0; i < vocab_size; ++i) {
    std::string token = tokenizer->IdToToken(static_cast<int32_t>(i));
    encoded_vocab.push_back(token);
  }
  // std::cout << "[xgrammar] Vocabulary size: " << vocab_size << "\n";

  // Step 2: Create TokenizerInfo from encoded vocabulary
  // std::cout << "[xgrammar] Creating TokenizerInfo...\n";
  tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(
    encoded_vocab, xgrammar::VocabType::BYTE_LEVEL, encoded_vocab.size());

  // Step 3: Create GrammarCompiler
  // std::cout << "[xgrammar] Creating GrammarCompiler...\n";
  grammar_compiler_ =
    std::make_unique<xgrammar::GrammarCompiler>(*tokenizer_info_);

  // Step 4: Compile grammar using the shared helper
  compileGrammar(grammar_type, json_schema, grammar_compiler_.get(),
                 vocab_size);
}

void XGrammar::initializeGrammar(const std::string &grammar_type,
                                 const std::string &json_schema,
                                 xgrammar::GrammarCompiler *grammar_compiler,
                                 unsigned int vocab_size) {
  if (grammar_compiler == nullptr) {
    throw std::runtime_error(
      "GrammarCompiler is null. Cannot initialize grammar.");
  }
  // Skip encoded vocab extraction - reuse shared GrammarCompiler
  // Compile grammar using the shared helper
  compileGrammar(grammar_type, json_schema, grammar_compiler, vocab_size);
}

void XGrammar::compileGrammar(const std::string &grammar_type,
                              const std::string &json_schema,
                              xgrammar::GrammarCompiler *grammar_compiler,
                              unsigned int vocab_size) {
  if (grammar_type == "json") {
    if (!json_schema.empty()) {
      // Compile from JSON schema
      compiled_grammar_ = std::make_unique<xgrammar::CompiledGrammar>(
        grammar_compiler->CompileJSONSchema(json_schema));
    } else {
      // Compile built-in JSON grammar
      compiled_grammar_ = std::make_unique<xgrammar::CompiledGrammar>(
        grammar_compiler->CompileBuiltinJSONGrammar());
      std::cout << "[xgrammar] Built-in JSON grammar compiled\n";
    }
  } else if (grammar_type == "ebnf") {
    // Example EBNF grammar
    std::string ebnf_grammar = R"(
      root ::= greeting name "!" " " message "!"
      greeting ::= "Hello" | "Hi"
      name ::= "I'm" | "My name is"
      message ::= "an AI assistant" | "a helpful AI" | "here to help you"
    )";
    compiled_grammar_ = std::make_unique<xgrammar::CompiledGrammar>(
      grammar_compiler->CompileGrammar(ebnf_grammar, "root"));
    std::cout << "[xgrammar] EBNF grammar compiled\n";
  } else if (grammar_type == "regex") {
    // Example regex pattern
    std::string regex = R"(\d{3}-\d{3}-\d{4})";
    compiled_grammar_ = std::make_unique<xgrammar::CompiledGrammar>(
      grammar_compiler->CompileRegex(regex));
    std::cout << "[xgrammar] Regex pattern compiled\n";
  } else {
    throw std::invalid_argument("Unsupported grammar type: " + grammar_type);
  }
  // std::cout << "[xgrammar] Grammar memory usage: "
  //           << compiled_grammar_->MemorySizeBytes() << " bytes\n";

  // Create GrammarMatcher
  // std::cout << "[xgrammar] Creating GrammarMatcher...\n";
  grammar_matcher_ =
    std::make_unique<xgrammar::GrammarMatcher>(*compiled_grammar_);

  // std::cout << "[xgrammar] Grammar constraints initialized
  // successfully!\n\n"; Allocate bitmask storage (optimized - allocated once)
  bitmask_size_ = (vocab_size + 31) / 32;
  bitmask_data_ = std::vector<int32_t>(bitmask_size_);
  bitmask_tensor_.data = bitmask_data_.data();
  bitmask_tensor_.device = {kDLCPU, 0};
  bitmask_tensor_.ndim = 1;
  bitmask_tensor_.shape = &bitmask_size_;
  bitmask_tensor_.dtype = {kDLInt, 32, 1};
  bitmask_tensor_.strides = nullptr;
  bitmask_tensor_.byte_offset = 0;
  grammar_matcher_->FillNextTokenBitmask(&bitmask_tensor_);

  grammar_enabled = true;
}

void XGrammar::resetGrammar() {
  if (grammar_matcher_ != nullptr) {
    grammar_matcher_->Reset();
    // Update bitmask for initial token constraint after reset
    grammar_matcher_->FillNextTokenBitmask(&bitmask_tensor_);
  }
}

bool XGrammar::loadFromCache(const std::string &serialized_json,
                             xgrammar::TokenizerInfo *tokenizer_info,
                             unsigned int vocab_size) {
  if (tokenizer_info == nullptr) {
    std::cerr << "[xgrammar] Error: TokenizerInfo is null for cache loading\n";
    return false;
  }

  auto result = xgrammar::CompiledGrammar::DeserializeJSON(serialized_json,
                                                           *tokenizer_info);
  if (!std::holds_alternative<xgrammar::CompiledGrammar>(result)) {
    std::cerr << "[xgrammar] Error: Failed to deserialize compiled grammar "
                 "from cache\n";
    return false;
  }

  compiled_grammar_ = std::make_unique<xgrammar::CompiledGrammar>(
    std::get<xgrammar::CompiledGrammar>(std::move(result)));

  // Create GrammarMatcher
  grammar_matcher_ =
    std::make_unique<xgrammar::GrammarMatcher>(*compiled_grammar_);

  // Allocate bitmask storage
  bitmask_size_ = (vocab_size + 31) / 32;
  bitmask_data_ = std::vector<int32_t>(bitmask_size_);
  bitmask_tensor_.data = bitmask_data_.data();
  bitmask_tensor_.device = {kDLCPU, 0};
  bitmask_tensor_.ndim = 1;
  bitmask_tensor_.shape = &bitmask_size_;
  bitmask_tensor_.dtype = {kDLInt, 32, 1};
  bitmask_tensor_.strides = nullptr;
  bitmask_tensor_.byte_offset = 0;
  grammar_matcher_->FillNextTokenBitmask(&bitmask_tensor_);

  grammar_enabled = true;
  return true;
}

std::string XGrammar::serialize() const {
  if (compiled_grammar_ == nullptr) {
    return "";
  }
  return compiled_grammar_->SerializeJSON();
}

std::vector<int32_t> &XGrammar::getBitmaskData() { return bitmask_data_; }

DLTensor &XGrammar::getBitmaskTensor() { return bitmask_tensor_; }

int64_t XGrammar::getBitmaskSize() { return bitmask_size_; }

xgrammar::GrammarMatcher *XGrammar::getGrammarMatcher() {
  return grammar_matcher_.get();
}

void XGrammar::applyGrammarMask(float *logits, int vocab_size) {
  for (int i = 0; i < vocab_size; ++i) {
    int32_t block = bitmask_data_[i / 32];
    bool is_accepted = (block >> (i % 32)) & 1;

    if (!is_accepted) {
      // logits[i] = std::numeric_limits<float>::infinity();
      logits[i] = -INFINITY;
    }
  }
}

void XGrammar::applyGrammarMask(uint16_t *logits, int vocab_size,
                                float logit_scale, int logit_offset) {
  // Apply grammar mask to quantized uint16_t logits
  // Set masked tokens to minimum value (will be rejected by sampling)
  for (int i = 0; i < vocab_size; ++i) {
    int32_t block = bitmask_data_[i / 32];
    bool is_accepted = (block >> (i % 32)) & 1;

    if (!is_accepted) {
      // Set to minimum uint16_t value (0) which will result in -inf after
      // scaling
      logits[i] = 0;
    }
  }
}

} // namespace causallm
