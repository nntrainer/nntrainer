// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_wrapper.h
 * @date   14 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @brief  The header for the support of grammar-guided generation.
 * @bug    No known bugs except for NYI items
 */

#ifndef XGRAMMAR_XGRAMMAR_H_
#define XGRAMMAR_XGRAMMAR_H_

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <memory>
#include <vector>

#include <transformer.h>

#include <dlpack/dlpack.h>
#include <xgrammar/compiler.h>
#include <xgrammar/config.h>
#include <xgrammar/exception.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

// Forward declarations
namespace tokenizers {
class Tokenizer;
}

namespace causallm {
/**
 * @brief Grammar-guided generation helper wrapping xgrammar's compiler and
 * matcher.
 */
WIN_EXPORT class XGrammar {

public:
  XGrammar();

  /**
   * @brief Destroy the CausalLM object
   */
  virtual ~XGrammar() {}

  /**
   * @brief Initialize xgrammar for grammar-guided generation
   * @param grammar_type Type of grammar ("json", "ebnf", "regex")
   * @param json_schema Optional JSON schema for JSON grammar type
   * @param tokenizer Tokenizer instance to extract vocabulary
   * @param vocab_size Size of the vocabulary
   */
  void initializeGrammar(const std::string &grammar_type = "json",
                         const std::string &json_schema = "",
                         tokenizers::Tokenizer *tokenizer = nullptr,
                         unsigned int vocab_size = 0);

  /**
   * @brief Initialize xgrammar with pre-created TokenizerInfo and
   * GrammarCompiler This is optimized for cases where multiple grammars share
   * the same tokenizer.
   * @param grammar_type Type of grammar ("json", "ebnf", "regex")
   * @param json_schema Optional JSON schema for JSON grammar type
   * @param grammar_compiler Pre-created GrammarCompiler (shared, not owned)
   * @param vocab_size Size of the vocabulary (for bitmask allocation)
   */
  void initializeGrammar(const std::string &grammar_type,
                         const std::string &json_schema,
                         xgrammar::GrammarCompiler *grammar_compiler,
                         unsigned int vocab_size);

  /**
   * @brief Reset grammar matcher state
   */
  void resetGrammar();

  /**
   * @brief Check if grammar constraints are enabled
   */
  bool isGrammarEnabled() const { return grammar_enabled; }

  /**
   * @brief Load grammar from serialized JSON cache
   * @param serialized_json The serialized compiled grammar JSON string
   * @param tokenizer_info TokenizerInfo needed for deserialization
   * @param vocab_size Size of the vocabulary (for bitmask allocation)
   * @return true on success, false on failure
   */
  bool loadFromCache(const std::string &serialized_json,
                     xgrammar::TokenizerInfo *tokenizer_info,
                     unsigned int vocab_size);

  /**
   * @brief Get serialized JSON of the compiled grammar
   * @return JSON string representation of the compiled grammar
   */
  std::string serialize() const;

  /**
   * @brief Get bitmask data
   */
  std::vector<int32_t> &getBitmaskData();

  /**
   * @brief Get bitmask tensor
   */
  DLTensor &getBitmaskTensor();

  /**
   * @brief Get bitmask size
   */
  int64_t getBitmaskSize();

  /**
   * @brief Get grammar matcher pointer
   */
  xgrammar::GrammarMatcher *getGrammarMatcher();

  void applyGrammarMask(float *logits, int vocab_size);
  void applyGrammarMask(uint16_t *logits, int vocab_size, float logit_scale,
                        int logit_offset);

private:
  /**
   * @brief Internal helper to compile grammar (shared between overloaded
   * methods)
   */
  void compileGrammar(const std::string &grammar_type,
                      const std::string &json_schema,
                      xgrammar::GrammarCompiler *grammar_compiler,
                      unsigned int vocab_size);

protected:
  // xgrammar components for grammar-guided generation
  std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::unique_ptr<xgrammar::GrammarCompiler> grammar_compiler_;
  std::unique_ptr<xgrammar::CompiledGrammar> compiled_grammar_;
  std::unique_ptr<xgrammar::GrammarMatcher> grammar_matcher_;

  // Optimized bitmask storage (reused across generations)
  DLTensor bitmask_tensor_;
  std::vector<int32_t> bitmask_data_;
  int64_t bitmask_size_;

  bool grammar_enabled = false;
};

} // namespace causallm

#endif // XGRAMMAR_XGRAMMAR_H_
