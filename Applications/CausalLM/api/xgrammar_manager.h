// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xgrammar_manager.h
 * @date   14 April 2026
 * @brief  XGrammar Manager for managing grammar-constrained generation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Quick.AI Team
 * @bug    No known bugs except for NYI items
 *
 * @note   This manager pre-compiles all tool grammars from Toolset.json
 *         and provides grammar instances for structured decoding.
 *         One model - one manager. Recompile when model changes.
 */

#ifndef __XGRAMMAR_MANAGER_H__
#define __XGRAMMAR_MANAGER_H__

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/grammar.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

// Forward declarations
namespace tokenizers {
class Tokenizer;
}

namespace causallm {

class XGrammar;

/**
 * @brief XGrammarManager class
 * @note  Singleton pattern. Manages pre-compiled grammars for tool-based
 *        structured generation. Must be initialized with tokenizer info
 *        from the current model.
 */
class XGrammarManager {
public:
  /**
   * @brief Get singleton instance
   */
  static XGrammarManager &Instance();

  /**
   * @brief Load toolset from JSON file and pre-compile all grammars
   * @param toolset_path Path to Toolset.json file
   * @param tokenizer Pointer to the tokenizer instance
   * @param vocab_size Size of the vocabulary
   * @return true on success, false on failure
   */
  bool loadToolset(const std::string &toolset_path,
                   tokenizers::Tokenizer *tokenizer, unsigned int vocab_size);

  /**
   * @brief Get pre-compiled XGrammar for a specific tool
   * @param tool_name Name of the tool (e.g., "alarm", "send_email", "memo")
   * @return Pointer to XGrammar, or nullptr if not found
   */
  XGrammar *getGrammar(const std::string &tool_name);

  /**
   * @brief Reset grammar matcher state for a specific tool
   * @param tool_name Name of the tool
   */
  void resetGrammar(const std::string &tool_name);

  /**
   * @brief Check if a tool exists in the loaded toolset
   * @param tool_name Name of the tool
   * @return true if tool exists, false otherwise
   */
  bool hasTool(const std::string &tool_name) const;

  /**
   * @brief Get list of all available tool names
   * @return Vector of tool names
   */
  std::vector<std::string> getToolNames() const;

  /**
   * @brief Initialize
   */
  bool initialize(tokenizers::Tokenizer *tokenizer, unsigned int vocab_size);

  /**
   * @brief Check if manager is initialized with a toolset
   * @return true if initialized, false otherwise
   */
  bool isInitialized() const { return initialized_; }

  /**
   * @brief Register a single tool with its JSON schema dynamically
   * @param tool_name Name of the tool
   * @param json_schema JSON schema string for the tool
   * @return true on success, false on failure
   * @note Requires loadToolset() to be called first to initialize
   * tokenizer/compiler
   */
  bool registerTool(const std::string &tool_name,
                    const std::string &json_schema);

  /**
   * @brief Clear all compiled grammars
   */
  void clear();

private:
  XGrammarManager() = default;
  ~XGrammarManager() = default;
  XGrammarManager(const XGrammarManager &) = delete;
  XGrammarManager &operator=(const XGrammarManager &) = delete;

  // Pre-compiled grammars: tool_name -> XGrammar
  std::unordered_map<std::string, std::unique_ptr<XGrammar>> compiled_grammars_;

  // Shared tokenizer info (created once per model)
  std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::unique_ptr<xgrammar::GrammarCompiler> grammar_compiler_;

  // Current toolset path (to detect if reload needed)
  std::string current_toolset_path_;

  // Initialization flag
  bool initialized_ = false;

  // Thread safety
  std::mutex mutex_;
};

} // namespace causallm

#endif // __XGRAMMAR_MANAGER_H__
