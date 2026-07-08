// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   tokenizer_loader.cpp
 * @date   07 Apr 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Tokenizer loading helpers for Quick.AI models
 */

#include <tokenizer_loader.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace causallm {
namespace {

using json = nlohmann::json;

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

std::string ToLowerString(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

bool HasSuffix(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsWordPieceType(const std::string &tokenizer_type) {
  const std::string lower = ToLowerString(tokenizer_type);
  return lower == "wordpiece" || lower == "bert" || lower == "tinybert";
}

bool IsBPEType(const std::string &tokenizer_type) {
  const std::string lower = ToLowerString(tokenizer_type);
  return lower == "bpe" || lower == "bytelevelbpe" || lower == "byte_level_bpe";
}

bool TryLoadBytesFromFile(const std::string &path, std::string &buffer) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  buffer.assign(size, '\0');
  return file.read(buffer.data(), size).good() || file.eof();
}

bool TryWriteBytesToFile(const std::string &path, const std::string &buffer) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    return false;
  }

  file.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return file.good();
}

bool IsCacheFresh(const std::string &cache_file,
                  const std::string &source_file) {
  struct stat cache_stat {};
  if (stat(cache_file.c_str(), &cache_stat) != 0) {
    return false;
  }

  struct stat source_stat {};
  if (stat(source_file.c_str(), &source_stat) != 0) {
    return true;
  }

  return cache_stat.st_mtime >= source_stat.st_mtime;
}

bool TryFindBoolByKey(const json &node, const std::string &key, bool &value) {
  if (node.is_object()) {
    auto it = node.find(key);
    if (it != node.end() && it->is_boolean()) {
      value = it->get<bool>();
      return true;
    }

    for (auto item = node.begin(); item != node.end(); ++item) {
      if (TryFindBoolByKey(item.value(), key, value)) {
        return true;
      }
    }
  } else if (node.is_array()) {
    for (const auto &item : node) {
      if (TryFindBoolByKey(item, key, value)) {
        return true;
      }
    }
  }

  return false;
}

/** @brief configuration for the WordPiece tokenizer */
struct WordPieceConfig {
  bool do_lower_case = true;
  std::string unk_token = "[UNK]";
  std::string continuing_subword_prefix = "##";
  uint32_t max_input_chars_per_word = 100;
  std::string cls_token = "[CLS]";
  std::string sep_token = "[SEP]";
};

bool IsWordPieceTokenizerJson(const json &tokenizer_json) {
  if (!tokenizer_json.contains("model") ||
      !tokenizer_json["model"].is_object()) {
    return false;
  }

  const json &model = tokenizer_json["model"];
  return model.contains("type") && model["type"].is_string() &&
         ToLowerString(model["type"].get<std::string>()) == "wordpiece";
}

bool IsBPETokenizerJson(const json &tokenizer_json) {
  if (!tokenizer_json.contains("model") ||
      !tokenizer_json["model"].is_object()) {
    return false;
  }

  const json &model = tokenizer_json["model"];
  return model.contains("type") && model["type"].is_string() &&
         ToLowerString(model["type"].get<std::string>()) == "bpe";
}

std::string BuildWordPieceVocabBlob(const json &tokenizer_json,
                                    WordPieceConfig &config) {
  const json &model = tokenizer_json["model"];
  if (!model.contains("vocab") || !model["vocab"].is_object()) {
    throw std::runtime_error("WordPiece tokenizer.json has no model.vocab");
  }

  if (model.contains("unk_token") && model["unk_token"].is_string()) {
    config.unk_token = model["unk_token"].get<std::string>();
  }
  if (model.contains("continuing_subword_prefix") &&
      model["continuing_subword_prefix"].is_string()) {
    config.continuing_subword_prefix =
      model["continuing_subword_prefix"].get<std::string>();
  }
  if (model.contains("max_input_chars_per_word") &&
      (model["max_input_chars_per_word"].is_number_integer() ||
       model["max_input_chars_per_word"].is_number_unsigned())) {
    config.max_input_chars_per_word =
      model["max_input_chars_per_word"].get<uint32_t>();
  }
  if (tokenizer_json.contains("normalizer")) {
    bool do_lower_case = config.do_lower_case;
    if (TryFindBoolByKey(tokenizer_json["normalizer"], "lowercase",
                         do_lower_case)) {
      config.do_lower_case = do_lower_case;
    }
  }

  size_t max_id = 0;
  for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
    if (!it.value().is_number_integer() && !it.value().is_number_unsigned()) {
      throw std::runtime_error("WordPiece vocab id must be an integer");
    }

    const int64_t id = it.value().get<int64_t>();
    if (id < 0) {
      throw std::runtime_error("WordPiece vocab id must be non-negative");
    }
    max_id = std::max(max_id, static_cast<size_t>(id));
  }

  std::vector<std::string> tokens(max_id + 1);
  for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
    tokens[static_cast<size_t>(it.value().get<int64_t>())] = it.key();
  }

  std::string vocab_blob;
  for (size_t id = 0; id < tokens.size(); ++id) {
    if (tokens[id].empty()) {
      std::ostringstream ss;
      ss << "WordPiece vocab is missing id " << id;
      throw std::runtime_error(ss.str());
    }
    vocab_blob += tokens[id];
    vocab_blob.push_back('\n');
  }

  return vocab_blob;
}

std::unique_ptr<tokenizers::Tokenizer>
TryLoadWordPieceTokenizerCache(const std::string &cache_file,
                               const std::string &source_file) {
  if (!IsCacheFresh(cache_file, source_file)) {
    return nullptr;
  }

  std::string cache_blob;
  if (TryLoadBytesFromFile(cache_file, cache_blob)) {
    try {
      return tokenizers::Tokenizer::FromBlobWordPieceCache(cache_blob);
    } catch (const std::exception &e) {
      std::cerr << "Ignoring invalid WordPiece tokenizer cache: " << e.what()
                << std::endl;
    }
  }

  return nullptr;
}

std::unique_ptr<tokenizers::Tokenizer> LoadWordPieceTokenizer(
  const std::string &vocab_blob, const std::string &cache_file,
  const std::string &source_file, const WordPieceConfig &config) {
  auto cached_tokenizer =
    TryLoadWordPieceTokenizerCache(cache_file, source_file);
  if (cached_tokenizer) {
    return cached_tokenizer;
  }

  auto tokenizer = tokenizers::Tokenizer::FromBlobWordPiece(
    vocab_blob, config.do_lower_case, config.unk_token,
    config.continuing_subword_prefix, config.max_input_chars_per_word,
    config.cls_token, config.sep_token);

  const std::string serialized = tokenizer->SerializeToCache();
  if (!serialized.empty() && !TryWriteBytesToFile(cache_file, serialized)) {
    std::cerr << "Failed to write WordPiece tokenizer cache: " << cache_file
              << std::endl;
  }

  return tokenizer;
}

std::unique_ptr<tokenizers::Tokenizer>
TryLoadBPETokenizerCache(const std::string &cache_file,
                         const std::string &source_file) {
  if (!IsCacheFresh(cache_file, source_file)) {
    return nullptr;
  }

  std::string cache_blob;
  if (TryLoadBytesFromFile(cache_file, cache_blob)) {
    try {
      return tokenizers::Tokenizer::FromBlobBPECache(cache_blob);
    } catch (const std::exception &e) {
      std::cerr << "Ignoring invalid BPE tokenizer cache: " << e.what()
                << std::endl;
    }
  }

  return nullptr;
}

std::vector<std::string>
BuildBPEConformancePrompts(const json &tokenizer_json) {
  std::vector<std::string> prompts = {
    "",
    "Hello world!",
    "  leading and trailing  ",
    "def foo(x):\n    return x + 1",
    "accent cafe\xCC\x81 nai\xCC\x88ve resume\xCC\x81",
    "vietnamese tie\xCC\x82\xCC\x81ng Vie\xCC\xA3t",
    "jamo \xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB"
    "\xE1\x84\x80\xE1\x85\xB3\xE1\x86\xAF",
    "kana \xE3\x81\x8B\xE3\x82\x99 \xE3\x82\xAB\xE3\x82\x99",
    "\xEC\x95\x88\xEB\x85\x95\xED\x95\x98\xEC\x84\xB8\xEC\x9A\x94 "
    "\xEC\x84\xB8\xEA\xB3\x84",
    "emoji \xF0\x9F\x98\x80 test",
    "tabs\tand\nnewlines\n",
  };

  if (tokenizer_json.contains("added_tokens") &&
      tokenizer_json["added_tokens"].is_array()) {
    size_t added = 0;
    for (const auto &token : tokenizer_json["added_tokens"]) {
      if (!token.is_object() || !token.contains("content") ||
          !token["content"].is_string()) {
        continue;
      }

      const std::string content = token["content"].get<std::string>();
      prompts.push_back(content);
      prompts.push_back(content + " user\nhello " + content);
      if (++added >= 8) {
        break;
      }
    }
  }

  return prompts;
}

bool HasSameEncoding(tokenizers::Tokenizer &lhs, tokenizers::Tokenizer &rhs,
                     const std::vector<std::string> &prompts) {
  for (const std::string &prompt : prompts) {
    for (bool add_special_tokens : {false, true}) {
      if (lhs.Encode(prompt, add_special_tokens) !=
          rhs.Encode(prompt, add_special_tokens)) {
        return false;
      }
    }
  }

  return true;
}

std::unique_ptr<tokenizers::Tokenizer> LoadExactBPETokenizer(
  const std::string &tokenizer_blob, const std::string &cache_file,
  const std::string &source_file, const json &tokenizer_json) {
  auto cached_tokenizer = TryLoadBPETokenizerCache(cache_file, source_file);
  if (cached_tokenizer) {
    return cached_tokenizer;
  }

  auto original_tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);

  try {
    auto candidate_tokenizer =
      tokenizers::Tokenizer::FromBlobBPEJSON(tokenizer_json.dump());
    if (HasSameEncoding(*original_tokenizer, *candidate_tokenizer,
                        BuildBPEConformancePrompts(tokenizer_json))) {
      const std::string cache_blob = candidate_tokenizer->SerializeToCache();
      if (!TryWriteBytesToFile(cache_file, cache_blob)) {
        std::cerr << "Failed to write BPE tokenizer cache: " << cache_file
                  << std::endl;
      }
      return candidate_tokenizer;
    }

    std::cerr << "Ignoring BPE tokenizer cache candidate: conformance check "
                 "failed"
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Ignoring BPE tokenizer cache candidate: " << e.what()
              << std::endl;
  }

  return original_tokenizer;
}

} // namespace

std::unique_ptr<tokenizers::Tokenizer> LoadTokenizer(nlohmann::json &nntr_cfg) {
  if (!nntr_cfg.contains("tokenizer_file") ||
      nntr_cfg["tokenizer_file"].is_null()) {
    return nullptr;
  }

  const std::string tokenizer_file =
    nntr_cfg["tokenizer_file"].get<std::string>();
  const std::string tokenizer_type =
    nntr_cfg.contains("tokenizer_type")
      ? nntr_cfg["tokenizer_type"].get<std::string>()
      : "";
  const bool has_custom_cache_file = nntr_cfg.contains("tokenizer_cache_file");
  const std::string wordpiece_cache_file =
    has_custom_cache_file ? nntr_cfg["tokenizer_cache_file"].get<std::string>()
                          : tokenizer_file + ".qaiwp";
  const std::string bpe_cache_file =
    has_custom_cache_file ? nntr_cfg["tokenizer_cache_file"].get<std::string>()
                          : tokenizer_file + ".qaibpe";

  const std::string lower_path = ToLowerString(tokenizer_file);
  const bool looks_like_vocab_txt =
    HasSuffix(lower_path, ".txt") || HasSuffix(lower_path, "vocab");
  const bool requested_wordpiece = IsWordPieceType(tokenizer_type);
  const bool requested_bpe = IsBPEType(tokenizer_type);

  WordPieceConfig config;
  if (nntr_cfg.contains("tokenizer_do_lower_case")) {
    config.do_lower_case = nntr_cfg["tokenizer_do_lower_case"].get<bool>();
  }
  if (nntr_cfg.contains("tokenizer_unk_token")) {
    config.unk_token = nntr_cfg["tokenizer_unk_token"].get<std::string>();
  }
  if (nntr_cfg.contains("tokenizer_continuing_subword_prefix")) {
    config.continuing_subword_prefix =
      nntr_cfg["tokenizer_continuing_subword_prefix"].get<std::string>();
  }
  if (nntr_cfg.contains("tokenizer_max_input_chars_per_word")) {
    config.max_input_chars_per_word =
      nntr_cfg["tokenizer_max_input_chars_per_word"].get<uint32_t>();
  }
  if (nntr_cfg.contains("tokenizer_cls_token")) {
    config.cls_token = nntr_cfg["tokenizer_cls_token"].get<std::string>();
  }
  if (nntr_cfg.contains("tokenizer_sep_token")) {
    config.sep_token = nntr_cfg["tokenizer_sep_token"].get<std::string>();
  }

  const bool looks_like_tokenizer_json = HasSuffix(lower_path, ".json");

  if (requested_wordpiece || looks_like_vocab_txt ||
      (!has_custom_cache_file && looks_like_tokenizer_json)) {
    auto cached_tokenizer =
      TryLoadWordPieceTokenizerCache(wordpiece_cache_file, tokenizer_file);
    if (cached_tokenizer) {
      return cached_tokenizer;
    }
  }

  if (requested_bpe || (!has_custom_cache_file && looks_like_tokenizer_json)) {
    auto cached_tokenizer =
      TryLoadBPETokenizerCache(bpe_cache_file, tokenizer_file);
    if (cached_tokenizer) {
      return cached_tokenizer;
    }
  }

  const std::string tokenizer_blob = LoadBytesFromFile(tokenizer_file);
  const bool may_be_wordpiece_json =
    looks_like_tokenizer_json &&
    tokenizer_blob.find("WordPiece") != std::string::npos;
  const bool may_be_bpe_json =
    looks_like_tokenizer_json &&
    tokenizer_blob.find("\"BPE\"") != std::string::npos;

  std::unique_ptr<json> tokenizer_json;
  auto get_tokenizer_json = [&]() -> json & {
    if (!tokenizer_json) {
      tokenizer_json = std::make_unique<json>(json::parse(tokenizer_blob));
    }
    return *tokenizer_json;
  };

  if (looks_like_tokenizer_json &&
      (requested_wordpiece || may_be_wordpiece_json)) {
    json &json_blob = get_tokenizer_json();
    if (IsWordPieceTokenizerJson(json_blob)) {
      std::string vocab_blob = BuildWordPieceVocabBlob(json_blob, config);
      return LoadWordPieceTokenizer(vocab_blob, wordpiece_cache_file,
                                    tokenizer_file, config);
    }
  }

  if (!looks_like_tokenizer_json &&
      (requested_wordpiece || looks_like_vocab_txt)) {
    return LoadWordPieceTokenizer(tokenizer_blob, wordpiece_cache_file,
                                  tokenizer_file, config);
  }

  if (requested_bpe || may_be_bpe_json) {
    json &json_blob = get_tokenizer_json();
    if (IsBPETokenizerJson(json_blob)) {
      return LoadExactBPETokenizer(tokenizer_blob, bpe_cache_file,
                                   tokenizer_file, json_blob);
    }
  }

  return tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);
}

} // namespace causallm
