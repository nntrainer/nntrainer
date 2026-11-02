// SPDX-License-Identifier: Apache-2.0
/** @file htp_model_config.h
 * @brief Existing text-model and generation defaults for HTP model loaders.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_MODEL_CONFIG_H
#define CAUSALLM_HTP_MODEL_CONFIG_H
#include <json.hpp>
#include <set>
#include <sstream>
#include <vector>
namespace causallm {
using json = nlohmann::json;
inline std::set<int> htpLayerIds(const std::string &csv) {
  std::set<int> ids;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty())
      ids.insert(std::stoi(tok));
  }
  return ids;
}

inline json &normalizeHtpModelConfig(json &cfg) {
  if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
    const auto &text_cfg = cfg["text_config"];
    for (auto it = text_cfg.begin(); it != text_cfg.end(); ++it) {
      if (!cfg.contains(it.key())) {
        cfg[it.key()] = it.value();
      }
    }
  }

  if (cfg.contains("use_bidirectional_attention") &&
      !cfg["use_bidirectional_attention"].is_boolean())
    cfg.erase("use_bidirectional_attention");

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }

  if (!cfg.contains("head_dim") && cfg.contains("hidden_size") &&
      cfg.contains("num_attention_heads")) {
    cfg["head_dim"] = cfg["hidden_size"].get<unsigned int>() /
                      cfg["num_attention_heads"].get<unsigned int>();
  }

  return cfg;
}

inline json &normalizeHtpGenerationConfig(json &gen_cfg, const json &cfg) {
  if (!gen_cfg.contains("eos_token_id")) {
    if (cfg.contains("eos_token_id")) {
      auto eos = cfg["eos_token_id"];
      if (eos.is_number()) {
        gen_cfg["eos_token_id"] =
          std::vector<unsigned int>{eos.get<unsigned int>()};
      } else {
        gen_cfg["eos_token_id"] = eos;
      }
    }
  } else {
    auto eos = gen_cfg["eos_token_id"];
    if (eos.is_number()) {
      gen_cfg["eos_token_id"] =
        std::vector<unsigned int>{eos.get<unsigned int>()};
    }
  }

  return gen_cfg;
}
} // namespace causallm
#endif
