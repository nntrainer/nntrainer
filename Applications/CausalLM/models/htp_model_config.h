// SPDX-License-Identifier: Apache-2.0
/** @file htp_model_config.h
 * @brief Existing text-model and generation defaults for HTP model loaders.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_MODEL_CONFIG_H
#define CAUSALLM_HTP_MODEL_CONFIG_H
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#if __has_include(<htp_runtime_config.h>)
#include <htp_runtime_config.h>
#else
#include <htp_backend/htp_runtime_config.h>
#endif
#include <json.hpp>
#include <limits>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>
namespace causallm {
using json = nlohmann::json;

namespace detail {
inline bool finiteHtpConfigNumber(double value) {
  static_assert(sizeof(double) == sizeof(uint64_t) &&
                std::numeric_limits<double>::is_iec559);
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

template <typename T>
inline void readHtpConfigValue(const json &value, T &out,
                               const std::string &key) {
  const auto invalid = [&] {
    throw std::invalid_argument("Invalid runtime configuration: " + key);
  };
  if constexpr (std::is_same_v<T, bool>) {
    if (!value.is_boolean())
      invalid();
    out = value.get<bool>();
  } else if constexpr (std::is_integral_v<T>) {
    if (!value.is_number_integer())
      invalid();
    if (value.is_number_unsigned()) {
      const auto number = value.get<uint64_t>();
      if (number > static_cast<uint64_t>(std::numeric_limits<T>::max()))
        invalid();
      out = static_cast<T>(number);
    } else {
      const auto number = value.get<int64_t>();
      if constexpr (std::is_unsigned_v<T>) {
        if (number < 0 ||
            static_cast<uint64_t>(number) > std::numeric_limits<T>::max())
          invalid();
      } else if (number < std::numeric_limits<T>::min() ||
                 number > std::numeric_limits<T>::max()) {
        invalid();
      }
      out = static_cast<T>(number);
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    if (!value.is_number())
      invalid();
    const auto number = value.get<double>();
    if (!finiteHtpConfigNumber(number) ||
        std::abs(number) > std::numeric_limits<T>::max())
      invalid();
    out = static_cast<T>(number);
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (!value.is_string())
      invalid();
    out = value.get<std::string>();
  }
}

template <typename T>
inline void readHtpConfigValue(const json &value, std::optional<T> &out,
                               const std::string &key) {
  if (value.is_null()) {
    out.reset();
    return;
  }
  T parsed{};
  readHtpConfigValue(value, parsed, key);
  out = std::move(parsed);
}

inline void readHtpConfigValue(const json &value, std::vector<int> &out,
                               const std::string &key) {
  if (!value.is_array())
    throw std::invalid_argument("Invalid runtime configuration: " + key);
  out.clear();
  for (const auto &entry : value) {
    int cpu = 0;
    readHtpConfigValue(entry, cpu, key);
    out.push_back(cpu);
  }
}
} // namespace detail

/** Parse the model's runtime policy before opening the DSP or weight files. */
inline nntrainer::HtpRuntimeConfig parseHtpRuntimeConfig(const json &nntr) {
  nntrainer::HtpRuntimeConfig result;
  result.moe.hadamard = "off";
  result.moe.hadamard_scope = "2bit";
  for (const char *section : {"htp", "moe"}) {
    if (!nntr.contains(section))
      continue;
    if (!nntr.at(section).is_object())
      throw std::invalid_argument(std::string(section) + " must be an object");
    for (const auto &[key, value] : nntr.at(section).items()) {
      bool known = false;
      if (std::string(section) == "htp") {
#define NNTR_READ_HTP(type, name, default_value)                               \
  if (key == #name) {                                                          \
    detail::readHtpConfigValue(value, result.htp.name, "htp." #name);          \
    known = true;                                                              \
  }
        NNTR_HTP_CONFIG_FIELDS(NNTR_READ_HTP)
#undef NNTR_READ_HTP
      } else {
#define NNTR_READ_MOE(type, name, default_value)                               \
  if (key == #name) {                                                          \
    detail::readHtpConfigValue(value, result.moe.name, "moe." #name);          \
    known = true;                                                              \
  }
        NNTR_MOE_CONFIG_FIELDS(NNTR_READ_MOE)
#undef NNTR_READ_MOE
      }
      if (!known)
        throw std::invalid_argument(
          "Unknown runtime configuration: " + std::string(section) + "." + key);
    }
  }
  const auto has_moe = [&](const char *key) {
    return nntr.contains("moe") && nntr.at("moe").contains(key);
  };
  if (nntr.contains("moe_cache_experts")) {
    unsigned count = 0;
    detail::readHtpConfigValue(nntr.at("moe_cache_experts"), count,
                               "moe_cache_experts");
    if (count == 0)
      count = 16;
    if (has_moe("cache_experts") && result.moe.cache_experts != count)
      throw std::invalid_argument("Conflicting moe.cache_experts settings");
    result.moe.cache_experts = count;
  }
  for (const char *key : {"hadamard", "hadamard_scope"}) {
    const auto flat = std::string("moe_") + key;
    if (!nntr.contains(flat))
      continue;
    std::string value;
    detail::readHtpConfigValue(nntr.at(flat), value, flat);
    auto &target = std::string(key) == "hadamard" ? result.moe.hadamard
                                                  : result.moe.hadamard_scope;
    if (has_moe(key) && !nntr.at("moe").at(key).is_null() && target != value)
      throw std::invalid_argument("Conflicting moe." + std::string(key) +
                                  " settings");
    target = value;
  }
  if (!result.moe.hadamard)
    result.moe.hadamard = "off";
  if (!result.moe.hadamard_scope)
    result.moe.hadamard_scope = "2bit";
  nntrainer::validateHtpRuntimeConfig(result);
  return result;
}

/** Preserve existing pack metadata while accepting its canonical MoE policy. */
inline nntrainer::HtpRuntimeConfig normalizeHtpRuntimeConfig(json &nntr) {
  auto result = parseHtpRuntimeConfig(nntr);
  nntr["moe_cache_experts"] = result.moe.cache_experts;
  nntr["moe_hadamard"] = *result.moe.hadamard;
  nntr["moe_hadamard_scope"] = *result.moe.hadamard_scope;
  return result;
}

/** Input artifacts follow the model pack; diagnostic output paths are literal.
 */
inline void resolveHtpRuntimeConfigPaths(json &nntr,
                                         const std::string &model_path) {
  for (const auto &[section, key] : {std::pair{"htp", "static_linear"},
                                     std::pair{"moe", "tl2_prepared_file"}}) {
    if (!nntr.contains(section) || !nntr.at(section).is_object() ||
        !nntr.at(section).contains(key) ||
        !nntr.at(section).at(key).is_string())
      continue;
    const std::filesystem::path path =
      nntr.at(section).at(key).get<std::string>();
    if (!path.empty() && path.is_relative())
      nntr[section][key] = (std::filesystem::absolute(model_path) / path)
                             .lexically_normal()
                             .string();
  }
}

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
