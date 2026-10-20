// SPDX-License-Identifier: Apache-2.0
/**
 * @file gemma4_htp_static_quant.h
 * @brief Checkpoint supplied U8 input encodings for Gemma4 HTP linear ops.
 * @author Codex <noreply@openai.com>
 * @bug See the repository issue tracker for known issues.
 */
#ifndef GEMMA4_HTP_STATIC_QUANT_H
#define GEMMA4_HTP_STATIC_QUANT_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <json.hpp>
#include <limits>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>

namespace causallm {

/** @brief An unsigned eight-bit affine input grid. */
struct HtpStaticEncoding {
  float scale;
  int zero_point;
};

/** @brief Only these graph weights consume a linear input encoding. */
inline bool htpStaticLinearWeight(const std::string &name) {
  static const std::regex pattern(
    R"((layer[0-9]+_(qkv:(qweight|kweight|vweight)|post_attention_norm:weight|ffn:(gate|up|down)|sparse_moe:(gate|expert_gate_up_[0-9]+|expert_down_[0-9]+)))|output_of_causallm:weight)");
  return std::regex_match(name, pattern);
}

/** @brief Convert either device sidecars or their logical-site provenance. */
inline std::map<std::string, HtpStaticEncoding>
parseHtpStaticLinear(const nlohmann::json &document, unsigned experts) {
  using Json = nlohmann::json;
  std::map<std::string, HtpStaticEncoding> result;
  const bool logical = document.contains("sites");
  const Json &entries = logical ? document.at("sites") : document;
  if (!entries.is_object() || entries.empty())
    throw std::invalid_argument(
      "static activation encodings must be a nonempty object");
  const std::regex site_pattern(R"(model\.layers\.([0-9]+)\.(.+)/in)");
  const std::map<std::string, std::string> suffixes = {
    {"self_attn.q_proj", "qkv:qweight"},
    {"self_attn.k_proj", "qkv:kweight"},
    {"self_attn.v_proj", "qkv:vweight"},
    {"self_attn.o_proj", "post_attention_norm:weight"},
    {"mlp.gate_proj", "ffn:gate"},
    {"mlp.up_proj", "ffn:up"},
    {"mlp.down_proj", "ffn:down"},
    {"router.proj", "sparse_moe:gate"}};
  for (const auto &entry : entries.items()) {
    const auto &value = entry.value();
    double scale, zp;
    if (logical) {
      if (!value.is_object() || value.value("bitwidth", 0) != 8 ||
          value.value("dtype", std::string()) != "int" ||
          !value.contains("scale") || !value.at("scale").is_number() ||
          !value.contains("offset") || !value.at("offset").is_number())
        throw std::invalid_argument("expected a static U8 encoding at " +
                                    entry.key());
      scale = value.at("scale").get<double>();
      zp = -value.at("offset").get<double>();
    } else {
      if (!value.is_array() || value.size() != 2 || !value[0].is_number() ||
          !value[1].is_number())
        throw std::invalid_argument("expected [scale, zero_point] at " +
                                    entry.key());
      scale = value[0].get<double>();
      zp = value[1].get<double>();
    }
    const float s = static_cast<float>(scale);
    uint32_t scale_bits;
    uint64_t zp_bits;
    std::memcpy(&scale_bits, &s, sizeof(s));
    std::memcpy(&zp_bits, &zp, sizeof(zp));
    // Build flags include fast-math: classify the representation, not isfinite.
    if ((scale_bits & 0x7f800000u) == 0x7f800000u ||
        (zp_bits & UINT64_C(0x7ff0000000000000)) ==
          UINT64_C(0x7ff0000000000000) ||
        s < std::numeric_limits<float>::min() ||
        s > std::numeric_limits<float>::max() / 255.0f || zp < 0 || zp > 255 ||
        zp != std::floor(zp))
      throw std::invalid_argument("invalid static U8 grid at " + entry.key());
    const HtpStaticEncoding encoding{s, static_cast<int>(zp)};
    auto add = [&](std::string name) {
      const std::string alias = "post_attention_norm:proj";
      const auto pos = name.find(alias);
      if (pos != std::string::npos)
        name.replace(pos, alias.size(), "post_attention_norm:weight");
      if (!htpStaticLinearWeight(name) ||
          !result.emplace(name, encoding).second)
        throw std::invalid_argument("unknown or duplicate static input " +
                                    name);
    };
    if (!logical) {
      add(entry.key());
    } else if (entry.key() == "lm_head/in") {
      add("output_of_causallm:weight");
    } else {
      std::smatch match;
      const std::string name = entry.key();
      if (!std::regex_match(name, match, site_pattern))
        throw std::invalid_argument("unsupported static linear site " + name);
      const std::string prefix = "layer" + match[1].str() + "_";
      const auto suffix = suffixes.find(match[2].str());
      if (suffix != suffixes.end()) {
        add(prefix + suffix->second);
      } else if (match[2] == "experts.gate_up_proj" ||
                 match[2] == "experts.down_proj") {
        if (!experts)
          throw std::invalid_argument("static expert encoding without experts");
        const auto kind = match[2] == "experts.gate_up_proj"
                            ? "sparse_moe:expert_gate_up_"
                            : "sparse_moe:expert_down_";
        for (unsigned e = 0; e < experts; ++e)
          add(prefix + kind + std::to_string(e));
      } else {
        throw std::invalid_argument("unsupported static linear site " + name);
      }
    }
  }
  return result;
}

} // namespace causallm
#endif
