// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    htp_pack_gate.cpp
 * @date    29 Oct 2026
 * @brief   Load-time contract of the HTP (Hexagon) model packs.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#include "htp_pack_gate.h"

#include <stdexcept>
#include <vector>

namespace quick_dot_ai {

namespace {

/** @brief String value of an nntr_config key, empty when absent. */
std::string str_of(const nlohmann::json &j, const char *key) {
  return j.value(key, std::string());
}

/** @brief Architecture list of config.json, empty when absent. */
std::vector<std::string> architectures_of(const nlohmann::json &cfg) {
  return cfg.value("architectures", std::vector<std::string>{});
}

void validate_lfm2(bool is_model, bool is_profile, BackendType compute,
                   ModelQuantizationType quant, const nlohmann::json &cfg,
                   const nlohmann::json &nntr_cfg) {
  if (!is_model || !is_profile || compute != CAUSAL_LM_BACKEND_NPU ||
      quant != CAUSAL_LM_QUANTIZATION_W4A32 ||
      architectures_of(cfg) != std::vector<std::string>{"Lfm2MoeForCausalLM"} ||
      str_of(nntr_cfg, "lfm2_weight_format") != "sr15_v2" ||
      nntr_cfg.value("use_kvcache", false) ||
      nntr_cfg.value("save_kvcache", false))
    throw std::invalid_argument(
      "LFM2 HTP requires the NPU/W4A32 sr15_v2 fused pack");
}

void validate_gemma4(bool is_model, bool is_profile, BackendType compute,
                     ModelQuantizationType quant, const nlohmann::json &cfg,
                     const nlohmann::json &nntr_cfg) {
  if (!is_model || !is_profile || compute != CAUSAL_LM_BACKEND_NPU ||
      quant != CAUSAL_LM_QUANTIZATION_W4A32 ||
      str_of(nntr_cfg, "gemma4_weight_format") != "htp_fused_v1" ||
      str_of(nntr_cfg, "moe_engine") != "htp" ||
      str_of(nntr_cfg, "attention_engine") != "htp" ||
      str_of(nntr_cfg, "attn_proj_engine") != "htp" ||
      str_of(nntr_cfg, "dense_ffn_engine") != "htp")
    throw std::invalid_argument(
      "Gemma4 HTP requires the validated NPU/W4A32 preset");

  /* QS2 experts with a CPU Q4_0 lm_head, or TL2 experts with the affine
   * QS4 tail and an HTP lm_head. */
  const bool pack_qs2 = str_of(nntr_cfg, "moe_layer_dtype") == "QS2CX_WH" &&
                        str_of(nntr_cfg, "fc_layer_dtype") == "QS4CX" &&
                        str_of(nntr_cfg, "lmhead_dtype") == "Q4_0";
  const bool pack_tl2 =
    str_of(nntr_cfg, "moe_layer_dtype") == "TL2CX_WH" &&
    str_of(nntr_cfg, "fc_layer_dtype") == "QS4CX_WH_AFFINE" &&
    str_of(nntr_cfg, "lmhead_dtype") == "QS4CX_WH_AFFINE" &&
    str_of(nntr_cfg, "lmhead_engine") == "htp";

  if (architectures_of(cfg) != std::vector<std::string>{"Gemma4ForCausalLM"} ||
      cfg.value("hidden_size", 0) != 2816 ||
      cfg.value("num_hidden_layers", 0) != 30 ||
      cfg.value("num_experts", 0) != 128 || !(pack_qs2 || pack_tl2) ||
      str_of(nntr_cfg, "embedding_dtype") != "Q4_0" ||
      nntr_cfg.value("use_kvcache", false) ||
      nntr_cfg.value("save_kvcache", false))
    throw std::invalid_argument(
      "Unsupported Gemma4 HTP pack or KV snapshot option");
}

} // namespace

HtpPackKind validate_htp_pack(const std::string &model_id_upper,
                              BackendType compute, ModelQuantizationType quant,
                              const nlohmann::json &cfg,
                              const nlohmann::json &nntr_cfg) {
  HtpPackKind kind = HtpPackKind::NONE;

  const bool lfm2_model = model_id_upper == "LFM2-24BA2B-HTP";
  const bool lfm2_profile =
    str_of(nntr_cfg, "lfm2_execution_profile") == "htp_fused";
  if (lfm2_model || lfm2_profile) {
    validate_lfm2(lfm2_model, lfm2_profile, compute, quant, cfg, nntr_cfg);
    kind = HtpPackKind::LFM2;
  }

  const bool gemma4_model = model_id_upper == "GEMMA4-26B-A4B-HTP" ||
                            model_id_upper == "GEMMA4-26B-A4B-HTP-LONG" ||
                            model_id_upper == "GEMMA4-26B-A4B-HTP-2BIT-4BIT";
  const bool gemma4_profile =
    str_of(nntr_cfg, "gemma4_execution_profile") == "htp_fused";
  if (gemma4_model || gemma4_profile) {
    validate_gemma4(gemma4_model, gemma4_profile, compute, quant, cfg,
                    nntr_cfg);
    kind = HtpPackKind::GEMMA4;
  }

  return kind;
}

} // namespace quick_dot_ai
