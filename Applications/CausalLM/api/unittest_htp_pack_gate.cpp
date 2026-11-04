// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_htp_pack_gate.cpp
 * @date    29 Oct 2026
 * @brief   Replays the HTP pack fixtures through validate_htp_pack().
 *
 *          The fixture file (testdata/htp_pack_fixtures.json) holds a base
 *          config per pack family and per-case overrides. Its "gate" field is
 *          the expected verdict of the unified gate: "accept", "reject" (with
 *          an optional message substring in "gate_error") or "none".
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "htp_pack_gate.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

#ifndef HTP_PACK_FIXTURES
#error "HTP_PACK_FIXTURES must name htp_pack_fixtures.json"
#endif

namespace {

using quick_dot_ai::HtpPackKind;
using quick_dot_ai::validate_htp_pack;

nlohmann::json load_fixtures() {
  std::ifstream in(HTP_PACK_FIXTURES);
  EXPECT_TRUE(in.good()) << "cannot open " << HTP_PACK_FIXTURES;
  return nlohmann::json::parse(in);
}

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

BackendType backend_of(const std::string &name) {
  if (name == "CPU")
    return CAUSAL_LM_BACKEND_CPU;
  if (name == "GPU")
    return CAUSAL_LM_BACKEND_GPU;
  EXPECT_EQ(name, "NPU");
  return CAUSAL_LM_BACKEND_NPU;
}

ModelQuantizationType quant_of(const std::string &name) {
  EXPECT_EQ(name, "W4A32");
  return CAUSAL_LM_QUANTIZATION_W4A32;
}

/** @brief Overlay the keys of @a over on @a base. */
void overlay(nlohmann::json &base, const nlohmann::json &over) {
  for (auto it = over.begin(); it != over.end(); ++it)
    base[it.key()] = it.value();
}

} // namespace

TEST(HtpPackGateTest, FixtureCasesMatchGateVerdict) {
  const nlohmann::json fx = load_fixtures();
  ASSERT_TRUE(fx.contains("cases"));
  ASSERT_GE(fx["cases"].size(), 13u);

  for (const auto &c : fx["cases"]) {
    const std::string name = c.at("name");
    SCOPED_TRACE(name);

    const auto &base = fx["bases"].at(c.at("base").get<std::string>());
    nlohmann::json cfg = base.at("config");
    nlohmann::json nntr = base.at("nntr_config");
    if (c.contains("config"))
      overlay(cfg, c["config"]);
    if (c.contains("nntr_config"))
      overlay(nntr, c["nntr_config"]);

    const nlohmann::json &req =
      c.contains("request") ? c["request"] : fx.at("request");
    const BackendType backend = backend_of(req.at("backend"));
    const ModelQuantizationType quant = quant_of(req.at("quantization"));
    const std::string id = upper(c.at("id"));
    const std::string gate = c.at("gate");

    if (gate == "accept") {
      HtpPackKind kind = HtpPackKind::NONE;
      EXPECT_NO_THROW(kind = validate_htp_pack(id, backend, quant, cfg, nntr));
      EXPECT_EQ(kind, base.at("nntr_config").contains("lfm2_execution_profile")
                        ? HtpPackKind::LFM2
                        : HtpPackKind::GEMMA4);
    } else if (gate == "reject") {
      try {
        validate_htp_pack(id, backend, quant, cfg, nntr);
        ADD_FAILURE() << "expected std::invalid_argument";
      } catch (const std::invalid_argument &e) {
        const std::string want = c.value("gate_error", std::string());
        EXPECT_NE(std::string(e.what()).find(want), std::string::npos)
          << "message was: " << e.what();
      }
    } else {
      ASSERT_EQ(gate, "none");
      EXPECT_EQ(validate_htp_pack(id, backend, quant, cfg, nntr),
                HtpPackKind::NONE);
    }
  }
}

/** @brief The three Gemma4 ids are the only ones the model-id arm accepts. */
TEST(HtpPackGateTest, Gemma4IdList) {
  const nlohmann::json fx = load_fixtures();
  const auto &base = fx["bases"]["gemma4"];
  nlohmann::json nntr = base["nntr_config"];
  overlay(nntr, {{"moe_layer_dtype", "QS2CX_WH"},
                 {"fc_layer_dtype", "QS4CX"},
                 {"lmhead_dtype", "Q4_0"}});

  for (const char *id : {"GEMMA4-26B-A4B-HTP", "GEMMA4-26B-A4B-HTP-LONG",
                         "GEMMA4-26B-A4B-HTP-2BIT-4BIT"}) {
    EXPECT_EQ(validate_htp_pack(id, CAUSAL_LM_BACKEND_NPU,
                                CAUSAL_LM_QUANTIZATION_W4A32, base["config"],
                                nntr),
              HtpPackKind::GEMMA4)
      << id;
  }
  /* The profile alone, under an id that is not an HTP id, is rejected. */
  EXPECT_THROW(
    validate_htp_pack("GEMMA4-26B-A4B-HTP-OTHER", CAUSAL_LM_BACKEND_NPU,
                      CAUSAL_LM_QUANTIZATION_W4A32, base["config"], nntr),
    std::invalid_argument);
}
