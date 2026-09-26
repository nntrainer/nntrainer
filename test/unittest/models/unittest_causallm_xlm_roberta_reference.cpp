// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_xlm_roberta_reference.cpp
 * @date   18 June 2026
 * @brief  Differential test for the tiny XLM-RoBERTa encoder.
 *
 * Compares nntrainer's XLM-RoBERTa encoder output (raw, unpooled last hidden
 * state) against a golden fixture generated from HuggingFace XLMRobertaModel.
 * The reference is the full [seq_len, hidden] hidden state flattened row-major,
 * since the nntrainer path performs no pooling.
 *
 * XLMRobertaForMaskedLM sanitizes its config in the constructor, so a thin
 * subclass owns the (virtual) Transformer mem-initializer.
 *
 * Guarded for non-Windows only, matching the factory registration in
 * Applications/CausalLM/main.cpp. The test skips when the fixture is absent.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#if !defined(_WIN32)

#include <causallm_test_utils.h>

#include <xlm_roberta.h>

#include <memory>

namespace {

/**
 * @brief Tiny XLM-RoBERTa differential adapter
 *
 * XLMRobertaForMaskedLM sanitizes its config before initializing the virtual
 * Transformer base, so this thin subclass owns that mem-initializer.
 */
class TinyXLMRobertaRefAdapter final
  : public causallm_test::EmbeddingTestAdapter<
      causallm::XLMRobertaForMaskedLM> {
public:
  TinyXLMRobertaRefAdapter(causallm::json &cfg, causallm::json &generation_cfg,
                           causallm::json &nntr_cfg) :
    causallm::Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    causallm_test::EmbeddingTestAdapter<causallm::XLMRobertaForMaskedLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Differential model descriptor for the tiny XLM-RoBERTa fixture
 */
causallm_test::DifferentialModel xlmRobertaModel() {
  return {
    "xlm_roberta_tiny",
    [](causallm::json &cfg, causallm::json &gen_cfg, causallm::json &nntr_cfg) {
      return std::make_unique<TinyXLMRobertaRefAdapter>(cfg, gen_cfg, nntr_cfg);
    },
  };
}

/**
 * @brief FP32 raw encoder output matches the HF XLMRobertaModel reference
 */
TEST(TinyXLMRobertaDifferentialTest, FP32MatchesHFReference) {
  causallm_test::runFp32EmbeddingDifferentialChecks(xlmRobertaModel());
}

/**
 * @brief Q4_0 encoder output is close to the HF FP32 reference
 *        (skips automatically if nntr_quantize does not support
 * XLMRobertaForMaskedLM)
 */
TEST(TinyXLMRobertaDifferentialTest, Q40CloseToFP32Reference) {
  causallm_test::runQ40EmbeddingDifferentialChecks(xlmRobertaModel());
}

} // namespace

#endif // !_WIN32
