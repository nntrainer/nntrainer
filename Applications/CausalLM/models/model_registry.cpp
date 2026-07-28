// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_registry.cpp
 * @brief  Registration of every runnable CausalLM model (moved verbatim from
 *         main.cpp).
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "model_registry.h"

#include <mutex>

#include <factory.h>

#include "causal_lm.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma2_causallm.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "xlm_roberta.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "lfm2_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "timm_vit/timm_vit_transformer.h"

namespace causallm {

void registerAllModels() {
  static std::once_flag once;
  std::call_once(once, []() {
    causallm::Factory::Instance().registerModel(
      "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                    nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                         nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Qwen3SlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#if !defined(_WIN32)
    causallm::Factory::Instance().registerModel(
      "Qwen3CachedSlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
    causallm::Factory::Instance().registerModel(
      "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssForCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#if !defined(_WIN32)
    causallm::Factory::Instance().registerModel(
      "GptOssCachedSlimCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
    causallm::Factory::Instance().registerModel(
      "Gemma2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma2CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Gemma4ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma4CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "DebertaV2", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg,
                                                     nntr_cfg);
      });
#if !defined(_WIN32) && !defined(__ANDROID__)
    causallm::Factory::Instance().registerModel(
      "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::MultilingualTinyBert>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
#if !defined(_WIN32)
    causallm::Factory::Instance().registerModel(
      "XLMRobertaForMaskedLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::XLMRobertaForMaskedLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
    causallm::Factory::Instance().registerModel(
      "TimmViT", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::TimmViTTransformer>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Lfm2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Lfm2CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
      });
  });
}

} // namespace causallm
