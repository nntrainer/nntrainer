// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	model_registry.cpp
 * @date	14 July 2026
 * @brief	Shared model-registration for the CausalLM Factory
 * @see		https://github.com/nnstreamer/
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include "model_registry.h"

#include <cstdlib>
#include <mutex>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

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
#if !defined(_WIN32)
#include "xlm_roberta.h"
#endif
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

namespace {

/**
 * @brief Cap the number of glibc malloc arenas to one.
 *
 * The weight loader fans out over hardware_concurrency() worker threads, and
 * glibc hands each of them its own per-thread arena. Those arenas are never
 * returned to the OS for the life of the process, so they sit on the peak: the
 * 8-worker loader costs ~120MB of host VmHWM purely in arena fragmentation
 * (measured 1206 -> 1084 MiB on a 29.2K-token summarization; the generated
 * tokens are byte-identical and prefill/decode TPS are unchanged, 6985 / 103.9
 * either way). One arena serializes malloc across the loader threads,
 * but the loaders are I/O- and memcpy-bound, not allocator-bound.
 *
 * Must run before the loader threads spawn -- glibc only honours a lowered
 * M_ARENA_MAX for arenas not yet created. If the user set MALLOC_ARENA_MAX in
 * the environment, glibc has already applied it at startup and their choice
 * wins; do nothing in that case.
 */
void capMallocArenas() {
#if defined(__GLIBC__)
  if (std::getenv("MALLOC_ARENA_MAX") != nullptr)
    return;
  mallopt(M_ARENA_MAX, 1);
#endif
}

} // namespace

void registerAllModels() {
  static std::once_flag once;
  std::call_once(once, []() {
    capMallocArenas();
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
