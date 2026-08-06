// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.cpp
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "quick_dot_ai_api.h"
#ifdef ENABLE_QNN_MODELS
#include "quick_dot_ai_qnn.h"
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "causal_lm.h"
#include "chat_template.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#include "gptoss_cached_slim_causallm.h"
#include "gptoss_causallm.h"
#include "json.hpp"
#include "model_callbacks.h"
#include "model_config_internal.h"
#include "model_descriptor.h"
#include "multilingual_tinybert_16mb.h"
#include "qwen2_causallm.h"
#include "qwen3_cached_slim_moe_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "sentence_transformer.h"
// V-JEPA2 + LFM2 fused video-language model. Depends on the nntrainer
// VjepaLfm2ForConditionalGeneration model + Transformer::run_video virtual,
// which are not present in this tree. Gated off until they are ported here;
// define QUICKAI_ENABLE_VJEPA_LFM2_VIDEO to re-enable the video path.
#if defined(QUICKAI_ENABLE_VJEPA_LFM2_VIDEO)
#include "vjepa_lfm2_vl/vjepa_lfm2_vl.h"
#endif
#include "xgrammar_manager.h"
#include "xgrammar_wrapper.h"
#include <factory.h>
#ifdef ENABLE_QNN_MODELS
#include "gemma4_e2b_qnn.h"
#include "quick_dot_ai_qnn.h"
// Multimodal vision QNN models are excluded: they depend on the OLD
// nntrainer multimodal interface (run_image / multimodal_pointer) absent
// from main nntrainer. See docs/qnn-model-main-adaptation-todo.ko.md.
#if defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
#endif

#endif
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

using json = nlohmann::json;

/**
 * @brief Per-handle state for a loaded CausalLM model instance.
 *
 * Each handle may carry one or more sub-models so that compositions like
 * vision-encoder + LLM can live behind a single handle. The vectors are
 * kept parallel: models[i] ↔ architectures[i] ↔ model_dirs[i] ↔
 * initialization_duration_ms[i]. The single-model API paths
 * (runModelHandleWithMessages / runModelHandleStreaming) operate on models[0]
 * and ignore the rest; the multimodal API drives the full set.
 *
 * Note: the legacy non-handle API (loadModel / ...) is
 * implemented on top of a single static "default" instance of this struct
 * so that existing callers (e.g. test_api) keep working unchanged.
 */
struct CausalLmModel {
  std::mutex mtx;
  std::vector<std::unique_ptr<causallm::Transformer>> models;
  std::vector<std::string> architectures;
  std::vector<std::string> model_dirs;
  std::string last_output;
  std::string native_lib_dir;
  std::vector<double> initialization_duration_ms;
  bool initialized = false;
  int kv_len = 0;
};

// Globals shared across all handles — options set via setOptions() apply
// process-wide regardless of which handle is active.
static std::mutex g_registry_mutex;
static bool g_use_chat_template = true;
static bool g_verbose = false;
static std::string g_last_output = "";
static std::optional<causallm::ChatTemplate> g_chat_template;
static std::string g_formatted_template;
static std::string g_chat_template_name = "default";

// Default handle backing the legacy non-handle API.
static CausalLmModel &get_default_handle() {
  static CausalLmModel instance;
  return instance;
}

class XGrammarLogitsProcessor final : public causallm::LogitsProcessor {
public:
  explicit XGrammarLogitsProcessor(causallm::XGrammar *grammar,
                                   std::function<void()> on_completed = {}) :
    grammar_(grammar), on_completed_(std::move(on_completed)) {}

  void process(float *logits, unsigned int vocab_size,
               unsigned int batch_index) override {
    if (batch_index != 0 || grammar_ == nullptr ||
        !grammar_->isGrammarEnabled()) {
      return;
    }
    grammar_->applyGrammarMask(logits, static_cast<int>(vocab_size));
  }

  void acceptToken(unsigned int token_id, unsigned int batch_index) override {
    if (batch_index != 0 || grammar_ == nullptr ||
        grammar_->getGrammarMatcher() == nullptr) {
      return;
    }
    auto *matcher = grammar_->getGrammarMatcher();
    if (!matcher->AcceptToken(static_cast<int32_t>(token_id))) {
      return;
    }
    if (matcher->IsCompleted() || matcher->IsTerminated()) {
      if (on_completed_) {
        on_completed_();
      }
      return;
    }
    grammar_->getGrammarMatcher()->FillNextTokenBitmask(
      &grammar_->getBitmaskTensor());
  }

  void reset() override {
    if (grammar_ != nullptr) {
      grammar_->resetGrammar();
    }
  }

private:
  causallm::XGrammar *grammar_;
  std::function<void()> on_completed_;
};

static causallm::CausalLM *as_causal_lm(causallm::Transformer *model) {
  return dynamic_cast<causallm::CausalLM *>(model);
}

#ifdef ENABLE_QNN_MODELS
static causallm::Quick_Dot_AI_QNN *as_qnn_model(causallm::Transformer *model) {
  return dynamic_cast<causallm::Quick_Dot_AI_QNN *>(model);
}
#endif

static bool model_supports_text_output(causallm::Transformer *model) {
  if (as_causal_lm(model) != nullptr)
    return true;
#ifdef ENABLE_QNN_MODELS
  if (as_qnn_model(model) != nullptr)
    return true;
#endif
  return false;
}

static bool get_model_output(causallm::Transformer *model,
                             std::string &output) {
  if (auto *causal_model = as_causal_lm(model)) {
    output = causal_model->getOutput(0);
    return true;
  }
#ifdef ENABLE_QNN_MODELS
  if (auto *qnn_model = as_qnn_model(model)) {
    output = qnn_model->getOutput(0);
    return true;
  }
#endif
  return false;
}

static bool set_model_streamer(causallm::Transformer *model,
                               ::BaseStreamer *streamer) {
  if (auto *causal_model = as_causal_lm(model)) {
    causal_model->setStreamer(streamer);
    return true;
  }
#ifdef ENABLE_QNN_MODELS
  if (auto *qnn_model = as_qnn_model(model)) {
    qnn_model->setStreamer(streamer);
    return true;
  }
#endif
  return false;
}

static bool request_model_stop(causallm::Transformer *model) {
  if (auto *causal_model = as_causal_lm(model)) {
    causal_model->requestStop();
    return true;
  }
#ifdef ENABLE_QNN_MODELS
  if (auto *qnn_model = as_qnn_model(model)) {
    qnn_model->requestStop();
    return true;
  }
#endif
  return false;
}

static std::map<std::string, std::string> g_model_path_map = {
  {"QWEN3-0.6B", "qwen3-0.6b"},
  {"QWEN3-1.7B-Q40", "qwen3-1.7b-q40-arm"},
  {"TINY_BERT", "tiny_bert"},
  {"FUNCTION_GEMMA", "function_gemma"},
  {"GEMMA4_CPU", "gemma4_cpu"},
  {"OURO_EMBEDDING", "ouro_embedding"},
#ifdef ENABLE_QNN_MODELS
  {"GEMMA4-E2B-QNN", "gemma-4-e2b-qnn"},
  {"VJEPA2-QNN", "vjepa2-qnn"},
#endif
};

/**
 * @brief RegisteredModel
 */
struct RegisteredModel {
  std::string arch_name;
  ModelRuntimeConfig config;
};
static std::map<std::string, RegisteredModel> g_model_registry;
static std::map<std::string, ModelArchConfig> g_arch_config_map;

// Internal C++ registration functions — called from model_config.cpp
// These bypass extern "C" PLT and write directly to our static maps.
namespace quick_dot_ai {

void register_arch(const char *arch_name, ModelArchConfig config) {
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
}

void register_model(const char *model_name, const char *arch_name,
                    ModelRuntimeConfig config) {
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);
  g_model_registry[name] = {aname, config};
}

} // namespace quick_dot_ai

// ---------------------------------------------------------------------------
// T4: string-id descriptor registry + catalog JSON
// ---------------------------------------------------------------------------

// Lazily-constructed (Meyers singleton) so cross-library self-registration
// from model plugins (libquick_dot_ai.so) — whose static constructors run
// BEFORE this lib's globals would be constructed — lands in a live registry
// instead of an uninitialized one (static-init-order fiasco).
static std::mutex &descriptor_mutex() {
  static std::mutex m;
  return m;
}
static std::vector<ModelDescriptor> &descriptor_registry() {
  static std::vector<ModelDescriptor> v;
  return v;
}

namespace quick_dot_ai {
void register_model_descriptor(const ModelDescriptor *desc) {
  if (!desc || !desc->id)
    return;
  std::lock_guard<std::mutex> lock(descriptor_mutex());
  for (auto &d : descriptor_registry()) {
    if (std::strcmp(d.id, desc->id) == 0) {
      LOGE("register_model_descriptor: duplicate id '%s', overwriting",
           desc->id);
      d = *desc;
      return;
    }
  }
  descriptor_registry().push_back(*desc);
}
} // namespace quick_dot_ai

/** Find a descriptor by string id. Returns a copy while locked, or nullopt. */
static std::optional<ModelDescriptor> find_descriptor_by_id(const char *id) {
  if (!id)
    return std::nullopt;
  std::lock_guard<std::mutex> lk(descriptor_mutex());
  for (const auto &d : descriptor_registry())
    if (std::strcmp(d.id, id) == 0)
      return d; // copy while locked
  return std::nullopt;
}

// Library-owned buffer: rebuilt on every call and returned via c_str().
// The pointer is valid only until the next call to getModelCatalogJson().
static std::string g_catalog_json_cache;

/**
 * Returns a pointer to a library-owned buffer containing a JSON array of
 * registered model descriptors. The buffer is valid only until the next
 * call to getModelCatalogJson(). Callers must copy the contents immediately
 * (e.g. via JNI NewStringUTF) and must not hold the pointer across calls.
 * Not safe for concurrent access to the returned pointer.
 */
extern "C" const char *getModelCatalogJson(void) {
  auto json_escape = [](const char *s) -> std::string {
    if (!s)
      return "";
    std::string out;
    for (; *s; ++s) {
      if (*s == '"')
        out += "\\\"";
      else if (*s == '\\')
        out += "\\\\";
      else
        out += *s;
    }
    return out;
  };

  std::lock_guard<std::mutex> lock(descriptor_mutex());
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < descriptor_registry().size(); ++i) {
    const auto &d = descriptor_registry()[i];
    if (i)
      os << ",";
    os << "{\"id\":\"" << json_escape(d.id) << "\",\"family\":\""
       << json_escape(d.family) << "\",\"display_name\":\""
       << json_escape(d.display_name ? d.display_name : d.id)
       << "\",\"runtime\":" << static_cast<int>(d.runtime)
       << ",\"backend_mask\":" << d.backend_mask
       << ",\"capabilities\":" << d.capabilities;
    if (d.sd_variant_id)
      os << ",\"sd_variant_id\":\"" << json_escape(d.sd_variant_id) << "\"";
    os << "}";
  }
  os << "]";
  g_catalog_json_cache = os.str();
  return g_catalog_json_cache.c_str();
}

// Helper to register models (similar to main.cpp) ensuring factory is
// populated. Factory registration is singleton and persistent, but we do it
// once here to be sure. Since mquiain.cpp is not linked, we must duplicate
// registration or share it. Assuming this lib is used independently of
// main.cpp.
static void register_models() {
  static std::once_flag flag;
  std::call_once(flag, []() {
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
    causallm::Factory::Instance().registerModel(
      "Qwen3CachedSlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssForCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "GptOssCachedSlimCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
          cfg, generation_cfg, nntr_cfg);
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
      "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::MultilingualTinyBert>(
          cfg, generation_cfg, nntr_cfg);
      });
    // Ouro (Universal-Transformer) self-registers "OuroModel" from its own
    // TU (src/models/ouro/ouro_embedding.cpp) via __attribute__((constructor)),
    // same pattern as the QNN models below.

    // V-JEPA2 + LFM2 fused video-language model (CPU). Sub-models (ViT,
    // projector, LFM2) are constructed internally, so only the combined
    // architecture needs a Factory entry.
    // Gated: see the QUICKAI_ENABLE_VJEPA_LFM2_VIDEO note at the include of
    // vjepa_lfm2_vl.h.
#if defined(QUICKAI_ENABLE_VJEPA_LFM2_VIDEO)
    causallm::Factory::Instance().registerModel(
      "Lfm2VLVJepa21BModel", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::VjepaLfm2ForConditionalGeneration>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif

#ifdef ENABLE_QNN_MODELS
    causallm::Factory::Instance().registerModel(
      "Gemma4_E2B_QNN", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Gemma4_E2B_QNN>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
#endif
    // Register built-in configurations
    quick_dot_ai::register_builtin_configs();
  });
}

static const char *get_model_name_from_type(ModelType type) {
  switch (type) {
  case CAUSAL_LM_MODEL_QWEN3_0_6B:
    return "QWEN3-0.6B";
  case CAUSAL_LM_MODEL_QWEN3_1_7B_Q40:
    return "QWEN3-1.7B-Q40";
  case CAUSAL_LM_MODEL_TINY_BERT:
    return "TINY_BERT";
  case CAUSAL_LM_MODEL_FUNCTION_GEMMA:
    return "FUNCTION_GEMMA";
  case CAUSAL_LM_MODEL_GEMMA4_CPU:
    return "GEMMA4_CPU";
  case CAUSAL_LM_MODEL_OURO_EMBEDDING:
    return "OURO_EMBEDDING";
#ifdef ENABLE_QNN_MODELS
  case CAUSAL_LM_MODEL_GEMMA4_E2B_QNN:
    return "GEMMA4-E2B-QNN";
  case CAUSAL_LM_MODEL_VJEPA2_QNN:
    return "VJEPA2-QNN";
#endif
  default:
    return nullptr;
  }
}

static std::string apply_chat_template(const std::string &architecture,
                                       const std::string &input) {
  // Use dynamic chat template from tokenizer_config.json if available
  if (g_chat_template) {
    nlohmann::json request;
    request["messages"] = nlohmann::json::array();
    request["messages"].push_back({{"role", "user"}, {"content", input}});
    request["add_generation_prompt"] = true;
    try {
      return g_chat_template->apply(request);
    } catch (const std::exception &e) {
      LOGE("Chat template apply failed: %s", e.what());
      // fallback to hardcoded
    }
  }

  LOGE("----------------APPLY CHAT FALLBACKS!!!!!!-------------");

  // Fallback: hardcoded per-architecture templates
  if (architecture == "LlamaForCausalLM") {
    // Llama 2/3 chat format: [INST] {prompt} [/INST]
    return "[INST] " + input + " [/INST]";
  } else if (architecture == "Qwen2ForCausalLM" ||
             architecture == "Qwen3ForCausalLM" ||
             architecture == "Qwen3MoeForCausalLM" ||
             architecture == "Qwen3SlimMoeForCausalLM" ||
             architecture == "Qwen3CachedSlimMoeForCausalLM") {
    // Qwen chat format
    // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    return "<|im_start|>user\n" + input + "<|im_end|>\n<|im_start|>assistant\n";
  } else if (architecture == "Gemma3ForCausalLM") {
    // Gemma chat format:
    // <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    return "<start_of_turn>user\n" + input +
           "<end_of_turn>\n<start_of_turn>model\n";
  } else if (architecture == "Gemma4ForCausalLM" ||
             architecture == "Gemma4_E2B_QNN") {
    // Gemma 4 requires the prompt to begin with the <bos> token. The model's
    // own nntr_config.json sample_input documents the canonical format as
    // "<bos><|turn>user\n...<turn|>\n<|turn>model\n". This model's
    // tokenizer.json post_processor does NOT add <bos>, and tokenizer_config
    // .json carries no chat_template / add_bos_token, so it must be added
    // here or the model receives a BOS-less prompt and emits garbage.
    // "<bos>" is a special added token (id 2) and encodes to that single id.
    return "<bos><|turn>user\n" + input + "<turn|>\n<|turn>model\n";
  } else {
    if (const auto *cb =
          ModelCallbackRegistry::instance().lookup(architecture)) {
      if (cb->format_prompt) {
        return cb->format_prompt(input);
      }
    }
  }
  return input;
}

static size_t text_generation_model_index(const CausalLmModel &h) {
  // Convention: a multi-model handle is [vision producer, text LLM, ...];
  // text generation runs on the LLM at index 1.
  return (h.models.size() > 1) ? 1 : 0;
}

static void reset_handle_session_state(CausalLmModel &h) { h.kv_len = 0; }

static void update_handle_session_after_run(CausalLmModel &h,
                                            size_t model_index) {
  if (model_index >= h.models.size() || model_index >= h.architectures.size())
    return;
  const auto *cb =
    ModelCallbackRegistry::instance().lookup(h.architectures[model_index]);
  if (!cb || !cb->read_kv_len)
    return;
  h.kv_len = cb->read_kv_len(h.models[model_index].get());
}

static constexpr size_t kAutoTextGenerationModelIndex =
  (std::numeric_limits<size_t>::max)();

#ifdef ENABLE_QNN_MODELS
static causallm::Quick_Dot_AI_QNN *find_qnn_kv_cache_model(CausalLmModel &h) {
  for (auto &m : h.models) {
    auto *q = dynamic_cast<causallm::Quick_Dot_AI_QNN *>(m.get());
    if (q && q->supportsKvCachePersistence())
      return q;
  }
  return nullptr;
}
#endif

static ErrorCode save_qnn_kv_cache_on_handle(CausalLmModel &h,
                                             const char *cache_path) {
#ifndef ENABLE_QNN_MODELS
  (void)h;
  (void)cache_path;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  if (cache_path == nullptr || cache_path[0] == '\0') {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *model = find_qnn_kv_cache_model(h);
  if (model == nullptr) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    model->saveKvCache(cache_path);
    h.kv_len = model->getKvLen();
  } catch (const std::exception &e) {
    LOGE("saveQnnKvCacheHandle failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
#endif
}

static ErrorCode load_qnn_kv_cache_on_handle(CausalLmModel &h,
                                             const char *cache_path) {
#ifndef ENABLE_QNN_MODELS
  (void)h;
  (void)cache_path;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  if (cache_path == nullptr || cache_path[0] == '\0') {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *model = find_qnn_kv_cache_model(h);
  if (model == nullptr) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    model->loadKvCache(cache_path);
    h.kv_len = model->getKvLen();
  } catch (const std::exception &e) {
    LOGE("loadQnnKvCacheHandle failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
#endif
}

static ErrorCode reset_qnn_kv_cache_on_handle(CausalLmModel &h) {
#ifndef ENABLE_QNN_MODELS
  (void)h;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  auto *model = find_qnn_kv_cache_model(h);
  if (model == nullptr) {
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    model->resetKvCache();
    reset_handle_session_state(h);
  } catch (const std::exception &e) {
    LOGE("resetQnnKvCacheHandle failed: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
#endif
}

static std::string prepare_input_for_model(CausalLmModel &h, size_t model_index,
                                           const std::string &input,
                                           bool input_already_formatted) {
  if (model_index >= h.architectures.size() || !g_use_chat_template) {
    return input;
  }

  const std::string &architecture = h.architectures[model_index];
  if (h.kv_len > 0) {
    const auto *cb = ModelCallbackRegistry::instance().lookup(architecture);
    if (cb && cb->incremental_prompt) {
      return cb->incremental_prompt(input);
    }
  }

  if (input_already_formatted) {
    return input;
  }

  return apply_chat_template(architecture, input);
}

static std::string get_quantization_suffix(ModelQuantizationType type) {
  return "";
  switch (type) {
  case CAUSAL_LM_QUANTIZATION_W4A32:
    return "-w4a32";
  case CAUSAL_LM_QUANTIZATION_W16A16:
    return "-w16a16";
  case CAUSAL_LM_QUANTIZATION_W8A16:
    return "-w8a16";
  case CAUSAL_LM_QUANTIZATION_W32A32:
    return "-w32a32";
  default: // W4A32 by default
    return "-w4a32";
  }
}

static std::string resolve_model_path(const std::string &model_key,
                                      ModelQuantizationType quant_type) {
  std::string path_upper = model_key;
  std::transform(path_upper.begin(), path_upper.end(), path_upper.begin(),
                 ::toupper);

  std::string base_dir_name = "";

  // 1. Try to find base directory name from map
  if (g_model_path_map.find(path_upper) != g_model_path_map.end()) {
    base_dir_name = g_model_path_map[path_upper];
  } else {
    // Fallback: use lowercased key as base dir name if not found in map
    // or just return empty? For restricted API, we should probably fail
    // earlier, but here we can return constructed path.
    base_dir_name = path_upper;
    std::transform(base_dir_name.begin(), base_dir_name.end(),
                   base_dir_name.begin(), ::tolower);
  }

  std::string model_path =
    "/" + base_dir_name + get_quantization_suffix(quant_type);

  return model_path;
}

/**
 * @brief Rebase path-like keys of a sub-model nntr_config.json onto @p sub_dir.
 *
 * Called once per sub-model inside the multi-model branch of
 * load_into_handle so that downstream code (Factory::create, load_weight)
 * sees absolute paths — mirrors the inline fixups the single-model path
 * already performs for model_file_name / binary_config_path / ...
 *
 * Absolute values (leading '/') are left untouched so the caller can
 * override a specific file with a system-wide path if they want.
 */
static bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

static std::string rebase_path(const std::string &path,
                               const std::string &base_dir) {
  if (path.empty() || is_absolute_path(path))
    return path;
  return base_dir + "/" + path;
}

static void fix_paths(json &nntr_cfg, const std::string &sub_dir) {
  static const char *kKeys[] = {
    "tokenizer_file",       "model_file_name",     "binary_config_path",
    "image_newline_path",   "embedding_file_name", "ple_file_name",
    "rotation_matrix_path",
  };
  for (const char *k : kKeys) {
    if (!nntr_cfg.contains(k) || !nntr_cfg[k].is_string())
      continue;
    std::string v = nntr_cfg[k].get<std::string>();
    nntr_cfg[k] = rebase_path(v, sub_dir);
  }
}

static bool check_file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static void validate_models() {
  LOGD("[DEBUG] Validating model files...");
  // Iterate over all known model names in map
  for (auto const &[key, val] : g_model_path_map) {
    // We want to check for each Quantization Type if it exists
    // List of quant types to check: UNKNOWN (default), W4A32, W16A16, W32A32
    std::vector<ModelQuantizationType> quant_types = {
      CAUSAL_LM_QUANTIZATION_UNKNOWN, CAUSAL_LM_QUANTIZATION_W4A32,
      CAUSAL_LM_QUANTIZATION_W16A16, CAUSAL_LM_QUANTIZATION_W32A32};

    for (auto qt : quant_types) {
      std::string quant_suffix = get_quantization_suffix(qt);

      std::string lookup_key = key;
      if (qt != CAUSAL_LM_QUANTIZATION_UNKNOWN) {
        std::transform(quant_suffix.begin(), quant_suffix.end(),
                       quant_suffix.begin(), ::toupper); // "-W4A32"
        lookup_key += quant_suffix;
      }

      // Resolve path for this combination
      std::string resolved_path = "./models" + resolve_model_path(key, qt);

      if (g_model_registry.find(lookup_key) != g_model_registry.end()) {
        // CASE 1: Configuration is registered in model_config.cpp
        // For these models, we only check if the binary weight file exists.
        // The configurations (config.json, etc.) are embedded in the library.
        RegisteredModel &rm = g_model_registry[lookup_key];
        std::string bin_file_name = rm.config.model_file_name;
        std::string full_path = resolved_path + "/" + bin_file_name;

        if (check_file_exists(full_path)) {
          LOGD("  [OK] Reg Config: %s -> %s", lookup_key.c_str(),
               full_path.c_str());
        } else {
          LOGD("  [FAIL] Reg Config: %s -> Missing binary: %s",
               lookup_key.c_str(), full_path.c_str());
        }
      } else {
        // CASE 2: No internal config, but model type exists (via map
        // iteration). For these models, we require external configuration files
        // (config.json, nntr_config.json) to be present in the directory.
        if (check_file_exists(resolved_path)) {
          bool has_config = check_file_exists(resolved_path + "/config.json");
          bool has_nntr =
            check_file_exists(resolved_path + "/nntr_config.json");

          if (has_config && has_nntr) {
            LOGD("  [OK] External Config: %s -> %s", lookup_key.c_str(),
                 resolved_path.c_str());
            // Optional: Parse nntr_config to check bin
            try {
              json nntr =
                causallm::LoadJsonFile(resolved_path + "/nntr_config.json");
              if (nntr.contains("model_file_name")) {
                std::string bin = nntr["model_file_name"];
                if (check_file_exists(resolved_path + "/" + bin)) {
                  LOGD("       (Binary confirmed: %s)", bin.c_str());
                } else {
                  LOGD("       (MISSING BINARY: %s)", bin.c_str());
                }
              }
            } catch (...) {
            }
          } else {
            LOGD("  [FAIL] External Config: %s -> Missing configs in %s",
                 lookup_key.c_str(), resolved_path.c_str());
          }
        }
      }
    }
  }
}

ErrorCode setOptions(Config config) {
  // Currently no options are being handled
  g_use_chat_template = config.use_chat_template;
  g_verbose = config.verbose;
  g_chat_template_name = (config.chat_template_name != nullptr)
                           ? config.chat_template_name
                           : "default";
  if (config.debug_mode) {
    // Ensure models are registered so we can validate them
    register_models();
    validate_models();
  }
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode loadToolset(const char *toolset_path,
                      tokenizers::Tokenizer *tokenizer,
                      unsigned int vocab_size) {
  if (toolset_path == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  if (tokenizer == nullptr) {
    std::cerr << "Error: Tokenizer is null" << std::endl;
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  LOGD("[LoadToolset] load toolset path: %s", toolset_path);

  try {
    // Load and pre-compile all tool grammars
    bool success = causallm::XGrammarManager::Instance().loadToolset(
      std::string(toolset_path), tokenizer, vocab_size);
    LOGD("causallm::XGrammarManager::loadToolset() done");
    if (!success) {
      return CAUSAL_LM_ERROR_UNKNOWN;
    }
  } catch (const std::exception &e) {
    std::cerr << "Exception in loadToolset: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config) {
  if (arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config) {
  if (model_name == nullptr || arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);

  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);

  g_model_registry[name] = {aname, config};
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core loader shared by loadModel and loadModelHandle.
 *
 * Populates the given handle's model / architecture / init-duration
 * vectors on success. Takes the handle's own mutex so two concurrent
 * loads on the same handle are serialized, while loads on different
 * handles run in parallel. A separate registry mutex protects
 * g_model_registry / g_arch_config_map during lookup.
 *
 * Dispatch in CASE 2 (file-based):
 *   - If the top-level nntr_config.json has both "architectures" (string
 *     array) and "model_dirs" (string array) of equal non-zero length,
 *     loads one sub-model per entry (e.g. vision encoder + LLM).
 *   - Otherwise loads a single model from the resolved directory using
 *     the pre-existing flow.
 */
#ifdef ENABLE_QNN_MODELS
// Point the QNN HTP backend at htp_backend_ext_config.json for ANY QNN arch.
// Must run BEFORE constructing/initializing a QNN model — single OR multi-model
// sub-models (e.g. the vision encoder of a multimodal pair) — otherwise
// QNNContext falls back to the process cwd ("/" for an installed APK) and QNN
// layer registration throws not_supported ("Unable to load backend extensions
// config"). Honors an externally configured path if already set.
static void ensure_qnn_backend_ext_config(const std::string &base_dir) {
  const char *configured = getenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH");
  if (configured != nullptr && configured[0] != '\0')
    return;
  std::string config_path = base_dir;
  // Strip a trailing "/models" so the config resolves next to the
  // model-collection root (mirrors the single-model resolution).
  while (!config_path.empty() && config_path.back() == '/')
    config_path.pop_back();
  if (config_path.length() >= 7 &&
      config_path.substr(config_path.length() - 7) == "/models") {
    config_path = config_path.substr(0, config_path.length() - 7);
  }
  config_path += "/htp_backend_ext_config.json";
  setenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH", config_path.c_str(), 1);
}
#endif

/** Internal overload: config_name already resolved (T4 byName path). */
static ErrorCode load_into_handle(CausalLmModel &h, BackendType compute,
                                  const char *target_model_name,
                                  ModelQuantizationType quant_type,
                                  const char *native_lib_dir,
                                  const char *model_base_path) {
  LOGD("[DEBUG] load_into_handle: START");
  LOGD("[DEBUG]   compute: %d", compute);
  LOGD("[DEBUG]   target_model_name: %s",
       target_model_name ? target_model_name : "(null)");
  LOGD("[DEBUG]   quant_type: %d", quant_type);

  auto start_init = std::chrono::high_resolution_clock::now();

  // Ensure models/configs are registered (thread-safe via call_once)
  LOGD("[DEBUG] load_into_handle: Calling register_models...");
  register_models();
  LOGD("[DEBUG] load_into_handle: register_models done");

  std::lock_guard<std::mutex> lock(h.mtx);
  try {
    h.models.clear();
    h.architectures.clear();
    h.model_dirs.clear();
    h.initialization_duration_ms.clear();
    h.initialized = false;
    reset_handle_session_state(h);

    // Check if it's a registered in-memory config
    std::string input_name = std::string(target_model_name);
    std::string input_name_upper = input_name;
    std::transform(input_name_upper.begin(), input_name_upper.end(),
                   input_name_upper.begin(), ::toupper);
    LOGD("[DEBUG] load_into_handle: input_name = %s", input_name.c_str());

    std::string quant_suffix = "";
    switch (quant_type) {
    case CAUSAL_LM_QUANTIZATION_W4A32:
      quant_suffix = "-W4A32";
      break;
    case CAUSAL_LM_QUANTIZATION_W16A16:
      quant_suffix = "-W16A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W8A16:
      quant_suffix = "-W8A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W32A32:
      quant_suffix = "-W32A32";
      break;
    default:
      break;
    }
    std::string lookup_name = input_name_upper + quant_suffix;
    LOGD("[DEBUG] load_into_handle: lookup_name = %s", lookup_name.c_str());

    json cfg;
    json generation_cfg;
    json nntr_cfg;
    std::string model_dir_path;
    std::string abs_model_dir;
    std::string base_dir =
      (model_base_path != nullptr && strlen(model_base_path) > 0)
        ? model_base_path
        : "/sdcard/Download/aistudio-mobile/models/";

#ifdef ENABLE_QNN_MODELS
    // Set the QNN backend-extensions config path up front so it is in effect
    // for BOTH the multi-model sub-model loop and the single-model path.
    ensure_qnn_backend_ext_config(base_dir);
#endif

    // Snapshot registry entries under the registry mutex so concurrent
    // loads on different handles don't race with each other (or with
    // registerModel / registerModelArchitecture).
    std::lock_guard<std::mutex> reg_lock(g_registry_mutex);

    // Check in-memory map first
    // if (g_model_registry.find(lookup_name) != g_model_registry.end()) {

    // always goto case2
    if (0) {
      LOGD("[DEBUG] load_into_handle: CASE 1 - Internal config found for %s",
           lookup_name.c_str());
      // ------------------------------------------------------------------------
      // CASE 1: Model Configuration is Internal (Registered in
      // model_config.cpp)
      // ------------------------------------------------------------------------
      // In this case, we do NOT load config.json or nntr_config.json from disk.
      // We only locate the binary weight file.
      RegisteredModel &rm = g_model_registry[lookup_name];

      // Find architecture config
      if (g_arch_config_map.find(rm.arch_name) == g_arch_config_map.end()) {
        LOGE("[DEBUG] load_into_handle: Architecture '%s' not found for model "
             "'%s'",
             rm.arch_name.c_str(), lookup_name.c_str());
        return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      }
      LOGD("[DEBUG] load_into_handle: arch_name = %s", rm.arch_name.c_str());
      ModelArchConfig &ac = g_arch_config_map[rm.arch_name];
      ModelRuntimeConfig &rc = rm.config;

      // Strategy: Resolve path to find the weight file
      model_dir_path =
        "./models" + resolve_model_path(target_model_name, quant_type);
      LOGD("[DEBUG] load_into_handle: model_dir_path = %s",
           model_dir_path.c_str());

      // Populate JSONs from Arch Struct
      cfg["vocab_size"] = ac.vocab_size;
      cfg["hidden_size"] = ac.hidden_size;
      cfg["intermediate_size"] = ac.intermediate_size;
      cfg["num_hidden_layers"] = ac.num_hidden_layers;
      cfg["num_attention_heads"] = ac.num_attention_heads;
      cfg["head_dim"] = ac.head_dim;
      cfg["num_key_value_heads"] = ac.num_key_value_heads > 0
                                     ? ac.num_key_value_heads
                                     : ac.num_attention_heads;
      cfg["max_position_embeddings"] = ac.max_position_embeddings;
      cfg["rope_theta"] = ac.rope_theta;
      cfg["rms_norm_eps"] = ac.rms_norm_eps;
      cfg["tie_word_embeddings"] = ac.tie_word_embeddings;
      if (ac.sliding_window != UINT_MAX) {
        cfg["sliding_window"] = ac.sliding_window;
      } else {
        cfg["sliding_window"] = nullptr;
      }
      cfg["sliding_window_pattern"] = ac.sliding_window_pattern;
      cfg["architectures"] = {std::string(ac.architecture)};

      if (ac.num_eos_token_ids > 0) {
        std::vector<unsigned int> eos_ids;
        for (unsigned int i = 0; i < ac.num_eos_token_ids; ++i)
          eos_ids.push_back(ac.eos_token_ids[i]);
        generation_cfg["eos_token_id"] = eos_ids;
      }
      generation_cfg["bos_token_id"] = ac.bos_token_id;

      // Populate JSONs from Runtime Struct
      generation_cfg["top_k"] = rc.top_k;
      generation_cfg["top_p"] = rc.top_p;
      generation_cfg["temperature"] = rc.temperature;
      generation_cfg["do_sample"] = false;

      nntr_cfg["batch_size"] = rc.batch_size;
      nntr_cfg["model_type"] = std::string(rc.model_type);
      nntr_cfg["model_tensor_type"] = std::string(rc.model_tensor_type);
      nntr_cfg["init_seq_len"] = rc.init_seq_len;
      nntr_cfg["max_seq_len"] = rc.max_seq_len;
      nntr_cfg["num_to_generate"] = rc.num_to_generate;
      nntr_cfg["fsu"] = rc.fsu;
      nntr_cfg["fsu_lookahead"] = rc.fsu_lookahead;
      nntr_cfg["embedding_dtype"] = std::string(rc.embedding_dtype);
      nntr_cfg["fc_layer_dtype"] = std::string(rc.fc_layer_dtype);
      nntr_cfg["model_file_name"] = std::string(rc.model_file_name);

      // tokenizer_file path is set later from abs_model_dir in the shared
      // post-processing block below.
      (void)rc.tokenizer_file;

      if (strlen(rc.lmhead_dtype) > 0) {
        nntr_cfg["lmhead_dtype"] = std::string(rc.lmhead_dtype);
      }

      std::vector<unsigned int> bad_ids;
      for (unsigned int i = 0; i < rc.num_bad_word_ids; ++i)
        bad_ids.push_back(rc.bad_word_ids[i]);
      nntr_cfg["bad_word_ids"] = bad_ids;
    } else {
      LOGD("[DEBUG] load_into_handle: CASE 2 - External config (file-based)");
      // --------------------------------------------------
      // CASE 2: External Model Configuration (File-based)
      // --------------------------------------------------
      // The model type is registered (enum), but specific configuration for
      // this quantization is not in memory. We must load config.json and
      // nntr_config.json from the model directory
      model_dir_path = resolve_model_path(target_model_name, quant_type);
      LOGD("[DEBUG] load_into_handle: model_dir_path = %s",
           model_dir_path.c_str());

      abs_model_dir = base_dir + model_dir_path;
      LOGD("[DEBUG] load_into_handle: abs_model_dir = %s",
           abs_model_dir.c_str());

      // Top-level nntr_config.json is read once and used for both
      //   (a) multi-model dispatch (architectures[] + model_dirs[]), and
      //   (b) the single-model fallback below.
      json top_nntr =
        causallm::LoadJsonFile(abs_model_dir + "/nntr_config.json");

      LOGD("[DEBUG] load_into_handle: abs_model_dir = %s",
           abs_model_dir.c_str());

      LOGD("[DEBUG] load_into_handle: top_nntr = %s",
           (abs_model_dir + "/nntr_config.json").c_str());

      const bool is_multi =
        top_nntr.contains("architectures") &&
        top_nntr["architectures"].is_array() &&
        top_nntr.contains("model_dirs") && top_nntr["model_dirs"].is_array() &&
        !top_nntr["architectures"].empty() &&
        top_nntr["architectures"].size() == top_nntr["model_dirs"].size();

      if (top_nntr.contains("use_chat_template")) {
        g_use_chat_template = top_nntr["use_chat_template"].get<bool>();
      }

      LOGD("[DEBUG] load_into_handle: abs_model_dir = %d %d %d %d %zu %zu",
           top_nntr.contains("architectures"),
           top_nntr["architectures"].is_array(),
           top_nntr.contains("model_dirs"), top_nntr["model_dirs"].is_array(),
           top_nntr["architectures"].size(), top_nntr["model_dirs"].size());

      if (is_multi) {
        // ----------------------------------------------------------------
        // Multi-model branch.
        //
        //   top_nntr_config.json:
        //     { "architectures": ["ArchA", "ArchB"],
        //       "model_dirs":   ["sub_a",  "sub_b"] }
        //
        // Each sub_dir = abs_model_dir + "/" + model_dirs[i] owns its own
        // config.json / generation_config.json / nntr_config.json +
        // weights. The top-level architectures[i] wins over any
        // "architectures" entry inside sub-config — one source of truth.
        // ----------------------------------------------------------------
        auto archs = top_nntr["architectures"].get<std::vector<std::string>>();
        auto dirs = top_nntr["model_dirs"].get<std::vector<std::string>>();
        LOGD("[DEBUG] load_into_handle: MULTI-MODEL spec (N=%zu)",
             archs.size());

        for (size_t i = 0; i < archs.size(); ++i) {
          const std::string &arch_i = archs[i];
          const std::string sub_dir = abs_model_dir + "/" + dirs[i];
          LOGD("[DEBUG]   [%zu] arch=%s dir=%s", i, arch_i.c_str(),
               sub_dir.c_str());

          json sub_cfg = causallm::LoadJsonFile(sub_dir + "/config.json");

          json sub_gen;
          if (check_file_exists(sub_dir + "/generation_config.json")) {
            sub_gen =
              causallm::LoadJsonFile(sub_dir + "/generation_config.json");
          } else {
            sub_gen = json::object();
          }

          json sub_nntr = causallm::LoadJsonFile(sub_dir + "/nntr_config.json");

          fix_paths(sub_nntr, sub_dir);

          // Optional per-sub-model overrides from the top-level config.
          // Lets callers flip flags like uses_embedding / add keys like
          // embedding_file_name without duplicating the sub-model's own
          // nntr_config.json. fix_paths is run again so any newly
          // introduced path-like key (e.g. embedding_file_name) is
          // resolved relative to sub_dir just like the native keys.
          if (top_nntr.contains("model_options") &&
              top_nntr["model_options"].is_array() &&
              i < top_nntr["model_options"].size() &&
              top_nntr["model_options"][i].is_object()) {
            for (auto it = top_nntr["model_options"][i].begin();
                 it != top_nntr["model_options"][i].end(); ++it) {
              sub_nntr[it.key()] = it.value();
              LOGD("[DEBUG]   override sub[%zu] %s", i, it.key().c_str());
            }
            fix_paths(sub_nntr, sub_dir);
          }
          if (sub_nntr.contains("lora_path")) {
            LOGD("lora_path : %s",
                 sub_nntr["lora_path"].get<std::string>().c_str());
            std::string lora_path =
              sub_dir + "/" + sub_nntr["lora_path"].get<std::string>();
            sub_nntr["lora_path"] = lora_path;
            LOGD("lora_path is now %s", lora_path.c_str());
          }

          auto m = causallm::Factory::Instance().create(arch_i, sub_cfg,
                                                        sub_gen, sub_nntr);
          if (!m) {
            LOGE("[DEBUG] load_into_handle: Factory::create returned nullptr "
                 "for sub-model %zu (arch=%s)",
                 i, arch_i.c_str());
            return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
          }

          auto sub_t0 = std::chrono::high_resolution_clock::now();
          if (native_lib_dir != nullptr && strlen(native_lib_dir) > 0) {
            setenv("ADSP_LIBRARY_PATH", native_lib_dir, 1);
          }
          m->initialize();

          std::string weight_file =
            sub_nntr.contains("model_file_name")
              ? sub_nntr["model_file_name"].get<std::string>()
              : (sub_dir + "/pytorch_model.bin");
          m->load_weight(weight_file);
          auto sub_t1 = std::chrono::high_resolution_clock::now();
          double sub_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            sub_t1 - sub_t0)
                            .count();

          h.models.push_back(std::move(m));
          h.architectures.push_back(arch_i);
          h.model_dirs.push_back(sub_dir);
          h.initialization_duration_ms.push_back(sub_ms);
          LOGD("[DEBUG]   [%zu] loaded (%.1f ms)", i, sub_ms);

          // Load chat template from model directory if available.
          if (causallm::ChatTemplate::Exists(sub_dir)) {
            try {
              g_chat_template = causallm::ChatTemplate::Load(sub_dir);
              std::cout << "[Info] Chat template loaded from " << sub_dir
                        << std::endl;
            } catch (const std::exception &e) {
              std::cerr << "[Warning] Chat template load failed: " << e.what()
                        << ". Falling back to hardcoded templates."
                        << std::endl;
              g_chat_template.reset();
            }
          }
        }

        if (native_lib_dir != nullptr)
          h.native_lib_dir = native_lib_dir;
        h.initialized = true;

        auto finish_init = std::chrono::high_resolution_clock::now();
        auto e2e = std::chrono::duration_cast<std::chrono::milliseconds>(
                     finish_init - start_init)
                     .count();
        LOGD("[DEBUG] load_into_handle: MULTI-MODEL SUCCESS "
             "(%zu models, %lld ms e2e)",
             h.models.size(), (long long)e2e);
        return CAUSAL_LM_ERROR_NONE;
      }

      // -------------------- single-model fallback --------------------
      LOGD("single cfg : %s", (abs_model_dir + "/config.json").c_str());
      cfg = causallm::LoadJsonFile(abs_model_dir + "/config.json");

      if (check_file_exists(abs_model_dir + "/generation_config.json")) {
        generation_cfg =
          causallm::LoadJsonFile(abs_model_dir + "/generation_config.json");
      }

      nntr_cfg = std::move(top_nntr);

      if (nntr_cfg.contains("lora_path")) {
        nntr_cfg["lora_path"] = "";
      }

      LOGD("single tokenizer : %s",
           (abs_model_dir + "/tokenizer.json").c_str());

      if (nntr_cfg.contains("tokenizer_file")) {
        nntr_cfg["tokenizer_file"] = abs_model_dir + "/tokenizer.json";
      }
    }

    // Load chat template from model directory if available.
    if (causallm::ChatTemplate::Exists(abs_model_dir)) {
      try {
        g_chat_template = causallm::ChatTemplate::Load(abs_model_dir);
        LOGD("[Info] Chat template loaded from %s", abs_model_dir.c_str());
      } catch (const std::exception &e) {
        LOGE("[Warning] Chat template load failed: %s. Falling back to "
             "hardcoded templates.",
             e.what());
        g_chat_template.reset();
      }
    } else {
      g_chat_template.reset();
      LOGE("[Warning] No chat template found in %s. Using hardcoded chat "
           "templates.",
           abs_model_dir.c_str());
    }

    // Construct weight file path
    std::string weight_file_name;
    if (nntr_cfg.contains("model_file_name")) {
      weight_file_name = nntr_cfg["model_file_name"].get<std::string>();
    } else {
      weight_file_name = "pytorch_model.bin";
    }

    const std::string weight_file =
      rebase_path(weight_file_name, abs_model_dir);
    LOGD("[DEBUG] load_into_handle: weight_file = %s", weight_file.c_str());
    std::cout << "-------------------" << abs_model_dir << "/" << std::endl;

    nntr_cfg["model_file_name"] = weight_file;
    if (nntr_cfg.contains("binary_config_path")) {
      std::string str = nntr_cfg["binary_config_path"].get<std::string>();
      nntr_cfg["binary_config_path"] = rebase_path(str, abs_model_dir);
      LOGD("[DEBUG] bianry config data: file = %s",
           nntr_cfg["binary_config_path"].get<std::string>().c_str());
    }
    if (nntr_cfg.contains("image_newline_path")) {
      std::string str = nntr_cfg["image_newline_path"].get<std::string>();
      nntr_cfg["image_newline_path"] = rebase_path(str, abs_model_dir);
      LOGD("[DEBUG] new line config data: file = %s",
           nntr_cfg["image_newline_path"].get<std::string>().c_str());
    }
    if (nntr_cfg.contains("embedding_file_name")) {
      std::string str = nntr_cfg["embedding_file_name"].get<std::string>();
      nntr_cfg["embedding_file_name"] = rebase_path(str, abs_model_dir);
    }
    if (nntr_cfg.contains("ple_file_name")) {
      std::string str = nntr_cfg["ple_file_name"].get<std::string>();
      nntr_cfg["ple_file_name"] = rebase_path(str, abs_model_dir);
    }

    LOGD("[DEBUG] -------------------------- asdfasdfasdfasdfasdfasdf ");

    // Determine architecture from config or ModelType
    // Priority: Config file architecture > ModelType mapping (fallback)
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else {
      // No fallback mapping from specific ModelType instances to generic
      // architecture strings for now, as specific types should have config or
      // be loaded from valid file with config.json
      LOGE("[DEBUG] load_into_handle: No architecture found in config");
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
    LOGD("[DEBUG] load_into_handle: architecture = %s", architecture.c_str());

    LOGD("[DEBUG] load_into_handle: Creating model via Factory...%s ",
         architecture.c_str());

    auto m = causallm::Factory::Instance().create(architecture, cfg,
                                                  generation_cfg, nntr_cfg);
    if (!m) {
      LOGE("[DEBUG] load_into_handle: Factory::create returned nullptr");
      return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
    }
    LOGD("[DEBUG] load_into_handle: Model created successfully");

    if (native_lib_dir != nullptr)
      h.native_lib_dir = native_lib_dir;

#ifdef ENABLE_QNN_MODELS
    // Point the QNN HTP backend at htp_backend_ext_config.json. This must run
    // for ANY QNN-backed architecture (e.g. Gemma4_E2B_QNN), not only
    // callback-registered ones: otherwise QNNContext falls back to the process
    // cwd — which is "/" for an installed APK — and QNN layer registration
    // throws not_supported ("Unable to load backend extensions config").
    // Honor an externally configured path if one is already set.
    ensure_qnn_backend_ext_config(base_dir);
#endif

    LOGD("[DEBUG] load_into_handle: Calling model->initialize()...");
    if (native_lib_dir != nullptr && strlen(native_lib_dir) > 0) {
      setenv("ADSP_LIBRARY_PATH", native_lib_dir, 1);
    }
    m->initialize();
    LOGD("[DEBUG] load_into_handle: model->initialize() done");

    LOGD("[DEBUG] load_into_handle: Calling model->load_weight()...");
    // Multi-component models (e.g. Lfm2VLVJepa21BModel = ViT + projector +
    // LFM2) have no single model_file_name; they resolve each component's
    // weight file relative to the model directory, so hand them the directory
    // instead.
    if (architecture == "Lfm2VLVJepa21BModel") {
      m->load_weight(abs_model_dir);
    } else {
      m->load_weight(weight_file);
    }
    LOGD("[DEBUG] load_into_handle: model->load_weight() done");

    auto finish_init = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_init - start_init);

    h.models.push_back(std::move(m));
    h.architectures.push_back(architecture);
    h.model_dirs.push_back(abs_model_dir);
    h.initialization_duration_ms.push_back(
      static_cast<double>(init_duration.count()));
    h.initialized = true;

    // XGrammarManager Initalize
    auto *tokenizer = h.models[0]->getTokenizer();
    unsigned int vocab_size = h.models[0]->getVocabSize();
    causallm::XGrammarManager::Instance().initialize(tokenizer, vocab_size);

    // XGrammarManager Toolset Load
    std::string default_toolset_path = abs_model_dir + "/Toolset.json";
    bool toolset_file_exists = check_file_exists(default_toolset_path);
    if (toolset_file_exists) {
      loadToolset(default_toolset_path.c_str(), tokenizer, vocab_size);
    }

    LOGD("[DEBUG] load_into_handle: SINGLE SUCCESS (init took %lld ms)",
         (long long)init_duration.count());
  } catch (...) {
    // RTTI may not match across shared libraries — query the current
    // exception's typeinfo directly via the Itanium ABI hook. This
    // works even when catching by concrete types fails due to typeinfo
    // duplication between libnntrainer.so and libquick_dot_ai_api.so.
    const std::type_info *ti = abi::__cxa_current_exception_type();
    const char *raw = ti ? ti->name() : "(null)";
    int status = 0;
    char *demangled = (ti != nullptr)
                        ? abi::__cxa_demangle(raw, nullptr, nullptr, &status)
                        : nullptr;
    LOGE("[DEBUG] load_into_handle: unknown exception, type=%s",
         demangled ? demangled : raw);
    std::free(demangled);

    // Also try once more via rethrow — in case std::exception RTTI does
    // match from this catch-site (we already tried above but leaving
    // this as a second chance is cheap).
    try {
      throw;
    } catch (const std::exception &e) {
      LOGE("[DEBUG] load_into_handle: rethrown std::exception what()=%s",
           e.what());
    } catch (...) {
      LOGE("[DEBUG] load_into_handle: rethrown still non-std");
    }
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }
  LOGD("[DEBUG] load_into_handle: END (returning CAUSAL_LM_ERROR_NONE)");
  return CAUSAL_LM_ERROR_NONE;
}

/** ModelType overload: translates enum → config_name then delegates. */
static ErrorCode load_into_handle(CausalLmModel &h, BackendType compute,
                                  ModelType modeltype,
                                  ModelQuantizationType quant_type,
                                  const char *native_lib_dir,
                                  const char *model_base_path) {
  const char *target_model_name = get_model_name_from_type(modeltype);
  if (!target_model_name) {
    LOGE("[DEBUG] load_into_handle: Invalid modeltype %d", modeltype);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  LOGD("[DEBUG] load_into_handle: target_model_name = %s %d", target_model_name,
       modeltype);
  return load_into_handle(h, compute, target_model_name, quant_type,
                          native_lib_dir, model_base_path);
}

/**
 * @brief Core runner shared by runModelHandleWithMessages.
 */
static ErrorCode
run_on_handle(CausalLmModel &h, const char *inputTextPrompt,
              const char **outputText, bool input_already_formatted = false,
              size_t model_index = kAutoTextGenerationModelIndex,
              causallm::XGrammar *tool_grammar = nullptr) {
  if (inputTextPrompt == nullptr || outputText == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  if (model_index == kAutoTextGenerationModelIndex)
    model_index = text_generation_model_index(h);

  if (!h.initialized || model_index >= h.models.size() ||
      !h.models[model_index]) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    auto *model = h.models[model_index].get();
    if (!model_supports_text_output(model)) {
      LOGE("run_on_handle: model[%zu] does not expose text output",
           model_index);
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    }

    std::unique_ptr<XGrammarLogitsProcessor> grammar_processor;

    struct ScopedRunProcessor {
      causallm::Transformer *model = nullptr;
      bool detach_logits = false;
#ifdef ENABLE_QNN_MODELS
      causallm::Quick_Dot_AI_QNN *qnn_model = nullptr;
#endif

      ~ScopedRunProcessor() {
        if (detach_logits && model != nullptr)
          model->setLogitsProcessor(nullptr);
#ifdef ENABLE_QNN_MODELS
        if (qnn_model != nullptr)
          qnn_model->resetXGrammar();
#endif
      }
    } scoped_processor{model};

    bool qnn_grammar_attached = false;
#ifdef ENABLE_QNN_MODELS
    if (tool_grammar != nullptr) {
      if (auto *qnn_model = as_qnn_model(model)) {
        qnn_model->setXGrammar(tool_grammar);
        scoped_processor.qnn_model = qnn_model;
        qnn_grammar_attached = true;
      }
    }
#endif
    if (tool_grammar != nullptr && !qnn_grammar_attached) {
      grammar_processor = std::make_unique<XGrammarLogitsProcessor>(
        tool_grammar, [model]() { request_model_stop(model); });
      model->setLogitsProcessor(grammar_processor.get());
      scoped_processor.detach_logits = true;
    }

    std::string input = prepare_input_for_model(
      h, model_index, std::string(inputTextPrompt), input_already_formatted);

// We assume single batch request for this API
#if defined(_WIN32)
    model->run(std::wstring(input.begin(), input.end()), false, L"", L"",
               g_verbose);
#else
    model->run(input, false, "", "", g_verbose);
#endif

    if (!get_model_output(model, h.last_output))
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    *outputText = h.last_output.c_str();
    update_handle_session_after_run(h, model_index);
  } catch (const std::exception &e) {
    LOGE("Exception in run_on_handle: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Core metrics fetcher shared by getPerformanceMetrics and its
 *        handle-based counterpart.
 *
 * Reports models[0] runtime metrics. initialization_duration_ms is the
 * sum over all sub-models this handle owns.
 */
static ErrorCode metrics_on_handle(CausalLmModel &h,
                                   PerformanceMetrics *metrics) {
  if (metrics == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  std::lock_guard<std::mutex> lock(h.mtx);
  size_t metrics_model_idx = text_generation_model_index(h);
  if (!h.initialized || h.models.size() <= metrics_model_idx ||
      !h.models[metrics_model_idx]) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    auto *model = h.models[metrics_model_idx].get();
    if (!model->hasRun()) {
      return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;
    }
    auto im = model->getPerformanceMetrics();
    metrics->prefill_tokens = im.prefill_tokens;
    metrics->prefill_duration_ms = im.prefill_duration_ms;
    metrics->generation_tokens = im.generation_tokens;
    metrics->generation_duration_ms = im.generation_duration_ms;
    metrics->total_duration_ms = im.total_duration_ms;
    metrics->peak_memory_kb = im.peak_memory_kb;

    double total_init = 0.0;
    for (double d : h.initialization_duration_ms)
      total_init += d;
    metrics->initialization_duration_ms = total_init;
  } catch (const std::exception &e) {
    LOGE("Exception in getPerformanceMetrics: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

/*****************************************************************************
 * Chat Template API - role + content message support
 *****************************************************************************/

// Internal ChatMessage struct for API use
struct ChatMessage {
  std::string role;
  std::string content;
};

static std::vector<ChatMessage>
convertMessages(const CausalLMChatMessage *messages, size_t num_messages) {
  std::vector<ChatMessage> result;
  result.reserve(num_messages);
  for (size_t i = 0; i < num_messages; ++i) {
    ChatMessage msg;
    msg.role = messages[i].role ? messages[i].role : "";
    msg.content = messages[i].content ? messages[i].content : "";
    result.push_back(std::move(msg));
  }
  return result;
}

/**
 * @brief Apply chat template to messages with hardcoded fallback
 *
 * @param model_dir Optional model directory to load tokenizer_config.json
 *        from if g_chat_template is not already loaded. This ensures
 *        registered models (and any other model) use their tokenizer's
 *        chat template when available.
 */
static std::string apply_chat_template_messages(
  const std::string &architecture, const std::vector<ChatMessage> &messages,
  bool add_generation_prompt, const std::string &model_dir = "") {
  // If g_chat_template is not loaded but a model_dir is provided,
  // try loading tokenizer_config.json from that directory at run time.
  if (!g_chat_template && !model_dir.empty()) {
    std::string tc_path = model_dir + "/tokenizer_config.json";
    if (check_file_exists(tc_path)) {
      try {
        g_chat_template = causallm::ChatTemplate::Load(model_dir);
        if (g_chat_template) {
          LOGD("[Info] Chat template loaded on-demand from %s",
               model_dir.c_str());
        } else {
          LOGE("[Warning] tokenizer_config.json found in %s but could not be "
               "loaded.",
               model_dir.c_str());
        }
      } catch (const std::exception &e) {
        LOGE("[Warning] Failed to load chat template from %s: %s",
             model_dir.c_str(), e.what());
      }
    } else {
      LOGE("[Warning] tokenizer_config.json not found in %s",
           model_dir.c_str());
    }
  }

  // Use Enhanced Chat Template if available
  if (g_chat_template) {
    nlohmann::json request;
    request["messages"] = nlohmann::json::array();
    for (const auto &msg : messages) {
      request["messages"].push_back(
        {{"role", msg.role}, {"content", msg.content}});
    }
    request["add_generation_prompt"] = add_generation_prompt;

    try {
      return g_chat_template->apply(request);
    } catch (const std::exception &e) {
      LOGE("Chat template apply failed: %s", e.what());
      // fallback to hardcoded
    }
  }

  LOGD("APPLYING HARD CODED FALLBACK");
  std::string result;

  if (architecture == "LlamaForCausalLM") {
    for (const auto &msg : messages) {
      if (msg.role == "system") {
        result += "<<SYS>>\n" + msg.content + "\n<</SYS>>\n\n";
      } else if (msg.role == "user") {
        result += "[INST] " + msg.content + " [/INST]";
      } else if (msg.role == "assistant") {
        result += msg.content + "\n";
      }
    }
  } else if (architecture == "Qwen2ForCausalLM" ||
             architecture == "Qwen3ForCausalLM" ||
             architecture == "Qwen3MoeForCausalLM" ||
             architecture == "Qwen3SlimMoeForCausalLM" ||
             architecture == "Qwen3CachedSlimMoeForCausalLM") {
    for (const auto &msg : messages) {
      result += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    if (add_generation_prompt) {
      result += "<|im_start|>assistant\n";
    }
  } else if (architecture == "Gemma3ForCausalLM") {
    for (const auto &msg : messages) {
      if (msg.role == "user") {
        result += "<start_of_turn>user\n" + msg.content + "<end_of_turn>\n";
      } else if (msg.role == "assistant") {
        result += "<start_of_turn>model\n" + msg.content + "<end_of_turn>\n";
      }
    }
    if (add_generation_prompt) {
      result += "<start_of_turn>model\n";
    }
  } else if (architecture == "Gemma4ForCausalLM" ||
             architecture == "Gemma4_E2B_QNN") {
    // Gemma 4 requires a single leading <bos> token (id 2) at the very start
    // of the prompt; see the canonical sample_input in the model's
    // nntr_config.json. Nothing downstream adds it (the tokenizer.json
    // post_processor is empty and tokenizer_config.json has no
    // add_bos_token), so prepend it exactly once here.
    result += "<bos>";
    for (const auto &msg : messages) {
      std::string role = msg.role;
      if (role == "assistant") {
        role = "model";
      }
      result += "<|turn>" + role + "\n" + msg.content + "<turn|>\n";
    }
    if (add_generation_prompt) {
      result += "<|turn>model\n";
    }
  } else {
    const auto *reg_cb = ModelCallbackRegistry::instance().lookup(architecture);
    if (reg_cb && reg_cb->format_prompt) {
      // Build a full multi-message prompt via the registered format_prompt.
      // Concatenate all message contents and delegate to the callback.
      std::string combined;
      for (const auto &msg : messages) {
        combined += msg.content + "\n";
      }
      result = reg_cb->format_prompt(combined);
      return result;
    }
    // Unknown architecture fallback
    for (const auto &msg : messages) {
      result += msg.content + "\n";
    }
  }

  return result;
}

ErrorCode applyChatTemplate(const CausalLMChatMessage *messages,
                            size_t num_messages, bool add_generation_prompt,
                            const char **formattedText) {
  if (messages == nullptr || num_messages == 0 || formattedText == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    auto &h = get_default_handle();
    std::lock_guard<std::mutex> lock(h.mtx);

    // Debug: print messages before convertMessages
    LOGD("[DEBUG] applyChatTemplate: num_messages=%zu", num_messages);
    for (size_t i = 0; i < num_messages; ++i) {
      LOGD("[DEBUG] applyChatTemplate: messages[%zu] role='%s' content='%s'", i,
           messages[i].role ? messages[i].role : "(null)",
           messages[i].content ? messages[i].content : "(null)");
    }

    auto chat_messages = convertMessages(messages, num_messages);
    std::string arch =
      h.architectures.empty() ? std::string() : h.architectures[0];
    std::string model_dir =
      h.model_dirs.empty() ? std::string() : h.model_dirs[0];
    std::string formattedInput = apply_chat_template_messages(
      arch, chat_messages, add_generation_prompt, model_dir);

    g_formatted_template = std::move(formattedInput);
    *formattedText = g_formatted_template.c_str();
  } catch (const std::exception &e) {
    LOGE("Exception in applyChatTemplate: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModelHandleWithMessages(CausalLmHandle handle,
                                     const CausalLMChatMessage *messages,
                                     size_t num_messages,
                                     bool add_generation_prompt,
                                     const char **outputText) {
  LOGD("[DEBUG] runModelHandleWithMessages: handle=%p", (void *)handle);

  if (handle == nullptr || outputText == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  size_t model_index = 0;
  std::string formattedInput;

  try {
    {
      std::lock_guard<std::mutex> lock(h.mtx);
      if (!h.initialized || h.models.empty()) {
        return CAUSAL_LM_ERROR_NOT_INITIALIZED;
      }
      model_index = text_generation_model_index(h);
      if (model_index >= h.models.size() || !h.models[model_index]) {
        return CAUSAL_LM_ERROR_NOT_INITIALIZED;
      }

      std::string model_dir = h.model_dirs.size() > model_index
                                ? h.model_dirs[model_index]
                                : std::string();

      auto chat_messages = convertMessages(messages, num_messages);
      std::string arch = h.architectures.size() > model_index
                           ? h.architectures[model_index]
                           : std::string();
      formattedInput = apply_chat_template_messages(
        arch, chat_messages, add_generation_prompt, model_dir);
    }

    return run_on_handle(h, formattedInput.c_str(), outputText,
                         /*input_already_formatted=*/true, model_index);
  } catch (const std::exception &e) {
    LOGE("Exception in runModelHandleWithMessages: %s", e.what());
    return CAUSAL_LM_ERROR_UNKNOWN;
  }
}

ErrorCode runModelHandleWithTool(CausalLmHandle handle,
                                 const char *inputTextPrompt,
                                 const char **outputText, const char *tool_name,
                                 const char *tool_schema) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  causallm::XGrammar *grammar = nullptr;
  // Step 1: Check if tool exists in XGrammarManager
  if (causallm::XGrammarManager::Instance().hasTool(tool_name)) {
    LOGD("[runModelWithToolHandle] Tool '%s' found in XGrammarManager, using "
         "existing grammar",
         tool_name);
    grammar = causallm::XGrammarManager::Instance().getGrammar(tool_name);
  } else {
    // Step 2: Tool doesn't exist, create and register it
    if (tool_schema == nullptr) {
      LOGE("Error: Tool '%s' not found and no schema provided", tool_name);
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }

    LOGD("[runModelWithToolHandle] Tool '%s' not found, creating new grammar",
         tool_name);
    bool registered = causallm::XGrammarManager::Instance().registerTool(
      tool_name, tool_schema);

    if (!registered) {
      LOGE("Error: Failed to register tool '%s'", tool_name);
      return CAUSAL_LM_ERROR_UNKNOWN;
    }

    grammar = causallm::XGrammarManager::Instance().getGrammar(tool_name);
  }

  if (grammar == nullptr) {
    LOGE("Error: Failed to get grammar for tool '%s'", tool_name);
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return run_on_handle(*handle, inputTextPrompt, outputText,
                       /*input_already_formatted=*/false,
                       kAutoTextGenerationModelIndex, grammar);
}

/**===========================================================================
 * Legacy non-handle API implementation
 *============================================================================*/

ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type,
                    const char *model_base_path) {
  return load_into_handle(get_default_handle(), compute, modeltype, quant_type,
                          nullptr, model_base_path);
}

ErrorCode saveQnnKvCache(const char *cache_path) {
  return save_qnn_kv_cache_on_handle(get_default_handle(), cache_path);
}

ErrorCode loadQnnKvCache(const char *cache_path) {
  return load_qnn_kv_cache_on_handle(get_default_handle(), cache_path);
}

ErrorCode resetQnnKvCache(void) {
  return reset_qnn_kv_cache_on_handle(get_default_handle());
}

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics) {
  return metrics_on_handle(get_default_handle(), metrics);
}

/**===========================================================================
 * Handle-based API implementation
 *============================================================================*/

ErrorCode loadModelHandle(BackendType compute, ModelType modeltype,
                          ModelQuantizationType quant_type,
                          const char *native_lib_dir,
                          const char *model_base_path,
                          CausalLmHandle *out_handle) {
  LOGD("[DEBUG] loadModelHandle:%d START", __LINE__);
  LOGD("[DEBUG] loadModelHandle:%d   compute: %d", __LINE__, compute);
  LOGD("[DEBUG] loadModelHandle:%d   modeltype: %d", __LINE__, modeltype);
  LOGD("[DEBUG] loadModelHandle:%d   quant_type: %d", __LINE__, quant_type);
  LOGD("[DEBUG] loadModelHandle:%d   native_lib_dir: %s", __LINE__,
       native_lib_dir ? native_lib_dir : "(null)");
  LOGD("[DEBUG] loadModelHandle:%d   out_handle ptr: %p", __LINE__,
       (void *)out_handle);

  if (out_handle == nullptr) {
    LOGE("[DEBUG] loadModelHandle:%d out_handle is nullptr", __LINE__);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  auto *h = new (std::nothrow) CausalLmModel();
  if (h == nullptr) {
    LOGE("[DEBUG] loadModelHandle:%d Failed to allocate CausalLmModel",
         __LINE__);
    return CAUSAL_LM_ERROR_UNKNOWN;
  }
  LOGD("[DEBUG] loadModelHandle:%d CausalLmModel allocated at %p", __LINE__,
       (void *)h);

  LOGD("[DEBUG] loadModelHandle:%d Calling load_into_handle...", __LINE__);
  ErrorCode ec = load_into_handle(*h, compute, modeltype, quant_type,
                                  native_lib_dir, model_base_path);
  LOGD("[DEBUG] loadModelHandle:%d load_into_handle returned: %d", __LINE__,
       ec);

  if (ec != CAUSAL_LM_ERROR_NONE) {
    LOGE("[DEBUG] loadModelHandle:%d load_into_handle failed, deleting handle",
         __LINE__);
    delete h;
    *out_handle = nullptr;
    return ec;
  }
  *out_handle = h;
  LOGD("[DEBUG] loadModelHandle:%d SUCCESS, handle set to %p", __LINE__,
       (void *)h);
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode loadModelHandleByName(BackendType compute, const char *model_id,
                                ModelQuantizationType quant_type,
                                const char *native_lib_dir,
                                const char *model_base_path,
                                CausalLmHandle *out_handle) {
  if (out_handle == nullptr || model_id == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  register_models(); // ensure factory + public descriptors registered

  auto d_opt = find_descriptor_by_id(model_id);
  if (!d_opt) {
    LOGE("loadModelHandleByName: unknown id '%s'", model_id);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  const ModelDescriptor &d = *d_opt;
  if (!d.config_name) {
    LOGE("loadModelHandleByName: descriptor '%s' has null config_name",
         model_id);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  if (((d.backend_mask >> (unsigned)compute) & 1u) == 0u) {
    LOGE("loadModelHandleByName: backend %d not in mask 0x%x for '%s'", compute,
         d.backend_mask, model_id);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto *h = new (std::nothrow) CausalLmModel();
  if (!h)
    return CAUSAL_LM_ERROR_UNKNOWN;

  ErrorCode ec = load_into_handle(*h, compute, d.config_name, quant_type,
                                  native_lib_dir, model_base_path);
  if (ec != CAUSAL_LM_ERROR_NONE) {
    delete h;
    *out_handle = nullptr;
    return ec;
  }
  *out_handle = h;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode configureSpeculativeDecoding(CausalLmHandle h, bool use_sd) {
  if (!h)
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  if (!use_sd)
    return CAUSAL_LM_ERROR_NONE;
  if (h->models.empty() || h->architectures.empty())
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  const auto *cb =
    ModelCallbackRegistry::instance().lookup(h->architectures[0]);
  if (!cb || !cb->configure_speculative_decoding)
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  return cb->configure_speculative_decoding(h->models[0].get(), use_sd);
}

ErrorCode loadMultimodalHandleByName(
  BackendType compute, const char *embedding_model_id, const char *llm_model_id,
  ModelQuantizationType quant_type, const char *native_lib_dir,
  const char *model_base_path, CausalLmHandle *out_handle) {
  if (out_handle == nullptr || embedding_model_id == nullptr ||
      llm_model_id == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

#ifndef QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API
  (void)compute;
  (void)quant_type;
  (void)native_lib_dir;
  (void)model_base_path;
  *out_handle = nullptr;
  LOGE("loadMultimodalHandleByName: experimental multimodal nntrainer API is "
       "not enabled");
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#else
  register_models();

  auto ev = find_descriptor_by_id(embedding_model_id);
  auto lv = find_descriptor_by_id(llm_model_id);
  if (!ev || !lv || !ev->config_name || !lv->config_name) {
    LOGE("loadMultimodalHandleByName: unknown id(s) emb='%s' llm='%s'",
         embedding_model_id, llm_model_id);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // Load each model into its own temporary single-model handle, then move the
  // sub-models into the combined handle in order [vision producer, LLM].
  // This reuses the proven single-model load path without modifying it.
  CausalLmModel tmp_vision;
  CausalLmModel tmp_llm;
  ErrorCode ec = load_into_handle(tmp_vision, compute, ev->config_name,
                                  quant_type, native_lib_dir, model_base_path);
  if (ec != CAUSAL_LM_ERROR_NONE) {
    LOGE("loadMultimodalHandleByName: vision '%s' load failed (%d)",
         embedding_model_id, ec);
    return ec;
  }
  ec = load_into_handle(tmp_llm, compute, lv->config_name, quant_type,
                        native_lib_dir, model_base_path);
  if (ec != CAUSAL_LM_ERROR_NONE) {
    LOGE("loadMultimodalHandleByName: llm '%s' load failed (%d)", llm_model_id,
         ec);
    return ec;
  }
  if (tmp_vision.models.empty() || tmp_llm.models.empty())
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;

  // Compatibility check (R5): the LLM must expose an embedding table so the
  // composer can interleave text + image embeddings.
  if (tmp_llm.models[0]->embeddingBytesPerToken() == 0) {
    LOGE("loadMultimodalHandleByName: LLM '%s' has no embedding table — "
         "incompatible pair",
         llm_model_id);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  auto *h = new (std::nothrow) CausalLmModel();
  if (!h)
    return CAUSAL_LM_ERROR_UNKNOWN;

  auto move_one = [](CausalLmModel &src, CausalLmModel &dst) {
    dst.models.push_back(std::move(src.models[0]));
    dst.architectures.push_back(
      src.architectures.empty() ? std::string() : src.architectures[0]);
    dst.model_dirs.push_back(src.model_dirs.empty() ? std::string()
                                                    : src.model_dirs[0]);
    if (!src.initialization_duration_ms.empty())
      dst.initialization_duration_ms.push_back(
        src.initialization_duration_ms[0]);
  };
  move_one(tmp_vision, *h); // index 0 = vision producer
  move_one(tmp_llm, *h);    // index 1 = LLM consumer
  if (native_lib_dir != nullptr)
    h->native_lib_dir = native_lib_dir;
  h->initialized = true;

  *out_handle = h;
  return CAUSAL_LM_ERROR_NONE;
#endif
}

ErrorCode saveQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return save_qnn_kv_cache_on_handle(*handle, cache_path);
}

ErrorCode loadQnnKvCacheHandle(CausalLmHandle handle, const char *cache_path) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return load_qnn_kv_cache_on_handle(*handle, cache_path);
}

ErrorCode resetQnnKvCacheHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return reset_qnn_kv_cache_on_handle(*handle);
}

ErrorCode getPerformanceMetricsHandle(CausalLmHandle handle,
                                      PerformanceMetrics *metrics) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  return metrics_on_handle(*handle, metrics);
}

/**===========================================================================
 * Internal streaming helper
 *============================================================================*/

static ErrorCode run_model_streaming_on_handle(CausalLmModel &h,
                                               const std::string &raw_input,
                                               CausalLmTokenCallback callback,
                                               void *user_data,
                                               bool input_already_formatted,
                                               size_t model_index) {
  if (model_index >= h.models.size() || !h.models[model_index]) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  auto *m = h.models[model_index].get();
  if (!model_supports_text_output(m)) {
    LOGE("[DEBUG] run_model_streaming_on_handle: model[%zu] does not expose "
         "text output",
         model_index);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  if (!set_model_streamer(m, &streamer.base))
    return CAUSAL_LM_ERROR_UNSUPPORTED;

  struct Detach {
    causallm::Transformer *t;
    ~Detach() { set_model_streamer(t, nullptr); }
  } detach_guard{m};

  try {
    std::string input = prepare_input_for_model(h, model_index, raw_input,
                                                input_already_formatted);

    LOGD("[DEBUG]   raw input length: %zu", raw_input.length());
    LOGD("[DEBUG]   g_use_chat_template: %d", g_use_chat_template);
    if (input_already_formatted) {
      LOGD("[DEBUG]   input_already_formatted=1, using pre-formatted input "
           "(length: %zu)",
           input.length());
    } else {
      LOGD("[DEBUG]   input_already_formatted=0, applying chat template");
      LOGD("[DEBUG]   model input length: %zu", input.length());
      LOGD("[DEBUG]   model input: %s", input.c_str());
    }

#if defined(_WIN32)
    m->run(std::wstring(input.begin(), input.end()), false, L"", L"",
           g_verbose);
#else
    m->run(input, false, "", "", true);
#endif

    if (!get_model_output(m, h.last_output))
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    update_handle_session_after_run(h, model_index);

    if (m->hasRun()) {
      auto im = m->getPerformanceMetrics();
      double total_init = 0.0;
      for (double d : h.initialization_duration_ms)
        total_init += d;

      LOGD("[PERF] Performance Metrics:");
      LOGD("[PERF]   prefill_tokens: %u", im.prefill_tokens);
      LOGD("[PERF]   prefill_duration_ms: %.2f", im.prefill_duration_ms);
      LOGD("[PERF]   generation_tokens: %u", im.generation_tokens);
      LOGD("[PERF]   generation_duration_ms: %.2f", im.generation_duration_ms);
      LOGD("[PERF]   total_duration_ms: %.2f", im.total_duration_ms);
      LOGD("[PERF]   peak_memory_kb: %zu", im.peak_memory_kb);
      LOGD("[PERF]   initialization_duration_ms: %.2f", total_init);

      if (im.prefill_duration_ms > 0) {
        double tokens_per_sec =
          (im.prefill_tokens * 1000.0) / im.prefill_duration_ms;
        LOGD("[PERF]   prefill_tokens_per_sec: %.2f", tokens_per_sec);
      }
      if (im.generation_duration_ms > 0) {
        double tokens_per_sec =
          (im.generation_tokens * 1000.0) / im.generation_duration_ms;
        LOGD("[PERF]   generation_tokens_per_sec: %.2f", tokens_per_sec);
      }
    }
  } catch (const std::exception &e) {
    LOGE("[DEBUG] run_model_streaming_on_handle: Exception caught: %s",
         e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("[DEBUG] run_model_streaming_on_handle: Unknown exception caught");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModelHandleStreaming(CausalLmHandle handle,
                                  const char *inputTextPrompt,
                                  CausalLmTokenCallback callback,
                                  void *user_data) {
  LOGD("[DEBUG] runModelHandleStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   inputTextPrompt: %.50s%s",
       inputTextPrompt ? inputTextPrompt : "(null)",
       inputTextPrompt && strlen(inputTextPrompt) > 50 ? "..." : "");

  if (handle == nullptr || inputTextPrompt == nullptr || callback == nullptr) {
    LOGE("[DEBUG] runModelHandleStreaming: INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);

  if (!h.initialized || h.models.empty()) {
    LOGE("[DEBUG] runModelHandleStreaming: NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  const size_t model_index = text_generation_model_index(h);

  ErrorCode ec = run_model_streaming_on_handle(
    h, std::string(inputTextPrompt), callback, user_data,
    /*input_already_formatted=*/false, model_index);

  LOGD("[DEBUG] runModelHandleStreaming: END (errorCode=%d)", ec);
  return ec;
}

ErrorCode encodeModelHandle(CausalLmHandle handle, const char *text,
                            float **out_embedding, int *out_dim) {
  if (handle == nullptr || text == nullptr || out_embedding == nullptr ||
      out_dim == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  *out_embedding = nullptr;
  *out_dim = 0;

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);

  if (!h.initialized || h.models.empty()) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Embedding models occupy models[0] (single-model embedding handle).
  auto *st = dynamic_cast<causallm::SentenceTransformer *>(h.models[0].get());
  if (st == nullptr) {
    LOGE("encodeModelHandle: models[0] is not a SentenceTransformer");
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  try {
    const int dim = st->getEmbeddingDim();
    if (dim <= 0) {
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }

    // WSTR is std::string in this codebase; pass the text directly,
    // consistent with runModelHandleStreaming.
    std::string s(text);

    std::vector<float *> results = st->encode(s);
    if (results.empty() || results[0] == nullptr) {
      for (auto *p : results)
        delete[] p;
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }

    // Copy the batch-0 embedding (first DIM floats) into a caller-owned buffer.
    float *buf = new float[dim];
    std::memcpy(buf, results[0], sizeof(float) * static_cast<size_t>(dim));

    // encode() allocates each pointer with new[]; release them all.
    for (auto *p : results)
      delete[] p;

    *out_embedding = buf;
    *out_dim = dim;
    return CAUSAL_LM_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGE("encodeModelHandle: exception: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("encodeModelHandle: unknown exception");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

void freeEmbedding(float *embedding) { delete[] embedding; }

ErrorCode unloadModelHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_NONE;
  }
  std::lock_guard<std::mutex> lock(handle->mtx);
  handle->models.clear();
  handle->architectures.clear();
  handle->model_dirs.clear();
  handle->initialization_duration_ms.clear();
  handle->initialized = false;
  reset_handle_session_state(*handle);
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode destroyModelHandle(CausalLmHandle handle) {
  if (handle == nullptr) {
    return CAUSAL_LM_ERROR_NONE;
  }
  // Take the mutex to make sure no in-flight call on this handle is still
  // running, then release and delete. Any caller that still holds a pointer
  // to the output buffer returned by runModelHandleWithMessages is reading
  // freed memory after this point — documented as "valid until destroy".
  {
    std::lock_guard<std::mutex> lock(handle->mtx);
    handle->models.clear();
    handle->architectures.clear();
    handle->model_dirs.clear();
    handle->initialization_duration_ms.clear();
    handle->initialized = false;
    reset_handle_session_state(*handle);
  }
  delete handle;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode cancelModelHandle(CausalLmHandle handle) {
  LOGD("[DEBUG] cancelModelHandle: handle=%p", (void *)handle);

  if (handle == nullptr) {
    LOGE("[DEBUG] cancelModelHandle: handle is nullptr, returning "
         "INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // NOTE: We intentionally do NOT take the mutex here to avoid blocking
  // when run() is holding the lock. The requestStop() method is thread-safe
  // (uses atomic<bool>), and the models vector is not modified during run()
  // (only during load/unload which do take the mutex). This allows immediate
  // cancellation from any thread (e.g., UI cancel button handler).
  LOGD("[DEBUG] cancelModelHandle: checking state without mutex, "
       "initialized=%d, models.size=%zu",
       handle->initialized, handle->models.size());

  if (!handle->initialized || handle->models.empty()) {
    LOGE(
      "[DEBUG] cancelModelHandle: not initialized, returning NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Set stop flag on all models (primarily affects models[0] for LLM)
  for (size_t i = 0; i < handle->models.size(); ++i) {
    if (handle->models[i]) {
      if (!request_model_stop(handle->models[i].get())) {
        LOGD("[DEBUG] cancelModelHandle: model[%zu] is not cancellable", i);
        continue;
      }
      LOGD("[DEBUG] cancelModelHandle: calling requestStop() on model[%zu]", i);
    }
  }

  LOGD("[DEBUG] cancelModelHandle: returning NONE (success)");
  return CAUSAL_LM_ERROR_NONE;
}

/**===========================================================================
 * Multimodal API Implementation
 *
 * Preconditions: the handle must have been loaded from a multi-model
 * nntr_config.json carrying at least two sub-models. The first sub-model
 * is expected to be the vision encoder and the second the LLM, though
 * the concrete integration (vision encoding + embedding fusion + LLM
 * generation) is still TODO. Single-model handles return
 * CAUSAL_LM_ERROR_UNSUPPORTED.
 *============================================================================*/

#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
/**
 * Model-agnostic multimodal composer. Works through base Transformer virtuals
 * only (no concrete-model casts), so any [vision producer, LLM consumer] pair
 * (e.g. future vJEPA/siglip+LFM) is driven identically.
 *
 *   llm:          embedding CONSUMER (lookupEmbedding / run_with_embeddings)
 *   image_embeds: producer output; ownership taken here (freed before return)
 */
static ErrorCode execute_multimodal(CausalLmModel &h,
                                    causallm::Transformer *llm,
                                    causallm::multimodal_pointer image_embeds,
                                    const std::string &prompt,
                                    CausalLmTokenCallback callback,
                                    void *user_data) {
  auto *tok = llm->getTokenizer();
  if (tok == nullptr) {
    LOGE("[MM] llm has no tokenizer");
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  std::vector<int> text_ids = tok->Encode(prompt);
  int32_t image_token_id = tok->TokenToId("<|image|>");

  const size_t bpt = llm->embeddingBytesPerToken();
  if (bpt == 0) {
    LOGE("[MM] llm embedding table not loaded (needs uses_embedding=false + "
         "embedding_file_name)");
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }
  if (image_embeds.second % bpt != 0) {
    LOGE("[MM] image_embeds.size=%zu not a multiple of bpt=%zu",
         image_embeds.second, bpt);
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  const size_t n_image = image_embeds.second / bpt;

  auto it_img = (image_token_id >= 0)
                  ? std::find(text_ids.begin(), text_ids.end(), image_token_id)
                  : text_ids.end();
  const bool has_placeholder = (it_img != text_ids.end());
  const size_t img_pos =
    has_placeholder
      ? static_cast<size_t>(std::distance(text_ids.begin(), it_img))
      : 0;
  const size_t n_text_kept = text_ids.size() - (has_placeholder ? 1 : 0);
  const size_t n_total = n_text_kept + n_image;
  LOGD("[MM] text=%zu image=%zu total=%zu placeholder=%d pos=%zu",
       text_ids.size(), n_image, n_total, has_placeholder, img_pos);

  std::vector<uint8_t> combined(n_total * bpt);
  uint8_t *dst = combined.data();
  auto copy_text_range = [&](size_t start, size_t end) -> bool {
    for (size_t i = start; i < end; ++i) {
      const void *e = llm->lookupEmbedding(text_ids[i]);
      if (e == nullptr) {
        LOGE("[MM] lookupEmbedding(%d) null", text_ids[i]);
        return false;
      }
      std::memcpy(dst, e, bpt);
      dst += bpt;
    }
    return true;
  };
  if (!copy_text_range(0, img_pos)) {
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  std::memcpy(dst, image_embeds.first, n_image * bpt);
  dst += n_image * bpt;
  const size_t after_start = has_placeholder ? img_pos + 1 : img_pos;
  if (!copy_text_range(after_start, text_ids.size())) {
    std::free(image_embeds.first);
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  std::free(image_embeds.first);
  image_embeds.first = nullptr;

  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  // Attach via the cast helper: setStreamer lives on Quick_Dot_AI_QNN /
  // CausalLM, not on the base Transformer the composer drives through.
  set_model_streamer(llm, &streamer.base);
  struct Detach {
    causallm::Transformer *t;
    ~Detach() { set_model_streamer(t, nullptr); }
  } detach_guard{llm};

  try {
    llm->run_with_embeddings(combined.data(), n_total, text_ids,
                             /*do_sample=*/false, /*log_output=*/g_verbose);
    h.kv_len = llm->getKvLen();
  } catch (const std::exception &e) {
    LOGE("[MM] llm threw: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * Run the vision encoder (producer) on raw pixels and return its image
 * embeddings, after matching its quant space to the LLM consumer.
 * @return image_embeds (ownership transferred to caller) or {nullptr,0}.
 */
static causallm::multimodal_pointer
run_vision_encoder(CausalLmModel &h, const char *prompt,
                   const float *pixelValues, int numPatches, int originalHeight,
                   int originalWidth) {
  const int PATCH_SIZE = 512; // pixel layout: numPatches*3*512*512 floats
  causallm::Transformer *vision = h.models[0].get();
  causallm::Transformer *llm = h.models[1].get();

  auto info = llm->get_embedding_info();
  vision->set_quant_param(info.first, info.second);

  const size_t pixel_bytes = static_cast<size_t>(numPatches) * 3 * PATCH_SIZE *
                             PATCH_SIZE * sizeof(float);
  causallm::multimodal_pointer image_in{const_cast<float *>(pixelValues),
                                        pixel_bytes};
  return vision->run_image(std::string(prompt ? prompt : ""), image_in,
                           originalHeight, originalWidth, /*do_sample=*/false,
                           "", "", g_verbose);
}
#endif // ENABLE_QNN_MODELS &&
       // QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API

#ifdef ENABLE_QNN_MODELS
/**
 * Standalone vision/video encoder run (no LLM). Wraps models[0]->run_image
 * and transfers the malloc'd output buffer to the caller. See header for the
 * ownership contract (free with freeImageEmbedding, not freeEmbedding).
 *
 * Always defined under ENABLE_QNN_MODELS (part of the public ABI). The
 * run_image- dependent body is gated on the experimental multimodal macro;
 * without it the function returns CAUSAL_LM_ERROR_UNSUPPORTED.
 */
ErrorCode encodeImageModelHandle(CausalLmHandle handle,
                                 const float *pixelValues, size_t numFloats,
                                 int height, int width, void **out_embedding,
                                 int *out_bytes) {
  if (handle == nullptr || pixelValues == nullptr || out_embedding == nullptr ||
      out_bytes == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  *out_embedding = nullptr;
  *out_bytes = 0;

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    LOGE("encodeImageModelHandle: handle not initialized or empty");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

#if defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
  causallm::Transformer *vision = h.models[0].get();
  // Intentionally do NOT call set_quant_param: standalone encode keeps
  // llm_quant_param_given_ == false so run_image returns the raw quantized
  // embedding via plain memcpy (no LLM consumer required).

  const size_t pixel_bytes = numFloats * sizeof(float);
  causallm::multimodal_pointer image_in{const_cast<float *>(pixelValues),
                                        pixel_bytes};
  try {
    causallm::multimodal_pointer embeds =
      vision->run_image(std::string(""), image_in, height, width,
                        /*do_sample=*/false, "", "", g_verbose);
    if (embeds.first == nullptr || embeds.second == 0) {
      LOGE("encodeImageModelHandle: run_image returned empty output");
      return CAUSAL_LM_ERROR_INFERENCE_FAILED;
    }
    *out_embedding = embeds.first; // ownership transferred to caller
    *out_bytes = static_cast<int>(embeds.second);
  } catch (const std::exception &e) {
    LOGE("encodeImageModelHandle: exception: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("encodeImageModelHandle: unknown exception");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  return CAUSAL_LM_ERROR_NONE;
#else
  // Multimodal vision encoding relies on Transformer::run_image, which the
  // main nntrainer API does not provide. The proprietary vision QNN models
  // that implemented it (e.g. vjepa2-qnn) are excluded from the build. See
  // docs/qnn-model-main-adaptation-todo.ko.md.
  (void)height;
  (void)width;
  (void)numFloats;
  LOGE("encodeImageModelHandle: multimodal vision encoder API is not enabled");
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
}

void freeImageEmbedding(void *embedding) { std::free(embedding); }
#endif // ENABLE_QNN_MODELS

#ifndef ENABLE_QNN_MODELS
ErrorCode encodeImageModelHandle(CausalLmHandle handle,
                                 const float *pixelValues, size_t numFloats,
                                 int height, int width, void **out_embedding,
                                 int *out_bytes) {
  (void)handle;
  (void)pixelValues;
  (void)numFloats;
  (void)height;
  (void)width;
  if (out_embedding)
    *out_embedding = nullptr;
  if (out_bytes)
    *out_bytes = 0;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
}

void freeImageEmbedding(void *embedding) { (void)embedding; }
#endif // !ENABLE_QNN_MODELS

ErrorCode runMultimodalHandleStreaming(CausalLmHandle handle,
                                       const char *prompt,
                                       const float *pixelValues, int numPatches,
                                       int originalHeight, int originalWidth,
                                       CausalLmTokenCallback callback,
                                       void *user_data) {
  LOGD("[DEBUG] runMultimodalHandleStreaming: START");
  LOGD("[DEBUG]   handle=%p", handle);
  LOGD("[DEBUG]   prompt=%s", prompt ? prompt : "(null)");
  LOGD("[DEBUG]   pixelValues=%p", pixelValues);
  LOGD("[DEBUG]   numPatches=%d", numPatches);
  LOGD("[DEBUG]   originalHeight=%d", originalHeight);
  LOGD("[DEBUG]   originalWidth=%d", originalWidth);
  LOGD("[DEBUG]   callback=%p", (void *)callback);
  LOGD("[DEBUG]   user_data=%p", user_data);

  if (handle == nullptr || prompt == nullptr || pixelValues == nullptr ||
      callback == nullptr) {
    LOGE("[DEBUG] runMultimodalHandleStreaming: INVALID_PARAMETER"
         " handle=%p prompt=%s pixelValues=%p callback=%p",
         handle, prompt, pixelValues, (void *)callback);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    LOGE("[DEBUG] runMultimodalHandleStreaming: NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  // Multimodal expects the handle to be loaded from a multi-model
  // nntr_config.json (architectures[] + model_dirs[]) with at least
  // [vision_encoder, llm]. A single-model handle cannot drive this path.
  if (h.models.size() < 2) {
    LOGE("[DEBUG] runMultimodalHandleStreaming: need >=2 sub-models "
         "(got %zu). Load with multi-model nntr_config.json.",
         h.models.size());
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  LOGD("[DEBUG] runMultimodalHandleStreaming: %zu sub-models loaded",
       h.models.size());
  for (size_t i = 0; i < h.architectures.size(); ++i) {
    LOGD("[DEBUG]   models[%zu]: arch=%s dir=%s", i, h.architectures[i].c_str(),
         h.model_dirs[i].c_str());
  }

  // Log pixel values summary (first few values)
  // Note: patch size is fixed at 512x512
  const int PATCH_SIZE = 512;
  long long totalValues = 1LL * numPatches * 3 * PATCH_SIZE * PATCH_SIZE;
  LOGD("[DEBUG]   totalPixelValues=%lld", totalValues);
  if (totalValues > 0 && pixelValues != nullptr) {
    LOGD("[DEBUG]   pixelValues[0..4]=%f, %f, %f, %f, %f", pixelValues[0],
         pixelValues[1], pixelValues[2],
         (totalValues > 3 ? pixelValues[3] : 0.0f),
         (totalValues > 4 ? pixelValues[4] : 0.0f));
  }

#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
  // Generic path: models[0]=vision producer, models[1]=LLM consumer.
  causallm::multimodal_pointer image_embeds{nullptr, 0};
  try {
    image_embeds = run_vision_encoder(h, prompt, pixelValues, numPatches,
                                      originalHeight, originalWidth);
  } catch (const std::exception &e) {
    LOGE("[DEBUG] runMultimodalHandleStreaming: vision threw: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  const std::string raw_input(prompt);
  const bool input_already_formatted =
    raw_input.find("<|turn_start|>") != std::string::npos ||
    raw_input.find("<|im_start|>") != std::string::npos ||
    raw_input.find("<start_of_turn>") != std::string::npos;
  std::string input =
    prepare_input_for_model(h, 1, raw_input, input_already_formatted);

  return execute_multimodal(h, h.models[1].get(), image_embeds, input, callback,
                            user_data);
#else
  LOGE("[DEBUG] runMultimodalHandleStreaming: experimental multimodal "
       "nntrainer API is not enabled");
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
}

ErrorCode runMultimodalHandleWithMessages(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const float *pixelValues,
  int numPatches, int originalHeight, int originalWidth,
  const char **outputText) {
  LOGD("[DEBUG] runMultimodalHandleWithMessages: START");
  LOGD("[DEBUG]   handle=%p", handle);
  LOGD("[DEBUG]   messages=%p", messages);
  LOGD("[DEBUG]   num_messages=%zu", num_messages);
  LOGD("[DEBUG]   add_generation_prompt=%d", add_generation_prompt);
  LOGD("[DEBUG]   pixelValues=%p", pixelValues);
  LOGD("[DEBUG]   numPatches=%d", numPatches);
  LOGD("[DEBUG]   originalHeight=%d", originalHeight);
  LOGD("[DEBUG]   originalWidth=%d", originalWidth);
  LOGD("[DEBUG]   outputText=%p", outputText);

  if (handle == nullptr || messages == nullptr || num_messages == 0 ||
      pixelValues == nullptr || outputText == nullptr) {
    LOGE("[DEBUG] runMultimodalHandleWithMessages: INVALID_PARAMETER"
         " handle=%p messages=%p num_messages=%zu pixelValues=%p outputText=%p",
         handle, messages, num_messages, pixelValues, outputText);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);
  if (!h.initialized || h.models.empty()) {
    LOGE("[DEBUG] runMultimodalHandleWithMessages: NOT_INITIALIZED");
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  if (h.models.size() < 2) {
    LOGE(
      "[DEBUG] runMultimodalHandleWithMessages: need >=2 sub-models (got %zu)",
      h.models.size());
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_UNSUPPORTED;
  }

  LOGD("[DEBUG] runMultimodalHandleWithMessages: %zu sub-models loaded",
       h.models.size());
  for (size_t i = 0; i < h.architectures.size(); ++i) {
    LOGD("[DEBUG]   models[%zu]: arch=%s dir=%s", i, h.architectures[i].c_str(),
         h.model_dirs[i].c_str());
  }

  // Apply chat template
  auto chat_messages = convertMessages(messages, num_messages);
  const size_t llm_index = h.architectures.size() > 1 ? 1 : 0;
  std::string arch = h.architectures.size() > llm_index
                       ? h.architectures[llm_index]
                       : std::string();
  std::string model_dir =
    h.model_dirs.size() > llm_index ? h.model_dirs[llm_index] : std::string();
  std::string prompt = apply_chat_template_messages(
    arch, chat_messages, add_generation_prompt, model_dir);
  LOGD("[DEBUG]   formatted prompt length: %zu", prompt.length());
  LOGD("[DEBUG]   formatted prompt preview: %.100s%s", prompt.c_str(),
       prompt.length() > 100 ? "..." : "");

  // Log pixel values summary (first few values)
  // Note: patch size is fixed at 512x512
  const int PATCH_SIZE = 512;
  long long totalValues = 1LL * numPatches * 3 * PATCH_SIZE * PATCH_SIZE;
  LOGD("[DEBUG]   totalPixelValues=%lld", totalValues);
  if (totalValues > 0 && pixelValues != nullptr) {
    LOGD("[DEBUG]   pixelValues[0..4]=%f, %f, %f, %f, %f", pixelValues[0],
         pixelValues[1], pixelValues[2],
         (totalValues > 3 ? pixelValues[3] : 0.0f),
         (totalValues > 4 ? pixelValues[4] : 0.0f));
  }

#if defined(ENABLE_QNN_MODELS) &&                                              \
  defined(QUICKAI_ENABLE_EXPERIMENTAL_MULTIMODAL_NNTRAINER_API)
  causallm::multimodal_pointer image_embeds{nullptr, 0};
  try {
    image_embeds =
      run_vision_encoder(h, prompt.c_str(), pixelValues, numPatches,
                         originalHeight, originalWidth);
  } catch (const std::exception &e) {
    LOGE("[DEBUG] runMultimodalHandleWithMessages: vision threw: %s", e.what());
    *outputText = nullptr;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  h.last_output.clear();
  auto accumulate_cb = [](const char *delta, void *ud) -> int {
    if (delta)
      static_cast<std::string *>(ud)->append(delta);
    return 0;
  };
  ErrorCode ec = execute_multimodal(h, h.models[1].get(), image_embeds, prompt,
                                    accumulate_cb, &h.last_output);
  if (ec != CAUSAL_LM_ERROR_NONE) {
    *outputText = nullptr;
    return ec;
  }
#else
  LOGE("[DEBUG] runMultimodalHandleWithMessages: experimental multimodal "
       "nntrainer API is not enabled");
  *outputText = nullptr;
  return CAUSAL_LM_ERROR_UNSUPPORTED;
#endif
  *outputText = h.last_output.c_str();
  return CAUSAL_LM_ERROR_NONE;
}

/**===========================================================================
 * OpenAI messages streaming variants
 *============================================================================*/

extern "C" {

ErrorCode runModelHandleWithMessagesStreaming(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt,
  CausalLmTokenCallback callback, void *user_data) {
  LOGD("[DEBUG] runModelHandleWithMessagesStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   num_messages: %zu", num_messages);

  if (handle == nullptr || messages == nullptr || num_messages == 0 ||
      callback == nullptr) {
    LOGE("[DEBUG] runModelHandleWithMessagesStreaming: INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);

  if (!h.initialized || h.models.empty()) {
    LOGE("[DEBUG] runModelHandleWithMessagesStreaming: NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  const size_t model_index = text_generation_model_index(h);
  if (model_index >= h.models.size() || !h.models[model_index]) {
    LOGE("[DEBUG] runModelHandleWithMessagesStreaming: text model is missing");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    LOGD("[DEBUG] runModelHandleWithMessagesStreaming: Formatting messages...");

    std::string model_dir = h.model_dirs.size() > model_index
                              ? h.model_dirs[model_index]
                              : std::string();

    // Use the *actual* handle's architecture so architecture-specific
    // chat template markers are generated.
    auto chat_messages = convertMessages(messages, num_messages);
    std::string arch = h.architectures.size() > model_index
                         ? h.architectures[model_index]
                         : std::string();
    std::string formattedInput = apply_chat_template_messages(
      arch, chat_messages, add_generation_prompt, model_dir);

    LOGD("[DEBUG]   raw messages count: %zu", num_messages);
    LOGD("[DEBUG]   formatted input length: %zu", formattedInput.length());
    LOGD("[DEBUG]   formatted input: %s", formattedInput.c_str());

    LOGD("[DEBUG] runModelHandleWithMessagesStreaming: Calling internal helper "
         "directly...");
    return run_model_streaming_on_handle(h, formattedInput, callback, user_data,
                                         /*input_already_formatted=*/true,
                                         model_index);
  } catch (const std::exception &e) {
    LOGE("[DEBUG] runModelHandleWithMessagesStreaming: Exception caught: %s",
         e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE(
      "[DEBUG] runModelHandleWithMessagesStreaming: Unknown exception caught");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

ErrorCode runMultimodalHandleWithMessagesStreaming(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const float *pixelValues,
  int numPatches, int originalHeight, int originalWidth,
  CausalLmTokenCallback callback, void *user_data) {
  LOGD("[DEBUG] runMultimodalHandleWithMessagesStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   num_messages: %zu", num_messages);

  if (handle == nullptr || messages == nullptr || num_messages == 0 ||
      pixelValues == nullptr || callback == nullptr) {
    LOGE("[DEBUG] runMultimodalHandleWithMessagesStreaming: INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    LOGD("[DEBUG] runMultimodalHandleWithMessagesStreaming: Formatting "
         "messages...");

    std::string formattedInput;
    {
      auto &h = *handle;
      std::lock_guard<std::mutex> lock(h.mtx);
      if (!h.initialized) {
        LOGE("[DEBUG] runMultimodalHandleWithMessagesStreaming: handle is not "
             "initialized for multimodal");
        return CAUSAL_LM_ERROR_NOT_INITIALIZED;
      }
      if (h.models.size() < 2) {
        LOGE("[DEBUG] runMultimodalHandleWithMessagesStreaming: need >=2 "
             "sub-models (got %zu)",
             h.models.size());
        return CAUSAL_LM_ERROR_UNSUPPORTED;
      }

      auto chat_messages = convertMessages(messages, num_messages);
      const size_t llm_index = h.architectures.size() > 1 ? 1 : 0;
      std::string arch = h.architectures.size() > llm_index
                           ? h.architectures[llm_index]
                           : std::string();
      std::string model_dir = h.model_dirs.size() > llm_index
                                ? h.model_dirs[llm_index]
                                : std::string();
      formattedInput = apply_chat_template_messages(
        arch, chat_messages, add_generation_prompt, model_dir);
    }

    LOGD("[DEBUG]   raw messages count: %zu", num_messages);
    LOGD("[DEBUG]   formatted input length: %zu", formattedInput.length());
    LOGD("[DEBUG]   formatted input preview: %.100s%s", formattedInput.c_str(),
         formattedInput.length() > 100 ? "..." : "");

    LOGD("[DEBUG] runMultimodalHandleWithMessagesStreaming: Delegating to "
         "runMultimodalHandleStreaming...");
    return runMultimodalHandleStreaming(handle, formattedInput.c_str(),
                                        pixelValues, numPatches, originalHeight,
                                        originalWidth, callback, user_data);
  } catch (const std::exception &e) {
    LOGE(
      "[DEBUG] runMultimodalHandleWithMessagesStreaming: Exception caught: %s",
      e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("[DEBUG] runMultimodalHandleWithMessagesStreaming: Unknown exception "
         "caught");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

/**===========================================================================
 * OpenAI JSON streaming API implementation
 *============================================================================*/

ErrorCode runModelHandleWithJsonStreaming(CausalLmHandle handle,
                                          const char *jsonRequest,
                                          CausalLmTokenCallback callback,
                                          void *user_data) {
  LOGD("[DEBUG] runModelHandleWithJsonStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   jsonRequest length: %zu",
       jsonRequest ? strlen(jsonRequest) : 0);

  if (handle == nullptr || jsonRequest == nullptr || callback == nullptr) {
    LOGE("[DEBUG] runModelHandleWithJsonStreaming: INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto &h = *handle;
  std::lock_guard<std::mutex> lock(h.mtx);

  if (!h.initialized || h.models.empty()) {
    LOGE("[DEBUG] runModelHandleWithJsonStreaming: NOT_INITIALIZED");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  const size_t model_index = text_generation_model_index(h);
  if (model_index >= h.models.size() || !h.models[model_index]) {
    LOGE("[DEBUG] runModelHandleWithJsonStreaming: text model is missing");
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    LOGD("[DEBUG] runModelHandleWithJsonStreaming: Parsing JSON request...");

    // Parse JSON request
    json request = json::parse(jsonRequest);
    LOGD("[DEBUG]   JSON parsed successfully");

    // Apply chat template using the existing g_chat_template
    // The chat_template.apply() method handles messages, tools, functions, etc.
    std::string formattedInput;
    if (g_chat_template.has_value()) {
      LOGD(
        "[DEBUG] runModelHandleWithJsonStreaming: Applying chat template...");
      formattedInput = g_chat_template->apply(request);
      LOGD("[DEBUG]   Formatted input length: %zu", formattedInput.length());
      LOGD("[DEBUG]   Formatted input preview: %.100s%s",
           formattedInput.c_str(), formattedInput.length() > 100 ? "..." : "");
    } else {
      LOGE(
        "[DEBUG] runModelHandleWithJsonStreaming: Chat template not available");
      return CAUSAL_LM_ERROR_UNSUPPORTED;
    }

    LOGD("[DEBUG] runModelHandleWithJsonStreaming: Running inference...");
    return run_model_streaming_on_handle(h, formattedInput, callback, user_data,
                                         /*input_already_formatted=*/true,
                                         model_index);
  } catch (const json::exception &e) {
    LOGE("[DEBUG] runModelHandleWithJsonStreaming: JSON parse error: %s",
         e.what());
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  } catch (const std::exception &e) {
    LOGE("[DEBUG] runModelHandleWithJsonStreaming: Exception caught: %s",
         e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("[DEBUG] runModelHandleWithJsonStreaming: Unknown exception caught");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
}

// ---------------------------------------------------------------------------
// Multi-image Multimodal API (V-JEPA)
// ---------------------------------------------------------------------------

// V-JEPA2 + LFM2 fused video path. Gated off (no VjepaLfm2 model /
// Transformer::run_video in this tree). See the include-site note for
// QUICKAI_ENABLE_VJEPA_LFM2_VIDEO.
#if defined(QUICKAI_ENABLE_VJEPA_LFM2_VIDEO)
// True when the handle holds a single self-contained video-language model
// (VJEPA2ViT + Projector + LFM2 fused in one Transformer) rather than the
// QNN [vision producer, LLM consumer] pair.
static bool is_vjepa_lfm2_combined(CausalLmModel &h) {
  return h.models.size() == 1 && !h.architectures.empty() &&
         h.architectures[0] == "Lfm2VLVJepa21BModel";
}

// Extract a plain text prompt from chat messages for the combined video path.
// The model applies its own <video> chat template internally, so we hand it
// the raw user text rather than a pre-formatted template.
static std::string
extract_prompt_from_messages(const CausalLMChatMessage *messages,
                             size_t num_messages) {
  std::string last_user;
  std::string concat;
  for (size_t i = 0; i < num_messages; ++i) {
    const char *role = messages[i].role;
    const char *content = messages[i].content;
    if (content == nullptr)
      continue;
    if (!concat.empty())
      concat += "\n";
    concat += content;
    if (role != nullptr && std::string(role) == "user")
      last_user = content;
  }
  return last_user.empty() ? concat : last_user;
}

// Extract the system-role text from chat messages (empty ⇒ model default).
static std::string
extract_system_from_messages(const CausalLMChatMessage *messages,
                             size_t num_messages) {
  std::string system;
  for (size_t i = 0; i < num_messages; ++i) {
    if (messages[i].role != nullptr && messages[i].content != nullptr &&
        std::string(messages[i].role) == "system") {
      system = messages[i].content; // last system message wins
    }
  }
  return system;
}

// Drive the fused video-language model: split the flat pixel buffer into
// @p numImages frames ([C,H,W] each), stream tokens through @p callback.
// @p system_prompt sets the chat-template system turn (empty ⇒ model default).
// Contract: @p numFloats is the TOTAL float count of @p pixelValues; it must
// be an exact multiple of @p numImages. The model validates each frame's size
// against its config (3 * img * img) and throws on mismatch.
static ErrorCode run_video_on_combined_handle(
  CausalLmModel &h, const std::string &prompt, const std::string &system_prompt,
  const float *pixelValues, int numFloats, int numImages,
  CausalLmTokenCallback callback, void *user_data) {
  if (numImages < 1 || numFloats < numImages || (numFloats % numImages) != 0) {
    LOGE("run_video_on_combined_handle: bad shape numFloats=%d numImages=%d",
         numFloats, numImages);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  auto *m = h.models[0].get();
  if (!m)
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;

  const size_t frame_size = static_cast<size_t>(numFloats) / numImages;
  std::vector<std::vector<float>> frames;
  frames.reserve(numImages);
  for (int i = 0; i < numImages; ++i) {
    const float *p = pixelValues + static_cast<size_t>(i) * frame_size;
    frames.emplace_back(p, p + frame_size);
  }

  CallbackStreamer streamer;
  callback_streamer_init(&streamer, callback, user_data);
  m->setStreamer(&streamer.base);
  struct Detach {
    causallm::Transformer *t;
    ~Detach() { t->setStreamer(nullptr); }
  } detach_guard{m};

  try {
    m->run_video(frames, prompt, system_prompt, /*do_sample=*/false,
                 /*log_output=*/false);
    h.last_output = m->getOutput(0);
    update_handle_session_after_run(h, 0);
  } catch (const std::exception &e) {
    LOGE("run_video_on_combined_handle: exception: %s", e.what());
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  } catch (...) {
    LOGE("run_video_on_combined_handle: unknown exception");
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }
  return CAUSAL_LM_ERROR_NONE;
}
#endif // QUICKAI_ENABLE_VJEPA_LFM2_VIDEO

ErrorCode runMultimodalMultiImageHandleStreaming(
  CausalLmHandle handle, const char *prompt, const float *pixelValues,
  int numPatches, int numImages, const int *patchesPerImage,
  const int *originalHeights, const int *originalWidths,
  CausalLmTokenCallback callback, void *user_data) {

  LOGD("[DEBUG] runMultimodalMultiImageHandleStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   numPatches: %d, numImages: %d", numPatches, numImages);

  if (handle == nullptr || prompt == nullptr || pixelValues == nullptr ||
      callback == nullptr || patchesPerImage == nullptr ||
      originalHeights == nullptr || originalWidths == nullptr) {
    LOGE("[DEBUG] runMultimodalMultiImageHandleStreaming: INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  if (numImages < 1 || numPatches < 1) {
    LOGE("[DEBUG] runMultimodalMultiImageHandleStreaming: "
         "numImages=%d, numPatches=%d — must be >= 1",
         numImages, numPatches);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // Validate handle and, for the fused video-language model, run the full
  // video pipeline while holding the handle lock.
  {
    auto &h = *reinterpret_cast<CausalLmModel *>(handle);
    std::lock_guard<std::mutex> lock(h.mtx);
    if (!h.initialized || h.models.empty()) {
      LOGE("[DEBUG] runMultimodalMultiImageHandleStreaming: NOT_INITIALIZED");
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }
#if defined(QUICKAI_ENABLE_VJEPA_LFM2_VIDEO)
    if (is_vjepa_lfm2_combined(h)) {
      LOGD("[DEBUG] runMultimodalMultiImageHandleStreaming: VJEPA-LFM2 video "
           "path (numImages=%d, numFloats=%d)",
           numImages, numPatches);
      // No messages here → use the model's default system persona.
      return run_video_on_combined_handle(h, std::string(prompt),
                                          /*system_prompt=*/std::string(),
                                          pixelValues, numPatches, numImages,
                                          callback, user_data);
    }
#endif
  }

  // Fallback (QNN [vision, LLM] pair): delegate to the single-image path using
  // the first image's metadata until the QNN multi-image encoder lands.
  LOGD("[DEBUG] runMultimodalMultiImageHandleStreaming: delegating to "
       "single-image runMultimodalHandleStreaming");

  return runMultimodalHandleStreaming(handle, prompt, pixelValues, numPatches,
                                      originalHeights[0], originalWidths[0],
                                      callback, user_data);
}

ErrorCode runMultimodalMultiImageHandleWithMessagesStreaming(
  CausalLmHandle handle, const CausalLMChatMessage *messages,
  size_t num_messages, bool add_generation_prompt, const float *pixelValues,
  int numPatches, int numImages, const int *patchesPerImage,
  const int *originalHeights, const int *originalWidths,
  CausalLmTokenCallback callback, void *user_data) {

  LOGD("[DEBUG] runMultimodalMultiImageHandleWithMessagesStreaming: START");
  LOGD("[DEBUG]   handle: %p", (void *)handle);
  LOGD("[DEBUG]   num_messages: %zu, numPatches: %d, numImages: %d",
       num_messages, numPatches, numImages);

  if (handle == nullptr || messages == nullptr || pixelValues == nullptr ||
      callback == nullptr || patchesPerImage == nullptr ||
      originalHeights == nullptr || originalWidths == nullptr) {
    LOGE("[DEBUG] runMultimodalMultiImageHandleWithMessagesStreaming: "
         "INVALID_PARAMETER");
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  if (numImages < 1 || numPatches < 1) {
    LOGE("[DEBUG] runMultimodalMultiImageHandleWithMessagesStreaming: "
         "numImages=%d, numPatches=%d — must be >= 1",
         numImages, numPatches);
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // Fused video-language model: extract the user prompt and run the video
  // pipeline (the model applies its own <video> chat template internally).
  {
    auto &h = *reinterpret_cast<CausalLmModel *>(handle);
    std::lock_guard<std::mutex> lock(h.mtx);
    if (!h.initialized || h.models.empty()) {
      LOGE("[DEBUG] runMultimodalMultiImageHandleWithMessagesStreaming: "
           "NOT_INITIALIZED");
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }
#if defined(QUICKAI_ENABLE_VJEPA_LFM2_VIDEO)
    if (is_vjepa_lfm2_combined(h)) {
      std::string prompt = extract_prompt_from_messages(messages, num_messages);
      std::string system = extract_system_from_messages(messages, num_messages);
      LOGD(
        "[DEBUG] runMultimodalMultiImageHandleWithMessagesStreaming: "
        "VJEPA-LFM2 video path (numImages=%d, numFloats=%d, system=%zu chars)",
        numImages, numPatches, system.size());
      return run_video_on_combined_handle(h, prompt, system, pixelValues,
                                          numPatches, numImages, callback,
                                          user_data);
    }
#endif
  }

  // Fallback (QNN [vision, LLM] pair): delegate to the single-image path.
  LOGD("[DEBUG] runMultimodalMultiImageHandleWithMessagesStreaming: delegating "
       "to single-image runMultimodalHandleWithMessagesStreaming");

  return runMultimodalHandleWithMessagesStreaming(
    handle, messages, num_messages, add_generation_prompt, pixelValues,
    numPatches, originalHeights[0], originalWidths[0], callback, user_data);
}

} // extern "C"
