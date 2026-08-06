// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_callbacks.h
 * @brief  Per-architecture callback registry bridging proprietary model TUs
 *         to the public C API.
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include <functional>
#include <string>
#include <unordered_map>

#include "quick_dot_ai_api.h" // ErrorCode, CausalLmTokenCallback, CausalLmHandle

namespace causallm {
class Transformer;
}

/**
 * @brief Per-architecture callbacks registered by proprietary model TU files.
 * When a proprietary TU is absent (public build), no callbacks are
 * registered for that architecture; callers should fall back to
 * CAUSAL_LM_ERROR_UNSUPPORTED.
 */
struct ModelCallbacks {
  /**
   * Apply architecture-specific chat template to a raw single-turn input.
   * Returns empty string if not registered (caller uses raw input).
   */
  std::function<std::string(const std::string &raw_input)> format_prompt;

  /** True when this architecture requires an HTP/QNN backend. */
  bool requires_htp = false;

  /**
   * Read the current KV-cache length from a loaded transformer.
   * Used for incremental-session tracking.
   * Returns 0 if not registered.
   */
  std::function<int(causallm::Transformer *model)> read_kv_len;

  /**
   * Given the full prompt history (already-formatted), extract the latest user
   * content and rebuild it as the minimal incremental prompt for next turn.
   * Returns empty string if not registered.
   */
  std::function<std::string(const std::string &full_prompt)> incremental_prompt;

  /**
   * Streaming multimodal execution.
   * `handle` is CausalLmHandle (= CausalLmModel*).
   * The registering TU casts it to CausalLmModel* and accesses h.models[0]/[1].
   */
  std::function<ErrorCode(CausalLmHandle handle, const float *pixel_values,
                          int num_patches, int orig_h, int orig_w,
                          const std::string &prompt, CausalLmTokenCallback cb,
                          void *user_data)>
    multimodal_streaming;

  /**
   * Blocking multimodal execution; appends generated text to *output.
   * `handle` is CausalLmHandle (= CausalLmModel*).
   */
  std::function<ErrorCode(CausalLmHandle handle, const float *pixel_values,
                          int num_patches, int orig_h, int orig_w,
                          const std::string &prompt, std::string *output)>
    multimodal_blocking;

  /**
   * Enable speculative decoding on an already-loaded model instance.
   * `model` is the handle's primary sub-model (handle->models[0]). The
   * registering TU casts it to its own concrete type to confirm draft-model
   * support before enabling. Returns CAUSAL_LM_ERROR_MODEL_LOAD_FAILED if the
   * model does not support speculative decoding; if no callback is
   * registered for the architecture, the caller treats it the same way
   * (current no-op behavior for unsupported architectures).
   */
  std::function<ErrorCode(causallm::Transformer *model, bool use_sd)>
    configure_speculative_decoding;
};

/**
 * @brief Registry keyed by architecture name string (e.g. "VendorArch_QNN").
 * Proprietary model TUs call register_for() at static-init time.
 * quick_dot_ai_api.cpp calls lookup() at runtime.
 */
class ModelCallbackRegistry {
public:
  static ModelCallbackRegistry &instance();

  /** Register callbacks for one architecture name. */
  void register_for(const std::string &architecture, ModelCallbacks cb);

  /**
   * Look up callbacks for the given architecture.
   * Returns nullptr if not registered (public architecture, or the
   * proprietary TU that would register it is absent from this build).
   */
  const ModelCallbacks *lookup(const std::string &architecture) const;

  /** True if ANY registered architecture has requires_htp = true. */
  bool any_requires_htp() const;

private:
  ModelCallbackRegistry() = default;
  ModelCallbackRegistry(const ModelCallbackRegistry &) = delete;
  ModelCallbackRegistry &operator=(const ModelCallbackRegistry &) = delete;

  std::unordered_map<std::string, ModelCallbacks> by_arch_;
};
