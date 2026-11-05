// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_callbacks.h
 * @brief  Per-architecture callback registry bridging proprietary model TUs
 *         to the public C API.
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "quick_dot_ai_api.h" // ErrorCode, CausalLmTokenCallback, CausalLmHandle

namespace causallm {
class Transformer;
}

/**
 * Type-erased image buffer: the same type as causallm::multimodal_pointer
 * (buffer, size in bytes). Spelled out so this header does not need the engine
 * headers.
 */
using MultimodalBuffer = std::pair<void *, size_t>;

/** Where the image rows sit in the combined [text, image, text] sequence. */
struct MultimodalSpan {
  size_t first_image_row; /**< index of the first image row in the sequence */
  size_t image_rows;      /**< number of image rows */
};

/**
 * @brief Extension points of the model-agnostic multimodal composer.
 *
 * The host composer drives a [vision producer, LLM consumer] pair through the
 * base Transformer virtuals only (run_image / lookupEmbedding /
 * run_with_embeddings). A model that needs more than that -- a retained
 * per-layer feature stream, delimiter tokens around the image span, a
 * different per-position token id for some image rows -- supplies it here,
 * without the host knowing the concrete type. Every member is optional; with
 * none registered the composer behaves exactly like the plain pair.
 *
 * Producer-side members are looked up under models[0]'s architecture, consumer
 * side members under models[1]'s.
 */
struct MultimodalCallbacks {
  /** Producer side. Replaces Transformer::run_image: produce the image
   *  embeddings of `pixels` (ownership of the returned malloc'd buffer goes to
   *  the caller) and keep whatever side state the consumer needs. */
  std::function<MultimodalBuffer(
    causallm::Transformer *producer, causallm::Transformer *consumer,
    const std::string &prompt, MultimodalBuffer pixels, int image_height,
    int image_width, bool verbose)>
    encode_image;

  /** Producer side. `ids` holds image_rows entries, all `placeholder_id` (the
   *  image placeholder token, or -1 when the tokenizer has none); overwrite the
   *  rows whose per-position token id differs (e.g. newline rows of a tiled
   *  image). These ids select per-layer rows of the consumer. */
  std::function<void(causallm::Transformer *producer, size_t image_rows,
                     int placeholder_id, std::vector<int> &ids)>
    image_row_token_ids;

  /** Consumer side. Token names that delimit the image span; both must exist
   *  in the tokenizer, otherwise no delimiter is inserted. Empty = none. */
  std::string image_begin_token;
  std::string image_end_token;

  /** Consumer side. Called after the sequence is built and before the LLM
   *  runs, to bind the producer's retained state to the span. The returned
   *  closure, when set, is called once the run ends (also on error). */
  std::function<std::function<void()>(causallm::Transformer *producer,
                                      causallm::Transformer *consumer,
                                      const MultimodalSpan &span)>
    bind_image_span;
};

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
   * Multimodal extension points. See MultimodalCallbacks. Replaces the former
   * multimodal_streaming / multimodal_blocking slots, which no API entry point
   * ever called.
   */
  MultimodalCallbacks multimodal;

  /**
   * Enable speculative decoding on an already-loaded model instance.
   *
   * configureSpeculativeDecoding() walks EVERY sub-model of the handle, in
   * order, and uses the first one whose architecture has this callback
   * registered, so a vision+LLM handle ([vision producer, LLM, ...]) reaches
   * the LLM's callback. `model` is that sub-model. The registering TU casts it
   * to its own concrete type to confirm draft-model support before enabling.
   * Returns CAUSAL_LM_ERROR_MODEL_LOAD_FAILED if the model does not support
   * speculative decoding; when no sub-model has a callback the caller treats
   * it the same way.
   */
  std::function<ErrorCode(causallm::Transformer *model, bool use_sd)>
    configure_speculative_decoding;
};

/*
 * ModelCallbacks is passed BY VALUE to register_for(), from a plugin compiled
 * separately from the host. Count of std::function members + the one bool, each
 * member padded to the function alignment. Adding a member means updating this
 * AND bumping QDA_EXT_ABI_VERSION (ext_model_abi.h): the host also compares
 * sizeof(ModelCallbacks) reported by the plugin when it loads it.
 */
static_assert(sizeof(ModelCallbacks) ==
                (4 * sizeof(std::function<void()>) +
                 sizeof(MultimodalCallbacks) + alignof(std::function<void()>)),
              "ModelCallbacks layout changed: update this assert and bump "
              "QDA_EXT_ABI_VERSION");

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

/**
 * @brief Index of the sub-model that speculative decoding is configured on:
 *        the first one, in the handle's order, whose architecture has a
 *        configure_speculative_decoding callback registered; -1 when none has.
 *
 * A vision+LLM handle is [vision producer, LLM, ...], so looking at the first
 * model only would never reach the LLM. No model index is hard-coded.
 */
int findSpeculativeDecodingModel(const std::vector<std::string> &architectures,
                                 const ModelCallbackRegistry &registry);
