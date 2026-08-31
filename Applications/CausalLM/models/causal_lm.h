// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 * Copyright (C) 2025 Seungback Hong <sb92.hong@samsung.com>
 * Copyright (C) 2025 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @brief  Base class for Transformer-based Causal Language Models (CausalLM).
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This causal_lm.h constructs a class for Transformer-based Causal
 * Language Model (CausalLM). It aims to support AutoModelForCausalLM with
 * nntrainer. It supports the following models:
 *          - Qwen3
 *          - Qwen3-MoE
 * @note   This CausalLM assumes the Decoder-based model, which structure is
 *
 *        [Transformer]
 *              |
 *           [LMHead]
 */

#ifndef __CAUSAL_LM_H__
#define __CAUSAL_LM_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::string
#define WCHAR_P std::string &
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <kv_cache_manager.h>
#include <transformer.h>

#include <atomic>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
struct BaseStreamer;
}

namespace causallm {

/**
 * @brief CausalLM Class
 */
WIN_EXPORT class CausalLM : virtual public Transformer {

public:
  /**
   * @brief Construct a new CausalLM object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation
   * (generation_config.json)
   * @param nntr_cfg Configuration for nntrainer (nntrainer_config.json)
   */
  CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);

#ifdef ENABLE_TEST
protected:
  /**
   * @brief Construct a lightweight CausalLM test double base.
   */
  CausalLM() : Transformer() { output_list.push_back(""); }

public:
#endif

  /**
   * @brief Destroy the CausalLM object
   */
  virtual ~CausalLM() {
    if (ids_history)
      free(ids_history);
    for (auto &kv : logits_pool_sizes_)
      delete[] kv.first;
  }

  /**
   * @brief run the CausalLM model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  /**
   * @brief Get the generated output text
   * @param batch_idx Index of the batch item
   * @return Generated text string
   */
  std::string getOutput(int batch_idx = 0) const;

  /**
   * @brief Attach or detach a non-owning streamer for decoded output deltas.
   * @param streamer Streamer owned by the caller, or nullptr to detach
   */
  void setStreamer(::BaseStreamer *streamer) { streamer_ = streamer; }

  /**
   * @brief Cooperatively request the active generation loop to stop.
   */
  void requestStop() { stop_requested_.store(true, std::memory_order_release); }

  /**
   * @brief Clear stale stop requests before publishing a new cancellable run.
   */
  void prepareForRun();

  /**
   * @brief Attach a non-owning logits processor
   * @param processor Processor pointer, or nullptr to detach
   */
  void setLogitsProcessor(LogitsProcessor *processor) override;

  /**
   * @brief Reset attached logits processor state
   */
  void resetLogitsProcessor() override;

  /**
   * @brief Current KV-cache write position (absolute token position)
   */
  int getKvLen() const override {
    return static_cast<int>(kv_cache.getPosition());
  }

  /**
   * @brief save kv cache (all layers, first @p to positions) to @p path
   */
  WIN_EXPORT virtual void save_kvcache(std::string path, int to);

  /**
   * @brief load kv cache from @p path and sync every layer's cache index
   *        to position @p to
   */
  WIN_EXPORT virtual void load_kvcache(std::string path, int to);

  /**
   * @brief Arm (or disarm) resume-from-saved-KV for the next run() call.
   * @param path Saved KV-cache file produced by save_kvcache(); an empty
   *             string disarms and restores plain-prefill behavior.
   * @param sys_prompt_token_len Absolute token position the cache was saved
   *             at; the next run() reloads the cache and prefills the new
   *             prompt starting from this offset.
   * @note  Drives the same USE_KVCACHE flow that nntr_config.json's
   *        system_prompt.kvcache block configures, without needing the
   *        config entry.
   */
  void setPrecomputedKVCache(const std::string &path,
                             unsigned int sys_prompt_token_len) {
    USE_KVCACHE = !path.empty();
    PRE_COMPUTED_CACHE_PATH = path;
    SYS_PROMP_LEN = USE_KVCACHE ? sys_prompt_token_len : 0;
    global_token_len = 0;
  }

protected:
  /**
   * @brief Setup the parameters for the CausalLM model
   */
  virtual void setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) override;

  /**
   * @brief Construct Model — extends Transformer's symbolic graph with the
   *        LM-head and returns the final {input, logits} pair.
   */
  virtual std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Build the output_of_causallm lm_head layer from the transformer
   *        hidden state and return its output tensor.
   * @param h hidden state produced by the transformer body
   * @param add_skip_prefill append the skip_prefill layer property. The gate
   *        differs per constructModel path (the generic path derives it from
   *        the model-level SKIP_PREFILL runtime flag; the diamond-inheritance
   *        models from their own skip-prefill option), so it is passed in
   *        rather than recomputed here.
   */
  Tensor buildLmHeadOutput(Tensor h, bool add_skip_prefill);

  /**
   * @brief register Outputs
   */
  virtual void
  registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                  std::vector<unsigned int> ids, unsigned int pos,
                  const std::vector<bool> &eos_list, bool log_output = true);

  /**
   * @brief generate
   */
  std::vector<unsigned int> generate(float *logits, bool do_sample,
                                     float repetition_penalty = 1,
                                     unsigned int *input_ids = nullptr,
                                     unsigned int NUM_INPUT_IDS = 0);

  /**
   * @brief registerCutomLayers
   */
  void registerCustomLayers() override;

  /**
   * @brief Clear stale stop state at run start unless caller prepared it.
   */
  void prepareStopRequestForRun();

  /** internal buffer */
  std::vector<std::string>
    output_list; /**< List of output names for the model */
  unsigned int *ids_history =
    nullptr; /**< History of input IDs for the model */

  /**
   * @brief Recycled host staging buffers for incrementalInference outputs.
   * @details Decode allocated (and mostly never even touched, under the
   *          deferred-logits path) a fresh 1MB float row EVERY token and
   *          run() freed it right after -- a per-token mmap/munmap pair.
   *          acquireLogitsBuf() hands back a previously released buffer of
   *          the same element count instead; releaseLogitsBuf() returns a
   *          buffer to the free list (or plain delete[]s a pointer the pool
   *          does not own). Every in-tree release site is converted together:
   *          a plain delete[] on a pooled pointer would corrupt the ownership
   *          map. All pool storage is freed in the destructor.
   */
  std::unordered_map<float *, size_t> logits_pool_sizes_;
  std::vector<std::pair<size_t, float *>> logits_pool_free_;

  /** @brief Pop a same-size recycled buffer or allocate a new pooled one. */
  float *acquireLogitsBuf(size_t count);
  /** @brief Return a pooled buffer to the free list (delete[]s foreign ptrs).
   */
  void releaseLogitsBuf(float *buf);

  std::vector<int> pending_ids_;

  ::BaseStreamer *streamer_ = nullptr;
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> stop_prepared_for_run_{false};

  std::string LMHEAD_DTYPE; /** embedding dtype */
  // LMHEAD_UNTIE moved to Transformer: embedding0's layer-type choice (tied
  // TieWordEmbedding vs untied embedding_layer) needs it in
  // <model>Transformer::constructModel scope, which does not see CausalLM
  // members (the diamond joins only at <Model>CausalLM).
  std::vector<unsigned int> EOS_TOKEN_ID;
  unsigned int BOS_TOKEN_ID;
  float TEMPERATURE;
  unsigned int TOP_K;
  float TOP_P;

  std::vector<unsigned int> BAD_WORD_IDS; /**< List of bad word IDs */
  unsigned int NUM_BADWORDS;              /**< Number of bad words */

  unsigned int SYS_PROMP_LEN;
  std::string PRE_COMPUTED_CACHE_PATH;
  bool SAVE_KVCACHE;
  bool USE_KVCACHE;
  bool SKIP_PREFILL;
  /**
   * @brief nntr_config.json "repetition_penalty". Divides positive logits of
   *        already-generated tokens (multiplies negative ones), so > 1
   *        discourages repeats. 1.0 -- the default, and what a config without
   *        the key gets -- is the identity transform and leaves the greedy
   *        fast path in generate() untouched.
   */
  float REPETITION_PENALTY;
  /**
   * @brief nntr_config.json "repetition_window": how many of the most recently
   *        generated tokens REPETITION_PENALTY is applied to. Ignored while
   *        REPETITION_PENALTY == 1.
   */
  unsigned int REPETITION_WINDOW;
  unsigned int global_token_len;

  std::mt19937 rng; /**< Random Number Gen */

  LogitsProcessor *logits_processor = nullptr; /**< Non-owning processor */

  /**
   * @brief Externalized KV cache (host-owned). Allocated by allocateKVCache()
   *        once the model has been compiled (so we have the layer count,
   *        head count, etc.) and bound to mha_core's input slots
   *        cache_k_l<i> / cache_v_l<i> via Model::setExternalTensors.
   */
  KVCacheManager kv_cache;
  bool kv_cache_bound = false; /**< True once KV cache tensors are bound */

  /**
   * @brief Allocate kv_cache and bind it to all mha_core layers via
   *        Model::setExternalTensors. Idempotent — safe to call once after
   *        initialize().
   */
  virtual void allocateAndBindKVCache();

  /**
   * @brief incremental_inference wrapper that feeds the REAL KV-cache tensors
   *        (with their original MemoryData, isSVM() intact) into the graph's
   *        input placeholders instead of letting the framework re-wrap the
   *        raw pointers in fresh (flag-less) Tensor::Map MemoryData.
   * @details With in-place input layers (NNTR_INPUT_INPLACE relaxation) the
   *          mha_core cache input views alias the input placeholder directly
   *          (view-of-view flattening), so whatever MemoryData fills the
   *          placeholder reaches mha_core's svm_ok gate. A Map wrap of the
   *          same pointer would report isSVM()=false and silently kill the
   *          GPU attention path. Non-cache inputs (the prompt sample) keep
   *          the framework's Map wrapping, byte-identical to
   *          Model::incremental_inference(float* ...).
   */
  std::vector<float *> incrementalInference(unsigned int batch_size,
                                            const std::vector<float *> &input,
                                            unsigned int init_seq_len,
                                            unsigned int from, unsigned int to);

  /**
   * @brief Reset all mha_core layers' cache_index to @p pos and the
   *        KVCacheManager's tracked write position.
   */
  void setKVCachePosition(unsigned int pos);

  /**
   * @brief Advance all mha_core layers' cache_index by @p step_size and
   *        update the KVCacheManager's tracked write position.
   */
  void advanceKVCachePosition(unsigned int step_size);
};

} // namespace causallm

#endif
