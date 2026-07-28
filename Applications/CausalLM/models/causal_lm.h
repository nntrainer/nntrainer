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
   * @brief Build the output_of_causallm lm_head from the transformer hidden
   *        state @p h. Shared by the generic and the model-override
   *        constructModel paths. The skip_prefill decision differs per path
   *        (generic gates on SKIP_PREFILL; Gemma4 on ENABLE_SKIP_PREFILL_OPT),
   *        so it is passed in rather than recomputed here.
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
   * @brief save kv cache
   */
  WIN_EXPORT virtual void save_kvcache(std::string path, int to);

  /**
   * @brief load kv cache
   */
  WIN_EXPORT virtual void load_kvcache(std::string path, int to);

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

  std::vector<int> pending_ids_;

  ::BaseStreamer *streamer_ = nullptr;
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> stop_prepared_for_run_{false};

  std::string LMHEAD_DTYPE; /** embedding dtype */
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
   * @brief Same contract as NeuralNetwork::incremental_inference(float* ...),
   *        except that an input whose raw pointer belongs to a KVCacheManager
   *        cache tensor is fed as the REAL tensor (sharing its MemoryData) so
   *        that isSVM(), set when the cache comes from the gpu-svm MemoryPool,
   *        survives the per-call fillPlaceholder()/syncDependents().
   *
   * A fresh Tensor::Map of the same host pointer builds a brand-new
   * MemoryData with svm_allocation=false; fillPlaceholder() then installs
   * that MemoryData on the placeholder and propagates it to every dependent
   * view (mha_core's cache_k/cache_v views are dependents of the input
   * placeholder). mha_core gates its whole GPU/flash attention chain on
   * `svm_ok` = q && k && v && o all isSVM(), so the clobbered flag silently
   * drops attention onto the host GEMM, which then reads a device-written
   * K/V plane through host pointers.
   *
   * @param batch_size batch size
   * @param input input buffers, in getInputDimension() order
   * @param init_seq_len initial sequence length
   * @param from start position of this step
   * @param to end position of this step
   * @return newly allocated host buffers holding the model outputs; the
   *         caller owns them (delete[]), exactly as with the base API.
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
