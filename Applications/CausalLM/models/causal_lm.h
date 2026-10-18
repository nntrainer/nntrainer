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
#include <token_sampler.h>
#include <transformer.h>

#include <atomic>
#include <cstdint>
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

  /**
   * @brief Model-teardown hook for the process-global device caches the
   *        CUDA and OpenCL lanes key by HOST pointer or handle value (CUDA:
   *        derived FC weights, scale side buffers, the KV mirror and split-KV
   *        scratch, the decode graph; OpenCL: the v8c weight packs and their
   *        aux arena, the lm_head residency, the RoPE LUTs and the V image
   *        view). Call after the model objects of a handle are destroyed and
   *        before the next load: a second load in the same process otherwise
   *        lands its weights on recycled addresses and takes stale cache hits.
   *        Pure reset -- every cache is rebuilt lazily or by the next load's
   *        prewarm. No driver call for a lane the process never used.
   */
  static void releaseDeviceCaches();

protected:
  bool htp_execution_profile_ = false;
  /** Model-local preparation immediately before a timed prefill block.
   * The default is empty; loaders and decode steps never call this hook. */
  virtual void beforePrefillBlock(unsigned int, unsigned int) {}
  /**
   * @brief One-shot CUDA prewarm guard, PER MODEL OBJECT. A function-local
   *        static was process-lifetime, so the second load of a load/destroy
   *        loop skipped the eager derived-weight prewarm.
   */
  bool cuda_prewarmed_ = false;

public:
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
   * @brief Load the weights, and build the KV cache while they load.
   * @details The KV cache used to be allocated and bound inside the first
   * run(), i.e. inside TTFT. Its size does not depend on the prompt unless the
   * prefill plane can still grow (prefillPlaneCanGrow()), so in every other
   * case it is allocated here, on a side thread that overlaps the weight load,
   * and bound once the load is done. NNTR_KV_EARLY=0 (or
   * setPrepareRunAtLoad(false)) restores the lazy allocation in run();
   * NNTR_KV_EARLY=sync allocates after the load on the calling thread.
   */
  void load_weight(const std::string &weight_path) override;

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
   * @brief Restore a saved KV cache AND position the session on it, so the
   *        next run() continues the cached conversation instead of prefilling
   *        it again.
   * @param path file written by save_kvcache()
   * @param token_len absolute token position to resume at, or 0 to take the
   *        position the file records (only a file with a header has one)
   * @return the absolute token position the session now holds
   * @throw std::runtime_error naming the field that disagreed when the file
   *        does not belong to this model
   * @details The difference from setPrecomputedKVCache(path, len), which arms a
   *          reload for the next run(): this one loads NOW, so a bad file is a
   *          failure of the call that named it rather than of the next
   *          generation, and getKvLen() answers correctly straight away.
   *          Positioning goes through the session token position (not
   *          SYS_PROMP_LEN), which is the one counter run() accumulates -- so a
   *          run AFTER the resumed one continues from the true position instead
   *          of rewinding onto the restored prefix.
   */
  WIN_EXPORT unsigned int resumeFromKVCacheFile(const std::string &path,
                                                unsigned int token_len);

  /**
   * @brief Continue the next run() at absolute token position @p pos in the KV
   *        cache this model already holds, without reloading anything.
   * @param pos absolute token position; 0 is "start a fresh conversation"
   * @details This is the in-memory half of resume: the `prev_idx` an SDK
   *          consumer passes back after a pause, or the position a multi-turn
   *          caller continues from. It only moves the bookkeeping -- the caller
   *          asserts the cache really holds a valid prefix of that length,
   *          which getKvLen() is how it knows.
   * @note Also disarms any armed precomputed-KV reload, so the two ways of
   *       saying "resume" cannot both apply to one run, and retires a captured
   *       decode graph (see the definition for why that is not optional).
   */
  WIN_EXPORT void setSessionTokenPosition(unsigned int pos);

  /**
   * @brief Absolute token position the next run() would continue from.
   * @note Equal to getKvLen() after a completed run; the two differ only while
   *       a resume has been armed but not yet consumed.
   */
  unsigned int getSessionTokenPosition() const {
    return SYS_PROMP_LEN + global_token_len;
  }

  /**
   * @brief Opaque identity written into a saved KV cache and checked when one
   * is loaded: everything about this model that decides how to read a KV plane.
   * @details Geometry alone is not enough. Two of this tree's models -- one
   * with 1 KV head x 256, one with 2 x 128 -- have the SAME per-layer KV width
   * and the same layer count, so a length check passes between them and the
   * loaded bytes are simply another model's attention state.
   */
  WIN_EXPORT std::string kvCacheModelTag() const;

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
   * @note  Also moves the session to 0, so it retires a captured decode graph
   *        for the same reason setSessionTokenPosition() does.
   */
  WIN_EXPORT void setPrecomputedKVCache(const std::string &path,
                                        unsigned int sys_prompt_token_len);

  /**
   * @brief The sampling knobs applyTKP() consumes, as a settable group.
   * @details Same three values generation_config.json seeds at construction
   *          (Hugging Face's temperature 1.0 / top_k 50 / top_p 1.0 when a key
   *          is missing). Held here
   *          rather than passed down through run() so that a host API can
   *          express "this request is creative, the next one is not" without
   *          every model's run() signature growing a parameter.
   */
  struct SamplingParams {
    float temperature;
    unsigned int top_k;
    float top_p;
  };

  /**
   * @brief Replace the sampling knobs for subsequent run() calls.
   * @note  Only consulted when run(..., do_sample=true ...): greedy decoding
   *        ignores all three. A temperature <= 1e-5 degenerates to argmax
   *        inside applyTKP(), which is the documented way to ask for
   *        "sampling requested but deterministic".
   * @note  Not synchronized: call between runs, not during one (the same
   *        contract as setStreamer / setLogitsProcessor).
   */
  void setSamplingParams(const SamplingParams &params) {
    TEMPERATURE = params.temperature;
    TOP_K = params.top_k;
    TOP_P = params.top_p;
  }

  /**
   * @brief The sampling knobs currently in effect (the config's values until
   *        someone calls setSamplingParams()).
   */
  SamplingParams getSamplingParams() const {
    return SamplingParams{TEMPERATURE, TOP_K, TOP_P};
  }

  /**
   * @brief Whether the model config asks for sampling.
   * @details generation_config.json "do_sample"; false when the key is
   *          missing (the Hugging Face default). Hosts that do not choose a
   *          decoding mode themselves pass this to run().
   */
  bool getDoSample() const { return DO_SAMPLE; }

  /**
   * @brief The seed every run() starts from unless setSamplingSeed() was
   *        called for that run: generation_config.json "seed", else
   *        nntr_config.json "seed", else causallm::kDefaultSamplingSeed.
   */
  uint64_t getSamplingSeed() const { return SAMPLING_SEED; }

  /**
   * @brief Seed the sampling RNG for the next run() only.
   * @details Every run() re-seeds the RNG with getSamplingSeed() before it
   *          decodes, so the same prompt + params give the same tokens on any
   *          run number and in any process. Calling this before a run
   *          replaces that seed for that one run; the run after it goes back
   *          to getSamplingSeed(). Inert for greedy decoding.
   */
  void setSamplingSeed(uint64_t seed) {
    rng.seed(seed);
    sampling_seed_pending_ = true;
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
  bool DO_SAMPLE = false; /**< generation_config.json "do_sample" */
  float TEMPERATURE;
  unsigned int TOP_K;
  float TOP_P;
  uint64_t SAMPLING_SEED = causallm::kDefaultSamplingSeed; /**< per-run seed */

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

  causallm::SamplingRng rng{causallm::kDefaultSamplingSeed}; /**< sampler RNG */
  bool sampling_seed_pending_ = false; /**< setSamplingSeed() for next run */

  /**
   * @brief Re-seed the sampling RNG at the start of a run, unless the caller
   *        seeded it for this run with setSamplingSeed().
   */
  void beginSamplingRun() {
    if (!sampling_seed_pending_)
      rng.seed(SAMPLING_SEED);
    sampling_seed_pending_ = false;
  }

  LogitsProcessor *logits_processor = nullptr; /**< Non-owning processor */

  /**
   * @brief Externalized KV cache (host-owned). Allocated by allocateKVCache()
   *        once the model has been compiled (so we have the layer count,
   *        head count, etc.) and bound to mha_core's input slots
   *        cache_k_l<i> / cache_v_l<i> by allocateAndBindKVCache().
   */
  KVCacheManager kv_cache;
  bool kv_cache_bound = false; /**< True once KV cache tensors are bound */

  /**
   * @brief Allocate kv_cache and bind it to all mha_core layers by pointing
   *        each cache placeholder at its buffer (Tensor::setData).
   *        Idempotent — safe to call once after initialize().
   */
  virtual void allocateAndBindKVCache();

  /**
   * @brief Before a step at absolute position pos: the windowed layers of
   *        kv_cache move their window to the front (KVCacheManager::compact)
   *        and the HTP attention layers get the new row bases.
   * @param[in] pos absolute position of the next row
   * @param[in] force hand the bases over even when none moved (after a load)
   */
  void compactKVCache(unsigned int pos, bool force = false);

  /**
   * @brief Per-layer KV geometry, read from the model's hooks.
   */
  struct KVCacheLayout {
    std::vector<unsigned int> widths; /**< per-layer K/V row width */
    unsigned int max_timestep = 0;    /**< MAX_SEQ_LEN */
  };

  /**
   * @brief Read the KV geometry hooks and declare the per-layer row caps and
   *        the KV-sharing map on kv_cache. Model-side only, no allocation.
   */
  KVCacheLayout planKVCache();

  /**
   * @brief Allocate (and zero) kv_cache for @a layout. Touches only kv_cache,
   *        so load_weight() can run it on a side thread.
   */
  void allocateKVCacheStorage(const KVCacheLayout &layout);

  /**
   * @brief incremental_inference wrapper that feeds the REAL KV-cache tensors
   *        (with their original MemoryData, isSVM() intact) into the graph's
   *        input placeholders instead of letting the framework re-wrap the
   *        raw pointers in fresh (flag-less) Tensor::Map MemoryData.
   * @details With in-place input layers (InputLayer::initializeInPlace) the
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
