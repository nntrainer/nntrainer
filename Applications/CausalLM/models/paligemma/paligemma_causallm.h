// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   paligemma_causallm.h
 * @brief  PaliGemma (Gemma text decoder) causal language model implementation.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __PALIGEMMA_CAUSAL_LM_H__
#define __PALIGEMMA_CAUSAL_LM_H__

#include <cmath>

#include "prefix_attention.h"
#include <causal_lm.h>
#include <llm_util.hpp>

namespace causallm {

/**
 * @brief PaliGemmaTransformer class — the Gemma1 text-decoder half of
 *        PaliGemma (google/paligemma-3b-pt-224).
 *
 * PaliGemma is a prefix-LM: the image prefix (bos + 256 image embeddings)
 * attends bidirectionally, the text suffix is causal. The graph is built with
 * is_causal=false so ONE bidirectional prefill forward matches the HF prefix
 * mask over the prompt rows, and the 1-row decode steps (seq_len == 1) read
 * the whole cache, which is exactly the HF causal suffix. The prefill
 * orchestration (bidirectional prefix block, then per-token causal suffix
 * rows) lives in PaliGemmaCausalLM::run_with_embeddings(); a prompt with no
 * suffix (prefix-only, e.g. the tiny tests' all-prefix token_type_ids) is one
 * bidirectional forward, same as HF.
 *
 * The prompt-text-as-prefix caveat of the plain CausalLM::run() path (no
 * image): the whole prompt is prefilled bidirectionally there, which matches
 * HF only when the caller marks the whole prompt as prefix (token_type_ids
 * zeros). The image-conditional path via run_with_embeddings() is exact.
 */
class PaliGemmaTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "PaliGemmaTransformer";

  PaliGemmaTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {
    HALF_ACTIVATIONS =
      nntr_cfg.at("model_tensor_type").get<std::string>().find("-FP16") !=
      std::string::npos;
    PREFILL_ONLY = nntr_cfg.value("prefill_only", false);
    WEIGHT_TRANSPOSED = nntr_cfg.value("weight_transposed", false);
    FUSE_MLP = nntr_cfg.value("fuse_mlp", false);
    FUSE_QKV = nntr_cfg.value("fuse_qkv", FUSE_MLP);
    CPU_EXTERNAL_HEAD = nntr_cfg.value("cpu_external_head", false);
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(DIM));
    USE_EMBEDDING = nntr_cfg.contains("use_embedding")
                      ? nntr_cfg["use_embedding"].get<bool>()
                      : false;
    if (CPU_EXTERNAL_HEAD &&
        (causallm_engine() != "cpu" || !USE_EMBEDDING || HALF_ACTIVATIONS ||
         !TIE_WORD_EMBEDDINGS || EMBEDDING_DTYPE != "FP32" ||
         nntr_cfg.value("lmhead_dtype", "FP32") != "FP32"))
      throw std::invalid_argument(
        "cpu_external_head requires a tied FP32 embedding-bypass CPU model");
    EMBEDDING_BIN_PATH = nntr_cfg.contains("embedding_bin_path")
                           ? nntr_cfg["embedding_bin_path"].get<std::string>()
                           : "";
  }

  virtual ~PaliGemmaTransformer() = default;

  std::vector<std::string> getGraphReplayFeedNodes() const override {
    if (USE_EMBEDDING && HALF_ACTIVATIONS && !PREFILL_ONLY)
      return {"input0", "input0/generated_out_0"};
    return {};
  }

protected:
  /**
   * @brief Flatten PaliGemma's nested config into the text-decoder view
   *
   * config.json carries text_config (HF PaliGemmaConfig) and omits several
   * keys the Gemma1 decoder needs; this fills them from the architecture
   * defaults so the plain setupParameters parsers see one flat object:
   * text_config keys are merged up (existing top-level keys win), then
   * rms_norm_eps 1e-6, rope_theta 10000, tie_word_embeddings true and
   * head_dim hidden/heads are defaulted, is_causal is forced false (the
   * prefix-LM prefill above) and num_image_tokens is dropped (the vision
   * half owns it).
   */
  static json &sanitizeConfig(json &cfg);

  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  /**
   * @brief Embedding bypass switches (same contract as Lfm2Transformer):
   *        with use_embedding=true the graph input is the pre-computed
   *        embedding plane and prompt rows come from EMBEDDING_BIN_PATH.
   */
  bool USE_EMBEDDING = false;
  bool HALF_ACTIVATIONS = false; /**< Graph input activation dtype */
  bool CPU_EXTERNAL_HEAD = false;
  bool FUSE_MLP = false;
  bool FUSE_QKV = false;
  bool WEIGHT_TRANSPOSED = false; /**< Checkpoint FC order [output,input] */
  LayerHandle createModelLayer(const std::string &type,
                               std::vector<std::string> properties);
  bool PREFILL_ONLY = false; /**< pi0.5 backbone has no vocabulary projection */
  std::string EMBEDDING_BIN_PATH; /**< Standalone embedding bin path */

public:
  std::vector<std::shared_ptr<PrefixKV>> prefix_cache_;
  Tensor createAttention(int layer_id, int seq_len, int heads, int head_dim,
                         Tensor query, Tensor key, Tensor value) override;

public:
  const PrefixKV &prefixKV(unsigned int layer) const {
    return *prefix_cache_.at(layer);
  }
  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;
};

/**
 * @brief PaliGemmaCausalLM class — Gemma text decoder with the embedding
 *        bypass needed for the multimodal composition.
 *
 * When nntr_config sets "use_embedding": true, the graph input becomes the
 * pre-computed embedding plane [batch, seq, dim] and the caller drives
 * generation through run_with_embeddings(), feeding [emb(bos), image embeds,
 * emb(text)...] produced by SiglipVisionModel + the lookupEmbedding() table.
 * The HF 1-indexed positions need no handling here: RoPE is relative, so the
 * one-position offset between HF and the cache-relative positions cancels in
 * every attention score.
 */
class PaliGemmaCausalLM : public CausalLM, public PaliGemmaTransformer {

public:
  static constexpr const char *architectures = "PaliGemmaForCausalLM";

  PaliGemmaCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    PaliGemmaTransformer(sanitizeConfig(cfg),
                         sanitizeGenerationConfig(generation_cfg, cfg),
                         nntr_cfg) {}

  virtual ~PaliGemmaCausalLM() = default;

  /**
   * @brief Construct the Gemma decoder graph (embedding0 always present,
   *        disconnected from the data path when use_embedding is on).
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Run the model with a text prompt.
   *
   * When USE_EMBEDDING=false this is the plain token-id path; with
   * USE_EMBEDDING=true the prompt is tokenized, embedded through
   * lookupEmbedding() (text tokens only — image rows must come from the
   * vision model) and generation runs through run_with_embeddings() with the
   * whole prompt treated as the prefix block.
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  /**
   * @brief Run prefill + generation from in-memory embeddings.
   *
   * Overload of the Transformer composition interface that adds @p
   * prefix_len: the FIRST prefix_len rows are prefilled in one bidirectional
   * forward (the image+bos prefix block, is_causal=false), the remaining
   * n_tokens - prefix_len rows are fed one at a time (causal suffix), and
   * generation starts from the last suffix row's logits. Passing prefix_len
   * == n_tokens keeps the LFM2 all-bidirectional prefill behaviour.
   *
   * @param inputs_embeds Pointer to merged embeddings [B * n_tokens * DIM]
   * @param n_tokens Number of prompt rows in inputs_embeds (<= INIT_SEQ_LEN)
   * @param prefix_len Rows prefilled as the bidirectional prefix block
   * @param seed_tokens Prompt token ids (repetition penalty bookkeeping)
   */
  void run_with_embeddings(const void *inputs_embeds, size_t n_tokens,
                           size_t prefix_len, std::vector<int> seed_tokens,
                           bool do_sample, bool log_output);

  /**
   * @copydoc Transformer::run_with_embeddings — delegates with the whole
   *          prompt as the prefix block (all-bidirectional prefill).
   */
  void run_with_embeddings(const void *inputs_embeds, size_t n_tokens,
                           std::vector<int> seed_tokens, bool do_sample,
                           bool log_output) override {
    run_with_embeddings(inputs_embeds, n_tokens, n_tokens,
                        std::move(seed_tokens), do_sample, log_output);
  }

  /**
   * @brief Look up the embedding vector for a given token ID (scaled by
   *        EMBEDDING_SCALE, consistent with the embedding layer).
   */
  std::vector<float> lookupEmbedding(unsigned int token_id) const;

  /**
   * @brief Multimodal composer interface (embedding consumer, see
   *        Transformer): bytes of one merged embedding-plane row, and a
   *        pointer to the scaled row for @p token_id (nullptr when the
   *        table is not loaded or the id is out of range).
   */
  size_t embeddingBytesPerToken() const override;
  const void *lookupEmbedding(int token_id) const override;

  /**
   * @brief Forward ONLY the merged embedding plane as one bidirectional
   *        prefix block and leave the resulting per-layer K/V in this
   *        model's KV cache. Used by the pi0.5 backbone producer: the
   *        PaliGemmaCausalLM cache rows [0, n_tokens) then hold exactly the
   *        prefix K/V openpi computes once in sample_actions(). No
   *        generation, no tokenizer, no output reads.
   *
   * @param inputs_embeds merged plane [n_tokens, DIM] (image rows raw,
   *                      text rows scaled by EMBEDDING_SCALE)
   */
  void prefillEmbeddings(const float *inputs_embeds, size_t n_tokens);

  /**
   * @brief Access this model's KV cache (pi0.5 backbone producer reads the
   *        prefix K/V rows out of it after prefillEmbeddings()).
   */
  causallm::KVCacheManager &kvCache();

  /**
   * @brief Get the generated token IDs from the last run_with_embeddings.
   */
  const std::vector<unsigned int> &getGeneratedIds() const {
    return generated_ids_;
  }

  /** @brief Last prompt row's logits, before sampling or penalties. */
  const std::vector<float> &prefillLogits() const { return prefill_logits_; }

  /**
   * @brief Load model weights and, with the embedding bypass on, cache the
   *        standalone embedding table for lookupEmbedding().
   */
  void load_weight(const std::string &weight_path) override;

private:
  std::vector<unsigned int>
    generated_ids_;                   /**< Generated token IDs from last run */
  std::vector<float> prefill_logits_; /**< Unmodified prefill logits */

  /** Embedding weight cache for lookupEmbedding() */
  bool embedding_weight_cached_ = false;
  nntrainer::TensorDim::DataType embedding_weight_dtype_ =
    nntrainer::TensorDim::DataType::FP32;
  std::vector<float> embedding_weight_cache_; /**< FP32 embedding weights */
  std::vector<uint8_t>
    embedding_weight_cache_uint8_; /**< Quantized embedding weights */
  size_t embedding_row_bytes_ = 0; /**< Bytes per row for quantized types */

  /** Scratch row backing the base-class const lookupEmbedding() override */
  mutable std::vector<float> lookup_scratch_;
  std::function<void(const std::vector<float> *)> bind_cpu_head_;

  /**
   * @brief Load embedding weights from EMBEDDING_BIN_PATH into cache
   *        (FP32/Q4_0/Q6_K, same format contract as LFM2).
   */
  void loadEmbeddingWeight();
};

} // namespace causallm

#endif /* __PALIGEMMA_CAUSAL_LM_H__ */
