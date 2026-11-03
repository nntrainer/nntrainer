// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   fp16_attention.h
 * @date   02 September 2024
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief Attention execution with an explicit FP16 KV-cache contract.
 * @note Used by the owning layer for prefill and decode. Backend handles and
 *       calibration belong to HtpAttentionState.
 */

#ifndef NNTR_FP16_ATTENTION_H
#define NNTR_FP16_ATTENTION_H

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <complex>
#include <memory>
#include <unordered_map>

#include <acti_func.h>
#include <attention_config.h>
#include <common_properties.h>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <fused_llm_properties.h>
#include <layer_impl.h>
#include <limits.h>
#include <map>
#include <util_simd.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nntrainer {
class HtpAttentionState;

namespace attention_props {
using namespace nntrainer::props;

using nntrainer::props::NumHeads_KV;

using nntrainer::props::SlidingWindow;

using nntrainer::props::MaxNewTokens;

using nntrainer::props::MaxPositionEmbeddings;

using nntrainer::props::UseRope;

using nntrainer::props::UseSink;

using nntrainer::props::AttnLogitSoftcapping;

/**
 * @brief KvCacheQuant: "" (off), "q8", "q8_row" or "q4". Keep an int8 / int4
 * mirror of this layer's KV cache on the accelerator and run attention over it
 * (ComputeOps::sdpa_q_kvcache) instead of over the fp16 cache. q8_row selects
 * per-row scales for prefill rather than the fixed-scale row-blocked kernel.
 * The fp16 cache stays the source of truth.
 */
class KvCacheQuant final : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "kv_cache_quant";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief AttnOutU16: the o-proj that consumes this layer's output runs on
 *        the same accelerator as a block call (gemm_qs4cx_res_add_fp32), so
 *        the quantized-cache attention may leave its u16 context there and
 *        never write the f32 output tensor (ComputeOps::sdpa_q2_kvcache_f32).
 *        The model builder sets it from the o-proj's engine; false (the
 *        default) keeps the f32 output on the host.
 */
class AttnOutU16 final : public nntrainer::Property<bool> {
public:
  AttnOutU16(bool value = false) { set(value); }
  static constexpr const char *key = "attn_out_u16";
  using prop_tag = nntrainer::bool_prop_tag;
};

using nntrainer::props::IsCausal;

using nntrainer::props::RopeScalingFactor;

using nntrainer::props::RopeScalingMaxPositionEmbeddings;

}; // namespace attention_props

/**
 * @brief Computes attention from projected Q/K/V and updates the FP16 cache.
 */
WIN_EXPORT class Fp16Attention : private AttentionConfig {
public:
  /**
   * @brief Register batch 0's quantized mirror after weight placement.
   * A refused registration is retried by the first attention call.
   */
  void registerQuantizedCache(nntrainer::RunLayerContext &context);

  WIN_EXPORT Fp16Attention();

  WIN_EXPORT ~Fp16Attention();

  /** Keep the execution state's reference to this object's config stable. */
  Fp16Attention(Fp16Attention &&rhs) = delete;

  Fp16Attention &operator=(Fp16Attention &&rhs) = delete;

  /** Validate input shapes and configure cache tensors. */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context, bool skip);

  /** Run attention with externally supplied cache tensors. */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training);

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim);

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step);
  /** Process prefill or decode rows in [from, to). */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training);

  /** Export attention properties for the owning layer. */
  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method,
                           const Layer *owner) const;

  /** Load attention properties and return unhandled layer properties. */
  WIN_EXPORT std::vector<std::string>
  setProperty(const std::vector<std::string> &values);

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch);

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions);

  /**
   * @brief Set the write position of the external KV cache. The host calls
   *        this when it repositions the cache -- after a load, a rewind, a
   *        session switch -- and any row of the fp16 cache may have been
   *        rewritten by then, so the quantized mirror
   * (attention_props::KvCacheQuant) is marked out of date in full and
   * re-appended from row 0 on the next step.
   */
  WIN_EXPORT void setCacheIndex(unsigned int idx);

  /**
   * @brief The absolute position of the external KV cache's row 0. A
   *        sliding layer's cache at a long context holds only its window
   *        and the step behind it (KVCacheManager::compact): the host moved
   *        the rows, so the quantized mirror is re-appended in full, like
   *        after setCacheIndex. Positions (cache_index, RoPE) stay absolute;
   *        only row numbers subtract the base.
   */
  WIN_EXPORT void setCacheBase(unsigned int base);

  /**
   * @brief |max| per column (kv head x head_dim) of the K
   *        and V rows the host's windowed cache dropped
   *        (KVCacheManager::compact), folded into the layer's floor by
   *        elementwise max. The decode seed hands the floor to the
   *        accelerator so its int8 scales cover the whole context, as the
   *        full cache's seed sets them; the q2 store keeps it beside the
   *        calibration, so a saved cache carries it.
   */
  WIN_EXPORT void foldDroppedAmax(const std::vector<float> &k_amax,
                                  const std::vector<float> &v_amax);

  /**
   * @brief Get the current cache index
   */
  WIN_EXPORT unsigned int getCacheIndex() const { return cache_index; }
  /**
   * @brief The fixed-scale int8 attention's calibration (calibrate_q2_scales)
   *        of every layer that has one, beside a saved KV cache: a run that
   *        loads the cache then quantizes with the scales the saving run
   *        used instead of calibrating again on its own (shorter) step, so
   *        it computes what the uninterrupted prefill would have.
   * @param[in] file stream to write: per layer the name and the floats
   */
  WIN_EXPORT static void saveQ2Calibration(std::ostream &file);

  /**
   * @brief Reads saveQ2Calibration's stream; a layer whose name and sizes
   *        match takes its scales from it at its first quantized step.
   * @param[in] file stream to read; a malformed one throws
   */
  WIN_EXPORT static void loadQ2Calibration(std::istream &file);

  /** Discard mirrors from the previous independent HTP prompt. */
  WIN_EXPORT void beginHtpPrompt();

  /** Whether the layer's CPU KV rows are gone (HtpAttentionState). */
  WIN_EXPORT bool kvDetached() const;

  /** See HtpAttentionState::setHtpPrefillSeed. */
  WIN_EXPORT void setHtpPrefillSeed(int ordinal, unsigned int decode_base);

private:
  std::tuple<
    nntrainer::props::NumHeads, attention_props::NumHeads_KV,
    nntrainer::props::ProjectedKeyDim, nntrainer::props::ProjectedValueDim,
    nntrainer::props::OutputShape, nntrainer::props::DropOutRate,
    nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep,
    attention_props::SlidingWindow, attention_props::MaxNewTokens,
    attention_props::RopeTheta, attention_props::UseRope,
    attention_props::MaxPositionEmbeddings, attention_props::UseSink,
    attention_props::RopeScalingType, attention_props::RopeScalingFactor,
    attention_props::RopePartialRotaryFactor,
    attention_props::RopeScalingMaxPositionEmbeddings,
    attention_props::AttnLogitSoftcapping, attention_props::IsCausal,
    attention_props::KvCacheQuant, attention_props::AttnOutU16>
    mha_core_props; /**< mha_core layer properties */

  /** softmax activation operation */
  nntrainer::ActiFunc sm;

  float epsilon;

  /**
   * @brief Whether to use externally provided cache tensors
   *        (true when num_inputs >= 5, i.e., Q, K, V + cache_key + cache_value)
   *        In external mode mha_core does not allocate its own cache tensors,
   *        and reads cache slots from input[3] (cache_key) and input[4]
   *        (cache_value) which are bound by the host via setExternalTensors.
   */
  bool use_external_cache = false;

  bool cache_shift;
  float theta;
  bool use_sink = false;
  bool skip_prefill = false;

  std::unique_ptr<HtpAttentionState> state_;

  enum INOUT_INDEX {
    /** input index */
    QUERY = 0,
    KEY = 1,
    VALUE = 2,
    MASK = 3,

    /** output index */
    OUTPUT = 0,
    RETURN_ATTENTION_WEIGHT = 1,
  };

  /**< indices of the weights and tensors */
  enum AttentionParams {
    cache_key,
    cache_value,
    projected_key,
    projected_value,
    attention_weight,
    dropout_mask,
    attention_output,
  };
  std::array<unsigned int, 7> tensor_idx;
  unsigned int sink_idx;

  /** attention parameters */
  unsigned int max_position_embeddings;

  /** rope_scaling parameters */
  std::string rope_scaling_type;
  float attention_scaling = 1.0f;
  float mscale = 1.0f;
  float scale = 1.0f;
  float rope_partial_rotary_factor = 1.0f;
  unsigned int original_max_position_embeddings = 4096;

  /** set by incremental_forwarding, used by forwarding */
  unsigned int incremental_step_size = 0;

  /****************** ROTARY EMBEDDING *****************/
  /**
   * @brief FP32 rotary embedding cache entries.
   */
  struct RopeCacheFP32 {
    std::vector<std::vector<float>> cos;
    std::vector<std::vector<float>> sin;
  };
  inline static std::unordered_map<std::string, std::shared_ptr<RopeCacheFP32>>
    rope_cache_fp32;
  std::shared_ptr<RopeCacheFP32> freqs_fp32 = nullptr;
#ifdef ENABLE_FP16
  /**
   * @brief FP16 rotary embedding cache entries.
   */
  struct RopeCacheFP16 {
    std::vector<std::vector<_FP16>> cos;
    std::vector<std::vector<_FP16>> sin;
  };
  inline static std::unordered_map<std::string, std::shared_ptr<RopeCacheFP16>>
    rope_cache_fp16;
  std::shared_ptr<RopeCacheFP16> freqs_fp16 = nullptr;
#endif
  std::vector<float> thetas;

  /**
   * @brief pre_compute frequencies for Rotary Embedding.
   * @param[in] head_dim dimension of head
   * @param[in] seq_len sequence length
   * @param[in] theta base of theta (default = 10000)
   */
  void precompute_freqs(int head_dim, unsigned int seq_len,
                        float theta = 10000.0, bool is_fp16 = false);

  /**
   * @brief The HTP per-token attention hook at one decode row:
   *        hands q | k | v, the RoPE table (once) and, when asked, the
   *        CPU cache's rows [0, pos) to the backend; on true the output
   *        row was written by the DSP and this layer's cache is stale
   *        from now on (the DSP owns it).
   */
  bool htpDecodeAttention(nntrainer::RunLayerContext &context,
                          unsigned int pos);

  /**
   * @brief _compute frequency parameters for default ROPE
   */
  void _compute_default_parameters(int head_dim, float theta);

  /** Compute YaRN RoPE frequencies. */
  void _compute_yarn_parameters(int head_dim, float theta);
  void _compute_proportional_parameters(int head_dim, float theta);

  std::string getRopeCacheKey(int head_dim, unsigned int seq_len,
                              float theta) const;

  /**
   * @brief     apply rotary embedding
   * @param[in] in input tensor
   * @param[out] out output tensor
   * @param[in] dim hidden dim size
   * @param[in] from sequence order
   * @param[in] convert_only true to only store the tensor into the cache dtype
   */
  void apply_rotary_emb_tensor_v2(nntrainer::Tensor &in, nntrainer::Tensor &out,
                                  unsigned int dim, unsigned int from,
                                  bool convert_only = false);

  template <typename BType>
  void compute(const float *A, const BType *B, float *output, int num_rows,
               int N, int chunk_size, int group_size, int tile_size,
               bool process_all);

  void compute_kcaches(nntrainer::Tensor &in, nntrainer::Tensor &cache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from, nntrainer::Tensor &sink_step);

  void compute_vcaches(nntrainer::Tensor &in, nntrainer::Tensor &vcache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  void compute_fp16vcache_transposed(nntrainer::Tensor &in,
                                     nntrainer::Tensor &vcache,
                                     nntrainer::Tensor &output, int from,
                                     int num_cache_head, int gqa_size,
                                     int head_dim, int to);

  /************** END OF  ROTARY EMBEDDING *************/

  void calcCommonDerivative(nntrainer::RunLayerContext &context);

  size_t calc_attn_index(size_t i);

  /**
   * @brief Windowed cumulative attention score index.
   *
   * Returns the cumulative number of attention scores before absolute query
   * row i, respecting a sliding window of size local_window_size (W):
   *
   *   S(i) = sum_{k=0}^{i-1} min(k+1, W)
   *        = (i <= W) ? i*(i+1)/2 : W*(W+1)/2 + (i - W)*W
   *
   * When W == UINT_MAX (full attention), this reduces exactly to
   * i*(i+1)/2 == calc_attn_index(i), preserving byte-identical behaviour.
   */
  size_t calc_windowed_attn_index(size_t i);

}; // end of class Fp16Attention
} // namespace nntrainer

#endif
