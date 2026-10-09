// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.h
 * @date   02 September 2024
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is custom_mha_core layer supports
 *         the work of multi_head_attention.
 * @note   Unlike custom_multi_head_attention_layer,
 *         which works all of the attention operations
 *         in a layer, this layer is attached after Q / K / V
 *         fully connected layer to post-process them
 *         including KV-Cache.
 *         For inference, incremental_forwarding is called,
 *         which takes inputs of seq_len = 1 via `from` / `to` param.
 *         For training, forwarding is called,
 *         which takes all input seqences at once.
 */

#ifndef __MHA_CORE_H__
#define __MHA_CORE_H__

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
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <layer_impl.h>
#include <limits.h>
#include <util_simd.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace causallm {

namespace props {

/**
 * @brief NumHeads property, NumHeads is number of head in multi head attention
 * of Q
 */
class NumHeads_KV : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new NumHeads object with default value 1
   */
  NumHeads_KV(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "num_heads_KV";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief SlidingWindow
 */
class SlidingWindow : public nntrainer::Property<unsigned int> {
public:
  SlidingWindow(unsigned int value = UINT_MAX) { set(value); };
  static constexpr const char *key =
    "sliding_window";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief MaxNewTokens
 */
class MaxNewTokens : public nntrainer::Property<unsigned int> {
public:
  MaxNewTokens(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "max_new_tokens";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief MaxNewTokens
 */
class MaxPositionEmbeddings : public nntrainer::Property<unsigned int> {
public:
  MaxPositionEmbeddings(unsigned int value = 40960) { set(value); };
  static constexpr const char *key =
    "max_position_embeddings";               /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief UseRope property
 */
class UseRope : public nntrainer::Property<bool> {
public:
  UseRope(bool value = true) { set(value); };
  static constexpr const char *key = "use_rope"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;     /**< property type */
};

/**
 * @brief UseSink property
 */
class UseSink : public nntrainer::Property<bool> {
public:
  UseSink(bool value = false) { set(value); };
  static constexpr const char *key = "use_sink"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;     /**< property type */
};

/**
 * @brief AttnLogitSoftcapping
 */
class AttnLogitSoftcapping : public nntrainer::Property<float> {
public:
  AttnLogitSoftcapping(float value = 0.0f) { set(value); };
  static constexpr const char *key =
    "attn_logit_softcapping";                 /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief KvCacheQuant: "" (off), "q8" or "q4". Keep an int8 / int4 mirror
 *        of this layer's KV cache on the accelerator and run attention
 *        over it (ComputeOps::sdpa_q_kvcache) instead of over the fp16
 *        cache. The fp16 cache stays the source of truth.
 */
class KvCacheQuant final : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "kv_cache_quant";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief IsCausal property
 */
class IsCausal : public nntrainer::Property<bool> {
public:
  IsCausal(bool value = true) { set(value); };
  static constexpr const char *key = "is_causal"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;      /**< property type */
};

/**
 * @brief RopeScalingFactor
 */
class RopeScalingFactor : public nntrainer::Property<float> {
public:
  RopeScalingFactor(float value = 1.0) { set(value); };
  static constexpr const char *key =
    "rope_scaling_factor";                    /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief RopeScalingMaxPositionEmbeddings
 */
class RopeScalingMaxPositionEmbeddings
  : public nntrainer::Property<unsigned int> {
public:
  RopeScalingMaxPositionEmbeddings(unsigned int value = 4096) { set(value); };
  static constexpr const char *key =
    "rope_scaling_max_position_embeddings";  /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

}; // namespace props

/**
 * @class MHA Core Layer
 * @brief Part of Multi-Head-Attention Layer.
 *        It should be attached after Q / K / V fc layers and before O fc layer.
 *        custom_mha_core_layer computes attention, while updating KV-cache for
 *        inference mode.
 *
 *    [ Q ]    [ K ]    [ V ]
 *      |        |        |
 *     [      mha_core      ]
 *               |
 *             [ O ]
 *
 */
WIN_EXPORT class MHACoreLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Registers batch 0's quantized KV cache on the accelerator now
   *        rather than inside the first attention call, where it cost the
   *        first prefill 1.9 ms a layer (doc 57 section 9.24).
   *        Called once at load after every weight is placed (the cache is
   *        DSP heap: grown between two arena chunk mappings it would strand
   *        address space, transformer.cpp). A refusal leaves the handle
   *        unset, and the first call registers and falls back as before.
   */
  void registerQuantizedCache(nntrainer::RunLayerContext &context);

  /**
   * @brief Constructor of MhaCore Layer
   */
  WIN_EXPORT MHACoreLayer();

  /**
   * @brief Destructor of MhaPost Layer
   */
  WIN_EXPORT ~MHACoreLayer();

  /**
   *  @brief  Move constructor of CustomMultiHeadAttentionLayer.
   *  @param[in] CustomMultiHeadAttentionLayer &&
   */
  WIN_EXPORT
  MHACoreLayer(MHACoreLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CustomMultiHeadAttentionLayer to be moved.
   */
  WIN_EXPORT MHACoreLayer &operator=(MHACoreLayer &&rhs) = default;

  /**
   * @brief Finalize funciton of MhaCore Layer
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @brief forwarding function of MhaCore Layer
   *        Please note that forwarding function is used only for training.
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

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
  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   * @note In current version, we do not support backwarding yet.
   * It will be updated ASAP.
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return MHACoreLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @brief Set the cache index for external cache mode.
   *        Must be called before forwarding() when use_external_cache is true.
   * @param[in] idx current write position in the KV cache
   */
  /**
   * @brief Set the write position of the external KV cache. The host calls
   *        this when it repositions the cache -- after a load, a rewind, a
   *        session switch -- and any row of the fp16 cache may have been
   *        rewritten by then, so the quantized mirror (props::KvCacheQuant)
   *        is marked out of date in full and re-appended from row 0 on the
   *        next step.
   */
  WIN_EXPORT void setCacheIndex(unsigned int idx) {
    cache_index = idx;
    for (auto &synced : q_cache_synced) {
      synced = 0;
    }
  }

  /**
   * @brief Get the current cache index
   */
  WIN_EXPORT unsigned int getCacheIndex() const { return cache_index; }

  inline static const std::string type = "mha_core";

private:
  std::tuple<
    nntrainer::props::NumHeads, props::NumHeads_KV,
    nntrainer::props::ProjectedKeyDim, nntrainer::props::ProjectedValueDim,
    nntrainer::props::OutputShape, nntrainer::props::DropOutRate,
    nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep,
    props::SlidingWindow, props::MaxNewTokens, props::RopeTheta, props::UseRope,
    props::MaxPositionEmbeddings, props::UseSink, props::RopeScalingType,
    props::RopeScalingFactor, props::RopePartialRotaryFactor,
    props::RopeScalingMaxPositionEmbeddings, props::AttnLogitSoftcapping,
    props::IsCausal, props::KvCacheQuant>
    mha_core_props; /**< mha_core layer properties */

  /** softmax activation operation */
  nntrainer::ActiFunc sm;

  float epsilon;            /** to avoid overflow */
  unsigned int cache_index; /** idx of kv cache */

  /**
   * @brief Whether to use externally provided cache tensors
   *        (true when num_inputs >= 5, i.e., Q, K, V + cache_key + cache_value)
   *        In external mode mha_core does not allocate its own cache tensors,
   *        and reads cache slots from input[3] (cache_key) and input[4]
   *        (cache_value) which are bound by the host via setExternalTensors.
   */
  bool use_external_cache = false;

  /** intermal info */
  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;
  bool cache_shift;
  float theta;
  size_t local_window_size;
  bool use_sink = false;
  bool use_rope = true;
  float attn_logit_softcapping = 0.0f;
  bool is_causal;
  bool skip_prefill = false;

  /**
   * @brief The ComputeOps this layer's context resolves to, captured at
   *        the start of each forwarding so the per-batch helpers (which
   *        take no context) can ask it for an accelerated attention.
   *        nullptr means "CPU only", which is also what get_cpu_ops()'s
   *        supports_sdpa_fp16_kvcache() answers.
   */
  nntrainer::ComputeOps *compute_ops_ = nullptr;

  /**
   * @brief The accelerator-resident quantized mirror of the KV cache
   *        (props::KvCacheQuant). One handle per batch; q_cache_synced is
   *        how many rows of that handle match the fp16 cache, so a rewind
   *        or a cache load simply re-appends from there. One failure
   *        anywhere releases everything and this layer stays on the fp16
   *        path for good: the CPU/fp16 result is always available.
   */
  int kv_cache_quant_kind = -1; /**< -1 off, 0 int8, 1 int4 */
  std::vector<int> q_cache_handles;
  std::vector<unsigned int> q_cache_synced;
  bool q_cache_failed = false;
  /**
   * @brief The fixed-scale (row-blocked, ComputeOps::sdpa_q2_kvcache)
   *        a16 / kv8 path's encodings: K per KV head and V per (KV head,
   *        dim) as symmetric int8 scales, Q and the context output per
   *        query head as 16-bit asymmetric (scale, zero point) pairs. A
   *        quantized model supplies these as encodings; this f32 model
   *        stands in for them by calibrating on the first rows it sees
   *        (K/V max / 127, Q's range, the output's from V's range since the
   *        context is a convex combination of V rows) and keeping them, so
   *        later steps pay nothing. Values beyond them saturate, as they
   *        would under encodings. The u16 Q and output of each step live in
   *        the two staging buffers.
   */
  std::vector<float> q2_scale_k, q2_scale_v, q2_q_enc, q2_out_enc;
  /** The a16 path's u16 Q and output, one pair for every layer: per layer
   *  each first resize zero-filled 8-17 MB twice, 151 ms of the first
   *  prefill over 30 layers (doc 57 section 9.24). ponytail: shared by all
   *  MHACoreLayer instances, so layers must not run concurrently (they run
   *  in graph order); a per-thread pair is the upgrade if they ever do. */
  static std::vector<uint16_t> q2_q_u16, q2_out_u16;
  bool q2_calibrated = false;
  std::vector<unsigned char> q2_scales_set; /**< per batch handle */
  bool accel_logged_ = false; /**< one info line the first time attention
                                   leaves the CPU, for run logs */

  /**
   * @brief Attention of one batch over the quantized mirror: registers the
   *        handle on first use, appends the fp16 cache rows [synced,
   *        cache_to) -- this step's rows included -- and runs
   *        sdpa_q_kvcache. Same fallback contract as
   *        try_accelerated_attention.
   */
  bool try_quantized_attention(unsigned int batch,
                               nntrainer::Tensor &query_step,
                               nntrainer::Tensor &cache_key,
                               nntrainer::Tensor &cache_value,
                               const ml::train::TensorDim &cache_key_dim,
                               nntrainer::Tensor &attention_output_step,
                               unsigned int cache_from, unsigned int cache_to,
                               const nntrainer::Tensor *sink);

  /** @brief Releases every quantized-cache handle. */
  void release_quantized_cache();

  /** @brief Fills q2_scale_* from the first rows (see the members). */
  void calibrate_q2_scales(const uint16_t *k_rows, const uint16_t *v_rows,
                           unsigned int n_rows, unsigned int kv_stride,
                           const float *q, unsigned int q_stride,
                           unsigned int n_q);

  /**
   * @brief Runs steps 2-4 (Q.K^T, softmax, .V) of one batch on the
   *        accelerator when the context's ComputeOps offers one and the
   *        shape fits its contract (causal, f32 query/output, fp16 cache).
   *
   * @return true if attention_output_step now holds the result; false
   *         means nothing was written and the CPU path must run.
   */
  bool try_accelerated_attention(nntrainer::Tensor &query_step,
                                 nntrainer::Tensor &cached_key,
                                 nntrainer::Tensor &cached_value,
                                 nntrainer::Tensor &attention_output_step,
                                 unsigned int cache_from, unsigned int cache_to,
                                 const nntrainer::Tensor *sink);

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
    /** intended comment for later use of attention_mask */
    // attention_mask,
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
   * @note it is expected to be called only once at the finalize.
   * @param[in] head_dim dimension of head
   * @param[in] seq_len sequence length
   * @param[in] theta base of theta (default = 10000)
   */
  void precompute_freqs(int head_dim, unsigned int seq_len,
                        float theta = 10000.0, bool is_fp16 = false);

  /**
   * @brief [#130] The HTP per-token attention hook at one decode row:
   *        hands q | k | v, the RoPE table (once) and, when asked, the
   *        CPU cache's rows [0, pos) to the backend; on true the output
   *        row was written by the DSP and this layer's cache is stale
   *        from now on (the DSP owns it, plan 130 section 3.3).
   */
  bool htpDecodeAttention(nntrainer::RunLayerContext &context,
                          unsigned int pos);
  std::vector<float>
    htp_rope_table_; /**< [max_timestep][cos hd/2 | sin hd/2] */
  bool htp_owns_cache_ = false;       /**< a hook returned 1: rows past the
                                           prompt are the DSP's, not this cache's */

  /**
   * @brief _compute frequency parameters for default ROPE
   */
  void _compute_default_parameters(int head_dim, float theta);

  /**
   * @brief _compute frequency parameters for default ROPE
   */
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

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
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

}; // end of class MHACoreLayer
} // namespace causallm

#endif
