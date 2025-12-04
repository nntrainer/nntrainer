// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file   mla_core.h
 * @date   03 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/2405.04434 (DeepSeek-V2)
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is mla_core layer supports
 *         the work of multi_head_latent_attention.
 * @note   This layer implements the core logic of MLA, including
 *         KV compression and decoupled RoPE.
 */

#ifndef __MLA_CORE_H__
#define __MLA_CORE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <complex>

#include <acti_func.h>
#include <bs_thread_pool_manager.hpp>
#include <common_properties.h>
#include <cpu_backend.h>
#include <layer_impl.h>
#include <limits.h>
#include <util_simd.h>

#include <utility>

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
 * @brief RopeTheta
 */
class RopeTheta : public nntrainer::Property<unsigned int> {
public:
  RopeTheta(unsigned int value = 500000) { set(value); };
  static constexpr const char *key = "rope_theta"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;       /**< property type */
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
 * @brief RopeScalingType
 * - default
 * - yarn
 */
class RopeScalingType : public nntrainer::Property<std::string> {
public:
  RopeScalingType(std::string value = "default") { set(value); };
  static constexpr const char *key =
    "rope_scaling_type";                    /**< unique key to access */
  using prop_tag = nntrainer::str_prop_tag; /**< property type */
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

/**
 * @brief KVLoRARank
 * Dimension of the compressed latent vector for KV
 */
class KVLoRARank : public nntrainer::PositiveIntegerProperty {
public:
  KVLoRARank(unsigned int value = 512) { set(value); };
  static constexpr const char *key = "kv_lora_rank"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;         /**< property type */
};

/**
 * @brief QKRoPEDim
 * Dimension of the RoPE part of Query and Key
 */
class QKRoPEDim : public nntrainer::PositiveIntegerProperty {
public:
  QKRoPEDim(unsigned int value = 64) { set(value); };
  static constexpr const char *key = "qk_rope_dim"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

/**
 * @brief QKNopeDim
 * Dimension of the non-RoPE content part of Query and Key
 */
class QKNopeDim : public nntrainer::PositiveIntegerProperty {
public:
  QKNopeDim(unsigned int value = 128) { set(value); };
  static constexpr const char *key = "qk_nope_dim"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

}; // namespace props

/**
 * @class MLA Core Layer
 * @brief Multi-Head Latent Attention Layer.
 *        Implements DeepSeek-V2/V3 style attention with latent KV compression.
 */
WIN_EXPORT class MLACoreLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of MlaCore Layer
   */
  WIN_EXPORT MLACoreLayer();

  /**
   * @brief Destructor of MlaCore Layer
   */
  WIN_EXPORT ~MLACoreLayer();

  /**
   *  @brief  Move constructor.
   */
  WIN_EXPORT
  MLACoreLayer(MLACoreLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   */
  WIN_EXPORT MLACoreLayer &operator=(MLACoreLayer &&rhs) = default;

  /**
   * @brief Finalize funciton of MlaCore Layer
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @brief forwarding function of MlaCore Layer
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @brief Incremental forwarding for inference
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @brief Calculate derivative (not supported yet)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @brief Calculate gradient (not supported yet)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @brief Support backwarding check
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @brief Export layer
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @brief Set properties
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Get layer type
   */
  WIN_EXPORT const std::string getType() const override {
    return MLACoreLayer::type;
  };

  /**
   * @brief Set batch size
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  /**
   * @brief Update tensors by input dimensions
   */
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "mla_core";

private:
  std::tuple<
    nntrainer::props::NumHeads, props::NumHeads_KV,
    nntrainer::props::ProjectedKeyDim, nntrainer::props::ProjectedValueDim,
    nntrainer::props::OutputShape, nntrainer::props::DropOutRate,
    nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep,
    props::SlidingWindow, props::MaxNewTokens, props::RopeTheta,
    props::MaxPositionEmbeddings, props::UseSink, props::RopeScalingType,
    props::RopeScalingFactor, props::RopeScalingMaxPositionEmbeddings,
    props::KVLoRARank, props::QKRoPEDim, props::QKNopeDim>
    mla_core_props; /**< mla_core layer properties */

  /** softmax activation operation */
  nntrainer::ActiFunc sm;

  float epsilon;            /** to avoid overflow */
  unsigned int cache_index; /** idx of kv cache */

  /** internal info */
  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;
  size_t kv_lora_rank;
  size_t qk_rope_dim;
  size_t qk_nope_dim;
  
  bool cache_shift;
  float theta;
  size_t local_window_size;
  bool use_sink = false;

  enum INOUT_INDEX {
    /** input index */
    QUERY = 0,      // [B, 1, 1, (QK_NOPE + QK_ROPE) * H_Q]
    LATENT_KV = 1,  // [B, 1, 1, KV_LORA_RANK]
    KEY_ROPE = 2,   // [B, 1, 1, QK_ROPE]
    MASK = 3,

    /** output index */
    OUTPUT = 0,
    RETURN_ATTENTION_WEIGHT = 1,
  };

  /**< indices of the weights and tensors */
  enum AttentionParams {
    cache_c_kv,      // Compressed KV cache
    cache_k_pe,      // Key RoPE cache
    weight_uv,       // Up-projection weight (Latent -> Value Head)
    weight_uk,       // Up-projection weight (Latent -> Key Head)
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
  unsigned int original_max_position_embeddings = 4096;

  /****************** ROTARY EMBEDDING *****************/
  inline static std::vector<float> thetas;
  inline static std::vector<std::vector<float>> *freqs_cos = {};
  inline static std::vector<std::vector<float>> *freqs_sin = {};
#ifdef ENABLE_FP16
  inline static std::vector<std::vector<_FP16>> *freqs_cos_fp16 = {};
  inline static std::vector<std::vector<_FP16>> *freqs_sin_fp16 = {};
#endif

  void precompute_freqs(int head_dim, unsigned int seq_len,
                        float theta = 10000.0);
  void _compute_default_parameters(int head_dim, float theta);
  void _compute_yarn_parameters(int head_dim, float theta);
  void apply_rotary_emb_tensor_v2(nntrainer::Tensor &in, nntrainer::Tensor &out,
                                  unsigned int dim, unsigned int from,
                                  bool convert_only = false);

  /************** HELPER FUNCTIONS *************/
  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &latent_kv_step, nntrainer::Tensor &key_rope_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_c_kv,
    nntrainer::Tensor &cache_k_pe, ml::train::TensorDim &cache_c_kv_dim,
    ml::train::TensorDim &cache_c_kv_step_dim,
    ml::train::TensorDim &cache_k_pe_dim,
    ml::train::TensorDim &cache_k_pe_step_dim);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from, BS::thread_pool<> &pool);

  size_t calc_attn_index(size_t i);

}; // end of class MLACoreLayer
} // namespace causallm

#endif
