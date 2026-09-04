// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_attention_layer.h
 * @date   14 January 2026
 * @see
 * https://github.com/huggingface/transformers/blob/5c1c72b/src/transformers/models/deberta/modeling_deberta.py
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  DeBERTa attention layer based on mha_core-style optimized path
 */

#ifndef __DEBERTA_ATTENTION_LAYER_H__
#define __DEBERTA_ATTENTION_LAYER_H__

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <array>
#include <mutex>
#include <tuple>
#include <vector>

#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/**
 * @brief MaxRelativePositions property
 */
class MaxRelativePositions : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "max_relative_positions";
  using prop_tag = nntrainer::uint_prop_tag;
  MaxRelativePositions(unsigned int value = 0) { set(value); }
};

/**
 * @brief C2P property
 */
class C2P : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "c2p";
  using prop_tag = nntrainer::bool_prop_tag;
  C2P(bool value = false) { set(value); }
};

/**
 * @brief P2C property
 */
class P2C : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "p2c";
  using prop_tag = nntrainer::bool_prop_tag;
  P2C(bool value = false) { set(value); }
};

/**
 * @brief MaxPositionEmbeddings property
 */
class MaxPositionEmbeddings : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "max_position_embeddings";
  using prop_tag = nntrainer::uint_prop_tag;
  MaxPositionEmbeddings(unsigned int value = 512) { set(value); }
};

/**
 * @brief ShareAttKey property
 */
class ShareAttKey : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "share_att_key";
  using prop_tag = nntrainer::bool_prop_tag;
  ShareAttKey(bool value = false) { set(value); }
};

/**
 * @brief RelativeAttention property
 */
class RelativeAttention : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "relative_attention";
  using prop_tag = nntrainer::bool_prop_tag;
  RelativeAttention(bool value = true) { set(value); }
};

/**
 * @brief PositionBuckets property
 */
class PositionBuckets : public nntrainer::Property<int> {
public:
  static constexpr const char *key = "position_buckets";
  using prop_tag = nntrainer::int_prop_tag;
  PositionBuckets(int value = -1) { set(value); }
};

/**
 * @brief InputLen property
 */
class InputLen : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "input_len";
  using prop_tag = nntrainer::uint_prop_tag;
  InputLen(unsigned int value = 0) { set(value); }
};

} // namespace props

/**
 * @class DebertaAttentionLayer
 * @brief DeBERTa Attention Layer
 */
class WIN_EXPORT DebertaAttentionLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Construct a new Deberta Attention Layer object
   */
  DebertaAttentionLayer();

  /**
   * @brief Destroy the Deberta Attention Layer object
   */
  ~DebertaAttentionLayer() override;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context,
   * unsigned int from, unsigned int to, bool training)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ExportMethods &method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return DebertaAttentionLayer::type;
  }

  static constexpr const char *type = "deberta_attention";

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  /**
   * @copydoc Layer::getLayerDimensions(InitLayerContext &context)
   */
  std::array<std::vector<nntrainer::TensorDim>, 3>
  getLayerDimensions(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::updateTensorsByInputDimensions(...)
   */
  std::vector<nntrainer::TensorDim> updateTensorsByInputDimensions(
    nntrainer::InitLayerContext &init_context,
    nntrainer::RunLayerContext &run_context) override;

  /**
   * @brief wrapper around nntrainer::compute_kcaches
   */
  void compute_kcaches(nntrainer::Tensor &in, nntrainer::Tensor &cache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  /**
   * @brief softmax helper for score tensor
   */
  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from);

  /**
   * @brief wrapper around nntrainer::compute_fp16vcache_transposed
   */
  void compute_fp16vcache_transposed(nntrainer::Tensor &in,
                                     nntrainer::Tensor &vcache,
                                     nntrainer::Tensor &output, int from,
                                     int num_cache_head, int gqa_size,
                                     int head_dim, int to);

private:
  enum InputIndex {
    INPUT_IDX_Q = 0,
    INPUT_IDX_K = 1,
    INPUT_IDX_V = 2,
  };

  enum OutputIndex {
    OUTPUT_IDX = 0,
  };

  enum AttentionParams { cache_key = 0, cache_value = 1, max_params };

  /**
   * @brief one-batch incremental forwarding path
   *
   * This follows mha_core-style execution:
   *   compute_kcaches()
   *   -> add_relative_attn_score()
   *   -> softmax_triangle()
   *   -> compute_fp16vcache_transposed()
   */
  void one_batch_incremental_forwarding(
    nntrainer::RunLayerContext &context, const unsigned int batch,
    const unsigned int _from, const unsigned int from, const unsigned int to,
    nntrainer::Tensor &query_step, nntrainer::Tensor &key_step,
    nntrainer::Tensor &value_step, nntrainer::Tensor &attention_output_step,
    nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
    ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim);

  /**
   * @brief add DeBERTa relative attention score (c2p / p2c)
   * into already computed qk score
   */
  void add_relative_attn_score(nntrainer::RunLayerContext &context,
                               nntrainer::Tensor &score,
                               nntrainer::Tensor &query_step,
                               nntrainer::Tensor &key_cache, unsigned int from,
                               unsigned int to);

  /**
   * @brief bucket helper
   */
  int make_log_bucket_position(int relative_pos, int bucket_size,
                               int max_position);

  /**
   * @brief precompute bucket lookup table
   */
  void prepare_bucket_table(unsigned int max_seq_len);

  /**
   * @brief lookup precomputed bucket
   */
  int lookup_bucket(int relative_pos) const;

private:
  std::tuple<nntrainer::props::NumHeads, props::MaxPositionEmbeddings,
             props::MaxRelativePositions, props::C2P, props::P2C,
             props::ShareAttKey, props::RelativeAttention,
             props::PositionBuckets, props::InputLen,
             nntrainer::props::DisableBias>
    deberta_props;

  /**
   * @brief internal tensor indices
   */
  std::array<unsigned int, max_params> tensor_idx;

  /**
   * @brief common runtime states, kept similar to mha_core
   */
  float epsilon;
  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;

  unsigned int max_position_embeddings;
  unsigned int max_relative_positions;
  int position_buckets;
  unsigned int local_window_size;

  float attn_logit_softcapping;

  /**
   * @brief relative bucket lookup table
   */
  std::vector<int> bucket_table;
  unsigned int bucket_table_max_seq_len;
  bool bucket_table_ready;
  mutable std::mutex bucket_mtx;
};

} // namespace causallm

#endif /* __DEBERTA_ATTENTION_LAYER_H__ */
