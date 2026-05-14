// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    embedding_pooling_layer.h
 * @date    02 Jan 2026
 * @brief   This is Embedding Pooling Layer Class (for sentence-transformer)
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __EMBEDDING_POOLING_LAYER_H__
#define __EMBEDDING_POOLING_LAYER_H__

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <base_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/**
 * @brief WordEmbeddingDimension property class to hold word embedding dimension
 */
class WordEmbeddingDimension : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "word_embedding_dimension";
  using prop_tag = nntrainer::uint_prop_tag;
  WordEmbeddingDimension(unsigned int value = 0) { set(value); }
};

/**
 * @brief PoolingModeClsToken property class to hold pooling mode cls token flag
 */
class PoolingModeClsToken : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "pooling_mode_cls_token";
  using prop_tag = nntrainer::bool_prop_tag;
  PoolingModeClsToken(bool value = false) { set(value); }
};

/**
 * @brief PoolingModeMeanTokens property class to hold pooling mode mean tokens
 * flag
 */
class PoolingModeMeanTokens : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "pooling_mode_mean_tokens";
  using prop_tag = nntrainer::bool_prop_tag;
  PoolingModeMeanTokens(bool value = false) { set(value); }
};

/**
 * @brief PoolingModeMaxTokens property class to hold pooling mode max tokens
 * flag
 */
class PoolingModeMaxTokens : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "pooling_mode_max_tokens";
  using prop_tag = nntrainer::bool_prop_tag;
  PoolingModeMaxTokens(bool value = false) { set(value); }
};

/**
 * @brief PoolingModeMeanSqrtLenTokens property class to hold pooling mode mean
 */
class PoolingModeMeanSqrtLenTokens : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "pooling_mode_mean_sqrt_len_tokens";
  using prop_tag = nntrainer::bool_prop_tag;
  PoolingModeMeanSqrtLenTokens(bool value = false) { set(value); }
};

/**
 * @brief PoolingModeWeightedMeanTokens property class to hold pooling mode
 * weighted mean tokens flag
 */
class PoolingModeWeightedMeanTokens : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "pooling_mode_weightedmean_tokens";
  using prop_tag = nntrainer::bool_prop_tag;
  PoolingModeWeightedMeanTokens(bool value = false) { set(value); }
};

/**
 * @brief PoolingModeLastToken property class to hold pooling mode last token
 * flag
 */
class PoolingModeLastToken : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "pooling_mode_lasttoken";
  using prop_tag = nntrainer::bool_prop_tag;
  PoolingModeLastToken(bool value = false) { set(value); }
};

/**
 * @brief IncludePrompt property class to hold include prompt flag (default
 * true)
 */
class IncludePrompt : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "include_prompt";
  using prop_tag = nntrainer::bool_prop_tag;
  IncludePrompt(bool value = true) { set(value); }
};
} // namespace props

/**
 * @brief Embedding Pooling Layer
 * @note This layer corresponds to sentence_transformers.models.Pooling.
 *       Currently, only pooling_mode_lasttoken with include_prompt is fully
 * implemented. Other pooling modes are defined as properties but their logic is
 * not yet implemented.
 */
class WIN_EXPORT EmbeddingPoolingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Construct a new Embedding Pooling Layer object
   */
  EmbeddingPoolingLayer();

  /**
   * @brief Destroy the Embedding Pooling Layer object
   */
  ~EmbeddingPoolingLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
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
    return EmbeddingPoolingLayer::type;
  }

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "embedding_pooling";

private:
  std::tuple<props::WordEmbeddingDimension, props::PoolingModeClsToken,
             props::PoolingModeMeanTokens, props::PoolingModeMaxTokens,
             props::PoolingModeMeanSqrtLenTokens,
             props::PoolingModeWeightedMeanTokens, props::PoolingModeLastToken,
             props::IncludePrompt>
    pooling_props;
};

} // namespace causallm

#endif
