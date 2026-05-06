// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen35_layers.h
 * @brief  Missing Qwen3.5 token mixer helper layers for CausalLM inference.
 */

#ifndef __QWEN35_LAYERS_H__
#define __QWEN35_LAYERS_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <common_properties.h>
#include <causallm_common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tensor.h>

namespace causallm {

namespace props {

class Qwen35LinearKeyHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  Qwen35LinearKeyHeadDim(unsigned int value = 128) { set(value); }
  static constexpr const char *key = "linear_key_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class Qwen35LinearValueHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  Qwen35LinearValueHeadDim(unsigned int value = 128) { set(value); }
  static constexpr const char *key = "linear_value_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class Qwen35LinearNumKeyHeads : public nntrainer::PositiveIntegerProperty {
public:
  Qwen35LinearNumKeyHeads(unsigned int value = 16) { set(value); }
  static constexpr const char *key = "linear_num_key_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

class Qwen35LinearNumValueHeads : public nntrainer::PositiveIntegerProperty {
public:
  Qwen35LinearNumValueHeads(unsigned int value = 16) { set(value); }
  static constexpr const char *key = "linear_num_value_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

class Qwen35LinearConvKernelDim : public nntrainer::PositiveIntegerProperty {
public:
  Qwen35LinearConvKernelDim(unsigned int value = 4) { set(value); }
  static constexpr const char *key = "linear_conv_kernel_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

class Qwen35PairSelectIndex : public nntrainer::Property<unsigned int> {
public:
  Qwen35PairSelectIndex(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "select_index";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

class ReshapedL2NormLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT ReshapedL2NormLayer();
  WIN_EXPORT ~ReshapedL2NormLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "reshaped_l2_norm";

private:
  std::tuple<props::FeatureSize, nntrainer::props::Epsilon> props_;
  unsigned int feature_size;
};

class FeatureBiasLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT FeatureBiasLayer();
  WIN_EXPORT ~FeatureBiasLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "feature_bias";

private:
  unsigned int bias_idx;
};

class FeatureScaleLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT FeatureScaleLayer();
  WIN_EXPORT ~FeatureScaleLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "feature_scale";

private:
  unsigned int scale_idx;
};

class HeadPairSplitLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT HeadPairSplitLayer();
  WIN_EXPORT ~HeadPairSplitLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "qwen35_head_pair_split";

private:
  std::tuple<props::FeatureSize, props::Qwen35PairSelectIndex> props_;
  unsigned int feature_size;
  unsigned int select_index;
};

class Qwen35CausalDepthwiseConv1DLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Qwen35CausalDepthwiseConv1DLayer();
  WIN_EXPORT ~Qwen35CausalDepthwiseConv1DLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "qwen35_causal_depthwise_conv1d";

private:
  enum WeightIndex { ConvWeight, WeightCount };
  enum TensorIndex { ConvState, TensorCount };

  std::array<unsigned int, WeightCount> wt_idx;
  std::array<unsigned int, TensorCount> tensor_idx;
  std::tuple<props::Qwen35LinearConvKernelDim> props_;
};

class Qwen35GatedDeltaCoreLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Qwen35GatedDeltaCoreLayer();
  WIN_EXPORT ~Qwen35GatedDeltaCoreLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override { return type; }
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "qwen35_gated_delta_core";

private:
  enum InputIndex { Query, Key, Value, Beta, Decay, InputCount };
  enum TensorIndex { RecurrentState, TensorCount };

  std::array<unsigned int, TensorCount> tensor_idx;
  std::tuple<props::Qwen35LinearNumKeyHeads,
             props::Qwen35LinearNumValueHeads,
             props::Qwen35LinearKeyHeadDim,
             props::Qwen35LinearValueHeadDim>
    props_;
};

} // namespace causallm

#endif /* __QWEN35_LAYERS_H__ */
