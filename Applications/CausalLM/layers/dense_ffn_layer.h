// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   dense_ffn_layer.h
 * @date   21 September 2026
 * @brief  The dense SwiGLU FFN (up, gate, SwiGLU, down) as one layer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __DENSE_FFN_LAYER_H__
#define __DENSE_FFN_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <tuple>

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/** glu_activation: the gate's activation, swish (default) or tanh_gelu.
 *  Its own key: "activation" belongs to the node and would append an
 *  activation layer after the block instead. */
class GluActivation final
  : public nntrainer::EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "glu_activation";
};

/** gate_first: the file holds gate, up, down (a converter that writes the
 *  projections in that order) instead of the up, gate, down default. */
class GateFirst : public nntrainer::Property<bool> {
public:
  GateFirst(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "gate_first";
};

} // namespace props

/**
 * @brief The dense gated FFN (SwiGLU, or GeGLU with activation=tanh_gelu) as
 * ONE layer, so an accelerator can take up, gate, SwiGLU and down in one call
 * (docs/htp_attention/51).
 *
 * Holds the same three Q4_0 weights, in the same order and shapes, as the
 * three fully_connected layers Transformer::createMlp builds otherwise
 * (up, gate, down -- the file's order), so a model file is read
 * identically. Where the layer's ComputeOps has the fused call and the
 * step is more than one row, the whole block goes out as one call; else
 * (decode, or a backend without it) it computes exactly what those layers
 * did, with the same deterministic SwiGLU the MoE layer uses.
 */
class WIN_EXPORT DenseFfnLayer : public nntrainer::LayerImpl {
public:
  DenseFfnLayer();
  ~DenseFfnLayer() = default;
  DenseFfnLayer(DenseFfnLayer &&rhs) noexcept = default;
  DenseFfnLayer &operator=(DenseFfnLayer &&rhs) = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  void calcGradient(nntrainer::RunLayerContext &context) override {}
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;
  const std::string getType() const override { return DenseFfnLayer::type; }
  void setProperty(const std::vector<std::string> &values) override;
  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "dense_ffn";

private:
  /** unit = the intermediate size (the width of up and gate); activation
   *  swish (default) or tanh_gelu; gate_first for the file's weight order */
  std::tuple<nntrainer::props::Unit, props::GluActivation, props::GateFirst,
             props::InNorm, props::OutNorm, nntrainer::props::Epsilon>
    dense_props;
  bool gelu = false; /**< tanh_gelu(gate) * up instead of silu(gate) * up */
  bool in_norm = false, out_norm = false;
  std::array<unsigned int, 5>
    weight_idx; /**< up, gate, down, [in_gamma, out_gamma] */
  std::array<unsigned int, 4>
    tensor_idx; /**< up_out, gate_out, act, the normed input/output */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __DENSE_FFN_LAYER_H__ */
