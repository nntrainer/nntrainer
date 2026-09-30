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

#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @brief The dense SwiGLU FFN as ONE layer, so an accelerator can take
 *        up, gate, SwiGLU and down in one call (docs/htp_attention/51).
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
  /** unit = the intermediate size (the width of up and gate) */
  std::tuple<nntrainer::props::Unit> dense_props;
  std::array<unsigned int, 3> weight_idx; /**< up, gate, down */
  std::array<unsigned int, 3> tensor_idx; /**< up_out, gate_out, act */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __DENSE_FFN_LAYER_H__ */
