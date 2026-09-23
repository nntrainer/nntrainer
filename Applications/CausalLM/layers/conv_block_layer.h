// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   conv_block_layer.h
 * @date   22 September 2026
 * @brief  The LFM2 conv block (in_proj, gate, causal conv1d, gate, out_proj)
 *         as one layer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CONV_BLOCK_LAYER_H__
#define __CONV_BLOCK_LAYER_H__
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
 * @brief The conv block as ONE layer, so an accelerator can take in_proj,
 *        the two gates, the causal conv1d and out_proj in one call
 *        (docs/htp_attention/51 section 2).
 *
 * Holds the same three weights, in the same order and shapes, as the
 * fully_connected / causal_conv1d / fully_connected layers
 * Lfm2Transformer::createConvBlock builds otherwise (in_proj [K x 3C]
 * Q4_0, conv [3 x C] FP32, out_proj [C x N] Q4_0 -- the file's order), so
 * a model file is read identically. Where the layer's ComputeOps has the
 * fused call and the step is more than one row, the whole block goes out
 * as one call and the conv state comes back with it; else (decode, or a
 * backend without it) it computes exactly what those layers did, with the
 * same CPU conv kernels and the same conv state.
 */
class WIN_EXPORT ConvBlockLayer : public nntrainer::LayerImpl {
public:
  ConvBlockLayer();
  ~ConvBlockLayer() = default;
  ConvBlockLayer(ConvBlockLayer &&rhs) noexcept = default;
  ConvBlockLayer &operator=(ConvBlockLayer &&rhs) = default;

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
  const std::string getType() const override { return ConvBlockLayer::type; }
  void setProperty(const std::vector<std::string> &values) override;
  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "conv_block";

  static constexpr unsigned int KERNEL_SIZE = 3;

private:
  /** unit = the conv width C (in_proj is 3C wide) */
  std::tuple<nntrainer::props::Unit> conv_props;
  std::array<unsigned int, 3> weight_idx; /**< in_proj, conv, out_proj */
  std::array<unsigned int, 4> tensor_idx; /**< proj, gated, conv_out, state */
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CONV_BLOCK_LAYER_H__ */
