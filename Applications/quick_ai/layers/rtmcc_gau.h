// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rtmcc_gau.h
 * @date   14 July 2026
 * @brief  RTMCC Gated Attention Unit (GAU) layer for the YOLOv7ReIDtiny pose
 *         head. Implements the RTMPose RTMCCBlock:
 *           x_ln  = ScaleNorm(x)                       (RMS over feature dim)
 *           uv    = SiLU(x_ln @ Wuv)                   -> split(u, v, base)
 *           q, k  = base * gamma + beta                (2 affine projections)
 *           attn  = relu(q k^T / sqrt(s))^2 @ v
 *           out   = res_scale * x + (u * attn) @ Wo
 *
 *         Inference only. Activations are FP32 (covers W32A32 and WQ8_0A32).
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __RTMCC_GAU_LAYER_H__
#define __RTMCC_GAU_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tuple>
#include <utility>

#include <base_properties.h>
#include <common_properties.h>

namespace quick_ai {

namespace props {

/**
 * @brief GAU expansion feature size e (= in_dims * expansion_factor)
 */
class GauExpansion : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gau_expansion";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief GAU query/key projection size s
 */
class GauKeyDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gau_key_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief RTMCC Gated Attention Unit layer.
 */
WIN_EXPORT class RTMCCGauLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT RTMCCGauLayer() :
    Layer(),
    gau_props(props::GauExpansion(), props::GauKeyDim(),
              nntrainer::props::Epsilon()) {}

  WIN_EXPORT ~RTMCCGauLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return false; };

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {};

  WIN_EXPORT const std::string getType() const override {
    return RTMCCGauLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "rtmcc_gau";

private:
  std::tuple<props::GauExpansion, props::GauKeyDim, nntrainer::props::Epsilon>
    gau_props;

  // weight indices
  enum GauParams { uv_w, o_w, gamma, beta, ln_g, res_scale, NUM_PARAMS };
  std::array<unsigned int, NUM_PARAMS> wt_idx;

  unsigned int in_dims = 0;  // D (token feature dim)
  unsigned int num_token = 0; // N (keypoints)
  unsigned int e_dims = 0;   // expansion e
  unsigned int s_dims = 0;   // key dim s
};

} // namespace quick_ai

#endif /* __RTMCC_GAU_LAYER_H__ */
