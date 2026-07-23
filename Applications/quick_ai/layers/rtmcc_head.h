// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rtmcc_head.h
 * @date   15 July 2026
 * @brief  RTMCC SimCC pose head as a single, layout-independent custom layer.
 *
 *         Consumes the final-layer conv output [B, K, H, W] (K = keypoints,
 *         H*W = spatial) and runs the whole RTMPose head internally:
 *           flatten (format-aware) -> ScaleNorm -> Linear(mlp) ->
 *           GAU (RTMCCBlock) -> cls_x / cls_y SimCC classifiers ->
 *           concat -> [B, 1, 2*K, SIMCC_BINS].
 *
 *         Because every token op is indexed explicitly, the layer is
 *         correct in both NCHW (x86) and NHWC (ARM Q8_0) — no transposes are
 *         needed around the conv/backbone, which stay channel-last natively.
 *         Inference only, FP32 activations.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __RTMCC_HEAD_LAYER_H__
#define __RTMCC_HEAD_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <tuple>

#include <base_properties.h>
#include <common_properties.h>

namespace quick_ai {

namespace props {

/** GAU expansion feature size e (= in_dims * expansion_factor) */
class RtmccExpansion : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gau_expansion";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** GAU query/key projection size s */
class RtmccKeyDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gau_key_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** GAU token feature dim D (mlp output / gau hidden) */
class RtmccHidden : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "gau_hidden";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** SimCC bins per axis */
class RtmccSimcc : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "simcc_bins";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief RTMCC SimCC pose head layer.
 */
WIN_EXPORT class RTMCCHeadLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT RTMCCHeadLayer() :
    Layer(),
    head_props(props::RtmccExpansion(), props::RtmccKeyDim(),
               props::RtmccHidden(), props::RtmccSimcc(),
               nntrainer::props::Epsilon()) {}

  WIN_EXPORT ~RTMCCHeadLayer() {}

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
    return RTMCCHeadLayer::type;
  };

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "rtmcc_head";

private:
  std::tuple<props::RtmccExpansion, props::RtmccKeyDim, props::RtmccHidden,
             props::RtmccSimcc, nntrainer::props::Epsilon>
    head_props;

  enum Params {
    mlp_ln_g,
    mlp_w,
    gau_uv,
    gau_o,
    gau_gamma,
    gau_beta,
    gau_ln_g,
    gau_res_scale,
    cls_x,
    cls_y,
    NUM_PARAMS
  };
  std::array<unsigned int, NUM_PARAMS> wt_idx;

  unsigned int num_token = 0; // K (keypoints)
  unsigned int flatten = 0;   // H*W (spatial feature)
  unsigned int hidden = 0;    // D
  unsigned int e_dims = 0;    // e
  unsigned int s_dims = 0;    // s
  unsigned int simcc = 0;     // bins per axis
  bool is_nchw = true;
};

} // namespace quick_ai

#endif /* __RTMCC_HEAD_LAYER_H__ */
