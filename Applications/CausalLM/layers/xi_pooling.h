// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   xi_pooling.h
 * @date   24 August 2026
 * @brief  Xi-vector Gaussian posterior inference pooling
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This layer only supports inference mode.
 */

#ifndef __XI_POOLING_LAYER_H__
#define __XI_POOLING_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

#include <common_properties.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace causallm {

namespace props {

/**
 * @brief Hidden width of the log-precision estimator
 */
class XiHiddenSize : public nntrainer::PositiveIntegerProperty {
public:
  XiHiddenSize(unsigned int value = 256) { set(value); }
  static constexpr const char *key = "hidden_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief Xi-vector pooling (Gaussian posterior inference over frames).
 *
 * Reduces a [tokens, dim] sequence to a single [dim] vector. Instead of a plain
 * mean it estimates a per-frame, per-channel precision and takes the posterior
 * mean under a learned Gaussian prior, so noisy frames are down-weighted
 * channel by channel:
 *
 *   r      = BatchNorm(ReLU(W1 x + b1))                  [hidden, tokens]
 *   lambda = softplus(W2 r + b2)                         [dim, tokens]
 *   L      = clamp(2 log lambda, -15, 15)                [dim, tokens]
 *   w      = softmax([L, prior_logprec], over tokens+1)  [dim, tokens+1]
 *   out    = sum(w * [x^T, prior_mean], over tokens+1)   [dim]
 *
 * The prior contributes one extra column to the softmax, which is what makes
 * this a posterior mean rather than an attention-weighted average: a frame set
 * whose precisions are all low falls back toward prior_mean.
 *
 * Input  [batch, 1, tokens, dim]
 * Output [batch, 1, 1, dim]
 */
WIN_EXPORT class XiPoolingLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Xi pooling layer object
   */
  WIN_EXPORT XiPoolingLayer();

  /**
   * @brief Destroy the Xi pooling layer object
   */
  WIN_EXPORT ~XiPoolingLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return XiPoolingLayer::type;
  }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "xi_pooling";

private:
  /** Weight slots, in the order they are read from the weight file. */
  enum XiParams {
    lin1_w = 0,  /**< [hidden, dim] log-precision estimator projection */
    lin1_b,      /**< [hidden] */
    bn_mean,     /**< [hidden] BatchNorm moving mean */
    bn_var,      /**< [hidden] BatchNorm moving variance */
    bn_gamma,    /**< [hidden] */
    bn_beta,     /**< [hidden] */
    lin2_w,      /**< [dim, hidden] second projection */
    lin2_b,      /**< [dim] */
    prior_mean,  /**< [dim] Gaussian prior mean */
    prior_logpr, /**< [dim] Gaussian prior log-precision */
    NUM_PARAMS
  };

  std::array<unsigned int, NUM_PARAMS> wt_idx;
  std::tuple<props::XiHiddenSize, nntrainer::props::Epsilon> xi_props;
  unsigned int dim = 0;    /**< channel width (input width) */
  unsigned int hidden = 0; /**< estimator hidden width */
};

} // namespace causallm

#endif /* __XI_POOLING_LAYER_H__ */
