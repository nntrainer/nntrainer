// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   cross_entropy_sigmoid_loss_layer.h
 * @date   24 June 2021
 * @brief  This is Cross Entropy Sigmoid with Sigmoid Loss Layer Class of Neural
 * Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CROSS_ENTROPY_SIGMOID_LOSS_LAYER_H__
#define __CROSS_ENTROPY_SIGMOID_LOSS_LAYER_H__
#ifdef __cplusplus

#include <loss_layer.h>

namespace nntrainer {

/**
 * @class   CrossEntropySigmoidLossLayer
 * @brief   Cross Entropy Sigmoid Loss Layer
 */
class CrossEntropySigmoidLossLayer : public LossLayer {
public:
  /**
   * @brief     Constructor of Cross Entropy Sigmoid Loss Layer
   */
  CrossEntropySigmoidLossLayer() : LossLayer() {}

  /**
   * @brief     Destructor of Cross Entropy Sigmoid Loss Layer
   */
  ~CrossEntropySigmoidLossLayer() = default;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::getInputLifespan()
   */
  std::optional<TensorLifespan> getInputLifespan() const override {
    return TensorLifespan::FORWARD_DERIV_LIFESPAN;
  }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return CrossEntropySigmoidLossLayer::type;
  };

  static constexpr const char *type = "cross_sigmoid";
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CROSS_ENTROPY_SIGMOID_LOSS_LAYER_H__ */
