// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   cross_entropy_Softmax_loss_layer.h
 * @date   24 June 2021
 * @brief  This is Cross Entropy Softmax with Softmax Loss Layer Class of Neural
 * Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CROSS_ENTROPY_SOFTMAX_LOSS_LAYER_H__
#define __CROSS_ENTROPY_SOFTMAX_LOSS_LAYER_H__
#ifdef __cplusplus

#include <loss_layer.h>

namespace nntrainer {

/**
 * @class   CrossEntropySoftmaxLossLayer
 * @brief   Cross Entropy Softmax Loss Layer
 */
class CrossEntropySoftmaxLossLayer : public LossLayer {
public:
  /**S
   * @brief     Constructor of Cross Entropy Softmax Loss Layer
   */
  CrossEntropySoftmaxLossLayer() : LossLayer() {}

  /**
   * @brief     Destructor of Cross Entropy Softmax Loss Layer
   */
  ~CrossEntropySoftmaxLossLayer() = default;

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
    return CrossEntropySoftmaxLossLayer::type;
  };

  static constexpr const char *type = "cross_softmax";
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CROSS_ENTROPY_SOFTMAX_LOSS_LAYER_H__ */
