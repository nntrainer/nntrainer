// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   mse_loss_layer.h
 * @date   24 June 2021
 * @brief  This is MSE Loss Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __MSE_LOSS_LAYER_H__
#define __MSE_LOSS_LAYER_H__
#ifdef __cplusplus

#include <loss_layer.h>

namespace nntrainer {

/**
 * @class   MSELossLayer
 * @brief   MSE Loss Layer
 */
class MSELossLayer : public LossLayer {
public:
  /**
   * @brief     Constructor of MSE Loss Layer
   */
  MSELossLayer() : LossLayer() {}

  /**
   * @brief     Destructor of MSE Loss Layer
   */
  ~MSELossLayer() = default;

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
  const std::string getType() const override { return MSELossLayer::type; };

  static constexpr const char *type = "mse";
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MSE_LOSS_LAYER_H__ */
