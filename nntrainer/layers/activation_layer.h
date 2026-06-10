// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   activation_layer.h
 * @date   17 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__
#ifdef __cplusplus

#include <memory>

#include <acti_func.h>
#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Activation Layer
 * @brief   Activation Layer
 */
class ActivationLayer : public Layer {

public:
  /**
   * @brief     Constructor of Activation Layer
   */
  ActivationLayer();

  /**
   * @brief     Destructor of Activation Layer
   */
  ~ActivationLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ActivationLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return acti_func.supportInPlace(); }

  static constexpr const char *type = "activation";

private:
  using PropTypes = std::tuple<props::Activation, props::SkipPrefill>;

  std::unique_ptr<PropTypes> activation_props; /**< activation props */

  ActiFunc acti_func; /**< activation function from activation type */
  bool skip_prefill = false;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ACTIVATION_LAYER_H__ */
