// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   plugged_layer.h
 * @date   27 January 2021
 * @brief  This file contains a wrapper for a plugged layer, INTERNAL USE ONLY
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __PLUGGED_LAYER_H__
#define __PLUGGED_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <nntrainer_error.h>

namespace nntrainer {
namespace internal {

/**
 * @brief PluggedLayer to wrap a layer from shared object file
 *
 */
class PluggedLayer : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Plugged Layer object
   *
   * @param pluggable LayerPluggable structure from the symbol
   */
  PluggedLayer(const nntrainer::LayerPluggable *pluggable) :
    layerImpl(pluggable->createfunc()),
    destroy_func(pluggable->destroyfunc) {
    NNTR_THROW_IF(layerImpl == nullptr, std::invalid_argument)
      << "either create_func_ failed or cannot dynamic cast to layer";
  }

  /**
   * @brief Destroy the Plugged Layer object
   *
   */
  ~PluggedLayer() override { destroy_func(layerImpl); }

  /**
   * @brief Move Contruct Plugged Layer object
   *
   * @param rhs layer to move
   */
  PluggedLayer(PluggedLayer &&rhs) noexcept = default;

  /**
   * @brief Move assign Plugged Layer Object
   *
   * @param rhs layer to move
   * @return PluggedLayer& *this
   */
  PluggedLayer &operator=(PluggedLayer &&rhs) = default;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return layerImpl->getType(); }

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override {
    layerImpl->finalize(context);
  }

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override {
    layerImpl->forwarding(context, training);
  }

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override {
    layerImpl->calcDerivative(context);
  }

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override {
    layerImpl->calcGradient(context);
  }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override {
    layerImpl->setProperty(values);
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {
    layerImpl->exportTo(exporter, method);
  }

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override {
    layerImpl->setBatch(context, batch);
  }

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return layerImpl->supportInPlace(); }

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return layerImpl->requireLabel(); }

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override {
    return layerImpl->supportBackwarding();
  }

  /**
   * @copydoc Layer::getInputLifespan()
   */
  std::optional<TensorLifespan> getInputLifespan() const override {
    return layerImpl->getInputLifespan();
  }

  /**
   * @copydoc Layer::getInputGradLifespan()
   */
  std::optional<TensorLifespan> getInputGradLifespan() const override {
    return layerImpl->getInputGradLifespan();
  }

private:
  nntrainer::Layer *layerImpl;
  nntrainer::DestroyLayerFunc destroy_func;
};
} // namespace internal
} // namespace nntrainer

#endif // __PLUGGED_LAYER_H__
