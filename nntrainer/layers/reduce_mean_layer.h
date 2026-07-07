// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   reduce_mean_layer.h
 * @date   25 Nov 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reduce Mean Layer Class for Neural Network
 *
 */

#ifndef __REDUCE_MEAN_LAYER_H__
#define __REDUCE_MEAN_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

class RunLayerContext;
class InitLayerContext;

/**
 * @class   Reduce Mean Layer
 * @brief   Reduce Mean Layer
 */
class ReduceMeanLayer : public Layer {
public:
  /**
   * @brief     Constructor of Reduce Mean Layer
   */
  ReduceMeanLayer() : Layer() {}

  /**
   * @brief     Destructor of Reduce Mean Layer
   */
  ~ReduceMeanLayer(){};

  /**
   *  @brief  Move constructor of ReduceMeanLayer.
   *  @param[in] rhs ReduceMeanLayer to be moved.
   */
  ReduceMeanLayer(ReduceMeanLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs ReduceMeanLayer to be moved.
   */
  ReduceMeanLayer &operator=(ReduceMeanLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ReduceMeanLayer::type; };

  static constexpr const char *type = "reduce_mean";

private:
  /** TODO: support scalar multiplier to simulate reduce_sum */
  std::tuple<props::ReduceDimension>
    reduce_mean_props; /**< reduce_mean properties : axis to reduce along */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __REDUCE_MEAN_LAYER_H__ */
