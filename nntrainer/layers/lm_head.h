// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.h
 * @date   16 Jan 2026
 * @brief  This is LM_Head Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LM_HEAD_H__
#define __LM_HEAD_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   LMHead layer
 * @brief   LMHead layer
 */
class LmHeadLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of LmHead Layer
   */
  LmHeadLayer();

  /**
   * @brief     Destructor of LmHead Layer
   */
  ~LmHeadLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] LmHeadLayer &&
   */
  LmHeadLayer(LmHeadLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LmHeadLayer to be moved.
   */
  LmHeadLayer &operator=(LmHeadLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return LmHeadLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  void updateTensorsByInputDimensions(
    RunLayerContext &context, std::vector<TensorDim> input_dimensions) override;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "lm_head";

private:
  std::tuple<props::Unit> lmhead_props;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};
} // namespace nntrainer

#endif
#endif
