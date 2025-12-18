// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   negative_layer.h
 * @date   3 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is negative layer class (operation layer)
 *
 */

#ifndef __NEGATIVE_LAYER_H__
#define __NEGATIVE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Negative Layer
 * @brief Negative Layer
 */
class NegativeLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Negative Layer
   */
  NegativeLayer() :
    UnaryOperationLayer(),
    negative_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Negative Layer
   */
  ~NegativeLayer(){};

  /**
   *  @brief  Move constructor of Negative Layer.
   *  @param[in] NegativeLayer &&
   */
  NegativeLayer(NegativeLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs NegativeLayer to be moved.
   */
  NegativeLayer &operator=(NegativeLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for negative
   *
   * @param input input tensor
   * @param hidden tensor to store the result value
   */
  void forwarding_operation(const Tensor &input, Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const final { return support_backwarding; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    if (std::get<props::InPlaceProp>(negative_props).empty() ||
        !std::get<props::InPlaceProp>(negative_props).get()) {
      is_inplace = false;
      support_backwarding = true;
    } else {
      is_inplace = true;
      support_backwarding = false;
    }

    if (!supportInPlace())
      return InPlaceType::NONE;
    else
      return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const final { return NegativeLayer::type; };

  std::tuple<props::Print, props::InPlaceProp> negative_props;
  bool support_backwarding; /**< support backwarding */

  static constexpr const char *type = "negative";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NEGATIVE_LAYER_H__ */
