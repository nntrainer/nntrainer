// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   cast_layer.h
 * @date   04 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is cast layer class (operation layer)
 *
 */

#ifndef __CAST_LAYER_H__
#define __CAST_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Cast Layer
 * @brief Cast Layer
 */
class CastLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Cast Layer
   */
  CastLayer() :
    UnaryOperationLayer(),
    cast_props(props::Print(), props::TensorDataType()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Cast Layer
   */
  ~CastLayer(){};

  /**
   *  @brief  Move constructor of Cast Layer.
   *  @param[in] CastLayer &&
   */
  CastLayer(CastLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CastLayer to be moved.
   */
  CastLayer &operator=(CastLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for cast
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
  const std::string getType() const final { return CastLayer::type; };

  std::tuple<props::Print, props::TensorDataType> cast_props;
  bool support_backwarding;

  static constexpr const char *type = "cast";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CAST_LAYER_H__ */
