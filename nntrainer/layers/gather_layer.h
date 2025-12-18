// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gather_layer.h
 * @date   02 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    It's not implemented operation yet. Just a draft for compilation.
 * @brief  This is gather layer class (operation layer)
 */

#ifndef __GATHER_LAYER_H__
#define __GATHER_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Gather Layer
 * @brief Gather Layer
 */
class GatherLayer : public BinaryOperationLayer {
public:
  /**
   * @brief Constructor of Gather Layer
   */
  GatherLayer() : support_backwarding(true) {}

  /**
   * @brief Destructor of Gather Layer
   */
  ~GatherLayer(){};

  /**
   *  @brief  Move constructor of Gather Layer.
   *  @param[in] GatherLayer &&
   */
  GatherLayer(GatherLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs GatherLayer to be moved.
   */
  GatherLayer &operator=(GatherLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for gather
   *
   * @param input tensor to be gathered from
   * @param indices tensor containing the indices of elements to gather
   * @param hidden tensor to store the result value
   */
  void forwarding_operation(const Tensor &input, const Tensor &indices,
                            Tensor &hidden) final;

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
  const std::string getType() const final { return GatherLayer::type; };

  std::tuple<props::Print, props::Axis> gather_props;
  unsigned int axis = 0;
  bool support_backwarding;

  inline static const std::string type = "gather";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GATHER_LAYER_H__ */
