// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   slice_layer.h
 * @date   07 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is slice layer class (operation layer)
 */

#ifndef __SLICE_LAYER_H__
#define __SLICE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Slice Layer
 * @brief Slice Layer
 */
class SliceLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Slice Layer
   */
  SliceLayer() :
    UnaryOperationLayer(),
    slice_props(props::Print(), props::StartIndex(), props::EndIndex(),
                props::Axis()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Slice Layer
   */
  ~SliceLayer(){};

  /**
   *  @brief  Move constructor of Slice Layer.
   *  @param[in] SliceLayer &&
   */
  SliceLayer(SliceLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs SliceLayer to be moved.
   */
  SliceLayer &operator=(SliceLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for slice
   *
   * @param input tensor to be sliced from
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
  const std::string getType() const final { return SliceLayer::type; };

  std::tuple<props::Print, props::StartIndex, props::EndIndex, props::Axis>
    slice_props;

  inline static const std::string type = "slice";
  bool support_backwarding;
  unsigned int axis;
  unsigned int start;
  // TensorDim starts;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SLICE_LAYER_H__ */
