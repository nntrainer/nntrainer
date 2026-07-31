// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gate_up_layer.h
 * @date   29 July 2026
 * @brief  Batched FFN up/gate projection layer: two independent Q4_0
 *         matmuls sharing one activation, dispatched together via
 *         Tensor::dot(vector<Tensor*>, vector<Tensor*>, ...) so a backend's
 *         ComputeOps can collapse them into one dispatch (e.g. the Hexagon
 *         cDSP bridge's gemm_q4_0_batch_fp32) instead of two.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 * Mirrors qkv_layer.h's structure (same pattern for the analogous Q/K/V
 * case), sized down to 2 outputs instead of 3. A first-class nntrainer layer
 * (not app-specific) so HexagonContext can register it alongside
 * FullyConnectedLayer without depending on application code.
 */

#ifndef __GATE_UP_LAYER_H__
#define __GATE_UP_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

namespace props {

class UpUnit : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "up_unit";
  using prop_tag = uint_prop_tag;
};

class GateUnit : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "gate_unit";
  using prop_tag = uint_prop_tag;
};

} // namespace props

/**
 * @class   GateUpLayer
 * @brief   Batched FFN up + gate projection (shared activation)
 */
class GateUpLayer : public LayerImpl {
public:
  GateUpLayer();
  ~GateUpLayer() = default;

  GateUpLayer(GateUpLayer &&rhs) noexcept = default;
  GateUpLayer &operator=(GateUpLayer &&rhs) = default;

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
  const std::string getType() const override { return GateUpLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   *
   * Inference-only - calcDerivative/calcGradient throw rather than silently
   * no-op.
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  void updateTensorsByInputDimensions(
    RunLayerContext &context,
    std::vector<TensorDim> input_dimensions) override;

  static constexpr const char *type = "gate_up_layer";

private:
  std::tuple<props::UpUnit, props::GateUnit> gate_up_props;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GATE_UP_LAYER_H__ */
