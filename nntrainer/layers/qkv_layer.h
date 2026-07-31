// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moone <jijoong.moon@samsung.com>
 *
 * @file   qkv_layer.h
 * @date   14 May 2020
 * @brief  Batched Q/K/V projection layer: three independent Q4_0 matmuls
 *         sharing one activation, dispatched together via
 *         Tensor::dot(vector<Tensor*>, vector<Tensor*>, ...) so a backend's
 *         ComputeOps can collapse them into one dispatch (e.g. the Hexagon
 *         cDSP bridge's gemm_q4_0_batch_fp32) instead of three.
 *
 *         A first-class nntrainer layer (not app-specific) so that
 *         HexagonContext - and any future non-CPU Context - can register it
 *         alongside FullyConnectedLayer without needing to depend on
 *         application code.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __QKV_LAYER_H__
#define __QKV_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

namespace props {

class QUnit : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "q_unit";
  using prop_tag = uint_prop_tag;
};

class KUnit : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "k_unit";
  using prop_tag = uint_prop_tag;
};

class VUnit : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "v_unit";
  using prop_tag = uint_prop_tag;
};

} // namespace props

/**
 * @class   QKVLayer
 * @brief   Batched Q/K/V projection (shared activation)
 */
class QKVLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of QKV Layer
   */
  QKVLayer();

  /**
   * @brief     Destructor of QKV Layer
   */
  ~QKVLayer() = default;

  /**
   *  @brief  Move constructor.
   */
  QKVLayer(QKVLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   */
  QKVLayer &operator=(QKVLayer &&rhs) = default;

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
  const std::string getType() const override { return QKVLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   *
   * Inference-only (this layer exists to batch Q4_0 accelerator dispatch;
   * this project never trains through it) - calcDerivative/calcGradient
   * throw rather than silently no-op.
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  void updateTensorsByInputDimensions(
    RunLayerContext &context,
    std::vector<TensorDim> input_dimensions) override;

  static constexpr const char *type = "qkv_layer";

private:
  std::tuple<props::QUnit, props::KUnit, props::VUnit> qkv_props;
  std::array<unsigned int, 3> weight_idx; /**< indices of the weights */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QKV_LAYER_H__ */
