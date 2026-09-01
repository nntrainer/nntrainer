// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   qkv_layer.h
 * @date   14 May 2020
 * @brief  Fused query/key/value projection layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __QKV_LAYER_H__
#define __QKV_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

namespace props {

class QUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "q_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class KUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "k_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class VUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "v_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   QKVLayer
 * @brief   One layer holding the query, key and value projection weights, so a
 *          single dot() call produces all three outputs from one input read.
 */
WIN_EXPORT class QKVLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of QKV Layer
   */
  WIN_EXPORT QKVLayer();

  /**
   * @brief     Destructor of QKV Layer
   */
  WIN_EXPORT ~QKVLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] rhs QKVLayer to be moved.
   */
  WIN_EXPORT QKVLayer(QKVLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs QKVLayer to be moved.
   */
  WIN_EXPORT QKVLayer &operator=(QKVLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return QKVLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::updateTensorsByInputDimensions(RunLayerContext &context,
   * std::vector<TensorDim> input_dimensions)
   */
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "qkv_layer";

private:
  std::tuple<props::QUnit, props::KUnit, props::VUnit> qkv_props;
  std::array<unsigned int, 3> weight_idx; /**< indices of the weights */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QKV_LAYER_H__ */
