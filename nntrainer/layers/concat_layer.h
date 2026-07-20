// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.h
 * @date   27 Oct 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 */

#ifndef __CONCAT_LAYER_H__
#define __CONCAT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <tensor_dim.h>
namespace nntrainer {

/**
 * @class   Concat Layer
 * @brief   Concat Layer
 */
class ConcatLayer : public Layer {
public:
  /**
   * @brief     Constructor of Concat Layer
   */
  ConcatLayer();

  /**
   * @brief     Destructor of Concat Layer
   */
  ~ConcatLayer() = default;

  /**
   *  @brief  Move constructor of ConcatLayer.
   *  @param[in] ConcatLayer &&
   */
  ConcatLayer(ConcatLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ConcatLayer to be moved.
   */
  ConcatLayer &operator=(ConcatLayer &&rhs) = default;

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
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ConcatLayer::type; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override {
    setBatch(batch);
  }

  static constexpr const char *type = "concat";

private:
  unsigned int leading_helper_dim; /**< batch dimension of helper dimension not
                                containing the actual batch */
  std::vector<TensorDim>
    input_reshape_helper;          /** helper dimension to reshape inputs */
  TensorDim output_reshape_helper; /** helper dimension to reshape outputs */
  std::tuple<props::ConcatDimension> concat_props;
  unsigned int concat_dimension_cache =
    1; /**< resolved in finalize for
      forwarding (NHWC channel-concat path) */

  /**
   * @brief set batch for the internal variables
   *
   * @param batch update batch size
   */
  void setBatch(unsigned int batch) {
    for (auto &irh : input_reshape_helper)
      irh.batch(batch * leading_helper_dim);
    output_reshape_helper.batch(batch * leading_helper_dim);
  }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONCAT_LAYER_H__ */
