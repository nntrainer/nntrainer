// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   positional_encoding_layer.h
 * @date   16 August 2022
 * @brief  This file contains the positional encoding layer in transformer
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1607.06450
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __POSITIONAL_ENCODING_LAYER_H__
#define __POSITIONAL_ENCODING_LAYER_H__
#ifdef __cplusplus

#include <base_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace nntrainer {

/**
 * @class   Positional encoding Layer
 * @brief   Implementation of positional encoding layer which is described in
 * paper "Attention is all you need"
 */
class PositionalEncodingLayer : public Layer {
public:
  /**
   * @brief     Constructor of PositionalEncodingLayer
   */
  PositionalEncodingLayer();

  /**
   * @brief     Destructor of PositionalEncodingLayer
   */
  ~PositionalEncodingLayer();

  /**
   *  @brief  Move constructor of PositionalEncodingLayer.
   *  @param[in] PositionalEncodingLayer &&
   */
  PositionalEncodingLayer(PositionalEncodingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PositionalEncodingLayer to be moved.
   */
  PositionalEncodingLayer &operator=(PositionalEncodingLayer &&rhs) = default;

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
   * @copydoc bool supportBackwarding() const
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
  const std::string getType() const override {
    return PositionalEncodingLayer::type;
  };

  static constexpr const char *type = "positional_encoding";

private:
  std::tuple<props::MaxTimestep> positional_encoding_props;
  std::array<unsigned int, 2> tensor_idx;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MULTI_HEAD_ATTENTION_LAYER_H__ */
