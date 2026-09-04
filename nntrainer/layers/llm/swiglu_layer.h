// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   swiglu_layer.h
 * @date   6 June 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 *
 * @details One thin Layer owning shape and orchestration, with the kernel
 * delegated to the active backend through the tensor's op table
 * (in1.getOps()->swiglu(...)). It replaces the OpenCL SwiGLULayerCl fork,
 * which duplicated the whole layer in order to change the two lines that did
 * the maths.
 */

#ifndef __SWIGLU_LAYER_H__
#define __SWIGLU_LAYER_H__
#ifdef __cplusplus

#include <tuple>
#include <vector>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   SwiGLULayer
 * @brief   SwiGLU activation layer, out = silu(gate) * up
 */
class SwiGLULayer final : public Layer {

public:
  /**
   * @brief Construct a new SwiGLU layer object
   */
  SwiGLULayer() : Layer(), swiglu_props(props::Print()) {}

  /**
   * @brief Destroy the SwiGLU layer object
   */
  ~SwiGLULayer() {}

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SwiGLULayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "swiglu";

private:
  std::tuple<props::Print> swiglu_props; /**< swiglu layer properties */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SWIGLU_LAYER_H__ */
