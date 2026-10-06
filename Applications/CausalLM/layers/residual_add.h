// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   residual_add.h
 * @date   28 Sep 2026
 * @brief  The residual add of a decoder block, with the HTP per-token
 *         decode hook (#132)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * out = in0; out += in1 -- AdditionLayer's two-input arithmetic, byte for
 * byte, with in0 the residual and in1 the addend. At one decode row it
 * first offers the row to the HTP ADD op (htpDecodeAdd), which holds the
 * residual on the DSP; on 1 the layer writes nothing. The models use it
 * only under NNTR_HTP_FORWARD (Transformer::RESIDUAL_ADD_TYPE), so the
 * core addition layer and every other model stay as they are.
 */

#ifndef __CAUSALLM_RESIDUAL_ADD_H__
#define __CAUSALLM_RESIDUAL_ADD_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief out = residual + addend, with the HTP ADD hook at a decode row.
 */
class ResidualAddLayer final : public nntrainer::Layer {
public:
  /** @brief Constructor */
  ResidualAddLayer() : Layer() {}

  /** @copydoc Layer::finalize(InitLayerContext &context) */
  void finalize(nntrainer::InitLayerContext &context) override;

  /** @copydoc Layer::forwarding(RunLayerContext &context, bool training) */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /** @copydoc Layer::incremental_forwarding() */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /** @copydoc Layer::calcDerivative(RunLayerContext &context) */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /** @copydoc bool supportBackwarding() const */
  bool supportBackwarding() const override { return true; }

  /** @copydoc Layer::exportTo() */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /** @copydoc Layer::getType() */
  const std::string getType() const override { return ResidualAddLayer::type; }

  /** @copydoc Layer::setProperty(const std::vector<std::string> &values) */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "residual_add";
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CAUSALLM_RESIDUAL_ADD_H__ */
