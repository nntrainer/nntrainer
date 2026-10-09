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
 *
 * With its options it is the block's whole epilogue (doc 57 section 5
 * step 4): out = scale * (in0 + rmsnorm(in1 [+ in2]) * gamma). in_norm
 * adds the gamma weight (the post-attention / post-FFN norm), use_weight
 * the scalar weight after it (the block's layer scalar), a third input
 * the second addend (the MoE output beside the dense one). At prefill the
 * accelerator runs all of it as one call (ComputeOps::rmsnorm_add_fp32);
 * a decode row and a backend without it run the three ops here.
 */

#ifndef __CAUSALLM_RESIDUAL_ADD_H__
#define __CAUSALLM_RESIDUAL_ADD_H__
#ifdef __cplusplus

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief out = scale * (residual + [rmsnorm](addend [+ addend2]) * [gamma]),
 *        with the HTP ADD hook at a plain decode row.
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

private:
  /** @brief The rows [from, to) of every batch, decode hook first. */
  void run(nntrainer::RunLayerContext &context, unsigned int from,
           unsigned int to);

  std::tuple<props::InNorm, props::UseWeight, props::Proj,
             nntrainer::props::Epsilon, nntrainer::props::SkipPrefill>
    props_;
  unsigned int proj_idx = 0; /**< proj: the projection's weight */
  bool proj = false;
  unsigned int gamma_idx = 0;
  unsigned int scale_idx = 0;
  unsigned int sum_idx = 0; /**< the CPU path's summed, normed addend */
  bool in_norm = false;
  bool use_scale = false;
  bool skip_prefill = false;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CAUSALLM_RESIDUAL_ADD_H__ */
