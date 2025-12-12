// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   gpt_oss_moe_layer.h
 * @date   09 June 2025
 * @brief  This is Mixture of Expert Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @todo   This layer does not support backwarding yet.
 */

#ifndef __GPT_OSS_MOE_LAYER_H__
#define __GPT_OSS_MOE_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @class   GptOssMoELayer
 * @brief   Mixture of Expert Layer
 */
class GptOssMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Mixture of Expert Layer
   */
  GptOssMoELayer();

  /**
   * @brief     Destructor of Mixture of Expert Layer
   */
  ~GptOssMoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] GptOssMoELayer &&
   */
  GptOssMoELayer(GptOssMoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs GptOssMoELayer to be moved.
   */
  GptOssMoELayer &operator=(GptOssMoELayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * &methods)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return GptOssMoELayer::type; };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "gpt_oss_moe"; /**< type of the layer */

private:
  unsigned int num_experts; /**< number of experts */
  unsigned int topk;        /**< number of experts per token, i.e., topk */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit>
    moe_props;

  // weight indeices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_gate_bias_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_up_bias_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  std::vector<unsigned int> expert_down_bias_indices;
  unsigned int gate_idx;
  unsigned int gate_bias_idx;

  // Intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;
  bool enable_bias = false;

  float alpha = 1.702;
  float limit = 7.0;

  /**
   * @brief expert forward computation without memory copies
   * @param input Input tensor (reshaped to [total_tokens, 1, 1, hidden_size])
   * @param output Output tensor to accumulate results
   * @param token_assignments Vector of (token_index, weight) pairs for this
   * expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param hidden_size Hidden dimension size
   */
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);

  /**
   * @brief expert forward computation without critical section
   * @param input Input tensor (reshaped to [total_tokens, 1, 1, hidden_size])
   * @param expert_output Expert-specific output tensor
   * @param token_assignments Vector of (token_index, weight) pairs for this
   * expert
   * @param gate_proj Gate projection weight tensor
   * @param up_proj Up projection weight tensor
   * @param down_proj Down projection weight tensor
   * @param hidden_size Hidden dimension size
   */
  inline void compute_expert_forward_no_critical(
    const nntrainer::Tensor &input, nntrainer::Tensor &expert_output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size,
    const nntrainer::Tensor &gate_bias = {},
    const nntrainer::Tensor &up_bias = {},
    const nntrainer::Tensor &down_bias = {});
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __GPT_OSS_MOE_LAYER_H__ */
