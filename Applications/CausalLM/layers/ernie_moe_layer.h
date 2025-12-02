// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   ernie_moe_layer.h
 * @brief  ernie 4.5 moe layer header
 * @date   02 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef NNTRAINER_ERNIE_MOE_LAYER_H
#define NNTRAINER_ERNIE_MOE_LAYER_H
#ifdef __cplusplus

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <list>

namespace causallm {

class ErnieMoELayer : public nntrainer::LayerImpl {
public:
  ErnieMoELayer();

  /**
   * @brief     Destructor of Mixture of Expert Layer
   */
  ~ErnieMoELayer() = default;

  /**
   * @brief  Move constructor.
   *  @param[in] ErnieMoELayer &&
   */
  ErnieMoELayer(ErnieMoELayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs ErnieMoELayer to be moved.
   */
  ErnieMoELayer &operator=(ErnieMoELayer &&rhs) = default;

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
  const std::string getType() const override { return ErnieMoELayer::type; };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @brief Update tensors by input dimensions
   * @param context RunLayerContext
   * @param input_dimensions Input dimensions
   */
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  static constexpr const char *type = "ernie_moe"; /**< type of the layer */

private:
  unsigned int num_experts;        /**< number of experts */
  unsigned int num_shared_experts; /**< number of shared experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             props::NumSharedExperts, props::MoENormMin>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int shared_gate_proj_idx;
  unsigned int shared_up_proj_idx;
  unsigned int shared_down_proj_idx;

  std::list<int> loaded_expert_deque;
  std::unordered_map<int, std::list<int>::iterator> iteration_map;
  std::unordered_map<int, double> expert_predict_scores;
  std::vector<bool> need_load;
  std::mutex cache_mutex;

  unsigned int gate_idx;
  unsigned int e_score_correction_bias_idx;

  unsigned int router_logits_idx;
  unsigned int expert_mask_idx;

  /**
   * @brief Compute expert forward pass
   * @param input Input tensor
   * @param output Output tensor
   * @param token_assignments Token assignments
   * @param gate_proj Gate projection weight
   * @param up_proj Up projection weight
   * @param down_proj Down projection weight
   * @param hidden_size Hidden size
   */
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* NNTRAINER_ERNIE_MOE_LAYER_H */