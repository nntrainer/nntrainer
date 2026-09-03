// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma4_moe_layer.h
 * @brief  Gemma4 sparse Mixture-of-Experts inference layer.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#ifndef __GEMMA4_MOE_LAYER_H__
#define __GEMMA4_MOE_LAYER_H__

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

#include <memory>

namespace causallm {

namespace props {

/**
 * @brief Maximum number of Gemma4 experts kept mmap'd by the LRU cache.
 *
 * A value of zero preserves the resident-weight path. A positive value makes
 * every expert projection virtual and enables on-demand mapping.
 */
class Gemma4MoECacheSize : public nntrainer::Property<unsigned int> {
public:
  Gemma4MoECacheSize(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "moe_cache_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

class Gemma4ExpertCache;

/**
 * @class Gemma4MoELayer
 * @brief Gemma4 sparse expert layer with its model-specific router.
 */
class WIN_EXPORT Gemma4MoELayer final : public nntrainer::LayerImpl {
public:
  Gemma4MoELayer();
  ~Gemma4MoELayer();

  Gemma4MoELayer(Gemma4MoELayer &&rhs) noexcept = delete;
  Gemma4MoELayer &operator=(Gemma4MoELayer &&rhs) = delete;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void setProperty(const std::vector<std::string> &values) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  const std::string getType() const override { return type; }
  bool supportBackwarding() const override { return false; }

  void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  static constexpr const char *type = "gemma4_moe";

private:
  unsigned int num_experts;
  unsigned int topk;
  unsigned int cache_size;
  float epsilon;
  nntrainer::ActiFunc acti_func;
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             nntrainer::props::Epsilon, props::Gemma4MoECacheSize>
    moe_props;

  std::unique_ptr<Gemma4ExpertCache> expert_cache;

  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int router_idx;
  unsigned int router_scale_idx;
  unsigned int per_expert_scale_idx;
  unsigned int router_input_scaled_idx;
  unsigned int router_logits_idx;

  void registerExpertCache(nntrainer::RunLayerContext &context);

  void forwardTensors(nntrainer::RunLayerContext &context,
                      nntrainer::Tensor &expert_input,
                      nntrainer::Tensor &router_input,
                      nntrainer::Tensor &output,
                      nntrainer::Tensor &router_input_scaled,
                      nntrainer::Tensor &router_logits);

  void computeExpertForward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned int, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};

} // namespace causallm

#endif /* __GEMMA4_MOE_LAYER_H__ */
