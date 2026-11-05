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
#include <tensor.h>
#include <utility>
#include <vector>

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

/**
 * @brief Whether the router RMSNorm carries a learned per-feature scale.
 *
 * The HF Gemma4 router normalizes its input without a scale (followed by the
 * router scale and hidden^-0.5, which this layer always applies). Set this
 * for checkpoints whose router norm has its own weight; it is then read as
 * the extra weight "router_norm" (fp32, [hidden]) and applied after the
 * normalization.
 */
class Gemma4RouterNormWeight : public nntrainer::Property<bool> {
public:
  Gemma4RouterNormWeight(bool value = false) { set(value); }
  static constexpr const char *key = "router_norm_weight";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief Stream the experts from the weight file into device slots (the gpu
 *        engine's expert slot arena) instead of keeping all of them resident.
 *
 * Needs a positive moe_cache_size: the slots per layer. The cpu engine maps
 * virtual experts through its own cache (moe_cache_size alone) and ignores
 * this.
 */
class Gemma4MoEFsu : public nntrainer::Property<bool> {
public:
  Gemma4MoEFsu(bool value = false) { set(value); }
  static constexpr const char *key = "moe_fsu";
  using prop_tag = nntrainer::bool_prop_tag;
};

} // namespace props

class Gemma4ExpertCache;

/**
 * @class Gemma4MoELayer
 * @brief Gemma4 sparse expert layer with its model-specific router.
 */
class WIN_EXPORT Gemma4MoELayer : public nntrainer::LayerImpl {
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

protected:
  unsigned int num_experts;
  unsigned int topk;
  unsigned int cache_size;
  float epsilon;
  nntrainer::ActiFunc acti_func;
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation,
             nntrainer::props::Epsilon, props::Gemma4MoECacheSize,
             props::Gemma4RouterNormWeight, props::Gemma4MoEFsu>
    moe_props;

  std::unique_ptr<Gemma4ExpertCache> expert_cache;

  std::vector<unsigned int> expert_gate_proj_indices;
  std::vector<unsigned int> expert_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int router_idx;
  unsigned int router_scale_idx;
  unsigned int per_expert_scale_idx;
  unsigned int router_norm_idx;
  unsigned int router_input_scaled_idx;
  unsigned int router_logits_idx;

  /**
   * @brief Per-expert working buffers, sized once and sliced per expert.
   *
   * The expert loop runs up to num_experts times per layer per token step, so
   * allocating these inside computeExpertForward cost ~11k heap allocations
   * per prefill. They are grown to the largest token count any single expert
   * has been handed and then reused via getSharedDataTensor views.
   */
  unsigned int scratch_capacity;
  nntrainer::Tensor scratch_gathered;
  nntrainer::Tensor scratch_gate_out;
  nntrainer::Tensor scratch_up_out;
  nntrainer::Tensor scratch_activated;
  nntrainer::Tensor scratch_down_out;

  void ensureExpertScratch(unsigned int tokens, unsigned int hidden_size,
                           unsigned int intermediate_size,
                           nntrainer::TensorDim::TensorType tensor_type);

  void registerExpertCache(nntrainer::RunLayerContext &context);

  /**
   * @brief Routing of one forward call: per expert, the (token row, routing
   *        weight) pairs assigned to it in ascending token order, and the
   *        experts with at least one row in ascending expert id.
   */
  struct Routing {
    std::vector<std::vector<std::pair<unsigned int, float>>> assignments;
    std::vector<unsigned int> active;
  };

  /**
   * @brief Route @a total_tokens rows: rms_norm, router scale, router GEMM,
   *        softmax, top-k, renormalisation and per-expert scale.
   * @param router_input router rows [T, 1, 1, hidden] (read on the host)
   * @param router_input_scaled scratch [T, 1, 1, hidden]
   * @param router_logits scratch [T, 1, 1, num_experts]
   * @param routing result
   */
  void routeTokens(nntrainer::RunLayerContext &context,
                   nntrainer::Tensor &router_input,
                   nntrainer::Tensor &router_input_scaled,
                   nntrainer::Tensor &router_logits, unsigned int total_tokens,
                   unsigned int hidden_size, Routing &routing);

  /**
   * @brief Whether this implementation computes with FP16 activations. The
   *        host expert loop is FP32 only; the gpu layer accepts both.
   */
  virtual bool acceptsFp16Activations() const { return false; }

  virtual void forwardTensors(nntrainer::RunLayerContext &context,
                              nntrainer::Tensor &expert_input,
                              nntrainer::Tensor &router_input,
                              nntrainer::Tensor &output,
                              nntrainer::Tensor &router_input_scaled,
                              nntrainer::Tensor &router_logits);

  /**
   * @brief Decode step (one token) with TQ2CX experts: the gate and up
   * projections of every selected expert go to the batched GEMM in one call
   * (one fork / join instead of two per expert), then activation, down and
   * the weighted scatter run per expert in ascending expert order exactly as
   * computeExpertForward does, so the result is bit-identical to it.
   * @return false when the compute ops cannot take the batched call
   */
  bool computeGroupedDecode(
    nntrainer::RunLayerContext &context, const nntrainer::Tensor &input,
    nntrainer::Tensor &output, const std::vector<unsigned int> &active_experts,
    const std::vector<std::vector<std::pair<unsigned int, float>>>
      &expert_assignments,
    unsigned int hidden_size);

  nntrainer::Tensor scratch_grouped_gate_up;

  void computeExpertForward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned int, float>> &token_assignments,
    const nntrainer::Tensor &gate_proj, const nntrainer::Tensor &up_proj,
    const nntrainer::Tensor &down_proj, unsigned int hidden_size);
};

} // namespace causallm

#endif /* __GEMMA4_MOE_LAYER_H__ */
