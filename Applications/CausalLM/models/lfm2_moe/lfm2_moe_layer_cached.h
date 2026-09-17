// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_layer_cached.h
 * @date   06 July 2026
 * @brief  Cached-Slim MoE layer for LFM2-MoE (FSU experts + LRU expert cache).
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Same LFM2 routing as Lfm2MoELayer (sigmoid router + expert bias for
 *         top-k selection). Expert projection weights are FSU (virtual) tensors
 *         kept in a bounded LRU cache: recently used experts stay mmap-resident
 *         while cold experts are evicted. Inference-only; uses only
 *         incremental_forwarding (forwarding() is a no-op, as in the Qwen
 *         cached-slim variant).
 */

#ifndef __LFM2_MOE_LAYER_CACHED_H__
#define __LFM2_MOE_LAYER_CACHED_H__
#ifdef __cplusplus

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <acti_func.h>
#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <list>
#include <unordered_map>

namespace causallm {

/**
 * @class   Lfm2CachedSlimMoELayer
 * @brief   Cached-Slim (LRU streaming-expert) MoE layer for LFM2-MoE
 */
class WIN_EXPORT Lfm2CachedSlimMoELayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of the LFM2 Cached-Slim MoE layer
   */
  Lfm2CachedSlimMoELayer();

  /**
   * @brief Destructor of the LFM2 Cached-Slim MoE layer
   */
  ~Lfm2CachedSlimMoELayer() = default;

  /**
   * @brief Move constructor (deleted: holds a mutex).
   */
  Lfm2CachedSlimMoELayer(Lfm2CachedSlimMoELayer &&rhs) = delete;

  /**
   * @brief Move assignment operator (deleted: holds a mutex).
   */
  Lfm2CachedSlimMoELayer &operator=(Lfm2CachedSlimMoELayer &&rhs) = delete;

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
   * @copydoc Layer::save(std::ofstream &file, RunLayerContext &run_context,
   * bool opt_var, ml::train::ExecutionMode mode, bool trainable,
   * TensorDim::DataType dtype, ml::train::ISA target_isa)
   * @note Overridden so the router gate and expert bias are never quantized
   *       (see Lfm2MoELayer::save for the full rationale).
   */
  void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return Lfm2CachedSlimMoELayer::type;
  };

  /**
   * @brief Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::updateTensorsByInputDimensions
   */
  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  static constexpr const char *type =
    "lfm2_moe_cached_slim"; /**< type of the layer */

private:
  unsigned int num_experts;      /**< number of experts */
  unsigned int topk;             /**< number of experts per token, i.e., topk */
  nntrainer::ActiFunc acti_func; /**< activation function for the expert */
  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::MoEActivation>
    moe_props;

  // weight indices
  std::vector<unsigned int> expert_gate_up_proj_indices;
  std::vector<unsigned int> expert_down_proj_indices;
  unsigned int gate_idx;
  unsigned int expert_bias_idx;

  // LRU expert cache bookkeeping
  std::list<int> loaded_expert_deque;
  std::unordered_map<int, std::list<int>::iterator> iteration_map;
  std::vector<bool> need_load;
  unsigned int cache_capacity; /**< mmap-resident expert limit */

  // intermediate tensor indices
  unsigned int router_logits_idx;
  unsigned int decode_expert_output_idx;
  unsigned int decode_gate_up_output_idx;
  unsigned int decode_activation_output_idx;

  /** Reusable backing tensors shared by all active experts in one pass. */
  struct ExpertWorkspace {
    nntrainer::Tensor *token_input;
    nntrainer::Tensor *expert_output;
    nntrainer::Tensor *gate_up_output;
    nntrainer::Tensor *activation_output;
  };

  /**
   * @brief Build the per-expert token assignments for LFM2 routing.
   * @param router_logits Raw router logits tensor [total_tokens, 1, 1, E]
   * @param expert_bias Per-expert bias tensor [1, 1, 1, E]
   * @param total_tokens number of tokens routed
   * @param[out] expert_assignments per-expert list of (token index, weight)
   * @param[out] extra_top_k flattened expert ids of the extended top-k set used
   *             only for LRU cache ordering (prefetch)
   */
  void buildExpertAssignments(
    const nntrainer::Tensor &router_logits,
    const nntrainer::Tensor &expert_bias, unsigned int total_tokens,
    std::vector<std::vector<std::pair<unsigned, float>>> &expert_assignments,
    std::vector<int> &extra_top_k);

  /**
   * @brief Run one expert as a token batch into compact output storage
   */
  inline void compute_expert_forward(
    const nntrainer::Tensor &input, nntrainer::Tensor &output,
    const std::vector<std::pair<unsigned, float>> &token_assignments,
    const nntrainer::Tensor &gate_up_proj, const nntrainer::Tensor &down_proj,
    unsigned int hidden_size, ExpertWorkspace &workspace);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __LFM2_MOE_LAYER_CACHED_H__ */
