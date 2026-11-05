// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   gemma4_moe_layer_gpu.h
 * @date   02 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Gemma4 sparse MoE layer for the OpenCL (gpu) engine.
 *
 * @details Same weights, properties and routing as Gemma4MoELayer; only the
 * expert half of the forward pass moves to the device:
 *
 *  - routing stays on the host (rms_norm, router GEMM, softmax, top-k,
 *    renormalisation, per-expert scale), on host scratch, so the selected
 *    experts and their weights are the CPU layer's bit for bit;
 *  - for each selected expert, in ascending expert id: its token rows are
 *    gathered into a device workspace, projected by the device FC
 *    (ComputeOps::fc_expert -- the Q4_0 / QS4CX / FP32 FC kernels, or a
 *    registered expert-only format), gated by the device GeGLU (tanh or exact
 * erf), projected down, and added back into the output rows with the routing
 * weights.
 *
 * Every expert weight stays resident (moe_cache_size = 0). The virtual expert
 * cache of the CPU layer has no device counterpart yet, so a non-zero cache
 * size is refused at finalize.
 */

#ifndef __GEMMA4_MOE_LAYER_GPU_H__
#define __GEMMA4_MOE_LAYER_GPU_H__

#include <compute_ops.h>
#include <gemma4_moe_layer.h>

namespace causallm {

/**
 * @brief Gemma4MoELayer whose experts run on the gpu engine.
 */
class WIN_EXPORT Gemma4MoELayerGpu final : public Gemma4MoELayer {
public:
  Gemma4MoELayerGpu();
  ~Gemma4MoELayerGpu();

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @brief Read the weights, building the device expert copy at load.
   * @details When the backend keeps its own copy of the experts (the TQ2CX
   * expert stack), the expert records are uploaded straight from the weight
   * file mapping and never copied into the host payloads (single copy, see
   * nntrainer::tq2cxSingleCopyOn); the other weights are read as usual.
   * Without a mapping the payloads are read, uploaded and then released.
   */
  void read(nntrainer::ReadSource src, nntrainer::RunLayerContext &run_context,
            bool opt_var, ml::train::ExecutionMode mode, bool trainable,
            nntrainer::TensorDim::DataType defineWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  /**
   * @copydoc read(ReadSource, ...)
   */
  void read(std::ifstream &file, nntrainer::RunLayerContext &run_context,
            bool opt_var, ml::train::ExecutionMode mode, bool trainable,
            nntrainer::TensorDim::DataType defineWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

protected:
  /** FP16 activations run here: device rows, a widened host router */
  bool acceptsFp16Activations() const override { return true; }

  /**
   * @copydoc Gemma4MoELayer::forwardTensors
   */
  void forwardTensors(nntrainer::RunLayerContext &context,
                      nntrainer::Tensor &expert_input,
                      nntrainer::Tensor &router_input,
                      nntrainer::Tensor &output,
                      nntrainer::Tensor &router_input_scaled,
                      nntrainer::Tensor &router_logits) override;

private:
  unsigned int gathered_idx; /**< [T, hidden] gathered token rows */
  unsigned int gate_out_idx; /**< [T, intermediate] gate projection */
  unsigned int up_out_idx;   /**< [T, intermediate] up projection */
  unsigned int act_out_idx;  /**< [T, intermediate] gelu(gate) * up */
  unsigned int down_out_idx; /**< [T, hidden] down projection */
  bool prebuilt;             /**< device FC backings built for the experts */
  bool exact_gelu;           /**< erf gelu gate (HF "gelu") */

  /** host scratch for the router (see routeTokens) */
  nntrainer::Tensor host_router_input; /**< FP32 copy of FP16 router rows */
  nntrainer::Tensor host_router_scaled;
  nntrainer::Tensor host_router_logits;

  /**
   * @brief Build the device FC backing of every expert weight once, so the
   *        first prefill does not pay the per-weight repack.
   */
  void prebuildExpertWeights(nntrainer::RunLayerContext &context);

  /**
   * @brief The expert weights of this layer as a group (weights only).
   */
  nntrainer::MoeExpertGroup
  expertWeightGroup(nntrainer::RunLayerContext &context);

  /**
   * @brief Build the backend's expert copy now and release the host
   *        payloads: @a records null = read from the filled payloads (whose
   *        pages are then released), else from those bytes (the payloads
   *        were never filled). No-op returning false when the backend keeps
   *        no copy or single copy is off.
   */
  bool buildDeviceExperts(nntrainer::RunLayerContext &context,
                          const std::vector<const void *> *records);

  /**
   * @brief Route @a total_tokens rows on the host (reads a device-only
   *        router row back first).
   */
  void hostRoute(nntrainer::RunLayerContext &context,
                 nntrainer::Tensor &router_input, unsigned int total_tokens,
                 unsigned int hidden_size, Routing &routing);

  /**
   * @brief The per-expert loop (formats the grouped launch does not serve).
   */
  void forwardExpertLoop(nntrainer::RunLayerContext &context,
                         nntrainer::Tensor &expert_input,
                         nntrainer::Tensor &output, const Routing &routing,
                         unsigned int total_tokens, unsigned int hidden_size);
};

} // namespace causallm

#endif /* __GEMMA4_MOE_LAYER_GPU_H__ */
