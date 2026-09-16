// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Hyeong-Gwon Hong
 *
 * @file   causal_conv1d_layer.h
 * @date   01 April 2026
 * @brief  Causal depthwise Conv1D layer with conv-state cache for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeong-Gwon Hong
 * @bug    No known bugs except for NYI items
 *
 * Input/output tensor layout: [B, 1, T, W]
 *   B = batch, T = sequence length (height), W = features/channels (width)
 *
 * Weight layout: [1, 1, KERNEL_SIZE, W] FP32
 *   position 0 (w0): applied to the current token x_t
 *   position 1 (w1): applied to x_{t-1}
 *   position 2 (w2): applied to x_{t-2}
 *
 * Conv-state cache: [B, 1, KERNEL_SIZE-1, W] FP32 (MAX_LIFESPAN)
 *   state[b, 0, 0, :] = x_{t-2}  (oldest cached token)
 *   state[b, 0, 1, :] = x_{t-1}  (newest cached token)
 *
 * During single-token decode (to - from == 1):
 *   output[b, 0, t, f] = w0[f]*x[t,f] + w1[f]*state[1,f] + w2[f]*state[0,f]
 *   then: state[0] <- state[1], state[1] <- x[t]  (O(W) update)
 *
 * During prefill (to > from + 1):
 *   Computes all positions 0..to-1 and saves last KERNEL_SIZE-1 to state.
 */

#ifndef __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
#define __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <layer_devel.h>
#include <layer_impl.h>
#include <tensor_dim.h>

namespace causallm {

/**
 * @brief Causal depthwise Conv1D layer with conv-state cache for CausalLM
 */
class CausalConv1DLayer : public nntrainer::LayerImpl {
public:
  CausalConv1DLayer();
  ~CausalConv1DLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::getLayerDimensions(InitLayerContext &context)
   */
  std::array<std::vector<nntrainer::TensorDim>, 3>
  getLayerDimensions(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::updateTensorsByInputDimensions(InitLayerContext &context,
   * RunLayerContext &run_context)
   */
  std::vector<nntrainer::TensorDim> updateTensorsByInputDimensions(
    nntrainer::InitLayerContext &init_context,
    nntrainer::RunLayerContext &run_context) override;

  /** Full forwarding is unused; use incremental_forwarding. */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @brief Incremental forward.
   *  - Prefill  (to - from > 1): compute positions [0, to) and save state.
   *  - Decode   (to - from == 1): O(1) single-token using cached state.
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;

  const std::string getType() const override { return type; }

  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  void setProperty(const std::vector<std::string> &values) override;

  bool supportBackwarding() const override { return false; }

  inline static const std::string type = "causal_conv1d";

private:
  /** Kernel size is fixed at 3 (matches LFM2 conv_L_cache=3). */
  static constexpr unsigned int KERNEL_SIZE = 3;
  static constexpr size_t SINGLE_INOUT_IDX = 0;

  enum WeightIdx { weight = 0, NUM_WEIGHTS };
  enum TensorIdx { conv_state = 0, NUM_TENSORS };

  std::array<unsigned int, NUM_WEIGHTS> weight_idx;
  std::array<unsigned int, NUM_TENSORS> tensor_idx;

  void validateInputShape(const nntrainer::TensorDim &input_dim) const;
};

} // namespace causallm

#endif // __CAUSAL_LM_CAUSAL_CONV1D_LAYER_H__
