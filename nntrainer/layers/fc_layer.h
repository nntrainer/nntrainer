// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Anirudh <b.saianirud@samsung.com>
 * @author Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <functional>
#include <layer_impl.h>
#include <limits>
#include <mutex>
#include <string>
#include <tensor.h>
#include <unordered_map>
#include <vector>

namespace nntrainer {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   * Prints final QAT calibration stats when lora_qat was active.
   */
  ~FullyConnectedLayer();

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayer(FullyConnectedLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayer &operator=(FullyConnectedLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  static constexpr const char *type = "fully_connected";

  struct LoRAQATStats {
    float a_min = 0, a_max = 0, a_scale = 0;
    float b_min = 0, b_max = 0, b_scale = 0;
    bool valid = false;
  };

  /** Look up per-block stats for a layer by name (updated each forward).
   * Thread-safe. */
  static LoRAQATStats getRegisteredStats(const std::string &layer_name);

  /**
   * Look up per-block EMA scales (N×K layout) for a layer. Returns empty
   * vectors if the layer hasn't run a QAT forward pass yet. Thread-safe.
   */
  static std::pair<std::vector<float>, std::vector<float>>
  getRegisteredBlockScales(const std::string &layer_name);

private:
  static std::mutex s_registry_mutex;
  static std::unordered_map<std::string, LoRAQATStats> s_qat_registry;
  static std::unordered_map<std::string,
                            std::pair<std::vector<float>, std::vector<float>>>
    s_block_d_registry;

  float lora_scaling;
  std::tuple<props::Unit, props::LoraRank, props::LoraAlpha, props::LoraQAT,
             props::LoraWeightQ4>
    fc_props;                             /**< fc layer properties :
                                                unit - number of output neurons,
                                                lora_rank - rank of lora (optional)
                                                lora_alpha - alpha for LoRA scaling
                                                lora_qat - enable per-block Q4_0 fake-quant
                                                lora_weight_q4 - Q4_0 tensors for inference */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 4> lora_idx;   /**< indices of the lora weights */
  std::unique_ptr<nntrainer::Quantizer> quantizer;
  bool skip_prefill = false;

  float momentum;          /**< EMA momentum for per-block scale update */
  std::string layer_name_; // captured in finalize(), keys s_qat_registry

  // QAT: per-block EMA scales for loraA and loraB (one float per 32-elem
  // block). Lazy-initialized on first forward pass when tensor sizes are known.
  std::vector<float> lora_a_block_d;
  std::vector<float> lora_b_block_d;

  // QAT: cached fake-quantized adapters from last forward pass (used in STE
  // backward)
  Tensor a_fq, b_fq;

  /**
   * @brief Per-block Q4_0 fake-quantization with EMA block scales.
   *        Training: updates EMA for each block, quantizes with EMA scale.
   *        Validation: uses current EMA scales without updating them.
   *        Backward is STE: gradient passes through unchanged.
   */
  Tensor fakeQuantizeQ4_0(const Tensor &x, std::vector<float> &block_d,
                          bool training);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
