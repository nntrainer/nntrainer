// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

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
   */
  ~FullyConnectedLayer() = default;

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

private:
  float lora_scaling;
  std::tuple<props::Unit, props::LoraRank, props::LoraAlpha>
    fc_props;                             /**< fc layer properties :
                                                unit - number of output neurons,
                                                lora_rank - rank of lora (optional)
                                                lora_scaling - scaling factor of LoRA apply, i.e.,
                                             lora_scaling = alpha / lora_rank */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 4> lora_idx;   /**< indices of the lora weights */
  std::unique_ptr<nntrainer::Quantizer> quantizer;
  bool skip_prefill = false;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */
