#ifndef __NNTR_QNNLINEAR_H__
#define __NNTR_QNNLINEAR_H__

#include <common_properties.h>
#include <iostream>
#include <layer_impl.h>
#include <qnn_context_var.h>
#include <qnn_rpc_manager.h>

namespace nntrainer {

/**
 * @class   QNNLinear
 * @brief   Linear (matmul + bias) layer offloaded to the QNN backend.
 *          Submits the op as a QNN graph fragment at forwarding time.
 */
class QNNLinear : public LayerImpl {
public:
  QNNLinear();
  virtual ~QNNLinear() = default;

  inline static const std::string type = "qnn_linear";

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return QNNLinear::type; };

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override{};

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override{};

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  void makeContext() {}

  void setRpcMem(QNNRpcManager &rpc_mem) {}

private:
  int in_features_;
  int out_features_;
  bool support_bias_;

  std::tuple<props::Unit, props::LoraRank, props::LoraAlpha>
    fc_props; /**< fc layer properties :
                    unit - number of output neurons,
                    lora_rank - rank of lora (optional)
                    lora_scaling - scaling factor of LoRA apply, i.e.,
                 lora_scaling = alpha / lora_rank */
  float lora_scaling;
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 4> lora_idx;   /**< indices of the lora weights */
                                          // Tensor weight_;
                                          // Tensor bias_;
};

} // namespace nntrainer

#endif
