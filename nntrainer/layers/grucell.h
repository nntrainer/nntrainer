// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   grucell.h
 * @date   28 Oct 2021
 * @brief  This is Gated Recurrent Unit Cell Layer Class of Neural Network
 * @see	   https://github.com/nntrainer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __GRUCELL_H__
#define __GRUCELL_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   GRUCellLayer
 * @brief   GRUCellLayer
 */
class GRUCellLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of GRUCellLayer
   */
  GRUCellLayer();

  /**
   * @brief     Destructor of GRUCellLayer
   */
  ~GRUCellLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] GRUCellLayer &&
   */
  GRUCellLayer(GRUCellLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs GRUCellLayer to be moved.
   */
  GRUCellLayer &operator=(GRUCellLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
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
  const std::string getType() const override { return GRUCellLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::getInputGradLifespan()
   */
  std::optional<TensorLifespan> getInputGradLifespan() const override {
    return TensorLifespan::CALC_GRAD_DERIV_LIFESPAN;
  }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "grucell";

private:
  static constexpr unsigned int NUM_GATE = 3;
  enum INOUT_INDEX {
    INPUT = 0,
    INPUT_HIDDEN_STATE = 1,
    OUTPUT = 0,
  };

  /**
   * Unit: number of output neurons
   * IntegrateBias: integrate bias_ih, bias_hh to bias_h
   * ResetAfter: Whether apply reset gate before/after the matrix
   * multiplication. Apply reset gate after the mulplication if true.
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * RecurrentActivation: activation type for recurrent. default is sigmoid
   * DropOutRate: dropout rate
   *
   * */
  std::tuple<props::Unit, props::IntegrateBias, props::ResetAfter,
             props::HiddenStateActivation, props::RecurrentActivation,
             props::DropOutRate>
    grucell_props;
  std::array<unsigned int, 7> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation function for h_t : default is sigmoid
   */
  ActiFunc acti_func;

  /**
   * @brief     activation function for recurrent : default is tanh
   */
  ActiFunc recurrent_acti_func;

  /**
   * @brief     to protect overflow
   */
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __GRUCELL_H__ */
