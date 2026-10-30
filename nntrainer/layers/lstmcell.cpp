// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lstmcell.cpp
 * @date   17 March 2021
 * @brief  This is LSTMCell Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <layer_context.h>
#include <lstmcell.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

enum LSTMCellParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  ifgo,
  dropout_mask
};

LSTMCellLayer::LSTMCellLayer() : lstmcell_props(props::DropOutRate()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void LSTMCellLayer::finalize(InitLayerContext &context) {
  const Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  const Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props).get();
  const WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props).get();
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props).get();
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::Unit>(lstmcore_props).empty(),
                std::invalid_argument)
    << "unit property missing for lstmcell layer";
  const unsigned int unit = std::get<props::Unit>(lstmcore_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcore_props).get();
  const ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(lstmcore_props).get();
  const ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(lstmcore_props).get();

  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "LSTMCell layer expects 3 inputs(one for the input and two for the "
       "hidden/cell state) but got " +
         std::to_string(context.getNumInputs()) + " input(s)";

  // input_dim = [ batch_size, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[INOUT_INDEX::INPUT];
  NNTR_THROW_IF(input_dim.channel() != 1 || input_dim.height() != 1,
                std::invalid_argument)
    << "Input must be single time dimension for LSTMCell (shape should be "
       "[batch_size, 1, 1, feature_size])";
  // input_hidden_state_dim = [ batch, 1, 1, unit ]
  const TensorDim &input_hidden_state_dim =
    context.getInputDimensions()[INOUT_INDEX::INPUT_HIDDEN_STATE];
  NNTR_THROW_IF(input_hidden_state_dim.channel() != 1 ||
                  input_hidden_state_dim.height() != 1,
                std::invalid_argument)
    << "Input hidden state's dimension should be [batch, 1, 1, unit] for "
       "LSTMCell";
  // input_cell_state_dim = [ batch, 1, 1, unit ]
  const TensorDim &input_cell_state_dim =
    context.getInputDimensions()[INOUT_INDEX::INPUT_CELL_STATE];
  NNTR_THROW_IF(input_cell_state_dim.channel() != 1 ||
                  input_cell_state_dim.height() != 1,
                std::invalid_argument)
    << "Input cell state's dimension should be [batch, 1, 1, unit] for "
       "LSTMCell";
  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  TensorDim::TensorType weight_tensor_type = {context.getFormat(),
                                              context.getWeightDataType()};

  // output_hidden_state_dim = [ batch_size, 1, 1, unit ]
  const TensorDim output_hidden_state_dim = input_hidden_state_dim;
  // output_cell_state_dim = [ batch_size, 1, 1, unit ]
  const TensorDim output_cell_state_dim = input_cell_state_dim;

  std::vector<VarGradSpecV2> out_specs;
  out_specs.push_back(
    InitLayerContext::outSpec(output_hidden_state_dim, "output_hidden_state",
                              TensorLifespan::FORWARD_GRAD_LIFESPAN));
  out_specs.push_back(
    InitLayerContext::outSpec(output_cell_state_dim, "output_cell_state",
                              TensorLifespan::FORWARD_GRAD_LIFESPAN));
  context.requestOutputs(std::move(out_specs));

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // - weight_ih ( input to hidden )
  //  : [ 1, 1, feature_size, NUM_GATE x unit ] -> i, f, g, o
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit}, weight_tensor_type);
  wt_idx[LSTMCellParams::weight_ih] = context.requestWeight(
    weight_ih_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_ih", true);
  // - weight_hh ( hidden to hidden )
  //  : [ 1, 1, unit, NUM_GATE x unit ] -> i, f, g, o
  TensorDim weight_hh_dim({unit, NUM_GATE * unit}, weight_tensor_type);
  wt_idx[LSTMCellParams::weight_hh] = context.requestWeight(
    weight_hh_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // - bias_h ( input bias, hidden bias are integrate to 1 bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_h_dim({NUM_GATE * unit}, weight_tensor_type);
      wt_idx[LSTMCellParams::bias_h] = context.requestWeight(
        bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
        "bias_h", true);
    } else {
      // - bias_ih ( input bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_ih_dim({NUM_GATE * unit}, weight_tensor_type);
      wt_idx[LSTMCellParams::bias_ih] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_ih", true);
      // - bias_hh ( hidden bias )
      //  : [ 1, 1, 1, NUM_GATE x unit ] -> i, f, g, o
      TensorDim bias_hh_dim({NUM_GATE * unit}, weight_tensor_type);
      wt_idx[LSTMCellParams::bias_hh] = context.requestWeight(
        bias_hh_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_hh", true);
    }
  }

  /** ifgo_dim = [ batch_size, 1, 1, NUM_GATE * unit ] */
  const TensorDim ifgo_dim(batch_size, 1, 1, NUM_GATE * unit,
                           weight_tensor_type);
  wt_idx[LSTMCellParams::ifgo] =
    context.requestTensor(ifgo_dim, "ifgo", Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ batch_size, 1, 1, unit ]
    const TensorDim dropout_mask_dim(batch_size, 1, 1, unit,
                                     weight_tensor_type);
    wt_idx[LSTMCellParams::dropout_mask] =
      context.requestTensor(dropout_mask_dim, "dropout_mask", Initializer::NONE,
                            false, TensorLifespan::ITERATION_LIFESPAN);
  }

  if (context.getActivationDataType() == TensorDim::DataType::FP32) {
    acti_func.setActiFunc<float>(hidden_state_activation_type);
    recurrent_acti_func.setActiFunc<float>(recurrent_activation_type);
  } else if (context.getActivationDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    acti_func.setActiFunc<_FP16>(hidden_state_activation_type);
    recurrent_acti_func.setActiFunc<_FP16>(recurrent_activation_type);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void LSTMCellLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, lstmcell_props);
  LSTMCore::setProperty(remain_props);
}

void LSTMCellLayer::exportTo(Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  LSTMCore::exportTo(exporter, method);
  exporter.saveResult(lstmcell_props, method, this);
}

void LSTMCellLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcore_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcore_props).get();

  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props).get();

  const Tensor &input = context.getInput(INOUT_INDEX::INPUT);
  const Tensor &prev_hidden_state =
    context.getInput(INOUT_INDEX::INPUT_HIDDEN_STATE);
  const Tensor &prev_cell_state =
    context.getInput(INOUT_INDEX::INPUT_CELL_STATE);
  Tensor &hidden_state = context.getOutput(INOUT_INDEX::OUTPUT_HIDDEN_STATE);
  Tensor &cell_state = context.getOutput(INOUT_INDEX::OUTPUT_CELL_STATE);

  const unsigned int batch_size = input.getDim().batch();

  const Tensor &weight_ih =
    context.getWeight(wt_idx[LSTMCellParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[LSTMCellParams::weight_hh]);

  Tensor empty =
    Tensor("empty", weight_ih.getFormat(), weight_ih.getDataType());

  const Tensor &bias_h = !disable_bias && integrate_bias
                           ? context.getWeight(wt_idx[LSTMCellParams::bias_h])
                           : empty;
  const Tensor &bias_ih = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[LSTMCellParams::bias_ih])
                            : empty;
  const Tensor &bias_hh = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[LSTMCellParams::bias_hh])
                            : empty;

  Tensor &ifgo = context.getTensor(wt_idx[LSTMCellParams::ifgo]);

  forwardLSTM(batch_size, unit, disable_bias, integrate_bias, acti_func,
              recurrent_acti_func, input, prev_hidden_state, prev_cell_state,
              hidden_state, cell_state, weight_ih, weight_hh, bias_h, bias_ih,
              bias_hh, ifgo);

  if (dropout_rate > epsilon && training) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[LSTMCellParams::dropout_mask]);
    dropout_mask.dropout_mask(dropout_rate);
    hidden_state.multiply_i(dropout_mask);
  }
}

void LSTMCellLayer::calcDerivative(RunLayerContext &context) {
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[LSTMCellParams::ifgo]);
  const Tensor &weight_ih =
    context.getWeight(wt_idx[LSTMCellParams::weight_ih]);
  Tensor &outgoing_derivative =
    context.getOutgoingDerivative(INOUT_INDEX::INPUT);

  calcDerivativeLSTM(outgoing_derivative, weight_ih, d_ifgo);
}

void LSTMCellLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(lstmcore_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(lstmcore_props).get();

  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);

  const Tensor &input = context.getInput(INOUT_INDEX::INPUT);
  const Tensor &prev_hidden_state =
    context.getInput(INOUT_INDEX::INPUT_HIDDEN_STATE);
  Tensor &d_prev_hidden_state =
    context.getOutgoingDerivative(INOUT_INDEX::INPUT_HIDDEN_STATE);
  const Tensor &prev_cell_state =
    context.getInput(INOUT_INDEX::INPUT_CELL_STATE);
  Tensor &d_prev_cell_state =
    context.getOutgoingDerivative(INOUT_INDEX::INPUT_CELL_STATE);
  const Tensor &d_hidden_state =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT_HIDDEN_STATE);
  const Tensor &cell_state = context.getOutput(INOUT_INDEX::OUTPUT_CELL_STATE);
  const Tensor &d_cell_state =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT_CELL_STATE);

  unsigned int batch_size = input.getDim().batch();

  Tensor &d_weight_ih =
    context.getWeightGrad(wt_idx[LSTMCellParams::weight_ih]);
  const Tensor &weight_hh =
    context.getWeight(wt_idx[LSTMCellParams::weight_hh]);
  Tensor &d_weight_hh =
    context.getWeightGrad(wt_idx[LSTMCellParams::weight_hh]);

  Tensor empty =
    Tensor("empty", weight_hh.getFormat(), weight_hh.getDataType());

  Tensor &d_bias_h = !disable_bias && integrate_bias
                       ? context.getWeightGrad(wt_idx[LSTMCellParams::bias_h])
                       : empty;
  Tensor &d_bias_ih = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMCellParams::bias_ih])
                        : empty;
  Tensor &d_bias_hh = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[LSTMCellParams::bias_hh])
                        : empty;

  const Tensor &ifgo = context.getTensor(wt_idx[LSTMCellParams::ifgo]);
  Tensor &d_ifgo = context.getTensorGrad(wt_idx[LSTMCellParams::ifgo]);

  if (context.isGradientFirstAccess(wt_idx[LSTMCellParams::weight_ih])) {
    d_weight_ih.setZero();
  }
  if (context.isGradientFirstAccess(wt_idx[LSTMCellParams::weight_hh])) {
    d_weight_hh.setZero();
  }
  if (!disable_bias) {
    if (integrate_bias) {
      if (context.isGradientFirstAccess(wt_idx[LSTMCellParams::bias_h])) {
        d_bias_h.setZero();
      }
    } else {
      if (context.isGradientFirstAccess(wt_idx[LSTMCellParams::bias_ih])) {
        d_bias_ih.setZero();
      }
      if (context.isGradientFirstAccess(wt_idx[LSTMCellParams::bias_hh])) {
        d_bias_hh.setZero();
      }
    }
  }

  Tensor d_hidden_state_masked = Tensor(
    "d_hidden_state_masked", weight_hh.getFormat(), weight_hh.getDataType());

  if (dropout_rate > epsilon) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[LSTMCellParams::dropout_mask]);
    d_hidden_state.multiply(dropout_mask, d_hidden_state_masked);
  }

  calcGradientLSTM(batch_size, unit, disable_bias, integrate_bias, acti_func,
                   recurrent_acti_func, input, prev_hidden_state,
                   d_prev_hidden_state, prev_cell_state, d_prev_cell_state,
                   dropout_rate > epsilon ? d_hidden_state_masked
                                          : d_hidden_state,
                   cell_state, d_cell_state, d_weight_ih, weight_hh,
                   d_weight_hh, d_bias_h, d_bias_ih, d_bias_hh, ifgo, d_ifgo);
}

void LSTMCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const float dropout_rate = std::get<props::DropOutRate>(lstmcell_props);
  context.updateTensor(wt_idx[LSTMCellParams::ifgo], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(wt_idx[LSTMCellParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
