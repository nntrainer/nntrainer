// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   rnncell.cpp
 * @date   29 Oct 2021
 * @brief  This is Recurrent Cell Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <common_properties.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <rnncell.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// - weight_ih ( weights of input to hidden )
// - weight_hh ( weights of hidden to hidden )
// - bias_h ( input bias, hidden_bias )
// - bias_ih ( input bias )
// - bias_hh ( hidden bias )
enum RNNCellParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  dropout_mask
};

RNNCellLayer::RNNCellLayer() :
  LayerImpl(),
  rnncell_props(props::Unit(), props::IntegrateBias(),
                props::HiddenStateActivation() = ActivationType::ACT_TANH,
                props::DropOutRate()),
  acti_func(ActivationType::ACT_NONE, true),
  epsilon(1e-3f) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void RNNCellLayer::finalize(InitLayerContext &context) {
  const nntrainer::WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  const Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  const Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  NNTR_THROW_IF(std::get<props::Unit>(rnncell_props).empty(),
                std::invalid_argument)
    << "unit property missing for rnncell layer";
  const unsigned int unit = std::get<props::Unit>(rnncell_props).get();
  const bool integrate_bias =
    std::get<props::IntegrateBias>(rnncell_props).get();
  const nntrainer::ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "RNNCell layer expects 2 inputs(one for the input and hidden state for "
       "the other) but got " +
         std::to_string(context.getNumInputs()) + " input(s)";

  // input_dim = [ batch, 1, 1, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[INOUT_INDEX::INPUT];
  NNTR_THROW_IF(input_dim.channel() != 1 || input_dim.height() != 1,
                std::invalid_argument)
    << "Input must be single time dimension for RNNCell (shape should be "
       "[batch_size, 1, 1, feature_size])";
  // input_hidden_state_dim = [ batch, 1, 1, unit ]
  const TensorDim &input_hidden_state_dim =
    context.getInputDimensions()[INOUT_INDEX::INPUT_HIDDEN_STATE];
  NNTR_THROW_IF(input_hidden_state_dim.channel() != 1 ||
                  input_hidden_state_dim.height() != 1,
                std::invalid_argument)
    << "Input hidden state's dimension should be [batch, 1, 1, unit] for "
       "RNNCell";

  const unsigned int batch_size = input_dim.batch();
  const unsigned int feature_size = input_dim.width();

  // output_hidden_state_dim = [ batch, 1, 1, unit ]
  TensorDim output_hidden_state_dim(batch_size, 1, 1, unit);
  std::vector<VarGradSpecV2> out_specs;
  out_specs.push_back(InitLayerContext::outSpec(
    output_hidden_state_dim, "out", TensorLifespan::FORWARD_GRAD_LIFESPAN));
  context.requestOutputs(std::move(out_specs));

  // weight_initializer can be set seperately. weight_ih initializer,
  // weight_hh initializer kernel initializer & recurrent_initializer in keras
  // for now, it is set same way.

  // weight_ih_dim : [ 1, 1, feature_size, unit ]
  const TensorDim weight_ih_dim({feature_size, unit});
  wt_idx[RNNCellParams::weight_ih] = context.requestWeight(
    weight_ih_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_ih", true);
  // weight_hh_dim : [ 1, 1, unit, unit ]
  const TensorDim weight_hh_dim({unit, unit});
  wt_idx[RNNCellParams::weight_hh] = context.requestWeight(
    weight_hh_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_hh", true);
  if (!disable_bias) {
    if (integrate_bias) {
      // bias_h_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_h_dim({unit});
      wt_idx[RNNCellParams::bias_h] = context.requestWeight(
        bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
        "bias_h", true);
    } else {
      // bias_ih_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_ih_dim({unit});
      wt_idx[RNNCellParams::bias_ih] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_ih", true);
      // bias_hh_dim : [ 1, 1, 1, unit ]
      const TensorDim bias_hh_dim({unit});
      wt_idx[RNNCellParams::bias_hh] = context.requestWeight(
        bias_hh_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_hh", true);
    }
  }

  if (dropout_rate > epsilon) {
    // dropout_mask_dim = [ batch, 1, 1, unit ]
    const TensorDim dropout_mask_dim(batch_size, 1, 1, unit);
    wt_idx[RNNCellParams::dropout_mask] =
      context.requestTensor(dropout_mask_dim, "dropout_mask", Initializer::NONE,
                            false, TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);

  if (!acti_func.supportInPlace()) {
    throw exception::not_supported(
      "Out of place activation functions not supported");
  }
}

void RNNCellLayer::setProperty(const std::vector<std::string> &values) {
  const std::vector<std::string> &remain_props =
    loadProperties(values, rnncell_props);
  LayerImpl::setProperty(remain_props);
}

void RNNCellLayer::exportTo(Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rnncell_props, method, this);
}

void RNNCellLayer::forwarding(RunLayerContext &context, bool training) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const bool integrate_bias =
    std::get<props::IntegrateBias>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();

  const Tensor &input = context.getInput(INOUT_INDEX::INPUT);
  const Tensor &prev_hidden_state =
    context.getInput(INOUT_INDEX::INPUT_HIDDEN_STATE);
  Tensor &hidden_state = context.getOutput(INOUT_INDEX::OUTPUT_HIDDEN_STATE);

  const Tensor &weight_ih = context.getWeight(wt_idx[RNNCellParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[RNNCellParams::weight_hh]);
  Tensor empty;
  const Tensor &bias_h = !disable_bias && integrate_bias
                           ? context.getWeight(wt_idx[RNNCellParams::bias_h])
                           : empty;
  const Tensor &bias_ih = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[RNNCellParams::bias_ih])
                            : empty;
  const Tensor &bias_hh = !disable_bias && !integrate_bias
                            ? context.getWeight(wt_idx[RNNCellParams::bias_hh])
                            : empty;

  input.dot(weight_ih, hidden_state);
  prev_hidden_state.dot(weight_hh, hidden_state, false, false, 1.0f);
  if (!disable_bias) {
    if (integrate_bias) {
      hidden_state.add_i(bias_h);
    } else {
      hidden_state.add_i(bias_ih);
      hidden_state.add_i(bias_hh);
    }
  }

  acti_func.run_fn(hidden_state, hidden_state);

  if (dropout_rate > epsilon && training) {
    Tensor &dropout_mask =
      context.getTensor(wt_idx[RNNCellParams::dropout_mask]);
    dropout_mask.dropout_mask(dropout_rate);
    hidden_state.multiply_i(dropout_mask);
  }
}

void RNNCellLayer::calcDerivative(RunLayerContext &context) {
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();

  Tensor &outgoing_derivative =
    context.getOutgoingDerivative(INOUT_INDEX::INPUT);
  Tensor &d_prev_hidden_state =
    context.getOutgoingDerivative(INOUT_INDEX::INPUT_HIDDEN_STATE);
  const Tensor &hidden_state =
    context.getOutput(INOUT_INDEX::OUTPUT_HIDDEN_STATE);
  const Tensor &d_hidden_state =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT_HIDDEN_STATE);
  const Tensor &weight_ih = context.getWeight(wt_idx[RNNCellParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[RNNCellParams::weight_hh]);

  /// @note calculate d_hidden_state is duplicated with calcGradient. Needs
  /// optimization
  Tensor d_hidden_state_;
  if (dropout_rate > epsilon) {
    const Tensor &dropout_mask =
      context.getTensor(wt_idx[RNNCellParams::dropout_mask]);
    d_hidden_state.multiply(dropout_mask, d_hidden_state_);
  } else {
    d_hidden_state_.copy(d_hidden_state);
  }

  Tensor hidden_state_;
  hidden_state_.copy(hidden_state);
  acti_func.run_prime_fn(hidden_state_, d_hidden_state_, d_hidden_state_);

  d_hidden_state_.dot(weight_ih, outgoing_derivative, false, true);
  d_hidden_state_.dot(weight_hh, d_prev_hidden_state, false, true);
}

void RNNCellLayer::calcGradient(RunLayerContext &context) {
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const bool integrate_bias =
    std::get<props::IntegrateBias>(rnncell_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();

  const Tensor &input = context.getInput(INOUT_INDEX::INPUT);
  const Tensor &prev_hidden_state =
    context.getInput(INOUT_INDEX::INPUT_HIDDEN_STATE);
  const Tensor &hidden_state =
    context.getOutput(INOUT_INDEX::OUTPUT_HIDDEN_STATE);
  const Tensor &d_hidden_state =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT_HIDDEN_STATE);

  Tensor &d_weight_ih = context.getWeightGrad(wt_idx[RNNCellParams::weight_ih]);
  Tensor &d_weight_hh = context.getWeightGrad(wt_idx[RNNCellParams::weight_hh]);
  Tensor empty;
  Tensor &d_bias_h = !disable_bias && integrate_bias
                       ? context.getWeightGrad(wt_idx[RNNCellParams::bias_h])
                       : empty;
  Tensor &d_bias_ih = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[RNNCellParams::bias_ih])
                        : empty;
  Tensor &d_bias_hh = !disable_bias && !integrate_bias
                        ? context.getWeightGrad(wt_idx[RNNCellParams::bias_hh])
                        : empty;

  if (context.isGradientFirstAccess(wt_idx[RNNCellParams::weight_ih])) {
    d_weight_ih.setZero();
  }
  if (context.isGradientFirstAccess(wt_idx[RNNCellParams::weight_hh])) {
    d_weight_hh.setZero();
  }
  if (!disable_bias) {
    if (integrate_bias) {
      if (context.isGradientFirstAccess(wt_idx[RNNCellParams::bias_h])) {
        d_bias_h.setZero();
      }
    } else {
      if (context.isGradientFirstAccess(wt_idx[RNNCellParams::bias_ih])) {
        d_bias_ih.setZero();
      }
      if (context.isGradientFirstAccess(wt_idx[RNNCellParams::bias_hh])) {
        d_bias_hh.setZero();
      }
    }
  }

  Tensor d_hidden_state_;
  if (dropout_rate > epsilon) {
    const Tensor &dropout_mask =
      context.getTensor(wt_idx[RNNCellParams::dropout_mask]);
    d_hidden_state.multiply(dropout_mask, d_hidden_state_);
  } else {
    d_hidden_state_.copy(d_hidden_state);
  }

  Tensor hidden_state_;
  hidden_state_.copy(hidden_state);
  acti_func.run_prime_fn(hidden_state_, d_hidden_state_, d_hidden_state_);

  input.dot(d_hidden_state_, d_weight_ih, true, false, 1.0);
  prev_hidden_state.dot(d_hidden_state_, d_weight_hh, true, false, 1.0);
  if (!disable_bias) {
    if (integrate_bias) {
      d_hidden_state_.sum(0, d_bias_h, 1.0, 1.0);
    } else {
      d_hidden_state_.sum(0, d_bias_ih, 1.0, 1.0);
      d_hidden_state_.sum(0, d_bias_hh, 1.0, 1.0);
    }
  }
}

void RNNCellLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  const float dropout_rate = std::get<props::DropOutRate>(rnncell_props).get();
  if (dropout_rate > epsilon) {
    context.updateTensor(wt_idx[RNNCellParams::dropout_mask], batch);
  }
}

} // namespace nntrainer
