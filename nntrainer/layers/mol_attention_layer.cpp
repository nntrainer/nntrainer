// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   mol_attention_layer.cpp
 * @date   11 November 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MoL Attention Layer Class for Neural Network
 *
 */

#include <math.h>

#include <layer_context.h>
#include <mol_attention_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

MoLAttentionLayer::MoLAttentionLayer() :
  helper_exec(false),
  softmax(ActivationType::ACT_SOFTMAX, false),
  tanh(ActivationType::ACT_TANH, false),
  sigmoid(ActivationType::ACT_SIGMOID, false) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

MoLAttentionLayer::~MoLAttentionLayer() {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum MoLAttentionParams {
  query = 0,
  value = 1,
  state = 2,
  mask_len = 3,
  fc_w,
  fc_bias,
  fc_proj_w,
  fc_out,
  fc_tanh,
  fc_proj_out,
  scores,
  prob,
  prob_left,
  prob_right,
  u_neg_div,
  u_pos_div,
  dstate,
};

void MoLAttentionLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "MoL Attention layer needs 3-4 inputs.";

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[MoLAttentionParams::query];
  auto const &value_dim = all_dims[MoLAttentionParams::value];
  auto const &state_dim = all_dims[MoLAttentionParams::state];

  wt_idx[MoLAttentionParams::query] = MoLAttentionParams::query;
  wt_idx[MoLAttentionParams::value] = MoLAttentionParams::value;
  wt_idx[MoLAttentionParams::state] = MoLAttentionParams::state;
  wt_idx[MoLAttentionParams::mask_len] = MoLAttentionParams::mask_len;

  NNTR_THROW_IF(query_dim.width() != value_dim.width(), std::invalid_argument)
    << "Query and Value dimension mismatch for layer " << context.getName();

  NNTR_THROW_IF(std::get<props::Unit>(mol_props).empty(), std::invalid_argument)
    << "Number of units not provided for layer " << context.getName();
  auto unit = std::get<props::Unit>(mol_props).get();

  NNTR_THROW_IF(std::get<props::MoL_K>(mol_props).empty(),
                std::invalid_argument)
    << "MoL_K property not provided for layer " << context.getName();
  auto mol_k = std::get<props::MoL_K>(mol_props).get();

  NNTR_THROW_IF(mol_k != state_dim.width(), std::invalid_argument)
    << "MoL_K property mismatches the provided state dimension for layer"
    << context.getName();

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);

  TensorDim fc_w_dim = {query_dim.width(), unit};
  wt_idx[MoLAttentionParams::fc_w] = context.requestWeight(
    fc_w_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "fc_w", true);
  TensorDim fc_bias_dim = {unit};
  wt_idx[MoLAttentionParams::fc_bias] = context.requestWeight(
    fc_bias_dim, bias_initializer, weight_regularizer,
    weight_regularizer_constant, bias_decay, "fc_bias", true);

  TensorDim fc_proj_w_dim = {unit, 3 * mol_k};
  wt_idx[MoLAttentionParams::fc_proj_w] = context.requestWeight(
    fc_proj_w_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "fc_proj_w", true);

  TensorDim fc_out_dim = query_dim;
  fc_out_dim.width(fc_w_dim.width());
  wt_idx[MoLAttentionParams::fc_out] =
    context.requestTensor(fc_out_dim, "fc_out", Initializer::NONE, false,
                          TensorLifespan::FORWARD_FUNC_LIFESPAN);

  wt_idx[MoLAttentionParams::fc_tanh] =
    context.requestTensor(fc_out_dim, "fc_tanh", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);

  TensorDim fc_proj_out_dim = fc_out_dim;
  fc_proj_out_dim.width(fc_proj_w_dim.width());
  wt_idx[MoLAttentionParams::fc_proj_out] =
    context.requestTensor(fc_proj_out_dim, "fc_proj_out", Initializer::NONE,
                          false, TensorLifespan::ITERATION_LIFESPAN);

  TensorDim scores_dim =
    TensorDim({value_dim.batch(), 1, 1, value_dim.height()});
  wt_idx[MoLAttentionParams::scores] =
    context.requestTensor(scores_dim, "scores", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);

  TensorDim prob_dim = value_dim;
  prob_dim.width(mol_k);
  wt_idx[MoLAttentionParams::prob] =
    context.requestTensor(prob_dim, "prob", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[MoLAttentionParams::prob_left] =
    context.requestTensor(prob_dim, "prob_left", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[MoLAttentionParams::prob_right] =
    context.requestTensor(prob_dim, "prob_right", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[MoLAttentionParams::u_neg_div] =
    context.requestTensor(prob_dim, "u_neg_div", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[MoLAttentionParams::u_pos_div] =
    context.requestTensor(prob_dim, "u_pos_div", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  wt_idx[MoLAttentionParams::dstate] =
    context.requestTensor(state_dim, "dstate", Initializer::NONE, false,
                          TensorLifespan::BACKWARD_FUNC_LIFESPAN);

  if (context.getNumRequestedOutputs() == 2)
    context.setOutputDimensions({query_dim, state_dim});
  else
    context.setOutputDimensions({query_dim});
}

void MoLAttentionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &query = context.getInput(wt_idx[MoLAttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[MoLAttentionParams::value]);
  Tensor &state = context.getInput(wt_idx[MoLAttentionParams::state]);

  Tensor &output = context.getOutput(0);
  Tensor &fc_w = context.getWeight(wt_idx[MoLAttentionParams::fc_w]);
  Tensor &fc_bias = context.getWeight(wt_idx[MoLAttentionParams::fc_bias]);
  Tensor &fc_proj_w = context.getWeight(wt_idx[MoLAttentionParams::fc_proj_w]);
  Tensor &fc_out = context.getTensor(wt_idx[MoLAttentionParams::fc_out]);
  Tensor &fc_tanh = context.getTensor(wt_idx[MoLAttentionParams::fc_tanh]);
  Tensor &fc_proj_out =
    context.getTensor(wt_idx[MoLAttentionParams::fc_proj_out]);
  Tensor &scores = context.getTensor(wt_idx[MoLAttentionParams::scores]);
  Tensor &prob = context.getTensor(wt_idx[MoLAttentionParams::prob]);
  Tensor &prob_left = context.getTensor(wt_idx[MoLAttentionParams::prob_left]);
  Tensor &prob_right =
    context.getTensor(wt_idx[MoLAttentionParams::prob_right]);
  Tensor &u_neg_div = context.getTensor(wt_idx[MoLAttentionParams::u_neg_div]);
  Tensor &u_pos_div = context.getTensor(wt_idx[MoLAttentionParams::u_pos_div]);

  const TensorDim &input_dim = query.getDim();
  unsigned int batch = input_dim.batch();
  auto mol_k = std::get<props::MoL_K>(mol_props).get();

  /** reset helper state */
  helper_exec = false;

  query.dot(fc_w, fc_out);
  fc_out.add_i(fc_bias);

  tanh.run_fn(fc_out, fc_tanh);

  fc_tanh.dot(fc_proj_w, fc_proj_out);

  Tensor kappa_src, beta_src, alpha_src;
  kappa_src.copy_with_stride(
    fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, 0, false));
  beta_src.copy_with_stride(
    fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k, false));
  alpha_src.copy_with_stride(
    fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k * 2, false));

  kappa_src.apply_i<float>(&expf);
  beta_src.apply_i<float>(&expf);
  Tensor kappa = kappa_src;
  Tensor beta = beta_src;

  Tensor alpha;
  softmax.run_fn(alpha_src, alpha);

  fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, 0, false)
    .copy_with_stride(kappa);
  fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k, false)
    .copy_with_stride(beta);
  fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k * 2, false)
    .copy_with_stride(alpha);

  /** @todo cache u_base, u_pos, u_neg */
  Tensor u_base = Tensor(TensorDim({batch, 1, value.height(), mol_k}));
  for (unsigned int b = 0; b < batch; b++) {
    for (unsigned int h = 0; h < u_base.height(); h++) {
      float *u_data = u_base.getAddress<float>(b, 0, h, 0);
      std::fill(u_data, u_data + u_base.width(), static_cast<float>(h + 1));
    }
  }

  Tensor u_pos = u_base.add(0.5f);
  u_base.add_i(-0.5f);
  Tensor u_neg = u_base;

  Tensor beta_eps = beta.add(1e-8f);

  Tensor u_pos_m, u_neg_m;
  if (context.getNumOutputs() == 2) {
    Tensor &updated_state = context.getOutput(1);
    state.add(kappa, updated_state);
    u_pos_m = u_pos.subtract(updated_state);
    u_neg_m = u_neg.subtract(updated_state);
  } else {
    Tensor updated_state = state.add(kappa);
    u_pos_m = u_pos.subtract(updated_state);
    u_neg_m = u_neg.subtract(updated_state);
  }

  u_pos_m.divide(beta_eps, u_pos_div);
  sigmoid.run_fn(u_pos_div, prob_left);

  u_neg_m.divide(beta_eps, u_neg_div);
  sigmoid.run_fn(u_neg_div, prob_right);

  prob_left.subtract(prob_right, prob);

  Tensor prob_scaled = prob.multiply(alpha);
  prob_scaled.sum(3, scores);

  if (context.getNumInputs() == 4) {
    Tensor mask = Tensor(scores.getDim());
    mask.filter_mask(context.getInput(wt_idx[MoLAttentionParams::mask_len]),
                     false);
    scores.multiply_i(mask);
  }

  scores.dotBatched(value, output);
}

void MoLAttentionLayer::calcDerivativeHelper(RunLayerContext &context,
                                             Tensor &dstate) {
  /** optimize temporary tensor usage here */
  Tensor &query = context.getInput(wt_idx[MoLAttentionParams::query]);
  Tensor &value = context.getInput(wt_idx[MoLAttentionParams::value]);

  const Tensor &derivative = context.getIncomingDerivative(0);

  Tensor &fc_proj_out =
    context.getTensor(wt_idx[MoLAttentionParams::fc_proj_out]);
  Tensor &dfc_proj_out =
    context.getTensor(wt_idx[MoLAttentionParams::fc_proj_out]);
  Tensor &scores = context.getTensor(wt_idx[MoLAttentionParams::scores]);
  Tensor &prob = context.getTensor(wt_idx[MoLAttentionParams::prob]);
  Tensor &prob_left = context.getTensor(wt_idx[MoLAttentionParams::prob_left]);
  Tensor &prob_right =
    context.getTensor(wt_idx[MoLAttentionParams::prob_right]);
  Tensor &u_neg_div = context.getTensor(wt_idx[MoLAttentionParams::u_neg_div]);
  Tensor &u_pos_div = context.getTensor(wt_idx[MoLAttentionParams::u_pos_div]);

  const TensorDim &input_dim = query.getDim();
  unsigned int batch = input_dim.batch();
  auto mol_k = std::get<props::MoL_K>(mol_props).get();

  Tensor kappa, beta, alpha;
  kappa.copy_with_stride(
    fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, 0, false));
  beta.copy_with_stride(
    fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k, false));
  alpha.copy_with_stride(
    fc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k * 2, false));

  Tensor dscores = Tensor(TensorDim({value.batch(), 1, 1, value.height()}));
  dscores.dot_batched_deriv_wrt_1(value, derivative);
  dscores.reshape(TensorDim({scores.batch(), 1, scores.width(), 1}));
  if (context.getNumInputs() == 4) {
    Tensor mask = Tensor(dscores.getDim());
    mask.filter_mask(context.getInput(wt_idx[MoLAttentionParams::mask_len]));
    dscores.multiply_i(mask);
  }

  Tensor dprob_scaled = Tensor(TensorDim({batch, 1, value.height(), mol_k}));
  dprob_scaled.setZero();
  dprob_scaled.add_i(dscores);

  Tensor dalpha = dprob_scaled.multiply(prob).sum(2);
  Tensor dprob = dprob_scaled.multiply(alpha);

  Tensor dprob_left = dprob;
  Tensor dprob_right = dprob.multiply(-1);

  Tensor beta_eps = beta.add(1e-8f);
  Tensor du_neg_div, du_pos_div;
  sigmoid.run_prime_fn(prob_right, du_neg_div, dprob_right);
  Tensor du_neg_m = du_neg_div.divide(beta_eps);
  Tensor dm_neg = du_neg_m.multiply(-1).sum(2);
  Tensor dbeta_eps_neg = du_neg_m.multiply(u_neg_div).multiply(-1).sum(2);

  sigmoid.run_prime_fn(prob_left, du_pos_div, dprob_left);
  Tensor du_pos_m = du_pos_div.divide(beta_eps);
  Tensor dm_pos = du_pos_m.multiply(-1).sum(2);
  Tensor dbeta_eps_pos = du_pos_m.multiply(u_pos_div).multiply(-1).sum(2);

  Tensor dbeta_eps = dbeta_eps_neg.add(dbeta_eps_pos);
  dm_neg.add(dm_pos, dstate);
  if (context.getNumOutputs() == 2) {
    const Tensor &derivative_state = context.getIncomingDerivative(1);
    dstate.add_i(derivative_state);
  }
  Tensor dkappa = dstate;
  Tensor dbeta = dbeta_eps;

  Tensor dalpha_src;
  softmax.run_prime_fn(alpha, dalpha_src, dalpha);

  Tensor dkappa_src = dkappa.multiply(kappa);
  Tensor dbeta_src = dbeta.multiply(beta);

  /** dfc_proj_out shares memory with fc_proj_out */
  dfc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, 0, false)
    .copy_with_stride(dkappa_src);
  dfc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k, false)
    .copy_with_stride(dbeta_src);
  dfc_proj_out.getSharedDataTensor({batch, 1, 1, mol_k}, mol_k * 2, false)
    .copy_with_stride(dalpha_src);

  /** update the helper state */
  helper_exec = true;
}

void MoLAttentionLayer::calcDerivative(RunLayerContext &context) {
  Tensor &dquery =
    context.getOutgoingDerivative(wt_idx[MoLAttentionParams::query]);
  Tensor &dvalue =
    context.getOutgoingDerivative(wt_idx[MoLAttentionParams::value]);
  Tensor &dstate =
    context.getOutgoingDerivative(wt_idx[MoLAttentionParams::state]);
  Tensor &dstate_local = context.getTensor(wt_idx[MoLAttentionParams::dstate]);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &fc_w = context.getWeight(wt_idx[MoLAttentionParams::fc_w]);
  Tensor &fc_proj_w = context.getWeight(wt_idx[MoLAttentionParams::fc_proj_w]);
  Tensor &fc_tanh = context.getTensor(wt_idx[MoLAttentionParams::fc_tanh]);
  Tensor &dfc_proj_out =
    context.getTensor(wt_idx[MoLAttentionParams::fc_proj_out]);
  Tensor &scores = context.getTensor(wt_idx[MoLAttentionParams::scores]);

  scores.dot_batched_deriv_wrt_2(dvalue, derivative);

  if (!helper_exec)
    calcDerivativeHelper(context, dstate);
  else
    dstate.copyData(dstate_local);

  Tensor dfc_tanh = Tensor(fc_tanh.getDim());
  dfc_tanh.dot_deriv_wrt_1(fc_proj_w, dfc_proj_out, false, false);

  Tensor dfc_out;
  tanh.run_prime_fn(fc_tanh, dfc_out, dfc_tanh);
  dquery.dot_deriv_wrt_1(fc_w, dfc_out, false, false);
}

void MoLAttentionLayer::calcGradient(RunLayerContext &context) {
  Tensor &query = context.getInput(wt_idx[MoLAttentionParams::query]);
  Tensor &dstate = context.getTensor(wt_idx[MoLAttentionParams::dstate]);

  Tensor &fc_proj_w = context.getWeight(wt_idx[MoLAttentionParams::fc_proj_w]);
  Tensor &dfc_w = context.getWeightGrad(wt_idx[MoLAttentionParams::fc_w]);
  Tensor &dfc_bias = context.getWeightGrad(wt_idx[MoLAttentionParams::fc_bias]);
  Tensor &dfc_proj_w =
    context.getWeightGrad(wt_idx[MoLAttentionParams::fc_proj_w]);
  Tensor &fc_tanh = context.getTensor(wt_idx[MoLAttentionParams::fc_tanh]);
  Tensor &dfc_proj_out =
    context.getTensor(wt_idx[MoLAttentionParams::fc_proj_out]);

  if (!helper_exec)
    calcDerivativeHelper(context, dstate);

  Tensor dfc_tanh = Tensor(fc_tanh.getDim());
  fc_tanh.dot_deriv_wrt_2(
    dfc_proj_w, dfc_proj_out, false, false,
    !context.isGradientFirstAccess(wt_idx[MoLAttentionParams::fc_proj_w]));
  dfc_tanh.dot_deriv_wrt_1(fc_proj_w, dfc_proj_out);

  Tensor dfc_out;
  tanh.run_prime_fn(fc_tanh, dfc_out, dfc_tanh);
  query.dot_deriv_wrt_2(
    dfc_w, dfc_out, false, false,
    !context.isGradientFirstAccess(wt_idx[MoLAttentionParams::fc_w]));

  if (context.isGradientFirstAccess(wt_idx[MoLAttentionParams::fc_bias])) {
    dfc_out.sum({0, 1, 2}, dfc_bias);
  } else {
    /// @todo optimize below by adding beta to Tensor::sum
    Tensor t = dfc_out.sum({0, 1, 2});
    dfc_bias.add_i(t);
  }
}

void MoLAttentionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mol_props);
  LayerImpl::setProperty(remain_props);
}

void MoLAttentionLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[MoLAttentionParams::fc_out], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::fc_tanh], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::fc_proj_out], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::scores], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::prob], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::prob_left], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::prob_right], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::u_neg_div], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::u_pos_div], batch);
  context.updateTensor(wt_idx[MoLAttentionParams::dstate], batch);
}

void MoLAttentionLayer::exportTo(Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mol_props, method, this);
}

} /* namespace nntrainer */
