// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.cpp
 * @date   16 Jan 2026
 * @brief  This is lmhead layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cpu_backend.h>
#include <layer_context.h>
#include <lm_head.h>
#include <nntr_htp_decode_hook.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LmHeadParams {
  weight,
  bias,
};

LmHeadLayer::LmHeadLayer() :
  LayerImpl(), lmhead_props(nntrainer::props::Unit()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void LmHeadLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(lmhead_props).get();
  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions.
  /// EffDimFlag shouldn't be fixed like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);
  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);

  /** set output dimensions */
  ///@note lm_head's output dimension (height is always 1 !)
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  if (is_nchw)
    output_dims[0].width(unit);
  else
    output_dims[0].channel(unit);
  output_dims[0].height(1);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ///@note LMHead layer's tensor dim is transposed dim of user-defined
  /// dim
  /// so it can reuse embedding layer.
  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[LmHeadParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[LmHeadParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }
}

void LmHeadLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, lmhead_props);
  LayerImpl::setProperty(remain_props);
}

void LmHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  throw nntrainer::exception::not_supported(
    "Forwarding for LMHead layer is not supported");
}

void LmHeadLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  nntrainer::Tensor &weight =
    context.getWeight(weight_idx[LmHeadParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + (to - from - 1) * input_.width(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);
    auto &disable_bias =
      std::get<nntrainer::props::DisableBias>(*layer_impl_props);
    const bool no_bias = !disable_bias.empty() && disable_bias.get();

    // [#276] An untied head's decode row on the HTP, as tie_word_embedding's
    // (the list's LM_HEAD op is this weight alone; the final norm is its own
    // RMSNORM op and the softcap stays in the CPU layer after this one).
    // With NNTR_HTP_E2E=1 it is the token's last hook: without it the
    // stretch never closes.
    if (b_size == 1 && to - from == 1 && no_bias &&
        input_step.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        nntrainer::htpDecodeLmHead(
          from, input_step.getData<float>(), input_step.width(),
          hidden_step.getData<float>(), hidden_step.width()))
      continue;

#ifdef ENABLE_HEXKL
    if (!training && no_bias &&
        input_step.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        nntrainer::get_htp_ops()->lm_head_shared_fp32(
          weight.getData(), input_step.width(), hidden_step.width(),
          input_step.getData<float>(), hidden_step.getData<float>()))
      continue;
#endif

    input_step.dot(weight, hidden_step, false, false);

    if (!no_bias) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[LmHeadParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void LmHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for LMHead layer is not supported");
}

void LmHeadLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcGradient for LMHead layer is not supported");
}

void LmHeadLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(lmhead_props, method, this);
}

void LmHeadLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  // output dim's height is always 1 !
  in_dim.height(height);
  context.updateInput(SINGLE_INOUT_IDX, in_dim);
}

} // namespace nntrainer
