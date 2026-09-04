// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   lm_head.cpp
 * @date   16 Jan 2026
 * @brief  This is lmhead layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Anirudh Bocha <b.saianirud@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cpu_backend.h>
#include <layer_context.h>
#include <limits>
#include <lm_head.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LmHeadParams {
  weight,
  bias,
};

std::vector<unsigned int> g_lm_head_read_row;

namespace {
inline unsigned int readRowForBatch(unsigned int b, unsigned int fallback) {
  return (b < g_lm_head_read_row.size()) ? g_lm_head_read_row[b] : fallback;
}
} // namespace

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

/**
 * @brief Full-sequence (training) forward: projects only a single row (the
 *        last real token under right-padded training, or the last row of
 *        the buffer by default) to vocab logits, matching the height=1
 *        output contract already established by incremental_forwarding.
 */
void LmHeadLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor weight =
    context.getWeight(weight_idx[LmHeadParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  const unsigned int fallback_row = input_dim.height() - 1;

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    const unsigned int read_row = readRowForBatch(b, fallback_row);
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() + read_row * input_dim.width(), true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[LmHeadParams::bias]);
      hidden_step.add_i(bias);
    }
  }
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

    input_step.dot(weight, hidden_step, false, false);

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[LmHeadParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

/**
 * @brief calcDerivative for LMHead.
 * @details forwarding() only ever reads one row (read_row) of the input
 *          per batch, so only that row receives a non-zero gradient; every
 *          other row's outgoing derivative is exactly zero by the chain
 *          rule (the output does not depend on them at all).
 */
void LmHeadLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor weight =
    context.getWeight(weight_idx[LmHeadParams::weight]);

  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = dx.getDim();
  ml::train::TensorDim dy_dim = dy.getDim();

  const unsigned int fallback_row = input_dim.height() - 1;

  dx.setZero();

  ml::train::TensorDim dx_step_dim = input_dim;
  ml::train::TensorDim dy_step_dim = dy_dim;

  dx_step_dim.batch(1);
  dx_step_dim.height(1);
  dy_step_dim.batch(1);

  unsigned int b_size = input_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    const unsigned int read_row = readRowForBatch(b, fallback_row);
    nntrainer::Tensor dx_step = dx.getSharedDataTensor(
      dx_step_dim,
      b * input_dim.getFeatureLen() + read_row * input_dim.width(), true);
    nntrainer::Tensor dy_step = dy.getSharedDataTensor(
      dy_step_dim, b * dy_dim.getFeatureLen(), true);

    // y = x . W  =>  dx = dy . W^T
    dy_step.dot(weight, dx_step, false, true);
  }
}

/**
 * @brief calcGradient for LMHead.
 * @details forwarding() computes y = x[read_row] . W (plus bias), reading a
 *          single input row, so
 *            dL/dW[i][j] += x[read_row][i] * dy[j]
 *            dL/dbias[j] += dy[j]
 *          Only that one row contributes; every other row of the input has
 *          no effect on the output and therefore none on dW.
 *
 * @note Only invoked when the layer is trainable. Under LoRA-only training
 *       the LM head is frozen (see causal_lm.cpp) so this never runs; it
 *       exists so full fine-tuning can train the head.
 * @note Safe to read the input: calcGradient runs before calcDerivative,
 *       which is what overwrites the input buffer with dx.
 */
void LmHeadLayer::calcGradient(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dw =
    context.getWeightGrad(weight_idx[LmHeadParams::weight]);

  NNTR_THROW_IF(input_.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  dw.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[lm_head] calcGradient only supports FP32 for now";

  const ml::train::TensorDim in_dim = input_.getDim();
  const ml::train::TensorDim dy_dim = dy.getDim();
  const unsigned int in_w = in_dim.width();
  const unsigned int unit = dy_dim.width();

  const unsigned int fallback_row = in_dim.height() - 1;

  const bool first =
    context.isGradientFirstAccess(weight_idx[LmHeadParams::weight]);
  if (first)
    dw.setZero();

  const float *x = input_.getData<float>();
  const float *dy_ = dy.getData<float>();
  float *dw_ = dw.getData<float>();

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    const unsigned int read_row = readRowForBatch(b, fallback_row);
    const float *x_row =
      x + b * in_dim.getFeatureLen() + read_row * in_w;
    const float *dy_row = dy_ + b * dy_dim.getFeatureLen();
    for (unsigned int i = 0; i < in_w; ++i) {
      const float xi = x_row[i];
      float *dw_row = dw_ + i * unit;
      for (unsigned int j = 0; j < unit; ++j)
        dw_row[j] += xi * dy_row[j];
    }
  }

  if (auto &disable_bias =
        std::get<nntrainer::props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    nntrainer::Tensor &db =
      context.getWeightGrad(weight_idx[LmHeadParams::bias]);
    if (context.isGradientFirstAccess(weight_idx[LmHeadParams::bias]))
      db.setZero();
    float *db_ = db.getData<float>();
    for (unsigned int b = 0; b < in_dim.batch(); ++b) {
      const float *dy_row = dy_ + b * dy_dim.getFeatureLen();
      for (unsigned int j = 0; j < unit; ++j)
        db_[j] += dy_row[j];
    }
  }
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

#ifdef PLUGGABLE

nntrainer::Layer *create_tie_word_embedding() {
  auto layer = new LmHeadLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_tie_word_embedding(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_tie_word_embedding,
                                                   destroy_tie_word_embedding};
}

#endif

} // namespace causallm
