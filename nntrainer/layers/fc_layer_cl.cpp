// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   fc_layer_cl.cpp
 * @date   7 May 2024
 * @brief  Backend-neutral quantized Fully Connected layer (op-table dispatch).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fc_layer_cl.h>

#include <cstdlib>
#include <limits>

#include <common_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };

FullyConnectedLayerCl::FullyConnectedLayerCl() :
  LayerImpl(), fc_props(props::Unit(), props::FusedActivation()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedLayerCl::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*layer_impl_props).get();
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(fc_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  // CausalLM lm_head (name "output_of_causallm"): an untied QINT4 vocab
  // projection (unit=vocab) that is skip_prefill, so it only ever computes the
  // last position (decode M=1). Planning its output at the full graph-build
  // height makes a vocab-wide dead activation plane (~1GB) that uniformly slows
  // every decode kernel via bandwidth pressure. Force height=1 (rows>0 are
  // never produced). NNTR_LMHEAD_OUT_FULL restores the old full-height
  // planning.
  if (skip_prefill && context.getName() == "output_of_causallm") {
    static const bool keep_full =
      std::getenv("NNTR_LMHEAD_OUT_FULL") != nullptr;
    if (!keep_full)
      output_dims[0].height(1);
  }

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  /// @note The bias follows the activation dtype, not the weight dtype -- see
  /// FullyConnectedLayer::finalize() for the on-disk contract. This layer backs
  /// LAYER_FC on both the OpenCL and the CUDA context, so it reads the same
  /// .bin the host FC does; asking for the weight dtype would size the bias as
  /// a quantized record (a QS4CX bias of unit elements is 5*unit bytes, not
  /// 2*unit) and shift every weight after it.
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }
}

void FullyConnectedLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayerCl::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayerCl::forwarding(RunLayerContext &context,
                                       bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  // output = input * weight. The backend's ComputeOps owns the GEMM (OpenCL v8c
  // / CUDA cuda_fc_qint4 / host dot).
  input_.getOps()->fc(input_, weight, hidden_);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    // Cast when a package pins its bias to a dtype other than the activation
    // one, matching FullyConnectedLayer's add site.
    if (bias.getDataType() != hidden_.getDataType()) {
      Tensor bias_cast = bias.clone(hidden_.getDataType());
      hidden_.add_i(bias_cast);
    } else {
      hidden_.add_i(bias);
    }
  }

  // fused activation epilogue via the op table — same backend-neutral
  // dispatch as the core FC (ClComputeOps on gpu, CudaComputeOps on cuda).
  // Inert for the LLM stack (no fc+activation); fires only when a
  // FusionRealizer sets fused_activation (a GPU CNN/MLP).
  auto &fused_act = std::get<props::FusedActivation>(fc_props);
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE)
    hidden_.getOps()->apply_activation(hidden_, (int)fused_act.get());
}

void FullyConnectedLayerCl::incremental_forwarding(RunLayerContext &context,
                                                   unsigned int from,
                                                   unsigned int to,
                                                   bool training) {
  // Multi-token steps count as prefill (a resumed multi-turn / KV-restored
  // prefill arrives as one from>0 block call).
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

  // by-reference so a quantized weight keeps its instance across forwards (a
  // Tensor::operator= on QINT4 deep-clones the Int4QTensor, wiping the
  // assembled weight cache every token and tanking decode throughput).
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    // Normalize to 0-based while preserving step size for multi-token
    // prefill (the producers write the live rows at the buffer base; the
    // absolute position only matters to the KV/attention layers).
    //
    to = to - from;
    from = 0;
  }

  // Only widen the step view when the tensor actually has rows to step over.
  // Mirrors the CPU twin (fc_layer.cpp FullyConnectedLayer::incremental_-
  // forwarding), which guards each step dim on its own tensor's height. An FC
  // whose input is already collapsed to a single row -- a sentence-transformer
  // Dense module sitting after the Pooling module -- would otherwise be asked
  // for a (to - from)-row view of a 1-row buffer, and getSharedDataTensor()
  // throws "Creating shared tensor of size bigger than tensor memory."
  if (input_dim.height() > 1)
    input_step_dim.height(to - from);
  if (hidden_dim.height() > 1)
    hidden_step_dim.height(to - from);

  // @todo: only correct for batch size 1
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);
  Tensor hidden_step = hidden_.getSharedDataTensor(hidden_step_dim, 0, true);

  input_step.getOps()->fc(input_step, weight, hidden_step);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    // See forwarding(): cast a bias whose dtype differs from the activation.
    if (bias.getDataType() != hidden_step.getDataType()) {
      Tensor bias_cast = bias.clone(hidden_step.getDataType());
      hidden_step.add_i(bias_cast);
    } else {
      hidden_step.add_i(bias);
    }
  }

  // fused activation epilogue via the op table (see forwarding()).
  auto &fused_act = std::get<props::FusedActivation>(fc_props);
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE)
    hidden_step.getOps()->apply_activation(hidden_step, (int)fused_act.get());
}

void FullyConnectedLayerCl::read(std::ifstream &file,
                                 RunLayerContext &run_context, bool opt_var,
                                 ml::train::ExecutionMode mode, bool trainable,
                                 TensorDim::DataType defineWeightDataType,
                                 bool fsu, size_t start_offset,
                                 bool read_from_offset, int file_fd) {
  Layer::read(file, run_context, opt_var, mode, trainable, defineWeightDataType,
              fsu, start_offset, read_from_offset, file_fd);
  // Eager backend weight build at load (see header). Not under FSU: the weight
  // data may be streamed back out, invalidating the host pointer the cache is
  // keyed on. No-op on backends without a prebuild (CPU/CUDA).
  if (!opt_var && !fsu) {
    Tensor &w = run_context.getWeight(weight_idx[FCParams::weight]);
    w.getOps()->fc_prebuild_weight(w);
  }
}

void FullyConnectedLayerCl::read(ReadSource src, RunLayerContext &run_context,
                                 bool opt_var, ml::train::ExecutionMode mode,
                                 bool trainable,
                                 TensorDim::DataType defineWeightDataType,
                                 bool fsu, size_t start_offset,
                                 bool read_from_offset, int file_fd) {
  Layer::read(src, run_context, opt_var, mode, trainable, defineWeightDataType,
              fsu, start_offset, read_from_offset, file_fd);
  if (!opt_var && !fsu) {
    Tensor &w = run_context.getWeight(weight_idx[FCParams::weight]);
    w.getOps()->fc_prebuild_weight(w);
  }
}

void FullyConnectedLayerCl::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void FullyConnectedLayerCl::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);

    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      /// @todo optimize below by adding beta to Tensor::sum
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} /* namespace nntrainer */
