/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Anirudh <b.saianirud@samsung.com>
 * @author	Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <unordered_map>

#include <common_properties.h>
#include <fc_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

// Static registries: layer_name → QAT stats / per-block EMA scales.
std::mutex FullyConnectedLayer::s_registry_mutex;
std::unordered_map<std::string, FullyConnectedLayer::LoRAQATStats>
  FullyConnectedLayer::s_qat_registry;
std::unordered_map<std::string,
  std::pair<std::vector<float>, std::vector<float>>>
  FullyConnectedLayer::s_block_d_registry;

FullyConnectedLayer::FullyConnectedLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  fc_props(props::Unit(), props::LoraRank(), props::LoraAlpha(), props::LoraQAT(),
           props::LoraWeightQ4()),
  quantizer(nullptr),
  momentum(0.1f) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

// Per-block Q4_0 fake-quantization with EMA block scales tracked in N×K layout.
//
// Blocks are defined in N×K layout (the transposed layout that build_q4_0_natural
// and the GEMM kernel use). This ensures the EMA scales exactly match the block
// boundaries used at save time, so force-feeding works correctly.
//
// Training:  compute fresh d_fresh per N×K block, bootstrap EMA on first call,
//            then update: block_d[b] = (1-m)*block_d[b] + m*d_fresh.
//            Quantize using updated EMA scale.
// Validation: use current EMA without updating.
// STE: gradient passes through the clamp+round unchanged.
Tensor FullyConnectedLayer::fakeQuantizeQ4_0(const Tensor &x,
                                              std::vector<float> &block_d,
                                              bool training) {
  // x is stored K×N in nntrainer (height=K, width=N)
  const size_t K = x.getDim().height();
  const size_t N = x.getDim().width();
  const size_t num_blocks_NK = (K * N) / 32;

  if (block_d.empty())
    block_d.resize(num_blocks_NK, 0.0f);

  // Compute fresh per-block scales in N×K layout.
  // For N×K linear index nk = b*32+j: n = nk/K, k = nk%K → K×N index = k*N+n.
  for (size_t b = 0; b < num_blocks_NK; ++b) {
    float max_abs = 0.0f;
    for (size_t j = 0; j < 32; ++j) {
      const size_t nk  = b * 32 + j;
      const size_t n   = nk / K;
      const size_t k   = nk % K;
      max_abs = std::max(max_abs, std::abs(x.getValue<float>(k * N + n)));
    }
    const float d_fresh = (max_abs > 1e-8f) ? max_abs / 8.0f : 0.0f;
    if (training) {
      if (block_d[b] < 1e-10f)
        block_d[b] = d_fresh;
      else
        block_d[b] = (1.0f - momentum) * block_d[b] + momentum * d_fresh;
    }
  }

  // Apply fake-quant iterating K×N order; map each element to its N×K block.
  // apply() iterates K×N linearly: index i = k*N + n → k = i/N, n = i%N
  // → N×K index nk = n*K + k → block = nk/32.
  Tensor x_fq = x.clone();
  size_t i = 0;
  std::function<float(float)> fn = [&i, K, N, &block_d](float v) -> float {
    const size_t k  = i / N;
    const size_t n  = i % N;
    const size_t b  = (n * K + k) / 32;
    ++i;
    const float d = block_d[b];
    if (d < 1e-10f) return v;
    float q = std::round(v / d);
    q = std::max(-8.0f, std::min(7.0f, q));
    return q * d;
  };
  x_fq.apply<float>(fn, x_fq);
  return x_fq;
}

FullyConnectedLayer::LoRAQATStats
FullyConnectedLayer::getRegisteredStats(const std::string &layer_name) {
  std::lock_guard<std::mutex> lock(s_registry_mutex);
  auto it = s_qat_registry.find(layer_name);
  if (it != s_qat_registry.end())
    return it->second;
  return {};
}

std::pair<std::vector<float>, std::vector<float>>
FullyConnectedLayer::getRegisteredBlockScales(const std::string &layer_name) {
  std::lock_guard<std::mutex> lock(s_registry_mutex);
  auto it = s_block_d_registry.find(layer_name);
  if (it != s_block_d_registry.end())
    return it->second;
  return {};
}

void FullyConnectedLayer::finalize(InitLayerContext &context) {
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

  const auto &unit = std::get<props::Unit>(fc_props).get();
  const auto &lora_rank = (std::get<props::LoraRank>(fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(fc_props).get();
  lora_scaling = (lora_rank && !std::get<props::LoraAlpha>(fc_props).empty())
                   ? (float)std::get<props::LoraAlpha>(fc_props) / lora_rank
                   : 1;
  if (!std::get<props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*layer_impl_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  /// @note bias is directly added to activation
  /// since we have no dequantizer for add operation,
  /// we have to set its data type as same as activation.
  /// This should be updated when the dequantizer is supported.
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getActivationDataType()),
    is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  // Base weight is trainable only when LoRA is not active for this layer.
  // When lora_rank > 0, only loraA/loraB update; W is frozen.
  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", (lora_rank == 0));

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", (lora_rank == 0));
  }

  /** create weights for LoRA */
  if (lora_rank) {

    const bool lora_qat_mode = !std::get<props::LoraQAT>(fc_props).empty() &&
                               std::get<props::LoraQAT>(fc_props).get();
    const bool lora_q4       = !std::get<props::LoraWeightQ4>(fc_props).empty() &&
                               std::get<props::LoraWeightQ4>(fc_props).get();
    // Inference with Q4_0: use Q4_0 tensor dtype → W4A8 kernel fires at runtime.
    // Training (QAT): keep FP32 for gradients; fake-quant range adjusted below.
    const auto lora_dtype = (lora_q4 && !lora_qat_mode)
                              ? TensorDim::DataType::Q4_0
                              : TensorDim::DataType::FP32;

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      TensorDim::TensorType(context.getFormat(), lora_dtype),
      is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      TensorDim::TensorType(context.getFormat(), lora_dtype),
      is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    // Q4_0 inference: NONE init (will be overwritten from file), not trainable.
    // QAT + FP32 training: A=random, B=zeros — standard LoRA init (Hu et al. 2022).
    // With A=zeros,B=random (old QAT init) loraB gets near-zero gradient since
    // grad_loraB = output_grad * loraA^T ≈ 0, so loraB never learns.
    const bool use_q4_tensors = lora_q4 && !lora_qat_mode;
    const Initializer loraA_init = use_q4_tensors
      ? Initializer::NONE : Initializer::LECUN_NORMAL;
    const Initializer loraB_init = use_q4_tensors
      ? Initializer::NONE : Initializer::ZEROS;

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, loraA_init,
      weight_regularizer, weight_regularizer_constant, weight_decay,
      "loraA", !use_q4_tensors);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, loraB_init,
      weight_regularizer, weight_regularizer_constant, weight_decay,
      "loraB", !use_q4_tensors);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);

    if (lora_qat_mode) {
      layer_name_ = context.getName();
      static int qat_layer_count = 0;
      if (++qat_layer_count == 1)
        std::cerr << "[QAT] LoRA QAT active: per-block Q4_0 fake-quant "
                     "(16 levels, block=32, symmetric).\n";
    }
  }

  ///@todo this quantizaer should be moved to tensor, not layer!
  switch (context.getWeightDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
  case ml::train::TensorDim::DataType::QINT8:
  case ml::train::TensorDim::DataType::QINT16:
    quantizer =
      Quantization::createQuantizer(nntrainer::QScheme::PER_TENSOR_AFFINE);
    break;
  default:
    quantizer = nullptr;
    break;
  }
}

void FullyConnectedLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayer::setBatch(nntrainer::RunLayerContext &context,
                                   unsigned int batch) {
  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  if (quantizer != nullptr) {
    Tensor weight_ = quantizer->dequantize(weight, input_.getDataType());
    input_.dot(weight_, hidden_, false, false);
  } else {
    input_.dot(weight, hidden_, false, false);
  }

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    Tensor &hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);

    const bool lora_qat = !std::get<props::LoraQAT>(fc_props).empty() &&
                           std::get<props::LoraQAT>(fc_props).get();
    if (lora_qat) {
      // Per-block EMA fake-quant: training updates EMA, validation reads it.
      a_fq = fakeQuantizeQ4_0(loraA, lora_a_block_d, training);
      b_fq = fakeQuantizeQ4_0(loraB, lora_b_block_d, training);
      // Push current EMA stats to both registries.
      if (!lora_a_block_d.empty() && !lora_b_block_d.empty()) {
        LoRAQATStats s;
        s.a_min = loraA.minValue();  s.a_max = loraA.maxValue();
        s.a_scale = std::accumulate(lora_a_block_d.begin(), lora_a_block_d.end(), 0.0f)
                    / static_cast<float>(lora_a_block_d.size());
        s.b_min = loraB.minValue();  s.b_max = loraB.maxValue();
        s.b_scale = std::accumulate(lora_b_block_d.begin(), lora_b_block_d.end(), 0.0f)
                    / static_cast<float>(lora_b_block_d.size());
        s.valid = true;
        std::lock_guard<std::mutex> lk(s_registry_mutex);
        s_qat_registry[layer_name_]    = s;
        s_block_d_registry[layer_name_] = {lora_a_block_d, lora_b_block_d};
      }
      input_.dot(a_fq, hidden_tmp_lora, false, false);
      hidden_tmp_lora.dot(b_fq, hidden_out_lora, false, false);
    } else {
      input_.dot(loraA, hidden_tmp_lora, false, false);
      hidden_tmp_lora.dot(loraB, hidden_out_lora, false, false);
    }
    hidden_out_lora.multiply_i(lora_scaling);
    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void FullyConnectedLayer::incremental_forwarding(RunLayerContext &context,
                                                 unsigned int from,
                                                 unsigned int to,
                                                 bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor loraA, loraB;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    // loraTmp/loraOut are NOT fetched from context here: they use
    // FORWARD_GRAD_LIFESPAN which may not be allocated in inference mode.
    // Instead, local tensors are allocated per batch step below.
  }

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  if (input_dim.height() > 1)
    input_step_dim.height(to - from);
  hidden_step_dim.batch(1);
  if (hidden_dim.height() > 1)
    hidden_step_dim.height(to - from);

  // @todo make it parallelized with batch axis
  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * hidden_dim.getFeatureLen(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (!std::get<props::LoraRank>(fc_props).empty()) {
      // Allocate local intermediates — avoids context.getTensor which may fail
      // in inference mode (FORWARD_GRAD_LIFESPAN not allocated without backward).
      TensorDim tmp_step_dim = input_step_dim;
      tmp_step_dim.width(loraA.getDim().width()); // lora_rank
      nntrainer::Tensor hidden_tmp_lora_step(tmp_step_dim);
      nntrainer::Tensor hidden_out_lora_step(hidden_step_dim);

      input_step.dot(loraA, hidden_tmp_lora_step, false, false);
      hidden_tmp_lora_step.dot(loraB, hidden_out_lora_step, false, false);
      hidden_out_lora_step.multiply_i(lora_scaling);
      hidden_step.add_i(hidden_out_lora_step);
    }

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // MODE 2 (LoRA QAT): effective weight = W_frozen + a_fq · b_fq · scaling
    // dL/dx = dL/dy * [W + a_fq · b_fq · scaling]^T
    // Using a_fq/b_fq (from forward) matches Pranjal's qat_fc_layer reference.
    // Base is frozen in LoRA training so this gradient feeds no weight update,
    // but using a_fq/b_fq is theoretically correct for the forward computation.
    Tensor w_fp32;
    using DT = TensorDim::DataType;
    if (quantizer != nullptr) {
      Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
      w_fp32 = quantizer->dequantize(weight, lora_A.getDataType());
    } else if (weight.getDataType() == DT::Q4_0) {
      auto dq = Quantization::createQuantizer(nntrainer::QScheme::Q4_0);
      w_fp32 = dq->dequantize(weight, DT::FP32);
    } else if (weight.getDataType() == DT::Q6_K) {
      auto dq = Quantization::createQuantizer(nntrainer::QScheme::Q6_K);
      w_fp32 = dq->dequantize(weight, DT::FP32);
    } else {
      w_fp32 = weight;
    }

    const bool lora_qat_deriv = !std::get<props::LoraQAT>(fc_props).empty() &&
                                std::get<props::LoraQAT>(fc_props).get();
    Tensor lora_contrib;
    if (lora_qat_deriv) {
      // chain-rule STE: dL/dx uses the same a_fq/b_fq that the forward used
      lora_contrib = a_fq.dot(b_fq).multiply(lora_scaling);
    } else {
      Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
      Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);
      lora_contrib = lora_A.dot(lora_B).multiply(lora_scaling);
    }
    ret_.dot_deriv_wrt_1(w_fp32.add(lora_contrib), derivative_, false, false);
  } else {
    ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
  }
}

void FullyConnectedLayer::calcGradient(RunLayerContext &context) {

  /** (default) calcGradient - compute gradient of weight and bias */
  if (std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
      djdb.setZero();

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
  } else {
    // LoRA calcGradient with chain-rule STE.
    // QAT path: backward uses b_fq (from forward) for dL/dA, matching
    // the actual computation in forwarding. B=LECUN_NORMAL init ensures
    // b_fq is non-zero from batch 1, so A gets gradient immediately.
    // Non-QAT path: uses raw loraB as before.

    Tensor &djdla = context.getWeightGrad(lora_idx[LORAParams::loraA]);
    Tensor &djdlb = context.getWeightGrad(lora_idx[LORAParams::loraB]);
    Tensor &djdtmp = context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    Tensor &loraTmp = context.getTensor(lora_idx[LORAParams::loraTmp]);
    const auto &lora_derivative_ = derivative_.multiply(lora_scaling);

    loraTmp.dot_deriv_wrt_2(
      djdlb, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]));

    const bool lora_qat_grad = !std::get<props::LoraQAT>(fc_props).empty() &&
                               std::get<props::LoraQAT>(fc_props).get();
    if (lora_qat_grad) {
      // chain-rule STE: dL/d(loraTmp) = dL/dy_lora · b_fq^T
      djdtmp.dot_deriv_wrt_1(
        b_fq, lora_derivative_, false, false,
        !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));
    } else {
      Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
      djdtmp.dot_deriv_wrt_1(
        loraB, lora_derivative_, false, false,
        !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]));
    }

    input_.dot_deriv_wrt_2(
      djdla, djdtmp, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
  }
}

} /* namespace nntrainer */
