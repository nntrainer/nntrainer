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
 * @file	qkv_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <htp_qkv_layer.h>

#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "htp_decode_hook.h"

#include <cpu_backend.h>
#include <engine.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum QKVParams { Q, K, V };
/** weight_idx slots; the file's order is q, q_gamma, k, k_gamma, v */
enum QKVWeights { WQ, WQ_GAMMA, WK, WK_GAMMA, WV, IN_GAMMA };
/** tensor_idx slots: the three raw projections, then the normed input */
enum QKVTensors { T_Q, T_K, T_V, T_IN };

HtpQKVLayer::HtpQKVLayer() :
  LayerImpl(),
  qkv_props(props::QUnit(), props::KUnit(), props::VUnit(),
            props::FeatureSize(), nntrainer::props::Epsilon(), props::VNorm(),
            props::VFromK(), props::QScale(), props::InNorm(), props::Rope(),
            props::RopeTheta(), props::RopeScalingType(),
            props::RopePartialRotaryFactor(), nntrainer::props::MaxTimestep(),
            props::NormInCall(), props::OutToHost()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

/** @brief The RoPE table in the attention core's own numbers (its
 *  precompute_freqs: the thetas of the default and proportional types, then
 *  calc_trigonometric_vals_dup per position), as [pos][cos | sin] rows of
 *  2 * hd floats, once per shape for every layer that asks. */
static std::shared_ptr<const std::vector<float>>
ropeTable(unsigned int hd, unsigned int rows, float theta,
          const std::string &type, float partial) {
  static std::mutex mtx;
  static std::map<std::string, std::shared_ptr<const std::vector<float>>> cache;
  const std::string key = std::to_string(hd) + ":" + std::to_string(rows) +
                          ":" + std::to_string(theta) + ":" + type + ":" +
                          std::to_string(partial);
  const std::lock_guard<std::mutex> lock(mtx);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  NNTR_THROW_IF(type != "default" && type != "proportional",
                std::invalid_argument)
    << "qkv_layer: rope_scaling_type " << type
    << " is not folded here; keep use_rope on the attention core";
  const unsigned int half = hd / 2;
  std::vector<float> thetas(half, 0.0f);
  if (type == "default") {
    for (unsigned int i = 0; i < half; ++i)
      thetas[i] = 1.0 / (std::pow(theta, (2 * i) / static_cast<float>(hd)));
  } else {
    const int angles = static_cast<int>((partial * hd) / 2.0f);
    for (int i = 0; i < angles; ++i)
      thetas[i] = 1.0f / (std::pow(theta, (2 * i) / static_cast<float>(hd)));
  }
  auto table =
    std::make_shared<std::vector<float>>(static_cast<size_t>(rows) * 2 * hd);
  for (unsigned int pos = 0; pos < rows; ++pos) {
    float *row = table->data() + static_cast<size_t>(pos) * 2 * hd;
    nntrainer::calc_trigonometric_vals_dup(half, thetas.data(), row, row + hd,
                                           pos, 1.0f);
  }
  cache[key] = table;
  return table;
}

/** @brief The attention core's RoPE kernel over rows [0, rows) of x at
 *  positions from + r, heads of hd across the row's width. */
static void ropeRows(float *x, unsigned int rows, unsigned int width,
                     unsigned int hd, const float *table, unsigned int from) {
  for (unsigned int r = 0; r < rows; ++r) {
    const float *cs = table + static_cast<size_t>(from + r) * 2 * hd;
    nntrainer::compute_rotary_emb_value(width, hd, hd / 2,
                                        x + static_cast<size_t>(r) * width,
                                        nullptr, cs, cs + hd, false);
  }
}

void HtpQKVLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &q_unit = std::get<props::QUnit>(qkv_props).get();
  const auto &k_unit = std::get<props::KUnit>(qkv_props).get();
  v_norm = std::get<props::VNorm>(qkv_props).get();
  v_from_k = std::get<props::VFromK>(qkv_props).get();
  q_scale = std::get<props::QScale>(qkv_props).get();
  in_norm = std::get<props::InNorm>(qkv_props).get();
  norm_in_call = std::get<props::NormInCall>(qkv_props).get();
  out_to_host = std::get<props::OutToHost>(qkv_props).get();
  rope = std::get<props::Rope>(qkv_props).get();
  // v from the raw k projection: v's width is k's, and the raw k lands in
  // the norm scratch below, so the norm (feature_size) must be on.
  const unsigned int v_unit =
    v_from_k ? k_unit : std::get<props::VUnit>(qkv_props).get();

  std::vector<nntrainer::TensorDim> output_dims(3);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];

  /** Q out */
  output_dims[QKVParams::Q] = in_dim;
  is_nchw ? output_dims[QKVParams::Q].width(q_unit)
          : output_dims[QKVParams::Q].channel(q_unit);
  output_dims[QKVParams::Q].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** K out */
  output_dims[QKVParams::K] = in_dim;
  is_nchw ? output_dims[QKVParams::K].width(k_unit)
          : output_dims[QKVParams::K].channel(k_unit);
  output_dims[QKVParams::K].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  /** V out */
  output_dims[QKVParams::V] = in_dim;
  is_nchw ? output_dims[QKVParams::V].width(v_unit)
          : output_dims[QKVParams::V].channel(v_unit);
  output_dims[QKVParams::V].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  feature_size = std::get<props::FeatureSize>(qkv_props).empty()
                   ? 0
                   : std::get<props::FeatureSize>(qkv_props).get();
  NNTR_THROW_IF(feature_size != 0 &&
                  (q_unit % feature_size != 0 || k_unit % feature_size != 0 ||
                   v_unit % feature_size != 0),
                std::invalid_argument)
    << "qkv_layer: feature_size must divide q_unit, k_unit and v_unit";
  if (rope) {
    NNTR_THROW_IF(feature_size == 0 || feature_size % 2 != 0,
                  std::invalid_argument)
      << "qkv_layer: rope takes feature_size as the head dim, an even one";
    NNTR_THROW_IF(std::get<nntrainer::props::MaxTimestep>(qkv_props).empty(),
                  std::invalid_argument)
      << "qkv_layer: rope needs max_timestep for its table";
    rope_rows = std::get<nntrainer::props::MaxTimestep>(qkv_props).get();
    rope_table =
      ropeTable(feature_size, rope_rows,
                static_cast<float>(std::get<props::RopeTheta>(qkv_props).get()),
                std::get<props::RopeScalingType>(qkv_props).get(),
                std::get<props::RopePartialRotaryFactor>(qkv_props).get());
  }
  NNTR_THROW_IF((v_from_k || v_norm) && feature_size == 0,
                std::invalid_argument)
    << "qkv_layer: v_from_k and v_norm need feature_size";

  // gamma is unquantized FP32 on disk, requested FP32 regardless of the
  // activation dtype -- ReshapedRMSNormLayer's rule, and its weight.
  const nntrainer::TensorDim gamma_dim(
    1, 1, 1, feature_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  auto request_gamma = [&](const char *name) {
    return context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, name, true);
  };

  // The input norm's gamma leads the weights, where the file has it.
  if (in_norm) {
    const nntrainer::TensorDim in_gamma_dim(
      1, 1, 1, in_dim.width(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       nntrainer::TensorDim::DataType::FP32));
    weight_idx[IN_GAMMA] = context.requestWeight(
      in_gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "in_norm_gamma", true);
  }

  /** Q */
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : q_unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? q_unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);
  weight_idx[WQ] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "qweight", true);
  if (feature_size)
    weight_idx[WQ_GAMMA] = request_gamma("q_norm_gamma");

  /** K */
  weight_dim.width(k_unit);
  weight_idx[WK] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "kweight", true);
  if (feature_size)
    weight_idx[WK_GAMMA] = request_gamma("k_norm_gamma");

  /** V */
  if (!v_from_k) {
    weight_dim.width(v_unit);
    weight_idx[WV] = context.requestWeight(
      weight_dim, weight_initializer, weight_regularizer,
      weight_regularizer_constant, weight_decay, "vweight", true);
  }

  lazy_scratch =
    context.getComputeEngineType() == ml::train::HTP &&
    context.getExecutionMode() == ml::train::ExecutionMode::INFERENCE;
  if (lazy_scratch)
    return;

  // The norm kernel is not in-place (its pointers are __restrict), so the
  // projections land here first and the outputs hold the normed rows.
  if (feature_size) {
    tensor_idx[QKVParams::Q] = context.requestTensor(
      output_dims[QKVParams::Q], "q_raw", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
    tensor_idx[QKVParams::K] = context.requestTensor(
      output_dims[QKVParams::K], "k_raw", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
    if (v_norm && !v_from_k) {
      tensor_idx[QKVParams::V] = context.requestTensor(
        output_dims[QKVParams::V], "v_raw", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
  }
  if (in_norm) {
    nntrainer::TensorDim normed_dim = in_dim;
    normed_dim.setTensorType(
      {context.getFormat(), context.getActivationDataType()});
    tensor_idx[T_IN] = context.requestTensor(
      normed_dim, "in_normed", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
}

/**
 * @brief out = rms_norm(in) * gamma * scale per feature_size-wide head,
 *        over the first @a rows rows. ReshapedRMSNormLayer's FP32
 *        arithmetic, call for call; without gamma (nullptr) it is that
 *        layer's use_gamma=false, and a scale other than 1 rides on a
 *        feature_size-long copy of gamma, not on the rows.
 */
static void headNorm(nntrainer::Tensor &in, nntrainer::Tensor &out,
                     const nntrainer::Tensor *gamma, unsigned int rows,
                     unsigned int feature_size, float epsilon,
                     float scale = 1.0f) {
  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "qkv_layer: the folded norm is FP32 only";
  const unsigned int width = in.getDim().width();
  ml::train::TensorDim dim(1, 1, rows * (width / feature_size), feature_size);
  nntrainer::Tensor in_step = in.getSharedDataTensor(dim, 0, true);
  nntrainer::Tensor out_step = out.getSharedDataTensor(dim, 0, true);
  nntrainer::rms_norm_wrt_width_fp32_intrinsic(
    in_step.getData<float>(), out_step.getData<float>(), dim.height(),
    dim.width(), epsilon);
  if (gamma == nullptr) {
    if (scale != 1.0f)
      out_step.multiply_i(scale);
  } else if (scale != 1.0f) {
    out_step.multiply_i(gamma->multiply(scale));
  } else {
    out_step.multiply_i(*gamma);
  }
}

void HtpQKVLayer::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(qkv_props, method, this);
}

void HtpQKVLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, qkv_props);
  LayerImpl::setProperty(remain_props);
}

void HtpQKVLayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  incremental_forwarding(context, 0,
                         context.getInput(SINGLE_INOUT_IDX).height(), training);
}

void HtpQKVLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  nntrainer::Tensor &Qweight = context.getWeight(weight_idx[WQ]);
  nntrainer::Tensor &Kweight = context.getWeight(weight_idx[WK]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);
  const float epsilon = std::get<nntrainer::props::Epsilon>(qkv_props).get();
  const unsigned int rows = to - from;

  auto step_dim = [&](unsigned index) {
    nntrainer::TensorDim dim = context.getOutput(index).getDim();
    dim.batch(1);
    dim.height(rows);
    return dim;
  };
  const auto Qhidden_step_dim = step_dim(QKVParams::Q);
  const auto Khidden_step_dim = step_dim(QKVParams::K);
  const auto Vhidden_step_dim = step_dim(QKVParams::V);

  // Every norm of the block in the projection call (doc 57 section 5 step
  // 4): the input norm before the quantizer, the per-head norms (q's gamma
  // carrying q_scale, v's gamma all ones) on the output rows. v from the
  // raw k projection is the k weight sent twice, its second copy normed
  // without a gamma -- one more projection on the accelerator instead of
  // a copy and a norm on the CPU.
  const auto q4 = ml::train::TensorDim::DataType::Q4_0;
  const auto qs4cx = ml::train::TensorDim::DataType::QS4CX;
  const bool wh =
    Qweight.getDataType() == ml::train::TensorDim::DataType::QS4CX_WH ||
    Qweight.getDataType() == ml::train::TensorDim::DataType::QS4CX_WH_AFFINE;
  const bool scaled = wh || Qweight.getDataType() == qs4cx;
  const auto wtype = Qweight.getDataType();
  auto *ops = context.getComputeOps();
  // Q4_0 or QS4CX weights, all three alike; decode's one row too when the
  // backend keeps QS4CX in its own format (accelerates_qs4cx_at_m1). The
  // accelerator's RoPE takes a head dim whose half is a multiple of 32.
  if ((in_norm || (feature_size && norm_in_call)) && ops != nullptr &&
      ops->supports_gemm_q4_0_batch_norm_fp32() &&
      (!rope || (feature_size / 2) % 32 == 0) &&
      (rows > 1 || (scaled && ops->accelerates_qs4cx_at_m1())) &&
      input_step.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      (wtype == q4 || scaled) && Kweight.getDataType() == wtype &&
      (v_from_k || context.getWeight(weight_idx[WV]).getDataType() == wtype)) {
    std::vector<void *> wdata = {Qweight.getData<char>(),
                                 Kweight.getData<char>()};
    std::vector<float *> wscale;
    if (scaled)
      wscale = {Qweight.getScale<float>(), Kweight.getScale<float>()};
    std::vector<unsigned int> widths = {
      static_cast<unsigned int>(Qhidden_step_dim.width()),
      static_cast<unsigned int>(Khidden_step_dim.width())};
    std::vector<float *> dsts = {
      context.getOutput(QKVParams::Q).getData<float>(),
      context.getOutput(QKVParams::K).getData<float>()};
    std::vector<unsigned int> chunks;
    std::vector<float> gammas;
    if (feature_size) {
      const float *gq =
        context.getWeight(weight_idx[WQ_GAMMA]).getData<float>();
      const float *gk =
        context.getWeight(weight_idx[WK_GAMMA]).getData<float>();
      chunks = {feature_size, feature_size};
      for (unsigned int j = 0; j < feature_size; ++j)
        gammas.push_back(gq[j] * q_scale);
      gammas.insert(gammas.end(), gk, gk + feature_size);
    } else {
      chunks = {0u, 0u};
    }
    nntrainer::Tensor &Vweight =
      context.getWeight(weight_idx[v_from_k ? WK : WV]);
    wdata.push_back(Vweight.getData<char>());
    if (scaled)
      wscale.push_back(Vweight.getScale<float>());
    widths.push_back(static_cast<unsigned int>(Vhidden_step_dim.width()));
    dsts.push_back(context.getOutput(QKVParams::V).getData<float>());
    if (v_norm) {
      chunks.push_back(feature_size);
      gammas.insert(gammas.end(), feature_size, 1.0f);
    } else {
      chunks.push_back(0u);
    }
    NNTR_THROW_IF(rope && from + rows > rope_rows, std::runtime_error)
      << "qkv_layer: rows " << from << ".." << from + rows
      << " past the RoPE table's " << rope_rows;
    ops->gemm_q4_0_batch_norm_fp32(
      wdata, wscale, input_step.getData<float>(), dsts, rows, widths,
      input_step_dim.width(),
      in_norm ? context.getWeight(weight_idx[IN_GAMMA]).getData<float>()
              : nullptr,
      chunks, gammas.data(), epsilon,
      rope ? rope_table->data() + static_cast<size_t>(from) * 2 * feature_size
           : nullptr,
      rope ? feature_size : 0u, 2u, out_to_host || rows == 1);
    return;
  }

  // Only the unfused path needs raw projection/norm buffers. HTP layers
  // allocate these locally on fallback; CPU layers retain pooled scratch.
  auto scratch = [&](unsigned index, const nntrainer::TensorDim &dim) {
    return lazy_scratch
             ? nntrainer::Tensor(dim, true)
             : context.getTensor(index).getSharedDataTensor(dim, 0, true);
  };
  nntrainer::Tensor Qhidden_ =
    feature_size ? scratch(tensor_idx[QKVParams::Q], Qhidden_step_dim)
                 : context.getOutput(QKVParams::Q);
  nntrainer::Tensor Khidden_ =
    feature_size ? scratch(tensor_idx[QKVParams::K], Khidden_step_dim)
                 : context.getOutput(QKVParams::K);
  nntrainer::Tensor Vhidden_ =
    v_norm && !v_from_k ? scratch(tensor_idx[QKVParams::V], Vhidden_step_dim)
                        : context.getOutput(QKVParams::V);
  nntrainer::Tensor Qhidden_step =
    Qhidden_.getSharedDataTensor(Qhidden_step_dim, 0, true);
  nntrainer::Tensor Khidden_step =
    Khidden_.getSharedDataTensor(Khidden_step_dim, 0, true);
  nntrainer::Tensor Vhidden_step =
    Vhidden_.getSharedDataTensor(Vhidden_step_dim, 0, true);

  // The CPU: the input norm first, into its scratch (the kernel is not in
  // place), then the projections read the normed rows.
  if (in_norm) {
    NNTR_THROW_IF(input_step.getDataType() !=
                    ml::train::TensorDim::DataType::FP32,
                  std::invalid_argument)
      << "qkv_layer: in_norm is FP32 only";
    nntrainer::Tensor normed = scratch(tensor_idx[T_IN], input_step_dim);
    // [#260] one decode row: the input norm is the layer's RMSNORM op (the
    // token's first op at layer 0); once the row is resident the DSP runs
    // the whole token -- norms, projections, RoPE -- and nothing below is
    // read
    const bool hooked =
      rows == 1 && input_dim.batch() == 1 &&
      htpDecodeRmsNorm(from, input_step.getData<float>(),
                       normed.getData<float>(),
                       context.getWeight(weight_idx[IN_GAMMA]).getData<float>(),
                       input_step_dim.width(), epsilon);
    if (hooked && htpDecodeRowResident(from))
      return;
    if (!hooked) {
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        input_step.getData<float>(), normed.getData<float>(), rows,
        input_step_dim.width(), epsilon);
      normed.multiply_i(context.getWeight(weight_idx[IN_GAMMA]));
    }
    input_step = normed;
  }

  std::vector<nntrainer::Tensor *> Weights({&Qweight, &Kweight});
  std::vector<nntrainer::Tensor *> Outputs({&Qhidden_step, &Khidden_step});
  if (!v_from_k) {
    Weights.push_back(&context.getWeight(weight_idx[WV]));
    Outputs.push_back(&Vhidden_step);
  }

  // [#132 Part B] every kind resident: the HTP projects this row itself
  // and the rows below are never read (the QK_NORM hook still binds its
  // gammas)
  if (!(to - from == 1 && input_dim.batch() == 1 && htpDecodeRowResident(from)))
    input_step.dot(Weights, Outputs);

  if (feature_size) {
    // [#130] one decode row: with QK_NORM resident the HTP norms q | k and
    // keeps the row for the attention hook, so the outputs stay unwritten
    if (to - from == 1 && input_dim.batch() == 1 && !v_from_k && !v_norm &&
        q_scale == 1.0f && !rope &&
        Qhidden_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const unsigned int wq = Qhidden_step_dim.width(),
                         wk = Khidden_step_dim.width(),
                         wv = Vhidden_step_dim.width();
      static thread_local std::vector<float> row, gammas;
      row.resize(wq + wk + wv);
      std::memcpy(row.data(), Qhidden_step.getData<float>(),
                  wq * sizeof(float));
      std::memcpy(row.data() + wq, Khidden_step.getData<float>(),
                  wk * sizeof(float));
      std::memcpy(row.data() + wq + wk, Vhidden_step.getData<float>(),
                  wv * sizeof(float));
      gammas.resize(2 * feature_size);
      std::memcpy(gammas.data(),
                  context.getWeight(weight_idx[WQ_GAMMA]).getData<float>(),
                  feature_size * sizeof(float));
      std::memcpy(gammas.data() + feature_size,
                  context.getWeight(weight_idx[WK_GAMMA]).getData<float>(),
                  feature_size * sizeof(float));
      if (htpDecodeQkNorm(from, row.data(), wq + wk + wv, gammas.data(),
                          2 * feature_size, epsilon))
        return;
    }
    headNorm(Qhidden_, context.getOutput(QKVParams::Q),
             &context.getWeight(weight_idx[WQ_GAMMA]), to - from, feature_size,
             epsilon, q_scale);
    headNorm(Khidden_, context.getOutput(QKVParams::K),
             &context.getWeight(weight_idx[WK_GAMMA]), to - from, feature_size,
             epsilon);
    if (rope) {
      NNTR_THROW_IF(from + rows > rope_rows, std::runtime_error)
        << "qkv_layer: rows " << from << ".." << from + rows
        << " past the RoPE table's " << rope_rows;
      ropeRows(context.getOutput(QKVParams::Q).getData<float>(), rows,
               Qhidden_step_dim.width(), feature_size, rope_table->data(),
               from);
      ropeRows(context.getOutput(QKVParams::K).getData<float>(), rows,
               Khidden_step_dim.width(), feature_size, rope_table->data(),
               from);
    }
    if (v_norm) {
      // v_from_k: the raw k projection is what v_raw would have held
      headNorm(v_from_k ? Khidden_ : Vhidden_, context.getOutput(QKVParams::V),
               nullptr, to - from, feature_size, epsilon);
    } else if (v_from_k) {
      nntrainer::Tensor v_out =
        context.getOutput(QKVParams::V)
          .getSharedDataTensor(Khidden_step_dim, 0, true);
      v_out.copyData(Khidden_step);
    }
  }
}

void HtpQKVLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  return;
}

void HtpQKVLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void HtpQKVLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim Qoutput_dim = context.getOutput(QKVParams::Q).getDim();
  ml::train::TensorDim Koutput_dim = context.getOutput(QKVParams::K).getDim();
  ml::train::TensorDim Voutput_dim = context.getOutput(QKVParams::V).getDim();

  input_dim.height(input_dimensions[0].height());
  Qoutput_dim.height(input_dimensions[0].height());
  Koutput_dim.height(input_dimensions[0].height());
  Voutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(QKVParams::Q, Qoutput_dim);
  context.updateOutput(QKVParams::K, Koutput_dim);
  context.updateOutput(QKVParams::V, Voutput_dim);
  if (!lazy_scratch && feature_size) {
    context.updateTensor(tensor_idx[QKVParams::Q], Qoutput_dim);
    context.updateTensor(tensor_idx[QKVParams::K], Koutput_dim);
    if (v_norm && !v_from_k)
      context.updateTensor(tensor_idx[QKVParams::V], Voutput_dim);
  }
  if (!lazy_scratch && in_norm)
    context.updateTensor(tensor_idx[T_IN], input_dim);
}
} // namespace causallm
