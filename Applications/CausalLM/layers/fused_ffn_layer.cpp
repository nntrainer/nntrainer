// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Anirudh Rajapakshe <anirudh.rajapakshe@samsung.com>
 *
 * @file   fused_ffn_layer.cpp
 * @date   4 August 2026
 * @brief  Fused FFN Layer with SwiGLU activation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Anirudh Rajapakshe <anirudh.rajapakshe@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <fstream>
#include <layer_context.h>
#include <limits>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <fused_ffn_layer.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace causallm {
using namespace nntrainer;

static constexpr size_t INPUT_IDX = 0;

// Weight request order matches .bin file: up (0), gate (1), down (2)
enum FFNParams { W_UP = 0, W_GATE = 1, W_DOWN = 2 };

// ── DSP Bridge ──────────────────────────────────────────────────────────────

using ffn_fn = int (*)(const void *, const void *, const void *,
                       const float *, float *, unsigned int,
                       unsigned int, unsigned int);

using upload_fn = int (*)(const void *, const void *, unsigned int,
                          unsigned int);

static void *get_hexagon_handle() {
  static void *handle = []() -> void * {
    void *h = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
      ml_logw("FusedFFN: dlopen(libggml-hexagon.so) failed: %s "
              "(fused FFN disabled, using CPU path)",
              dlerror());
    }
    return h;
  }();
  return handle;
}

static ffn_fn get_ffn_bridge() {
  static ffn_fn fn = []() -> ffn_fn {
    void *handle = get_hexagon_handle();
    if (!handle)
      return nullptr;
    void *s = dlsym(handle, "nntr_htp_bridge_ffn_swiglu");
    if (!s) {
      ml_logw("FusedFFN: dlsym(nntr_htp_bridge_ffn_swiglu) failed: %s "
              "(fused FFN disabled, using CPU path)",
              dlerror());
      return nullptr;
    }
    ml_logi("FusedFFN: bridge loaded successfully");
    return reinterpret_cast<ffn_fn>(s);
  }();
  return fn;
}

static upload_fn get_upload_fn() {
  static upload_fn fn = []() -> upload_fn {
    void *handle = get_hexagon_handle();
    if (!handle)
      return nullptr;
    void *s = dlsym(handle, "nntr_htp_bridge_upload_weight_q4x4x2");
    if (!s) {
      ml_logw("FusedFFN: dlsym(nntr_htp_bridge_upload_weight_q4x4x2) failed: %s",
              dlerror());
      return nullptr;
    }
    return reinterpret_cast<upload_fn>(s);
  }();
  return fn;
}

/// Upload a Q4_0 weight to the DSP if not already uploaded (idempotent by key).
static void ensure_weight_uploaded(const void *key, const void *data,
                                   unsigned int N, unsigned int K) {
  static const upload_fn &upload = get_upload_fn();
  if (!upload)
    return;
  int rc = upload(key, data, N, K);
  if (rc != 0) {
    ml_logw("FusedFFN: upload_weight_q4x4x2 failed (rc=%d)", rc);
  }
}

static bool should_use_fused_ffn(unsigned int step_size, bool is_prefill) {
  static const char *env = std::getenv("NNTR_HEXAGON_FUSED_FFN");
  bool enabled = (env && std::atoi(env) == 1);
  if (!enabled)
    return false;

  static const char *cdsp_env = std::getenv("NNTR_USE_HEXAGON_CDSP");
  if (!cdsp_env)
    return false;

  static std::atomic<bool> logged_accept{false};
  if (!logged_accept.exchange(true)) {
    fprintf(stderr, "[FUSED_FFN] gate: ACCEPT (step_size=%u, prefill=%d)\n",
            step_size, (int)is_prefill);
  }

  return get_ffn_bridge() != nullptr;
}

// ── CPU FFN forward (fallback when DSP bridge is unavailable) ───────────────

static void cpu_ffn_forward(Tensor &input, Tensor &w_gate, Tensor &w_up,
                            Tensor &w_down, Tensor &output,
                            unsigned int from = 0, unsigned int to = 0) {
  unsigned int M = (to > from) ? (to - from) : input.height();
  unsigned int K = input.width();
  // Weights are [N, K] → height=N, width=K
  unsigned int N = w_gate.height();
  unsigned int K_out = w_down.height();

  // Dequantize Q4_0 → FP32 for CPU dot (weights are [N,K], need transpose)
  auto quantizer = Quantization::createQuantizer(QScheme::Q4_0);
  Tensor w_gate_fp32 =
    quantizer->dequantize(w_gate, ml::train::TensorDim::DataType::FP32);
  Tensor w_up_fp32 =
    quantizer->dequantize(w_up, ml::train::TensorDim::DataType::FP32);
  Tensor w_down_fp32 =
    quantizer->dequantize(w_down, ml::train::TensorDim::DataType::FP32);

  // gate = input[M,K] · w_gate[N,K]^T = [M,N]  (transpose=true)
  Tensor gate_out(TensorDim(1, 1, M, N,
                            TensorDim::TensorType(
                              input.getFormat(),
                              ml::train::TensorDim::DataType::FP32)));
  Tensor up_out(TensorDim(1, 1, M, N,
                          TensorDim::TensorType(
                            input.getFormat(),
                            ml::train::TensorDim::DataType::FP32)));

  if (to > from) {
    Tensor input_step = input.getSharedDataTensor({1, 1, M, K}, from * K);
    input_step.dot(w_gate_fp32, gate_out, false, true);
    input_step.dot(w_up_fp32, up_out, false, true);
  } else {
    input.dot(w_gate_fp32, gate_out, false, true);
    input.dot(w_up_fp32, up_out, false, true);
  }

  // SwiGLU: gate_out * sigmoid(up_out) → [M,N]
  Tensor act(TensorDim(1, 1, M, N,
                       TensorDim::TensorType(
                         input.getFormat(),
                         ml::train::TensorDim::DataType::FP32)));
  up_out.apply(std::function<float(float)>(
    [](float x) -> float { return 1.0f / (1.0f + std::exp(-x)); }), act);
  act.multiply_i(gate_out);

  // down = act[M,N] · w_down[K_out,N]^T = [M,K_out]  (transpose=true)
  if (to > from) {
    Tensor output_step =
      output.getSharedDataTensor({1, 1, M, K_out}, from * K_out);
    act.dot(w_down_fp32, output_step, false, true);
  } else {
    act.dot(w_down_fp32, output, false, true);
  }
}

// ── Layer Implementation ────────────────────────────────────────────────────

FusedFFNLayer::FusedFFNLayer() : nntrainer::LayerImpl() {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void FusedFFNLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 1, std::invalid_argument)
    << "FusedFFNLayer requires at least one input";

  const auto &hidden_dim = std::get<props::HiddenDim>(ffn_props).get();
  const auto &output_dim = std::get<props::OutputDim>(ffn_props).get();

  int N = hidden_dim;
  if (N <= 0) {
    N = std::get<nntrainer::props::Unit>(ffn_props).get();
  }
  int K_out = (output_dim > 0) ? output_dim : 0;

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  const auto &in_dim = context.getInputDimensions()[0];
  int K = (int)in_dim.width();
  if (K_out == 0)
    K_out = K;

  auto weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(ffn_props);

  // Weights stored as [N, K] (matching .bin file layout) for gate/up
  // and [K_out, N] for down. The DSP bridge expects this [N,K] layout.
  ml::train::TensorDim weight_dim_gate(
    {1, 1, (unsigned int)N, (unsigned int)K},
    {context.getFormat(), ml::train::TensorDim::DataType::Q4_0});

  ml::train::TensorDim weight_dim_down(
    {1, 1, (unsigned int)K_out, (unsigned int)N},
    {context.getFormat(), ml::train::TensorDim::DataType::Q4_0});

  // Weight request order MUST match .bin file: up, gate, down
  weight_idx[W_UP] = context.requestWeight(
    weight_dim_gate, weight_initializer, nntrainer::WeightRegularizer::NONE,
    0.0f, 0.0f, "ffn_up");

  weight_idx[W_GATE] = context.requestWeight(
    weight_dim_gate, weight_initializer, nntrainer::WeightRegularizer::NONE,
    0.0f, 0.0f, "ffn_gate");

  weight_idx[W_DOWN] = context.requestWeight(
    weight_dim_down, weight_initializer, nntrainer::WeightRegularizer::NONE,
    0.0f, 0.0f, "ffn_down");

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = in_dim;
  output_dims[0].width((unsigned int)K_out);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

void FusedFFNLayer::forwarding(nntrainer::RunLayerContext &context,
                               bool training) {
  Tensor &input = context.getInput(INPUT_IDX);
  Tensor &output = context.getOutput(0);
  Tensor &w_gate = context.getWeight(weight_idx[W_GATE]);
  Tensor &w_up = context.getWeight(weight_idx[W_UP]);
  Tensor &w_down = context.getWeight(weight_idx[W_DOWN]);

  unsigned int step_size = input.height();

  if (should_use_fused_ffn(step_size, true)) {
    // DSP fused path: single FastRPC call for all 3 GEMMs + SwiGLU
    ffn_fn bridge = get_ffn_bridge();
    if (bridge) {
      unsigned int M = step_size;
      unsigned int K = input.width();
      unsigned int N = w_gate.height();
      unsigned int K_out = w_down.height();

      ensure_weight_uploaded(w_gate.getData(), w_gate.getData(), N, K);
      ensure_weight_uploaded(w_up.getData(), w_up.getData(), N, K);
      ensure_weight_uploaded(w_down.getData(), w_down.getData(), K_out, N);

      int rc = bridge(w_gate.getData(), w_up.getData(), w_down.getData(),
                      (const float *)input.getData(), (float *)output.getData(),
                      M, K, N);
      if (rc == 0)
        return;
      ml_logw("FusedFFN: DSP bridge failed (rc=%d), falling back to CPU", rc);
    }
  }

  // CPU fallback
  cpu_ffn_forward(input, w_gate, w_up, w_down, output);
}

void FusedFFNLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  Tensor &input = context.getInput(INPUT_IDX);
  Tensor &output = context.getOutput(0);
  Tensor &w_gate = context.getWeight(weight_idx[W_GATE]);
  Tensor &w_up = context.getWeight(weight_idx[W_UP]);
  Tensor &w_down = context.getWeight(weight_idx[W_DOWN]);

  unsigned int step_size = to - from;

  if (should_use_fused_ffn(step_size, false)) {
    // DSP fused path for decode: single FastRPC call
    ffn_fn bridge = get_ffn_bridge();
    if (bridge) {
      unsigned int M = step_size;
      unsigned int K = input.width();
      unsigned int N = w_gate.height();
      unsigned int K_out = w_down.height();

      ensure_weight_uploaded(w_gate.getData(), w_gate.getData(), N, K);
      ensure_weight_uploaded(w_up.getData(), w_up.getData(), N, K);
      ensure_weight_uploaded(w_down.getData(), w_down.getData(), K_out, N);

      int rc = bridge(w_gate.getData(), w_up.getData(), w_down.getData(),
                      (const float *)input.getData() + from * K,
                      (float *)output.getData() + from * K_out,
                      M, K, N);
      if (rc == 0)
        return;
      ml_logw("FusedFFN: DSP bridge failed (rc=%d), falling back to CPU", rc);
    }
  }

  // CPU fallback
  cpu_ffn_forward(input, w_gate, w_up, w_down, output, from, to);
}

void FusedFFNLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void FusedFFNLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void FusedFFNLayer::exportTo(nntrainer::Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(ffn_props, method, this);
}

void FusedFFNLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, ffn_props);
  LayerImpl::setProperty(remain_props);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_fused_ffn_layer() {
  auto layer = new FusedFFNLayer();
  return layer;
}

void destroy_fused_ffn_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_fused_ffn_layer,
                                                   destroy_fused_ffn_layer};
}

#endif

} // namespace causallm
