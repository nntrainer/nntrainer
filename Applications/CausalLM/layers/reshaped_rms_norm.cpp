// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reshaped_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <reshaped_rms_norm.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// ---------------------------------------------------------------------------
// DSP bridge for q_norm/k_norm (the RMSNorm applied inside attention, on the
// reshaped-per-head view of Q/K - distinct from the residual-stream
// attention_norm/ffn_norm in rms_norm.cpp, which already dispatches). Same
// nntr_htp_bridge_rms_norm bridge function, same fused normalize+scale
// (HTP_OP_RMS_NORM_MUL) kernel - but that kernel is F32-only, and this
// layer's actual input here is FP16 (it operates directly on the Q/K
// projection GEMM's FP16 output, before mha_core ever sees it). So unlike
// rms_norm.cpp's direct F32 dispatch, this goes through the same
// cast-rotate-cast-style chain used for FP16 RoPE in mha_core.cpp: cast
// F16->F32 into a scratch buffer, normalize the scratch buffer on the DSP,
// cast the result back to F16 at the destination - all three ops are
// existing DSP kernels (htp/cpy-ops.c's F16<->F32 conversion, htp/
// unary-ops.c's F32 rms_norm_mul), enqueued into the same batch with no
// flush needed between or before them (same FIFO-chaining guarantee
// nntr_htp_bridge_ffn_swiglu already relies on for its 5 dependent ops).
// ---------------------------------------------------------------------------
using rms_norm_fn = int (*)(const float *, const float *, float *,
                            unsigned int, unsigned int, float);

static rms_norm_fn get_rms_norm_bridge() {
  static rms_norm_fn fn = []() -> rms_norm_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("ReshapedRMSNorm: dlopen(libggml-hexagon.so) failed: %s "
              "(DSP offload disabled, using CPU path)", dlerror());
      return nullptr;
    }
    void *s = dlsym(handle, "nntr_htp_bridge_rms_norm");
    if (!s) {
      ml_logw("ReshapedRMSNorm: dlsym(nntr_htp_bridge_rms_norm) failed: %s "
              "(DSP offload disabled, using CPU path)", dlerror());
      return nullptr;
    }
    return reinterpret_cast<rms_norm_fn>(s);
  }();
  return fn;
}

using cpy_fn = int (*)(const void *, void *, unsigned int, int, int);

static cpy_fn get_cpy_bridge() {
  static cpy_fn fn = []() -> cpy_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return nullptr;
    return reinterpret_cast<cpy_fn>(dlsym(handle, "nntr_htp_bridge_cpy"));
  }();
  return fn;
}

namespace {

// One shared F32 scratch buffer, allocated and registered with the bridge
// ONCE (not per layer/call - registering many small pools was found to hang
// the DSP after ~12-13 registrations, see RMSNormLayer's disabled
// gamma-rpcmem attempt in rms_norm.cpp), reused for every q_norm/k_norm call
// across every block. Two halves so the cast-in destination and the
// rms_norm-mul output don't alias the same memory the DSP is still reading.
class NormScratchRpcMem {
public:
  static NormScratchRpcMem &global() {
    static NormScratchRpcMem inst;
    return inst;
  }

  // Returns {in_buf, out_buf}, each with room for at least `n_elems` floats,
  // or {nullptr, nullptr} if unavailable or n_elems exceeds capacity.
  std::pair<float *, float *> get(unsigned int n_elems) {
    if (!usable() || n_elems > kMaxElemsPerHalf) {
      return {nullptr, nullptr};
    }
    return {buf_, buf_ + kMaxElemsPerHalf};
  }

  bool usable() const { return buf_ != nullptr; }

private:
  // Sized for one block's largest single q_norm/k_norm call: reshaped to
  // (n_tokens * num_heads) rows of feature_size (head_dim) each. 32 heads *
  // 128 head_dim * 1024 tokens covers Qwen3-0.6B (16 Q heads) with margin,
  // matching the same cap used for mha_core.cpp's RoPE scratch buffer (the
  // existing gemm_q4_0/ffn_swiglu "M>1024 activation rows" limit already
  // forces CPU fallback wholesale past that length anyway).
  static constexpr unsigned int kMaxElemsPerHalf = 32u * 128u * 1024u;

  using AllocFn = void *(*)(int, uint32_t, int);
  using RegisterFn = int (*)(const void *, size_t);

  static constexpr int kHeapIdSystem = 25;
  static constexpr int kDefaultFlags = 1;

  NormScratchRpcMem() {
    void *rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rpc) {
      ml_logw("ReshapedRMSNorm: dlopen(libcdsprpc.so) failed: %s", dlerror());
      return;
    }
    auto alloc = reinterpret_cast<AllocFn>(dlsym(rpc, "rpcmem_alloc"));
    if (!alloc) {
      ml_logw("ReshapedRMSNorm: dlsym(rpcmem_alloc) failed: %s", dlerror());
      return;
    }

    void *bridge = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!bridge) {
      ml_logw("ReshapedRMSNorm: dlopen(libggml-hexagon.so) failed: %s",
              dlerror());
      return;
    }
    auto register_pool = reinterpret_cast<RegisterFn>(
      dlsym(bridge, "nntr_htp_bridge_register_activation_pool"));
    if (!register_pool) {
      ml_logw("ReshapedRMSNorm: dlsym(nntr_htp_bridge_register_activation_"
              "pool) failed: %s",
              dlerror());
      return;
    }

    size_t bytes = static_cast<size_t>(kMaxElemsPerHalf) * 2 * sizeof(float);
    void *p = alloc(kHeapIdSystem, kDefaultFlags, static_cast<int>(bytes));
    if (!p) {
      ml_logw("ReshapedRMSNorm: rpcmem_alloc(%zu) failed for scratch buffer; "
              "FP16 DSP dispatch will stay on CPU",
              bytes);
      return;
    }
    if (register_pool(p, bytes) != 0) {
      ml_logw("ReshapedRMSNorm: bridge rejected scratch pool %p (%zu bytes); "
              "FP16 DSP dispatch will stay on CPU",
              p, bytes);
      return;
    }
    buf_ = static_cast<float *>(p);
  }

  float *buf_ = nullptr;
};

} // namespace

/**
 * @brief Try DSP dispatch for q_norm/k_norm. On this model, in_step/out_step
 * turn out to be FP32 here (the FP16 downcast mha_core.cpp sees happens
 * later, at the connection into mha_core - not in q_norm/k_norm itself,
 * confirmed by direct on-device dtype tracing), so the direct dispatch path
 * mirrors rms_norm.cpp exactly: no cast needed, just call
 * nntr_htp_bridge_rms_norm straight on in_step/out_step. The FP16
 * cast-rotate-cast chain (cast in, normalize on a scratch F32 buffer, cast
 * out - all existing DSP kernels, same pattern as mha_core.cpp's FP16 RoPE
 * attempt) is kept as a fallback for the FP16 case in case a future model
 * config actually exercises it.
 *
 * Returns false (caller falls back to CPU) if any step can't be dispatched;
 * `in_step`/`gamma` are never mutated by the FP16 path (only the scratch
 * buffer is), so the CPU fallback re-reads unmodified original data.
 */
bool try_dsp_fp16_reshaped_rms_norm(bool is_cdsp_engine,
                                    nntrainer::Tensor &in_step,
                                    nntrainer::Tensor &out_step,
                                    const nntrainer::Tensor &gamma,
                                    unsigned int M, unsigned int W,
                                    float epsilon) {
  if (!is_cdsp_engine)
    return false;

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      out_step.getDataType() == ml::train::TensorDim::DataType::FP32 &&
      gamma.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rms_norm_fn norm = get_rms_norm_bridge();
    if (!norm)
      return false;
    int rc = norm(in_step.getData<float>(), gamma.getData<float>(),
                  out_step.getData<float>(), M, W, epsilon);
    if (rc != 0) {
      ml_logw("ReshapedRMSNorm: DSP rms_norm_mul failed (rc=%d), falling "
              "back to CPU",
              rc);
      return false;
    }
    return true;
  }

#ifdef ENABLE_FP16
  if (in_step.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      out_step.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP32) {
    return false;
  }
  cpy_fn cpy = get_cpy_bridge();
  rms_norm_fn norm = get_rms_norm_bridge();
  if (!cpy || !norm) {
    return false;
  }
  unsigned int n_elems = M * W;
  auto [scratch_in, scratch_out] = NormScratchRpcMem::global().get(n_elems);
  if (!scratch_in) {
    return false;
  }
  int rc = cpy(static_cast<const void *>(in_step.getData<_FP16>()),
              static_cast<void *>(scratch_in), n_elems, /*src_is_fp16=*/1,
              /*dst_is_fp16=*/0);
  if (rc != 0) {
    ml_logw("ReshapedRMSNorm: DSP cast-in (F16->F32) failed (rc=%d), "
            "falling back to CPU",
            rc);
    return false;
  }

  rc = norm(scratch_in, gamma.getData<float>(), scratch_out, M, W, epsilon);
  if (rc != 0) {
    ml_logw("ReshapedRMSNorm: DSP rms_norm_mul failed (rc=%d), falling back "
            "to CPU",
            rc);
    return false;
  }

  rc = cpy(static_cast<const void *>(scratch_out),
          static_cast<void *>(out_step.getData<_FP16>()), n_elems,
          /*src_is_fp16=*/0, /*dst_is_fp16=*/1);
  if (rc != 0) {
    ml_logw("ReshapedRMSNorm: DSP cast-out (F32->F16) failed (rc=%d), "
            "falling back to CPU",
            rc);
    return false;
  }
  return true;
#else
  return false;
#endif
}

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);
  use_gamma = std::get<props::UseGamma>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  is_cdsp_engine =
    context.getComputeEngineType() == ml::train::LayerComputeEngine::CDSP;

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  if (use_gamma) {
    // gamma is unquantized FP32 on disk; request FP32 regardless of activation
    // dtype (FP16 would reinterpret the FP32 bytes and corrupt gamma). The FP16
    // path casts gamma down at the multiply site.
    nntrainer::TensorDim gamma_dim(
      1, 1, 1, feature_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       nntrainer::TensorDim::DataType::FP32));
    wt_idx[RMSParams::gamma] = context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
  }
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    // Try DSP dispatch first (prefill-only, matching every other NPU
    // dispatch gate in this codebase: decode is a single row, all
    // round-trip cost, no compute to amortize it against). Only attempted
    // when use_gamma is true - the DSP bridge always applies gamma, no
    // no-gamma variant exists.
    bool dsp_done = false;
    if (is_prefill && is_cdsp_engine && use_gamma &&
        !getenv("NNTR_HEXAGON_NO_ELEM_OPS")) {
      nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
      unsigned int M = step_reshaped_dim.height();
      unsigned int W = step_reshaped_dim.width();
      dsp_done = try_dsp_fp16_reshaped_rms_norm(is_cdsp_engine, in_step,
                                                out_step, gamma, M, W,
                                                epsilon);
    }
    if (dsp_done) {
      // gamma already applied by the DSP kernel - skip the CPU compute and
      // the CPU gamma multiply below entirely.
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      ///@todo rms_norm_wrt_width_something() should be refactored to
      /// nntrainer::Tensor operation.
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);

      // DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!

      // nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      //   in_step.getData<float>(), out_step.getData<float>(),
      //   in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      // FP16 activation: kernel accumulates squares in FP32 (no overflow).
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(),
        in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    if (use_gamma && !dsp_done) {
      nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
      if (gamma.getDataType() != out_step.getDataType()) {
        nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
        out_step.multiply_i(gamma_cast);
      } else {
        out_step.multiply_i(gamma);
      }
    }

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
