// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
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
#include <iostream>

#include "rms_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// ---------------------------------------------------------------------------
// DSP bridge for RMSNorm: dlopen libggml-hexagon.so and dlsym
// nntr_htp_bridge_rms_norm, which dispatches a fused normalize+scale op
// (HTP_OP_RMS_NORM_MUL) to the cDSP in one FastRPC round trip.
// Same lazy-load pattern as mha_core.cpp's flash_attn bridge.
// Returns nullptr if the library or symbol is not found (graceful CPU fallback).
// ---------------------------------------------------------------------------
using rms_norm_fn = int (*)(const float *, const float *, float *,
                            unsigned int, unsigned int, float);

static rms_norm_fn get_rms_norm_bridge() {
  static rms_norm_fn fn = []() -> rms_norm_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("RMSNorm: dlopen(libggml-hexagon.so) failed: %s "
              "(DSP offload disabled, using CPU path)", dlerror());
      return nullptr;
    }
    void *s = dlsym(handle, "nntr_htp_bridge_rms_norm");
    if (!s) {
      ml_logw("RMSNorm: dlsym(nntr_htp_bridge_rms_norm) failed: %s "
              "(DSP offload disabled, using CPU path)", dlerror());
      return nullptr;
    }
    ml_logi("RMSNorm: DSP bridge loaded successfully");
    return reinterpret_cast<rms_norm_fn>(s);
  }();
  return fn;
}

// ---------------------------------------------------------------------------
// rpcmem allocation for gamma. gamma is a weight tensor, so it's never part
// of the graph's shared rpcmem-backed activation pool (weights are kept off
// rpcmem by design - see docs/backend_guide, CMA budget concern for the
// multi-hundred-MB GEMM weight matrices). Confirmed by
// NNTR_HTP_BRIDGE_POOL_DEBUG=1 tracing: "rms_norm:gamma" misses the pool on
// every call while "rms_norm:in"/"rms_norm:out" (activations) hit it. gamma
// is tiny by comparison (one row, width floats), so give it its own small
// persistent rpcmem copy - same allocate-once-register-once pattern
// KVCacheManager uses for the KV cache (kv_cache_manager.cpp), just resolved
// by dlopen/dlsym rather than linking HexagonRpcAllocator directly, for the
// same reason documented there: ENABLE_HEXAGON_CDSP is not propagated to
// this Application build.
// ---------------------------------------------------------------------------
namespace {

class GammaRpcMem {
public:
  static GammaRpcMem &global() {
    static GammaRpcMem inst;
    return inst;
  }

  bool usable() const { return alloc_ && register_pool_; }

  /** @brief rpcmem_alloc + register with the bridge; nullptr on any failure */
  void *allocAndRegister(size_t bytes) {
    if (!usable()) {
      return nullptr;
    }
    void *p = alloc_(kHeapIdSystem, kDefaultFlags, static_cast<int>(bytes));
    if (!p) {
      ml_logw("RMSNorm: rpcmem_alloc(%zu) failed for gamma; falling back to "
              "the bridge's staging memcpy path",
              bytes);
      return nullptr;
    }
    // Registering a new pool while a DSP batch is open hung the device
    // (observed on-device: dspqueue_write succeeds but the following
    // flush_pending never completes) - flush first so registration always
    // happens with no in-flight ops, exactly like KVCacheManager's pool
    // (registered once at model init, before any batch is ever opened).
    // This is a one-time cost per layer (cached after the first call).
    if (flush_fn_) {
      flush_fn_();
    }
    if (register_pool_(p, bytes) != 0) {
      ml_logw("RMSNorm: bridge rejected gamma rpcmem pool %p (%zu bytes); "
              "keeping it but it will not be a zero-copy hit",
              p, bytes);
    }
    return p;
  }

private:
  using AllocFn = void *(*)(int, uint32_t, int);
  using RegisterFn = int (*)(const void *, size_t);
  using FlushFn = int (*)(void);

  static constexpr int kHeapIdSystem = 25;
  static constexpr int kDefaultFlags = 1;

  GammaRpcMem() {
    void *rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rpc) {
      ml_logw("RMSNorm: dlopen(libcdsprpc.so) failed: %s", dlerror());
      return;
    }
    alloc_ = reinterpret_cast<AllocFn>(dlsym(rpc, "rpcmem_alloc"));

    void *bridge = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!bridge) {
      ml_logw("RMSNorm: dlopen(libggml-hexagon.so) failed: %s", dlerror());
      return;
    }
    register_pool_ = reinterpret_cast<RegisterFn>(
      dlsym(bridge, "nntr_htp_bridge_register_activation_pool"));
    flush_fn_ = reinterpret_cast<FlushFn>(
      dlsym(bridge, "nntr_htp_bridge_flush_if_batch_active"));
  }

  AllocFn alloc_ = nullptr;
  RegisterFn register_pool_ = nullptr;
  FlushFn flush_fn_ = nullptr;
};

} // namespace

float *RMSNormLayer::getOrCreateGammaRpcmem(const nntrainer::Tensor &gamma) {
  // Disabled: on-device testing found that registering one small (4KB)
  // rpcmem pool per RMSNormLayer instance works for the first ~12-13 layers,
  // then hangs the DSP/FastRPC driver permanently on the next
  // dspqueue_write/flush_pending (observed at exactly this point across
  // repeated runs - the pool sizes/addresses are consistent with a fixed
  // FastRPC buffer-registration/mmap slot limit being exhausted, not a
  // logic bug in the copy/registration sequence itself). Qwen3-0.6B has 56
  // RMSNormLayer instances, well past that point.
  //
  // The fix is a shared rpcmem arena for all gamma vectors (one
  // rpcmem_alloc + one register_activation_pool call for the whole model,
  // each layer copying gamma into its own offset - the same pattern already
  // used for the graph's activation tensor_pool and for KVCacheManager's
  // K/V cache), not one registration per layer. That needs coordination
  // across RMSNormLayer instances (a shared allocator keyed by model/session,
  // sized once all layers are known) which is a bigger change than this
  // session's scope - falling back to the bridge's existing staging memcpy
  // path for gamma in the meantime; RMSNorm still dispatches to the DSP
  // (in/out activations still hit zero-copy), gamma alone stays staged.
  return nullptr;

  unsigned int width = gamma.getDim().width();
  if (gamma_rpcmem && gamma_rpcmem_width == width) {
    return gamma_rpcmem;
  }

  void *p = GammaRpcMem::global().allocAndRegister(
    static_cast<size_t>(width) * sizeof(float));
  if (!p) {
    return nullptr;
  }

  std::memcpy(p, gamma.getData<float>(), static_cast<size_t>(width) * sizeof(float));
  gamma_rpcmem = static_cast<float *>(p);
  gamma_rpcmem_width = width;
  return gamma_rpcmem;
}


void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  // gamma is unquantized and stored as FP32 in the bin. Request it as FP32
  // regardless of the activation dtype; declaring it FP16 reinterprets the
  // on-disk FP32 bytes as FP16 and corrupts gamma (≈FP16-max garbage). The
  // FP16 forward path casts gamma down to FP16 at the multiply site.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

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

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const auto &dim = in_step.getDim();
      unsigned int M = dim.height();
      unsigned int W = dim.width();

      // Try DSP bridge first (fused normalize + gamma scale in one op).
      // Prefill-only: decode is a single row (M=1), all compute, no
      // amortization for the FastRPC round trip - keep it on CPU, matching
      // every other NPU dispatch gate in this codebase (should_use_flash_attn,
      // should_use_fused_ffn).
      // Primary gate is compute_engine (set from this layer's "engine="
      // property, see withHexagonEngine() at its construction site) rather
      // than an independently-probed env var - this is the same source of
      // truth LayerNode's flush guard uses, so a layer tagged engine=cdsp
      // both skips the pre-layer flush AND actually dispatches here.
      // NNTR_HEXAGON_NO_ELEM_OPS remains as a manual kill-switch on top, for
      // benchmarking without rebuilding.
      const rms_norm_fn &dsp_fn = get_rms_norm_bridge();
      bool dsp_done = false;
      if (is_prefill && dsp_fn &&
          context.getComputeEngineType() ==
            ml::train::LayerComputeEngine::CDSP &&
          gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
          !getenv("NNTR_HEXAGON_NO_ELEM_OPS")) {

        // gamma is a weight tensor and never in the graph's rpcmem-backed
        // activation pool - route it through its own persistent rpcmem copy
        // instead so this operand is a zero-copy pool hit too, not just
        // in/out. Falls back to gamma's own pointer (staged by the bridge,
        // as before) if rpcmem/the bridge isn't available.
        const float *gamma_ptr = getOrCreateGammaRpcmem(gamma);
        if (!gamma_ptr) {
          gamma_ptr = gamma.getData<float>();
        }
        int rc = dsp_fn(in_step.getData<float>(), gamma_ptr,
                        out_step.getData<float>(), M, W, epsilon);
        if (rc == 0) {
          dsp_done = true;
        } else {
          ml_logw("RMSNorm: DSP bridge failed (rc=%d), falling back to CPU", rc);
        }
      }

      if (!dsp_done) {
        // CPU fallback: normalize then multiply by gamma
#ifdef ENABLE_FP16
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(), dim.height(),
          dim.width(), epsilon);
#else
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(), dim.height(),
          dim.width(), epsilon);
#endif
        if (gamma.getDataType() != out_step.getDataType()) {
          nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
          out_step.multiply_i(gamma_cast);
        } else {
          out_step.multiply_i(gamma);
        }
      }
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const auto &dim = in_step.getDim();
      // FP16 activation: this kernel accumulates the sum-of-squares in FP32
      // (so a wide residual row cannot overflow FP16) and reads/writes FP16.
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(), dim.height(),
        dim.width(), epsilon);
      // gamma (unquantized) may be stored at a different dtype than the FP16
      // activation; cast it to match before the elementwise multiply.
      if (gamma.getDataType() != out_step.getDataType()) {
        nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
        out_step.multiply_i(gamma_cast);
      } else {
        out_step.multiply_i(gamma);
      }
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
