// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_compute_ops.cpp
 * @date   22 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  CUDA ComputeOps subclass (mirror of ClComputeOps). P1 provides only
 *         the host-side copy ops so Tensor::copy() works on engine=cuda tensors
 *         (their memory is Unified/managed, hence host-addressable). The
 *         accelerator quantized GEMM/GEMV predicates are left at the base
 *         default (false), so float_tensor.cpp falls back to the CPU path until
 *         the CUDA kernels land in P3 (cuda_operations/).
 */

#include <common_properties.h> // ActivationType (the act_type int encoding)
#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <env_compat.h>
#include <nntrainer_log.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <tensor.h>

#include <cuda_stream_manager.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_blas_manager.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_fc_qs4cx.h>
#include <cuda_gelu.h>
#include <cuda_layernorm.h>
#include <cuda_runtime.h>
#include <fp16.h>
#include <int4_utils.h>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#endif

namespace nntrainer {

// CudaComputeOps derives from CpuComputeOps (not the abstract ComputeOps base):
// engine=cuda tensors are Unified Memory (host-coherent), so every standard op
// runs correctly via the CPU implementations; this class only overrides the
// host-side copy ops for now. Inheriting CpuComputeOps means get_cuda_ops() can
// be installed without throwing on the un-accelerated ops (prereq for the CUDA
// op kernels in a later phase).
class CudaComputeOps : public CpuComputeOps {
public:
  // Plain elementwise copy (Y = X). Tensor::copy() calls this unconditionally
  // (no supports_*() guard); correct for host and (host-coherent) managed
  // pointers. Under the device-only pools (NNTR_CUDA_DEV_ACT / KV_DEV) either
  // endpoint may be cudaMalloc memory the host loop below would fault on --
  // device_copy() routes contiguous same-type copies through a stream-ordered
  // cudaMemcpyAsync (legal inside graph capture, ordered against the
  // producing kernels on the same stream); a copy the host reads next (D2H)
  // drains first. Strided device copies do not occur in the forward path --
  // fail loudly rather than fault.
  static bool device_copy(const void *X, void *Y, size_t bytes,
                          bool contiguous) {
    if (!(cuda::dev_only(X) || cuda::dev_only(Y)))
      return false;
    if (!contiguous)
      throw std::runtime_error(
        "CudaComputeOps: strided copy on device-only memory is unsupported");
    auto &sm = cuda::StreamManager::Global();
    if (cudaMemcpyAsync(Y, X, bytes, cudaMemcpyDefault, sm.GetStream()) !=
        cudaSuccess) {
      cudaGetLastError();
      throw std::runtime_error(
        "CudaComputeOps: device copy (cudaMemcpyAsync) failed");
    }
    if (!cuda::dev_only(Y))
      sm.finish(); // D2H: the host consumes the destination immediately
    return true;
  }

  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    if (device_copy(X, Y, (size_t)N * sizeof(float), incX == 1 && incY == 1))
      return;
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    if (device_copy(X, Y, (size_t)N * sizeof(_FP16), incX == 1 && incY == 1))
      return;
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  // Converting copies with a device-only endpoint: stage through host temps
  // (synchronous; these do not occur inside graph capture today).
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    if (cuda::dev_only(X) || cuda::dev_only(Y)) {
      if (incX != 1 || incY != 1)
        throw std::runtime_error(
          "CudaComputeOps: strided converting copy on device-only memory");
      cuda::StreamManager::Global().finish();
      std::vector<float> xs;
      const float *xp = X;
      if (cuda::dev_only(X)) {
        xs.resize(N);
        cuda::copy_any(xs.data(), X, (size_t)N * sizeof(float));
        xp = xs.data();
      }
      std::vector<_FP16> ys(N);
      for (unsigned int i = 0; i < N; ++i)
        ys[i] = static_cast<_FP16>(xp[i]);
      if (cuda::dev_only(Y))
        cuda::copy_any(Y, ys.data(), (size_t)N * sizeof(_FP16));
      else
        std::memcpy(Y, ys.data(), (size_t)N * sizeof(_FP16));
      return;
    }
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<_FP16>(X[i * incX]);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    if (cuda::dev_only(X) || cuda::dev_only(Y)) {
      if (incX != 1 || incY != 1)
        throw std::runtime_error(
          "CudaComputeOps: strided converting copy on device-only memory");
      cuda::StreamManager::Global().finish();
      std::vector<_FP16> xs;
      const _FP16 *xp = X;
      if (cuda::dev_only(X)) {
        xs.resize(N);
        cuda::copy_any(xs.data(), X, (size_t)N * sizeof(_FP16));
        xp = xs.data();
      }
      std::vector<float> ys(N);
      for (unsigned int i = 0; i < N; ++i)
        ys[i] = static_cast<float>(xp[i]);
      if (cuda::dev_only(Y))
        cuda::copy_any(Y, ys.data(), (size_t)N * sizeof(float));
      else
        std::memcpy(Y, ys.data(), (size_t)N * sizeof(float));
      return;
    }
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<float>(X[i * incX]);
  }
#endif

  // ── Whole-op (Tensor-level) ───────────────────────────────────────────────
  // GeGLU: out = gelu_tanh(gate) * up. Device-resident fp16 kernel (opt-in via
  // NNTR_CUDA_GEGLU until the whole decode chain is on-GPU); otherwise the host
  // gelu loop on the host-coherent UVM tensors (CpuComputeOps::geglu). Matches
  // the former forked GeGLU layer's math byte-for-byte.
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    const unsigned int dim2 = in1.width();
    const size_t elem_off = (size_t)row_offset * dim2;
    const size_t n = (size_t)active_rows * dim2;
    const auto dt = in1.getDataType();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // GPU geglu (device-resident fp16): one kernel instead of the host loop, so
    // the FFN/PLE activation stays on the device. NNTR_CUDA_ASYNC governs the
    // drain.
    if (dt == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = nntr_env_on("NNTR_CUDA_GEGLU");
      if (gpu && n > 0) {
        auto *a = reinterpret_cast<const unsigned short *>(
          in1.getData<_FP16>() + elem_off);
        auto *b = reinterpret_cast<const unsigned short *>(
          in2.getData<_FP16>() + elem_off);
        auto *o =
          reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
        const bool dev = nntrainer::cuda::dev_accessible(a);
        if (dev && cuda::cuda_geglu_fp16(a, b, o, (unsigned int)n))
          return;
      }
    }
#endif

    // Host gelu fallback: sync first so the host read of GPU-produced gate/up
    // is coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::geglu(in1, in2, out, active_rows, row_offset);
  }

  // SwiGLU: out = silu(gate) * up. Mirrors the geglu entry above. The cuda
  // activation pool is device-resident, so the DEVICE kernel is the primary
  // path -- the base CpuComputeOps host loop faults on a device-only buffer.
  // The neutral SwiGLULayer reaches the device through this entry rather than
  // through a branch of its own, so the layer stays free of backend code.
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override {
    const unsigned int dim2 = in1.width();
    const size_t elem_off = (size_t)row_offset * dim2;
    const size_t n = (size_t)active_rows * dim2;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      // dev_accessible() is the whole gate: a tensor this engine did not
      // allocate falls through to the host loop below, which is correct for it.
      if (nntrainer::cuda::dev_accessible(a) &&
          cuda::cuda_swiglu_fp16(a, b, o, (unsigned int)n))
        return;
    }
#endif
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::swiglu(in1, in2, out, active_rows, row_offset);
  }

  // LayerNorm: out = (x-mean)*rsqrt(var+eps)*gamma + beta per row over width.
  // Device fp16 kernel for all-FP16 in/gamma/beta/out within the row gate;
  // everything else (FP32, every mixed activation/weight dtype combo, and
  // rows > gate) runs the INHERITED host loop CpuComputeOps::layer_norm over
  // the host-coherent UVM tensors — i.e. UNACCELERATED rather than
  // "CUDA support". cuda_layernorm_fp32 exists and is
  // covered by unittest_cuda_kernels_layernorm, but is deliberately not routed
  // from here yet: it has had no in-graph validation, unlike the fp16 path.
  //
  // The row gate is a CUDA-specific PERFORMANCE POLICY and belongs here, in the
  // op, never in the Layer (a Layer branching on backend behaviour is exactly
  // the fork smell this collapse removes). Rationale: the kernel syncs per
  // call, so for a wide prefill norm (rows = seq_len) the multi-threaded host
  // loop over UVM wins; gating by rows gives the decode speedup with no prefill
  // regression (same tradeoff as CudaRMSNormLayer). ClComputeOps gets NO
  // equivalent gate and that is correct, not an oversight — it has no host
  // fallback to fall back to. Replaces the former forked LayerNorm layer.
  void layer_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                  const Tensor &beta, float epsilon, unsigned int active_rows,
                  unsigned int row_offset) override {
    const unsigned int width = in.width();
    const size_t elem_off = (size_t)row_offset * width;

    if (std::getenv("NNTR_CUDA_DBG")) {
      static int _n = 0;
      if (_n++ < 3)
        std::fprintf(stderr,
                     "[CUDA-DBG] CudaComputeOps::layer_norm rows=%u width=%u\n",
                     active_rows, width);
    }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    using DT = ml::train::TensorDim::DataType;
    // NNTR_LAYERNORM_CUDA_OFF: unset => 32-row decode-only cap, "a"/"all" =>
    // uncapped, anything else => off. CudaContext::initialize() sets "all" on
    // discrete GPUs next to the RMSNorm cap raise.
    static const int gpu_max_rows = []() {
      const char *e = std::getenv("NNTR_LAYERNORM_CUDA_OFF");
      if (e && e[0] == 'a')
        return 1 << 30; // "all"
      if (e)
        return 0; // off
      return 32;  // decode-only default
    }();
    if (in.getDataType() == DT::FP16 && gamma.getDataType() == DT::FP16 &&
        beta.getDataType() == DT::FP16 && out.getDataType() == DT::FP16 &&
        (int)active_rows <= gpu_max_rows && active_rows > 0) {
      auto *xi = reinterpret_cast<const unsigned short *>(in.getData<_FP16>() +
                                                          elem_off);
      auto *gi =
        reinterpret_cast<const unsigned short *>(gamma.getData<_FP16>());
      auto *bi =
        reinterpret_cast<const unsigned short *>(beta.getData<_FP16>());
      auto *yi =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(xi) &&
          nntrainer::cuda::dev_accessible(gi) &&
          nntrainer::cuda::dev_accessible(bi) &&
          nntrainer::cuda::dev_accessible(yi) &&
          cuda::cuda_layernorm_fp16(xi, gi, bi, yi, epsilon, active_rows,
                                    width))
        return;
    }
#endif
    // Host layernorm fallback (UNACCELERATED): sync first so the host read of a
    // GPU-produced input is coherent under NNTR_CUDA_ASYNC (no-op in sync
    // mode).
    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::layer_norm(in, out, gamma, beta, epsilon, active_rows,
                              row_offset);
  }

  // Element-wise activation. Device fp16 GELU/tanh-GELU kernel; every other
  // mode and every other dtype runs the INHERITED host ActiFunc over UVM —
  // UNACCELERATED, so say so rather than claiming CUDA support.
  // Note there is NO row gate here and there should not be one: this is a flat
  // 1-D elementwise map with no host-wins crossover (mirrors the ungated
  // swiglu/geglu CUDA fast paths). Replaces the former CudaActivationLayer
  // fork; the ActivationType -> mode mapping (its getGeluMode) lives here now,
  // because it is a backend concern.
  void activation(const Tensor &in, Tensor &out, int act_type,
                  unsigned int active_rows, unsigned int row_offset) override {
    const auto at = static_cast<ActivationType>(act_type);
    const unsigned int width = in.width();
    const size_t elem_off = (size_t)row_offset * width;
    const size_t n = (size_t)active_rows * width;
    const bool is_gelu =
      (at == ActivationType::ACT_GELU || at == ActivationType::ACT_TANH_GELU);

    if (std::getenv("NNTR_CUDA_DBG")) {
      static int _n = 0;
      if (_n++ < 3)
        std::fprintf(stderr,
                     "[CUDA-DBG] CudaComputeOps::activation n=%zu act=%d\n", n,
                     act_type);
    }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    using DT = ml::train::TensorDim::DataType;
    if (is_gelu && n > 0 && in.getDataType() == DT::FP16 &&
        out.getDataType() == DT::FP16) {
      const int mode = (at == ActivationType::ACT_TANH_GELU) ? 1 : 0;
      auto *xi = reinterpret_cast<const unsigned short *>(in.getData<_FP16>() +
                                                          elem_off);
      auto *yi =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(xi) &&
          nntrainer::cuda::dev_accessible(yi) &&
          cuda::cuda_gelu_fp16(xi, yi, mode, (unsigned int)n))
        return;
    }
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // Under a device-only activation pool (NNTR_CUDA_DEV_ACT) the host loop
    // below would FAULT on a non-UVM pointer. Fail loudly instead: the caller
    // must either use an accelerated mode/dtype or turn the pool off.
    if (n > 0 && (cuda::dev_only(in.getData<uint8_t>()) ||
                  cuda::dev_only(out.getData<uint8_t>())))
      throw std::runtime_error(
        "CudaComputeOps::activation: this activation mode/dtype has no device "
        "kernel and the tensors are device-only (NNTR_CUDA_DEV_ACT); the host "
        "path would fault");
#endif

    cuda::StreamManager::Global().finishIfAsync();
    CpuComputeOps::activation(in, out, act_type, active_rows, row_offset);
  }

  // FC GEMM: output = input * weight. The former CudaFcLayer::cudaFcGemm body
  // — QS4CX fused dequant-GEMM on device (plain payload consumed in place,
  // w4a8 dp4a / cuBLAS int8 IMMA) with host-weight/host-input staging, an FP32
  // cuBLAS path, and a host Tensor::dot fallback (correct on the host-coherent
  // UVM).
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    using DT = ml::train::TensorDim::DataType;
    Tensor &input_ = input;
    Tensor &hidden_ = output;
    const DT wt = weight.getDataType();
    const DT at = input_.getDataType();

    const auto &id = input_.getDim();
    const auto &od = hidden_.getDim();
    const int K = (int)id.width();
    const int N = (int)od.width();
    const int M = (int)(id.batch() * id.channel() * id.height());

    static const bool fc_dbg = std::getenv("NNTR_FC_DEBUG") != nullptr;
    if (fc_dbg) {
      auto ptype = [](const void *p) {
        cudaPointerAttributes a{};
        bool ok = cudaPointerGetAttributes(&a, p) == cudaSuccess;
        cudaGetLastError();
        if (!ok)
          return 'u';
        switch (a.type) {
        case cudaMemoryTypeManaged:
          return 'm';
        case cudaMemoryTypeDevice:
          return 'd';
        case cudaMemoryTypeHost:
          return 'h';
        default:
          return '0';
        }
      };
      fprintf(stderr,
              "[FCDBG] wt=%d at=%d ot=%d M=%d N=%d K=%d in=%c w=%c out=%c\n",
              (int)wt, (int)at, (int)hidden_.getDataType(), M, N, K,
              ptype(input_.getData<float>()), ptype(weight.getData<uint8_t>()),
              ptype(hidden_.getData<float>()));
    }

    // QS4CX weight: fused dequant-GEMM on device, consuming the PLAIN nibble
    // payload in place -- the derived dp4a/cuBLAS device caches are keyed by
    // its pointer, so the payload is never copied. Default on; the host
    // Tensor::dot fallback below has no x86 implementation for this dtype.
    if (wt == DT::QS4CX && (at == DT::FP32 || at == DT::FP16) && M > 0 &&
        N > 0 && K > 0) {
      static const bool qs4cx_enabled = []() {
        const char *e = std::getenv("NNTR_FC_CUDA_QS4CX");
        return !(e != nullptr && e[0] == '0');
      }();
      const uint8_t *W = weight.getData<uint8_t>();
#ifdef ENABLE_FP16
      // [lmhead-tie-lut] Re-tied head: the lm_head's weight IS the embedding
      // sidecar LUT, resident in VRAM since load, and W is a pure routing key
      // -- no derived cache, no scale side buffer, and (with the plain payload
      // dropped) no dereferenceable bytes behind W at all. That is why this
      // arm sits BEFORE the scale/cache entry ticket below: a tied head passes
      // neither test by construction (no fp16-scale cache is built for it, and
      // the staging that would follow a cache miss would upload dropped pages).
      // Shape gate only (the fp-act decode shape); the callee refuses unless
      // the app registered THIS weight's LUT at load, so with the tie inactive
      // the arm costs one mutexed pointer compare and falls through unchanged.
      if (M == 1 && N >= (int)cuda::CUDA_FC_FPACT_MIN_N && at == DT::FP16 &&
          hidden_.getDataType() == DT::FP16 &&
          (int)weight.getDim().height() == K) {
        auto *Xh =
          reinterpret_cast<const unsigned short *>(input_.getData<_FP16>());
        auto *Yh = reinterpret_cast<unsigned short *>(hidden_.getData<_FP16>());
        if (nntrainer::cuda::dev_accessible(Xh) &&
            cuda::cuda_fc_lmhead_tie_gemv_fp16(W, Xh, Yh, (unsigned)N,
                                               (unsigned)K))
          return;
      }
#endif
      // The only per-weight side buffer: the N fp16 per-channel scales the
      // dequant kernels read every call (cached UVM; built at load by the
      // prewarm, so this is a pure cache hit -- a miss under graph capture
      // returns false and the FC falls to the host path).
      const uint16_t *S = nullptr;
      if (qs4cx_enabled && (int)weight.getDim().height() == K &&
          cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(weight.getScale<float>(),
                                                 (unsigned)N, &S)) {
        // A prewarmed dp4a cache makes W a pure lookup key -- no kernel
        // dereferences the payload -- so a host-resident W must NOT be staged
        // in that case: staging would both miss the prewarmed cache, which is
        // keyed by the original pointer, and read a payload the caller may no
        // longer be keeping. Stage only when there is no cache to hit.
        const bool w_keyed = cuda::cuda_fc_qs4cx_has_cache(W);
        if (!w_keyed && !nntrainer::cuda::dev_accessible(W)) {
          const uint8_t *dW = nullptr;
          const uint16_t *dS = nullptr;
          if (cuda::cuda_fc_qs4cx_stage_host_weight(W, S, (unsigned)N,
                                                    (unsigned)K, &dW, &dS)) {
            W = dW;
            S = dS;
          }
        }
        const bool fp16 = (at == DT::FP16);
        const void *Xp = fp16 ? (const void *)input_.getData<uint16_t>()
                              : (const void *)input_.getData<float>();
        void *Yp = fp16 ? (void *)hidden_.getData<uint16_t>()
                        : (void *)hidden_.getData<float>();
        static const bool use_dp4a = []() {
          const char *e = std::getenv("NNTR_FC_CUDA_DP4A");
          return !(e != nullptr && e[0] == '0');
        }();
        static const bool use_cublas_i8 = []() {
          const char *e = std::getenv("NNTR_FC_CUDA_CUBLAS");
          return e != nullptr && e[0] == '1';
        }();
        const bool x_dev = nntrainer::cuda::dev_accessible(Xp);
        const bool wy_dev = (w_keyed || nntrainer::cuda::dev_accessible(W)) &&
                            nntrainer::cuda::dev_accessible(Yp);
        bool all_dev = x_dev && wy_dev;
        if (!x_dev && wy_dev && fp16) {
          if (const uint16_t *Xd = cuda::cuda_fc_qs4cx_stage_host_x_fp16(
                (const uint16_t *)Xp, (unsigned)M, (unsigned)K)) {
            Xp = (const void *)Xd;
            all_dev = true;
          }
        }
        if (std::getenv("NNTR_FC_HOSTDBG") && !x_dev) {
          std::fprintf(stderr,
                       "[FC-HOSTDBG] host-input M=%u K=%u N=%u wy_dev=%d "
                       "fp16=%d staged=%d capturing=%d\n",
                       (unsigned)M, (unsigned)K, (unsigned)N, (int)wy_dev,
                       (int)fp16, (int)all_dev,
                       (int)cuda::StreamManager::Global().isCapturing());
        }
        bool ok = false;
        if (all_dev && fp16) {
          static const unsigned cublas_kmax = []() {
            const char *e = std::getenv("NNTR_FC_CUBLAS_KMAX");
            return e ? (unsigned)atoi(e) : (1u << 20);
          }();
          // Decode on a vocab-wide output (the untied int4 lm_head) goes to
          // the fp-ACTIVATION int4 GEMV ahead of everything else: at
          // N = 262144 the int8 activation quant's per-logit noise outweighs
          // the argmax margin and costs sampled tokens (see
          // cuda_fc_qs4cx_fpact_gemv_fp16). Ordinary projections
          // (N ~ 1536-12288) stay on dp4a, which is faster and whose noise
          // they can afford. Like the cuBLAS gate below this is the SHAPE
          // only: NNTR_CUDA_LMHEAD_FPACT=0 is enforced inside the callee,
          // which then reports failure and lets the usual route take the call.
          if (M == 1 && N >= (int)cuda::CUDA_FC_FPACT_MIN_N)
            ok = cuda::cuda_fc_qs4cx_fpact_gemv_fp16(
              (const uint16_t *)Xp, W, weight.getScale<float>(), (uint16_t *)Yp,
              (unsigned)N, (unsigned)K);
          const bool tried_cublas = !ok && use_cublas_i8 && use_dp4a &&
                                    M >= 32 && K <= (int)cublas_kmax;
          if (tried_cublas)
            ok = cuda::cuda_fc_qs4cx_cublas_i8_gemm_fp16(
              (const uint16_t *)Xp, W, S, (uint16_t *)Yp, (unsigned)M,
              (unsigned)N, (unsigned)K);
          if (tried_cublas && !ok) {
            // One-time: cuBLAS was attempted (BlasManager initialized OK,
            // NNTR_FC_CUDA_CUBLAS=1) but this call failed, so every
            // subsequent qualifying GEMM in this process silently falls back
            // to dp4a (~10x slower prefill) unless the caller is watching
            // logs -- surface it once instead of letting it pass unnoticed.
            static bool warned = false;
            if (!warned) {
              warned = true;
              ml_logw("[CUDA] cuBLAS unavailable, falling back to dp4a GEMM "
                      "(~10x slower prefill); check that the driver "
                      "supports CUDA 13.x");
            }
          }
          if (!ok)
            ok = use_dp4a ? cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(
                              (const uint16_t *)Xp, W, S, (uint16_t *)Yp,
                              (unsigned)M, (unsigned)N, (unsigned)K)
                          : cuda::cuda_fc_qs4cx_gemm_fp16_naive(
                              (const uint16_t *)Xp, W, S, (uint16_t *)Yp,
                              (unsigned)M, (unsigned)N, (unsigned)K);
        } else if (all_dev) {
          ok = use_dp4a
                 ? cuda::cuda_fc_qs4cx_dp4a_gemm_fp32((const float *)Xp, W, S,
                                                      (float *)Yp, (unsigned)M,
                                                      (unsigned)N, (unsigned)K)
                 : cuda::cuda_fc_qs4cx_gemm_fp32((const float *)Xp, W, S,
                                                 (float *)Yp, (unsigned)M,
                                                 (unsigned)N, (unsigned)K);
        } else if (!fp16) {
          ok = cuda::cuda_fc_qs4cx_gemm_fp32_resident((const float *)Xp, W, S,
                                                      (float *)Yp, (unsigned)M,
                                                      (unsigned)N, (unsigned)K);
        }
        if (std::getenv("NNTR_FC_HOSTDBG") && !ok) {
          cudaPointerAttributes aw{}, ay{};
          cudaError_t ew = cudaPointerGetAttributes(&aw, W);
          cudaError_t ey = cudaPointerGetAttributes(&ay, Yp);
          cudaGetLastError();
          std::fprintf(
            stderr,
            "[FC-GPUFAIL] ok=0 -> HOST i8mm: M=%u K=%u N=%u x_dev=%d wy_dev=%d "
            "| W: err=%d type=%d  Y: err=%d type=%d  cap=%d\n",
            (unsigned)M, (unsigned)K, (unsigned)N, (int)x_dev, (int)wy_dev,
            (int)ew, (int)aw.type, (int)ey, (int)ay.type,
            (int)cuda::StreamManager::Global().isCapturing());
        }
        if (ok)
          return;
      }
    }

    // FP32 weight: cuBLAS SGEMM on the UVM pointers.
    if (wt == DT::FP32 && at == DT::FP32 && M > 0 && N > 0 && K > 0 &&
        nntrainer::cuda::dev_accessible(input_.getData<float>()) &&
        nntrainer::cuda::dev_accessible(weight.getData<float>()) &&
        nntrainer::cuda::dev_accessible(hidden_.getData<float>()) &&
        cuda::BlasManager::Global().sgemmRowMajor(
          M, N, K, input_.getData<float>(), weight.getData<float>(),
          hidden_.getData<float>())) {
      cuda::StreamManager::Global().maybeFinish();
      return;
    }
#endif

    // Host fallback: correct for FP16 / Q4_x / Q6_K / cross-engine host input
    // (and any GPU-path failure) on the host-coherent UVM tensors.
    CpuComputeOps::fc(input, weight, output);
  }
};

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer
