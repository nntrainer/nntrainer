// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_compute_ops.cpp
 * @date    29 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA ComputeOps subclass for the cuda context. Inherits
 *          CpuComputeOps: cuda tensors default to Unified Memory
 *          (host-coherent), so every op the CUDA backend does not accelerate
 *          runs correctly via the CPU implementation over the managed buffers.
 *          Overrides the element-wise decode dispatches (behind the same
 *          runtime gates the neutral layers used to open-code), the rms_norm
 *          whole-op, the FC GEMM dispatch, and the copy ops (the latter so a
 *          Tensor::copy is correct on the device-only activation pool).
 */

#include "cuda_compute_ops.h"

#include <cmath>
#include <stdexcept>

#include <compute_ops.h>
#include <env_compat.h>
#include <tensor.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_fc_qint4.h>
#include <cuda_rmsnorm.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>

namespace nntrainer {

void CudaComputeOps::swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                            unsigned int active_rows, unsigned int row_offset) {
#ifdef ENABLE_FP16
  // engine=cuda device-resident fp16: one kernel instead of the host loop
  // (the host body below would fault on the device-only activation pool
  // under NNTR_CUDA_DEV_ACT). Gated on FP16 + batch/channel==1 +
  // row_offset==0 -- the batch/channel==1 gate mirrors the layer-side gate
  // this override replaces (with it, active_rows * width() equals the
  // (to - from) * width() element count the layer's former open-coded block
  // launched); falls through to the host body for non-device tensors.
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      in1.batch() == 1 && in1.channel() == 1 && row_offset == 0) {
    const size_t n = (size_t)active_rows * in1.width();
    auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>());
    auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>());
    auto *o = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    const bool dev = a && nntrainer::cuda::dev_accessible(a);
    if (dev && n > 0 &&
        nntrainer::cuda::cuda_swiglu_fp16(a, b, o, (unsigned int)n))
      return;
  }
#endif
  CpuComputeOps::swiglu(in1, in2, out, active_rows, row_offset);
}

// GeGLU: out = gelu_tanh(gate) * up. Device-resident fp16 kernel (opt-in via
// NNTR_CUDA_GEGLU until the whole decode chain is on-GPU); otherwise the host
// gelu loop on the host-coherent UVM tensors (CpuComputeOps::geglu).
void CudaComputeOps::geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
  const auto dt = in1.getDataType();

#ifdef ENABLE_FP16
  // GPU geglu (device-resident fp16): one kernel instead of the host loop, so
  // the FFN/PLE activation stays on the device. NNTR_CUDA_ASYNC governs the
  // drain.
  if (dt == ml::train::TensorDim::DataType::FP16) {
    static const bool gpu = std::getenv("NNTR_CUDA_GEGLU") != nullptr;
    if (gpu && n > 0) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      const bool dev = nntrainer::cuda::dev_accessible(a);
      if (dev && nntrainer::cuda::cuda_geglu_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif

  // Host gelu fallback: sync first so the host read of GPU-produced gate/up
  // is coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
  nntrainer::cuda::drain_if_async();
  CpuComputeOps::geglu(in1, in2, out, active_rows, row_offset);
}

// Fused sigmoid gates on cuda (mirror of geglu above). A device-resident
// activation pool makes the DEVICE kernel the primary path (the base
// CpuComputeOps host loop faults on a device-only activation in runDecode).
// Host loop only for genuinely host tensors.
// Kill-switch: NNTR_CUDA_SIGMOID_GATE=0.
void CudaComputeOps::sigmoid_glu(const Tensor &in1, const Tensor &in2,
                                 Tensor &out, unsigned int active_rows,
                                 unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
#ifdef ENABLE_FP16
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
    static const bool gpu = []() {
      const char *e = std::getenv("NNTR_CUDA_SIGMOID_GATE");
      return !(e && e[0] == '0');
    }();
    if (gpu) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(a) &&
          nntrainer::cuda::cuda_sigmoid_glu_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif
  nntrainer::cuda::drain_if_async();
  CpuComputeOps::sigmoid_glu(in1, in2, out, active_rows, row_offset);
}

void CudaComputeOps::sigmoid_add(const Tensor &in1, const Tensor &in2,
                                 Tensor &out, unsigned int active_rows,
                                 unsigned int row_offset) {
  const unsigned int dim2 = in1.width();
  const size_t elem_off = (size_t)row_offset * dim2;
  const size_t n = (size_t)active_rows * dim2;
#ifdef ENABLE_FP16
  if (in1.getDataType() == ml::train::TensorDim::DataType::FP16 && n > 0) {
    static const bool gpu = []() {
      const char *e = std::getenv("NNTR_CUDA_SIGMOID_GATE");
      return !(e && e[0] == '0');
    }();
    if (gpu) {
      auto *a = reinterpret_cast<const unsigned short *>(in1.getData<_FP16>() +
                                                         elem_off);
      auto *b = reinterpret_cast<const unsigned short *>(in2.getData<_FP16>() +
                                                         elem_off);
      auto *o =
        reinterpret_cast<unsigned short *>(out.getData<_FP16>() + elem_off);
      if (nntrainer::cuda::dev_accessible(a) &&
          nntrainer::cuda::cuda_sigmoid_add_fp16(a, b, o, (unsigned int)n))
        return;
    }
  }
#endif
  nntrainer::cuda::drain_if_async();
  CpuComputeOps::sigmoid_add(in1, in2, out, active_rows, row_offset);
}

void CudaComputeOps::scalar_mul(const Tensor &in, Tensor &out, float scale) {
#ifdef ENABLE_FP16
  if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
    if (gpu) {
      auto *ip = reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
      auto *op = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
      const bool dev = nntrainer::cuda::dev_accessible(ip);
      if (dev && nntrainer::cuda::cuda_scalar_mul_fp16(
                   ip, op, (unsigned int)in.size(), scale))
        return;
    }
  }
#endif
  // Host multiply reads the GPU-produced UVM input on the CPU; sync first
  // in async mode (no-op in default sync mode).
  nntrainer::cuda::drain_if_async();
  CpuComputeOps::scalar_mul(in, out, scale);
}

void CudaComputeOps::softcap(const Tensor &in, Tensor &out, float cap,
                             int act_type) {
  // Terminal drain for the selective-sync (NNTR_CUDA_ASYNC) path: the softcap
  // input is the first host-read point of the lm_head logits, so the
  // one-per-token GPU pipeline drains here. Per call (the layer chunks are
  // per batch/channel); the drain is idempotent and a no-op in default mode
  // (every GPU op already drained). cuda runs only: StreamManager::Global()
  // would CREATE the CUDA context.
  if (nntrainer::cuda::engine_selected())
    nntrainer::cuda::StreamManager::Global().finish();
#ifdef ENABLE_FP16
  // Device-only activation pool: the logits are real device memory; the host
  // Tensor ops in the fallback would fault. out = cap * tanh(in / cap) in one
  // GPU kernel -- the kernel realizes tanh, the activation every reachable
  // configuration sets; the routing (device kernel regardless of act_type) is
  // the same the layer's former open-coded block applied.
  if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    auto *ip = reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
    auto *op = reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    cudaPointerAttributes pa{};
    // Accept Managed (UVM) too, not just Device: on integrated GPUs the
    // activation pool is cudaMallocManaged, so a Device-only gate sends the
    // softcap to the host fallback -- which, inside a CUDA-graph capture,
    // reads the not-yet-run lm_head logits (stale) and is itself not
    // captured -> garbage output. Managed pointers run the GPU kernel fine.
    if (nntrainer::cuda::engine_selected() &&
        cudaPointerGetAttributes(&pa, ip) == cudaSuccess &&
        (pa.type == cudaMemoryTypeDevice || pa.type == cudaMemoryTypeManaged) &&
        nntrainer::cuda::cuda_softcap_fp16(ip, op, (unsigned int)in.size(),
                                           cap)) {
      cudaGetLastError();
      return;
    }
    cudaGetLastError();
  }
#endif
  CpuComputeOps::softcap(in, out, cap, act_type);
}

namespace {
// x * rsqrt(mean(x^2)+eps) * gamma, sum-of-squares accumulated in FP32 (no
// fp16 overflow). rows = leading dims folded, width = feature size.
template <typename T, typename G>
void rmsnorm_rows(const T *x, const G *g, T *y, unsigned int rows,
                  unsigned int width, float eps) {
  for (unsigned int r = 0; r < rows; ++r) {
    const T *xr = x + (size_t)r * width;
    T *yr = y + (size_t)r * width;
    float ss = 0.f;
    for (unsigned int k = 0; k < width; ++k) {
      float v = (float)xr[k];
      ss += v * v;
    }
    float inv = 1.0f / std::sqrt(ss / (float)width + eps);
    for (unsigned int k = 0; k < width; ++k)
      yr[k] = (T)(((float)xr[k] * inv) * (float)g[k]);
  }
}

#ifdef ENABLE_FP16
bool dev_ok(const void *p) { return nntrainer::cuda::dev_accessible(p); }
#endif

void rmsnorm_dispatch(const Tensor &in, const Tensor &gamma, Tensor &out,
                      unsigned int rows, unsigned int width, float eps) {
  using DT = ml::train::TensorDim::DataType;
  const DT dt = in.getDataType();
  const DT gt = gamma.getDataType();
#ifdef ENABLE_FP16
  // GPU path: fp16 in/out/gamma all device-resident (UVM). Block-per-row, FP32
  // sum-of-squares. Used only for small row counts (decode, rows~1): the kernel
  // syncs per call, so for the wide prefill norm (rows=seq_len) the
  // multi-thread host norm wins -- gating by rows gives the decode speedup
  // without a prefill regression.
  static constexpr int gpu_max_rows = 32;
  if (dt == DT::FP16 && gt == DT::FP16 && out.getDataType() == DT::FP16 &&
      (int)rows <= gpu_max_rows) {
    const unsigned short *xi =
      reinterpret_cast<const unsigned short *>(in.getData<_FP16>());
    const unsigned short *gi =
      reinterpret_cast<const unsigned short *>(gamma.getData<_FP16>());
    unsigned short *yi =
      reinterpret_cast<unsigned short *>(out.getData<_FP16>());
    if (dev_ok(xi) && dev_ok(gi) && dev_ok(yi) &&
        cuda::cuda_rmsnorm_fp16(xi, gi, yi, eps, rows, width))
      return;
  }
#endif
  // Host rmsnorm fallback: sync first so the host read of GPU-produced input is
  // coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
  cuda::StreamManager::Global().finishIfAsync();
  if (dt == DT::FP32 && gt == DT::FP32) {
    rmsnorm_rows(in.getData<float>(), gamma.getData<float>(),
                 out.getData<float>(), rows, width, eps);
#ifdef ENABLE_FP16
  } else if (dt == DT::FP16 && gt == DT::FP16) {
    rmsnorm_rows(in.getData<_FP16>(), gamma.getData<_FP16>(),
                 out.getData<_FP16>(), rows, width, eps);
  } else if (dt == DT::FP16 && gt == DT::FP32) {
    rmsnorm_rows(in.getData<_FP16>(), gamma.getData<float>(),
                 out.getData<_FP16>(), rows, width, eps);
  } else if (dt == DT::FP32 && gt == DT::FP16) {
    rmsnorm_rows(in.getData<float>(), gamma.getData<_FP16>(),
                 out.getData<float>(), rows, width, eps);
#endif
  } else {
    throw std::invalid_argument(
      "CudaComputeOps::rms_norm: unsupported data type");
  }
}
} // namespace

void CudaComputeOps::rms_norm(const Tensor &in, Tensor &out,
                              const Tensor &gamma, float epsilon,
                              unsigned int active_rows,
                              unsigned int row_offset) {
  // rmsnorm_dispatch consumes base pointers + a row count, so the
  // (active_rows, row_offset) window becomes a shared-data view at the row
  // offset. Every in-tree caller passes row_offset 0, where the views alias
  // the arguments' own buffers -- the same pointers the former per-backend
  // layer handed the dispatch.
  const unsigned int width = in.width();
  const size_t elem_off = (size_t)row_offset * width;
  Tensor in_win = in.getSharedDataTensor(
    TensorDim(1, 1, active_rows, width, in.getDim().getTensorType()), elem_off,
    true);
  Tensor out_win = out.getSharedDataTensor(
    TensorDim(1, 1, active_rows, width, out.getDim().getTensorType()), elem_off,
    true);
  rmsnorm_dispatch(in_win, gamma, out_win, active_rows, width, epsilon);
}

// FC GEMM: output = input * weight. QS4CX weight -> fused dequant-GEMM on
// device, consuming the PLAIN nibble payload in place (single weight copy, no
// UVM copy). QINT4 never reaches here: layer_context coerces it to QS4CX at
// init.
void CudaComputeOps::fc(Tensor &input, Tensor &weight, Tensor &output) {
  using DT = ml::train::TensorDim::DataType;
  const DT wt = weight.getDataType();
  const DT at = input.getDataType();

  const auto &id = input.getDim();
  const auto &od = output.getDim();
  const int K = (int)id.width();
  const int N = (int)od.width();
  const int M = (int)(id.batch() * id.channel() * id.height());

  if (wt == DT::QS4CX && M > 0 && N > 0 && K > 0 &&
      (int)weight.getDim().height() == K) {
    const uint8_t *W = weight.getData<uint8_t>();
    // The per-weight fp16 scale buffer the dequant kernel reads every call.
    const uint16_t *S = nullptr;
    if (nntrainer::cuda::dev_accessible(W) &&
        cuda::cuda_fc_qs4cx_scales_to_uvm_fp16(weight.getScale<float>(),
                                               (unsigned)N, &S)) {
#ifdef ENABLE_FP16
      if (at == DT::FP16 && output.getDataType() == DT::FP16) {
        auto *Xh =
          reinterpret_cast<const unsigned short *>(input.getData<_FP16>());
        auto *Yh = reinterpret_cast<unsigned short *>(output.getData<_FP16>());
        // Prefill (M>=32): w4a8 on the INT8 Tensor Cores via cuBLAS (~10x the
        // dp4a int-ALU GEMM, bit-identical). Then the dp4a fast path, then
        // the naive plain GEMM -- each falls to the next on failure.
        const bool prefill = M >= 32;
        if (nntrainer::cuda::dev_accessible(Xh) &&
            ((prefill &&
              cuda::cuda_fc_qs4cx_cublas_i8_gemm_fp16(
                Xh, W, S, Yh, (unsigned)M, (unsigned)N, (unsigned)K)) ||
             cuda::cuda_fc_qs4cx_dp4a_gemm_fp16(Xh, W, S, Yh, (unsigned)M,
                                                (unsigned)N, (unsigned)K) ||
             cuda::cuda_fc_qs4cx_gemm_fp16_naive(Xh, W, S, Yh, (unsigned)M,
                                                 (unsigned)N, (unsigned)K)))
          return;
      }
#endif
      if (at == DT::FP32 && output.getDataType() == DT::FP32) {
        const float *X = input.getData<float>();
        float *Y = output.getData<float>();
        // w4a8 dp4a fast path; falls to the naive plain GEMM on failure.
        if (nntrainer::cuda::dev_accessible(X) &&
            (cuda::cuda_fc_qs4cx_dp4a_gemm_fp32(X, W, S, Y, (unsigned)M,
                                                (unsigned)N, (unsigned)K) ||
             cuda::cuda_fc_qs4cx_gemm_fp32(X, W, S, Y, (unsigned)M, (unsigned)N,
                                           (unsigned)K)))
          return;
      }
    }
  }

  // Host fallback: the input is host-coherent UVM, so the CPU dot is correct.
  // Drain first in async mode so the host read sees the produced input.
  // NNTR_CUDA_FC_DBG=1 prints WHY a call fell off the device fast paths --
  // the fall-through above is silent by design (checklist B.15: a CUDA op
  // falling to the host loop is invisible without a runtime trace).
  static const bool fc_dbg = []() {
    const char *e = std::getenv("NNTR_CUDA_FC_DBG");
    return e && e[0] == '1';
  }();
  if (fc_dbg) {
    static int n_prints = 0;
    if (n_prints < 64) {
      ++n_prints;
      std::fprintf(
        stderr,
        "[CUDA-FC-DBG] host-dot fallback: wdt=%d adt=%d odt=%d M=%d N=%d "
        "K=%d w_h=%d dev(W)=%d dev(X)=%d\n",
        (int)wt, (int)at, (int)output.getDataType(), M, N, K,
        (int)weight.getDim().height(),
        (int)nntrainer::cuda::dev_accessible(weight.getData<uint8_t>()),
        (int)nntrainer::cuda::dev_accessible(input.getData<char>()));
    }
  }
  cuda::StreamManager::Global().finishIfAsync();
  input.dot(weight, output, false, false);
}

// ── Copy ops (device-only aware) ─────────────────────────────────────────
// Under the device-only activation pool (NNTR_CUDA_DEV_ACT) an activation is
// real device memory; Tensor::copy() -> the CpuComputeOps host loop would
// fault on it. Route contiguous device-only copies through a stream-ordered
// cudaMemcpyAsync; host / host-coherent UVM keep the CPU path.
static bool device_copy(const void *X, void *Y, size_t bytes, bool contiguous) {
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
  if (!cuda::dev_only(Y)) {
    if (sm.isCapturing())
      std::fprintf(
        stderr,
        "[CAP-AUDIT] scopy D2H (host-consumed) during capture: %zu bytes\n",
        bytes);
    sm.finish(); // D2H: the host consumes the destination immediately
  }
  return true;
}

void CudaComputeOps::scopy_fp32(const unsigned int N, const float *X,
                                const unsigned int incX, float *Y,
                                const unsigned int incY) {
  if (device_copy(X, Y, (size_t)N * sizeof(float), incX == 1 && incY == 1))
    return;
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

#ifdef ENABLE_FP16
void CudaComputeOps::scopy_fp16(const unsigned int N, const _FP16 *X,
                                const unsigned int incX, _FP16 *Y,
                                const unsigned int incY) {
  if (device_copy(X, Y, (size_t)N * sizeof(_FP16), incX == 1 && incY == 1))
    return;
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}
// Converting copies with a device-only endpoint: stage through host temps
// (synchronous; these do not occur inside graph capture today).
void CudaComputeOps::scopy_fp32_to_fp16(const unsigned int N, const float *X,
                                        const unsigned int incX, _FP16 *Y,
                                        const unsigned int incY) {
  if (cuda::dev_only(X) || cuda::dev_only(Y)) {
    if (incX != 1 || incY != 1)
      throw std::runtime_error(
        "CudaComputeOps: strided converting copy on device-only memory");
    if (cuda::StreamManager::Global().isCapturing())
      std::fprintf(stderr,
                   "[CAP-AUDIT] converting scopy fp32->fp16 during capture: "
                   "N=%u (host convert frozen into graph)\n",
                   N);
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
void CudaComputeOps::scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                                        const unsigned int incX, float *Y,
                                        const unsigned int incY) {
  if (cuda::dev_only(X) || cuda::dev_only(Y)) {
    if (incX != 1 || incY != 1)
      throw std::runtime_error(
        "CudaComputeOps: strided converting copy on device-only memory");
    if (cuda::StreamManager::Global().isCapturing())
      std::fprintf(stderr,
                   "[CAP-AUDIT] converting scopy fp16->fp32 during capture: "
                   "N=%u (host convert frozen into graph)\n",
                   N);
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

// Load-time device-residency action, executed through the op-table prebuild
// seam: FullyConnectedLayerCl::read() calls w.getOps()->fc_prebuild_weight(w)
// per weight inside the parallel load worker, right after the weight bytes
// are read (skipped under FSU/opt_var). Only engine=cuda tensors resolve to
// this table, so no engine scan is needed to keep the call off gpu/cpu runs.
// Prebuild contract: a prebuild may create derived device state but must NOT
// invalidate the host payload -- cudaMemPrefetchAsync is a migration of the
// managed pages to the device, never an invalidation; the pointer stays
// host-accessible.
void CudaComputeOps::fc_prebuild_weight(Tensor &w) {
  if (w.getDataType() != ml::train::TensorDim::DataType::QS4CX)
    return;
  // NNTR_CUDA_WPREFETCH >= 2 opts in; unset -> 0 (default off).
  static const int wpf = []() {
    const char *e = std::getenv("NNTR_CUDA_WPREFETCH");
    return e ? atoi(e) : 0;
  }();
  if (wpf < 2)
    return;
  // The primitive is self-guarding (cuda_fc_qint4.cpp): integrated GPU ->
  // false, non-managed pointer -> false, and it computes its own byte extent
  // (the N*(K+1)/2 nibble payload + the N*4 fp32 scale tail).
  (void)cuda::cuda_fc_qs4cx_prefetch_weight(w.getData<uint8_t>(), w.width(),
                                            w.height());
}

ComputeOps *get_cuda_ops() {
  static CudaComputeOps instance;
  return &instance;
}

} // namespace nntrainer
