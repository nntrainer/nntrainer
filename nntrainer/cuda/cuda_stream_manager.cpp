// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_stream_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA stream/dispatch management implementation.
 */

#include "cuda_stream_manager.h"
#include "cuda_common.h"
#include "cuda_context_manager.h"
#include "cuda_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <env_compat.h>

namespace nntrainer::cuda {

// [CAP-AUDIT] prints are opt-in diagnostics (NNTR_CUDA_CAP_AUDIT=1): with the
// M2-B decode graph default-ON they fire on every capture (one per shared-
// layer drain preamble, ~68 lines/run) and read as errors to SDK consumers,
// while in sync mode (ASYNC off, the Windows default) the skipped drains are
// no-ops to begin with. Correctness is covered by the replay validations
// (byte-identical vs the sync path, 6-run determinism); flip the env on when
// hunting a NEW capture-time host-fallback hazard.
static bool cap_audit_on() {
  static const bool on = nntr_env_on("NNTR_CUDA_CAP_AUDIT");
  return on;
}

void StreamManager::initialize() noexcept {
  // make sure the device + primary context exist before creating a stream
  ContextManager::Global().EnsureCurrent();
  if (!cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate"))
    stream_ = nullptr;
}

bool StreamManager::EnqueueWriteBuffer(void *dst_dev, size_t size,
                                       const void *src_host, bool async) {
  if (!cudaCheck(cudaMemcpyAsync(dst_dev, src_host, size,
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync H2D"))
    return false;
  if (!async)
    return cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

bool StreamManager::EnqueueReadBuffer(const void *src_dev, size_t size,
                                      void *dst_host, bool async) {
  if (!cudaCheck(cudaMemcpyAsync(dst_host, src_dev, size,
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync D2H"))
    return false;
  if (!async)
    return cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return true;
}

bool StreamManager::DispatchCommand(Kernel &kernel, const int (&grid)[3],
                                    const int (&block)[3],
                                    unsigned int shared_bytes) {
  if (!kernel.valid()) {
    ml_loge("[CUDA] DispatchCommand: invalid kernel");
    return false;
  }
  ContextManager::Global().EnsureCurrent();
  auto params = kernel.getKernelParams();
  CUresult r = cuLaunchKernel(
    kernel.GetFunction(), (unsigned)grid[0], (unsigned)grid[1],
    (unsigned)grid[2], (unsigned)block[0], (unsigned)block[1],
    (unsigned)block[2], shared_bytes, reinterpret_cast<CUstream>(stream_),
    params.empty() ? nullptr : params.data(), nullptr);
  return cuCheck(r, "cuLaunchKernel");
}

void StreamManager::finish() {
  if (capturing_) { // an in-capture cudaStreamSynchronize is illegal; the drain
    // is deferred to after the graph replay (endCapture caller). A host read
    // that depended on this drain now consumes stale bytes -- audit-log the
    // skip so capture-time host fallbacks are visible.
    static int audit_n = 0;
    if (++audit_n <= 32 && cap_audit_on())
      std::fprintf(
        stderr, "[CAP-AUDIT] finish() skipped during capture (#%d)\n", audit_n);
    return;
  }
  if (stream_) {
    cudaStreamSynchronize(stream_);
    // concurrentManagedAccess==0 (Windows WDDM / pre-Pascal model) device-sync
    // add-on. HISTORY: added when host reads of kernel-written managed pages
    // appeared stale on WDDM -- but the actual culprit turned out to be the
    // unified-binary isSVM hijack (outputs were never written at all; see
    // CudaMemAllocator::isSVM). With that fixed, the stream-sync alone may be
    // sufficient (pre-Pascal launch migration + stream drain), and the per-op
    // cudaDeviceSynchronize goes through the WDDM OS scheduler = measurable
    // cost. The =0 variant was field-validated golden on the WDDM box (1K
    // 63.0/5.90 TPS, +7% decode vs devsync-on; pinned zero-copy pool), so the
    // DEFAULT IS OFF -- NNTR_CUDA_WDDM_DEVSYNC=1 re-arms the drain if a future
    // cMA==0 device shows a genuine post-kernel host-visibility gap.
    static const bool wddm_devsync = []() {
      const char *e = std::getenv("NNTR_CUDA_WDDM_DEVSYNC");
      return e != nullptr && e[0] == '1';
    }();
    if (wddm_devsync && !ContextManager::Global().concurrentManagedAccess())
      cudaDeviceSynchronize();
  }
}

static bool cuda_async_mode() {
  static const bool async = []() {
    const char *e = std::getenv("NNTR_CUDA_ASYNC");
    if (e == nullptr || e[0] != '1')
      return false;
    // Integrated GPU (Tegra/Jetson Orin): async drops the per-op stream drain,
    // but on the shared-memory iGPU there is no UVM page-fault ordering to
    // order a host read against an in-flight kernel write -> the host fallbacks
    // read half-written buffers = corrupted tokens. Force SYNC on integrated
    // regardless of the env (re-enable per-Orin only after a dedicated
    // coherence benchmark). Discrete GPUs honor NNTR_CUDA_ASYNC.
    return !ContextManager::Global().isIntegrated();
  }();
  return async;
}

void StreamManager::maybeFinish() {
  if (capturing_)
    return;
  // NNTR_CUDA_PACE=<N> (default off): depth-N submission pacing -- the middle
  // ground between the full per-op drain (sync mode; WDDM decode ~29 TPS) and
  // no drain at all (no-drain modes corrupt on WDDM). Bounds the un-drained op
  // window to N by waiting the (i-N)th op's event instead of draining op i.
  // Host-read boundaries still use the full finish(), so correctness of host
  // consumption is unchanged. If the WDDM corruption scales with N, the driver
  // chokes on deep unpaced queues (pacing = fix); if even N=4 corrupts while
  // full drain is clean, the defect is at kernel-boundary granularity (driver
  // bug class).
  static const int pace_n = []() {
    const char *e = std::getenv("NNTR_CUDA_PACE");
    const int v = e ? std::atoi(e) : 0;
    return v > 120 ? 120 : v; // ring headroom
  }();
  if (pace_n > 0 && stream_) {
    constexpr int RING = 128;
    static cudaEvent_t ring[RING] = {};
    static unsigned long long idx = 0;
    const int slot = (int)(idx % RING);
    if (ring[slot] == nullptr)
      cudaEventCreateWithFlags(&ring[slot], cudaEventDisableTiming);
    cudaEventRecord(ring[slot], stream_);
    if (idx >= (unsigned long long)pace_n) {
      const int wslot = (int)((idx - (unsigned long long)pace_n) % RING);
      if (ring[wslot])
        cudaEventSynchronize(ring[wslot]);
    }
    ++idx;
    return;
  }
  if (!cuda_async_mode())
    finish();
}

void StreamManager::finishIfAsync() {
  if (capturing_) {
    // Same audit as finish(): callers of finishIfAsync are host-fallback
    // preambles -- a hit during capture means a host op ran inside the graph.
    static int audit_n = 0;
    if (++audit_n <= 32 && cap_audit_on())
      std::fprintf(stderr,
                   "[CAP-AUDIT] finishIfAsync() skipped during capture (#%d)\n",
                   audit_n);
    return;
  }
  if (cuda_async_mode())
    finish();
}

bool StreamManager::beginCapture() {
  if (!stream_)
    return false;
  cudaStreamSynchronize(stream_); // drain pre-capture work; start from idle
  if (!cudaCheck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed),
                 "cudaStreamBeginCapture"))
    return false;
  capturing_ = true;
  return true;
}

bool StreamManager::endCapture(cudaGraph_t *graph) {
  capturing_ = false;
  if (!stream_ || graph == nullptr)
    return false;
  return cudaCheck(cudaStreamEndCapture(stream_, graph),
                   "cudaStreamEndCapture");
}

StreamManager::~StreamManager() {
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

int *cuda_pos_buffer() {
  static int *g_pos_dev = []() -> int * {
    int *p = nullptr;
    cudaMalloc((void **)&p, 2 * sizeof(int));
    return p;
  }();
  return g_pos_dev;
}

void cuda_set_pos(int pos, int n_kv) {
  // Pinned host source so the H2D is a real async DMA (also keeps it capturable
  // should it ever be issued inside a capture). The copy is on the backend
  // stream, so it is ordered before a subsequent cudaGraphLaunch on the same
  // stream -- the replayed kernels read the fresh pos.
  static int *g_pos_host = []() -> int * {
    int *p = nullptr;
    cudaHostAlloc((void **)&p, 2 * sizeof(int), cudaHostAllocDefault);
    return p;
  }();
  int *d = cuda_pos_buffer();
  if (!d || !g_pos_host)
    return;
  g_pos_host[0] = pos;
  g_pos_host[1] = n_kv;
  cudaMemcpyAsync(d, g_pos_host, 2 * sizeof(int), cudaMemcpyHostToDevice,
                  StreamManager::Global().GetStream());
}

} // namespace nntrainer::cuda
