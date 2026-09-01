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

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

namespace nntrainer::cuda {

namespace {

/**
 * @brief The CUDA state that must be ONE per process, not one per module.
 *
 * See StreamManager::initialize() for why "per module" is the default here and
 * what it breaks. Three fields, and each is shared for the same reason: the
 * M2-B capture records ONE stream, the capture flag decides whether a guard
 * fires, and the decode-position buffer is the fixed device address the
 * captured RoPE/attention/KV nodes were recorded against.
 */
struct SharedCudaState {
  cudaStream_t stream = nullptr;
  int capturing = 0;
  int *pos_dev = nullptr; // device int[2] {pos, n_kv}
};

#ifdef _WIN32
/**
 * @brief Process-wide lock for the publish-or-adopt handshake below.
 *
 * A plain std::mutex cannot serialise this: it would itself be one lock per
 * module, which is precisely the problem being solved. The name carries the PID
 * so two nntrainer processes never wait on each other.
 */
class ProcLock {
public:
  ProcLock() {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "nntr_cuda_shared_state_%lu",
                  (unsigned long)GetCurrentProcessId());
    h_ = CreateMutexA(nullptr, FALSE, nm);
    if (h_ != nullptr)
      WaitForSingleObject(h_, INFINITE);
  }
  ~ProcLock() {
    if (h_ != nullptr) {
      ReleaseMutex(h_);
      CloseHandle(h_);
    }
  }

private:
  HANDLE h_ = nullptr;
};
#else
// POSIX: the layer plugins link nntrainer DYNAMICALLY and ELF symbol
// interposition already collapses these singletons to one copy, so there is
// nothing to hand over -- keep the plain per-process object and no lock.
class ProcLock {};
#endif

/**
 * @brief The one SharedCudaState of this process, found by every module.
 *
 * The Win32 environment block is the lookup table used deliberately: it is
 * per-process, and every module in this process binds the SAME UCRT, so it is
 * the one place they can all read without importing anything from each other.
 * The value is the address of a heap object -- all modules share one address
 * space and one UCRT heap, so a raw pointer is all that has to travel.
 */
SharedCudaState *shared_cuda_state() {
  static SharedCudaState *s = []() -> SharedCudaState * {
#ifdef _WIN32
    static const char *KEY = "__NNTR_CUDA_SHARED_STATE";
    ProcLock lk;
    char buf[32] = {0};
    if (GetEnvironmentVariableA(KEY, buf, (DWORD)sizeof(buf)) > 0) {
      const auto v = (uintptr_t)_strtoui64(buf, nullptr, 16);
      if (v != 0)
        return reinterpret_cast<SharedCudaState *>(v);
    }
    auto *p = new SharedCudaState();
    char out[32];
    std::snprintf(out, sizeof(out), "%llx", (unsigned long long)(uintptr_t)p);
    SetEnvironmentVariableA(KEY, out);
    return p;
#else
    static SharedCudaState local;
    return &local;
#endif
  }();
  return s;
}

} // namespace

void StreamManager::initialize() noexcept {
  // make sure the device + primary context exist before creating a stream
  ContextManager::Global().EnsureCurrent();
  auto *sh = shared_cuda_state();
  capture_flag_ = &sh->capturing;
  {
    ProcLock lk;
    if (sh->stream != nullptr) {
      stream_ = sh->stream; // adopt: another module published first
      owns_stream_ = false;
      return;
    }
    if (!cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate")) {
      stream_ = nullptr;
      return;
    }
    sh->stream = stream_;
    owns_stream_ = true;
  }
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
  // Counted before the launch so a caller that stamps dispatchSeq() AFTER its
  // own dispatches sees a value no other dispatch can reproduce.
  ++dispatch_seq_;
  auto params = kernel.getKernelParams();
  CUresult r = cuLaunchKernel(
    kernel.GetFunction(), (unsigned)grid[0], (unsigned)grid[1],
    (unsigned)grid[2], (unsigned)block[0], (unsigned)block[1],
    (unsigned)block[2], shared_bytes, reinterpret_cast<CUstream>(stream_),
    params.empty() ? nullptr : params.data(), nullptr);
  return cuCheck(r, "cuLaunchKernel");
}

// Whether to log the diagnostics around graph capture. Off unless asked for.
static bool graph_debug() {
  static const bool v = std::getenv("NNTR_CUDA_GRAPH_DBG") != nullptr;
  return v;
}

void StreamManager::finish() {
  if (isCapturing()) {
    // An in-capture cudaStreamSynchronize is illegal; the drain is deferred to
    // after the graph replay. A host read that depended on this drain now
    // consumes stale bytes, which is a bug in the caller rather than here, so
    // the skip is loggable under NNTR_CUDA_GRAPH_DBG.
    if (graph_debug()) {
      static int n = 0;
      if (++n <= 32)
        std::fprintf(stderr,
                     "[cuda] finish() skipped during graph capture (#%d)\n", n);
    }
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
  if (isCapturing())
    return;
  // NNTR_CUDA_PACE=<N> (default off): depth-N submission pacing -- the middle
  // ground between the full per-op drain (sync mode; WDDM decode ~29 TPS) and
  // no drain at all (ASYNC/DRAINSKIP), which is fast but corrupts on WDDM.
  // Bounds the un-drained op window to N by waiting the (i-N)th
  // op's event instead of draining op i. Host-read boundaries still use the
  // full finish(), so correctness of host consumption is unchanged. If the
  // WDDM corruption scales with N, the driver chokes on deep unpaced queues
  // (pacing = fix); if even N=4 corrupts while full drain is clean, the defect
  // is at kernel-boundary granularity (driver bug class).
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
  if (isCapturing()) {
    // Same as finish(): callers of finishIfAsync are host-fallback preambles,
    // so a hit during capture means a host op ran inside the graph.
    if (graph_debug()) {
      static int n = 0;
      if (++n <= 32)
        std::fprintf(
          stderr, "[cuda] finishIfAsync() skipped during graph capture (#%d)\n",
          n);
    }
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
  *capture_flag_ = 1;
  return true;
}

bool StreamManager::endCapture(cudaGraph_t *graph) {
  *capture_flag_ = 0;
  if (!stream_ || graph == nullptr)
    return false;
  return cudaCheck(cudaStreamEndCapture(stream_, graph),
                   "cudaStreamEndCapture");
}

StreamManager::~StreamManager() {
  // Only the module that CREATED the process stream destroys it; the adopters
  // merely borrowed the handle (see initialize()). Destroying it from each of
  // them would hand the survivors a dangling stream during static teardown,
  // whose order across modules is unspecified.
  if (stream_ && owns_stream_) {
    shared_cuda_state()->stream = nullptr;
    cudaStreamDestroy(stream_);
  }
  stream_ = nullptr;
}

StreamManager &StreamManager::Global() {
  // Out-of-line + intentionally leaked (see header note): never destroyed,
  // so ~StreamManager (cudaStreamDestroy) never runs at process exit
  // (2026-07-20 field crash fix, same convention as ClContext).
  static StreamManager *instance = new StreamManager();
  instance->initializeOnce();
  return *instance;
}

int *cuda_pos_buffer() {
  // Process-wide, for the same reason as the stream: the captured RoPE /
  // attention / KV-write nodes bake THIS device address in, and the scaffold
  // that rewrites the 8 bytes per token (cuda_set_pos, called from the CUDA
  // context) lives in a different module from the kernels that read them
  // (mha_core's DLL). A per-module buffer left every replay reading the
  // position frozen at capture time.
  auto *sh = shared_cuda_state();
  if (sh->pos_dev == nullptr) {
    ProcLock lk;
    if (sh->pos_dev == nullptr) {
      int *p = nullptr;
      if (cudaMalloc((void **)&p, 2 * sizeof(int)) == cudaSuccess)
        sh->pos_dev = p;
      else
        cudaGetLastError();
    }
  }
  return sh->pos_dev;
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
