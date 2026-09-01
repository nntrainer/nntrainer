// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA device/context management implementation.
 */

#include "cuda_context_manager.h"
#include "cuda_common.h"
#include "cuda_stream_manager.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#ifdef _WIN32
#include <delayimp.h>
#include <windows.h>
#endif

namespace nntrainer::cuda {

#ifdef _WIN32
// [delay-load failure hook] Without this, a missing CUDA runtime DLL (the
// binary /DELAYLOAD-imports cublas64_13.dll, cublasLt64_13.dll,
// nvrtc64_130_0.dll and nvcuda.dll -- see meson.build cuda_delayload_args)
// makes the FIRST call through that import raise an MSVC delay-load SEH
// exception
// (0xC06D007E) instead of a normal error return. Unhandled, that looks like
// a silent hang/crash rather than a diagnosable failure. Installing
// __pfnDliFailureHook2 intercepts dliFailLoadLib/dliFailGetProc so we can
// print one clear message naming the DLL before the loader's own error path
// runs.
static bool is_cuda_delayload_dll(const char *dll) {
  static const char *kOurs[] = {"cublas64_13.dll", "cublasLt64_13.dll",
                                "nvrtc64_130_0.dll", "nvcuda.dll"};
  for (const char *n : kOurs)
    if (_stricmp(dll, n) == 0)
      return true;
  return false;
}

static FARPROC WINAPI nntr_dli_failure_hook(unsigned dliNotify,
                                            PDelayLoadInfo pdli) {
  const char *dll = (pdli && pdli->szDll) ? pdli->szDll : "(unknown)";
  const char *verb = dliNotify == dliFailGetProc ? "resolved" : "loaded";
  // __pfnDliFailureHook2 is per MODULE, not per DLL: once linked in it sees
  // the failure of EVERY delay-loaded import in this binary, including ones
  // that have nothing to do with CUDA and including runs on another engine.
  // Decline those -- returning nullptr leaves the loader's own error path
  // intact, which is what an embedding application can actually handle.
  if (!is_cuda_delayload_dll(dll))
    return nullptr;
  // nvcuda.dll ships with the NVIDIA driver itself (not the CUDA toolkit),
  // so its absence means "no NVIDIA GPU / no driver" rather than "missing
  // redistributable DLL" -- point the user at the right fix.
  if (_stricmp(dll, "nvcuda.dll") == 0) {
    fprintf(stderr,
            "[NNTRAINER][ERROR] %s could not be %s: NVIDIA driver not "
            "installed or no NVIDIA GPU -- the cuda backend needs an "
            "NVIDIA GPU with a current driver\n",
            dll, verb);
  } else {
    fprintf(stderr,
            "[NNTRAINER][ERROR] CUDA runtime DLL could not be %s: %s\n"
            "  The CUDA backend needs the NVIDIA driver plus the CUDA runtime "
            "DLLs shipped in the SDK bin/ directory\n"
            "  (cudart64_13.dll, cublas64_13.dll, cublasLt64_13.dll, "
            "nvrtc64_130_0.dll, nvrtc-builtins64_133.dll).\n"
            "  Keep them next to the executable or on PATH.\n",
            verb, dll);
  }
  fflush(stderr);
  // Do NOT terminate. This is a library: killing the host process denies an
  // embedding application any chance to report the failure or fall back to
  // another engine. Returning nullptr lets the delay-load machinery raise its
  // own structured exception, which the caller can handle -- and the message
  // above is what makes that failure diagnosable rather than a silent crash,
  // which was the whole reason for installing this hook.
  return nullptr;
}
extern "C" const PfnDliHook __pfnDliFailureHook2 = nntr_dli_failure_hook;
#endif

void ContextManager::initialize() noexcept {
  initialized_ok_ = CreateDefaultGPUDevice();
#ifdef _WIN32
  // [W2 delay-load follow-up] The unified binary delay-loads cuBLAS/NVRTC
  // (meson cuda_delayload_args) so XMX runs never map their DLL images. On
  // CUDA runs the deferred LoadLibrary then landed inside the first in-forward
  // call, i.e. mid-prefill: +67ms inside the timed 1K window (measured
  // 4210 -> 3300 TPS; clocks were flat P0/2805MHz, so the "sustained
  // clock band" reading was this load tax, not a clock state). Pull the DLLs
  // in at context bring-up, outside any timed phase. NVRTC alone was not
  // enough (still 3330): the mid-prefill loader is the cuBLAS pair. LoadLibrary
  // maps without prefaulting, so most of the delay-load WS win survives —
  // measured on cuda-a2 right after this change.
  if (initialized_ok_) {
    if (!LoadLibraryA("nvrtc64_130_0.dll"))
      ml_logw("[CUDA] preload of nvrtc64_130_0.dll failed (err=%lu); the "
              "CUDA backend will fail hard on first use if it is still "
              "missing",
              GetLastError());
    if (!LoadLibraryA("cublas64_13.dll"))
      ml_logw("[CUDA] preload of cublas64_13.dll failed (err=%lu); the CUDA "
              "backend will fail hard on first use if it is still missing",
              GetLastError());
    if (!LoadLibraryA("cublasLt64_13.dll"))
      ml_logw("[CUDA] preload of cublasLt64_13.dll failed (err=%lu); the "
              "CUDA backend will fail hard on first use if it is still "
              "missing",
              GetLastError());
  }
#endif
}

bool ContextManager::CreateDefaultGPUDevice() {
  if (!cuCheck(cuInit(0), "cuInit"))
    return false;

  int count = 0;
  if (!cuCheck(cuDeviceGetCount(&count), "cuDeviceGetCount") || count == 0) {
    ml_loge("[CUDA] no CUDA-capable device found");
    return false;
  }

  device_ordinal_ = 0;
  if (const char *e = getenv("NNTR_CUDA_DEVICE"))
    device_ordinal_ = atoi(e);
  if (device_ordinal_ < 0 || device_ordinal_ >= count)
    device_ordinal_ = 0;

  // bind the Runtime API to this device (creates/uses its primary context) ...
  if (!cudaCheck(cudaSetDevice(device_ordinal_), "cudaSetDevice"))
    return false;

  // ... and retain the SAME primary context for the Driver API so module loads
  // and kernel launches share allocations made through the Runtime API.
  if (!cuCheck(cuDeviceGet(&device_, device_ordinal_), "cuDeviceGet"))
    return false;
  if (!cuCheck(cuDevicePrimaryCtxRetain(&context_, device_),
               "cuDevicePrimaryCtxRetain"))
    return false;
  if (!cuCheck(cuCtxSetCurrent(context_), "cuCtxSetCurrent"))
    return false;

  cudaDeviceProp prop{};
  if (cudaCheck(cudaGetDeviceProperties(&prop, device_ordinal_),
                "cudaGetDeviceProperties")) {
    device_name_ = prop.name;
    cc_major_ = prop.major;
    cc_minor_ = prop.minor;
    // Integrated GPU (Tegra/Jetson Orin): host+device share one physical memory
    // pool. prop.integrated is 1 there, 0 on discrete GPUs (RTX4070). This bit
    // gates every "is this discrete VRAM?" residency assumption (device-only
    // activation pool, KV mirror copies, MemAdvise device-pin) so the same
    // binary stays coherent on both -- see isIntegrated().
    integrated_ = prop.integrated != 0;
    // Windows WDDM reports 0 here (Linux discrete: 1). Everything that lets a
    // host thread touch managed memory while kernels may be in flight -- async
    // submission, and by extension the discrete "FAST" env add-ons tuned
    // around it -- presumes 1, so the profile gates consult this bit.
    int cma = 0;
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess,
                           device_ordinal_);
    concurrent_managed_access_ = cma != 0;
  }
  cudaDriverGetVersion(&driver_version_);

  ml_logi("[CUDA] device %d: %s (sm_%d%d, driver %d, %.1f GiB, %s)",
          device_ordinal_, device_name_.c_str(), cc_major_, cc_minor_,
          driver_version_, prop.totalGlobalMem / 1073741824.0,
          integrated_ ? "integrated" : "discrete");

  // NNTR_CUDA_DBG: a VISIBLE (stderr, logger-independent) dump of the residency
  // facts the GPU-vs-host dispatch gates depend on. On Tegra/Orin the critical
  // unknown is whether cudaMallocManaged memory reports as
  // cudaMemoryTypeManaged
  // (==2) -- if it instead reports Host(1)/Unregistered(0), every
  // dev()/dev_ok() residency gate fails and the GPU ops silently fall to the
  // host => deterministic garbage + low GPU% + slow. This self-probe prints the
  // actual type so that hypothesis is confirmable in one run.
  if (std::getenv("NNTR_CUDA_DBG") != nullptr) {
    int cma = 0, pma = 0;
    cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess,
                           device_ordinal_);
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess,
                           device_ordinal_);
    int mtype = -2, dtype = -2;
    void *mp = nullptr, *dp = nullptr;
    if (cudaMallocManaged(&mp, 256) == cudaSuccess && mp) {
      cudaPointerAttributes a{};
      if (cudaPointerGetAttributes(&a, mp) == cudaSuccess)
        mtype = (int)a.type;
      cudaFree(mp);
    }
    if (cudaMalloc(&dp, 256) == cudaSuccess && dp) {
      cudaPointerAttributes a{};
      if (cudaPointerGetAttributes(&a, dp) == cudaSuccess)
        dtype = (int)a.type;
      cudaFree(dp);
    }
    cudaGetLastError();
    std::fprintf(
      stderr,
      "[CUDA-DBG] %s sm_%d%d integrated=%d concurrentManagedAccess=%d "
      "pageableMemoryAccess=%d | cudaPointerGetAttributes.type: "
      "managed=%d device=%d (expect managed==2 device==2; "
      "type enum 0=unreg 1=host 2=device... NOTE managed reports as "
      "type 2/Device OR 3 depending on driver -- gates accept 2&3)\n",
      device_name_.c_str(), cc_major_, cc_minor_, (int)integrated_, cma, pma,
      mtype, dtype);
    std::fflush(stderr);
  }
  return true;
}

void ContextManager::EnsureCurrent() {
  if (context_)
    cuCtxSetCurrent(context_);
}

std::string ContextManager::GetDeviceSignature() const {
  return device_name_ + "|drv" + std::to_string(driver_version_) + "|sm_" +
         std::to_string(cc_major_) + std::to_string(cc_minor_);
}

std::string ContextManager::GetComputeArch() const {
  return "compute_" + std::to_string(cc_major_) + std::to_string(cc_minor_);
}

ContextManager::~ContextManager() {
  if (context_) {
    cuDevicePrimaryCtxRelease(device_);
    context_ = nullptr;
  }
}

ContextManager &ContextManager::Global() {
  // Out-of-line + intentionally leaked (see header note): never destroyed,
  // so ~ContextManager (cuDevicePrimaryCtxRelease) never runs at process
  // exit (2026-07-20 field crash fix, same convention as ClContext).
  static ContextManager *instance = new ContextManager();
  instance->initializeOnce();
  return *instance;
}

bool engine_selected() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_ENGINE");
    return e != nullptr && std::string(e) == "cuda";
  }();
  return on;
}

void drain_if_async() {
  if (!engine_selected())
    return;
  StreamManager::Global().finishIfAsync();
}

bool dev_accessible(const void *p) {
  if (!engine_selected())
    return false;
  if (p == nullptr)
    return false;
  cudaPointerAttributes a{};
  const bool ok = cudaPointerGetAttributes(&a, p) == cudaSuccess;
  cudaGetLastError(); // clear the benign "invalid pointer" state for host ptrs
  if (!ok)
    return false;
  if (a.type == cudaMemoryTypeManaged || a.type == cudaMemoryTypeDevice)
    return true;
  // Pinned host-mapped (zero-copy) pool: reports Host but carries a valid
  // devicePointer (UVA) -- kernel-reachable on ANY GPU. This is the cMA==0
  // (WDDM) replacement for the incoherent managed pool.
  if (a.type == cudaMemoryTypeHost && a.devicePointer != nullptr)
    return true;
  // Integrated GPU (Tegra/Orin): managed memory may report as Host, but it is
  // GPU-accessible (shared physical pool) -- accept so the kernel engages.
  if (a.type == cudaMemoryTypeHost && ContextManager::Global().isIntegrated())
    return true;
  return false;
}

bool dev_only(const void *p) {
  if (!engine_selected())
    return false;
  if (p == nullptr)
    return false;
  cudaPointerAttributes a{};
  const bool ok = cudaPointerGetAttributes(&a, p) == cudaSuccess;
  cudaGetLastError();
  return ok && a.type == cudaMemoryTypeDevice;
}

bool device_memset0(void *p, size_t bytes) {
  if (p == nullptr || bytes == 0)
    return false;
  const bool ok = cudaMemset(p, 0, bytes) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

bool copy_any(void *dst, const void *src, size_t bytes) {
  if (dst == nullptr || src == nullptr || bytes == 0)
    return false;
  // cudaMemcpyDefault resolves H2D/D2H/D2D/H2H from the pointer attributes;
  // synchronous, so the caller may immediately consume the destination.
  const bool ok = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) == cudaSuccess;
  cudaGetLastError();
  return ok;
}

} // namespace nntrainer::cuda
