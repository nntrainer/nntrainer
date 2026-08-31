// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for context/device management. Peer of
 *          nntrainer::opencl::ContextManager. Retains the device PRIMARY
 * context so the Driver API (cuModuleLoad/cuLaunchKernel) and the Runtime API
 *          (cudaMalloc/cudaMemcpy) share one context.
 */

#ifndef __CUDA_CONTEXT_MANAGER_H__
#define __CUDA_CONTEXT_MANAGER_H__

#include <string>

#include <cuda.h>

#include "singleton.h"

namespace nntrainer::cuda {

/**
 * @class ContextManager
 * @brief Singleton wrapper around the selected CUDA device + primary context.
 */
class ContextManager : public Singleton<ContextManager> {
public:
  /**
   * @brief true if the device + primary context were created successfully
   */
  bool isAvailable() const { return initialized_ok_; }

  /**
   * @brief Get the active primary CUDA context
   */
  CUcontext GetContext() const { return context_; }

  /**
   * @brief Get the active CUDA device handle
   */
  CUdevice GetDevice() const { return device_; }

  /**
   * @brief Get the active device ordinal (cudaSetDevice index)
   */
  int GetDeviceOrdinal() const { return device_ordinal_; }

  /**
   * @brief Get the active device name (e.g. "NVIDIA GeForce RTX 4070 Laptop
   * GPU")
   */
  const std::string &GetDeviceName() const { return device_name_; }

  /**
   * @brief true if the device is an INTEGRATED GPU (Tegra/Jetson Orin etc.)
   *        where host and device share one physical memory pool. On such
   *        devices the discrete-GPU residency tricks (device-only cudaMalloc
   *        activation pool, KV mirror copies, MemAdvise device-pin) give no
   *        bandwidth benefit and BREAK host-coherence -- callers gate those
   *        off when this returns true. Read from cudaDevAttrIntegrated once.
   */
  bool isIntegrated() const { return integrated_; }

  /**
   * @brief true when the device supports CONCURRENT host access to managed
   *        memory while kernels are in flight
   * (cudaDevAttrConcurrentManagedAccess). Linux discrete GPUs: 1. Windows WDDM:
   * 0 -- there, host access to ANY managed allocation is only legal while the
   * device is idle (pre-Pascal model), so async submission and the discrete env
   *        add-ons tuned around cMA=1 must be gated off.
   */
  bool concurrentManagedAccess() const { return concurrent_managed_access_; }

  /**
   * @brief Stable signature used to key the on-disk PTX cache so a module built
   *        for a different GPU / driver / arch is never loaded.
   * @return "<name>|drv<driver>|sm_<cc>"
   */
  std::string GetDeviceSignature() const;

  /**
   * @brief NVRTC --gpu-architecture target for this device, e.g. "compute_89".
   */
  std::string GetComputeArch() const;

  /**
   * @brief Make the primary context current on the calling thread. Cheap; safe
   *        to call before any Driver-API op (module load / kernel launch).
   */
  void EnsureCurrent();

  /**
   * @brief Release the primary context.
   */
  ~ContextManager() override;

  /**
   * @brief Get the process-wide instance (out-of-line override of
   *        Singleton<T>::Global(), intentionally-leaked heap instance --
   *        never destroyed).
   * @note  Mirrors nntrainer::ClContext::Global() (cl_context.h): the 2026-
   *        07-20 field crash showed that a driver-release call inside a
   *        static-teardown-time destructor can abort when the GPU driver's
   *        worker threads are already dead (Windows DLL_PROCESS_DETACH,
   *        observed on the Intel OpenCL path). ~ContextManager calls
   *        cuDevicePrimaryCtxRelease; leaking the singleton keeps that call
   *        out of CRT teardown for CUDA too.
   */
  static ContextManager &Global();

protected:
  /**
   * @brief Singleton hook: create device + primary context once.
   */
  void initialize() noexcept override;

private:
  bool CreateDefaultGPUDevice();

  CUdevice device_{0};
  int device_ordinal_{0};
  CUcontext context_{nullptr};
  std::string device_name_;
  int cc_major_{0};
  int cc_minor_{0};
  int driver_version_{0};
  bool integrated_{false};
  bool concurrent_managed_access_{true};
  bool initialized_ok_{false};
};

/**
 * @brief True iff this run selected the CUDA engine (NNTR_ENGINE=cuda,
 *        static-cached). The residency probes below and every shared-layer
 *        CUDA touch must short-circuit on this: cudart is statically linked,
 *        so on a non-cuda run of the unified binary the first cudart call
 *        boots the runtime (LoadLibrary nvcuda + driver init) wherever it
 *        lands — with the engine-gated context bring-up (engine.cpp) no longer
 *        hiding it at startup, that was the first forward = inside the timed
 *        prefill window (measured -55% XMX prefill).
 */
bool engine_selected();

/**
 * @brief Engine-gated stream drain for the host-fallback paths of SHARED
 *        layers (residual add, slicing, norms, logits readback): no-op unless
 *        the cuda engine is selected. Calling
 *        StreamManager::Global().finishIfAsync() directly there CREATES the
 *        CUDA stream/context on a non-cuda run — use this instead.
 */
void drain_if_async();

/**
 * @brief  Is pointer @p p reachable by a CUDA kernel? Accepts Managed/Device
 *         always; on an INTEGRATED GPU (Tegra/Orin) also accepts Host, because
 *         there cudaMallocManaged memory can report as cudaMemoryTypeHost yet
 * is GPU-accessible (one shared physical pool). Without this every dev() gate
 * rejects the (managed) activation pool on Orin and the GPU ops silently fall
 * to the host => correct-but-slow (2 TPS). Single source of truth for the
 * residency gates. Always false when the cuda engine was not selected (see
 * engine_selected()).
 */
bool dev_accessible(const void *p);

/**
 * @brief True iff the pointer is DEVICE-ONLY memory (cudaMalloc): the host
 *        cannot dereference it, so every host read/write must stage. False for
 *        managed/pinned/host memory. Companion of dev_accessible() for the
 *        device-resident (NNTR_CUDA_DEV_ACT / NNTR_CUDA_KV_DEV) pools -- the
 *        gate for "auto-route to the GPU kernel / staged copy instead of the
 *        host fallback" decisions.
 */
bool dev_only(const void *p);

/**
 * @brief Staged-copy + memset helpers for device-only pool tensors, so app
 *        code (KV manager, logits readback) does not include cuda_runtime.
 *        All return false on failure (caller keeps its host path).
 */
bool device_memset0(void *p, size_t bytes);
bool copy_any(void *dst, const void *src, size_t bytes); // cudaMemcpyDefault

} // namespace nntrainer::cuda

#endif // __CUDA_CONTEXT_MANAGER_H__
