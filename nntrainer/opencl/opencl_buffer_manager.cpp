// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_buffer_manager.cpp
 * @date    01 Dec 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#include <cstring>
#include <mutex>

#include <opencl_buffer_manager.h>
#include <opencl_loader.h>

namespace nntrainer {

namespace {
// Guards the lazy allocation below (getters may be hit from worker threads).
std::mutex &bm_mtx() {
  static std::mutex m;
  return m;
}
} // namespace

void ClBufferManager::initBuffers() {
  // [lazy] Allocation is deferred to first use (the getSVM*/getBuffer
  // getters). The eager version allocated data_input + max_qs x
  // (scale+quant+output) = 204.5MB of SVM here, and initBuffers() runs once
  // per ClContext init -- which on Windows means once per DLL MODULE:
  // Singleton<T> is a function-local static, so the exe + every layer/model
  // DLL owns a private instance. With several modules loaded in one process,
  // each instance eagerly reserving 204.5MB of SVM adds up to a large
  // resident, ALL-ZERO (never touched) working set -- these pools serve only
  // the GGML Q4_0/Q6_K CL path, which not every model calls. Lazy allocation
  // makes unused instances cost 0 bytes and used ones materialize only in
  // the one module that actually runs the GGML path. (Linux never showed
  // this: shared .so = one process-wide singleton.)
}

opencl::Buffer *ClBufferManager::getInBufferA() {
  if (inBufferA == nullptr) {
    inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferA;
}

opencl::Buffer *ClBufferManager::getInBufferB() {
  if (inBufferB == nullptr) {
    inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferB;
}

opencl::Buffer *ClBufferManager::getInBufferC() {
  if (inBufferC == nullptr) {
    inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferC;
}

opencl::Buffer *ClBufferManager::getOutBufferA() {
  if (outBufferA == nullptr) {
    outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferA;
}

opencl::Buffer *ClBufferManager::getOutBufferB() {
  if (outBufferB == nullptr) {
    outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferB;
}

void *ClBufferManager::getSVMInput() {
  std::lock_guard<std::mutex> lk(bm_mtx());
  if (data_input == nullptr)
    data_input = context_inst_.createSVMRegion(buffer_size_bytes);
  return data_input;
}

void *ClBufferManager::getSVMScale(unsigned int idx) {
  if (idx >= max_qs)
    return nullptr;
  std::lock_guard<std::mutex> lk(bm_mtx());
  while (scale_vec.size() <= idx)
    scale_vec.push_back(context_inst_.createSVMRegion(scale_q4_0_size));
  return scale_vec[idx];
}

void *ClBufferManager::getSVMQuant(unsigned int idx) {
  if (idx >= max_qs)
    return nullptr;
  std::lock_guard<std::mutex> lk(bm_mtx());
  while (quant_vec.size() <= idx)
    quant_vec.push_back(context_inst_.createSVMRegion(quant_q4_0_size));
  return quant_vec[idx];
}

void *ClBufferManager::getSVMOutput(unsigned int idx) {
  if (idx >= max_qs)
    return nullptr;
  std::lock_guard<std::mutex> lk(bm_mtx());
  while (output_vec.size() <= idx)
    output_vec.push_back(context_inst_.createSVMRegion(buffer_size_bytes));
  return output_vec[idx];
}

ClBufferManager &ClBufferManager::Global() {
  // Out-of-line on purpose — single process-wide instance (see header note).
  static ClBufferManager instance;
  instance.initializeOnce();
  return instance;
}

ClBufferManager::~ClBufferManager() {
  /** Intentionally a no-op (leak at process exit).
   *
   * The singleton lives in a function-local static
   * (Singleton<T>::Global), so this destructor only ever runs from
   * __cxa_finalize at process teardown. By then the Adreno user-mode
   * driver has already run its own finalizers, and releasing CL
   * resources here (clSVMFree via releaseSVMRegion, clReleaseMemObject
   * via the Buffer deletes) null-derefs inside libgsl
   * (gsl_memory_free_pure): every on-device CausalLM run exited with
   * SIGSEGV after printing its results, and the crash raced the stdio
   * flush, randomly truncating redirected output. The OS reclaims all
   * GPU memory at process death, so the right move is to not touch the
   * driver at all. (Trade-off: a dlclose() of the library mid-process
   * would now leak these regions until exit; nntrainer is not used that
   * way.)
   */
}

} // namespace nntrainer
