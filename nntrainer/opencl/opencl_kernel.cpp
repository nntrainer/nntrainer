// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#include "opencl_kernel.h"

#include "opencl_loader.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include <nntrainer_log.h>

namespace nntrainer::opencl {

// Set whenever a shared-virtual-memory pointer is bound as a kernel argument,
// read and cleared by CommandQueueManager when the kernel is enqueued, so the
// coherence drain runs only after a dispatch that actually touched shared
// memory. The dispatch path is single threaded -- a kernel is fully bound and
// then enqueued before the next one is bound -- so a file-scope flag is enough.
static bool s_bind_touched_svm = false;

/**
 * @brief Create a Kernel From Program object
 *
 * @param program
 * @param function_name the kernel string name
 * @return true if successful or false otherwise
 */
bool Kernel::CreateKernelFromProgram(Program program,
                                     const std::string &function_name) {
  int error_code;
  // get the OpenCL program
  cl_program prgm = program.GetProgram();

  // returns NULL with error code if fails
  kernel_ = clCreateKernel(prgm, function_name.c_str(), &error_code);
  if (!kernel_ || error_code != CL_SUCCESS) {
    kernel_ = nullptr;
    ml_loge("Failed to create %s. OpenCL error code: %d : %s",
            function_name.c_str(), error_code,
            OpenCLErrorCodeToString(error_code));
    return false;
  }
  // increments the program reference count.
  clRetainProgram(prgm);

  return true;
}

/**
 * @brief Set the Kernel Arguments
 *
 * @param arg_index index of the argument
 * @param arg_value value of the argument
 * @param size size of the argument
 * @return true if successful or false otherwise
 */
bool Kernel::SetKernelArguments(cl_uint arg_index, const void *arg_value,
                                size_t size) {
  int error_code;
  // returns NULL with error code if fails
  error_code = clSetKernelArg(kernel_, arg_index, size, arg_value);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to set argument: %u = %p. OpenCL error code: %d : %s",
            arg_index, arg_value, error_code,
            OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Set the Kernel Arguments
 *
 * @param arg_index index of the argument
 * @param arg_value value of the argument
 * @param size size of the argument
 * @return true if successful or false otherwise
 */
bool Kernel::SetKernelSVMArguments(cl_uint arg_index, const void *arg_value) {
  // This dispatch reads or writes shared virtual memory, so its
  // producer-to-consumer handoff needs an explicit flush on a device without
  // fine-grain coherence. The flag accumulates until the next dispatch
  // consumes it, so the order the arguments are bound in does not matter.
  s_bind_touched_svm = true;
  // Remember WHICH plane, not only that there was one: the hazard test below is
  // a containment question, and it is asked once per dispatch rather than once
  // per argument. Same bind-then-enqueue pairing, same thread.
  noteBoundSvmPointer(arg_value);
  int error_code;
  // returns NULL with error code if fails
  error_code = clSetKernelArgSVMPointer(kernel_, arg_index, arg_value);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to set argument. OpenCL error code: %d : %s", error_code,
            OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Get the Kernel object
 *
 * @return const cl_kernel
 */
const cl_kernel Kernel::GetKernel() { return kernel_; }

bool Kernel::takeDispatchTouchedSVM() {
  const bool touched = s_bind_touched_svm;
  s_bind_touched_svm = false;
  return touched;
}

// ---------------------------------------------------------------------------
// Undrained shared-memory planes: the risk condition, decided in code.
//
// A pack-level flag can record where a missing drain has been OBSERVED to
// decide a token, but the hazard itself is structural rather than per model: a
// consumer binding a shared-memory plane that an earlier, undrained submission
// wrote. This is that condition, asked where the chain is built.
//
// Two halves.
//   * A producer that writes a shared plane and only flushes says so
//     (noteUndrainedSvmPlane), naming the plane's byte range.
//   * Every dispatch that binds a shared pointer asks whether any bound pointer
//     lands inside an outstanding plane (takeDispatchSvmHazard). On a hit the
//     caller drains BEFORE enqueueing the consumer, and the set is cleared: one
//     clFinish retires every outstanding write, so the set has no reason to
//     keep the others.
//
// Why a range and not a pointer. Producer and consumer do not name the same
// address here. A prefill K rotation binds an offset-baked slice of the KV
// cache; the scatter that reads it binds the cache's stable base with the row
// offset as a kernel scalar (NNTR_KV_SCALAR_OFF, the recordable form). A
// pointer-equality test would miss exactly that pair.
//
// Cost. One pointer compare per bound shared argument against a set that is
// almost always empty or of size one or two, plus a clFinish exactly where a
// consumer meets an undrained write.
//
// NNTR_SVM_HAZARD_DRAIN=1 opts in; it is off by default (see below).
namespace {
constexpr int kSvmPlaneSlots = 8; // outstanding undrained planes
constexpr int kSvmBoundSlots = 8; // shared pointers bound for one dispatch
struct SvmPlane {
  const char *base = nullptr;
  const char *end = nullptr;
};
thread_local SvmPlane s_svm_planes[kSvmPlaneSlots];
thread_local int s_svm_plane_n = 0;
thread_local const void *s_bound_svm[kSvmBoundSlots];
thread_local int s_bound_svm_n = 0;
// Reported once at teardown, so it is process-wide rather than per-thread and
// counted only when someone asked for the ledger (NNTR_SVM_HAZARD_STAT).
std::atomic<unsigned long long> s_svm_hazard_drains{0};
std::atomic<unsigned long long> s_svm_dispatches{0};

bool svmHazardStatOn() {
  static const bool on = std::getenv("NNTR_SVM_HAZARD_STAT") != nullptr;
  return on;
}

/** `site` (default) or `all`; see noteUndrainedSvmPlane's contract. */
bool svmHazardScopeAll() {
  static const bool all = []() {
    const char *e = std::getenv("NNTR_SVM_HAZARD_SCOPE");
    return e != nullptr && e[0] == 'a';
  }();
  return all;
}

/**
 * DEFAULT OFF. The check is correct and it is not free: one model measured 39 %
 * of its prefill for it at `site` scope, and the evidence available when this
 * landed does not earn that price -- the misses it was built for did not
 * reproduce in any configuration on the following day, and the one miss that
 * did land came from a BLANKET DRAIN arm, which is the coarse version of what
 * this check does precisely. The one model whose misses ARE measurable reaches
 * this code with nothing to do: its pack declares "prefill_kv_drain", the
 * producer drains at the site and clears the set, and the run reports drains=0.
 * So there is no model on this chain the shipping default would help today.
 * Opt in with NNTR_SVM_HAZARD_DRAIN=1; revisit when a trigger that reproduces
 * the misses exists.
 */
bool svmHazardEnabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_SVM_HAZARD_DRAIN");
    return e != nullptr && e[0] != '0';
  }();
  return on;
}

/** prints the run's ledger at exit, only under NNTR_SVM_HAZARD_STAT */
struct SvmHazardReport {
  ~SvmHazardReport() {
    if (!svmHazardStatOn())
      return;
    fprintf(stderr, "[SVM-HAZARD] drains=%llu dispatches=%llu enabled=%d\n",
            (unsigned long long)s_svm_hazard_drains.load(),
            (unsigned long long)s_svm_dispatches.load(),
            (int)svmHazardEnabled());
  }
};
SvmHazardReport s_svm_hazard_report;

} // namespace

void Kernel::noteBoundSvmPointer(const void *p) {
  if (p == nullptr || s_bound_svm_n >= kSvmBoundSlots)
    return;
  s_bound_svm[s_bound_svm_n++] = p;
}

void Kernel::noteUndrainedSvmPlane(const void *base, size_t bytes,
                                   bool declared_by_consumer_site) {
  if (!svmHazardEnabled() || base == nullptr || bytes == 0)
    return;
  if (!declared_by_consumer_site && !svmHazardScopeAll())
    return;
  const char *b = static_cast<const char *>(base);
  const char *e = b + bytes;
  for (int i = 0; i < s_svm_plane_n; ++i) {
    // Same plane, or one that already covers this write: widen rather than add
    // a slot. The KV cache is written a slice at a time and the slices share a
    // base, so without this the set would fill with one graph layer.
    if (b >= s_svm_planes[i].base && b < s_svm_planes[i].end) {
      if (e > s_svm_planes[i].end)
        s_svm_planes[i].end = e;
      return;
    }
    if (s_svm_planes[i].base >= b && s_svm_planes[i].base < e) {
      s_svm_planes[i].base = b;
      if (e > s_svm_planes[i].end)
        s_svm_planes[i].end = e;
      return;
    }
  }
  if (s_svm_plane_n >= kSvmPlaneSlots) {
    // The set is full. Dropping a plane would drop a drain, which is the one
    // direction that costs correctness, so drop the NEWEST slot's identity by
    // widening it instead: a wider plane over-drains, it never under-drains.
    SvmPlane &last = s_svm_planes[kSvmPlaneSlots - 1];
    if (b < last.base)
      last.base = b;
    if (e > last.end)
      last.end = e;
    return;
  }
  s_svm_planes[s_svm_plane_n].base = b;
  s_svm_planes[s_svm_plane_n].end = e;
  ++s_svm_plane_n;
}

void Kernel::clearUndrainedSvmPlanes() { s_svm_plane_n = 0; }

unsigned long long Kernel::svmHazardDrains() {
  return s_svm_hazard_drains.load(std::memory_order_relaxed);
}

bool Kernel::takeDispatchSvmHazard() {
  const int bound_n = s_bound_svm_n;
  s_bound_svm_n = 0; // one question per dispatch, whatever the answer
  if (svmHazardStatOn())
    s_svm_dispatches.fetch_add(1, std::memory_order_relaxed);
  if (s_svm_plane_n == 0 || bound_n == 0)
    return false;
  for (int a = 0; a < bound_n; ++a) {
    const char *p = static_cast<const char *>(s_bound_svm[a]);
    for (int i = 0; i < s_svm_plane_n; ++i) {
      if (p >= s_svm_planes[i].base && p < s_svm_planes[i].end) {
        // The caller drains the whole queue, which makes every outstanding
        // write visible, not only this plane.
        s_svm_plane_n = 0;
        if (svmHazardStatOn())
          s_svm_hazard_drains.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
    }
  }
  return false;
}

} // namespace nntrainer::opencl
