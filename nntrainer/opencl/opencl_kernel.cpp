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
// Dispatch write log.
//
// A producer kernel can hand a consumer more than its named output -- a norm
// that also emits the int8 quantisation of the row it just wrote, a quantiser
// whose result a sibling FC can reuse. The consumer is a different call site
// and identifies the data only by the device buffer it reads, and on this
// backend a pooled sub-buffer handle does NOT identify a tensor: the planner
// hands the same handle back for tensors that merely share an offset. Keying
// a reuse on the handle alone is the KV-sharing defect this guards against --
// two different activations at one address, the second FC silently multiplying
// the first one's quantisation.
//
// What actually has to be true for the reuse to be correct is narrower and
// checkable: the BYTES at that handle must not have changed since the
// producer wrote them. So each dispatch appends one entry saying which device
// buffer it wrote, and a reuse is allowed only when every entry since the
// producer is present and none of them names the consumer's buffer. A site
// that does not declare its writes logs UNKNOWN and invalidates every
// outstanding handoff -- the failure direction is a redundant recomputation,
// not a wrong answer, so an un-annotated writer anywhere in the tree (present
// or future) is safe by default.
//
// The log is a small ring: a handoff spans a handful of dispatches, and a
// window longer than the ring simply answers "changed".
namespace {
constexpr int kWriteLogSize = 64;
struct WriteLogEntry {
  void *written = nullptr; // buffer this dispatch wrote (when known)
  bool known = false;      // false => the dispatch did not declare
};
// Same threading argument as s_bind_touched_svm above: declaration and
// dispatch are one pairing on one thread.
thread_local WriteLogEntry s_wlog[kWriteLogSize];
thread_local unsigned long long s_wlog_seq = 0;
thread_local void *s_pending_write = nullptr;
thread_local bool s_pending_known = false;
} // namespace

void Kernel::noteDispatchWrites(void *written) {
  s_pending_write = written;
  s_pending_known = true;
}

void Kernel::noteDispatchWritesNothing() {
  s_pending_write = nullptr;
  s_pending_known = true;
}

void Kernel::commitDispatch() {
  ++s_wlog_seq;
  WriteLogEntry &e = s_wlog[s_wlog_seq % kWriteLogSize];
  e.written = s_pending_write;
  e.known = s_pending_known;
  s_pending_write = nullptr;
  s_pending_known = false;
}

unsigned long long Kernel::dispatchSeq() { return s_wlog_seq; }

bool Kernel::bufferUnchangedSince(void *handle, unsigned long long since_seq) {
  if (handle == nullptr || since_seq == 0 || since_seq > s_wlog_seq)
    return false;
  if (s_wlog_seq - since_seq >= (unsigned long long)kWriteLogSize)
    return false;
  for (unsigned long long s = since_seq + 1; s <= s_wlog_seq; ++s) {
    const WriteLogEntry &e = s_wlog[s % kWriteLogSize];
    if (!e.known)
      return false;
    if (e.written == handle)
      return false;
  }
  return true;
}

} // namespace nntrainer::opencl
