// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_loader.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Load required OpenCL functions
 *
 */

#include "opencl_loader.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dynamic_library_loader.h>
#include <mutex>
#include <nntrainer_log.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace nntrainer::opencl {

#define LoadFunction(function)                                                 \
  function = reinterpret_cast<PFN_##function>(                                 \
    DynamicLibraryLoader::loadSymbol(libopencl, #function));

/// Same, for an entry point this file wraps: the variable carries a _raw
/// suffix so the wrapper can own the plain name, but the SYMBOL to resolve is
/// still the driver's.
#define LoadRawFunction(function)                                              \
  function##_raw = reinterpret_cast<PFN_##function>(                           \
    DynamicLibraryLoader::loadSymbol(libopencl, #function));

/// The two driver entry points this file wraps for the memory ledger. They are
/// defined with the other globals below; the loader needs them named first.
extern PFN_clReleaseMemObject clReleaseMemObject_raw;
extern PFN_clSVMFree clSVMFree_raw;

/**
 * @brief Declaration of loading function for OpenCL APIs
 *
 * @param libopencl
 */
void LoadOpenCLFunctions(void *libopencl);

static bool open_cl_initialized = false;

static bool opencl_init_failed = false;

/**
 * @brief Loading OpenCL libraries and required function
 *
 * @return true if successfull or false otherwise
 */
bool LoadOpenCL() {
  // check if already loaded
  if (open_cl_initialized) {
    return true;
  }
  // if OpenCL is not available
  if (opencl_init_failed) {
    return false;
  }

  void *libopencl = nullptr;

#if defined(_WIN32)
  static const char *kClLibName = "OpenCL.dll";
#else
  static const char *kClLibName = "libOpenCL.so";
#endif

  libopencl =
    DynamicLibraryLoader::loadLibrary(kClLibName, RTLD_NOW | RTLD_LOCAL);
  if (libopencl) {
    LoadOpenCLFunctions(libopencl);
    open_cl_initialized = true;
    return true;
  }

#if !defined(_WIN32)
  // Android Qualcomm/Adreno: the vendor's libOpenCL.so is not always reachable
  // through the default linker namespace from a shell-launched executable, so
  // try the well-known vendor paths explicitly. The alternative is asking the
  // caller to set LD_LIBRARY_PATH=/system/vendor/lib64, which on some devices
  // drags in libandroid_runtime.so with unresolved symbols.
  static const char *kAndroidVendorPaths[] = {
    "/vendor/lib64/libOpenCL.so",
    "/system/vendor/lib64/libOpenCL.so",
    "/vendor/lib/libOpenCL.so",
    "/system/vendor/lib/libOpenCL.so",
  };
  for (const char *p : kAndroidVendorPaths) {
    libopencl = DynamicLibraryLoader::loadLibrary(p, RTLD_NOW | RTLD_LOCAL);
    if (libopencl) {
      LoadOpenCLFunctions(libopencl);
      open_cl_initialized = true;
      return true;
    }
  }
#endif

  // record error
  std::string error(DynamicLibraryLoader::getLastError());
  ml_loge("Cannot open OpenCL library on this device - %s", error.c_str());
  opencl_init_failed = true;
  return false;
}

/**
 * @brief Retrieves string representation of OpenCL status code
 *
 * @return OpenCL status code as string
 */
const char *OpenCLErrorCodeToString(const cl_int code) {
#define SWITCH_CASE_RETURN(ENUM)                                               \
  case ENUM:                                                                   \
    return #ENUM

  switch (code) {
    SWITCH_CASE_RETURN(CL_SUCCESS);
    SWITCH_CASE_RETURN(CL_DEVICE_NOT_FOUND);
    SWITCH_CASE_RETURN(CL_DEVICE_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_COMPILER_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    SWITCH_CASE_RETURN(CL_OUT_OF_RESOURCES);
    SWITCH_CASE_RETURN(CL_OUT_OF_HOST_MEMORY);
    SWITCH_CASE_RETURN(CL_PROFILING_INFO_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_MEM_COPY_OVERLAP);
    SWITCH_CASE_RETURN(CL_IMAGE_FORMAT_MISMATCH);
    SWITCH_CASE_RETURN(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    SWITCH_CASE_RETURN(CL_BUILD_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
    SWITCH_CASE_RETURN(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    SWITCH_CASE_RETURN(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#ifdef CL_VERSION_1_2
    SWITCH_CASE_RETURN(CL_COMPILE_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_LINKER_NOT_AVAILABLE);
    SWITCH_CASE_RETURN(CL_LINK_PROGRAM_FAILURE);
    SWITCH_CASE_RETURN(CL_DEVICE_PARTITION_FAILED);
    SWITCH_CASE_RETURN(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
    SWITCH_CASE_RETURN(CL_INVALID_VALUE);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_TYPE);
    SWITCH_CASE_RETURN(CL_INVALID_PLATFORM);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE);
    SWITCH_CASE_RETURN(CL_INVALID_CONTEXT);
    SWITCH_CASE_RETURN(CL_INVALID_QUEUE_PROPERTIES);
    SWITCH_CASE_RETURN(CL_INVALID_COMMAND_QUEUE);
    SWITCH_CASE_RETURN(CL_INVALID_HOST_PTR);
    SWITCH_CASE_RETURN(CL_INVALID_MEM_OBJECT);
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_SAMPLER);
    SWITCH_CASE_RETURN(CL_INVALID_BINARY);
    SWITCH_CASE_RETURN(CL_INVALID_BUILD_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_PROGRAM);
    SWITCH_CASE_RETURN(CL_INVALID_PROGRAM_EXECUTABLE);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_NAME);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_DEFINITION);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_INDEX);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_VALUE);
    SWITCH_CASE_RETURN(CL_INVALID_ARG_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_KERNEL_ARGS);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_DIMENSION);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_GROUP_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_WORK_ITEM_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_GLOBAL_OFFSET);
    SWITCH_CASE_RETURN(CL_INVALID_EVENT_WAIT_LIST);
    SWITCH_CASE_RETURN(CL_INVALID_EVENT);
    SWITCH_CASE_RETURN(CL_INVALID_OPERATION);
    SWITCH_CASE_RETURN(CL_INVALID_GL_OBJECT);
    SWITCH_CASE_RETURN(CL_INVALID_BUFFER_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_MIP_LEVEL);
    SWITCH_CASE_RETURN(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
    SWITCH_CASE_RETURN(CL_INVALID_PROPERTY);
#endif
#ifdef CL_VERSION_1_2
    SWITCH_CASE_RETURN(CL_INVALID_IMAGE_DESCRIPTOR);
    SWITCH_CASE_RETURN(CL_INVALID_COMPILER_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_LINKER_OPTIONS);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#ifdef CL_VERSION_2_0
    SWITCH_CASE_RETURN(CL_INVALID_PIPE_SIZE);
    SWITCH_CASE_RETURN(CL_INVALID_DEVICE_QUEUE);
#endif
#ifdef CL_VERSION_2_2
    SWITCH_CASE_RETURN(CL_INVALID_SPEC_ID);
    SWITCH_CASE_RETURN(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif
  default:
    return "(unknown)";
  }
#undef SWITCH_CASE_RETURN
}

/**
 * @brief Utility to load the required OpenCL APIs
 *
 * @param libopencl
 */
void LoadOpenCLFunctions(void *libopencl) {
  LoadFunction(clGetPlatformIDs);
  LoadFunction(clGetDeviceIDs);
  LoadFunction(clGetDeviceInfo);
  LoadFunction(clGetImageInfo);
  LoadFunction(clCreateContext);
  LoadFunction(clCreateCommandQueue);
  LoadFunction(clCreateBuffer);
  LoadFunction(clCreateSubBuffer);
  LoadFunction(clCreateImage);
  LoadFunction(clEnqueueWriteBuffer);
  LoadFunction(clEnqueueFillBuffer);
  LoadFunction(clEnqueueReadBuffer);
  LoadFunction(clEnqueueMapBuffer);
  LoadFunction(clEnqueueUnmapMemObject);
  LoadFunction(clEnqueueWriteBufferRect);
  LoadFunction(clEnqueueReadBufferRect);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clCreateProgramWithBinary);
  LoadFunction(clBuildProgram);
  LoadFunction(clGetProgramInfo);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clRetainProgram);
  LoadFunction(clCreateKernel);
  LoadFunction(clSetKernelArg);
  LoadFunction(clEnqueueNDRangeKernel);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadRawFunction(clReleaseMemObject);
  LoadFunction(clFlush);
  LoadFunction(clFinish);
  LoadFunction(clSVMAlloc);
  LoadRawFunction(clSVMFree);
  LoadFunction(clEnqueueSVMMap);
  LoadFunction(clEnqueueSVMUnmap);
  LoadFunction(clSetKernelArgSVMPointer);
  LoadFunction(clWaitForEvents);
  LoadFunction(clReleaseEvent);
  LoadFunction(clEnqueueBarrierWithWaitList);
}

PFN_clGetPlatformIDs clGetPlatformIDs;
PFN_clGetDeviceIDs clGetDeviceIDs;
PFN_clGetDeviceInfo clGetDeviceInfo;
PFN_clGetImageInfo clGetImageInfo;
PFN_clCreateContext clCreateContext;
PFN_clCreateCommandQueue clCreateCommandQueue;
PFN_clCreateBuffer clCreateBuffer;
PFN_clCreateSubBuffer clCreateSubBuffer;
PFN_clCreateImage clCreateImage;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer;
PFN_clEnqueueFillBuffer clEnqueueFillBuffer;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject;
PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect;
PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect;
PFN_clCreateProgramWithSource clCreateProgramWithSource;
PFN_clCreateProgramWithBinary clCreateProgramWithBinary;
PFN_clBuildProgram clBuildProgram;
PFN_clGetProgramInfo clGetProgramInfo;
PFN_clGetProgramBuildInfo clGetProgramBuildInfo;
PFN_clRetainProgram clRetainProgram;
PFN_clCreateKernel clCreateKernel;
PFN_clSetKernelArg clSetKernelArg;
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
PFN_clGetEventProfilingInfo clGetEventProfilingInfo;
PFN_clRetainContext clRetainContext;
PFN_clReleaseContext clReleaseContext;
PFN_clRetainCommandQueue clRetainCommandQueue;
PFN_clReleaseCommandQueue clReleaseCommandQueue;
PFN_clReleaseMemObject clReleaseMemObject_raw;
PFN_clSVMFree clSVMFree_raw;
PFN_clFlush clFlush;
PFN_clFinish clFinish;
PFN_clSVMAlloc clSVMAlloc;
PFN_clEnqueueSVMMap clEnqueueSVMMap;
PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap;
PFN_clSetKernelArgSVMPointer clSetKernelArgSVMPointer;
PFN_clWaitForEvents clWaitForEvents;
PFN_clReleaseEvent clReleaseEvent;
PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;

// ---------------------------------------------------------------------------
// Handle epoch.
//
// Caches that key on a cl_mem / SVM pointer VALUE (the kernel-argument cache in
// opencl_kernel.cpp, the image->backing-buffer cache in blas_kernels.cpp) are
// only sound while a value identifies one object. The single way that breaks is
// release-then-recreate at the same address, so every creation entry point in
// this tree goes through the wrappers below and bumps this counter; a cache
// stores the epoch it was filled at and drops itself when the epoch moves.
// Bumping on CREATION rather than release is what makes the record complete:
// a release alone cannot make a stale value point at a different object.
//
// Relaxed atomics: the dispatch path is single threaded, and a cache that
// observes a stale epoch merely re-issues the call it could have skipped.
static std::atomic<unsigned long long> g_handle_epoch{1};

unsigned long long clHandleEpoch() {
  return g_handle_epoch.load(std::memory_order_relaxed);
}

void clBumpHandleEpoch() {
  g_handle_epoch.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// GPU memory ledger (NNTR_GPU_MEM_ACCT).
//
// /sys/class/kgsl/kgsl/page_alloc is the honest GPU footprint on this driver
// and says nothing about composition. This records the same allocations from
// the inside, tagged by call site, so the two can be put side by side.
// ---------------------------------------------------------------------------
namespace {

struct AcctTagStat {
  size_t live = 0;     ///< bytes currently outstanding
  size_t peak = 0;     ///< high-water of `live`
  size_t cum = 0;      ///< bytes ever requested (grow-only caches show here)
  unsigned n = 0;      ///< allocations
  unsigned n_view = 0; ///< of which views (sub-buffers, image-from-buffer)
  unsigned n_free = 0;
};

struct AcctRec {
  std::string tag;
  size_t bytes;
};

class MemAcct {
public:
  static MemAcct &get() {
    static MemAcct a;
    return a;
  }

  ~MemAcct() {
    if (dump_)
      dump("atexit");
  }

  bool enabled() const { return enabled_; }

  /// Bytes outstanding right now, and the high-water of that. This is the
  /// quantity the honest footprint wants: what this process has asked the
  /// driver for and not given back.
  size_t liveBytes() {
    if (!enabled_)
      return 0;
    std::lock_guard<std::mutex> lk(m_);
    return live_bytes_;
  }
  size_t peakBytes() {
    if (!enabled_)
      return 0;
    std::lock_guard<std::mutex> lk(m_);
    return peak_bytes_;
  }

  void note(const char *kind, const void *handle, size_t bytes, bool view) {
    if (!enabled_ || handle == nullptr)
      return;
    view = view || viewDepth() > 0;
    const std::string tag = std::string(kind) + "|" + currentTag();
    std::lock_guard<std::mutex> lk(m_);
    auto &st = tags_[tag];
    ++st.n;
    if (view) {
      ++st.n_view;
    } else {
      st.cum += bytes;
      st.live += bytes;
      // (std::max): this TU pulls in windows.h transitively on MSVC, whose
      // min/max macros would otherwise eat the bare call.
      st.peak = (std::max)(st.peak, st.live);
      live_bytes_ += bytes;
      peak_bytes_ = (std::max)(peak_bytes_, live_bytes_);
    }
    live_[handle] = AcctRec{tag, view ? 0u : bytes};
  }

  void release(const void *handle) {
    if (!enabled_ || handle == nullptr)
      return;
    std::lock_guard<std::mutex> lk(m_);
    auto it = live_.find(handle);
    if (it == live_.end())
      return;
    auto &st = tags_[it->second.tag];
    ++st.n_free;
    st.live -= (std::min)(st.live, it->second.bytes);
    live_bytes_ -= (std::min)(live_bytes_, it->second.bytes);
    live_.erase(it);
  }

  void push(const char *tag) {
    if (enabled_)
      stack().push_back(tag);
  }
  void pushView() {
    if (enabled_)
      ++viewDepth();
  }
  void popView() {
    if (enabled_ && viewDepth() > 0)
      --viewDepth();
  }
  void pop() {
    if (enabled_ && !stack().empty())
      stack().pop_back();
  }

  void dump(const char *phase) {
    if (!dump_)
      return;
    std::vector<std::pair<std::string, AcctTagStat>> rows;
    size_t live = 0, peak = 0;
    {
      std::lock_guard<std::mutex> lk(m_);
      rows.assign(tags_.begin(), tags_.end());
      live = live_bytes_;
      peak = peak_bytes_;
    }
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
      return a.second.live > b.second.live;
    });
    std::fprintf(stderr, "[gpumem] ==== %s ==== live %.1f MiB, peak %.1f MiB\n",
                 phase, live / 1048576.0, peak / 1048576.0);
    std::fprintf(stderr, "[gpumem] %-38s %10s %10s %10s %6s %6s %6s\n",
                 "kind|tag", "live_MiB", "peak_MiB", "cum_MiB", "n", "views",
                 "freed");
    for (const auto &r : rows)
      std::fprintf(stderr, "[gpumem] %-38s %10.2f %10.2f %10.2f %6u %6u %6u\n",
                   r.first.c_str(), r.second.live / 1048576.0,
                   r.second.peak / 1048576.0, r.second.cum / 1048576.0,
                   r.second.n, r.second.n_view, r.second.n_free);
    std::fflush(stderr);
  }

private:
  MemAcct() {
    const char *e = std::getenv("NNTR_GPU_MEM_ACCT");
    /** The counting is ON by default and NNTR_GPU_MEM_ACCT=0 opts out; the
     *  stderr dumps stay opt-in and need the flag set to something else.
     *
     *  The reason the ledger cannot stay opt-in: it is the only per-process
     *  GPU byte count the app can read. /sys/class/kgsl/kgsl/proc/<pid>/gpumem
     *  is Permission denied to the shell uid AND to the application uid on the
     *  Android devices this was measured on, so the global page_alloc counter
     *  -- which is not ours alone -- is all an outside observer gets.
     *  A run that has to report its own honest footprint has to count from
     *  the inside, on every run, not only when a flag is set. */
    enabled_ = (e == nullptr || (e[0] != 0 && e[0] != '0'));
    dump_ = (e != nullptr && e[0] != 0 && e[0] != '0');
  }

  static std::vector<const char *> &stack() {
    static thread_local std::vector<const char *> s;
    return s;
  }
  /** Depth of nested "this allocation is a VIEW" scopes on this thread.
   *  A view allocates no memory of its own -- a sub-buffer, an image over a
   *  buffer, or a CL_MEM_USE_HOST_PTR buffer over memory that is already
   *  allocated and already counted. Counting its bytes again would make the
   *  ledger disagree with the driver by exactly the amount the aliasing saves,
   *  and the app's Peak Mem is computed FROM this ledger. */
  static int &viewDepth() {
    static thread_local int d = 0;
    return d;
  }
  static std::string currentTag() {
    const auto &s = stack();
    return s.empty() ? std::string("untagged") : std::string(s.back());
  }

  bool enabled_ = false;
  bool dump_ = false;
  std::mutex m_;
  std::unordered_map<const void *, AcctRec> live_;
  std::unordered_map<std::string, AcctTagStat> tags_;
  size_t live_bytes_ = 0;
  size_t peak_bytes_ = 0;
};

/// One cheap load on every allocation when the ledger is off.
const bool g_acct_on = MemAcct::get().enabled();

} // namespace

bool clMemAcctOn() { return g_acct_on; }
void clMemAcctPush(const char *tag) { MemAcct::get().push(tag); }
void clMemAcctPop() { MemAcct::get().pop(); }
void clMemAcctPushView() { MemAcct::get().pushView(); }
void clMemAcctPopView() { MemAcct::get().popView(); }
void clMemAcctDump(const char *phase) { MemAcct::get().dump(phase); }
size_t clMemAcctLiveBytes() {
  return g_acct_on ? MemAcct::get().liveBytes() : 0;
}
size_t clMemAcctPeakBytes() {
  return g_acct_on ? MemAcct::get().peakBytes() : 0;
}

cl_int clReleaseMemObjectT(cl_mem memobj) {
  if (g_acct_on)
    MemAcct::get().release(memobj);
  return clReleaseMemObject_raw(memobj);
}

void clSVMFreeT(cl_context context, void *svm_pointer) {
  if (g_acct_on)
    MemAcct::get().release(svm_pointer);
  clSVMFree_raw(context, svm_pointer);
}

cl_mem clCreateBufferT(cl_context context, cl_mem_flags flags, size_t size,
                       void *host_ptr, cl_int *errcode_ret) {
  clBumpHandleEpoch();
  cl_mem m = clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
  if (g_acct_on)
    MemAcct::get().note("buf", m, size, /*view=*/false);
  return m;
}

cl_mem clCreateSubBufferT(cl_mem buffer, cl_mem_flags flags,
                          cl_buffer_create_type type, const void *info,
                          cl_int *errcode_ret) {
  clBumpHandleEpoch();
  cl_mem m = clCreateSubBuffer(buffer, flags, type, info, errcode_ret);
  /** A sub-buffer is a window on its parent: it allocates nothing, so it is
   *  counted and charged zero bytes. */
  if (g_acct_on)
    MemAcct::get().note("subbuf", m, 0, /*view=*/true);
  return m;
}

cl_mem clCreateImageT(cl_context context, cl_mem_flags flags,
                      const cl_image_format *format, const cl_image_desc *desc,
                      void *host_ptr, cl_int *errcode_ret) {
  clBumpHandleEpoch();
  cl_mem m = clCreateImage(context, flags, format, desc, host_ptr, errcode_ret);
  if (g_acct_on) {
    /** image2d-from-buffer allocates nothing -- it reinterprets a buffer this
     *  ledger already charged. Every image in this tree is that shape today;
     *  a standalone one would be a real allocation, so size it rather than
     *  assume, from the pitch the driver was given (row_pitch 0 means "packed",
     *  which cannot be reconstructed from the desc alone -- charge 0 and let
     *  the count show it). */
    const bool view = desc != nullptr && desc->buffer != nullptr;
    size_t bytes = 0;
    if (!view && desc != nullptr)
      bytes =
        desc->image_row_pitch * (desc->image_height ? desc->image_height : 1);
    MemAcct::get().note("img", m, bytes, view);
  }
  return m;
}

void *clSVMAllocT(cl_context context, cl_svm_mem_flags flags, size_t size,
                  unsigned int alignment) {
  clBumpHandleEpoch();
  void *p = clSVMAlloc(context, flags, size, alignment);
  if (g_acct_on)
    MemAcct::get().note("svm", p, size, /*view=*/false);
  return p;
}
} // namespace nntrainer::opencl
