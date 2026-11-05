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
#include <chrono>
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

/// The queue entry points the launch ledger wraps (NNTR_CL_LAUNCH_STAT). They
/// are loaded as *_raw and the plain names are bound by
/// InstallLaunchStatWrappers.
extern PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel_raw;
extern PFN_clFinish clFinish_raw;
extern PFN_clFlush clFlush_raw;
extern PFN_clEnqueueReadBuffer clEnqueueReadBuffer_raw;
extern PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer_raw;
extern PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect_raw;
extern PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect_raw;
extern PFN_clEnqueueFillBuffer clEnqueueFillBuffer_raw;
extern PFN_clEnqueueMapBuffer clEnqueueMapBuffer_raw;
extern PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject_raw;
extern PFN_clEnqueueSVMMap clEnqueueSVMMap_raw;
extern PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap_raw;
extern PFN_clWaitForEvents clWaitForEvents_raw;
extern PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList_raw;
extern PFN_clCreateCommandQueue clCreateCommandQueue_raw;
void InstallLaunchStatWrappers();

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

#if defined(__ANDROID__)
  // Android Qualcomm/Adreno: the vendor's libOpenCL.so is not always reachable
  // through the default linker namespace from a shell-launched executable, so
  // try the well-known vendor paths explicitly. The alternative is asking the
  // caller to set LD_LIBRARY_PATH=/system/vendor/lib64, which on some devices
  // drags in libandroid_runtime.so with unresolved symbols.
  //
  // Guarded on Android rather than on "not Windows", which is what the paths
  // themselves say. A desktop Linux build was trying all four of them after
  // the normal load already failed, and the last failure is the one reported
  // below -- so a box with no ICD logged a missing /system/vendor path instead
  // of the real libOpenCL.so error.
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
  LoadRawFunction(clCreateCommandQueue);
  LoadFunction(clCreateBuffer);
  LoadFunction(clCreateSubBuffer);
  LoadFunction(clCreateImage);
  LoadRawFunction(clEnqueueWriteBuffer);
  LoadRawFunction(clEnqueueFillBuffer);
  LoadRawFunction(clEnqueueReadBuffer);
  LoadRawFunction(clEnqueueMapBuffer);
  LoadRawFunction(clEnqueueUnmapMemObject);
  LoadRawFunction(clEnqueueWriteBufferRect);
  LoadRawFunction(clEnqueueReadBufferRect);
  LoadFunction(clCreateProgramWithSource);
  LoadFunction(clCreateProgramWithBinary);
  LoadFunction(clBuildProgram);
  LoadFunction(clGetProgramInfo);
  LoadFunction(clGetProgramBuildInfo);
  LoadFunction(clRetainProgram);
  LoadFunction(clCreateKernel);
  LoadFunction(clSetKernelArg);
  LoadRawFunction(clEnqueueNDRangeKernel);
  LoadFunction(clGetEventInfo);
  LoadFunction(clGetEventProfilingInfo);
  LoadFunction(clRetainContext);
  LoadFunction(clReleaseContext);
  LoadFunction(clRetainCommandQueue);
  LoadFunction(clReleaseCommandQueue);
  LoadRawFunction(clReleaseMemObject);
  LoadRawFunction(clFlush);
  LoadRawFunction(clFinish);
  LoadFunction(clSVMAlloc);
  LoadRawFunction(clSVMFree);
  LoadRawFunction(clEnqueueSVMMap);
  LoadRawFunction(clEnqueueSVMUnmap);
  LoadFunction(clSetKernelArgSVMPointer);
  LoadRawFunction(clWaitForEvents);
  LoadFunction(clReleaseEvent);
  LoadFunction(clRetainEvent);
  LoadFunction(clCreateUserEvent);
  LoadFunction(clSetUserEventStatus);
  LoadFunction(clSetEventCallback);
  LoadRawFunction(clEnqueueBarrierWithWaitList);
  LoadFunction(clGetKernelInfo);
  LoadFunction(clGetKernelWorkGroupInfo);
  InstallLaunchStatWrappers();
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
PFN_clGetEventInfo clGetEventInfo;
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
PFN_clRetainEvent clRetainEvent;
PFN_clCreateUserEvent clCreateUserEvent;
PFN_clSetUserEventStatus clSetUserEventStatus;
PFN_clSetEventCallback clSetEventCallback;
PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList;
PFN_clGetKernelInfo clGetKernelInfo;
PFN_clGetKernelWorkGroupInfo clGetKernelWorkGroupInfo;

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
// ---------------------------------------------------------------------------
// Launch ledger (NNTR_CL_LAUNCH_STAT).
//
// The question this answers is "how many times per decode token does the host
// talk to the queue, and how long does each of those calls hold the host":
// the difference between a token's wall and the GPU's busy time is host gaps,
// and a host gap is made of exactly these calls. The wrappers are installed
// at load time only when the variable is set; otherwise the plain driver
// pointers are bound and the dispatch path is what it always was.
//
// Attribution: every NDRange is keyed by its kernel's function name, and a
// clFinish/clFlush is charged both to its own API row and to the kernel that
// was enqueued last -- a per-dispatch drain is a property of that dispatch.
// Blocking reads, maps and SVM maps are the other ways a host call waits on
// the device, so they are counted as API rows with their wall time.
//
// Level 2 asks the queue for CL_QUEUE_PROFILING_ENABLE and hands the driver an
// event per NDRange that the caller did not want one for; the tick collects
// them (the token is complete by then, so the wait is free) and sums each
// kernel's START..END. An in-order queue runs kernels back to back, so the
// sum is the GPU busy time and wall - sum is the idle the host left it.
namespace {

/** wall clock for the wrappers */
inline double launchNowMs() {
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

struct LaunchRow {
  unsigned long long n = 0;        ///< calls
  double host_ms = 0;              ///< host wall inside the call
  unsigned long long n_finish = 0; ///< clFinish issued right after (kernels)
  double finish_ms = 0;            ///< host wall of those finishes
  unsigned long long n_flush = 0;  ///< clFlush issued right after (kernels)
  double gpu_ms = 0;               ///< device START..END (level 2)
  unsigned long long n_gpu = 0;    ///< events collected (level 2)
};

struct PendingEvent {
  cl_event ev;
  const std::string *name;
};

struct LaunchStat {
  static LaunchStat &get() {
    static LaunchStat s;
    return s;
  }
  int level = 0;
  std::atomic<bool> counting{false};
  std::mutex mu;
  std::unordered_map<std::string, LaunchRow> kernels;
  std::unordered_map<std::string, LaunchRow> calls;
  std::unordered_map<cl_kernel, std::string> names;
  const std::string *last_kernel = nullptr;
  std::vector<PendingEvent> pending;
  unsigned long long tokens = 0;
  unsigned long long dropped_tokens = 0;
  double phase_begin_ms = 0;
  double tokens_begin_ms = 0;
  double last_tick_ms = 0;
  std::string phase;

  LaunchStat() {
    if (const char *e = std::getenv("NNTR_CL_LAUNCH_STAT"))
      level = std::atoi(e);
    if (level < 0)
      level = 0;
  }

  const std::string &kernelName(cl_kernel k) {
    auto it = names.find(k);
    if (it != names.end())
      return it->second;
    char buf[256] = {0};
    size_t ret = 0;
    std::string name;
    if (clGetKernelInfo != nullptr &&
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, sizeof(buf) - 1, buf,
                        &ret) == CL_SUCCESS &&
        ret > 0) {
      name.assign(buf, ret > 0 && buf[ret - 1] == '\0' ? ret - 1 : ret);
    } else {
      char tmp[64];
      snprintf(tmp, sizeof(tmp), "kernel@%p", static_cast<void *>(k));
      name = tmp;
    }
    return names.emplace(k, std::move(name)).first->second;
  }

  void noteKernel(cl_kernel k, double ms, cl_event own) {
    std::lock_guard<std::mutex> lk(mu);
    const std::string &name = kernelName(k);
    LaunchRow &r = kernels[name];
    ++r.n;
    r.host_ms += ms;
    last_kernel = &kernels.find(name)->first;
    if (own != nullptr)
      pending.push_back({own, last_kernel});
  }

  /** an API call; `after_kernel` charges it to the last dispatch too */
  void noteCall(const char *api, double ms, int after_kernel /*1 fin 2 fl*/) {
    std::lock_guard<std::mutex> lk(mu);
    LaunchRow &r = calls[api];
    ++r.n;
    r.host_ms += ms;
    if (after_kernel != 0 && last_kernel != nullptr) {
      LaunchRow &k = kernels[*last_kernel];
      if (after_kernel == 1) {
        ++k.n_finish;
        k.finish_ms += ms;
      } else {
        ++k.n_flush;
      }
    }
  }

  void reset() {
    kernels.clear();
    calls.clear();
    pending.clear();
    last_kernel = nullptr;
    tokens = 0;
  }

  /** level 2: fold the completed events into their kernels' gpu_ms */
  void collectEvents() {
    if (pending.empty())
      return;
    std::vector<cl_event> evs;
    evs.reserve(pending.size());
    for (const PendingEvent &p : pending)
      evs.push_back(p.ev);
    // The raw entry point: the wrapper would count this wait as a token cost
    // and take the ledger mutex this thread already holds.
    if (clWaitForEvents_raw != nullptr)
      clWaitForEvents_raw(static_cast<cl_uint>(evs.size()), evs.data());
    for (const PendingEvent &p : pending) {
      cl_ulong t0 = 0, t1 = 0;
      if (clGetEventProfilingInfo != nullptr &&
          clGetEventProfilingInfo(p.ev, CL_PROFILING_COMMAND_START, sizeof(t0),
                                  &t0, nullptr) == CL_SUCCESS &&
          clGetEventProfilingInfo(p.ev, CL_PROFILING_COMMAND_END, sizeof(t1),
                                  &t1, nullptr) == CL_SUCCESS &&
          t1 >= t0) {
        LaunchRow &r = kernels[*p.name];
        r.gpu_ms += static_cast<double>(t1 - t0) * 1e-6;
        ++r.n_gpu;
      }
      if (clReleaseEvent != nullptr)
        clReleaseEvent(p.ev);
    }
    pending.clear();
  }

  void begin(const char *ph) {
    std::lock_guard<std::mutex> lk(mu);
    reset();
    dropped_tokens = 0;
    phase = ph ? ph : "";
    phase_begin_ms = tokens_begin_ms = last_tick_ms = launchNowMs();
    counting.store(true, std::memory_order_release);
  }

  void tick() {
    if (!counting.load(std::memory_order_acquire))
      return;
    const double now = launchNowMs();
    std::lock_guard<std::mutex> lk(mu);
    if (level >= 2)
      collectEvents();
    if (dropped_tokens == 0) {
      // The first token of a phase pays for whatever the previous phase left
      // in flight; it is not a steady token, so it is measured and dropped.
      reset();
      dropped_tokens = 1;
      tokens_begin_ms = now;
    } else {
      ++tokens;
    }
    last_tick_ms = now;
  }

  static void printRow(const char *name, const LaunchRow &r, double per,
                       bool gpu) {
    if (gpu)
      fprintf(stderr,
              "[CL-LAUNCH]   %-44s n/tok=%8.2f enq_ms/tok=%7.3f "
              "fin/tok=%7.2f fin_ms/tok=%7.3f flush/tok=%6.2f "
              "gpu_ms/tok=%7.3f\n",
              name, r.n / per, r.host_ms / per, r.n_finish / per,
              r.finish_ms / per, r.n_flush / per, r.gpu_ms / per);
    else
      fprintf(stderr,
              "[CL-LAUNCH]   %-44s n/tok=%8.2f enq_ms/tok=%7.3f "
              "fin/tok=%7.2f fin_ms/tok=%7.3f flush/tok=%6.2f\n",
              name, r.n / per, r.host_ms / per, r.n_finish / per,
              r.finish_ms / per, r.n_flush / per);
  }

  void dump(const char *ph) {
    std::lock_guard<std::mutex> lk(mu);
    counting.store(false, std::memory_order_release);
    if (level >= 2)
      collectEvents();
    const double per = tokens > 0 ? static_cast<double>(tokens) : 1.0;
    const double wall = tokens > 0 ? (last_tick_ms - tokens_begin_ms) / per
                                   : (launchNowMs() - phase_begin_ms);
    LaunchRow tot;
    for (const auto &kv : kernels) {
      tot.n += kv.second.n;
      tot.host_ms += kv.second.host_ms;
      tot.n_finish += kv.second.n_finish;
      tot.finish_ms += kv.second.finish_ms;
      tot.n_flush += kv.second.n_flush;
      tot.gpu_ms += kv.second.gpu_ms;
      tot.n_gpu += kv.second.n_gpu;
    }
    double api_ms = 0;
    unsigned long long api_n = 0;
    for (const auto &kv : calls) {
      api_ms += kv.second.host_ms;
      api_n += kv.second.n;
    }
    fprintf(stderr,
            "[CL-LAUNCH] phase=%s(%s) level=%d tokens=%llu (first dropped) "
            "wall_ms/tok=%.3f launches/tok=%.1f enq_ms/tok=%.3f "
            "finish/tok=%.1f finish_ms/tok=%.3f flush/tok=%.1f "
            "api_calls/tok=%.1f api_ms/tok=%.3f host_in_cl_ms/tok=%.3f",
            phase.c_str(), ph ? ph : "", level, tokens, wall, tot.n / per,
            tot.host_ms / per, tot.n_finish / per, tot.finish_ms / per,
            tot.n_flush / per, api_n / per, api_ms / per,
            (tot.host_ms + api_ms) / per);
    if (level >= 2)
      fprintf(stderr, " gpu_ms/tok=%.3f gpu_busy=%.1f%% events=%llu",
              tot.gpu_ms / per, wall > 0 ? 100.0 * tot.gpu_ms / per / wall : 0,
              tot.n_gpu);
    fprintf(stderr, "\n");
    std::vector<std::pair<std::string, LaunchRow>> rows(kernels.begin(),
                                                        kernels.end());
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
      const double ca = a.second.host_ms + a.second.finish_ms;
      const double cb = b.second.host_ms + b.second.finish_ms;
      return ca != cb ? ca > cb : a.second.n > b.second.n;
    });
    fprintf(stderr, "[CL-LAUNCH] kernels by host cost (enqueue + finish "
                    "charged to it), per token:\n");
    for (const auto &kv : rows)
      printRow(kv.first.c_str(), kv.second, per, level >= 2);
    std::vector<std::pair<std::string, LaunchRow>> apis(calls.begin(),
                                                        calls.end());
    std::sort(apis.begin(), apis.end(), [](const auto &a, const auto &b) {
      return a.second.host_ms > b.second.host_ms;
    });
    fprintf(stderr, "[CL-LAUNCH] queue API calls, per token:\n");
    for (const auto &kv : apis)
      fprintf(stderr, "[CL-LAUNCH]   %-44s n/tok=%8.2f ms/tok=%7.3f\n",
              kv.first.c_str(), kv.second.n / per, kv.second.host_ms / per);
  }
};

const int g_launch_level = LaunchStat::get().level;

inline bool launchCounting() {
  return LaunchStat::get().counting.load(std::memory_order_acquire);
}

} // namespace

// The driver entry points the ledger wraps. Bound by LoadOpenCLFunctions to
// the driver symbol; the plain name is the wrapper only when the ledger is on.
PFN_clEnqueueNDRangeKernel clEnqueueNDRangeKernel_raw;
PFN_clFinish clFinish_raw;
PFN_clFlush clFlush_raw;
PFN_clEnqueueReadBuffer clEnqueueReadBuffer_raw;
PFN_clEnqueueWriteBuffer clEnqueueWriteBuffer_raw;
PFN_clEnqueueReadBufferRect clEnqueueReadBufferRect_raw;
PFN_clEnqueueWriteBufferRect clEnqueueWriteBufferRect_raw;
PFN_clEnqueueFillBuffer clEnqueueFillBuffer_raw;
PFN_clEnqueueMapBuffer clEnqueueMapBuffer_raw;
PFN_clEnqueueUnmapMemObject clEnqueueUnmapMemObject_raw;
PFN_clEnqueueSVMMap clEnqueueSVMMap_raw;
PFN_clEnqueueSVMUnmap clEnqueueSVMUnmap_raw;
PFN_clWaitForEvents clWaitForEvents_raw;
PFN_clEnqueueBarrierWithWaitList clEnqueueBarrierWithWaitList_raw;
PFN_clCreateCommandQueue clCreateCommandQueue_raw;

namespace {

cl_int CL_API_CALL clEnqueueNDRangeKernel_stat(
  cl_command_queue q, cl_kernel k, cl_uint work_dim, const size_t *goff,
  const size_t *gws, const size_t *lws, cl_uint n_wait, const cl_event *wait,
  cl_event *event) {
  if (!launchCounting())
    return clEnqueueNDRangeKernel_raw(q, k, work_dim, goff, gws, lws, n_wait,
                                      wait, event);
  cl_event own = nullptr;
  cl_event *evp = event;
  if (g_launch_level >= 2 && event == nullptr)
    evp = &own;
  const double t0 = launchNowMs();
  const cl_int rc = clEnqueueNDRangeKernel_raw(q, k, work_dim, goff, gws, lws,
                                               n_wait, wait, evp);
  const double ms = launchNowMs() - t0;
  LaunchStat::get().noteKernel(k, ms, rc == CL_SUCCESS ? own : nullptr);
  return rc;
}

cl_int CL_API_CALL clFinish_stat(cl_command_queue q) {
  if (!launchCounting())
    return clFinish_raw(q);
  const double t0 = launchNowMs();
  const cl_int rc = clFinish_raw(q);
  LaunchStat::get().noteCall("clFinish", launchNowMs() - t0, 1);
  return rc;
}

cl_int CL_API_CALL clFlush_stat(cl_command_queue q) {
  if (!launchCounting())
    return clFlush_raw(q);
  const double t0 = launchNowMs();
  const cl_int rc = clFlush_raw(q);
  LaunchStat::get().noteCall("clFlush", launchNowMs() - t0, 2);
  return rc;
}

cl_int CL_API_CALL clEnqueueReadBuffer_stat(
  cl_command_queue q, cl_mem b, cl_bool blocking, size_t off, size_t size,
  void *ptr, cl_uint n_wait, const cl_event *wait, cl_event *event) {
  if (!launchCounting())
    return clEnqueueReadBuffer_raw(q, b, blocking, off, size, ptr, n_wait, wait,
                                   event);
  const double t0 = launchNowMs();
  const cl_int rc = clEnqueueReadBuffer_raw(q, b, blocking, off, size, ptr,
                                            n_wait, wait, event);
  LaunchStat::get().noteCall(blocking ? "clEnqueueReadBuffer(blocking)"
                                      : "clEnqueueReadBuffer",
                             launchNowMs() - t0, blocking ? 1 : 0);
  return rc;
}

cl_int CL_API_CALL clEnqueueWriteBuffer_stat(
  cl_command_queue q, cl_mem b, cl_bool blocking, size_t off, size_t size,
  const void *ptr, cl_uint n_wait, const cl_event *wait, cl_event *event) {
  if (!launchCounting())
    return clEnqueueWriteBuffer_raw(q, b, blocking, off, size, ptr, n_wait,
                                    wait, event);
  const double t0 = launchNowMs();
  const cl_int rc = clEnqueueWriteBuffer_raw(q, b, blocking, off, size, ptr,
                                             n_wait, wait, event);
  LaunchStat::get().noteCall(blocking ? "clEnqueueWriteBuffer(blocking)"
                                      : "clEnqueueWriteBuffer",
                             launchNowMs() - t0, 0);
  return rc;
}

cl_int CL_API_CALL clEnqueueReadBufferRect_stat(
  cl_command_queue q, cl_mem b, cl_bool blocking, const size_t *boff,
  const size_t *hoff, const size_t *region, size_t brp, size_t bsp, size_t hrp,
  size_t hsp, void *ptr, cl_uint n_wait, const cl_event *wait,
  cl_event *event) {
  if (!launchCounting())
    return clEnqueueReadBufferRect_raw(q, b, blocking, boff, hoff, region, brp,
                                       bsp, hrp, hsp, ptr, n_wait, wait, event);
  const double t0 = launchNowMs();
  const cl_int rc =
    clEnqueueReadBufferRect_raw(q, b, blocking, boff, hoff, region, brp, bsp,
                                hrp, hsp, ptr, n_wait, wait, event);
  LaunchStat::get().noteCall(blocking ? "clEnqueueReadBufferRect(blocking)"
                                      : "clEnqueueReadBufferRect",
                             launchNowMs() - t0, blocking ? 1 : 0);
  return rc;
}

cl_int CL_API_CALL clEnqueueWriteBufferRect_stat(
  cl_command_queue q, cl_mem b, cl_bool blocking, const size_t *boff,
  const size_t *hoff, const size_t *region, size_t brp, size_t bsp, size_t hrp,
  size_t hsp, const void *ptr, cl_uint n_wait, const cl_event *wait,
  cl_event *event) {
  if (!launchCounting())
    return clEnqueueWriteBufferRect_raw(q, b, blocking, boff, hoff, region, brp,
                                        bsp, hrp, hsp, ptr, n_wait, wait,
                                        event);
  const double t0 = launchNowMs();
  const cl_int rc =
    clEnqueueWriteBufferRect_raw(q, b, blocking, boff, hoff, region, brp, bsp,
                                 hrp, hsp, ptr, n_wait, wait, event);
  LaunchStat::get().noteCall(blocking ? "clEnqueueWriteBufferRect(blocking)"
                                      : "clEnqueueWriteBufferRect",
                             launchNowMs() - t0, 0);
  return rc;
}

cl_int CL_API_CALL clEnqueueFillBuffer_stat(
  cl_command_queue q, cl_mem b, const void *pattern, size_t psz, size_t off,
  size_t size, cl_uint n_wait, const cl_event *wait, cl_event *event) {
  if (!launchCounting())
    return clEnqueueFillBuffer_raw(q, b, pattern, psz, off, size, n_wait, wait,
                                   event);
  const double t0 = launchNowMs();
  const cl_int rc =
    clEnqueueFillBuffer_raw(q, b, pattern, psz, off, size, n_wait, wait, event);
  LaunchStat::get().noteCall("clEnqueueFillBuffer", launchNowMs() - t0, 0);
  return rc;
}

void *CL_API_CALL clEnqueueMapBuffer_stat(cl_command_queue q, cl_mem b,
                                          cl_bool blocking, cl_map_flags flags,
                                          size_t off, size_t size,
                                          cl_uint n_wait, const cl_event *wait,
                                          cl_event *event, cl_int *err) {
  if (!launchCounting())
    return clEnqueueMapBuffer_raw(q, b, blocking, flags, off, size, n_wait,
                                  wait, event, err);
  const double t0 = launchNowMs();
  void *p = clEnqueueMapBuffer_raw(q, b, blocking, flags, off, size, n_wait,
                                   wait, event, err);
  LaunchStat::get().noteCall(blocking ? "clEnqueueMapBuffer(blocking)"
                                      : "clEnqueueMapBuffer",
                             launchNowMs() - t0, blocking ? 1 : 0);
  return p;
}

cl_int CL_API_CALL clEnqueueUnmapMemObject_stat(cl_command_queue q, cl_mem b,
                                                void *ptr, cl_uint n_wait,
                                                const cl_event *wait,
                                                cl_event *event) {
  if (!launchCounting())
    return clEnqueueUnmapMemObject_raw(q, b, ptr, n_wait, wait, event);
  const double t0 = launchNowMs();
  const cl_int rc = clEnqueueUnmapMemObject_raw(q, b, ptr, n_wait, wait, event);
  LaunchStat::get().noteCall("clEnqueueUnmapMemObject", launchNowMs() - t0, 0);
  return rc;
}

cl_int CL_API_CALL clEnqueueSVMMap_stat(cl_command_queue q, cl_bool blocking,
                                        cl_map_flags flags, void *ptr,
                                        size_t size, cl_uint n_wait,
                                        const cl_event *wait, cl_event *event) {
  if (!launchCounting())
    return clEnqueueSVMMap_raw(q, blocking, flags, ptr, size, n_wait, wait,
                               event);
  const double t0 = launchNowMs();
  const cl_int rc =
    clEnqueueSVMMap_raw(q, blocking, flags, ptr, size, n_wait, wait, event);
  LaunchStat::get().noteCall(blocking ? "clEnqueueSVMMap(blocking)"
                                      : "clEnqueueSVMMap",
                             launchNowMs() - t0, blocking ? 1 : 0);
  return rc;
}

cl_int CL_API_CALL clEnqueueSVMUnmap_stat(cl_command_queue q, void *ptr,
                                          cl_uint n_wait, const cl_event *wait,
                                          cl_event *event) {
  if (!launchCounting())
    return clEnqueueSVMUnmap_raw(q, ptr, n_wait, wait, event);
  const double t0 = launchNowMs();
  const cl_int rc = clEnqueueSVMUnmap_raw(q, ptr, n_wait, wait, event);
  LaunchStat::get().noteCall("clEnqueueSVMUnmap", launchNowMs() - t0, 0);
  return rc;
}

cl_int CL_API_CALL clWaitForEvents_stat(cl_uint n, const cl_event *evs) {
  if (!launchCounting())
    return clWaitForEvents_raw(n, evs);
  const double t0 = launchNowMs();
  const cl_int rc = clWaitForEvents_raw(n, evs);
  LaunchStat::get().noteCall("clWaitForEvents", launchNowMs() - t0, 1);
  return rc;
}

cl_int CL_API_CALL clEnqueueBarrierWithWaitList_stat(cl_command_queue q,
                                                     cl_uint n_wait,
                                                     const cl_event *wait,
                                                     cl_event *event) {
  if (!launchCounting())
    return clEnqueueBarrierWithWaitList_raw(q, n_wait, wait, event);
  const double t0 = launchNowMs();
  const cl_int rc = clEnqueueBarrierWithWaitList_raw(q, n_wait, wait, event);
  LaunchStat::get().noteCall("clEnqueueBarrierWithWaitList", launchNowMs() - t0,
                             0);
  return rc;
}

/** level 2 needs event profiling on the queue; the queue is created once */
cl_command_queue CL_API_CALL
clCreateCommandQueue_stat(cl_context ctx, cl_device_id dev,
                          cl_command_queue_properties props, cl_int *err) {
  if (g_launch_level >= 2)
    props |= CL_QUEUE_PROFILING_ENABLE;
  return clCreateCommandQueue_raw(ctx, dev, props, err);
}

} // namespace

/// Bind the ledger wrappers over the driver symbols loaded as *_raw.
void InstallLaunchStatWrappers() {
#define BindStat(function)                                                     \
  function = g_launch_level > 0 ? &function##_stat : function##_raw
  BindStat(clEnqueueNDRangeKernel);
  BindStat(clFinish);
  BindStat(clFlush);
  BindStat(clEnqueueReadBuffer);
  BindStat(clEnqueueWriteBuffer);
  BindStat(clEnqueueReadBufferRect);
  BindStat(clEnqueueWriteBufferRect);
  BindStat(clEnqueueFillBuffer);
  BindStat(clEnqueueMapBuffer);
  BindStat(clEnqueueUnmapMemObject);
  BindStat(clEnqueueSVMMap);
  BindStat(clEnqueueSVMUnmap);
  BindStat(clWaitForEvents);
  BindStat(clEnqueueBarrierWithWaitList);
  BindStat(clCreateCommandQueue);
#undef BindStat
  if (g_launch_level > 0)
    ml_logi("NNTR_CL_LAUNCH_STAT=%d: OpenCL launch ledger installed",
            g_launch_level);
}

bool clLaunchStatOn() { return g_launch_level > 0; }
void clLaunchStatBegin(const char *phase) {
  if (g_launch_level > 0)
    LaunchStat::get().begin(phase);
}
void clLaunchStatTick() {
  if (g_launch_level > 0)
    LaunchStat::get().tick();
}
void clLaunchStatDump(const char *phase) {
  if (g_launch_level > 0)
    LaunchStat::get().dump(phase);
}

} // namespace nntrainer::opencl
