// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_buffer_pool.cpp
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of the device cl_mem plane.
 */

#include "cl_buffer_pool.h"

#include <CL/cl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unwind.h>
#endif

#include <cl_context.h>
#include <engine.h>
#include <nntrainer_log.h>
#include <opencl_loader.h>
#include <residency_policy.h>

namespace nntrainer {

namespace {

/** @brief the OpenCL context this process' gpu Context owns */
ClContext *clContext() {
  return static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
}

// ---------------------------------------------------------------------------
// NNTR_CLMEM_PLANE_CANARY -- which pooled planes does the HOST actually touch?
//
// Every planner offset the residency planner classified GPU_CLMEM gets a device
// cl_mem sub-buffer AND keeps its slice of the shared SVM plane, so the
// activation plane is charged twice. Dropping the shared half wholesale
// (NNTR_CLMEM_SKIP_SHARED=1) saves 96.6 MiB and
// changes the answer, because a handful of those offsets DO have a host reader
// or writer and the drop leaves it addressing nothing.
//
// This is the measurement that separates the two sets. Arm it and every
// candidate offset's shared slice is mprotect()ed; a host access faults, the
// handler names the offset and unprotects it (permanently, so the run
// completes), and the exit report lists touched vs untouched with bytes. The
// untouched set is exactly the set a correct skip may drop.
//
// The device is unaffected: it reaches these bytes through the kgsl mapping,
// not through this process' CPU page tables, so a fault here is a HOST access
// by construction -- the same property NNTR_QP_CANARY relies on.
//
//   =1  PROT_READ  -- host WRITES only
//   =2  PROT_NONE  -- any host access, read or write (this is the set that
//                     decides the skip; a read of a slice that does not exist
//                     is just as fatal as a write)
//
// Diagnostic only, off by default, and nothing is armed when the env is absent.
#ifndef _WIN32
struct PlaneCanaryRange {
  size_t offset;    /**< planner offset */
  size_t bytes;     /**< largest token the planner placed there */
  uintptr_t lo, hi; /**< page-aligned protected interior */
  int hits;         /**< host accesses seen */
  bool touched;     /**< at least one host access */
};

/** Fixed after arming; the handler only mutates hits/touched, never the
 *  container, so no allocation happens on the signal path. */
std::vector<PlaneCanaryRange> g_pc_ranges;
size_t g_pc_untouched_bytes = 0; /**< bytes still protected, for the report */
bool g_pc_armed = false;
struct sigaction g_pc_prev_segv {};
struct sigaction g_pc_prev_bus {};

int planeCanaryMode() {
  static const int m = [] {
    const char *e = std::getenv("NNTR_CLMEM_PLANE_CANARY");
    return (e != nullptr && e[0] != 0) ? std::atoi(e) : 0;
  }();
  return m;
}

_Unwind_Reason_Code pc_unwind_cb(struct _Unwind_Context *ctx, void *arg) {
  int *n = static_cast<int *>(arg);
  const uintptr_t pc = _Unwind_GetIP(ctx);
  if (pc == 0 || *n >= 20)
    return _URC_END_OF_STACK;
  Dl_info info{};
  char line[256];
  if (dladdr(reinterpret_cast<void *>(pc), &info) &&
      info.dli_sname != nullptr) {
    std::snprintf(line, sizeof(line), "[PLANE-CANARY]  #%02d %s + 0x%lx (%s)\n",
                  *n, info.dli_sname,
                  (unsigned long)((uintptr_t)pc - (uintptr_t)info.dli_saddr),
                  info.dli_fname ? info.dli_fname : "?");
  } else if (dladdr(reinterpret_cast<void *>(pc), &info)) {
    std::snprintf(line, sizeof(line), "[PLANE-CANARY]  #%02d +0x%lx (%s)\n", *n,
                  (unsigned long)((uintptr_t)pc - (uintptr_t)info.dli_fbase),
                  info.dli_fname ? info.dli_fname : "?");
  } else {
    std::snprintf(line, sizeof(line), "[PLANE-CANARY]  #%02d %p\n", *n,
                  (void *)pc);
  }
  ssize_t w = write(2, line, std::strlen(line));
  (void)w;
  ++(*n);
  return _URC_NO_REASON;
}

void pc_sigsegv(int sig, siginfo_t *si, void *uc) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(si->si_addr);
  for (auto &r : g_pc_ranges) {
    if (a < r.lo || a >= r.hi)
      continue;
    const bool first = !r.touched;
    r.touched = true;
    ++r.hits;
    /** Unprotect for the rest of the run: the question is WHICH offsets the
     *  host touches, and one fault per offset answers it. Leaving it protected
     *  would turn a hot loop into a fault storm and change what the run
     *  measures. */
    mprotect(reinterpret_cast<void *>(r.lo), (size_t)(r.hi - r.lo),
             PROT_READ | PROT_WRITE);
    if (first) {
      char head[256];
      std::snprintf(head, sizeof(head),
                    "\n[PLANE-CANARY] HOST access at %p -- planner offset %zu "
                    "(%.2f MiB), byte %ld of the slice\n",
                    si->si_addr, r.offset, r.bytes / 1048576.0,
                    (long)(a - r.lo));
      ssize_t w = write(2, head, std::strlen(head));
      (void)w;
      int n = 0;
      _Unwind_Backtrace(pc_unwind_cb, &n);
    }
    return;
  }

  /** Not one of ours: restore the previous disposition and let the real fault
   *  happen, so this diagnostic cannot swallow a genuine crash. */
  const struct sigaction *prev =
    (sig == SIGBUS) ? &g_pc_prev_bus : &g_pc_prev_segv;
  sigaction(sig, prev, nullptr);
  (void)uc;
}

struct PlaneCanaryReport {
  ~PlaneCanaryReport() {
    if (!g_pc_armed)
      return;
    size_t touched_bytes = 0, untouched_bytes = 0;
    int touched_n = 0;
    for (const auto &r : g_pc_ranges) {
      if (r.touched) {
        ++touched_n;
        touched_bytes += r.bytes;
        std::fprintf(
          stderr, "[PLANE-CANARY] TOUCHED offset %10zu  %8.2f MiB  hits=%d\n",
          r.offset, r.bytes / 1048576.0, r.hits);
      } else {
        untouched_bytes += r.bytes;
      }
    }
    for (const auto &r : g_pc_ranges)
      if (!r.touched)
        std::fprintf(stderr, "[PLANE-CANARY] clean   offset %10zu  %8.2f MiB\n",
                     r.offset, r.bytes / 1048576.0);
    std::fprintf(stderr,
                 "[PLANE-CANARY] mode=%d: %d of %zu candidate offsets touched "
                 "by the host (%.1f MiB); %zu clean (%.1f MiB droppable)\n",
                 planeCanaryMode(), touched_n, g_pc_ranges.size(),
                 touched_bytes / 1048576.0, g_pc_ranges.size() - touched_n,
                 untouched_bytes / 1048576.0);
    std::fflush(stderr);
    (void)g_pc_untouched_bytes;
  }
};
PlaneCanaryReport g_pc_report;
#endif // _WIN32

} // namespace

ClBufferPool::~ClBufferPool() { ClBufferPool::deallocate(); }

void ClBufferPool::recordPlannerLayout() {
  /** Record where the planner put each token. Tokens that share an offset have
   *  disjoint lifetimes, so one buffer sized to the largest of them backs all
   *  of them -- the device-side expression of the planner's reuse. */
  const auto &offsets = getMemoryOffset();
  const auto &sizes = getMemorySize();

  token_offset_.assign(offsets.begin(), offsets.end());
  offset_size_.clear();
  for (size_t i = 0; i < offsets.size(); ++i) {
    const size_t bytes = (i < sizes.size()) ? sizes[i] : 0;
    auto it = offset_size_.find(offsets[i]);
    if (it == offset_size_.end() || bytes > it->second)
      offset_size_[offsets[i]] = bytes;
  }
}

bool ClBufferPool::sharedSliceNeeded(size_t offset) const {
  return shared_slice_skipped_.find(offset) == shared_slice_skipped_.end();
}

void ClBufferPool::noteDeviceOnlyTokens(
  const std::vector<unsigned int> &tokens) {
  std::lock_guard<std::mutex> lk(device_mtx_);
  device_only_tokens_.assign(tokens.begin(), tokens.end());
}

void ClBufferPool::allocate() {
  /** The base MemoryPool::allocate() requests ONE contiguous clSVMAlloc of the
   *  whole plane. When that plane exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE,
   *  clSVMAlloc returns null and ClSVMAllocator falls back to plain host
   *  memory -- which the device cannot use as SVM at all: map/unmap return
   *  CL_INVALID_VALUE and every kernel reads zeros. The fallback is silent by
   *  design (correctness over speed for a host-only run), so on a GPU run it
   *  surfaces only as output collapsing to a single repeated token.
   *
   *  The per-offset cl_mem buffers below already dodge this cap; the SVM plane
   *  has to as well. Above the cap, take the per-offset (shared-objects)
   *  allocateFSU() path so every SVM buffer is a single tensor wide -- far
   *  under the cap -- and stays REAL SVM. A plane under the cap keeps the
   *  single-buffer path and is byte-identical to before.
   *
   *  A process with no registered gpu Context reports no cap and keeps the
   *  single-buffer path, which is the same answer it gave before: on such a
   *  run the SVM allocator is already the host fallback and there is nothing
   *  to protect. */
  cl_ulong plane_cap = 0;
  if (auto *cc = clContext())
    opencl::clGetDeviceInfo(cc->context_inst_.GetDeviceId(),
                            CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(plane_cap),
                            &plane_cap, nullptr);
  const bool over_cap =
    plane_cap > 0 && static_cast<cl_ulong>(size()) > plane_cap;

  /** NNTR_CLMEM_SKIP_SHARED -- what to do about the tensors that live on the
   *  device plane and nowhere else.
   *
   *   1           leave their shared slice unallocated,
   *   2           allocate per offset but skip nothing (the control arm that
   *               isolates the per-offset allocation shape from the skip),
   *   0           the pre-existing single-buffer plane.
   *
   *  Unset, the answer is the application's: ResidencyPolicy::skip_shared_slice
   *  means "I have declared every tensor of mine the host touches
   *  (host_plane_patterns), so drop the rest". No model in tree sets it, so the
   *  default is 0; the env overrides in both directions and stays the A/B arm.
   *
   *  Measured on an Adreno 840 device: the shared plane holds 132.5 MiB that
   *  the host touches 24.0 MiB of, because every offset the planner classified
   *  GPU_CLMEM gets a device cl_mem AND keeps its slice of the shared plane. On
   *  this driver an SVM allocation is a kernel-side mapping, so those bytes are
   *  charged to the GPU whether or not anything reads them.
   *
   *  WHAT QUALIFIES, and how it is known. The premise -- that a GPU_CLMEM
   *  tensor is one no host code touches -- is a statement about CONSUMERS, and
   *  the planner only checks consumers. A host PRODUCER that writes its output
   *  with CPU stores classifies device-only and is not: an embedding lookup
   *  does exactly that (EmbeddingLayer::incremental_forwarding stores into its
   *  own output from worker threads), and a model whose attention lowers an
   *  input back to the host (clmem_lower_cl -> clEnqueueReadBuffer into
   *  Tensor::getData()) reads one. With the slice skipped and neither declared,
   *  one measured model died 3/3 with "clmem_lower_cl: buffer read-back failed
   *  for layer0_attention:input0" and another survived but answered differently
   *  -- 1 363 bytes instead of the reference 821, in both of two runs and at
   *  both prompt lengths. So an application opts in by DECLARING what the host
   *  touches, and TensorPool::allocate() keeps the shared slice for those.
   *  NNTR_CLMEM_PLANE_CANARY is the instrument that checks a declaration is
   *  complete -- it mprotects exactly the set that would be dropped and names
   *  whoever faults. On the measured lane it reports 2 of 31 candidate offsets
   *  touched, both by the embedding, in prefill and decode and at 1 and 4
   *  chunks alike.
   *
   *  Even with a complete declaration this lever is NOT sound wherever the
   *  attention path binds a device-only tensor by SVM pointer rather than by
   *  handle: the OHWI GPU-RoPE in mha_core does that for Q/K/V, so a dropped
   *  slice hands a kernel a null SVM pointer, the device reads zeros through
   *  its own mapping and the model answers something else -- which no host-page
   *  canary can see, the access being a DEVICE read. That is why no model in
   *  tree sets skip_shared_slice, and why the plane's double charge is removed
   *  by aliasing the two planes instead (see devicePlaneBaseLocked).
   *
   *  Skipping needs the per-offset path: a hole cannot be left in one
   *  contiguous allocation. */
  const int skip_mode = [] {
    const char *e = std::getenv("NNTR_CLMEM_SKIP_SHARED");
    if (e != nullptr && e[0] != 0)
      return std::atoi(e);
    return ResidencyPolicy::global().skip_shared_slice ? 1 : 0;
  }();

  std::lock_guard<std::mutex> lk(device_mtx_);
  recordPlannerLayout();

  if (skip_mode == 1 && !device_only_tokens_.empty()) {
    /** An offset qualifies only if EVERY token the planner placed there is
     *  device-only. Tokens at one offset alias one region; one host-addressable
     *  tensor among them means the region has to exist on the shared plane. */
    std::unordered_set<size_t> device_only_offsets;
    for (unsigned int tok : device_only_tokens_) {
      const size_t i = tok - 1;
      if (i < token_offset_.size())
        device_only_offsets.insert(token_offset_[i]);
    }
    std::unordered_set<size_t> host_offsets;
    {
      std::unordered_set<unsigned int> dev_tok(device_only_tokens_.begin(),
                                               device_only_tokens_.end());
      for (size_t i = 0; i < token_offset_.size(); ++i)
        if (dev_tok.find(static_cast<unsigned int>(i + 1)) == dev_tok.end())
          host_offsets.insert(token_offset_[i]);
    }

    /** Create each candidate's device buffer HERE, before the shared plane is
     *  allocated, and skip the slice only for the ones that succeeded. Doing it
     *  the other way round -- skip first, ask the driver later -- leaves a
     *  tensor with no memory on either plane when clCreateBuffer says no. */
    for (size_t off : device_only_offsets) {
      if (host_offsets.find(off) != host_offsets.end())
        continue;
      if (createDeviceBufferLocked(off) != nullptr)
        shared_slice_skipped_.insert(off);
    }

    /** getMemory() refuses a pool that allocated nothing, and a graph whose
     *  every offset is device-only would produce exactly that. Give the
     *  largest offset its slice back rather than trip that check. */
    if (!shared_slice_skipped_.empty() &&
        shared_slice_skipped_.size() == offset_size_.size()) {
      size_t biggest = *shared_slice_skipped_.begin();
      for (size_t off : shared_slice_skipped_)
        if (offset_size_[off] > offset_size_[biggest])
          biggest = off;
      shared_slice_skipped_.erase(biggest);
    }

    size_t skipped_bytes = 0;
    for (size_t off : shared_slice_skipped_)
      skipped_bytes += offset_size_[off];
    /** stderr, not ml_logi: on Android ml_logi is __android_log_print, so this
     *  line would land in logcat and never in a run's captured output -- and a
     *  saving nobody can see in the run that made it is a claim, not a
     *  measurement. Same reason KVCacheManager prints its kv-share banner
     *  here. */
    std::fprintf(stderr,
                 "[clmempool] %zu of %zu planner offsets are device-only; "
                 "%.1f MB of shared plane not allocated\n",
                 shared_slice_skipped_.size(), offset_size_.size(),
                 skipped_bytes / 1048576.0);
    std::fflush(stderr);
  }

  if (over_cap) {
    ml_logi("ClBufferPool: the %.1f MB SVM plane exceeds the device maximum "
            "allocation %.1f MB; allocating it per offset (shared objects) so "
            "every buffer stays real SVM",
            size() / 1048576.0, plane_cap / 1048576.0);
    MemoryPool::allocateFSU();
  } else if (skip_mode != 0) {
    MemoryPool::allocateFSU();
  } else {
    MemoryPool::allocate();
  }

  /** Re-read the layout: allocateFSU()/allocate() do not change it, but
   *  recordPlannerLayout() before them is what the skip decision needed, and
   *  running it again keeps the post-condition ("the maps describe the plane
   *  that now exists") true from either path. */
  recordPlannerLayout();

  /** Under the GPU ledger, say how the two planes relate. The device plane is
   *  one buffer per planner offset, sized to the largest token there, so its
   *  TOTAL is the sum over offsets -- which is not the same number as the
   *  shared plane's span, and the ledger showing 310 MiB of device buffers
   *  against a 132 MiB shared plane is otherwise unreadable. */
  if (opencl::clMemAcctOn()) {
    size_t sum = 0, biggest = 0;
    cl_uint align_bits = 0;
    auto *cc_a = clContext();
    if (cc_a != nullptr)
      opencl::clGetDeviceInfo(cc_a->context_inst_.GetDeviceId(),
                              CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits),
                              &align_bits, nullptr);
    const size_t align = align_bits ? align_bits / 8u : 0u;
    size_t aligned = 0;
    for (const auto &kv : offset_size_) {
      sum += kv.second;
      biggest = std::max(biggest, kv.first + kv.second);
      if (align && kv.first % align == 0)
        ++aligned;
    }
    std::fprintf(stderr,
                 "[clmempool] shared plane %.1f MiB; %zu planner offsets, "
                 "sum %.1f MiB, top-of-plane %.1f MiB; sub-buffer align %zu B, "
                 "%zu/%zu offsets aligned\n",
                 size() / 1048576.0, offset_size_.size(), sum / 1048576.0,
                 biggest / 1048576.0, align, aligned, offset_size_.size());
    std::fflush(stderr);
  }

  registerPlaneCanaryCandidates();
}

/**
 * @brief NNTR_CLMEM_SUBBUF: carve the device plane out of one buffer.
 *
 * @details ON by default; =0 restores a private clCreateBuffer per offset in
 * the same binary, which is the arm the previous device baselines were taken
 * with.
 */
static bool clmemSubbufOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_CLMEM_SUBBUF");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

void *ClBufferPool::devicePlaneBaseLocked(size_t span) {
  if (device_plane_ != nullptr || device_plane_failed_)
    return device_plane_;

  auto *cc = clContext();
  if (cc == nullptr)
    return nullptr;

  cl_ulong max_alloc = 0;
  opencl::clGetDeviceInfo(cc->context_inst_.GetDeviceId(),
                          CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                          &max_alloc, nullptr);
  if (span == 0 || (max_alloc > 0 && static_cast<cl_ulong>(span) > max_alloc)) {
    device_plane_failed_ = true;
    return nullptr;
  }

  /** NNTR_CLMEM_ALIAS_SVM=1 -- one PHYSICAL plane, two views. Opt-in.
   *
   *  Opt-in, not default, because making the two views one plane also makes
   *  every undrained write through one view a real hazard for the other: what
   *  was a stale copy nobody read becomes the bytes the next kernel reads. On
   *  one driver the answer is not stable with this on. It stays off until the
   *  long-prefill correctness series that fixes exactly those drain hazards
   *  lands, and the default can be flipped then.
   *
   *  The double charge this pool is measured for is not one layout held twice
   *  by accident: it is two real allocations of the same bytes. The shared
   *  plane is a clSVMAlloc the attention path reads by pointer (mha_core binds
   *  query/key/value_step's SVM addresses, and dispatches the image attention
   *  with q_clmem = nullptr so qk reads the same plane), and the device plane
   *  is a separate clCreateBuffer the FC/v8c kernels bind as cl_mem. Dropping
   *  either one breaks whoever reads it.
   *
   *  So do not drop one -- make them the SAME bytes. CL_MEM_USE_HOST_PTR over
   *  the SVM pointer asks the driver to back this buffer with memory that
   *  already exists, and the per-offset sub-buffers are then windows on the
   *  very slices the SVM plane hands out. Every kernel reads exactly what it
   *  reads today, nothing is reclassified, and act:device_plane should vanish
   *  from the kgsl counter.
   *
   *  THE ASSUMPTION THAT HAS TO BE MEASURED: that the driver aliases rather
   *  than shadow-copies. CL_MEM_USE_HOST_PTR permits a copy, and a copy here
   *  is a net LOSS -- the same bytes a third time, plus a coherence problem.
   *  The GPU ledger and /sys/class/kgsl/kgsl/page_alloc answer it directly:
   *  aliasing shows act:device_plane charged and the kgsl counter NOT moving;
   *  a shadow copy shows both moving. Default OFF until that measurement
   *  exists on the device in question.
   *
   *  Alignment is already satisfied: every planner offset is a multiple of
   *  CL_DEVICE_MEM_BASE_ADDR_ALIGN (128 B on Adreno, 36 of 36 offsets), and
   *  clSVMAlloc returns page-aligned memory for a plane this size, which is
   *  what USE_HOST_PTR wants for a zero-copy mapping. */
  static const bool alias_svm = [] {
    const char *e = std::getenv("NNTR_CLMEM_ALIAS_SVM");
    return e != nullptr && e[0] != 0 && e[0] != '0';
  }();

  cl_int err = CL_SUCCESS;
  cl_mem base = nullptr;
  bool aliased = false;
  if (alias_svm) {
    void *host = getMemoryPoolAddress();
    if (host == nullptr) {
      ml_logw("ClBufferPool: NNTR_CLMEM_ALIAS_SVM=1 but there is no contiguous "
              "shared plane to alias (per-offset path); using a private plane");
    } else if (size() < span) {
      ml_logw("ClBufferPool: NNTR_CLMEM_ALIAS_SVM=1 but the shared plane "
              "(%zu B) is smaller than the device span (%zu B); using a "
              "private plane",
              size(), span);
    } else {
      /** A VIEW, not an allocation: these bytes are the shared plane's, which
       *  the ledger already counted as `svm tensor_pool`. Charging them again
       *  would make the ledger overstate the footprint by exactly what the
       *  aliasing saves -- and the app's Peak Mem is computed from the ledger,
       *  so the saving would never reach the number anyone reads. */
      opencl::ClMemAcctScope _acct_alias("act:device_plane_alias");
      opencl::ClMemAcctViewScope _acct_view;
      base = opencl::clCreateBufferT(cc->context_inst_.GetContext(),
                                     CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                     span, host, &err);
      if (err != CL_SUCCESS || base == nullptr) {
        ml_logw("ClBufferPool: USE_HOST_PTR over the SVM plane failed with %d; "
                "falling back to a private device plane",
                err);
        base = nullptr;
      } else {
        aliased = true;
        /** No zero-fill: the bytes ARE the shared plane's, which
         *  ClSVMAllocator::alloc() already memset once at model load
         *  (NNTR_SVM_ZERO). Filling here would be correct but redundant, and
         *  on an aliased buffer it would also race the host's own view. */
        std::fprintf(stderr,
                     "[clmempool] device plane ALIASED onto the SVM plane at "
                     "%p (%.1f MiB, USE_HOST_PTR). Check kgsl page_alloc: it "
                     "must NOT grow by this much, or the driver shadow-copied "
                     "and this lever is a loss.\n",
                     host, span / 1048576.0);
        std::fflush(stderr);
      }
    }
  }

  if (base == nullptr) {
    opencl::ClMemAcctScope _acct("act:device_plane");
    base = opencl::clCreateBufferT(cc->context_inst_.GetContext(),
                                   CL_MEM_READ_WRITE, span, nullptr, &err);
    if (err != CL_SUCCESS || base == nullptr) {
      ml_logw("ClBufferPool: device plane of %.1f MB failed with %d; falling "
              "back to one buffer per planner offset",
              span / 1048576.0, err);
      device_plane_failed_ = true;
      return nullptr;
    }
  }

  /** One fill for the whole plane, for the reason the per-offset fill below
   *  gives: a producer writes only the rows it has and an element-wise
   *  consumer reads the padded rows too. Doing it once over the span also
   *  covers the bytes between two offsets that no token claims.
   *
   *  Skipped when the plane is aliased: those bytes are the shared plane's and
   *  ClSVMAllocator::alloc() already zeroed them at model load. */
  const cl_uchar zero = 0;
  if (!aliased && opencl::clEnqueueFillBuffer(
                    cc->command_queue_inst_.GetCommandQueue(), base, &zero,
                    sizeof(zero), 0, span, 0, nullptr, nullptr) != CL_SUCCESS) {
    opencl::clReleaseMemObjectT(base);
    ml_logw("ClBufferPool: zero-filling the %.1f MB device plane failed; "
            "falling back to one buffer per planner offset",
            span / 1048576.0);
    device_plane_failed_ = true;
    return nullptr;
  }

  device_plane_ = static_cast<void *>(base);
  device_plane_bytes_ = span;
  return device_plane_;
}

void *ClBufferPool::createDeviceBufferLocked(size_t offset) {
  auto hit = offset_buffer_.find(offset);
  if (hit != offset_buffer_.end())
    return hit->second;

  auto sit = offset_size_.find(offset);
  if (sit == offset_size_.end() || sit->second == 0)
    return nullptr;
  const size_t bytes = sit->second;

  auto *cc = clContext();
  cl_device_id dev = cc->context_inst_.GetDeviceId();

  /** ---- The device plane as ONE buffer with a sub-buffer per offset. ----
   *
   *  A buffer per planner offset throws away the only thing the planner did:
   *  two tokens whose lifetimes are disjoint are given offsets whose byte
   *  ranges OVERLAP, and the shared plane gets that reuse for free because it
   *  is one region. Measured on the Adreno 840 gemma4 E2B cell -- 36 planner
   *  offsets, 328.5 MiB of per-offset sizes over a 132.5 MiB plane span -- so
   *  the private-buffer plane costs 2.5x what the layout it is copying does.
   *
   *  A sub-buffer allocates nothing; it is a window on the base. Binding one
   *  as a kernel argument is binding the region the planner assigned, and the
   *  aliasing it re-creates is exactly the aliasing the shared plane has
   *  always relied on -- same offsets, same disjoint lifetimes, same in-order
   *  queue sequencing the tokens that share a region.
   *
   *  Two conditions, both checked rather than assumed: the region origin must
   *  be a multiple of CL_DEVICE_MEM_BASE_ADDR_ALIGN (128 B on this device;
   *  36 of 36 offsets satisfy it), and it must fit inside the span. Anything
   *  that fails either -- or a driver that refuses the sub-buffer -- falls
   *  through to the private buffer below for that offset alone, so the plane
   *  degrades one tensor at a time instead of all at once. */
  if (clmemSubbufOn()) {
    cl_uint align_bits = 0;
    opencl::clGetDeviceInfo(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                            sizeof(align_bits), &align_bits, nullptr);
    const size_t align = align_bits ? align_bits / 8u : 0u;
    const size_t span = size();
    if (align != 0 && offset % align == 0 && bytes <= span &&
        offset <= span - bytes) {
      cl_mem base = static_cast<cl_mem>(devicePlaneBaseLocked(span));
      if (base != nullptr) {
        cl_buffer_region region{offset, bytes};
        cl_int serr = CL_SUCCESS;
        cl_mem sub = opencl::clCreateSubBufferT(base, CL_MEM_READ_WRITE,
                                                CL_BUFFER_CREATE_TYPE_REGION,
                                                &region, &serr);
        if (serr == CL_SUCCESS && sub != nullptr) {
          offset_buffer_[offset] = static_cast<void *>(sub);
          return offset_buffer_[offset];
        }
        ml_logw("ClBufferPool: clCreateSubBuffer at offset %zu (%zu B) failed "
                "with %d; this offset takes a private buffer",
                offset, bytes, serr);
      }
    }
  }

  /** A single allocation larger than the device can hold is not a device
   *  buffer at all. Report it and leave the tensor on the shared plane, which
   *  is where a buffer this size (a host-dequantized weight table) belongs
   *  anyway -- placing it nowhere would be worse than placing it there. */
  cl_ulong max_alloc = 0;
  opencl::clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc),
                          &max_alloc, nullptr);
  if (max_alloc > 0 && static_cast<cl_ulong>(bytes) > max_alloc) {
    ml_logw("ClBufferPool: %.1f MB exceeds the device maximum allocation "
            "%.1f MB; the tensor stays on the shared plane",
            bytes / 1048576.0, max_alloc / 1048576.0);
    return nullptr;
  }

  cl_int err = CL_SUCCESS;
  opencl::ClMemAcctScope _acct("act:device_plane");
  cl_mem buf = opencl::clCreateBufferT(cc->context_inst_.GetContext(),
                                       CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || buf == nullptr) {
    ml_logw("ClBufferPool: clCreateBuffer for %zu bytes failed with %d; the "
            "tensor stays on the shared plane",
            bytes, err);
    return nullptr;
  }

  /** Match the shared plane's zero-initialisation: a producer writes only the
   *  rows it has, and an element-wise consumer reads the padded rows too. The
   *  in-order queue orders this fill ahead of every kernel, since allocation
   *  precedes the first forward. */
  const cl_uchar zero = 0;
  if (opencl::clEnqueueFillBuffer(cc->command_queue_inst_.GetCommandQueue(),
                                  buf, &zero, sizeof(zero), 0, bytes, 0,
                                  nullptr, nullptr) != CL_SUCCESS) {
    opencl::clReleaseMemObjectT(buf);
    ml_logw("ClBufferPool: zero-filling a %zu byte device buffer failed; the "
            "tensor stays on the shared plane",
            bytes);
    return nullptr;
  }

  offset_buffer_[offset] = static_cast<void *>(buf);
  return offset_buffer_[offset];
}

/**
 * @brief Hand the plane canary the offsets a skip would drop.
 *
 * @details Same qualification as NNTR_CLMEM_SKIP_SHARED=1 -- an offset counts
 * only when EVERY token the planner put there is device-only -- so what the
 * canary protects is exactly what the skip would leave unallocated. Nothing is
 * protected yet: arming happens at the first forward (clPlaneCanaryArm), after
 * the initialisers and the weight load have had their legitimate go at the
 * plane.
 */
void ClBufferPool::registerPlaneCanaryCandidates() {
#ifndef _WIN32
  if (planeCanaryMode() == 0 || device_only_tokens_.empty())
    return;
  void *base = getMemoryPoolAddress();
  if (base == nullptr)
    return; /** allocateFSU() path: no contiguous plane to slice */

  std::unordered_set<size_t> device_only_offsets;
  for (unsigned int tok : device_only_tokens_) {
    const size_t i = tok - 1;
    if (i < token_offset_.size())
      device_only_offsets.insert(token_offset_[i]);
  }
  std::unordered_set<unsigned int> dev_tok(device_only_tokens_.begin(),
                                           device_only_tokens_.end());
  std::unordered_set<size_t> host_offsets;
  for (size_t i = 0; i < token_offset_.size(); ++i)
    if (dev_tok.find(static_cast<unsigned int>(i + 1)) == dev_tok.end())
      host_offsets.insert(token_offset_[i]);

  const long pg = sysconf(_SC_PAGESIZE);
  for (size_t off : device_only_offsets) {
    if (host_offsets.find(off) != host_offsets.end())
      continue;
    auto sit = offset_size_.find(off);
    if (sit == offset_size_.end() || sit->second == 0)
      continue;
    const uintptr_t p = reinterpret_cast<uintptr_t>(base) + off;
    const uintptr_t lo = (p + pg - 1) & ~(uintptr_t)(pg - 1);
    const uintptr_t hi = (p + sit->second) & ~(uintptr_t)(pg - 1);
    if (hi <= lo)
      continue; /** shorter than a page interior: nothing to protect */
    g_pc_ranges.push_back(PlaneCanaryRange{off, sit->second, lo, hi, 0, false});
  }
  std::fprintf(stderr,
               "[PLANE-CANARY] %zu candidate offsets registered on plane %p "
               "(mode %d); arming at the first forward\n",
               g_pc_ranges.size(), base, planeCanaryMode());
  std::fflush(stderr);
#endif
}

void clPlaneCanaryArm() {
#ifndef _WIN32
  const int mode = planeCanaryMode();
  if (mode == 0 || g_pc_armed || g_pc_ranges.empty())
    return;
  g_pc_armed = true;

  struct sigaction sa {};
  sa.sa_sigaction = pc_sigsegv;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, &g_pc_prev_segv);
  sigaction(SIGBUS, &sa, &g_pc_prev_bus);

  const int prot = (mode >= 2) ? PROT_NONE : PROT_READ;
  size_t armed_bytes = 0;
  size_t failed = 0;
  for (auto &r : g_pc_ranges) {
    if (mprotect(reinterpret_cast<void *>(r.lo), (size_t)(r.hi - r.lo), prot) !=
        0) {
      /** Treat a refusal as "host-touchable": an offset we could not watch
       *  must not be reported clean. */
      r.touched = true;
      ++failed;
      continue;
    }
    armed_bytes += r.bytes;
  }
  std::fprintf(stderr,
               "[PLANE-CANARY] armed %zu ranges (%.1f MiB) with prot=%s; "
               "%zu mprotect failures\n",
               g_pc_ranges.size() - failed, armed_bytes / 1048576.0,
               prot == PROT_NONE ? "NONE" : "READ", failed);
  std::fflush(stderr);
#endif
}

void *ClBufferPool::deviceMemory(unsigned int idx) {
  std::lock_guard<std::mutex> lk(device_mtx_);

  const size_t i = idx - 1;
  if (i >= token_offset_.size())
    return nullptr;

  return createDeviceBufferLocked(token_offset_[i]);
}

void ClBufferPool::deallocate() {
  {
    std::lock_guard<std::mutex> lk(device_mtx_);
    /** Sub-buffers first, then the base they are windows on: releasing the
     *  base while a sub-buffer still references it is the one ordering the
     *  runtime does not have to survive. */
    for (auto &entry : offset_buffer_)
      if (entry.second != nullptr)
        opencl::clReleaseMemObjectT(static_cast<cl_mem>(entry.second));
    offset_buffer_.clear();
    if (device_plane_ != nullptr) {
      opencl::clReleaseMemObjectT(static_cast<cl_mem>(device_plane_));
      device_plane_ = nullptr;
      device_plane_bytes_ = 0;
    }
    device_plane_failed_ = false;
    offset_size_.clear();
    token_offset_.clear();
    shared_slice_skipped_.clear();
    device_only_tokens_.clear();
  }
  MemoryPool::deallocate();
}

} // namespace nntrainer
