// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   load_trace.h
 * @date   06 September 2026
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Env-gated stopwatch for the phases inside the weight load.
 *
 * `model->load_weight()` is 78% of a warm init on a handset, and until now it
 * was one number. This splits it into the file map, the byte read, the GPU
 * weight prebuild (fingerprint / pack lookup / permute / upload / image view)
 * and the page reaper, accumulating nanoseconds per slot across every loader
 * worker so a serial pole shows up as busy-time close to wall-time.
 *
 * Everything here is inert unless NNTR_LOAD_TRACE is set: `on()` is one
 * function-local static bool and every Scope's constructor returns
 * immediately when it is false.
 */
#ifndef __LOAD_TRACE_H__
#define __LOAD_TRACE_H__
#ifdef __cplusplus

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace nntrainer {
namespace load_trace {

enum Slot {
  PRESCAN = 0,  /**< graph walk that computes the per-weight file offsets */
  PACK_OPEN,    /**< v8c_pack::set_source: validate + mmap the .v8cpack */
  MAP,          /**< open + fstat + mmap of the weight file, per node */
  MADV,         /**< posix_madvise(SEQUENTIAL) on that mapping */
  NODE_READ,    /**< node->read(): file bytes -> tensor storage + prebuild */
  PREBUILD,     /**< the GPU weight build inside NODE_READ */
  FINGERPRINT,  /**< sampled source fnv used as the pack key */
  LOOKUP,       /**< pack index lookup */
  HIT_UPLOAD,   /**< cache hit: clEnqueueWriteBuffer straight from the pack */
  WBUF_CREATE,  /**< clCreateBuffer for the device weight */
  PERMUTE,      /**< CPU nibble repack + row-sum fold (cache miss only) */
  PACK_WRITE,   /**< tee of a derived chunk into the pack temp file */
  MISS_UPLOAD,  /**< cache miss: clEnqueueWriteBuffer of a derived chunk */
  AUX_CREATE,   /**< scale + row-sum device buffers */
  AUX_SUB,      /**< of which: the two clCreateSubBuffer calls (arena arm) */
  AUX_STAGE,    /**< of which: the arena reservation + the pair's host copy */
  IMAGE_VIEW,   /**< image2d-from-buffer view creation */
  DROP,         /**< madvise(DONTNEED) + munmap at the end of a node */
  T_GETDATA,    /**< Tensor::getData(): commit/validate the tensor storage */
  T_COPY,       /**< the payload memcpy from the mapping into that storage */
  CTX_CREATE,   /**< OpenCL platform/device/context/queue bring-up */
  KRN_BLAS,     /**< ClContext::initBlasClKernels */
  KRN_ATTN,     /**< ClContext::initAttentionClKernels (incl. the prewarms) */
  KRN_BIN_READ, /**< reading one kernel binary out of the on-disk cache */
  KRN_PROG_BIN, /**< clCreateProgramWithBinary + clBuildProgram */
  KRN_PROG_SRC, /**< clCreateProgramWithSource + clBuildProgram */
  KRN_OBJ,      /**< clCreateKernel off an already-built program */
  G_ADD,        /**< ccapi Model::compile: addLayer over the symbolic graph */
  G_COMPILE,    /**< NeuralNetwork::compile: graph realization */
  G_INIT,       /**< NeuralNetwork::initialize: the graph's own realization */
  G_ALLOC,      /**< NeuralNetwork::allocate: the tensor pools */
  WALL,         /**< wall clock of the worker fan-out (main thread) */
  N_SLOTS
};

inline const char *slot_name(int s) {
  static const char *n[N_SLOTS] = {
    "prescan",      "pack_open",    "map",         "madvise",
    "node_read",    "prebuild",     "fingerprint", "lookup",
    "hit_upload",   "wbuf_create",  "permute",     "pack_write",
    "miss_upload",  "aux_create",   "aux_sub",     "aux_stage",
    "image_view",   "drop",         "t_getdata",   "t_copy",
    "ctx_create",   "krn_blas",     "krn_attn",    "krn_bin_read",
    "krn_prog_bin", "krn_prog_src", "krn_obj",     "g_add",
    "g_compile",    "g_init",       "g_alloc",     "wall"};
  return (s >= 0 && s < N_SLOTS) ? n[s] : "?";
}

inline bool on() {
  // NNTR_INIT_TRACE turns it on too: the slots below the load are init phases
  // (context bring-up, kernel programs, graph compile) and the init dissection
  // asks for them by that name.
  static const bool v = std::getenv("NNTR_LOAD_TRACE") != nullptr ||
                        std::getenv("NNTR_INIT_TRACE") != nullptr;
  return v;
}

inline std::atomic<uint64_t> &slot_ns(int s) {
  static std::atomic<uint64_t> a[N_SLOTS];
  return a[s];
}
inline std::atomic<uint64_t> &slot_cnt(int s) {
  static std::atomic<uint64_t> a[N_SLOTS];
  return a[s];
}
inline std::atomic<uint64_t> &slot_bytes(int s) {
  static std::atomic<uint64_t> a[N_SLOTS];
  return a[s];
}

inline void add(int s, uint64_t ns, uint64_t bytes = 0) {
  if (!on())
    return;
  slot_ns(s).fetch_add(ns, std::memory_order_relaxed);
  slot_cnt(s).fetch_add(1, std::memory_order_relaxed);
  if (bytes)
    slot_bytes(s).fetch_add(bytes, std::memory_order_relaxed);
}

/** RAII stopwatch; `bytes` is charged to the slot when the scope closes. */
class Scope {
public:
  explicit Scope(int slot) : slot_(slot), enabled_(on()) {
    if (enabled_)
      t0_ = std::chrono::steady_clock::now();
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
  void bytes(uint64_t b) { bytes_ = b; }
  ~Scope() {
    if (!enabled_)
      return;
    const uint64_t ns =
      (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0_)
        .count();
    add(slot_, ns, bytes_);
  }

private:
  int slot_;
  bool enabled_;
  uint64_t bytes_ = 0;
  std::chrono::steady_clock::time_point t0_;
};

/** One line per non-empty slot, on stderr, at the end of the load. */
inline void dump(const char *tag) {
  if (!on())
    return;
  std::fprintf(stderr, "[load-trace] %s  (busy ms summed over all workers)\n",
               tag ? tag : "");
  for (int s = 0; s < N_SLOTS; ++s) {
    const uint64_t ns = slot_ns(s).load(std::memory_order_relaxed);
    const uint64_t c = slot_cnt(s).load(std::memory_order_relaxed);
    const uint64_t b = slot_bytes(s).load(std::memory_order_relaxed);
    if (ns == 0 && c == 0)
      continue;
    std::fprintf(stderr, "[load-trace]   %-12s %10.1f ms  n=%-6llu %.1f MiB\n",
                 slot_name(s), ns / 1e6, (unsigned long long)c, b / 1048576.0);
  }
  std::fflush(stderr);
}

} // namespace load_trace
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOAD_TRACE_H__ */
