// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon Tensor Processor) backend lifecycle.
 *
 * Process-wide owner of the HexKL micro-API FastRPC session. If the DSP
 * skel cannot be opened, the backend stays disabled and HtpComputeOps
 * falls back to CPU operations. The interface is dsp/nntr_hvx.idl.
 *
 * Do not open a HexKL macro-API session after this micro-API session;
 * that ordering permanently fails macro-API initialization.
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_BACKEND_H__
#define __HTP_BACKEND_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace nntrainer {

struct HtpRuntimeConfig;

/**
 * @class HtpBackend
 * @brief Process-wide owner of the HTP (HexKL micro-API/FastRPC) session.
 */
class HtpBackend {
public:
  /**
   * @brief Access the process-wide singleton. The first call attempts
   *        nntr_hvx_open() exactly once (thread-safe).
   */
  static HtpBackend &global();

  /** Open a fresh session after model teardown or a failed initial open. */
  void open();
  /** Run shutdown hooks and close the session; no inference may be active. */
  void close();

  /**
   * @brief Whether the HTP session is initialized and usable. When false,
   *        all HTP ops must defer to the CPU fallback.
   */
  bool enabled() const { return enabled_; }

  /**
   * @brief The FastRPC session handle every nntr_hvx_* call dispatches
   *        through (HtpComputeOps, from the first accelerated kernel
   *        onward). Only meaningful when enabled() is true.
   */
  uint64_t handle() const { return handle_; }

  /**
   * @brief Accepted latency QoS mode: 2 = poll, 1 = PM, 0 = neither
   *        (interrupt-driven). Included in transport profiling output.
   */
  int qosMode() const { return qos_mode_; }

  /**
   * @brief The poll-QoS window the driver accepted, in us (default 5000),
   *        or 0 when poll QoS was refused (qosMode() != 2):
   *        how long a FastRPC call spins before it blocks. The
   *        dspqueue MoE call spins its response wait for the same window.
   */
  uint32_t pollUs() const { return poll_us_; }

  /**
   * @brief DSP-visible memory: an rpcmem block, i.e. a dma-buf the rpcmem
   *        library registers with FastRPC as it allocates. Any range of it
   *        passed as a FastRPC buffer argument is then mapped into the DSP
   *        and cache-maintained instead of copied -- the KV cache goes
   *        here so attention never copies the used cache per call.
   * @return the block, or nullptr when the backend is disabled, @a bytes
   *         is 0 or above the rpcmem limit, or the allocation failed
   */
  void *alloc_shared(size_t bytes);

  /**
   * @brief Frees a block from alloc_shared(); nullptr is a no-op, and so is
   *        a block drop_shared() freed already.
   */
  void free_shared(void *block);

  /**
   * @brief Frees a block from alloc_shared() before its owner does
   *        (the prefill's KV cache once the DSP holds the rows); the owner's
   *        later free_shared() of it is then a no-op. A pointer that is not
   *        a live block is ignored.
   */
  void drop_shared(void *block);

  /**
   * @brief Runs fn in ~HtpBackend, in registration order, before
   *        the session is closed: the shutdown hook for state that makes
   *        RPC calls on the session (the dspqueue thread). fn must own
   *        what it touches; HtpComputeOps may already be destroyed.
   */
  void atClose(std::function<void()> fn) { at_close_.push_back(std::move(fn)); }
  /**
   * @brief As atClose, but after every atClose hook:
   *        for what the others' DSP threads use (S1's arena, released on
   *        the DSP and unmapped before the session closes).
   */
  void atCloseLast(std::function<void()> fn) {
    at_close_last_.push_back(std::move(fn));
  }

  ~HtpBackend();

  HtpBackend(const HtpBackend &) = delete;
  HtpBackend &operator=(const HtpBackend &) = delete;

private:
  HtpBackend();

  bool enabled_ = false;
  std::mutex shared_mu_;               /**< guards the two sets */
  std::unordered_set<void *> shared_;  /**< live alloc_shared blocks */
  std::unordered_set<void *> dropped_; /**< drop_shared()'s, till freed */
  uint64_t handle_ = 0; ///< remote_handle64 from nntr_hvx_open; opaque here
                        ///< so this header does not need <remote.h>.
  int qos_mode_ = 0;
  uint32_t poll_us_ = 0;
  std::vector<std::function<void()>> at_close_, at_close_last_;
};

/** Lifecycle hooks for a single exclusively owned HTP model. */
void htp_prepare_runtime();
void htp_prepare_runtime(const HtpRuntimeConfig &config);
void htp_begin_prompt();
void htp_stop_workers();
/** Call only after all model layers and shared KV buffers are destroyed. */
void htp_release_runtime();

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HTP_BACKEND_H__
