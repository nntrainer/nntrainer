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
 * Process-wide singleton owning the one HexKL micro-API FastRPC session a
 * process opens: nntr_hvx_open against libnntr_hvx_skel.so, the skel
 * test/htp/build.sh builds from test/htp/nntr_hvx.idl. This directory's
 * generate_stub.sh produces the matching client stub from the same IDL, so
 * the interface has one source of truth. If the skel is not reachable
 * (not on ADSP_LIBRARY_PATH, no device, driver error) construction leaves
 * the backend DISABLED and every HtpComputeOps::supports_*() reports
 * false, so callers transparently take the CPU path.
 *
 * This used to own a HexKL CPU Macro API (sdkl.h / libsdkl.so) session
 * instead. Nothing on this branch computed through it, and keeping it was
 * a standing hazard rather than a convenience: once any process has run
 * the micro API's hexkl_micro_hw_init (which the skel does in its open),
 * a macro-API session fails permanently -- measured on the prior attention
 * branch, see docs/backend_guide/htp_backend/20_hmx_flash_attention_plan.md.
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_BACKEND_H__
#define __HTP_BACKEND_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <cstdint>

namespace nntrainer {

/**
 * @class HtpBackend
 * @brief Process-wide owner of the HexKL micro-API FastRPC session.
 */
class HtpBackend {
public:
  /**
   * @brief Access the process-wide singleton. The first call attempts
   *        nntr_hvx_open() exactly once (thread-safe).
   */
  static HtpBackend &global();

  /**
   * @brief Whether the HTP session is open and usable. When false, all
   *        HTP ops must defer to the CPU fallback.
   */
  bool enabled() const { return enabled_; }

  /**
   * @brief The FastRPC session handle every nntr_hvx_* call dispatches
   *        through. Only meaningful when enabled() is true.
   */
  uint64_t handle() const { return handle_; }

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
   * @brief Frees a block from alloc_shared(); nullptr is a no-op.
   */
  void free_shared(void *block);

  ~HtpBackend();

  HtpBackend(const HtpBackend &) = delete;
  HtpBackend &operator=(const HtpBackend &) = delete;

private:
  HtpBackend();

  bool enabled_ = false;
  uint64_t handle_ = 0;
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HTP_BACKEND_H__
