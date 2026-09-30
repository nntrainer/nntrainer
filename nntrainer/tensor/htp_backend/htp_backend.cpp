// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP backend lifecycle implementation (HexKL micro-API FastRPC).
 */

#ifdef ENABLE_HEXKL

#include <htp_backend.h>

#include <nntrainer_log.h>

#include <string>

// remote_handle64, CDSP_DOMAIN_ID, remote_session_control -- Hexagon SDK,
// not the HexKL addon. nntr_hvx.h is generated from test/htp/nntr_hvx.idl
// by this directory's generate_stub.sh (the same IDL the DSP skel is built
// from).
#include <AEEStdErr.h>
#include <remote.h>
#include <rpcmem.h>

#include <nntr_hvx.h>

#include <climits>

namespace nntrainer {

HtpBackend &HtpBackend::global() {
  static HtpBackend instance;
  return instance;
}

HtpBackend::HtpBackend() {
  // Enables the unsigned-PD CDSP session the dev/bring-up skel needs. Not
  // fatal if it fails -- a signed production skel does not need it, and
  // nntr_hvx_open below is the real pass/fail signal either way.
  remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  int pd_err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                      &unsigned_pd, sizeof(unsigned_pd));
  if (pd_err != AEE_SUCCESS) {
    ml_logw("remote_session_control(unsigned PD) failed (err=0x%x); "
            "continuing -- a signed skel does not need it.",
            pd_err);
  }

  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  remote_handle64 h = 0;
  int err = nntr_hvx_open(uri.c_str(), &h);
  if (err != AEE_SUCCESS) {
    // Graceful disable: enabled_ stays false so supports_*() reports false
    // and callers fall back to CPU. Not fatal.
    ml_logw("nntr_hvx_open failed (err=0x%x); HTP backend disabled, falling "
            "back to CPU. Is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?",
            err);
    return;
  }

  handle_ = static_cast<uint64_t>(h);
  enabled_ = true;
  ml_logi("HTP backend: FastRPC session to libnntr_hvx_skel.so open");
}

void *HtpBackend::alloc_shared(size_t bytes) {
  // rpcmem_alloc takes an int; rpcmem_alloc2 (size_t) is not exported by
  // every device's libcdsprpc.so, and no single KV cache slab approaches
  // 2 GiB. RPCMEM_DEFAULT_FLAGS is the cached mapping: the CPU writes new
  // K/V rows into this memory every token, and FastRPC cleans the passed
  // range before each call.
  if (!enabled_ || bytes == 0 || bytes > static_cast<size_t>(INT_MAX)) {
    return nullptr;
  }
  void *block = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                             static_cast<int>(bytes));
  if (!block) {
    ml_logw("rpcmem_alloc(%zu bytes) failed; this buffer stays on the heap "
            "and FastRPC copies it per call",
            bytes);
  }
  return block;
}

void HtpBackend::free_shared(void *block) {
  if (block) {
    rpcmem_free(block);
  }
}

HtpBackend::~HtpBackend() {
  if (enabled_) {
    nntr_hvx_close(static_cast<remote_handle64>(handle_));
    enabled_ = false;
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
