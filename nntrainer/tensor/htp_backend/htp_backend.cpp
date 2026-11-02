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
#include <htp_backend/htp_fixed_opts.h>

#include <nntrainer_log.h>

#include <cstdlib>
#include <cstring>
#include <string>

// remote_handle64, CDSP_DOMAIN_ID, remote_session_control -- Hexagon SDK,
// not the HexKL addon. nntr_hvx.h is generated from test/htp/nntr_hvx.idl by
// this directory's meson.build (the same IDL the DSP skel is built from).
#include <remote.h>

#include <nntr_hvx.h>

#include <climits>

#include <htp_rpcmem.h>

namespace nntrainer {

HtpBackend &HtpBackend::global() {
  static HtpBackend instance;
  return instance;
}

bool HtpBackend::e2eRequested() {
  // built in: the one-PD decode graph is the HTP path's configuration
  return htp_fixed::E2E;
}

HtpBackend::HtpBackend() { open(); }

void HtpBackend::open() {
  if (enabled_)
    return;
  // The MoE frame plus scratch allocation can overflow the default 16 KiB
  // FastRPC stack. Match the dspqueue thread's 64 KiB before any RPC.
  // Failure is logged but does not prevent opening the session.
  remote_rpc_thread_params thread_params = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  int tp_err = remote_session_control(FASTRPC_THREAD_PARAMS, &thread_params,
                                      sizeof(thread_params));
  if (tp_err != AEE_SUCCESS)
    ml_logw("remote_session_control(thread stack 64 KiB) failed (err=%d)",
            tp_err);

  // Enables the unsigned-PD CDSP session the dev/bring-up skel needs.
  // Not fatal if it fails -- a signed production skel does not need it,
  // and nntr_hvx_open below is the real pass/fail signal either way.
  remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  int pd_err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                      &unsigned_pd, sizeof(unsigned_pd));
  if (pd_err != AEE_SUCCESS) {
    ml_logw("remote_session_control(unsigned PD) failed (err=%d); "
            "continuing -- a signed skel does not need it.",
            pd_err);
  }

  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  remote_handle64 h = 0;
  int err = nntr_hvx_open(uri.c_str(), &h);
  if (err != AEE_SUCCESS) {
    // Graceful disable: leave enabled_ = false so supports_*() reports
    // false and callers fall back to CPU. Not fatal.
    ml_logw("nntr_hvx_open failed (err=%d); HTP backend disabled, falling "
            "back to CPU. Is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?",
            err);
    return;
  }

  handle_ = static_cast<uint64_t>(h);
  enabled_ = true;

  // Poll before blocking to reduce interrupt-wake latency. The default
  // window is 5000 us; NNTR_HTP_POLL_US overrides it. If polling is refused,
  // try PM QoS, then retain interrupt-driven behavior and report the mode.
  struct remote_rpc_control_latency lat;
  std::memset(&lat, 0, sizeof(lat));
  lat.enable = RPC_POLL_QOS;
  lat.latency = 5000;
  if (const char *poll_us = std::getenv("NNTR_HTP_POLL_US")) {
    lat.latency = static_cast<uint32_t>(std::strtoul(poll_us, nullptr, 10));
    ml_logi("HtpBackend: poll QoS latency %u us (NNTR_HTP_POLL_US)",
            static_cast<unsigned>(lat.latency));
  }
  int qos_err =
    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
  if (qos_err == AEE_SUCCESS) {
    qos_mode_ = 2;
    poll_us_ = lat.latency;
  } else {
    std::memset(&lat, 0, sizeof(lat));
    lat.enable = RPC_PM_QOS;
    lat.latency = 100;
    qos_err =
      remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
    qos_mode_ = (qos_err == AEE_SUCCESS) ? 1 : 0;
  }
  if (qos_mode_ == 0) {
    ml_logw("HtpBackend: poll and PM latency QoS both rejected (err=%d); "
            "transport will pay the interrupt-wake tail (34_fc_measured.md "
            "section4 item F).",
            qos_err);
  }
}

void *HtpBackend::alloc_shared(size_t bytes) {
  // rpcmem_alloc takes an int, and no single KV cache slab approaches
  // 2 GiB. The default flags are the cached mapping: the CPU writes new K/V
  // rows into this memory every token, and FastRPC cleans the passed range
  // before each call. rpcmem comes through HtpRpcMemApi (dlsym) like every
  // other rpcmem user here, so libnntrainer.so gains no import for it.
  const HtpRpcMemApi &api = HtpRpcMemApi::get();
  if (!enabled_ || api.alloc == nullptr || bytes == 0 ||
      bytes > static_cast<size_t>(INT_MAX)) {
    return nullptr;
  }
  void *block = api.alloc(HTP_RPC_HEAP_ID_SYSTEM, HTP_RPC_FLAGS_DEFAULT,
                          static_cast<int>(bytes));
  if (!block) {
    ml_logw("rpcmem_alloc(%zu bytes) failed; this buffer stays on the heap "
            "and FastRPC copies it per call",
            bytes);
  } else {
    std::lock_guard<std::mutex> lock(shared_mu_);
    shared_.insert(block);
  }
  return block;
}

void HtpBackend::free_shared(void *block) {
  if (!block)
    return;
  {
    std::lock_guard<std::mutex> lock(shared_mu_);
    if (dropped_.erase(block) != 0)
      return; // drop_shared freed it
    shared_.erase(block);
  }
  HtpRpcMemApi::get().free_(block);
}

void HtpBackend::drop_shared(void *block) {
  {
    std::lock_guard<std::mutex> lock(shared_mu_);
    if (!block || shared_.erase(block) == 0)
      return;
    dropped_.insert(block);
  }
  HtpRpcMemApi::get().free_(block);
}

HtpBackend::~HtpBackend() { close(); }

void HtpBackend::close() {
  if (enabled_) {
    for (auto &fn : at_close_) {
      fn();
    }
    for (auto &fn : at_close_last_) {
      fn();
    }
    nntr_hvx_close(static_cast<remote_handle64>(handle_));
    enabled_ = false;
  }
  at_close_.clear();
  at_close_last_.clear();
  handle_ = 0;
  qos_mode_ = 0;
  poll_us_ = 0;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
