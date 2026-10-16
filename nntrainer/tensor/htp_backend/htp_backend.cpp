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
#include <htp_fixed_opts.h>

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
  // [#260 r3] The FastRPC thread runs every call that does not ride the
  // dspqueue, the load-time MoE warm-up included. Its default stack is the
  // 16 KiB minimum; hexkl_mm_u8i4_moe_layer_run's frame alone is 14.6 KiB
  // since #4415's down ring joined ours, and the first scratch malloc that
  // grows the heap then overflowed it (DSP crash in apps_mem_request_map64,
  // 0x8000040d). 64 KiB as the dspqueue thread (nntr_hvx_dspq.c). Must
  // precede every other RPC of the process; not fatal if refused.
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

  // Poll-mode QoS: the host half of the measured 90 -> 3,900 us transport
  // spread (docs/htp_attention/34_fc_measured.md section4 item F; the
  // control call itself is test/unittest/htp_rpc_bench.h's
  // htp_set_latency_qos, ported here because this is the first production
  // caller -- every 34_fc_measured.md transport number assumes this is on,
  // and until now nothing in the model path turned it on). The DSP-side
  // half (DCVS/apptype vote) already runs unconditionally in
  // nntr_hvx_open's session setup; this is the other half of the same pair.
  // Best effort: an SDK or device without the control just keeps the old
  // interrupt-driven behavior, logged so a slow transport number is
  // explainable rather than silently misread as the kernel being slow.
  // NNTR_HTP_POLL_US: how long the host polls for the DSP's reply before
  // falling back to the interrupt wait. Measured on device (doc 51
  // section 2.20): at 100 us the MoE layer call's transport read 2.19 ms
  // and decode's 158 us; at 5000, 1.13 ms and 83 us, which is prefill
  // -30 ms and decode 20.6 -> 23.7 TPS. 10000 was refused by the driver
  // (the control call fails and the session falls back to PM QoS, which
  // the profile's first line reports as qos_mode=1), so 5000 is the
  // default and the ceiling between the two is still to be found. The
  // cost is a core spinning for up to 5 ms per call while it waits.
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
      return; // [#289] drop_shared freed it
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
