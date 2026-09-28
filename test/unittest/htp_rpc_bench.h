// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_rpc_bench.h
 * @date   08 Aug 2026
 * @brief  FastRPC transport controls the device benchmarks share
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The attention benchmark and the fully-connected benchmark are compared
 * against each other and against the same QNN table, so they have to run under
 * the same transport conditions -- poll-mode QoS and ION-backed payload
 * buffers, or neither. Keeping one definition is what makes that true by
 * construction instead of by remembering to keep two copies in step.
 *
 * Header-only and included by one translation unit each.
 */

#ifndef __NNTRAINER_HTP_RPC_BENCH_H__
#define __NNTRAINER_HTP_RPC_BENCH_H__

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <dlfcn.h>

#include <AEEStdErr.h>
#include <remote.h>

/** @brief Renders a FastRPC error as hex so the code is searchable. */
inline std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/**
 * @brief rpcmem, resolved at runtime.
 *
 * The functions live in the DEVICE's libcdsprpc.so, which exports them; the
 * SDK's link-time stub of the same library does not, and its librpcmem.a
 * hides in version-dependent paths -- both already broke one build each.
 * dlsym against the already-loaded process image depends on neither.
 */
struct RpcMemApi {
  void *(*alloc)(int heap, uint32_t flags, int size) = nullptr;
  void (*free_)(void *p) = nullptr;

  static const RpcMemApi &get() {
    static RpcMemApi api = [] {
      RpcMemApi a;
      void (*init)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "rpcmem_init");
      a.alloc =
        (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
      a.free_ = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
      if (a.alloc == nullptr || a.free_ == nullptr) {
        a.alloc = nullptr;
        a.free_ = nullptr;
      } else if (init != nullptr) {
        init();
      }
      return a;
    }();
    return api;
  }
};

/** rpcmem.h's values, restated because that header is deliberately not
 * included (see RpcMemApi). Stable ABI constants, not tunables. */
constexpr int kRpcHeapIdSystem = 25;
constexpr uint32_t kRpcFlagsDefault = 1; /* RPCMEM_DEFAULT_FLAGS: cached */

/**
 * @brief A buffer the FastRPC driver can map once instead of per call.
 *
 * rpcmem/ION-backed memory is recognized by the driver and keeps its SMMU
 * mapping across calls; plain heap is pinned and mapped on EVERY call,
 * which is part of the measured per-call transport. Plain heap remains the
 * fallback when the device library exports no rpcmem -- ion says which one
 * this run actually got, and the tests report it.
 */
struct RpcBuf {
  void *p = nullptr;
  bool ion = false;
  explicit RpcBuf(size_t bytes) {
    const RpcMemApi &api = RpcMemApi::get();
    if (api.alloc != nullptr) {
      p = api.alloc(kRpcHeapIdSystem, kRpcFlagsDefault, (int)bytes);
      ion = (p != nullptr);
    }
    if (p == nullptr) {
      p = std::malloc(bytes);
    }
  }
  ~RpcBuf() {
    if (ion) {
      RpcMemApi::get().free_(p);
      return;
    }
    std::free(p);
  }
  RpcBuf(const RpcBuf &) = delete;
  RpcBuf &operator=(const RpcBuf &) = delete;
};

/**
 * @brief Asks the FastRPC driver to POLL for completion instead of sleeping
 *        on the interrupt.
 *
 * The interrupt path is the host half of the measured 90 -> 3,900 us transport
 * spread: an idle CPU parks in a deep C-state and the wakeup pays for it. Best
 * effort -- an SDK or device without the control just keeps the old behavior.
 *
 * These identifiers are ENUMS in remote.h, not #defines; an earlier
 * #if defined() guard around this compiled the whole thing out and then
 * reported "not in this SDK" for an SDK that has them.
 *
 * @return 2 poll mode, 1 PM mode, 0 every mode rejected. Report it: a run's
 *         transport numbers can only be read knowing which mode produced them.
 */
inline int htp_set_latency_qos(remote_handle64 h) {
  struct remote_rpc_control_latency lat;
  std::memset(&lat, 0, sizeof(lat));
  lat.enable = RPC_POLL_QOS;
  lat.latency = 100;
  int rc =
    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
  if (rc == AEE_SUCCESS) {
    return 2;
  }
  std::cout << "  (poll QoS rejected: " << hex(rc) << ")\n";
  std::memset(&lat, 0, sizeof(lat));
  lat.enable = RPC_PM_QOS;
  lat.latency = 100;
  rc = remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
  if (rc == AEE_SUCCESS) {
    return 1;
  }
  std::cout << "  (PM QoS rejected too: " << hex(rc) << ")\n";
  return 0;
}

/** @brief Enables the unsigned PD on the CDSP domain. */
inline int htp_enable_unsigned_pd() {
  remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &unsigned_pd,
                                sizeof(unsigned_pd));
}

#endif /* __NNTRAINER_HTP_RPC_BENCH_H__ */
