// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_rpcmem.h
 * @brief  rpcmem/ION-backed scratch buffer for the HTP weight-registration
 *         path.
 *
 * Ported from test/unittest/htp_rpc_bench.h's RpcBuf/RpcMemApi (a test-only
 * header) so production code can use the same allocator without depending
 * on a test target. The logic is unchanged: resolve rpcmem_alloc/free via
 * dlsym against the process's already-loaded libcdsprpc.so (the SDK's
 * link-time stub does not export them), fall back to plain heap when the
 * device library has none.
 *
 * Why this exists at all: plain heap memory is pinned and mapped on EVERY
 * FastRPC call, which is part of the measured per-call transport; rpcmem/ION
 * memory is recognized by the driver and keeps its SMMU mapping across
 * calls. This branch's own harness already measured the gap
 * (docs/htp_attention/34_fc_measured.md section4 item F: 44.6 GB/s sustained
 * with ION-backed payload buffers) -- the LFM2.5-8B-A1B MoE FFN's own weight
 * registration transport measured at ~155 MB/s before this, on plain heap.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 */
#ifndef __NNTRAINER_HTP_RPCMEM_H__
#define __NNTRAINER_HTP_RPCMEM_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>

namespace nntrainer {

/**
 * @brief rpcmem, resolved at runtime -- lives in the device's libcdsprpc.so,
 *        which the process already has loaded via HtpBackend's FastRPC
 *        session by the time any weight is registered.
 */
struct HtpRpcMemApi {
  void *(*alloc)(int heap, uint32_t flags, int size) = nullptr;
  void (*free_)(void *p) = nullptr;

  static const HtpRpcMemApi &get() {
    static HtpRpcMemApi api = [] {
      HtpRpcMemApi a;
      void (*init)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "rpcmem_init");
      a.alloc =
        (void *(*)(int, uint32_t, int))dlsym(RTLD_DEFAULT, "rpcmem_alloc");
      a.free_ = (void (*)(void *))dlsym(RTLD_DEFAULT, "rpcmem_free");
      if (a.alloc == nullptr || a.free_ == nullptr) {
        // Partial API is not usable -- treat as absent rather than mixing
        // an rpcmem_alloc with a libc free or vice versa.
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

constexpr int HTP_RPC_HEAP_ID_SYSTEM = 25;    // rpcmem.h RPCMEM_HEAP_ID_SYSTEM
constexpr uint32_t HTP_RPC_FLAGS_DEFAULT = 1; // rpcmem.h RPCMEM_DEFAULT_FLAGS

/**
 * @brief A byte buffer the FastRPC driver can map once instead of per call.
 *        Falls back to plain heap when the device library exports no
 *        rpcmem (older device libraries, or a host build).
 */
class HtpRpcBuffer {
public:
  explicit HtpRpcBuffer(size_t bytes) : bytes_(bytes) {
    const HtpRpcMemApi &api = HtpRpcMemApi::get();
    if (api.alloc != nullptr) {
      data_ = static_cast<uint8_t *>(api.alloc(HTP_RPC_HEAP_ID_SYSTEM,
                                               HTP_RPC_FLAGS_DEFAULT,
                                               static_cast<int>(bytes)));
      ion_ = (data_ != nullptr);
    }
    if (data_ == nullptr) {
      data_ = static_cast<uint8_t *>(std::malloc(bytes));
      ion_ = false;
    }
  }

  ~HtpRpcBuffer() {
    if (ion_) {
      HtpRpcMemApi::get().free_(data_);
    } else {
      std::free(data_);
    }
  }

  HtpRpcBuffer(const HtpRpcBuffer &) = delete;
  HtpRpcBuffer &operator=(const HtpRpcBuffer &) = delete;

  uint8_t *data() { return data_; }
  size_t size() const { return bytes_; }
  /** @brief Whether this buffer actually landed on rpcmem/ION, for the
   *  profile dump to report which transport a run actually got. */
  bool isIon() const { return ion_; }

private:
  uint8_t *data_ = nullptr;
  size_t bytes_ = 0;
  bool ion_ = false;
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __NNTRAINER_HTP_RPCMEM_H__
