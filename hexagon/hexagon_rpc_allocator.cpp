// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_rpc_allocator.cpp
 * @date   31 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  See hexagon_rpc_allocator.h.
 */

#include <hexagon_rpc_allocator.h>

#include <nntrainer_log.h>

#include <dlfcn.h>

#include <cstdint>
#include <stdexcept>

namespace nntrainer {

namespace {

using RpcMemAllocFn = void *(*)(int, uint32_t, int);
using RpcMemFreeFn = void (*)(void *);
using RegisterPoolFn = int (*)(const void *, size_t);

// Hexagon system heap / default flags for rpcmem_alloc - same constants
// nntrainer/qnn/jni/rpc_mem.h uses (kRpcMemHeapIdSystem, kRpcMemDefaultFlags).
constexpr int kRpcMemHeapIdSystem = 25;
constexpr int kRpcMemDefaultFlags = 1;

struct RpcMemApi {
  RpcMemAllocFn alloc = nullptr;
  RpcMemFreeFn free = nullptr;
};

const RpcMemApi &get_rpcmem_api() {
  static RpcMemApi api = [] {
    void *handle = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      throw std::runtime_error(std::string("HexagonRpcAllocator: dlopen("
                                           "libcdsprpc.so) failed: ") +
                               dlerror());
    }

    auto sym = [handle](const char *name) {
      void *s = dlsym(handle, name);
      if (!s) {
        throw std::runtime_error(std::string("HexagonRpcAllocator: dlsym(") +
                                 name + ") failed: " + dlerror());
      }
      return s;
    };

    RpcMemApi a;
    a.alloc = reinterpret_cast<RpcMemAllocFn>(sym("rpcmem_alloc"));
    a.free = reinterpret_cast<RpcMemFreeFn>(sym("rpcmem_free"));
    return a;
  }();

  return api;
}

// libggml-hexagon.so is already dlopen'd (RTLD_GLOBAL) by
// hexagon_compute_ops.cpp's own bridge loader; dlopen-ing it again here just
// returns the same cached handle and shares the same nntr_htp_bridge_state
// singleton, so registering a pool here is visible to the GEMM calls that
// bridge makes. Registration failing here is not fatal - the bridge's own
// find_ext_pool() fallback keeps the memcpy path correct - so this logs and
// continues rather than throwing, unlike get_rpcmem_api() above.
RegisterPoolFn get_register_pool_fn() {
  static RegisterPoolFn fn = [] {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("HexagonRpcAllocator: dlopen(libggml-hexagon.so) failed: %s "
              "(zero-copy activation registration disabled, falling back to "
              "the bridge's memcpy path)",
              dlerror());
      return static_cast<RegisterPoolFn>(nullptr);
    }
    void *s = dlsym(handle, "nntr_htp_bridge_register_activation_pool");
    if (!s) {
      ml_logw("HexagonRpcAllocator: dlsym(nntr_htp_bridge_register_activation_"
              "pool) failed: %s (zero-copy activation registration disabled)",
              dlerror());
      return static_cast<RegisterPoolFn>(nullptr);
    }
    ml_logi("HexagonRpcAllocator: register_pool fn resolved at %p", s);
    return reinterpret_cast<RegisterPoolFn>(s);
  }();

  return fn;
}

} // namespace

HexagonRpcAllocator::HexagonRpcAllocator() {
  // Resolve (or throw) at construction, not on first alloc() - fail fast
  // during context initialization instead of mid-inference.
  get_rpcmem_api();
}

void HexagonRpcAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  ml_logi("HexagonRpcAllocator::alloc: size=%zu alignment=%zu", size, alignment);
  const RpcMemApi &api = get_rpcmem_api();
  void *p = api.alloc(kRpcMemHeapIdSystem, kRpcMemDefaultFlags,
                      static_cast<int>(size));
  if (!p) {
    throw std::runtime_error("HexagonRpcAllocator::alloc: rpcmem_alloc failed");
  }
  ml_logi("HexagonRpcAllocator::alloc: rpcmem_alloc returned %p", p);

  if (RegisterPoolFn register_pool = get_register_pool_fn()) {
    ml_logi("HexagonRpcAllocator::alloc: calling register_pool(%p, %zu)", p, size);
    if (register_pool(p, size) != 0) {
      ml_logw("HexagonRpcAllocator::alloc: nntr_htp_bridge_register_activation_"
              "pool failed for %p (size %zu) - falling back to the bridge's "
              "memcpy path for this pool",
              p, size);
    }
  } else {
    ml_logw("HexagonRpcAllocator::alloc: register_pool fn is null - "
            "libggml-hexagon.so not loaded yet?");
  }

  *ptr = p;
}

void HexagonRpcAllocator::free(void *ptr) {
  if (!ptr) {
    return;
  }
  get_rpcmem_api().free(ptr);
}

} // namespace nntrainer
