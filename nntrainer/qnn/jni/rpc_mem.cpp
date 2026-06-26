// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    rpc_mem.cpp
 * @date    21 May 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  haehun.yang <haehun.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Single source-of-truth loader for Hexagon RPC memory.
 */
#include "rpc_mem.h"

#include <dynamic_library_loader.h>
#include <nntrainer_log.h>

namespace nntrainer {

RpcMem::RpcMem() {
#ifdef ENABLE_QNN
  void *handle =
    DynamicLibraryLoader::loadLibrary("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    ml_loge("RpcMem: failed to dlopen libcdsprpc.so: %s",
            DynamicLibraryLoader::getLastError());
    return;
  }

  alloc_ = reinterpret_cast<RpcMemAllocFn_t>(
    DynamicLibraryLoader::loadSymbol(handle, "rpcmem_alloc"));
  free_ = reinterpret_cast<RpcMemFreeFn_t>(
    DynamicLibraryLoader::loadSymbol(handle, "rpcmem_free"));
  to_fd_ = reinterpret_cast<RpcMemToFdFn_t>(
    DynamicLibraryLoader::loadSymbol(handle, "rpcmem_to_fd"));

  if (alloc_ == nullptr || free_ == nullptr || to_fd_ == nullptr) {
    ml_loge("RpcMem: failed to resolve rpcmem_alloc/free/to_fd symbols");
  }
#endif
}

RpcMem &RpcMem::global() {
  static RpcMem instance;
  return instance;
}

} // namespace nntrainer
