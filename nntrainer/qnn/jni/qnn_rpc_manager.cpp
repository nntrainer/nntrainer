// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_rpc_manager.cpp
 * @date    06 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains qnn rpc memory manager
 */
#include "qnn_rpc_manager.h"
#include "PAL/DynamicLoading.hpp"
#include "QnnTypes.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <nntrainer_log.h>
#include <utility>

namespace nntrainer {

template <class T>
static inline T resolveSymbol(void *libHandle, const char *sym) {
  T ptr = (T)pal::dynamicloading::dlSym(libHandle, sym);
  if (ptr == nullptr) {
    ml_loge("Unable to access symbol %s. pal::dynamicloading::dlError()", sym);
  }
  return ptr;
}

QNNRpcManager::QNNRpcManager() {
#ifdef ENABLE_QNN
  // load libcdsprpc.so
  void *libCdspHandle = pal::dynamicloading::dlOpen(
    "libcdsprpc.so",
    pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
  if (nullptr == libCdspHandle) {
    ml_loge("dlopen libcdsprpc.so failed");
    exit(-1);
  }

  rpcmem_alloc = (RpcMemAllocFn_t)dlsym(libCdspHandle, "rpcmem_alloc");
  rpcmem_free = (RpcMemFreeFn_t)dlsym(libCdspHandle, "rpcmem_free");
  rpcmem_to_fd = (RpcMemToFdFn_t)dlsym(libCdspHandle, "rpcmem_to_fd");

  if (nullptr == rpcmem_alloc || nullptr == rpcmem_free ||
      nullptr == rpcmem_to_fd) {
    dlclose(libCdspHandle);
    ml_loge("dlsym failed");
    exit(-1);
  }
#endif
  // Get QNN Interface
  void *libBackendHandle = pal::dynamicloading::dlOpen(
    "libQnnHtp.so",
    pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_GLOBAL);
  QnnInterfaceGetProvidersFn_t getInterfaceProviders{nullptr};
  getInterfaceProviders = resolveSymbol<QnnInterfaceGetProvidersFn_t>(
    libBackendHandle, "QnnInterface_getProviders");
  QnnInterface_t **interfaceProviders{nullptr};
  uint32_t numProviders{0};
  if (QNN_SUCCESS !=
      getInterfaceProviders((const QnnInterface_t ***)&interfaceProviders,
                            &numProviders)) {
    ml_loge("Failed to get interface providers.");
  }
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    if (QNN_API_VERSION_MAJOR ==
          interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <=
          interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
      qnnInterface_ = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
}

QNNRpcManager::~QNNRpcManager() {
#ifdef ENABLE_QNN
  // Deregister + free each entry. The map itself is destroyed when this
  // object dies, so we must NOT erase entries while iterating (range-for
  // would invalidate the iterator on erase) — just leave the map intact
  // and let it deallocate normally.
  for (auto &mem : ptrToFdAndMemHandleMap_) {
    Qnn_ErrorHandle_t deregisterRet =
      qnnInterface_.memDeRegister(&mem.second.second.second, 1);
    if (QNN_SUCCESS != deregisterRet) {
      ml_loge("qnnInterface_.memDeRegister failed");
    }
    rpcmem_free(mem.first);
  }
#endif
}

void QNNRpcManager::alloc(void **ptr, size_t size, size_t alignment) {
  assert(size > 0);
#ifdef DEBUGPRINT
  std::cout << "QNN alloc size: " << size << std::endl;
#endif
#ifdef ENABLE_QNN
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1
  // Allocate the shared buffer
  uint8_t *memPointer =
    (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
  if (nullptr == memPointer) {
    ml_loge("rpcmem_alloc failed");
    exit(-1);
  }
  qnnMemPtrMap_.insert(memPointer);

  *ptr = memPointer;
#else
  *ptr = calloc(size, 1);
  assert(ptr != nullptr);
#endif
}

bool QNNRpcManager::findMatchingPtr(void *ptr, Qnn_ContextHandle_t &context,
                                    Qnn_Tensor_t &qnnTensor) {
  auto mapIt = ptrToFdAndMemHandleMap_.find(ptr);

  if (mapIt != ptrToFdAndMemHandleMap_.end()) {
    if (mapIt->second.first == context) {
      qnnTensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
      qnnTensor.v1.memHandle = mapIt->second.second.second;
      return true;
    }
  }
  return false;
}

void QNNRpcManager::registerQnnTensor(void *ptr, Qnn_Tensor_t &qnnTensor,
                                      Qnn_ContextHandle_t &context_) {

  auto tensor_reg_start = std::chrono::system_clock::now();
  // auto it = qnnMemPtrMap_.find(ptr);
  // if (it == qnnMemPtrMap_.end()) {
  //   ml_loge("Ptr is not resistered. Registering");
  //   exit(-1);
  //   return;
  // }

  if (findMatchingPtr(ptr, context_, qnnTensor)) {
    return;
  }

  auto start = std::chrono::system_clock::now();
  int memFd = rpcmem_to_fd(ptr);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;

  if (-1 == memFd) {
    ml_loge("rpcmem_to_fd failed");
    exit(-1);
    return;
  }

  Qnn_MemDescriptor_t memDescriptor = QNN_MEM_DESCRIPTOR_INIT;
  memDescriptor.memShape = {qnnTensor.v1.rank, qnnTensor.v1.dimensions,
                            nullptr};
  memDescriptor.dataType = qnnTensor.v1.dataType;
  memDescriptor.memType = QNN_MEM_TYPE_ION;
  memDescriptor.ionInfo.fd = memFd;
  qnnTensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;

  Qnn_ErrorHandle_t registRet = qnnInterface_.memRegister(
    context_, &memDescriptor, 1u, &(qnnTensor.v1.memHandle));
  if (registRet != QNN_SUCCESS) {
    ml_loge("qnnInterface memRegister failed");
    exit(-1);
    return;
  }

  ptrToFdAndMemHandleMap_.insert(std::make_pair(
    ptr,
    std::make_pair(context_, std::make_pair(memFd, qnnTensor.v1.memHandle))));
  auto tensor_reg_end = std::chrono::system_clock::now();
  std::chrono::duration<double> tensor_reg_elapsed_seconds =
    tensor_reg_end - tensor_reg_start;
  // std::cout << "finished rpcmem_to_fd "
  //           << "elapsed time: " << elapsed_seconds.count() << "s\n";
  // std::cout << "finished tensor_registration "
  //           << "elapsed time: " << tensor_reg_elapsed_seconds.count() <<
  //           "s\n";
}

void QNNRpcManager::deRegisterQnnTensor() {
#ifdef ENABLE_QNN
  // free all buffers if it's not being used
  for (auto &mem : ptrToFdAndMemHandleMap_) {
    Qnn_ErrorHandle_t deregisterRet =
      qnnInterface_.memDeRegister(&mem.second.second.second, 1);
    if (QNN_SUCCESS != deregisterRet) {
      ml_loge("qnnInterface_.memDeRegister failed");
    }
    // rpcmem_free(mem.first);
    // clear the map outside the loop.
    // ptrToFdAndMemHandleMap_.erase(mem.first);
  }
  ptrToFdAndMemHandleMap_.clear();
#endif
}

void QNNRpcManager::free(void *ptr) {
#ifdef ENABLE_QNN
  // if the ptr has been registered, deregister it
  auto it = ptrToFdAndMemHandleMap_.find(ptr);
  if (it != ptrToFdAndMemHandleMap_.end()) {
    Qnn_ErrorHandle_t deregisterRet =
      qnnInterface_.memDeRegister(&it->second.second.second, 1);
    if (QNN_SUCCESS != deregisterRet) {
      // handle errors
      ml_loge("qnnInterface_.memDeRegister failed");
    }
    ptrToFdAndMemHandleMap_.erase(it);
  }
  rpcmem_free(ptr);
#else
  free(ptr);
#endif
}

} // namespace nntrainer
