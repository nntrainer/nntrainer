#include "android_memory_allocator.h"

#include <dynamic_library_loader.h>
#include <iostream>

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

typedef void *(*RpcMemAllocFn_t)(int, unsigned int, int);
typedef void (*RpcMemFreeFn_t)(void *);

enum {
  DL_NOW = 0x0001,
  DL_LOCAL = 0x0002,
  DL_GLOBAL = 0x0004,
};

bool inited = false;
void *handle;
RpcMemAllocFn_t rpcmem_alloc;
RpcMemFreeFn_t rpcmem_free;

void init_RPCMEM() {
  void *handle = nntrainer::DynamicLibraryLoader::loadLibrary(
    "libcdsprpc.so", DL_NOW | DL_LOCAL);
  rpcmem_alloc = (RpcMemAllocFn_t)nntrainer::DynamicLibraryLoader::loadSymbol(
    handle, "rpcmem_alloc");
  rpcmem_free = (RpcMemFreeFn_t)nntrainer::DynamicLibraryLoader::loadSymbol(
    handle, "rpcmem_free");

  if (rpcmem_alloc == nullptr || rpcmem_free == nullptr) {
    std::cerr << "open rpc mem failed" << std::endl;
  }
}

void *allocate(size_t fileSize) {
  if (!inited) {
    init_RPCMEM();
    inited = true;
  }
  void *buffer =
    rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, fileSize + 140);

  return buffer;
}

void deallocate(void *pointer) { rpcmem_free(pointer); }
