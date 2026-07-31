// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_rpc_allocator.h
 * @date   31 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  MemAllocator that backs activation tensors with rpcmem, so the
 *         Hexagon cDSP bridge can hand the DSP a pointer it can map
 *         directly instead of memcpy-ing into a separate staging buffer
 *         on every accelerated GEMM call.
 *
 * Deliberately independent of nntrainer/qnn/jni/rpc_mem.h's RpcMem/
 * QNNRpcManager: that pair is compiled only into the optional QNN module
 * (gated on the QNN SDK being present), while HexagonContext is a core
 * nntrainer feature with no such gate. Same dlopen(libcdsprpc.so) pattern
 * as HexagonComputeOps's own bridge-library loader
 * (hexagon_compute_ops.cpp's get_bridge_api()) - a second dlopen of an
 * already-loaded soname is cheap and returns the same handle, so load
 * order relative to libggml-hexagon.so's own internal use of
 * libcdsprpc.so does not matter here.
 */

#ifndef __HEXAGON_RPC_ALLOCATOR_H__
#define __HEXAGON_RPC_ALLOCATOR_H__

#include <mem_allocator.h>

namespace nntrainer {

/**
 * @class HexagonRpcAllocator
 * @brief Allocates activation tensors from libcdsprpc.so's rpcmem heap.
 */
class HexagonRpcAllocator : public MemAllocator {
public:
  HexagonRpcAllocator();
  ~HexagonRpcAllocator() override = default;

  void alloc(void **ptr, size_t size, size_t alignment) override;
  void free(void *ptr) override;

  std::string getName() override { return "cdsp"; }
};

} // namespace nntrainer

#endif /* __HEXAGON_RPC_ALLOCATOR_H__ */
