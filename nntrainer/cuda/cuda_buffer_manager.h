// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_buffer_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Global CUDA device-buffer pool manager. Peer of ClBufferManager.
 *          Skeleton for P0: prealloc pools are filled in during P1 (memory /
 *          residency). Kept as the structural mirror so later phases plug in.
 */

#ifndef __CUDA_BUFFER_MANAGER_H__
#define __CUDA_BUFFER_MANAGER_H__

#include "singleton.h"

namespace nntrainer {

/**
 * @class CudaBufferManager
 * @brief Singleton managing reusable CUDA device buffers (pools added in P1).
 */
class CudaBufferManager : public Singleton<CudaBufferManager> {
public:
  /**
   * @brief Initialize device buffer pools. No-op placeholder until P1.
   */
  void initBuffers();

  /**
   * @brief Destroy pooled buffers.
   */
  ~CudaBufferManager() override;
};

} // namespace nntrainer

#endif // __CUDA_BUFFER_MANAGER_H__
