// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_buffer_manager.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Global CUDA device-buffer pool manager implementation (P0 skeleton).
 */

#include "cuda_buffer_manager.h"

#include <nntrainer_log.h>

namespace nntrainer {

void CudaBufferManager::initBuffers() {
  ml_logd("[CUDA] buffer manager initialized (prealloc pools deferred to P1)");
}

CudaBufferManager::~CudaBufferManager() {}

} // namespace nntrainer
