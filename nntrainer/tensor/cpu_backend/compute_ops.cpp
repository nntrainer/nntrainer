// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 NNtrainer contributors
 *
 * @file   compute_ops.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 * @brief  Global ComputeOps pointer definition
 */

#include <compute_ops.h>

namespace nntrainer {

ComputeOps *g_compute_ops = nullptr;

void ensureComputeOps() {
  if (g_compute_ops == nullptr) {
    init_backend();
  }
}

} // namespace nntrainer
