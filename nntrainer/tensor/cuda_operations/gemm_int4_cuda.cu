// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	gemm_int4_cuda.cu
 * @date	28 Nov 2025
 * @brief	CUDA implementation of int4 GEMM operation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	[Your Name] <[your.email@samsung.com]>
 * @bug		No known bugs except for NYI items
 *
 */

#include "gemm_int4_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace nntrainer {

void gemm_int4_cuda(void *input, void *weights, void *scales, void *output,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int quantization_group_size) {
  // todo:
}

} // namespace nntrainer
