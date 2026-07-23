// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_compute_ops.h
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Accessor for the Hexagon cDSP ComputeOps table. Mirrors
 * get_cl_ops()/get_cpu_ops() - see compute_ops.h for the dispatch contract.
 */

#ifndef __HEXAGON_COMPUTE_OPS_H__
#define __HEXAGON_COMPUTE_OPS_H__

namespace nntrainer {

class ComputeOps;

/**
 * @brief Get the Hexagon cDSP ComputeOps table.
 *
 * Stage 1: overrides only the Q4_0 accel predicate/impl, same as
 * ClComputeOps. gemm_q4_0_accel_fp32() currently throws not_implemented -
 * the real FastRPC/dspqueue bridge into ggml-hexagon's cDSP session lands in
 * a follow-up once the Hexagon SDK is available to build/test against.
 *
 * @return ComputeOps* pointer to the ops table
 */
ComputeOps *get_hexagon_ops();

} // namespace nntrainer

#endif /* __HEXAGON_COMPUTE_OPS_H__ */
