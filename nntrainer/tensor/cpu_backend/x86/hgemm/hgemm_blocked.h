// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_blocked.h
 * @date   01 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  C32 panel orchestration for x86 FP16 GEMM
 */

#ifndef __X86_HGEMM_BLOCKED_H_
#define __X86_HGEMM_BLOCKED_H_

#include "hgemm_workspace.h"

#include <tensor_dim.h>

namespace nntrainer::hgemm::internal {

template <typename AType, typename BType, typename CType>
void run_hgemm_blocked(bool TransA, bool TransB, unsigned int M, unsigned int N,
                       unsigned int K, float alpha, const AType *A,
                       unsigned int a_stride, const BType *B,
                       unsigned int b_stride, float beta, CType *C,
                       unsigned int c_stride, HgemmWorkspace &workspace);

} /* namespace nntrainer::hgemm::internal */

#endif /* __X86_HGEMM_BLOCKED_H_ */
