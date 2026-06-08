// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_fast_path.h
 * @date   01 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Small/skinny x86 FP16 GEMM fast-path dispatcher
 */

#ifndef __X86_HGEMM_FAST_PATH_H_
#define __X86_HGEMM_FAST_PATH_H_

#include "hgemm_workspace.h"

#include <tensor_dim.h>

namespace nntrainer::avx2::internal {

template <typename AType, typename BType, typename CType>
bool try_hgemm_fast_path(bool TransA, bool TransB, unsigned int M,
                         unsigned int N, unsigned int K, float alpha,
                         const AType *A, unsigned int a_stride, const BType *B,
                         unsigned int b_stride, float beta, CType *C,
                         unsigned int c_stride, HgemmWorkspace &workspace);

} /* namespace nntrainer::avx2::internal */

#endif /* __X86_HGEMM_FAST_PATH_H_ */
