// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_gemv_q6k.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Q6_K (GGML K-quant) lm_head GEMV for the CUDA backend. Split out
 *          of cuda_fc_qint4.{h,cpp}: Q6_K is a different weight layout,
 *          dequant, and consumer than the QS4CX plain-nibble int4 FC path
 *          (this file has no QS4CX dependency), mirroring the per-format
 *          split already used on the OpenCL side (q6_k_sgemv.cl vs
 *          int8_int4_gemm_v8c.cl).
 */

#ifndef __CUDA_GEMV_Q6K_H__
#define __CUDA_GEMV_Q6K_H__

namespace nntrainer::cuda {

/**
 * @brief Q6_K lm_head GEMV on the GPU (port of OpenCL kernel_mul_mv_q6_K_f32).
 *        Reads the FP16 hidden + (managed) Q6_K weight on the device and writes
 *        FP16 logits to the device output -- no host bounce, so the Q6_K
 *        lm_head (gemma2/qwen3) works under a device-only activation pool
 *        (NNTR_CUDA_DEV_ACT) where the host GEMV would fault.
 * @param w_q6k_dev       Q6_K weight, [vocab, hidden] row-major, device/managed
 * @param hidden_fp16_dev FP16 hidden (single row), device-resident, len=hidden
 * @param logits_fp16_dev FP16 logits out, device-resident, len=vocab
 * @return true on success; false (caller falls back to host) if hidden%256!=0
 *         or the kernel could not be dispatched.
 */
bool lmhead_gemv_q6_k_cuda(const void *w_q6k_dev,
                           const unsigned short *hidden_fp16_dev,
                           unsigned short *logits_fp16_dev, int vocab,
                           int hidden);

} // namespace nntrainer::cuda

#endif // __CUDA_GEMV_Q6K_H__
