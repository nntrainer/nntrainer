// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qint4.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QS4CX dequant-GEMM for the CUDA FC layer:
 *          Y[M,N] = X[M,K] * dequant(W), where W is the QS4CX PLAIN payload
 *          (row-major [N][(K+1)/2] nibbles, uint4 = int4+8) with an
 *          N-entry per-channel fp16 scale. The int4 weight is read and
 *          dequantized inline in the kernel; float accumulation. Callers must
 *          pass device-accessible (UVM) pointers.
 */

#ifndef __CUDA_FC_QINT4_H__
#define __CUDA_FC_QINT4_H__

namespace nntrainer::cuda {

/**
 * @brief Build (and cache) the N-entry UVM fp16 per-channel scale buffer from
 *        the tensor's fp32 scales. The dequant kernels read the scale on device
 *        every call; the tensor stores fp32, so the fp16 copy is made once at
 *        first use and cached by the fp32-scale pointer (weights live for the
 *        process lifetime). @p out_sc receives the cached device pointer.
 * @return false on allocation failure (caller falls back to the host path).
 */
bool cuda_fc_qs4cx_scales_to_uvm_fp16(const float *fp32_scales, unsigned int N,
                                      const unsigned short **out_sc);

/**
 * @brief Y[M,N] = X[M,K] * dequant(QS4CX W) where W is the PLAIN QS4CX payload
 *        and @p scales_fp16 is the N-entry fp16 scale buffer (from
 *        cuda_fc_qs4cx_scales_to_uvm_fp16). FP32 activation, FP32 output.
 *        One thread per output element; float accumulation.
 * @return true on success.
 */
bool cuda_fc_qs4cx_gemm_fp32(const float *X, const unsigned char *plain_w,
                             const unsigned short *scales_fp16, float *Y,
                             unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief fp16-activation variant of cuda_fc_qs4cx_gemm_fp32: fp16 in / fp16
 *        out, staged through fp32 for the plain-decode GEMM (float
 *        accumulation, no int8 activation quantization -- the accuracy
 *        reference for the int4 FC).
 * @return true on success.
 */
bool cuda_fc_qs4cx_gemm_fp16_naive(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const unsigned short *scales_fp16,
                                   unsigned short *Yh, unsigned int M,
                                   unsigned int N, unsigned int K);

/**
 * @brief w4a8 dp4a fast path: Y[M,N] = X[M,K] * dequant(QS4CX W), FP32
 *        activation. Per-row asymmetric int8 activation quant + symmetric int4
 *        weight, int8xint8 dot via __dp4a on the int ALU. The int4 weight is
 *        repacked to signed packed int4 once and cached on device (keyed by
 *        @p plain_w). The int32 accumulate is exact.
 * @return true on success.
 */
bool cuda_fc_qs4cx_dp4a_gemm_fp32(const float *X, const unsigned char *plain_w,
                                  const unsigned short *scales_fp16, float *Y,
                                  unsigned int M, unsigned int N,
                                  unsigned int K);

/** @brief fp16-activation variant of cuda_fc_qs4cx_dp4a_gemm_fp32: fp16 in /
 *  fp16 out (the conversion folded into the GEMM epilogue). */
bool cuda_fc_qs4cx_dp4a_gemm_fp16(const unsigned short *Xh,
                                  const unsigned char *plain_w,
                                  const unsigned short *scales_fp16,
                                  unsigned short *Yh, unsigned int M,
                                  unsigned int N, unsigned int K);

/**
 * @brief w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant
 *        scheme as the dp4a path (per-row asym int8 activation x symmetric int4
 *        weight) but the int8xint8->int32 GEMM runs on the IMMA Tensor Cores
 *        (~10x the dp4a int-ALU GEMM at prefill M). The int32 accumulate is
 *        exact, so the result is bit-identical to dp4a; the int4->int8 weight
 *        unpack is cached once. Returns false (caller falls to dp4a) on any
 *        cuBLAS/runtime failure.
 */
bool cuda_fc_qs4cx_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                       const unsigned char *plain_w,
                                       const unsigned short *scales_fp16,
                                       unsigned short *Yh, unsigned int M,
                                       unsigned int N, unsigned int K);

/**
 * @brief [wprefetch] Migrate a QS4CX weight's managed plain payload (+ its
 *        fp32 scale tail) to the device with cudaMemPrefetchAsync, so the FC
 *        bytes leave host RSS and the GEMM reads them from VRAM. Discrete GPU
 *        only (a no-op / false on integrated, where managed pages don't
 *        migrate). @p plain_w must be a managed (UVM) pointer.
 * @return true if the prefetch was issued.
 */
bool cuda_fc_qs4cx_prefetch_weight(const unsigned char *plain_w, unsigned int N,
                                   unsigned int K);

/**
 * @brief Build the dp4a derived weight cache (packed int4 + rowsum) for one
 *        QS4CX plain payload at load time, off the first prefill. GPU repack,
 *        no host transient; idempotent (pointer-keyed). Returns false only on
 *        a device allocation / dispatch failure (the lazy in-path build then
 *        remains the fallback).
 */
bool cuda_fc_qs4cx_prewarm(const unsigned char *plain_w, unsigned int N,
                           unsigned int K);

/**
 * @brief Mark a weight exempt from the eager load-time cuBLAS-i8 [K,N] cache
 *        build (skip_prefill towers / untied lm_head cannot reach the M>=32
 *        cuBLAS gate -- their int8 cache is dead VRAM). Lazy build self-heals.
 */
void cuda_fc_qs4cx_prewarm_exempt_i8(const void *plain_w);

/**
 * @brief Pre-grow the dp4a activation-quant scratch to the given decode
 *        bounds so the M=1 decode FC never cudaMallocs inside a CUDA-graph
 *        capture. maxN is accepted for signature stability; the decode path
 *        has no N-sized scratch.
 */
bool cuda_fc_qint4_dp4a_prewarm(unsigned int maxM, unsigned int maxK,
                                unsigned int maxN);

/**
 * @brief Free every pointer-keyed derived weight cache (dp4a packed int4 +
 *        cuBLAS int8) -- the model-reload teardown. The fp16-scale UVM side
 *        buffers are process-lifetime by design and are not freed.
 */
void cuda_fc_qs4cx_release_weight_caches();

} // namespace nntrainer::cuda

#endif // __CUDA_FC_QINT4_H__
