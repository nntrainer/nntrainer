// SPDX-License-Identifier: Apache-2.0
#ifndef __HEXKL_BACKEND_H__
#define __HEXKL_BACKEND_H__
#ifdef __cplusplus

#ifdef USE_HMX

namespace nntrainer {
namespace hexkl {

void initialize();
void finalize();

/**
 * @brief Pre-build the WH-layout cache entry for a FP32 weight (no dispatch).
 *
 * Call this on M=1 decode steps or right after model loading so that the first
 * real prefill (M > 1) skips the build and goes straight to HMX dispatch.
 */
void preload_weight_f32(bool TransB, unsigned N, unsigned K,
                        const float *B, unsigned ldb);

/** @brief Returns true if this weight's WH-layout is already cached. */
bool is_weight_cached(bool TransB, unsigned N, unsigned K, const float *B);

/**
 * @brief Dispatch a prefill-phase SGEMM (FP32×FP32) to HMX via INT8.
 *
 * Quantizes FP32 weight B to symmetric INT8 WH-layout on first call (cached),
 * quantizes FP32 activation A to UINT8 per-dispatch, then dispatches via
 * sdkl_npu_mm_u8i8_i32. Output INT32 is dequantized back to FP32.
 * Bias correction (zero_point=128) is pre-computed with the weight cache.
 *
 * @return true if dispatched to HMX; false to fall back to CPU.
 */
bool sgemm_hmx_i8(bool TransB,
                  unsigned int M, unsigned int N, unsigned int K,
                  const float *A, unsigned int lda,
                  const float *B, unsigned int ldb,
                  float *C, unsigned int ldc);

/**
 * @brief Dispatch a prefill-phase SGEMM (FP32×FP32) to HMX via FP16.
 *
 * Converts FP32 weight B to FP16 WH-layout on first call (cached for reuse),
 * then dispatches C[M×N] = A[M×K] * op(B) via sdkl_npu_mm_f32f16_f32.
 *
 * @return true if dispatched to HMX; false to fall back to CPU.
 */
bool sgemm_hmx(bool TransB,
               unsigned int M, unsigned int N, unsigned int K,
               const float *A, unsigned int lda,
               const float *B, unsigned int ldb,
               float *C, unsigned int ldc);

/**
 * @brief Dispatch a SGEMM with FP16 weights to HMX (shgemm path).
 *
 * B is already FP16 — only needs WH layout transform (cached).
 * No FP32→FP16 conversion overhead.  Used by the shgemm code path
 * (FP32 activations × FP16 weights → FP32 output).
 *
 * @return true if dispatched to HMX; false to fall back to CPU.
 */
bool shgemm_hmx(bool TransB,
                unsigned int M, unsigned int N, unsigned int K,
                const float *A, unsigned int lda,
                const __fp16 *B, unsigned int ldb,
                float *C, unsigned int ldc);

} // namespace hexkl
} // namespace nntrainer

#endif // USE_HMX

#endif // __cplusplus
#endif // __HEXKL_BACKEND_H__