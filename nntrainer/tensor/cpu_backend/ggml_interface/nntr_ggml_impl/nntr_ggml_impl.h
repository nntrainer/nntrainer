// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   nntr_ggml_impl.h
 * @date   13 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#ifndef __NNTR_GGML_IMPL__
#define __NNTR_GGML_IMPL__

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Initialize ggml implementation state
 */
void nntr_ggml_init();

/**
 * @brief Compute Q4_0 4x8 weights by Q8_0 activations GEMM
 */
void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

#ifdef ENABLE_FP16
// Pick the half type the same way tensor_dim.h does, so this header stays
// self-contained even if a caller pulls it in without tensor_dim.h. On
// ARM/Android (USE__FP16) it is __fp16; on x86_64 fp16 builds it is _Float16,
// except on compilers without native _Float16 (MSVC), where the build defines
// USE_HALF_WRAPPER and the uint16-backed nntrainer::Half stands in.
#ifdef USE__FP16
#define NNTR_GGML_FP16 __fp16
#elif defined(USE_HALF_WRAPPER)
#include "half_fp16.h" // self-contained leaf: uint16-backed nntrainer::Half
#define NNTR_GGML_FP16 ::nntrainer::Half
#else
#define NNTR_GGML_FP16 _Float16
#endif
void nntr_gemm_q4_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q8_0 weights by Q8_0 activations GEMM, FP16 output (SMMLA)
 */
void nntr_gemm_q8_0_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s, size_t bs,
                              const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute interleaved (q8_0x4) Q8_0 weights by q8_0x4 activations GEMM,
 *        FP16 output (register-blocked 4x4 SMMLA). Both operands pre-packed.
 */
void nntr_gemm_q8_0_q8_0_4x4_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

/**
 * @brief Hand-written ARM64 SMMLA asm port of nntr_gemm_q4_0_4x8_q8_0_fp16
 *        for Q8_0x4 weights (no nibble-unpack) by Q8_0x4 activations GEMM,
 *        FP16 output. Same 4x8-interleaved block-loop structure/register
 *        schedule as the Q4_0 kernel, with 8 direct block loads (stride
 *        0x88) replacing the 4-load+unpack sequence and no /16 scvtf shift.
 */
void nntr_gemm_q8_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);

void nntr_gemv_q4_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc);
#endif

/**
 * @brief Compute Q4_0 8x8 weights by Q8_0 activations GEMM
 */
void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q4_K 8x8 weights by Q8_K activations GEMM
 */
void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q4_0 4x8 weights by Q8_0 activations GEMV
 */
void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q4_0 8x8 weights by Q8_0 activations GEMV
 */
void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Compute Q8_0 weights by Q8_0 activations GEMM
 */
void nntr_gemm_q8_0_q8_0(int n, float *__restrict s, size_t bs,
                         const void *__restrict vx, const void *__restrict vy,
                         int nr, int nc);

/**
 * @brief Repack plain block_q8_0 rows [nrow x k/QK8_0] to the q8_0x4
 *        interleaved layout (block_q8_0x4: d[4] + qs[32*sub + row*8 + c]).
 *        Byte-for-byte the layout of nntr_quantize_mat_q8_0_4x8 and the
 *        offline Python repack_q8_0 (YOLO conv path). Total bytes unchanged.
 */
int nntr_repack_q8_0_to_q8_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief FP32-output Q8_0 x Q8_0 GEMM. Same numerics as nntr_gemm_q8_0_q8_0 but
 * implemented via the proven SMMLA path on NEON (i8mm) and correct scalar
 * elsewhere; used by the W8A32 indirect conv. vx=weight[nc x nb],
 * vy=act[nr x nb], s=FP32 out[nr x bs].
 */
void nntr_gemm_q8_0_q8_0_f32(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

/**
 * @brief Interleaved (q8_0x4) FP32-output 4x4 SMMLA GEMM. Both operands are
 * pre-packed block_q8_0x4 (weight via repack_q8_0, activation via
 * nntr_quantize_mat_q8_0_4x8). NEON uses the proven i8mm kernel with FP32
 * store; other ISAs use a correct scalar. Requires nr % 4 == 0.
 * vx=weight[nc/4][nb], vy=act[nr/4][nb], s=FP32 out[nr x bs].
 */
void nntr_gemm_q8_0_q8_0_4x4_f32(int n, float *__restrict s, size_t bs,
                                 const void *__restrict vx,
                                 const void *__restrict vy, int nr, int nc);

/**
 * @brief Per-CHANNEL W8A8 GEMM. Activation is int8 with ONE per-tensor scale
 * (a_scale), weight is int8 with ONE scale per output channel (w_scale[nc]);
 * both operands are int8 packed in the block_q8_0x4 qs layout (the fp16 d
 * fields are ignored). The whole K reduction accumulates in int32 and the
 * scale a_scale*w_scale[col] is applied once at the end -- no per-32-block
 * float folding. NEON uses SMMLA; other ISAs a correct scalar. Requires
 * nr,nc arbitrary (tail handled), n % 32 == 0.
 * vx=weight qs[nc/4][nb], vy=act qs[nr/4][nb], s=FP32 out[nr x bs].
 */
void nntr_gemm_q8ch_4x4_f32(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx,
                            const float *__restrict w_scale,
                            const void *__restrict vy, float a_scale, int nr,
                            int nc);

/**
 * @brief Per-channel W8A8 GEMM with a PLAIN row-major int8 activation.
 *
 * Identical math and output to nntr_gemm_q8ch_4x4_f32, but the activation is
 * read from a plain [nr x n] int8 matrix `A` (row stride lda) instead of the
 * pre-interleaved block_q8_0x4 stream. This removes the activation packing pass
 * entirely for the 1x1 unit-stride convs whose gather is the identity (the
 * input already IS the [M, in_ch] int8 matrix). The weight `vx` stays in the
 * q8_0x4 qs layout (its fp16 d ignored); w_scale is per-output-channel FP32.
 * The NEON SMMLA path assembles the 2x8 row tiles on the fly from two 8-byte
 * plain loads + vcombine. n % 32 == 0; nr,nc arbitrary (tails handled); every
 * activation row [0, nr) must be in-bounds (the caller zero-pads the M%4 tail).
 */
void nntr_gemm_q8ch_plainA_f32(int n, float *__restrict s, size_t bs,
                               const void *__restrict vx,
                               const float *__restrict w_scale,
                               const int8_t *__restrict A, size_t lda,
                               float a_scale, int nr, int nc);

/**
 * @brief Compute Q4_K 8x8 weights by Q8_K activations GEMV
 */
void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q8_0_4x4_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q8_0_4x4_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemm_q8_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_gemv_q8_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc);

void nntr_quantize_mat_q8_0_4x4(const float *__restrict x, void *__restrict vy,
                                int64_t k);

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

/**
 * @brief Quantize FP32 matrix rows to Q8_K 4x8 layout
 */
void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k);

/**
 * @brief Repack Q4_0 data to Q4_0 4-block layout
 */
int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief Repack Q4_0 data to Q4_0 8-block layout
 */
int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief Repack Q4_K data to Q4_K 8-block layout
 */
int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k);

/**
 * @brief Quantize FP32 rows to Q4_0
 */
size_t nntr_quantize_q4_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize FP32 rows to Q4_K
 */
size_t nntr_quantize_q4_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize FP32 rows to Q6_K
 */
size_t nntr_quantize_q6_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize FP32 rows to Q8_0
 */
size_t nntr_quantize_q8_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrows, int64_t n_per_row,
                          const float *imatrix);

/**
 * @brief Quantize one FP32 row to Q8_0
 */
void nntr_quantize_row_q8_0(const float *__restrict x, void *__restrict y,
                            int64_t k);

/**
 * @brief Quantize one FP32 row to Q8_K
 */
void nntr_quantize_row_q8_K(const float *__restrict x, void *__restrict y,
                            int64_t k);

/**
 * @brief Dequantize one Q4_0 row to FP32
 */
void nntr_dequantize_row_q4_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q4_K row to FP32
 */
void nntr_dequantize_row_q4_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q6_K row to FP32
 */
void nntr_dequantize_row_q6_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q8_0 row to FP32
 */
void nntr_dequantize_row_q8_0(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Dequantize one Q8_K row to FP32
 */
void nntr_dequantize_row_q8_K(const void *__restrict x, float *__restrict y,
                              int64_t k);

/**
 * @brief Compute dot product between Q6_K and Q8_K vectors
 */
void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc);

#endif
