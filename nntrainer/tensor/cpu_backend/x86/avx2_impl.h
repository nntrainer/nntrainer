// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl.h
 * @date   20 Feb 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header for AVX implementation
 *
 */

#ifndef __AVX2_IMPL_H_
#define __AVX2_IMPL_H_
#ifdef __cplusplus

#include <cstdint>
#include <limits.h>
#include <limits>
#include <stddef.h>

namespace nntrainer::avx2 {

#ifdef ENABLE_FP16
/**
 * @brief Converts half-precision floating point values to single-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing 16-bit floating point values
 * @param[out] output vector containing single-precision floating point values.
 */
void vcvt_f16_f32(unsigned int N, const _Float16 *input, float *output);

/**
 * @brief  Converts single-precision floating point values to half-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing single-precision floating point values
 * @param[out] output vector containing 16-bit floating point values
 */
void vcvt_f32_f16(unsigned int N, const float *input, _Float16 *output);

/**
 * @brief     check if the X has NaN value
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] X half-precision * for Vector X
 * @param[out] false if it has NaN or inf
 */
bool is_valid(const unsigned int N, const _Float16 *X);
#endif

/**
 * @copydoc unpack_q4_0x8_transpose16 in cpu_backend.h
 */
void unpack_q4_0x8_transpose16(const void *src, unsigned short *__restrict dT,
                               unsigned short *__restrict qsT, int N, int K,
                               int CT = 1);

/**
 * @brief convert q4_0x8 data to quants and scales
 *
 * @note this func is reserved for the performance comparison
 */
void convert_q4_0x8_shuffle_dispatch_avx(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out, int N, int K);

/**
 * @brief     check if the X has NaN value
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[out] false if it has NaN or inf
 */
bool is_valid(const unsigned int N, const float *X);

/**
 * @brief cblas_scopy occasionally emits SIGSEGV, so implement a custom version.
 *
 * @param N length of the vector
 * @param X float * for Vector X (input)
 * @param Y float * for Vector Y (output)
 */
void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY);

/**
 * @brief Matrix transpose / 2D Tensor transpose
 *
 * @param M row length of input matrix
 * @param N col length of input matrix
 * @param src src data of input matrix
 * @param ld_src data offset of input matrix
 * @param dst destination of output matrix
 * @param ld_dst data offset of output matrix
 */
void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst);

/**
 * @brief swiglu function with AVX : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
void swiglu(const unsigned int N, float *X, const float *Y, const float *Z);

/**
 * @brief swiglu function with alpha and AVX : X = (Y / (1 + exp(- alpha * Y)))
 * * Z
 * @param N number of elements in X
 * @param X float* for Vector X
 * @param Y float* for Vector Y
 * @param Z float* for Vector Z
 * @param alpha float
 */
void swiglu(const unsigned int N, float *X, const float *Y, const float *Z,
            float alpha);

/**
 * @brief     elementwise vector multiplication : Z = X ⊙ alpha * Y +
 * beta * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha = 1.f, float beta = 0.f, unsigned int i_stride = 1,
             unsigned int o_stride = 1);

/**
 * @brief     elementwise vector addition : Z = X + alpha * Y + beta *
 * Z
 * @param[in] N  length of the vector
 * @param[in] X float * for Vector X
 * @param[in] Y float * for Vector Y
 * @param[in] Z float * for Vector Z
 * @param[in] alpha scalar multiplier for input
 * @param[in] beta scalar multiplier for output
 * @param[in] i_stride input stride
 * @param[in] o_stride output stride
 */
void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride);

/**
 * @brief Multihead softmax, exp(x_i) / sum(exp(x_i)), inplace version
 * @param[in/out] qk_out float* input/output values
 * @param[in] start_row start row number
 * @param[in] end_row end row number
 * @param[in] num_heads heads number
 */
template <typename T = float>
void softmax_row_inplace(T *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, T *sink = nullptr);

/**
 * @brief Multihead softmax, exp(x_i) / sum(exp(x_i))
 * @param[in/out] qk_out float* input/output values
 * @param[in] start_row start row number
 * @param[in] end_row end row number
 * @param[in] num_heads heads number
 */
template <typename T = float>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, T *sink = nullptr);

/**
 * @brief Compute vcache for one row transposed
 * @param[in] row_num row number
 * @param[in] in float* input vector
 * @param[in] vcache uint16_t* input vector
 * @param[out] output float* output vector
 * @param[in] num_cache_head number head of cache
 * @param[in] gqa_size size of group
 * @param[in] head_dim head dimension
 * @param[in] local_window_size windows size for local attention
 */
void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim,
                                        size_t local_window_size = UINT_MAX);

/**
 * @brief Compute kcaches
 * @tparam BType type of B vector element
 * @param[in] in float* input vector
 * @param[in] kcache BType* input vector with keys cache
 * @param[out] output float* output float vector
 * @param[in] num_rows number of row
 * @param[in] num_cache_head number head of cache
 * @param[in] head_dim head dimension
 * @param[in] gqa_size size of group
 * @param[in] tile_size size of tile
 * @param[in] local_window_size windows size for local attention
 */
template <typename BType>
void compute_kcaches(const float *in, const BType *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size,
                     size_t local_window_size = UINT_MAX);

/**
 * @brief Compute rotary embedding value
 * @param[in] width current w value from b, c, h, w
 * @param[in] dim unit length of simd computation
 * @param[in] half_ criterion for rotational direction of embedding
 * @param[in/out] inout float* uesed also as output when expected output float*
 * values
 * @param[out] output void* output values, used when expected output __fp16*
 * values
 * @param[in] cos_ float* input con values
 * @param[in] sin_ float* input sin values
 * @param[in] only_convert_to_fp16 equal true if method is used only for
 * conversion
 */
void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16);
/**
 * @brief rms normalization computation w.r.t. width in H*W matrix input
 *
 * @param X input
 * @param Y output
 * @param H height of input matrix
 * @param W width of input matrix
 * @param epsilon epsilon of root mean squared dividing scale
 */
void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon);

/**
 * @brief fallback for clamping function.
 *
 * @tparam T Type of input data
 * @param input input vector
 * @param output output vector
 * @param length length of IO
 * @param lower_bound ditto
 * @param upper_bound ditto
 */
template <typename T = float>
void clamp(const T *input, T *output, size_t length,
           T lower_bound = std::numeric_limits<T>::lowest(),
           T upper_bound = std::numeric_limits<T>::max());

/**
 * @brief Copy uint16_t to float
 *
 * @param N length of the vector
 * @param input input data
 * @param output output data
 */
void copy_f16_f32(unsigned int N, const uint16_t *input, float *output);

/**
 * @brief Copy float to uint16_t
 *
 * @param N length of the vector
 * @param input input data
 * @param output output data
 */
void copy_f32_f16(unsigned int N, const float *input, uint16_t *output);

/**
 * @brief     Create a Q4_0 weights (without XOR 0x88) from int4 weights
 *
 * @param[in] int4_weight Pointer to the input 4-bit quantized weights array.
 * The array should contain 16 bytes representing 32 4-bit values. Each byte
 * contains two 4-bit quantized values packed together.
 * @param[out] q4_0_weight Pointer to the output 4-bit quantized weights
 * array. The array should contain 16 bytes representing 32 4-bit values. Each
 * byte contains two 4-bit quantized values packed together.
 * @note      The input int4_weight array should contain exactly 32 4-bit
 * values (16 bytes) to match the weight of Q4_0 block size (32 elements per
 * block).
 * Input:  | 0, 1 | 2, 3 | 4, 5 | ... |14,15 |16,17 | ... |28,29 |30,31 |
 *         | A, B | A, B | A, B | ... | A, B | C, D | ... | C, D | C, D |
 *
 * Output: | 0,16 | 1,17 | 2,18 | 3,19 | ...          ... |14,30 |15,31 |
 *         | A, C | B, D | A, C | B, D | ...          ... | A, C | B, D |
 */
void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight);

/**
 * @brief Transform data from in-memory layout osv32_isv2 to block_q4_0x8
 * in-memory layout.
 *
 * @param N number of rows
 * @param K number of columns
 * @param osv32_weights uint8_t* data of weights in osv32_isv2 layout
 * @param osv32_scales fp16* scales
 * @param scale_group_size group size (32 or 64 or 128)
 * @param dst_q4_0x void * output data in block_q4_0x8 or block_q4_0x4 layout
 */
void transform_int4_osv32_isv2_to_q4_0x8(size_t N, size_t K,
                                         const uint8_t *osv32_weights,
                                         const uint16_t *osv32_scales,
                                         size_t scale_group_size,
                                         void *dst_q4_0x);

} // namespace nntrainer::avx2

#endif /* __cplusplus */
#endif /* __BLAS_AVX_H_ */
