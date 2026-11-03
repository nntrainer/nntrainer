// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl.h
 * @date   20 Feb 2024
 * @see    https://github.com/nntrainer/nntrainer
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
#include <tensor_dim.h>

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
void vcvt_f16_f32(unsigned int N, const _FP16 *input, float *output);

/**
 * @brief  Converts single-precision floating point values to half-precision
 * floating point values.
 *
 * @param[in]  N number of elements in input vector
 * @param[in]  input vector containing single-precision floating point values
 * @param[out] output vector containing 16-bit floating point values
 */
void vcvt_f32_f16(unsigned int N, const float *input, _FP16 *output);

/**
 * @brief     check if the X has NaN value
 * @note it compare (x!=x || x == inf)
 * @param[in] N  length of the vector
 * @param[in] X half-precision * for Vector X
 * @param[out] false if it has NaN or inf
 */
bool is_valid(const unsigned int N, const _FP16 *X);
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
 * @brief swiglu function with AVX : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
void tanh_gelu_v2(const unsigned int N, const float *X, float *Y);

/**
 * @brief fused tanh-gelu and multiply with AVX :
 *        X = 0.5 * Y * (1 + tanh(0.7978845608 * (Y + 0.044715 * Y^3))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X (output)
 * @param Y float * for Vector Y (input, gelu applied to this)
 * @param Z float * for Vector Z (input, multiplied in)
 */
void tanh_gelu_mul(const unsigned int N, float *X, const float *Y,
                   const float *Z);

/**
 * @brief swiglu function with AVX : X = (Y / (1 + exp( -Y ))) * Z
 *
 * @param N number of elements in X
 * @param X float * for Vector X
 * @param Y float * for Vector Y
 * @param Z float * for Vector Z
 */
void gelu_v2(const unsigned int N, const float *X, float *Y);

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

/**f
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
 * @brief AVX2 fp32 causal depthwise Conv1D prefill for kernel size 3.
 *
 * Input and output are contiguous [B, H, W]. For each channel c, the kernel
 * uses packed_weight [w0 | w1 | w2] and computes the causal recurrence over H:
 * y_t = w0*x_t + w1*x_{t-1} + w2*x_{t-2} (+ bias).
 */
void causal_depthwise_conv1d_k3(const float *input, const float *packed_weight,
                                const float *bias, float *output,
                                unsigned int B, unsigned int H, unsigned int W);

/**
 * @brief AVX2 fp32 single-token decode for causal depthwise Conv1D.
 *
 * Reads state [x_{t-2} | x_{t-1}], writes y_cur for x_cur, and shifts the
 * state in-place to [x_{t-1} | x_t].
 */
void causal_depthwise_conv1d_k3_decode(const float *x_cur,
                                       const float *packed_weight, float *state,
                                       float *y_cur, unsigned int W);

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
 * @param[in] head_start start index of KV heads to process (default 0)
 *            Used for head-direction parallelization during decoding.
 * @param[in] head_end end index of KV heads to process (default num_cache_head)
 *            The range is [head_start, head_end), i.e., head_end is exclusive.
 *            Default -1 means process all heads from head_start to
 *            num_cache_head. No other negative values are accepted.
 * @note Caller must ensure head_start < head_end when head_end != -1.
 */
void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim,
                                        size_t local_window_size = UINT_MAX,
                                        int head_start = 0, int head_end = -1);

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
 * @param[in] head_start start index of KV heads to process (default 0).
 *            Used for head-direction parallelization during decoding.
 * @param[in] head_end end index (exclusive) of KV heads to process.
 *            The range is [head_start, head_end), i.e., head_end is exclusive.
 *            Default -1 means process all heads from head_start to
 *            num_cache_head. No other negative values are accepted.
 * @note Caller must ensure head_start < head_end when head_end != -1.
 */
template <typename BType>
void compute_kcaches(const float *in, const BType *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size,
                     size_t local_window_size = UINT_MAX, int head_start = 0,
                     int head_end = -1);

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
 * @brief RoPE (optional) + symmetric int8 snap + fp16 store, one pass: the
 *        KV-cache write of a model trained with an int8 KV cache.
 *
 * For every head h (columns [h * dim, (h + 1) * dim) of each row) and
 * element x (rotated with cos_ / sin_ first when cos_ != nullptr, using the
 * rotate-half form of compute_rotary_emb_value):
 *   q = nearbyint(x / head_scales[h])   (round half to even)
 *   x = clamp(q, -128, 127) * head_scales[h]
 * and x is stored to @a output as IEEE fp16. Every implementation must use
 * true division and round-half-to-even so all paths give identical bits.
 *
 * @param[in] width row width (number of heads * dim)
 * @param[in] dim head dimension
 * @param[in] half_ dim / 2
 * @param[in] input fp32 row (not modified)
 * @param[out] output fp16 (uint16_t bits) row
 * @param[in] cos_ cos table of this position or nullptr (no RoPE)
 * @param[in] sin_ sin table of this position (ignored when cos_ == nullptr)
 * @param[in] head_scales one scale per head
 */
void compute_rotary_emb_value_snap(unsigned int width, unsigned int dim,
                                   unsigned int half_, const float *input,
                                   uint16_t *output, const float *cos_,
                                   const float *sin_, const float *head_scales);
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

/**
 * @brief Number of QS4CX weight rows one call of qs4cx_f32acc_rows() or
 * qs4cx_qa8dx_rows() covers at most.
 */
constexpr size_t QS4CX_ROWS_PER_BLOCK = 32;

/**
 * @brief Power-of-two factor qs4cx_prepare_lhs_f16() folds into the
 * activation. The weight side decodes each nibble as (int4 << 28), so the
 * pair multiplies back to activation * int4 exactly.
 */
constexpr float QS4CX_LHS_SCALE = 1.0f / 268435456.0f; // 2^-28

/**
 * @brief Convert an fp16 activation to the fp32 operand qs4cx_f32acc_rows()
 * reads: every element is widened (F16C, exact) and multiplied by
 * QS4CX_LHS_SCALE (a power of two, exact for every fp16 value).
 *
 * @param[in] N number of elements
 * @param[in] input fp16 values as raw IEEE half bits
 * @param[out] output fp32 values, input * 2^-28
 */
void qs4cx_prepare_lhs_f16(size_t N, const uint16_t *input, float *output);

/**
 * @brief Raw fp32 dot products of up to QS4CX_ROWS_PER_BLOCK QS4CX weight
 * rows with every activation row.
 *
 * acc[m * QS4CX_ROWS_PER_BLOCK + i] = sum over k of x[m][k] * (w[i][k] - 8)
 * for m < M and i < nrows. Each sum is accumulated in fp32 strictly in
 * ascending k, starting from 0, one rounding per k: the order the scalar
 * reference loop uses. The vector lanes run across weight rows (outputs),
 * never across k, and every product is exact (an fp16 value times a 4-bit
 * integer), so a fused multiply-add rounds exactly like the reference's
 * multiply then add. The result is therefore bit-identical to the scalar
 * loop for any M, K and nrows.
 *
 * @param[in] M number of activation rows
 * @param[in] K reduction length
 * @param[in] x activation rows prepared by qs4cx_prepare_lhs_f16()
 * @param[in] ldx leading dimension of @a x
 * @param[in] w first weight row: plain QS4CX nibbles, ceil(K/2) bytes a
 * row, even k in the low nibble, each nibble int4 + 8
 * @param[in] row_bytes bytes between weight rows
 * @param[in] nrows number of weight rows, 1 .. QS4CX_ROWS_PER_BLOCK
 * @param[out] acc M x QS4CX_ROWS_PER_BLOCK sums; lanes past @a nrows are
 * scratch
 */
void qs4cx_f32acc_rows(size_t M, size_t K, const float *x, size_t ldx,
                       const uint8_t *w, size_t row_bytes, size_t nrows,
                       float *acc);

/**
 * @brief Integer dot products of up to QS4CX_ROWS_PER_BLOCK QS4CX weight
 * rows with qa8dx-quantized activation rows.
 *
 * For m < M and i < nrows, writes
 * iacc[m * QS4CX_ROWS_PER_BLOCK + i] = sum over k of
 *   (x[m][k] + offset[m]) * (w[i][k] - 8)
 * which is the integer the scalar qa8dx x qs4cx reference accumulates. The
 * arithmetic is exact int32, so the order of the sum does not matter.
 *
 * @param[in] M number of activation rows
 * @param[in] K reduction length
 * @param[in] lhs qa8dx rows: fp32 scale, int32 offset, then K int8 values
 * @param[in] lhs_stride bytes between qa8dx rows
 * @param[in] w first weight row (plain QS4CX nibbles)
 * @param[in] row_bytes bytes between weight rows
 * @param[in] nrows number of weight rows, 1 .. QS4CX_ROWS_PER_BLOCK
 * @param[out] iacc M x QS4CX_ROWS_PER_BLOCK sums; lanes past @a nrows are
 * scratch
 */
void qs4cx_qa8dx_rows(size_t M, size_t K, const int8_t *lhs, size_t lhs_stride,
                      const uint8_t *w, size_t row_bytes, size_t nrows,
                      int32_t *iacc);

} // namespace nntrainer::avx2

#endif /* __cplusplus */
#endif /* __BLAS_AVX_H_ */
