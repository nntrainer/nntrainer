// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.h
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_H__
#define __BLAS_KERNELS_H__

#include <cl_buffer_manager.h>
#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

#include <string>

namespace nntrainer {

/**
 * @brief     signed 4-bit integer gemv async computation : C = A*B
 * @param[in] weight std::vector<void *> for int4 quantized weight
 * @param[in] scale std::vector<uint16_t *> for scales
 * @param[in] input uint16_t * for input
 * @param[in] output std::vector<uint16_t *> for output
 * @param[in] K hidden dimension
 * @param[in] Ns output dimensions
 */
void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, uint16_t *input,
                        std::vector<uint16_t *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv async computation : C = A*B
 * @param[in] weight std::vector<void *> for int4 quantized weight
 * @param[in] scale std::vector<uint16_t *> for scales
 * @param[in] input float * for input
 * @param[in] output std::vector<float *> for output
 * @param[in] K hidden dimension
 * @param[in] Ns output dimensions
 */
void gemv_int4_async_cl(std::vector<void *> weights,
                        std::vector<uint16_t *> scales, float *input,
                        std::vector<float *> outputs, unsigned int K,
                        std::vector<unsigned int> Ns,
                        unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv computation : C = A*B
 * @param[in] weight char * for int4 quantized weight
 * @param[in] scale uint16_t * for scales
 * @param[in] input uint16_t * for input
 * @param[in] output uint16_t * for output
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemv_int4_cl(char *weight, uint16_t *scale, uint16_t *input,
                  uint16_t *output, unsigned int K, unsigned int N,
                  unsigned int quantization_group_size);

/**
 * @brief     signed 4-bit integer gemv computation : C = A*B
 * @param[in] weight char * for int4 quantized weight
 * @param[in] scale uint16_t * for scales
 * @param[in] input float * for input
 * @param[in] output float * for output
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemv_int4_cl(char *weight, uint16_t *scale, float *input, float *output,
                  unsigned int K, unsigned int N,
                  unsigned int quantization_group_size);

/**
 * @brief     Q4_0 gemm async computation : C = A*B
 * @param[in] matAdata std::vector<void *> for Matrix A
 * @param[in] matBdata float * for Matrix B
 * @param[in] matCdata std::vector<float *> for Matrix C
 * @param[in] M input dimension
 * @param[in] N output dimensions of As
 * @param[in] K hidden dimension
 */
void gemm_q4_0_async_cl(std::vector<void *> matAdata, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

/**
 * @brief     Q4_0 gemm computation : C = A*B
 * @param[in] matAdata void * for Matrix A
 * @param[in] matBdata float * for Matrix B
 * @param[in] matCdata float * for Matrix C
 * @param[in] M input dimension
 * @param[in] K hidden dimension
 * @param[in] N output dimension
 */
void gemm_q4_0_cl(void *matAdata, float *matBdata, float *matCdata,
                  unsigned int M, unsigned int N, unsigned int K);

/**
 * @brief INT4 GEMM computation for float input / output
 */
void openvino_sgemm_cl(float *input, char *weight, uint16_t *scale,
                       float *output, unsigned int M, unsigned int N,
                       unsigned int K, unsigned int quantization_group_size);
/**
 * @brief INT4 GEMM computation for fp16 input / output
 */
void openvino_gemm_cl(void *input, void *weights, void *scales, void *output,
                      unsigned int M, unsigned int N, unsigned int K,
                      unsigned int quantization_group_size);

/**
 * @brief INT4 input quantization using quantize_input_int4_pad kernel
 */
void openvino_quantize_input_int4_pad(void *input, void *quantized_input, void *scales,
                                      unsigned int M, unsigned int K,
                                      unsigned int quantization_group_size);

/**
 * @brief INT4 GEMM async computation
 */
void openvino_gemm_async_cl(float *input, std::vector<void *> weights,
                            std::vector<uint16_t *> scales,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> Ns, unsigned int K,
                            unsigned int quantization_group_size);

/**
 * @brief     Q6_K sgemv computation : Y = A*X
 * @param[in] matAdata void * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] M number of rows in matrix A
 * @param[in] N number of columns in matrix A
 */
void sgemv_q6_k_cl(void *matAdata, float *vecXdata, float *vecYdata,
                   unsigned int M, unsigned int N);

/**
 * @brief     sgemv computation : Y = A*X + Y
 * @param[in] matAdata float * for Matrix A
 * @param[in] vecXdata float * for Vector X
 * @param[in] vecYdata float * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     dot computation : sum of all X * Y
 * @param[in] vecAdata float * for Vector A
 * @param[in] vecXdata float * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    float dot product result
 */
float dot_cl(const float *vecAdata, const float *vecXdata, unsigned int dim1);

/**
 * @brief     sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A float * for Matrix A
 * @param[in] B float * for Matrix B
 * @param[in] C float * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const float *A, const float *B,
              float *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     addition : sum of all input vectors
 * @param[in] input float * for input
 * @param[in] res float * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cl(const float *input, float *res, unsigned int size_input,
                 unsigned int size_res);

/**
 * @brief rmsnorm each row of the tensor
 * @param[in] input float * for input
 * @param[in] gamma float * for gamma multiplier for each row
 * @param[in] result float * for result
 * @param[in] epsilon epsilon to add to each row sum to prevent division by zero
 * @param[in] height height of the tensor
 * @param[in] width width of the tensor
 * @param[in] use_svm whether to treat pointers as SVM
 */
void rmsnorm_cl(const float *input, const float *gamma, float *result,
                const float epsilon, unsigned int height, unsigned int width,
                const bool use_svm = true);

/**
 * @brief     sscal value element by element immediately
 * @param[in] X float * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(float *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input float * for Input Tensor
 * @param[in] res float * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels & width
 */
void transpose_cl_axis(const float *in, float *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
/**
 * @brief  Separate the quantized bits and scale from block_q4_0
 *
 * @param src source pointer to the block_q4_0 data
 * @param dst_q destination pointer for the quantized bits
 * @param dst_d destination pointer for the scale
 * @param num_blocks number of blocks to process
 */
void flatten_block_q4_0_cl(const void *src, void *dst_q, void *dst_d,
                           unsigned int num_blocks);

/**
 * @brief Restore the original block_q4_0 from the quantized bits and scale
 *
 * @param src_q source pointer to the quantized bits
 * @param src_d source pointer to the scale
 * @param dst destination pointer for the restored block_q4_0
 * @param num_blocks number of blocks to process
 */
void restore_block_q4_0_cl(const void *src_q, const void *src_d, void *dst,
                           unsigned int num_blocks);

/**
 * @brief This kernel load & store a 4x4 tile of elements
 *
 * @param data Input FP32 matrix data
 * @param M width (row)
 * @param K height (col)
 *
 * @note This kernel is only used for activations
 * Activation is coverted to FP16 and adds zero padding for non multiple of 8
 * Output is not returned and instead saved to outBufferB
 */
void transpose_32_16(float *data, int M, int K);

/**
 * @brief This kernel transpose fp16 type
 *
 * @param data input fp16 matrix data
 * @param output output fp16 matrix data
 * @param width widh
 * @param height height
 * @param size_bytes data size in bytes
 *
 * @note Temporary disable transpose 16
 */
// void transpose_16(void *data, void *output, int width, int height,
//                   int size_bytes, bool isQuant = false);

#ifdef ENABLE_FP16

/**
 * @brief     fp16 sgemv computation : Y = A*X + Y
 * @param[in] matAdata fp16 * for Matrix A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] vecYdata fp16 * for Vector Y
 * @param[in] transA bool transpose
 * @param[in] dim1 number of A's columns
 * @param[in] dim2 number of A's rows
 * @param[in] lda number of X's columns
 * @param[in] context RunLayerContext reference
 */
void sgemv_cl(const _FP16 *matAdata, const _FP16 *vecXdata, _FP16 *vecYdata,
              bool TransA, unsigned int dim1, unsigned int dim2,
              unsigned int lda);

/**
 * @brief     fp16 dot computation : sum of all X * Y
 * @param[in] vecAdata fp16 * for Vector A
 * @param[in] vecXdata fp16 * for Vector X
 * @param[in] dim1 number of elements in both input vectors
 * @param[in] context RunLayerContext reference
 * @return    fp16 dot product result
 */
_FP16 dot_cl(const _FP16 *vecAdata, const _FP16 *vecXdata, unsigned int dim1);

/**
 * @brief     fp16 sgemm computation : Y = op(A)*op(B) + C,
 * where op(X) is one of X or X**T
 * @param[in] transA bool transpose
 * @param[in] transB bool transpose
 * @param[in] A fp16 * for Matrix A
 * @param[in] B fp16 * for Matrix B
 * @param[in] C fp16 * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] lda number of A's columns
 * @param[in] ldb number of B's columns
 * @param[in] ldc number of C's columns
 * @param[in] context RunLayerContext reference
 */
void sgemm_cl(bool TransA, bool TransB, const _FP16 *A, const _FP16 *B,
              _FP16 *C, unsigned int M, unsigned int N, unsigned int K,
              unsigned int lda, unsigned int ldb, unsigned int ldc);

/**
 * @brief     fp16 addition : sum of all input vectors
 * @param[in] input fp16 * for input
 * @param[in] res fp16 * for result/output
 * @param[in] size_input number of elements in input vector
 * @param[in] size_res number of elements in result vector
 */
void addition_cl(const _FP16 *input, _FP16 *res, unsigned int size_input,
                 unsigned int size_res);

/**
 * @brief     fp16 sscal value element by element immediately
 * @param[in] X _FP16 * input
 * @param[in] N unsigned int number of elements
 * @param[in] alpha float multiplier
 * @param[in] context RunLayerContext reference
 */
void sscal_cl(_FP16 *X, const unsigned int N, const float alpha);

/**
 * @brief     transpose computation
 * @param[in] input fp16 * for Input Tensor
 * @param[in] res fp16 * for Output Tensor
 * @param[in] input_batch_size  represents the number of samples in the input
 * tensor
 * @param[in] input_channels   represents the channels of the input tensor
 * @param[in] input_height   represents the height of the input tensor
 * @param[in] input_width   represents the width of the input tensor
 * @param[in] axis   transpose about axis, 0-> channels & height, 1-> height &
 * width, 2-> channels and width
 */
void transpose_cl_axis(const _FP16 *in, _FP16 *res,
                       unsigned int input_batch_size,
                       unsigned int input_channels, unsigned int input_height,
                       unsigned int input_width, unsigned int axis);
#endif

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */
