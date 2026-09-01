// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernels.h
 * @date	14 May 2024
 * @brief	Common blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BLAS_KERNELS_H__
#define __BLAS_KERNELS_H__

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_buffer_manager.h>
#include <opencl_kernel.h>

#include <functional>
#include <string>
#include <vector>

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
void sgemm_int4_cl(float *input, char *weight, uint16_t *scale, float *output,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int quantization_group_size);
/**
 * @brief INT4 GEMM computation for fp16 input / output
 */
void gemm_int4_cl(void *input, void *weights, void *scales, void *output,
                  unsigned int M, unsigned int N, unsigned int K,
                  unsigned int quantization_group_size);

/**
 * @brief INT4 GEMM async computation
 */
void gemm_int4_async_cl(float *input, std::vector<void *> weights,
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
              unsigned int lda, bool out_svm = false);

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
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              bool out_svm = false);

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
              unsigned int lda, bool out_svm = false);

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
              unsigned int lda, unsigned int ldb, unsigned int ldc,
              bool out_svm = false);

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

/**
 * @brief v8c int8 act × int4(channel-wise QINT4, offset-encoded) GEMM
 *        (paper-aligned 8/4/4 prefill path for quantized FC).
 * @param[in] act_image image2d_from_buffer view over int8 act buffer
 *            (CL_RGBA UINT32, width=K/16, height=M)
 * @param[in] weight_image image2d_from_buffer view over int4-offset weight buf
 *            (CL_RGBA UINT32, width=K/32, height=N)
 * @param[in] scale_act per-row fp32 act recip-scale buffer [M]
 * @param[in] scale_wgt per-channel fp32 weight recip-scale buffer [N]
 * @param[in] row_sum_act per-row int32 sum of int8 acts [M]
 * @param[in] zp_act per-row int32 asymmetric zero-point [M]
 * @param[in] row_sum_w_int4 per-channel int32 sum_k(int4 w_nk) [N], precomputed
 *            once at weight upload (depends only on weight bytes).
 * @param[out] output_fp16 fp16 output buffer [M*N]
 * @param[in] M,N,K shape; K must be multiple of 32
 * @param[in] M_valid valid output rows (0 = M): stores with row >= M_valid
 *            are skipped by the TM=4 kernels. Direct-output mode passes the
 *            unpadded row count with output_fp16 = the FC's planner
 *            sub-buffer (sized M_valid x N), so the M_pad padding rows
 *            cannot write out of bounds.
 */
void gemm_int8_v8c_cl(cl_mem act_image, cl_mem weight_image, cl_mem scale_act,
                      cl_mem scale_wgt, cl_mem row_sum_act, cl_mem zp_act,
                      cl_mem row_sum_w_int4, cl_mem output_fp16, unsigned int M,
                      unsigned int N, unsigned int K, unsigned int M_valid = 0);

/**
 * @brief #46l: v8c GEMM with fused OHWI-reversed V scatter. Same compute as
 *        gemm_int8_v8c_cl but writes outputs directly into V cache at the
 *        OHWI-reversed layout positions, eliminating the separate
 *        v_scatter_ohwi_t pass. Requires N == hKV * head_dim and
 *        V8C_TN (= 8) divides head_dim (so per-WI 8-wide n-tile stays in
 *        one head). N is the same FC output channel count; head_dim and
 *        S_max parameterize the OHWI cache geometry; position is the start
 *        token offset in the t-axis.
 *
 * @param[out] v_ohwi  cl_mem [hKV * head_dim * S_max] fp16
 * @param[in]  position  start token offset along the S_max axis
 */
void gemm_int8_v8c_v_ohwi_cl(cl_mem act_image, cl_mem weight_image,
                             cl_mem scale_act, cl_mem scale_wgt,
                             cl_mem row_sum_act, cl_mem zp_act,
                             cl_mem row_sum_w_int4, cl_mem v_ohwi,
                             unsigned int M_pad, unsigned int N, unsigned int K,
                             unsigned int head_dim, unsigned int S_max,
                             unsigned int position, unsigned int M_real);

/**
 * @brief Asymmetric int8 activation quantization for v8c.
 *        fp16/fp32 → int8 + per-row recip-scale + per-row int32 zero-point
 *        + per-row int32 sum. The scale/zp form matches KAI's qai8dxp_f32
 *        host packer so the v8c GPU path has the same robustness to
 *        single-sided outliers (post-SwiGLU activations etc.).
 * @param[in] act_fp16 or act_fp32 input buffer [M*K]
 * @param[out] out_int8 [M*K] int8 (row-major; later wrapped in image2d view)
 * @param[out] out_scale [M] fp32 per-row recip-scale = (rmax-rmin)/255
 * @param[out] out_zp [M] int32 per-row nudged zero-point
 * @param[out] out_row_sum [M] int32 sum_k(int8_value)
 * @param[in] M,K shape
 */
void quantize_act_v8c_fp16_cl(cl_mem act_fp16, cl_mem out_int8,
                              cl_mem out_scale, cl_mem out_zp,
                              cl_mem out_row_sum, unsigned int M,
                              unsigned int K);
void quantize_act_v8c_fp32_cl(cl_mem act_fp32, cl_mem out_int8,
                              cl_mem out_scale, cl_mem out_zp,
                              cl_mem out_row_sum, unsigned int M,
                              unsigned int K);

} // namespace nntrainer

#include "cl_tensor_view.h"
#include <memory>
namespace nntrainer {
/**
 * @brief Convert a channel-wise QINT4 weight (Int4QTensor osv32_isv2 + fp16
 *        per-group scales) into a v8c-ready backing: row-major + offset-encoded
 *        int4 in a single cl_mem buffer (image2d view created on demand via
 *        TensorBacking::imageView), plus a fp32 per-channel scale cl_mem.
 *        Paper §4.2 alignment: re-quantize from per-group (32) → per-channel
 *        (one scale per output row) during the conversion. ONE-TIME at FC init.
 * @param[in] osv32_packed   pointer to osv32 packed int4 bytes (N*K/2 bytes)
 * @param[in] fp16_scales    pointer to fp16 scales (N*K/group_size values)
 * @param[in] group_size     32 (Int4QTensor default)
 * @param[in] N              output channels
 * @param[in] K              input dim
 * @param[out] out_scale_buf cl_mem (fp32, [N], CL_MEM_READ_ONLY) — caller owns
 * @return TensorBacking holding the v8c row-major+offset weight buffer
 *         (Encoding::INT4_OFFSET, Layout::ROW_MAJOR, bytes = N*K/2)
 */
std::unique_ptr<tv::TensorBacking>
make_v8c_weight_backing(const uint8_t *osv32_packed,
                        const uint16_t *fp16_scales, unsigned int group_size,
                        unsigned int N, unsigned int K, cl_mem *out_scale_buf);

/**
 * @brief Build the v8c GEMM weight backing from an upstream
 * QS4CX plain weight: row-major nibbles (N x (K+1)/2 bytes, even k=low nibble,
 * uint4=int4+8, no XOR) + per-output-channel fp32 scale (range/15). Produces
 * the identical v8c backing/scale/row-sum the GEMM consumes.
 * @param[in] plain_nibbles QS4CX nibble payload (length N*((K+1)/2))
 * @param[in] fp32_scales   per-channel fp32 dequant scales (length N)
 * @param[in] cache_name    stable weight identity (the tensor name) for the
 *                          derive-once pack cache (v8c_pack_cache.h); nullptr
 *                          disables caching for this build. On a validated hit
 *                          the nibble permute and the row-sum fold are skipped
 *                          and the upload streams from the mapped pack; on a
 *                          miss the derive is teed to the pack writer. The
 *                          result is byte-identical either way.
 */
std::unique_ptr<tv::TensorBacking> make_v8c_weight_backing_from_qs4cx(
  const uint8_t *plain_nibbles, const float *fp32_scales, unsigned int N,
  unsigned int K, cl_mem *out_scale_buf, cl_mem *out_row_sum_w_int4_buf,
  const char *cache_name = nullptr);

/**
 * @brief Wait out and free any submit-and-go weight-upload staging queued by
 *        make_v8c_weight_backing_from_qs4cx (NNTR_V8C_UPLOAD_ASYNC, default
 *        on). Memory hygiene only: the in-order queue already sequences the
 *        writes ahead of any later GEMM. Cheap no-op when nothing is pending.
 */
void v8c_flush_pending_uploads();

/**
 * @brief 8/4/4 paper attention path: int8(act) × int8(weight) channel-wise
 * GEMM. Signature mirrors gemm_int8_v8c_cl (row_sum_act ignored). Weight image
 *        must be the plain row-major int8 view (width K/16). Dispatches the
 *        v8c_gemm_int8_int8 kernel (or _m1 for M<=4).
 */
void gemm_int8_int8_v8c_cl(cl_mem act_image, cl_mem weight_image,
                           cl_mem scale_act, cl_mem scale_wgt,
                           cl_mem row_sum_act, cl_mem zp_act, cl_mem row_sum_w,
                           cl_mem output_fp16, unsigned int M, unsigned int N,
                           unsigned int K);

/** @brief int8×int8 variant of gemm_int8_v8c_v_ohwi_cl (fused OHWI V scatter).
 */
void gemm_int8_int8_v8c_v_ohwi_cl(
  cl_mem act_image, cl_mem weight_image, cl_mem scale_act, cl_mem scale_wgt,
  cl_mem row_sum_act, cl_mem zp_act, cl_mem row_sum_w, cl_mem v_ohwi,
  unsigned int M_pad, unsigned int N, unsigned int K, unsigned int head_dim,
  unsigned int S_max, unsigned int position, unsigned int M_real);

/**
 * @brief Build a v8c int8-weight backing from a plain row-major int8 weight
 *        buffer + per-channel fp16 scales. Computes the fp32 scale buffer and
 *        per-channel signed int8 row sums (row_sum_w[n] = Σ_k int8 w[n,k]).
 *        Caller creates the image2d view (width=K/16, CL_UNSIGNED_INT32).
 */
std::unique_ptr<tv::TensorBacking> make_v8c_int8_weight_backing(
  const int8_t *int8_weights, const uint16_t *fp16_scales, unsigned int N,
  unsigned int K, cl_mem *out_scale_buf, cl_mem *out_row_sum_w_buf);

/**
 * @brief Decode lm_head GEMV on a Q6_K weight: logits[v] = Σ_k act[k] *
 *        dequant_q6_K(w[v,k]). One fp32 act row in, vocab fp32 logits out
 *        (blocking readback). The kernel is the gpu_native q6k_gemv_lmhead
 *        (ML Drift reaudit #1) verbatim: it replicates
 *        dequantize_row_q6_K_impl exactly; only the fp32 summation ORDER
 *        differs from the host dequant+sdot loop (per-WI partials + LDS
 *        tree), so logits drift in the lsb — the verification gate is
 *        token-ID equality over the greedy sequence. The Q6_K table
 *        (210 B per 256-elem block) is uploaded to a cached device buffer
 *        on first call (keyed by host pointer).
 * @return false when the GPU path is unavailable (no CL context, or hidden
 *         not a multiple of 256) — the caller must fall back to the host
 *         loop.
 */
bool lmhead_gemv_q6_k_cl(const void *w_q6k_host, const float *act_f32_host,
                         float *logits_f32_host, unsigned int vocab,
                         unsigned int hidden);

/**
 * @brief High-precision lm_head GEMV on an UNQUANTIZED FP32 weight (the tied
 *        embed table, --embd_dtype FP32): logits[v] = Σ_k W_fp32[v,k]*act[k],
 *        fp32 W × fp16 act, fp32 accumulate. The Q6_K lm_head loses ~1.66
 *        logit on the first-token argmax (the <think> vs garbage decision on
 *        Qwen3 thinking models => a garbage "noise prefix"); this reads the
 *        full-precision weight and matches the HF reference that ranks the
 *        correct token first. One 64-WI workgroup per vocab row, LDS tree
 *        reduce; weight uploaded to a cached device buffer on first call.
 * @param act_fp16_host  pointer to the fp16 activation row (hidden halfs).
 * @return false when the GPU path is unavailable — caller falls back.
 */
bool lmhead_gemv_fp32w_cl(const void *w_fp32_host, const void *act_fp16_host,
                          float *logits_f32_host, unsigned int vocab,
                          unsigned int hidden);

/**
 * @brief Decode lm_head GEMV on a QINT4 (v8c row-major) weight buffer, for the
 *        untied int4 lm_head whose N=vocab (262144) exceeds the image2d height
 *        cap so dotCl_v8c's image path cannot run. Reads the already-built v8c
 *        row-major nibble buffer directly (offset-encoded uint4, value = nibble
 *        - 8), with the activation kept fp16 and accumulated in fp32 (no int8
 *        act quant => best argmax fidelity, matching the q6k/fp32w lm_head
 *        kernels): logits[n] = scale_w[n] * Σ_k act[k] * (w_nibble[n,k] - 8).
 *        One 64-WI workgroup per vocab row, LDS-tree reduce. Reuses dotCl_v8c's
 *        cached weight_buf + scale_buf cl_mem (no extra upload).
 * Self-contained like lmhead_gemv_q6_k_cl: dispatches to a cached device logits
 * buffer and reads back to the host output (the lm_head output is consumed by
 * the host argmax/sampler), so the caller just passes its output host pointer.
 * @param w_buf_clmem     cl_mem: v8c row-major nibbles [N][K/2]
 * (offset-encoded)
 * @param scale_buf_clmem cl_mem: fp32 per-channel scale [N]
 * @param act             fp16 activation row [K] -- cl_mem if act_is_clmem else
 *                        an SVM pointer
 * @param logits_host     output host buffer [N], written as fp16 (out_fp16) or
 *                        fp32
 * @return false when the GPU path is unavailable (no CL context / kernel build
 *         failure) -- the caller falls back.
 */
bool lmhead_int4_v8c_gemv_cl(void *w_buf_clmem, void *scale_buf_clmem,
                             void *act, bool act_is_clmem, void *logits_host,
                             bool out_fp16, unsigned int N, unsigned int K);

/**
 * @brief Whether the v8c FC GEMM / KV attention use the cl_mem BUFFER path
 *        (Intel NEO) instead of the image2d path (Adreno). The env var
 * NNTR_V8C_BUF overrides; unset ⇒ derived from DeviceCaps::image_v8c
 * (vendor_id). Resolved once per process. Single source of truth for the
 * V8C_BUF cell.
 */
bool v8c_use_buffer_path();

/**
 * @brief Collect eager-build tasks for the v8c programs that would otherwise
 *        be compiled on their first dispatch, inside the first prefill and the
 *        first decode step.
 *
 * This translation unit owns both their sources and the compile options their
 * dispatch passes, so the prewarmed program keys match the hot path's exactly;
 * a prewarm built with different options produces a program that is never
 * looked up, paying the compile twice and saving nothing.
 *
 * @note Runs inside ClContext bring-up: nothing on this path may call
 *       ClContext::Global() (v8c_use_buffer_path() does), which would re-enter
 *       the context's one-time initialization and deadlock.
 *
 * @param cc context being brought up, whose caps decide the compile options
 * @param out task list to append to
 */
void v8c_collect_lazy_program_tasks(ClContext &cc,
                                    std::vector<std::function<void()>> &out);

/**
 * @brief Block until the GPU command queue drains (clFinish).
 *
 * A host-coherence barrier for the point where a host operation is about to
 * read a buffer that a previous GPU operation wrote asynchronously. The queue
 * is in-order, so draining it is the whole ordering guarantee the caller needs.
 */
void cl_queue_finish();

/**
 * @brief Take a shared-virtual-memory buffer back for the host.
 *
 * The counterpart of cl_svm_unmap_force below, and the reason both have to
 * exist: at a genuine GPU->host boundary the host must map the buffer before
 * it reads it, or it reads a stale shadow of the bytes the device wrote --
 * fluent, wrong output with no crash to notice it by. Blocking, so the
 * mapping is in place when the call returns. A no-op when there is no GPU
 * context, which is what a CPU-only run has.
 *
 * @param ptr shared-virtual-memory pointer to map; null is ignored
 * @param bytes length of the region to map; zero is ignored
 * @param read_only true when the host only reads through the mapping
 */
void cl_svm_map_force(void *ptr, size_t bytes, bool read_only);

/**
 * @brief Hand a shared-virtual-memory buffer back to the device.
 *
 * Use after a genuine HOST write to an SVM activation, so the GPU kernels that
 * read it next see coherent data. A no-op when there is no GPU context, which
 * is what a CPU-only run has.
 *
 * @param ptr shared-virtual-memory pointer to unmap; null is ignored
 */
void cl_svm_unmap_force(void *ptr);

} // namespace nntrainer
#endif /* __BLAS_KERNELS_H__ */
