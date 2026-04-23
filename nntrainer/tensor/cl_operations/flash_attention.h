// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Tiwari <anup.tiwari@samsung.com>
 *
 * @file	flash_attention.h
 * @date	25 March 2026
 * @brief	Common flash attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Anup Tiwari <anup.tiwari@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __FLASH_ATTENTION_H__
#define __FLASH_ATTENTION_H__

#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

#ifdef ENABLE_FP16
#ifndef _FP16
#ifdef USE__FP16
#define _FP16 __fp16
#else
#define _FP16 _Float16
#endif
#endif
#endif

#include <string>

namespace nntrainer {


/**
 * @brief CPU reference implementation of flash attention with Grouped-Query Attention (GQA) support
 * @detail This function implements the flash attention algorithm on CPU for verification purposes.
 *         It supports Grouped-Query Attention where the number of query heads can be different
 *         from the number of key/value heads. The attention computation follows the standard
 *         scaled dot-product attention mechanism with softmax normalization.
 * @param[in] query Pointer to the query matrix data (float)
 * @param[in] key Pointer to the key matrix data (float)
 * @param[in] value Pointer to the value matrix data (float)
 * @param[out] output Pointer to the output matrix data (float)
 * @param[in] seqlen_q Sequence length of the query matrix
 * @param[in] seqlen_k Sequence length of the key matrix
 * @param[in] head_dim Dimension of each attention head
 * @param[in] num_heads_q Number of query attention heads
 * @param[in] num_heads_kv Number of key/value attention heads
 * @param[in] batch Batch size
 * @param[in] scale Scaling factor applied to the dot products (typically 1/sqrt(head_dim))
 * @note This is a reference implementation for correctness verification and not optimized for performance
 */
void flash_attention_cpu(const float *query, const float *key, 
                         const float *value, float *output,
                         int seqlen_q, int seqlen_k, int head_dim, 
                         int num_heads_q, int num_heads_kv, int batch,
                         float scale);

/**
 * @brief Flash Attention FP32 dispatch — selects prefill or decode kernel
 * @detail Top-level dispatch function that routes to the appropriate kernel
 *         based on seqlen_q. For seqlen_q == 1 (decode), uses the decode-optimized
 *         path. For seqlen_q > 1 (prefill), uses the prefill-optimized path.
 * @param[in] query float * for Query matrix
 * @param[in] key float * for Key matrix
 * @param[in] value float * for Value matrix
 * @param[out] output float * for Output matrix
 * @param[in] seqlen_q sequence length of query
 * @param[in] seqlen_k sequence length of key
 * @param[in] head_dim dimension of each attention head
 * @param[in] num_heads_q number of query attention heads
 * @param[in] num_heads_kv number of key/value attention heads
 * @param[in] batch batch size
 * @param[in] scale scaling factor for attention scores
 */
void flash_attention_fp32_cl(float *query, float *key, float *value, float *output,
                             unsigned int seqlen_q, unsigned int seqlen_k,
                             unsigned int head_dim, unsigned int num_heads_q,
                             unsigned int num_heads_kv, unsigned int batch,
                             float scale);

/**
 * @brief Flash Attention FP32 prefill kernel — optimized for many Q tokens
 * @detail Uses tiled GEMM approach for KQ and VKQ computation with online softmax.
 *         Each work-group processes multiple Q tokens (ncols1) against tiles of K/V.
 * @param[in] query float * for Query matrix
 * @param[in] key float * for Key matrix
 * @param[in] value float * for Value matrix
 * @param[out] output float * for Output matrix
 * @param[in] seqlen_q sequence length of query
 * @param[in] seqlen_k sequence length of key
 * @param[in] head_dim dimension of each attention head
 * @param[in] num_heads_q number of query attention heads
 * @param[in] num_heads_kv number of key/value attention heads
 * @param[in] batch batch size
 * @param[in] scale scaling factor for attention scores
 */
void flash_attention_prefill_fp32_cl(float *query, float *key, float *value,
                                     float *output, unsigned int seqlen_q,
                                     unsigned int seqlen_k, unsigned int head_dim,
                                     unsigned int num_heads_q,
                                     unsigned int num_heads_kv,
                                     unsigned int batch, float scale);

/**
 * @brief Flash Attention FP32 decode kernel — optimized for single Q token
 * @detail Uses split-KV approach where KV sequence is split across work-groups,
 *         each computing partial (max, sum, VKQ), then reduced via log-sum-exp.
 * @param[in] query float * for Query matrix
 * @param[in] key float * for Key matrix
 * @param[in] value float * for Value matrix
 * @param[out] output float * for Output matrix
 * @param[in] seqlen_q sequence length of query (must be 1 for decode)
 * @param[in] seqlen_k sequence length of key
 * @param[in] head_dim dimension of each attention head
 * @param[in] num_heads_q number of query attention heads
 * @param[in] num_heads_kv number of key/value attention heads
 * @param[in] batch batch size
 * @param[in] scale scaling factor for attention scores
 */
void flash_attention_decode_fp32_cl(float *query, float *key, float *value,
                                    float *output, unsigned int seqlen_q,
                                    unsigned int seqlen_k, unsigned int head_dim,
                                    unsigned int num_heads_q,
                                    unsigned int num_heads_kv,
                                    unsigned int batch, float scale);


#ifdef ENABLE_FP16
/**
 * @brief Flash Attention FP16 dispatch — selects prefill or decode kernel
 * @detail Top-level dispatch function that routes to the appropriate kernel
 *         based on seqlen_q. For seqlen_q == 1 (decode), uses the decode-optimized
 *         path. For seqlen_q > 1 (prefill), uses the prefill-optimized path.
 * @param[in] query _FP16 * for Query matrix
 * @param[in] key _FP16 * for Key matrix
 * @param[in] value _FP16 * for Value matrix
 * @param[out] output _FP16 * for Output matrix
 * @param[in] seqlen_q sequence length of query
 * @param[in] seqlen_k sequence length of key
 * @param[in] head_dim dimension of each attention head
 * @param[in] num_heads_q number of query attention heads
 * @param[in] num_heads_kv number of key/value attention heads
 * @param[in] batch batch size
 * @param[in] scale scaling factor for attention scores
 */
void flash_attention_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value, _FP16 *output,
                             unsigned int seqlen_q, unsigned int seqlen_k,
                             unsigned int head_dim, unsigned int num_heads_q,
                             unsigned int num_heads_kv, unsigned int batch,
                             float scale);

/**
 * @brief Flash Attention FP16 prefill kernel — optimized for many Q tokens
 * @detail Uses tiled GEMM approach with FP16 storage and FP32 accumulation.
 *         Each work-group processes multiple Q tokens against tiles of K/V.
 * @param[in] query _FP16 * for Query matrix
 * @param[in] key _FP16 * for Key matrix
 * @param[in] value _FP16 * for Value matrix
 * @param[out] output _FP16 * for Output matrix
 * @param[in] seqlen_q sequence length of query
 * @param[in] seqlen_k sequence length of key
 * @param[in] head_dim dimension of each attention head
 * @param[in] num_heads_q number of query attention heads
 * @param[in] num_heads_kv number of key/value attention heads
 * @param[in] batch batch size
 * @param[in] scale scaling factor for attention scores
 */
void flash_attention_prefill_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value,
                                     _FP16 *output, unsigned int seqlen_q,
                                     unsigned int seqlen_k, unsigned int head_dim,
                                     unsigned int num_heads_q,
                                     unsigned int num_heads_kv,
                                     unsigned int batch, float scale);

/**
 * @brief Flash Attention FP16 decode kernel — optimized for single Q token
 * @detail Uses split-KV approach where KV sequence is split across work-groups,
 *         each computing partial (max, sum, VKQ), then reduced via log-sum-exp.
 * @param[in] query _FP16 * for Query matrix
 * @param[in] key _FP16 * for Key matrix
 * @param[in] value _FP16 * for Value matrix
 * @param[out] output _FP16 * for Output matrix
 * @param[in] seqlen_q sequence length of query (must be 1 for decode)
 * @param[in] seqlen_k sequence length of key
 * @param[in] head_dim dimension of each attention head
 * @param[in] num_heads_q number of query attention heads
 * @param[in] num_heads_kv number of key/value attention heads
 * @param[in] batch batch size
 * @param[in] scale scaling factor for attention scores
 */
void flash_attention_decode_fp16_cl(_FP16 *query, _FP16 *key, _FP16 *value,
                                    _FP16 *output, unsigned int seqlen_q,
                                    unsigned int seqlen_k, unsigned int head_dim,
                                    unsigned int num_heads_q,
                                    unsigned int num_heads_kv,
                                    unsigned int batch, float scale);
#endif /* ENABLE_FP16 */


} // namespace nntrainer
#endif /* __FLASH_ATTENTION_H__ */