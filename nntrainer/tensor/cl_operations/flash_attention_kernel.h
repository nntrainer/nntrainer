// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Anup Kumar Tiwari(anup.tiwari@samsung.com)
 *
 * @file	flash_attention_kernel.h
 * @date	23 March 2026
 * @brief	Flash attention OpenCL kernel interface
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Anup Kumar Tiwari(anup.tiwari@samsung.com)
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __FLASH_ATTENTION_KERNEL_H__
#define __FLASH_ATTENTION_KERNEL_H__

#include <cl_kernels/flash_attention.h>
#include <cl_kernels/flash_attention_fp16.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Flash Attention kernel for FP32
 * @param[in] query query tensor
 * @param[in] key key tensor
 * @param[in] value value tensor
 * @param[out] output output tensor
 * @param[in] attention_mask attention mask tensor (can be null)
 * @param[in] batch_size batch size
 * @param[in] num_heads number of attention heads
 * @param[in] seq_len sequence length
 * @param[in] head_dim head dimension
 * @param[in] scale scaling factor
 */
void flash_attention_cl(const Tensor &query, const Tensor &key, const Tensor &value,
                        Tensor &output, const Tensor *attention_mask,
                        int batch_size, int num_heads, int seq_len, int head_dim,
                        float scale);

#ifdef ENABLE_FP16
/**
 * @brief     Flash Attention kernel for FP16
 * @param[in] query query tensor
 * @param[in] key key tensor
 * @param[in] value value tensor
 * @param[out] output output tensor
 * @param[in] attention_mask attention mask tensor (can be null)
 * @param[in] batch_size batch size
 * @param[in] num_heads number of attention heads
 * @param[in] seq_len sequence length
 * @param[in] head_dim head dimension
 * @param[in] scale scaling factor
 */
void flash_attention_cl_fp16(const Tensor &query, const Tensor &key, const Tensor &value,
                             Tensor &output, const Tensor *attention_mask,
                             int batch_size, int num_heads, int seq_len, int head_dim,
                             float scale);
#endif

} // namespace nntrainer

#endif /* __FLASH_ATTENTION_KERNEL_H__ */