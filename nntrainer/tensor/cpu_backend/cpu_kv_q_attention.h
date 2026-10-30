// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   cpu_kv_q_attention.h
 * @date   29 Sep 2026
 * @brief  A8W8 attention over the int8 KV cache registry, on the CPU
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The CPU counterpart of the HTP quantized attention (hvx_attn_decode_q /
 * hexkl_attn_q_prefill): the same registry (hexkl_kv_q, masters
 * offset-binary, per-token K scales with column sums, per-token-and-group
 * V scales), the same arithmetic (Q per-row asymmetric uint8, integer
 * dot products, f32 base-2 softmax, P times the V scale quantized to uint8
 * per 32-row block and group), so a model configured with
 * attention_kv_dtype but without attention_engine runs the identical
 * scheme on the CPU. That is what makes a CPU-vs-DSP timing a comparison
 * of the kernels rather than of the data types.
 *
 * The dot products are udot (ARMv8.2 dotprod) where the build has it and
 * scalar otherwise; the work is split over ThreadManager by (query chunk,
 * head).
 */

#ifndef __NNTRAINER_CPU_KV_Q_ATTENTION_H__
#define __NNTRAINER_CPU_KV_Q_ATTENTION_H__

#include <cstdint>

namespace nntrainer {

/**
 * @brief Registers an int8 (kind 0) or int4 (kind 1) cache of @a max_rows
 *        rows. Only kind 0 has a CPU kernel; kind 1 is refused.
 * @return handle >= 0, or -1
 */
int cpu_kv_q_register(unsigned int kind, unsigned int max_rows,
                      unsigned int n_head_kv, unsigned int head_dim);

/**
 * @brief Quantizes fp16 rows [row0, row0 + n_rows) into the registry, in
 *        parallel over rows.
 */
bool cpu_kv_q_append(int handle, unsigned int row0, unsigned int n_rows,
                     unsigned int kv_stride, const uint16_t *k_rows,
                     const uint16_t *v_rows);

void cpu_kv_q_release(int handle);

/**
 * @brief Appends @a append_rows rows (may be 0) and attends: the contract
 *        of ComputeOps::sdpa_q_kvcache.
 */
bool cpu_sdpa_q_kvcache(int handle, unsigned int append_row0,
                        unsigned int append_rows, unsigned int kv_stride,
                        const uint16_t *k_rows, const uint16_t *v_rows,
                        const float *q, unsigned int q_stride, unsigned int n_q,
                        unsigned int cache_from, unsigned int cache_to,
                        unsigned int n_head_q, unsigned int n_head_kv,
                        unsigned int head_dim, unsigned int window,
                        float softcap, const float *sinks, float *out,
                        unsigned int out_stride);

} // namespace nntrainer

#endif /* __NNTRAINER_CPU_KV_Q_ATTENTION_H__ */
