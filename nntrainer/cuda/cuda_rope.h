// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_rope.h
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device RoPE (rotary position embedding) for the gemma4 path.
 *
 * Matches the host compute_rotary_emb_value split-half convention exactly:
 * for each row (token), head and each k in [0, head_dim/2):
 *   out[k]      = in[k]*cos[k] - in[k+half]*sin[k]
 *   out[k+half] = in[k]*sin[k] + in[k+half]*cos[k]
 * Full rotation over head_dim; FP32 math, fp16 I/O. Handles num_rows>1
 * (prefill) with a per-row position from+row. The cos/sin LUTs are flat device
 * buffers [num_positions * head_dim/2] (uploaded once by the caller).
 */

#ifndef __CUDA_ROPE_H__
#define __CUDA_ROPE_H__

namespace nntrainer::cuda {

/**
 * @brief  Apply RoPE on the device to interleaved fp16 rows
 *         [num_rows, num_heads*head_dim]. in/out + the LUTs are
 * device-resident.
 * @param in        [num_rows, num_heads*head_dim] fp16 bits (device)
 * @param out       same shape (device); may == in
 * @param cos_lut   flat device LUT [num_positions, head_dim/2] fp16 bits
 * @param sin_lut   flat device LUT [num_positions, head_dim/2] fp16 bits
 * @param num_heads heads packed per row
 * @param head_dim  per-head dim (256 sliding / 512 full); half = head_dim/2
 * @param num_rows  number of token rows (1 = decode, >1 = prefill big-step)
 * @param from      absolute position of row 0 (LUT row = from + row index)
 * @return true on success
 */
bool cuda_rope_fp16(const unsigned short *in, unsigned short *out,
                    const unsigned short *cos_lut,
                    const unsigned short *sin_lut, int num_heads, int head_dim,
                    int num_rows, int from);

/**
 * @brief Device-pos variant of cuda_rope_fp16: the RoPE position `from` is read
 * from the device cuda_pos_buffer() ([0]) instead of a baked int, so a captured
 *        CUDA graph stays valid across decode tokens. When @p out_slot_dpos !=
 * 0, each input row is written to OUTPUT row (from + row) -- pass @p out as the
 *        cache BASE pointer for the K-into-cache write so the slot is computed
 *        on-device from the live position. out_slot_dpos == 0 keeps
 * row-relative output (query, in-place). Same math as cuda_rope_fp16.
 */
bool cuda_rope_fp16_dpos(const unsigned short *in, unsigned short *out,
                         const unsigned short *cos_lut,
                         const unsigned short *sin_lut, int num_heads,
                         int head_dim, int num_rows, int out_slot_dpos,
                         int ring_cap = 0);

} // namespace nntrainer::cuda

#endif // __CUDA_ROPE_H__
