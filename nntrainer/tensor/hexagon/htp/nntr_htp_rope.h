// SPDX-License-Identifier: Apache-2.0
/**
 * @file	nntr_htp_rope.h
 * @date	17 September 2026
 * @brief	RoPE cos/sin table row shared by the host packer
 *		(graph_lowering.cpp) and the simulator reference (ref_ops.c):
 *		one fp32 generator so the two tables cannot drift. Each side
 *		keeps its own fp32 -> fp16 conversion. Plain C; needs <math.h>,
 *		which is why it is not part of nntr_htp_common.h.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_ROPE_H
#define NNTR_HTP_ROPE_H

#include <math.h>
#include <stdint.h>

/**
 * @brief Row p of the RoPE table for head_dim 128:
 *        row[i] = cos(p * inv_freq_i), row[64 + i] = sin(p * inv_freq_i),
 *        inv_freq_i = theta^(-2i/128), i in [0, 64). fp32 throughout.
 */
static inline void nntr_htp_rope_row_f32(float row[128], uint32_t p,
                                         float theta) {
  uint32_t i;
  for (i = 0; i < 64u; ++i) {
    float inv_freq = powf(theta, -2.0f * (float)i / 128.0f);
    row[i] = cosf((float)p * inv_freq);
    row[64u + i] = sinf((float)p * inv_freq);
  }
}

#endif /* NNTR_HTP_ROPE_H */
