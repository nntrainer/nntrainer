// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_softcap_f32.h
 * @date   7 October 2026
 * @brief  Logit softcap over a row of f32 on HVX, in place
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * y = cap * tanh(y / cap), the final logit softcap a model applies after
 * its lm_head (the CPU's logit_softcapping layer), so the logits can leave
 * the accelerator capped. The exp and the reciprocal are the GLU
 * epilogue's (hvx_swiglu_det.h), so the lane emulation runs it as is.
 */

#ifndef __NNTRAINER_HVX_SOFTCAP_F32_H__
#define __NNTRAINER_HVX_SOFTCAP_F32_H__

#include <stdint.h>

/**
 * @brief y[i] = cap * tanh(y[i] / cap) for i < n.
 * @param n    any count; a tail past the last whole vector is scalar
 * @param cap  > 0
 * @return 0, or -1 for cap <= 0 or a null y (nothing written)
 */
int hvx_softcap_f32(float *y, uint32_t n, float cap);

#endif /* __NNTRAINER_HVX_SOFTCAP_F32_H__ */
