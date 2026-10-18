// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/**
 * @file lfm2_router_arm.h
 * @brief Exact FP32 four-row dot kernel for the opt-in LFM prefill router.
 */
#ifndef __LFM2_ROUTER_ARM_H__
#define __LFM2_ROUTER_ARM_H__

namespace causallm {
#if defined(__aarch64__) && defined(ENABLE_HEXKL)
/** The caller validates disjoint contiguous x[4,2048], w[2048,64] and
 * out[4,64]. The separate translation unit is compiled without contraction
 * or fast math: every k contributes one FP32 multiply followed by one add.
 * Returns false for subnormal input or any non-finite output; the caller uses
 * the existing DSP router to replace every logit and preserve its special-value
 * bits.
 */
/** False for subnormal weights; the caller retains the existing DSP route. */
bool lfm2RouterArmWeightSupported(const float *w);
bool lfm2RouterArmBlock(const float *x, const float *w, float *out);
#endif
} // namespace causallm
#endif
