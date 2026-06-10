// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Convenience greedy (temperature-0) sampling helpers for DDTree
 * @file   ddtree_sampling.h
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __NNTRAINER_DDTREE_SAMPLING_H__
#define __NNTRAINER_DDTREE_SAMPLING_H__

#include <cstdint>

namespace nntrainer {
namespace ddtree {

/** argmax over one row of `vocab` logits; ties resolve to the lowest index. */
int32_t argmaxRow(const float *logits, int vocab);

/** Per-row argmax for `rows` rows of `vocab` logits into out[rows]. */
void sampleGreedy(const float *logits, int rows, int vocab, int32_t *out);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_SAMPLING_H__
