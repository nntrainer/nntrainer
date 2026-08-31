// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qwen3_lowering.h
 * @date	19 August 2026
 * @brief	qwen3 model-specific lowering recipe: builds the v2 op-list
 *		and the WEIGHTS layout plan for the qwen3 transformer shape,
 *		using the shared types/pack_weights() declared in
 *		nntrainer/tensor/hexagon/host/graph_lowering.h.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __CAUSALLM_HEXAGON_QWEN3_LOWERING_H__
#define __CAUSALLM_HEXAGON_QWEN3_LOWERING_H__

#include "graph_lowering.h"

namespace nntrainer::hexagon {

/**
 * @brief Lower a qwen3 model shape into the v2 op-list plus the WEIGHTS
 *        and ACT layout plans. Pure shape computation, no source weight
 *        data is read.
 */
HexLoweredGraph lower_qwen3(const HexModelConfig &cfg);

} // namespace nntrainer::hexagon
#endif // __CAUSALLM_HEXAGON_QWEN3_LOWERING_H__
