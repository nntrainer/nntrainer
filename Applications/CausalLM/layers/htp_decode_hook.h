// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_decode_hook.h
 * @brief Bind fused CausalLM layers to the native HTP decode hooks.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#pragma once
#include <nntr_htp_decode_hook.h>
namespace causallm {
using nntrainer::htpDecodeAdd;
using nntrainer::htpDecodeAttn;
using nntrainer::htpDecodeBan;
using nntrainer::htpDecodeConvGate;
using nntrainer::htpDecodeHasTokenId;
using nntrainer::htpDecodeKvAmaxFloor;
using nntrainer::htpDecodeKvSeed;
using nntrainer::htpDecodeKvSeedF16;
using nntrainer::htpDecodeKvSeedKvq;
using nntrainer::htpDecodeKvWindow;
using nntrainer::htpDecodeLmHead;
using nntrainer::htpDecodeOp;
using nntrainer::htpDecodeQkNorm;
using nntrainer::htpDecodeRmsNorm;
using nntrainer::htpDecodeRouter;
using nntrainer::htpDecodeRowResident;
using nntrainer::htpDecodeTokenId;
using nntrainer::htpDecodeWantLogits;
} // namespace causallm
