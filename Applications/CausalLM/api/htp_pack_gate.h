// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    htp_pack_gate.h
 * @date    29 Oct 2026
 * @brief   Load-time contract of the HTP (Hexagon) model packs.
 *
 *          The check is a pure function of the requested model id, the
 *          execution profile and the pack's config files. It does not depend
 *          on ENABLE_HEXKL, the environment or the file system, so every
 *          build variant (and a host unit test) evaluates the same rules; only
 *          the caller decides what an accepted HTP pack means in its build.
 *
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __HTP_PACK_GATE_H__
#define __HTP_PACK_GATE_H__

#include <string>

#include "json.hpp"
#include "quick_dot_ai_api.h"

namespace quick_dot_ai {

/**
 * @brief Family of the HTP pack a load request resolved to.
 */
enum class HtpPackKind {
  NONE,   /**< not an HTP load; nothing to validate */
  GEMMA4, /**< Gemma4 26B-A4B htp_fused pack */
  LFM2,   /**< LFM2 24B-A2B htp_fused pack (sr15_v2 weight format) */
};

/**
 * @brief Validate the HTP pack contract of a load request.
 *
 * A request is an HTP request when its model id is one of the HTP ids or when
 * the nntr_config selects an htp_fused execution profile. Anything else
 * returns HtpPackKind::NONE without looking at the pack. An HTP request must
 * be a matching id + profile pair on the NPU backend with W4A32 weights and a
 * pack that passes the family's checks; the first violation throws.
 *
 * Accepted Gemma4 ids: GEMMA4-26B-A4B-HTP, GEMMA4-26B-A4B-HTP-LONG and
 * GEMMA4-26B-A4B-HTP-2BIT-4BIT. The LFM2 id is LFM2-24BA2B-HTP and its pack
 * must be the sr15_v2 format.
 *
 * Only the keys the load path itself reads are checked; keys that only the
 * engine consumes (for example static_activation_file or kv_window_ring) are
 * validated by the model, not here.
 *
 * @param model_id_upper requested model id, upper-cased
 * @param compute requested compute backend
 * @param quant requested weight/activation quantization
 * @param cfg the model's config.json
 * @param nntr_cfg the model's nntr_config.json
 * @return the pack family, HtpPackKind::NONE for a non-HTP request
 * @throws std::invalid_argument if the request selects an HTP pack that does
 *         not satisfy the contract
 */
HtpPackKind validate_htp_pack(const std::string &model_id_upper,
                              BackendType compute, ModelQuantizationType quant,
                              const nlohmann::json &cfg,
                              const nlohmann::json &nntr_cfg);

} // namespace quick_dot_ai

#endif /* __HTP_PACK_GATE_H__ */
