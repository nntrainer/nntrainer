// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    model_features.h
 * @date    29 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   What a model IS — the model-side half of the ExecPlan resolver
 * input.
 *
 * Kept in its own dependency-free header so a model can declare its features
 * without including the Context header: these fields are model topology, not
 * device capability, and the model layer must not have to reach into the
 * backend layer to describe itself. The resolver (context.h) includes this
 * header, not the other way round.
 */

#ifndef __MODEL_FEATURES_H__
#define __MODEL_FEATURES_H__

#include <sstream>
#include <string>

namespace nntrainer {

/** @brief MLP kind a model uses (the gate activation). */
enum class MlpKind { SWIGLU, GEGLU };
/** @brief Transformer-block normalization placement. */
enum class NormStyle { PRE, SANDWICH };
/** @brief LM-head weight scheme. */
enum class LmHeadKind { TIED, UNTIED_QINT4 };

inline const char *toString(MlpKind k) {
  return k == MlpKind::SWIGLU ? "swiglu" : "geglu";
}
inline const char *toString(NormStyle s) {
  return s == NormStyle::SANDWICH ? "sandwich" : "pre";
}
inline const char *toString(LmHeadKind k) {
  return k == LmHeadKind::UNTIED_QINT4 ? "untied_qint4" : "tied";
}

/**
 * @struct ModelFeatures
 * @brief What the model IS, declared by the model itself (NOT inferred from a
 *        model-name/`is_gemma2` proxy). This is the other half of the resolver
 *        input: `ModelFeatures × DeviceCaps → ExecPlan`. Fields are attributes
 *        (independent feature combos), so a new model is "set the flags" with
 *        no backend edit. Currently consumed only by the SHADOW matcher
 *        overload in context.h (log-only, byte-identical).
 *        docs/backend_guide/ARCHITECTURE_REFACTOR.md §7.
 */
struct ModelFeatures {
  bool has_qk_norm = false; /**< per-head q/k RMSNorm (qwen3, gemma4) */
  bool has_v_norm = false;  /**< gamma-free v-norm (gemma4) */
  MlpKind mlp_kind = MlpKind::SWIGLU;
  NormStyle norm_style = NormStyle::PRE;
  bool sliding_window = false; /**< any sliding-window attention layers */
  bool kv_share_skip_prefill =
    false;                    /**< KV-shared layers skip prefill (gemma4) */
  bool dual_head_dim = false; /**< two head_dims (gemma4 sliding/global) */
  bool ple = false;           /**< per-layer input embedding (gemma4) */
  bool attn_softcap = false;  /**< QK logit soft-cap (gemma2/gemma4) */
  bool final_softcap = false; /**< final-logit soft-cap */
  LmHeadKind lmhead_kind = LmHeadKind::TIED;
  bool decode_gpu = false;      /**< GPU attn + OHWI-rope at decode (off for
                                     d=128); drives gpu_decode_attn /
                                     gpu_ohwi_rope on the attention layer */
  bool decode_rope_gpu = false; /**< GPU decode-RoPE (a strict subset of
                                     decode_gpu: gemma2 diverges, so attn=GPU
                                     but rope=HOST); drives gpu_decode_rope */
  unsigned int head_dim = 0;    /**< attention head dim (0 = derive) */

  /**
   * @brief One-line dump for the shadow log.
   */
  std::string toString() const {
    std::ostringstream os;
    os << "ModelFeatures{qk_norm=" << has_qk_norm << ", v_norm=" << has_v_norm
       << ", mlp=" << nntrainer::toString(mlp_kind)
       << ", norm=" << nntrainer::toString(norm_style)
       << ", sliding=" << sliding_window
       << ", kv_share=" << kv_share_skip_prefill
       << ", dual_head_dim=" << dual_head_dim << ", ple=" << ple
       << ", attn_softcap=" << attn_softcap
       << ", final_softcap=" << final_softcap
       << ", lmhead=" << nntrainer::toString(lmhead_kind)
       << ", decode_gpu=" << decode_gpu
       << ", decode_rope_gpu=" << decode_rope_gpu << ", head_dim=" << head_dim
       << "}";
    return os.str();
  }
};

} // namespace nntrainer

#endif // __MODEL_FEATURES_H__
