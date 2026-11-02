// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_htp_moe_causallm.h
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#ifndef __GEMMA4_HTP_MOE_CAUSAL_LM_H__
#define __GEMMA4_HTP_MOE_CAUSAL_LM_H__

#include <gemma4_htp_causallm.h>

#include <string>
#include <vector>

namespace causallm {

/**
 * @brief Gemma4 sparse MoE variant.
 */
class Gemma4HtpMoECausalLM : public Gemma4HtpCausalLM {
public:
  Gemma4HtpMoECausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    Gemma4HtpCausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Gemma4HtpMoECausalLM() override = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Load weights, then build the HTP decode graph when htp_e2e is
   *        enabled. The graph needs the checkpoint's layer_scalar and
   *        binds its parameters by weight name.
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Gemma4HtpCausalLM::repack_weight (the experts' registration), then
   *        the E2E FC arena (finish_decode_graph_q4_0).
   */
  void repack_weight() override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = false) override;

private:
  /** Run the whole decode token on HTP when moe_engine=htp. */
  bool htp_e2e = false;
  bool loading_weights_ = false;
  bool weights_prepared_ = false;
  bool htp_run_started_ = false;
  /** Parameters assembled for the backend (q | k gammas, the
   *  router's g | per-expert scale), kept until its graph init */
  std::vector<std::vector<float>> htp_params;
};

} // namespace causallm

#endif // __GEMMA4_HTP_MOE_CAUSAL_LM_H__
