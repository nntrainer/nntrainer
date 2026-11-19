// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   pi05_model.h
 * @brief  pi0.5 (Physical Intelligence) flow-matching VLA model for
 *         nntrainer: PaliGemma backbone graph + host-side Gemma action
 *         expert with adaRMS + Euler flow loop.
 * @date   12 Nov 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Architecture (openpi pi05_base, lerobot/pi05_base weights):
 *   - backbone = PaliGemma (SigLIP vision + Gemma1 18L MQA, width 2048) —
 *     runs as a REAL PaliGemmaCausalLM graph (use_embedding bypass), driven
 *     once over image rows followed by the scaled language embeddings.
 *     FP32 prefix K/V are exported directly into the expert cache; the
 *     vocabulary projection is omitted.
 *   - expert   = Gemma 18L MQA width 1024, adaRMS norms (Dense(3w)+bias,
 *     gate multiplies the branch at the residual), NO token embedder —
 *     evaluated with CPU BLAS or resident CUDA FP32 weights, cuBLAS, and
 *     device kernels for normalization, RoPE, attention and residuals.
 *   - flow matching: num_steps Euler iterations; step k runs the expert on
 *     the horizon noisy-action tokens (K/V appended to the shared cache
 *     rows [prefix_len, prefix_len + horizon)), positions = prefix_len + i,
 *     x_t += dt * v_t with dt = -1/num_steps from t = 1 to 0.
 *
 * Weights (res/pi05/weight_converter.py):
 *   model_file_name       backbone bin (PaliGemma bypass layout)
 *   embedding_bin_path    backbone embedding table (lookupEmbedding)
 *   expert_file_name      expert + flow-head records (graph order)
 *
 * The tiny fixture (pi05_tiny) injects golden vision embeddings into the
 * prefix plane (setVisionEmbeddings), so the CI test needs no vision graph;
 * the 3B e2e uses SiglipVisionModel exactly like the PaliGemma mm driver.
 */
#ifndef __PI05_MODEL_H__
#define __PI05_MODEL_H__

#include <model.h>
#include <transformer.h>

#include "paligemma/paligemma_causallm.h"

#include <memory>
#include <vector>

namespace causallm {
class PI05CudaExpert;

/**
 * @brief pi0.5 flow-matching model (backbone graph + CPU/CUDA expert loop).
 */
class PI05Model : virtual public Transformer {
public:
  /**
   * @brief Construct a new PI05Model object
   */
  PI05Model(json &cfg, json &generation_cfg, json &nntr_cfg);

  virtual ~PI05Model() = default;

  /**
   * @brief static architectures string used by the model registry
   */
  static constexpr const char *architectures = "PI05Model";

  /**
   * @brief flatten the HF config (text_config merge) and apply pi0.5
   *        defaults, in place (the Transformer ctor binds json&)
   */
  static json &sanitizeConfig(json &cfg);

  /**
   * @brief sanitize the generation config in place (openpi euler defaults)
   */
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  /**
   * @brief the base machinery still builds a (never-used) trivial graph so
   *        the Transformer contract holds; the real backbone graph lives in
   *        backbone_
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief load backbone weights (delegated) and the expert/flow records
   */
  void load_weight(const std::string &weight_file) override;

  /**
   * @brief delegate repacking to the backbone graph
   */
  void repack_weight() override;

  /**
   * @brief run flow-matching action generation
   *
   * @param prompt_ids  tokenized prompt INCLUDING bos (2) and any <image>
   *                    placeholders the caller wants replaced by vision rows
   * @param noise       [horizon, action_dim] initial noise
   * @param num_steps   Euler steps (openpi default 10)
   * @return [horizon, action_dim] action chunk (row-major)
   */
  std::vector<float> sampleActions(const std::vector<unsigned int> &prompt_ids,
                                   const std::vector<float> &noise,
                                   unsigned int num_steps);

  /**
   * @brief inject precomputed vision embeddings (tiny-test path); rows are
   *        consumed by the <image> placeholder positions in prompt_ids
   */
  void setVisionEmbeddings(const std::vector<float> &embs, unsigned int rows) {
    vision_embs_ = embs;
    vision_rows_ = rows;
  }

  /**
   * @brief per-step velocity field for the tiny test (exposed for the
   *        golden diff: v_t after each Euler step)
   */
  const std::vector<std::vector<float>> &perStepVelocities() const {
    return per_step_v_t_;
  }

private:
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  friend class PI05CudaExpert;
  std::shared_ptr<PI05CudaExpert> cuda_expert_;
  void prepareCudaExpert();
  void loadCudaExpertWeights();
  std::vector<float> denoiseStepCuda(const std::vector<float> &x,
                                     const std::vector<float> &cond);
#endif
  // ---- expert / flow hyperparameters (parsed from the sanitized cfg) ----
  unsigned int EXPERT_DIM;      /**< expert hidden width (1024 / 64 tiny) */
  unsigned int EXPERT_MLP_DIM;  /**< expert mlp width (4096 / 128 tiny) */
  unsigned int EXPERT_HEADS;    /**< expert query heads (8) */
  unsigned int EXPERT_KV_HEADS; /**< expert kv heads (1, MQA) */
  unsigned int EXPERT_HEAD_DIM; /**< expert head dim (256 / 16 tiny) */
  unsigned int ACTION_DIM;      /**< action chunk dim (32 / 8 tiny) */
  unsigned int ACTION_HORIZON;  /**< action tokens per chunk (50 / 4) */
  double TIME_MIN_PERIOD;       /**< sincos min period (4e-3) */
  double TIME_MAX_PERIOD;       /**< sincos max period (4.0) */
  float RMS_EPS;                /**< rms epsilon (1e-6) */
  float ROPE_THETA;             /**< rope base (10000) */

  // prefix geometry
  unsigned int prefix_len_;        /**< bos + images + prompt rows */
  std::vector<float> vision_embs_; /**< injected vision embeddings */
  unsigned int vision_rows_;
  std::string expert_file_name_; /**< expert + flow-head bin path */
  json sanitized_cfg_;           /**< pi0.5-specific config values */
  json gen_cfg_;                 /**< generation config (ctor copy) */
  json nntr_cfg_;                /**< nntr config (ctor copy) */

  // backbone producer (owns the real graph + embedding table)
  std::unique_ptr<PaliGemmaCausalLM> backbone_;

  // ---- expert weights (host, sliced from the expert bin in graph order) --
  struct ExpertLayer {
    std::vector<float> n1_w, n1_b;     /**< adaRMS dense [d,3d], [3d] */
    std::vector<float> qw, kw, vw, ow; /**< attention FCs (in-major) */
    std::vector<float> n2_w, n2_b;     /**< adaRMS dense */
    std::vector<float> gw, uw, dw;     /**< MLP FCs (in-major) */
  };
  std::vector<ExpertLayer> layers_;
  std::vector<float> fn_w, fn_b;           /**< final adaRMS dense */
  std::vector<float> act_in_w, act_in_b;   /**< action_in_proj [32,d] */
  std::vector<float> act_out_w, act_out_b; /**< action_out_proj [d,32] */
  std::vector<float> t_in_w, t_in_b;       /**< time_mlp_in [d,d] */
  std::vector<float> t_out_w, t_out_b;     /**< time_mlp_out [d,d] */
  bool expert_loaded_ = false;
  static void dumpDiagnostic(const std::string &name,
                             const std::vector<float> &values);

  std::vector<std::vector<float>> per_step_v_t_;
  unsigned int modulation_steps_ = 0;
  std::vector<std::vector<float>> time_conditions_, modulation_cache_;
  void prepareModulations(unsigned int steps);
  const float *modulation(size_t norm) const {
    return modulation_cache_.at(norm).data() +
           per_step_v_t_.size() * 3 * EXPERT_DIM;
  }

  /**
   * @brief sincos time embedding -> time MLP -> adaRMS cond vector [d]
   */
  std::vector<float> timeCondition(float t) const;

  /**
   * @brief one expert denoise step. backbone_kv_[l] holds the transplanted
   *        prefix K/V rows [0, prefix_len); the expert's own suffix K/V is
   *        written at [prefix_len, prefix_len + horizon).
   * @return v_t [horizon, action_dim]
   */
  std::vector<float> denoiseStep(const std::vector<float> &x_t,
                                 const std::vector<float> &cond);

  /**
   * @brief Host BLAS helpers using the loaded expert's CPU weight layout.
   */
  void matvec(const std::vector<float> &w, const std::vector<float> &x,
              std::vector<float> &y, unsigned int in_dim,
              unsigned int out_dim) const;
  void matmat(const std::vector<float> &w, const std::vector<float> &x,
              std::vector<float> &y, unsigned int rows, unsigned int in_dim,
              unsigned int out_dim) const;
  bool expert_mlp_fused_ = false;
  bool expert_weights_transposed_ = false;

  // shared cache (backbone prefix + expert suffix), per layer:
  // k_cache_[l] / v_cache_[l] sized [(prefix_len + horizon) * kv_width]
  std::vector<std::vector<float>> k_cache_, v_cache_;
  unsigned int kv_width_; /**< kv_heads * head_dim */
};

} // namespace causallm

#endif // __PI05_MODEL_H__
