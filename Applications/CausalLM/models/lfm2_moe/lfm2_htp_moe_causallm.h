// SPDX-License-Identifier: Apache-2.0
/** @file lfm2_htp_moe_causallm.h
 * @brief LFM2 hybrid MoE using shared HTP preparation services.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef __LFM2_HTP_MOE_CAUSALLM_H__
#define __LFM2_HTP_MOE_CAUSALLM_H__
#include <causal_lm.h>
#include <climits>
#include <functional>
#include <htp_weight_preparation.h>
namespace causallm {
class ConvBlockLayer;
/** LFM2 retains sigmoid+bias routing, SwiGLU, unscaled embeddings and
 * alternating conv/attention blocks. Only the HTP implementation is shared. */
class Lfm2HtpMoECausalLM : public CausalLM {
public:
  Lfm2HtpMoECausalLM(json &cfg, json &gen, json &nntr);
  void setupParameters(json &cfg, json &gen, json &nntr) override;
  std::pair<Tensor, Tensor> constructModel() override;
  void registerCustomLayers() override;
  unsigned int getLayerSlidingWindow(int) const override { return UINT_MAX; }
  void load_weight(const std::string &path) override;
  void repack_weight() override;
  void run(const WSTR prompt, bool sample = false, const WSTR system = "",
           const WSTR tail = "", bool log = false) override;
  /** The attention caches as CausalLM writes them, plus every conv block's
   *  state in <path>.conv: a position of this model is both. */
  void save_kvcache(std::string path, int to) override;
  void load_kvcache(std::string path, int to) override;

protected:
  bool chunkLongPrompts() const override { return true; }
  void beforePrefillBlock(unsigned int from, unsigned int to) override;
  void allocateAndBindKVCache() override;

private:
  HtpWeightPreparation preparation_{};
  std::vector<std::string> layer_types;
  std::string ATTENTION_ENGINE, ATTENTION_KV_DTYPE, MOE_LAYER_DTYPE;
  unsigned NUM_EXPERTS = 0, NUM_EXPERTS_PER_TOK = 0;
  unsigned MOE_INTERMEDIATE_SIZE = 0, MOE_CACHE_EXPERTS = 0;
  unsigned dense_layers_ = 0;
  bool loading_ = false, prepared_ = false, ran_ = false;
  bool prefill_early_attempted_ = false;
  std::vector<std::vector<float>> params_;
  Tensor attentionBlock(unsigned layer, Tensor input);
  void configureActiveKVLayers();
  void prepareDecodeGraph();
  void forEachConvBlock(
    const std::function<void(ConvBlockLayer &, nntrainer::RunLayerContext &)>
      &fn);
};
} // namespace causallm
#endif
