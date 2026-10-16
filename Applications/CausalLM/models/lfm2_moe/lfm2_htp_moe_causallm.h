// SPDX-License-Identifier: Apache-2.0
/** @file lfm2_htp_moe_causallm.h
 * @brief LFM2 hybrid MoE using the shared Gemma4 HTP weight backend.
 * @author NNTrainer contributors
 * @bug Chunked prefill is not supported.
 */
#ifndef __LFM2_HTP_MOE_CAUSALLM_H__
#define __LFM2_HTP_MOE_CAUSALLM_H__
#include <climits>
#include <gemma4_htp_causallm.h>
namespace causallm {
/** LFM2 retains sigmoid+bias routing, SwiGLU, unscaled embeddings and
 * alternating conv/attention blocks. Only the HTP implementation is shared. */
class Lfm2HtpMoECausalLM : public Gemma4HtpCausalLM {
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

protected:
  void allocateAndBindKVCache() override;
  // Conv history is seeded once for this profile; prefill is a single block.
  bool chunkLongPrompts() const override { return false; }

private:
  unsigned dense_layers_ = 0;
  bool loading_ = false, prepared_ = false, ran_ = false;
  std::vector<std::vector<float>> params_;
  Tensor attentionBlock(unsigned layer, Tensor input);
  void prepareDecodeGraph();
};
} // namespace causallm
#endif
