// SPDX-License-Identifier: Apache-2.0
/** @file htp_attention_state.h
 * @brief Owns attention cache handles, calibration and resident decode handoff.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_ATTENTION_STATE_H
#define NNTR_HTP_ATTENTION_STATE_H
#include <attention_config.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tensor.h>
#include <vector>
namespace nntrainer {
class HtpAttentionState {
public:
  explicit HtpAttentionState(AttentionConfig &config) : config_(config) {}
  ~HtpAttentionState() { release_quantized_cache(); }
  HtpAttentionState(const HtpAttentionState &) = delete;
  HtpAttentionState &operator=(const HtpAttentionState &) = delete;
  void invalidate() {
    for (auto &row : q_cache_synced)
      row = 0;
  }
  void name(const std::string &value) {
    if (q2_name_.empty())
      q2_name_ = value;
  }
  bool failed() const { return q_cache_failed; }
  const std::string &name() const { return q2_name_; }
  bool ownsCache() const { return htp_owns_cache_; }
  /** Whether the layer's CPU KV rows are gone (its decode cache holds
   *  them): the model's compaction moves this layer's base only. */
  bool kvDetached() const { return kv_detached_; }
  /** htpKvAdopt with the decode caches registered before the prefill
   *  (htpDecodeKvPrepare): the layer's prefill seeds its decode cache
   *  (attention layer @a ordinal in graph order, based at absolute
   *  position @a decode_base) and releases its int8 cache there.
   *  @a ordinal -1: the first token seeds from the CPU rows. */
  void setHtpPrefillSeed(int ordinal, unsigned int decode_base) {
    htp_seed_ord_ = ordinal;
    htp_seed_base_ = decode_base;
  }
  /** Whether a call that attended is still holding its int8 cache: under
   *  int8-only KV every call ends seeding the decode cache, which releases
   *  it, so one left means a path returned without it. */
  bool holdsQuantizedCache(unsigned int batch) const {
    return batch < q_cache_handles.size() && q_cache_handles[batch] >= 0;
  }
  void markResident() { htp_owns_cache_ = true; }
  void beginHtpPrompt();
  bool htpDecodeAttention(
    Tensor &query, Tensor &key, Tensor &value, Tensor &output, Tensor &ck,
    Tensor &cv, unsigned int pos,
    const std::function<std::shared_ptr<const std::vector<float>>()>
      &make_rope);
  bool try_accelerated_attention(nntrainer::Tensor &query_step,
                                 nntrainer::Tensor &cached_key,
                                 nntrainer::Tensor &cached_value,
                                 nntrainer::Tensor &attention_output_step,
                                 unsigned int cache_from, unsigned int cache_to,
                                 const nntrainer::Tensor *sink);
  void registerQuantizedCache(unsigned int cache_rows);
  void release_quantized_cache();
  /** K's int8 scale per (KV head, dim) instead of per head, the decode
   *  cache's granularity: with the KV adoption (htp.kv_adopt). Both Q
   *  paths fold fs_kd / fs_k into Q's inverse step. */
  static bool q2_k_per_dim();
  void calibrate_q2_scales(const uint16_t *k_rows, const uint16_t *v_rows,
                           unsigned int n_rows, unsigned int kv_stride,
                           const float *q, unsigned int q_stride,
                           unsigned int n_q);
  bool q2_grow(const uint16_t *k_rows, const uint16_t *v_rows,
               unsigned int n_rows, unsigned int kv_stride);
  void q2_derive_kv();
  void q2_derive_q(const float *q, unsigned int q_stride, unsigned int n_q);
  void q2_keep();
  void foldDroppedAmax(const std::vector<float> &k_amax,
                       const std::vector<float> &v_amax);
  void q2_maxima_from_scales();
  bool take_q2_calibration();
  static void saveQ2Calibration(std::ostream &file);
  static void loadQ2Calibration(std::istream &file);
  bool try_quantized_attention(unsigned int batch,
                               nntrainer::Tensor &query_step,
                               nntrainer::Tensor &cache_key,
                               nntrainer::Tensor &cache_value,
                               const ml::train::TensorDim &cache_key_dim,
                               nntrainer::Tensor &attention_output_step,
                               unsigned int cache_from, unsigned int cache_to,
                               const nntrainer::Tensor *sink);

private:
  AttentionConfig &config_;
  std::vector<int> q_cache_handles;
  std::vector<unsigned int> q_cache_synced;
  bool q_cache_failed = false;
  std::vector<float> q2_scale_k, q2_scale_v, q2_q_enc, q2_out_enc;
  std::vector<float> q2_kmax, q2_vmax;
  static std::vector<uint16_t> q2_q_u16, q2_out_u16;
  bool q2_calibrated = false;
  inline static std::map<std::string, std::vector<float>> q2_store_;
  inline static bool q2_from_file_ = false;
  std::string q2_name_;
  std::vector<float> kv_floor_k_, kv_floor_v_;
  std::vector<unsigned char> q2_scales_set;
  bool accel_logged_ = false;
  /** [max_timestep][cos hd/2 | sin hd/2], shared by the layers of one RoPE
   *  cache over the first decode row (Fp16Attention's table cache) */
  std::shared_ptr<const std::vector<float>> htp_rope_table_;
  bool htp_owns_cache_ = false;
  bool kv_detached_ = false;       /**< see kvDetached */
  int htp_seed_ord_ = -1;          /**< see setHtpPrefillSeed */
  unsigned int htp_seed_base_ = 0; /**< see setHtpPrefillSeed */
};
} // namespace nntrainer
#endif
