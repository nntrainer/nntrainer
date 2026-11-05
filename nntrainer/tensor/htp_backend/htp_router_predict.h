// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_router_predict.h
 * @brief Prediction-only top-k on softmax and sigmoid router dot products.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#ifndef NNTR_HTP_ROUTER_PREDICT_H
#define NNTR_HTP_ROUTER_PREDICT_H
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <htp_wh_layout.h>
#include <numeric>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace nntrainer {
/** An inverse gamma is used only by speculative normalized-row prediction.
 * Reject the whole representation change when any channel cannot be inverted
 * safely, rather than mixing normalized and legacy residual coordinates. */
inline bool htpRouterGammaUsable(const float *gamma, unsigned width) {
  if (!gamma || !width)
    return false;
  for (unsigned h = 0; h < width; ++h)
    if (!std::isfinite(gamma[h]) || std::abs(gamma[h]) < 1.0e-6f)
      return false;
  return true;
}

/** Previous-token layer residuals, used exclusively for speculative routing.
 * A skipped observation or token never joins unrelated hidden states. */
class HtpRouterResidual {
  struct Row {
    std::vector<float> delta;
    uint64_t token = 0;
    bool valid = false;
  };
  std::vector<Row> rows_;
  std::vector<float> previous_;
  unsigned previous_layer_ = 0;
  uint64_t previous_token_ = 0;
  bool valid_ = false;

public:
  HtpRouterResidual(unsigned layers, unsigned width) :
    rows_(layers), previous_(width) {
    for (auto &row : rows_)
      row.delta.resize(width);
  }
  void observe(unsigned layer, uint64_t token, const float *input) {
    if (valid_ && token == previous_token_ && layer == previous_layer_ + 1u &&
        layer < rows_.size()) {
      auto &row = rows_[layer];
      for (size_t h = 0; h < previous_.size(); ++h)
        row.delta[h] = input[h] - previous_[h];
      row.token = token;
      row.valid = true;
    }
    std::copy(input, input + previous_.size(), previous_.begin());
    previous_layer_ = layer;
    previous_token_ = token;
    valid_ = true;
  }
  void apply(unsigned target, uint64_t token, const float *input, float blend,
             float *out) const {
    const Row *row = target < rows_.size() ? &rows_[target] : nullptr;
    const bool use = row && row->valid && token && row->token == token - 1u;
    for (size_t h = 0; h < previous_.size(); ++h)
      out[h] = input[h] + (use ? blend * row->delta[h] : 0.0f);
  }
};

/** Learn precision by layer and rank using completed real routes. Rejected
 * candidates are still observed, so a rank can recover without being read.
 * The caller serializes this with the expert table. */
class HtpPrefetchConfidence {
  struct Rank {
    float hits = 2, count = 3;
  };
  struct Row {
    std::array<Rank, 16> ranks;
    std::vector<unsigned> ids;
    std::vector<bool> absent;
    uint64_t epoch = 0;
    bool pending = false;
  };
  std::vector<Row> rows_;
  float threshold_;

public:
  HtpPrefetchConfidence(size_t layers, float threshold) :
    rows_(layers), threshold_(threshold) {}
  bool allow(size_t layer, size_t rank) const {
    const auto &r = rows_.at(layer).ranks.at(rank);
    return r.hits >= threshold_ * r.count;
  }
  void record(size_t layer, uint64_t epoch, const std::vector<unsigned> &ids,
              const std::vector<bool> &absent) {
    auto &r = rows_.at(layer);
    r.ids = ids;
    r.absent = absent;
    r.epoch = epoch;
    r.pending = true;
  }
  void observe(size_t layer, uint64_t epoch, const uint32_t *actual, size_t n) {
    auto &r = rows_.at(layer);
    if (!r.pending || r.epoch != epoch)
      return;
    for (size_t k = 0; k < r.ids.size() && k < r.ranks.size(); ++k)
      if (k < r.absent.size() && r.absent[k]) {
        auto &rank = r.ranks[k];
        const bool hit = std::find(actual, actual + n, r.ids[k]) != actual + n;
        rank.hits = 0.95f * rank.hits + float(hit);
        rank.count = 0.95f * rank.count + 1.0f;
      }
    r.pending = false;
  }
};

/** Positive RMS normalization and softmax preserve logit ranking. Per-expert
 * scales apply after selection. A sigmoid router instead ranks sigmoid(logit)
 * plus its expert bias; its input has already passed the layer's RMS norm.
 * This approximation is only an I/O hint; the real NPU router still supplies
 * all expert IDs and mixture weights.
 */
inline std::vector<unsigned>
htpRankPredictedExperts(std::vector<float> &scores, unsigned top,
                        const float *sigmoid_bias) {
  const unsigned experts = scores.size();
  for (float score : scores)
    if (!std::isfinite(score))
      return {};
  if (sigmoid_bias) {
    for (unsigned e = 0; e < experts; ++e) {
      if (!std::isfinite(sigmoid_bias[e]))
        return {};
      const float ex = std::exp(-std::abs(scores[e]));
      scores[e] = (scores[e] >= 0.0f ? 1.0f / (1.0f + ex) : ex / (1.0f + ex)) +
                  sigmoid_bias[e];
    }
  }
  std::vector<unsigned> ids(experts);
  std::iota(ids.begin(), ids.end(), 0u);
  top = std::min(top, experts);
  std::partial_sort(
    ids.begin(), ids.begin() + top, ids.end(), [&](unsigned a, unsigned b) {
      return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    });
  ids.resize(top);
  return ids;
}
/** Rank a borrowed row-major FP32 gate; no expanded model copy is retained. */
inline std::vector<unsigned>
htpPredictExperts(const float *input, const float *weights, unsigned hidden,
                  unsigned experts, unsigned top,
                  const float *sigmoid_bias = nullptr) {
  if (!input || !weights || !hidden || !experts || !top)
    return {};
  std::vector<float> scores(experts, 0.0f);
  for (unsigned h = 0; h < hidden; ++h)
    for (unsigned e = 0; e < experts; ++e)
      scores[e] += input[h] * weights[static_cast<size_t>(h) * experts + e];
  return htpRankPredictedExperts(scores, top, sigmoid_bias);
}

/** Prediction-only dot directly from the affine WH image. Keep the small
 * packed router borrowed; do not retain an FP32 expansion per layer. The
 * exact DSP router still selects all experts used for inference. */
inline std::vector<unsigned>
htpPredictExpertsWh(const float *input, const void *weights, unsigned hidden,
                    unsigned experts, unsigned top,
                    const float *sigmoid_bias = nullptr) {
  if (!input || !weights || !hidden || !experts || !top || hidden % WH_TILE ||
      experts % WH_TILE)
    return {};
  const auto *packed = static_cast<const uint8_t *>(weights);
  const auto *scale =
    reinterpret_cast<const float *>(packed + whBytes(hidden, experts));
  const float *offset = scale + 2 * experts;
  std::vector<float> scores(experts, 0.0f);
#if defined(__aarch64__) && !defined(__AARCH64EB__) &&                         \
  !defined(NNTR_ROUTER_PREDICT_SCALAR)
  // Keep one WH panel's accumulators and affine offsets in registers.
  // A word contains four consecutive K values for one expert; its high
  // nibbles hold the next four. Decode four experts with one vector load,
  // preserving the original K order without expanding the packed gate.
  const uint32x4_t mask = vdupq_n_u32(15u);
  const int32x4_t sign = vdupq_n_s32(8);
  for (unsigned block = 0; block < experts; block += WH_TILE) {
    float32x4_t sum[WH_TILE / 4], bias[WH_TILE / 4];
    for (unsigned c = 0; c < WH_TILE / 4; ++c) {
      sum[c] = vdupq_n_f32(0.0f);
      bias[c] = vld1q_f32(offset + block + c * 4u);
    }
    for (unsigned h = 0; h < hidden; ++h) {
      const uint8_t *row =
        packed +
        ((size_t)(h / WH_TILE) * (experts / WH_TILE) + block / WH_TILE) *
          WH_TILE_BYTES +
        (h % WH_TILE / 8u) * 128u;
      const int32x4_t shift =
        vdupq_n_s32(-int((h % 4u) * 8u + ((h / 4u) % 2u) * 4u));
      for (unsigned c = 0; c < WH_TILE / 4; ++c) {
        const uint32x4_t word = vreinterpretq_u32_u8(vld1q_u8(row + c * 16u));
        const int32x4_t nibble =
          vreinterpretq_s32_u32(vandq_u32(vshlq_u32(word, shift), mask));
        const int32x4_t q = vsubq_s32(veorq_s32(nibble, sign), sign);
        const float32x4_t value = vaddq_f32(vcvtq_f32_s32(q), bias[c]);
        sum[c] = vaddq_f32(sum[c], vmulq_n_f32(value, input[h]));
      }
    }
    for (unsigned c = 0; c < WH_TILE / 4; ++c)
      vst1q_f32(scores.data() + block + c * 4u, sum[c]);
  }
#else
  for (unsigned h = 0; h < hidden; ++h) {
    const unsigned shift = ((h / 4u) % 2u) * 4u;
    for (unsigned block = 0; block < experts; block += WH_TILE) {
      const uint8_t *row =
        packed +
        ((size_t)(h / WH_TILE) * (experts / WH_TILE) + block / WH_TILE) *
          WH_TILE_BYTES +
        (h % WH_TILE / 8u) * 128u + h % 4u;
      for (unsigned e = 0; e < WH_TILE; ++e) {
        const unsigned nibble = (row[e * 4u] >> shift) & 15u;
        const int q = nibble < 8u ? (int)nibble : (int)nibble - 16;
        scores[block + e] += input[h] * (q + offset[block + e]);
      }
    }
  }
#endif
  for (unsigned e = 0; e < experts; ++e)
    scores[e] *= scale[e];
  return htpRankPredictedExperts(scores, top, sigmoid_bias);
}

} // namespace nntrainer
#endif
