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
#include <numeric>
#include <vector>
namespace nntrainer {
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
htpPredictExperts(const float *input, const float *weights, unsigned hidden,
                  unsigned experts, unsigned top,
                  const float *sigmoid_bias = nullptr) {
  if (!input || !weights || !hidden || !experts || !top)
    return {};
  std::vector<float> scores(experts, 0.0f);
  for (unsigned h = 0; h < hidden; ++h)
    for (unsigned e = 0; e < experts; ++e)
      scores[e] += input[h] * weights[static_cast<size_t>(h) * experts + e];
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
} // namespace nntrainer
#endif
