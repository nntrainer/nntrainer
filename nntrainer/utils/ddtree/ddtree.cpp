// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  DDTree core implementation (buildTree / compile / followVerified)
 * @file   ddtree.cpp
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <ddtree.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include <thread_manager.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace nntrainer {
namespace ddtree {

// Dequantize one UFIXED16 logit: (q + offset) * scale, fp32.
inline float dequantU16(uint16_t q, float scale, int32_t offset) {
  return (static_cast<float>(q) + static_cast<float>(offset)) * scale;
}

namespace {

// Top-k candidate; "better" == (higher logit, lower token on a tie), matching
// torch.topk's (logit desc, token index asc) total order.
struct Cand {
  float logit;
  int32_t token;
};
inline bool candBetter(const Cand &a, const Cand &b) {
  if (a.logit != b.logit)
    return a.logit > b.logit; // higher logit is better
  return a.token < b.token;   // tie: lower token index is better
}

// One-pass top-k + double-precision logsumexp over one logits row.
// `rowAt(i)` returns the fp32 logit of vocab index i — a plain load for the
// float overload, a dequantU16 for the uint16 overload — so BOTH overloads run
// byte-identical selection and logZ math (fp32 compares, token-index-order
// double exp accumulation). Fills logProbs/tokens best-first (torch.topk
// parity, golden §6.2).
template <typename RowAt>
void scanRowTopK(RowAt rowAt, int vocab, int topk, std::vector<float> &logProbs,
                 std::vector<int32_t> &tokens) {
  // Single sequential pass top-k via a threshold + small array: the hot path
  // is one well-predicted `x < thresh` compare that rejects the ~99.98% of
  // tokens outside the top-k, so it avoids the per-element priority_queue
  // push/pop and Cand churn. `thresh` is the worst kept logit once `buf` is
  // full; ties (x == thresh) fall through to the full `candBetter` comparison
  // so the lower token index still wins. The selected set/order is identical
  // to a partial_sort (logit/token is a total order), preserving golden
  // parity.
  std::vector<Cand> buf;
  buf.reserve(topk);
  float thresh = -std::numeric_limits<float>::infinity();
  int worstPos = 0;
  for (int i = 0; i < vocab; ++i) {
    const float x = rowAt(i);
    if (static_cast<int>(buf.size()) < topk) {
      buf.push_back(Cand{x, i});
      if (static_cast<int>(buf.size()) == topk) {
        worstPos = 0;
        for (int j = 1; j < topk; ++j)
          if (candBetter(buf[worstPos], buf[j]))
            worstPos = j;
        thresh = buf[worstPos].logit;
      }
      continue;
    }
    if (x < thresh)
      continue; // outside top-k: single-compare fast reject
    const Cand c{x, i};
    if (!candBetter(c, buf[worstPos]))
      continue; // x == thresh tie resolved by token index
    buf[worstPos] = c;
    worstPos = 0; // recompute worst (rare: only on an actual insertion)
    for (int j = 1; j < topk; ++j)
      if (candBetter(buf[worstPos], buf[j]))
        worstPos = j;
    thresh = buf[worstPos].logit;
  }
  std::sort(buf.begin(), buf.end(), candBetter); // best-first (== partial_sort)
  // The global max is always kept, so it is the best candidate; reuse it for
  // logsumexp instead of a separate full-row max scan.
  const float maxLogit = buf[0].logit;

  // logsumexp in double for accuracy (token-index order, parity §6.2);
  // result stored as fp32.
  double sumExp = 0.0;
  for (int i = 0; i < vocab; ++i)
    sumExp += std::exp(static_cast<double>(rowAt(i)) - maxLogit);
  const float logZ = static_cast<float>(maxLogit + std::log(sumExp));

  logProbs.resize(topk);
  tokens.resize(topk);
  for (int r = 0; r < topk; ++r) {
    tokens[r] = buf[r].token;
    logProbs[r] = buf[r].logit - logZ; // fp32 subtraction
  }
}

#if defined(__aarch64__)
// Vectorized expf (Cephes-style two-part-ln2 range reduction + degree-6
// polynomial). Inputs on the speculative-decoding hot
// path are always <= 0 (x = logit - running max), so only underflow matters;
// the clamps keep the 2^n reconstruction in range. Accuracy is well within
// what the tree tolerates: output correctness is target-verified, only the
// acceptance length depends on tree precision.
inline float32x4_t vexpf(float32x4_t x) {
  x = vminq_f32(x, vdupq_n_f32(88.376259f));
  x = vmaxq_f32(x, vdupq_n_f32(-87.336548f));
  const float32x4_t LOG2EF = vdupq_n_f32(1.44269504088896341f);
  float32x4_t fx = vrndnq_f32(vmulq_f32(x, LOG2EF));  // n = round(x*log2e)
  x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));    // x -= n*C1
  x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f)); // x -= n*C2
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
  y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
  y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
  y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
  y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
  y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
  y = vfmaq_f32(x, y, z);              // y*z + x
  y = vaddq_f32(y, vdupq_n_f32(1.0f)); // + 1
  int32x4_t n =
    vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f)), 23); // 2^n
  return vmulq_f32(y, vreinterpretq_f32_s32(n));
}

// Fused single-pass scan for the uint16 speculative-decoding hot path: ONE
// read+dequant of the vocab-wide row replaces the previous top-k pass +
// full-row double-exp logsumexp pass (the dominant [raw buildTree] cost, ~8M
// scalar double exps per round).
//   - logsumexp uses ONLINE softmax (flash-style: running max `m`, rescale the
//     partial sums when it grows) so no precomputed max is needed. 4
//     independent NEON accumulators keep SIMD width and lower summation error;
//     the final reduction promotes to double.
//   - top-k rejects on the RAW uint16 (dequant (q+offset)*scale is monotone in
//     q since scale>0), so a NEON gate skips whole 16-lane chunks and only the
//     ~topk survivors are ever dequantized for selection.
// SELECTION is byte-identical to scanRowTopK (same candBetter tie-break; a
// q == qthresh tie falls through to the full comparison). Only logZ differs,
// by float rounding of the online rescale, which shifts every candidate of the
// row by the same epsilon — acceptance-validated on device, not bit-exact
// (the target verify step guarantees output correctness regardless).
inline void scanRowTopKU16Fused(const uint16_t *row, int vocab, float scale,
                                int32_t offset, int topk,
                                std::vector<float> &logProbs,
                                std::vector<int32_t> &tokens) {
  std::vector<Cand> buf;
  buf.reserve(topk);
  uint16_t qthresh = 0; // raw q of the current worst kept candidate
  int worstPos = 0;
  bool full = false;
  auto refreshWorst = [&]() {
    worstPos = 0;
    for (int j = 1; j < topk; ++j)
      if (candBetter(buf[worstPos], buf[j]))
        worstPos = j;
    qthresh = row[buf[worstPos].token];
  };
  auto consider = [&](int i, float x) { // top-k insert (vector body + tail)
    if (!full) {
      buf.push_back(Cand{x, i});
      if (static_cast<int>(buf.size()) == topk) {
        full = true;
        refreshWorst();
      }
      return;
    }
    const Cand c{x, i};
    if (!candBetter(c, buf[worstPos]))
      return;
    buf[worstPos] = c;
    refreshWorst();
  };

  float m = -std::numeric_limits<float>::infinity(); // running max
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
  const float32x4_t vscale = vdupq_n_f32(scale);
  const float32x4_t voff = vdupq_n_f32(static_cast<float>(offset));
  auto deqv = [&](uint32x4_t q) { // (q+offset)*scale — matches dequantU16
    return vmulq_f32(vaddq_f32(vcvtq_f32_u32(q), voff), vscale);
  };
  int i = 0;
  for (; i + 16 <= vocab; i += 16) {
    uint16x8_t q01 = vld1q_u16(row + i), q23 = vld1q_u16(row + i + 8);
    float32x4_t v0 = deqv(vmovl_u16(vget_low_u16(q01)));
    float32x4_t v1 = deqv(vmovl_u16(vget_high_u16(q01)));
    float32x4_t v2 = deqv(vmovl_u16(vget_low_u16(q23)));
    float32x4_t v3 = deqv(vmovl_u16(vget_high_u16(q23)));
    // online softmax: grow the running max, rescale partials only on growth
    const float cmax =
      vmaxvq_f32(vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
    if (cmax > m) {
      if (m > -std::numeric_limits<float>::infinity()) {
        const float32x4_t f = vdupq_n_f32(std::exp(m - cmax));
        a0 = vmulq_f32(a0, f);
        a1 = vmulq_f32(a1, f);
        a2 = vmulq_f32(a2, f);
        a3 = vmulq_f32(a3, f);
      }
      m = cmax;
    }
    const float32x4_t vm = vdupq_n_f32(m);
    a0 = vaddq_f32(a0, vexpf(vsubq_f32(v0, vm)));
    a1 = vaddq_f32(a1, vexpf(vsubq_f32(v1, vm)));
    a2 = vaddq_f32(a2, vexpf(vsubq_f32(v2, vm)));
    a3 = vaddq_f32(a3, vexpf(vsubq_f32(v3, vm)));
    // top-k: raw-uint16 gate — skip the chunk when nothing can beat the worst
    if (full && vmaxvq_u16(vmaxq_u16(q01, q23)) < qthresh)
      continue;
    for (int j = 0; j < 16; ++j) {
      const uint16_t q = row[i + j];
      if (full && q < qthresh)
        continue;
      consider(i + j, dequantU16(q, scale, offset));
    }
  }
  double se = static_cast<double>(
    vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))));
  for (; i < vocab; ++i) { // defensive tail (vocab % 16 == 0 here)
    const float x = dequantU16(row[i], scale, offset);
    if (x > m) {
      se *= std::exp(static_cast<double>(m) - static_cast<double>(x));
      m = x;
    }
    se += std::exp(static_cast<double>(x) - static_cast<double>(m));
    if (full && row[i] < qthresh)
      continue;
    consider(i, x);
  }
  std::sort(buf.begin(), buf.end(), candBetter); // best-first (rank order)
  const float logZ = static_cast<float>(static_cast<double>(m) + std::log(se));

  logProbs.resize(topk);
  tokens.resize(topk);
  for (int r = 0; r < topk; ++r) {
    tokens[r] = buf[r].token;
    logProbs[r] = buf[r].logit - logZ; // fp32 subtraction, as scanRowTopK
  }
}
#endif // __aarch64__

// Heap expansion + visibility from precomputed per-depth top-k — the part of
// buildTree that is independent of the logits representation. Shared by the
// float and uint16 overloads so they produce identical trees for identical
// dequantized logits (and match the reference builder's tie rule, which uses
// the same ddtree.py heap discipline).
DDTreeStructure
buildTreeFromTopK(const std::vector<std::vector<float>> &topLogProbs,
                  const std::vector<std::vector<int32_t>> &topTokenIds,
                  int depthLimit, int topk, int budget) {
  DDTreeStructure t;

  // Heap entry mirrors the Python tuple (-logw, ranks, parent, depth, rank,
  // logw). Comparison is the Python tuple order; ranks is variable-length
  // lexicographic.
  struct Entry {
    double negLogw;
    std::vector<int32_t> ranks;
    int parentIndex;
    int depth;
    int rank;
    double logw;
  };
  // pythonLess(a,b) == (a < b) under Python tuple comparison.
  auto pythonLess = [](const Entry &a, const Entry &b) {
    if (a.negLogw != b.negLogw)
      return a.negLogw < b.negLogw;
    if (a.ranks != b.ranks)
      return a.ranks <
             b.ranks; // std::vector lexicographic (prefix sorts first)
    if (a.parentIndex != b.parentIndex)
      return a.parentIndex < b.parentIndex;
    if (a.depth != b.depth)
      return a.depth < b.depth;
    if (a.rank != b.rank)
      return a.rank < b.rank;
    return a.logw < b.logw;
  };
  // priority_queue top == Python-smallest -> comp(x,y) = pythonLess(y,x).
  auto comp = [&pythonLess](const Entry &x, const Entry &y) {
    return pythonLess(y, x);
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(comp)> heap(comp);

  const double firstLogw = static_cast<double>(topLogProbs[0][0]);
  heap.push(
    Entry{-firstLogw, {0}, /*parent*/ 0, /*depth*/ 1, /*rank*/ 0, firstLogw});

  t.parents.assign(budget + 1, 0);
  t.parents[0] = -1;
  t.childMaps.clear();
  t.childMaps.emplace_back(); // root placeholder child map
  t.nodeTokenIds.clear();
  t.nodeDepths.clear();

  int nodeCount = 0;
  while (!heap.empty() && nodeCount < budget) {
    Entry e = heap.top();
    heap.pop();

    const int32_t tokenId = topTokenIds[e.depth - 1][e.rank];
    const int currentIndex = nodeCount + 1;
    t.nodeTokenIds.push_back(tokenId);
    t.nodeDepths.push_back(e.depth);
    t.parents[currentIndex] = e.parentIndex;
    t.childMaps.emplace_back();
    t.childMaps[e.parentIndex][tokenId] = currentIndex;
    ++nodeCount;

    if (e.rank + 1 < topk) {
      std::vector<int32_t> siblingRanks = e.ranks;
      siblingRanks.back() = e.rank + 1; // ranks[:-1] + (rank+1,)
      const double siblingLogw =
        e.logw - static_cast<double>(topLogProbs[e.depth - 1][e.rank]) +
        static_cast<double>(topLogProbs[e.depth - 1][e.rank + 1]);
      heap.push(Entry{-siblingLogw, std::move(siblingRanks), e.parentIndex,
                      e.depth, e.rank + 1, siblingLogw});
    }

    if (e.depth < depthLimit) {
      std::vector<int32_t> childRanks = e.ranks;
      childRanks.push_back(0); // ranks + (0,)
      const double childLogw =
        e.logw + static_cast<double>(topLogProbs[e.depth][0]);
      heap.push(Entry{-childLogw, std::move(childRanks), currentIndex,
                      e.depth + 1, 0, childLogw});
    }
  }

  // Visibility bitmap (ddtree.py 160-167).
  const int currentLength = 1 + nodeCount;
  t.nodeCount = nodeCount;
  t.currentLength = currentLength;
  t.parents.resize(currentLength);
  t.visibility.assign(static_cast<size_t>(currentLength) * currentLength, 0);
  t.visibility[0] = 1; // vis[0,0]
  for (int index = 1; index < currentLength; ++index) {
    const int parent = t.parents[index];
    for (int j = 0; j < index; ++j)
      t.visibility[static_cast<size_t>(index) * currentLength + j] =
        t.visibility[static_cast<size_t>(parent) * currentLength + j];
    t.visibility[static_cast<size_t>(index) * currentLength + index] = 1;
  }
  return t;
}

} // namespace

DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg) {
  DDTreeStructure t;

  // Empty case (ddtree.py 98-108): budget<=0 or depth_limit==0 -> root only.
  if (cfg.budget <= 0 || depthLimit == 0) {
    t.nodeCount = 0;
    t.currentLength = 1;
    t.parents = {-1};
    t.childMaps.resize(1);
    t.visibility = {1};
    return t;
  }

  const int topk = std::min(cfg.budget, vocab);

  // Per-row top-k: topLogProbs[d][r] = top_logit - logsumexp(row d), sorted by
  // (logit desc, token index asc) to match torch.topk on CPU (ddtree.py
  // 114-117). Each depth row is independent -> parallel via ThreadManager.
  // Selection + logZ math live in scanRowTopK (shared with the uint16
  // overload for byte-identical trees).
  std::vector<std::vector<float>> topLogProbs(depthLimit);
  std::vector<std::vector<int32_t>> topTokenIds(depthLimit);

  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, static_cast<size_t>(depthLimit), [&](size_t d) {
    const float *row = draftLogits + d * static_cast<size_t>(vocab);
    scanRowTopK([row](int i) { return row[i]; }, vocab, topk, topLogProbs[d],
                topTokenIds[d]);
  });

  return buildTreeFromTopK(topLogProbs, topTokenIds, depthLimit, topk,
                           cfg.budget);
}

// uint16 overload: identical tree to the float overload for the same
// dequantized logits. Dequantizes (q + offset) * scale on the fly inside the
// shared scanRowTopK (fp32 values, so selection ties and logZ accumulation are
// byte-identical to running the float overload on a pre-dequantized buffer) —
// no horizon*vocab fp32 staging buffer.
DDTreeStructure buildTree(const uint16_t *draftLogits, float scale,
                          int32_t offset, int depthLimit, int vocab,
                          int budget) {
  // Empty case (ddtree.py 98-108): budget<=0 or depth_limit==0 -> root only.
  if (budget <= 0 || depthLimit == 0) {
    DDTreeStructure t;
    t.nodeCount = 0;
    t.currentLength = 1;
    t.parents = {-1};
    t.childMaps.resize(1);
    t.visibility = {1};
    return t;
  }

  const int topk = std::min(budget, vocab);

  std::vector<std::vector<float>> topLogProbs(depthLimit);
  std::vector<std::vector<int32_t>> topTokenIds(depthLimit);

  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, static_cast<size_t>(depthLimit), [&](size_t d) {
    const uint16_t *row = draftLogits + d * static_cast<size_t>(vocab);
#if defined(__aarch64__)
    // Speculative-decoding hot path: fused single-pass NEON scan (selection
    // byte-identical to scanRowTopK; logZ differs only by online-softmax float
    // rounding).
    scanRowTopKU16Fused(row, vocab, scale, offset, topk, topLogProbs[d],
                        topTokenIds[d]);
#else
    scanRowTopK([row, scale, offset](int i) { return dequantU16(row[i], scale,
                                                                offset); },
                vocab, topk, topLogProbs[d], topTokenIds[d]);
#endif
  });

  return buildTreeFromTopK(topLogProbs, topTokenIds, depthLimit, topk, budget);
}

CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride) {
  const int cur = tree.currentLength;

  // verify_input_ids: [root, node tokens...] (ddtree.py 197-200).
  verifyInputIds[0] = rootTokenId;
  for (int i = 1; i < cur; ++i)
    verifyInputIds[i] = tree.nodeTokenIds[i - 1];

  // verify_position_ids: [start, start+depth...] (ddtree.py 202-206).
  verifyPositionIds[0] = start;
  for (int i = 1; i < cur; ++i)
    verifyPositionIds[i] = start + tree.nodeDepths[i - 1];

  // Mask (ddtree.py 195, 211-213): prefix block 0 (visible); tree block
  // filled with maskFillValue then opened (0) where visibility == 1.
  for (int i = 0; i < cur; ++i) {
    float *r = attentionMask + static_cast<size_t>(i) * attnMaskRowStride;
    for (int j = 0; j < pastLength; ++j)
      r[j] = 0.0f;
    for (int j = 0; j < cur; ++j)
      r[pastLength + j] = tree.visibility[static_cast<size_t>(i) * cur + j]
                            ? 0.0f
                            : cfg.maskFillValue;
  }

  return CompiledTree{pastLength, cur};
}

Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior) {
  // ddtree.py 282-293.
  Accepted a;
  a.indices.push_back(0);
  int currentIndex = 0;
  int32_t nextToken = posterior[currentIndex];
  while (true) {
    auto it = childMaps[currentIndex].find(nextToken);
    if (it == childMaps[currentIndex].end())
      break;
    currentIndex = it->second;
    a.indices.push_back(currentIndex);
    nextToken = posterior[currentIndex];
  }
  a.nextToken = nextToken;
  return a;
}

} // namespace ddtree
} // namespace nntrainer
