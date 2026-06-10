// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Unit tests for the DDTree speculative-decoding core
 * @file   unittest_ddtree.cpp
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <build_golden.h>
#include <ddtree.h>
#include <ddtree_compact.h>
#include <ddtree_sampling.h>
#include <ddtree_sliding.h>
#include <ddtree_types.h>

TEST(DDTreeScaffold, ConfigDefaults) {
  nntrainer::ddtree::DDTreeConfig cfg;
  EXPECT_EQ(cfg.budget, 31);
  EXPECT_EQ(cfg.depthLimit, 0);

  nntrainer::ddtree::DDTreeStructure root;
  EXPECT_EQ(root.currentLength, 1);
  EXPECT_EQ(root.nodeCount, 0);
}

using nntrainer::ddtree::buildTree;
using nntrainer::ddtree::DDTreeConfig;
using nntrainer::ddtree::DDTreeStructure;

TEST(DDTreeBuild, EmptyBudgetReturnsRootOnly) {
  std::vector<float> logits(8, 0.0f); // depthLimit=2, vocab=4, unused here
  DDTreeConfig cfg;
  cfg.budget = 0;
  DDTreeStructure t = buildTree(logits.data(), 2, 4, cfg);
  EXPECT_EQ(t.nodeCount, 0);
  EXPECT_EQ(t.currentLength, 1);
  ASSERT_EQ(t.parents.size(), 1u);
  EXPECT_EQ(t.parents[0], -1);
  ASSERT_EQ(t.childMaps.size(), 1u);
  EXPECT_TRUE(t.childMaps[0].empty());
  ASSERT_EQ(t.visibility.size(), 1u);
  EXPECT_EQ(t.visibility[0], 1u);
}

TEST(DDTreeBuild, ZeroDepthReturnsRootOnly) {
  std::vector<float> logits(1, 0.0f);
  DDTreeConfig cfg;
  cfg.budget = 8;
  DDTreeStructure t = buildTree(logits.data(), 0, 4, cfg);
  EXPECT_EQ(t.currentLength, 1);
  EXPECT_EQ(t.visibility[0], 1u);
}

// 2 depths, vocab 3. Row-major [depth, vocab].
// Depth 0 logits: token0 highest, token1 next, token2 lowest.
// Depth 1 logits: token2 highest, token0 next, token1 lowest.
static std::vector<float> kSmallLogits() {
  return {
    2.0f, 1.0f,  0.0f, // depth 0
    0.0f, -1.0f, 1.0f, // depth 1
  };
}

TEST(DDTreeBuild, SmallTreeStructure) {
  DDTreeConfig cfg;
  cfg.budget = 3;
  auto logits = kSmallLogits();
  DDTreeStructure t = buildTree(logits.data(), 2, 3, cfg);

  // budget 3 -> 3 nodes, currentLength 4.
  EXPECT_EQ(t.nodeCount, 3);
  EXPECT_EQ(t.currentLength, 4);

  // First popped node: depth 1, rank 0 -> token_ids[0][0] = token 0.
  EXPECT_EQ(t.nodeDepths[0], 1);
  EXPECT_EQ(t.nodeTokenIds[0], 0);
  EXPECT_EQ(t.parents[1], 0); // child of root placeholder (index 0)

  // parents/visibility well-formed: root visible to all; each node sees itself.
  ASSERT_EQ(t.parents.size(), 4u);
  EXPECT_EQ(t.parents[0], -1);
  ASSERT_EQ(t.visibility.size(), 16u);
  int L = t.currentLength;
  EXPECT_EQ(t.visibility[0 * L + 0], 1u);
  for (int i = 1; i < L; ++i) {
    EXPECT_EQ(t.visibility[i * L + 0], 1u); // root always visible
    EXPECT_EQ(t.visibility[i * L + i], 1u); // self visible
  }
  // A node never sees a strictly-later node (upper triangle = 0).
  for (int i = 0; i < L; ++i)
    for (int j = i + 1; j < L; ++j)
      EXPECT_EQ(t.visibility[i * L + j], 0u);
}

TEST(DDTreeBuild, BudgetExceedsVocabTopkClamped) {
  DDTreeConfig cfg;
  cfg.budget = 100; // > vocab
  auto logits = kSmallLogits();
  // Must not read past topk = min(budget, vocab) = 3 columns; no crash.
  DDTreeStructure t = buildTree(logits.data(), 2, 3, cfg);
  EXPECT_GE(t.nodeCount, 1);
  EXPECT_LE(t.nodeCount, 100);
}

TEST(DDTreeCompile, IdsPositionsAndMask) {
  using nntrainer::ddtree::compile;
  using nntrainer::ddtree::CompiledTree;

  DDTreeConfig cfg;
  cfg.budget = 3;
  cfg.maskFillValue = -1e30f;
  auto logits = kSmallLogits();
  DDTreeStructure t = buildTree(logits.data(), 2, 3, cfg);

  const int past = 2;
  const int start = 5;
  const int32_t rootToken = 99;
  const int L = t.currentLength;
  std::vector<int32_t> ids(L), pos(L);
  const int stride = past + L;
  std::vector<float> mask(static_cast<size_t>(L) * stride, 7.0f); // sentinel

  CompiledTree c = compile(rootToken, start, past, t, cfg, ids.data(),
                           pos.data(), mask.data(), stride);
  EXPECT_EQ(c.pastLength, past);
  EXPECT_EQ(c.currentLength, L);

  // ids[0] = root; ids[i] = node token.
  EXPECT_EQ(ids[0], rootToken);
  for (int i = 1; i < L; ++i)
    EXPECT_EQ(ids[i], t.nodeTokenIds[i - 1]);

  // pos[0] = start; pos[i] = start + depth.
  EXPECT_EQ(pos[0], start);
  for (int i = 1; i < L; ++i)
    EXPECT_EQ(pos[i], start + t.nodeDepths[i - 1]);

  // Prefix block [0, past) fully visible (== 0) for every row.
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < past; ++j)
      EXPECT_EQ(mask[i * stride + j], 0.0f);

  // Tree block [past, past+L) == visibility ? 0 : fill.
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      float expected = t.visibility[i * L + j] ? 0.0f : cfg.maskFillValue;
      EXPECT_EQ(mask[i * stride + past + j], expected);
    }
}

TEST(DDTreeSliding, NoSlidingLayersPassthrough) {
  using nntrainer::ddtree::makeSlidingMasks;
  using nntrainer::ddtree::SlidingMasks;
  std::vector<float> full(4, 0.0f);
  std::vector<int32_t> pos = {0, 1};
  std::vector<float> slide(4, -1.0f);
  DDTreeConfig cfg;
  SlidingMasks m = makeSlidingMasks(full.data(), pos.data(), 2, 2, 8, false,
                                    cfg, slide.data());
  EXPECT_FALSE(m.hasSliding);
  EXPECT_EQ(m.full, full.data());
}

TEST(DDTreeSliding, WindowZeroFullEqualsSliding) {
  using nntrainer::ddtree::makeSlidingMasks;
  using nntrainer::ddtree::SlidingMasks;
  std::vector<float> full(4, 0.0f);
  std::vector<int32_t> pos = {0, 1};
  std::vector<float> slide(4, -1.0f);
  DDTreeConfig cfg;
  SlidingMasks m =
    makeSlidingMasks(full.data(), pos.data(), 2, 2, 0, true, cfg, slide.data());
  EXPECT_TRUE(m.hasSliding);
  EXPECT_EQ(m.sliding, m.full);
}

TEST(DDTreeSliding, PositiveWindowVisibility) {
  using nntrainer::ddtree::makeSlidingMasks;
  using nntrainer::ddtree::SlidingMasks;
  // past=1, cur=2, kv=3. query=[5,6]; key[:past]=arange=[0], key[past:]=[5,6].
  const int past = 1, cur = 2, kv = 3, window = 2;
  std::vector<int32_t> pos = {5, 6};
  std::vector<float> full(static_cast<size_t>(cur) * kv, 0.0f); // all visible
  std::vector<float> slide(static_cast<size_t>(cur) * kv, 123.0f);
  DDTreeConfig cfg;
  cfg.maskFillValue = -1e30f;
  SlidingMasks m = makeSlidingMasks(full.data(), pos.data(), cur, kv, window,
                                    true, cfg, slide.data());
  ASSERT_TRUE(m.hasSliding);
  ASSERT_NE(m.sliding, m.full);

  auto vis = [&](int qi, int kj) {
    int q = pos[qi];
    int k = (kj < past) ? kj : pos[kj - past];
    return (k <= q) && (k > q - window);
  };
  for (int i = 0; i < cur; ++i)
    for (int j = 0; j < kv; ++j) {
      float expected = vis(i, j) ? 0.0f : cfg.maskFillValue;
      EXPECT_EQ(m.sliding[i * kv + j], expected) << "i=" << i << " j=" << j;
    }
}

TEST(DDTreeSliding, VisibilityMatchesThresholdedMask) {
  using nntrainer::ddtree::makeSlidingMasks;
  using nntrainer::ddtree::makeSlidingVisibility;
  using nntrainer::ddtree::SlidingMasks;
  // Linear chain: node0(root), node1 child of 0, node2 child of 1.
  // tree visibility (cur=3) is lower-triangular incl diagonal.
  const int past = 2, cur = 3, kv = past + cur, window = 3;
  std::vector<uint8_t> treeVis = {1, 0, 0, 1, 1, 0, 1, 1, 1};
  std::vector<int32_t> pos = {10, 11, 12}; // tree query/key positions

  // The 0/1 helper must equal makeSlidingMasks()'s fp32 sliding output
  // thresholded, for ANY non-zero maskFillValue (it must be fill-independent).
  for (float fill : {-1e30f, -7.5f}) {
    DDTreeConfig cfg;
    cfg.maskFillValue = fill;
    std::vector<float> full(static_cast<size_t>(cur) * kv, 0.0f);
    for (int i = 0; i < cur; ++i)
      for (int j = 0; j < cur; ++j)
        full[i * kv + past + j] = treeVis[i * cur + j] ? 0.0f : fill;
    std::vector<float> slide(static_cast<size_t>(cur) * kv, 0.0f);
    SlidingMasks m = makeSlidingMasks(full.data(), pos.data(), cur, kv, window,
                                      true, cfg, slide.data());
    ASSERT_TRUE(m.hasSliding);

    std::vector<uint8_t> vis01(static_cast<size_t>(cur) * kv, 9);
    makeSlidingVisibility(treeVis.data(), pos.data(), cur, kv, window,
                          vis01.data());
    for (int idx = 0; idx < cur * kv; ++idx) {
      uint8_t expect = (m.sliding[idx] == 0.0f) ? 1 : 0;
      EXPECT_EQ(vis01[idx], expect) << "fill=" << fill << " idx=" << idx;
    }
  }
}

TEST(DDTreeSliding, VisibilityWindowUnrestrictedEqualsTree) {
  using nntrainer::ddtree::makeSlidingVisibility;
  // window <= 0 -> no window restriction -> out == tree visibility (prefix
  // columns always visible, tree columns follow treeVis).
  const int past = 1, cur = 2, kv = past + cur;
  std::vector<uint8_t> treeVis = {1, 0, 1, 1};
  std::vector<int32_t> pos = {3, 4};
  std::vector<uint8_t> out(static_cast<size_t>(cur) * kv, 9);
  makeSlidingVisibility(treeVis.data(), pos.data(), cur, kv, 0, out.data());
  EXPECT_EQ(out, (std::vector<uint8_t>{1, 1, 0, 1, 1, 1}));
}

// root(0) -> {10:1}; node1 -> {20:2}; node2 -> {}.
static std::vector<std::unordered_map<int32_t, int32_t>> kChain() {
  std::vector<std::unordered_map<int32_t, int32_t>> cm(3);
  cm[0][10] = 1;
  cm[1][20] = 2;
  return cm;
}

TEST(DDTreeFollow, FullAccept) {
  using nntrainer::ddtree::Accepted;
  using nntrainer::ddtree::followVerified;
  auto cm = kChain();
  std::vector<int32_t> posterior = {10, 20, 77}; // root->1->2, then bonus 77
  Accepted a = followVerified(cm, posterior.data());
  EXPECT_EQ(a.indices, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(a.nextToken, 77);
}

TEST(DDTreeFollow, PartialAccept) {
  using nntrainer::ddtree::Accepted;
  using nntrainer::ddtree::followVerified;
  auto cm = kChain();
  std::vector<int32_t> posterior = {10, 999,
                                    77}; // root->1, then 999 not a child
  Accepted a = followVerified(cm, posterior.data());
  EXPECT_EQ(a.indices, (std::vector<int32_t>{0, 1}));
  EXPECT_EQ(a.nextToken, 999);
}

TEST(DDTreeFollow, ImmediateReject) {
  using nntrainer::ddtree::Accepted;
  using nntrainer::ddtree::followVerified;
  auto cm = kChain();
  std::vector<int32_t> posterior = {5, 20,
                                    77}; // root token 5 not a child of root
  Accepted a = followVerified(cm, posterior.data());
  EXPECT_EQ(a.indices, (std::vector<int32_t>{0}));
  EXPECT_EQ(a.nextToken, 5);
}

TEST(DDTreeCompact, SubsetReorderRowMajor) {
  using nntrainer::ddtree::compactTail;
  // 1 prefix row then a tail of 4 rows, each row = 2 elems. seqStride=2.
  const int rowElems = 2, seqStride = 2, past = 1, tailLen = 4;
  std::vector<float> buf = {
    -1, -1, // row 0 (prefix, untouched)
    10, 11, // tail idx 0
    20, 21, // tail idx 1
    30, 31, // tail idx 2
    40, 41, // tail idx 3
  };
  std::vector<int32_t> keep = {0, 2, 3}; // keep tail rows 0,2,3
  compactTail(buf.data(), sizeof(float), seqStride, rowElems, past, tailLen,
              keep.data(), (int)keep.size());
  EXPECT_EQ(buf[2], 10);
  EXPECT_EQ(buf[3], 11); // dst0 = src0
  EXPECT_EQ(buf[4], 30);
  EXPECT_EQ(buf[5], 31); // dst1 = src2
  EXPECT_EQ(buf[6], 40);
  EXPECT_EQ(buf[7], 41); // dst2 = src3
  EXPECT_EQ(buf[0], -1); // prefix untouched
}

TEST(DDTreeCompact, IdentityNoOp) {
  using nntrainer::ddtree::compactTail;
  std::vector<float> buf = {0, 1, 2, 3};
  std::vector<int32_t> keep = {0, 1};
  compactTail(buf.data(), sizeof(float), 2, 2, 0, 2, keep.data(), 2);
  EXPECT_EQ(buf, (std::vector<float>{0, 1, 2, 3}));
}

TEST(DDTreeCompact, EmptyKeepNoOp) {
  using nntrainer::ddtree::compactTail;
  std::vector<float> buf = {9, 9};
  std::vector<int32_t> keep;
  compactTail(buf.data(), sizeof(float), 2, 2, 0, 1, keep.data(), 0);
  EXPECT_EQ(buf, (std::vector<float>{9, 9}));
}

TEST(DDTreeCompact, WorksWithSeqStrideGreaterThanRow) {
  using nntrainer::ddtree::compactTail;
  // seqStride 3 but rowElems 2 (padded rows). 3 tail rows.
  const int rowElems = 2, seqStride = 3, past = 0, tailLen = 3;
  std::vector<float> buf = {
    10, 11, 99, // tail 0 (col 2 = pad)
    20, 21, 99, // tail 1
    30, 31, 99, // tail 2
  };
  std::vector<int32_t> keep = {2, 0};
  compactTail(buf.data(), sizeof(float), seqStride, rowElems, past, tailLen,
              keep.data(), 2);
  EXPECT_EQ(buf[0], 30);
  EXPECT_EQ(buf[1], 31); // dst0 = src2
  EXPECT_EQ(buf[3], 10);
  EXPECT_EQ(buf[4], 11); // dst1 = src0
}

TEST(DDTreeSampling, ArgmaxAndGreedyRows) {
  using nntrainer::ddtree::argmaxRow;
  using nntrainer::ddtree::sampleGreedy;
  std::vector<float> logits = {
    0.1f, 0.9f, 0.2f, // row 0 -> token 1
    5.0f, 1.0f, 2.0f, // row 1 -> token 0
  };
  EXPECT_EQ(argmaxRow(logits.data(), 3), 1);
  std::vector<int32_t> out(2);
  sampleGreedy(logits.data(), 2, 3, out.data());
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 0);
}

TEST(DDTreeSampling, ArgmaxTieLowestIndex) {
  using nntrainer::ddtree::argmaxRow;
  std::vector<float> logits = {2.0f, 2.0f, 1.0f}; // tie -> lowest index 0
  EXPECT_EQ(argmaxRow(logits.data(), 3), 0);
}

// Parses the embedded golden vectors (build_golden.h, generated by
// gen_golden.py) and asserts buildTree reproduces each case. The vectors are
// embedded rather than read from a file so the test stays self-contained when a
// cross-compiled binary runs on a separate test runner. See gen_golden.py for
// the format.
TEST(DDTreeBuild, MatchesPythonGolden) {
  std::istringstream in(kDDTreeBuildGolden);

  std::string tok;
  int ncases = 0;
  in >> tok >> ncases;
  ASSERT_EQ(tok, "NCASES");
  ASSERT_GT(ncases, 0);

  for (int ci = 0; ci < ncases; ++ci) {
    std::string name;
    int depth = 0, vocab = 0, budget = 0;
    in >> tok >> name;
    ASSERT_EQ(tok, "CASE") << "case " << ci;
    in >> tok >> depth >> vocab >> budget;
    ASSERT_EQ(tok, "DIMS") << name;

    std::vector<float> logits(static_cast<size_t>(depth) * vocab);
    in >> tok;
    ASSERT_EQ(tok, "LOGITS") << name;
    for (auto &v : logits)
      in >> v;

    auto readVec = [&](const char *label) {
      std::string t;
      int n = 0;
      in >> t >> n;
      EXPECT_EQ(t, label) << name;
      std::vector<int32_t> v(n);
      for (auto &x : v)
        in >> x;
      return v;
    };
    std::vector<int32_t> expNodes = readVec("NODES");
    std::vector<int32_t> expDepths = readVec("DEPTHS");
    std::vector<int32_t> expParents = readVec("PARENTS");

    int L = 0;
    in >> tok >> L;
    ASSERT_EQ(tok, "VIS") << name;
    std::vector<int32_t> expVis(static_cast<size_t>(L) * L);
    for (auto &x : expVis)
      in >> x;

    DDTreeConfig cfg;
    cfg.budget = budget;
    DDTreeStructure t = buildTree(logits.data(), depth, vocab, cfg);

    EXPECT_EQ(t.nodeTokenIds, expNodes) << name;
    EXPECT_EQ(t.nodeDepths, expDepths) << name;
    EXPECT_EQ(t.parents, expParents) << name;
    EXPECT_EQ(t.currentLength, L) << name;
    ASSERT_EQ(t.visibility.size(), expVis.size()) << name;
    for (size_t k = 0; k < t.visibility.size(); ++k)
      EXPECT_EQ(static_cast<int32_t>(t.visibility[k]), expVis[k])
        << name << " vis@" << k;
  }
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to init gtest\n";
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run test.\n";
  }
  return result;
}
