// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_embedding_pool_ops.cpp
 * @date   28 July 2026
 * @brief  CPU-vs-OpenCL differential for the embedding pooling / normalize
 *         whole-ops (ComputeOps::mean_rows, ComputeOps::l2_normalize_rows).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * These are the two ops behind the sentence-embedding tail. The backend is
 * selected by the NNTR_OPS_BACKEND env var ("cpu" | "gpu") so the two engines
 * are compared across SEPARATE PROCESSES, never two engines in one process.
 * With NNTR_OPS_DUMP=<path> the computed vectors are written out so an
 * external differ can compute the cpu-vs-gpu error metrics.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <compute_ops.h>
#include <context_data.h>
#include <tensor.h>

namespace {

/** deterministic, sign-varied, non-trivial magnitudes */
float genValue(unsigned int i) {
  return std::sin(0.37f * (float)i) * (1.0f + 0.01f * (float)(i % 17)) +
         0.05f * (float)((int)(i % 5) - 2);
}

std::string backendName() {
  const char *e = std::getenv("NNTR_OPS_BACKEND");
  return e ? std::string(e) : std::string("cpu");
}

/**
 * @brief Ops table for the selected backend, plus a ContextData carrying it so
 *        the tensors under test dispatch exactly like a real layer's tensors.
 *        Returns nullptr when the backend is unavailable in this build.
 */
nntrainer::ComputeOps *selectedOps() {
  const std::string b = backendName();
  if (b == "gpu") {
#ifdef ENABLE_OPENCL
    return nntrainer::get_cl_ops();
#else
    return nullptr;
#endif
  }
  return nntrainer::get_cpu_ops();
}

/** attach the selected backend's ops to a tensor (mirrors LayerNode) */
void attachOps(nntrainer::Tensor &t, nntrainer::ComputeOps *ops) {
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(ops);
  t.setContextData(ct);
}

/**
 * @brief Fill a tensor with a sentinel so a silently-skipped GPU dispatch
 *        cannot pass: the CL op helpers bail out of their dispatch block on
 *        any OpenCL failure and leave the destination untouched, which would
 *        otherwise read as uninitialised-but-plausible memory.
 */
void poison(nntrainer::Tensor &t) {
  for (unsigned int i = 0; i < t.size(); ++i)
    t.getData()[i] = -123456.0f;
}

void dumpIfRequested(const std::string &tag, const std::vector<float> &v) {
  const char *path = std::getenv("NNTR_OPS_DUMP");
  if (!path)
    return;
  std::ofstream f(std::string(path) + "." + tag + "." + backendName(),
                  std::ios::trunc);
  f.precision(9);
  for (float x : v)
    f << x << "\n";
}

} // namespace

TEST(EmbeddingPoolOps, L2NormalizeRowsMatchesContract) {
  auto *ops = selectedOps();
  if (!ops)
    GTEST_SKIP() << "backend " << backendName() << " unavailable in this build";

  // [B=2, 1, H=3, W=128]: several independent rows, width a typical hidden size
  const unsigned int B = 2, H = 3, W = 128;
  nntrainer::Tensor in({B, 1, H, W}, true);
  nntrainer::Tensor out({B, 1, H, W}, true);
  for (unsigned int i = 0; i < in.size(); ++i)
    in.getData()[i] = genValue(i);

  attachOps(in, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(in.getOps()->l2_normalize_rows(in, out, 1e-12f));

  std::vector<float> got(out.getData(), out.getData() + out.size());
  dumpIfRequested("l2", got);

  // Independent closed-form check: every row must have unit L2 norm and stay
  // parallel to the input row. This is an absolute contract, so it holds for
  // whichever backend this process ran.
  for (unsigned int r = 0; r < B * H; ++r) {
    const float *x = in.getData() + (size_t)r * W;
    const float *y = got.data() + (size_t)r * W;
    double ss_x = 0.0, ss_y = 0.0;
    for (unsigned int i = 0; i < W; ++i) {
      ss_x += (double)x[i] * x[i];
      ss_y += (double)y[i] * y[i];
    }
    EXPECT_NEAR(std::sqrt(ss_y), 1.0, 1e-5) << "row " << r << " not unit-norm";

    const double scale = 1.0 / std::sqrt(ss_x);
    for (unsigned int i = 0; i < W; ++i)
      ASSERT_NEAR(y[i], (float)(x[i] * scale), 1e-5)
        << "row " << r << " element " << i;
  }
}

TEST(EmbeddingPoolOps, L2NormalizeRowsEpsilonFloorsTheNorm) {
  auto *ops = selectedOps();
  if (!ops)
    GTEST_SKIP() << "backend " << backendName() << " unavailable in this build";

  // A tiny row: ||x|| < epsilon, so the divisor must clamp to epsilon (a FLOOR
  // on the norm) rather than blowing the row up to unit length. This is the
  // semantic that separates this op from RMSNorm (epsilon under the sqrt).
  const unsigned int W = 64;
  const float eps = 1e-3f;
  nntrainer::Tensor in({1, 1, 1, W}, true);
  nntrainer::Tensor out({1, 1, 1, W}, true);
  for (unsigned int i = 0; i < W; ++i)
    in.getData()[i] = 1e-8f;

  attachOps(in, ops);
  attachOps(out, ops);
  poison(out);
  ASSERT_NO_THROW(in.getOps()->l2_normalize_rows(in, out, eps));

  std::vector<float> got(out.getData(), out.getData() + out.size());
  dumpIfRequested("l2eps", got);

  for (unsigned int i = 0; i < W; ++i)
    EXPECT_NEAR(got[i], 1e-8f / eps, 1e-9f) << "element " << i;
}

TEST(EmbeddingPoolOps, MeanRowsMatchesContract) {
  auto *ops = selectedOps();
  if (!ops)
    GTEST_SKIP() << "backend " << backendName() << " unavailable in this build";

  const unsigned int H = 7, W = 128;
  nntrainer::Tensor in({1, 1, H, W}, true);
  nntrainer::Tensor out({1, 1, 1, W}, true);
  for (unsigned int i = 0; i < in.size(); ++i)
    in.getData()[i] = genValue(i);

  attachOps(in, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(in.getOps()->mean_rows(in, out, H, 0));

  std::vector<float> got(out.getData(), out.getData() + out.size());
  dumpIfRequested("mean", got);

  for (unsigned int c = 0; c < W; ++c) {
    double acc = 0.0;
    for (unsigned int r = 0; r < H; ++r)
      acc += in.getData()[(size_t)r * W + c];
    EXPECT_NEAR(got[c], (float)(acc / H), 1e-5) << "column " << c;
  }
}

TEST(EmbeddingPoolOps, MeanRowsHonorsRowOffset) {
  auto *ops = selectedOps();
  if (!ops)
    GTEST_SKIP() << "backend " << backendName() << " unavailable in this build";

  const unsigned int H = 8, W = 96, OFF = 3, N = 4;
  nntrainer::Tensor in({1, 1, H, W}, true);
  nntrainer::Tensor out({1, 1, 1, W}, true);
  for (unsigned int i = 0; i < in.size(); ++i)
    in.getData()[i] = genValue(i);

  attachOps(in, ops);
  attachOps(out, ops);

  poison(out);
  ASSERT_NO_THROW(in.getOps()->mean_rows(in, out, N, OFF));

  std::vector<float> got(out.getData(), out.getData() + out.size());
  dumpIfRequested("meanoff", got);

  for (unsigned int c = 0; c < W; ++c) {
    double acc = 0.0;
    for (unsigned int r = OFF; r < OFF + N; ++r)
      acc += in.getData()[(size_t)r * W + c];
    EXPECT_NEAR(got[c], (float)(acc / N), 1e-5) << "column " << c;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  std::printf("[ops-backend] %s\n", backendName().c_str());
  return RUN_ALL_TESTS();
}
