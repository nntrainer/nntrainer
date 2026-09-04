// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	unittest_attention_kernels_cl.cpp
 * @date	28 August 2024
 * @brief	Test setup for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <vector>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <attention_kernel_interface.h>
#include <attention_kernels.h>
#include <cl_context.h>
#include <engine.h>
#include <layer_context.h>
#include <tensor.h>

#include "testing_rotary_emb.cpp"

#define EXPECT_IN_RANGE(VAL, MIN, MAX)                                         \
  EXPECT_GE((VAL), (MIN));                                                     \
  EXPECT_LE((VAL), (MAX))

using namespace nntrainer;

TEST(attention_kernels, rotary_emb_kernel_FP32) {
  int batch = 1;
  int channel = 1;
  int height = 4;
  int width = 4;

  unsigned int dim = 2;
  unsigned int from = 4;
  unsigned int max_timestep = 4;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  B_fp32.copy(A_fp32);

  apply_rotary_emb_cl(A_fp32, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp32, dim, from, max_timestep);

  float mseErrorNeon_fp32 =
    mse<float>(A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  double cosSimNeon_fp32 = cosine_similarity<float>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp32, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp32, 0.99, 1);
}

TEST(attention_kernels, rotary_emb_kernel_FP32_case2) {
  int batch = 4;
  int channel = 4;
  int height = 8;
  int width = 8;

  unsigned int dim = 2;
  unsigned int from = 2;
  unsigned int max_timestep = 4;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height, width, t_type_nchw_fp32);

  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  B_fp32.copy(A_fp32);

  apply_rotary_emb_cl(A_fp32, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp32, dim, from, max_timestep);

  float mseErrorNeon_fp32 =
    mse<float>(A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  double cosSimNeon_fp32 = cosine_similarity<float>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), A_fp32.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp32, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp32, 0.99, 1);
}

#ifdef ENABLE_FP16

TEST(attention_kernels, rotary_emb_kernel_FP16) {
  int batch = 1;
  int channel = 1;
  int height = 4;
  int width = 4;

  unsigned int dim = 2;
  unsigned int from = 4;
  unsigned int max_timestep = 4;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, i * (batch * height * channel) * alpha +
                           j * (batch * height) * alpha + k * (width)*alpha +
                           l + 1);

  B_fp16.copy(A_fp16);

  apply_rotary_emb_cl(A_fp16, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp16, dim, from, max_timestep);

  float mseErrorNeon_fp16 =
    mse<_FP16>(A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>(), A_fp16.size());

  double cosSimNeon_fp16 = cosine_similarity<_FP16>(
    A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp16, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp16, 0.99, 1);
}

TEST(attention_kernels, rotary_emb_kernel_FP16_case2) {
  int batch = 4;
  int channel = 4;
  int height = 8;
  int width = 8;

  unsigned int dim = 4;
  unsigned int from = 4;
  unsigned int max_timestep = 8;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_fp16(batch, channel, height, width, t_type_nchw_fp16);

  GEN_TEST_INPUT(A_fp16, i * (batch * height * channel) * alpha +
                           j * (batch * height) * alpha + k * (width)*alpha +
                           l + 1);

  B_fp16.copy(A_fp16);

  apply_rotary_emb_cl(A_fp16, dim, from, max_timestep);
  apply_rotary_emb_tensor(B_fp16, dim, from, max_timestep);

  float mseErrorNeon_fp16 =
    mse<_FP16>(A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>(), A_fp16.size());

  double cosSimNeon_fp16 = cosine_similarity<_FP16>(
    A_fp16.getData<_FP16>(), B_fp16.getData<_FP16>(), A_fp16.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon_fp16, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon_fp16, 0.99, 1);
}

#endif

#ifdef ENABLE_FP16

/**
 * @brief Convert one IEEE binary16 bit pattern to float, without relying on a
 *        compiler half type: this test has to read the kernel's raw fp16
 *        output on hosts where _Float16 is unavailable.
 */
static float halfBitsToFloat(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize it by hand.
      uint32_t e = 0, m = man;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x3FFu;
      bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/**
 * @brief Convert a float to an IEEE binary16 bit pattern (round to nearest,
 *        ties to even). Only the normal range is exercised by this test.
 */
static uint16_t floatToHalfBits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
  int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = bits & 0x7FFFFFu;
  if (exp <= 0)
    return sign;
  if (exp >= 0x1F)
    return (uint16_t)(sign | 0x7C00u);
  const uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
  // Round to nearest, ties to even on the truncated mantissa bits.
  const uint32_t rest = man & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u)))
    return (uint16_t)(h + 1);
  return h;
}

/**
 * @brief The flash-attention prefill kernel must agree with a CPU reference.
 *
 * This runs whichever flash variant the device selects -- on a GPU that
 * advertises the sub-group matrix-multiply-accumulate extension that is the
 * XMX/DPAS tile kernel, elsewhere the scalar block-Q walk -- so the same case
 * covers both. The operands are device-resident SVM buffers, which is the only
 * shape this entry point accepts. The test skips rather than fails where the
 * driver has no SVM, since that is a device property and not a defect.
 */
TEST(attention_kernels, flash_attention_prefill_matches_cpu_FP16) {
  const unsigned int M = 32, N_kv = 32, Hq = 4, Hkv = 2, d = 64;
  const unsigned int HDq = Hq * d, HDkv = Hkv * d;
  const bool causal = true;
  const float scale = 1.0f / std::sqrt((float)d);

  auto fill = [](size_t n, unsigned seed) {
    std::vector<uint16_t> v(n);
    unsigned x = seed;
    for (size_t i = 0; i < n; ++i) {
      x = x * 1103515245u + 12345u;
      v[i] = floatToHalfBits((float)((int)((x >> 16) & 0xFFu) - 128) / 256.0f);
    }
    return v;
  };
  const std::vector<uint16_t> Q = fill((size_t)M * HDq, 1u);
  const std::vector<uint16_t> K = fill((size_t)N_kv * HDkv, 2u);
  const std::vector<uint16_t> V = fill((size_t)N_kv * HDkv, 3u);

  // CPU reference: per query row, softmax over the causally visible keys.
  std::vector<float> ref((size_t)M * HDq, 0.0f);
  for (unsigned hq = 0; hq < Hq; ++hq) {
    const unsigned hkv = hq / (Hq / Hkv);
    for (unsigned m = 0; m < M; ++m) {
      const unsigned n_last = causal ? m : (N_kv - 1);
      std::vector<float> s(n_last + 1);
      float mx = -std::numeric_limits<float>::infinity();
      for (unsigned n = 0; n <= n_last; ++n) {
        float dot = 0.0f;
        for (unsigned x = 0; x < d; ++x)
          dot += halfBitsToFloat(Q[(size_t)m * HDq + hq * d + x]) *
                 halfBitsToFloat(K[(size_t)n * HDkv + hkv * d + x]);
        s[n] = dot * scale;
        mx = std::max(mx, s[n]);
      }
      float sum = 0.0f;
      for (unsigned n = 0; n <= n_last; ++n) {
        s[n] = std::exp(s[n] - mx);
        sum += s[n];
      }
      for (unsigned n = 0; n <= n_last; ++n) {
        const float p = s[n] / sum;
        for (unsigned x = 0; x < d; ++x)
          ref[(size_t)m * HDq + hq * d + x] +=
            p * halfBitsToFloat(V[(size_t)n * HDkv + hkv * d + x]);
      }
    }
  }

  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(cc, nullptr);
  cc->context_inst_.GetContext();

  const size_t qB = Q.size() * sizeof(uint16_t);
  const size_t kB = K.size() * sizeof(uint16_t);
  const size_t vB = V.size() * sizeof(uint16_t);
  const size_t oB = (size_t)M * HDq * sizeof(uint16_t);
  auto *Qs = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(qB));
  auto *Ks = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(kB));
  auto *Vs = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(vB));
  auto *Os = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(oB));
  if (!Qs || !Ks || !Vs || !Os) {
    GTEST_SKIP() << "device has no shared virtual memory; flash prefill needs "
                    "device-resident operands";
  }

  std::memcpy(Qs, Q.data(), qB);
  std::memcpy(Ks, K.data(), kB);
  std::memcpy(Vs, V.data(), vB);
  std::memset(Os, 0, oB);
  cc->command_queue_inst_.enqueueSVMUnmap(Qs);
  cc->command_queue_inst_.enqueueSVMUnmap(Ks);
  cc->command_queue_inst_.enqueueSVMUnmap(Vs);
  cc->command_queue_inst_.enqueueSVMUnmap(Os);

  const bool ok = nntrainer::flash_attention_prefill_f16_cl(
    Qs, Ks, Vs, Os, M, N_kv, Hq, Hkv, d, /*max_seq_len=*/0, causal,
    /*svm_inputs=*/true);
  ASSERT_TRUE(ok) << "flash attention prefill dispatch failed";

  cc->command_queue_inst_.enqueueSVMMap(Os, oB, /*read_only=*/true);
  std::vector<uint16_t> O(Os, Os + (size_t)M * HDq);
  cc->command_queue_inst_.enqueueSVMUnmap(Os);

  cc->context_inst_.releaseSVMRegion(Qs);
  cc->context_inst_.releaseSVMRegion(Ks);
  cc->context_inst_.releaseSVMRegion(Vs);
  cc->context_inst_.releaseSVMRegion(Os);

  // Relative L2, not an element-wise bound: the kernel accumulates in a
  // different order than the reference and stores fp16, so a few units in the
  // last place per element are expected and only a systematic error matters.
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < O.size(); ++i) {
    const double g = halfBitsToFloat(O[i]) - (double)ref[i];
    num += g * g;
    den += (double)ref[i] * (double)ref[i];
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
  EXPECT_LT(rel, 3e-2) << "flash attention prefill diverges from the CPU "
                          "reference, relative L2 = "
                       << rel;
}

#endif

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
