// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_layers_tie_word_embedding.cpp
 * @date 01 October 2026
 * @brief Host lm_head of the tied word embedding with a Q6_K weight.
 * @see   https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug   No known bugs except for NYI items
 *
 * @details The tied Q6_K lm_head runs on the host as the q6_K x q8_K integer
 * dot (the hidden row quantized to q8_K once, one dot per vocab row). These
 * cases drive the layer itself and check that
 *   (1) every logit equals dot_q6_K_q8_K of that row against the q8_K hidden,
 *       bit for bit, on the backends that implement the integer dot, and
 *   (2) every logit stays within the q8_K quantization bound of the fp32
 *       reference (dequantize the row, fp32 dot).
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <cpu_backend.h>
#include <htp_tie_word_embedding.h>
#include <layer_context.h>
#include <tensor.h>
#include <tie_word_embedding.h>
#include <var_grad.h>
#include <weight.h>

/**
 * The OpenCL / CUDA builds may take a device lm_head GEMV ahead of the host
 * one, so the bit-exact host check runs only on CPU-only builds; the fp32
 * reference bound holds everywhere.
 */
#if defined(NNTR_CPU_HAS_Q6_K_Q8_K_DOT) && !defined(ENABLE_OPENCL) &&          \
  !(defined(ENABLE_CUDA) && ENABLE_CUDA == 1)
#define TIED_LMHEAD_HOST_Q8K_EXACT 1
#endif

namespace {

constexpr unsigned int kHidden = 512; /**< two q6_K / q8_K blocks per row */
constexpr unsigned int kVocab = 301;  /**< not a multiple of any thread count */
constexpr size_t kQ6KBlockBytes = 210;
constexpr size_t kQ8KBlockBytes = 4 + 256 + 16 * sizeof(int16_t);

/**
 * @brief Logits of a tied Q6_K lm_head driven through the layer.
 */
struct TiedLmHeadRun {
  std::vector<float> hidden; /**< fp32 copy of the hidden fed in */
  std::vector<uint8_t> q6k;  /**< the Q6_K weight bytes */
  std::vector<float> logits; /**< the layer's logits as fp32 */
};

/**
 * @brief Build a tied lm_head with a random Q6_K weight, run one decode step
 * and return what it produced.
 *
 * @param act activation dtype ("FP32" or "FP16")
 */
TiedLmHeadRun runTiedLmHead(const std::string &act, bool htp = false) {
  using nntrainer::Tensor;
  using nntrainer::TensorDim;

  const TensorDim::DataType act_dt =
    act == "FP16" ? TensorDim::DataType::FP16 : TensorDim::DataType::FP32;
  TensorDim in_dim(1, 1, 1, kHidden,
                   TensorDim::TensorType(nntrainer::Tformat::NCHW, act_dt));

  std::unique_ptr<nntrainer::LayerImpl> layer;
  if (htp)
    layer = std::make_unique<nntrainer::HtpTieWordEmbedding>();
  else
    layer = std::make_unique<nntrainer::TieWordEmbedding>();
  layer->setProperty({"unit=" + std::to_string(kVocab), "disable_bias=true"});

  nntrainer::InitLayerContext init_ctx({in_dim}, {true}, false, "tied_lm_head",
                                       "", 0.0, {"NCHW", "Q6_K", act}, 1.0,
                                       ml::train::ExecutionMode::INFERENCE);
  layer->finalize(init_ctx);

  EXPECT_EQ(init_ctx.getWeightsSpec().size(), 1u);
  const TensorDim w_dim = std::get<0>(init_ctx.getWeightsSpec()[0]);
  EXPECT_EQ(w_dim.getDataType(), TensorDim::DataType::Q6_K);
  EXPECT_EQ(w_dim.height(), kVocab);
  EXPECT_EQ(w_dim.width(), kHidden);

  TiedLmHeadRun r;
  std::mt19937 rng(20261001);
  std::uniform_real_distribution<float> wdist(-0.5f, 0.5f);
  std::uniform_real_distribution<float> hdist(-2.0f, 2.0f);

  std::vector<float> w_f32((size_t)kVocab * kHidden);
  for (auto &v : w_f32)
    v = wdist(rng);
  r.q6k.resize((size_t)kVocab * (kHidden / 256) * kQ6KBlockBytes);
  nntrainer::quantize_q6_K(w_f32.data(), r.q6k.data(), kVocab, kHidden,
                           nullptr);

  nntrainer::Weight weight(w_dim, nntrainer::Initializer::ZEROS,
                           nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, 0.0f,
                           false, true, "tied_lm_head:weight");
  EXPECT_EQ(weight.getVariableRef().getMemoryBytes(), r.q6k.size());
  std::memcpy(weight.getVariableRef().getData<uint8_t>(), r.q6k.data(),
              r.q6k.size());

  nntrainer::Var_Grad input(in_dim, nntrainer::Initializer::NONE, false, true,
                            "tied_lm_head:input");
  r.hidden.resize(kHidden);
  for (unsigned int k = 0; k < kHidden; ++k) {
    float v = hdist(rng);
    if (act_dt == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      _FP16 h = static_cast<_FP16>(v);
      input.getVariableRef().getData<_FP16>()[k] = h;
      v = static_cast<float>(h); /** the layer sees the rounded value */
#endif
    } else {
      input.getVariableRef().getData<float>()[k] = v;
    }
    r.hidden[k] = v;
  }

  const TensorDim out_dim = init_ctx.getOutSpecs()[0].variable_spec.dim;
  EXPECT_EQ(out_dim.width(), kVocab);
  EXPECT_EQ(out_dim.getDataType(), htp ? TensorDim::DataType::FP32 : act_dt);
  nntrainer::Var_Grad output(out_dim, nntrainer::Initializer::NONE, false, true,
                             "tied_lm_head:output");

  nntrainer::RunLayerContext run_ctx("tied_lm_head", false, 0.0f, false, 1.0f,
                                     nullptr, false, {&weight}, {&input},
                                     {&output}, {});
  layer->incremental_forwarding(run_ctx, 0, 1, false);

  r.logits.resize(kVocab);
  for (unsigned int v = 0; v < kVocab; ++v) {
    if (out_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      r.logits[v] =
        static_cast<float>(output.getVariableRef().getData<_FP16>()[v]);
#endif
    } else {
      r.logits[v] = output.getVariableRef().getData<float>()[v];
    }
  }
  return r;
}

/**
 * @brief Compare a run against the fp32 dequant reference and, where the
 * backend has it, against the q6_K x q8_K row dot.
 *
 * @param r the layer run
 * @param out_rel extra relative slack for a narrowed (FP16) logit
 */
void checkTiedLmHead(const TiedLmHeadRun &r, float out_rel) {
  const size_t row_bytes = (kHidden / 256) * kQ6KBlockBytes;

  float amax = 0.0f;
  for (float h : r.hidden)
    amax = std::max(amax, std::fabs(h));
  /**
   * q8_K scales the row by 128 / amax and rounds, so each hidden value moves
   * by at most half a step of about amax / 127, except the one value clamped
   * from +128 to +127, which moves by a whole step.
   */
  const float half_step = 0.5f * amax / 127.0f;

#ifdef TIED_LMHEAD_HOST_Q8K_EXACT
  std::vector<uint8_t> q8k((kHidden / 256) * kQ8KBlockBytes);
  nntrainer::quantize_row_q8_K<float>(r.hidden.data(), q8k.data(), kHidden);
#endif

  std::vector<float> row(kHidden);
  for (unsigned int v = 0; v < kVocab; ++v) {
    const void *wrow = r.q6k.data() + row_bytes * v;
    nntrainer::dequantize_row_q6_K(wrow, row.data(), kHidden);
    double ref = 0.0, abs_w = 0.0, max_w = 0.0;
    for (unsigned int k = 0; k < kHidden; ++k) {
      ref += (double)row[k] * r.hidden[k];
      abs_w += std::fabs(row[k]);
      max_w = std::max(max_w, (double)std::fabs(row[k]));
    }
    const float bound =
      (float)(half_step * (abs_w + 2.0 * max_w) * 1.01 + 1e-3) +
      out_rel * std::fabs((float)ref);
    EXPECT_NEAR(r.logits[v], (float)ref, bound) << "vocab row " << v;

#ifdef TIED_LMHEAD_HOST_Q8K_EXACT
    const float q8 = nntrainer::dot_q6_K_q8_K(kHidden, wrow, q8k.data());
    if (out_rel == 0.0f)
      EXPECT_EQ(r.logits[v], q8) << "vocab row " << v;
    else
      EXPECT_NEAR(r.logits[v], q8, out_rel * std::fabs(q8) + 1e-3f)
        << "vocab row " << v;
#endif
  }
}

} // namespace

/**
 * @brief FP32 activation: logits are the q6_K x q8_K dot exactly, and within
 *        the q8_K bound of the fp32 dequant reference.
 */
TEST(TieWordEmbeddingLmHead, q6_K_host_fp32_activation_p) {
  TiedLmHeadRun r = runTiedLmHead("FP32");
  checkTiedLmHead(r, 0.0f);
}

TEST(HtpTieWordEmbeddingLmHead, q6_K_host_fp32_activation_p) {
  checkTiedLmHead(runTiedLmHead("FP32", true), 0.0f);
}

#ifdef ENABLE_FP16
/**
 * @brief FP16 activation: the hidden is widened, the integer dot runs as for
 *        FP32 and each logit is narrowed to the FP16 output.
 */
TEST(TieWordEmbeddingLmHead, q6_K_host_fp16_activation_p) {
  TiedLmHeadRun r = runTiedLmHead("FP16");
  checkTiedLmHead(r, 1.0f / 1024.0f);
}

TEST(HtpTieWordEmbeddingLmHead, q6_K_host_fp16_activation_fp32_logits_p) {
  checkTiedLmHead(runTiedLmHead("FP16", true), 0.0f);
}
#endif
