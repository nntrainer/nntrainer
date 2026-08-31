// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_layers_norm_activation_neutral.cpp
 * @date 28 July 2026
 * @brief LayerNorm / Activation collapse: ONE neutral Layer per op,
 *        registered under the same type string on every context, with the math
 *        reaching the backend through ComputeOps::layer_norm / ::activation.
 * @see   https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug   No known bugs except for NYI items
 *
 * @details The superseded fork branches shipped kernel-level tests only, which
 * is precisely why the fork pattern was never caught by CI: a kernel test
 * cannot see which Layer class a context hands out. These cases assert the
 * collapse itself --
 *   (1) createLayerObject("layer_normalization"/"activation") returns the SAME
 *       core C++ class on cpu and on gpu (no LayerNormLayerCl and no
 *       ActivationLayerCl),
 *   (2) each engine's ComputeOps table actually implements the two new
 *       whole-ops (or throws per its documented contract),
 *   (3) the OpenCL contract is throw-not-silently-fall-back for the activation
 *       modes it has no kernel for.
 */
#include <typeinfo>

#include <gtest/gtest.h>

#include <activation_layer.h>
#include <common_properties.h>
#include <compute_ops.h>
#include <context_data.h>
#include <engine.h>
#include <layer_normalization_layer.h>
#include <tensor.h>

namespace {

nntrainer::ComputeOps *opsOf(const char *engine) {
  auto *ct = nntrainer::Engine::Global().getRegisteredContext(engine);
  EXPECT_NE(ct, nullptr);
  return ct->getContextData()->getComputeOps();
}

} // namespace

/**
 * @brief The cpu context hands out the core LayerNormalizationLayer /
 *        ActivationLayer. Baseline for the gpu comparison below.
 */
TEST(NeutralNormActivation, CpuContextYieldsCoreClasses) {
  auto ln = nntrainer::Engine::Global().createLayerObject(
    nntrainer::LayerNormalizationLayer::type, {"engine=cpu", "axis=3"});
  ASSERT_NE(ln, nullptr);
  EXPECT_EQ(ln->getType(), nntrainer::LayerNormalizationLayer::type);
  EXPECT_NE(dynamic_cast<nntrainer::LayerNormalizationLayer *>(ln.get()),
            nullptr);

  auto act = nntrainer::Engine::Global().createLayerObject(
    nntrainer::ActivationLayer::type, {"engine=cpu", "activation=gelu"});
  ASSERT_NE(act, nullptr);
  EXPECT_EQ(act->getType(), nntrainer::ActivationLayer::type);
  EXPECT_NE(dynamic_cast<nntrainer::ActivationLayer *>(act.get()), nullptr);
}

#ifdef ENABLE_OPENCL
/**
 * @brief THE collapse assertion: the gpu context hands out the very same C++
 *        class as the cpu context for both type strings. A returned
 *        LayerNormLayerCl / ActivationLayerCl would fail the dynamic_cast and
 *        the typeid comparison — that is the regression guard against the fork
 *        pattern coming back.
 */
TEST(NeutralNormActivation, GpuContextYieldsTheSameCoreClasses) {
  auto ln_cpu = nntrainer::Engine::Global().createLayerObject(
    nntrainer::LayerNormalizationLayer::type, {"engine=cpu", "axis=3"});
  auto ln_gpu = nntrainer::Engine::Global().createLayerObject(
    nntrainer::LayerNormalizationLayer::type, {"engine=gpu", "axis=3"});
  ASSERT_NE(ln_gpu, nullptr);
  EXPECT_EQ(ln_gpu->getType(), nntrainer::LayerNormalizationLayer::type);
  EXPECT_NE(dynamic_cast<nntrainer::LayerNormalizationLayer *>(ln_gpu.get()),
            nullptr);
  EXPECT_STREQ(typeid(*ln_gpu).name(), typeid(*ln_cpu).name());

  auto act_cpu = nntrainer::Engine::Global().createLayerObject(
    nntrainer::ActivationLayer::type, {"engine=cpu", "activation=gelu"});
  auto act_gpu = nntrainer::Engine::Global().createLayerObject(
    nntrainer::ActivationLayer::type, {"engine=gpu", "activation=gelu"});
  ASSERT_NE(act_gpu, nullptr);
  EXPECT_EQ(act_gpu->getType(), nntrainer::ActivationLayer::type);
  EXPECT_NE(dynamic_cast<nntrainer::ActivationLayer *>(act_gpu.get()), nullptr);
  EXPECT_STREQ(typeid(*act_gpu).name(), typeid(*act_cpu).name());
}

/**
 * @brief The gpu ComputeOps really implements layer_norm — i.e. the neutral
 *        Layer's in.getOps()->layer_norm(...) lands on an OpenCL kernel, not on
 *        the throwing ComputeOps base. Checked through the ops table the gpu
 *        ContextData carries, which is exactly what a gpu-attached Tensor
 *        resolves to.
 */
TEST(NeutralNormActivation, GpuOpsImplementLayerNormAndGelu) {
  auto *ops = opsOf("gpu");
  ASSERT_NE(ops, nullptr);

  constexpr unsigned int H = 3, W = 8;
  const float eps = 1e-3f;
  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::Tensor in(1, 1, H, W, t_fp32), out(1, 1, H, W, t_fp32);
  nntrainer::Tensor gamma(1, 1, 1, W, t_fp32), beta(1, 1, 1, W, t_fp32);

  float *ip = in.getData<float>();
  for (unsigned int i = 0; i < H * W; ++i)
    ip[i] = (i % 7) * 0.5f - 1.0f;
  gamma.setValue(1.25f);
  beta.setValue(-0.5f);

  ASSERT_NO_THROW(
    ops->layer_norm(in, out, gamma, beta, eps, H, /*row_offset=*/0));

  for (unsigned int h = 0; h < H; ++h) {
    double mean = 0.0;
    for (unsigned int w = 0; w < W; ++w)
      mean += ip[h * W + w];
    mean /= W;
    double var = 0.0;
    for (unsigned int w = 0; w < W; ++w) {
      double d = ip[h * W + w] - mean;
      var += d * d;
    }
    var /= W;
    const double inv = 1.0 / std::sqrt(var + (double)eps);
    for (unsigned int w = 0; w < W; ++w) {
      const double ref = (ip[h * W + w] - mean) * inv * 1.25 - 0.5;
      EXPECT_NEAR(out.getData<float>()[h * W + w], (float)ref, 1e-4f);
    }
  }

  nntrainer::Tensor gin(1, 1, 1, 8, t_fp32), gout(1, 1, 1, 8, t_fp32);
  for (unsigned int i = 0; i < 8; ++i)
    gin.getData<float>()[i] = -2.0f + 0.5f * i;
  ASSERT_NO_THROW(
    ops->activation(gin, gout, (int)nntrainer::ActivationType::ACT_GELU, 1, 0));
  for (unsigned int i = 0; i < 8; ++i) {
    const double x = gin.getData<float>()[i];
    const double ref = 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752));
    EXPECT_NEAR(gout.getData<float>()[i], (float)ref, 1e-4f);
  }
}

/**
 * @brief R3 / checklist item 16: OpenCL has kernels for gelu and tanh_gelu
 *        only, and THROWS for the rest — no silent host fallback onto a
 *        possibly cl_mem/SVM-resident tensor. supports_activation() is the
 *        non-throwing query. This is the former ActivationLayerCl::getGeluMode
 *        contract, now enforced in the op table instead of in a forked Layer.
 */
TEST(NeutralNormActivation, GpuActivationThrowsForUnacceleratedModes) {
  auto *ops = opsOf("gpu");
  ASSERT_NE(ops, nullptr);

  EXPECT_TRUE(
    ops->supports_activation((int)nntrainer::ActivationType::ACT_GELU));
  EXPECT_TRUE(
    ops->supports_activation((int)nntrainer::ActivationType::ACT_TANH_GELU));
  EXPECT_FALSE(
    ops->supports_activation((int)nntrainer::ActivationType::ACT_RELU));
  EXPECT_FALSE(
    ops->supports_activation((int)nntrainer::ActivationType::ACT_SOFTMAX));

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::Tensor in(1, 1, 1, 8, t_fp32), out(1, 1, 1, 8, t_fp32);
  in.setValue(0.5f);
  EXPECT_THROW(
    ops->activation(in, out, (int)nntrainer::ActivationType::ACT_RELU, 1, 0),
    std::invalid_argument);
}

#ifdef ENABLE_FP16
/**
 * @brief The OpenCL LayerNorm kernels are single-dtype: a mixed
 *        activation/weight dtype throws rather than silently producing garbage
 *        (the CPU op table handles all four combos, so engine=cpu is the fix).
 */
TEST(NeutralNormActivation, GpuLayerNormRejectsMixedWeightDtype) {
  auto *ops = opsOf("gpu");
  ASSERT_NE(ops, nullptr);

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::TensorDim::TensorType t_fp16 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP16};
  nntrainer::Tensor in(1, 1, 2, 8, t_fp32), out(1, 1, 2, 8, t_fp32);
  nntrainer::Tensor gamma(1, 1, 1, 8, t_fp16), beta(1, 1, 1, 8, t_fp16);
  in.setValue(1.0f);
  gamma.setValue((_FP16)1.0f);
  beta.setValue((_FP16)0.0f);

  EXPECT_THROW(ops->layer_norm(in, out, gamma, beta, 1e-3f, 2, 0),
               std::invalid_argument);
}
#endif // ENABLE_FP16
#endif // ENABLE_OPENCL

/**
 * @brief The cpu ops table services EVERY activation mode (that is why
 *        supports_activation() defaults to true there) and matches the host
 *        ActiFunc for the non-gelu modes.
 */
TEST(NeutralNormActivation, CpuOpsServiceEveryActivationMode) {
  auto *ops = opsOf("cpu");
  ASSERT_NE(ops, nullptr);

  EXPECT_TRUE(
    ops->supports_activation((int)nntrainer::ActivationType::ACT_RELU));

  nntrainer::TensorDim::TensorType t_fp32 = {nntrainer::Tformat::NCHW,
                                             nntrainer::Tdatatype::FP32};
  nntrainer::Tensor in(1, 1, 1, 8, t_fp32), out(1, 1, 1, 8, t_fp32);
  for (unsigned int i = 0; i < 8; ++i)
    in.getData<float>()[i] = -2.0f + 0.5f * i;

  ASSERT_NO_THROW(
    ops->activation(in, out, (int)nntrainer::ActivationType::ACT_RELU, 1, 0));
  for (unsigned int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(out.getData<float>()[i],
                    std::max(0.0f, in.getData<float>()[i]));

  // ACT_NONE is a straight copy over the window.
  out.setValue(-99.0f);
  ASSERT_NO_THROW(
    ops->activation(in, out, (int)nntrainer::ActivationType::ACT_NONE, 1, 0));
  for (unsigned int i = 0; i < 8; ++i)
    EXPECT_FLOAT_EQ(out.getData<float>()[i], in.getData<float>()[i]);
}
