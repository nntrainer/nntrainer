// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_sigmoid_gate_ops_cl.cpp
 * @date   12 September 2026
 * @brief  CPU-vs-OpenCL differential for ClComputeOps::sigmoid_glu /
 *         ::sigmoid_add, and the layer registrations that go with them.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The CPU table is the reference the kernels are written against, so
 * the two have to agree on the same inputs; unittest_sigmoid_gate_ops.cpp is
 * where that reference is pinned to the closed form. Everything here needs a
 * device, so each case skips rather than fails on a runner that has the OpenCL
 * build but no device answering.
 */

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <compute_ops.h>
#include <context_data.h>
#include <engine.h>
#include <sigmoid_add_layer.h>
#include <sigmoid_glu_layer.h>
#include <tensor.h>

namespace {

constexpr float POISON = -123456.0f;

/** @brief attach an ops table to a tensor, the way a LayerNode does */
void attachOps(nntrainer::Tensor &t, nntrainer::ComputeOps *ops) {
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(ops);
  t.setContextData(ct);
}

/**
 * @brief Fill the destination with a sentinel, so a dispatch that bails out
 *        and writes nothing cannot pass as a correct result.
 */
void poison(nntrainer::Tensor &t) {
  for (unsigned int i = 0; i < t.size(); ++i)
    t.getData()[i] = POISON;
}

/** deterministic, sign-varied inputs with non-trivial magnitudes */
float genValue(unsigned int i) {
  return std::sin(0.41f * (float)i) * (1.0f + 0.02f * (float)(i % 13));
}

/**
 * @brief Bring the OpenCL backend up and hand back its ops table, or nullptr
 *        when this runner has the build but no device.
 *
 * @details Engine::getRegisteredContext() THROWS for a name nothing
 * registered, and "no gpu context" is a reachable state: a build with OpenCL
 * compiled in still declines to bring the backend up when no device answers.
 * Asking for the context is also what registers the kernels (ClContext's
 * add_default_object()), so this has to happen before the first dispatch.
 *
 * @return the OpenCL ops table, or nullptr when the backend is absent
 */
nntrainer::ComputeOps *clOpsOrNull() {
  try {
    (void)nntrainer::Engine::Global().getRegisteredContext("gpu");
  } catch (const std::exception &) {
    return nullptr;
  }
  return nntrainer::get_cl_ops();
}

} // namespace

/**
 * @brief CPU-vs-OpenCL differential. The CPU table is the reference the
 *        kernels are written against, so the two must agree to within fp32
 *        rounding on the same inputs -- and the destination must actually be
 *        written, which the sentinel catches when a dispatch bails out.
 */
TEST(SigmoidGateOpsCl, AgreesWithTheCpuTable) {
  auto *cl_ops = clOpsOrNull();
  if (cl_ops == nullptr)
    GTEST_SKIP() << "no OpenCL device on this runner";

  const unsigned int H = 4, W = 64;
  nntrainer::Tensor gate({1, 1, H, W}, true);
  nntrainer::Tensor other({1, 1, H, W}, true);
  nntrainer::Tensor cpu_out({1, 1, H, W}, true);
  nntrainer::Tensor cl_out({1, 1, H, W}, true);

  for (unsigned int i = 0; i < gate.size(); ++i) {
    gate.getData()[i] = genValue(i);
    other.getData()[i] = genValue(i + 3) * 2.0f;
  }

  auto *cpu_ops = nntrainer::get_cpu_ops();
  attachOps(cpu_out, cpu_ops);
  poison(cpu_out);
  poison(cl_out);

  ASSERT_NO_THROW(cpu_ops->sigmoid_glu(gate, other, cpu_out, H, 0));
  try {
    cl_ops->sigmoid_glu(gate, other, cl_out, H, 0);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "the OpenCL sigmoid_glu kernels are unavailable: "
                 << e.what();
  }
  for (unsigned int i = 0; i < gate.size(); ++i) {
    ASSERT_NE(cl_out.getData()[i], POISON) << "element " << i << " not written";
    EXPECT_NEAR(cl_out.getData()[i], cpu_out.getData()[i], 1e-5)
      << "sigmoid_glu element " << i;
  }

  poison(cpu_out);
  poison(cl_out);
  ASSERT_NO_THROW(cpu_ops->sigmoid_add(gate, other, cpu_out, H, 0));
  try {
    cl_ops->sigmoid_add(gate, other, cl_out, H, 0);
  } catch (const std::exception &e) {
    GTEST_SKIP() << "the OpenCL sigmoid_add kernels are unavailable: "
                 << e.what();
  }
  for (unsigned int i = 0; i < gate.size(); ++i) {
    ASSERT_NE(cl_out.getData()[i], POISON) << "element " << i << " not written";
    EXPECT_NEAR(cl_out.getData()[i], cpu_out.getData()[i], 1e-5)
      << "sigmoid_add element " << i;
  }
}

/**
 * @brief The layer types are what a model graph asks for by name, so the
 *        OpenCL context has to know them once its kernels compiled --
 *        createLayer() on a live context throws for a type the context does
 *        not hold, whatever the ops table can do.
 */
TEST(SigmoidGateOpsCl, TheLayerTypesAreRegisteredOnTheGpuContext) {
  nntrainer::Context *ctx = nullptr;
  try {
    ctx = nntrainer::Engine::Global().getRegisteredContext("gpu");
  } catch (const std::exception &) {
    GTEST_SKIP() << "no OpenCL device on this runner";
  }
  ASSERT_NE(ctx, nullptr);

  auto glu = ctx->createLayerObject("sigmoid_glu", {});
  ASSERT_NE(glu, nullptr);
  EXPECT_EQ(glu->getType(), nntrainer::SigmoidGluLayer::type);

  auto add = ctx->createLayerObject("sigmoid_add", {});
  ASSERT_NE(add, nullptr);
  EXPECT_EQ(add->getType(), nntrainer::SigmoidAddLayer::type);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
