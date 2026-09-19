// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_residency_planner.cpp
 * @date   30 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Tests for ResidencyPlanner::classify(), the pure function that
 *         decides where a planned tensor lives.
 *
 * @details Deliberately built WITHOUT OpenCL and without FP16, and it includes
 * nothing that would pull either in. classify() is a header-only pure function
 * of five facts the graph already knows, so it needs no device, no driver and
 * no GPU build to exercise -- and the CI jobs that run the test suite are the
 * ones with OpenCL off. The allocator's demote step that follows classify()
 * is driven here too, through a host allocator that declines a class rather
 * than through a real device. Keeping the decision table
 * here means the rules are covered on every runner; the companion suite
 * (unittest_cl_residency) covers what only a device can answer: that a placed
 * tensor really comes back as a cl_mem, that offset reuse shares one handle,
 * and that a real graph reaches the plane at all.
 */

#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <mem_allocator.h>
#include <residency_planner.h>
#include <residency_policy.h>
#include <tensor_pool.h>

namespace {

constexpr auto GPU = ml::train::LayerComputeEngine::GPU;
constexpr auto CPU = ml::train::LayerComputeEngine::CPU;
using RC = nntrainer::ResidencyClass;

/** @brief a planner on an allocator that has both a shared and a device plane
 */
nntrainer::ResidencyPlanner devicePlanner() {
  nntrainer::ResidencyPlanner p;
  p.device_backed = true;
  p.device_pool = true;
  return p;
}

/** @brief classify with the arguments this suite varies one at a time */
RC classify(const nntrainer::ResidencyPlanner &p,
            ml::train::LayerComputeEngine e, bool all_consumers_device,
            bool is_fp16, bool needs_host_init,
            const std::string &name = "act") {
  return p.classify(e, all_consumers_device, is_fp16, needs_host_init, name);
}

} // namespace

/**
 * @brief A host-only allocator has one plane, so there is nothing to place and
 *        every tensor keeps host residency whatever the graph says.
 */
TEST(ResidencyPlanner, a_host_only_allocator_places_everything_on_the_host) {
  nntrainer::ResidencyPlanner p; /** device_backed stays false */
  EXPECT_EQ(classify(p, GPU, true, true, false), RC::HOST);
  EXPECT_EQ(classify(p, CPU, false, false, true), RC::HOST);
}

/**
 * @brief The heuristic itself: written on the device, read only on the device,
 *        in the type the kernels compute in.
 */
TEST(ResidencyPlanner, a_gpu_written_gpu_read_fp16_tensor_is_device_resident) {
  EXPECT_EQ(classify(devicePlanner(), GPU, true, true, false), RC::GPU_CLMEM);
}

/**
 * @brief Each of the three facts the heuristic needs, withheld one at a time.
 *        A tensor may only be placed where every one of its readers can reach
 *        it, so each of these keeps the shared plane.
 */
TEST(ResidencyPlanner, withholding_any_one_fact_keeps_the_shared_plane) {
  const auto p = devicePlanner();
  /** a host producer: the bytes are written where the host can write them */
  EXPECT_EQ(classify(p, CPU, true, true, false), RC::SVM);
  /** one host consumer: the placement has to be one every reader can reach */
  EXPECT_EQ(classify(p, GPU, false, true, false), RC::SVM);
  /** not the type the kernels that read the device plane compute in */
  EXPECT_EQ(classify(p, GPU, true, false, false), RC::SVM);
}

/**
 * @brief An allocator with no device plane to place tensors in downgrades the
 *        heuristic's answer rather than refusing it.
 */
TEST(ResidencyPlanner, no_device_pool_downgrades_to_the_shared_plane) {
  nntrainer::ResidencyPlanner p;
  p.device_backed = true;
  p.device_pool = false;
  EXPECT_EQ(classify(p, GPU, true, true, false), RC::SVM);
}

/**
 * @brief A declared input boundary raises a host-produced tensor onto the
 *        device plane, because the application's own upload is the point at
 *        which the two planes agree. Only tensors it names are raised.
 */
TEST(ResidencyPlanner, a_declared_raise_boundary_lifts_a_host_producer) {
  auto p = devicePlanner();
  p.raise = "uploaded";
  EXPECT_EQ(classify(p, CPU, true, true, false, "uploaded_w"), RC::GPU_CLMEM);
  EXPECT_EQ(classify(p, CPU, true, true, false, "other"), RC::SVM);
  /** a raise still needs every consumer on the device: the boundary is the
   *  producer's, and says nothing about a host reader downstream */
  EXPECT_EQ(classify(p, CPU, false, true, false, "uploaded_w"), RC::SVM);
}

/**
 * @brief A declared output boundary keeps a device-produced tensor on the
 *        device plane despite the one host consumer that reads it back.
 */
TEST(ResidencyPlanner, a_declared_lower_boundary_keeps_a_device_producer) {
  auto p = devicePlanner();
  p.lower = "readback";
  EXPECT_EQ(classify(p, GPU, false, true, false, "readback_0"), RC::GPU_CLMEM);
  EXPECT_EQ(classify(p, GPU, false, true, false, "other"), RC::SVM);
  /** and only in the declared direction: a host producer is not lowered */
  EXPECT_EQ(classify(p, CPU, false, true, false, "readback_0"), RC::SVM);
}

/**
 * @brief An exclusion overrides the heuristic and both boundaries: it is the
 *        application saying it also touches the tensor from the host.
 */
TEST(ResidencyPlanner, a_declared_exclusion_overrides_every_promotion) {
  auto p = devicePlanner();
  p.raise = "cache_";
  p.exclude = "cache_";
  EXPECT_EQ(classify(p, GPU, true, true, false, "cache_k"), RC::SVM);
  EXPECT_EQ(classify(p, CPU, true, true, false, "cache_k"), RC::SVM);
  EXPECT_EQ(classify(p, GPU, true, true, false, "act"), RC::GPU_CLMEM);
}

/**
 * @brief A tensor that declares an Initializer is never device-resident.
 *
 * The initializer writes the host side of the allocation and core owns no
 * upload path, so a device placement would hand the kernels a buffer that
 * never saw those bytes. The combination is refused in the planner rather
 * than half honoured at allocation.
 */
TEST(ResidencyPlanner, an_initialised_tensor_is_refused_the_device_plane) {
  const auto p = devicePlanner();
  EXPECT_EQ(classify(p, GPU, true, true, true), RC::SVM);
  /** including one a declared boundary would otherwise have promoted */
  auto raised = devicePlanner();
  raised.raise = "uploaded";
  EXPECT_EQ(classify(raised, CPU, true, true, true, "uploaded_w"), RC::SVM);
}

/**
 * @brief The pattern list is comma-separated and matched as a substring, and
 *        an empty token never matches, so a stray comma is harmless.
 */
TEST(ResidencyPlanner, pattern_lists_are_comma_separated_substrings) {
  auto p = devicePlanner();
  p.exclude = "cache_,scratch";
  EXPECT_EQ(classify(p, GPU, true, true, false, "layer0/cache_k"), RC::SVM);
  EXPECT_EQ(classify(p, GPU, true, true, false, "scratchpad"), RC::SVM);
  EXPECT_EQ(classify(p, GPU, true, true, false, "act"), RC::GPU_CLMEM);

  p.exclude = ",,cache_,,";
  EXPECT_EQ(classify(p, GPU, true, true, false, "cache_v"), RC::SVM);
  EXPECT_EQ(classify(p, GPU, true, true, false, "act"), RC::GPU_CLMEM);

  p.exclude = nullptr;
  EXPECT_EQ(classify(p, GPU, true, true, false, "cache_v"), RC::GPU_CLMEM);
}

/**
 * @brief The step after classify(): the allocator has the last word, and a
 *        class it cannot back is demoted rather than bound.
 *
 * @details TensorPool::allocate() asks the allocator whether the class the
 * planner arrived at is one it can actually produce, because a placement is
 * only available if the memory behind it is. That step sits outside
 * classify() and was reachable only through the device-gated suite. Here it
 * is driven from a host build with an allocator that reports device-visible
 * memory without shared virtual memory: the planner says SVM, the allocator
 * cannot back SVM, and the tensor lands on the host plane with a pointer its
 * reader can dereference -- demoted, never left half-placed.
 */
TEST(ResidencyPlanner, the_allocator_demotes_a_class_it_cannot_back) {
  /**
   * @brief device-visible memory with no shared plane, allocated on the host
   */
  class DeviceVisibleNotShared : public nntrainer::MemAllocator {
  public:
    /** @copydoc MemAllocator::isDeviceVisible */
    bool isDeviceVisible() const override { return true; }
    /** @copydoc MemAllocator::isSVM */
    bool isSVM() const override { return false; }
  };

  nntrainer::TensorPool pool(false, "", "residency_demote",
                             ml::train::ExecutionMode::INFERENCE,
                             std::make_shared<DeviceVisibleNotShared>());
  const nntrainer::TensorDim dim(
    1, 1, 4, 16, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  auto *t = pool.request("act", dim, {0},
                         nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN,
                         nntrainer::Initializer::NONE,
                         /*is_weight_grad=*/false, GPU);
  pool.view("act_view", "act", dim, {1},
            nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN, 0, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  ASSERT_NE(t->getMemoryData(), nullptr);
  EXPECT_EQ(t->getMemoryData()->residency(), RC::HOST);
  EXPECT_FALSE(t->isClMem());
  EXPECT_NE(t->getData<float>(), nullptr);

  pool.deallocate();
}

/**
 * @brief The application's engine-neutral list is an exact type-name match,
 *        not a substring one: a layer type either is neutral or is not.
 */
TEST(ResidencyPolicyTest, engine_neutral_types_match_exactly) {
  nntrainer::ResidencyPolicy policy;
  policy.engine_neutral_types = {"multiout", "activation"};
  EXPECT_TRUE(policy.isEngineNeutral("multiout"));
  EXPECT_TRUE(policy.isEngineNeutral("activation"));
  EXPECT_FALSE(policy.isEngineNeutral("multiout_2"));
  EXPECT_FALSE(policy.isEngineNeutral("fully_connected"));
}

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
