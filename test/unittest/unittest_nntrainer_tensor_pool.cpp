// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_nntrainer_tensor_pool.cpp
 * @date 20 August 2021
 * @brief Tensor Pool Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstring>
#include <random>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <optimized_v1_planner.h>
#include <tensor_pool.h>

TEST(TensorPool, release_inference_scratch_preserves_state_and_shared_pages) {
  using namespace nntrainer;
#if defined(__linux__)
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#else
  const size_t page = 4096;
#endif
  const unsigned words = static_cast<unsigned>(page / sizeof(float));
  TensorPool pool;
  EXPECT_EQ(pool.releaseInferenceScratch(), 0u);
  auto *head = pool.request("head", TensorDim({words + 1}), {0},
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
  auto *state = pool.request("state", TensorDim({8}), {0},
                             TensorLifespan::FORWARD_FUNC_LIFESPAN);
  auto *alias = pool.view("state_alias", "state", TensorDim({8}), {1},
                          TensorLifespan::MAX_LIFESPAN);
  auto *tail = pool.request("tail", TensorDim({3 * words + 1}), {1},
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.finalize(BasicPlanner(), 0, 2);
  pool.allocate();
  std::fill_n(head->getData(), words + 1, 11.0f);
  std::fill_n(state->getData(), 8, 77.0f);
  std::fill_n(tail->getData(), 3 * words + 1, 22.0f);
  float *const address = head->getData();
#if defined(__linux__)
  EXPECT_EQ(pool.releaseInferenceScratch(), 3 * page);
  EXPECT_EQ(head->getData()[0], 0.0f);
  EXPECT_EQ(tail->getData()[words], 0.0f);
#else
  EXPECT_EQ(pool.releaseInferenceScratch(), 0u);
  EXPECT_EQ(head->getData()[0], 11.0f);
#endif
  EXPECT_EQ(head->getData(), address);
  // A live tensor protects its entire page, including scratch sharing it.
  EXPECT_EQ(head->getData()[words], 11.0f);
  EXPECT_EQ(tail->getData()[0], 22.0f);
  // The allocation's partial last page may contain allocator metadata.
  EXPECT_EQ(tail->getData()[3 * words], 22.0f);
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_EQ(state->getData()[i], 77.0f);
    EXPECT_EQ(alias->getData()[i], 77.0f);
  }
  pool.deallocate();
  EXPECT_EQ(pool.releaseInferenceScratch(), 0u);
}

TEST(TensorPool, release_inference_scratch_leaves_device_allocator_untouched) {
  class DeviceAllocator : public nntrainer::MemAllocator {
  public:
    std::string getName() override { return "test-device"; }
  };
  nntrainer::TensorPool pool(false, "", "", ml::train::ExecutionMode::INFERENCE,
                             std::make_shared<DeviceAllocator>());
  auto *scratch =
    pool.request("scratch", nntrainer::TensorDim({8192}), {0},
                 nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.finalize(nntrainer::BasicPlanner(), 0, 1);
  pool.allocate();
  std::fill_n(scratch->getData(), 8192, 33.0f);
  EXPECT_EQ(pool.releaseInferenceScratch(), 0u);
  for (unsigned i = 0; i < 8192; ++i)
    EXPECT_EQ(scratch->getData()[i], 33.0f);
}

constexpr unsigned int MEM_BYTES = 128;
constexpr unsigned int MEM_QUANT = 100;
constexpr unsigned int INTERVAL_SIZE = 5;
constexpr static auto max_ls = nntrainer::TensorLifespan::MAX_LIFESPAN;

/**
 * @brief creation and destruction
 */
TEST(TensorPool, create_destroy) { EXPECT_NO_THROW(nntrainer::TensorPool()); }

/**
 * @brief request empty tensor
 */
TEST(TensorPool, request_01_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.request("abc", nntrainer::TensorDim(), {},
                            nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request empty name
 */
TEST(TensorPool, request_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.request("", nntrainer::TensorDim({1}), {},
                            nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, request_03_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.request("abc", nntrainer::TensorDim({2}), {},
                            nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request tensor
 */
TEST(TensorPool, request_04_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t = nullptr;

  EXPECT_NO_THROW(t = pool.request("abc", nntrainer::TensorDim({1}), {},
                                   nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t, nullptr);
  EXPECT_FALSE(t->isAllocated());
}

/**
 * @brief request bigger size for view
 */
TEST(TensorPool, view_01_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("abc1", "abc", nntrainer::TensorDim({2}), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request view non existing tensor
 */
TEST(TensorPool, view_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_ANY_THROW(pool.view("abc1", "not_exist", nntrainer::TensorDim({1}), {},
                             nntrainer::TensorLifespan::UNMANAGED));
}

/**
 * @brief request with clashing name
 */
TEST(TensorPool, view_03_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("abc", "abc", nntrainer::TensorDim({1}), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief view with empty tensor
 */
TEST(TensorPool, view_04_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("abc1", "abc", nntrainer::TensorDim({}), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief view with empty name
 */
TEST(TensorPool, view_05_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("", "abc", nntrainer::TensorDim({1}), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request view of managed tensor
 */
TEST(TensorPool, view_06_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("abc", nntrainer::TensorDim({1}), {},
                               nntrainer::TensorLifespan::MAX_LIFESPAN));
  EXPECT_THROW(pool.view("abc1", "abc", nntrainer::TensorDim({1}), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, view_07_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("abc", nntrainer::TensorDim({1}), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(t2 = pool.view("abc1", "abc", nntrainer::TensorDim({1}), {},
                                 nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);
}

/**
 * @brief request try extending lifespan of unmanaged
 */
TEST(TensorPool, view_08_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("abc", nntrainer::TensorDim({1}), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.view("abc1", "abc", nntrainer::TensorDim({1}), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(
    t2 = pool.view("abc2", "abc", nntrainer::TensorDim({1}), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);
}

/**
 * @brief set batch
 */
TEST(TensorPool, set_batch_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("abc", nntrainer::TensorDim({1}), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_EQ(t1->batch(), 1u);
  EXPECT_NO_THROW(pool.setBatchSize("abc", 10));
  EXPECT_EQ(t1->batch(), 10u);
}

/**
 * @brief set batch for not exist tensor
 */
TEST(TensorPool, set_batch_02_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("abc", nntrainer::TensorDim({1}), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_THROW(pool.setBatchSize("not_exist", 10), std::invalid_argument);
  EXPECT_EQ(t1->batch(), 1u);
}

/**
 * @brief set batch for allocated tensor
 */
TEST(TensorPool, set_batch_03_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr;
  nntrainer::BasicPlanner basic_planner;

  EXPECT_NO_THROW(
    t1 = pool.request("abc", nntrainer::TensorDim({1}), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_NO_THROW(pool.finalize(basic_planner, 0, 1));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.setBatchSize("abc", 10), std::invalid_argument);
}

/**
 * @brief zero size pool as no usage
 */
TEST(TensorPool, finalize_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("abc1", nntrainer::TensorDim({1}), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("abc2", nntrainer::TensorDim({1}), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), 0u);

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief max lifespan tensors
 */
TEST(TensorPool, finalize_02_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("abc1", nntrainer::TensorDim({1}), {0}, max_ls));
  EXPECT_NE(t1, nullptr);

  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("abc2", nntrainer::TensorDim({1}), {1}, max_ls));
  EXPECT_NE(t2, nullptr);

  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief max lifespan tensors
 */
TEST(TensorPool, finalize_03_p) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
}

/**
 * @brief allocate
 */
TEST(TensorPool, allocate_deallocate_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("abc1", nntrainer::TensorDim({1}), {0}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("abc2", nntrainer::TensorDim({1}), {1}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_TRUE(t1->isAllocated());
  EXPECT_TRUE(t2->isAllocated());

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief allocate
 */
TEST(TensorPool, allocate_deallocate_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief request already allocated tensor
 */
TEST(TensorPool, allocate_deallocate_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("abc", nntrainer::TensorDim({1}), {0}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.view("abc1", "abc", nntrainer::TensorDim({1}), {1}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(
    t3 = pool.view("abc2", "abc", nntrainer::TensorDim({1}), {0}, max_ls));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NE(t1, t3);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_TRUE(t1->isAllocated());
  EXPECT_TRUE(t2->isAllocated());
  EXPECT_TRUE(t3->isAllocated());

  EXPECT_EQ(t1->getData<float>(), t2->getData<float>());
  EXPECT_EQ(t1->getData<float>(), t3->getData<float>());

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
  EXPECT_FALSE(t3->isAllocated());
}

/**
 * @brief validate memory full overlap
 */
TEST(TensorPool, validate_memory) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("abc1", nntrainer::TensorDim({100}), {0}, max_ls));

  EXPECT_NO_THROW(
    t2 = pool.request("abc2", nntrainer::TensorDim({100}), {1}, max_ls));

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_NO_THROW(pool.allocate());

  nntrainer::Tensor g1 = nntrainer::Tensor(nntrainer::TensorDim({100}));
  g1.setRandNormal();
  nntrainer::Tensor g2 = nntrainer::Tensor(nntrainer::TensorDim({100}));
  g2.setRandNormal();

  t1->copy(g1);
  t2->copy(g2);

  EXPECT_EQ(*t1, g1);
  EXPECT_EQ(*t2, g2);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief qint8 tensors reuse fp32 tensor memory space
 */
TEST(TensorPool, validate_memory_reuse_01_p) {
  // |--------- t1 ---------|
  // |-t2-||-t3-||-t4-||-t5-|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr,
                    *t5 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({4}), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2",
                      nntrainer::TensorDim({4}, {nntrainer::Tformat::NCHW,
                                                 nntrainer::Tdatatype::QINT8}),
                      {1}, nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3",
                      nntrainer::TensorDim({4}, {nntrainer::Tformat::NCHW,
                                                 nntrainer::Tdatatype::QINT8}),
                      {1}, nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(
    t4 = pool.request("t4",
                      nntrainer::TensorDim({4}, {nntrainer::Tformat::NCHW,
                                                 nntrainer::Tdatatype::QINT8}),
                      {1}, nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t4, nullptr);
  EXPECT_FALSE(t4->isAllocated());

  EXPECT_NO_THROW(
    t5 = pool.request("t5",
                      nntrainer::TensorDim({4}, {nntrainer::Tformat::NCHW,
                                                 nntrainer::Tdatatype::QINT8}),
                      {1}, nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t5, nullptr);
  EXPECT_FALSE(t5->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(),
            t1->bytes() + (t2->scale_size() + t3->scale_size() +
                           t4->scale_size() + t5->scale_size()) *
                            sizeof(float));

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());
}

// /**
//  * @brief qint4 tensors reuse fp32 tensor memory space
//  */
// TEST(TensorPool, validate_memory_reuse_02_p) {
//   // |--------- t1 ---------|
//   // |-t2-||-t3-||-t4-||-t5-|
//   nntrainer::TensorPool pool;
//   nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 =
//   nullptr,
//                     *t5 = nullptr;

//   EXPECT_NO_THROW(
//     t1 = pool.request("t1", nntrainer::TensorDim({4}), {0},
//                       nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
//   EXPECT_NE(t1, nullptr);
//   EXPECT_FALSE(t1->isAllocated());

//   EXPECT_NO_THROW(
//     t2 = pool.request("t2",
//                       nntrainer::TensorDim({8}, {nntrainer::Tformat::NCHW,
//                                                  nntrainer::Tdatatype::QINT4}),
//                       {1},
//                       nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
//   EXPECT_NE(t2, nullptr);
//   EXPECT_FALSE(t2->isAllocated());

//   EXPECT_NO_THROW(
//     t3 = pool.request("t3",
//                       nntrainer::TensorDim({7}, {nntrainer::Tformat::NCHW,
//                                                  nntrainer::Tdatatype::QINT4}),
//                       {1},
//                       nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
//   EXPECT_NE(t3, nullptr);
//   EXPECT_FALSE(t3->isAllocated());

//   EXPECT_NO_THROW(
//     t4 = pool.request("t4",
//                       nntrainer::TensorDim({8}, {nntrainer::Tformat::NCHW,
//                                                  nntrainer::Tdatatype::QINT4}),
//                       {1},
//                       nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
//   EXPECT_NE(t4, nullptr);
//   EXPECT_FALSE(t4->isAllocated());

//   EXPECT_NO_THROW(
//     t5 = pool.request("t5",
//                       nntrainer::TensorDim({7}, {nntrainer::Tformat::NCHW,
//                                                  nntrainer::Tdatatype::QINT4}),
//                       {1},
//                       nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
//   EXPECT_NE(t5, nullptr);
//   EXPECT_FALSE(t5->isAllocated());

//   EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
//   EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes());

//   EXPECT_NO_THROW(pool.allocate());

//   EXPECT_EQ(t1->getAddress<float>(0), (float *)t2->getAddress<uint8_t>(0));
//   EXPECT_EQ(t1->getAddress<float>(1), (float *)t3->getAddress<uint8_t>(0));
//   EXPECT_EQ(t1->getAddress<float>(2), (float *)t4->getAddress<uint8_t>(0));
//   EXPECT_EQ(t1->getAddress<float>(3), (float *)t5->getAddress<uint8_t>(0));

//   EXPECT_NO_THROW(pool.deallocate());
// }

/**
 * @brief check if data span of two tensor testOverlap
 *
 * @param t1 tensor1
 * @param t2 tensor2
 */
static void testNoOverlap(nntrainer::Tensor *t1, nntrainer::Tensor *t2) {
  float *t1_start = t1->getData<float>();
  float *t1_end = t1_start + t1->size();

  float *t2_start = t2->getData<float>();
  float *t2_end = t2_start + t2->size();

  EXPECT_NE(t1_start, nullptr);
  EXPECT_NE(t2_start, nullptr);
  EXPECT_TRUE(!(t1_start < t2_end && t2_start < t1_end))
    << "t1 and t2 overlaps";
}

/**
 * @brief test if t1 is including t2
 *
 * @param t1 t1 tensor 1
 * @param t2 t2 tensor 2
 */
static void testSubset(nntrainer::Tensor *t1, nntrainer::Tensor *t2) {
  float *t1_start = t1->getData<float>();
  float *t1_end = t1_start + t1->size();

  float *t2_start = t2->getData<float>();
  float *t2_end = t2_start + t2->size();

  EXPECT_NE(t1_start, nullptr);
  EXPECT_NE(t2_start, nullptr);
  EXPECT_TRUE(t1_start <= t2_start && t2_end <= t1_end)
    << "t2 is not subset of t1";
}

TEST(TensorPool, create_allocate_has_data_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  t1 = pool.request("a", {10}, {0}, max_ls);
  t2 = pool.request("b", {10}, {1}, max_ls);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  testNoOverlap(t1, t2);
  pool.deallocate();
}

TEST(TensorPool, create_clashing_name_n) {
  nntrainer::TensorPool pool;
  auto t1 = pool.request("a", {10}, {0}, max_ls);
  EXPECT_NE(t1, nullptr);
  EXPECT_ANY_THROW(pool.request("a", {10}, {1}, max_ls));
}

TEST(TensorPool, placeholder_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("a", {10});
  pool.placeholder("b", {10});
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  EXPECT_NO_THROW(pool.allocate());
}

TEST(TensorPool, placeholder_clashing_name_n) {
  nntrainer::TensorPool pool;
  auto t1 = pool.placeholder("a", {10});
  EXPECT_NE(t1, nullptr);
  EXPECT_ANY_THROW(pool.placeholder("a", {10}));
}

TEST(TensorPool, view_is_same_p) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-------- t2 -------|
  auto t1 = pool.request("t1", {10}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {10}, {1}, max_ls);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getDim(), t2->getDim());
  EXPECT_EQ(t1->getData<float>(), t2->getData<float>());
  pool.deallocate();
}

TEST(TensorPool, view_is_subset_p) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-t2-|
  //       |-t3-|
  auto t1 = pool.request("t1", {10}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {3}, {1}, max_ls);
  auto t3 = pool.view("t3", "t1", {3}, {1}, max_ls, 3);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  testSubset(t1, t2);
  testSubset(t1, t3);
  testNoOverlap(t2, t3);
  pool.deallocate();
}

TEST(TensorPool, view_is_subset_n) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-t2-|
  //       |-t3-|
  auto t1 = pool.request("t1", {10}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t2", "unknown", {3}, {1}, max_ls));
  EXPECT_ANY_THROW(pool.view("t3", "unknown", {3}, {1}, max_ls, 3));
  pool.deallocate();
}

TEST(TensorPool, view_is_view_of_view_and_subset_p) {
  // |-------- t1-------|
  // |-t2-|(offset)
  //               |-t3-|
  nntrainer::TensorPool pool;
  auto t1 = pool.request("t1", {10}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {3}, {1}, max_ls);
  auto t3 = pool.view("t3", "t2", {3}, {1}, max_ls, 3);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  testSubset(t1, t2);
  testSubset(t1, t3);
  testNoOverlap(t2, t3);
  pool.deallocate();
}

TEST(TensorPool, view_is_view_of_view_and_subset_n) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-t2-|
  //       |-t3-|
  auto t1 = pool.request("t1", {10}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {3}, {1}, max_ls);
  EXPECT_ANY_THROW(pool.view("t3", "unknown", {3}, {1}, max_ls, 3));
  pool.deallocate();
}

TEST(TensorPool, view_of_placeholder_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {10}, {0}, max_ls);
  auto t1 = pool.placeholder("t1", {10});
  auto t2 = pool.view("t2", "t1", {10}, {0}, max_ls);
  auto t3 = pool.view("t3", "t1", {2}, {0}, max_ls, 2);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getData<float>(), nullptr);
  EXPECT_EQ(t2->getData<float>(), nullptr);

  /// t_original: 0 1 2 3 4 5 6 7 8 9
  /// t1        : 0 1 2 3 4 5 6 7 8 9
  /// t2        : 0 1 2 3 4 5 6 7 8 9
  /// t3        :     2 3
  nntrainer::Tensor t_original(t1->getDim());
  t_original.apply_i(
    (std::function<float(float)>)[i = 0u](float _) mutable { return ++i; });
  pool.fillPlaceholder("t1", t_original);

  testSubset(t1, &t_original);
  testSubset(t1, t2);
  testSubset(t1, t3);

  EXPECT_EQ(*t1, t_original);
  EXPECT_FLOAT_EQ(t2->getData<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(t3->getData<float>()[0], 3.0f);

  pool.deallocate();
}

TEST(TensorPool, view_clashing_name_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {10}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t0", "t0", {10}, {0}, max_ls));
}

TEST(TensorPool, view_out_of_range_n) {
  // |-------- t0 -------|
  //                     |-t1-|
  nntrainer::TensorPool pool;
  pool.request("t0", {10}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t1", "t0", {1}, {0}, max_ls, 10));
}
TEST(TensorPool, view_of_view_out_of_range_n) {
  nntrainer::TensorPool pool;
  // |-------- t0 -------|
  //                |-t1-|
  //                     |-t2-|
  pool.request("t0", {10}, {0}, max_ls);
  pool.view("t1", "t0", {1}, {0}, max_ls, 9);
  EXPECT_ANY_THROW(pool.view("t2", "t1", {1}, {0}, max_ls, 1));
}

TEST(TensorPool, view_of_placeholder_out_of_range_n) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {10});
  EXPECT_ANY_THROW(pool.view("t1", "t0", {1}, {0}, max_ls, 11));
}

TEST(TensorPool, extend_source_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {10}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.extend("t0", {10}, {1},
              nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
}

TEST(TensorPool, extend_view_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {10}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.view("t1", "t0", {10}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.extend("t1", {10}, {2}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 2),
            exec_order.end());
}

TEST(TensorPool, extend_placeholder_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {10});
  pool.extend("t0", {10}, {2}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_EQ(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 2),
            exec_order.end());
}

TEST(TensorPool, extend_view_of_placeholder_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {10});
  pool.view("t1", "t0", {10}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.extend("t1", {10}, {2}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_EQ(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 2),
            exec_order.end());
}

TEST(TensorPool, extend_out_of_range_n) {
  nntrainer::TensorPool pool;
  EXPECT_ANY_THROW(pool.extend("t1", {10}, {2}, max_ls));
}

TEST(TensorPool, extend_unmanged_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {10}, {0}, nntrainer::TensorLifespan::UNMANAGED);
  EXPECT_ANY_THROW(pool.extend("t1", {10}, {2}, max_ls));
}

TEST(TensorPool, createOrExtend_p) {
  nntrainer::TensorPool pool;
  auto t1 = pool.requestOrExtend("t", {10}, {0}, max_ls);
  auto t2 = pool.requestOrExtend("t", {10}, {1}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t");
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
  EXPECT_EQ(t1, t2);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();
  EXPECT_EQ(*t1, *t2);
  pool.deallocate();
}

TEST(TensorPool, createOrExtend_different_dim_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t", {10, 1}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {1, 10}, {1}, max_ls));
}

TEST(TensorPool, createOrExtend_init_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t", {10}, {0}, max_ls, nntrainer::Initializer::ONES);
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {10}, {1}, max_ls,
                                        nntrainer::Initializer::ZEROS));
}
TEST(TensorPool, createOrExtend_unmanaged_n) {
  nntrainer::TensorPool pool;
  EXPECT_ANY_THROW(
    pool.requestOrExtend("t", {10}, {0}, nntrainer::TensorLifespan::UNMANAGED));
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
