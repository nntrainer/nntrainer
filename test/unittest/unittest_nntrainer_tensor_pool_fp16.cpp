// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file unittest_nntrainer_tensor_pool_fp16.cpp
 * @date 23 August 2023
 * @brief Mixed Tensor Pool Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <optimized_v1_planner.h>
#include <tensor_pool.h>

#define FP32_                       \
  nntrainer::TensorDim::TensorType( \
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32})
#define FP16_                       \
  nntrainer::TensorDim::TensorType( \
    {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16})

constexpr unsigned int MEM_BYTES = 128;
constexpr unsigned int MEM_QUANT = 100;
constexpr unsigned int INTERVAL_SIZE = 5;
constexpr static auto max_ls = nntrainer::TensorLifespan::MAX_LIFESPAN;

/**
 * @brief request empty fp16 tensor
 */
TEST(TensorPool, request_01_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.request("t0", nntrainer::TensorDim({}, FP16_), {},
                            nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request empty name
 */
TEST(TensorPool, request_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_THROW(pool.request("", nntrainer::TensorDim({1}, FP16_), {},
                            nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request already allocated fp16 tensor
 */
TEST(TensorPool, request_03_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.request("t0", nntrainer::TensorDim({2}, FP16_), {},
                            nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request fp16 tensor
 */
TEST(TensorPool, request_04_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t = nullptr;

  EXPECT_NO_THROW(t = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                   nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t, nullptr);
  EXPECT_FALSE(t->isAllocated());
}

/**
 * @brief request fp16 and fp32 tensors
 */
TEST(TensorPool, request_05_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(t1 = pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(t2 = pool.request("t2", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief request bigger size for view
 */
TEST(TensorPool, view_01_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t1", "t0", nntrainer::TensorDim({2}, FP16_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);

  EXPECT_NO_THROW(pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t2", "t1", nntrainer::TensorDim({2}, FP32_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request view non existing tensor
 */
TEST(TensorPool, view_02_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_ANY_THROW(pool.view("t1", "not_exist",
                             nntrainer::TensorDim({1}, FP16_), {},
                             nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_NO_THROW(pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_ANY_THROW(pool.view("t2", "not_exist",
                             nntrainer::TensorDim({1}, FP32_), {},
                             nntrainer::TensorLifespan::UNMANAGED));
}

/**
 * @brief request with clashing name
 */
TEST(TensorPool, view_03_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t0", "t0", nntrainer::TensorDim({1}, FP16_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);

  EXPECT_NO_THROW(pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_ANY_THROW(pool.view("t1", "t1", nntrainer::TensorDim({1}, FP32_), {},
                             nntrainer::TensorLifespan::UNMANAGED));
}

/**
 * @brief view with empty tensor
 */
TEST(TensorPool, view_04_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t1", "t0", nntrainer::TensorDim({}, FP16_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);

  EXPECT_NO_THROW(pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t2", "t1", nntrainer::TensorDim({}, FP32_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief view with empty name
 */
TEST(TensorPool, view_05_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("", "t0", nntrainer::TensorDim({1}, FP16_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);

  EXPECT_NO_THROW(pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("", "t1", nntrainer::TensorDim({1}, FP32_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request view of managed tensor
 */
TEST(TensorPool, view_06_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::MAX_LIFESPAN));
  EXPECT_THROW(pool.view("t1", "t0", nntrainer::TensorDim({1}, FP16_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
  EXPECT_NO_THROW(pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::MAX_LIFESPAN));
  EXPECT_THROW(pool.view("t2", "t1", nntrainer::TensorDim({1}, FP32_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request different data type for view
 */
TEST(TensorPool, view_07_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t1", "t0", nntrainer::TensorDim({1}, FP32_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request different data type for view
 */
TEST(TensorPool, view_08_n) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.request("t0", nntrainer::TensorDim({1}, FP32_), {},
                               nntrainer::TensorLifespan::UNMANAGED));

  EXPECT_THROW(pool.view("t1", "t0", nntrainer::TensorDim({1}, FP16_), {},
                         nntrainer::TensorLifespan::UNMANAGED),
               std::invalid_argument);
}

/**
 * @brief request fp16 tensor
 */
TEST(TensorPool, view_09_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(t1 = pool.view("t1", "t0", nntrainer::TensorDim({1}, FP16_),
                                 {}, nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NE(t0, t1);
}

/**
 * @brief request try extending lifespan of unmanaged
 */
TEST(TensorPool, view_10_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(
    t1 = pool.view("t1", "t0", nntrainer::TensorDim({1}, FP16_), {}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NE(t0, t1);

  EXPECT_NO_THROW(
    t1 = pool.view("t2", "t0", nntrainer::TensorDim({1}, FP16_), {}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NE(t0, t1);
}

/**
 * @brief request mixed tensor
 */
TEST(TensorPool, view_11_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(t1 = pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(t2 = pool.view("t2", "t0", nntrainer::TensorDim({1}, FP16_),
                                 {}, nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t0, t2);

  EXPECT_NO_THROW(t3 = pool.view("t3", "t1", nntrainer::TensorDim({1}, FP32_),
                                 {}, nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NE(t1, t3);
}

/**
 * @brief request try extending lifespan of unmanaged
 */
TEST(TensorPool, view_12_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(t1 = pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.view("t2", "t0", nntrainer::TensorDim({1}, FP16_), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t0, t2);

  EXPECT_NO_THROW(
    t3 = pool.view("t3", "t1", nntrainer::TensorDim({1}, FP32_), {}, max_ls));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NE(t1, t3);

  EXPECT_NO_THROW(
    t2 = pool.view("t4", "t0", nntrainer::TensorDim({1}, FP16_), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t0, t2);

  EXPECT_NO_THROW(
    t3 = pool.view("t5", "t1", nntrainer::TensorDim({1}, FP32_), {}, max_ls));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NE(t1, t3);
}

/**
 * @brief set batch
 */
TEST(TensorPool, set_batch_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_EQ(t0->batch(), 1u);
  EXPECT_NO_THROW(pool.setBatchSize("t0", 10));
  EXPECT_EQ(t0->batch(), 10u);
}

/**
 * @brief set batch for not exist tensor
 */
TEST(TensorPool, set_batch_02_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_THROW(pool.setBatchSize("not_exist", 10), std::invalid_argument);
  EXPECT_EQ(t0->batch(), 1u);
}

/**
 * @brief set batch for allocated fp16 tensor
 */
TEST(TensorPool, set_batch_03_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr;
  nntrainer::BasicPlanner basic_planner;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t0, nullptr);
  EXPECT_NO_THROW(pool.finalize(basic_planner, 0, 1));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.setBatchSize("t0", 10), std::invalid_argument);
}

/**
 * @brief set batch for mixed tensor pool
 */
TEST(TensorPool, set_batch_04_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;

  EXPECT_NO_THROW(t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(t1 = pool.request("t1", nntrainer::TensorDim({1}, FP32_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_EQ(t0->batch(), 1u);
  EXPECT_NO_THROW(pool.setBatchSize("t0", 10));
  EXPECT_EQ(t0->batch(), 10u);

  EXPECT_EQ(t1->batch(), 1u);
  EXPECT_NO_THROW(pool.setBatchSize("t1", 10));
  EXPECT_EQ(t1->batch(), 10u);
}

/**
 * @brief set batch for allocated tensor
 */
TEST(TensorPool, set_batch_05_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;
  nntrainer::BasicPlanner basic_planner;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({1}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t0, nullptr);
  EXPECT_NE(t1, nullptr);
  EXPECT_NO_THROW(pool.finalize(basic_planner, 0, 1));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.setBatchSize("t0", 10), std::invalid_argument);
  EXPECT_THROW(pool.setBatchSize("t1", 10), std::invalid_argument);
}

/**
 * @brief zero size pool as no usage
 */
TEST(TensorPool, finalize_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("t1", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({1}, FP16_), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), 0u);

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief zero size mixed pool as no usage
 */
TEST(TensorPool, finalize_02_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(t1 = pool.request("t1", nntrainer::TensorDim({1}, FP16_), {},
                                    nntrainer::TensorLifespan::UNMANAGED));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({1}, FP32_), {}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), 0u);

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief max lifespan fp16 tensors
 */
TEST(TensorPool, finalize_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t1, nullptr);

  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({1}, FP16_), {1}, max_ls));
  EXPECT_NE(t2, nullptr);

  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t1, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief max lifespan mixed tensors
 */
TEST(TensorPool, finalize_04_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t1, nullptr);

  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({1}, FP32_), {1}, max_ls));
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
TEST(TensorPool, finalize_05_p) {
  nntrainer::TensorPool pool;

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
}

/**
 * @brief cannot reuse fp32 tensor memory space
 */
TEST(TensorPool, finalize_06_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_NE(pool.minMemoryRequirement(), t1->bytes());
}

/**
 * @brief cannot reuse fp32 tensor memory space
 */
TEST(TensorPool, finalize_07_n) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {1}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1}, max_ls));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_NE(pool.minMemoryRequirement(), t1->bytes());
}

/**
 * @brief reuse fp32 tensor memory space
 */
TEST(TensorPool, finalize_08_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes());
}

/**
 * @brief reuse fp32 tensor memory space
 */
TEST(TensorPool, finalize_09_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes());
}

/**
 * @brief reuse fp32 tensor memory space
 */
TEST(TensorPool, finalize_10_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  // This is the theoretical minimum memory requirement which differs from the
  // actual memory layout. The actual memory required would be 16.
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes());
}

/**
 * @brief allocate
 */
TEST(TensorPool, allocate_deallocate_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({1}, FP32_), {1}, max_ls));
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
 * @brief request already allocated fp16 tensor
 */
TEST(TensorPool, allocate_deallocate_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr, *t2 = nullptr;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(
    t1 = pool.view("t1", "t0", nntrainer::TensorDim({1}, FP16_), {1}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NE(t0, t1);

  EXPECT_NO_THROW(
    t2 = pool.view("t2", "t0", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t0, t2);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_TRUE(t0->isAllocated());
  EXPECT_TRUE(t1->isAllocated());
  EXPECT_TRUE(t2->isAllocated());

  EXPECT_EQ(t0->getData<_FP16>(), t1->getData<_FP16>());
  EXPECT_EQ(t0->getData<_FP16>(), t2->getData<_FP16>());

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(t0->isAllocated());
  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
}

/**
 * @brief request already allocated mixed tensor
 */
TEST(TensorPool, allocate_deallocate_04_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr, *t2 = nullptr, *t3 = nullptr,
                    *t4 = nullptr, *t5 = nullptr;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t0, nullptr);
  EXPECT_FALSE(t0->isAllocated());

  EXPECT_NO_THROW(
    t1 = pool.view("t1", "t0", nntrainer::TensorDim({1}, FP16_), {1}, max_ls));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NE(t0, t1);

  EXPECT_NO_THROW(
    t2 = pool.view("t2", "t0", nntrainer::TensorDim({1}, FP16_), {0}, max_ls));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NE(t0, t2);

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({1}, FP32_), {0}, max_ls));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(
    t4 = pool.view("t4", "t3", nntrainer::TensorDim({1}, FP32_), {1}, max_ls));
  EXPECT_NE(t4, nullptr);
  EXPECT_FALSE(t4->isAllocated());

  EXPECT_NE(t3, t4);

  EXPECT_NO_THROW(
    t5 = pool.view("t5", "t3", nntrainer::TensorDim({1}, FP32_), {0}, max_ls));
  EXPECT_NE(t5, nullptr);
  EXPECT_FALSE(t5->isAllocated());

  EXPECT_NE(t3, t5);

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));

  EXPECT_NO_THROW(pool.allocate());
  EXPECT_TRUE(t0->isAllocated());
  EXPECT_TRUE(t1->isAllocated());
  EXPECT_TRUE(t2->isAllocated());
  EXPECT_TRUE(t3->isAllocated());
  EXPECT_TRUE(t4->isAllocated());
  EXPECT_TRUE(t5->isAllocated());

  EXPECT_EQ(t0->getData<_FP16>(), t1->getData<_FP16>());
  EXPECT_EQ(t0->getData<_FP16>(), t2->getData<_FP16>());
  EXPECT_EQ(t3->getData<float>(), t4->getData<float>());
  EXPECT_EQ(t3->getData<float>(), t5->getData<float>());

  EXPECT_NO_THROW(pool.deallocate());
  EXPECT_FALSE(t0->isAllocated());
  EXPECT_FALSE(t1->isAllocated());
  EXPECT_FALSE(t2->isAllocated());
  EXPECT_FALSE(t3->isAllocated());
  EXPECT_FALSE(t4->isAllocated());
  EXPECT_FALSE(t5->isAllocated());
}

/**
 * @brief validate memory full overlap
 */
TEST(TensorPool, validate_memory_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({100}, FP16_), {0}, max_ls));

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({100}, FP16_), {1}, max_ls));

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_NO_THROW(pool.allocate());

  nntrainer::Tensor g1 = nntrainer::Tensor(nntrainer::TensorDim({100}, FP16_));
  g1.setRandNormal();
  nntrainer::Tensor g2 = nntrainer::Tensor(nntrainer::TensorDim({100}, FP16_));
  g2.setRandNormal();

  t0->copy(g1);
  t1->copy(g2);

  EXPECT_EQ(*t0, g1);
  EXPECT_EQ(*t1, g2);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief validate memory full overlap
 */
TEST(TensorPool, validate_memory_02_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({100}, FP16_), {0}, max_ls));

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({100}, FP32_), {1}, max_ls));

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_NO_THROW(pool.allocate());

  nntrainer::Tensor g1 = nntrainer::Tensor(nntrainer::TensorDim({100}, FP16_));
  g1.setRandNormal();
  nntrainer::Tensor g2 = nntrainer::Tensor(nntrainer::TensorDim({100}, FP32_));
  g2.setRandNormal();

  t0->copy(g1);
  t1->copy(g2);

  EXPECT_EQ(*t0, g1);
  EXPECT_EQ(*t1, g2);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief validate memory full overlap in fsu
 */
TEST(TensorPool, validate_memory_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t0 = nullptr, *t1 = nullptr;

  EXPECT_NO_THROW(
    t0 = pool.request("t0", nntrainer::TensorDim({100}, FP16_), {0}, max_ls));

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({100}, FP32_), {1}, max_ls));

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_NO_THROW(pool.allocate());

  nntrainer::Tensor g1 = nntrainer::Tensor(nntrainer::TensorDim({100}, FP32_));
  g1.setRandNormal();
  nntrainer::Tensor g2 = nntrainer::Tensor(nntrainer::TensorDim({100}, FP16_));
  g2.setRandNormal();

  t0->copy(g1);
  t1->copy(g2);

  EXPECT_EQ(*t0, g1);
  EXPECT_EQ(*t1, g2);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp16 tensors cannot reuse fp32 tensor memory space
 */
TEST(TensorPool, validate_memory_reuse_01_n) {
  // |--------- t1 ---------|
  //                        |--- t2 ---||--- t3 ---|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::BasicPlanner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  EXPECT_NE(t2, t3);

  // BasicPlanner
  EXPECT_NE(t1->getAddress<float>(0), (float *)t2->getAddress<_FP16>(0));
  EXPECT_NE(t1->getAddress<float>(1), (float *)t3->getAddress<_FP16>(0));

  EXPECT_EQ(t1->getAddress<float>(0) + t1->size(),
            (float *)t2->getAddress<_FP16>(0));
  EXPECT_EQ(t1->getAddress<float>(1) + t1->size(),
            (float *)t3->getAddress<_FP16>(0));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp16 tensors reuse fp32 tensor memory space
 */
TEST(TensorPool, validate_memory_reuse_02_p) {
  // |--------- t1 ---------|
  // |--- t2 ---||--- t3 ---|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  EXPECT_NE(t2, t3);

  EXPECT_EQ(t1->getAddress<float>(0), (float *)t2->getAddress<_FP16>(0));
  EXPECT_EQ(t1->getAddress<float>(1), (float *)t3->getAddress<_FP16>(0));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp32 tensor reuses fp16 tensors memory spaces
 */
TEST(TensorPool, validate_memory_reuse_03_p) {
  // |--- t1 ---||--- t2 ---|
  // |--------- t3 ---------|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  EXPECT_NE(t2, t3);

  EXPECT_EQ((float *)t1->getAddress<_FP16>(0), t3->getAddress<float>(0));
  EXPECT_EQ((float *)t2->getAddress<_FP16>(0), t3->getAddress<float>(1));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp32 and fp16 tensors reuse fp16 and fp32 tensor memory space
 */
TEST(TensorPool, validate_memory_reuse_04_p) {
  // |--------- t1 ---------||--- t2 ---|
  // |--- t3 ---||--------- t4 ---------|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(
    t4 = pool.request("t4", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t4, nullptr);
  EXPECT_FALSE(t4->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_EQ(t1->getAddress<float>(0), (float *)t3->getAddress<_FP16>(0));
  EXPECT_EQ(t1->getAddress<float>(1), t4->getAddress<float>(0));
  EXPECT_EQ((float *)t2->getAddress<_FP16>(0), t4->getAddress<float>(1));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp16 and fp32 tensors reuse fp32 and fp16 tensor memory space
 */
TEST(TensorPool, validate_memory_reuse_05_p) {
  // |--- t1 ---||--------- t2 ---------|
  // |--------- t3 ---------||--- t4 ---|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(
    t4 = pool.request("t4", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t4, nullptr);
  EXPECT_FALSE(t4->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_EQ((float *)t1->getAddress<_FP16>(0), t3->getAddress<float>(0));
  EXPECT_EQ(t2->getAddress<float>(0), t3->getAddress<float>(1));
  EXPECT_EQ((float *)t4->getAddress<_FP16>(0), t2->getAddress<float>(1));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp16 and fp32 tensors reuse fp32 tensor memory space
 */
TEST(TensorPool, validate_memory_reuse_06_p) {
  // |--------- t1 ---------||--------- t2 ---------|
  // |--- t3 ---||--------- t4 ---------||--- t5 ---|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr,
                    *t5 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(
    t4 = pool.request("t4", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t4, nullptr);
  EXPECT_FALSE(t4->isAllocated());

  EXPECT_NO_THROW(
    t5 = pool.request("t5", nntrainer::TensorDim({2}, FP16_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t5, nullptr);
  EXPECT_FALSE(t5->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(), t1->bytes() + t2->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_EQ(t1->getAddress<float>(0), (float *)t3->getAddress<_FP16>(0));
  EXPECT_EQ(t1->getAddress<float>(1), t4->getAddress<float>(0));
  EXPECT_EQ(t2->getAddress<float>(1), (float *)t5->getAddress<_FP16>(0));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief fp32 tensors reuse fp16 and fp32 tensors memory spaces
 */
TEST(TensorPool, validate_memory_reuse_07_p) {
  // |--- t1 ---||--------- t2 ---------||--- t3 ---|
  // |--------- t4 ---------||--------- t5 ---------|
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr, *t3 = nullptr, *t4 = nullptr,
                    *t5 = nullptr;

  EXPECT_NO_THROW(
    t1 = pool.request("t1", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t1, nullptr);
  EXPECT_FALSE(t1->isAllocated());

  EXPECT_NO_THROW(
    t2 = pool.request("t2", nntrainer::TensorDim({2}, FP32_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t2, nullptr);
  EXPECT_FALSE(t2->isAllocated());

  EXPECT_NO_THROW(
    t3 = pool.request("t3", nntrainer::TensorDim({2}, FP16_), {0},
                      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
  EXPECT_NE(t3, nullptr);
  EXPECT_FALSE(t3->isAllocated());

  EXPECT_NO_THROW(
    t4 = pool.request("t4", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t4, nullptr);
  EXPECT_FALSE(t4->isAllocated());

  EXPECT_NO_THROW(
    t5 = pool.request("t5", nntrainer::TensorDim({2}, FP32_), {1},
                      nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN));
  EXPECT_NE(t5, nullptr);
  EXPECT_FALSE(t5->isAllocated());

  EXPECT_NO_THROW(pool.finalize(nntrainer::OptimizedV1Planner(), 0, 2));
  EXPECT_EQ(pool.minMemoryRequirement(),
            t1->bytes() + t2->bytes() + t3->bytes());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_EQ((float *)t1->getAddress<_FP16>(0), t4->getAddress<float>(0));
  EXPECT_EQ((float *)t2->getAddress<_FP16>(0), t4->getAddress<float>(1));
  EXPECT_EQ(t2->getAddress<float>(1), t5->getAddress<float>(0));
  EXPECT_EQ((float *)t3->getAddress<_FP16>(0), t5->getAddress<float>(1));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief check if data span of two tensor testOverlap
 *
 * @param t1 tensor1
 * @param t2 tensor2
 */
static void testNoOverlap(nntrainer::Tensor *t1, nntrainer::Tensor *t2) {
  _FP16 *t1_start = t1->getData<_FP16>();
  _FP16 *t1_end = t1_start + t1->size();

  _FP16 *t2_start = t2->getData<_FP16>();
  _FP16 *t2_end = t2_start + t2->size();

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
  if (t1->getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *t1_start = t1->getData<float>();
    float *t1_end = t1_start + t1->size();

    float *t2_start = t2->getData<float>();
    float *t2_end = t2_start + t2->size();

    EXPECT_NE(t1_start, nullptr);
    EXPECT_NE(t2_start, nullptr);
    EXPECT_TRUE(t1_start <= t2_start && t2_end <= t1_end)
      << "t2 is not subset of t1";
  } else {
    _FP16 *t1_start = t1->getData<_FP16>();
    _FP16 *t1_end = t1_start + t1->size();

    _FP16 *t2_start = t2->getData<_FP16>();
    _FP16 *t2_end = t2_start + t2->size();

    EXPECT_NE(t1_start, nullptr);
    EXPECT_NE(t2_start, nullptr);
    EXPECT_TRUE(t1_start <= t2_start && t2_end <= t1_end)
      << "t2 is not subset of t1";
  }
}

TEST(TensorPool, create_allocate_has_data_01_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  t1 = pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  t2 = pool.request("t1", {{10}, FP16_}, {1}, max_ls);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  testNoOverlap(t1, t2);
  pool.deallocate();
}

TEST(TensorPool, create_allocate_has_data_02_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  t1 = pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  t2 = pool.request("t1", {{10}, FP32_}, {1}, max_ls);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  testNoOverlap(t1, t2);
  pool.deallocate();
}

TEST(TensorPool, create_allocate_has_data_03_p) {
  nntrainer::TensorPool pool;
  nntrainer::Tensor *t1 = nullptr, *t2 = nullptr;

  t1 = pool.request("t0", {{10}, FP32_}, {0}, max_ls);
  t2 = pool.request("t1", {{10}, FP16_}, {1}, max_ls);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  testNoOverlap(t1, t2);
  pool.deallocate();
}

TEST(TensorPool, create_clashing_name_n) {
  nntrainer::TensorPool pool;
  auto t1 = pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  EXPECT_NE(t1, nullptr);
  EXPECT_ANY_THROW(pool.request("t0", {{10}, FP16_}, {1}, max_ls));
}

TEST(TensorPool, placeholder_01_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.placeholder("t1", {{10}, FP16_});
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  EXPECT_NO_THROW(pool.allocate());
}

TEST(TensorPool, placeholder_02_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP32_});
  pool.placeholder("t1", {{10}, FP16_});
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  EXPECT_NO_THROW(pool.allocate());
}

TEST(TensorPool, placeholder_03_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.placeholder("t1", {{10}, FP32_});
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  EXPECT_NO_THROW(pool.allocate());
}

TEST(TensorPool, placeholder_clashing_name_n) {
  nntrainer::TensorPool pool;
  auto t1 = pool.placeholder("t1", {{10}, FP16_});
  auto t2 = pool.placeholder("t2", {{10}, FP32_});
  EXPECT_NE(t1, nullptr);
  EXPECT_NE(t2, nullptr);
  EXPECT_ANY_THROW(pool.placeholder("t1", {{10}, FP16_}));
  EXPECT_ANY_THROW(pool.placeholder("t2", {{10}, FP32_}));
}

TEST(TensorPool, placeholder_no_name_n) {
  nntrainer::TensorPool pool;
  auto t1 = pool.placeholder("t0", {{10}, FP16_});
  auto t2 = pool.placeholder("t1", {{10}, FP32_});
  EXPECT_NE(t1, nullptr);
  EXPECT_NE(t2, nullptr);
  EXPECT_ANY_THROW(pool.placeholder("", {{10}, FP16_}));
  EXPECT_ANY_THROW(pool.placeholder("", {{10}, FP32_}));
}

TEST(TensorPool, placeholder_empty_n) {
  nntrainer::TensorPool pool;
  auto t1 = pool.placeholder("t0", {{10}, FP16_});
  auto t2 = pool.placeholder("t1", {{10}, FP32_});
  EXPECT_NE(t1, nullptr);
  EXPECT_NE(t2, nullptr);
  EXPECT_ANY_THROW(pool.placeholder("t2", {{}, FP16_}));
  EXPECT_ANY_THROW(pool.placeholder("t3", {{}, FP32_}));
}

TEST(TensorPool, view_is_same_01_p) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-------- t2 -------|
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{10}, FP16_}, {1}, max_ls);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getDim(), t2->getDim());
  EXPECT_EQ(t1->getData<_FP16>(), t2->getData<_FP16>());
  pool.deallocate();
}

TEST(TensorPool, view_is_same_02_p) {
  nntrainer::TensorPool pool;
  // |---- t1 ----| |-------- t3 --------|
  // |---- t2 ----| |-------- t4 --------|
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{10}, FP16_}, {1}, max_ls);
  auto t3 = pool.request("t3", {{10}, FP32_}, {0}, max_ls);
  auto t4 = pool.view("t4", "t3", {{10}, FP32_}, {1}, max_ls);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getDim(), t2->getDim());
  EXPECT_EQ(t1->getData<_FP16>(), t2->getData<_FP16>());

  EXPECT_NE(t3, t4);
  EXPECT_EQ(t3->getDim(), t4->getDim());
  EXPECT_EQ(t3->getData<float>(), t4->getData<float>());

  testSubset(t1, t2);
  testSubset(t3, t4);
  testNoOverlap(t1, t3);
  testNoOverlap(t2, t4);

  pool.deallocate();
}

TEST(TensorPool, view_is_same_03_p) {
  nntrainer::TensorPool pool;
  // |-------- t1 --------| |---- t3 ----|
  // |-------- t2 --------| |---- t4 ----|
  auto t1 = pool.request("t1", {{10}, FP32_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{10}, FP32_}, {1}, max_ls);
  auto t3 = pool.request("t3", {{10}, FP16_}, {0}, max_ls);
  auto t4 = pool.view("t4", "t3", {{10}, FP16_}, {1}, max_ls);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getDim(), t2->getDim());
  EXPECT_EQ(t1->getData<float>(), t2->getData<float>());

  EXPECT_NE(t3, t4);
  EXPECT_EQ(t3->getDim(), t4->getDim());
  EXPECT_EQ(t3->getData<_FP16>(), t4->getData<_FP16>());

  testSubset(t1, t2);
  testSubset(t3, t4);
  testNoOverlap(t1, t3);
  testNoOverlap(t2, t4);

  pool.deallocate();
}

TEST(TensorPool, view_is_subset_01_p) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-t2-|
  //       |-t3-|
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{3}, FP16_}, {1}, max_ls);
  auto t3 = pool.view("t3", "t1", {{3}, FP16_}, {1}, max_ls, 3);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  testSubset(t1, t2);
  testSubset(t1, t3);
  testNoOverlap(t2, t3);
  pool.deallocate();
}

TEST(TensorPool, view_is_subset_02_p) {
  nntrainer::TensorPool pool;
  // |------ t1 ------|  |-------- t4 -------|
  // |-t2-|              |-t5-|
  //       |-t3-|              |-t6-|
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{3}, FP16_}, {1}, max_ls);
  auto t3 = pool.view("t3", "t1", {{3}, FP16_}, {1}, max_ls, 3);
  auto t4 = pool.request("t4", {{10}, FP32_}, {0}, max_ls);
  auto t5 = pool.view("t5", "t4", {{3}, FP32_}, {1}, max_ls);
  auto t6 = pool.view("t6", "t4", {{3}, FP32_}, {1}, max_ls, 3);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  testSubset(t1, t2);
  testSubset(t1, t3);
  testNoOverlap(t2, t3);

  EXPECT_NE(t4, t5);
  EXPECT_NE(t4, t6);
  testSubset(t4, t5);
  testSubset(t4, t6);
  testNoOverlap(t5, t6);

  testNoOverlap(t1, t4);
  testNoOverlap(t1, t5);
  testNoOverlap(t3, t4);

  pool.deallocate();
}

TEST(TensorPool, view_is_subset_n) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-t2-|
  //       |-t3-|
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t2", "unknown", {{3}, FP16_}, {1}, max_ls));
  EXPECT_ANY_THROW(pool.view("t3", "unknown", {{3}, FP16_}, {1}, max_ls, 3));
  pool.deallocate();
}

TEST(TensorPool, view_is_view_of_view_and_subset_01_p) {
  // |-------- t1-------|
  // |-t2-|(offset)
  //               |-t3-|
  nntrainer::TensorPool pool;
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{3}, FP16_}, {1}, max_ls);
  auto t3 = pool.view("t3", "t2", {{3}, FP16_}, {1}, max_ls, 3);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  testSubset(t1, t2);
  testSubset(t1, t3);
  testNoOverlap(t2, t3);
  pool.deallocate();
}

TEST(TensorPool, view_is_view_of_view_and_subset_02_p) {
  // |-------- t1 -------|  |-------- t4 -------|
  // |-t2-|( offset )       |-t5-|(offset)
  //                |-t3-|                 |-t6-|
  nntrainer::TensorPool pool;
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{3}, FP16_}, {1}, max_ls);
  auto t3 = pool.view("t3", "t2", {{3}, FP16_}, {1}, max_ls, 4);
  auto t4 = pool.request("t4", {{10}, FP32_}, {0}, max_ls);
  auto t5 = pool.view("t5", "t4", {{3}, FP32_}, {1}, max_ls);
  auto t6 = pool.view("t6", "t4", {{3}, FP32_}, {1}, max_ls, 4);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_NE(t1, t3);
  testSubset(t1, t2);
  testSubset(t1, t3);
  testNoOverlap(t2, t3);

  EXPECT_NE(t4, t5);
  EXPECT_NE(t4, t6);
  testSubset(t4, t5);
  testSubset(t4, t6);
  testNoOverlap(t5, t6);

  testNoOverlap(t1, t4);
  testNoOverlap(t1, t5);
  testNoOverlap(t3, t4);

  pool.deallocate();
}

TEST(TensorPool, view_is_view_of_view_and_subset_03_n) {
  nntrainer::TensorPool pool;
  // |-------- t1 -------|
  // |-t2-|
  //       |-t3-|
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{3}, FP16_}, {1}, max_ls);
  EXPECT_ANY_THROW(pool.view("t3", "unknown", {{3}, FP16_}, {1}, max_ls, 3));
  pool.deallocate();
}

TEST(TensorPool, view_is_view_of_view_and_subset_04_n) {
  // |-------- t1 -------|  |-------- t4 -------|
  // |-t2-|( offset )       |-t5-|(offset)
  //                |-t3-|                 |-t6-|
  nntrainer::TensorPool pool;
  auto t1 = pool.request("t1", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.view("t2", "t1", {{3}, FP16_}, {1}, max_ls);
  EXPECT_ANY_THROW(pool.view("t3", "unknown", {{3}, FP16_}, {1}, max_ls, 4));
  auto t4 = pool.request("t4", {{10}, FP32_}, {0}, max_ls);
  auto t5 = pool.view("t5", "t4", {{3}, FP32_}, {1}, max_ls);
  EXPECT_ANY_THROW(pool.view("t6", "unknown", {{3}, FP32_}, {1}, max_ls, 4));

  pool.deallocate();
}

TEST(TensorPool, view_of_placeholder_01_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  auto t1 = pool.placeholder("t1", {{10}, FP16_});
  auto t2 = pool.view("t2", "t1", {{10}, FP16_}, {0}, max_ls);
  auto t3 = pool.view("t3", "t1", {{2}, FP16_}, {0}, max_ls, 2);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getData<_FP16>(), nullptr);
  EXPECT_EQ(t2->getData<_FP16>(), nullptr);

  /// t_original: 1 2 3 4 5 6 7 8 9 10
  /// t1        : 1 2 3 4 5 6 7 8 9 10
  /// t2        : 1 2 3 4 5 6 7 8 9 10
  /// t3        :     3 4
  nntrainer::Tensor t_original(t1->getDim());
  t_original.apply_i(
    (std::function<_FP16(_FP16)>)[i = 0u](_FP16 _) mutable { return ++i; });
  pool.fillPlaceholder("t1", t_original);

  testSubset(t1, &t_original);
  testSubset(t1, t2);
  testSubset(t1, t3);

  EXPECT_EQ(*t1, t_original);
  EXPECT_FLOAT_EQ(t2->getData<_FP16>()[0], 1.0f);
  EXPECT_FLOAT_EQ(t3->getData<_FP16>()[0], 3.0f);

  pool.deallocate();
}

TEST(TensorPool, view_of_placeholder_02_p) {
  nntrainer::TensorPool pool;
  pool.request("tensor_fp16", {{10}, FP16_}, {0}, max_ls);
  auto t1 = pool.placeholder("t1", {{10}, FP16_});
  auto t2 = pool.view("t2", "t1", {{10}, FP16_}, {0}, max_ls);
  auto t3 = pool.view("t3", "t1", {{2}, FP16_}, {0}, max_ls, 2);

  pool.request("tensor_fp32", {{10}, FP32_}, {0}, max_ls);
  auto t4 = pool.placeholder("t4", {{10}, FP32_});
  auto t5 = pool.view("t5", "t4", {{10}, FP32_}, {0}, max_ls);
  auto t6 = pool.view("t6", "t4", {{2}, FP32_}, {0}, max_ls, 3);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_NE(t1, t2);
  EXPECT_EQ(t1->getData<_FP16>(), nullptr);
  EXPECT_EQ(t2->getData<_FP16>(), nullptr);
  EXPECT_EQ(t3->getData<_FP16>(), nullptr);

  EXPECT_NE(t4, t5);
  EXPECT_EQ(t4->getData<float>(), nullptr);
  EXPECT_EQ(t5->getData<float>(), nullptr);
  EXPECT_EQ(t6->getData<float>(), nullptr);

  /// t_original: 1 2 3 4 5 6 7 8 9 10
  /// t1        : 1 2 3 4 5 6 7 8 9 10
  /// t2        : 1 2 3 4 5 6 7 8 9 10
  /// t3        :     3 4
  /// t4        : 1 2 3 4 5 6 7 8 9 10
  /// t5        : 1 2 3 4 5 6 7 8 9 10
  /// t6        :       4 5
  nntrainer::Tensor t_original_fp16(t1->getDim());
  t_original_fp16.apply_i(
    (std::function<_FP16(_FP16)>)[i = 0u](_FP16 _) mutable { return ++i; });
  pool.fillPlaceholder("t1", t_original_fp16);

  nntrainer::Tensor t_original_fp32(t4->getDim());
  t_original_fp32.apply_i(
    (std::function<float(float)>)[i = 0u](float _) mutable { return ++i; });
  pool.fillPlaceholder("t4", t_original_fp32);

  testSubset(t1, &t_original_fp16);
  testSubset(t1, t2);
  testSubset(t1, t3);

  testSubset(t4, &t_original_fp32);
  testSubset(t4, t5);
  testSubset(t4, t6);

  EXPECT_EQ(*t1, t_original_fp16);
  EXPECT_FLOAT_EQ(t2->getData<_FP16>()[0], 1.0f);
  EXPECT_FLOAT_EQ(t3->getData<_FP16>()[0], 3.0f);
  EXPECT_FLOAT_EQ(t3->getData<_FP16>()[1], 4.0f);

  EXPECT_EQ(*t4, t_original_fp32);
  EXPECT_FLOAT_EQ(t5->getData<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(t6->getData<float>()[0], 4.0f);
  EXPECT_FLOAT_EQ(t6->getData<float>()[1], 5.0f);

  pool.deallocate();
}

TEST(TensorPool, view_clashing_name_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t0", "t0", {{10}, FP16_}, {0}, max_ls));
  pool.request("t1", {{10}, FP32_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t1", "t1", {{10}, FP32_}, {0}, max_ls));
}

TEST(TensorPool, view_out_of_range_01_n) {
  // |-------- t0 -------|      |-------- t1 -------|
  //                     |-t2-|                     |-t3-|
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  pool.request("t1", {{10}, FP32_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t2", "t0", {{1}, FP16_}, {0}, max_ls, 10));
  EXPECT_ANY_THROW(pool.view("t2", "t1", {{1}, FP32_}, {0}, max_ls, 10));
}

TEST(TensorPool, view_out_of_range_02_n) {
  // |-------- t0 -------|     |-------- t1 -------|
  //                  |--t2--|                  |--t3--|
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  pool.request("t1", {{10}, FP32_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.view("t2", "t0", {{2}, FP16_}, {0}, max_ls, 9));
  EXPECT_ANY_THROW(pool.view("t3", "t1", {{2}, FP32_}, {0}, max_ls, 9));
}

TEST(TensorPool, view_of_view_out_of_range_01_n) {
  // |-------- t0 -------|        |-------- t1 -------|
  //                |-t2-|                       |-t3-|
  //                     |-t4-|                       |-t5-|
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  pool.request("t1", {{10}, FP32_}, {0}, max_ls);
  pool.view("t2", "t0", {{1}, FP16_}, {0}, max_ls, 9);
  pool.view("t3", "t1", {{1}, FP32_}, {0}, max_ls, 9);
  EXPECT_ANY_THROW(pool.view("t4", "t2", {{1}, FP16_}, {0}, max_ls, 1));
  EXPECT_ANY_THROW(pool.view("t5", "t3", {{1}, FP32_}, {0}, max_ls, 1));
}

TEST(TensorPool, view_of_view_out_of_range_02_n) {
  nntrainer::TensorPool pool;
  // |-------- t0 -------|      |-------- t1 -------|
  //             |-t2-|                     |-t3-|
  //                  |--t4--|                   |--t5--|
  pool.request("t0", {{10}, FP16_}, {0}, max_ls);
  pool.request("t1", {{10}, FP32_}, {0}, max_ls);
  pool.view("t2", "t0", {{1}, FP16_}, {0}, max_ls, 8);
  pool.view("t3", "t1", {{1}, FP32_}, {0}, max_ls, 8);
  EXPECT_ANY_THROW(pool.view("t4", "t2", {{2}, FP16_}, {0}, max_ls, 1));
  EXPECT_ANY_THROW(pool.view("t5", "t3", {{2}, FP32_}, {0}, max_ls, 1));
}

TEST(TensorPool, view_of_placeholder_out_of_range_n) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.placeholder("t1", {{10}, FP32_});
  EXPECT_ANY_THROW(pool.view("t2", "t0", {{1}, FP16_}, {0}, max_ls, 10));
  EXPECT_ANY_THROW(pool.view("t2", "t1", {{1}, FP32_}, {0}, max_ls, 10));
}

TEST(TensorPool, extend_source_01_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.extend("t0", {{10}, FP16_}, {1},
              nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
}

TEST(TensorPool, extend_source_02_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.request("t1", {{10}, FP32_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.extend("t0", {{10}, FP16_}, {1},
              nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.extend("t1", {{10}, FP32_}, {1},
              nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);

  auto &exec_order_fp16 = pool.getExecutionOrder("t0");
  auto &exec_order_fp32 = pool.getExecutionOrder("t1");
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 0),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 1),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 0),
            exec_order_fp32.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 1),
            exec_order_fp32.end());
}

TEST(TensorPool, extend_source_03_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  EXPECT_ANY_THROW(
    pool.extend("t0", {{10}, FP32_}, {1},
                nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
}

TEST(TensorPool, extend_source_04_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP32_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  EXPECT_ANY_THROW(
    pool.extend("t0", {{10}, FP16_}, {1},
                nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN));
}

TEST(TensorPool, extend_view_01_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.view("t1", "t0", {{10}, FP16_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.extend("t1", {{10}, FP16_}, {2}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 2),
            exec_order.end());
}

TEST(TensorPool, extend_view_02_p) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.request("t1", {{10}, FP32_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.view("t2", "t0", {{10}, FP16_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.view("t3", "t1", {{10}, FP32_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.extend("t2", {{10}, FP16_}, {2}, max_ls);
  pool.extend("t3", {{10}, FP32_}, {2}, max_ls);

  auto &exec_order_fp16 = pool.getExecutionOrder("t0");
  auto &exec_order_fp32 = pool.getExecutionOrder("t1");
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 0),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 1),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 0),
            exec_order_fp32.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 1),
            exec_order_fp32.end());
}

TEST(TensorPool, extend_view_03_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.view("t1", "t0", {{10}, FP16_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  EXPECT_ANY_THROW(pool.extend("t1", {{10}, FP32_}, {2}, max_ls));
}

TEST(TensorPool, extend_view_04_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP32_}, {0},
               nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  pool.view("t1", "t0", {{10}, FP32_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  EXPECT_ANY_THROW(pool.extend("t1", {{10}, FP16_}, {2}, max_ls));
}

TEST(TensorPool, extend_placeholder_01_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.extend("t0", {{10}, FP16_}, {2}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_EQ(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 2),
            exec_order.end());
}

TEST(TensorPool, extend_placeholder_02_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.placeholder("t1", {{10}, FP32_});
  pool.extend("t0", {{10}, FP16_}, {2}, max_ls);
  pool.extend("t1", {{10}, FP32_}, {2}, max_ls);

  auto &exec_order_fp16 = pool.getExecutionOrder("t0");
  auto &exec_order_fp32 = pool.getExecutionOrder("t1");
  EXPECT_EQ(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 0),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 2),
            exec_order_fp16.end());
  EXPECT_EQ(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 0),
            exec_order_fp32.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 2),
            exec_order_fp32.end());
}

TEST(TensorPool, extend_view_of_placeholder_01_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.view("t1", "t0", {{10}, FP16_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.extend("t1", {{10}, FP16_}, {2}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t0");
  EXPECT_EQ(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 2),
            exec_order.end());
}

TEST(TensorPool, extend_view_of_placeholder_02_p) {
  nntrainer::TensorPool pool;
  pool.placeholder("t0", {{10}, FP16_});
  pool.placeholder("t1", {{10}, FP32_});
  pool.view("t2", "t0", {{10}, FP16_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.view("t3", "t1", {{10}, FP32_}, {1},
            nntrainer::TensorLifespan::BACKWARD_FUNC_LIFESPAN);
  pool.extend("t2", {{10}, FP16_}, {2}, max_ls);
  pool.extend("t3", {{10}, FP32_}, {2}, max_ls);

  auto &exec_order_fp16 = pool.getExecutionOrder("t0");
  auto &exec_order_fp32 = pool.getExecutionOrder("t1");
  EXPECT_EQ(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 0),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 1),
            exec_order_fp16.end());
  EXPECT_NE(std::find(exec_order_fp16.begin(), exec_order_fp16.end(), 2),
            exec_order_fp16.end());
  EXPECT_EQ(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 0),
            exec_order_fp32.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 1),
            exec_order_fp32.end());
  EXPECT_NE(std::find(exec_order_fp32.begin(), exec_order_fp32.end(), 2),
            exec_order_fp32.end());
}

TEST(TensorPool, extend_out_of_range_n) {
  nntrainer::TensorPool pool;
  EXPECT_ANY_THROW(pool.extend("t1", {{10}, FP16_}, {2}, max_ls));
  EXPECT_ANY_THROW(pool.extend("t1", {{10}, FP32_}, {2}, max_ls));
}

TEST(TensorPool, extend_unmanged_01_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, nntrainer::TensorLifespan::UNMANAGED);
  EXPECT_ANY_THROW(pool.extend("t1", {{10}, FP16_}, {2}, max_ls));
}

TEST(TensorPool, extend_unmanged_02_n) {
  nntrainer::TensorPool pool;
  pool.request("t0", {{10}, FP16_}, {0}, nntrainer::TensorLifespan::UNMANAGED);
  pool.request("t1", {{10}, FP32_}, {0}, nntrainer::TensorLifespan::UNMANAGED);
  EXPECT_ANY_THROW(pool.extend("t2", {{10}, FP16_}, {2}, max_ls));
  EXPECT_ANY_THROW(pool.extend("t2", {{10}, FP32_}, {2}, max_ls));
}

TEST(TensorPool, createOrExtend_p) {
  nntrainer::TensorPool pool;
  auto t1 = pool.requestOrExtend("t", {{10}, FP16_}, {0}, max_ls);
  auto t2 = pool.requestOrExtend("t", {{10}, FP16_}, {1}, max_ls);

  auto &exec_order = pool.getExecutionOrder("t");
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 0),
            exec_order.end());
  EXPECT_NE(std::find(exec_order.begin(), exec_order.end(), 1),
            exec_order.end());
  EXPECT_EQ(t1, t2);
  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();
  /// t1 and t2 alias the same pool memory, which pool.allocate() leaves
  /// uninitialized. Give it a deterministic, non-NaN value before comparing
  /// contents: HalfTensor::operator== treats NaN as always-unequal (even to
  /// itself), so comparing raw uninitialized memory is flaky depending on
  /// what garbage bytes happen to land in the fp16 exponent field.
  t1->setZero();
  EXPECT_EQ(*t1, *t2);
  pool.deallocate();
}

TEST(TensorPool, createOrExtend_different_dim_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t", {{10, 1}, FP16_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {{1, 10}, FP16_}, {1}, max_ls));
}

TEST(TensorPool, createOrExtend_different_type_01_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t", {{10}, FP16_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {{10}, FP32_}, {1}, max_ls));
}

TEST(TensorPool, createOrExtend_different_type_02_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t", {{10}, FP32_}, {0}, max_ls);
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {{10}, FP16_}, {1}, max_ls));
}

TEST(TensorPool, createOrExtend_init_01_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t", {{10}, FP16_}, {0}, max_ls,
                       nntrainer::Initializer::ONES);
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {{10}, FP16_}, {1}, max_ls,
                                        nntrainer::Initializer::ZEROS));
}

TEST(TensorPool, createOrExtend_init_02_n) {
  nntrainer::TensorPool pool;
  pool.requestOrExtend("t0", {{10}, FP16_}, {0}, max_ls,
                       nntrainer::Initializer::ONES);
  EXPECT_ANY_THROW(pool.requestOrExtend("t0", {{10}, FP16_}, {1}, max_ls,
                                        nntrainer::Initializer::ZEROS));
  pool.requestOrExtend("t1", {{10}, FP32_}, {0}, max_ls,
                       nntrainer::Initializer::ONES);
  EXPECT_ANY_THROW(pool.requestOrExtend("t1", {{10}, FP32_}, {1}, max_ls,
                                        nntrainer::Initializer::ZEROS));
}

TEST(TensorPool, createOrExtend_unmanaged_01_n) {
  nntrainer::TensorPool pool;
  EXPECT_ANY_THROW(pool.requestOrExtend("t", {{10}, FP16_}, {0},
                                        nntrainer::TensorLifespan::UNMANAGED));
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
