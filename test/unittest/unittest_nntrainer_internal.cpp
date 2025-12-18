/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_nntrainer_internal.cpp
 * @date        10 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <fstream>

#include <neuralnet.h>
#include <nntrainer_error.h>
#include <optimizer.h>
#include <util_func.h>

#include <nntrainer_test_util.h>

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_01_p) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_NO_THROW(op = ac->createOptimizerObject("adam", {}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_02_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("adam", {"unknown"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_03_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("adam", {"lr=0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_04_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op =
                     ac->createOptimizerObject("adam", {"learning_rate:0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_05_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_NO_THROW(op = ac->createOptimizerObject("sgd", {}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_06_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("sgd", {"lr=0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_07_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  // auto &ac = nntrainer::AppContext::Global();
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op =
                     ac->createOptimizerObject("sgd", {"learning_rate:0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_08_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("sgd", {"unknown"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_09_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("non-existing type", {}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_adamw_01_p) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_NO_THROW(op =
                    ac->createOptimizerObject("adamw", {"weight_decay=0.01"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_adamw_02_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("adamw", {"unknown"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_adamw_03_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op =
                     ac->createOptimizerObject("adamw", {"learning_rate:0.1"}));
}

TEST(nntrainer_throw_if, throw_invalid_arg_p) {
  try {
    NNTR_THROW_IF(1 == 1, std::invalid_argument) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
  }

  try {
    NNTR_THROW_IF(true, std::invalid_argument) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
  }

  bool hit = false;
  auto cleanup = [&hit] { hit = true; };
  try {
    NNTR_THROW_IF_CLEANUP(true, std::invalid_argument, cleanup) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
    EXPECT_TRUE(hit);
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
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
