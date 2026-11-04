// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_engine_request.cpp
 * @brief   nntrainer::requestEngine(): a compute-engine request that does not
 *          go through the process environment.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * The request state is process-wide and latches on the first engine read, so
 * the whole sequence is one test.
 */

#include <env_compat.h>

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

TEST(EngineRequestTest, RequestWinsBeforeFirstReadThenLatches) {
  unsetenv("NNTR_ENGINE");

  std::string effective;
  // nothing read the engine yet: the request is taken, in any case
  EXPECT_EQ(nntrainer::requestEngine("CPU", &effective),
            nntrainer::EngineRequestResult::APPLIED);
  EXPECT_EQ(effective, "cpu");

  // a later request before the first read replaces it
  EXPECT_EQ(nntrainer::requestEngine("gpu", &effective),
            nntrainer::EngineRequestResult::APPLIED);
  EXPECT_EQ(nntrainer::requestEngine("cpu", &effective),
            nntrainer::EngineRequestResult::APPLIED);

  // the request answers nntr_engine_env(), ahead of the environment, and the
  // environment is left alone
  setenv("NNTR_ENGINE", "gpu", 1);
  EXPECT_EQ(nntr_engine_env(), "cpu");
  EXPECT_STREQ(std::getenv("NNTR_ENGINE"), "gpu");

  // the first read latched the engine: same engine is a no-op, another is late
  EXPECT_EQ(nntrainer::requestEngine("cpu", &effective),
            nntrainer::EngineRequestResult::UNCHANGED);
  EXPECT_EQ(effective, "cpu");
  EXPECT_EQ(nntrainer::requestEngine("gpu", &effective),
            nntrainer::EngineRequestResult::TOO_LATE);
  EXPECT_EQ(effective, "cpu");
  EXPECT_EQ(nntr_engine_env(), "cpu");
  unsetenv("NNTR_ENGINE");
}
