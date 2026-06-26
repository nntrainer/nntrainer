// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_cancel_api.cpp
 * @brief   Focused tests for CausalLM cancellation API contracts.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "causal_lm.h"
#include "causal_lm_api.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>

namespace causal_lm_api_test {
using ActiveRunPublishHook = void (*)(void *);
using BeforeCancelRequestHook = void (*)(void *);
void setAfterActiveRunPublishHookForTest(ActiveRunPublishHook hook,
                                         void *user_data);
void setBeforeCancelRequestHookForTest(BeforeCancelRequestHook hook,
                                       void *user_data);
void setModelForTest(std::unique_ptr<causallm::Transformer> model,
                     const std::string &architecture);
void resetForTest();
std::string resolveNntrConfigPathForTest(const std::string &value,
                                         const std::string &model_dir_path);
} // namespace causal_lm_api_test

namespace {

/** @brief Shared state for synchronising fake run and cancel threads. */
struct FakeRunState {
  std::mutex mutex;
  std::condition_variable cv;
  bool run_entered = false;
  bool allow_finish = false;
  bool cancel_ready_to_request = false;
  bool allow_cancel_to_request = false;
  bool replacement_completed = false;
  bool stop_seen_at_run_entry = false;
  bool stop_seen_before_finish = false;
  ErrorCode hook_cancel_result = CAUSAL_LM_ERROR_UNKNOWN;
};

FakeRunState *g_fake_run_state = nullptr;

/** @brief Fake CausalLM implementation for cancel API unit tests. */
class FakeCausalLM final : public causallm::CausalLM {
public:
  FakeCausalLM() : CausalLM() {}

  void initialize() override { is_initialized = true; }

  void load_weight(const std::string &) override {}

  void run(const WSTR, bool = false, const WSTR = "", const WSTR = "",
           bool = true) override {
    prepareStopRequestForRun();

    auto *state = g_fake_run_state;
    if (state != nullptr) {
      std::unique_lock<std::mutex> lock(state->mutex);
      state->stop_seen_at_run_entry =
        stop_requested_.load(std::memory_order_acquire);
      state->run_entered = true;
      state->cv.notify_all();
      state->cv.wait(lock, [state]() { return state->allow_finish; });
      state->stop_seen_before_finish =
        stop_requested_.load(std::memory_order_acquire);
    }

    if (output_list.empty())
      output_list.push_back("cancel-test-output");
    else
      output_list[0] = "cancel-test-output";
    has_run_ = true;
  }
};

std::unique_ptr<causallm::Transformer> makeFakeModel() {
  return std::make_unique<FakeCausalLM>();
}

bool waitForRunEntered(FakeRunState &state) {
  std::unique_lock<std::mutex> lock(state.mutex);
  return state.cv.wait_for(lock, std::chrono::seconds(5),
                           [&state]() { return state.run_entered; });
}

void allowRunToFinish(FakeRunState &state) {
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.allow_finish = true;
  }
  state.cv.notify_all();
}

void cancelFromPublishHook(void *user_data) {
  auto *state = static_cast<FakeRunState *>(user_data);
  state->hook_cancel_result = cancelModel();
}

bool waitForCancelReadyToRequest(FakeRunState &state) {
  std::unique_lock<std::mutex> lock(state.mutex);
  return state.cv.wait_for(lock, std::chrono::seconds(5), [&state]() {
    return state.cancel_ready_to_request;
  });
}

void blockBeforeCancelRequestHook(void *user_data) {
  auto *state = static_cast<FakeRunState *>(user_data);
  std::unique_lock<std::mutex> lock(state->mutex);
  state->cancel_ready_to_request = true;
  state->cv.notify_all();
  state->cv.wait(lock, [state]() { return state->allow_cancel_to_request; });
}

bool waitForReplacementCompleted(FakeRunState &state,
                                 std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(state.mutex);
  return state.cv.wait_for(lock, timeout,
                           [&state]() { return state.replacement_completed; });
}

void allowCancelToRequest(FakeRunState &state) {
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.allow_cancel_to_request = true;
  }
  state.cv.notify_all();
}

} // namespace

TEST(CausalLmCancelApiTest, HeaderExposesCancelModelBeforeLoad) {
  causal_lm_api_test::resetForTest();

  ErrorCode (*cancel_fn)(void) = cancelModel;

  EXPECT_EQ(cancel_fn(), CAUSAL_LM_ERROR_NOT_INITIALIZED);
}

TEST(CausalLmCancelApiTest, CausalLmExposesCrossThreadStopRequest) {
  static_assert(std::is_same<decltype(&causallm::CausalLM::requestStop),
                             void (causallm::CausalLM::*)()>::value,
                "CausalLM::requestStop must be a public void method");
}

TEST(CausalLmCancelApiTest, ResolvesRelativeNntrConfigPathsAgainstModelDir) {
  EXPECT_EQ(causal_lm_api_test::resolveNntrConfigPathForTest(
              "sidecars/embedding.bin", "/models/gemma4"),
            "/models/gemma4/sidecars/embedding.bin");
  EXPECT_EQ(causal_lm_api_test::resolveNntrConfigPathForTest(
              "/absolute/embedding.bin", "/models/gemma4"),
            "/absolute/embedding.bin");
  EXPECT_EQ(
    causal_lm_api_test::resolveNntrConfigPathForTest("", "/models/gemma4"), "");
}

TEST(CausalLmCancelApiTest, LoadedModelWithoutActiveRunCancelsAsNoOpSuccess) {
  causal_lm_api_test::resetForTest();
  causal_lm_api_test::setModelForTest(makeFakeModel(), "CancelApiTestCausalLM");

  EXPECT_EQ(cancelModel(), CAUSAL_LM_ERROR_NONE);

  causal_lm_api_test::resetForTest();
}

TEST(CausalLmCancelApiTest, ActiveRunCanBeCancelledAcrossThreads) {
  causal_lm_api_test::resetForTest();
  causal_lm_api_test::setModelForTest(makeFakeModel(), "CancelApiTestCausalLM");

  FakeRunState state;
  g_fake_run_state = &state;
  const char *output = nullptr;
  ErrorCode run_result = CAUSAL_LM_ERROR_UNKNOWN;
  std::thread runner(
    [&]() { run_result = runModel("cancel test prompt", &output); });

  if (!waitForRunEntered(state)) {
    allowRunToFinish(state);
    runner.join();
    g_fake_run_state = nullptr;
    causal_lm_api_test::resetForTest();
    FAIL() << "fake model run did not enter";
  }
  EXPECT_FALSE(state.stop_seen_at_run_entry);
  EXPECT_EQ(cancelModel(), CAUSAL_LM_ERROR_NONE);
  allowRunToFinish(state);
  runner.join();

  EXPECT_EQ(run_result, CAUSAL_LM_ERROR_NONE);
  EXPECT_STREQ(output, "cancel-test-output");
  EXPECT_TRUE(state.stop_seen_before_finish);

  g_fake_run_state = nullptr;
  causal_lm_api_test::resetForTest();
}

TEST(CausalLmCancelApiTest, CancelAfterActivePublishIsNotClearedAtRunStart) {
  causal_lm_api_test::resetForTest();
  causal_lm_api_test::setModelForTest(makeFakeModel(), "CancelApiTestCausalLM");

  FakeRunState state;
  g_fake_run_state = &state;
  causal_lm_api_test::setAfterActiveRunPublishHookForTest(cancelFromPublishHook,
                                                          &state);

  const char *output = nullptr;
  ErrorCode run_result = CAUSAL_LM_ERROR_UNKNOWN;
  std::thread runner(
    [&]() { run_result = runModel("cancel race prompt", &output); });

  if (!waitForRunEntered(state)) {
    allowRunToFinish(state);
    runner.join();
    causal_lm_api_test::setAfterActiveRunPublishHookForTest(nullptr, nullptr);
    g_fake_run_state = nullptr;
    causal_lm_api_test::resetForTest();
    FAIL() << "fake model run did not enter";
  }
  allowRunToFinish(state);
  runner.join();

  EXPECT_EQ(state.hook_cancel_result, CAUSAL_LM_ERROR_NONE);
  EXPECT_TRUE(state.stop_seen_at_run_entry);
  EXPECT_TRUE(state.stop_seen_before_finish);
  EXPECT_EQ(run_result, CAUSAL_LM_ERROR_NONE);
  EXPECT_STREQ(output, "cancel-test-output");

  causal_lm_api_test::setAfterActiveRunPublishHookForTest(nullptr, nullptr);
  g_fake_run_state = nullptr;
  causal_lm_api_test::resetForTest();
}

TEST(CausalLmCancelApiTest, ModelReplacementWaitsForInFlightCancelDereference) {
  causal_lm_api_test::resetForTest();
  causal_lm_api_test::setModelForTest(makeFakeModel(), "CancelApiTestCausalLM");

  FakeRunState state;
  g_fake_run_state = &state;
  causal_lm_api_test::setBeforeCancelRequestHookForTest(
    blockBeforeCancelRequestHook, &state);

  const char *output = nullptr;
  ErrorCode run_result = CAUSAL_LM_ERROR_UNKNOWN;
  ErrorCode cancel_result = CAUSAL_LM_ERROR_UNKNOWN;

  std::thread runner(
    [&]() { run_result = runModel("cancel lifetime prompt", &output); });
  if (!waitForRunEntered(state)) {
    allowRunToFinish(state);
    runner.join();
    causal_lm_api_test::setBeforeCancelRequestHookForTest(nullptr, nullptr);
    g_fake_run_state = nullptr;
    causal_lm_api_test::resetForTest();
    FAIL() << "fake model run did not enter";
  }

  std::thread canceller([&]() { cancel_result = cancelModel(); });
  if (!waitForCancelReadyToRequest(state)) {
    allowRunToFinish(state);
    allowCancelToRequest(state);
    canceller.join();
    runner.join();
    causal_lm_api_test::setBeforeCancelRequestHookForTest(nullptr, nullptr);
    g_fake_run_state = nullptr;
    causal_lm_api_test::resetForTest();
    FAIL() << "cancelModel did not reach requestStop";
  }

  allowRunToFinish(state);
  std::thread replacer([&]() {
    causal_lm_api_test::setModelForTest(makeFakeModel(),
                                        "CancelApiTestCausalLM");
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.replacement_completed = true;
    }
    state.cv.notify_all();
  });

  EXPECT_FALSE(
    waitForReplacementCompleted(state, std::chrono::milliseconds(50)))
    << "model replacement completed while cancelModel still held a selected "
       "active model pointer";

  allowCancelToRequest(state);
  canceller.join();
  runner.join();
  replacer.join();

  EXPECT_EQ(cancel_result, CAUSAL_LM_ERROR_NONE);
  EXPECT_EQ(run_result, CAUSAL_LM_ERROR_NONE);
  EXPECT_TRUE(state.replacement_completed);

  causal_lm_api_test::setBeforeCancelRequestHookForTest(nullptr, nullptr);
  g_fake_run_state = nullptr;
  causal_lm_api_test::resetForTest();
}
