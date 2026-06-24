// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_callback_streamer.cpp
 * @brief   Focused tests for CausalLM callback streamer adapter behavior.
 */

#include "callback_streamer.h"

#include <gtest/gtest.h>

#include <string>

namespace {

struct CallbackState {
  int calls = 0;
  std::string text;
};

int append_callback(const char *delta, void *user_data) {
  auto *state = static_cast<CallbackState *>(user_data);
  ++state->calls;
  state->text += delta;
  return 0;
}

int cancel_on_first_callback(const char *delta, void *user_data) {
  auto *state = static_cast<CallbackState *>(user_data);
  ++state->calls;
  state->text += delta;
  return 7;
}

} // namespace

TEST(CausalLmCallbackStreamerTest, ForwardsDeltaAndUserData) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "hello"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, " world"), 0);

  EXPECT_EQ(state.calls, 2);
  EXPECT_EQ(state.text, "hello world");
  EXPECT_EQ(streamer.cancelled, 0);
}

TEST(CausalLmCallbackStreamerTest, CallbackNonZeroIsStickyCancellation) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, cancel_on_first_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "first"), 7);
  EXPECT_EQ(streamer_put(&streamer.base, "second"), 7);

  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(state.text, "first");
  EXPECT_EQ(streamer.cancelled, 7);
}

TEST(CausalLmCallbackStreamerTest, NullSafeWrappersIgnoreMissingHooks) {
  CallbackState state;
  BaseStreamer empty = {nullptr};

  EXPECT_EQ(streamer_put(nullptr, "ignored"), 0);
  EXPECT_EQ(streamer_put(&empty, "ignored"), 0);
  EXPECT_EQ(streamer_put(nullptr, nullptr), 0);
  EXPECT_EQ(streamer_put(&empty, nullptr), 0);
  EXPECT_NO_THROW(streamer_end(nullptr));
  EXPECT_NO_THROW(streamer_end(&empty));
  EXPECT_EQ(state.calls, 0);
}

TEST(CausalLmCallbackStreamerTest, NullInitIsIgnored) {
  EXPECT_NO_THROW(callback_streamer_init(nullptr, append_callback, nullptr));
}
