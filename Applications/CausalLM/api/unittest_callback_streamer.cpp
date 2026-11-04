// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_callback_streamer.cpp
 * @brief   Focused tests for CausalLM callback streamer adapter behavior.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "callback_streamer.h"

#include <gtest/gtest.h>

#include <string>

namespace {

/** @brief State captured by the test token callback. */
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

TEST(CausalLmCallbackStreamerTest, StripsTurnMarkersFromDeltas) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "answer.<turn|>"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, "<|im_start|>a<|im_end|>b"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, "x<|turn>y<|turn>z"), 0);

  EXPECT_EQ(state.calls, 3);
  EXPECT_EQ(state.text, "answer.abxyz");
  EXPECT_EQ(streamer.cancelled, 0);
}

TEST(CausalLmCallbackStreamerTest, MarkerOnlyDeltaIsDroppedWithoutCancelling) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "<turn|>"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, "<|im_end|><|im_start|>"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, "after"), 0);

  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(state.text, "after");
  EXPECT_EQ(streamer.cancelled, 0);
}

TEST(CausalLmCallbackStreamerTest, TextWithPipeButNoMarkerIsUntouched) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "a | b <|not a marker|> c"), 0);

  EXPECT_EQ(state.text, "a | b <|not a marker|> c");
}

TEST(CausalLmCallbackStreamerTest, PadDeltasAreSwallowedThenCancelAtEight) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  for (int i = 0; i < 7; ++i)
    EXPECT_EQ(streamer_put(&streamer.base, "<pad>"), 0) << "pad " << i;
  EXPECT_EQ(state.calls, 0);
  EXPECT_EQ(streamer.cancelled, 0);

  EXPECT_EQ(streamer_put(&streamer.base, "<pad>"), 1);
  EXPECT_EQ(streamer.cancelled, 1);

  // sticky afterwards, and nothing reaches the callback
  EXPECT_EQ(streamer_put(&streamer.base, "text"), 1);
  EXPECT_EQ(state.calls, 0);
}

TEST(CausalLmCallbackStreamerTest, MultiPadDeltaCountsOnceAndTextResetsRun) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  // a delta of several pad tokens is one delta
  for (int i = 0; i < 7; ++i)
    EXPECT_EQ(streamer_put(&streamer.base, "<pad><pad>"), 0);
  EXPECT_EQ(streamer.cancelled, 0);

  // real text resets the run, so seven more pads still do not cancel
  EXPECT_EQ(streamer_put(&streamer.base, "word"), 0);
  for (int i = 0; i < 7; ++i)
    EXPECT_EQ(streamer_put(&streamer.base, "<pad>"), 0);
  EXPECT_EQ(streamer.cancelled, 0);

  EXPECT_EQ(state.calls, 1);
  EXPECT_EQ(state.text, "word");
}

TEST(CausalLmCallbackStreamerTest, PadInsideTextIsForwarded) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, append_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "<pad>x"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, "<pa"), 0);
  EXPECT_EQ(streamer_put(&streamer.base, ""), 0);

  EXPECT_EQ(state.calls, 3);
  EXPECT_EQ(state.text, "<pad>x<pa");
}

TEST(CausalLmCallbackStreamerTest, CallbackCancellationSurvivesMarkerStrip) {
  CallbackState state;
  CallbackStreamer streamer;
  callback_streamer_init(&streamer, cancel_on_first_callback, &state);

  EXPECT_EQ(streamer_put(&streamer.base, "hi<turn|>"), 7);
  EXPECT_EQ(state.text, "hi");
  EXPECT_EQ(streamer.cancelled, 7);
}
