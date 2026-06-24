// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    callback_streamer.cpp
 * @brief   Callback-backed BaseStreamer implementation.
 */

#include "callback_streamer.h"

extern "C" {

static int callback_streamer_put(BaseStreamer *self,
                                 const char *decoded_utf8) {
  auto *streamer = reinterpret_cast<CallbackStreamer *>(self);
  if (streamer == nullptr || streamer->callback == nullptr) {
    return 0;
  }

  if (streamer->cancelled != 0) {
    return streamer->cancelled;
  }

  const int ret = streamer->callback(decoded_utf8, streamer->user_data);
  if (ret != 0) {
    streamer->cancelled = ret;
  }
  return ret;
}

static void callback_streamer_end(BaseStreamer * /*self*/) {}

static const BaseStreamerVTable callback_streamer_vtable = {
  &callback_streamer_put,
  &callback_streamer_end,
};

void callback_streamer_init(CallbackStreamer *self,
                            CausalLmTokenCallback callback,
                            void *user_data) {
  if (self == nullptr) {
    return;
  }

  self->base.vtable = &callback_streamer_vtable;
  self->callback = callback;
  self->user_data = user_data;
  self->cancelled = 0;
}

} // extern "C"
