// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    callback_streamer.cpp
 * @brief   Implementation of the CallbackStreamer vtable, which routes
 *          every decoded-token delta to a user-supplied callback.
 *          See AsyncAndStreaming.md §3.2 at the repo root.
 * @author  junbong.yu <junbong.yu@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "callback_streamer.h"

extern "C" {

static int callback_streamer_put(BaseStreamer *self, const char *decoded_utf8) {
  CallbackStreamer *cs = reinterpret_cast<CallbackStreamer *>(self);
  if (cs == nullptr || cs->callback == nullptr) {
    return 0;
  }
  // Once the user has asked us to cancel, keep returning the sticky
  // cancellation flag — this protects against the (cheap but real)
  // race where the CausalLM generation loop emits one extra token
  // between setting stop_requested_ and actually breaking out.
  if (cs->cancelled != 0) {
    return cs->cancelled;
  }
  int rc = cs->callback(decoded_utf8, cs->user_data);
  if (rc != 0) {
    cs->cancelled = rc;
  }
  return rc;
}

static void callback_streamer_end(BaseStreamer * /*self*/) {
  // Intentionally empty. Stream termination is reported to the caller
  // through the return value of runModelHandleStreaming(); there is no
  // "done" payload to forward here.
}

static const BaseStreamerVTable kCallbackStreamerVTable = {
  /*.put =*/&callback_streamer_put,
  /*.end =*/&callback_streamer_end,
};

void callback_streamer_init(CallbackStreamer *self, CausalLmTokenCallback cb,
                            void *user_data) {
  if (self == nullptr) {
    return;
  }
  self->base.vtable = &kCallbackStreamerVTable;
  self->callback = cb;
  self->user_data = user_data;
  self->cancelled = 0;
}

} // extern "C"
