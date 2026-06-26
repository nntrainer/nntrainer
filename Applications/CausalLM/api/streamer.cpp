// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    streamer.cpp
 * @brief   Null-safe BaseStreamer vtable helpers.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "streamer.h"

extern "C" {

int streamer_put(BaseStreamer *self, const char *decoded_utf8) {
  if (self == nullptr || self->vtable == nullptr ||
      self->vtable->put == nullptr || decoded_utf8 == nullptr) {
    return 0;
  }

  return self->vtable->put(self, decoded_utf8);
}

void streamer_end(BaseStreamer *self) {
  if (self == nullptr || self->vtable == nullptr ||
      self->vtable->end == nullptr) {
    return;
  }

  self->vtable->end(self);
}

} // extern "C"
