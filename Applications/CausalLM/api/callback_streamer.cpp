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

#include <cstring>
#include <string>

extern "C" {

/** Maximum consecutive pad-only deltas before generation is cancelled. */
static const int kPadRunLimit = 8;

/** True iff the delta is one or more literal "<pad>" tokens and nothing
 *  else. A degenerated generation streams these forever (integration
 *  4,462-token pad floods have reached the UI). */
static int is_pad_only(const char *s) {
  if (s == nullptr || s[0] == '\0') {
    return 0;
  }
  while (s[0] != '\0') {
    if (s[0] != '<' || s[1] != 'p' || s[2] != 'a' || s[3] != 'd' ||
        s[4] != '>') {
      return 0;
    }
    s += 5;
  }
  return 1;
}

/** Turn delimiters. They are real tokens of the prompt grammar, so they must
 *  stay in the token sequence (and in the KV cache), but they are not words:
 *  a UI that renders the delta verbatim shows a trailing "<turn|>" at the end
 *  of every answer. The QNN loops happen to hide the trailing one because
 *  `<turn|>` (id 106) is normally listed in generation_config.json's
 *  eos_token_id and the loop breaks before decoding it — but that is a
 *  property of the pack, not a guarantee, and the GPU/CPU path emits it. Strip
 * them here, at the one funnel every backend's stream passes through. */
static const char *const kTurnMarkers[] = {"<turn|>", "<|turn>",
                                           // ChatML (LFM2): "<|im_end|>" is
                                           // its EOS and reaches the stream
                                           "<|im_end|>", "<|im_start|>"};

/** Remove every turn delimiter from `s`. Substring (not whole-string) removal
 *  on purpose: the punctuation-hold in CausalLM::registerOutputs can make one
 *  delta span several tokens, so the marker often arrives glued to real text
 *  ("...습니다.<turn|>"). Returns true when anything was removed. */
static bool strip_turn_markers(std::string &s) {
  bool changed = false;
  for (const char *m : kTurnMarkers) {
    const std::string marker(m);
    for (std::string::size_type at = s.find(marker); at != std::string::npos;
         at = s.find(marker, at)) {
      s.erase(at, marker.size());
      changed = true;
    }
  }
  return changed;
}

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
  // Pad tokens are an artifact of a degenerated generation, never real
  // output: swallow them, and once a run of them shows the model is
  // stuck, cancel the generation instead of streaming pads for minutes.
  if (is_pad_only(decoded_utf8)) {
    if (++cs->pad_run >= kPadRunLimit) {
      cs->cancelled = 1;
      return 1;
    }
    return 0;
  }
  cs->pad_run = 0;
  // Turn delimiters are grammar, not text: never show them to the caller.
  // Only pay for the copy when the delta actually contains one.
  const char *out = decoded_utf8;
  std::string stripped;
  // Every marker contains '|' ("<turn|>", "<|im_end|>", ...).
  if (decoded_utf8 != nullptr && std::strchr(decoded_utf8, '|') != nullptr) {
    stripped.assign(decoded_utf8);
    if (strip_turn_markers(stripped)) {
      if (stripped.empty()) {
        // The whole delta was a delimiter — swallow it, but do not report a
        // cancellation: generation ends on the model's own EOS check.
        return 0;
      }
      out = stripped.c_str();
    }
  }
  int rc = cs->callback(out, cs->user_data);
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
  self->pad_run = 0;
}

} // extern "C"