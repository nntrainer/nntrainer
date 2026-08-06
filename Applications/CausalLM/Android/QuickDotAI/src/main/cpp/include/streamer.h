// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    streamer.h
 * @brief   Minimal C-callable base streamer used by the handle-based
 *          `runModelHandleStreaming` entry point in quick_dot_ai_api.h.
 * @author  junbong.yu <junbong.yu@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is intentionally a very thin vtable-based polymorphism layer so
 * that:
 *   - the CausalLM inference loop can push decoded tokens through a
 *     single pointer,
 *   - concrete streamers (currently only CallbackStreamer) can be
 *     implemented in plain C without dragging C++ headers into the
 *     CausalLM internals,
 *   - the same mechanism is reusable from JNI callers (the JNI bridge
 *     instantiates a CallbackStreamer on the stack and lets the C API
 *     drive it).
 *
 * See AsyncAndStreaming.md §3.1 at the repo root for the full design
 * rationale.
 */
#ifndef __QUICK_DOT_AI_STREAMER_H__
#define __QUICK_DOT_AI_STREAMER_H__

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BaseStreamer BaseStreamer;

/**
 * @brief Vtable for a BaseStreamer.
 *
 * Both function pointers may be NULL — the streamer_put / streamer_end
 * helpers below are null-safe so the caller never has to check.
 */
typedef struct {
  /**
   * @brief Forward one UTF-8 delta string to the streamer.
   *
   * The pointer is only valid for the duration of the call; the
   * streamer implementation must copy if it needs to retain the data.
   *
   * @return 0 to continue generation, non-zero to request cancellation
   *         at the next token boundary.
   */
  int (*put)(BaseStreamer *self, const char *decoded_utf8);

  /**
   * @brief Called exactly once after the last put, regardless of whether
   *        generation finished normally, was cancelled via the callback
   *        return value, or ended because an exception propagated out of
   *        the run loop.
   */
  void (*end)(BaseStreamer *self);
} BaseStreamerVTable;

/**
 * @brief Base streamer. Concrete streamers embed this as their first
 *        field and set @c vtable to a static const instance of
 *        BaseStreamerVTable.
 */
struct BaseStreamer {
  const BaseStreamerVTable *vtable;
};

/**
 * @brief NULL-safe wrapper around the vtable's put() hook. Returns
 *        non-zero if the streamer requested cancellation.
 */
WIN_EXPORT int streamer_put(BaseStreamer *self, const char *decoded_utf8);

/**
 * @brief NULL-safe wrapper around the vtable's end() hook. Idempotent
 *        from the caller's perspective — concrete implementations
 *        should tolerate being called multiple times, but the CausalLM
 *        inference path calls this at most once.
 */
WIN_EXPORT void streamer_end(BaseStreamer *self);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __QUICK_DOT_AI_STREAMER_H__