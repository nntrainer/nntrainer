// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    callback_streamer.h
 * @brief   BaseStreamer implementation that forwards every delta to a
 *          user-supplied C function pointer.
 * @author  junbong.yu <junbong.yu@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the streamer used by the JNI bridge in QuickAI: the Kotlin
 * side hands the JNI entry point a listener object, and the JNI entry
 * point wraps the listener in a CausalLmTokenCallback + user_data pair
 * and pushes a CallbackStreamer onto its own stack frame for the
 * duration of the call.
 *
 * See AsyncAndStreaming.md §3.2 at the repo root.
 */
#ifndef __QUICK_DOT_AI_CALLBACK_STREAMER_H__
#define __QUICK_DOT_AI_CALLBACK_STREAMER_H__

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include "streamer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Token callback signature.
 *
 * @param delta     UTF-8 text produced for this token boundary. Valid
 *                  only for the duration of the call — copy before
 *                  retaining.
 * @param user_data Opaque pointer passed through from the
 *                  runModelHandleStreaming() caller.
 * @return 0 to continue generation, non-zero to request cancellation.
 */
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);

/**
 * @brief A BaseStreamer that forwards every put() to a
 *        CausalLmTokenCallback.
 *
 * Layout note: @c base MUST be the first member so that a
 * `CallbackStreamer*` can be safely cast to `BaseStreamer*`.
 */
typedef struct {
  BaseStreamer base;
  CausalLmTokenCallback callback;
  void *user_data;
  int cancelled; /**< sticky: once set to non-zero, put() becomes a no-op. */
} CallbackStreamer;

/**
 * @brief Initialize a CallbackStreamer in-place. Does not allocate.
 *
 * @param self      Storage owned by the caller (typically stack).
 * @param cb        Callback to invoke for every delta. Must be non-NULL.
 * @param user_data Opaque pointer forwarded to @c cb.
 */
WIN_EXPORT void callback_streamer_init(CallbackStreamer *self,
                                       CausalLmTokenCallback cb,
                                       void *user_data);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __QUICK_DOT_AI_CALLBACK_STREAMER_H__