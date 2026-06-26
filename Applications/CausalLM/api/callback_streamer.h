// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    callback_streamer.h
 * @brief   BaseStreamer implementation backed by a C token callback.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CAUSAL_LM_CALLBACK_STREAMER_H__
#define __CAUSAL_LM_CALLBACK_STREAMER_H__

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include "causal_lm_callback.h"
#include "streamer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Callback-backed streamer. The base member must remain first.
 */
typedef struct {
  BaseStreamer base;
  CausalLmTokenCallback callback;
  void *user_data;
  int cancelled;
} CallbackStreamer;

/**
 * @brief Initialize a CallbackStreamer in caller-owned storage.
 */
WIN_EXPORT void callback_streamer_init(CallbackStreamer *self,
                                       CausalLmTokenCallback callback,
                                       void *user_data);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_CALLBACK_STREAMER_H__
