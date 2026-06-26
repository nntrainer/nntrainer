// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    streamer.h
 * @brief   Minimal non-owning streamer abstraction for CausalLM decoded deltas.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CAUSAL_LM_STREAMER_H__
#define __CAUSAL_LM_STREAMER_H__

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
 */
typedef struct {
  /**
   * @brief Forward one UTF-8 decoded delta to the streamer.
   * @return 0 to continue, non-zero to request cooperative stop.
   */
  int (*put)(BaseStreamer *self, const char *decoded_utf8);

  /**
   * @brief Notify the streamer that the current run has ended.
   */
  void (*end)(BaseStreamer *self);
} BaseStreamerVTable;

/**
 * @brief Base streamer. Concrete streamers embed this as their first field.
 */
struct BaseStreamer {
  const BaseStreamerVTable *vtable;
};

/**
 * @brief NULL-safe wrapper around the streamer's put hook.
 */
WIN_EXPORT int streamer_put(BaseStreamer *self, const char *decoded_utf8);

/**
 * @brief NULL-safe wrapper around the streamer's end hook.
 */
WIN_EXPORT void streamer_end(BaseStreamer *self);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_STREAMER_H__
