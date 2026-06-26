// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    causal_lm_callback.h
 * @brief   Public callback types for the CausalLM C API.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CAUSAL_LM_CALLBACK_H__
#define __CAUSAL_LM_CALLBACK_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Token callback invoked for each UTF-8 decoded delta.
 * @param delta UTF-8 decoded delta valid only for the duration of the callback.
 * Clients must copy it if they need to keep it after returning.
 * @param user_data Opaque pointer passed from runModelStreaming().
 *
 * @return 0 to continue, non-zero to request cooperative stop.
 */
typedef int (*CausalLmTokenCallback)(const char *delta, void *user_data);

#ifdef __cplusplus
}
#endif

#endif // __CAUSAL_LM_CALLBACK_H__
