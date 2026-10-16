// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexagon_types.h
 * @date   22 Sep 2026
 * @brief  Host stand-in: an HVX vector as 32 words (dma_replay_host_check)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include <stdint.h>
/** @brief 128 bytes, the HVX vector size. */
typedef struct {
  int32_t w[32];
} HVX_Vector;
