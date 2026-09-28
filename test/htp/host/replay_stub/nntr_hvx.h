// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx.h
 * @date   22 Sep 2026
 * @brief  Host stand-in for the qaic header: the two DMA entry points
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include "remote.h"
#include <stdint.h>
typedef uint32_t uint32;
int nntr_hvx_dma_probe(remote_handle64 h, uint32 arena, uint32 src_off,
                       uint32 bytes, uint32 row_size, uint32 nrows,
                       uint32 src_stride, uint32 workers, uint32 passes,
                       uint32 *res, int resLen);
int nntr_hvx_dma_replay(remote_handle64 h, uint32 arena, uint32 region_bytes,
                        const uint32 *schedule, int scheduleLen, uint32 workers,
                        uint32 load, uint32 pace, uint32 fresh, uint32 gap_us,
                        uint32 calls, uint32 *res, int resLen);
