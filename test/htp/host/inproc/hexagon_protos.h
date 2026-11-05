// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexagon_protos.h
 * @date   27 Sep 2026
 * @brief  Host stand-in for the scalar Hexagon intrinsics the DMA ring uses,
 *         on top of hvx_emu/ for the vector ones (in-process HTP build)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */
#pragma once
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief No cache to clean on the host. */
static inline void Q6_dccleaninva_A(void *p) { (void)p; }

#include "../replay_stub/hexkl_dma_standin.h"
