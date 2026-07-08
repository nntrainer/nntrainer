// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_session.h
 * @date   06 Aug 2026
 * @brief  Per-session HMX/VTCM state and the u8i4 weight registry
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTR_HVX_SESSION_H__
#define __NNTR_HVX_SESSION_H__

#include <stdint.h>

#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_mm_u8i8_dma.h"
#include "hvx_worker_pool.h"

/**
 * @brief State held for the lifetime of one nntr_hvx_open()/close() pair.
 *
 * hw_init and the HMX lock happen once in open() rather than per call (doc15
 * §3/§4): every FastRPC entry point in this file reaches its VTCM arena and
 * weight table through the session instead of re-acquiring either. This
 * assumes one open session at a time -- hexkl_micro_hw_init is a singleton
 * DSP resource, so a second concurrent open would contend for the same VTCM
 * arena and HMX lock. nntrainer opens exactly one HTP session per process,
 * so that is not a real constraint today; it would need addressing before
 * this skel served more than one client process at once.
 */
typedef struct {
  uint8_t *vtcm_base;
  uint32_t vtcm_size;
  uint32_t config_off; /**< session-constant: depends only on vtcm_size */
  int hmx_locked;      /**< close() only unlocks/finalizes what open() set up */
  hexkl_weight_u8i4_table weights_u8i4;
  hexkl_weight_u8i8_table weights_u8i8;
  hvx_worker_pool *quant_pool; /**< sized from the HVX unit count in open() */
} nntr_hvx_session;

#endif /* __NNTR_HVX_SESSION_H__ */
