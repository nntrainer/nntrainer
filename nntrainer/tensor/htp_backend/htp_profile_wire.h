// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_profile_wire.h
 * @brief SDK-independent profiling slots shared by ARM and DSP RPC entries.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_PROFILE_WIRE_H
#define NNTR_HTP_PROFILE_WIRE_H

/* Wire ordinals: append new entries only with a matching RPC count change. */
enum {
  HTP_T_DSP_TOTAL = 0,
  HTP_T_QUANT,
  HTP_T_DEQUANT,
  HTP_T_ACC_READ,
  HTP_T_ACC_COPY,
  HTP_T_DRAIN,
  HTP_T_ACC_STRIDE,
  HTP_N_STAGES
};

enum {
  HTP_FU_T_DSP_TOTAL = 0,
  HTP_FU_T_QUANT,
  HTP_FU_T_SWIGLU,
  HTP_FU_T_DEQUANT,
  HTP_FU_T_ACC_READ,
  HTP_FU_T_ACC_COPY,
  HTP_FU_T_DRAIN,
  HTP_FU_T_ACC_STRIDE,
  HTP_FU_N_STAGES
};

enum {
  HTP_GU_T_DSP_TOTAL = 0,
  HTP_GU_T_QUANT,
  HTP_GU_T_SWIGLU,
  HTP_GU_T_DEQUANT,
  HTP_GU_T_ACC_READ,
  HTP_GU_T_DRAIN,
  HTP_GU_T_ACC_STRIDE,
  HTP_GU_N_STAGES
};

enum {
  HTP_MOE_T_DSP_TOTAL = 0,
  HTP_MOE_T_QUANT,
  HTP_MOE_T_SWIGLU,
  HTP_MOE_T_DEQUANT,
  HTP_MOE_T_ACC_READ,
  HTP_MOE_T_DRAIN,
  HTP_MOE_T_SCATTER,
  HTP_MOE_T_GATHER,
  HTP_MOE_T_REQUANT,
  HTP_MOE_T_BLOCKS,
  HTP_MOE_T_MM,
  HTP_MOE_T_DMA_KB,
  HTP_MOE_T_DMA_FIRST,
  HTP_MOE_T_ALLOC, /**< the layer call's own malloc and free */
  HTP_MOE_T_DMA_FIRST_KB,
  HTP_MOE_T_DRAIN_DN,
  HTP_MOE_T_PUSH,
  HTP_MOE_T_STAGE,
  HTP_MOE_T_ACC_STRIDE,
  /** The DMA ring trace's per-call numbers, hexkl_probe.h's
      HEXKL_PROBE_DMA_* in the same order. Counts unless named _US. */
  HTP_MOE_T_DMA_DESC,
  HTP_MOE_T_DMA_WAITS,
  HTP_MOE_T_DMA_WAITS_BLOCKED,
  HTP_MOE_T_DMA_WAIT_US,
  HTP_MOE_T_DMA_WAIT_ACT_US,
  HTP_MOE_T_DMA_BUSY_LO_US,
  HTP_MOE_T_DMA_BUSY_HI_US,
  HTP_MOE_T_DMA_DEPTH_MAX,
  HTP_MOE_T_DMA_FIRST_READY_US,
  HTP_MOE_T_DMA_LAST_ISSUE_US,
  HTP_MOE_T_PATH,    /**< NOT us: 0 = HMX block loop, 1 = M=1 HVX GEMV */
  HTP_MOE_T_M1_FEED, /**< NOT us: the DMA queues the VTCM feed used (#117:
                          1; #177: 1..4), 0 = the GEMV read the arena */
  HTP_MOE_N_STAGES
};

#endif
