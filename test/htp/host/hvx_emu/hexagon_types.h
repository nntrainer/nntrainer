// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexagon_types.h
 * @date   27 Sep 2026
 * @brief  Host emulation of the HVX vector types, so a real HVX kernel
 *         source compiles on x86 for m1_ops_host_check.c and
 *         attn_m1_host_check.c
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * This directory shadows the Hexagon SDK's <hexagon_types.h> and
 * <hvx_hexagon_protos.h> for the hvx_emu host checks (plan 82 section 3.2,
 * plan 81 section 3.3). It is
 * not on any other host check's include path on purpose: stub/ replaces
 * kernels with scalar stand-ins, this replaces the instruction set under
 * the real kernel. One 128-byte vector is 32 int32 lanes; the unaligned
 * type is the same struct, since a struct of int32 has 4-byte alignment
 * and the kernels load from float pointers.
 */

#ifndef __NNTRAINER_HVX_EMU_HEXAGON_TYPES_H__
#define __NNTRAINER_HVX_EMU_HEXAGON_TYPES_H__

#include <stdint.h>

#define HVX_EMU_LANES 32

typedef struct {
  int32_t w[HVX_EMU_LANES];
} HVX_Vector;

typedef HVX_Vector HVX_UVector;

/** @brief A register pair Vdd = V(d+1):V(d): lo is V(d), what Q6_V_lo_W
 *         returns and the second operand of Q6_W_vcombine_VV (#170). */
typedef struct {
  HVX_Vector lo, hi;
} HVX_VectorPair;

/** @brief A vector predicate: one flag per byte, as the ISA defines it. The
 *         word-lane ops below set and test all four bytes of a lane. */
typedef struct {
  uint8_t q[4 * HVX_EMU_LANES];
} HVX_VectorPred;

#endif /* __NNTRAINER_HVX_EMU_HEXAGON_TYPES_H__ */
