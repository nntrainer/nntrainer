// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexagon_types.h
 * @date   27 Sep 2026
 * @brief  Host emulation of the HVX vector types, so a real HVX kernel
 *         source compiles on x86 for m1_ops_host_check.c
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * This directory shadows the Hexagon SDK's <hexagon_types.h> and
 * <hvx_hexagon_protos.h> for ONE host check (plan 82 section 3.2). It is
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

#endif /* __NNTRAINER_HVX_EMU_HEXAGON_TYPES_H__ */
