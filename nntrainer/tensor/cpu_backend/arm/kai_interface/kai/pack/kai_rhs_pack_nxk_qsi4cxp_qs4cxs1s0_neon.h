/**
 * @file   kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon.h
 * @date   02 July 2026
 * @see    https://github.com/ARM-software/kleidiai
 * @author Jaemin Shin <jaemin980311@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom NEON-optimized variant of kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0
 *         specialized for (nr, kr, sr) = (8, 16, 2).
 */
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Run the micro-kernel to pack the RHS matrix (NEON-optimized).
///
/// Drop-in replacement for kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0. The fast
/// path covers (nr, kr, sr) = (8, 16, 2) with rhs_zero_point == 8; any other
/// configuration is delegated to the reference implementation. The packed
/// output is byte-identical to the reference implementation.
///
/// The size/offset/stride helper functions of the reference implementation
/// (kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0, ...) remain valid
/// for this variant.
///
/// @param[in]  num_groups  The number of groups. It must be 1.
/// @param[in]  n           The number of rows.
/// @param[in]  k           The common dimension between the LHS and RHS matrix
/// (K). It must be an even value.
/// @param[in]  nr          The number of N rows to interleave on the same
/// output output row.
/// @param[in]  kr          The number of K values loaded in the single inner
/// most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up
/// to kr.
///                         However, kr must be multiple of sr.
/// @param[in]  rhs         The RHS matrix containing the 4-bit values.
///                         Size in bytes is expected to be greater than or
///                         equal to n * k * (sizeof(uint8_t) / 2).
/// @param[in]  bias        The biases.
/// @param[in]  scale       The scale for each output channel.
/// @param[out] rhs_packed  The packed RHS matrix.
/// @param[in]  extra_bytes Extra bytes to append to the end of each row of the
/// packed RHS matrix.
/// @param[in]  params      Parameters for the micro-kernel.
void kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_neon(
  size_t num_groups,  //
  size_t n,           //
  size_t k,           //
  size_t nr,          //
  size_t kr,          //
  size_t sr,          //
  const uint8_t *rhs, //
  const float *bias,  //
  const float *scale, //
  void *rhs_packed,   //
  size_t extra_bytes, //
  const struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params *params);

#ifdef __cplusplus
}
#endif
