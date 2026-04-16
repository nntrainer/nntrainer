// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Arm Limited and/or its affiliates
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   kleidiai_interface_qai8dxp_qsi4cxp.cpp
 * @date   15 September 2025
 * @see    https://github.com/ARM-software/kleidiai
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong
 *
 * @brief  Modified computational backend components of
 * kleidiai. Portions of this file are derived from Arm
 * Limited code licensed under the Apache License, Version 2.0, with
 * modifications
 *
 * @note   Licensed under the Apache License, Version 2.0 (the "License");
 *         you may not use this file except in compliance with the License.
 *         You may obtain a copy of the License at
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * @modifications
 *   - [2025-09-15] Integrated and adapted Arm-provided code into
 *     nntrainer CPU backend
 *
 * @bug    No known bugs except for NYI items
 */
//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates
// <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <kleidiai_interface.h>
#include <limits>
#include <mutex>
#include <string>
#include <thread_manager.h>
#include <unordered_map>
#include <vector>

// Runtime ARM feature detection via auxv HWCAP bits. SME/SME2 require
// both CPU support AND OS kernel enablement (kernel 6.3+ for SME),
// so compile-time flags alone are not sufficient — fall back to NEON
// if the running kernel doesn't expose SME.
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::nanoseconds;  // or microseconds
using std::chrono::seconds;      // or microseconds

// Include micro-kernel variants
// NEON dotprod variants
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.h"
// NEON i8mm variants
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
// SME variants (from upstream KleidiAI, C6)
#ifdef ENABLE_SME
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot.h"
#endif
#ifdef ENABLE_SME2
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#endif
// Interface
#include "kai/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"
#include "kai/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#define INT4_MIN (-8)
#define INT4_MAX (7)
/**
 * @brief rhs_format
 *
 */
enum class rhs_format {
  nxk,
  kxn,
};

// Micro-kernel interface
/**
 * @brief kai_matmul_ukernel_f32_qa8dxp_qs4cxp
 *
 */
struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {
  kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel ukernel;
  std::string name = {};
};

kai_matmul_ukernel_f32_qa8dxp_qs4cxp ukernel_variants[] = {
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
   "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod"},
#ifdef ENABLE_SME
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
   "matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot,
   "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme_dot"},
#endif
#ifdef ENABLE_SME2
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
   "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa"},
  {kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
   "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot"},
#endif
};

static size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

// Static RHS-pack cache: maps (rhs_data_ptr, variant_idx, transB) to
// the KleidiAI-packed RHS buffer. Weight tensors are static in
// inference — their data pointers don't change after load — so we can
// pack the weight ONCE at first call and reuse for every subsequent
// forward pass. This eliminates the per-call RHS packing overhead
// that was previously dominating the int4 path (the weight is ~1MB
// per FC and repacking it every forward wasted ~50% of GEMM time).
//
// Cache key: packs (ptr_low32, variant_idx, transB_bit) into uint64.
// Two different weights can share a cache slot only if they have the
// same pointer — extremely unlikely in practice for simultaneously-
// live tensors.
static std::mutex kai_rhs_cache_mutex;
static std::unordered_map<uint64_t, std::vector<uint8_t>> kai_rhs_cache;

static inline uint64_t kai_rhs_cache_key(const void *rhs_ptr,
                                          uint32_t variant_idx, bool transB) {
  return (reinterpret_cast<uint64_t>(rhs_ptr) << 8) |
         (static_cast<uint64_t>(variant_idx) << 1) |
         (transB ? 1ull : 0ull);
}

uint32_t nntr_kai_gemm_qai8dxp_qsi4cxp_rtp(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_native_mtx_qs4cx, void *rhs_scales_f32, float *dst_act_mtx_f32,
  bool transB, float lower_bound, float upper_bound) {
  // Variant selection: compile-time feature availability + RUNTIME
  // CPU feature check. An SME2-compiled binary can run on CPUs that
  // only have SME, or pure NEON, or where the OS kernel hasn't
  // enabled SME userspace access yet (common on current Android
  // devices — Linux kernel SME support shipped in 6.3 but many
  // phones still run older kernels). Without runtime fallback the
  // kernel SIGILLs on SMSTART / SME2 instructions.
  //
  // Variant layout after C6 upstream sync:
  //   [0..7]   NEON dotprod + i8mm (always compiled, always safe)
  //   [8..9]   SME  mopa, SME  dot    (needs runtime SME)
  //   [10..11] SME2 mopa, SME2 sdot   (needs runtime SME2)
  uint32_t ret_idx;

  // Runtime feature detection (aarch64 Linux/Android)
  bool has_sme  = false;
  bool has_sme2 = false;
#if defined(__aarch64__) && defined(__linux__)
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
# ifdef HWCAP2_SME
  has_sme  = (hwcap2 & HWCAP2_SME)  != 0;
# endif
# ifdef HWCAP2_SME2
  has_sme2 = (hwcap2 & HWCAP2_SME2) != 0;
# endif
#endif

#if defined(ENABLE_SME2)
  if (has_sme2) {
    ret_idx = (m == 1) ? 11 : 10;  // SME2 sdot / mopa
  } else
#endif
#if defined(ENABLE_SME)
  if (has_sme) {
    ret_idx = (m == 1) ? 9 : 8;    // SME dot / mopa
  } else
#endif
  {
    // NEON fallback. Index 0 (1x8_qsi4cxp4x8_1x4x32_neon_dotprod) is
    // GEMV-tuned; index 5 (qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm)
    // has the largest GEMM tile and is fastest on armv8.6-a+ cores
    // that support i8mm (Cortex-X series, all recent Snapdragon 8).
    ret_idx = (m == 1) ? 0 : 5;
  }
  // Selected variant params (compile+runtime selected above)
  const uint32_t idx_variant = ret_idx;
  const rhs_format format = transB ? rhs_format::nxk : rhs_format::kxn;

  const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  // ==========================================================
  // RHS PACKING — cached across calls, done ONCE per weight
  // ==========================================================
  // Pack the static weight into KleidiAI layout on first call,
  // then reuse the packed buffer for every subsequent forward.
  // For LLM decode (M=1) this turns ~196 FC calls per token from
  // "pack+matmul" into just "matmul", halving the CPU time.
  const uint64_t cache_key =
    kai_rhs_cache_key(rhs_native_mtx_qs4cx, idx_variant, transB);
  const uint8_t *rhs_packed_cached = nullptr;
  {
    std::lock_guard<std::mutex> lk(kai_rhs_cache_mutex);
    auto it = kai_rhs_cache.find(cache_key);
    if (it == kai_rhs_cache.end()) {
      // First time seeing this (weight_ptr, variant, transB) combo:
      // allocate + pack.
      const size_t rhs_packed_size =
        (format == rhs_format::nxk)
          ? kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, nr,
                                                                    kr, sr)
          : kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(n, k, nr,
                                                                    kr, sr);
      std::vector<uint8_t> buf(rhs_packed_size);
      if (format == rhs_format::nxk) {
        struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params p;
        p.lhs_zero_point = 1;
        p.rhs_zero_point = 8;
        kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
          1, n, k, nr, kr, sr,
          (const uint8_t *)(rhs_native_mtx_qs4cx), NULL,
          (const float *)(rhs_scales_f32), buf.data(), 0, &p);
      } else {
        struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params p;
        p.lhs_zero_point = 1;
        p.rhs_zero_point = 8;
        kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
          1, n, k, nr, kr, sr,
          (const uint8_t *)(rhs_native_mtx_qs4cx), NULL,
          (const float *)(rhs_scales_f32), buf.data(), 0, &p);
      }
      auto [inserted_it, ok] =
        kai_rhs_cache.emplace(cache_key, std::move(buf));
      rhs_packed_cached = inserted_it->second.data();
    } else {
      rhs_packed_cached = it->second.data();
    }
  }

  // ==========================================================
  // LHS PACKING — per-call (activation changes every token)
  // ==========================================================
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
  kai_run_lhs_quant_pack_qai8dxp_f32(
    m, k, mr, kr, sr, 0,
    (const float *)lhs_native_mtx_f32, k * sizeof(float),
    lhs_packed_mtx_qa8dx);

  // ==========================================================
  // MATMUL — parallelized over N output columns
  // ==========================================================
  // Split N across ThreadManager worker threads. Each thread runs
  // the same ukernel on its slice of output columns, reading from
  // the same cached packed RHS (different slice offset per thread).
  // This matches Q4_0's parallel_for_chunked pattern — without it,
  // KleidiAI is single-threaded and runs 4-8x slower than Q4_0 on
  // multi-core devices.
  //
  // Constraint: each thread's N-slice must be a multiple of the
  // kernel's nr (output-column tile). For unaligned N we let the
  // last thread handle the remainder.
  const size_t nr_tile = nr;
  auto &tm = nntrainer::ThreadManager::Global();
  unsigned int thread_num = tm.getComputeThreadCount() + 1;

  // Don't bother spawning threads for tiny GEMMs.
  if (n < nr_tile * 2 || thread_num <= 1) {
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
    const size_t rhs_offset =
      ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k);
    const size_t dst_offset =
      ukernel_variants[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

    ukernel_variants[idx_variant].ukernel.run_matmul(
      m, n, k,
      (const char *)lhs_packed_mtx_qa8dx + lhs_offset,
      (const char *)rhs_packed_cached + rhs_offset,
      (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset),
      dst_stride, sizeof(float), lower_bound, upper_bound);
  } else {
    // Chunk N into thread_num parts, aligned to nr_tile.
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
    const size_t n_tiles = n / nr_tile;
    const size_t tiles_per_thread = (n_tiles + thread_num - 1) / thread_num;

    tm.parallel_for_chunked(thread_num, [=](size_t thread_idx) {
      const size_t tile_start = thread_idx * tiles_per_thread;
      if (tile_start >= n_tiles) return;
      const size_t tile_end =
        std::min(tile_start + tiles_per_thread, n_tiles);
      const size_t n_start = tile_start * nr_tile;
      size_t n_end = tile_end * nr_tile;
      // Last thread picks up any N-remainder past the tile boundary.
      if (thread_idx == thread_num - 1 && n_end < n) {
        n_end = n;
      }
      const size_t n_this = n_end - n_start;

      const size_t rhs_offset =
        ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_start, k);
      const size_t dst_offset =
        ukernel_variants[idx_variant].ukernel.get_dst_offset(0, n_start,
                                                              dst_stride);
      ukernel_variants[idx_variant].ukernel.run_matmul(
        m, n_this, k,
        (const char *)lhs_packed_mtx_qa8dx + lhs_offset,
        (const char *)rhs_packed_cached + rhs_offset,
        (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset),
        dst_stride, sizeof(float), lower_bound, upper_bound);
    });
  }

  delete[] lhs_packed_mtx_qa8dx;
  // RHS buffer is owned by kai_rhs_cache — never freed here.
  return ret_idx;
}

size_t nntr_kai_get_rhs_packed_size_qsi4cxp_qs4cxs1s0(size_t n, size_t k,
                                                      uint32_t idx_variant,
                                                      bool transB) {
  ///@note Packing arguments are identical among all ukernel idx_variants
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();
  if (transB) {
    return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, nr, kr,
                                                                  sr);
  } else {
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(n, k, nr, kr,
                                                                  sr);
  }
}

void nntr_kai_qsi4cxp_qs4cxs1s0_rhs_pack(size_t n, size_t k,
                                         void *rhs_packed_mtx_qs4cx,
                                         void *rhs_native_mtx_qs4cx,
                                         void *rhs_scales_f32,
                                         uint32_t idx_variant, bool transB) {
  ///@note Packing arguments are identical among all ukernel idx_variants
  rhs_format format = rhs_format::nxk;
  if (!transB) {
    format = rhs_format::kxn;
  }

  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  if (format == rhs_format::nxk) {
    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params nxk_params;

    nxk_params.lhs_zero_point = 1;
    nxk_params.rhs_zero_point = 8;
    // RHS packing
    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
      1, n, k, nr, kr, sr,                     // Packing arguments
      (const uint8_t *)(rhs_native_mtx_qs4cx), // RHS
      NULL,                                    // Bias
      (const float *)(rhs_scales_f32),         // Scale
      rhs_packed_mtx_qs4cx,                    // RHS packed
      0, &nxk_params);

  } else {
    struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params;
    kxn_params.lhs_zero_point = 1;
    kxn_params.rhs_zero_point = 8;
    // RHS packing
    kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
      1, n, k, nr, kr, sr,                     // Packing arguments
      (const uint8_t *)(rhs_native_mtx_qs4cx), // RHS
      NULL,                                    // Bias
      (const float *)(rhs_scales_f32),         // Scale
      rhs_packed_mtx_qs4cx,                    // RHS packed
      0, &kxn_params);
  }
}

void nntr_kai_gemm_qai8dxp_qsi4cxp_olp_single_thread(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_packed_mtx_qs4cx, float *dst_act_mtx_f32, uint32_t idx_variant,
  bool transB, float lower_bound, float upper_bound) {
  rhs_format format = rhs_format::nxk;
  if (!transB) {
    format = rhs_format::kxn;
  }

  const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  // LHS packing
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
  kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, // Packing arguments
                                     (const float *)lhs_native_mtx_f32, // LHS
                                     k * sizeof(float),     // LHS stride
                                     lhs_packed_mtx_qa8dx); // LHS packed
  {
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
    const size_t rhs_offset =
      ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k);
    const size_t dst_offset =
      ukernel_variants[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

    const void *lhs_ptr =
      (const void *)((const char *)lhs_packed_mtx_qa8dx + lhs_offset);
    const void *rhs_ptr =
      (const void *)((const char *)rhs_packed_mtx_qs4cx + rhs_offset);
    float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

    ukernel_variants[idx_variant].ukernel.run_matmul(
      m, n, k,                 // Dimensions
      lhs_ptr,                 // LHS packed
      rhs_ptr,                 // RHS packed
      dst_ptr,                 // DST
      dst_stride,              // DST stride (row)
      sizeof(float),           // DST stride (col)
      lower_bound, upper_bound // Min and max for the clamp operation
    );
  }

  delete[] lhs_packed_mtx_qa8dx;
}

void nntr_kai_gemm_qai8dxp_qsi4cxp_olp_n_parallel(
  size_t m, size_t n, size_t k, void *lhs_native_mtx_f32,
  void *rhs_packed_mtx_qs4cx, float *dst_act_mtx_f32, uint32_t idx_variant,
  bool transB, float lower_bound, float upper_bound) {
  rhs_format format = rhs_format::nxk;
  if (!transB) {
    format = rhs_format::kxn;
  }

  const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
  const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
  const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
  const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

  // LHS packing
  const size_t lhs_packed_size =
    kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
  uint8_t *lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
  kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr, 0, // Packing arguments
                                     (const float *)lhs_native_mtx_f32, // LHS
                                     k * sizeof(float),     // LHS stride
                                     lhs_packed_mtx_qa8dx); // LHS packed
  int n_threads = 4;
  assert(n % n_threads == 0);
  size_t n_ukernel = n / n_threads;
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, static_cast<size_t>(n_threads), static_cast<unsigned int>(n_threads), [&](size_t current_thread) {
    const size_t dst_stride = n * sizeof(float);
    const size_t lhs_offset =
      ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
    const size_t rhs_offset =
      ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(
        n_ukernel * current_thread, k);
    const size_t dst_offset =
      ukernel_variants[idx_variant].ukernel.get_dst_offset(
        0, n_ukernel * current_thread, dst_stride);

    const void *lhs_ptr =
      (const void *)((const char *)lhs_packed_mtx_qa8dx + lhs_offset);
    const void *rhs_ptr =
      (const void *)((const char *)rhs_packed_mtx_qs4cx + rhs_offset);
    float *dst_ptr = (float *)((uint8_t *)dst_act_mtx_f32 + dst_offset);

    ukernel_variants[idx_variant].ukernel.run_matmul(
      m, n / n_threads, k,     // Dimensions
      lhs_ptr,                 // LHS packed
      rhs_ptr,                 // RHS packed
      dst_ptr,                 // DST
      dst_stride,              // DST stride (row)
      sizeof(float),           // DST stride (col)
      lower_bound, upper_bound // Min and max for the clamp operation
    );
  });

  delete[] lhs_packed_mtx_qa8dx;
}

void nntr_kai_gemm_qai8dxp_qsi4cxp_olp(size_t m, size_t n, size_t k,
                                       void *lhs_native_mtx_f32,
                                       void *rhs_packed_mtx_qs4cx,
                                       float *dst_act_mtx_f32,
                                       uint32_t idx_variant, bool transB,
                                       float lower_bound, float upper_bound) {
  if (m == 1) {
    return nntr_kai_gemm_qai8dxp_qsi4cxp_olp_single_thread(
      m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs4cx, dst_act_mtx_f32,
      idx_variant, transB, lower_bound, upper_bound);
  } else {
    return nntr_kai_gemm_qai8dxp_qsi4cxp_olp_n_parallel(
      m, n, k, lhs_native_mtx_f32, rhs_packed_mtx_qs4cx, dst_act_mtx_f32,
      idx_variant, transB, lower_bound, upper_bound);
  }
}
