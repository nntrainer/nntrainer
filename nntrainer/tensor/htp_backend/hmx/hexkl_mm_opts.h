// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_opts.h
 * @date   08 Aug 2026
 * @brief  Optional behaviors of hexkl_mm_u8iX_layer_run, as one parameter
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Three callers want three different extras from layer_run -- the FC path
 * wants the quant worker pool, attention's P.V wants that plus caller-known
 * quantization params and accumulate-into-output -- and growing the
 * signature once per extra had it at twelve positional arguments. One
 * struct, NULL meaning "all defaults", ends that; the next extra is a field,
 * not a churn through every caller.
 */

#ifndef __NNTRAINER_HEXKL_MM_OPTS_H__
#define __NNTRAINER_HEXKL_MM_OPTS_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

typedef struct {
  /** Splits the quant passes by row/k-tile. NULL runs them single-threaded. */
  hvx_worker_pool *pool;

  /** Both NULL, or both m_pad = ROUND_UP(M, 64) entries: per-row activation
      quant params to use INSTEAD of scanning act_f32 for min/max. Only valid
      when the caller can produce exactly what the scan would -- attention's
      P.V does, because the blocked softmax already computed each (row, block)
      maximum while storing P, and p >= 0 pins rmin at 0. Anything short of
      exact reproduction is an accuracy change, not an optimization
      (hexkl_attn_u8.h records the constant-params version of that mistake). */
  const float *act_scale;
  const int32_t *act_zp;

  /** Nonzero: out_cat += result instead of =. The adds land per element in
      the same per-call order either way, so accumulating here is bitwise
      identical to staging into a scratch and adding afterwards -- which is
      the caller loop this flag deletes. */
  int accumulate;
} hexkl_mm_opts;

#endif /* __NNTRAINER_HEXKL_MM_OPTS_H__ */
