// SPDX-License-Identifier: Apache-2.0
/**
 * @file hexkl_acc_i24.h
 * @brief Exact signed 24-bit HMX accumulator extraction for bounded int4 dots.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HEXKL_ACC_I24_H__
#define __NNTRAINER_HEXKL_ACC_I24_H__

#include "hexkl_acc_tile.h"
#include "hexkl_cvt.h"

/** Three bias blocks, separate from all staging rings, native sources, and
 * HexKL's system-managed config. Output planes live in each ring tile. */
#define HEXKL_ACC_I24_SCRATCH_BYTES 3072u

/** Place scratch at the aligned end of usable VTCM. occupied_end includes
 * all staging rings and optional native packed sources. UINT32_MAX refuses
 * undersized arenas and overlaps without changing their existing layout. */
static inline uint32_t hexkl_acc_i24_scratch_offset(uint32_t arena,
                                                    uint32_t occupied_end) {
  if (arena < HEXKL_ACC_I24_SCRATCH_BYTES)
    return UINT32_MAX;
  const uint32_t off =
    (arena - HEXKL_ACC_I24_SCRATCH_BYTES) & ~(HEXKL_CVT_PLANE_ALIGN - 1u);
  return occupied_end <= off ? off : UINT32_MAX;
}

/** Size two gate/up and two foreground-down accumulator banks at their
 * actual three-plane storage size. Native weight sources follow the banks;
 * the separate convert config must still fit at the aligned arena end. */
static inline int hexkl_acc_i24_banks(uint32_t arena, uint32_t first,
                                      uint32_t tiles, uint32_t native_bytes,
                                      uint32_t *down, uint32_t *end) {
  if (!down || !end || first % HEXKL_CVT_PLANE_ALIGN || tiles < 2u ||
      tiles % 2u)
    return 0;
  const uint64_t bank = (uint64_t)tiles * 3u * HEXKL_CVT_PLANE_BYTES;
  const uint64_t total = first + 4u * bank;
  const uint64_t occupied = total + native_bytes;
  if (occupied > UINT32_MAX ||
      hexkl_acc_i24_scratch_offset(arena, (uint32_t)occupied) == UINT32_MAX)
    return 0;
  *down = (uint32_t)(first + 2u * bank);
  *end = (uint32_t)total;
  return 1;
}

/** The complete u8 * signed-int4 dot must fit before any zero-point or
 * affine correction. Check both signs in uint64 so large K cannot wrap. */
static inline int hexkl_acc_i24_range(uint32_t K, int32_t min_w,
                                      int32_t max_w) {
  if (!K || min_w < -8 || min_w > 0 || max_w < 0 || max_w > 7)
    return 0;
  return (uint64_t)K * 255u * (uint32_t)(-min_w) <= 8388608u &&
         (uint64_t)K * 255u * (uint32_t)max_w <= 8388607u;
}

/** The convert planes are row-major. Keep the vendor read if its runtime
 * layout probe finds any different permutation, even one usable by dequant. */
static inline int hexkl_acc_i24_usable(uint32_t K, int32_t min_w, int32_t max_w,
                                       const hexkl_acc_layout *layout) {
  return layout && layout->usable && layout->base == 0u &&
         layout->row_stride == 32u && hexkl_acc_i24_range(K, min_w, max_w);
}

/** Prepare once per call. scratch is 2 KiB aligned and at least
 * HEXKL_ACC_I24_SCRATCH_BYTES bytes. No vendor config bytes are touched. */
static inline void hexkl_acc_i24_prepare(uint8_t *scratch) {
  const uint16_t scales[3] = {0x6000u, 0x4000u, 0x2000u};
  for (uint32_t p = 0; p < 3u; ++p) {
#if defined(__hexagon__)
    HVX_Vector *v = (HVX_Vector *)(scratch + p * HEXKL_CVT_BLOCK_BYTES);
    const HVX_Vector scale = Q6_V_vsplat_R(scales[p]);
    v[0] = scale;
    v[1] = scale;
    for (uint32_t j = 2u; j < HEXKL_CVT_BLOCK_BYTES / 128u; ++j)
      v[j] = Q6_V_vzero();
#else
    hexkl_cvt_block_set_uniform(scratch + p * HEXKL_CVT_BLOCK_BYTES, scales[p]);
#endif
  }
}

#if defined(__hexagon__)
/** Retains acc and writes its three bytes directly into an existing 8 KiB
 * ring tile. Worker epilogues reconstruct int32 in registers; the last 2 KiB
 * is untouched. Caller must establish hexkl_acc_i24_usable and hold HMX. */
static inline void hexkl_acc_i24_read_planes(const uint8_t *scratch,
                                             uint8_t *lo) {
  uint8_t *mid = lo + HEXKL_CVT_PLANE_BYTES;
  uint8_t *hi = mid + HEXKL_CVT_PLANE_BYTES;
  hexkl_cvt_issue(scratch, lo);
  hexkl_cvt_issue(scratch + HEXKL_CVT_BLOCK_BYTES, mid);
  hexkl_cvt_issue(scratch + 2u * HEXKL_CVT_BLOCK_BYTES, hi);
  /** A load of the final vector waits for the final convert to finish
   * reading acc, so the next tile's clear cannot overtake it. */
  (void)*(const volatile HVX_Vector *)(hi + HEXKL_CVT_PLANE_BYTES - 128u);
}

#endif

#endif /* __NNTRAINER_HEXKL_ACC_I24_H__ */
