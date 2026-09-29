// SPDX-License-Identifier: Apache-2.0
/**
 * @file	htp_hmx.h
 * @date	18 September 2026
 * @brief	HMX resources of the graph executor (issue #65 S0): the VTCM
 *		arena HexKL's micro API works in, the lazy per-thread HMX lock
 *		on worker 0, the accumulator-layout probe and the HexKL
 *		version string. Compiled only with -DHTP_HMX=1 (build scripts
 *		set it when libhexkl_micro.a is on the addon mount).
 * @see		https://github.com/nnstreamer/nntrainer
 * @see		docs/plans/65-w4-htp-port.md section 2.2 and 3.1
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_HMX_H
#define NNTR_HTP_HMX_H

#include <stdint.h>

#include "hexkl_acc_tile.h"

/** Alignment HexKL needs for an activation tile and for an int32 result
 * tile (HEXKL_HMX_ACTIVATION_ALIGNMENT); the arena base and every tile
 * offset below keep it. */
#define HTP_HMX_ACT_ALIGN 2048u
/** One 64x32 int32 accumulator tile. */
#define HTP_HMX_ACC_TILE_BYTES (HEXKL_ACC_TILE_ROWS * HEXKL_ACC_TILE_COLS * 4u)
/** One 32x32 packed int4 weight tile in the WH layout. */
#define HTP_HMX_W4_TILE_BYTES 512u
/** One 64x32 uint8 activation tile (flat row-major). */
#define HTP_HMX_ACT_TILE_BYTES 2048u

/**
 * @brief HMX state of one graph session. Lives in htp_exec_ctx; zeroed by
 *        htp_graph_init_ex, laid out by htp_hmx_arena_init once the VTCM
 *        grant is known. base == NULL means "no HMX in this session" (no
 *        HexKL library, the part has no HMX, or the compute-res manager
 *        refused the HMX attribute); a W4A8 op-list is then rejected at
 *        init (S2), never run silently on HVX.
 */
struct htp_hmx {
  uint8_t *base;       /**< arena start in VTCM, HTP_HMX_ACT_ALIGN aligned */
  uint32_t size;       /**< arena bytes */
  uint32_t cfg_off;    /**< HexKL config region (256-aligned, at the top) */
  uint32_t res_off[2]; /**< two int32 result tiles below the config */
  uint32_t free_off;   /**< [0, free_off) is unassigned: S2's activation
                          tiles and the two weight strips go there */
  unsigned ctx_id;     /**< compute-res context that holds the HMX grant */
  int locked;          /**< worker 0 holds HAP_compute_res_hmx_lock */
  int lock_rc;         /**< the lock call's rc (0 ok), valid once tried */
  int lock_tried;      /**< the lazy lock ran at least once */
  int acc_setup;       /**< hexkl_micro_hmx_setup_acc_read_int32 done */
  char version[48];    /**< "1.0.0-beta1 hexagon v79" from HexKL */
};

/**
 * @brief Bytes the fixed part of the arena needs (config + two result
 *        tiles + alignment slack): the 4 MB fallback carves this much off
 *        the HVX region.
 */
uint32_t htp_hmx_arena_min_bytes(void);

/**
 * @brief Lay the arena out over [base, base + size): config at the top,
 *        result tiles below it, the rest free. Runs on any thread (no HMX
 *        instruction is issued).
 * @return 0 ok, 1 when size < htp_hmx_arena_min_bytes() or base is not
 *         HTP_HMX_ACT_ALIGN aligned (h->base stays NULL)
 */
int htp_hmx_arena_init(struct htp_hmx *h, uint8_t *base, uint32_t size);

/**
 * @brief Fill buf with HexKL's version ("major.minor.patch-prerel hexagon
 *        vNN"), or "hexkl unavailable (rc=N)".
 * @return hexkl_micro_get_version's rc
 */
int htp_hmx_version(char *buf, uint32_t n);

/**
 * @brief Worker-0 entry of every HMX section: takes the HMX lock for this
 *        thread on the first call (HEXAGON.md section 7: HMX instructions
 *        are issued by worker 0 only; the lock is per thread, so it cannot
 *        be taken on the RPC thread at init) and sets the int32
 *        accumulator read up once. Must run inside a wp_run job.
 * @return 0 ok; 1 not worker 0 or no arena; else the HAP / HexKL rc
 */
int htp_hmx_worker_acquire(struct htp_hmx *h, int wid);

/**
 * @brief wp_run job that releases the lock from the thread that took it
 *        (htp_graph_destroy runs it before HAP_compute_res_release).
 */
void htp_hmx_release_job(void *arg, int wid, int nw);

/**
 * @brief Probe the int32 result tile's layout (hexkl_acc_tile.c) on result
 *        tile 0; clobbers that tile. Worker 0 after htp_hmx_worker_acquire.
 */
const hexkl_acc_layout *htp_hmx_acc_layout(struct htp_hmx *h);

#endif /* NNTR_HTP_HMX_H */
