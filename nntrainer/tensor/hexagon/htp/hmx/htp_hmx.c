// SPDX-License-Identifier: Apache-2.0
/**
 * @file	htp_hmx.c
 * @date	18 September 2026
 * @brief	HMX resources of the graph executor: arena layout, lazy
 *		worker-0 lock, accumulator-layout probe, HexKL version
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <stdio.h>
#include <string.h>

#include "hexkl_micro.h"
#include "htp_hmx.h"

#define HTP_HMX_CFG_ALIGN 256u /* HEXKL_HMX_CONFIG_ALIGNMENT */

static uint32_t round_up(uint32_t v, uint32_t a) {
  return (v + a - 1u) & ~(a - 1u);
}

uint32_t htp_hmx_arena_min_bytes(void) {
  /** config (rounded to its alignment) + two result tiles, then one
   * activation alignment of slack so the top-down layout never runs into
   * the alignment of the arena base. */
  return round_up(hexkl_micro_hmx_config_size(), HTP_HMX_CFG_ALIGN) +
         2u * HTP_HMX_ACC_TILE_BYTES + HTP_HMX_ACT_ALIGN;
}

int htp_hmx_arena_init(struct htp_hmx *h, uint8_t *base, uint32_t size) {
  const uint32_t cfg = hexkl_micro_hmx_config_size();

  h->base = NULL;
  if (!base || ((uintptr_t)base % HTP_HMX_ACT_ALIGN) != 0u ||
      size < htp_hmx_arena_min_bytes())
    return 1;
  h->size = size;
  h->cfg_off = (size - cfg) & ~(HTP_HMX_CFG_ALIGN - 1u);
  h->res_off[1] =
    (h->cfg_off - HTP_HMX_ACC_TILE_BYTES) & ~(HTP_HMX_ACT_ALIGN - 1u);
  h->res_off[0] = h->res_off[1] - HTP_HMX_ACC_TILE_BYTES;
  h->free_off = h->res_off[0];
  h->base = base;
  return 0;
}

int htp_hmx_version(char *buf, uint32_t n) {
  int major = 0, minor = 0, patch = 0, hex = 0, rc;
  char prerel[HEXKL_PREREL_STR_LEN];

  memset(prerel, 0, sizeof(prerel));
  rc = hexkl_micro_get_version(&major, &minor, &patch, prerel, &hex);
  if (rc == AEE_SUCCESS)
    snprintf(buf, n, "%d.%d.%d-%s hexagon v%d", major, minor, patch, prerel,
             hex);
  else
    snprintf(buf, n, "hexkl unavailable (rc=%d)", rc);
  return rc;
}

int htp_hmx_worker_acquire(struct htp_hmx *h, int wid) {
  int rc;

  if (wid != 0 || !h->base)
    return 1;
  if (!h->locked) {
    h->lock_tried = 1;
    h->lock_rc = HAP_compute_res_hmx_lock(h->ctx_id);
    if (h->lock_rc != 0)
      return h->lock_rc;
    h->locked = 1;
  }
  if (!h->acc_setup) {
    rc = hexkl_micro_hmx_setup_acc_read_int32(h->base, h->cfg_off);
    if (rc != AEE_SUCCESS)
      return rc;
    h->acc_setup = 1;
  }
  return 0;
}

void htp_hmx_release_job(void *arg, int wid, int nw) {
  struct htp_hmx *h = arg;

  (void)nw;
  if (wid != 0 || !h->locked)
    return;
  (void)HAP_compute_res_hmx_unlock(h->ctx_id);
  h->locked = 0;
  h->acc_setup = 0;
}

const hexkl_acc_layout *htp_hmx_acc_layout(struct htp_hmx *h) {
  return hexkl_acc_layout_get(h->base, h->res_off[0]);
}
