// SPDX-License-Identifier: Apache-2.0
/**
 * @file   swap_host_check.c
 * @brief  weight_swap_u8i4_arena (doc 52 section 10.12) on the host: the
 *         real entry point in nntr_hvx_mm_u8i4.c over the real weight
 *         registry (hexkl_mm_u8i4_dma.c), with an arena that is plain
 *         aligned memory set straight into the session instead of an
 *         HAP_mmap'd ION buffer. What it checks is the bookkeeping the host
 *         relies on: the new pair registered in place with the given
 *         scales and column sums and a zero bias, the old pair released
 *         only after, and on every error nothing released and nothing left
 *         registered.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

/* Referenced by nntr_hvx_mm_u8i4.c's other entry points, none of which
   this check calls. Declared without their headers on purpose: C links
   them by name, and reaching one is a bug in the check. */
#define NOT_HERE(fn)                                                           \
  void fn(void) {                                                              \
    fprintf(stderr, "swap_host_check: %s called\n", #fn);                      \
    abort();                                                                   \
  }
NOT_HERE(hexkl_conv_block_run)
NOT_HERE(hexkl_mm_u8i4_bake_weights)
NOT_HERE(hexkl_mm_u8i4_plan)
NOT_HERE(hexkl_mm_u8i4_run)
/* These two are declared by headers the session struct pulls in, so they
   get their real signatures. */
int hexkl_mm_u8i4_moe_layer_run(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, uint32_t inter, uint32_t N_out,
  uint32_t n_experts, const uint32_t *h_gate_up, const uint32_t *h_down,
  const uint32_t *row_index, const uint32_t *row_count, const float *row_weight,
  const float *act_f32, float *out_f32, hvx_worker_pool *pool,
  hexkl_moe_scratch *scratch) {
  fprintf(stderr, "swap_host_check: hexkl_mm_u8i4_moe_layer_run called\n");
  abort();
}
void hexkl_probe_reset(int enable) {
  (void)enable;
  fprintf(stderr, "swap_host_check: hexkl_probe_reset called\n");
  abort();
}

#define NONE 0xFFFFFFFFu
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "swap_host_check FAILED %s:%d: %s\n", __FILE__,          \
              __LINE__, #cond);                                                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static nntr_hvx_session g_s; /* big: the weight tables live inline */

static unsigned live(void) {
  unsigned n = 0;
  for (unsigned i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i)
    n += g_s.weights_u8i4.slots[i].in_use ? 1u : 0u;
  return n;
}

enum { K = 64, INTER = 32, NOUT = 64, NGU = 2 * INTER };
enum { GU_BYTES = (K / 32) * (NGU / 32) * 512 };      /* 2048 */
enum { DN_BYTES = (INTER / 32) * (NOUT / 32) * 512 }; /* 1024 */
enum { ARENA = 4 * 4096 };

static float gu_s[NGU], dn_s[NOUT];
static int32_t gu_c[NGU], dn_c[NOUT];

static int swap(uint32_t og, uint32_t od, uint32_t off_gu, uint32_t off_dn,
                uint32_t *g, uint32_t *d) {
  return nntr_hvx_weight_swap_u8i4_arena(
    (remote_handle64)(uintptr_t)&g_s, og, od, K, INTER, NOUT, 0, off_gu, off_dn,
    gu_s, NGU, gu_c, NGU, dn_s, NOUT, dn_c, NOUT, g, d);
}

int main(void) {
  static uint8_t va[ARENA] __attribute__((aligned(4096)));
  g_s.vtcm_size = 8u << 20;
  g_s.arenas[0].fd = 3;
  g_s.arenas[0].va = va;
  g_s.arenas[0].bytes = ARENA;
  for (int i = 0; i < NGU; ++i) {
    gu_s[i] = 0.5f + (float)i;
    gu_c[i] = i - 7;
  }
  for (int i = 0; i < NOUT; ++i) {
    dn_s[i] = 2.0f + (float)i;
    dn_c[i] = 3 - i;
  }

  /* 1. Nothing to release: a plain two-weight registration in place. */
  uint32_t g = NONE, d = NONE;
  CHECK(swap(NONE, NONE, 0, 4096, &g, &d) == AEE_SUCCESS);
  CHECK(live() == 2);
  const hexkl_weight_u8i4 *wg = &g_s.weights_u8i4.slots[g];
  const hexkl_weight_u8i4 *wd = &g_s.weights_u8i4.slots[d];
  CHECK(wg->borrowed && wg->wh_bytes == va && wg->K == K && wg->N == NGU);
  CHECK(wd->borrowed && wd->wh_bytes == va + 4096 && wd->K == INTER &&
        wd->N == NOUT);
  CHECK(memcmp(wg->w_scale, gu_s, sizeof(gu_s)) == 0);
  CHECK(memcmp(wg->colsum_w, gu_c, sizeof(gu_c)) == 0);
  CHECK(memcmp(wd->w_scale, dn_s, sizeof(dn_s)) == 0);
  CHECK(memcmp(wd->colsum_w, dn_c, sizeof(dn_c)) == 0);
  for (int i = 0; i < NGU; ++i)
    CHECK(wg->bias[i] == 0.0f);

  /* 2. The swap proper: new pair in, old pair out, in one call. */
  uint32_t g2 = NONE, d2 = NONE;
  CHECK(swap(g, d, 8192, 12288, &g2, &d2) == AEE_SUCCESS);
  CHECK(live() == 2);
  CHECK(!g_s.weights_u8i4.slots[g].in_use || g == g2 || g == d2);
  CHECK(!g_s.weights_u8i4.slots[d].in_use || d == g2 || d == d2);
  CHECK(g_s.weights_u8i4.slots[g2].wh_bytes == va + 8192);
  CHECK(g_s.weights_u8i4.slots[d2].wh_bytes == va + 12288);

  /* 3. Refusals before anything is registered or released. */
  uint32_t x = 1234, y = 1234;
  CHECK(swap(g2, g2, 0, 4096, &x, &y) == AEE_EBADPARM);   /* same twice */
  CHECK(swap(g2, NONE, 0, 4096, &x, &y) == AEE_EBADPARM); /* half a pair */
  CHECK(swap(g2, 2000, 0, 4096, &x, &y) == AEE_EBADPARM); /* not live */
  CHECK(nntr_hvx_weight_swap_u8i4_arena(
          (remote_handle64)(uintptr_t)&g_s, g2, d2, K, INTER, NOUT, 0, 0, 4096,
          gu_s, NGU - 1, gu_c, NGU, dn_s, NOUT, dn_c, NOUT, &x,
          &y) == AEE_EBADPARM); /* wrong length */
  CHECK(live() == 2 && x == 1234 && y == 1234);
  CHECK(g_s.weights_u8i4.slots[g2].in_use && g_s.weights_u8i4.slots[d2].in_use);

  /* 4. gate_up registers, down does not (past the arena): gate_up is
        rolled back and the old pair stays live. */
  CHECK(swap(g2, d2, 0, ARENA - 512, &x, &y) != AEE_SUCCESS);
  CHECK(live() == 2);
  CHECK(g_s.weights_u8i4.slots[g2].in_use && g_s.weights_u8i4.slots[d2].in_use);

  /* 5. Unaligned gate_up: refused by the registry, nothing changes. */
  CHECK(swap(g2, d2, 100, 4096, &x, &y) != AEE_SUCCESS);
  CHECK(live() == 2);

  printf("WEIGHT SWAP OK\n");
  return 0;
}
