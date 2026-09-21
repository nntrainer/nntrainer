// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_hmx.c
 * @date	18 September 2026
 * @brief	Hexagon-sim probe of the HMX path (issue #65 S0): a header-only
 *		graph session acquires VTCM + HMX and splits the arena; on
 *		worker 0 (the only HMX issuer, under its lazy lock) the test
 *		(1) probes the int32 accumulator tile layout (hexkl_acc_tile),
 *		(2) runs one 64x32 (u8) x 32x32 (i4) micro-mm against a scalar
 *		int32 reference, read both in place through the probed layout
 *		and through HexKL's vendor copy, (3) relocates the baked WH
 *		weight tile VTCM -> DDR -> DMA -> VTCM and re-runs the mm on
 *		the copy, and (4) bakes three nibble-field ramps through
 *		hexkl_micro_hmx_rm_to_wh_i4 to print the exact WH permutation
 *		(destination nibble -> source (k, n)) that S3's HVX reader of
 *		WH tiles needs. Built without HTP_HMX the test reports itself
 *		skipped with rc 2 so it is never mistaken for a pass.
 * @see		https://github.com/nnstreamer/nntrainer
 * @see		docs/plans/65-w4-htp-port.md section 4 (S0) and 5
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_graph.h"
#include "sim_test_util.h"

#if HTP_HMX
#include <AEEStdErr.h>

#include "dma-queue.h"
#include "hexkl_micro.h"
#include "htp_hmx.h"

#define ROWS HEXKL_ACC_TILE_ROWS /* 64 activation rows */
#define KK 32u                   /* one k tile */
#define NN HEXKL_ACC_TILE_COLS   /* 32 output columns */
#define TILE_N (ROWS * NN)       /* 2048 int32 in one result tile */
#define W_N (KK * NN)            /* 1024 int4 in one weight tile */

/** Arena offsets the probe uses inside [0, free_off). */
#define OFF_ACT 0u
#define OFF_W (HTP_HMX_ACT_TILE_BYTES)
#define OFF_W_RELOC (2u * HTP_HMX_ACT_TILE_BYTES)
#define OFF_NEED (3u * HTP_HMX_ACT_TILE_BYTES)

struct hmx_job {
  struct htp_exec_ctx *c;
  uint8_t *ddr_tile; /* 512 B, 128-aligned */
  int step;          /* first failing step (0 = all ok) */
  int rc;            /* its rc */
  int acq_rc;
  int usable;
  uint32_t base, stride;
  uint8_t act[ROWS * KK];
  int8_t w[KK * NN]; /* [k][n], values in [-8, 7] */
  int32_t ref[TILE_N];
  int32_t got_copy[TILE_N];
  int32_t got_inplace[TILE_N];
  int32_t got_reloc[TILE_N];
  int reloc_bytes_same;
  uint16_t perm[W_N]; /* destination nibble -> source index k*32+n */
  int perm_bijective;
};

static void fail(struct hmx_job *j, int step, int rc) {
  if (!j->step) {
    j->step = step;
    j->rc = rc;
  }
}

/** rm_to_wh_i4 packs the low nibble of each int8: encode a 4-bit field f
 * as the int8 whose low nibble is f (f >= 8 as f - 16). */
static int8_t nib_to_i8(unsigned f) {
  return (int8_t)(f >= 8u ? (int)f - 16 : (int)f);
}

static int mm_one(struct hmx_job *j, uint32_t w_off, uint32_t res,
                  int32_t *out) {
  struct htp_hmx *h = &j->c->hmx;
  int rc;
  hexkl_micro_hmx_acc_clear_int32();
  rc = hexkl_micro_hmx_mm_u8i4(h->base, OFF_ACT, w_off);
  if (rc != AEE_SUCCESS)
    return rc;
  rc = hexkl_micro_hmx_acc_read_int32(h->base, h->cfg_off, h->res_off[res]);
  if (rc != AEE_SUCCESS)
    return rc;
  return hexkl_micro_hmx_copy_32b_to_submatrix(h->base, h->res_off[res], out,
                                               0u, 0u, ROWS, NN);
}

static void hmx_job(void *arg, int wid, int nw) {
  struct hmx_job *j = arg;
  struct htp_hmx *h = &j->c->hmx;
  const hexkl_acc_layout *lay;
  int rc;
  uint32_t r, c, b, d;

  (void)nw;
  if (wid != 0)
    return;

  /* 1. lazy lock + acc-read setup, then the layout probe (clobbers res 0) */
  j->acq_rc = htp_hmx_worker_acquire(h, wid);
  if (j->acq_rc) {
    fail(j, 1, j->acq_rc);
    return;
  }
  lay = htp_hmx_acc_layout(h);
  j->usable = lay->usable;
  j->base = lay->base;
  j->stride = lay->row_stride;

  /* 2. one micro-mm: activation flat row-major u8, weight baked to WH */
  memcpy(h->base + OFF_ACT, j->act, sizeof(j->act));
  rc = hexkl_micro_hmx_rm_to_wh_i4(h->base, OFF_W, j->w, 0u, 0u, NN);
  if (rc != AEE_SUCCESS) {
    fail(j, 2, rc);
    return;
  }
  rc = mm_one(j, OFF_W, 0, j->got_copy);
  if (rc != AEE_SUCCESS) {
    fail(j, 2, rc);
    return;
  }
  if (lay->usable) {
    const int32_t *tile =
      (const int32_t *)(h->base + h->res_off[0]) + lay->base;
    for (r = 0; r < ROWS; ++r)
      for (c = 0; c < NN; ++c)
        j->got_inplace[r * NN + c] = tile[r * lay->row_stride + c];
  }

  /** 3. WH relocate round trip: VTCM -> DDR (memcpy) -> VTCM (DMA, worker
   * 0's graph-lifetime queue) -> the same mm on the copy */
  memcpy(j->ddr_tile, h->base + OFF_W, HTP_HMX_W4_TILE_BYTES);
  memset(h->base + OFF_W_RELOC, 0, HTP_HMX_W4_TILE_BYTES);
  {
    dma_queue_t q = j->c->dmaq[wid];
    if (!dma_queue_push_ddr_to_vtcm(
          q, dma_make_ptr(h->base + OFF_W_RELOC, j->ddr_tile),
          HTP_HMX_W4_TILE_BYTES, HTP_HMX_W4_TILE_BYTES, 1u)) {
      fail(j, 3, -1);
      return;
    }
    (void)dma_queue_pop(q);
  }
  j->reloc_bytes_same =
    memcmp(h->base + OFF_W_RELOC, j->ddr_tile, HTP_HMX_W4_TILE_BYTES) == 0;
  rc = mm_one(j, OFF_W_RELOC, 1, j->got_reloc);
  if (rc != AEE_SUCCESS) {
    fail(j, 3, rc);
    return;
  }

  /** 4. WH permutation: three ramps carry bits [0,4), [4,8), [8,10) of the
   * source index k*32+n; the destination nibble (byte*2 + high) then
   * reads back the full source index. */
  memset(j->perm, 0, sizeof(j->perm));
  for (b = 0; b < 3u; ++b) {
    int8_t src[KK * NN];
    const uint8_t *wh = h->base + OFF_W;
    for (r = 0; r < KK * NN; ++r)
      src[r] = nib_to_i8((r >> (4u * b)) & 0xFu);
    rc = hexkl_micro_hmx_rm_to_wh_i4(h->base, OFF_W, src, 0u, 0u, NN);
    if (rc != AEE_SUCCESS) {
      fail(j, 4, rc);
      return;
    }
    for (d = 0; d < W_N; ++d) {
      const unsigned nib =
        (d & 1u) ? (wh[d >> 1] >> 4) & 0xFu : wh[d >> 1] & 0xFu;
      j->perm[d] |= (uint16_t)(nib << (4u * b));
    }
  }
  {
    uint8_t seen[W_N];
    memset(seen, 0, sizeof(seen));
    j->perm_bijective = 1;
    for (d = 0; d < W_N; ++d) {
      if (j->perm[d] >= W_N || seen[j->perm[d]]) {
        j->perm_bijective = 0;
        break;
      }
      seen[j->perm[d]] = 1;
    }
  }
}

static int cmp_i32(const char *tag, const int32_t *ref, const int32_t *got,
                   uint32_t n) {
  uint32_t i, wi = 0;
  int64_t worst = 0;
  for (i = 0; i < n; ++i) {
    int64_t d = (int64_t)ref[i] - got[i];
    if (d < 0)
      d = -d;
    if (d > worst) {
      worst = d;
      wi = i;
    }
  }
  printf("SIM_TEST %s STAT max_abs=%lld\n", tag, (long long)worst);
  if (worst) {
    printf("SIM_TEST %s FAIL i=%u ref=%d got=%d\n", tag, (unsigned)wi,
           (int)ref[wi], (int)got[wi]);
    return 1;
  }
  return 0;
}

/** dst = P(src) over GF(2): if every source bit lands on exactly one
 * destination bit the WH bake is a pure bit permutation of the index and
 * S3 can express it as a fixed intra-tile shuffle. */
static void print_perm(const struct hmx_job *j) {
  uint16_t inv[W_N];
  uint32_t d, s, bit;
  int linear = 1;
  for (d = 0; d < W_N; ++d)
    inv[j->perm[d]] = (uint16_t)d;
  for (d = 0; d < W_N; d += 32u) {
    printf("SIM_TEST hmx wh_perm dst=%4u src:", (unsigned)d);
    for (s = 0; s < 32u; ++s)
      printf(" %u", (unsigned)j->perm[d + s]);
    printf("\n");
  }
  for (s = 0; s < W_N && linear; ++s) {
    uint32_t x = inv[0], t = s;
    for (bit = 0; bit < 10u; ++bit)
      if (t & (1u << bit))
        x ^= inv[1u << bit] ^ inv[0];
    if (x != inv[s])
      linear = 0;
  }
  if (linear && inv[0] == 0u) {
    int pure = 1;
    for (bit = 0; bit < 10u; ++bit)
      if (inv[1u << bit] & (inv[1u << bit] - 1u))
        pure = 0;
    printf("SIM_TEST hmx wh_perm %s: src bit -> dst bit",
           pure ? "is a bit permutation" : "is GF(2)-linear");
    for (bit = 0; bit < 10u; ++bit)
      printf(" %s%u:0x%x", bit < 5u ? "n" : "k",
             (unsigned)(bit < 5u ? bit : bit - 5u), (unsigned)inv[1u << bit]);
    printf("\n");
  } else {
    printf(
      "SIM_TEST hmx wh_perm is not GF(2)-linear in the index (inv[0]=%u)\n",
      (unsigned)inv[0]);
  }
}

int test_hmx(void) {
  struct htp_graph g;
  struct hmx_job *j;
  uint8_t ol[64], w[128] __attribute__((aligned(128))),
    kv[128] __attribute__((aligned(128))),
    act[128] __attribute__((aligned(128)));
  struct nntr_htp_oplist_header *h =
    (struct nntr_htp_oplist_header *)(void *)ol;
  int rc = 0;
  uint32_t r, c, k;

  /* A header-only op-list: init acquires the resources, runs no op. */
  memset(ol, 0, sizeof(ol));
  h->magic = NNTR_HTP_OPLIST_MAGIC;
  h->version = NNTR_HTP_ABI_VERSION;
  h->n_ops = 0u;
  h->n_layers = 1u;
  h->n_heads = 1u;
  h->n_kv_heads = 1u;
  h->head_dim = 128u;
  h->hidden = 128u;
  h->ffn = 128u;
  h->vocab = 64u;
  h->max_seq = 8u;
  h->max_chunk = 1u;
  h->weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;
  memset(w, 0, sizeof(w));
  memset(kv, 0, sizeof(kv));
  memset(act, 0, sizeof(act));
  rc = htp_graph_init_ex(&g, ol, sizeof(ol), w, sizeof(w), kv, sizeof(kv), act,
                         sizeof(act), 0);
  if (rc) {
    printf("SIM_TEST hmx FAIL graph init rc=%d\n", rc);
    return 1;
  }
  printf(
    "SIM_TEST hmx hexkl=\"%s\" hmx=%d ctx=%u vtcm_hvx=%u arena=%u cfg_off=%u "
    "res0=%u res1=%u free=%u workers=%d\n",
    g.ctx.hmx.version, g.ctx.hmx.base ? 1 : 0, g.ctx.hmx.ctx_id,
    (unsigned)g.ctx.vtcm_size, (unsigned)g.ctx.hmx.size,
    (unsigned)g.ctx.hmx.cfg_off, (unsigned)g.ctx.hmx.res_off[0],
    (unsigned)g.ctx.hmx.res_off[1], (unsigned)g.ctx.hmx.free_off,
    wp_size(g.ctx.pool));
  if (!g.ctx.hmx.base || !g.ctx.dmaq) {
    printf("SIM_TEST hmx FAIL no HMX arena or DMA queue in this session\n");
    htp_graph_destroy(&g);
    return 1;
  }
  if (g.ctx.hmx.free_off < OFF_NEED) {
    printf("SIM_TEST hmx FAIL arena free region %u < %u\n",
           (unsigned)g.ctx.hmx.free_off, (unsigned)OFF_NEED);
    htp_graph_destroy(&g);
    return 1;
  }

  j = calloc(1, sizeof(*j));
  if (j)
    j->ddr_tile = memalign(128, HTP_HMX_W4_TILE_BYTES);
  if (!j || !j->ddr_tile) {
    printf("SIM_TEST hmx FAIL alloc\n");
    free(j);
    htp_graph_destroy(&g);
    return 1;
  }
  j->c = &g.ctx;
  for (r = 0; r < ROWS * KK; ++r)
    j->act[r] = (uint8_t)((frand() * 0.5f + 0.5f) * 255.f);
  for (k = 0; k < KK * NN; ++k)
    j->w[k] = (int8_t)((int)((frand() * 0.5f + 0.5f) * 16.f) - 8);
  for (k = 0; k < KK * NN; ++k)
    if (j->w[k] > 7)
      j->w[k] = 7;
  for (r = 0; r < ROWS; ++r)
    for (c = 0; c < NN; ++c) {
      int32_t acc = 0;
      for (k = 0; k < KK; ++k)
        acc += (int32_t)j->act[r * KK + k] * (int32_t)j->w[k * NN + c];
      j->ref[r * NN + c] = acc;
    }

  wp_run(g.ctx.pool, hmx_job, j);

  printf("SIM_TEST hmx lock rc=%d acc_layout usable=%d base=%u row_stride=%u\n",
         j->acq_rc, j->usable, (unsigned)j->base, (unsigned)j->stride);
  if (j->step) {
    printf("SIM_TEST hmx FAIL step=%d rc=%d\n", j->step, j->rc);
    rc = 1;
    goto out;
  }
  if (!j->usable) {
    printf("SIM_TEST hmx FAIL acc layout not usable in place\n");
    rc = 1;
  }
  rc |= cmp_i32("hmx_mm_copy", j->ref, j->got_copy, TILE_N);
  if (j->usable)
    rc |= cmp_i32("hmx_mm_inplace", j->ref, j->got_inplace, TILE_N);
  printf("SIM_TEST hmx wh_reloc bytes_same=%d\n", j->reloc_bytes_same);
  if (!j->reloc_bytes_same)
    rc = 1;
  rc |= cmp_i32("hmx_mm_reloc", j->ref, j->got_reloc, TILE_N);
  printf("SIM_TEST hmx wh_perm bijective=%d\n", j->perm_bijective);
  if (!j->perm_bijective)
    rc = 1;
  else
    print_perm(j);

out : {
  const int locked = g.ctx.hmx.locked;
  htp_graph_destroy(&g); /* runs the worker-0 unlock job, then releases */
  printf("SIM_TEST hmx lock held=%d released by destroy\n", locked);
}
  free(j->ddr_tile);
  free(j);
  if (rc)
    return 1;
  printf("SIM_TEST hmx PASS\n");
  return 0;
}

#else /* !HTP_HMX */

int test_hmx(void) {
  printf("SIM_TEST hmx SKIP: built without HTP_HMX (no HexKL addon on the "
         "mount); not a pass\n");
  return 2;
}

#endif
