// SPDX-License-Identifier: Apache-2.0
/**
 * @file hexkl_hadamard_f16.h
 * @brief Full H44 x H_p online rotation on FP16 HMX, with HVX packing.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 *
 * Each row is a 44 x p matrix (p=16 or 64). Compute X H_p/sqrt(p), then
 * (H44/8) X, then scale by 8/sqrt(44). Both coefficient scales are exact
 * powers of two in binary16. The last scale can move after INT8 packing.
 * H44 is not symmetric: never transpose it or replace it with H64 padding.
 *
 * For p=16 two independent tokens occupy the two halves of a tile, using
 * diag(H16,H16); H44 still mixes all 44 blocks of each complete 704 row.
 * The padded 20 rows are zero, and no token is mixed with another token.
 *
 * Caller owns the HMX lock, an aligned HAD_HMX_SPACE-byte VTCM region,
 * and the foreground worker pool. Only the caller issues HMX instructions.
 * FP16 input/intermediate range and precision apply; this is opt-in.
 */
#ifndef NNTR_HEXKL_HADAMARD_F16_H
#define NNTR_HEXKL_HADAMARD_F16_H

#include "hexkl_hadamard_coeffs.inc"
#include "hexkl_micro.h"
#include "hvx_convert.h"
#include "hvx_quant_u8.h"
#include "hvx_rmsnorm_rows_f32.h"
#include "hvx_tile_f16.h"
#include "hvx_worker_pool.h"
#include <AEEStdErr.h>
#include <hexagon_protos.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#define HAD_HMX_SPACE 335872u
#define HAD_HMX_H44 0u
#define HAD_HMX_H64 8192u
#define HAD_HMX_H16 16384u
#define HAD_HMX_CFG 32768u
/* Two 16-token Up banks (or two 64-token Down banks). The HMX caller
 * reuses one 8 KiB intermediate tile matrix; workers never touch it. */
#define HAD_HMX_DATA 65536u
#define HAD_HMX_TEMP 327680u
#define HAD_HMX_POST_SCALE 1.2060453783110545f
static int had_hmx_init(uint8_t *vb) {
  if (hexkl_micro_hmx_config_size() > HAD_HMX_DATA - HAD_HMX_CFG)
    return AEE_EUNSUPPORTED;
  memcpy(vb + HAD_HMX_H44, had_hmx_h44, sizeof had_hmx_h44);
  memcpy(vb + HAD_HMX_H64, had_hmx_h64, sizeof had_hmx_h64);
  memcpy(vb + HAD_HMX_H16, had_hmx_h16, sizeof had_hmx_h16);
  return hexkl_micro_hmx_setup_acc_read_f16(vb, HAD_HMX_CFG);
}
static inline void had_hmx_clear(void) {
  __asm__ volatile("mxclracc.hf" ::: "memory");
}
static inline void had_hmx_mm(uint8_t *vb, uint32_t a, uint32_t w) {
  __asm__ volatile(
    "{ activation.hf = mxmem(%0,%2)\n weight.hf = mxmem(%1,%3) }" ::"r"(vb + a),
    "r"(vb + w), "r"(0x7fff), "r"(0x780)
    : "memory");
}
static inline void had_hmx_read(uint8_t *vb, uint32_t o) {
  __asm__ volatile("mxmem(%0,%1):after.hf = acc" ::"r"(vb + o), "r"(0)
                   : "memory");
}
typedef struct {
  uint8_t *vb;
  const float *x;
  uint32_t rows, n, groups;
  uint32_t defer_scale;
  float *scale;
  int32_t *zp;
  uint8_t *u8;
  uint32_t bank;
  const float *gamma;
  float eps;
  float *out;
} had_hmx_ctx;
static inline __attribute__((always_inline)) void
had_hmx_pack_group(const had_hmx_ctx *ctx, uint32_t g, const float *in) {
  const uint32_t n = ctx->n, cols = n == 2816 ? 2 : 1, p = n / 44;
  const uint32_t m = g * (n == 704 ? 2 : 1);
  uint8_t *vb = ctx->vb;
  const uint32_t a_off = HAD_HMX_DATA + (ctx->bank + g) * cols * 4096;
  const int pair = n == 704 && m + 1 < ctx->rows;
  const HVX_VectorPred low = Q6_Q_vsetq_R(64);
  const HVX_Vector one = hvx_splat_sf(1.0f);
  for (uint32_t rt = 0; rt < 2; ++rt)
    for (uint32_t ct = 0; ct < cols; ++ct) {
      HVX_Vector *dst = (HVX_Vector *)(vb + a_off + (rt * cols + ct) * 2048);
      for (uint32_t i = 0; i < 16; ++i) {
        uint32_t r = rt * 32 + 2 * i;
        HVX_Vector a = Q6_V_vzero(), b = a;
        if (r < 44) {
          a = hvx_tile_load_u(in + r * p + ct * 32);
          if (n == 704) {
            b = Q6_V_vror_VR(a, 64);
            HVX_Vector ab =
              pair ? hvx_tile_load_u(in + 704 + r * 16) : Q6_V_vzero();
            a = Q6_V_vmux_QVV(low, a, Q6_V_vror_VR(ab, 64));
            b = Q6_V_vmux_QVV(low, b, ab);
          } else
            b = hvx_tile_load_u(in + (r + 1) * p + ct * 32);
        }
        HVX_Vector q0 = Q6_Vqf32_vmpy_VsfVsf(a, one),
                   q1 = Q6_Vqf32_vmpy_VsfVsf(b, one);
        dst[i] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
      }
    }
}
static inline void had_hmx_compute_group(const had_hmx_ctx *ctx, uint32_t g) {
  const uint32_t cols = ctx->n == 2816 ? 2 : 1;
  uint8_t *vb = ctx->vb;
  const uint32_t a_off = HAD_HMX_DATA + (ctx->bank + g) * cols * 4096;
  const uint32_t t_off = HAD_HMX_TEMP;
  /* INT32 GEMM replaces the FP16 bias registers. The read API does not
   * restore them; setup must run at each transition back to FP16. It touches
   * only a small config record. Never depend on its private byte offset. */
  if (g == 0u)
    (void)hexkl_micro_hmx_setup_acc_read_f16(vb, HAD_HMX_CFG);
  for (uint32_t rt = 0; rt < 2; ++rt)
    for (uint32_t ct = 0; ct < cols; ++ct) {
      had_hmx_clear();
      for (uint32_t k = 0; k < cols; ++k)
        had_hmx_mm(vb, a_off + (rt * cols + k) * 2048,
                   (cols == 1 ? HAD_HMX_H16 : HAD_HMX_H64) +
                     (k * cols + ct) * 2048);
      had_hmx_read(vb, t_off + (rt * cols + ct) * 2048);
    }
  for (uint32_t rt = 0; rt < 2; ++rt)
    for (uint32_t ct = 0; ct < cols; ++ct) {
      had_hmx_clear();
      for (uint32_t k = 0; k < 2; ++k)
        had_hmx_mm(vb, HAD_HMX_H44 + (rt * 2 + k) * 2048,
                   t_off + (k * cols + ct) * 2048);
      had_hmx_read(vb, a_off + (rt * cols + ct) * 2048);
    }
}
static inline __attribute__((always_inline)) void
had_hmx_store_group(const had_hmx_ctx *ctx, uint32_t g, float *in) {
  const uint32_t n = ctx->n, cols = n == 2816 ? 2 : 1;
  const uint32_t m = g * (n == 704 ? 2 : 1);
  uint8_t *vb = ctx->vb;
  const uint32_t a_off = HAD_HMX_DATA + (ctx->bank + g) * cols * 4096;
  const int pair = n == 704 && m + 1 < ctx->rows;
  const HVX_VectorPred low = Q6_Q_vsetq_R(64);
  const HVX_Vector norm = hvx_splat_sf(HAD_HMX_POST_SCALE);
  HVX_Vector vmin = Q6_V_vzero(), vmax = vmin;
  const HVX_Vector hf_one = Q6_V_vsplat_R(0x3c003c00);
  for (uint32_t rt = 0; rt < 2; ++rt)
    for (uint32_t ct = 0; ct < cols; ++ct) {
      const HVX_Vector *src =
        (const HVX_Vector *)(vb + a_off + (rt * cols + ct) * 2048);
      for (uint32_t i = 0; i < 16 && rt * 32 + 2 * i < 44; ++i) {
        const uint32_t r = rt * 32 + 2 * i;
        const HVX_VectorPair q = Q6_Wqf32_vmpy_VhfVhf(src[i], hf_one);
        HVX_Vector a = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(q));
        HVX_Vector b = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(q));
        if (!ctx->defer_scale) {
          a = Q6_Vsf_vmpy_VsfVsf(a, norm);
          b = Q6_Vsf_vmpy_VsfVsf(b, norm);
        }
        if (ctx->scale) {
          vmin = Q6_Vsf_vmin_VsfVsf(vmin, Q6_Vsf_vmin_VsfVsf(a, b));
          vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_Vsf_vmax_VsfVsf(a, b));
        }
        if (n == 704) {
          hvx_tile_store_u(in + r * 16,
                           Q6_V_vmux_QVV(low, a, Q6_V_vror_VR(b, 64)));
          if (pair)
            hvx_tile_store_u(in + 704 + r * 16,
                             Q6_V_vmux_QVV(low, Q6_V_vror_VR(a, 64), b));
        } else {
          hvx_tile_store_u(in + r * 64 + ct * 32, a);
          hvx_tile_store_u(in + (r + 1) * 64 + ct * 32, b);
        }
      }
    }
  if (ctx->scale) {
    for (uint32_t rot = n == 704u ? 32u : 64u; rot >= 4u; rot >>= 1) {
      vmin = Q6_Vsf_vmin_VsfVsf(vmin, Q6_V_vror_VR(vmin, rot));
      vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vror_VR(vmax, rot));
    }
    for (uint32_t r = 0; r < 1u + (uint32_t)pair; ++r) {
      uint32_t lb = Q6_R_vextract_VR(vmin, r * 64u),
               hb = Q6_R_vextract_VR(vmax, r * 64u);
      float l, h;
      memcpy(&l, &lb, 4);
      memcpy(&h, &hb, 4);
      float sc = h == l ? 1.0f : (h - l) / 255.0f;
      int32_t z = h == l ? 0 : (int32_t)nearbyintf(-l / sc);
      ctx->scale[m + r] = sc;
      ctx->zp[m + r] = z < 0 ? 0 : z > 255 ? 255 : z;
    }
  }
}
static void had_hmx_pack_worker(uint32_t threads, uint32_t id, void *v) {
  const had_hmx_ctx *ctx = v;
  float normed[2816] __attribute__((aligned(128)));
  for (uint32_t g = ctx->groups * id / threads;
       g < ctx->groups * (id + 1) / threads; ++g) {
    if (ctx->n == 2816u && g + 3u < ctx->groups) {
      uint64_t box =
        ((uint64_t)128 << 32) | ((uint64_t)128 << 16) | (2u * 2816u / 32u);
      Q6_l2fetch_AP((void *)(ctx->x + (size_t)(g + 2u) * 2816u), box);
    }
    const float *in = ctx->x + (size_t)g * (ctx->n == 704u ? 2u : 1u) * ctx->n;
    /* Only Up (2816) supplies gamma; one normalized row stays on this lane. */
    if (ctx->gamma) {
      hvx_rmsnorm_row_f32(in, normed, ctx->gamma, ctx->n, ctx->eps);
      in = normed;
    }
    if (ctx->n == 704) {
      had_hmx_ctx c = *ctx;
      c.n = 704;
      had_hmx_pack_group(&c, g, in);
    } else {
      had_hmx_ctx c = *ctx;
      c.n = 2816;
      had_hmx_pack_group(&c, g, in);
    }
  }
}
static void had_hmx_store_worker(uint32_t threads, uint32_t id, void *v) {
  const had_hmx_ctx *ctx = v;
  float scratch[2816] __attribute__((aligned(128)));
  for (uint32_t g = ctx->groups * id / threads;
       g < ctx->groups * (id + 1) / threads; ++g) {
    const uint32_t m = g * (ctx->n == 704u ? 2u : 1u);
    float *out = ctx->u8 ? scratch : ctx->out + (size_t)m * ctx->n;
    if (ctx->n == 704) {
      had_hmx_ctx c = *ctx;
      c.n = 704;
      had_hmx_store_group(&c, g, out);
    } else {
      had_hmx_ctx c = *ctx;
      c.n = 2816;
      had_hmx_store_group(&c, g, out);
    }
    if (ctx->u8) {
      const uint32_t count = ctx->n == 704u && m + 1 < ctx->rows ? 2u : 1u;
      for (uint32_t r = 0; r < count; ++r) {
        hvx_quant_row_u8_rm(out + (size_t)r * ctx->n, ctx->n, ctx->scale[m + r],
                            ctx->zp[m + r], ctx->u8 + (size_t)(m + r) * ctx->n);
        ctx->scale[m + r] *= HAD_HMX_POST_SCALE;
      }
    }
  }
}
typedef struct {
  had_hmx_ctx store, pack;
  uint32_t have_store, have_pack;
} had_hmx_exchange_ctx;
static void had_hmx_exchange_worker(uint32_t threads, uint32_t id, void *v) {
  had_hmx_exchange_ctx *ctx = v;
  if (ctx->have_store)
    had_hmx_store_worker(threads, id, &ctx->store);
  if (ctx->have_pack)
    had_hmx_pack_worker(threads, id, &ctx->pack);
}
/* Positive rows, n in {704,2816}; gamma is supported only for n=2816.
 * Quantizing call: u8/scale/zp non-NULL, out NULL, input is read-only.
 * Standalone call: out non-NULL (may equal x), u8/scale/zp/gamma NULL.
 * A batch overlaps HMX with the workers storing the previous bank and then
 * packing the next into that bank. The wait precedes every bank reuse. */
static void had_hmx_rows(uint8_t *vb, const float *x, uint32_t rows, uint32_t n,
                         hvx_worker_pool *pool, float *scale, int32_t *zp,
                         uint8_t *u8, const float *gamma, float eps,
                         float *out) {
  const uint32_t bs = n == 704 ? 64 : 16;
  had_hmx_ctx previous = {0};
  const uint32_t first = rows < bs ? rows : bs;
  had_hmx_ctx current = {vb,
                         x,
                         first,
                         n,
                         n == 704 ? (first + 1) / 2 : first,
                         scale != NULL,
                         scale,
                         zp,
                         u8,
                         0u,
                         gamma,
                         eps,
                         out};
  hvx_worker_pool_run(pool, had_hmx_pack_worker, &current, current.groups);
  for (uint32_t m = 0; m < rows; m += bs) {
    const uint32_t remaining = rows - m - current.rows;
    const uint32_t count = remaining < bs ? remaining : bs;
    const uint32_t next_m = m + current.rows;
    had_hmx_ctx next = {vb,
                        x + (size_t)next_m * n,
                        count,
                        n,
                        n == 704 ? (count + 1) / 2 : count,
                        scale != NULL,
                        scale ? scale + next_m : NULL,
                        zp ? zp + next_m : NULL,
                        u8 ? u8 + (size_t)next_m * n : NULL,
                        current.bank ? 0u : (n == 704 ? 32u : 16u),
                        gamma,
                        eps,
                        out ? out + (size_t)next_m * n : NULL};
    if (count) {
      uint64_t box =
        ((uint64_t)128 << 32) | ((uint64_t)128 << 16) | (count * n / 32u);
      Q6_l2fetch_AP((void *)next.x, box);
    }
    had_hmx_exchange_ctx exchange = {previous, next, m != 0u, count != 0u};
    if (exchange.have_store || exchange.have_pack)
      hvx_worker_pool_submit(pool, had_hmx_exchange_worker, &exchange,
                             n == 704 ? 32u : 16u);
    for (uint32_t g = 0; g < current.groups; ++g)
      had_hmx_compute_group(&current, g);
    hvx_worker_pool_wait(pool);
    previous = current;
    current = next;
  }
  hvx_worker_pool_run(pool, had_hmx_store_worker, &previous, previous.groups);
}

#endif
