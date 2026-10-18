// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_add_f32.c
 * @date   03 Aug 2026
 * @brief  DSP-side implementation of the nntr_hvx FastRPC interface
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>

#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_vtcm_mgr.h>

#include "hexkl_micro.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#if __has_include(<HAP_power.h>)
#include <HAP_power.h>
#define NNTR_HVX_HAVE_HAP_POWER 1
#endif

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 lanes per HVX vector. */
/** Kept as an int-casted expression (not the unsigned LANES form used
    elsewhere) because n_vec/i in this file's loops are int. */
#define LANES ((int)(VLEN / sizeof(float)))

/** [#178] Not every image exports it; the lite open then takes no VTCM. */
#pragma weak HAP_query_avail_VTCM
#pragma weak HAP_mem_get_stats

/**
 * @brief [#178] The lite open's VTCM: the largest block the resource manager
 *        has left (HAP_query_avail_VTCM), acquired with no HMX and a 100 ms
 *        timeout; none at all is allowed (vtcm_size 0, feeds that need VTCM
 *        refuse). config_off = vtcm_size: there is no HMX config block.
 */
static void lite_vtcm(nntr_hvx_session *s) {
  unsigned int avail = 0, max_page = 0, n_pages = 0;
  s->open_path = 1;
  s->vtcm_base = NULL;
  s->vtcm_size = 0;
  if (HAP_query_avail_VTCM &&
      HAP_query_avail_VTCM(&avail, &max_page, &n_pages) == 0 && avail > 0) {
    compute_res_attr_t attr;
    if (HAP_compute_res_attr_init(&attr) == 0 &&
        HAP_compute_res_attr_set_vtcm_param(&attr, avail, 0) == 0) {
      s->vtcm_ctx = HAP_compute_res_acquire(&attr, 100000);
      if (s->vtcm_ctx) {
        s->vtcm_base = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
        s->vtcm_size = s->vtcm_base ? (uint32_t)avail : 0u;
      }
    }
  }
  FARF(HIGH, "nntr_hvx_open: lite VTCM avail=%u max_page=%u got=%u ctx=%u",
       avail, max_page, (unsigned)s->vtcm_size, (unsigned)s->vtcm_ctx);
  s->config_off = s->vtcm_size;
}

int nntr_hvx_session_info(remote_handle64 handle, uint32 *res, int resLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  unsigned int avail = 0, max_page = 0, n_pages = 0;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (resLen != 7 && resLen != 8) {
    return AEE_EINVALIDFORMAT;
  }
  if (HAP_query_avail_VTCM) {
    (void)HAP_query_avail_VTCM(&avail, &max_page, &n_pages);
  }
  res[0] = (uint32)s->hmx_locked;
  res[1] = s->vtcm_size;
  res[2] = (uint32)(avail >> 10);
  res[3] = (uint32)(max_page >> 10);
  {
    /** [#132 Part B E5g] the PD's heap in use, KiB (HAP_mem_get_stats; 0
       where the image lacks it) -- the E path's close line prints it */
    struct HAP_mem_stats st;
    res[4] = (HAP_mem_get_stats && HAP_mem_get_stats(&st) == 0)
               ? (uint32)(st.bytes_used >> 10)
               : 0u;
  }
  res[5] = s->open_path;
  res[6] = (uint32)((qurt_hvx_get_units() >> 8) & 0xFF);
  if (resLen == 8) {
    res[7] = s->hmx_fp16_rate;
  }
  return AEE_SUCCESS;
}

/** Release only ephemeral workspace, between completed synchronous calls.
 * The host holds invoke_mutex_ across this RPC and each queue response; an
 * installed token driver may remain idle across independent prompts. */
int nntr_hvx_release_prefill_scratch(remote_handle64 handle, uint64 *bytes) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  const size_t keep_bytes = 64u << 10;
  if (!s || !bytes)
    return AEE_EBADPARM;
  *bytes = 0;
  if (s->moe_scratch.pre_gamma || s->moe_scratch.out_heap ||
      s->moe_scratch.rotated_input || s->moe_scratch.router_act)
    return AEE_EBADSTATE;
  if (s->moe_scratch.cap > keep_bytes) {
    *bytes += s->moe_scratch.cap + 128u;
    hexkl_moe_scratch_free(&s->moe_scratch);
  }
  if ((size_t)s->norm_rows_n * sizeof(float) > keep_bytes) {
    *bytes += (uint64)s->norm_rows_n * sizeof(float);
    free(s->norm_rows);
    s->norm_rows = NULL;
    s->norm_rows_n = 0;
  }
  if ((size_t)s->moe_res_n * sizeof(float) > keep_bytes) {
    *bytes += (uint64)s->moe_res_n * sizeof(float);
    free(s->moe_res);
    s->moe_res = NULL;
    s->moe_res_n = 0;
  }
  return AEE_SUCCESS;
}

int nntr_hvx_open(const char *uri, remote_handle64 *handle) {
  (void)uri;

  nntr_hvx_session *s = (nntr_hvx_session *)calloc(1, sizeof(nntr_hvx_session));
  if (!s) {
    return AEE_ENOMEMORY;
  }

  /** Bits 15:8 hold the number of 128-byte HVX contexts. Checked once for
     the session: every entry in this skel uses HVX and the unit count does
     not change while a session is open. */
  if (((qurt_hvx_get_units() >> 8) & 0xFF) == 0) {
    free(s);
    return AEE_EUNSUPPORTED;
  }

  // hw_init and the HMX lock happen once here, for the session's whole
  // lifetime, instead of per call (doc15 §3/§4) -- every other entry point
  // in this skel reaches vtcm_base/vtcm_size/config_off through the
  // session rather than re-acquiring either.
  // hexkl_micro_hw_init's arity is not stable across hexkl_addon drops: the
  // beta2 header ref_08's docs describe takes three args (the third reports
  // the HMX fp16 throughput rate), an older checkout this skel was first
  // built against took two. The header carries no version macro to branch
  // on, so this defaults to the three-arg form -- what hxkl-beta2 ships --
  // and leaves -DNNTR_HEXKL_HW_INIT_2ARG as the escape hatch for the older
  // one. hmx_fp16_rate stays on the session: the f16 attention entries
  // gate on it (0, which is also what the two-arg form and the lite open
  // leave, means "no fp16 HMX here"). If a future drop changes the arity
  // again, this is the one line to touch.
#ifdef NNTR_HEXKL_HW_INIT_2ARG
  int res = hexkl_micro_hw_init(&s->vtcm_base, &s->vtcm_size);
#else
  int res =
    hexkl_micro_hw_init(&s->vtcm_base, &s->vtcm_size, &s->hmx_fp16_rate);
#endif
  if (res != AEE_SUCCESS) {
    /** [#178] a second session (its own PD) beside one that holds the HMX
       and most of VTCM: open without HMX instead of failing */
    FARF(HIGH, "nntr_hvx_open: hexkl_micro_hw_init 0x%08x: lite open", res);
    lite_vtcm(s);
  } else {
    // config_off depends only on vtcm_size (see hexkl_mm_u8i4_plan), so it
    // is computed once here rather than at every mm_u8i4_layer call.
    const uint32_t config_size = hexkl_micro_hmx_config_size();
    if (s->vtcm_size < config_size) {
      free(s);
      return AEE_ENOMEMORY;
    }
    s->config_off =
      (s->vtcm_size - config_size) & ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1u);

    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
      /* [#178] keep hw_init's VTCM below config_off, no HMX entries */
      FARF(HIGH, "nntr_hvx_open: hexkl_micro_hmx_lock 0x%08x: lite open", res);
      s->open_path = 1;
    } else {
      s->hmx_locked = 1;
      res = hexkl_micro_hmx_setup_acc_read_int32(s->vtcm_base, s->config_off);
      if (res != AEE_SUCCESS) {
        FARF(ERROR, "nntr_hvx_open: setup_acc_read_int32 failed: 0x%08x", res);
        hexkl_micro_hmx_unlock();
        free(s);
        return res;
      }
    }
  }

  // Sized from the real HVX context count (bits 15:8, same decode
  // nntr_hvx_add_f32 uses to check HVX is present at all) minus one, since
  // this session's own FastRPC thread uses an HVX context too when it runs
  // quant's own share of the work.
  const int hvx_units_raw = qurt_hvx_get_units();
  const uint32_t n_hvx = (qurt_hvx_get_units() >> 8) & 0xFFu;
  (void)hvx_units_raw; // used only by FARF below, which may compile to nothing
  FARF(HIGH,
       "nntr_hvx_open: qurt_hvx_get_units()=0x%08x n_hvx=%u pool_workers=%u",
       hvx_units_raw, (unsigned)n_hvx, (unsigned)(n_hvx > 1 ? n_hvx - 1 : 0));
  s->quant_pool = hvx_worker_pool_create(n_hvx > 1 ? n_hvx - 1 : 0);
  if (!s->quant_pool) {
    FARF(ERROR, "nntr_hvx_open: hvx_worker_pool_create failed");
    if (s->hmx_locked) {
      hexkl_micro_hmx_unlock();
    }
    if (s->vtcm_ctx) {
      HAP_compute_res_release(s->vtcm_ctx);
    }
    free(s);
    return AEE_ENOMEMORY;
  }

#ifdef NNTR_HVX_HAVE_HAP_POWER
  /** Vote the DSP's clocks up and its wake latency down, once, for the
   * session's lifetime. This is the DSP half of the measured 90 -> 3,900 us
   * transport spread: without a vote the core power-collapses between
   * FastRPC calls and every call pays the DCVS ramp back up. Best effort by
   * design -- a device that rejects the vote runs exactly as before, only
   * slower to wake, so failures are logged and ignored. The votes die with
   * this PD; close() does not need to unwind them. */
  {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: apptype vote rejected (continuing)");
    }
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v2;
    req.dcvs_v2.dcvs_enable = 1;
    req.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v2.set_latency = 1;
    req.dcvs_v2.latency = 100; /* us of wake latency we are willing to pay */
    req.dcvs_v2.set_dcvs_params = 1;
    req.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_NOM;
    req.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: DCVS vote rejected (continuing)");
    }
#if defined(HAP_POWER_SET_HMX_V2_DEFINED) && !defined(NNTR_HVX_NO_HMX_VOTE)
    /** The HMX clock, separately from the core's: with no clock vote the
       HMX_v2 default is its lowest frequency, and the prefill MoE call is
       HMX-bound (28 of 43.6 ms, doc 57 section 9.35). The maximum corner at
       its highest frequency, the same best-effort policy as above.
       -DNNTR_HVX_NO_HMX_VOTE builds the skel without it. */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX_v2;
    req.hmx_v2.set_power = 0;
    req.hmx_v2.set_clock = 1;
    req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_DISABLE;
    req.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: HMX clock vote rejected (continuing)");
    }
#endif
    /** The bus, separately from the core. The DCVS vote above raises the
       DSP clock, and mm at 17.5 ns a tile says it holds; it says nothing
       about DDR. The MoE layer's weight DMA measured 16-18 GB/s in situ
       (doc 46 section 50.7) against 38.8 for the same arena in isolation,
       and a decode token leaves the SoC mostly idle -- the bus is free to
       clock down between 2 ms bursts. This asks for it at 40 GB/s, roughly
       what the isolated probe reached. Same best-effort policy as above.
       ponytail: an unconditional vote for the session's lifetime, which
       is the measurement; if it is what moves DMA_FIRST, the product
       shape is a vote around the layer call, or a lower figure.
       -DNNTR_HVX_NO_BUS_VOTE (HEX_EXTRA_CFLAGS in build.sh) builds the
       skel without this vote so the DMA probe can measure its worth. */
#ifndef NNTR_HVX_NO_BUS_VOTE
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_mips_bw;
    req.mips_bw.set_bus_bw = 1;
    req.mips_bw.bwBytePerSec = 40000000000ull;
    req.mips_bw.busbwUsagePercentage = 100;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: bus bandwidth vote rejected (continuing)");
    }
#endif
  }
#endif

  *handle = (remote_handle64)s;
  return AEE_SUCCESS;
}

int nntr_hvx_arenas_release(remote_handle64 handle, uint32 *n) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !n) {
    return AEE_EBADPARM;
  }
  /* [#141] The queue thread runs kernels on this session: it stops first. */
  nntr_hvx_dspq_shutdown(s);
  /* [#132 Part B E2] The driver's page; the dspq thread above ran it. */
  nntr_hvx_token_shutdown(s);
  /* [#85] The graph names weight handles, so it goes before the tables. */
  hexkl_graph_free(s->graph);
  s->graph = NULL;
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (s->weights_u8i4.slots[i].in_use) {
      hexkl_weight_u8i4_release(&s->weights_u8i4, i);
    }
  }
  for (uint32_t i = 0; i < HEXKL_MM_U8I8_MAX_WEIGHTS; ++i) {
    if (s->weights_u8i8.slots[i].in_use) {
      hexkl_weight_u8i8_release(&s->weights_u8i8, i);
    }
  }
  /* the slots borrow from the arenas, so they go before the puts */
  nntr_hvx_q4m1_free_all(s);
  *n = 0;
  for (uint32_t i = 0; i < NNTR_HVX_MAX_ARENAS; ++i) {
    *n += s->arenas[i].va != NULL;
  }
  nntr_hvx_arenas_put_all(s);
  return AEE_SUCCESS;
}

int nntr_hvx_close(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  uint32 n = 0;
  if (!s) {
    return AEE_SUCCESS;
  }
  /* [#132 Part B E5i] a no-op when the ARM side ran it before its munmaps */
  (void)nntr_hvx_arenas_release(handle, &n);
  hexkl_moe_scratch_free(&s->moe_scratch);
  hexkl_moe_set_hadamard_signs(NULL, 0u, NULL, 0u);
  free(s->had_sign_in);
  free(s->had_sign_down);
  s->had_sign_in = s->had_sign_down = NULL;
  /* [#81] The attention cache borrows the pool, so it goes first. */
  hvx_attn_m1_free(s->attn_m1);
  s->attn_m1 = NULL;
  hvx_attn_m1_free(s->attn_m1_b);
  s->attn_m1_b = NULL;
  for (uint32_t i = 0; i < HEXKL_KV_TILES_MAX; ++i) {
    if (s->kv_tiles.slots[i].in_use) {
      hexkl_kv_tiles_f16_release(&s->kv_tiles, i);
    }
  }
  for (uint32_t i = 0; i < HEXKL_KV_Q_MAX; ++i) {
    if (s->kv_q.slots[i].in_use) {
      hexkl_kv_q_release(&s->kv_q, i);
    }
  }
  hvx_worker_pool_destroy(s->quant_pool);
  free(s->fc_l2);
  free(s->norm_rows);
  free(s->moe_res);
  if (s->vtcm_ctx) {
    HAP_compute_res_release(s->vtcm_ctx);
  }
  int res = AEE_SUCCESS;
  if (s->hmx_locked) {
    res = hexkl_micro_hmx_unlock();
    if (res != AEE_SUCCESS) {
      FARF(ERROR, "nntr_hvx_close: hexkl_micro_hmx_unlock failed: 0x%08x", res);
    }
  }
  free(s);
  return res;
}

int nntr_hvx_add_f32(remote_handle64 handle, const float *a, int aLen,
                     const float *b, int bLen, float *c, int cLen) {
  (void)handle;

  if (aLen != bLen || aLen != cLen) {
    return AEE_EBADPARM;
  }

  // FastRPC buffers carry no vector alignment guarantee, so the unaligned
  // vector type is what keeps this from faulting on a misaligned input.
  const HVX_UVector *va = (const HVX_UVector *)a;
  const HVX_UVector *vb = (const HVX_UVector *)b;
  HVX_UVector *vc = (HVX_UVector *)c;

  const int n_vec = aLen / LANES;
  for (int i = 0; i < n_vec; ++i) {
    vc[i] = Q6_Vsf_vadd_VsfVsf(va[i], vb[i]);
  }

  // ponytail: scalar tail; a masked vector store would fold it in, add
  // that only if the tail shows up in a profile.
  for (int i = n_vec * LANES; i < aLen; ++i) {
    c[i] = a[i] + b[i];
  }

  return AEE_SUCCESS;
}
