// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   nntr_hvx_session.h
 * @date   06 Aug 2026
 * @brief  Per-session HMX/VTCM state and the u8i4 weight registry
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTR_HVX_SESSION_H__
#define __NNTR_HVX_SESSION_H__

#include <stdint.h>

#include "hexkl_graph.h"
#include "hexkl_mm_u8i4_dma.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_mm_u8i8_dma.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_worker_pool.h"

/**
 * @brief State held for the lifetime of one nntr_hvx_open()/close() pair.
 *
 * hw_init and the HMX lock happen once in open() rather than per call (doc15
 * §3/§4): every FastRPC entry point in this file reaches its VTCM arena and
 * weight table through the session instead of re-acquiring either. This
 * assumes one open session at a time -- hexkl_micro_hw_init is a singleton
 * DSP resource, so a second concurrent open would contend for the same VTCM
 * arena and HMX lock. nntrainer opens exactly one HTP session per process,
 * so that is not a real constraint today; it would need addressing before
 * this skel served more than one client process at once.
 * [#178] A reserved second session is a second PD with its own copy of this
 * struct; when hw_init or the HMX lock is refused there, open() takes the
 * lite path (open_path 1: no HMX, VTCM from HAP_compute_res, maybe none).
 */
/** @brief How many host arenas one session can have mapped at once. The
 *  whole-model plan needs 3.9 GB in chunks of at most 1 GiB (rpcmem/ION
 *  allocation size, doc 45 Gate 0b), so four -- but the DSP refused the
 *  fourth 1 GiB mapping (doc 46 section 39), and the host now halves its
 *  request down to 64 MiB to find where that ceiling actually falls. The
 *  tail of a halving sequence is what needs the room: this is a table of
 *  three-word structs, so the slots cost nothing next to being wrong. */
#define NNTR_HVX_MAX_ARENAS 32

/** @brief [#132 PR 2] How many Q4M1 weights (q4m1_register) one session
 *  holds at once. [#132 Part B] 80 (the #178 probe's): the resident graph
 *  holds the whole FC set -- LFM2.5's 66 weights plus the lm_head's 8
 *  slices of 16384 rows -- in one session; the slots are 12 bytes each,
 *  the weights themselves are on the heap. */
#define NNTR_HVX_Q4M1_SLOTS 80

/** @brief One Q4M1 weight on the DSP heap (memalign 128). w NULL = free. */
typedef struct {
  uint8_t *w;
  uint32_t K, N;
} nntr_hvx_q4m1_slot;

/** @brief One host rpcmem buffer as the DSP sees it. va NULL means free. */
typedef struct {
  int fd;
  uint8_t *va;
  uint32_t bytes;
} nntr_hvx_arena;

typedef struct {
  uint8_t *vtcm_base;
  uint32_t vtcm_size;
  uint32_t config_off; /**< session-constant: depends only on vtcm_size */
  int hmx_locked;      /**< close() only unlocks/finalizes what open() set up;
                            0 after the lite open, where every HMX entry is
                            AEE_EUNSUPPORTED */
  uint32_t open_path;  /**< [#178] 0 the full open (hw_init + HMX lock), 1 the
                            lite open (no HMX; VTCM from HAP_compute_res) */
  uint32_t vtcm_ctx;   /**< [#178] the lite open's HAP_compute_res context
                            (0 = none), released in close() */
  hexkl_weight_u8i4_table weights_u8i4;
  hexkl_weight_u8i8_table weights_u8i8;
  hvx_worker_pool *quant_pool; /**< sized from the HVX unit count in open() */
  nntr_hvx_arena arenas[NNTR_HVX_MAX_ARENAS];
  hexkl_moe_scratch moe_scratch; /**< the MoE layer call's heap scratch,
                                      grown on demand, freed in close() */
  uint32_t moe_flags;       /**< HEXKL_MOE_FLAG_* bits from moe_set_opts; the
                                 session is calloc'd, so 0 = the HMX loop */
  hexkl_graph *graph;       /**< [#85] the decode op table from graph_init; NULL
                                 = none. Freed in close() before the weight
                                 tables, since its MoE ops name their handles */
  hvx_attn_m1_ctx *attn_m1; /**< [#81] the m=1 attention KV cache from
                                 attn_m1_register; NULL = none. Borrows
                                 quant_pool, so it is freed in close()
                                 before hvx_worker_pool_destroy */
  struct nntr_hvx_dspq *dspq; /**< [#141] the MoE call's dspqueue thread
                                 (nntr_hvx_dspq.c); NULL = no queue.
                                 Stopped first in close() */
  nntr_hvx_q4m1_slot q4m1[NNTR_HVX_Q4M1_SLOTS]; /**< [#132 PR 2] the Q4M1
                                 weights of fc_q4m1_f32 and (Part B) the
                                 graph's FC kinds, freed in close() */
  uint8_t *fc_l2; /**< [#132 Part B, #178] the FC runner's L2 feed scratch
                       (2 MiB of DSP heap, first use), freed in close() */
  struct nntr_hvx_token *token; /**< [#132 Part B E2] the token driver
                       (nntr_hvx_token.c): the mailbox page and the role;
                       NULL = none. Stopped in close() after the dspq
                       thread that runs it */
} nntr_hvx_session;

/** @brief [#85] mm_u8i4_moe_layer_timed's stage table, shared with
 *  forward_debug so the two cannot drift: the slot count the ARM side
 *  must pass, and the fill from the probe tables after a bracket
 *  [t0, t1] (hexkl_probe_now ticks) around the run. Both live in
 *  nntr_hvx_mm_u8i4.c, next to the enum that names the slots. */
uint32_t nntr_hvx_moe_stage_count(void);
int nntr_hvx_moe_stage_fill(uint32_t *stage_us, uint64_t t0, uint64_t t1,
                            int rc);

/** @brief [#141] Stops and frees the session's dspqueue thread, if any.
 *  close() calls it first, while the tables the thread reads still exist.
 *  Lives in nntr_hvx_dspq.c. */
void nntr_hvx_dspq_shutdown(nntr_hvx_session *s);

/** @brief [#132 Part B E2] The graph kernels' view of the session
 *  (hexkl_graph_env), for forward and the token driver. Lives in
 *  nntr_hvx_graph.c. */
void nntr_hvx_graph_env(const nntr_hvx_session *s, hexkl_graph_env *env);

/** @brief [#132 Part B E2] One token of the session's role
 *  (token_driver_start) on its graph: S2 runs from op 0 on @a act (the
 *  embedding row) and returns the id, the logits into @a logits when it
 *  is not NULL; S1 serves its rounds (@a act, @a logits unused).
 *  res (4): [id, hops, wait_us, pcycles of the token's ops]. The dspq
 *  thread's HTP_DSPQ_OP_TOKEN calls it. Lives in nntr_hvx_token.c.
 *  @return 0, AEE_EBADSTATE (no driver or no graph), AEE_EINVALIDFORMAT
 *          (a length), or hexkl_token_main / hexkl_token_serve's code */
int nntr_hvx_token_run(nntr_hvx_session *s, uint32_t tok, uint32_t pos,
                       const float *act, uint32_t act_len, float *logits,
                       uint32_t logits_len, uint32_t res[4]);

/** @brief [#132 Part B E2] Stops the token driver, if any (close()).
 *  Lives in nntr_hvx_token.c. */
void nntr_hvx_token_shutdown(nntr_hvx_session *s);

/** @brief [#132 PR 2] Frees every Q4M1 slot and the L2 feed scratch. Lives
 *  in nntr_hvx_fc_q4.c. */
void nntr_hvx_q4m1_free_all(nntr_hvx_session *s);

/**
 * @brief [#132 Part B] One Q4M1 FC over @a lanes pool threads: y (N) =
 *        weight @a h times the prepared activation @a a. @a feed 0 reads
 *        the weight from DDR in place, FC_Q4_FEED_VTCM (1 << 16) stages
 *        each lane's next 32-column group into VTCM by DMA, FC_Q4_FEED_L2
 *        (1 << 17) the same into the session's L2 scratch.
 * @return 0, AEE_EBADITEM (a free or out-of-range handle),
 *         AEE_EINVALIDFORMAT (lanes, feed, or a feed that cannot hold two
 *         groups per lane), AEE_ENOMEMORY, AEE_EEXPIRED (a DMA never
 *         completed: y is void). Lives in nntr_hvx_fc_q4.c.
 */
int nntr_hvx_fc_q4m1_run(nntr_hvx_session *s, uint32_t h, const hvx_q4m1_act *a,
                         float *y, uint32_t lanes, uint32_t feed,
                         uint32_t *lanes_used);

/** @brief [#132 Part B] The graph's hexkl_graph_fc_fn over @a ctx = the
 *  session: the op's feed 0 takes VTCM at 6 lanes when two groups per lane
 *  fit below the HMX config block, else the L2 scratch at 3 lanes; feed 1
 *  the L2 scratch. Lives in nntr_hvx_fc_q4.c. */
int nntr_hvx_fc_q4m1_graph(void *ctx, uint32_t h, uint32_t feed,
                           const hvx_q4m1_act *a, float *y);

/** @brief HAP_mmap_put on every attached arena. close() calls it after the
 *  weight tables are released, since a borrowed slot points into one. Lives
 *  in nntr_hvx_mm_u8i4.c, the one file that includes HAP_mem.h. */
void nntr_hvx_arenas_put_all(nntr_hvx_session *s);

#endif /* __NNTR_HVX_SESSION_H__ */
