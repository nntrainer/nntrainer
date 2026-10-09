// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_token.h
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2, #211] The one-PD token driver: one decode token
 *         over the session's whole graph, the expert pool's miss rounds
 *         through a shared page
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_token_main runs every op of the session's graph once (every op
 * must be resident), the embedding row in, the logits and LM_HEAD's
 * argmax out. [plan 201 S1] A MOE op whose routed experts are not all in
 * its EXPERTS table runs a miss round (protocol P-A) on the page's miss
 * lines: the request, the present experts, the answer (each eviction
 * cleared from its table, each load rebound through the env's rebind and
 * entered), the rest. The graph's route_log holds the token's routed sets
 * after the call.
 *
 * A request is written body first and cleaned out of the data cache, then
 * its sequence word; the answer's word is flush-invalidated until it
 * equals the expected value, then the answer, which is refused when its
 * second copy (seq2) does not carry it (a stale read:
 * HEXKL_TOKEN_E_STALE). seq = tok x 256 + round + 1, so no value of an
 * earlier round or token can pass for this one. A wait spins spin_us,
 * with a pause between reads, then sleeps HEXKL_TOKEN_POLL_US between
 * reads, and gives up HEXKL_TOKEN_TIMEOUT_US after the spin window with
 * AEE_EEXPIRED -- the ARM turns that into its throw, a lost answer is
 * never a hang. An answer carrying a code ends the token with it.
 *
 * The page's first HEXKL_MBOX_MISS_REQ bytes held the two-PD path's
 * ping / pong words and row slots (removed by #211); the miss lines stay
 * where they were, so the wire (htp_dspq_wire.h) does not move.
 *
 * No heap, no VTCM, no DMA: the page is the caller's (an ION buffer the
 * PD maps), the miss lines live in it.
 */

#ifndef __NNTRAINER_HEXKL_TOKEN_H__
#define __NNTRAINER_HEXKL_TOKEN_H__

#include <stdint.h>

#include "hexkl_graph.h"
#include "htp_dspq_wire.h"

/** @brief [plan 201 S1] The expert pool's miss round (protocol P-A): the
 *  request (htp_miss_req) and the pool owner's answer (htp_miss_ans),
 *  each in its own lines. */
#define HEXKL_MBOX_MISS_REQ HTP_MBOX_MISS_REQ
#define HEXKL_MBOX_MISS_ANS HTP_MBOX_MISS_ANS
/** @brief The page size the driver needs (18 560 B, [#282 B] the hint). */
#define HEXKL_MBOX_BYTES (HTP_MBOX_HINT + HTP_MBOX_HINT_BYTES)
/** @brief A wait gives up this long after its spin window. */
#define HEXKL_TOKEN_TIMEOUT_US 1000000u
/** @brief Poll period after the spin window: a sleep, so the waiting
 *  thread holds no hardware thread the ARM's pool reads need. */
#define HEXKL_TOKEN_POLL_US 20u
/** @brief Miss rounds one token may have: seq's low byte. */
#define HEXKL_TOKEN_MAX_ROUNDS 255u
/** @brief An answer whose seq2 does not carry the expected seq. */
#define HEXKL_TOKEN_E_STALE HTP_GRAPH_E_INCOMPLETEITEM

/* [plan 201 S1] the miss lines' structs (htp_miss_req / _ans) are the
   ARM's too: htp_dspq_wire.h. [#211] Their offset is a skel / lib
   contract: a skel and a lib that disagree would time out silently. */
typedef char hexkl_miss_at[HTP_MBOX_MISS_REQ == 17152u ? 1 : -1];
typedef char hexkl_miss_max[HTP_MBOX_MISS_MAX == HEXKL_GRAPH_MISS_MAX ? 1 : -1];

/** @brief The driver's counters, accumulated over its calls. */
typedef struct {
  uint32_t tokens;   /**< calls that returned 0 */
  uint32_t timeouts; /**< waits that gave up (AEE_EEXPIRED) */
  uint32_t stale;    /**< answers refused as stale */
  uint64_t pcycles;  /**< the op_pcycles of every op the token ran */
  uint64_t kind_pcycles[HTP_OP_KIND_N]; /**< the same, per op kind */
  uint32_t misses;  /**< [plan 201 S1] experts loaded by miss rounds */
  uint32_t miss_us; /**< the waits for their answers */
  /** [#267 L0] the ops' wall QTimer ticks per kind (hexkl_graph op_qt);
   *  MOE's hold its miss waits */
  uint64_t kind_qt[HTP_OP_KIND_N];
  uint64_t miss_pcyc; /**< [#267 L0] the pcycles over those waits */
} hexkl_token_stats;

/** @brief The sequence value of round @a round of token @a tok. */
static inline uint32_t hexkl_token_seq(uint32_t tok, uint32_t round) {
  return tok * 256u + round + 1u;
}

/**
 * @brief Token @a tok at position @a pos: forward over the whole graph on
 *        @a act_in (the embedding row, op 0's input width), the MOE ops'
 *        miss rounds on @a mbox.
 * @param logits the last op's output (its width), or NULL: then the last
 *               op must be LM_HEAD and the logits stay in g->logits
 * @param id     g->lm_id after the token (the last op is LM_HEAD), else 0
 * @return 0; AEE_EEXPIRED (a miss wait gave up); HEXKL_TOKEN_E_STALE; the
 *         owner's code from its answer; hexkl_graph_forward's code;
 *         AEE_EBADSTATE (an op not resident); AEE_EINVALIDFORMAT
 *         (@a logits NULL without an LM_HEAD at the end, or a request
 *         past HEXKL_TOKEN_MAX_ROUNDS)
 */
int hexkl_token_main(hexkl_graph *g, const hexkl_graph_env *env, uint8_t *mbox,
                     uint32_t tok, uint32_t pos, const float *act_in,
                     uint32_t act_len, float *logits, uint32_t logits_len,
                     uint32_t spin_us, hexkl_token_stats *st, uint32_t *id);

#endif /* __NNTRAINER_HEXKL_TOKEN_H__ */
