// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_token.h
 * @date   30 Sep 2026
 * @brief  [#132 Part B E2] The two-session token driver: one decode token
 *         over two sessions' graphs split by resident mask, their MoE
 *         hops through a shared-page mailbox
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/132-part-b-two-session-e2e.md section 3.2. Both sessions
 * hold the same description with complementary masks (htp_graph_desc.h
 * HTP_GRAPH_KINDS_S1 / _S2), so hexkl_graph_forward's stretch rule cuts
 * the token at the hops. S2 (hexkl_token_main) runs from op 0; at every
 * stretch end short of the list's end it posts its output row to S1 and
 * waits for S1's, then runs on from where S1 stopped; its last stretch
 * ends in LM_HEAD, whose argmax is the token id. S1 (hexkl_token_serve)
 * serves one round per resident stretch of its own graph: wait, run its
 * stretch on the row, post the output. Neither side computes anything
 * the one-session run does not: every op runs the same kernel on the same
 * bytes, and the rows cross the page by memcpy (hexkl_graph_forward's own
 * act_in / act_out copies, pointed at the page).
 *
 * The page (the #178 probe's nntr_hvx_mailbox.c layout): ping (S2 writes)
 * at byte 0, pong (S1 writes) at byte 128, each on its own 128-byte line;
 * S2's slot at 256, S1's slot after it. A slot is a 128-byte header line
 * {seq, op, n, rc}, the row (at most HEXKL_MBOX_ROW_MAX bytes) and a
 * 128-byte trailer line whose word 0 repeats seq. A post writes the slot,
 * cleans it out of the writer's data cache (FLUSH), then writes and
 * cleans the sequence word; a read flush-invalidates the other side's
 * sequence word until it equals the expected value, then the slot, and
 * refuses a header or trailer that does not carry it (a stale read:
 * HEXKL_TOKEN_E_STALE). seq = tok x 256 + round + 1, so no value of an
 * earlier round or token can pass for this one. A wait spins spin_us,
 * then polls every 50 us, and gives up HEXKL_TOKEN_TIMEOUT_US after the
 * spin window with AEE_EEXPIRED -- the ARM turns that into its throw, a
 * lost post is never a hang. A side whose forward fails posts its code in
 * the header, so the other returns it at once instead of timing out.
 *
 * No heap, no VTCM, no DMA: the page is the caller's (an ION buffer both
 * PDs map), the rows live in it.
 */

#ifndef __NNTRAINER_HEXKL_TOKEN_H__
#define __NNTRAINER_HEXKL_TOKEN_H__

#include <stdint.h>

#include "hexkl_graph.h"

/** @brief The page's byte layout. */
#define HEXKL_MBOX_PING 0u
#define HEXKL_MBOX_PONG 128u
#define HEXKL_MBOX_LINE 128u
/** @brief The widest row a hop carries: LFM2.5's hidden, 2048 f32. */
#define HEXKL_MBOX_ROW_MAX 8192u
#define HEXKL_MBOX_SLOT (2u * HEXKL_MBOX_LINE + HEXKL_MBOX_ROW_MAX)
#define HEXKL_MBOX_S2_SLOT 256u
#define HEXKL_MBOX_S1_SLOT (HEXKL_MBOX_S2_SLOT + HEXKL_MBOX_SLOT)
/** @brief The page size the driver needs (16 896 B). */
#define HEXKL_MBOX_BYTES (HEXKL_MBOX_S1_SLOT + HEXKL_MBOX_SLOT)
/** @brief A wait gives up this long after its spin window. */
#define HEXKL_TOKEN_TIMEOUT_US 1000000u
/** @brief Poll period after the spin window. */
#define HEXKL_TOKEN_POLL_US 50u
/** @brief Rounds (S1 stretches) one token may have: seq's low byte. */
#define HEXKL_TOKEN_MAX_ROUNDS 255u
/** @brief A header or trailer that does not carry the posted seq. */
#define HEXKL_TOKEN_E_STALE HTP_GRAPH_E_INCOMPLETEITEM

/** @brief A slot's header line, word by word. */
typedef struct {
  uint32_t seq; /**< the round's sequence value */
  uint32_t op;  /**< S2 -> S1: where S1 starts; S1 -> S2: where S2 resumes */
  uint32_t n;   /**< the row's f32 count */
  int32_t rc;   /**< 0, or the poster's failure (the row is void) */
} hexkl_mbox_hdr;

/** @brief One side's counters, accumulated over its calls. */
typedef struct {
  uint32_t tokens;   /**< calls that returned 0 */
  uint32_t hops;     /**< rows posted plus rows received */
  uint32_t wait_us;  /**< time spent waiting for the other side */
  uint32_t timeouts; /**< waits that gave up (AEE_EEXPIRED) */
  uint32_t stale;    /**< reads refused as stale */
  uint64_t pcycles;  /**< the op_pcycles of every stretch this side ran */
} hexkl_token_stats;

/** @brief The sequence value of round @a round of token @a tok. */
static inline uint32_t hexkl_token_seq(uint32_t tok, uint32_t round) {
  return tok * 256u + round + 1u;
}

/** @brief How many resident stretches @a g has: S1's rounds per token. */
uint32_t hexkl_token_rounds(const hexkl_graph *g);

/**
 * @brief S2's side of token @a tok at position @a pos: forward from op 0
 *        on @a act_in (the embedding row, op 0's input width), a hop at
 *        every stretch end short of the list's end.
 * @param logits the last stretch's output (its width), or NULL: then the
 *               last op must be LM_HEAD and the logits stay in g->logits
 * @param id     g->lm_id after the token (the last op is LM_HEAD), else 0
 * @return 0; AEE_EEXPIRED (a wait gave up); HEXKL_TOKEN_E_STALE; S1's code
 *         from its header; hexkl_graph_forward's code; AEE_EBADSTATE (op
 *         0 not resident, or S1 resumed where S2 cannot run);
 *         AEE_EINVALIDFORMAT (a row wider than HEXKL_MBOX_ROW_MAX, or
 *         @a logits NULL without an LM_HEAD at the end)
 */
int hexkl_token_main(hexkl_graph *g, const hexkl_graph_env *env, uint8_t *mbox,
                     uint32_t tok, uint32_t pos, const float *act_in,
                     uint32_t act_len, float *logits, uint32_t logits_len,
                     uint32_t spin_us, hexkl_token_stats *st, uint32_t *id);

/**
 * @brief S1's side of token @a tok: hexkl_token_rounds(g) rounds of wait,
 *        forward over the stretch S2 names, post.
 * @return 0; AEE_EEXPIRED; HEXKL_TOKEN_E_STALE; S2's code from its header;
 *         hexkl_graph_forward's code (posted to S2 too); AEE_EBADSTATE (no
 *         resident op, or S2 named an op S1 does not run);
 *         AEE_EINVALIDFORMAT (a row wider than HEXKL_MBOX_ROW_MAX)
 */
int hexkl_token_serve(hexkl_graph *g, const hexkl_graph_env *env, uint8_t *mbox,
                      uint32_t tok, uint32_t pos, uint32_t spin_us,
                      hexkl_token_stats *st);

#endif /* __NNTRAINER_HEXKL_TOKEN_H__ */
