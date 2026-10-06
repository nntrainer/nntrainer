// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_dspq_wire.h
 * @date   28 Sep 2026
 * @brief  [#141] The dspqueue packet of one M==1 MoE layer call, shared by
 *         the ARM side (htp_compute_ops.cpp) and the DSP side
 *         (test/htp/nntr_hvx_dspq.c), as pure C99
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/141-dspq-moe.md section 3.1. Request message: the
 * header below, then h_gate_up[n_experts], h_down[n_experts],
 * row_count[n_experts], row_index[n_rows] (u32) and row_weight[n_rows]
 * (f32), in that order; buffer reference 0 is the M x K f32 activation,
 * 1 the M x N_out f32 output. Response message: {seq, rc} and, when the
 * request's HTP_DSPQ_FLAG_TIMED is set, the HTP_DSPQ_STAGES stage slots of
 * mm_u8i4_moe_layer_timed. The DSP calls the FastRPC method's own C
 * function with these arguments, so the arithmetic is the same.
 */
#ifndef __HTP_DSPQ_WIRE_H__
#define __HTP_DSPQ_WIRE_H__

#include <stdint.h>

#define HTP_DSPQ_OP_MOE 1u
#define HTP_DSPQ_OP_QUIT 2u
/** @brief [#132 Part B E2] One decode token of the session's token driver
 *  role (nntr_hvx_token.c): htp_dspq_token_req, answered with
 *  htp_dspq_token_resp. S2's packet carries buffer 0, the embedding row
 *  (op 0's width in f32), and with HTP_DSPQ_TOKEN_LOGITS buffer 1, the
 *  logits (vocab f32); S1's carries none. */
#define HTP_DSPQ_OP_TOKEN 3u
#define HTP_DSPQ_FLAG_TIMED 1u
/** @brief OP_TOKEN: S2 writes the logits into buffer 1 (NNTR_PPL_DECODE,
 *  the shadows); without it only the id travels. */
#define HTP_DSPQ_TOKEN_LOGITS 1u
/** @brief mm_u8i4_moe_layer_timed's slot count; both sides check theirs. */
#define HTP_DSPQ_STAGES 31u
#define HTP_DSPQ_MAX_MSG 4096u
#define HTP_DSPQ_REQ_QUEUE_BYTES (16u * 1024u)
#define HTP_DSPQ_RESP_QUEUE_BYTES (4u * 1024u)
/** @brief Each of the two queue-owned ION buffers (act, out). */
#define HTP_DSPQ_BUF_BYTES (64u * 1024u)

/** @brief The request message's header, 9 x u32. */
typedef struct {
  uint32_t op, seq, flags, M, K, inter, N_out, n_experts, n_rows;
} htp_dspq_req_hdr;

/** @brief The response message; stage_us travels only for a timed call. */
typedef struct {
  uint32_t seq;
  int32_t rc;
  uint32_t stage_us[HTP_DSPQ_STAGES];
} htp_dspq_resp;

#define HTP_DSPQ_RESP_BASE_BYTES 8u

/** @brief [#132 Part B E2] OP_TOKEN's request: seq is the token number
 *  both sessions' packets carry (the mailbox's sequence values derive
 *  from it), pos the token's position. */
typedef struct {
  uint32_t op, seq, flags, pos;
} htp_dspq_token_req;

/** @brief The graph's op kinds (htp_graph_desc.h HTP_OP_KIND_N; the ARM
 *  side asserts they agree). */
#define HTP_DSPQ_TOKEN_KINDS 11u
/** @brief [plan 201 S1] The routed sets' bytes (hexkl_graph.h
 *  HEXKL_GRAPH_ROUTE_LOG; the DSP side asserts they agree). */
#define HTP_DSPQ_TOKEN_ROUTE 320u

/** @brief OP_TOKEN's response: id (S2: LM_HEAD's argmax), the hops this
 *  side made, its wait for the other side, the pcycles of its ops; the
 *  token's wall time on this side and the pcycles over it (their ratio is
 *  the clock), and the op pcycles per kind; [#194 L0] of the wait, the
 *  time from each of the other side's posts to this side's wake-up. */
typedef struct htp_dspq_token_resp_s {
  uint32_t seq;
  int32_t rc;
  uint32_t id, hops, wait_us, pcycles;
  uint32_t wall_us, wall_pcyc;
  uint32_t kind_pcyc[HTP_DSPQ_TOKEN_KINDS];
  uint32_t hop_us;
  uint32_t t_in_us, t_out_us; /**< [#194 L0] QTimer us (low 32 bits) when
                                   the packet was read / the response is
                                   written: the ARM's dispatch and return */
  uint32_t misses, miss_us;   /**< [plan 201 S1] S1: experts its miss rounds
                                   loaded this token, and the waits */
  uint32_t route_n;           /**< [plan 201 S1] S1: bytes of route */
  uint8_t route[HTP_DSPQ_TOKEN_ROUTE]; /**< per MOE op in list order: its
                                            routed count, then the ids */
} htp_dspq_token_resp;

/** @brief [plan 201 S1] The token driver's mailbox page (hexkl_token.h):
 *  after its two row slots, the expert pool's miss round (protocol P-A):
 *  S1's request and the pool owner's (the ARM's) answer, each in its own
 *  lines. The most experts a request names: the M = 1 MoE path's bound. */
#define HTP_MBOX_MISS_REQ 17152u
#define HTP_MBOX_MISS_REQ_BYTES 256u
#define HTP_MBOX_MISS_ANS (HTP_MBOX_MISS_REQ + HTP_MBOX_MISS_REQ_BYTES)
#define HTP_MBOX_MISS_ANS_BYTES 1024u
#define HTP_MBOX_MISS_MAX 16u

/**
 * @brief [plan 201 S1] S1 -> the pool's owner (the ARM): MOE op @a op of
 *        this token routes to @a routed, and its EXPERTS table lacks
 *        @a miss. Written body first and cleaned, then @a seq (word 0);
 *        @a seq2 repeats it, so a torn read is refused.
 */
typedef struct {
  uint32_t seq; /**< hexkl_token_seq(tok, k), k the token's k-th round */
  uint32_t op;  /**< the MOE op, graph index */
  uint32_t n_routed, n_miss;
  uint32_t routed[HTP_MBOX_MISS_MAX]; /**< ascending: never evicted */
  uint32_t miss[HTP_MBOX_MISS_MAX];
  uint32_t seq2;
} htp_miss_req;

/** @brief One expert the answer brought: read into the slot of the pair
 *  @a old_gu / @a old_dn (hexkl_graph.h hexkl_graph_rebind_fn); S1 writes the
 * pair's numbers after the rebind into @a h_gu / @a h_dn (NO_HANDLE before),
 * for the owner to file. */
typedef struct {
  uint32_t e, old_gu, old_dn, arena, off_gu, off_dn;
  uint32_t h_gu, h_dn;
  /** [plan 229] 2: QS2CX_WH codes with these palettes; 4 (or 0): QS4CX_WH,
   *  the palettes unused */
  uint32_t bits;
  int8_t pal_gu[4], pal_dn[4];
} htp_miss_load;

/** @brief The owner -> S1: every expert it evicted (MOE op, expert), then
 *  the misses loaded, or @a rc (the round fails the token). */
typedef struct {
  uint32_t seq;
  int32_t rc;
  uint32_t n_evict, n_load;
  uint32_t evict[HTP_MBOX_MISS_MAX][2];
  htp_miss_load load[HTP_MBOX_MISS_MAX];
  uint32_t seq2;
} htp_miss_ans;

typedef char
  htp_miss_req_fits[sizeof(htp_miss_req) <= HTP_MBOX_MISS_REQ_BYTES ? 1 : -1];
typedef char
  htp_miss_ans_fits[sizeof(htp_miss_ans) <= HTP_MBOX_MISS_ANS_BYTES ? 1 : -1];

/** @brief The request message length for n_experts experts and n_rows
 *  routed rows (u64, so a hostile count cannot wrap). */
static inline uint64_t htp_dspq_req_bytes(uint32_t n_experts, uint32_t n_rows) {
  return (uint64_t)sizeof(htp_dspq_req_hdr) + 12u * (uint64_t)n_experts +
         8u * (uint64_t)n_rows;
}

#endif /* __HTP_DSPQ_WIRE_H__ */
