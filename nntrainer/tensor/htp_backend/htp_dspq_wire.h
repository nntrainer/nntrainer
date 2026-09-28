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
#define HTP_DSPQ_FLAG_TIMED 1u
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

/** @brief The request message length for n_experts experts and n_rows
 *  routed rows (u64, so a hostile count cannot wrap). */
static inline uint64_t htp_dspq_req_bytes(uint32_t n_experts, uint32_t n_rows) {
  return (uint64_t)sizeof(htp_dspq_req_hdr) + 12u * (uint64_t)n_experts +
         8u * (uint64_t)n_rows;
}

#endif /* __HTP_DSPQ_WIRE_H__ */
