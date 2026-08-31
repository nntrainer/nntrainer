// SPDX-License-Identifier: Apache-2.0
/**
 * @file	nntr_htp_common.h
 * @date	15 August 2026
 * @brief	Op-list wire format (v2: op descriptors) shared by host (arm64)
 *		and DSP (hexagon-clang). Plain C - compiled by both toolchains.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_COMMON_H
#define NNTR_HTP_COMMON_H

#include <stdint.h>
#include <string.h>

#define NNTR_HTP_OPLIST_MAGIC 0x5054484Eu /* "NHTP" little-endian */
#define NNTR_HTP_ABI_VERSION 3u /* v3: forward pcycles + forward_debug */

enum nntr_htp_buf_id {
  NNTR_HTP_BUF_WEIGHTS = 0,
  NNTR_HTP_BUF_KV = 1,
  NNTR_HTP_BUF_ACT = 2,
  NNTR_HTP_BUF_TOKENS =
    3, /* forward() token_ids argument, updated every call */
  NNTR_HTP_BUF_LOGITS = 4, /* forward() logits argument, updated every call */
  NNTR_HTP_BUF_COUNT = 5
};

enum nntr_htp_op_kind {
  NNTR_HTP_OP_EMBED = 0,
  NNTR_HTP_OP_RMSNORM = 1,
  NNTR_HTP_OP_MATMUL_W8A8 = 2,
  NNTR_HTP_OP_ROPE = 3,
  NNTR_HTP_OP_ATTN = 4,
  NNTR_HTP_OP_SILU_MUL = 5,
  NNTR_HTP_OP_ADD = 6,
  NNTR_HTP_OP_MATMUL_LOGITS = 7,
  NNTR_HTP_OP_MATMUL_W8A16 = 8, /* fp16 x (no act quant) . int8 w, fp16 y */
  NNTR_HTP_OP_KIND_COUNT = 9
};

#define NNTR_HTP_FLAG_PER_HEAD 0x1u /* RMSNORM: per head_dim QK-Norm */

/**
 * @brief One tensor reference: which buffer plus a 128B-aligned byte offset.
 */
struct nntr_htp_tensor_ref {
  uint32_t buf;    /**< enum nntr_htp_buf_id */
  uint32_t offset; /**< bytes, 128B aligned */
};                 /* 8B */

/**
 * @brief One op entry of the op-list. Fixed 64B wire record.
 */
struct nntr_htp_op_desc {
  uint32_t kind, flags, layer;
  uint32_t m, k, n; /* m==0 -> substituted with n_tokens at execution time */
  struct nntr_htp_tensor_ref in0, in1, in2, out;
  uint32_t param0, param1; /* op specific (e.g. fp32 bit pattern) */
};                         /* 64B, size check required */

/**
 * @brief Leading header of every op-list buffer sent from host to DSP.
 */
struct nntr_htp_oplist_header {
  uint32_t magic, version, n_ops, reserved; /* v1 prefix kept for compat */
  uint32_t n_layers, n_heads, n_kv_heads, head_dim;
  uint32_t hidden, ffn, vocab, max_seq;
  uint32_t max_chunk;
  uint32_t reserved2[3];
}; /* 64B, size check required */

/* The struct is the wire ABI: all three toolchains must agree on 64 bytes. */
typedef char nntr_htp_oplist_header_size_check
  [(sizeof(struct nntr_htp_oplist_header) == 64) ? 1 : -1];
typedef char
  nntr_htp_op_desc_size_check[(sizeof(struct nntr_htp_op_desc) == 64) ? 1 : -1];

/**
 * @brief Validate an op-list buffer header.
 * @return 0 ok, 1 bad pointer/size, 2 bad magic, 3 version mismatch
 */
static inline int nntr_htp_oplist_check(const void *buf, uint32_t len) {
  struct nntr_htp_oplist_header h;
  if (buf == 0 || len < (uint32_t)sizeof(h))
    return 1;
  memcpy(&h, buf, sizeof(h));
  if (h.magic != NNTR_HTP_OPLIST_MAGIC)
    return 2;
  if (h.version != NNTR_HTP_ABI_VERSION)
    return 3;
  return 0;
}

/**
 * @brief Bounds-check one tensor reference against its target buffer.
 * @return 0 ok, 1 out of bounds
 */
static inline int
nntr_htp_check_ref(uint32_t buf_id, uint32_t offset, uint64_t bytes,
                   const uint32_t buf_size[NNTR_HTP_BUF_COUNT]) {
  if ((uint64_t)offset + bytes > (uint64_t)buf_size[buf_id])
    return 1;
  return 0;
}

/**
 * @brief Validate an op-list buffer: header fields, op kinds/refs, and
 * per-op tensor bounds against the caller-provided buffer sizes.
 * @return 0 ok, 1 bad pointer/size, 2 bad magic, 3 version mismatch,
 * 4 bad header field, 5 bad op
 */
static inline int
nntr_htp_oplist_validate(const void *buf, uint32_t len,
                         const uint32_t buf_size[NNTR_HTP_BUF_COUNT]) {
  struct nntr_htp_oplist_header h;
  uint32_t i;

  if (buf == 0 || len < (uint32_t)sizeof(h))
    return 1;
  memcpy(&h, buf, sizeof(h));
  if (h.magic != NNTR_HTP_OPLIST_MAGIC)
    return 2;
  if (h.version != NNTR_HTP_ABI_VERSION)
    return 3;
  if ((uint64_t)len !=
      (uint64_t)sizeof(h) + (uint64_t)h.n_ops * sizeof(struct nntr_htp_op_desc))
    return 1;
  if (h.head_dim != 128u || h.hidden % 64u != 0u || h.ffn % 64u != 0u ||
      h.n_kv_heads == 0u || h.n_heads % h.n_kv_heads != 0u || h.max_chunk < 1u)
    return 4;

  for (i = 0; i < h.n_ops; ++i) {
    struct nntr_htp_op_desc d;
    uint32_t m;
    memcpy(&d, (const uint8_t *)buf + sizeof(h) + (uint64_t)i * sizeof(d),
           sizeof(d));

    if (d.kind >= (uint32_t)NNTR_HTP_OP_KIND_COUNT)
      return 5;
    if (d.in0.buf >= (uint32_t)NNTR_HTP_BUF_COUNT ||
        d.in1.buf >= (uint32_t)NNTR_HTP_BUF_COUNT ||
        d.in2.buf >= (uint32_t)NNTR_HTP_BUF_COUNT ||
        d.out.buf >= (uint32_t)NNTR_HTP_BUF_COUNT)
      return 5;
    if (d.in0.offset % 128u != 0u || d.in1.offset % 128u != 0u ||
        d.in2.offset % 128u != 0u || d.out.offset % 128u != 0u)
      return 5;
    if ((d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A8 ||
         d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A16 ||
         d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS) &&
        d.k % 128u != 0u)
      return 5;

    m = d.m ? d.m : h.max_chunk;

    switch (d.kind) {
    case NNTR_HTP_OP_EMBED:
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset, (uint64_t)m * 4u,
                             buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset, (uint64_t)h.vocab * d.k,
                             buf_size) ||
          nntr_htp_check_ref(d.in2.buf, d.in2.offset, (uint64_t)h.vocab * 4u,
                             buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset, (uint64_t)m * d.k * 2u,
                             buf_size))
        return 5;
      break;
    case NNTR_HTP_OP_RMSNORM: {
      uint64_t gamma_bytes = (d.flags & NNTR_HTP_FLAG_PER_HEAD)
                               ? (uint64_t)h.head_dim * 2u
                               : (uint64_t)d.n * 2u;
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset, (uint64_t)m * d.n * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset, gamma_bytes, buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset, (uint64_t)m * d.n * 2u,
                             buf_size))
        return 5;
      break;
    }
    case NNTR_HTP_OP_MATMUL_W8A8:
    case NNTR_HTP_OP_MATMUL_W8A16: /* same operand layout */
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset, (uint64_t)m * d.k * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset, (uint64_t)d.n * d.k,
                             buf_size) ||
          nntr_htp_check_ref(d.in2.buf, d.in2.offset, (uint64_t)d.n * 4u,
                             buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset, (uint64_t)m * d.n * 2u,
                             buf_size))
        return 5;
      break;
    case NNTR_HTP_OP_ROPE:
      /* out is in-place (== in0); skip the out bounds check. */
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset,
                             (uint64_t)m * h.n_heads * 128u * 2u, buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset,
                             (uint64_t)m * h.n_kv_heads * 128u * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.in2.buf, d.in2.offset,
                             (uint64_t)h.max_seq * 128u * 2u, buf_size))
        return 5;
      break;
    case NNTR_HTP_OP_ATTN:
      /* K/V are addressed by layer into the KV buffer, not by in1/in2. */
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset,
                             (uint64_t)m * h.n_heads * 128u * 2u, buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset,
                             (uint64_t)m * h.n_heads * 128u * 2u, buf_size))
        return 5;
      if (2ull * h.n_layers * h.n_kv_heads * h.max_seq * h.head_dim * 2u >
          (uint64_t)buf_size[NNTR_HTP_BUF_KV])
        return 5;
      if (h.max_seq % 64u) /* K^T rows are streamed 64 positions/vector */
        return 5;
      break;
    case NNTR_HTP_OP_SILU_MUL:
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset, (uint64_t)m * d.n * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset, (uint64_t)m * d.n * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset, (uint64_t)m * d.n * 2u,
                             buf_size))
        return 5;
      break;
    case NNTR_HTP_OP_ADD:
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset, (uint64_t)m * d.n * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset, (uint64_t)m * d.n * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset, (uint64_t)m * d.n * 2u,
                             buf_size))
        return 5;
      break;
    case NNTR_HTP_OP_MATMUL_LOGITS:
      if (nntr_htp_check_ref(d.in0.buf, d.in0.offset, (uint64_t)m * d.k * 2u,
                             buf_size) ||
          nntr_htp_check_ref(d.in1.buf, d.in1.offset, (uint64_t)d.n * d.k,
                             buf_size) ||
          nntr_htp_check_ref(d.in2.buf, d.in2.offset, (uint64_t)d.n * 4u,
                             buf_size) ||
          nntr_htp_check_ref(d.out.buf, d.out.offset, (uint64_t)d.n * 4u,
                             buf_size))
        return 5;
      break;
    default:
      return 5;
    }
  }
  return 0;
}

#endif /* NNTR_HTP_COMMON_H */
