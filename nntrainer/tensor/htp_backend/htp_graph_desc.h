// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_graph_desc.h
 * @date   23 Sep 2026
 * @brief  The per-token decode graph as the DSP receives it: wire format,
 *         validator and the LFM2 builder, as pure C99 (#85)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Header-only, like htp_moe_opts.h, so the three consumers compile one
 * source: the skel (hexkl_graph.c validates at graph_init), the ARM side
 * (htp_compute_ops.cpp validates before it sends, lfm2_moe_causallm.cpp
 * builds) and test/htp/host/graph_host_check.c. No Hexagon or SDK header
 * is included: the error codes are restated below with AEEStdErr.h's
 * values, and hexkl_graph.c checks them against the SDK's at compile time.
 *
 * Wire format, uint32 words:
 *   [0..6]  magic, version, n_layers, n_ops, hidden, vocab, max_seq
 *   [7..]   layer_kind[n_layers]  (HTP_GRAPH_LAYER_*)
 *           ffn_kind[n_layers]    (HTP_GRAPH_FFN_*)
 *           ops[n_ops]            (HTP_GRAPH_OP_WORDS each, htp_graph_op)
 *
 * The list carries every op of the decode step with its real kind, so the
 * validator can check layer consistency, and a per-op resident bit says
 * whether this build's kernel table runs it. forward() runs from a start
 * op while ops are resident and returns the index of the first op that is
 * not (plan 85 section 3.1: layer-granular resume).
 */
#ifndef __HTP_GRAPH_DESC_H__
#define __HTP_GRAPH_DESC_H__

#include <stdint.h>
#include <string.h>

#define HTP_GRAPH_MAGIC 0x47505448u /* "HTPG" */
#define HTP_GRAPH_VERSION 1u
#define HTP_GRAPH_HEADER_WORDS 7u
/** @brief LFM2.5-8B-A1B is 228 ops (section htp_graph_lfm2_build); the
 *  table is HTP_GRAPH_OP_WORDS x 4 B per op, so 256 is 76 KiB of DSP
 *  heap at graph_init (address-space note in hexkl_graph.c). */
#define HTP_GRAPH_MAX_OPS 256u
#define HTP_GRAPH_MAX_LAYERS 64u
#define HTP_GRAPH_MAX_EXPERTS 32u
#define HTP_GRAPH_N_SLOTS 3u
#define HTP_GRAPH_NO_OP 0xFFFFFFFFu

/** @brief Op kinds. The order is the wire value; append, never reorder. */
enum {
  HTP_OP_RMSNORM = 0,
  HTP_OP_FC,
  HTP_OP_CONV1D_GATE,
  HTP_OP_QK_NORM,
  HTP_OP_ROPE,
  HTP_OP_ATTN_M1,
  HTP_OP_ADD,
  HTP_OP_ROUTER_TOPK,
  HTP_OP_MOE,
  HTP_OP_DENSE_FFN,
  HTP_OP_LM_HEAD,
  HTP_OP_KIND_N
};
#define HTP_GRAPH_KIND_BIT(k) (1u << (k))

enum { HTP_GRAPH_LAYER_CONV = 0, HTP_GRAPH_LAYER_ATTN = 1 };
enum { HTP_GRAPH_FFN_DENSE = 0, HTP_GRAPH_FFN_MOE = 1 };

/**
 * @brief One op record, HTP_GRAPH_OP_WORDS words on the wire.
 *
 * K is the input width, N the output width (inter for MOE / DENSE_FFN,
 * whose output width is N_out). in_slot / out_slot name the session's
 * activation slots (plan 85 section 3.1); ADD reads slot 0 as its second
 * operand implicitly. next_mm names the next weight-streaming op (FC,
 * MOE, DENSE_FFN, LM_HEAD) or HTP_GRAPH_NO_OP; nothing consumes it yet
 * (the cross-op prefetch hook is a later issue), the validator only
 * requires it to point forward. h_gu / h_dn are the MoE op's registered
 * weight handles, bound by the ARM before graph_init; other kinds leave
 * them 0.
 */
typedef struct {
  uint32_t kind;
  uint32_t layer; /**< 0..n_layers-1, or n_layers for the tail */
  uint32_t resident;
  uint32_t K;
  uint32_t N;
  uint32_t N_out;
  uint32_t n_experts;
  uint32_t top_k;
  uint32_t in_slot;
  uint32_t out_slot;
  uint32_t next_mm;
  uint32_t rsv;
  uint32_t h_gu[HTP_GRAPH_MAX_EXPERTS];
  uint32_t h_dn[HTP_GRAPH_MAX_EXPERTS];
} htp_graph_op;
#define HTP_GRAPH_OP_WORDS (12u + 2u * HTP_GRAPH_MAX_EXPERTS)
typedef char
  htp_graph_op_size_check[sizeof(htp_graph_op) == HTP_GRAPH_OP_WORDS * 4u ? 1
                                                                          : -1];

/** @brief AEEStdErr.h's codes, restated so this header needs no SDK. The
 *  offset mirrors AEEStdErr.h: 0x80000400 on the DSP, 0 on every other
 *  side, so on each side HTP_GRAPH_E_x == AEE_x (hexkl_graph.c asserts
 *  it). AEE_EBADPARM is deliberately absent: no validator path returns
 *  it, so it stays the stale-skel symptom (LEDGER rule 3). */
#if defined(__hexagon__)
#define HTP_GRAPH_EOFFSET 0x80000400u
#else
#define HTP_GRAPH_EOFFSET 0u
#endif
#define HTP_GRAPH_E_CLASSNOTSUPPORT (HTP_GRAPH_EOFFSET + 0x003u)
#define HTP_GRAPH_E_BADSTATE (HTP_GRAPH_EOFFSET + 0x00Du)
#define HTP_GRAPH_E_BADITEM (HTP_GRAPH_EOFFSET + 0x010u)
#define HTP_GRAPH_E_INVALIDFORMAT (HTP_GRAPH_EOFFSET + 0x011u)
#define HTP_GRAPH_E_INCOMPLETEITEM (HTP_GRAPH_EOFFSET + 0x012u)
#define HTP_GRAPH_E_UNSUPPORTED (HTP_GRAPH_EOFFSET + 0x014u)
#define HTP_GRAPH_E_NOTYPE (HTP_GRAPH_EOFFSET + 0x022u)
#define HTP_GRAPH_E_INVALIDITEM (HTP_GRAPH_EOFFSET + 0x02Au)
#define HTP_GRAPH_E_INVHANDLE (HTP_GRAPH_EOFFSET + 0x02Cu)

/** @brief The code's name for a log line, on either side's offset. */
static inline const char *htp_graph_err_name(uint32_t code) {
  switch (code & 0xFFFu) {
  case 0:
    return "AEE_SUCCESS";
  case 0x003u:
    return "AEE_ECLASSNOTSUPPORT";
  case 0x00Du:
    return "AEE_EBADSTATE";
  case 0x00Eu:
    return "AEE_EBADPARM";
  case 0x010u:
    return "AEE_EBADITEM";
  case 0x011u:
    return "AEE_EINVALIDFORMAT";
  case 0x012u:
    return "AEE_EINCOMPLETEITEM";
  case 0x014u:
    return "AEE_EUNSUPPORTED";
  case 0x022u:
    return "AEE_ENOTYPE";
  case 0x02Au:
    return "AEE_EINVALIDITEM";
  case 0x02Cu:
    return "AEE_EINVHANDLE";
  default:
    return "AEE_?";
  }
}

static inline uint32_t htp_graph_words_for(uint32_t n_layers, uint32_t n_ops) {
  return HTP_GRAPH_HEADER_WORDS + 2u * n_layers + n_ops * HTP_GRAPH_OP_WORDS;
}
static inline uint32_t htp_graph_op_offset(const uint32_t *w, uint32_t i) {
  return HTP_GRAPH_HEADER_WORDS + 2u * w[2] + i * HTP_GRAPH_OP_WORDS;
}
static inline htp_graph_op *htp_graph_op_at(uint32_t *w, uint32_t i) {
  return (htp_graph_op *)(w + htp_graph_op_offset(w, i));
}
static inline const htp_graph_op *htp_graph_op_cat(const uint32_t *w,
                                                   uint32_t i) {
  return (const htp_graph_op *)(w + htp_graph_op_offset(w, i));
}
static inline int htp_graph_kind_streams_weights(uint32_t kind) {
  return kind == HTP_OP_FC || kind == HTP_OP_MOE || kind == HTP_OP_DENSE_FFN ||
         kind == HTP_OP_LM_HEAD;
}
/** @brief f32 words an op reads from its in_slot and writes to its
 *  out_slot: what forward()'s act_in / act_out must be sized to. */
static inline uint32_t htp_graph_op_in_words(const htp_graph_op *op) {
  return op->K;
}
static inline uint32_t htp_graph_op_out_words(const htp_graph_op *op) {
  return (op->kind == HTP_OP_MOE || op->kind == HTP_OP_DENSE_FFN) ? op->N_out
                                                                  : op->N;
}

/**
 * @brief Validates a description. Pure: no table, no hardware.
 * @param w           the words
 * @param n_words     how many the caller has
 * @param resident_ok HTP_GRAPH_KIND_BIT mask of the kinds this build runs
 * @param n_ops_out   the op count on success (may be NULL)
 * @return 0, or one code per finding (the mutation table of
 *         graph_host_check.c): INVALIDFORMAT for magic and every shape,
 *         UNSUPPORTED for the version, INCOMPLETEITEM for a short list,
 *         NOTYPE for a kind this header does not know, INVALIDITEM for an
 *         op in the wrong layer kind, BADITEM for a next_mm that does not
 *         point forward at a weight-streaming op, CLASSNOTSUPPORT for a
 *         resident bit on a kind with no kernel here
 */
static inline uint32_t htp_graph_validate(const uint32_t *w, uint32_t n_words,
                                          uint32_t resident_ok,
                                          uint32_t *n_ops_out) {
  uint32_t n_layers, n_ops, hidden, vocab, max_seq, i, prev_layer = 0;
  if (w == NULL || n_words < HTP_GRAPH_HEADER_WORDS)
    return HTP_GRAPH_E_INCOMPLETEITEM;
  if (w[0] != HTP_GRAPH_MAGIC)
    return HTP_GRAPH_E_INVALIDFORMAT;
  if (w[1] != HTP_GRAPH_VERSION)
    return HTP_GRAPH_E_UNSUPPORTED;
  n_layers = w[2];
  n_ops = w[3];
  hidden = w[4];
  vocab = w[5];
  max_seq = w[6];
  if (n_layers == 0u || n_layers > HTP_GRAPH_MAX_LAYERS || n_ops == 0u ||
      n_ops > HTP_GRAPH_MAX_OPS || hidden == 0u || hidden % 32u != 0u ||
      vocab == 0u || max_seq == 0u)
    return HTP_GRAPH_E_INVALIDFORMAT;
  if (n_words < htp_graph_words_for(n_layers, n_ops))
    return HTP_GRAPH_E_INCOMPLETEITEM;
  for (i = 0; i < 2u * n_layers; ++i)
    if (w[HTP_GRAPH_HEADER_WORDS + i] > 1u)
      return HTP_GRAPH_E_INVALIDFORMAT;

  for (i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(w, i);
    const uint32_t k = op->kind;
    uint32_t layer_kind, ffn_kind;
    if (k >= HTP_OP_KIND_N)
      return HTP_GRAPH_E_NOTYPE;
    if (op->layer > n_layers || op->layer < prev_layer)
      return HTP_GRAPH_E_INVALIDITEM;
    prev_layer = op->layer;
    if (op->layer == n_layers) {
      /* The tail: final norm then lm_head, the last op. */
      if (k != HTP_OP_RMSNORM && k != HTP_OP_LM_HEAD)
        return HTP_GRAPH_E_INVALIDITEM;
      layer_kind = HTP_GRAPH_LAYER_CONV;
      ffn_kind = HTP_GRAPH_FFN_DENSE;
    } else {
      layer_kind = w[HTP_GRAPH_HEADER_WORDS + op->layer];
      ffn_kind = w[HTP_GRAPH_HEADER_WORDS + n_layers + op->layer];
    }
    if ((k == HTP_OP_LM_HEAD) != (i == n_ops - 1u))
      return HTP_GRAPH_E_INVALIDITEM;
    if ((k == HTP_OP_QK_NORM || k == HTP_OP_ROPE || k == HTP_OP_ATTN_M1) &&
        layer_kind != HTP_GRAPH_LAYER_ATTN)
      return HTP_GRAPH_E_INVALIDITEM;
    if (k == HTP_OP_CONV1D_GATE && layer_kind != HTP_GRAPH_LAYER_CONV)
      return HTP_GRAPH_E_INVALIDITEM;
    if ((k == HTP_OP_MOE || k == HTP_OP_ROUTER_TOPK) &&
        ffn_kind != HTP_GRAPH_FFN_MOE)
      return HTP_GRAPH_E_INVALIDITEM;
    if (k == HTP_OP_DENSE_FFN && ffn_kind != HTP_GRAPH_FFN_DENSE)
      return HTP_GRAPH_E_INVALIDITEM;

    if (op->K == 0u || op->N == 0u || op->in_slot >= HTP_GRAPH_N_SLOTS ||
        op->out_slot >= HTP_GRAPH_N_SLOTS)
      return HTP_GRAPH_E_INVALIDFORMAT;
    switch (k) {
    case HTP_OP_RMSNORM:
    case HTP_OP_QK_NORM:
    case HTP_OP_ROPE:
      if (op->K != op->N)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    case HTP_OP_CONV1D_GATE:
      if (op->K != 3u * op->N || op->N != hidden)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    case HTP_OP_ATTN_M1:
    case HTP_OP_ADD:
      if (op->N != hidden || (k == HTP_OP_ADD && op->K != hidden))
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    case HTP_OP_ROUTER_TOPK:
      if (op->K != hidden || op->N != op->n_experts || op->n_experts == 0u ||
          op->n_experts > HTP_GRAPH_MAX_EXPERTS || op->top_k == 0u ||
          op->top_k > op->n_experts)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    case HTP_OP_MOE:
    case HTP_OP_DENSE_FFN:
      if (k == HTP_OP_MOE &&
          (op->n_experts == 0u || op->n_experts > HTP_GRAPH_MAX_EXPERTS ||
           op->top_k == 0u || op->top_k > op->n_experts))
        return HTP_GRAPH_E_INVALIDFORMAT;
      if (op->K != hidden || op->N_out != hidden || op->N % 32u != 0u)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    case HTP_OP_FC:
      if (op->K % 32u != 0u || op->N % 32u != 0u)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    case HTP_OP_LM_HEAD:
      if (op->K != hidden || op->N != vocab)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    default:
      break;
    }
    if (op->next_mm != HTP_GRAPH_NO_OP &&
        (op->next_mm <= i || op->next_mm >= n_ops ||
         !htp_graph_kind_streams_weights(
           htp_graph_op_cat(w, op->next_mm)->kind)))
      return HTP_GRAPH_E_BADITEM;
    if (op->resident != 0u && (resident_ok & HTP_GRAPH_KIND_BIT(k)) == 0u)
      return HTP_GRAPH_E_CLASSNOTSUPPORT;
  }
  if (n_ops_out != NULL)
    *n_ops_out = n_ops;
  return 0u;
}

/** @brief The LFM2 decode step's shape, from config.json / nntr_config.json. */
typedef struct {
  uint32_t n_layers;
  uint32_t n_dense_layers; /**< layers [0, n) keep the dense SwiGLU FFN */
  uint32_t hidden;
  uint32_t inter_dense; /**< intermediate_size */
  uint32_t inter_moe;   /**< moe_intermediate_size */
  uint32_t n_experts;
  uint32_t top_k;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t vocab;
  uint32_t max_seq;
} htp_graph_lfm2_shape;

static inline htp_graph_op *
htp_graph_lfm2_emit(uint32_t *w, uint32_t *n, uint32_t kind, uint32_t layer,
                    uint32_t K, uint32_t N, uint32_t in_slot, uint32_t out_slot,
                    uint32_t resident_mask) {
  htp_graph_op *op = htp_graph_op_at(w, (*n)++);
  memset(op, 0, sizeof(*op));
  op->kind = kind;
  op->layer = layer;
  op->resident = (resident_mask & HTP_GRAPH_KIND_BIT(kind)) != 0u;
  op->K = K;
  op->N = N;
  op->N_out = N;
  op->in_slot = in_slot;
  op->out_slot = out_slot;
  op->next_mm = HTP_GRAPH_NO_OP;
  return op;
}

/**
 * @brief Builds the LFM2 decode op list (lfm2_causallm.cpp's block order:
 *        operator_norm, in_proj / qkv, conv1d+gate / qk norm + RoPE +
 *        attention, out_proj / o_proj, residual add, ffn_norm, the FFN,
 *        residual add; then the final norm and the tied lm_head).
 * @param w             HTP_GRAPH_MAX_OPS' worth of words or more (cap)
 * @param cap           words available in @a w
 * @param s             the shape
 * @param layer_is_attn n_layers bytes, 1 for an attention layer
 * @param resident_mask HTP_GRAPH_KIND_BIT mask: which kinds get the bit
 * @return the words written, or 0 when @a cap or HTP_GRAPH_MAX_OPS is
 *         too small. Slot use: 0 = the residual stream, 1 / 2 = working.
 *         MoE handles are left 0 for the ARM to bind (htp_compute_ops.cpp).
 */
static inline uint32_t htp_graph_lfm2_build(uint32_t *w, uint32_t cap,
                                            const htp_graph_lfm2_shape *s,
                                            const uint8_t *layer_is_attn,
                                            uint32_t resident_mask) {
  const uint32_t h = s->hidden;
  const uint32_t qkv = (s->n_heads + 2u * s->n_kv_heads) * s->head_dim;
  uint32_t n = 0, l, i, last_mm;
  /* 11 per attention layer, 9 per conv layer, 2 for the tail */
  uint32_t n_ops_max = 0;
  if (s->n_layers == 0u || s->n_layers > HTP_GRAPH_MAX_LAYERS)
    return 0u;
  for (l = 0; l < s->n_layers; ++l)
    n_ops_max += layer_is_attn[l] ? 11u : 9u;
  n_ops_max += 2u;
  if (n_ops_max > HTP_GRAPH_MAX_OPS ||
      cap < htp_graph_words_for(s->n_layers, n_ops_max))
    return 0u;
  w[0] = HTP_GRAPH_MAGIC;
  w[1] = HTP_GRAPH_VERSION;
  w[2] = s->n_layers;
  w[3] = 0u; /* patched below; htp_graph_op_at only reads w[2] */
  w[4] = h;
  w[5] = s->vocab;
  w[6] = s->max_seq;
  for (l = 0; l < s->n_layers; ++l) {
    w[HTP_GRAPH_HEADER_WORDS + l] =
      layer_is_attn[l] ? HTP_GRAPH_LAYER_ATTN : HTP_GRAPH_LAYER_CONV;
    w[HTP_GRAPH_HEADER_WORDS + s->n_layers + l] =
      l < s->n_dense_layers ? HTP_GRAPH_FFN_DENSE : HTP_GRAPH_FFN_MOE;
  }
  for (l = 0; l < s->n_layers; ++l) {
    htp_graph_lfm2_emit(w, &n, HTP_OP_RMSNORM, l, h, h, 0, 1, resident_mask);
    if (layer_is_attn[l]) {
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, qkv, 1, 2, resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_QK_NORM, l, qkv, qkv, 2, 2,
                          resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_ROPE, l, qkv, qkv, 2, 2, resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_ATTN_M1, l, qkv, h, 2, 1,
                          resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, h, 1, 2, resident_mask);
    } else {
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, 3u * h, 1, 2, resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_CONV1D_GATE, l, 3u * h, h, 2, 1,
                          resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, h, 1, 2, resident_mask);
    }
    htp_graph_lfm2_emit(w, &n, HTP_OP_ADD, l, h, h, 2, 0, resident_mask);
    htp_graph_lfm2_emit(w, &n, HTP_OP_RMSNORM, l, h, h, 0, 1, resident_mask);
    if (l < s->n_dense_layers) {
      htp_graph_op *op = htp_graph_lfm2_emit(
        w, &n, HTP_OP_DENSE_FFN, l, h, s->inter_dense, 1, 2, resident_mask);
      op->N_out = h;
    } else {
      htp_graph_op *op = htp_graph_lfm2_emit(w, &n, HTP_OP_ROUTER_TOPK, l, h,
                                             s->n_experts, 1, 2, resident_mask);
      op->n_experts = s->n_experts;
      op->top_k = s->top_k;
      op = htp_graph_lfm2_emit(w, &n, HTP_OP_MOE, l, h, s->inter_moe, 1, 2,
                               resident_mask);
      op->N_out = h;
      op->n_experts = s->n_experts;
      op->top_k = s->top_k;
    }
    htp_graph_lfm2_emit(w, &n, HTP_OP_ADD, l, h, h, 2, 0, resident_mask);
  }
  htp_graph_lfm2_emit(w, &n, HTP_OP_RMSNORM, s->n_layers, h, h, 0, 1,
                      resident_mask);
  htp_graph_lfm2_emit(w, &n, HTP_OP_LM_HEAD, s->n_layers, h, s->vocab, 1, 2,
                      resident_mask);
  w[3] = n;
  /* next_mm: walk backwards, each op points at the nearest weight
     streamer after it. */
  last_mm = HTP_GRAPH_NO_OP;
  for (i = n; i-- > 0u;) {
    htp_graph_op *op = htp_graph_op_at(w, i);
    op->next_mm = last_mm;
    if (htp_graph_kind_streams_weights(op->kind))
      last_mm = i;
  }
  return htp_graph_words_for(s->n_layers, n);
}

#endif /* __HTP_GRAPH_DESC_H__ */
