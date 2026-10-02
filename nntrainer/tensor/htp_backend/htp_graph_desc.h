// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_graph_desc.h
 * @date   23 Sep 2026
 * @brief  The per-token decode graph as the DSP receives it: wire format,
 *         validator and the LFM2 builder, as pure C99 (#85, v2 in #130)
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
 * Version 2 (#130) adds four words to the op record -- n_kv, gqa,
 * head_dim for the attention kinds and eps_bits for the norms -- and the
 * kernel shape rules of #81 / #82 to the validator. A v1 skel refuses a
 * v2 list with AEE_EUNSUPPORTED, never AEE_EBADPARM.
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
#define HTP_GRAPH_VERSION 2u
#define HTP_GRAPH_HEADER_WORDS 7u
/** @brief LFM2.5-8B-A1B is 228 ops (section htp_graph_lfm2_build),
 *  Gemma-4-26B-A4B about 25 a layer x 30 (plan 201 section 2.4); the table
 *  is HTP_GRAPH_OP_WORDS x 4 B per op, so 1024 is 320 KiB of DSP heap at
 *  graph_init (address-space note in hexkl_graph.c). */
#define HTP_GRAPH_MAX_OPS 1024u
#define HTP_GRAPH_MAX_LAYERS 64u
/** @brief A MOE / ROUTER_TOPK op's expert count (Gemma: 128). A MOE op's
 *  handles are not in its record: they are its HTP_GRAPH_PARAM_EXPERTS
 *  table (plan 201 S1, the expert pool's table). */
#define HTP_GRAPH_MAX_EXPERTS 128u
/** @brief A resident sigmoid ROUTER_TOPK's expert count: one vector of
 *  lanes (hvx_router_topk_f32, LFM2). The softmax router (eps_bits set,
 *  plan 201 S4) takes HTP_GRAPH_MAX_EXPERTS. */
#define HTP_GRAPH_ROUTER_MAX_EXPERTS 32u
/** @brief The op record's handle arrays: the Q4M1 kinds' parts. */
#define HTP_GRAPH_MAX_PARTS 32u
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
/** @brief Every kind: the one-session all-resident mask (#132 Part B E1). */
#define HTP_GRAPH_KINDS_ALL ((1u << HTP_OP_KIND_N) - 1u)
/** @brief The kinds whose weights are Q4M1 handles (#132 Part B): the
 *  CPU-exact Q4_0 FC and its two compositions. */
#define HTP_GRAPH_KINDS_Q4M1                                                   \
  (HTP_GRAPH_KIND_BIT(HTP_OP_FC) | HTP_GRAPH_KIND_BIT(HTP_OP_DENSE_FFN) |      \
   HTP_GRAPH_KIND_BIT(HTP_OP_LM_HEAD))

/** @brief The kind's name, for log lines and NNTR_HTP_FORWARD_KINDS. */
static inline const char *htp_graph_kind_name(uint32_t k) {
  static const char *const names[HTP_OP_KIND_N] = {
    "RMSNORM", "FC",          "CONV1D_GATE", "QK_NORM",   "ROPE",   "ATTN_M1",
    "ADD",     "ROUTER_TOPK", "MOE",         "DENSE_FFN", "LM_HEAD"};
  return k < HTP_OP_KIND_N ? names[k] : "?";
}

/** @brief Parses a comma-separated list of kind names ("MOE,RMSNORM")
 *  into a HTP_GRAPH_KIND_BIT mask. @return the mask, or 0 for an unknown
 *  or empty name (a caller with a default checks for NULL first). */
static inline uint32_t htp_graph_kinds_parse(const char *s) {
  uint32_t mask = 0;
  while (s != NULL && *s != '\0') {
    const char *e = s;
    uint32_t k;
    size_t n;
    while (*e != '\0' && *e != ',')
      ++e;
    n = (size_t)(e - s);
    for (k = 0; k < HTP_OP_KIND_N; ++k) {
      const char *name = htp_graph_kind_name(k);
      if (strlen(name) == n && strncmp(name, s, n) == 0)
        break;
    }
    if (k == HTP_OP_KIND_N)
      return 0u;
    mask |= HTP_GRAPH_KIND_BIT(k);
    s = (*e == ',') ? e + 1 : e;
  }
  return mask;
}

/** @brief Writes the mask as "MOE|RMSNORM|..." ("-" when empty) into
 *  @a buf of @a cap bytes (128 holds every kind) and returns it. */
static inline const char *htp_graph_kinds_str(uint32_t mask, char *buf,
                                              size_t cap) {
  size_t off = 0;
  uint32_t k;
  buf[0] = '\0';
  for (k = 0; k < HTP_OP_KIND_N && off + 1 < cap; ++k) {
    const char *name;
    size_t n;
    if ((mask & HTP_GRAPH_KIND_BIT(k)) == 0u)
      continue;
    name = htp_graph_kind_name(k);
    n = strlen(name);
    if (off + n + 2 > cap)
      break;
    if (off != 0)
      buf[off++] = '|';
    memcpy(buf + off, name, n + 1);
    off += n;
  }
  if (off == 0 && cap > 1) {
    buf[0] = '-';
    buf[1] = '\0';
  }
  return buf;
}

/** @brief graph_set_param's `which` (#130): the f32 parameters an op
 *  reads, bound once after graph_init. GAMMA is K floats for RMSNORM and
 *  2 x head_dim (q gamma | k gamma) for QK_NORM; CONV_W is 3 x N
 *  (w0 | w1 | w2) and CONV_STATE 2 x N (x_{t-2} | x_{t-1}) for
 *  CONV1D_GATE; ROPE_TABLE is max_seq x 64 (cos[32] | sin[32] per
 *  position) with op == HTP_GRAPH_NO_OP, the head_dim 64 table every ROPE
 *  op without its own reads; [plan 201 S4] on a ROPE op it is that op's own
 *  table, max_seq x head_dim (cos[hd / 2] | sin[hd / 2] per position, from
 *  the CPU's precompute_freqs: the layer's theta and rotary variant, e.g.
 *  Gemma 4's proportional RoPE with partial factor 0.25 as zero angles),
 *  shared between ROPE ops that bind equal tables; ROUTER_W is K x n_experts
 *  (the gate weight, row-major [K][E]) and ROUTER_BIAS n_experts for
 *  ROUTER_TOPK (#132); [plan 201 S4] for the softmax router (eps_bits set)
 *  ROUTER_BIAS is K + n_experts: the input scale g[K]
 *  (m1_router_input_scale_det: router_scale / sqrt(K)) then the per-expert
 *  scale. [#132 Part B E3] LM_BAN is 0..HTP_GRAPH_MAX_BAN
 *  token ids (u32 bits in the f32 words, each < N; 0 clears) for LM_HEAD:
 *  its argmax
 *  skips them, as the CPU's bad-words penalty sets those logits to -inf
 *  before its first-maximum pick; the logits themselves stay raw. A later
 *  LM_BAN replaces the list. [plan 201 S1] EXPERTS is a MOE op's pool
 *  table, 2 x n_experts u32 handle words (h_gu[0..E) then h_dn[0..E)), each
 *  a registered weight of the op's shape or HTP_GRAPH_NO_HANDLE (not
 *  resident: forward refuses a token that routes to it); a later EXPERTS
 *  replaces the table. Append, never reorder. */
enum {
  HTP_GRAPH_PARAM_GAMMA = 0,
  HTP_GRAPH_PARAM_CONV_W,
  HTP_GRAPH_PARAM_CONV_STATE,
  HTP_GRAPH_PARAM_ROPE_TABLE,
  HTP_GRAPH_PARAM_ROUTER_W,
  HTP_GRAPH_PARAM_ROUTER_BIAS,
  HTP_GRAPH_PARAM_LM_BAN,
  HTP_GRAPH_PARAM_EXPERTS,
  HTP_GRAPH_PARAM_N
};
/** @brief LM_BAN's longest list. */
#define HTP_GRAPH_MAX_BAN 32u
/** @brief An EXPERTS table entry for an expert that is not resident. */
#define HTP_GRAPH_NO_HANDLE 0xFFFFFFFFu

enum { HTP_GRAPH_LAYER_CONV = 0, HTP_GRAPH_LAYER_ATTN = 1 };
enum { HTP_GRAPH_FFN_DENSE = 0, HTP_GRAPH_FFN_MOE = 1 };

/**
 * @brief One op record, HTP_GRAPH_OP_WORDS words on the wire.
 *
 * K is the input width, N the output width (inter for MOE / DENSE_FFN,
 * whose output width is N_out; vocab for LM_HEAD). in_slot / out_slot name
 * the session's
 * activation slots (plan 85 section 3.1); ADD reads slot 0 as its second
 * operand implicitly. next_mm names the next weight-streaming op (FC,
 * MOE, DENSE_FFN, LM_HEAD) or HTP_GRAPH_NO_OP; nothing consumes it yet
 * (the cross-op prefetch hook is a later issue), the validator only
 * requires it to point forward. h_gu / h_dn are the Q4M1 kinds' weight
 * handles, bound by the ARM before graph_init (a MOE op leaves them unused:
 * its handles are its HTP_GRAPH_PARAM_EXPERTS table, plan 201 S1), with
 * n_experts the part count (#132 Part B): an FC's parts
 * are h_gu[0..n_experts), K x N_p each, their outputs concatenated (q | k
 * | v: three parts, one quantization), an LM_HEAD's are its vocab slices
 * likewise, and a DENSE_FFN has h_gu[0] = up, h_gu[1] = gate (K x N each)
 * and h_dn[0] = down (N x N_out), n_experts 3. feed says how those
 * kinds' weights reach the vector unit: 0 VTCM by DMA when the session
 * has 2 groups per lane of it, else the L2 scratch; 1 the L2 scratch
 * (hexkl_graph.h); [#132 Part B E5f] bits 8-11 / 12-15 the L2 feed's
 * lanes for a part with K <= 2048 / K > 2048, 0 = the runner's default
 * (HTP_GRAPH_FEED_*); [#194 L1, htp_moe_ppl] bit 16 the native kernels
 * (q4_gemv_native_det.h: vector quantizer, no emulated FMA) in place of the
 * CPU-exact ones. Other kinds leave all of these 0. n_kv / gqa /
 * head_dim describe the attention kinds (QK_NORM,
 * ROPE, ATTN_M1: K == (gqa + 2) n_kv head_dim, ATTN_M1's N == gqa n_kv
 * head_dim) and eps_bits holds the f32 bits of the norm epsilon (RMSNORM,
 * QK_NORM); both are 0 elsewhere. [plan 201 S4] An ATTN_M1's eps_bits is
 * its score scale, which must be an fp16 value (0: 1/sqrt(head_dim),
 * LFM2's 0.125; Gemma 4 sets 1.0), and its top_k the sliding window (0:
 * every position; Gemma's sliding layers 1024). A ROUTER_TOPK with eps_bits
 * set is Gemma 4's softmax router: it RMS-norms its input itself (that
 * epsilon, ROUTER_BIAS's g as gamma), so it reads the un-normed stream,
 * then softmax, top-k, renormalise, per-expert scale (m1_ops_det.h); with
 * eps_bits 0 it is LFM2's sigmoid router. An RMSNORM's feed bit 0
 * (HTP_GRAPH_NORM_N1) selects N1, the order-free integer sum of squares
 * (opt-in, not the CPU's bits).
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
  uint32_t feed; /**< Q4M1 kinds: bit 0 the L2 feed (else VTCM if it
                     fits); bits 8-11 / 12-15 the L2 lanes, small / large K;
                     bit 16 the native kernels (#194 L1) */
  uint32_t n_kv;
  uint32_t gqa;
  uint32_t head_dim;
  uint32_t eps_bits;
  uint32_t h_gu[HTP_GRAPH_MAX_PARTS];
  uint32_t h_dn[HTP_GRAPH_MAX_PARTS];
} htp_graph_op;
#define HTP_GRAPH_OP_WORDS (16u + 2u * HTP_GRAPH_MAX_PARTS)
/** @brief The Q4M1 kinds' feed word (htp_graph_op.feed). */
#define HTP_GRAPH_FEED_L2 1u
#define HTP_GRAPH_FEED_NATIVE (1u << 16)
#define HTP_GRAPH_FEED_LANES_SMALL(f) (((f) >> 8) & 0xFu)
#define HTP_GRAPH_FEED_LANES_LARGE(f) (((f) >> 12) & 0xFu)
/** @brief An RMSNORM's feed word: N1 (hvx_rmsnorm_n1_f32, plan 201 S4).
 *  ponytail: RMSNORM only; QK_NORM and the router's own norm take it when
 *  N1 is chosen for the model. */
#define HTP_GRAPH_NORM_N1 1u
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
#define HTP_GRAPH_E_SCHEMENOTSUPPORTED (HTP_GRAPH_EOFFSET + 0x00Fu)
#define HTP_GRAPH_E_BADITEM (HTP_GRAPH_EOFFSET + 0x010u)
#define HTP_GRAPH_E_INVALIDFORMAT (HTP_GRAPH_EOFFSET + 0x011u)
#define HTP_GRAPH_E_INCOMPLETEITEM (HTP_GRAPH_EOFFSET + 0x012u)
#define HTP_GRAPH_E_UNSUPPORTED (HTP_GRAPH_EOFFSET + 0x014u)
#define HTP_GRAPH_E_NOTYPE (HTP_GRAPH_EOFFSET + 0x022u)
#define HTP_GRAPH_E_INVALIDITEM (HTP_GRAPH_EOFFSET + 0x02Au)
#define HTP_GRAPH_E_NOTALLOWED (HTP_GRAPH_EOFFSET + 0x02Bu)
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
  case 0x00Fu:
    return "AEE_ESCHEMENOTSUPPORTED";
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
  case 0x02Bu:
    return "AEE_ENOTALLOWED";
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
 *         op in the wrong layer kind or an attention op out of the
 *         QK_NORM -> ROPE -> ATTN_M1 order, BADITEM for a next_mm that
 *         does not point forward at a weight-streaming op, CLASSNOTSUPPORT
 *         for a resident bit on a kind with no kernel here,
 *         SCHEMENOTSUPPORTED for a resident op outside its kernel's shape
 *         rule (RMSNORM: K a multiple of 32 (plan 201 S4: any width,
 *         Gemma's 2816), its feed 0 or HTP_GRAPH_NORM_N1; QK_NORM:
 *         head_dim 32, 64 or 128 -- the per-head norm's chunk must be a
 *         power of two too -- gqa <= 8, max_seq a multiple of 32; ROPE
 *         and ATTN_M1 (#152: the fp16 CPU order): head_dim 64, since plan
 *         201 S4 any multiple of 64 up to 512 -- a ROPE at another than 64
 *         reads its own ROPE_TABLE, an ATTN_M1's scale is an fp16 value),
 *         NOTALLOWED for
 *         a resident
 *         ATTN_M1 whose layer's ROPE is not resident (the DSP stretch must
 *         apply RoPE, since mha_core does on the CPU). #132's rules: an ADD
 *         writes slot 0 and reads another slot (INVALIDFORMAT: slot 0 is
 *         its implicit second operand, the residual); a resident ADD needs
 *         every RMSNORM resident (NOTALLOWED: slot 0 holds the residual
 *         across calls only if op 0 seeds it each token and no CPU norm
 *         leaves a stretch start that would re-seed it); a resident
 *         ROUTER_TOPK needs the next op, its MOE, resident (NOTALLOWED:
 *         the routing has no other consumer). #132 Part B's rules for
 *         the Q4M1 kinds: n_experts (the part count) at most 32 and feed
 *         0 or 1 (INVALIDFORMAT); [plan 201 S1] a resident sigmoid
 *         ROUTER_TOPK over more than 32 experts (SCHEMENOTSUPPORTED: its
 *         vector width; [plan 201 S4] the softmax router takes 128, needs
 *         a valid eps and in_slot != out_slot, INVALIDFORMAT: it norms
 *         into its out slot); a resident Q4M1 kind needs K % 64 == 0 and K
 *         <= 8192, a DENSE_FFN also N (the down weight's K) % 64 == 0 and
 *         N <= 8192 -- the Q4M1 layout's pair and the quantizer's scratch
 *         (SCHEMENOTSUPPORTED)
 */
static inline uint32_t htp_graph_validate(const uint32_t *w, uint32_t n_words,
                                          uint32_t resident_ok,
                                          uint32_t *n_ops_out) {
  uint32_t n_layers, n_ops, hidden, vocab, max_seq, i, prev_layer = 0;
  uint32_t attn_layer = HTP_GRAPH_NO_OP, attn_stage = 0, rope_resident = 0;
  uint32_t add_resident = 0, rmsnorm_cpu = 0;
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
    case HTP_OP_ADD:
      if (op->N != hidden || op->K != hidden)
        return HTP_GRAPH_E_INVALIDFORMAT;
      if (op->out_slot != 0u || op->in_slot == 0u)
        return HTP_GRAPH_E_INVALIDFORMAT;
      break;
    /* ATTN_M1's N is gqa n_kv head_dim (the record rule below), the o
       projection's K: LFM2's 32 x 64 happens to be its hidden, Gemma 4's
       16 x 256 and 16 x 512 are not (plan 201 S4) */
    case HTP_OP_ROUTER_TOPK:
      if (op->K != hidden || op->N != op->n_experts || op->n_experts == 0u ||
          op->n_experts > HTP_GRAPH_MAX_EXPERTS || op->top_k == 0u ||
          op->top_k > op->n_experts ||
          (op->eps_bits != 0u && op->in_slot == op->out_slot))
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
    if ((HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(k)) != 0u) {
      if (op->n_experts > HTP_GRAPH_MAX_PARTS ||
          (op->feed & ~(HTP_GRAPH_FEED_L2 | 0xFF00u | HTP_GRAPH_FEED_NATIVE)) !=
            0u ||
          HTP_GRAPH_FEED_LANES_SMALL(op->feed) > 8u ||
          HTP_GRAPH_FEED_LANES_LARGE(op->feed) > 8u)
        return HTP_GRAPH_E_INVALIDFORMAT;
      if (op->resident != 0u &&
          (op->K % 64u != 0u || op->K > 8192u ||
           (k == HTP_OP_DENSE_FFN && (op->N % 64u != 0u || op->N > 8192u))))
        return HTP_GRAPH_E_SCHEMENOTSUPPORTED;
    }
    /* v2: the attention record, its in-layer order and the norm epsilon */
    if (k == HTP_OP_QK_NORM || k == HTP_OP_ROPE || k == HTP_OP_ATTN_M1) {
      if (op->n_kv == 0u || op->gqa == 0u || op->head_dim == 0u ||
          op->K != (op->gqa + 2u) * op->n_kv * op->head_dim ||
          (k == HTP_OP_ATTN_M1 && op->N != op->gqa * op->n_kv * op->head_dim))
        return HTP_GRAPH_E_INVALIDFORMAT;
      if (op->layer != attn_layer) {
        attn_layer = op->layer;
        attn_stage = 0u;
        rope_resident = 0u;
      }
      if ((k == HTP_OP_QK_NORM && attn_stage != 0u) ||
          (k == HTP_OP_ROPE && attn_stage != 1u) ||
          (k == HTP_OP_ATTN_M1 && attn_stage != 2u))
        return HTP_GRAPH_E_INVALIDITEM;
      ++attn_stage;
      if (k == HTP_OP_ROPE)
        rope_resident = op->resident;
      if (op->resident != 0u) {
        if (op->gqa > 8u || max_seq % 32u != 0u ||
            (k == HTP_OP_QK_NORM &&
             (op->head_dim % 32u != 0u || op->head_dim > 128u ||
              (op->head_dim & (op->head_dim - 1u)) != 0u)) ||
            ((k == HTP_OP_ROPE || k == HTP_OP_ATTN_M1) &&
             (op->head_dim % 64u != 0u || op->head_dim > 512u)))
          return HTP_GRAPH_E_SCHEMENOTSUPPORTED;
        /* [plan 201 S4] the kernel's scale is one hf multiply: eps_bits an
           fp16 normal (sign clear, 13 low mantissa bits clear, 2^-14 ..
           65504), or 0 only where 1/sqrt(head_dim) is one (64, 256) */
        if (k == HTP_OP_ATTN_M1 &&
            (op->eps_bits != 0u
               ? ((op->eps_bits & 0x80001FFFu) != 0u ||
                  ((op->eps_bits >> 23) & 0xFFu) < 113u ||
                  ((op->eps_bits >> 23) & 0xFFu) > 142u)
               : (op->head_dim != 64u && op->head_dim != 256u)))
          return HTP_GRAPH_E_SCHEMENOTSUPPORTED;
        if (k == HTP_OP_ATTN_M1 && rope_resident == 0u)
          return HTP_GRAPH_E_NOTALLOWED;
      }
    }
    if (k == HTP_OP_RMSNORM || k == HTP_OP_QK_NORM ||
        (k == HTP_OP_ROUTER_TOPK && op->eps_bits != 0u)) {
      /* a positive finite f32: sign clear, exponent neither 0 nor 0xFF */
      const uint32_t exp = op->eps_bits & 0x7F800000u;
      if ((op->eps_bits & 0x80000000u) != 0u || exp == 0u || exp == 0x7F800000u)
        return HTP_GRAPH_E_INVALIDFORMAT;
    }
    if (k == HTP_OP_RMSNORM && (op->feed & ~HTP_GRAPH_NORM_N1) != 0u)
      return HTP_GRAPH_E_INVALIDFORMAT;
    if (k == HTP_OP_RMSNORM && op->resident != 0u && op->K % 32u != 0u)
      return HTP_GRAPH_E_SCHEMENOTSUPPORTED;
    if (k == HTP_OP_ROUTER_TOPK && op->resident != 0u && op->eps_bits == 0u &&
        op->n_experts > HTP_GRAPH_ROUTER_MAX_EXPERTS)
      return HTP_GRAPH_E_SCHEMENOTSUPPORTED;
    if (k == HTP_OP_ROUTER_TOPK && op->resident != 0u &&
        (i + 1u >= n_ops || htp_graph_op_cat(w, i + 1u)->kind != HTP_OP_MOE ||
         htp_graph_op_cat(w, i + 1u)->resident == 0u))
      return HTP_GRAPH_E_NOTALLOWED;
    add_resident |= k == HTP_OP_ADD && op->resident != 0u;
    rmsnorm_cpu |= k == HTP_OP_RMSNORM && op->resident == 0u;
    if (op->next_mm != HTP_GRAPH_NO_OP &&
        (op->next_mm <= i || op->next_mm >= n_ops ||
         !htp_graph_kind_streams_weights(
           htp_graph_op_cat(w, op->next_mm)->kind)))
      return HTP_GRAPH_E_BADITEM;
    if (op->resident != 0u && (resident_ok & HTP_GRAPH_KIND_BIT(k)) == 0u)
      return HTP_GRAPH_E_CLASSNOTSUPPORT;
  }
  if (add_resident && rmsnorm_cpu)
    return HTP_GRAPH_E_NOTALLOWED;
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
  float eps; /**< rms_norm_eps (Transformer::NORM_EPS), every norm */
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
 *         A MOE op's handles are its EXPERTS table, bound by the ARM
 *         after graph_init (htp_compute_ops.cpp).
 */
static inline uint32_t htp_graph_lfm2_build(uint32_t *w, uint32_t cap,
                                            const htp_graph_lfm2_shape *s,
                                            const uint8_t *layer_is_attn,
                                            uint32_t resident_mask) {
  const uint32_t h = s->hidden;
  const uint32_t qkv = (s->n_heads + 2u * s->n_kv_heads) * s->head_dim;
  uint32_t n = 0, l, i, last_mm, eps_bits;
  htp_graph_op *op;
  /* 11 per attention layer, 9 per conv layer, 2 for the tail */
  uint32_t n_ops_max = 0;
  if (s->n_layers == 0u || s->n_layers > HTP_GRAPH_MAX_LAYERS ||
      s->n_kv_heads == 0u || s->n_heads % s->n_kv_heads != 0u)
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
  memcpy(&eps_bits, &s->eps, sizeof(eps_bits));
  for (l = 0; l < s->n_layers; ++l) {
    w[HTP_GRAPH_HEADER_WORDS + l] =
      layer_is_attn[l] ? HTP_GRAPH_LAYER_ATTN : HTP_GRAPH_LAYER_CONV;
    w[HTP_GRAPH_HEADER_WORDS + s->n_layers + l] =
      l < s->n_dense_layers ? HTP_GRAPH_FFN_DENSE : HTP_GRAPH_FFN_MOE;
  }
  for (l = 0; l < s->n_layers; ++l) {
    op =
      htp_graph_lfm2_emit(w, &n, HTP_OP_RMSNORM, l, h, h, 0, 1, resident_mask);
    op->eps_bits = eps_bits;
    if (layer_is_attn[l]) {
      uint32_t a;
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, qkv, 1, 2, resident_mask);
      op = htp_graph_lfm2_emit(w, &n, HTP_OP_QK_NORM, l, qkv, qkv, 2, 2,
                               resident_mask);
      op->eps_bits = eps_bits;
      htp_graph_lfm2_emit(w, &n, HTP_OP_ROPE, l, qkv, qkv, 2, 2, resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_ATTN_M1, l, qkv, h, 2, 1,
                          resident_mask);
      for (a = n - 3u; a < n; ++a) {
        op = htp_graph_op_at(w, a);
        op->n_kv = s->n_kv_heads;
        op->gqa = s->n_heads / s->n_kv_heads;
        op->head_dim = s->head_dim;
      }
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, h, 1, 2, resident_mask);
    } else {
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, 3u * h, 1, 2, resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_CONV1D_GATE, l, 3u * h, h, 2, 1,
                          resident_mask);
      htp_graph_lfm2_emit(w, &n, HTP_OP_FC, l, h, h, 1, 2, resident_mask);
    }
    htp_graph_lfm2_emit(w, &n, HTP_OP_ADD, l, h, h, 2, 0, resident_mask);
    op =
      htp_graph_lfm2_emit(w, &n, HTP_OP_RMSNORM, l, h, h, 0, 1, resident_mask);
    op->eps_bits = eps_bits;
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
  op = htp_graph_lfm2_emit(w, &n, HTP_OP_RMSNORM, s->n_layers, h, h, 0, 1,
                           resident_mask);
  op->eps_bits = eps_bits;
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
