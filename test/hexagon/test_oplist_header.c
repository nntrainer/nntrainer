// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_oplist_header.c
 * @date	15 August 2026
 * @brief	x86 self-check for the op-list header validation shared by host and
 * DSP.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "../../nntrainer/tensor/hexagon/htp/nntr_htp_common.h"
#include "../../nntrainer/tensor/hexagon/htp/nntr_htp_rope.h"

/**
 * @brief Build a valid 2-op list: RMSNORM followed by MATMUL_W8A8.
 */
static void build_valid(struct nntr_htp_oplist_header *h,
                        struct nntr_htp_op_desc ops[2],
                        uint32_t buf_size[NNTR_HTP_BUF_COUNT]) {
  memset(h, 0, sizeof(*h));
  h->magic = NNTR_HTP_OPLIST_MAGIC;
  h->version = NNTR_HTP_ABI_VERSION;
  h->n_ops = 2;
  h->n_layers = 1;
  h->n_heads = 2;
  h->n_kv_heads = 2;
  h->head_dim = 128;
  h->hidden = 128;
  h->ffn = 128;
  h->vocab = 32;
  h->max_seq = 8;
  h->max_chunk = 4;
  h->weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;

  memset(ops, 0, 2 * sizeof(*ops));
  /** op0: RMSNORM, x[ACT@0] * gamma[WEIGHTS@0] -> out[ACT@1024] */
  ops[0].kind = NNTR_HTP_OP_RMSNORM;
  ops[0].m = 0;
  ops[0].n = h->hidden;
  ops[0].in0.buf = NNTR_HTP_BUF_ACT;
  ops[0].in0.offset = 0;
  ops[0].in1.buf = NNTR_HTP_BUF_WEIGHTS;
  ops[0].in1.offset = 0;
  ops[0].out.buf = NNTR_HTP_BUF_ACT;
  ops[0].out.offset = 1024;

  /** op1: MATMUL_W8A8, X[ACT@1024] x W[WEIGHTS@256] -> out[ACT@2048] */
  ops[1].kind = NNTR_HTP_OP_MATMUL_W8A8;
  ops[1].m = 0;
  ops[1].k = h->hidden;
  ops[1].n = 256;
  ops[1].in0.buf = NNTR_HTP_BUF_ACT;
  ops[1].in0.offset = 1024;
  ops[1].in1.buf = NNTR_HTP_BUF_WEIGHTS;
  ops[1].in1.offset = 256;
  ops[1].in2.buf = NNTR_HTP_BUF_WEIGHTS;
  ops[1].in2.offset = 33024;
  ops[1].out.buf = NNTR_HTP_BUF_ACT;
  ops[1].out.offset = 2048;

  buf_size[NNTR_HTP_BUF_WEIGHTS] = 65536;
  buf_size[NNTR_HTP_BUF_KV] = 0;
  buf_size[NNTR_HTP_BUF_ACT] = 8192;
  buf_size[NNTR_HTP_BUF_TOKENS] = h->max_chunk * 4;
  buf_size[NNTR_HTP_BUF_LOGITS] = h->vocab * 4;
}

int main(void) {
  struct nntr_htp_oplist_header h = {NNTR_HTP_OPLIST_MAGIC,
                                     NNTR_HTP_ABI_VERSION, 0, 0};

  /** v4: reserved2[0] became weight_layout; the record stays 64 bytes.
   * v5 (#65 S1): kind 9 and two layout ids, no record change. */
  assert(NNTR_HTP_ABI_VERSION == 5u);
  assert(sizeof(struct nntr_htp_oplist_header) == 64u);
  assert(offsetof(struct nntr_htp_oplist_header, weight_layout) == 52u);
  assert(NNTR_HTP_OP_MATMUL_W4A8 == 9u && NNTR_HTP_OP_KIND_COUNT == 10u);
  assert(NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8 == 2u &&
         NNTR_HTP_WEIGHT_LAYOUT_W4CX == 3u);

  /* tile_off by example (spec P2 "tile definition"): K = 256 -> k_tiles = 2. */
  assert(nntr_htp_tile_off(0u, 0u, 256u) == 0u);
  assert(nntr_htp_tile_off(0u, 1u, 256u) == 1u);   /* k%4 is the byte */
  assert(nntr_htp_tile_off(1u, 0u, 256u) == 4u);   /* row r -> lane 4r */
  assert(nntr_htp_tile_off(0u, 4u, 256u) == 128u); /* 4k group g -> vector g */
  assert(nntr_htp_tile_off(31u, 127u, 256u) ==
         4095u); /* last byte of tile(0,0) */
  assert(nntr_htp_tile_off(0u, 128u, 256u) == 4096u); /* tile(0,1) */
  assert(nntr_htp_tile_off(32u, 0u, 256u) ==
         8192u); /* tile(1,0) = k_tiles*4096 */

  /* repack is a bijection onto [0, N*K) and tile_off is its inverse. */
  {
    enum { TN = 64, TK = 256 };
    static uint8_t src[TN * TK], dst[TN * TK], seen[TN * TK];
    uint32_t n, k;
    for (n = 0; n < (uint32_t)(TN * TK); ++n)
      src[n] = (uint8_t)(n * 31u + 7u);
    memset(seen, 0, sizeof(seen));
    nntr_htp_repack_tiled32(dst, src, TN, TK);
    for (n = 0; n < TN; ++n)
      for (k = 0; k < TK; ++k) {
        uint32_t o = nntr_htp_tile_off(n, k, TK);
        assert(o < (uint32_t)(TN * TK) && !seen[o]);
        seen[o] = 1;
        assert(dst[o] == src[n * TK + k]);
      }
  }

  /** w4cx tile index: K = 64 -> k_tiles = 2; nibble index k*32 + n inside
   * a 512 B tile, n even = low nibble. */
  assert(nntr_htp_w4_tile_off(0u, 0u, 64u) == 0u);
  assert(nntr_htp_w4_tile_off(1u, 0u, 64u) == 0u);     /* same byte, high */
  assert(nntr_htp_w4_tile_off(2u, 0u, 64u) == 1u);     /* next byte */
  assert(nntr_htp_w4_tile_off(0u, 1u, 64u) == 16u);    /* next k row */
  assert(nntr_htp_w4_tile_off(31u, 31u, 64u) == 511u); /* last byte */
  assert(nntr_htp_w4_tile_off(0u, 32u, 64u) == 512u);  /* tile(0,1) */
  assert(nntr_htp_w4_tile_off(32u, 0u, 64u) == 1024u); /* tile(1,0) */

  /** repack_w4cx: every code lands where w4_get reads it, colsum is the
   * row sum, the [-8, 7] range is enforced. */
  {
    enum { WN = 64, WK = 64 };
    static int8_t src[WN * WK];
    static uint8_t dst[WN * WK / 2];
    static int32_t cs[WN];
    uint32_t n, k;
    for (n = 0; n < (uint32_t)(WN * WK); ++n)
      src[n] = (int8_t)((int)(n * 7u % 16u) - 8);
    assert(nntr_htp_repack_w4cx(dst, cs, src, WN, WK) == 0);
    for (n = 0; n < WN; ++n) {
      int32_t sum = 0;
      for (k = 0; k < WK; ++k) {
        assert(nntr_htp_w4_get(dst, n, k, WK) == src[n * WK + k]);
        sum += src[n * WK + k];
      }
      assert(cs[n] == sum);
    }
    src[5] = 8; /* not an int4 code */
    assert(nntr_htp_repack_w4cx(dst, cs, src, WN, WK) == 1);
  }

  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 0);

  h.version = 999u;
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 3);
  h.version = NNTR_HTP_ABI_VERSION;

  h.magic = 0;
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 2);
  h.magic = NNTR_HTP_OPLIST_MAGIC;

  assert(nntr_htp_oplist_check(&h, 3) == 1);
  assert(nntr_htp_oplist_check(0, sizeof(h)) == 1);

  {
    struct {
      struct nntr_htp_oplist_header h;
      struct nntr_htp_op_desc ops[2];
    } wire;
    uint32_t buf_size[NNTR_HTP_BUF_COUNT];

    /* valid list -> 0 */
    build_valid(&wire.h, wire.ops, buf_size);
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);

    /* length mismatch -> 1 */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire) - 1, buf_size) == 1);

    /* version != 2 -> 3 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.h.version = 1u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 3);

    /* head_dim != 128 -> 4 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.h.head_dim = 64u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 4);

    /* unknown kind -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].kind = 99u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* unaligned offset -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].in0.offset = 4u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* ACT bounds overflow -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    buf_size[NNTR_HTP_BUF_ACT] = 10u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* v4: unknown weight_layout -> 4 (a v3-era list has 0 here) */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.h.weight_layout = 0u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 4);
    /* v5: w4cx_down8 is a known layout; w4cx (3) is reserved -> 4 */
    wire.h.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);
    wire.h.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_W4CX;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 4);

    /** v5 MATMUL_W4A8: on a w4cx_down8 header with a 128B-aligned colsum
     * of n int32 inside WEIGHTS -> 0; on a tiled32 header -> 5; k % 32,
     * n % 32, k > 16384, an unaligned or out-of-range colsum -> 5. The
     * tiles are n*k/2 bytes (256*128/2 = 16384 at WEIGHTS@256). */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_W4A8;
    wire.ops[1].param0 = 34048u; /* 33024 + 1024 scale bytes */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.h.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);
    wire.ops[1].k = 96u; /* % 32 ok, % 128 not: W4A8 needs only 32 */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);
    wire.ops[1].k = 80u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].k = 0u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].k = 128u;
    wire.ops[1].n = 240u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].n = 256u;
    wire.ops[1].param0 = 34052u; /* unaligned colsum */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].param0 = 65536u - 512u; /* 1024 B of colsum past WEIGHTS */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].param0 = 65536u - 1024u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);
    wire.ops[1].m = 1u; /* the per-call row rule holds for W4A8 too */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* tiled kinds need n % 32 == 0 -> 5; W8A16 (down, row-major) is exempt */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].n = 250u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_W8A16;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);

    /** EMBED needs k % 128 == 0 -> 5; k=4u (a multiple of 4, not 128) keeps
     * the vocab*k bounds check from tripping first. */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_EMBED;
    wire.ops[1].k = 4u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* W8A16 int32 accumulation is only exact for k <= 16384 -> 5 above it */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_W8A16;
    wire.ops[1].k = 16512u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /** k == 0 passes k % 128 but no kernel can run it -> 5 on all four
     * int8 kinds (the k-dependent bounds shrink to 0 bytes, so only the k
     * rule can reject). */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].k = 0u; /* W8A8 */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_W8A16;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_LOGITS;
    wire.ops[1].n = 32u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].kind = NNTR_HTP_OP_EMBED;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* RMSNORM / ADD / SILU_MUL consume whole 64-half vectors: n % 64 -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].n = 65u; /* RMSNORM; 4 x 65 halves still fit the ACT slot */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].kind = NNTR_HTP_OP_ADD;
    wire.ops[0].n = 65u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].kind = NNTR_HTP_OP_SILU_MUL;
    wire.ops[0].n = 65u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /** PER_HEAD RMSNORM chunks each row by head_dim: n = 192 is a multiple
     * of 64, so only the head_dim rule can reject it -> 5; n = 256 -> 0. */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].flags = NNTR_HTP_FLAG_PER_HEAD;
    wire.ops[0].n = 192u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[0].n = 256u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);

    /** ATTN indexes the KV cache by layer: layer >= n_layers -> 5. KV is
     * sized for exactly n_layers (1) at max_seq 64, so layer 0 -> 0. */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_ATTN;
    wire.ops[1].layer = 1u;
    wire.h.max_seq = 64u;
    buf_size[NNTR_HTP_BUF_KV] = 65536u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].layer = 0u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);
    /* ATTN in1/in2 are the fresh K/V rows: max_chunk * n_kv_heads * 256 B */
    wire.ops[1].in2.offset = 65536u - 128u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /** m is the per-call token count on every kind but LOGITS: a fixed m
     * would bypass the forward() gates -> 5 (RMSNORM); LOGITS m = 1 -> 0
     * is the control below. */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].m = 2u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /** LOGITS reads row n_tokens - 1 of in0, so in0 must hold max_chunk
     * rows whatever m says: at offset 7936 one 256 B row fits the 8192 B
     * ACT, four (max_chunk) do not -> 5; at offset 1024 -> 0. */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_LOGITS;
    wire.ops[1].m = 1u;
    wire.ops[1].k = 128u;
    wire.ops[1].n = 32u;
    wire.ops[1].in0.offset = 7936u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].in0.offset = 1024u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);
  }

  /** The op extent table the validator, hexagon_ref_run, ref_ops.c and
   * sim_model.c share, by literal numbers at rows = max_chunk = 4 on the
   * build_valid header (hidden 128, vocab 32, 2 heads, 2 kv heads, max_seq
   * 8, head_dim 128). used bit i: in0, in1, in2, out. */
  {
    struct nntr_htp_oplist_header h4;
    struct nntr_htp_op_desc ops[2], d;
    struct nntr_htp_op_extent e;
    uint32_t buf_size[NNTR_HTP_BUF_COUNT];
    build_valid(&h4, ops, buf_size);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_EMBED; /* ids, vocab*k int8, vocab scales, rows*k */
    d.k = 128u;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 16u && e.in1 == 4096u && e.in2 == 128u && e.out == 1024u);
    assert(e.used == 0xFu && e.out_alias_in0 == 0u);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_RMSNORM;
    d.n = 128u;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 256u && e.in2 == 0u && e.out == 1024u);
    assert(e.used == 0xBu && e.out_alias_in0 == 0u);
    d.flags = NNTR_HTP_FLAG_PER_HEAD; /* gamma stays head_dim long */
    d.n = 256u;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 2048u && e.in1 == 256u && e.in2 == 0u && e.out == 2048u);
    assert(e.used == 0xBu);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_MATMUL_W8A8;
    d.k = 128u;
    d.n = 256u;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 32768u && e.in2 == 1024u &&
           e.out == 2048u);
    assert(e.used == 0xFu && e.out_alias_in0 == 0u);
    d.kind = NNTR_HTP_OP_MATMUL_W8A16; /* same operand layout */
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 32768u && e.in2 == 1024u &&
           e.out == 2048u);
    assert(e.used == 0xFu && e.out_alias_in0 == 0u);
    d.kind = NNTR_HTP_OP_MATMUL_W4A8; /* nibble tiles: half the bytes */
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 16384u && e.in2 == 1024u &&
           e.out == 2048u);
    assert(e.used == 0xFu && e.out_alias_in0 == 0u);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_ROPE; /* q rows, k rows, max_seq-row table */
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 2048u && e.in1 == 2048u && e.in2 == 2048u);
    assert(e.out == e.in0 && e.out_alias_in0 == 1u && e.used == 0x7u);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_ATTN;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 2048u && e.in1 == 2048u && e.in2 == 2048u &&
           e.out == 2048u);
    assert(e.used == 0xFu && e.out_alias_in0 == 0u);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_SILU_MUL;
    d.n = 128u;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 1024u && e.in2 == 0u && e.out == 1024u);
    assert(e.used == 0xBu && e.out_alias_in0 == 0u);
    d.kind = NNTR_HTP_OP_ADD;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 1024u && e.in2 == 0u && e.out == 1024u);
    assert(e.used == 0xBu && e.out_alias_in0 == 0u);

    memset(&d, 0, sizeof(d));
    d.kind = NNTR_HTP_OP_MATMUL_LOGITS; /* m is ignored: rows decides in0 */
    d.m = 1u;
    d.k = 128u;
    d.n = 32u;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 0);
    assert(e.in0 == 1024u && e.in1 == 4096u && e.in2 == 128u && e.out == 128u);
    assert(e.used == 0xFu && e.out_alias_in0 == 0u);
    assert(nntr_htp_op_extent(&h4, &d, 1u, &e) == 0);
    assert(e.in0 == 256u && e.out == 128u);

    d.kind = NNTR_HTP_OP_KIND_COUNT;
    assert(nntr_htp_op_extent(&h4, &d, 4u, &e) == 1);

    /* row rule and the two size helpers */
    memset(&d, 0, sizeof(d));
    assert(nntr_htp_op_rows(&d, 3u) == 3u);
    d.m = 1u;
    assert(nntr_htp_op_rows(&d, 3u) == 1u);
    assert(nntr_htp_kv_bytes(1u, 2u, 8u, 128u) == 8192u);
    assert(nntr_htp_oplist_bytes(2u) == 192u);
  }

  /* RoPE row at p = 0: cos 1, sin 0 for every i, whatever libm does. */
  {
    float row[128];
    uint32_t i;
    memset(row, 0x7f, sizeof(row));
    nntr_htp_rope_row_f32(row, 0u, 1e6f);
    for (i = 0; i < 64u; ++i)
      assert(row[i] == 1.0f && row[64u + i] == 0.0f);
  }

  /* token-id gate: shared by htp_graph.c and HexagonBackend::forward */
  {
    static const int32_t ok[3] = {0, 17, 31};
    static const int32_t big[3] = {0, 32, 31};
    static const int32_t neg[3] = {0, 17, -1};
    assert(nntr_htp_token_ids_ok(ok, 3u, 32u) == 1);
    assert(nntr_htp_token_ids_ok(ok, 0u, 32u) == 1);
    assert(nntr_htp_token_ids_ok(big, 3u, 32u) == 0);
    assert(nntr_htp_token_ids_ok(neg, 3u, 32u) == 0);
  }

  puts("oplist header check: PASS");
  return 0;
}
