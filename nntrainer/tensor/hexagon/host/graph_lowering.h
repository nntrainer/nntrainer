// SPDX-License-Identifier: Apache-2.0
/**
 * @file	graph_lowering.h
 * @date	19 August 2026
 * @brief	Shared wire-layout types (WEIGHTS/ACT offsets, lowered-graph
 *		result) and pack_weights() consumed by every model-specific
 *		lowering recipe (e.g. Applications/CausalLM/hexagon/
 *		qwen3_lowering.h) and by HexagonRunner::init() (M4). No
 *		SDK/rpcmem dependency.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __HEXAGON_GRAPH_LOWERING_H__
#define __HEXAGON_GRAPH_LOWERING_H__

#include <cstdint>
#include <cstring>
#include <vector>

#include "nntr_htp_common.h"

namespace nntrainer::hexagon {

/**
 * @brief One bit per weight tensor class, in the order the names are
 *        spelled in a .hexcfg `i8_tensors=` line and in
 *        make_w8cx_bin.py --i8-tensors: embed,q,k,v,o,gate,up,down. A set
 *        bit means "int8 (tiled32 / row-major), as in a W8 image"; a
 *        clear bit means "int4 w4cx tiles" (issue #65 S1).
 */
enum HexTensorBit : uint32_t {
  kHexEmbed = 1u << 0,
  kHexQ = 1u << 1,
  kHexK = 1u << 2,
  kHexV = 1u << 3,
  kHexO = 1u << 4,
  kHexGate = 1u << 5,
  kHexUp = 1u << 6,
  kHexDown = 1u << 7,
  kHexAllI8 = 0xFFu,
  /** The w4cx_down8 split of #65: every per-layer projection int4, the
   * tied embed / LOGITS table and down int8. */
  kHexW4cxDown8I8 = kHexEmbed | kHexDown,
};

/**
 * @brief Round up to the next 128B boundary, all math in uint64_t. Shared
 *        by every lowering recipe's WEIGHTS/ACT cursor and by
 *        pack_weights(), which is why it lives here instead of with a
 *        single model's lowering .cpp.
 */
/**
 * @brief Convert an fp32 value to its IEEE fp16 bit pattern (round to
 *        nearest even) with plain integer arithmetic, so the host side
 *        builds on any compiler regardless of _Float16 support.
 */
inline uint16_t f32_to_f16_bits(float v) {
  uint32_t x;
  std::memcpy(&x, &v, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t exp = (x >> 23) & 0xffu;
  uint32_t mant = x & 0x7fffffu;
  if (exp == 0xff) /* inf / nan */
    return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
  int32_t e = static_cast<int32_t>(exp) - 127 + 15;
  if (e >= 0x1f) /* overflow -> inf */
    return static_cast<uint16_t>(sign | 0x7c00u);
  if (e <= 0) { /* subnormal or zero */
    if (e < -10)
      return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - e);
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u)))
      ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(e) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
    ++half; /* carries into the exponent correctly */
  return static_cast<uint16_t>(sign | half);
}

inline uint64_t align128(uint64_t x) {
  return (x + 127ull) & ~static_cast<uint64_t>(127ull);
}

/**
 * @brief qwen3 shape/hparam config. Mirrors nntr_htp_oplist_header plus
 *        the rope/eps values needed only at lowering time.
 */
struct HexModelConfig {
  uint32_t n_layers, n_heads, n_kv_heads, head_dim;
  uint32_t hidden, ffn, vocab, max_seq, max_chunk;
  float rms_eps, rope_theta;
  /** WEIGHTS image family (NNTR_HTP_WEIGHT_LAYOUT_*, the header field) and,
   * for W4CX_DOWN8, which tensor classes stay int8 (HexTensorBit; embed and
   * down must be set in S1). TILED32 ignores the mask: everything is int8.
   * Both are read from the .bin by Qwen3W8cxBin and written to the .hexcfg
   * so every consumer re-lowers the same op-list. */
  uint32_t weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;
  uint32_t i8_mask = kHexAllI8;
};

/** @brief True when tensor class `bit` of cfg is packed as int4 w4cx. */
inline bool hex_is_w4(const HexModelConfig &cfg, uint32_t bit) {
  return cfg.weight_layout != NNTR_HTP_WEIGHT_LAYOUT_TILED32 &&
         (cfg.i8_mask & bit) == 0u;
}

/**
 * @brief One transformer layer's source weights. All pointers are
 *        non-owning; int8 matrices are N-major [N][K] (row = out chan).
 *        A tensor whose HexTensorBit is clear in HexModelWeights::i8_mask
 *        holds int4 codes ([-7, 7]) one per byte at the same pointer.
 */
struct HexLayerWeights {
  const int8_t *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;
  const float *wq_s, *wk_s, *wv_s, *wo_s, *w_gate_s, *w_up_s, *w_down_s;
  const float *attn_norm, *ffn_norm, *q_norm, *k_norm; // fp32 gamma
};

/**
 * @brief Full qwen3 source weights (fp32 gammas/scales, int8 matrices).
 */
struct HexModelWeights {
  const int8_t *embed;  // [vocab][hidden], tied -> also used as lm_head
  const float *embed_s; // [vocab]
  const float *final_norm;
  std::vector<HexLayerWeights> layers; // size == n_layers
  uint32_t i8_mask = kHexAllI8;        // which classes are int8 (else int4)
};

/**
 * @brief Byte offsets of every tensor inside the packed WEIGHTS buffer.
 *        All offsets are 128B aligned.
 */
struct HexWeightOffsets {
  uint32_t embed, embed_scale, rope_table, final_norm;
  struct PerLayer {
    uint32_t wq, wq_s, wk, wk_s, wv, wv_s, wo, wo_s;
    uint32_t gate, gate_s, up, up_s, down, down_s;
    uint32_t attn_norm, ffn_norm, q_norm, k_norm; // stored fp16
    /** int32 colsum[N] of a w4cx projection (MATMUL_W4A8 param0); 0 and
     * unallocated for a tensor that is int8. */
    uint32_t wq_cs, wk_cs, wv_cs, wo_cs, gate_cs, up_cs;
  };
  std::vector<PerLayer> layers;
};

/**
 * @brief The result of lowering: the wire-format op-list bytes, the
 *        WEIGHTS layout plan, and the three buffer sizes it implies.
 */
struct HexLoweredGraph {
  std::vector<uint8_t> oplist; // header(64B) + n_ops*64B
  HexWeightOffsets woff;
  uint64_t weights_size, kv_size, act_size;
};

/**
 * @brief Pack source weights into dst according to a lowered graph's
 *        WEIGHTS layout. int8 projections except down_proj are stored
 *        tiled32 (see nntr_htp_tile_off in nntr_htp_common.h); int4
 *        projections (hex_is_w4) become w4cx nibble tiles plus their
 *        int32 colsum (nntr_htp_repack_w4cx); scales, norms and the RoPE
 *        table are unchanged.
 * @param g lowered graph carrying the WEIGHTS offsets/sizes.
 * @param cfg the same config passed to lower_qwen3().
 * @param w source weights to pack; w.i8_mask must match cfg on a w4cx
 *          layout.
 * @param dst destination buffer, at least g.weights_size bytes.
 * @throw std::runtime_error when the source is not the int width the
 *        layout expects (a W8 .bin packed as w4cx, or the reverse).
 */
void pack_weights(const HexLoweredGraph &g, const HexModelConfig &cfg,
                  const HexModelWeights &w, uint8_t *dst);

} // namespace nntrainer::hexagon
#endif // __HEXAGON_GRAPH_LOWERING_H__
