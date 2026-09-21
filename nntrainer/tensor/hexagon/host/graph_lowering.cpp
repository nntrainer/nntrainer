// SPDX-License-Identifier: Apache-2.0
/**
 * @file	graph_lowering.cpp
 * @date	19 August 2026
 * @brief	pack_weights(): copies/converts model-agnostic source weights
 *		into the WEIGHTS byte image at the offsets a lowering recipe
 *		(e.g. lower_qwen3()) already computed, plus the precomputed
 *		RoPE cos/sin table. No shape knowledge beyond HexModelConfig
 *		and HexWeightOffsets; walks whatever a lowering produced.
 *		int8 projections other than down_proj are written tiled32
 *		(nntr_htp_tile_off); int4 projections of a w4cx image are
 *		written as nibble tiles + int32 colsum (nntr_htp_w4_tile_off).
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "graph_lowering.h"
#include "nntr_htp_common.h"
#include "nntr_htp_rope.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace nntrainer::hexagon {

namespace {

/** @brief Convert n fp32 values to fp16 and write them at dst+off. */
void write_f16_vec(uint8_t *dst, uint32_t off, const float *src, uint64_t n) {
  uint16_t *out = reinterpret_cast<uint16_t *>(dst + off);
  for (uint64_t i = 0; i < n; ++i)
    out[i] = f32_to_f16_bits(src[i]);
}

/**
 * @brief Fill the RoPE table at dst+off: max_seq rows, each row
 *        [cos64||sin64] fp16 from nntr_htp_rope_row_f32 (the row the
 *        simulator reference also fills).
 */
void write_rope_table(uint8_t *dst, uint32_t off, uint32_t max_seq,
                      float theta) {
  uint16_t *out = reinterpret_cast<uint16_t *>(dst + off);
  float f[128];
  for (uint32_t p = 0; p < max_seq; ++p) {
    uint16_t *row = out + static_cast<uint64_t>(p) * 128u;
    nntr_htp_rope_row_f32(f, p, theta);
    for (uint32_t i = 0; i < 128u; ++i)
      row[i] = f32_to_f16_bits(f[i]);
  }
}

/**
 * @brief Write an int8 [n][k] projection in the tiled32 WEIGHTS layout
 *        (nntr_htp_tile_off): same byte count as the row-major source,
 *        only the order changes. down_proj is the one projection that
 *        stays row-major (MATMUL_W8A16 reads rows).
 */
void write_tiled(uint8_t *dst, uint32_t off, const int8_t *src, uint32_t n,
                 uint32_t k) {
  nntr_htp_repack_tiled32(dst + off, reinterpret_cast<const uint8_t *>(src), n,
                          k);
}

/**
 * @brief Write an int4 [n][k] projection (codes one per byte) as w4cx
 *        nibble tiles at off and its int32 colsum[n] at cs_off.
 */
void write_w4cx(uint8_t *dst, uint32_t off, uint32_t cs_off, const int8_t *src,
                uint32_t n, uint32_t k, const char *name) {
  if (nntr_htp_repack_w4cx(dst + off, reinterpret_cast<int32_t *>(dst + cs_off),
                           src, n, k))
    throw std::runtime_error(std::string("pack_weights: ") + name +
                             " holds codes outside [-8, 7], not an int4 "
                             "tensor (W8 checkpoint packed as w4cx?)");
}

/**
 * @brief One projection, tiled32 int8 or w4cx int4 by the class bit.
 */
void write_proj(const HexModelConfig &cfg, uint32_t bit, uint8_t *dst,
                uint32_t off, uint32_t cs_off, const int8_t *src, uint32_t n,
                uint32_t k, const char *name) {
  if (hex_is_w4(cfg, bit))
    write_w4cx(dst, off, cs_off, src, n, k, name);
  else
    write_tiled(dst, off, src, n, k);
}

} // namespace

void pack_weights(const HexLoweredGraph &g, const HexModelConfig &cfg,
                  const HexModelWeights &w, uint8_t *dst) {
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  /** The int width of every source tensor must be the one the layout
   * packs: a mismatch would pass the [-8, 7] check on most int8 tensors
   * only by luck, so it is refused up front (TILED32 needs all-int8). */
  const uint32_t want = cfg.weight_layout == NNTR_HTP_WEIGHT_LAYOUT_TILED32
                          ? kHexAllI8
                          : cfg.i8_mask;
  if (w.i8_mask != want)
    throw std::runtime_error(
      "pack_weights: checkpoint int8 set " + std::to_string(w.i8_mask) +
      " does not match the layout's " + std::to_string(want));

  write_tiled(dst, g.woff.embed, w.embed, cfg.vocab, cfg.hidden);
  std::memcpy(dst + g.woff.embed_scale, w.embed_s,
              static_cast<uint64_t>(cfg.vocab) * 4u);
  write_rope_table(dst, g.woff.rope_table, cfg.max_seq, cfg.rope_theta);
  write_f16_vec(dst, g.woff.final_norm, w.final_norm, cfg.hidden);

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    const HexLayerWeights &lw = w.layers[l];

    const uint32_t n_q32 = static_cast<uint32_t>(n_q);
    const uint32_t n_kv32 = static_cast<uint32_t>(n_kv);

    write_proj(cfg, kHexQ, dst, pl.wq, pl.wq_cs, lw.wq, n_q32, cfg.hidden,
               "wq");
    std::memcpy(dst + pl.wq_s, lw.wq_s, n_q * 4u);
    write_proj(cfg, kHexK, dst, pl.wk, pl.wk_cs, lw.wk, n_kv32, cfg.hidden,
               "wk");
    std::memcpy(dst + pl.wk_s, lw.wk_s, n_kv * 4u);
    write_proj(cfg, kHexV, dst, pl.wv, pl.wv_cs, lw.wv, n_kv32, cfg.hidden,
               "wv");
    std::memcpy(dst + pl.wv_s, lw.wv_s, n_kv * 4u);
    write_proj(cfg, kHexO, dst, pl.wo, pl.wo_cs, lw.wo, cfg.hidden, n_q32,
               "wo");
    std::memcpy(dst + pl.wo_s, lw.wo_s, static_cast<uint64_t>(cfg.hidden) * 4u);
    write_proj(cfg, kHexGate, dst, pl.gate, pl.gate_cs, lw.w_gate, cfg.ffn,
               cfg.hidden, "gate");
    std::memcpy(dst + pl.gate_s, lw.w_gate_s,
                static_cast<uint64_t>(cfg.ffn) * 4u);
    write_proj(cfg, kHexUp, dst, pl.up, pl.up_cs, lw.w_up, cfg.ffn, cfg.hidden,
               "up");
    std::memcpy(dst + pl.up_s, lw.w_up_s, static_cast<uint64_t>(cfg.ffn) * 4u);
    std::memcpy(dst + pl.down, lw.w_down,
                static_cast<uint64_t>(cfg.hidden) * cfg.ffn); /* row-major */
    std::memcpy(dst + pl.down_s, lw.w_down_s,
                static_cast<uint64_t>(cfg.hidden) * 4u);

    write_f16_vec(dst, pl.attn_norm, lw.attn_norm, cfg.hidden);
    write_f16_vec(dst, pl.ffn_norm, lw.ffn_norm, cfg.hidden);
    write_f16_vec(dst, pl.q_norm, lw.q_norm, cfg.head_dim);
    write_f16_vec(dst, pl.k_norm, lw.k_norm, cfg.head_dim);
  }
}

} // namespace nntrainer::hexagon
