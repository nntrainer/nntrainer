// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qwen3_w8cx_bin.h
 * @date	31 August 2026
 * @brief	Read-only mmap view over an nntr_quantize W8_CX qwen3 checkpoint
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __CAUSALLM_HEXAGON_QWEN3_W8CX_BIN_H__
#define __CAUSALLM_HEXAGON_QWEN3_W8CX_BIN_H__

#include <cstdint>
#include <string>

#include "graph_lowering.h"

namespace nntrainer::hexagon {

/** qwen3-0.6b checkpoint shape, the only W8_CX .bin this reader knows. */
inline const HexModelConfig kQwen3_0_6b = {
  /*n_layers=*/28,   /*n_heads=*/16,    /*n_kv_heads=*/8,
  /*head_dim=*/128,
  /*hidden=*/1024,   /*ffn=*/3072,      /*vocab=*/151936,
  /*max_seq=*/2048,
  /*max_chunk=*/128, /*rms_eps=*/1e-6f, /*rope_theta=*/1e6f};

/**
 * @class Qwen3W8cxBin
 * @brief Read-only view over an nntr_quantize W8_CX .bin. mmaps the file and
 *        hands out non-owning pointers into it; the object must outlive the
 *        HexModelWeights it returns.
 *
 * The file is a header-less stream in graph layer order: embedding
 * (int8 [vocab][hidden] + fp32 [vocab]), then per layer attn_norm, wq,
 * q_norm, wk, k_norm, wv, wo, ffn_norm, ffn_up, ffn_gate, ffn_down (2D
 * tensors as int8 [N][K] + fp32 [N], norms as fp32), then output_norm.
 */
class Qwen3W8cxBin {
public:
  /** @throw std::runtime_error on open/size/structure mismatch. */
  Qwen3W8cxBin(const std::string &path, const HexModelConfig &cfg);
  ~Qwen3W8cxBin();
  Qwen3W8cxBin(const Qwen3W8cxBin &) = delete;
  Qwen3W8cxBin &operator=(const Qwen3W8cxBin &) = delete;

  const HexModelWeights &weights() const { return w_; }

  /** Byte size the checkpoint must have for this shape. */
  static uint64_t expected_size(const HexModelConfig &cfg);

private:
  int fd_ = -1;
  uint8_t *base_ = nullptr;
  uint64_t size_ = 0;
  HexModelWeights w_;
};

} // namespace nntrainer::hexagon
#endif // __CAUSALLM_HEXAGON_QWEN3_W8CX_BIN_H__
