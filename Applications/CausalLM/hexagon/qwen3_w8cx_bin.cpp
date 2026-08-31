// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qwen3_w8cx_bin.cpp
 * @date	31 August 2026
 * @brief	Read-only mmap view over an nntr_quantize W8_CX qwen3 checkpoint
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "qwen3_w8cx_bin.h"

#include <cassert>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nntrainer::hexagon {

namespace {

struct Cursor {
  uint8_t *p;
  uint8_t *end;
  const int8_t *i8(uint64_t n) { return (const int8_t *)take(n); }
  const float *f32(uint64_t n) {
    // fp32 blocks are only aligned when every preceding int8 blob is a
    // multiple of 4 bytes; true for qwen3-0.6b, asserted rather than assumed.
    assert(((uintptr_t)p & 3u) == 0);
    return (const float *)take(n * 4ull);
  }
  void *take(uint64_t bytes) {
    if ((uint64_t)(end - p) < bytes)
      throw std::runtime_error("w8cx bin: truncated");
    void *r = p;
    p += bytes;
    return r;
  }
};

/** int8 [n][k] blob followed by n fp32 scales. */
void quantized(Cursor &c, const int8_t *&q, const float *&s, uint64_t n,
               uint64_t k) {
  q = c.i8(n * k);
  s = c.f32(n);
}

uint64_t q_bytes(uint64_t n, uint64_t k) { return n * k + n * 4ull; }

} // namespace

uint64_t Qwen3W8cxBin::expected_size(const HexModelConfig &c) {
  const uint64_t qdim = (uint64_t)c.n_heads * c.head_dim;
  const uint64_t kvdim = (uint64_t)c.n_kv_heads * c.head_dim;
  const uint64_t per_layer =
    4ull * c.hidden + q_bytes(qdim, c.hidden) + 4ull * c.head_dim +
    q_bytes(kvdim, c.hidden) + 4ull * c.head_dim + q_bytes(kvdim, c.hidden) +
    q_bytes(c.hidden, qdim) + 4ull * c.hidden + q_bytes(c.ffn, c.hidden) +
    q_bytes(c.ffn, c.hidden) + q_bytes(c.hidden, c.ffn);
  return q_bytes(c.vocab, c.hidden) + per_layer * c.n_layers + 4ull * c.hidden;
}

Qwen3W8cxBin::Qwen3W8cxBin(const std::string &path, const HexModelConfig &cfg) {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0)
    throw std::runtime_error("w8cx bin: cannot open " + path);
  struct stat st;
  if (fstat(fd_, &st) != 0) {
    close(fd_);
    throw std::runtime_error("w8cx bin: fstat failed " + path);
  }
  size_ = (uint64_t)st.st_size;
  const uint64_t want = expected_size(cfg);
  if (size_ != want) {
    close(fd_);
    throw std::runtime_error("w8cx bin: size " + std::to_string(size_) +
                             " != expected " + std::to_string(want));
  }
  void *m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (m == MAP_FAILED) {
    close(fd_);
    throw std::runtime_error("w8cx bin: mmap failed " + path);
  }
  base_ = (uint8_t *)m;

  const uint64_t qdim = (uint64_t)cfg.n_heads * cfg.head_dim;
  const uint64_t kvdim = (uint64_t)cfg.n_kv_heads * cfg.head_dim;
  Cursor c{base_, base_ + size_};

  quantized(c, w_.embed, w_.embed_s, cfg.vocab, cfg.hidden);
  w_.layers.resize(cfg.n_layers);
  for (auto &l : w_.layers) {
    l.attn_norm = c.f32(cfg.hidden);
    quantized(c, l.wq, l.wq_s, qdim, cfg.hidden);
    l.q_norm = c.f32(cfg.head_dim);
    quantized(c, l.wk, l.wk_s, kvdim, cfg.hidden);
    l.k_norm = c.f32(cfg.head_dim);
    quantized(c, l.wv, l.wv_s, kvdim, cfg.hidden);
    quantized(c, l.wo, l.wo_s, cfg.hidden, qdim);
    l.ffn_norm = c.f32(cfg.hidden);
    // nntrainer stores the mlp weights in up, gate order.
    quantized(c, l.w_up, l.w_up_s, cfg.ffn, cfg.hidden);
    quantized(c, l.w_gate, l.w_gate_s, cfg.ffn, cfg.hidden);
    quantized(c, l.w_down, l.w_down_s, cfg.hidden, cfg.ffn);
  }
  w_.final_norm = c.f32(cfg.hidden);
  if (c.p != c.end)
    throw std::runtime_error("w8cx bin: trailing bytes");
}

Qwen3W8cxBin::~Qwen3W8cxBin() {
  if (base_)
    munmap(base_, size_);
  if (fd_ >= 0)
    close(fd_);
}

} // namespace nntrainer::hexagon
