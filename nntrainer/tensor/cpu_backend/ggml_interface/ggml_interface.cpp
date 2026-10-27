// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   ggml_interface.cpp
 * @date   13 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface to use ggml lib from cpu_backend
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ggml_interface.h>
#include <mutex>
#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>
#include <thread_manager.h>
#include <unordered_map>
#ifdef Q4_0
#undef Q4_0
#endif
#ifdef Q8_0
#undef Q8_0
#endif
#include <q8_0_tensor.h>
#include <string>
#include <thread>
#include <vector>

namespace nntrainer {

void __ggml_init() { nntr_ggml_init(); }

size_t __ggml_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q4_0(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q4_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q6_K(src, dst, nrow, n_per_row, quant_weights);
}

size_t __ggml_quantize_q8_0(const float *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  return nntr_quantize_q8_0(src, dst, nrow, n_per_row, quant_weights);
}

void __ggml_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  __ggml_quantize_q6_K(src, dst, 1, k, nullptr);
}

template <>
void __ggml_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  nntr_quantize_row_q8_K(src, dst, k);
}

void __ggml_dequantize_row_q4_0(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q4_0(x_raw, y, k);
}

void __ggml_dequantize_row_q8_0(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q8_0(x_raw, y, k);
}

void __ggml_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  nntr_dequantize_row_q4_K(x_raw, y, k);
}

void __ggml_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  nntr_dequantize_row_q6_K(x, y, k);
}

template <>
void __ggml_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  nntr_dequantize_row_q8_K(x, y, k);
}

float __ggml_vec_dot_q6_K_q8_K(const unsigned int K,
                               const void *__restrict v_q6_K,
                               const void *__restrict v_q8_K) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K
  nntr_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_K, by, nrc);
  return result;
}

float __ggml_vec_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  // Quantization of activations
  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(f, v_q8_activation.data(), K);

  return __ggml_vec_dot_q6_K_q8_K(K, v_q6_K, v_q8_activation.data());
}

float __ggml_vec_dot_q6_K(const unsigned int K, const void *__restrict v_q6_K,
                          const float *__restrict activation) {
  float result;
  int bs = 1, bx = 1, by = 1,
      nrc = 1; // unused variables in ggml_vec_dot_q6_K_q8_K

  int blocks_per_row = (K + QK_K - 1) / QK_K;
  int q8_K_activation_size = sizeof(block_q8_K) * blocks_per_row;
  std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
  __ggml_quantize_row_q8_K(activation, v_q8_activation.data(), K);

  nntr_vec_dot_q6_K_q8_K(K, &result, bs, v_q6_K, bx, v_q8_activation.data(), by,
                         nrc);
  return result;
}

void __ggml_repack_q4_0_to_q4_0_4(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_0_to_q4_0_4_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q4_0_to_q4_0_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_0_to_q4_0_8_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q8_0_to_q8_0_4x4(void *dst, void *src, size_t data_size,
                                    const unsigned int M,
                                    const unsigned int N) {
  nntr_repack_q8_0_to_q8_0_4_bl(dst, 4, src, data_size, M, N);
}

void __ggml_repack_q8_0_to_q8_0_4x8(void *dst, void *src, size_t data_size,
                                    const unsigned int M,
                                    const unsigned int N) {
  nntr_repack_q8_0_to_q8_0_4_bl(dst, 8, src, data_size, M, N);
}

void __ggml_repack_q4_K_to_q4_K_8(void *dst, void *src, size_t data_size,
                                  const unsigned int M, const unsigned int N) {
  nntr_repack_q4_K_to_q4_K_8_bl(dst, 8, src, data_size, M, N);
}

/**
 * @brief De-interleave + gather + plain-SMMLA W8A32 fallback
 * (NNTR_Q8A32_PLAIN).
 *
 * The original W8A32 pipeline: de-interleaves the pre-repacked q8_0x4 weight
 * back to plain block_q8_0 (cached per weight pointer), quantizes each gathered
 * FP32 activation row to plain block_q8_0, and feeds nntr_gemm_q8_0_q8_0_f32.
 * Kept as a numerical reference / fallback under NNTR_Q8A32_PLAIN. The default
 * path below is ~2x faster (interleaved 4-row quantize + 4x4 SMMLA, no
 * de-interleave), so this stays behind the env only.
 */
static void __q8a32_indirect_plain(const unsigned int M, const unsigned int N,
                                   const unsigned int K, const float *in,
                                   const ConvGatherParams &geom, const void *B,
                                   float *C) {
  auto &tm = ThreadManager::Global();
  const unsigned int nb = K / QK8_0;

  // 1) De-interleave weight q8_0x4 -> plain block_q8_0 [N][nb], cached per
  //    weight-buffer pointer so it runs once instead of every forward.
  const block_q8_0 *Wp_ptr;
  {
    static std::mutex wcache_mtx;
    static std::unordered_map<const void *, std::vector<block_q8_0>> wcache;
    std::lock_guard<std::mutex> lk(wcache_mtx);
    auto it = wcache.find(B);
    if (it == wcache.end()) {
      std::vector<block_q8_0> Wp((size_t)N * nb);
      const block_q8_0x4 *bx4 = (const block_q8_0x4 *)B;
      const unsigned int NB4 = N / 4;
      for (unsigned int sc = 0; sc < NB4; ++sc)
        for (unsigned int j = 0; j < nb; ++j) {
          const block_q8_0x4 &sb = bx4[(size_t)sc * nb + j];
          for (unsigned int r = 0; r < 4; ++r) {
            block_q8_0 &p = Wp[(size_t)(sc * 4 + r) * nb + j];
            p.d = sb.d[r];
            for (unsigned int sub = 0; sub < 4; ++sub)
              std::memcpy(&p.qs[8 * sub], &sb.qs[32 * sub + 8 * r], 8);
          }
        }
      it = wcache.emplace(B, std::move(Wp)).first;
    }
    Wp_ptr = it->second.data();
  }

  // 2) Gather FP32 activation rows -> plain block_q8_0 [M][nb].
  std::vector<block_q8_0> Ap((size_t)M * nb);
  {
    block_q8_0 *Ap_ptr = Ap.data();
    const unsigned int QCHUNK = 64;
    const size_t qloops = (M + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      std::vector<float> tile((size_t)K);
      const unsigned int r0 = (unsigned int)q * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, M);
      for (unsigned int r = r0; r < r1; ++r) {
        gather_conv_act_rows_fp32(tile.data(), in, geom, (int)r, 1);
        nntr_quantize_row_q8_0(tile.data(), &Ap_ptr[(size_t)r * nb],
                               (int64_t)K);
      }
    });
  }

  // 3) Plain Q8_0 x Q8_0 GEMM -> FP32 C, tiled 16x16.
  const unsigned int row_chunk = 16, col_chunk = 16;
  const size_t row_loop = (M + row_chunk - 1) / row_chunk;
  const size_t col_loop = (N + col_chunk - 1) / col_chunk;
  const block_q8_0 *Ap_ptr = Ap.data();
  tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
    unsigned int r = (unsigned int)(i / col_loop);
    unsigned int c = (unsigned int)(i % col_loop);
    unsigned int r0 = r * row_chunk, r1 = std::min(r0 + row_chunk, M);
    unsigned int c0 = c * col_chunk, c1 = std::min(c0 + col_chunk, N);
    nntr_gemm_q8_0_q8_0_f32((int)K, C + (size_t)r0 * N + c0, N,
                            (const void *)&Wp_ptr[(size_t)c0 * nb],
                            (const void *)&Ap_ptr[(size_t)r0 * nb],
                            (int)(r1 - r0), (int)(c1 - c0));
  });
}

/**
 * @brief FP32-activation Q8_0-weight indirect conv GEMM (W8A32).
 *
 * The W8A16 (FP16-activation) path stores activations as FP16 between layers,
 * which accumulates rounding error across the deep backbone and collapses pose
 * confidence. This FP32 variant keeps activations in FP32 end-to-end (matching
 * an ONNX-Runtime int8 model's accuracy) while still doing int8 SMMLA compute.
 *
 * The default path mirrors the fast W8A16 pipeline exactly, only in FP32:
 * the FP32 input is gathered 4 rows at a time (no im2col materialization) and
 * quantized straight into the interleaved block_q8_0x4 layout via
 * nntr_quantize_mat_q8_0_4x8, so the pre-repacked q8_0x4 weight can be consumed
 * *directly* (no per-forward de-interleave) by the register-blocked 4x4 SMMLA
 * kernel nntr_gemm_q8_0_q8_0_4x4_f32, writing FP32 output. This removes the two
 * costs that made the earlier plain path 10-28x slower than W8A16: the 1-row
 * scalar quantize and the weight de-interleave. It reuses the identical weight
 * file as the FP16 path (no re-quantize) and is portable across ISAs (all
 * primitives have neon/avx/sve/fallback impls). Set NNTR_Q8A32_PLAIN to fall
 * back to the reference de-interleave pipeline.
 *
 * CRS-padding: the caller may pass a K that is LARGER than the geometry's real
 * CRS = in_ch*k_h*k_w. The gather still produces only the real CRS values per
 * row and zero-fills the tail up to K (dst_stride = K), and the weight B is
 * expected to be quantized with the same K-padding; the zero tail contributes 0
 * to every dot so the result is the un-padded convolution. This lets a conv
 * whose real CRS is not a multiple of 32 run on the int8 GEMM by rounding K up
 * to a block multiple. Likewise N may be smaller than the weight's padded
 * column count -- only the first N columns are ever tiled/written, so padded
 * out-channels are never computed (free out_ch strip).
 */
void __ggml_q8_0_q8_0_indirect_GEMM_fp32(const unsigned int M,
                                         const unsigned int N,
                                         const unsigned int K, const float *in,
                                         const ConvGatherParams &geom,
                                         const void *B, const unsigned int ldb,
                                         float *C, const unsigned int ldc) {
  (void)ldb;
  (void)ldc;

  static const bool use_plain = (std::getenv("NNTR_Q8A32_PLAIN") != nullptr);
  if (use_plain) {
    __q8a32_indirect_plain(M, N, K, in, geom, B, C);
    return;
  }

  auto &tm = ThreadManager::Global();
  const unsigned int nb = K / QK8_0; // blocks per row (K multiple of 32)
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * nb;
  const unsigned int Mfull = (M / 4) * 4; // 4-row-divisible part
  const unsigned int rem = M % 4;
  const unsigned int M4 = M / 4;
  const unsigned int M4c = (M + 3) / 4; // groups incl. padded tail

  // 1) Fused gather + Q8_0 quantize to interleaved q8_0x4, 4 rows at a time.
  //    Parallelized over 64-row chunks (16 tiles) to amortize dispatch, with a
  //    reusable per-thread tile buffer.
  std::vector<char> QA((size_t)M4c * qa_4_rows_size);
  char *QA_ptr = QA.data();

  const unsigned int QCHUNK = 64; // multiple of 4
  if (Mfull > 0) {
    const size_t qloops = (Mfull + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, Mfull);
      std::vector<float> tilebuf((size_t)4 * K);
      float *tile = tilebuf.data();
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows_fp32(tile, in, geom, (int)r, 4, (int)K);
        nntr_quantize_mat_q8_0_4x8(
          tile, QA_ptr + (size_t)(r / 4) * qa_4_rows_size, K);
      }
    });
  }

  // Handle M-tail (rem 1..3) gather and quantization (zero-padded to 4 rows).
  if (rem > 0) {
    std::vector<float> tilebuf((size_t)4 * K, 0.0f);
    gather_conv_act_rows_fp32(tilebuf.data(), in, geom, (int)Mfull, (int)rem,
                              (int)K);
    nntr_quantize_mat_q8_0_4x8(tilebuf.data(),
                               QA_ptr + (size_t)M4 * qa_4_rows_size, K);
  }

  // 2) Tiled 4x4 SMMLA GEMM over the 4-row-divisible part, direct to C.
  //    Weight B is consumed in its native q8_0x4 layout (stride = nb plain
  //    blocks per 4-column super-block).
  const size_t B_step = (size_t)nb * sizeof(block_q8_0);
  const unsigned int row_chunk_size = 16; // multiple of 4
  const unsigned int col_chunk_size = 16; // multiple of 4

  if (Mfull > 0) {
    const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
      unsigned int r = static_cast<unsigned int>(i / col_loop);
      unsigned int c = static_cast<unsigned int>(i % col_loop);
      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

      nntr_gemm_q8_0_q8_0_4x4_f32(
        (int)K, C + (size_t)r_start * N + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)(QA_ptr + (size_t)(r_start / 4) * qa_4_rows_size),
        (int)(r_end - r_start), (int)(c_end - c_start));
    });
  }

  // 3) M-tail (rem 1..3): run the padded last group into a 4xN FP32 scratch,
  //    then copy only the valid rem rows into C.
  if (rem > 0) {
    std::vector<float> scratch((size_t)4 * N);
    float *scratch_ptr = scratch.data();
    const char *tail_a = QA_ptr + (size_t)M4 * qa_4_rows_size;
    const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, col_loop, [=](size_t c) {
      unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * ((unsigned int)c + 1), N);
      nntr_gemm_q8_0_q8_0_4x4_f32(
        (int)K, scratch_ptr + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)tail_a, 4, (int)(c_end - c_start));
    });
    for (unsigned int rr = 0; rr < rem; ++rr)
      std::memcpy(C + (size_t)(Mfull + rr) * N, scratch_ptr + (size_t)rr * N,
                  (size_t)N * sizeof(float));
  }
}

/**
 * @brief Pack 4 gathered int8 rows [4][K] into the interleaved block_q8_0x4
 * stream with one constant scale d for every block.
 *
 * A per-tensor-scale int8 activation is numerically identical to a block_q8_0
 * stream whose blocks all carry d = scale, so the pre-quantized W8A8 activation
 * feeds the existing q8_0x4 SMMLA kernel with a pure byte shuffle -- no amax
 * scan, no rounding math (that happened once at the producing layer).
 * Layout matches nntr_quantize_mat_q8_0_4x8 / repack_q8_0:
 * qs[32*sub + 8*row + lane], sub = 8-lane chunk 0..3 within the 32-block.
 */
static inline void pack_i8_rows_q8_0x4(const int8_t *rows, unsigned int K,
                                       uint16_t d_fp16, char *dst) {
  const unsigned int nb = K / QK8_0;
  block_q8_0x4 *out = (block_q8_0x4 *)dst;
#if defined(__ARM_NEON)
  // The interleave qs[32*sub + 8*r + lane] is, per 32-block, four 32-byte
  // groups [r0|r1|r2|r3] of the rows' 8-byte sub-chunks. Loading each row's
  // 32 bytes once (two q-registers) and storing combined 16-byte halves does
  // the whole block in 8 loads + 8 stores -- the scalar 8-byte memcpy version
  // ran well below memory bandwidth and was the largest cost of the 1x1-conv
  // fast path (head.final gemm.pack).
  for (unsigned int j = 0; j < nb; ++j) {
    block_q8_0x4 &sb = out[j];
    sb.d[0] = sb.d[1] = sb.d[2] = sb.d[3] = d_fp16;
    const uint8_t *p0 =
      (const uint8_t *)(rows + (size_t)0 * K + (size_t)j * QK8_0);
    const uint8_t *p1 =
      (const uint8_t *)(rows + (size_t)1 * K + (size_t)j * QK8_0);
    const uint8_t *p2 =
      (const uint8_t *)(rows + (size_t)2 * K + (size_t)j * QK8_0);
    const uint8_t *p3 =
      (const uint8_t *)(rows + (size_t)3 * K + (size_t)j * QK8_0);
    const uint8x16_t r0lo = vld1q_u8(p0), r0hi = vld1q_u8(p0 + 16);
    const uint8x16_t r1lo = vld1q_u8(p1), r1hi = vld1q_u8(p1 + 16);
    const uint8x16_t r2lo = vld1q_u8(p2), r2hi = vld1q_u8(p2 + 16);
    const uint8x16_t r3lo = vld1q_u8(p3), r3hi = vld1q_u8(p3 + 16);
    uint8_t *q = (uint8_t *)sb.qs;
    vst1q_u8(q + 0, vcombine_u8(vget_low_u8(r0lo), vget_low_u8(r1lo)));
    vst1q_u8(q + 16, vcombine_u8(vget_low_u8(r2lo), vget_low_u8(r3lo)));
    vst1q_u8(q + 32, vcombine_u8(vget_high_u8(r0lo), vget_high_u8(r1lo)));
    vst1q_u8(q + 48, vcombine_u8(vget_high_u8(r2lo), vget_high_u8(r3lo)));
    vst1q_u8(q + 64, vcombine_u8(vget_low_u8(r0hi), vget_low_u8(r1hi)));
    vst1q_u8(q + 80, vcombine_u8(vget_low_u8(r2hi), vget_low_u8(r3hi)));
    vst1q_u8(q + 96, vcombine_u8(vget_high_u8(r0hi), vget_high_u8(r1hi)));
    vst1q_u8(q + 112, vcombine_u8(vget_high_u8(r2hi), vget_high_u8(r3hi)));
  }
#else
  for (unsigned int j = 0; j < nb; ++j) {
    block_q8_0x4 &sb = out[j];
    for (int r = 0; r < 4; ++r) {
      sb.d[r] = d_fp16;
      const int8_t *src = rows + (size_t)r * K + (size_t)j * QK8_0;
      for (int sub = 0; sub < 4; ++sub)
        std::memcpy(&sb.qs[32 * sub + 8 * r], src + 8 * sub, 8);
    }
  }
#endif
}

/**
 * @brief True when a conv's im2col gather is the identity map.
 *
 * A 1x1, unit-stride, zero-pad, unit-dilation NHWC conv gathers output pixel m
 * as exactly the contiguous in_ch vector at input pixel m (out_w == in_w and
 * (oh,ow)->(oh,ow)), so the K=in_ch activation row is already laid out at
 * `in + m*K` with stride K -- the exact stride pack_i8_rows_q8_0x4 expects.
 * The whole gather pass (per-row memset + scatter into a tile) can then be
 * skipped and the pack reads straight from the input. Requires K == in_ch (no
 * CRS zero-padding), otherwise the real row stride would be in_ch, not K.
 */
static inline bool conv_gather_is_identity(const ConvGatherParams &g,
                                           unsigned int K) {
  return g.is_nhwc && g.k_h == 1 && g.k_w == 1 && g.stride_h == 1 &&
         g.stride_w == 1 && g.pad_t == 0 && g.pad_l == 0 && g.dil_h == 1 &&
         g.dil_w == 1 && g.out_w == g.in_w && (int)K == g.in_ch;
}

/**
 * @brief W8A8 indirect conv GEMM: pre-quantized int8 activation (per-tensor
 * scale) x q8_0x4 weight -> FP32 output.
 *
 * Identical structure to __ggml_q8_0_q8_0_indirect_GEMM_fp32, but the input is
 * already int8 with one FP32 scale (the producing layer's fused quantize
 * epilogue wrote it), so the per-conv activation quantization disappears: the
 * gather collects int8 bytes and pack_i8_rows_q8_0x4 shuffles them into the
 * kernel layout with constant d. Same CRS-padding contract: K may exceed the
 * geometry's real CRS (zero tail via the gather's dst_stride), and only the
 * first N weight columns are tiled (free out_ch strip).
 *
 * @param act_scale the producer's per-tensor scale. It is converted to fp16
 * once here (block_q8_0.d is fp16); the producer must quantize with the SAME
 * fp16-rounded value so (q, d) reproduces its FP32 values exactly.
 */
void __ggml_q8_0_q8_0_indirect_GEMM_i8a(
  const unsigned int M, const unsigned int N, const unsigned int K,
  const int8_t *in, const float act_scale, const ConvGatherParams &geom,
  const void *B, const unsigned int ldb, float *C, const unsigned int ldc) {
  (void)ldb;
  (void)ldc;
  auto &tm = ThreadManager::Global();
  const unsigned int nb = K / QK8_0;
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * nb;
  const unsigned int Mfull = (M / 4) * 4;
  const unsigned int rem = M % 4;
  const unsigned int M4 = M / 4;
  const unsigned int M4c = (M + 3) / 4;
  const uint16_t d16 = nntr_compute_fp32_to_fp16(act_scale);

  // 1) Gather int8 rows 4 at a time and byte-pack to q8_0x4 (constant d).
  std::vector<char> QA((size_t)M4c * qa_4_rows_size);
  char *QA_ptr = QA.data();

  // 1x1 unit-stride convs gather the identity: pack straight from `in`, no
  // tile.
  const bool ident = conv_gather_is_identity(geom, K);

  const unsigned int QCHUNK = 64; // multiple of 4
  if (Mfull > 0) {
    const size_t qloops = (Mfull + QCHUNK - 1) / QCHUNK;
    tm.parallel_for(0, qloops, [=](size_t q) {
      const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
      const unsigned int r1 = std::min(r0 + QCHUNK, Mfull);
      if (ident) {
        for (unsigned int r = r0; r < r1; r += 4)
          pack_i8_rows_q8_0x4(in + (size_t)r * K, K, d16,
                              QA_ptr + (size_t)(r / 4) * qa_4_rows_size);
        return;
      }
      std::vector<int8_t> tile((size_t)4 * K);
      for (unsigned int r = r0; r < r1; r += 4) {
        gather_conv_act_rows<int8_t>(tile.data(), in, geom, (int)r, 4, (int)K);
        pack_i8_rows_q8_0x4(tile.data(), K, d16,
                            QA_ptr + (size_t)(r / 4) * qa_4_rows_size);
      }
    });
  }
  if (rem > 0) {
    std::vector<int8_t> tile((size_t)4 * K, 0);
    gather_conv_act_rows<int8_t>(tile.data(), in, geom, (int)Mfull, (int)rem,
                                 (int)K);
    pack_i8_rows_q8_0x4(tile.data(), K, d16,
                        QA_ptr + (size_t)M4 * qa_4_rows_size);
  }

  // 2) Same tiled 4x4 SMMLA GEMM as the FP32-activation path.
  const size_t B_step = (size_t)nb * sizeof(block_q8_0);
  const unsigned int row_chunk_size = 16, col_chunk_size = 16;

  if (Mfull > 0) {
    const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
      unsigned int r = static_cast<unsigned int>(i / col_loop);
      unsigned int c = static_cast<unsigned int>(i % col_loop);
      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);
      nntr_gemm_q8_0_q8_0_4x4_f32(
        (int)K, C + (size_t)r_start * N + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)(QA_ptr + (size_t)(r_start / 4) * qa_4_rows_size),
        (int)(r_end - r_start), (int)(c_end - c_start));
    });
  }
  if (rem > 0) {
    std::vector<float> scratch((size_t)4 * N);
    float *scratch_ptr = scratch.data();
    const char *tail_a = QA_ptr + (size_t)M4 * qa_4_rows_size;
    const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, col_loop, [=](size_t c) {
      unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * ((unsigned int)c + 1), N);
      nntr_gemm_q8_0_q8_0_4x4_f32(
        (int)K, scratch_ptr + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        (const void *)tail_a, 4, (int)(c_end - c_start));
    });
    for (unsigned int rr = 0; rr < rem; ++rr)
      std::memcpy(C + (size_t)(Mfull + rr) * N, scratch_ptr + (size_t)rr * N,
                  (size_t)N * sizeof(float));
  }
}

// ---- Per-channel W8A8 conv weight preparation ------------------------------
// This module owns the block_q8_0x4 layout the kernels consume, so the
// per-channel weight conversion (dequant -> per-output-channel int8 requant ->
// q8_0x4 pack, with the taps-last K permutation and the zero-copy in-place
// mode) lives here; the conv layer only calls __ggml_q8ch_prepare_conv_weight.

// fp16 <-> fp32 bit conversions, IEEE-correct including subnormals. Conv
// per-channel scales d = amax(row)/127 are frequently in the fp16 subnormal
// range (amax < ~0.0078 -> d < 2^-14): those MUST be encoded as fp16
// subnormals, not flushed to zero, or the whole output channel would quantize
// to all-zero weights (measured as a systematic score drop / lost keypoints).
static inline float perch_fp16_to_fp32_bits(uint16_t h) {
  uint32_t sign = (uint32_t)(h >> 15) << 31;
  uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
  if (exp == 0) {
    if (man == 0)
      bits = sign;
    else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) {
        man <<= 1;
        --exp;
      }
      man &= 0x3ff;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31)
    bits = sign | 0x7f800000u | (man << 13);
  else
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

static inline uint16_t perch_fp32_to_fp16_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t e32 = (bits >> 23) & 0xffu;
  uint32_t man = bits & 0x7fffffu;
  if (e32 == 0xff) // Inf / NaN
    return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
  int32_t exp = (int32_t)e32 - 127 + 15;
  if (exp >= 31)
    return (uint16_t)(sign | 0x7c00u); // overflow -> inf
  if (exp <= 0) {
    // subnormal (or zero): shift the significand (with implicit 1) into the
    // 10-bit fp16 mantissa at the denormal position, round-to-nearest-even.
    if (exp < -10)
      return (uint16_t)sign; // magnitude below the smallest fp16 subnormal -> 0
    man |= 0x800000u;        // restore the implicit leading 1
    const uint32_t shift = (uint32_t)(14 - exp); // in [14, 24]
    uint32_t hm = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hm & 1u)))
      ++hm; // ties-to-even; may carry hm up to 0x400 = smallest normal
            // (correct)
    return (uint16_t)(sign | hm);
  }
  uint32_t m = man >> 13;
  const uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (m & 1u)))
    ++m;
  if (m == 0x400u) {
    m = 0;
    if (++exp >= 31)
      return (uint16_t)(sign | 0x7c00u);
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | m);
}

void __ggml_quantize_q8_0_per_channel(const float *src, void *dst,
                                      unsigned int nrow, unsigned int ncol) {
  auto &tm = ThreadManager::Global();
  const unsigned int nb = ncol / QK8_0;
  block_q8_0 *out = reinterpret_cast<block_q8_0 *>(dst);
  tm.parallel_for(0, nrow, [=](size_t n) {
    const float *row = src + (size_t)n * ncol;
    float amax = 0.f;
    for (unsigned int k = 0; k < ncol; ++k)
      amax = std::max(amax, std::fabs(row[k]));
    const float d = amax > 0.f ? amax / 127.f : 1.f;
    const uint16_t d16 = perch_fp32_to_fp16_bits(d);
    const float d_used = perch_fp16_to_fp32_bits(d16); // exact runtime value
    const float inv = d_used > 0.f ? 1.f / d_used : 0.f;
    for (unsigned int j = 0; j < nb; ++j) {
      block_q8_0 &b = out[(size_t)n * nb + j];
      b.d = d16;
      for (int c = 0; c < QK8_0; ++c) {
        const float v = row[j * QK8_0 + c] * inv;
        b.qs[c] = (int8_t)std::max(-128.f, std::min(127.f, std::round(v)));
      }
    }
  });
}

const PerChConvWeight &
__ggml_q8ch_prepare_conv_weight(const void *key, const void *q8_src_x4,
                                const float *fp32_src, unsigned int out_ch,
                                unsigned int CRS, unsigned int khkw,
                                unsigned int in_ch, const void *q4_src_x4) {
  static std::mutex mtx;
  static std::unordered_map<const void *, PerChConvWeight> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  const block_q8_0x4 *q8_src = (const block_q8_0x4 *)q8_src_x4;

  // Parallelized so the one-time conversion (first forward) is not a big
  // single-threaded warmup stall. Batch is 1 in the W8A8 path, so this never
  // nests inside another parallel_for.
  auto &tm = ThreadManager::Global();

  PerChConvWeight w;
  w.Nreal = out_ch;
  w.Kpad = (CRS + QK8_0 - 1) / QK8_0 * QK8_0;
  w.Npad = (out_ch + 3) / 4 * 4;
  const unsigned int nb = w.Kpad / QK8_0;
  const unsigned int NB4 = w.Npad / 4;
  const unsigned int nbq = CRS / QK8_0; // q8 source blocks (CRS % 32 == 0)

  w.scale.assign(out_ch, 1.f);
  w.colsum.assign(out_ch, 0);
  float *scale_ptr = w.scale.data();
  int32_t *colsum_ptr = w.colsum.data();

  // Taps-last K permutation (kernels > 1x1, NNTR_W8A8_PACKA=1 reverts): the
  // filter's K order [in_ch][k_h][k_w] forces the NHWC activation gather into
  // a byte-wise transposing scatter plus a q8_0x4 interleave (gemm.pack).
  // Permuting the weight's K to [k_h][k_w][in_ch] lets the gather copy
  // contiguous NHWC spans and feed the plain-activation kernel directly.
  // The dot product sums the same K multiset and int32 accumulation is exact,
  // so outputs are bit-identical; per-row amax/scale/colsum are order-
  // invariant and unchanged.
  static const bool packa = std::getenv("NNTR_W8A8_PACKA") != nullptr;
  const bool taps =
    !packa && khkw > 1 && in_ch > 0 && (size_t)khkw * in_ch == CRS;
  w.taps_last = taps;
  std::vector<unsigned int> perm; // dst (taps-last) k -> src (filter-order) k
  if (taps) {
    perm.resize(CRS);
    for (unsigned int c = 0; c < in_ch; ++c)
      for (unsigned int t = 0; t < khkw; ++t)
        perm[(size_t)t * in_ch + c] = c * khkw + t;
  }
  const unsigned int *perm_ptr = taps ? perm.data() : nullptr;

  // Memory optimization (opt-in via NNTR_W8A8_PERCH_INPLACE): the repacked int8
  // stream is byte-for-byte the same layout and size as the source Q8_0x4
  // weight when out_ch % 4 == 0 and CRS % 32 == 0 (the pack writes every qs
  // byte, and the kernel ignores the d field). Rather than allocate a second
  // ~equal-sized buffer, requantize in place into the source and share it
  // zero-copy, eliminating the persistent per-conv weight duplicate. Each
  // 4-row group's source blocks are fully dequantized before that group's
  // pack overwrites them, and groups touch disjoint block ranges, so the
  // in-place path stays parallel-safe. Numerically identical to the
  // owned-buffer path.
  static const bool perch_inplace =
    std::getenv("NNTR_W8A8_PERCH_INPLACE") != nullptr;
  const bool use_inplace =
    perch_inplace && q8_src != nullptr && w.Npad == out_ch && w.Kpad == CRS;
  block_q8_0x4 *dst;
  if (use_inplace) {
    dst = const_cast<block_q8_0x4 *>(q8_src);
    w.qs_ext = dst;
  } else {
    w.qs.assign((size_t)NB4 * nb * sizeof(block_q8_0x4), 0);
    dst = (block_q8_0x4 *)w.qs.data();
  }

  // Fast path detection for PER-CHANNEL Q8_0 sources (the "pch" file the
  // offline __ggml_quantize_q8_0_per_channel writes): every block of an
  // output-channel row carries the SAME fp16 scale d. For such a row the
  // generic dequant -> amax -> requant round trip is the identity on the int8
  // quants (that is the pch file's design point), so the FP32 intermediate can
  // be skipped entirely: copy the int8 quants, and reproduce the generic
  // path's scale bit-exactly by replaying its float ops on the row maximum
  // (amax == fl(d * max|q|) because x -> fl(d*x) is monotone and |fl(d*q)| ==
  // fl(d*|q|); qrow == q because |(d*q)*fl(1/scv) - q| << 0.5 so the round
  // returns the same integer). This roughly halves the one-time conversion
  // cost. NNTR_W8A8_NOFASTCONV=1 reverts to the generic path; per-block Q8_0
  // sources (non-uniform d) are detected and take the generic path anyway.
  static const bool no_fastconv =
    std::getenv("NNTR_W8A8_NOFASTCONV") != nullptr;
  bool fast_pch = false;
  if (!no_fastconv && q8_src != nullptr && nbq > 0) {
    fast_pch = true;
    const unsigned int full_groups = out_ch / 4;
    for (unsigned int sc = 0; sc < full_groups && fast_pch; ++sc) {
      const block_q8_0x4 *row = &q8_src[(size_t)sc * nbq];
      for (unsigned int r = 0; r < 4 && fast_pch; ++r) {
        const uint16_t d0 = row[0].d[r];
        for (unsigned int j = 1; j < nbq; ++j)
          if (row[j].d[r] != d0) {
            fast_pch = false;
            break;
          }
      }
    }
  }

  // Streamed per 4-row group: dequant -> per-channel quantize -> pack with a
  // 4 x CRS scratch, instead of materializing the whole FP32 weight plus a
  // whole int8 copy. The monolithic buffers spiked warmup peak RSS by ~5x the
  // weight's int8 size on the largest conv and set the process peak; streaming
  // keeps the transient in the hundreds of KB. The per-row math and ordering
  // are unchanged, so scale/colsum/qs stay bit-identical. A q8 source only
  // feeds full 4-row groups (out_ch / 4); a partial tail group
  // (out_ch % 4 != 0) dequantizes to zeros.
  tm.parallel_for(0, NB4, [=](size_t sc) {
    std::vector<int8_t> wi((size_t)4 * CRS);
    int8_t *wi_ptr = wi.data();

    if (q8_src && fast_pch) {
      // Per-channel source: int8 straight through (no FP32 round trip). The
      // whole group's int8 -> wi_ptr, scale/colsum reproduced bit-exactly.
      if ((unsigned int)(sc + 1) * 4 <= out_ch) {
        for (unsigned int r = 0; r < 4; ++r) {
          const unsigned int n = (unsigned int)sc * 4 + r;
          const float d =
            perch_fp16_to_fp32_bits(q8_src[(size_t)sc * nbq].d[r]);
          int8_t *qrow = &wi_ptr[(size_t)r * CRS];
          int maxq = 0;
          int32_t qsum = 0;
          for (unsigned int j = 0; j < nbq; ++j) {
            const block_q8_0x4 &sb = q8_src[(size_t)sc * nbq + j];
            for (unsigned int sub = 0; sub < 4; ++sub)
              for (unsigned int c = 0; c < 8; ++c) {
                const int8_t v = sb.qs[32 * sub + 8 * r + c];
                qrow[j * QK8_0 + sub * 8 + c] = v;
                const int a = v < 0 ? -(int)v : (int)v;
                if (a > maxq)
                  maxq = a;
                qsum += v;
              }
          }
          const float amax = d * (float)maxq; // == generic path's row amax
          scale_ptr[n] = amax > 0.f ? amax / 127.f : 1.f;
          colsum_ptr[n] = qsum;
        }
      } else {
        // q8 tail group (out_ch % 4 != 0): zeros, scale 1, colsum 0 --
        // exactly what the generic path produces for it.
        std::memset(wi_ptr, 0, (size_t)4 * CRS);
      }
    } else {
      // Generic path: dequant -> per-channel quantize with a 4 x CRS FP32
      // scratch. Used for FP32 sources and per-BLOCK Q8_0 (non-uniform d).
      std::vector<float> wf((size_t)4 * CRS);
      float *wf_ptr = wf.data();

      // 1) This group's 4 rows as FP32 (rows past out_ch / q8 tail -> 0).
      if (q8_src) {
        if ((unsigned int)(sc + 1) * 4 <= out_ch) {
          for (unsigned int j = 0; j < nbq; ++j) {
            const block_q8_0x4 &sb = q8_src[(size_t)sc * nbq + j];
            for (unsigned int r = 0; r < 4; ++r) {
              const float d = perch_fp16_to_fp32_bits(sb.d[r]);
              for (unsigned int sub = 0; sub < 4; ++sub)
                for (unsigned int c = 0; c < 8; ++c)
                  wf_ptr[(size_t)r * CRS + j * QK8_0 + sub * 8 + c] =
                    d * (float)sb.qs[32 * sub + 8 * r + c];
            }
          }
        } else {
          std::memset(wf_ptr, 0, (size_t)4 * CRS * sizeof(float));
        }
      } else if (q4_src_x4) {
        if ((unsigned int)(sc + 1) * 4 <= out_ch) {
          const uint8_t *q4_data = (const uint8_t *)q4_src_x4;
          for (unsigned int j = 0; j < nbq; ++j) {
            size_t base_off = ((size_t)sc * nbq + j) * 72;
            float d_vals[4];
            for (unsigned int r = 0; r < 4; ++r) {
              uint16_t d16;
              std::memcpy(&d16, q4_data + base_off + r * 2, 2);
              d_vals[r] = perch_fp16_to_fp32_bits(d16);
            }
            const uint8_t *qs_block = q4_data + base_off + 8;
            for (unsigned int i = 0; i < 8; ++i) {
              unsigned int r = i % 4;
              unsigned int src_offset = (i / 4) * 8;
              for (unsigned int b = 0; b < 8; ++b) {
                uint8_t q = qs_block[i * 8 + b] ^ 0x88;
                int x0 = (q & 0x0F) - 8;
                int x1 = (q >> 4) - 8;
                unsigned int col_start = j * QK8_0 + src_offset;
                wf_ptr[(size_t)r * CRS + col_start + b] = (float)x0 * d_vals[r];
                wf_ptr[(size_t)r * CRS + col_start + b + 16] =
                  (float)x1 * d_vals[r];
              }
            }
          }
        } else {
          std::memset(wf_ptr, 0, (size_t)4 * CRS * sizeof(float));
        }
      } else {
        for (unsigned int r = 0; r < 4; ++r) {
          const unsigned int n = (unsigned int)sc * 4 + r;
          if (n < out_ch)
            std::memcpy(&wf_ptr[(size_t)r * CRS], &fp32_src[(size_t)n * CRS],
                        (size_t)CRS * sizeof(float));
          else
            std::memset(&wf_ptr[(size_t)r * CRS], 0,
                        (size_t)CRS * sizeof(float));
        }
      }

      // 2) Per-output-channel int8 quantize (+ per-row quant sum for the
      // asymmetric-activation bias fold).
      for (unsigned int r = 0; r < 4; ++r) {
        const unsigned int n = (unsigned int)sc * 4 + r;
        if (n >= out_ch) {
          std::memset(&wi_ptr[(size_t)r * CRS], 0, CRS);
          continue;
        }
        float amax = 0.f;
        const float *row = &wf_ptr[(size_t)r * CRS];
        for (unsigned int k = 0; k < CRS; ++k)
          amax = std::max(amax, std::fabs(row[k]));
        const float scv = amax > 0.f ? amax / 127.f : 1.f;
        scale_ptr[n] = scv;
        const float inv = 1.f / scv;
        int8_t *qrow = &wi_ptr[(size_t)r * CRS];
        int32_t qsum = 0;
        for (unsigned int k = 0; k < CRS; ++k) {
          qrow[k] =
            (int8_t)std::max(-128.f, std::min(127.f, std::round(row[k] * inv)));
          qsum += qrow[k];
        }
        colsum_ptr[n] = qsum;
      }
    }

    // 3) Pack the group's int8 rows -> q8_0x4 qs (d ignored by the kernel),
    // applying the taps-last K permutation when enabled. Shared by both paths.
    for (unsigned int j = 0; j < nb; ++j) {
      block_q8_0x4 &sb = dst[(size_t)sc * nb + j];
      for (unsigned int r = 0; r < 4; ++r) {
        for (unsigned int sub = 0; sub < 4; ++sub)
          for (unsigned int c = 0; c < 8; ++c) {
            const unsigned int k = j * QK8_0 + sub * 8 + c;
            const unsigned int ks = (k < CRS && perm_ptr) ? perm_ptr[k] : k;
            sb.qs[32 * sub + 8 * r + c] =
              (k < CRS) ? wi_ptr[(size_t)r * CRS + ks] : 0;
          }
      }
    }
  });
  it = cache.emplace(key, std::move(w)).first;
  return it->second;
}

/**
 * @brief Per-CHANNEL W8A8 indirect conv GEMM: int8 activation (per-tensor
 * scale) x int8 weight (per-output-channel scale) -> FP32.
 *
 * Same gather + q8_0x4 int8 packing as __ggml_q8_0_q8_0_indirect_GEMM_i8a, but
 * the micro-kernel is nntr_gemm_q8ch_4x4_f32 (int32 accumulate over all K, one
 * scale a_scale*w_scale[col] at the end) -- no per-32-block float folding. The
 * weight B is int8 in the q8_0x4 qs layout (its fp16 d fields are ignored) and
 * w_scale is the per-output-channel FP32 scale [N]. K may exceed the geometry's
 * real CRS (zero-padded via the gather stride), and N is the real (unpadded)
 * out_ch count so padded weight columns are never tiled.
 */
void __ggml_q8ch_indirect_GEMM(const unsigned int M, const unsigned int N,
                               const unsigned int K, const int8_t *in,
                               const float act_scale,
                               const ConvGatherParams &geom, const void *B,
                               const float *w_scale, float *C, int8_t pad_value,
                               bool taps_last) {
  auto &tm = ThreadManager::Global();
  const unsigned int nb = K / QK8_0;
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4) * nb;
  const unsigned int Mfull = (M / 4) * 4;
  const unsigned int rem = M % 4;
  const unsigned int M4 = M / 4;
  const unsigned int M4c = (M + 3) / 4;
  const uint16_t d16 = 0; // d ignored by the per-channel kernel

  const size_t B_step = (size_t)nb * sizeof(block_q8_0);
  const unsigned int row_chunk_size = 16, col_chunk_size = 16;

  // 1x1 unit-stride convs gather the identity: `in` already IS the plain
  // [M, K] int8 activation matrix, so the plain-activation kernel reads it
  // directly and the whole packing pass (+ its QA buffer) disappears.
  const bool ident = conv_gather_is_identity(geom, K);
  if (ident) {
    if (Mfull > 0) {
      const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
      const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
      tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
        unsigned int r = static_cast<unsigned int>(i / col_loop);
        unsigned int c = static_cast<unsigned int>(i % col_loop);
        unsigned int r_start = r * row_chunk_size;
        unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
        unsigned int c_start = c * col_chunk_size;
        unsigned int c_end = std::min(col_chunk_size * (c + 1), N);
        nntr_gemm_q8ch_plainA_f32(
          (int)K, C + (size_t)r_start * N + c_start, N,
          (const void *)((const char *)B + (size_t)c_start * B_step),
          w_scale + c_start, in + (size_t)r_start * K, K, act_scale,
          (int)(r_end - r_start), (int)(c_end - c_start));
      });
    }
    if (rem > 0) {
      // The M%4 tail's 4-row group must not read past row M, so copy the rem
      // rows into a zero-padded 4xK plain buffer.
      static thread_local std::vector<int8_t> tail;
      if (tail.size() < (size_t)4 * K)
        tail.resize((size_t)4 * K);
      std::memset(tail.data(), 0, (size_t)4 * K);
      std::memcpy(tail.data(), in + (size_t)Mfull * K, (size_t)rem * K);
      std::vector<float> scratch((size_t)4 * N);
      float *scratch_ptr = scratch.data();
      const int8_t *tail_a = tail.data();
      const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
      tm.parallel_for(0, col_loop, [=](size_t c) {
        unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
        unsigned int c_end =
          std::min(col_chunk_size * ((unsigned int)c + 1), N);
        nntr_gemm_q8ch_plainA_f32(
          (int)K, scratch_ptr + c_start, N,
          (const void *)((const char *)B + (size_t)c_start * B_step),
          w_scale + c_start, tail_a, K, act_scale, 4, (int)(c_end - c_start));
      });
      for (unsigned int rr = 0; rr < rem; ++rr)
        std::memcpy(C + (size_t)(Mfull + rr) * N, scratch_ptr + (size_t)rr * N,
                    (size_t)N * sizeof(float));
    }
    return;
  }

  // Pack-free gathered path (weight K permuted taps-last): gather plain
  // [M, K] int8 rows -- for NHWC each tap is a contiguous in_ch copy, unit
  // width-dilation collapses a whole k_w run into one memcpy -- and feed the
  // plain-activation kernel directly. The q8_0x4 interleave pass (gemm.pack)
  // and its transposing byte scatter disappear; results are bit-identical to
  // the packed path (int32 accumulation over a permuted K is exact).
  if (taps_last) {
    static thread_local std::vector<int8_t> QAp;
    const size_t qa_p_bytes = (size_t)M4c * 4 * K;
    if (QAp.size() < qa_p_bytes)
      QAp.resize(qa_p_bytes);
    int8_t *QAp_ptr = QAp.data();

    {
      const unsigned int QCHUNK = 64;
      const size_t qloops = (M + QCHUNK - 1) / QCHUNK;
      tm.parallel_for(0, qloops, [=](size_t q) {
        const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
        const unsigned int r1 = std::min(r0 + QCHUNK, M);
        gather_conv_act_rows_tapslast<int8_t>(QAp_ptr + (size_t)r0 * K, in,
                                              geom, (int)r0, (int)(r1 - r0),
                                              (int)K, pad_value);
      });
      if (rem > 0) // zero the tail group's rows past M (kernel reads 4 rows)
        std::memset(QAp_ptr + (size_t)M * K, 0, (size_t)(4 - rem) * K);
    }

    if (Mfull > 0) {
      const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
      const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
      tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
        unsigned int r = static_cast<unsigned int>(i / col_loop);
        unsigned int c = static_cast<unsigned int>(i % col_loop);
        unsigned int r_start = r * row_chunk_size;
        unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
        unsigned int c_start = c * col_chunk_size;
        unsigned int c_end = std::min(col_chunk_size * (c + 1), N);
        nntr_gemm_q8ch_plainA_f32(
          (int)K, C + (size_t)r_start * N + c_start, N,
          (const void *)((const char *)B + (size_t)c_start * B_step),
          w_scale + c_start, QAp_ptr + (size_t)r_start * K, K, act_scale,
          (int)(r_end - r_start), (int)(c_end - c_start));
      });
    }
    if (rem > 0) {
      std::vector<float> scratch((size_t)4 * N);
      float *scratch_ptr = scratch.data();
      const int8_t *tail_a = QAp_ptr + (size_t)Mfull * K;
      const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
      tm.parallel_for(0, col_loop, [=](size_t c) {
        unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
        unsigned int c_end =
          std::min(col_chunk_size * ((unsigned int)c + 1), N);
        nntr_gemm_q8ch_plainA_f32(
          (int)K, scratch_ptr + c_start, N,
          (const void *)((const char *)B + (size_t)c_start * B_step),
          w_scale + c_start, tail_a, K, act_scale, 4, (int)(c_end - c_start));
      });
      for (unsigned int rr = 0; rr < rem; ++rr)
        std::memcpy(C + (size_t)(Mfull + rr) * N, scratch_ptr + (size_t)rr * N,
                    (size_t)N * sizeof(float));
    }
    return;
  }

  // Reused thread_local packing buffer (per-forward malloc + page-fault churn
  // showed up as unattributed per-conv time). Every super-block [0, M4c) is
  // rewritten below, so no clearing is needed on reuse.
  static thread_local std::vector<char> QA;
  const size_t qa_bytes = (size_t)M4c * qa_4_rows_size;
  if (QA.size() < qa_bytes)
    QA.resize(qa_bytes);
  char *QA_ptr = QA.data();

  {
    const unsigned int QCHUNK = 64;
    if (Mfull > 0) {
      const size_t qloops = (Mfull + QCHUNK - 1) / QCHUNK;
      tm.parallel_for(0, qloops, [=](size_t q) {
        const unsigned int r0 = static_cast<unsigned int>(q) * QCHUNK;
        const unsigned int r1 = std::min(r0 + QCHUNK, Mfull);
        std::vector<int8_t> tile((size_t)4 * K);
        for (unsigned int r = r0; r < r1; r += 4) {
          gather_conv_act_rows<int8_t>(tile.data(), in, geom, (int)r, 4, (int)K,
                                       pad_value);
          pack_i8_rows_q8_0x4(tile.data(), K, d16,
                              QA_ptr + (size_t)(r / 4) * qa_4_rows_size);
        }
      });
    }
    if (rem > 0) {
      std::vector<int8_t> tile((size_t)4 * K, 0);
      gather_conv_act_rows<int8_t>(tile.data(), in, geom, (int)Mfull, (int)rem,
                                   (int)K, pad_value);
      pack_i8_rows_q8_0x4(tile.data(), K, d16,
                          QA_ptr + (size_t)M4 * qa_4_rows_size);
    }
  }

  if (Mfull > 0) {
    const size_t row_loop = (Mfull + row_chunk_size - 1) / row_chunk_size;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, row_loop * col_loop, [=](size_t i) {
      unsigned int r = static_cast<unsigned int>(i / col_loop);
      unsigned int c = static_cast<unsigned int>(i % col_loop);
      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), Mfull);
      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);
      nntr_gemm_q8ch_4x4_f32(
        (int)K, C + (size_t)r_start * N + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        w_scale + c_start,
        (const void *)(QA_ptr + (size_t)(r_start / 4) * qa_4_rows_size),
        act_scale, (int)(r_end - r_start), (int)(c_end - c_start));
    });
  }
  if (rem > 0) {
    std::vector<float> scratch((size_t)4 * N);
    float *scratch_ptr = scratch.data();
    const char *tail_a = QA_ptr + (size_t)M4 * qa_4_rows_size;
    const unsigned int col_loop = (N + col_chunk_size - 1) / col_chunk_size;
    tm.parallel_for(0, col_loop, [=](size_t c) {
      unsigned int c_start = static_cast<unsigned int>(c) * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * ((unsigned int)c + 1), N);
      nntr_gemm_q8ch_4x4_f32(
        (int)K, scratch_ptr + c_start, N,
        (const void *)((const char *)B + (size_t)c_start * B_step),
        w_scale + c_start, (const void *)tail_a, act_scale, 4,
        (int)(c_end - c_start));
    });
    for (unsigned int rr = 0; rr < rem; ++rr)
      std::memcpy(C + (size_t)(Mfull + rr) * N, scratch_ptr + (size_t)rr * N,
                  (size_t)N * sizeof(float));
  }
}

} // namespace nntrainer
