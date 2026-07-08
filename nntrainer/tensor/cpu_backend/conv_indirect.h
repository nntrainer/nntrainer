// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   conv_indirect.h
 * @date   25 June 2026
 * @brief  Indirect (im2col-fused) convolution activation gather.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NaN or Inf
 *
 * @details This header provides a single pure-C++ gather primitive used by the
 * indirect quantized convolution path: instead of materializing the full FP32
 * im2col column buffer and feeding it to the GEMM, the GEMM's activation packer
 * gathers each row tile directly from the NCHW input on demand. The gather is
 * factored here (platform-agnostic, no SIMD/backend dependency) so it can be
 * verified bit-identically against a naive im2col reference on every platform,
 * independent of the ARM-only quantize/GEMM micro-kernels that consume it.
 */

#ifndef __NNTR_CONV_INDIRECT_H__
#define __NNTR_CONV_INDIRECT_H__

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <tensor_dim.h> // for _FP16 (defined under ENABLE_FP16)

namespace nntrainer {

/**
 * @brief Geometry describing one conv layer's im2col gather.
 *
 * Mirrors the parameters Conv2DLayer feeds to im2col(). out_w is carried so a
 * flat output-spatial row index m can be split into (oh, ow) = (m / out_w,
 * m % out_w). K (the gathered row length) is in_ch * k_h * k_w.
 */
struct ConvGatherParams {
  int in_ch;    /**< input channels */
  int in_h;     /**< input height */
  int in_w;     /**< input width */
  int k_h;      /**< kernel height */
  int k_w;      /**< kernel width */
  int pad_t;    /**< top padding */
  int pad_l;    /**< left padding */
  int stride_h; /**< vertical stride */
  int stride_w; /**< horizontal stride */
  int dil_h;    /**< vertical dilation */
  int dil_w;    /**< horizontal dilation */
  int out_w;    /**< output width (maps row index -> output column) */
  bool is_nhwc = false;
};

/**
 * @brief Gather nrows im2col activation rows into a contiguous buffer.
 *
 * Reproduces im2col's [OH*OW, CRS] layout exactly for the row range
 * [m0, m0 + nrows): output-spatial row m maps to (oh, ow) = (m / out_w,
 * m % out_w); within a row the K = in_ch*k_h*k_w values are laid out as
 * [channel][k_h][k_w]; positions falling outside the input (padding) are
 * written as 0. The destination is fully written, so no pre-zeroing is needed.
 *
 * Templated over the activation element type so the same byte-identical gather
 * serves the FP32 indirect-conv path (float, gathered tile feeds the FP32-input
 * Q8_0 quantizer) and the FP16 indirect-conv path (_FP16, tile feeds the
 * FP16-input Q8_0 quantizer __ggml_quantize_*_q8_0(const _FP16*)). Padding is
 * zero-initialised via memset typed to sizeof(T), so FP16 padding is +0.0 too.
 *
 * @param[out] dst   buffer of nrows*K elements of type T, fully overwritten
 * @param[in]  in    batch-sliced contiguous NCHW input base pointer (type T)
 * @param[in]  p     gather geometry
 * @param[in]  m0    first output-spatial row to gather
 * @param[in]  nrows number of rows to gather
 */
template <typename T>
inline void gather_conv_act_rows(T *dst, const T *in, const ConvGatherParams &p,
                                 int m0, int nrows) {
  const int K = p.in_ch * p.k_h * p.k_w;
  const long inHW = (long)p.in_h * (long)p.in_w;
  const bool unit_dil = (p.dil_h == 1 && p.dil_w == 1);
  const int khkw = p.k_h * p.k_w;

  for (int r = 0; r < nrows; ++r) {
    const int m = m0 + r;
    const int oh = m / p.out_w;
    const int ow = m % p.out_w;
    T *row = dst + (long)r * K;
    /// every K element is written below for unit dilation only along valid
    /// runs, so zero the row up front: padding positions stay 0 (matches
    /// im2col). sizeof(T) so FP16 padding is +0.0 just like FP32.
    std::memset(row, 0, (size_t)K * sizeof(T));

    const int h0 = oh * p.stride_h - p.pad_t;
    const int w0 = ow * p.stride_w - p.pad_l;

    if (p.is_nhwc) {
      for (int kh = 0; kh < p.k_h; ++kh) {
        const int h = h0 + kh * p.dil_h;
        if (h < 0 || h >= p.in_h)
          continue;
        for (int kw = 0; kw < p.k_w; ++kw) {
          const int w = w0 + kw * p.dil_w;
          if (w < 0 || w >= p.in_w)
            continue;
          const T *in_ptr = in + (long)(h * p.in_w + w) * p.in_ch;
          for (int c = 0; c < p.in_ch; ++c) {
            row[c * khkw + kh * p.k_w + kw] = in_ptr[c];
          }
        }
      }
    } else {
      for (int c = 0; c < p.in_ch; ++c) {
        const T *in_c = in + (long)c * inHW;
        T *row_c = row + (long)c * khkw;
        for (int kh = 0; kh < p.k_h; ++kh) {
          const int h = h0 + kh * p.dil_h;
          if (h < 0 || h >= p.in_h)
            continue; /// whole kernel row is outside the input -> left as 0
          const T *in_row = in_c + (long)h * p.in_w;
          T *dst_run = row_c + (long)kh * p.k_w;
          if (unit_dil) {
            /// the kernel-width window maps a contiguous input run to a
            /// contiguous dest run: copy the in-bounds span in one memcpy.
            int wlo = w0 < 0 ? 0 : w0;
            int whi = w0 + p.k_w;
            if (whi > p.in_w)
              whi = p.in_w;
            if (whi > wlo)
              std::memcpy(dst_run + (wlo - w0), in_row + wlo,
                          (size_t)(whi - wlo) * sizeof(T));
          } else {
            for (int kw = 0; kw < p.k_w; ++kw) {
              const int w = w0 + kw * p.dil_w;
              if (w >= 0 && w < p.in_w)
                dst_run[kw] = in_row[w];
            }
          }
        }
      }
    }
  }
}

/**
 * @brief FP32 gather (legacy named entry; alias of the template instantiation).
 *
 * Kept as a concrete non-template function so existing FP32-indirect-conv
 * call sites (gemm_q4_0_indirect_conv_fp32 ->
 * __ggml_q4_0_4x8_q8_0_indirect_GEMM) keep their symbol and address-takability.
 */
inline void gather_conv_act_rows_fp32(float *dst, const float *in,
                                      const ConvGatherParams &p, int m0,
                                      int nrows) {
  gather_conv_act_rows<float>(dst, in, p, m0, nrows);
}

#ifdef ENABLE_FP16
/**
 * @brief FP16 gather for the FP16-activation indirect-conv path.
 *
 * Identical layout to gather_conv_act_rows_fp32 but _FP16 typed, so the
 * gathered tile feeds __ggml_quantize_row_q8_0(const _FP16*, ...) /
 * __ggml_quantize_mat_q8_0_4x8(const _FP16*, ...) directly without an FP32
 * staging copy — preserving the indirect path's no-col-materialization memory
 * win for FP16 activations too.
 */
inline void gather_conv_act_rows_fp16(_FP16 *dst, const _FP16 *in,
                                      const ConvGatherParams &p, int m0,
                                      int nrows) {
  gather_conv_act_rows<_FP16>(dst, in, p, m0, nrows);
}
#endif

/**
 * @brief Pre-quantized Q8_0 gather for the Q8_0-activation indirect-conv path.
 *
 * Gathers pre-quantized block_q8_0 blocks from the Q8_0 input tensor and
 * interleaves them directly into the block_q8_0x4 format expected by the SMMLA
 * GEMM kernel. Bypasses all FP16 dequantization and Q8_0 re-quantization,
 * executing as a pure byte-copy (memcpy) operation.
 */
inline void gather_conv_act_rows_q8_0(void *vy, const void *vx,
                                      const ConvGatherParams &p, int m0,
                                      int nrows) {
  struct local_block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };

  struct local_block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };

  const local_block_q8_0 *in = (const local_block_q8_0 *)vx;
  local_block_q8_0x4 *y = (local_block_q8_0x4 *)vy;

  const int ch_blocks = p.in_ch / 32;
  const int blocks_per_row = p.k_h * p.k_w * ch_blocks;

  const int hstride = p.stride_h;
  const int wstride = p.stride_w;

  for (int b_idx = 0; b_idx < blocks_per_row; ++b_idx) {
    int khkw_idx = b_idx / ch_blocks;
    int cb = b_idx % ch_blocks;
    int kh = khkw_idx / p.k_w;
    int kw = khkw_idx % p.k_w;

    local_block_q8_0x4 &dst_block = y[b_idx];

    // block_q8_0x4.qs[128] layout expected by nntr_gemm_q4_0_4x8_q8_0_fp16:
    //   qs[32*j + 8*row + lane], j=8-element chunk (0..3), row=0..3, lane=0..7.
    // (matches __ggml_quantize_mat_q8_0_4x8 and Q8_0_Tensor::dot interleave.)
    // Each source plain block_q8_0 row holds qs[32*j + lane]; scatter its 8
    // lanes per (j,row) into the interleaved dst — do NOT write row-contiguous.
    for (int r = 0; r < 4; ++r) {
      if (r >= nrows) {
        dst_block.d[r] = 0;
        for (int j = 0; j < 4; ++j)
          std::memset(&dst_block.qs[32 * j + 8 * r], 0, 8);
        continue;
      }

      const int m = m0 + r;
      const int oh = m / p.out_w;
      const int ow = m % p.out_w;

      const int h = oh * hstride - p.pad_t + kh * p.dil_h;
      const int w = ow * wstride - p.pad_l + kw * p.dil_w;

      if (h >= 0 && h < p.in_h && w >= 0 && w < p.in_w) {
        const local_block_q8_0 &src_block =
          in[(h * p.in_w + w) * ch_blocks + cb];
        dst_block.d[r] = src_block.d;
        for (int j = 0; j < 4; ++j)
          std::memcpy(&dst_block.qs[32 * j + 8 * r], &src_block.qs[8 * j], 8);
      } else {
        dst_block.d[r] = 0;
        for (int j = 0; j < 4; ++j)
          std::memset(&dst_block.qs[32 * j + 8 * r], 0, 8);
      }
    }
  }
}

/**
 * @brief Pre-quantized Q8_0 gather for single row (leftover / remainder).
 *
 * Gathers a single row of pre-quantized block_q8_0 blocks from the Q8_0 input
 * tensor without interleaving. Used by remainder rows in GEMV computation.
 */
inline void gather_conv_act_rows_q8_0_single(void *vy, const void *vx,
                                             const ConvGatherParams &p, int m) {
  struct local_block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };

  const local_block_q8_0 *in = (const local_block_q8_0 *)vx;
  local_block_q8_0 *y = (local_block_q8_0 *)vy;

  const int ch_blocks = p.in_ch / 32;
  const int blocks_per_row = p.k_h * p.k_w * ch_blocks;

  const int hstride = p.stride_h;
  const int wstride = p.stride_w;

  const int oh = m / p.out_w;
  const int ow = m % p.out_w;

  const int h0 = oh * hstride - p.pad_t;
  const int w0 = ow * wstride - p.pad_l;

  for (int b_idx = 0; b_idx < blocks_per_row; ++b_idx) {
    int khkw_idx = b_idx / ch_blocks;
    int cb = b_idx % ch_blocks;
    int kh = khkw_idx / p.k_w;
    int kw = khkw_idx % p.k_w;

    const int h = h0 + kh * p.dil_h;
    const int w = w0 + kw * p.dil_w;

    if (h >= 0 && h < p.in_h && w >= 0 && w < p.in_w) {
      y[b_idx] = in[(h * p.in_w + w) * ch_blocks + cb];
    } else {
      y[b_idx].d = 0;
      std::memset(y[b_idx].qs, 0, 32);
    }
  }
}

} // namespace nntrainer

#endif /* __NNTR_CONV_INDIRECT_H__ */
