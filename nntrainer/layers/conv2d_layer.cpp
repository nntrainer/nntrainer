// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   02 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <conv2d_layer.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <q8_0_tensor.h>
#include <tensor_dim.h>
#include <thread>
#include <thread_manager.h>
#include <util_func.h>

namespace nntrainer {

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

/**
 * @brief In-place SiLU / swish (x * sigmoid(x)) over a contiguous buffer.
 */
template <typename T>
static inline void convApplySwishInplace(T *data, size_t n) {
  auto &tm = ThreadManager::Global();
  const size_t nthreads = std::max<size_t>(1, tm.getComputeThreadCount());
  const size_t chunk = (n + nthreads - 1) / nthreads;
  
  static const bool use_approx = []() {
    if (const char* env_p = std::getenv("YOLO_APPROX_SILU")) {
      return std::string(env_p) == "1";
    }
    return false;
  }();

  tm.parallel_for(0, nthreads, [&](size_t t) {
    const size_t start = t * chunk;
    if (start >= n)
      return;
    const size_t end = std::min(start + chunk, n);
    size_t i = start;

    if (use_approx) {
#if defined(__ARM_NEON)
      if constexpr (std::is_same_v<T, _FP16>) {
        const float32x4_t v_three = vdupq_n_f32(3.0f);
        const float32x4_t v_six_inv = vdupq_n_f32(1.0f / 6.0f);
        const float32x4_t v_zero = vdupq_n_f32(0.0f);
        const float32x4_t v_six = vdupq_n_f32(6.0f);
        
        for (; i + 7 < end; i += 8) {
          float16x8_t vx = vld1q_f16(reinterpret_cast<const __fp16*>(&data[i]));

          float32x4_t vx_lo = vcvt_f32_f16(vget_low_f16(vx));
          float32x4_t v_add_lo = vaddq_f32(vx_lo, v_three);
          float32x4_t v_relu6_lo = vminq_f32(vmaxq_f32(v_add_lo, v_zero), v_six);
          float32x4_t vres_lo = vmulq_f32(vmulq_f32(vx_lo, v_relu6_lo), v_six_inv);

          float32x4_t vx_hi = vcvt_f32_f16(vget_high_f16(vx));
          float32x4_t v_add_hi = vaddq_f32(vx_hi, v_three);
          float32x4_t v_relu6_hi = vminq_f32(vmaxq_f32(v_add_hi, v_zero), v_six);
          float32x4_t vres_hi = vmulq_f32(vmulq_f32(vx_hi, v_relu6_hi), v_six_inv);

          vst1q_f16(reinterpret_cast<__fp16*>(&data[i]), vcombine_f16(vcvt_f16_f32(vres_lo), vcvt_f16_f32(vres_hi)));
        }
      }
#endif
      for (; i < end; ++i) {
        const float x = static_cast<float>(data[i]);
        float relu6 = std::max(0.0f, std::min(x + 3.0f, 6.0f));
        data[i] = static_cast<T>(x * relu6 / 6.0f);
      }
    } else {
      // EXACT original bca91663 baseline loop!
      for (; i < end; ++i) {
        const float x = static_cast<float>(data[i]);
        data[i] = static_cast<T>(x / (1.0f + std::exp(-x)));
      }
    }
  });
}

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()},
                   out.getTensorType());
}

#ifdef ENABLE_FP16
/**
 * @brief Quantize FP16 NHWC [n_spatial, in_ch] -> plain block_q8_0
 * [n_spatial][in_ch/32].
 *
 * NHWC input is already row-major (channel innermost): src[r * in_ch + c].
 * No transpose needed — each row r has in_ch contiguous FP16 channels.
 * Q8_0 requires in_ch % 32 == 0 (caller must check).
 * dst must hold n_spatial * (in_ch/32) * sizeof(block_q8_0) bytes.
 */
static inline void quantize_nhwc_q8_0_rows(const _FP16 *src, int n_spatial,
                                           int in_ch,
                                           ::nntrainer::block_q8_0 *dst) {
  const int nb = in_ch / 32;
  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 512;
  const size_t loops = ((size_t)n_spatial + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned r0 = (unsigned)idx * chunk;
    unsigned r1 = std::min(r0 + chunk, (unsigned)n_spatial);
    for (unsigned r = r0; r < r1; ++r) {
      const _FP16 *row = src + (size_t)r * in_ch;
      for (int b = 0; b < nb; ++b) {
        const _FP16 *blk = row + b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) {
          float v = std::abs(static_cast<float>(blk[j]));
          if (v > amax)
            amax = v;
        }
        const float d = amax / 127.f;
        const float id = d ? 1.f / d : 0.f;
        _FP16 d_h = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_h, 2);
        ::nntrainer::block_q8_0 &out_blk = dst[(size_t)r * nb + b];
        out_blk.d = d_u16;
        for (int j = 0; j < 32; ++j)
          out_blk.qs[j] = (int8_t)std::roundf(static_cast<float>(blk[j]) * id);
      }
    }
  });
}

/**
 * @brief Quantize FP16 NHWC [owoh, in_ch] directly into the block_q8_0x4
 *        (4-row interleaved) layout the SMMLA GEMM consumes — single pass.
 *
 * NHWC source is row-major (channel innermost): element (r, c) at
 * src[r*in_ch+c]. This is the NHWC-read counterpart of
 * transpose_quantize_q8_0x4_act (which reads NCHW channel-major). It fuses the
 * two passes the prior 1x1 W4A8 path performed (quantize_nhwc_q8_0_rows ->
 * plain block_q8_0, then Q8_0_Tensor::dot repacks to x4) into one, and lets the
 * caller invoke Q8_0_Tensor::dot_prepacked_x4 (no per-conv repack, no per-conv
 * QA malloc). Output bytes are identical to what the two-pass path produced.
 * dst layout: M4=owoh/4 groups of block_q8_0x4 (136 B/blk) followed by (owoh %
 * 4) remainder rows as plain block_q8_0 (34 B/blk) — exactly what
 * dot_prepacked_x4 expects. dst must hold the same total as the block_q8_0
 * buffer (136 B per 4 rows == 4 * 34 B). Q8_0 requires in_ch % 32 == 0.
 */
static inline void quantize_nhwc_q8_0x4_rows(const _FP16 *src, int in_ch,
                                             int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  struct block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };
  const int qk = 32;
  const int nb = in_ch / qk;
  const int M4 = owoh / 4;
  const int rem = owoh % 4;
  block_q8_0x4 *y4 = static_cast<block_q8_0x4 *>(dst);
  const size_t qa_4_rows_size = sizeof(block_q8_0x4) * nb;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 256; // groups of 4 rows per task
  const size_t loops = (M4 + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int g0 = idx * chunk;
    unsigned int g1 = std::min(g0 + chunk, (unsigned int)M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      for (int b = 0; b < nb; ++b) {
        block_q8_0x4 &dst_b = y4[g * nb + b];
        for (unsigned int row = 0; row < 4; ++row) {
          const _FP16 *blk = src + (size_t)(r0 + row) * in_ch + b * qk;
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            float val = std::abs(static_cast<float>(blk[j]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          dst_b.d[row] = d_u16;
          for (int j = 0; j < qk; ++j) {
            // qs[32*(j/8) + 8*row + (j%8)] — matches the SMMLA x4 layout.
            dst_b.qs[32 * (j / 8) + 8 * row + (j % 8)] =
              static_cast<int8_t>(std::roundf(static_cast<float>(blk[j]) * id));
          }
        }
      }
    }
  });

  // Remainder rows (owoh % 4): plain block_q8_0 for the GEMV tail.
  if (rem > 0) {
    block_q8_0 *yrem = reinterpret_cast<block_q8_0 *>(
      reinterpret_cast<char *>(dst) + (size_t)M4 * qa_4_rows_size);
    for (int i = 0; i < rem; ++i) {
      unsigned int r = M4 * 4 + i;
      for (int b = 0; b < nb; ++b) {
        const _FP16 *blk = src + (size_t)r * in_ch + b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
          float val = std::abs(static_cast<float>(blk[j]));
          if (val > amax)
            amax = val;
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        _FP16 d_half = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_half, 2);
        yrem[i * nb + b].d = d_u16;
        for (int j = 0; j < qk; ++j)
          yrem[i * nb + b].qs[j] =
            static_cast<int8_t>(std::roundf(static_cast<float>(blk[j]) * id));
      }
    }
  }
}

/**
 * @brief Transpose-and-quantize FP16 NCHW [in_ch, owoh] -> Q8_0 [owoh, in_ch]
 *        in a single fused pass (no intermediate transpose copy).
 *
 * Each output row r (a spatial position) is quantized per 32-channel block:
 * block (r, b) covers channels [b*32, b*32+32). The FP16 source is NCHW
 * (channel-major), so channel c at position r lives at src[c*owoh + r]. This
 * gathers a 32-wide channel run with a strided read and writes a packed
 * block_q8_0 (fp16 scale + 32 int8). Parallelized over spatial positions.
 *
 * dst must hold (owoh * in_ch/32) block_q8_0 = owoh*in_ch/32*34 bytes, laid out
 * row-major as [owoh][in_ch/32] blocks — exactly the [M,K] block_q8_0 layout
 * Q8_0_Tensor::dot / the indirect GEMM consumes (M=owoh, K=in_ch).
 */
static inline void transpose_quantize_q8_0_act(const _FP16 *src, int in_ch,
                                               int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  block_q8_0 *y = static_cast<block_q8_0 *>(dst);
  const int qk = 32;
  const int nb = in_ch / qk;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 1024;
  const size_t loops = (owoh + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int r0 = idx * chunk;
    unsigned int r1 = std::min(r0 + chunk, (unsigned int)owoh);
    for (unsigned int r = r0; r < r1; ++r) {
      for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
          int c = b * qk + j;
          float val = std::abs(static_cast<float>(src[c * owoh + r]));
          if (val > amax)
            amax = val;
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        _FP16 d_half = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_half, 2);
        y[r * nb + b].d = d_u16;
        for (int j = 0; j < qk; ++j) {
          int c = b * qk + j;
          float x0 = static_cast<float>(src[c * owoh + r]) * id;
          y[r * nb + b].qs[j] = std::roundf(x0);
        }
      }
    }
  });
}

/**
 * @brief Fused transpose + quantize FP16 NCHW [in_ch, owoh] directly into the
 *        block_q8_0x4 (4-row interleaved) layout the SMMLA GEMM consumes, with
 *        NO intermediate plain-block pass and NO separate interleave copy.
 *
 * Outputs (M4 = owoh/4) groups of 4 rows; each group packs nb=in_ch/32
 * block_q8_0x4. block_q8_0x4.qs[128] layout = qs[32*j + 8*row + lane],
 * j=8-element chunk (0..3), row=0..3 (matches __ggml_quantize_mat_q8_0_4x8).
 * Remainder (owoh % 4) rows packed as plain block_q8_0 afterward for the GEMV
 * tail. dst must hold the block_q8_0x4 region followed by the remainder
 * block_q8_0 region (same total as Q8_0_Tensor::dot's QA buffer).
 */
static inline void transpose_quantize_q8_0x4_act(const _FP16 *src, int in_ch,
                                                 int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  struct block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };
  const int qk = 32;
  const int nb = in_ch / qk;
  const int M4 = owoh / 4;
  const int rem = owoh % 4;
  block_q8_0x4 *y4 = static_cast<block_q8_0x4 *>(dst);
  const size_t qa_4_rows_size = sizeof(block_q8_0x4) * nb;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 256; // groups of 4 rows per task
  const size_t loops = (M4 + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int g0 = idx * chunk;
    unsigned int g1 = std::min(g0 + chunk, (unsigned int)M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      for (int b = 0; b < nb; ++b) {
        block_q8_0x4 &dst_b = y4[g * nb + b];
        for (unsigned int row = 0; row < 4; ++row) {
          unsigned int r = r0 + row;
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float val = std::abs(static_cast<float>(src[c * owoh + r]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          dst_b.d[row] = d_u16;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float x0 = static_cast<float>(src[c * owoh + r]) * id;
            // qs[32*chunk + 8*row + lane], chunk = j/8, lane = j%8
            dst_b.qs[32 * (j / 8) + 8 * row + (j % 8)] =
              static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    }
  });

  // Remainder rows (owoh % 4): plain block_q8_0 for the GEMV tail.
  if (rem > 0) {
    block_q8_0 *yrem = reinterpret_cast<block_q8_0 *>(
      reinterpret_cast<char *>(dst) + (size_t)M4 * qa_4_rows_size);
    const unsigned int rchunk = 1024;
    const size_t rloops = (rem + rchunk - 1) / rchunk;
    tm.parallel_for(0, rloops, [=](size_t idx) {
      unsigned int i0 = idx * rchunk;
      unsigned int i1 = std::min(i0 + rchunk, (unsigned int)rem);
      for (unsigned int i = i0; i < i1; ++i) {
        unsigned int r = M4 * 4 + i;
        for (int b = 0; b < nb; ++b) {
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float val = std::abs(static_cast<float>(src[c * owoh + r]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          yrem[i * nb + b].d = d_u16;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float x0 = static_cast<float>(src[c * owoh + r]) * id;
            yrem[i * nb + b].qs[j] = static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    });
  }
}
#endif

/**
 * @brief     reconstruct image data from 2d column matrix
 *
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] image image tensor to put
 */
static void col2im(const Tensor &col_matrix, const TensorDim &kdim,
                   const std::array<unsigned, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &image) {

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned k_height = kdim.height();
  unsigned k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned im_channel = image.channel();
  int im_height = image.height();
  int im_width = image.width();

  unsigned hstride = mstride[0];
  unsigned wstride = mstride[1];

  unsigned hdilation = dilation[0];
  unsigned wdilation = dilation[1];

  /// image considering padding
  unsigned im_eff_height = im_height + pt + pb;
  unsigned im_eff_width = im_width + pl + pr;
  image.setZero();

  int h_stride_end = im_eff_height - eff_k_height - pt;
  int w_stride_end = im_eff_width - eff_k_width - pl;

  /** @todo We need to implement way to use this kind of function to work inside
   * of Tensor. Then we could remove to access the getData or getValue which has
   * dependecy of data type.
   */
  auto apply_data = [&](auto *val) {
    using T = std::decay_t<decltype(*val)>;
    unsigned col_w = 0;
    for (int hs = -(int)pt; hs <= h_stride_end; hs += hstride) {
      for (int ws = -(int)pl; ws <= w_stride_end; ws += wstride) {
        unsigned col_h = 0;
        int patch_height_end = hs + eff_k_height;
        int patch_width_end = ws + eff_k_width;
        for (unsigned c = 0; c < im_channel; c++) {
          for (int h = hs; h < patch_height_end; h += hdilation) {
            if (h < 0 || im_height <= h) {
              col_h += k_width;
              continue;
            }
            for (int w = ws; w < patch_width_end; w += wdilation) {
              if (w < 0 || im_width <= w) {
                col_h++;
                continue;
              }

              val = image.getAddress<T>(0, c, h, w);
              *val += col_matrix.getValue<T>(0, 0, col_h, col_w);
              col_h++;
            }
          }
        }
        col_w++;
      }
    }
  };

  if (image.getDataType() == nntrainer::Tdatatype::FP32) {
    float val;
    apply_data(&val);
  }
#ifdef ENABLE_FP16
  else if (image.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 val;
    apply_data(&val);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

/**
 * @brief     reform the data to 2d matrix
 * a region is sampled considering @a padding, @a mstride of unit @a kdim
 * Each region is mapped to one column,
 * if channel mode, kernel channel is considered part of kernel feature
 * if not, kernel channel is consider part of output dimension
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] out out tensor, padding set each time for now
 * @note if out is initialized tensor, setting padding is skipped.
 */
static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &out) {
  /// for channel last mode, this is deprecated for now, leaving here on
  /// purpose.
  /** @code
  //   ================ initialize part ====================
  //   out_height -= 2;
  //   out =
  //     Tensor(k_height * k_width, in.channel() * (out_height) *
  //     (out_width));
  //   unsigned int im_w = 0;
  //   ================ loop part ====================
  //   if (eff_k_height > height || eff_k_width > width)
  //     throw std::runtime_error("Kernel shape bigger than input shape");

  //   for (unsigned int c = 0; c < channel; ++c) {
  //     for (unsigned int hs = 0; hs <= height - eff_k_height; hs +=
  //     mstride[0]) {
  //       for (unsigned int ws = 0; ws <= width - eff_k_width; ws +=
  //       mstride[1]) {
  //         unsigned int im_h = 0;
  //         unsigned int patch_height_end = eff_k_height + hs;
  //         unsigned int patch_width_end = eff_k_width + ws;

  //         for (unsigned int h = hs; h < patch_height_end; h += dilation[0]) {
  //           if (h < ph || in_height + ph <= h) {
  //             im_h += k_width;
  //             continue;
  //           }

  //           for (unsigned int w = ws; w < patch_width_end; w += dilation[1])
  //           {
  //             if (w < pw || in_width + pw <= w) {
  //               im_h++;
  //               continue;
  //             }

  //             float val = in.getValue(0, c, h - ph, w - pw);
  //             out.setValue(0, 0, im_h, im_w, val);
  //             im_h++;
  //           }
  //         }
  //         im_w++;
  //       }
  //     }
  //   }
  */

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int out_height = (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  out.reshape(
    TensorDim({out_height * out_width, in.channel() * k_height * k_width},
              in.getTensorType()));
  // float *out_data = out.getData();

  auto apply_data = [&]<typename T>(T *out_data) {
    int h_stride_end = height - eff_k_height - pt;
    int w_stride_end = width - eff_k_width - pl;

    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int owidth = out.width();
    const int hstride = mstride[0];

    /// Raw contiguous-NCHW input base + inner strides. `in` is a (batch-sliced)
    /// contiguous NCHW tensor, so element (0,c,h,w) lives at
    /// in_base + c*inHW + h*inW + w. Hoisting these out of the inner loop turns
    /// the old per-element in.getValue() -- which re-fetched the data pointer
    /// (getData<T>()) and recomputed a 4-D linear offset (getIndex(): a format
    /// branch + 4 muls + 3 adds) plus a per-element padding branch -- into one
    /// contiguous run copy. im2col is pure data movement; the previous form ran
    /// far below memory bandwidth because that overhead dominated the 4-byte
    /// move. Padding columns are left untouched (the caller zeroes the buffer
    /// once before im2col), so the fast path only writes the valid span.
    const T *in_base = in.getData<T>();
    const size_t inW = (size_t)in_width;
    const size_t inHW = (size_t)in_height * (size_t)in_width;
    const bool unit_dil =
      ((unsigned int)dilation[0] == 1 && (unsigned int)dilation[1] == 1);
    const bool is_nhwc = (in.getFormat() == ml::train::TensorDim::Format::NHWC);

    /// Each output row (oh) writes a disjoint band of `out_width` columns
    /// (rows [oh*out_width, (oh+1)*out_width) of the [OH*OW, CRS] matrix), so
    /// the per-row work is independent and bit-identical when parallelized.
    /// hs and base_im_w are derived from oh directly (no sequential carry).
    auto fill_row = [&](size_t oh) {
      int hs = -(int)pt + (int)oh * hstride;
      unsigned int base_im_w = (unsigned int)oh * out_width;
      unsigned int base_im_h = 0;
      int patch_height_end = eff_k_height + hs;
      /// map the patch to a single line looping through channel
      for (unsigned int c = 0; c < channel; ++c) {
        for (int h = hs; h < patch_height_end; h += dilation[0]) {
          if (h < 0 || in_height <= h) {
            base_im_h += k_width;
            continue;
          }

          if (unit_dil) {
            /// Fast path (dilation == 1): for each output column position the
            /// kernel-width window maps a contiguous source run
            /// in_row[w_lo,w_hi) to a contiguous dest run; copy it in one
            /// memcpy.
            const T *in_row = in_base + (size_t)c * inHW + (size_t)h * inW;
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              int w_lo = ws < 0 ? 0 : ws;
              int w_hi = ws + (int)k_width;
              if (w_hi > in_width)
                w_hi = in_width;
              if (w_hi > w_lo) {
                T *dst =
                  out_data + (size_t)im_w * owidth + base_im_h + (w_lo - ws);
                if (!is_nhwc) {
                  const T *in_row =
                    in_base + (size_t)c * inHW + (size_t)h * inW;
                  std::memcpy(dst, in_row + w_lo,
                              (size_t)(w_hi - w_lo) * sizeof(T));
                } else {
                  /// NHWC: channel is innermost, so the w-run is NOT
                  /// contiguous in source. Gather per (w, channel) element.
                  for (int w = w_lo; w < w_hi; ++w) {
                    dst[w - w_lo] =
                      in_base[((size_t)h * inW + (size_t)w) * channel + c];
                  }
                }
              }
              im_w++;
            }
          } else {
            /// General (dilated) path: original scalar gather, but via the
            /// hoisted base pointer (no per-element getData()/getIndex()).
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              unsigned int im_h = base_im_h;
              int patch_width_end = eff_k_width + ws;

              for (int w = ws; w < patch_width_end; w += dilation[1]) {
                if (w < 0 || in_width <= w) {
                  im_h++;
                  continue;
                }
                if (!is_nhwc) {
                  out_data[(size_t)im_w * owidth + im_h] =
                    in_base[(size_t)c * inHW + (size_t)h * inW + w];
                } else {
                  out_data[(size_t)im_w * owidth + im_h] =
                    in_base[((size_t)h * inW + (size_t)w) * channel + c];
                }
                im_h++;
              }
              im_w++;
            }
          }
          base_im_h += k_width;
        }
      }
    };

    ThreadManager::Global().parallel_for(0, out_height, fill_row);
  };

  if (out.getDataType() == nntrainer::Tdatatype::FP32) {
    float *out_data = out.getData<float>();
    apply_data(out_data);
  }
#ifdef ENABLE_FP16
  else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 *out_data = out.getData<_FP16>();
    apply_data(out_data);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}
} // namespace

enum ConvParams { weight, bias, im2col_scratch, qgemm_scratch, q8act_scratch };

Conv2DLayer::Conv2DLayer(
  const std::array<unsigned int, CONV2D_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), std::array<props::KernelSize, CONV2D_DIM>(),
             std::array<props::Stride, CONV2D_DIM>(), props::Padding2D(),
             std::array<props::Dilation, CONV2D_DIM>(), props::ConvGroups(),
             props::FusedActivation()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void Conv2DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();
  NNTR_THROW_IF(in_dim.channel() % groups != 0 || filter_size % groups != 0,
                std::invalid_argument)
    << "[Conv2D] input channels (" << in_dim.channel() << ") and filters ("
    << filter_size << ") must both be divisible by groups (" << groups << ")";

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // A quantized (Q4_0/QINT4) 1x1 conv is computed as a matmul, so its filter is
  // stored as a [in_ch, out_ch] (K, N) weight that the quantized GEMM consumes
  // directly (no im2col-style [out_ch, CRS] squeeze). Non-quantized or larger
  // kernels keep the standard [out_ch, in_ch/groups, kh, kw] layout.
  const bool quant_matmul_filter =
    (in_t_type.data_type == nntrainer::Tdatatype::Q4_0 ||
     in_t_type.data_type == nntrainer::Tdatatype::QINT4 ||
     in_t_type.data_type == nntrainer::Tdatatype::Q8_0) &&
    groups == 1;

  // Real conv kernel geometry — used for padding/output-size computation even
  // when the quantized weight is stored flattened as [CRS, out_ch].
  TensorDim real_kernel_dim(filter_size, in_dim.channel() / groups,
                            kernel_size[0], kernel_size[1], in_t_type);
  // A quantized (groups==1) conv stores its filter as a [CRS, out_ch] matmul
  // weight (CRS = in_ch*kh*kw), consumed by the quantized GEMM after im2col.
  TensorDim kernel_dim = quant_matmul_filter
                           ? TensorDim(1, 1,
                                       in_dim.channel() * kernel_size[0].get() *
                                         kernel_size[1].get(),
                                       filter_size, in_t_type)
                           : real_kernel_dim;

  // Bias is never quantized (no dequantizer for add); follow activation dtype
  // like other compute layers so a Q4_0/QINT4 weight does not force a Q4_0
  // bias.
  auto bias_t_type = in_dim.getTensorType();
  bias_t_type.data_type = context.getActivationDataType();
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1, bias_t_type);

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, real_kernel_dim, {stride[0], stride[1]},
                       {dilation[0], dilation[1]});

  wt_idx[ConvParams::weight] = context.requestWeight(
    kernel_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  // this output_dim must be the same with dimension of hidden
  unsigned int eff_in_height = in_dim.height() + padding[0] + padding[1];
  unsigned int eff_in_width = in_dim.width() + padding[2] + padding[3];

  unsigned int eff_k_height = (kernel_size[0] - 1) * dilation[0] + 1;
  unsigned int eff_k_width = (kernel_size[1] - 1) * dilation[1] + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height((eff_in_height - eff_k_height) / stride[0] + 1);
  out_dim.width((eff_in_width - eff_k_width) / stride[1] + 1);

  out_dim.setTensorType(in_dim.getTensorType());

  context.setOutputDimensions({out_dim});

  NNTR_THROW_IF(eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1],
                std::invalid_argument)
    << "Failed to initialize: in size + padding is smaller than effective "
       "kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(eff_in_height - padding[0] - kernel_size[0] > IM ||
                  eff_in_width - padding[2] - kernel_size[1] > IM,
                std::invalid_argument)
    << "Failed to initialize: Calculated patch end is over int max";

  // Forward scratch (groups==1 path only): the im2col column buffer and the
  // quantized-GEMM output are otherwise heap-allocated on every forwarding()
  // call. Request them once here (planned into the shared activation arena,
  // FORWARD_FUNC_LIFESPAN) and reuse — no per-forward malloc/free churn. The
  // grouped path keeps its local buffer. NOTE: the im2col buffer must still be
  // re-zeroed each forward (im2col skips padding positions and the arena is
  // reused across layers), so this saves the allocation, not the zeroing.
  wt_idx[ConvParams::im2col_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::qgemm_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::q8act_scratch] = std::numeric_limits<unsigned int>::max();
  if (groups == 1) {
    auto scratch_type = in_dim.getTensorType();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // im2col column buffer [batch, 1, CRS, OH*OW]. Unused by the quant paths
    // that never materialize a col buffer: the 1x1 path (im2col is an identity
    // handled by an input transpose) and, where the fused backend op exists,
    // the non-1x1 path (gather is fused into the q8_0 activation packing).
    if (!(quant_matmul_filter && (is_1x1_s1 || NNTR_HAS_Q4_0_INDIRECT_CONV))) {
      // FP path or quant fallback: materialize the im2col column buffer
      // [batch, 1, CRS, OH*OW] once (planned into the activation arena). The
      // quant 1x1 path (identity input transpose) and the quant indirect path
      // (gather fused into the GEMM's q8_0 packing) never materialize a col
      // buffer, so they request no im2col_scratch here.
      TensorDim col_dim(in_dim.batch(), 1, real_kernel_dim.getFeatureLen(),
                        owoh, scratch_type);
      wt_idx[ConvParams::im2col_scratch] =
        context.requestTensor(col_dim, "im2col", Initializer::NONE, false,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
    // quantized GEMM output [batch, 1, OH*OW, out_ch] (quant path only).
    if (quant_matmul_filter) {
      TensorDim tmp_dim(in_dim.batch(), 1, owoh, filter_size, scratch_type);
      wt_idx[ConvParams::qgemm_scratch] =
        context.requestTensor(tmp_dim, "qgemm_out", Initializer::NONE, false,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
    // Q8_0 activation scratch for NHWC W4A8 path: pre-allocated once so
    // forwarding never calls malloc. Size = max(owoh, in_h*in_w) * nb blocks,
    // stored as a plain float buffer and reinterpret-cast to block_q8_0*.
    // FORWARD_INFER_LIFESPAN (LongTerm) is used instead of
    // FORWARD_FUNC_LIFESPAN (ShortTerm) because ShortTerm scratch shares memory
    // with activation tensors in the pool. Writing Q8_0 bytes there corrupts
    // skip-connection activations that are still live when this layer's forward
    // runs. LongTerm gives this scratch its own allocation that never overlaps
    // with activation memory.
    const bool nhwc_layout =
      (in_dim.getFormat() == ml::train::TensorDim::Format::NHWC);
    if (quant_matmul_filter && nhwc_layout && NNTR_HAS_Q4_0_INDIRECT_CONV) {
      const int in_ch_i = (int)in_dim.channel();
      if (in_ch_i % 32 == 0) {
        const unsigned int max_sp =
          std::max(owoh, (unsigned int)(in_dim.height() * in_dim.width()));
        const unsigned int nb = (unsigned int)in_ch_i / 32;
        // block_q8_0 = 34 bytes; use scratch_type (FP16=2 bytes) for compat.
        const unsigned int n_elems =
          (max_sp * nb * 34 + 1) / 2; // round up to FP16 elements
        TensorDim q8dim(1, 1, 1, n_elems, scratch_type);
        wt_idx[ConvParams::q8act_scratch] =
          context.requestTensor(q8dim, "q8act", Initializer::NONE, false,
                                TensorLifespan::FORWARD_INFER_LIFESPAN);
      }
    }
  }
}

void Conv2DLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  /** Calculate Convolution 2D
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                     [Input Channel * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   * Kernel_size[0]   |_|_|_|_|      |_|_|_|_|
   *  * Kenel_size[1]     | | | | | .... | | | | |
   *    (height)]         |_|_|_|_|      |_|_|_|_|
   *                      | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimention
   *   -> [Channel ( = filter_size = output_dim.channel )]
   *       x [output_dim.height x output_dim.width]
   */
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();
  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();

  if (groups == 1) {
    // A quantized 1x1 conv stores its filter as a [in_ch, out_ch] matmul weight
    // (K, N). The quantized GEMM (dotQnK for Q4_0) takes the weight as the dot
    // *input* (activation is the receiver), so we keep that layout as-is and do
    // NOT squeeze it to [out_ch, CRS] like the FP32 path.
    const auto weight_dtype = filter_kernel.getDataType();
    const bool weight_is_q8 = (weight_dtype == nntrainer::Tdatatype::Q8_0);
    const bool weight_is_quant =
      (weight_dtype == nntrainer::Tdatatype::Q4_0 ||
       weight_dtype == nntrainer::Tdatatype::QINT4 || weight_is_q8);
    const unsigned int owoh = out_dim.width() * out_dim.height();

    TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                  filter_kernel.getDim().getFeatureLen()};
    filter_dim_squeezed.setTensorType(filter_kernel.getTensorType());
    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim_squeezed);
    }

    /**
     * Below sets the pad area values to zero
     * it is faster to do this way than seting selective area to zero
     */
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // Pre-allocated forward scratch (requested once in finalize). The im2col
    // column buffer is used by the FP32 path and the quant non-1x1 *fallback*;
    // the quant 1x1 path (identity input transpose) and the quant non-1x1 fused
    // path (gather folded into the GEMM) need no col buffer — finalize skips
    // im2col_scratch is materialized by finalize only for the FP/fallback path.
    // The quant 1x1 path (identity transpose) and the quant indirect path
    // (gather fused into the GEMM) requested no col buffer in finalize, so the
    // pointer stays null for them.
    const bool use_im2col_scratch =
      !(weight_is_quant && (is_1x1_s1 || NNTR_HAS_Q4_0_INDIRECT_CONV));
    Tensor *col_scratch =
      use_im2col_scratch
        ? &context.getTensor(wt_idx[ConvParams::im2col_scratch])
        : nullptr;
    Tensor *qgemm_scratch =
      weight_is_quant ? &context.getTensor(wt_idx[ConvParams::qgemm_scratch])
                      : nullptr;
    if (col_scratch != nullptr) {
      col_scratch->setZero();
    }

    auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                              void *user_data) {
      for (unsigned int b = s; b < e; ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        Tensor in_sub = input_.getBatchSlice(b, 1);

        if (weight_is_quant) {
          if (in_sub.getFormat() == ml::train::TensorDim::Format::NHWC) {
            // NHWC channel-last quantized convolution:
            // Since physical layout is [owoh, filter_size], we reshape `out` to
            // flat NCHW [1, 1, owoh, filter_size] and write directly,
            // completely bypassing qgemm_scratch and transposes!
            Tensor out_flat = out;
            out_flat.reshape(TensorDim(
              1, 1, owoh, filter_size,
              {ml::train::TensorDim::Format::NCHW, out.getDataType()}));

            const int in_ch_i = (int)in_dim.channel();
            // Q8_0-activation path is an env opt-in only. Q8_0 weights use the
            // proven FP16-activation indirect path (W8A16), dispatched by
            // weight dtype inside convQ4_0Indirect, so they must NOT force q8
            // act here.
            const bool can_q8act = (in_ch_i % 32 == 0) &&
                                   (std::getenv("NNTR_CONV_Q8ACT") != nullptr);
            // Pre-allocated Q8_0 scratch (no per-forward malloc).
            // Only consumed under ENABLE_FP16 (Q8_0 activation path); guard the
            // declaration too so non-FP16 builds don't trip
            // -Werror=unused-but-set-variable.
#ifdef ENABLE_FP16
            ::nntrainer::block_q8_0 *q8_buf = nullptr;
            if (can_q8act && wt_idx[ConvParams::q8act_scratch] !=
                               std::numeric_limits<unsigned int>::max()) {
              q8_buf = reinterpret_cast<::nntrainer::block_q8_0 *>(
                context.getTensor(wt_idx[ConvParams::q8act_scratch]).getData());
            }
#endif
            if (is_1x1_s1 && !weight_is_q8) {
#ifdef ENABLE_FP16
              if (can_q8act && q8_buf) {
                // Fused: quantize NHWC activation directly into the
                // block_q8_0x4 SMMLA layout and run the prepacked GEMM. This
                // replaces the two passes (quantize -> block_q8_0, then dot()
                // repacks to x4 with a per-call malloc) with one; output is
                // bit-identical.
                quantize_nhwc_q8_0x4_rows(in_sub.getData<_FP16>(), in_ch_i,
                                          owoh, q8_buf);
                Q8_0_Tensor::dot_prepacked_x4(
                  (unsigned)owoh, (unsigned)in_ch_i, filter_size, q8_buf,
                  filter_kernel.getData(), out_flat.getData<_FP16>(),
                  filter_size);
              } else {
#endif
                Tensor act = in_sub;
                act.reshape(TensorDim(
                  1, 1, owoh, in_dim.channel(),
                  {ml::train::TensorDim::Format::NCHW, in_sub.getDataType()}));
                act.dot(filter_kernel, out_flat, false, false);
#ifdef ENABLE_FP16
              }
#endif
            } else if (NNTR_HAS_Q4_0_INDIRECT_CONV) {
              ConvGatherParams geom;
              geom.in_ch = in_ch_i;
              geom.in_h = in_dim.height();
              geom.in_w = in_dim.width();
              geom.k_h = kernel_size[0].get();
              geom.k_w = kernel_size[1].get();
              geom.pad_t = padding[0];
              geom.pad_l = padding[2];
              geom.stride_h = stride[0].get();
              geom.stride_w = stride[1].get();
              geom.dil_h = dilation[0].get();
              geom.dil_w = dilation[1].get();
              geom.out_w = out_dim.width();
#ifdef ENABLE_FP16
              if (can_q8act && q8_buf) {
                const int n_sp = geom.in_h * geom.in_w;
                quantize_nhwc_q8_0_rows(in_sub.getData<_FP16>(), n_sp, in_ch_i,
                                        q8_buf);
                TensorDim q8dim({1, 1, (unsigned)n_sp, (unsigned)in_ch_i},
                                {ml::train::TensorDim::Format::NCHW,
                                 nntrainer::Tdatatype::Q8_0});
                Q8_0_Tensor q8act(q8dim, q8_buf);
                q8act.convQ4_0Indirect(filter_kernel, out_flat, geom);
              } else {
#endif
                geom.is_nhwc = true;
                in_sub.convQ4_0Indirect(filter_kernel, out_flat, geom);
#ifdef ENABLE_FP16
              }
#endif
            } else {
              throw std::runtime_error(
                "Fallback quantized NHWC conv is not supported (requires "
                "indirect conv on ARM).");
            }
          } else {
            // Q8_0 weights are only wired for the NHWC q8-activation indirect
            // path; the NCHW quant matmul/indirect fallbacks below assume a
            // Q4_0 weight operand, so reject Q8_0 here instead of silently
            // dispatching to the Q4_0 kernels.
            if (weight_is_q8) {
              throw std::runtime_error(
                "Q8_0 conv weights require the NHWC indirect path.");
            }
            // Quantized conv as matmul: act [OH*OW, CRS] . weight [CRS, out_ch]
            // -> [OH*OW, out_ch] -> out [out_ch, OH*OW]. CRS = in_ch*kh*kw.
            // NOTE: col must outlive `act` (act aliases col's storage); here
            // col is a view into the context-owned scratch, so its storage
            // outlives the loop iteration regardless.
            Tensor tmp = qgemm_scratch->getBatchSlice(b, 1);
            tmp.reshape(
              TensorDim(1, 1, owoh, filter_size, in_sub.getTensorType()));
            if (is_1x1_s1 && !weight_is_q8) {
              // 1x1 stride-1: im2col is an identity. The raw input is laid out
              // as [in_ch, OH*OW] (NCHW), so transpose to the act layout
              // [OH*OW, CRS] (CRS == in_ch here).
              in_sub.reshape({in_dim.channel(), owoh});
              Tensor act = in_sub.transpose("0:2:1");
              act.dot(filter_kernel, tmp, false, false);
            } else if (NNTR_HAS_Q4_0_INDIRECT_CONV) {
              // Quantized 3x3+ indirect: fold im2col gather into the q8_0
              // activation quantization so the activation matrix is never
              // materialized (the FP16 input is gathered on the fly and
              // quantized per tile inside the indirect GEMM). Output tmp is
              // FP16 [OH*OW, out_ch].
              ConvGatherParams geom;
              geom.in_ch = in_dim.channel();
              geom.in_h = in_dim.height();
              geom.in_w = in_dim.width();
              geom.k_h = kernel_size[0].get();
              geom.k_w = kernel_size[1].get();
              geom.pad_t = padding[0];
              geom.pad_l = padding[2];
              geom.stride_h = stride[0].get();
              geom.stride_w = stride[1].get();
              geom.dil_h = dilation[0].get();
              geom.dil_w = dilation[1].get();
              geom.out_w = out_dim.width();
              geom.is_nhwc = false;
              in_sub.convQ4_0Indirect(filter_kernel, tmp, geom);
            } else {
              // Fallback (no fused backend op): materialize im2col into the col
              // scratch, then the standard quant GEMM.
              // build the real kernel geometry (filter is stored as
              // [CRS,out_ch])
              TensorDim kdim(filter_size, in_dim.channel(),
                             kernel_size[0].get(), kernel_size[1].get(),
                             in_sub.getTensorType());
              Tensor col = col_scratch->getBatchSlice(b, 1);
              // im2col reshapes col in place to [OH*OW, CRS] (spatial-major),
              // which is ALREADY the act layout — no transpose (unlike the
              // raw-input 1x1 branch above). Transposing here gives [CRS,
              // OH*OW] and makes the GEMM emit CRS rows into the owoh-row
              // `tmp`, overflowing it whenever CRS > owoh (deep convs) -> heap
              // corruption.
              im2col(in_sub, kdim, padding, stride, dilation, col);
              col.dot(filter_kernel, tmp, false, false);
            }
            // [OH*OW, out_ch] -> [out_ch, OH*OW] written straight into the
            // (memory-planned) output. `tmp` is a separate scratch buffer and
            // `out` is a separate output view, so there is no aliasing.
            out.reshape({filter_size, owoh});
            tmp.transpose("0:2:1", out);
          }
        } else {
          Tensor result = col_scratch->getBatchSlice(b, 1);
          out.reshape({filter_size, owoh});
          im2col(in_sub, filter_dim, padding, stride, dilation, result);
          // filter kernel is (K, CRS), result is (CRS, OH*OW)
          if (out.getFormat() == ml::train::TensorDim::Format::NCHW) {
            filter_kernel.dot(result, out, false, true);
          } else {
            // NHWC: out's physical layout is [OH,OW,C] (channel innermost), so
            // a dot writing [out_ch, OH*OW] NCHW-order would land in the wrong
            // physical cells. Compute into a plain NCHW-order buffer (format
            // NCHW so dot writes channel-major), then scatter channel c of
            // spatial r to out[r*out_ch + c] (NHWC physical).
            // Compute into a plain NCHW-order buffer (same result feeding as
            // the NCHW path above — do NOT reshape result, im2col already laid
            // it out as the dot expects). nchw_out is format NCHW so dot writes
            // channel-major [out_ch, OH*OW]; then scatter channel c of spatial
            // r to out[r*out_ch + c] (NHWC physical).
            auto nchw_type = out.getTensorType();
            nchw_type.format = ml::train::TensorDim::Format::NCHW;
            // nchw_out holds the GEMM result [out_ch, OH*OW] in channel-major
            // (NCHW) order. Shape it [1, 1, out_ch, OH*OW] so width()==OH*OW is
            // the row stride the GEMM writes with (ldc); a [1,out_ch,OH*OW,1]
            // shape would give width()==1 and stride the output wrong.
            TensorDim nchw_dim({1, 1, filter_size, owoh}, nchw_type);
            Tensor nchw_out(nchw_dim, true);
            // filter_kernel (weight [out_ch, CRS]) and result (im2col columns
            // [OH*OW, CRS]) are plain 2D matmul matrices whose image format is
            // irrelevant to this GEMM. dot() derives the contraction axis from
            // the tensor format: their inherited NHWC tag makes it contract
            // over channel() (==1) instead of width() (==CRS) -> zero output.
            // Re-map the same bytes as NCHW-format views (no copy) so the GEMM
            // contracts over CRS==width(). getSharedDataTensor can't relabel
            // format (it enforces a match), so use Tensor::Map.
            auto fnchw_type = filter_kernel.getTensorType();
            fnchw_type.format = ml::train::TensorDim::Format::NCHW;
            auto cnchw_type = result.getTensorType();
            cnchw_type.format = ml::train::TensorDim::Format::NCHW;
            TensorDim fdim_nchw(filter_kernel.batch(), filter_kernel.channel(),
                                filter_kernel.height(), filter_kernel.width(),
                                fnchw_type);
            TensorDim cdim_nchw(result.batch(), result.channel(),
                                result.height(), result.width(), cnchw_type);
            Tensor filt_nchw = Tensor::Map<float>(
              filter_kernel.getData<float>(), filter_kernel.bytes(), fdim_nchw);
            if (out.getDataType() == nntrainer::Tdatatype::FP32) {
              Tensor col_nchw = Tensor::Map<float>(result.getData<float>(),
                                                   result.bytes(), cdim_nchw);
              filt_nchw.dot(col_nchw, nchw_out, false, true);
            }
#ifdef ENABLE_FP16
            else {
              Tensor col_nchw = Tensor::Map<_FP16>(result.getData<_FP16>(),
                                                   result.bytes(), cdim_nchw);
              filt_nchw.dot(col_nchw, nchw_out, false, true);
            }
#endif
            if (out.getDataType() == nntrainer::Tdatatype::FP32) {
              const float *s = nchw_out.getData<float>();
              float *d = out.getData<float>();
              for (unsigned int oc = 0; oc < filter_size; ++oc)
                for (unsigned int r = 0; r < owoh; ++r)
                  d[r * filter_size + oc] = s[oc * owoh + r];
            }
#ifdef ENABLE_FP16
            else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
              const _FP16 *s = nchw_out.getData<_FP16>();
              _FP16 *d = out.getData<_FP16>();
              for (unsigned int oc = 0; oc < filter_size; ++oc)
                for (unsigned int r = 0; r < owoh; ++r)
                  d[r * filter_size + oc] = s[oc * owoh + r];
            }
#endif
          }
        }
      }
    };

    auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

    if (workers.getNumWorkers() > 1) {
      workers.run();
    } else {
      forwarding_job(0, in_dim.batch(), 0, nullptr);
    }

    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim);
    }
  } else {
    // Grouped convolution: split channels into `groups` independent groups.
    const unsigned int ocg = filter_size / groups;      // out ch per group
    const unsigned int icg = in_dim.channel() / groups; // in ch per group
    const unsigned int fh = filter_dim.height(), fw = filter_dim.width();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const unsigned int ihw = in_dim.height() * in_dim.width();
    TensorDim fdim_g(ocg, icg, fh, fw, filter_dim.getTensorType());

    const bool is_true_depthwise = ocg == 1 && icg == 1;
    const bool nhwc_depthwise =
      is_true_depthwise &&
      in_dim.getFormat() == ml::train::TensorDim::Format::NHWC;

    if (nhwc_depthwise) {
      // NHWC depthwise convolution: input/output are channel-last, but the
      // backend depthwise ops (depthwise_conv2d_fp32/fp16) and the generic
      // grouped fallback below are all NCHW-planar (channel-major). Handle the
      // channel-last layout inline here. Filter keeps the standard
      // [out_ch, 1, fh, fw] layout, so channel c tap (kh,kw) is at
      // filt[c*fh*fw + kh*fw + kw]. Accumulate in float for parity with the
      // NCHW paths regardless of activation precision.
      const unsigned int B = in_dim.batch();
      const unsigned int C = filter_size; // channels (== groups)
      const int IH = static_cast<int>(in_dim.height());
      const int IW = static_cast<int>(in_dim.width());
      const int OH = static_cast<int>(out_dim.height());
      const int OW = static_cast<int>(out_dim.width());
      const int sh = static_cast<int>(stride[0].get());
      const int sw = static_cast<int>(stride[1].get());
      const int ph = static_cast<int>(padding[0]);
      const int pw = static_cast<int>(padding[2]);
      const int dh = static_cast<int>(dilation[0].get());
      const int dw_ = static_cast<int>(dilation[1].get());
      const unsigned int fhfw = fh * fw;
      const float *filt = filter_kernel.getData<float>();

      auto run = [&]<typename T>(const T *in, T *out) {
        std::vector<float> acc(C);
        for (unsigned int b = 0; b < B; ++b) {
          const T *inb = in + static_cast<size_t>(b) * IH * IW * C;
          T *outb = out + static_cast<size_t>(b) * OH * OW * C;
          for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
              std::fill(acc.begin(), acc.end(), 0.0f);
              for (unsigned int kh = 0; kh < fh; ++kh) {
                const int ih = oh * sh - ph + static_cast<int>(kh) * dh;
                if (ih < 0 || ih >= IH)
                  continue;
                for (unsigned int kw = 0; kw < fw; ++kw) {
                  const int iw = ow * sw - pw + static_cast<int>(kw) * dw_;
                  if (iw < 0 || iw >= IW)
                    continue;
                  const T *id = inb + (static_cast<size_t>(ih) * IW + iw) * C;
                  const float *fk = filt + kh * fw + kw;
                  for (unsigned int c = 0; c < C; ++c)
                    acc[c] += static_cast<float>(id[c]) *
                              fk[static_cast<size_t>(c) * fhfw];
                }
              }
              T *od = outb + (static_cast<size_t>(oh) * OW + ow) * C;
              for (unsigned int c = 0; c < C; ++c)
                od[c] = static_cast<T>(acc[c]);
            }
          }
        }
      };

      if (in_dim.getDataType() == nntrainer::Tdatatype::FP32)
        run(input_.getData<float>(), hidden_.getData<float>());
#ifdef ENABLE_FP16
      else
        run(input_.getData<_FP16>(), hidden_.getData<_FP16>());
#endif
    } else if (is_true_depthwise &&
               in_dim.getDataType() == nntrainer::Tdatatype::FP32) {
      // True depthwise (groups == channels): delegate to the CPU backend op so
      // the optimised kernel lives in the backend, not in the layer.
      nntrainer::getComputeOps()->depthwise_conv2d_fp32(
        input_.getData<float>(), filter_kernel.getData<float>(),
        hidden_.getData<float>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#ifdef ENABLE_FP16
    } else if (is_true_depthwise &&
               in_dim.getDataType() == nntrainer::Tdatatype::FP16 &&
               hidden_.getDataType() == nntrainer::Tdatatype::FP16 &&
               filter_kernel.getDataType() == nntrainer::Tdatatype::FP32) {
      // FP16-activation depthwise: weights are never Q4_0 for groups>1 and stay
      // FP32 (BN-folded), so this is FP16 input/output x FP32 kernel. Keep it
      // on the tight channel-parallel direct-loop kernel instead of falling
      // into the generic grouped else-branch (per-channel im2col + FP16 GEMV),
      // which was ~2x slower for YOLOv11's detect-head depthwise convs.
      nntrainer::getComputeOps()->depthwise_conv2d_fp16(
        input_.getData<_FP16>(), filter_kernel.getData<float>(),
        hidden_.getData<_FP16>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#endif
    } else {
      // getSharedDataTensor()/reshape() adopt the *passed* TensorDim's dtype
      // (TensorBase::getSharedDataTensor: ret->dim = dim_). A bare {..} dim
      // literal defaults to FP32, so on an FP16 activation graph every sub-view
      // below would silently relabel FP16-backed storage as FP32 -- im2col
      // would gather at the wrong precision and the dot would write 4-byte
      // floats into 2-byte half storage (overflow / garbage). Carry each
      // parent's real dtype onto its view. (For an all-FP32 graph these are
      // no-ops, so the historical path is unchanged.)
      const auto in_dt = input_.getDataType();
      const auto filt_dt = filter_kernel.getDataType();
      const auto out_dt = hidden_.getDataType();
      for (unsigned int b = 0; b < in_dim.batch(); ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        TensorDim out_rdim({filter_size, owoh});
        out_rdim.setDataType(out_dt);
        out.reshape(out_rdim);
        Tensor in_sub = input_.getBatchSlice(b, 1);
        TensorDim col_dim = calcCol2ImOutputDim(out_dim, fdim_g);
        col_dim.setDataType(in_dt);
        Tensor result = Tensor(col_dim);
        for (unsigned int g = 0; g < groups; ++g) {
          TensorDim ing_dim({1, icg, in_dim.height(), in_dim.width()});
          ing_dim.setDataType(in_dt);
          Tensor in_g =
            in_sub.getSharedDataTensor(ing_dim, (size_t)g * icg * ihw);
          TensorDim filtg_dim({ocg, (size_t)icg * fh * fw});
          filtg_dim.setDataType(filt_dt);
          Tensor filt_g = filter_kernel.getSharedDataTensor(
            filtg_dim, (size_t)g * ocg * icg * fh * fw);
          TensorDim outg_dim({ocg, owoh});
          outg_dim.setDataType(out_dt);
          Tensor out_g =
            out.getSharedDataTensor(outg_dim, (size_t)g * ocg * owoh);
          result.setZero();
          im2col(in_g, fdim_g, padding, stride, dilation, result);
          filt_g.dot(result, out_g, false, true);
        }
        result.deallocate();
      }
    }
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
    if (hidden_.getFormat() == ml::train::TensorDim::Format::NCHW) {
      status = hidden_.add_i(bias_kernel);
      if (status != ML_ERROR_NONE) {
        throw std::invalid_argument("[Conv2D] adding bias failed");
      }
    } else {
      // NHWC: channel is innermost. bias [out_ch] must be added per (n,h,w,c).
      // add_i assumes NCHW channel-major broadcast, so do it inline.
      const unsigned int C = out_dim.channel();
      const unsigned int HW = out_dim.height() * out_dim.width();
      const unsigned int B = out_dim.batch();
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP32) {
        float *d = hidden_.getData<float>();
        const float *bias = bias_kernel.getData<float>();
        for (unsigned int b = 0; b < B; ++b)
          for (unsigned int p = 0; p < HW; ++p)
            for (unsigned int c = 0; c < C; ++c)
              d[((size_t)b * HW + p) * C + c] += bias[c];
      }
#ifdef ENABLE_FP16
      else if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
        _FP16 *d = hidden_.getData<_FP16>();
        const _FP16 *bias = bias_kernel.getData<_FP16>();
        for (unsigned int b = 0; b < B; ++b)
          for (unsigned int p = 0; p < HW; ++p)
            for (unsigned int c = 0; c < C; ++c)
              d[((size_t)b * HW + p) * C + c] += bias[c];
      }
#endif
    }
  }

  // Fused activation epilogue. When the graph sets activation=swish on the
  // conv, apply SiLU in-place on the freshly written output instead of
  // materializing a separate Activation layer (which would read the conv
  // output back from memory and write a second full tensor). Only SiLU is
  // fused here (YOLOv11's conv activation); any other activation type is left
  // to a dedicated Activation layer in the graph.
  if (auto &act = std::get<props::FusedActivation>(conv_props);
      !act.empty() && act.get() == ActivationType::ACT_SWISH) {
    const size_t n = hidden_.size();
#ifdef ENABLE_FP16
    if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
      convApplySwishInplace(hidden_.getData<_FP16>(), n);
    } else
#endif
    {
      convApplySwishInplace(hidden_.getData<float>(), n);
    }
  }

  if (std::getenv("NNTR_CONV_PROBE") && context.getName() == "conv0/conv") {
    std::cerr << "[CONV_PROBE] " << context.getName()
              << " format=" << (int)hidden_.getFormat()
              << " C=" << out_dim.channel() << " H=" << out_dim.height()
              << " W=" << out_dim.width() << std::endl;
    const float *p = hidden_.getData<float>();
    for (int i = 0; i < 8; ++i)
      std::cerr << "  [" << i << "]=" << p[i] << std::endl;
  }
}

void Conv2DLayer::calcDerivative(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(
        {filter_size, derivative.width() * derivative.height()});
      // filter_kernel is (K, CRS), deriv_sub is (K, OH*OW), result is (CRS,
      // OH*OW)
      filter_kernel.dot(deriv_sub, result, true, false);
      col2im(result, filter_dim, padding, stride, dilation, in_deriv_sub);
      // in_derv_sub is (C,H,W)
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DLayer::calcGradient(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());
  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  if (workers.getNumWorkers() > 1) {

    TensorDim delK_ext = filter_dim_squeezed;
    delK_ext.batch(input_.batch());

    Tensor delK_par = Tensor(delK_ext);
    delK_par.setZero();

    auto calc_grad_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                             void *user_data) {
      Tensor result =
        Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
      result.setZero();
      for (unsigned int b = s; b < e; ++b) {
        Tensor deriv_sub = derivative.getBatchSlice(b, 1);
        Tensor delK_sub = delK_par.getBatchSlice(b, 1);
        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1);

        /**
         * @todo this result can be cached from the forward iteration at the
         * expense of memory. In this case, memory of im2col_result must be
         * saved for the whole batch. try this while benchmarking.
         */
        // deriv_sub is (K, OH*OW) and result is (CRS, OH*OW)
        im2col(in_sub, filter_dim, padding, stride, dilation, result);
        deriv_sub.dot(result, delK_sub, false, false);
      }
      result.deallocate();
    };

    workers.setCallback(calc_grad_job, nullptr);

    workers.run();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor delK_sub = delK_par.getBatchSlice(b, 1);
      delK.add_i(delK_sub);
    }

  } else {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
    result.setZero();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(out_dim_squeezed);

      Tensor in_sub = input_.getBatchSlice(b, 1);

      /**
       * @todo this result can be cached from the forward iteration at the
       * expense of memory. In this case, memory of im2col_result must be saved
       * for the whole batch. try this while benchmarking.
       */
      im2col(in_sub, filter_dim, padding, stride, dilation, result);
      deriv_sub.dot(result, delK, false, false, b == 0 ? 0.0f : 1.0f);
    }
    result.deallocate();
  }
  delK.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
    delBias.setZero();
    derivative.sum({0, 2, 3}, delBias);
  }
}

void Conv2DLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  // The forward scratch buffers (im2col column buffer and quantized-GEMM
  // output) are requested in finalize() sized to the batch present at init.
  // When the runtime batch changes the framework resizes inputs/outputs but
  // not these layer-private scratch tensors, so rebatch them here — otherwise
  // forwarding()'s getBatchSlice(b, 1) for b >= the init batch reads past the
  // planned storage and aborts ("shared tensor bigger than tensor memory").
  if (wt_idx[ConvParams::im2col_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::im2col_scratch], batch);
  if (wt_idx[ConvParams::qgemm_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::qgemm_scratch], batch);
}

void Conv2DLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv2DLayer::save(std::ofstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, ml::train::TensorDim::DataType dtype,
                       ml::train::ISA target_isa) const {
  // Optimizer-variable save (training only) has no conv-specific layout, so
  // defer to the base implementation.
  if (opt_var) {
    Layer::save(file, run_context, opt_var, mode, trainable, dtype, target_isa);
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;

    auto &weight = run_context.getWeight(i);

    // No conversion requested, or already the target dtype: save as-is.
    if (dtype == TensorDim::DataType::NONE || weight.getDataType() == dtype) {
      weight.save(file);
      continue;
    }

    NNTR_THROW_IF(dtype != TensorDim::DataType::Q4_0, std::runtime_error)
      << "[Conv2D] save: unsupported quantization dtype";
    NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                  std::runtime_error)
      << "[Conv2D] Q4_0 save only supports FP32 source weight.";

    // A conv FP32 filter is [out_ch, in_ch, kh, kw] in NCHW, i.e. already
    // row-major [out_ch, CRS] (CRS = in_ch*kh*kw) = [N rows, K cols]. This is
    // exactly the layout quantize_q4_0 consumes (N rows of K), so no transpose
    // is needed (the FC path in the base class transposes because its weight is
    // stored [K, N]). The bias and any non-matmul weight (CRS == 1 or
    // out_ch == 1) are kept FP32.
    const TensorDim dim = weight.getDim();
    const unsigned int out_ch = dim.batch();
    const unsigned int CRS = dim.channel() * dim.height() * dim.width();

    if (out_ch <= 1 || CRS <= 1 || out_ch % 32 != 0 || CRS % 32 != 0) {
      // Not Q4_0-eligible (bias, or block-misaligned): keep FP32 so the saved
      // tensor still matches what the runtime layer allocates for it.
      weight.save(file);
      continue;
    }

    // [1, 1, K=CRS, N=out_ch] is the matmul-weight shape the quantized conv
    // consumes at load (Conv2DLayer::finalize builds the same shape).
    Tensor quant_weight(1, 1, CRS, out_ch, {Tformat::NCHW, dtype});
    std::vector<char> tmp(quant_weight.size());

    quantize_q4_0(weight.getData<float>(), tmp.data(), out_ch, CRS, nullptr);
    repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                quant_weight.size(), out_ch, CRS, target_isa);
    quant_weight.save(file);
  }
}

void Conv2DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */
