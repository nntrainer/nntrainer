// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_utils.cpp
 * @date	15 October 2025
 * @brief	This is Int4Utils class for utils for INT4 quantization format.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Grzegorz Kisala <gkisala@gmail.com>
 * @bug		No known bugs
 */

#include "int4_utils.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>

#include "cpu_backend.h"
#include "fp16.h"
#include "nntrainer_error.h"
#include "quantizer.h"
#include "util_func.h"

namespace nntrainer {

float Int4Utils::computeScaleForGroup(const float *group_weights,
                                      const size_t group_size) {
  auto max_absolute_weight = 0.0f;

  for (size_t i = 0; i < group_size; ++i) {
    auto weight = group_weights[i];

    NNTR_THROW_IF(!std::isfinite(weight), std::invalid_argument)
      << "Weight is not finite value";

    const auto absolute_weight = std::abs(weight);

    if (absolute_weight > max_absolute_weight) {
      max_absolute_weight = absolute_weight;
    }
  }

  auto group_scale =
    (max_absolute_weight == 0.0f) ? 1.0f : (max_absolute_weight / 7.0f);

  NNTR_THROW_IF(!std::isfinite(group_scale), std::invalid_argument)
    << "Scale is not finite value";

  return group_scale;
}

void Int4Utils::computeScales(const float *weights, const size_t rows_count,
                              const size_t columns_count,
                              const size_t group_size,
                              std::vector<float> &scales) {
  // NNTR_THROW_IF(columns_count % group_size, std::invalid_argument)
  //   << "Columns size not divisible by group size";
  NNTR_THROW_IF(columns_count % 4, std::invalid_argument)
    << "Columns size not divisible by 4";

  const auto full_groups_per_row = columns_count / group_size;
  const auto last_group_size = columns_count % group_size;
  const auto padded_groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  scales.resize(static_cast<size_t>(rows_count_pad) * padded_groups_per_row,
                1.0f);

  for (size_t row_id = 0; row_id < rows_count; ++row_id) {
    const auto *weights_row = weights + (row_id * columns_count);

    for (size_t group_id = 0; group_id < full_groups_per_row; ++group_id) {
      const auto *weights_group = weights_row + (group_id * group_size);
      scales[(group_id * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, group_size);
    }

    // Compute scale for the last padded group
    if (last_group_size > 0) {
      const auto *weights_group =
        weights_row + (full_groups_per_row * group_size);
      scales[(full_groups_per_row * rows_count_pad) + row_id] =
        computeScaleForGroup(weights_group, last_group_size);
    }
  }
}

uint8_t Int4Utils::pack(const float *weights, const float *scales,
                        const size_t row_id, const size_t column_id,
                        const size_t groups_per_row, const size_t group_size,
                        const size_t rows_count, const size_t columns_count) {
  {
    const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
    const float scale =
      scales[row_id + ((column_id / group_size) * rows_count_pad)];
    const float weight = weights[(row_id * columns_count) + column_id];
    return quantizeToInt4(weight, scale);
  }
}

void Int4Utils::quantizeAndRepack(const float *weights, const size_t rows_count,
                                  const size_t columns_count,
                                  const size_t group_size,
                                  std::vector<uint8_t> &out_weights,
                                  std::vector<uint16_t> &out_scales) {
  NNTR_THROW_IF(!weights, std::invalid_argument) << "Weight cannot be null";

  NNTR_THROW_IF((rows_count <= 0), std::invalid_argument)
    << "Rows count needs to be greater than 0";

  NNTR_THROW_IF((columns_count <= 0), std::invalid_argument)
    << "Columns count needs to be greater than 0";

  NNTR_THROW_IF((!(group_size == 32 || group_size == 64 || group_size == 128)),
                std::invalid_argument)
    << "Group size must be 32/64/128";

  std::vector<float> scales_fp32;
  computeScales(weights, rows_count, columns_count, group_size, scales_fp32);

  out_scales.resize(scales_fp32.size());
  for (size_t scale_id = 0; scale_id < scales_fp32.size(); ++scale_id) {
    out_scales[scale_id] = compute_fp32_to_fp16(scales_fp32[scale_id]);
  }

  NNTR_THROW_IF(columns_count % COLUMN_BLOCK_SIZE, std::invalid_argument)
    << "Columns size not divisible by column block size";

  // Prepare output buffer in OS_IS_YX_OSV32_ISV2 layout
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);
  const auto rows_count_pad = row_blocks_count * ROW_BLOCK_SIZE;

  out_weights.resize((rows_count_pad * columns_count_pad) / 2, 0);

  size_t out_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            lo = pack(weights, scales_fp32.data(), row_id_absolute,
                      column_id_absolute_lo, groups_per_row, group_size,
                      rows_count, columns_count);

            const auto column_id_absolute_hi = column_id_absolute_lo + 1;
            if (column_id_absolute_hi < columns_count) {
              hi = pack(weights, scales_fp32.data(), row_id_absolute,
                        column_id_absolute_hi, groups_per_row, group_size,
                        rows_count, columns_count);
            }
          }
        }

        out_weights[out_idx++] = uint8_t((hi << 4) | lo);
      }
    }
  }
}

uint8_t Int4Utils::quantizeToInt4(const float weight, const float scale) {
  auto div = std::nearbyintf(weight / scale);

  if (std::isnan(div)) {
    div = 0.0f;
  }

  div = std::clamp(div, -8.0f, 7.0f);
  int quantized = (int)div;
  return uint8_t(quantized & 0xF);
}

int Int4Utils::convertInt4ToInt(const uint8_t int4_value) {
  static int lookup[] = {0,  1,  2,  3,  4,  5,  6,  7,
                         -8, -7, -6, -5, -4, -3, -2, -1};

  return lookup[int4_value];
}

size_t Int4Utils::kaiNibblePayloadBytes(size_t rows_count,
                                        size_t columns_count) {
  const size_t k_internal =
    ((columns_count + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (rows_count + KAI_NR - 1) / KAI_NR;
  return super_row_count * KAI_NR * (k_internal / 2);
}

size_t Int4Utils::kaiRhsPackedBytes(size_t rows_count, size_t columns_count) {
  // Match the generic-packer stride (which is what the matmul micro-kernel
  // actually walks per column tile): nr * (k_internal/2 + sums(4) +
  // scales(4) + bias(4)) per super-row.
  const size_t k_internal =
    ((columns_count + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (rows_count + KAI_NR - 1) / KAI_NR;
  const size_t trailer_bytes_per_super_row = KAI_NR * (4 + 4 + 4);
  return super_row_count *
         (KAI_NR * (k_internal / 2) + trailer_bytes_per_super_row);
}

void Int4Utils::assembleKaiRhsPacked(const uint8_t *section_a,
                                     const uint16_t *fp16_scales,
                                     size_t rows_count, size_t columns_count,
                                     std::vector<uint8_t> &out_kai_packed) {
  const size_t k_internal =
    ((columns_count + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (rows_count + KAI_NR - 1) / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (k_internal / 2);
  const size_t trailer_bytes_per_super_row = KAI_NR * (4 + 4 + 4);
  const size_t super_row_stride =
    nibble_bytes_per_super_row + trailer_bytes_per_super_row;

  out_kai_packed.assign(super_row_count * super_row_stride, 0u);

  const size_t block_length_in_bytes = KAI_KR / KAI_SR; // = 8

  for (size_t sr_idx = 0; sr_idx < super_row_count; ++sr_idx) {
    uint8_t *dst_row = out_kai_packed.data() + sr_idx * super_row_stride;
    const uint8_t *src_row = section_a + sr_idx * nibble_bytes_per_super_row;

    // Nibbles are byte-identical to Section A on disk — just copy them.
    std::memcpy(dst_row, src_row, nibble_bytes_per_super_row);

    // Per-channel sums = sum_k int4[n][k] * 16. Decode each Section A byte
    // back to two int4 lanes via XOR 0x88 -> uint4 -> int4 = uint4 - 8.
    int32_t sums[KAI_NR] = {0, 0, 0, 0};
    for (size_t dst_byte_idx = 0; dst_byte_idx < nibble_bytes_per_super_row;
         ++dst_byte_idx) {
      const size_t block_idx = dst_byte_idx / block_length_in_bytes;
      const size_t nr_idx = block_idx % KAI_NR;

      const uint8_t stored = src_row[dst_byte_idx] ^ 0x88;
      const int u_lo = stored & 0x0F;
      const int u_hi = (stored >> 4) & 0x0F;
      // Both nibbles belong to the same n0 (= sr_idx * KAI_NR + nr_idx) —
      // the K interleave shuffles k0/k1 within the same channel.
      sums[nr_idx] += (u_lo - 8) + (u_hi - 8);
    }

    int32_t *sums_dst = (int32_t *)(dst_row + nibble_bytes_per_super_row);
    for (size_t i = 0; i < KAI_NR; ++i) {
      sums_dst[i] = sums[i] * 16;
    }

    // Scales: per-channel fp16 -> fp32 -> baked × 0.0625 (matches KAI
    // packer line 212).
    float *scales_dst = (float *)(dst_row + nibble_bytes_per_super_row +
                                  KAI_NR * sizeof(int32_t));
    for (size_t i = 0; i < KAI_NR; ++i) {
      const size_t n_idx = std::min(sr_idx * KAI_NR + i, rows_count - 1);
      const float s = compute_fp16_to_fp32(fp16_scales[n_idx]);
      scales_dst[i] = s * 0.0625F;
    }

    // Bias: always zero for FC int4 weights — the bias tensor (if any) is
    // a separate FP16 weight that the layer adds outside this matmul.
    // 4 floats already memset to 0 by assign(.., 0u).
  }
}

size_t Int4Utils::plainNibbleBytes(size_t rows_count, size_t columns_count) {
  return rows_count * ((columns_count + 1) / 2);
}

size_t Int4Utils::plainScalesOffsetBytes(size_t rows_count,
                                         size_t columns_count) {
  // Mirrors the PR#3978 writer/reader expression `N * (K + 1) / 2` exactly
  // (left-to-right integer math) — for even K this lands N/2 bytes past the
  // nibble end, leaving a zero gap that is part of the on-disk format.
  return rows_count * (columns_count + 1) / 2;
}

size_t Int4Utils::plainRecordPayloadBytes(size_t rows_count,
                                          size_t columns_count) {
  // kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0 for the PR
  // writer's idx_variant=8 kernel: nr=8, k rounded up to 32, and a 12-byte
  // per-channel trailer slot (fp32 sum + fp32 scale + fp32 bias).
  const size_t k_internal =
    ((columns_count + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t n_rounded =
    ((rows_count + PLAIN_KAI_NR - 1) / PLAIN_KAI_NR) * PLAIN_KAI_NR;
  return n_rounded * (k_internal / 2 + 12);
}

void Int4Utils::quantizePlain(const float *weights, const size_t rows_count,
                              const size_t columns_count,
                              std::vector<uint8_t> &out_plain_nibbles,
                              std::vector<uint16_t> &out_scales, bool range15) {
  // Per-channel scales. Two formulas:
  //  - absmax/7 (default): the historical Section A writer formula — keeps
  //    plain nibbles bit-identical to existing qscheme-0x06 bins.
  //  - range15: the PR#3978 / KAI quantizer (__fallback_quant_nxk_qs4cx_f32)
  //    mirrored exactly — scale0 = 15 / (max(0,max)-min(0,min)), value =
  //    clamp(round(w * scale0)), stored scale = 1/scale0. Uses the int4 grid
  //    far better on asymmetric channels; quality-critical for small models.
  std::vector<float> scales_fp32(rows_count);
  std::vector<float> mul_fp32(range15 ? rows_count : 0);
  out_scales.resize(rows_count);
  for (size_t n = 0; n < rows_count; ++n) {
    if (range15) {
      float max0 = -FLT_MAX, min0 = FLT_MAX;
      const float *row = weights + n * columns_count;
      for (size_t k = 0; k < columns_count; ++k) {
        max0 = std::max(row[k], max0);
        min0 = std::min(row[k], min0);
      }
      const float rmin0 = std::min(0.0f, min0);
      const float rmax0 = std::max(0.0f, max0);
      const float scale0 =
        rmin0 == rmax0 ? 1.f : (7.0f - (-8.0f)) / (rmax0 - rmin0);
      mul_fp32[n] = scale0;
      scales_fp32[n] = scale0 != 0.0f ? 1.0f / scale0 : 0.0f;
    } else {
      scales_fp32[n] =
        computeScaleForGroup(weights + n * columns_count, columns_count);
    }
    out_scales[n] = compute_fp32_to_fp16(scales_fp32[n]);
  }

  // Build raw nibble matrix in N x roundup(K,2)/2 bytes. Each nibble
  // holds uint4 = int4 + 8 (the rhs_zero_point=8 input form expected
  // by kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0).
  const size_t rhs_row_bytes = (columns_count + 1) / 2;
  out_plain_nibbles.assign(rows_count * rhs_row_bytes, 0u);
  auto quantize_one = [&](float w, size_t n) -> uint8_t {
    if (range15) {
      // PR expression: round (half away from zero) of w * scale0.
      int32_t v = (int32_t)std::round(w * mul_fp32[n]);
      v = std::max(v, -8);
      v = std::min(v, 7);
      return (uint8_t)(v & 0xF);
    }
    // absmax/7: nearbyint of w / scale — the historical writer expression.
    return quantizeToInt4(w, scales_fp32[n]);
  };
  for (size_t n = 0; n < rows_count; ++n) {
    for (size_t k = 0; k < columns_count; k += 2) {
      const uint8_t u0 = quantize_one(weights[n * columns_count + k], n);
      uint8_t byte = (uint8_t)((u0 + 8) & 0x0F);
      if (k + 1 < columns_count) {
        const uint8_t u1 = quantize_one(weights[n * columns_count + k + 1], n);
        byte |= (uint8_t)(((u1 + 8) & 0x0F) << 4);
      } else {
        byte |= (uint8_t)(8u << 4); // pad nibble = uint4(0)
      }
      out_plain_nibbles[n * rhs_row_bytes + (k / 2)] = byte;
    }
  }
}

void Int4Utils::packPlainToSectionA(const uint8_t *plain_nibbles,
                                    size_t rows_count, size_t columns_count,
                                    uint8_t *out_section_a) {
  // Permute into KAI Section A super-row form (no trailer). Mirrors the
  // rhs_zero_point=8 path of kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0 from
  // nntrainer/tensor/cpu_backend/arm/kai/pack/, with the trailer (sums,
  // scales, bias) elided — those are reassembled at CPU load time on
  // ARM, and ignored entirely on x86/GPU paths.
  const uint8_t *raw_nibbles = plain_nibbles;
  const size_t rhs_row_bytes = (columns_count + 1) / 2;
  const size_t k_internal =
    ((columns_count + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (rows_count + KAI_NR - 1) / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (k_internal / 2);
  const size_t dst_num_bytes_per_row = nibble_bytes_per_super_row;
  const size_t block_length_in_bytes = KAI_KR / KAI_SR; // = 8
  const uint8_t pad_byte = (uint8_t)(8u | (8u << 4));   // uint4(0)|uint4(0)

  for (size_t dst_row_idx = 0; dst_row_idx < super_row_count; ++dst_row_idx) {
    uint8_t *dst_row = out_section_a + dst_row_idx * nibble_bytes_per_super_row;

    for (size_t dst_byte_idx = 0; dst_byte_idx < dst_num_bytes_per_row;
         ++dst_byte_idx) {
      const size_t block_idx = dst_byte_idx / block_length_in_bytes;
      const size_t block_byte_idx = dst_byte_idx % block_length_in_bytes;
      const size_t super_block_idx = block_idx / KAI_NR;
      const size_t nr_idx = block_idx % KAI_NR;

      const size_t k_adjustment =
        ((block_byte_idx + super_block_idx * block_length_in_bytes) /
         KAI_K_INTERLEAVE) *
        KAI_K_INTERLEAVE;
      const size_t k0_idx =
        block_byte_idx + super_block_idx * block_length_in_bytes + k_adjustment;
      const size_t k1_idx = k0_idx + KAI_K_INTERLEAVE;
      const size_t n0_idx = dst_row_idx * KAI_NR + nr_idx;
      const size_t n0_valid_idx =
        (n0_idx < rows_count) ? n0_idx : (rows_count - 1);

      uint8_t byte0 = pad_byte;
      uint8_t byte1 = pad_byte;
      if (k0_idx < columns_count) {
        byte0 = raw_nibbles[n0_valid_idx * rhs_row_bytes + (k0_idx / 2)];
      }
      if (k1_idx < columns_count) {
        byte1 = raw_nibbles[n0_valid_idx * rhs_row_bytes + (k1_idx / 2)];
      }
      const size_t shift_right_x0 = (k0_idx % 2) * 4;
      const size_t shift_right_x1 = (k1_idx % 2) * 4;
      const uint8_t src_x0_lo = (uint8_t)((byte0 >> shift_right_x0) & 0x0F);
      const uint8_t src_x0_hi = (uint8_t)((byte1 >> shift_right_x1) & 0x0F);
      const uint8_t dst_qs0 = (uint8_t)(src_x0_lo | (src_x0_hi << 4));
      *dst_row = (uint8_t)(dst_qs0 ^ 0x88);
      ++dst_row;
    }
  }
}

void Int4Utils::sectionAToPlain(const uint8_t *section_a, size_t rows_count,
                                size_t columns_count,
                                uint8_t *out_plain_nibbles) {
  // Exact inverse of packPlainToSectionA: walk the same Section A super-row /
  // block iteration, un-XOR 0x88, and scatter each recovered nibble back to its
  // plain [n][k] position. Padding entries (n >= rows_count, k >=
  // columns_count) come from clamped / pad reads in the forward and carry no
  // unique data, so they are skipped. A valid plain nibble may be produced by
  // more than one Section A byte (the forward reads it as both a k0 and a k1);
  // every such copy holds the same value, so writing with |= over a zeroed
  // buffer is idempotent and reproduces the plain layout byte-for-byte.
  const size_t rhs_row_bytes = (columns_count + 1) / 2;
  std::memset(out_plain_nibbles, 0, rows_count * rhs_row_bytes);

  const size_t k_internal =
    ((columns_count + KAI_K_PAD_MULTIPLE - 1) / KAI_K_PAD_MULTIPLE) *
    KAI_K_PAD_MULTIPLE;
  const size_t super_row_count = (rows_count + KAI_NR - 1) / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (k_internal / 2);
  const size_t dst_num_bytes_per_row = nibble_bytes_per_super_row;
  const size_t block_length_in_bytes = KAI_KR / KAI_SR; // = 8

  for (size_t dst_row_idx = 0; dst_row_idx < super_row_count; ++dst_row_idx) {
    const uint8_t *src_row =
      section_a + dst_row_idx * nibble_bytes_per_super_row;

    for (size_t dst_byte_idx = 0; dst_byte_idx < dst_num_bytes_per_row;
         ++dst_byte_idx) {
      const size_t block_idx = dst_byte_idx / block_length_in_bytes;
      const size_t block_byte_idx = dst_byte_idx % block_length_in_bytes;
      const size_t super_block_idx = block_idx / KAI_NR;
      const size_t nr_idx = block_idx % KAI_NR;

      const size_t k_adjustment =
        ((block_byte_idx + super_block_idx * block_length_in_bytes) /
         KAI_K_INTERLEAVE) *
        KAI_K_INTERLEAVE;
      const size_t k0_idx =
        block_byte_idx + super_block_idx * block_length_in_bytes + k_adjustment;
      const size_t k1_idx = k0_idx + KAI_K_INTERLEAVE;
      const size_t n0_idx = dst_row_idx * KAI_NR + nr_idx;

      if (n0_idx >= rows_count)
        continue; // padding super-row (clamped in the forward)

      const uint8_t qs0 = (uint8_t)(src_row[dst_byte_idx] ^ 0x88);
      const uint8_t src_x0_lo = (uint8_t)(qs0 & 0x0F); // nibble at (n0,k0)
      const uint8_t src_x0_hi =
        (uint8_t)((qs0 >> 4) & 0x0F); // nibble at (n0,k1)

      if (k0_idx < columns_count) {
        const size_t bi = n0_idx * rhs_row_bytes + (k0_idx / 2);
        out_plain_nibbles[bi] |= (uint8_t)(src_x0_lo << ((k0_idx % 2) * 4));
      }
      if (k1_idx < columns_count) {
        const size_t bi = n0_idx * rhs_row_bytes + (k1_idx / 2);
        out_plain_nibbles[bi] |= (uint8_t)(src_x0_hi << ((k1_idx % 2) * 4));
      }
    }
  }
}

void Int4Utils::readLegacyQint4RecordToQs4cx(
  const uint8_t *record, size_t record_bytes, size_t rows_count,
  size_t columns_count, uint8_t *out_plain_nibbles, float *out_fp32_scales) {
  const size_t N = rows_count;
  const size_t K = columns_count;
  const size_t hdr = sizeof(uint16_t);

  NNTR_THROW_IF(record_bytes < hdr, std::runtime_error)
    << "[readLegacyQint4RecordToQs4cx] record too small for qscheme header";

  uint16_t scheme = 0;
  std::memcpy(&scheme, record, hdr);
  const uint8_t *body = record + hdr;

  if (scheme == static_cast<uint16_t>(QScheme::QS4CX)) {
    // 0x06: Section A nibbles + per-channel fp16 scales.
    const size_t nib = kaiNibblePayloadBytes(N, K);
    NNTR_THROW_IF(record_bytes < hdr + nib + N * sizeof(uint16_t),
                  std::runtime_error)
      << "[readLegacyQint4RecordToQs4cx] Section A record truncated: have "
      << record_bytes << " need " << (hdr + nib + N * sizeof(uint16_t));
    // Lossless re-layout of the int4 values.
    sectionAToPlain(body, N, K, out_plain_nibbles);
    // Exact fp16 -> fp32 widening of the per-channel scales.
    const uint8_t *scale_src = body + nib;
    for (size_t n = 0; n < N; ++n) {
      uint16_t h;
      std::memcpy(&h, scale_src + n * sizeof(uint16_t), sizeof(uint16_t));
      out_fp32_scales[n] = compute_fp16_to_fp32(h);
    }
  } else if (scheme == static_cast<uint16_t>(QScheme::PER_CHANNEL_AFFINE)) {
    // 0x01: PR#3978 plain container (padded plain nibbles + fp32 scales).
    const size_t pay = plainRecordPayloadBytes(N, K);
    NNTR_THROW_IF(record_bytes < hdr + pay, std::runtime_error)
      << "[readLegacyQint4RecordToQs4cx] plain-container record truncated: "
         "have "
      << record_bytes << " need " << (hdr + pay);
    // Normalize the padded plain nibbles to the canonical (unpadded) plain
    // layout by round-tripping through Section A (drops the KAI nr=8 / K->32
    // padding); both directions are lossless int4 re-layouts.
    std::vector<uint8_t> sec_a(kaiNibblePayloadBytes(N, K));
    packPlainToSectionA(body, N, K, sec_a.data());
    sectionAToPlain(sec_a.data(), N, K, out_plain_nibbles);
    // The plain container already stores fp32 scales — copy verbatim.
    std::memcpy(out_fp32_scales, body + plainScalesOffsetBytes(N, K),
                N * sizeof(float));
  } else {
    NNTR_THROW_IF(true, std::runtime_error)
      << "[readLegacyQint4RecordToQs4cx] unsupported on-disk qscheme 0x"
      << std::hex << scheme << std::dec << " (N=" << N << " K=" << K << ")";
  }
}

void Int4Utils::dequantizeSectionAToFp32(const uint8_t *section_a,
                                         const uint16_t *fp16_scales, size_t N,
                                         size_t K, float *out_nk) {
  // Inverse of the Section A super-row packing (mirrors the GPU v8c builder
  // decode in blas_kernels.cpp): undo the XOR 0x88 + the nr=4 / kr=16 / sr=2
  // 16-K interleave to recover row-major int4(n,k) = uint4 - 8, then scale.
  constexpr size_t KAI_KR_BY_SR = 8; // KR/SR
  const size_t k_blocks = K / 32;
  const size_t super_row_count = N / KAI_NR;
  const size_t nibble_bytes_per_super_row = KAI_NR * (K / 2);
  const size_t KAI_BYTES_PER_KBLK = 2 * KAI_NR * KAI_KR_BY_SR; // 64
  for (size_t sr = 0; sr < super_row_count; ++sr) {
    const uint8_t *sr_base = section_a + sr * nibble_bytes_per_super_row;
    for (size_t nr = 0; nr < KAI_NR; ++nr) {
      const size_t n = sr * KAI_NR + nr;
      const float scale = nntrainer::compute_fp16_to_fp32(fp16_scales[n]);
      float *out_row = out_nk + n * K;
      for (size_t kblk = 0; kblk < k_blocks; ++kblk) {
        const uint8_t *blk_a =
          sr_base + kblk * KAI_BYTES_PER_KBLK + nr * KAI_KR_BY_SR;
        const uint8_t *blk_b = blk_a + KAI_NR * KAI_KR_BY_SR;
        const size_t kbase = kblk * 32;
        for (size_t i = 0; i < KAI_KR_BY_SR; ++i) {
          const uint8_t ba = (uint8_t)(blk_a[i] ^ 0x88);
          out_row[kbase + i] = ((int)(ba & 0x0F) - 8) * scale;
          out_row[kbase + 16 + i] = ((int)((ba >> 4) & 0x0F) - 8) * scale;
          const uint8_t bb = (uint8_t)(blk_b[i] ^ 0x88);
          out_row[kbase + 8 + i] = ((int)(bb & 0x0F) - 8) * scale;
          out_row[kbase + 24 + i] = ((int)((bb >> 4) & 0x0F) - 8) * scale;
        }
      }
    }
  }
}

void Int4Utils::quantizeAndPackKai(const float *weights,
                                   const size_t rows_count,
                                   const size_t columns_count,
                                   std::vector<uint8_t> &out_weights,
                                   std::vector<uint16_t> &out_scales) {
  std::vector<uint8_t> plain_nibbles;
  quantizePlain(weights, rows_count, columns_count, plain_nibbles, out_scales);
  out_weights.assign(kaiNibblePayloadBytes(rows_count, columns_count), 0u);
  packPlainToSectionA(plain_nibbles.data(), rows_count, columns_count,
                      out_weights.data());
}

void Int4Utils::dequantizePacked(const std::vector<uint8_t> &weights,
                                 const std::vector<uint16_t> &scales,
                                 const size_t rows_count,
                                 const size_t columns_count,
                                 const size_t group_size,
                                 std::vector<float> &dequantized_weights) {
  const auto groups_per_row = ceilDiv(columns_count, group_size);
  const auto rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  const auto row_blocks_count = ceilDiv(rows_count, ROW_BLOCK_SIZE);
  const auto columns_count_pad = align(columns_count, group_size);
  const auto column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE);

  dequantized_weights.resize(rows_count * columns_count);

  size_t weights_idx = 0;

  for (size_t row_block_id = 0; row_block_id < row_blocks_count;
       ++row_block_id) {
    for (size_t column_block_id = 0; column_block_id < column_blocks_count;
         ++column_block_id) {
      for (size_t i = 0; i < ROW_BLOCK_SIZE; ++i) {
        uint8_t lo = 0, hi = 0;
        const auto row_id_absolute = (row_block_id * ROW_BLOCK_SIZE) + i;
        if (row_id_absolute < rows_count) {
          const auto column_id_absolute_lo =
            (column_block_id * COLUMN_BLOCK_SIZE);
          if (column_id_absolute_lo < columns_count) {
            const auto column_id_absolute_hi = column_id_absolute_lo + 1;

            const auto scale_lo =
              scales[row_id_absolute +
                     ((column_id_absolute_lo / group_size) * rows_count_pad)];

            const auto scale_hi =
              scales[row_id_absolute +
                     ((column_id_absolute_hi / group_size) * rows_count_pad)];

            const auto weight = weights[weights_idx];
            const auto weight_lo = weight & 0xF;
            const auto weight_hi = (weight >> 4) & 0xF;

            dequantized_weights[(row_id_absolute * columns_count) +
                                column_id_absolute_lo] =
              Int4Utils::convertInt4ToInt(weight_lo) *
              nntrainer::compute_fp16_to_fp32(scale_lo);

            if (column_id_absolute_hi < columns_count) {
              dequantized_weights[(row_id_absolute * columns_count) +
                                  column_id_absolute_hi] =
                Int4Utils::convertInt4ToInt(weight_hi) *
                nntrainer::compute_fp16_to_fp32(scale_hi);
            }
          }
        }
        weights_idx++;
      }
    }
  }
}

void Int4Utils::dequantizePackedRow(uint8_t *weights, uint16_t *scales,
                                    const size_t rows_count,
                                    const size_t columns_count,
                                    const size_t group_size,
                                    const size_t row_index,
                                    float *dequantized_row) {
  // --- Validate ---
  NNTR_THROW_IF(rows_count == 0 || columns_count == 0, std::invalid_argument)
    << "rows_count and columns_count must be > 0";
  NNTR_THROW_IF(row_index >= rows_count, std::out_of_range)
    << "row_index out of range";
  NNTR_THROW_IF(!(group_size == 32 || group_size == 64 || group_size == 128),
                std::invalid_argument)
    << "group_size must be 32/64/128";

  // --- Layout ---
  const size_t rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(columns_count, group_size);
  const size_t column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE); // COLUMN_BLOCK_SIZE == 2
  const size_t padded_groups_per_row = ceilDiv(columns_count, group_size);

  // Address the bytes for this row
  const size_t row_block_id = row_index / ROW_BLOCK_SIZE;
  const size_t i_in_block = row_index % ROW_BLOCK_SIZE;
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;
  const size_t row_block_base =
    row_block_id * bytes_per_row_block_span + i_in_block;

  for (size_t column_block_id = 0; column_block_id < column_blocks_count;
       ++column_block_id) {
    const size_t weights_idx =
      row_block_base + column_block_id * ROW_BLOCK_SIZE;
    const uint8_t packed_byte = weights[weights_idx];

    const size_t col_lo = column_block_id * COLUMN_BLOCK_SIZE;
    const size_t col_hi = col_lo + 1;

    const int q_lo = Int4Utils::convertInt4ToInt(packed_byte & 0xF);
    const int q_hi = Int4Utils::convertInt4ToInt((packed_byte >> 4) & 0xF);

    if (col_lo < columns_count) {
      const size_t g_lo = col_lo / group_size;
      const float s_lo = nntrainer::compute_fp16_to_fp32(
        scales[row_index + g_lo * rows_count_pad]);
      dequantized_row[col_lo] = static_cast<float>(q_lo) * s_lo;
    }
    if (col_hi < columns_count) {
      const size_t g_hi = col_hi / group_size;
      const float s_hi = nntrainer::compute_fp16_to_fp32(
        scales[row_index + g_hi * rows_count_pad]);
      dequantized_row[col_hi] = static_cast<float>(q_hi) * s_hi;
    }
  }
}

void Int4Utils::dequantizePackedRow32ToInt4Scale(
  const uint8_t *weights, const uint16_t *scales, const size_t rows_count,
  const size_t columns_count, const size_t group_size, const size_t row_index,
  const size_t column_index, uint8_t *weight_int4_row32, uint16_t *scale) {
  // --- Validate ---
  NNTR_THROW_IF(rows_count == 0 || columns_count == 0, std::invalid_argument)
    << "rows_count and columns_count must be > 0";
  NNTR_THROW_IF(row_index >= rows_count, std::out_of_range)
    << "row_index out of range";
  NNTR_THROW_IF(!(group_size == 32 || group_size == 64 || group_size == 128),
                std::invalid_argument)
    << "group_size must be 32/64/128";
  NNTR_THROW_IF(columns_count % 32 != 0, std::invalid_argument)
    << "columns_count must be divisible by 32";

  // --- Layout ---
  const size_t rows_count_pad = align(rows_count, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(columns_count, group_size);
  const size_t column_blocks_count =
    ceilDiv(columns_count_pad, COLUMN_BLOCK_SIZE); // COLUMN_BLOCK_SIZE == 2
  const size_t padded_groups_per_row = ceilDiv(columns_count, group_size);

  // Address the bytes for this row
  const size_t row_block_id = row_index / ROW_BLOCK_SIZE;
  const size_t i_in_block = row_index % ROW_BLOCK_SIZE;
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;
  const size_t row_block_base =
    row_block_id * bytes_per_row_block_span + i_in_block;

  for (size_t column_block_id = 0; column_block_id < 16; ++column_block_id) {
    const size_t weights_idx =
      row_block_base + (column_index / 2 + column_block_id) * ROW_BLOCK_SIZE;
    const uint8_t packed_byte = weights[weights_idx];

    weight_int4_row32[column_block_id] = packed_byte;
  }

  *scale = scales[row_index + (column_index / group_size) * rows_count_pad];
}
} // namespace nntrainer
