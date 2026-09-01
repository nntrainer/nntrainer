// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_utils.h
 * @date	15 October 2025
 * @brief	This is Int4Utils class for utils for INT4 quantization format.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Grzegorz Kisala <gkisala@gmail.com>
 * @bug		No known bugs
 */

#ifndef __NNTRAINER_INT4_UTILS_H__
#define __NNTRAINER_INT4_UTILS_H__

#include <algorithm>
#include <cstdint>
#include <vector>

namespace nntrainer {

/**
 * @class Int4Utils class
 * @brief Int4Utils class with helpers for 4-bit integers calculation,
 * quantization and dequantization methods.
 *
 * Two on-disk layouts are recognised:
 *  - legacy osv32_isv2 (OpenVINO OS_IS_YX_OSV32_ISV2)
 *  - KAI qsi4cxp super-row (nr=4, kr=16, sr=2, idx_variant=2 of the
 *    qai8dxp_qsi4cxp interface). Section A nibble payload only — scales,
 *    sums and bias trailers are reassembled at load time.
 */
class Int4Utils {
public:
  /// @brief Block size used in the osv32_isv2 layout
  static constexpr const size_t ROW_BLOCK_SIZE = 32;

  /// @brief Numbers of element in one byte of date in the osv32_isv2 layout
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  /// @brief KAI qsi4cxp constants for matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_
  ///        4x4x32_neon_i8mm (idx_variant=2 in cpu_backend wrapper).
  static constexpr const size_t KAI_NR = 4;
  static constexpr const size_t KAI_KR = 16;
  static constexpr const size_t KAI_SR = 2;
  static constexpr const size_t KAI_K_INTERLEAVE = 16;
  static constexpr const size_t KAI_K_PAD_MULTIPLE = 32;
  static constexpr const uint32_t KAI_QSI4CXP_VARIANT_4x4x32 = 2;

  /**
   * @brief     Compute scale for input weights
   * @param[in] group_weights float * inout vector of weights
   * @param[in] group_size group size (32 or 64 or 128)
   * @return computed scale
   */
  static float computeScaleForGroup(const float *group_weights,
                                    const size_t group_size);

  /**
   * @brief     Compute scales for float* matrix weghts
   * @param[in] weights float * input matrix
   * @param[in] rows_count number of rows of input matrix
   * @param[in] columns_count number of columns of input matrix
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[out] scales float vector output scales
   */
  static void computeScales(const float *weights, const size_t rows_count,
                            const size_t columns_count, const size_t group_size,
                            std::vector<float> &scales);

  /**
   * @brief     Pack one weight from position (row_id, column_id) into 4-bits
   * value
   * @param[in] weights float * input matrix
   * @param[in] scales float * input vector os scales
   * @param[in] row_id number of row
   * @param[in] column_id number of column
   * @param[in] groups_per_row number of groups pre row
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[in] rows_count number of rows of input matrix
   * @param[in] columns_count number of columns of input matrix
   * @return
   */
  static uint8_t pack(const float *weights, const float *scales,
                      const size_t row_id, const size_t column_id,
                      const size_t groups_per_row, const size_t group_size,
                      const size_t rows_count, const size_t columns_count);

  /**
   * @brief Quantize weights float* matrix to OpenVINO layout:
   * OS_IS_YX_OSV32_ISV2, osv32_isv2 layout for int4 packed weight:
   *
   * y0_x0x1 | y1_x0x1 | ....  | y15_x0x1|| y16_x0x1 | y17_x0x1 | ... | y31_x0x1
   * y0_x2x3 | y1_x2x3 | ....  | y15_x2x3|| y16_x2x3 | y17_x2x3 | ... | y31_x2x3
   * ...
   * @param weights float * input matrix
   * @param rows_count number of rows of input matrix
   * @param columns_count number of columns of input matrix
   * @param group_size group size (32 or 64 or 128)
   * @param out_weights output quantized weights in layout osv**_isv2
   * @param out_scales output scales
   */
  static void quantizeAndRepack(const float *weights, const size_t rows_count,
                                const size_t columns_count,
                                const size_t group_size,
                                std::vector<uint8_t> &out_weights,
                                std::vector<uint16_t> &out_scales);

  /**
   * @brief Quantize float* matrix into KAI qsi4cxp Section A nibble payload.
   *
   * Layout: per super-row of nr=4 output channels, contiguous block layout
   * with block_length = kr/sr = 8 bytes, super-block stride, and a 16-K
   * interleave (one byte stores nibbles at k0 and k0+16). Each stored
   * uint4 = int4 + 8 (the rhs_zero_point=8 form, XOR 0x88 applied to the
   * sign-bit pattern as in kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0).
   * No sums/scales/bias trailer is written — reassemble at load time.
   *
   * @param weights float * input matrix (rows_count x columns_count)
   * @param rows_count N (output channels)
   * @param columns_count K (input channels)
   * @param out_weights output nibble payload, size
   *        = ceil(N/KAI_NR) * KAI_NR * (roundup(K, KAI_K_PAD_MULTIPLE) / 2)
   * @param out_scales output per-channel fp16 scales, size = rows_count
   */
  static void quantizeAndPackKai(const float *weights, const size_t rows_count,
                                 const size_t columns_count,
                                 std::vector<uint8_t> &out_weights,
                                 std::vector<uint16_t> &out_scales);

  /**
   * @brief Shared plain QINT4 container (engine-neutral, byte-compatible
   *        with the upstream PR#3978 writer/reader):
   *        record = [u16 qscheme = PER_CHANNEL_AFFINE (0x01)]
   *                 [N x ceil(K/2) plain nibbles, row-major; even k in the
   *                  low nibble; each stored uint4 = int4 + 8]
   *                 [N fp32 per-channel scales at byte offset (N*(K+1))/2
   *                  — the PR writer/reader expression, which for even K
   *                  leaves an N/2-byte zero gap after the nibbles]
   *                 [zero pad to plainRecordPayloadBytes]
   *        The pad mirrors the PR writer's KAI packed size for its
   *        idx_variant=8 micro-kernel (nr=8) — NOT our 4x4x32 (nr=4) —
   *        because the disk record must stay byte-identical to PR#3978.
   *        Loaders repack at read time: CPU -> KAI rhs-packed, GPU -> the
   *        Section A form (packPlainToSectionA).
   */
  static constexpr const size_t PLAIN_KAI_NR = 8;

  /// @brief Bytes of the plain nibble matrix: N * ceil(K/2).
  static size_t plainNibbleBytes(size_t rows_count, size_t columns_count);

  /// @brief Byte offset of the fp32 scales inside the plain payload.
  static size_t plainScalesOffsetBytes(size_t rows_count, size_t columns_count);

  /// @brief Full plain payload size (excluding the u16 qscheme header):
  ///        ceil(N/8)*8 * (roundup(K,32)/2 + 12).
  static size_t plainRecordPayloadBytes(size_t rows_count,
                                        size_t columns_count);

  /**
   * @brief Quantize float* matrix into the plain nibble form (stage 1 of
   *        quantizeAndPackKai). With range15=false (default) uses the same
   *        per-channel absmax/7 scales and quantizeToInt4 values as the
   *        Section A path, so a plain record repacked at load time is
   *        bit-identical to a Section A record. With range15=true mirrors
   *        the PR#3978/KAI quantizer (scale = range/15 over min(0,min)..
   *        max(0,max), round half-away-from-zero) — better int4 grid use on
   *        asymmetric channels (quality-critical for small models), at the
   *        cost of token-id baselines vs absmax/7 bins.
   * @param out_plain_nibbles N x ceil(K/2) bytes, row-major, even k = low
   *        nibble, uint4 = int4 + 8
   * @param out_scales per-channel fp16 scales, size = rows_count
   */
  static void quantizePlain(const float *weights, const size_t rows_count,
                            const size_t columns_count,
                            std::vector<uint8_t> &out_plain_nibbles,
                            std::vector<uint16_t> &out_scales,
                            bool range15 = false);

  /**
   * @brief Permute plain nibbles into the KAI Section A super-row payload
   *        (stage 2 of quantizeAndPackKai; also the GPU/lg load-time
   *        repack for plain-container records).
   * @param out_section_a caller-provided buffer of
   *        kaiNibblePayloadBytes(rows_count, columns_count) bytes
   */
  static void packPlainToSectionA(const uint8_t *plain_nibbles,
                                  size_t rows_count, size_t columns_count,
                                  uint8_t *out_section_a);

  /**
   * @brief LOSSLESS inverse of packPlainToSectionA: de-permute a KAI Section A
   *        nibble payload back into plain row-major [rows_count(N)][ceil(K/2)]
   *        nibbles (uint4 = int4+8, no XOR), byte-identical to the plain form
   *        that produced it. Unlike dequantizeSectionAToFp32 this does NOT
   *        touch scales or dequantize — it only re-lays-out the int4 nibbles,
   *        so a QINT4 (Section A) weight becomes the QS4CX plain nibble layout
   *        with its exact original values preserved (the int4-format
   *        collapse: one in-memory int4 form).
   * @param out_plain_nibbles caller-provided buffer of
   *        rows_count * ((columns_count + 1) / 2) bytes
   */
  static void sectionAToPlain(const uint8_t *section_a, size_t rows_count,
                              size_t columns_count, uint8_t *out_plain_nibbles);

  /**
   * @brief Standalone reader for a legacy on-disk QINT4 weight record (the
   *        format Int4QTensor::read parses) that emits the CANONICAL QS4CX
   *        in-memory form WITHOUT any tensor class. @a record points at the
   *        start of the record (the u16 qscheme header). Dispatches on it:
   *          - QS4CX (0x06): Section A nibbles + per-channel fp16
   *            scales -> plain row-major nibbles (sectionAToPlain, lossless) +
   *            fp32 scales (compute_fp16_to_fp32, exact).
   *          - PER_CHANNEL_AFFINE (0x01): PR#3978 plain container -> canonical
   *            plain nibbles (packPlainToSectionA -> sectionAToPlain, drops the
   *            KAI padding) + its fp32 scales copied as-is.
   *        LOSSLESS: preserves the exact int4 values and scale magnitudes. The
   *        output is byte-identical to what layer_devel.h's QINT4->QS4CX
   *        transcode produces from a loaded Int4QTensor, so it can replace that
   *        load path once Int4QTensor is removed. Depends on no
   *        tensor class. Throws on any other on-disk qscheme.
   * @param record       start of the on-disk record (the u16 header).
   * @param record_bytes size of the @a record buffer (bounds-checked).
   * @param rows_count   N (output channels); N % 4 == 0.
   * @param columns_count K (input channels); K % 32 == 0.
   * @param out_plain_nibbles caller buffer of rows_count * ((K + 1) / 2) bytes.
   * @param out_fp32_scales   caller buffer of rows_count floats.
   */
  static void readLegacyQint4RecordToQs4cx(
    const uint8_t *record, size_t record_bytes, size_t rows_count,
    size_t columns_count, uint8_t *out_plain_nibbles, float *out_fp32_scales);

  /**
   * @brief Inverse of packPlainToSectionA + dequant: decode a KAI Section A
   *        nibble payload (+ per-channel fp16 scales) back to a row-major
   *        [rows_count(N) x columns_count(K)] fp32 weight (out[n*K+k] =
   *        (int4(n,k)) * scale[n]). Used to transcode a loaded QINT4 weight
   *        to fp32 so it can be re-quantized into another int4 format (QS4CX).
   *        rows_count must be a multiple of 4 and columns_count a multiple of
   *        32 (the Section A packing constraint).
   * @param out_nk caller-provided buffer of rows_count*columns_count floats
   */
  static void dequantizeSectionAToFp32(const uint8_t *section_a,
                                       const uint16_t *fp16_scales,
                                       size_t rows_count, size_t columns_count,
                                       float *out_nk);

  /**
   * @brief Convenience: byte size of the KAI Section A nibble payload for
   *        the given (N, K) shape.
   */
  static size_t kaiNibblePayloadBytes(size_t rows_count, size_t columns_count);

  /**
   * @brief Convenience: byte size of the KAI full rhs_packed buffer (with
   *        the per-super-row sums + scales*0.0625 + bias trailer) for the
   *        given (N, K) shape. Matches what the matmul micro-kernel walks
   *        per column tile (= generic packer stride = nr*(k_internal/2 +
   *        sums + scales + bias) = nr*(k_internal/2 + 12)).
   */
  static size_t kaiRhsPackedBytes(size_t rows_count, size_t columns_count);

  /**
   * @brief Reassemble disk Section A nibbles + per-channel fp16 scales
   *        into a full KAI rhs_packed buffer suitable for direct use with
   *        nntr_kai_gemm_qai8dxp_qsi4cxp_olp / _olp_f16. Per super-row of
   *        nr=4 output channels the layout is
   *          [nibbles : 4*(k_internal/2) bytes (copied as-is)]
   *          [sums    : 4 * int32, each = sum_k int4[n][k] * 16]
   *          [scales  : 4 * fp32,  each = (fp16->fp32 scale) * 0.0625]
   *          [bias    : 4 * fp32 = 0]
   *
   * @param section_a       nibble payload as stored on disk; size must
   *                        be kaiNibblePayloadBytes(rows_count,
   *                        columns_count).
   * @param fp16_scales     per-output-channel fp16 scales, size N.
   * @param rows_count      N (output channels). Must be a multiple of 4.
   * @param columns_count   K (input channels). Must be a multiple of 32.
   * @param out_kai_packed  output buffer, sized to kaiRhsPackedBytes.
   */
  static void assembleKaiRhsPacked(const uint8_t *section_a,
                                   const uint16_t *fp16_scales,
                                   size_t rows_count, size_t columns_count,
                                   std::vector<uint8_t> &out_kai_packed);

  /**
   * @brief     Quantize one float value to 4-bits integer
   * @param[in] weight input weight
   * @param[in] scale input scale
   * @return 4-bit integer
   */
  static uint8_t quantizeToInt4(const float weight, const float scale);

  /**
   * @brief     Convert 4-bit integer value to 32-bit integer
   * @param[in] int4_value input 4-bit signed integer value
   * @return output int value
   */
  static int convertInt4ToInt(const uint8_t int4_value);

  /**
   * @brief     Dequantize weights in osv32_isv2 layout and scales to float
   * weights
   * @param[in] weights input matrix with quantized weights in osv32_isv2 layout
   * @param[in] scales fp16 vector input scales
   * @param[in] rows_count number of rows of data
   * @param[in] columns_count number of columns of data
   * @param[in] group_size group size (32 or 64 or 128)
   * @param[out] dequantized_weights float vector of dequantized_weights
   */
  static void dequantizePacked(const std::vector<uint8_t> &weights,
                               const std::vector<uint16_t> &scales,
                               const size_t rows_count,
                               const size_t columns_count,
                               const size_t group_size,
                               std::vector<float> &dequantized_weights);

  /**
   * @brief Dequantize weights in osv32_isv2 layout by row
   *
   * @param weights quantized weights in osv32_isv2 layout
   * @param scales fp16 scales
   * @param rows_count number of rows of data
   * @param columns_count number of columns of data
   * @param group_size group size (32 or 64 or 128)
   * @param row_index row index to dequantize
   * @param dequantized_row dequantized_weights
   */
  static void dequantizePackedRow(uint8_t *weights, uint16_t *scales,
                                  const size_t rows_count,
                                  const size_t columns_count,
                                  const size_t group_size,
                                  const size_t row_index,
                                  float *dequantized_row);

  /**
   * @brief Dequantize weights in osv32_isv2 layout by row
   *
   * @param weights quantized weights in osv32_isv2 layout
   * @param scales fp16 scales
   * @param rows_count number of rows of data
   * @param columns_count number of columns of data
   * @param group_size group size (32 or 64 or 128)
   * @param row_index row index to dequantize
   * @param column_index column start index
   * @param weight_int4_row32 output 32xint4 (16 bytes)
   * @param scale output scale
   */
  static void dequantizePackedRow32ToInt4Scale(
    const uint8_t *weights, const uint16_t *scales, const size_t rows_count,
    const size_t columns_count, const size_t group_size, const size_t row_index,
    const size_t column_index, uint8_t *weight_int4_row32, uint16_t *scale);
};

} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
