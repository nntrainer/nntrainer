// SPDX-License-Identifier: Apache-2.0
/**
 * @file        q8_0_tensor.h
 * @date        14 May 2026
 * @brief       Q8_0_Tensor class for Q8_0 quantised weights.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * Q8_0 layout (matches llama.cpp / GGML):
 *   struct block_q8_0 {
 *     fp16  d;           //   2 bytes — block-level scale
 *     int8  qs[32];      //  32 bytes — 32 int8 quants
 *   };                   //  34 bytes per 32-element block
 *
 * As with Q4_0_Tensor this class only stores the byte buffer and reports its
 * shape & size; the actual matmul is dispatched from Tensor's GEMM path to
 * gemm_q8_0_fp32() in the cpu backend. Mutating operations throw — Q8_0 is
 * a load-only weight tensor.
 */

#ifndef __Q8_0_TENSOR_H__
#define __Q8_0_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

#ifdef ENABLE_FP16
#include <tensor_dim.h> // for _FP16
#endif

namespace nntrainer {

#define QK8_0 32

/**
 * @brief Q8_0 block. Mirrors GGML's block_q8_0.
 * @note  This struct is reference-only; the tensor buffer is a flat byte array.
 */
struct block_q8_0 {
  uint16_t d;       // fp16 scale
  int8_t qs[QK8_0]; // 32 signed int8 quants
};

/// @note Fully qualified so Q8_0_SIZE expands correctly even when used from
/// a different namespace (e.g. causallm). An unqualified `struct block_q8_0`
/// would declare a *new* incomplete type in that namespace and fail sizeof.
#define Q8_0_SIZE sizeof(::nntrainer::block_q8_0)

/**
 * @class Q8_0_Tensor
 * @brief Holder for a tensor whose data is laid out as packed block_q8_0
 *        records. Width must be a multiple of QK8_0.
 */
class Q8_0_Tensor : public TensorBase {

public:
  Q8_0_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  Q8_0_Tensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "");

  Q8_0_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief View constructor — wraps an external pre-allocated block_q8_0
   *        buffer without allocating or copying. Caller owns the lifetime.
   */
  Q8_0_Tensor(const TensorDim &d, void *external_buf);

  Q8_0_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

  void allocate() override;

  void deallocate() override {
    data = nullptr;
    offset = 0;
  }

  void *getData() const override;

  void *getData(size_t idx) const override {
    throw std::invalid_argument(
      "Q8_0_Tensor::getData(idx) is not supported. Use getData() instead.");
  }

  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("Q8_0_Tensor::getAddress() is not supported.");
  }

  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("Q8_0_Tensor::getAddress() is not supported.");
  }

  void setValue(float value) override {
    throw std::invalid_argument("Q8_0_Tensor::setValue() is not supported.");
  }

  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("Q8_0_Tensor::setValue() is not supported.");
  }

  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("Q8_0_Tensor::addValue() is not supported.");
  }

  void setZero() override;

  void initialize(Initializer init) override {
    throw std::invalid_argument(
      "Q8_0_Tensor::initialize(init) is not supported.");
  }

  void initialize() override;

  void print(std::ostream &out) const override {
    throw std::invalid_argument("Q8_0_Tensor::print() is not supported.");
  }

  void copy(const Tensor &from) override {
    throw std::invalid_argument("Q8_0_Tensor::copy() is not supported.");
  }

  void copyData(const Tensor &from) override {
    throw std::invalid_argument("Q8_0_Tensor::copyData() is not supported.");
  }

  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "Q8_0_Tensor::copy_with_stride() is not supported.");
  }

  float max_abs() const override {
    throw std::invalid_argument("Q8_0_Tensor::max_abs() is not supported.");
  }

  float maxValue() const override {
    throw std::invalid_argument("Q8_0_Tensor::maxValue() is not supported.");
  }

  float minValue() const override {
    throw std::invalid_argument("Q8_0_Tensor::minValue() is not supported.");
  }

  size_t size() const override;

  size_t getMemoryBytes() const override;

  QScheme q_scheme() const override;

  Tensor &convQ4_0Indirect(Tensor const &weight, Tensor &output,
                           const ConvGatherParams &geom) const override;

  /**
   * @brief Q8_0 (this, [M,K]) x Q4_0 (input, [K,N]) -> FP16 (output, [M,N]).
   *
   * Repacks the Q8_0 activation rows into block_q8_0x4 (4-row interleave) and
   * calls the SMMLA (i8mm) GEMM kernel directly on int8, bypassing all
   * FP16 dequantization / re-quantization. Used by the W4A8 1x1 conv path.
   */
  Tensor &dot(Tensor const &input, Tensor &output, bool trans = false,
              bool trans_in = false, float beta = 0.0f) const override;

  /**
   * @brief Direct Q8_0x4 (pre-interleaved) act x Q4_0 weight -> FP16 GEMM.
   *
   * @a QA already holds the activation as block_q8_0x4 (4-row interleaved) for
   * the first M4*4 rows followed by plain block_q8_0 for the M%4 remainder
   * (produced by transpose_quantize_q8_0x4_act). Runs only the SMMLA GEMM +
   * GEMV tail (no allocation, no interleave) — the fast W4A8 1x1 path.
   */
#ifdef ENABLE_FP16
  static void dot_prepacked_x4(unsigned int M, unsigned int K, unsigned int N,
                               const void *QA, const void *B, _FP16 *C,
                               unsigned int ldc);

  /**
   * @brief Quantize FP16 NHWC [n_spatial, in_ch] -> plain block_q8_0
   *        [n_spatial][in_ch/32].
   *
   * NHWC input is already row-major (channel innermost): src[r * in_ch + c].
   * No transpose needed — each row r has in_ch contiguous FP16 channels.
   * Q8_0 requires in_ch % 32 == 0 (caller must check).
   * dst must hold n_spatial * (in_ch/32) * sizeof(block_q8_0) bytes.
   */
  static void quantize_nhwc_q8_0_rows(const _FP16 *src, int n_spatial,
                                      int in_ch, block_q8_0 *dst);

  /**
   * @brief Quantize FP16 NHWC [owoh, in_ch] directly into the block_q8_0x4
   *        (4-row interleaved) layout the SMMLA GEMM consumes — single pass.
   *
   * NHWC source is row-major (channel innermost): element (r, c) at
   * src[r*in_ch+c]. This is the NHWC-read counterpart of
   * transpose_quantize_q8_0x4_act (which reads NCHW channel-major). It fuses
   * the two passes the prior 1x1 W4A8 path performed (quantize_nhwc_q8_0_rows
   * -> plain block_q8_0, then Q8_0_Tensor::dot repacks to x4) into one, and
   * lets the caller invoke Q8_0_Tensor::dot_prepacked_x4 (no per-conv repack,
   * no per-conv QA malloc). Output bytes are identical to what the two-pass
   * path produced. dst layout: M4=owoh/4 groups of block_q8_0x4 (136 B/blk)
   * followed by (owoh % 4) remainder rows as plain block_q8_0 (34 B/blk) —
   * exactly what dot_prepacked_x4 expects. dst must hold the same total as the
   * block_q8_0 buffer (136 B per 4 rows == 4 * 34 B). Q8_0 requires in_ch % 32
   * == 0.
   */
  static void quantize_nhwc_q8_0x4_rows(const _FP16 *src, int in_ch, int owoh,
                                        void *dst);

  /**
   * @brief Fused NHWC 1x1 Q8_0-activation convolution for the W8A8 path.
   *
   * Quantizes the NHWC FP16 activation (in_ch-innermost, [owoh, in_ch])
   * directly into the block_q8_0x4 SMMLA layout in @a qa_buf (single pass) and
   * runs the prepacked Q8_0 x Q8_0 / Q4_0 GEMM, writing FP16 [owoh, out_ch]
   * into @a out. Encapsulates the quantize_nhwc_q8_0x4_rows + dot_prepacked_x4
   * pair so conv layers stay layout/dtype-general and do not own
   * activation-quantization logic. Requires in_ch % 32 == 0 and ENABLE_FP16; @a
   * qa_buf must hold the full block_q8_0x4 region (owoh/4 * 136 B) plus the
   * owoh%4 remainder.
   *
   * @param in_ch   input channel count (must be divisible by 32)
   * @param owoh    output spatial count (M)
   * @param out_ch  output channel count (N == filter_size)
   * @param act_nhwc FP16 NHWC activation, [owoh, in_ch] row-major
   * @param qa_buf  caller-owned scratch (one batch slice) as block_q8_0 bytes
   * @param weight  quantized weight tensor (Q8_0 or Q4_0) as [in_ch, out_ch]
   * @param out     FP16 output, [1,1,owoh,out_ch] (NCHW flat)
   */
  static void conv_nhwc_q8act_1x1(int in_ch, int owoh, int out_ch,
                                  const _FP16 *act_nhwc, void *qa_buf,
                                  Tensor const &weight, _FP16 *out);

  /**
   * @brief Fused NHWC non-1x1 Q8_0-activation convolution via indirect GEMM.
   *
   * Quantizes the NHWC FP16 activation into plain block_q8_0 (n_spatial rows),
   * wraps it as a Q8_0_Tensor view over @a qa_buf, and runs the indirect conv
   * GEMM (im2col gather folded in) against @a weight, writing FP16 [owoh,
   * out_ch] into @a out. Encapsulates quantize_nhwc_q8_0_rows +
   * convQ4_0Indirect so conv layers do not handle activation quantization or
   * raw block_q8_0 casting. Requires in_ch % 32 == 0 and ENABLE_FP16; @a qa_buf
   * must hold n_spatial * (in_ch/32) block_q8_0 bytes.
   *
   * @param geom     indirect-conv geometry
   * (in_ch/in_h/in_w/kernel/pad/stride/...)
   * @param act_nhwc FP16 NHWC activation, [n_spatial, in_ch] row-major
   * @param qa_buf   caller-owned scratch (one batch slice) as block_q8_0 bytes
   * @param weight   quantized weight tensor (Q8_0 or Q4_0)
   * @param out      FP16 output, [1,1,owoh,out_ch] (NCHW flat)
   */
  static void conv_nhwc_q8act_indirect(const ConvGatherParams &geom,
                                       const _FP16 *act_nhwc, void *qa_buf,
                                       Tensor const &weight, Tensor &out);

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
   * dst must hold (owoh * in_ch/32) block_q8_0 = owoh*in_ch/32*34 bytes, laid
   * out row-major as [owoh][in_ch/32] blocks — exactly the [M,K] block_q8_0
   * layout Q8_0_Tensor::dot / the indirect GEMM consumes (M=owoh, K=in_ch).
   */
  static void transpose_quantize_q8_0_act(const _FP16 *src, int in_ch, int owoh,
                                          void *dst);

  /**
   * @brief Fused transpose + quantize FP16 NCHW [in_ch, owoh] directly into
   *        the block_q8_0x4 (4-row interleaved) layout the SMMLA GEMM consumes,
   *        with NO intermediate plain-block pass and NO separate interleave
   * copy.
   *
   * Outputs (M4 = owoh/4) groups of 4 rows; each group packs nb=in_ch/32
   * block_q8_0x4. block_q8_0x4.qs[128] layout = qs[32*j + 8*row + lane],
   * j=8-element chunk (0..3), row=0..3 (matches
   * __ggml_quantize_mat_q8_0_4x8). Remainder (owoh % 4) rows packed as plain
   * block_q8_0 afterward for the GEMV tail. dst must hold the block_q8_0x4
   * region followed by the remainder block_q8_0 region (same total as
   * Q8_0_Tensor::dot's QA buffer).
   */
  static void transpose_quantize_q8_0x4_act(const _FP16 *src, int in_ch,
                                            int owoh, void *dst);
#endif
private:
  void copy_q80(const void *buf);

  std::string getStringDataType() const override { return "Q8_0"; }

  bool isValid() const override { return true; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q8_0_TENSOR_H__ */
