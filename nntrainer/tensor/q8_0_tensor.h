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
#endif
private:
  void copy_q80(const void *buf);

  std::string getStringDataType() const override { return "Q8_0"; }

  bool isValid() const override { return true; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q8_0_TENSOR_H__ */
