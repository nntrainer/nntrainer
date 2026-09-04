// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs4cx_tensor.h
 * @date	17 June 2026
 * @brief	This is QS4CX_Tensor class for QS4CX quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@google.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __QS4CX_TENSOR_H__
#define __QS4CX_TENSOR_H__
#ifdef __cplusplus

#include <atomic>

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class QS4CX_Tensor class
 * @brief QS4CX_Tensor class for QS4CX quantized tensor
 */
class QS4CX_Tensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  QS4CX_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new QS4CX_Tensor object
   *
   * @param d Tensor dim for this qs4cx tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  QS4CX_Tensor(const TensorDim &d, bool alloc_now,
               Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new QS4CX_Tensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  QS4CX_Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new QS4CX_Tensor object
   * @param rhs TensorBase object to copy
   */
  QS4CX_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override {
    data = nullptr;
    offset = 0;
  }

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData(size_t idx) const override {
    throw std::invalid_argument(
      "QS4CX_Tensor::getData() is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getPackedData()
   */
  void *getPackedData() const override;

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument("QS4CX_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument("QS4CX_Tensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("QS4CX_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("QS4CX_Tensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("QS4CX_Tensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument("QS4CX_Tensor::initialize() is not supported.");
  }

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::print()
   */
  void print(std::ostream &out) const override;

  /**
   * @copydoc Tensor::copy()
   */
  void copy(const Tensor &from) override {
    throw std::invalid_argument("QS4CX_Tensor::copy() is not supported.");
  }

  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from) override {
    throw std::invalid_argument("QS4CX_Tensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "QS4CX_Tensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override {
    throw std::invalid_argument("QS4CX_Tensor::max_abs() is not supported.");
  }

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override {
    throw std::invalid_argument("QS4CX_Tensor::maxValue() is not supported.");
  }

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override {
    throw std::invalid_argument("QS4CX_Tensor::minValue() is not supported.");
  }

  /**
   * @brief Bytes of the nibble payload of one QS4CX record, i.e. the offset of
   *        the per-channel fp32 scales inside it.
   * @param K input channels (height)
   * @param N output channels (width), one fp32 scale each
   * @param padded true for the padded layout `N * (K + 1) / 2`, false for the
   *        trimmed one `N * ((K + 1) / 2)`. The nibbles themselves are the
   *        same N rows of ceil(K/2) bytes either way; the padded layout adds
   *        floor(N/2) unused bytes before the scales when K is even (the two
   *        expressions coincide for odd K).
   * @return offset of the scales in bytes
   */
  static size_t nibbleBytes(size_t K, size_t N, bool padded) {
    return padded ? N * (K + 1) / 2 : N * ((K + 1) / 2);
  }

  /**
   * @brief Bytes of one whole QS4CX record: nibble payload + N fp32 scales.
   *        This is both the on-disk record stride and the buffer this tensor
   *        needs, so NeuralNetwork::load() can size a record it has not built
   *        a tensor for yet.
   * @param K input channels (height)
   * @param N output channels (width)
   * @param padded see nibbleBytes()
   * @return record size in bytes
   */
  static size_t recordBytes(size_t K, size_t N, bool padded) {
    return nibbleBytes(K, N, padded) + N * sizeof(float);
  }

  /**
   * @copydoc TensorBase::size()
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::getScale()
   */
  void *getScale() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  /**
   * @brief Eagerly pack the weight data after loading
   * @note Must be called after load_weight() to prepare for computation
   * @note Prepares weight data for efficient matrix multiplication
   */
  void pack() override;

  /**
   * @brief Eagerly build the fp16-activation KAI rhs after loading
   * @note Byte-identical to the buffer HalfTensor::dot's QS4CX case assembles
   * lazily on its first call (plain -> Section A -> fp16 scales ->
   * assembleKaiRhsPacked), so this only moves WHEN it is built. The layout is
   * NOT interchangeable with pack()'s fp32-facade rhs; repack_weight() picks
   * one of the two by the model's activation dtype. ARM-only (no-op
   * elsewhere, mirroring pack()'s NYI behavior on x86).
   */
  void packF16Activation() override;

  /**
   * @copydoc TensorBase::isPackedF16Activation()
   */
  bool isPackedF16Activation() const override {
    return packed_f16.load(std::memory_order_acquire) && packed_data != nullptr;
  }

  /**
   * @copydoc TensorBase::isPacked()
   * @note packed_data holds either pack()'s fp32-activation rhs or
   * packF16Activation()'s fp16-scale rhs, and the two layouts are not
   * interchangeable, so the fp16 one must not be reported here.
   */
  bool isPacked() const override {
    return !packed_f16.load(std::memory_order_acquire) &&
           packed_data != nullptr;
  }

  /**
   * @copydoc Tensor::read(std::ifstream &file, size_t, bool)
   * @note When this tensor is flagged on-disk-legacy-QINT4
   *   (setOnDiskLegacyQint4), the record is a legacy QINT4 (u16 header + KAI
   *   Section A / plain container) and is transcoded losslessly to the QS4CX
   *   in-memory layout via Int4Utils::readLegacyQint4RecordToQs4cx; otherwise a
   *   plain TensorBase::read.
   */
  void read(std::ifstream &file, size_t start_offset = 0,
            bool read_from_offset = false) override;

  /**
   * @copydoc Tensor::read(ReadSource, size_t, bool)
   */
  void read(ReadSource src, size_t start_offset = 0,
            bool read_from_offset = false) override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy_qs4cx(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS4CX)
   */
  std::string getStringDataType() const override { return "QS4CX"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }

  std::unique_ptr<uint8_t[]> packed_data = nullptr;
  /**
   * @brief packed_data holds the fp16-activation KAI rhs (packF16Activation),
   * not pack()'s fp32 layout
   * @note Atomic because HalfTensor::dot's QS4CX case reads it outside the lock
   * that serialises the lazy fill (a double-checked lock). packF16Activation()
   * publishes it with release AFTER filling packed_data, and
   * isPackedF16Activation() acquires it before looking at packed_data, so a
   * thread that sees the flag set also sees the buffer it names.
   */
  std::atomic<bool> packed_f16 = {false};
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QS4CX_TENSOR_H__ */
