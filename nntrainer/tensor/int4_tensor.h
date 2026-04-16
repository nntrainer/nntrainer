// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_tensor.h
 * @date	23 January 2025
 * @brief	This is Int4QTensor class for quantized 4-bit integer calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __INT4_TENSOR_H__
#define __INT4_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class Int4QTensor class
 * @brief Int4QTensor class for quantized 4-bit integer calculation
 *
 * @note Int4QTensor stores symmetric signed int4 data inside int8 memory
 * space. Each int8 byte carries two int4 values packed together; the high
 * nibble is the element at even index, the low nibble is the element at
 * odd index. E.g. the byte 01011001 (0x59) represents 0101 (+5) at index
 * 2*i and 1001 (-1, two's complement) at index 2*i+1. The class supports
 * both PER_TENSOR_AFFINE (one scale for the entire tensor) and
 * PER_CHANNEL_AFFINE (grouped per-channel) via QScheme.
 *
 * @note CANONICAL IN-MEMORY / ON-DISK LAYOUT (must match all three
 * backends: KleidiAI CPU, LiteRT-LM/Adreno GPU repackers, QNN HTP):
 *
 *   Offset                               | Content
 *   -------------------------------------+-------------------------------
 *   0 .. ceil(N/2) - 1                   | packed int4 values, row-major
 *                                        | in the tensor's natural order
 *                                        | (output-channel first). Two
 *                                        | nibbles per byte, high nibble
 *                                        | = even index.
 *   ceil(N/2) .. ceil(N/2) + 4*S - 1     | per-scale fp32 array of
 *                                        | length S = scale_size(),
 *                                        | contiguous, row-major.
 *
 * NOTE on scale dtype: scales are fp32, not fp16. This matches (a) the
 * memory that Int4QTensor::allocate() actually reserves via
 * `sizeof(float) * scale_size()`, (b) KleidiAI's qai8dxp_qsi4cxp_unpacked
 * variant which takes `rhs_scales_f32`, and (c) the existing
 * char_tensor/quantizer code that accesses scales via getScale<float>().
 * Using 2 bytes/scale (fp16) would save memory but would conflict with
 * all three existing consumers and break the round-trip test suite.
 *
 * Where N = dim.getDataLen() and S is determined by the QScheme:
 *   PER_TENSOR_AFFINE  -> S = 1
 *   PER_CHANNEL_AFFINE -> S = height * width / group_size_
 *                         (if group_size_ == row_width, this collapses
 *                          to S = height, i.e. one scale per output
 *                          channel = pure per-channel / qsi4cxp)
 *
 * group_size_ is a PER-INSTANCE member so one process can hold tensors
 * with different group sizes (e.g. 0 / pure, 32, 64, 128) simultaneously.
 * The previous implementation used a static class member, which forced
 * all Int4QTensors in a process to share a single group size and made
 * mixed-quantization models impossible.
 *
 * This canonical layout is also what safetensors schema_version 2 writes
 * for dtype "I4" + quant.encoding == "axis_scale_offset". See
 * neuralnet.cpp::NeuralNetwork::save for the writer and the
 * documentation note in P4 for details.
 */
class Int4QTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  Int4QTensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
              QScheme qscheme_ = QScheme::PER_CHANNEL_AFFINE,
              size_t g_size = 0);

  /**
   * @brief Construct a new Int4QTensor object
   *
   * @param d Tensor dim for this qint4 tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   * @param qscheme_ Quantization scheme of the tensor
   */
  Int4QTensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "",
              QScheme qscheme_ = QScheme::PER_CHANNEL_AFFINE,
              size_t g_size = 0);

  /**
   * @brief Construct a new Int4QTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @param qscheme_ quantization scheme of the tensor
   */
  Int4QTensor(const TensorDim &d, const void *buf = nullptr,
              QScheme qscheme_ = QScheme::PER_CHANNEL_AFFINE,
              size_t g_size = 0);

  /**
   * @brief Construct a new Int4QTensor object
   *
   * @param d data for the Tensor
   * @param scales scale factors for the Tensor
   * @param fm format for the Tensor
   * @param qscheme_ quantization scheme of the tensor
   */
  Int4QTensor(
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
    std::vector<float> const &scales, Tformat fm, QScheme qscheme_,
    size_t g_size = 0);

  /**
   * @brief Construct a new Int4QTensor object
   * @param rhs TensorBase object to copy
   */
  Int4QTensor(TensorBase &rhs) :
    TensorBase(rhs), qscheme(QScheme::PER_CHANNEL_AFFINE) {}

  /**
   * @brief Basic Destructor
   */
  ~Int4QTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator==(const Int4QTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator!=(const Int4QTensor &rhs) const { return !(*this == rhs); }

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData(size_t idx)
   */
  void *getData(size_t idx) const override;

  /**
   * @copydoc Tensor::getScale()
   */
  void *getScale() const override;

  /**
   * @copydoc Tensor::getScale(size_t idx)
   */
  void *getScale(size_t idx) const override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  void *getAddress(unsigned int i) override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const void *getAddress(unsigned int i) const override;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  const int8_t getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  int8_t getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const int8_t getValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  int8_t getValue(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w);

  /**
   * @copydoc Tensor::setValue(float value)
   */
  void setValue(float value) override;

  /**
   * @copydoc Tensor::setValue(b, c, h, w, value)
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override;

  /**
   * @copydoc Tensor::addValue(b, c, h, w, value, beta)
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override;

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc Tensor::copy(const Tensor &from)
   */
  void copy(const Tensor &from, ComputeOps *ops = nullptr) override;

  /**
   * @copydoc Tensor::copyData(const Tensor &from)
   */
  void copyData(const Tensor &from, ComputeOps *ops = nullptr) override;

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override;

  /**
   * @copydoc Tensor::save(std::ostream &file)
   */
  void save(std::ostream &file) override;

  /**
   * @copydoc Tensor::read(std::ifstream &file)
   */
  void read(std::ifstream &file, size_t start_offset,
            bool read_from_offset) override;

  /**
   * @brief     Read the Tensor from file
   * @param[in] src input file stream
   */
  void read(ReadSource src, size_t start_offset = 0,
            bool read_from_offset = false) override;

  /**
   * @copydoc Tensor::argmax()
   */
  std::vector<unsigned int> argmax() const override;

  /**
   * @copydoc Tensor::argmin()
   */
  std::vector<unsigned int> argmin() const override;

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs(ComputeOps *ops = nullptr) const override;

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override;

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override;

  /**
   * @copydoc Tensor::print(std::ostream &out)
   */
  void print(std::ostream &out) const override;

  /**
   * @copydoc TensorBase::save_quantization_info()
   */
  void save_quantization_info(std::ostream &file) override;

  /**
   * @copydoc TensorBase::read_quantization_info()
   */
  void read_quantization_info(std::ifstream &file, size_t start_offset,
                              bool read_from_offset) override;

  /**
   * @copydoc TensorBase::read_quantization_info()
   */
  void read_quantization_info(ReadSource src, size_t start_offset,
                              bool read_from_offset) override;
  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::scale_size()
   */
  size_t scale_size() const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  /**
   * @brief     return the quantization group size stored on this instance.
   * @retval    group size in elements
   * @note      This is the number of elements that share one fp16 scale
   *            factor within a channel (or across the whole tensor when
   *            scheme == PER_TENSOR_AFFINE). See the CANONICAL LAYOUT
   *            note on the class for full semantics.
   */
  size_t group_size() const override { return group_size_; }

  /**
   * @brief Build a Q4_0 interleaved repack cache from the canonical
   *        qsi4cxp kxn data. This enables fast GEMM on x86 (via the
   *        GGML AVX2 kernel) without KleidiAI. The cache is persistent
   *        — call once at load time and reuse across all forward() calls.
   *
   * Layout produced (matches GGML gemm_q4_0):
   *   N rows of (K/32) block_q4_0 each. Each block = 2B fp16 scale +
   *   16B packed nibbles (32 int4 values). Total = N * (K/32) * 18 bytes.
   *
   * The per-output-column fp32 scale is TRUNCATED to fp16 and duplicated
   * into every block within that column (since the original data is
   * pure per-channel = same scale for all K elements in one column).
   *
   * @pre width() must be divisible by 32 (K%32==0 for Q4_0 blocks).
   *      Throws std::invalid_argument otherwise.
   * @note Thread-safe for concurrent reads after build (const accessor).
   */
  void buildQ4_0RepackCache();

  /**
   * @brief Return the Q4_0 repack buffer (nullptr if not built).
   */
  const void *getQ4_0RepackData() const {
    return q4_0_repack_cache_.empty() ? nullptr : q4_0_repack_cache_.data();
  }

  /**
   * @brief Check whether the Q4_0 repack cache has been built.
   */
  bool hasQ4_0RepackCache() const { return !q4_0_repack_cache_.empty(); }

private:
  /**
   * @brief quantization scheme
   */
  QScheme qscheme;

  /**
   * @brief per-instance quantization group size (elements per scale).
   *        Default 0, which is the canonical signal for "pure per-channel"
   *        (= one scale per output row, a.k.a. qsi4cxp / QNN
   *        AXIS_SCALE_OFFSET with numScaleOffsets == height). A non-zero
   *        value means "group_size_ elements share one scale" within
   *        each output row (e.g. 32 -> qsi4c32p / GGML Q4_0 block layout).
   *        Set via constructor argument g_size; no longer static so
   *        multiple Int4QTensors with different group sizes can coexist
   *        in one process (required for per-layer mixed quant and for
   *        safetensors schema_version 2 round-tripping).
   */
  size_t group_size_ = 0;

  /**
   * @brief Cached Q4_0 interleaved repack for x86 GGML GEMM. Built
   *        once at load time via buildQ4_0RepackCache(), then used by
   *        FloatTensor::dotQInteger on non-KleidiAI platforms.
   */
  std::vector<uint8_t> q4_0_repack_cache_;

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QINT4)
   */
  std::string getStringDataType() const override { return "QINT4"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid(ComputeOps *ops = nullptr) const override { return true; };
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __INT4_TENSOR_H__ */
