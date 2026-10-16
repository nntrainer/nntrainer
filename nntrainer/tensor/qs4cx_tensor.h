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
   * @brief May allocate() leave this payload uninitialized (a load-destined
   *        heap-bypass weight, filled in full by its reader)?
   */
  bool allocUninitialized() const;

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
   * @note Prepares the fp32-activation KAI rhs once, so the GEMM does not pack
   * the weight on every call. FloatTensor::dotQs4cx() calls it lazily when the
   * load path did not. The pack runs in parallel over the ThreadManager, so
   * it must not be called from inside a parallel_for. The buffer is published
   * only after it is complete, and nothing is kept if the backend has no
   * packer (x86 throws NYI).
   */
  void pack() override;

  /**
   * @brief Eagerly build the fp16-activation KAI rhs after loading
   * @note Byte-identical to the buffer HalfTensor::dot's QS4CX case assembles
   * lazily on its first call (plain -> Section A -> fp16 scales ->
   * assembleKaiRhsPacked), so this only moves WHEN it is built. The layout is
   * NOT interchangeable with pack()'s fp32-facade rhs; repack_weight() picks
   * one of the two by the model's activation dtype. ARM-only (no-op
   * elsewhere, mirroring pack()'s NYI behavior on x86). Built in parallel
   * over the ThreadManager straight into the final buffer, so the same
   * no-nested-parallel_for rule as pack() applies.
   */
  void packF16Activation() override;

  /**
   * @copydoc TensorBase::isPackedF16Activation()
   */
  bool isPackedF16Activation() const override {
    return packed_f16.load(std::memory_order_acquire);
  }

  /**
   * @copydoc TensorBase::isPacked()
   * @note packed_data holds either pack()'s fp32-activation rhs or
   * packF16Activation()'s fp16-scale rhs, and the two layouts are not
   * interchangeable, so the fp16 one must not be reported here.
   */
  bool isPacked() const override {
    return packed_f32.load(std::memory_order_acquire);
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

protected:
  /**
   * @brief copy a buffer to @a this -- protected rather than private because
   * QS4CX_WH_Tensor has the same buffer layout below its extra column sums
   * and fills itself the same way. The caller has to ensure that @a this is
   * initialized, otherwise undefined behavior
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
  /**
   * @brief packed_data holds pack()'s fp32-activation KAI rhs
   * @note Same double-checked-lock contract as packed_f16, for the lazy pack
   * in FloatTensor::dotQs4cx(). At most one of the two flags is ever set.
   */
  std::atomic<bool> packed_f32 = {false};
};

/**
 * @brief Record that a QS4CX weight's plain payload pages have been released.
 *
 * A GPU backend that repacks a QS4CX weight into a device-side form may drop
 * the plain nibble+scale pages afterwards to reclaim the duplicate residency
 * (see NNTR_V8C_DROP_PLAIN). Dropped pages are still mapped -- the address has
 * to stay valid because the device-side weight caches are keyed on it -- but
 * on an anonymous mapping they read back as ZEROS. Any host consumer that
 * reads the payload after the drop would therefore compute a well-formed,
 * entirely wrong result.
 *
 * This registry is what lets those consumers refuse instead. It is written
 * only by the dropping backend and read on the host GEMM path; when nothing
 * has ever been dropped (the default on every backend that does not drop) the
 * query is a single acquire load, so the check costs nothing on the normal
 * path.
 *
 * @param base  first byte of the plain payload (nibbles)
 * @param bytes payload length (nibbles + fp32 scale tail)
 */
void markQs4cxPayloadDropped(const void *base, size_t bytes);

/**
 * @brief Has any QS4CX plain payload been dropped in this process?
 * @note Cheap fast-path guard for the containment query below.
 */
bool anyQs4cxPayloadDropped();

/**
 * @brief Does @a ptr fall inside a QS4CX payload whose pages were dropped?
 * @param ptr candidate payload base
 * @retval true reading through @a ptr may return released pages
 */
bool isQs4cxPayloadDropped(const void *ptr);

/**
 * @brief Forget the dropped range that starts at @a base.
 *
 * Called from the payload deleter: the ranges are keyed by host address, and
 * the allocator hands a freed address to the next weight (a model reloaded
 * into the same process lands its weights on the previous model's heap
 * blocks). Without this a fresh, fully readable payload would be refused as
 * "dropped" the moment it recycled a released one's address. No-op for an
 * address that was never marked.
 *
 * @param base first byte of the payload that is being freed
 */
void unmarkQs4cxPayloadDropped(const void *base);

/**
 * @brief Refuse a host read of a QS4CX payload whose pages were released.
 *
 * The one guard every host consumer of the plain payload calls before it
 * reads the nibbles. It throws instead of returning a flag so that no caller
 * can forget to act on the answer: a released payload reads back as zeros on
 * an anonymous mapping, and a GEMM over zeros finishes with rc = 0 and a
 * plausible-looking wrong answer, which no automated gate catches.
 *
 * @param payload the payload base the caller is about to read
 * @param who     the consumer, for the message (e.g. "FloatTensor::dotQs4cx")
 * @throws std::runtime_error when @a payload lies in a dropped range
 */
void refuseIfQs4cxPayloadDropped(const void *payload, const char *who);

/**
 * @brief Resolved NNTR_QS4CX_HEAP_BYPASS: does a QS4CX weight get its own
 *        heap allocation instead of a slice of the shared weight pool?
 *
 * Set in the environment, the value wins (=0 opts out, anything else opts
 * in). Unset, the default is ON on Android and OFF elsewhere. On Android the
 * weight pool is an OpenCL SVM arena that the kgsl driver maps and pins for
 * the GPU, so a QS4CX weight placed there stays resident on the GPU for the
 * life of the process even though the v8c kernels only ever read the device
 * repack built from it: a full second copy of every FC weight. A self-owned
 * heap payload is the only shape whose pages can be given back after that
 * repack (see NNTR_V8C_DROP_PLAIN). Elsewhere the bypass stays opt-in, as the
 * SDK bundles that were measured with it already set the variable.
 *
 * Read once; the first call latches the answer for the process.
 */
bool qs4cxHeapBypassOn();

/**
 * @brief Resolved NNTR_V8C_DROP_PLAIN: is the plain QS4CX payload dead once
 *        the device repack has been built from it -- released after the
 *        build, or, on the zero-copy load, never filled at all?
 *
 * Set in the environment, the value wins (=1 on, anything else off). Unset,
 * it follows qs4cxHeapBypassOn() on Android -- a heap payload is the only
 * shape whose release is real there -- and is OFF elsewhere, where the SDK
 * bundles that were measured with it set the variable themselves.
 *
 * The drop after the v8c build and the default of the zero-copy load
 * (NNTR_LOAD_ZEROCOPY) both ask this one function, so the two cannot
 * disagree. Read once; the first call latches the answer for the process.
 */
bool qs4cxDropPlainOn();

/**
 * @brief The heap-bypass decision for an int4 FC weight of type @a dt.
 *
 * QS4CX: qs4cxHeapBypassOn(). QS4ACX: NNTR_QS4CX_HEAP_BYPASS when set, else
 * ON on every platform -- a QS4ACX weight is read by the v8c device repack
 * only (the host GEMM is its correctness reference), so a pool slice of it is
 * a second copy of every such weight for the life of the process. Any other
 * type: false.
 */
bool int4HeapBypassOn(ml::train::TensorDim::DataType dt);

/**
 * @brief The drop-after-repack decision for an int4 FC weight of type @a dt.
 *
 * QS4CX: qs4cxDropPlainOn(). QS4ACX: NNTR_V8C_DROP_PLAIN when set, else ON
 * (see int4HeapBypassOn). Any other type: false.
 */
bool int4DropPlainOn(ml::train::TensorDim::DataType dt);

/**
 * @class   QS4CX_WH_Tensor
 * @brief   QS4CX whose nibbles are already in the HMX WH tile layout.
 *
 * Same values, same per-output-channel scales, same number of nibble bytes --
 * a WH tile is 32x32 i4 in 512 bytes, so the packing costs nothing over
 * row-major. Two things differ:
 *
 *  - the nibbles are arranged as htp_wh_layout.h describes, which is what the
 *    HTP backend registers directly instead of converting and baking on the
 *    DSP (doc 46 section 35);
 *  - a per-output-channel column sum follows the scales, because the matmul
 *    needs it to correct for the activation zero point and recomputing it
 *    from the packed nibbles costs about 20 seconds across this model.
 *
 * So the only layout difference is the extra N floats, which is why getData
 * and getScale are inherited unchanged. Nothing on the CPU can read this
 * tensor: the layout is the accelerator's, and a CPU kernel handed one would
 * compute a wrong answer rather than fail, so loading a model with these
 * weights commits it to the HTP path.
 */
class QS4CX_WH_Tensor : public QS4CX_Tensor {
public:
  /** @brief Basic constructor */
  QS4CX_WH_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW) :
    QS4CX_Tensor(name_, fm) {}

  /**
   * @brief Construct a new QS4CX_WH_Tensor
   *
   * Allocation is deferred out of the base constructor and done here.
   * QS4CX_Tensor::allocate() sizes the buffer with size(), and a virtual call
   * from a base constructor runs the BASE override -- which would quietly
   * allocate N floats too few and leave every column sum reading past the end.
   */
  QS4CX_WH_Tensor(const TensorDim &d, bool alloc_now,
                  Initializer init = Initializer::NONE, std::string name = "") :
    QS4CX_Tensor(d, false, init, name) {
    if (alloc_now)
      allocate();
  }

  /** @brief Construct from a buffer */
  QS4CX_WH_Tensor(const TensorDim &d, const void *buf = nullptr) :
    QS4CX_WH_Tensor(d, true, Initializer::NONE, "") {
    if (d.getDataLen() != 0 && buf != nullptr)
      copy_qs4cx(buf);
  }

  /** @brief Copy constructor from TensorBase */
  QS4CX_WH_Tensor(TensorBase &rhs) : QS4CX_Tensor(rhs) {}

  /**
   * @copydoc Tensor::size()
   * @note The nibble half matches QS4CX exactly; the extra N floats are the
   *       column sums.
   */
  size_t size() const override {
    return QS4CX_Tensor::size() + width() * sizeof(float);
  }

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override { return size() * sizeof(uint8_t); }

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS4CX_WH)
   */
  std::string getStringDataType() const override { return "QS4CX_WH"; }
};

/** WH int4 with an integer affine offset per output channel.
 * W[k,n] = scale[n] * (q[k,n] + offset[n]). The offset tail is explicit
 * so a symmetric WH reader cannot silently misinterpret this record. */
class QS4CX_WH_AFFINE_Tensor : public QS4CX_WH_Tensor {
public:
  QS4CX_WH_AFFINE_Tensor(std::string name = "", Tformat fm = Tformat::NCHW) :
    QS4CX_WH_Tensor(name, fm) {}
  QS4CX_WH_AFFINE_Tensor(const TensorDim &d, bool alloc_now,
                         Initializer init = Initializer::NONE,
                         std::string name = "") :
    QS4CX_WH_Tensor(d, false, init, name) {
    if (alloc_now)
      allocate();
  }
  QS4CX_WH_AFFINE_Tensor(const TensorDim &d, const void *buf = nullptr) :
    QS4CX_WH_AFFINE_Tensor(d, true, Initializer::NONE, "") {
    if (d.getDataLen() != 0 && buf)
      copy_qs4cx(buf);
  }
  QS4CX_WH_AFFINE_Tensor(TensorBase &rhs) : QS4CX_WH_Tensor(rhs) {}
  size_t size() const override {
    // WH records contain whole tiles and never QS4CX's legacy row padding.
    return height() * width() / 2 + 3 * width() * sizeof(uint32_t);
  }
  size_t getMemoryBytes() const override { return size(); }
  void *getScale() const override {
    if (!data)
      return nullptr;
    data->validate();
    return static_cast<uint8_t *>(getData()) + height() * width() / 2;
  }
  std::string getStringDataType() const override { return "QS4CX_WH_AFFINE"; }
};

/**
 * @class QS2CX_WH_Tensor
 * @brief QS4CX_WH at two bits a weight
 *
 * Same tile order, same per-output-channel scale, same column sums -- only
 * the codes are narrower. Each code indexes one of four int4 values in a
 * palette that covers the whole tensor, and the HTP expands the codes back
 * to the int4 lattice in VTCM before the matmul, so everything downstream
 * of the matmul is bit-identical to QS4CX_WH (htp_wh_palette.h).
 *
 * Layout: [K*N/4 codes][4 palette bytes][N f32 scales][N f32 column sums].
 *
 * The palette is per tensor rather than per k-group because doc 54 5.3
 * measured group size against the real checkpoint and it changed the SNR by
 * nothing -- the code histogram has the same shape everywhere, so the fit
 * picks the same four levels. A recipe that makes groups differ (GPTQ would)
 * needs a group axis here; this one does not, and carrying an axis nothing
 * varies along would be carried into the DSP expansion as a table swap.
 */
class QS2CX_WH_Tensor : public QS4CX_WH_Tensor {
public:
  /** @brief Palette entries stored after the codes */
  static constexpr size_t PALETTE_BYTES = 4;

  /** @brief Basic constructor */
  QS2CX_WH_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW) :
    QS4CX_WH_Tensor(name_, fm) {}

  /**
   * @brief Construct a new QS2CX_WH_Tensor
   * @note Allocation is deferred for the reason QS4CX_WH_Tensor documents:
   *       a virtual size() called from a base constructor runs the base
   *       override and under-allocates.
   */
  QS2CX_WH_Tensor(const TensorDim &d, bool alloc_now,
                  Initializer init = Initializer::NONE, std::string name = "") :
    QS4CX_WH_Tensor(d, false, init, name) {
    if (alloc_now)
      allocate();
  }

  /** @brief Construct from a buffer */
  QS2CX_WH_Tensor(const TensorDim &d, const void *buf = nullptr) :
    QS2CX_WH_Tensor(d, true, Initializer::NONE, "") {
    if (d.getDataLen() != 0 && buf != nullptr)
      copy_qs4cx(buf);
  }

  /** @brief Copy constructor from TensorBase */
  QS2CX_WH_Tensor(TensorBase &rhs) : QS4CX_WH_Tensor(rhs) {}

  /**
   * @copydoc Tensor::size()
   */
  size_t size() const override {
    const size_t K = height();
    const size_t N = width();
    return N * ((K + 3) / 4) + PALETTE_BYTES + 2 * width() * sizeof(float);
  }

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override { return size() * sizeof(uint8_t); }

  /**
   * @copydoc QS4CX_Tensor::getScale()
   * @note The palette sits between the codes and the scales, so the base
   *       class's offset would land four bytes short.
   */
  void *getScale() const override {
    if (!data)
      return nullptr;
    data->validate();
    const size_t K = height();
    const size_t N = width();
    return ((int8_t *)getData()) + N * ((K + 3) / 4) + PALETTE_BYTES;
  }

  /** @brief The four int4 codes the 2-bit indices name, ascending */
  const int8_t *getPalette() const {
    if (!data)
      return nullptr;
    data->validate();
    const size_t K = height();
    const size_t N = width();
    return ((const int8_t *)getData()) + N * ((K + 3) / 4);
  }

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QS2CX_WH)
   */
  std::string getStringDataType() const override { return "QS2CX_WH"; }
};

/** Native ternary TL2 payload followed by N fp32 scales and N fp32 sums.
 * The palette is implicitly {-1,0,1}; there is no QS2 companion on disk. */
class TL2CX_WH_Tensor : public QS4CX_WH_Tensor {
public:
  TL2CX_WH_Tensor(std::string name = "", Tformat fm = Tformat::NCHW) :
    QS4CX_WH_Tensor(name, fm) {}
  TL2CX_WH_Tensor(const TensorDim &d, bool alloc_now,
                  Initializer init = Initializer::NONE, std::string name = "") :
    QS4CX_WH_Tensor(d, false, init, name) {
    if (alloc_now)
      allocate();
  }
  TL2CX_WH_Tensor(const TensorDim &d, const void *buf = nullptr) :
    TL2CX_WH_Tensor(d, true, Initializer::NONE, "") {
    if (d.getDataLen() && buf)
      copy_qs4cx(buf);
  }
  TL2CX_WH_Tensor(TensorBase &rhs) : QS4CX_WH_Tensor(rhs) {}
  static size_t codeBytes(size_t K, size_t N) {
    return ((K + 23) / 24) * ((N + 127) / 128) * 640;
  }
  size_t size() const override {
    return codeBytes(height(), width()) + 2 * width() * sizeof(float);
  }
  size_t getMemoryBytes() const override { return size(); }
  void *getScale() const override {
    if (!data)
      return nullptr;
    data->validate();
    return static_cast<uint8_t *>(getData()) + codeBytes(height(), width());
  }
  std::string getStringDataType() const override { return "TL2CX_WH"; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QS4CX_TENSOR_H__ */
