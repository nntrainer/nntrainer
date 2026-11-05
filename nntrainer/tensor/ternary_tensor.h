// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ternary_tensor.h
 * @date	02 October 2026
 * @brief	TernaryTensor -- the TQ2CX ternary weight type (two 1-bit planes
 *              plus one fp32 scale per output channel).
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @details
 * TQ2CX stores a weight whose values are exactly {-s, 0, +s} for a
 * per-output-channel scale s. Instead of a 2-bit code per weight it keeps TWO
 * plain 1-bit planes over the same index space:
 *
 *   plane_pos bit set  <=>  code == +1
 *   plane_neg bit set  <=>  code == -1
 *   both clear         <=>  code ==  0
 *   both set           <=>  ILLEGAL (never produced; rejected by validate())
 *
 *   w[out][in] = scale[out] * (pos[out][in] - neg[out][in])
 *
 * The point of two plain masks rather than a {-1,+1} alphabet (which is what
 * BCQ uses, see bcq_tensor.h) is that the inner product needs no multiply and
 * no table:
 *
 *   acc = sum(x[in] : pos bit set) - sum(x[in] : neg bit set)
 *   y[out] = scale[out] * acc
 *
 * i.e. a masked add/subtract over the activation row. On AVX-512 a 32-bit mask
 * word is exactly two __mmask16 operands for two vaddps/vsubps; on NEON it is
 * four 4-lane predicated adds.
 *
 * ---------------------------------------------------------------------------
 * ON-DISK / IN-MEMORY LAYOUT  (one single byte stream, no header, no version)
 * ---------------------------------------------------------------------------
 * Dimension mapping. The canonical weight is [OUT, IN] row major -- OUT rows
 * of IN input taps -- which is the SAME storage convention QS4CX uses (see
 * qs4cx_tensor.cpp:239-241 and the layout note at half_tensor.cpp:778-783: the
 * TensorDim is the logical FC weight [K=in, N=out] while the bytes are the
 * transpose [N][K]). So:
 *
 *   height() == IN  == K   (input taps, the contracted dimension)
 *   width()  == OUT == N   (output channels, the per-channel scale axis)
 *   channel()== E          (expert count; 1 for a plain FC weight)
 *   batch()  == 1          (enforced)
 *
 * Expert stacking lives on CHANNEL, not batch, so that batch stays 1 for every
 * weight (nothing in the graph may resize a weight's batch) and so that
 * width() keeps meaning "output channel" exactly as it does for QS4CX.
 *
 * Word and unit. One MaskUnit = { uint32 pos; uint32 neg; } covers 32
 * consecutive input taps. uint32 (not uint64) because the consumer of a mask
 * word is a SIMD predicate register -- 32 bits is two AVX-512 __mmask16 or
 * eight NEON 4-lane groups -- and because 4-byte alignment is trivially
 * satisfied on an mmap'd record while 8-byte is not. Interleaving pos with neg
 * (rather than a pos plane followed by a neg plane) is deliberate: both masks
 * for the same 32 activations arrive in ONE 8-byte load on one sequential
 * stream. The kernel is activation-bandwidth bound anyway -- 8 bytes of weight
 * per 128 bytes of activation -- so there is nothing to gain from being able
 * to vector-load a run of pos words.
 *
 * Row stride. No padding: a row is exactly ceil(IN/32) MaskUnits, which is
 * already a multiple of 8 bytes. FORMAT INVARIANT: when IN % 32 != 0 the
 * unused high bits of the last unit of every row MUST be zero in BOTH planes,
 * so the kernel never needs a tail mask. validate() checks it.
 *
 * Stored layout. Where unit (out, u) sits inside a record is the stored
 * layout (tq2cx_layout.h): "rm" (v1) keeps the rows one after another,
 * "t2r16v2" (v2) interleaves them in 16-row tiles of 2-unit blocks and pads
 * the rows to a whole tile. The process runs one layout,
 * tq2cxStoredLayout(), and every size and index below goes through it.
 *
 * Record. The per-expert record is the whole 2D weight followed by its own
 * fp32 scale tail, and the stacked tensor is just those records concatenated:
 *
 *   recordBytes(IN, OUT) = tq2cxRecordBytes(layout, OUT, IN)
 *                        = OUT * rowBytes(IN) + OUT * sizeof(float)  ("rm")
 *   size()               = E * recordBytes(IN, OUT)
 *
 *   expert 0 : [ OUT rows x ceil(IN/32) MaskUnits ][ OUT fp32 scales ]
 *   expert 1 : [ OUT rows x ceil(IN/32) MaskUnits ][ OUT fp32 scales ]
 *   ...
 *
 * Two consequences this layout was chosen FOR:
 *  - expert e's record is byte-identical to a standalone single-expert TQ2CX
 *    record, so the offline packer emits one format and a runtime that wants
 *    per-expert 2D weights (which is what nntrainer's MoE layers actually
 *    request -- see gemma4_moe_layer.cpp:477-503 in PR #4296, 3 weights per
 *    expert at context.getWeightDataType()) slices it with pointer arithmetic
 *    only;
 *  - there is exactly ONE stride expression, derived from the dims alone. That
 *    is a direct reaction to QS4CX, whose padded-vs-trimmed ambiguity
 *    (qs4cx_tensor.h:204-220, and the per-tensor setQs4cxRecordPadded() flag
 *    plus two-pass file walk it forced at qs4cx_tensor.cpp:305-325) exists
 *    only because its record size was not a pure function of its dims.
 *
 * Scales sit in the TAIL of each record, following QS4CX (nibbles then fp32
 * scales, qs4cx_tensor.cpp:329-341), which keeps the "read only the scale
 * tail" zero-copy load optimization (qs4cx_tensor.cpp:404-419) available.
 */

#ifndef __TERNARY_TENSOR_H__
#define __TERNARY_TENSOR_H__
#ifdef __cplusplus

#include <cstdint>

#include <quantizer.h>
#include <tensor_base.h>
#include <tq2cx_layout.h>

namespace nntrainer {

/**
 * @brief Input taps covered by one MaskUnit.
 */
constexpr size_t TQ2CX_TAPS_PER_UNIT = 32;

/**
 * @brief One interleaved (pos, neg) mask pair covering TQ2CX_TAPS_PER_UNIT
 *        consecutive input taps of one output channel.
 * @note  Bit b of @a pos / @a neg is input tap (unit_index * 32 + b).
 *        (pos & neg) must be 0: the (1,1) combination is not a code.
 */
struct TQ2CX_MaskUnit {
  uint32_t pos; /**< bit set where the ternary code is +1 */
  uint32_t neg; /**< bit set where the ternary code is -1 */
};

static_assert(sizeof(TQ2CX_MaskUnit) == 8,
              "TQ2CX_MaskUnit must be exactly 8 bytes: it is the on-disk unit");

/**
 * @class TernaryTensor
 * @brief TQ2CX ternary weight tensor
 */
class TernaryTensor : public TensorBase {

public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  TernaryTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new TernaryTensor object
   *
   * @param d Tensor dim for this ternary tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  TernaryTensor(const TensorDim &d, bool alloc_now,
                Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new TernaryTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  TernaryTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new TernaryTensor object
   * @param rhs TensorBase object to copy
   */
  TernaryTensor(TensorBase &rhs) : TensorBase(rhs) {}

  /*======================= layout arithmetic (static) =====================*/

  /**
   * @brief Mask units in one output channel's row.
   * @param in_dim input taps (height())
   */
  static size_t unitsPerRow(size_t in_dim) {
    return (in_dim + TQ2CX_TAPS_PER_UNIT - 1) / TQ2CX_TAPS_PER_UNIT;
  }

  /**
   * @brief Bytes of one output channel's row of mask units.
   * @param in_dim input taps (height())
   * @note Always a multiple of 8; there is no row padding.
   */
  static size_t rowBytes(size_t in_dim) {
    return unitsPerRow(in_dim) * sizeof(TQ2CX_MaskUnit);
  }

  /**
   * @brief Bytes of the mask half of ONE expert's record, i.e. the offset of
   *        that record's fp32 scale tail.
   * @param in_dim input taps (height())
   * @param out_dim output channels (width())
   * @param lay stored layout
   */
  static size_t maskBytes(size_t in_dim, size_t out_dim,
                          Tq2cxLayout lay = tq2cxStoredLayout()) {
    return tq2cxMaskBytes(lay, out_dim, in_dim);
  }

  /**
   * @brief Bytes of ONE expert's whole record: masks + OUT fp32 scales (+ the
   *        64-byte padding of a tiled layout).
   * @param in_dim input taps (height())
   * @param out_dim output channels (width())
   * @param lay stored layout
   * @note This is also the byte size of a standalone single-expert TQ2CX
   *       weight, which is what makes an expert slice a pure subrange.
   */
  static size_t recordBytes(size_t in_dim, size_t out_dim,
                            Tq2cxLayout lay = tq2cxStoredLayout()) {
    return tq2cxRecordBytes(lay, out_dim, in_dim);
  }

  /**
   * @brief Bytes of a whole TQ2CX tensor, scales included. Exposed statically
   *        so a loader can size a record before it has built the tensor (the
   *        role QS4CX_Tensor::recordBytes plays for NeuralNetwork::load).
   * @param E   expert count (channel())
   * @param in_dim input taps (height())
   * @param out_dim output channels (width())
   * @param lay stored layout
   */
  static size_t totalBytes(size_t E, size_t in_dim, size_t out_dim,
                           Tq2cxLayout lay = tq2cxStoredLayout()) {
    return E * recordBytes(in_dim, out_dim, lay);
  }

  /*============================ TensorBase ================================*/

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
      "TernaryTensor::getData(idx) is not supported. Use getData() instead.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  void *getAddress(unsigned int i) override {
    throw std::invalid_argument(
      "TernaryTensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::getAddress()
   */
  const void *getAddress(unsigned int i) const override {
    throw std::invalid_argument(
      "TernaryTensor::getAddress() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(float value) override {
    throw std::invalid_argument("TernaryTensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setValue()
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("TernaryTensor::setValue() is not supported.");
  }

  /**
   * @copydoc Tensor::addValue()
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("TernaryTensor::addValue() is not supported.");
  }

  /**
   * @copydoc Tensor::setZero()
   * @note Zero is a representable code (both planes clear), so this really
   *       does produce an all-zero weight, scales included.
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize(Initializer init) override {
    throw std::invalid_argument(
      "TernaryTensor::initialize(Initializer) is not supported: a ternary "
      "weight is produced by an offline packer, not by an initializer.");
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
    throw std::invalid_argument("TernaryTensor::copy() is not supported.");
  }

  /**
   * @copydoc Tensor::copyData()
   */
  void copyData(const Tensor &from) override {
    throw std::invalid_argument("TernaryTensor::copyData() is not supported.");
  }

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "TernaryTensor::copy_with_stride() is not supported.");
  }

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override;

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override;

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override;

  /**
   * @copydoc TensorBase::size()
   * @note BYTES of the whole record (masks + fp32 scale tails), not an element
   *       count, because TensorDim::getDataTypeSize() is 1 for TQ2CX and
   *       TensorBase::bytes() -- which drives save()/read() and the loader's
   *       record-offset accumulation -- is size() * getDataTypeSize().
   */
  size_t size() const override;

  /**
   * @copydoc Tensor::getMemoryBytes()
   */
  size_t getMemoryBytes() const override;

  /**
   * @copydoc Tensor::scale_size()
   * @return E * OUT -- one fp32 scale per (expert, output channel)
   */
  size_t scale_size() const override;

  /**
   * @copydoc Tensor::getScale()
   * @return the fp32 scale tail of EXPERT 0's record
   * @note The scales of a stacked (E > 1) tensor are NOT one contiguous array:
   *       each expert carries its own tail. Use getScaleOfExpert().
   */
  void *getScale() const override;

  /**
   * @copydoc Tensor::getScale(size_t idx)
   * @param idx flat (expert * OUT + out) scale index
   */
  void *getScale(size_t idx) const override;

  /**
   * @copydoc Tensor::q_scheme()
   */
  QScheme q_scheme() const override;

  /*========================= TQ2CX-specific views =========================*/

  /**
   * @brief First byte of expert @a e's record.
   * @param e expert index, < channel()
   */
  const uint8_t *expertRecord(size_t e) const;

  /**
   * @brief Mask unit @a u of expert @a e, output channel @a out, wherever the
   *        stored layout puts it.
   * @param e   expert index, < channel()
   * @param out output channel, < width()
   * @param u   unit index, < unitsPerRow(height())
   */
  const TQ2CX_MaskUnit &maskUnit(size_t e, size_t out, size_t u) const;

  /**
   * @brief fp32 scale tail of expert @a e. @a width() floats follow.
   * @param e expert index, < channel()
   */
  const float *getScaleOfExpert(size_t e) const;

  /**
   * @brief Decode one weight. Reference accessor for tests and debugging; this
   *        is deliberately the slow, obvious path.
   * @param e   expert index
   * @param out output channel
   * @param in  input tap
   */
  float decode(size_t e, size_t out, size_t in) const;

  /**
   * @brief Check the two format invariants: (pos & neg) == 0 everywhere, and
   *        the unused high bits of a partial trailing unit are clear in both
   *        planes.
   * @retval true the record is a well-formed TQ2CX record
   */
  bool validate() const;

  /**
   * @copydoc Tensor::dot(Tensor const &input, Tensor &output, bool trans, bool
   *          trans_in, float beta)
   *
   * @details Weight-as-receiver form, as BCQTensor::dot is
   * (bcq_tensor.cpp:200): @a this is the TQ2CX weight and @a input is the FP32
   * activation [M, K]. The form the FC layers actually take is the opposite
   * one -- activation.dot(weight) -- which lands in FloatTensor::dot's switch;
   * both call the same reference kernel below.
   *
   * @note Reference implementation only. Single expert (channel() == 1),
   * trans == trans_in == false, FP32 activation.
   */
  Tensor &dot(Tensor const &input, Tensor &output, bool trans, bool trans_in,
              float beta) const override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy_tq2cx(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (TQ2CX)
   */
  std::string getStringDataType() const override { return "TQ2CX"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }
};

/*=============================== free kernels =============================*/

/**
 * @brief REFERENCE TQ2CX GEMM: out[M, N] = lhs[M, K] * W[K, N], where W is
 *        held as the canonical [N rows][K taps] TQ2CX record.
 *
 * @param M       activation rows
 * @param N       output channels (the record's OUT / the tensor's width())
 * @param K       input taps   (the record's IN  / the tensor's height())
 * @param lhs     activation, [M, K] row major, fp32
 * @param lda     leading dimension of @a lhs (>= K)
 * @param record  first byte of ONE expert's TQ2CX record (masks then scales)
 * @param out     output, [M, N] row major, fp32
 * @param ldc     leading dimension of @a out (>= N)
 * @param beta    out = acc + beta * out
 * @param lay     stored layout of @a record
 *
 * @details The contract a tuned kernel has to match, term for term:
 *
 *   row_units = ceil(K / 32)
 *   masks     = record
 *   scales    = record + maskBytes(K, N, lay)
 *   for n in [0, N):
 *     w[u]  = masks[tq2cxUnitIndex(lay, n, u, row_units)]
 *     s     = scales[n]
 *     for m in [0, M):
 *       acc = 0
 *       for u in [0, row_units):
 *         for b in [0, 32):
 *           k = u * 32 + b
 *           if (k >= K) break                // guaranteed dead: tail bits are
 *                                            // zero by format invariant
 *           if (w[u].pos >> b) & 1: acc += lhs[m * lda + k]
 *           if (w[u].neg >> b) & 1: acc -= lhs[m * lda + k]
 *       out[m * ldc + n] = s * acc + beta * out[m * ldc + n]
 *
 * NOTE ON ACCUMULATION ORDER. acc is an fp32 running sum over K terms with no
 * multiply, so it is NOT reassociation-safe against a different blocking: a
 * vectorized kernel that splits acc into 16 partial sums will differ from this
 * in the last bits. The unittest therefore compares against a float reference
 * that sums IN THIS ORDER for exactness, and uses a tolerance for anything
 * else.
 *
 * @note Unoptimized on purpose: scalar, no blocking, no threading. This is the
 * correctness oracle, not the production path.
 */
void tq2cx_gemm_ref_fp32(size_t M, size_t N, size_t K, const float *lhs,
                         size_t lda, const void *record, float *out, size_t ldc,
                         float beta, Tq2cxLayout lay = tq2cxStoredLayout());

/**
 * @brief REFERENCE TQ2CX packer: encode a dense [N, K] fp32 weight whose values
 *        are already ternary-with-per-row-scale into one TQ2CX record.
 *
 * @param N       output channels
 * @param K       input taps
 * @param src     dense weight, [N, K] row major, fp32
 * @param lds     leading dimension of @a src (>= K)
 * @param record  destination, TernaryTensor::recordBytes(K, N, lay) bytes
 * @param lay     stored layout to write
 *
 * @details For each row n: scale = max |src[n][*]|; code = round(src / scale)
 * clamped to {-1, 0, +1}. An all-zero row gets scale 0 and both planes clear.
 * Lossless exactly when every src[n][k] is one of {-scale, 0, +scale}, which
 * is the premise of the format; otherwise it is a nearest-of-three rounding and
 * the caller owns the error.
 *
 * @note Host-side reference, used by the unittest and usable by an offline
 * converter. It is NOT the production packer (which should read the already
 * ternary codes and scales straight out of the source checkpoint rather than
 * re-deriving them from a dense materialization).
 */
void tq2cx_pack_ref_fp32(size_t N, size_t K, const float *src, size_t lds,
                         void *record, Tq2cxLayout lay = tq2cxStoredLayout());

/**
 * @brief Resolved NNTR_TQ2CX_SINGLE_COPY: is a TQ2CX weight kept in one copy?
 *
 * On (the default; =0 opts out) a TQ2CX weight gets its own uninitialized
 * heap payload outside the shared weight pool. A backend that builds a
 * device copy of the weight at load (the OpenCL expert stack) reads the
 * record straight from the file mapping, or from this payload and then
 * releases its pages, and records the range as released (see
 * markQs4cxPayloadDropped); every host reader refuses a released payload
 * through refuseIfTq2cxPayloadReleased. Where no device copy is built (the
 * cpu engine) the payload is filled and read as before. Read once.
 */
bool tq2cxSingleCopyOn();

/**
 * @brief Throw instead of reading a TQ2CX payload whose pages were released
 *        (or never filled) because its device copy owns the weight.
 * @param payload the record base the caller is about to read
 * @param who     the consumer, for the message
 */
void refuseIfTq2cxPayloadReleased(const void *payload, const char *who);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TERNARY_TENSOR_H__ */
