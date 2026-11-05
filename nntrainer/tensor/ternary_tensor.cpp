// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ternary_tensor.cpp
 * @date	02 October 2026
 * @brief	TernaryTensor -- the TQ2CX ternary weight type.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @note	SCAFFOLDING. The dot below is the scalar correctness oracle, not
 *              a production kernel: no blocking, no SIMD, no threading. See
 *              ternary_tensor.h for the layout and the GEMM contract a tuned
 *              kernel has to match.
 */

#include <ternary_tensor.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include <cpu_backend.h>
#include <qs4cx_tensor.h> // the released-payload registry
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

TernaryTensor::TernaryTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

TernaryTensor::TernaryTensor(const TensorDim &d, bool alloc_now,
                             Initializer init, std::string name) :
  TensorBase(d, false, init, name) {
  /// @note batch stays 1 and experts stack on CHANNEL: nothing in the graph
  /// may resize a weight's batch, and keeping width() == output channel
  /// preserves the QS4CX convention the scale axis is defined against.
  NNTR_THROW_IF(d.batch() != 1, std::invalid_argument)
    << "TernaryTensor must have batch size 1; stack experts on the channel "
       "axis ([1, E, IN, OUT])";

  if (alloc_now)
    allocate();
  offset = 0;
}

TernaryTensor::TernaryTensor(const TensorDim &d, const void *buf) :
  TernaryTensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_tq2cx(buf);
  }
}

void TernaryTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
    // Single copy: the payload is load-destined (filled in full by its
    // reader) or never touched at all (built into a device copy straight from
    // the file), so a zero fill would only fault every page resident.
    const bool uninit = tq2cxSingleCopyOn();
    MemoryData *mem_data = new MemoryData(
      uninit ? (void *)(new uint8_t[size()]) : (void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      unmarkQs4cxPayloadDropped(mem_data->template getAddr<uint8_t>());
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });
    offset = 0;
    if (uninit)
      putData();
    else
      initialize();
  }
}

namespace {
/** R and V of the stored layout, packed: R << 16 | V (R = 0: row-major) */
std::atomic<uint32_t> &tq2cxLayoutWord() {
  static std::atomic<uint32_t> w{0u << 16 | 1u};
  return w;
}
} // namespace

Tq2cxLayout tq2cxStoredLayout() {
  const uint32_t w = tq2cxLayoutWord().load(std::memory_order_relaxed);
  return Tq2cxLayout{w >> 16, w & 0xFFFFu};
}

void setTq2cxStoredLayout(const Tq2cxLayout &l) {
  NNTR_THROW_IF(l.R > 0xFFFFu || l.V == 0u || l.V > 0xFFFFu,
                std::invalid_argument)
    << "setTq2cxStoredLayout: bad layout R " << l.R << " V " << l.V;
  tq2cxLayoutWord().store(l.R << 16 | l.V, std::memory_order_relaxed);
}

void tq2cxRelayout(const Tq2cxLayout &from, const Tq2cxLayout &to,
                   const void *src, void *dst, size_t N, size_t K) {
  const size_t U = tq2cxUnits(K);
  const auto *s = static_cast<const TQ2CX_MaskUnit *>(src);
  auto *d = static_cast<TQ2CX_MaskUnit *>(dst);
  std::memset(dst, 0, tq2cxRecordBytes(to, N, K));
  for (size_t n = 0; n < N; ++n)
    for (size_t u = 0; u < U; ++u)
      d[tq2cxUnitIndex(to, n, u, U)] = s[tq2cxUnitIndex(from, n, u, U)];
  std::memcpy(static_cast<uint8_t *>(dst) + tq2cxMaskBytes(to, N, K),
              static_cast<const uint8_t *>(src) + tq2cxMaskBytes(from, N, K),
              N * sizeof(float));
}

bool tq2cxSingleCopyOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_TQ2CX_SINGLE_COPY");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

void refuseIfTq2cxPayloadReleased(const void *payload, const char *who) {
  if (!isQs4cxPayloadDropped(payload))
    return;
  throw std::runtime_error(
    std::string("[") + (who ? who : "TQ2CX") +
    "] refusing to read a TQ2CX weight whose host payload was released (or "
    "never filled) because its device copy owns it: the bytes are not the "
    "weight. A host consumer reached an expert the device path was expected "
    "to own; run with NNTR_TQ2CX_SINGLE_COPY=0 to keep the payload resident");
}

void *TernaryTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

size_t TernaryTensor::size() const {
  /**
   * @note BYTES, not an element count. TensorDim::getDataTypeSize() is 1 for
   * TQ2CX, so TensorBase::bytes() == size(), which is what save()/read() and
   * the loader's record-offset accumulation consume. There is exactly one
   * stride expression and it is a pure function of the dims -- the property
   * QS4CX lacks (qs4cx_tensor.h:204-220).
   */
  return totalBytes(channel(), height(), width());
}

size_t TernaryTensor::getMemoryBytes() const {
  return size() * sizeof(uint8_t);
}

size_t TernaryTensor::scale_size() const { return channel() * width(); }

const uint8_t *TernaryTensor::expertRecord(size_t e) const {
  NNTR_THROW_IF(e >= channel(), std::invalid_argument)
    << "TernaryTensor::expertRecord(): expert " << e << " >= " << channel();
  return reinterpret_cast<const uint8_t *>(getData()) +
         e * recordBytes(height(), width());
}

const TQ2CX_MaskUnit &TernaryTensor::maskUnit(size_t e, size_t out,
                                              size_t u) const {
  NNTR_THROW_IF(out >= width(), std::invalid_argument)
    << "TernaryTensor::maskUnit(): output channel " << out << " >= " << width();
  const size_t U = unitsPerRow(height());
  return reinterpret_cast<const TQ2CX_MaskUnit *>(
    expertRecord(e))[tq2cxUnitIndex(tq2cxStoredLayout(), out, u, U)];
}

const float *TernaryTensor::getScaleOfExpert(size_t e) const {
  return reinterpret_cast<const float *>(expertRecord(e) +
                                         maskBytes(height(), width()));
}

void *TernaryTensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  /// @note Expert 0's tail only. A stacked tensor's scales are NOT contiguous
  /// -- each expert carries its own tail so that an expert record is a pure
  /// subrange. Use getScaleOfExpert().
  return const_cast<float *>(getScaleOfExpert(0));
}

void *TernaryTensor::getScale(size_t idx) const {
  NNTR_THROW_IF(idx >= scale_size(), std::invalid_argument)
    << "TernaryTensor::getScale(): index " << idx << " >= " << scale_size();

  if (!data)
    return nullptr;

  data->validate();
  const size_t out_dim = width();
  return const_cast<float *>(getScaleOfExpert(idx / out_dim) + (idx % out_dim));
}

float TernaryTensor::decode(size_t e, size_t out, size_t in) const {
  NNTR_THROW_IF(in >= height(), std::invalid_argument)
    << "TernaryTensor::decode(): input tap " << in << " >= " << height();

  const TQ2CX_MaskUnit &unit = maskUnit(e, out, in / TQ2CX_TAPS_PER_UNIT);
  const uint32_t bit = 1u << (in % TQ2CX_TAPS_PER_UNIT);

  const int code = ((unit.pos & bit) ? 1 : 0) - ((unit.neg & bit) ? 1 : 0);
  return static_cast<float>(code) * getScaleOfExpert(e)[out];
}

bool TernaryTensor::validate() const {
  if (!data)
    return true;

  const size_t E = channel(), in_dim = height(), out_dim = width();
  const size_t units = unitsPerRow(in_dim);
  const size_t rem = in_dim % TQ2CX_TAPS_PER_UNIT;
  // Bits of the last unit that are NOT real taps must be clear in both planes.
  const uint32_t tail_mask =
    (rem == 0) ? 0u : static_cast<uint32_t>(~((1ull << rem) - 1ull));

  for (size_t e = 0; e < E; ++e) {
    for (size_t n = 0; n < out_dim; ++n) {
      for (size_t u = 0; u < units; ++u) {
        const TQ2CX_MaskUnit &w = maskUnit(e, n, u);
        // (1,1) is not a code.
        if ((w.pos & w.neg) != 0u)
          return false;
        if (u + 1 == units && tail_mask != 0u &&
            ((w.pos | w.neg) & tail_mask) != 0u)
          return false;
      }
    }
  }
  return true;
}

void TernaryTensor::setZero() {
  uint8_t *d = (uint8_t *)getData();
  std::fill(d, d + size(), 0);
}

void TernaryTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

void TernaryTensor::copy_tq2cx(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData())
    return;
  std::memcpy(getData(), buf, size());
}

float TernaryTensor::max_abs() const {
  // The largest |w| is the largest scale that is actually reachable, i.e. the
  // largest scale on a channel that has at least one nonzero code.
  float m = 0.0f;
  const size_t E = channel(), out_dim = width(), units = unitsPerRow(height());
  for (size_t e = 0; e < E; ++e) {
    const float *s = getScaleOfExpert(e);
    for (size_t n = 0; n < out_dim; ++n) {
      bool nonzero = false;
      for (size_t u = 0; u < units && !nonzero; ++u) {
        const TQ2CX_MaskUnit &w = maskUnit(e, n, u);
        nonzero = (w.pos | w.neg) != 0u;
      }
      if (nonzero)
        m = std::max(m, std::fabs(s[n]));
    }
  }
  return m;
}

float TernaryTensor::maxValue() const {
  // A ternary channel with scale s spans {-|s|, 0, +|s|} whenever it has a
  // code of each sign, so the max over the tensor is max_abs() unless every
  // code is zero -- and zero is always representable, so 0 is the floor.
  return max_abs();
}

float TernaryTensor::minValue() const { return -max_abs(); }

void TernaryTensor::print(std::ostream &out) const {
  out << "data addr: " << getData() << '\n';
  out << dim;
  out << "[TQ2CX] experts=" << channel() << " IN=" << height()
      << " OUT=" << width() << " units/row=" << unitsPerRow(height())
      << " layout=" << tq2cxLayoutName(tq2cxStoredLayout())
      << " record=" << recordBytes(height(), width()) << "B total=" << size()
      << "B\n";
  out << "[TQ2CX data print skipped]" << std::endl;
}

QScheme TernaryTensor::q_scheme() const { return QScheme::TQ2CX; }

Tensor &TernaryTensor::dot(Tensor const &input, Tensor &output, bool trans,
                           bool trans_in, float beta) const {
  NNTR_THROW_IF(trans || trans_in, std::invalid_argument)
    << "[TernaryTensor::dot] a TQ2CX weight is already stored as the "
       "transposed rhs ([OUT][IN]); trans and trans_in must both be false";
  NNTR_THROW_IF(channel() != 1, std::invalid_argument)
    << "[TernaryTensor::dot] a stacked (E > 1) TQ2CX weight has no single "
       "operand: route one expert at a time through tq2cx_gemm_ref_fp32() "
       "with expertRecord(e)";
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32 ||
                  output.getDataType() != Tdatatype::FP32,
                std::invalid_argument)
    << "[TernaryTensor::dot] reference kernel takes FP32 activations only";

  const size_t K = height();
  const size_t N = width();
  const size_t M = input.getDim().height();

  NNTR_THROW_IF(input.getDim().width() != K, std::invalid_argument)
    << "[TernaryTensor::dot] activation width " << input.getDim().width()
    << " != weight IN " << K;
  NNTR_THROW_IF(output.getDim().width() != N, std::invalid_argument)
    << "[TernaryTensor::dot] output width " << output.getDim().width()
    << " != weight OUT " << N;

  tq2cx_gemm_ref_fp32(M, N, K, input.getData<float>(), K, getData(),
                      output.getData<float>(), N, beta);
  return output;
}

/*=============================== free kernels =============================*/

void tq2cx_gemm_ref_fp32(size_t M, size_t N, size_t K, const float *lhs,
                         size_t lda, const void *record, float *out, size_t ldc,
                         float beta, Tq2cxLayout lay) {
  const size_t units = TernaryTensor::unitsPerRow(K);
  const TQ2CX_MaskUnit *masks =
    reinterpret_cast<const TQ2CX_MaskUnit *>(record);
  const float *scales =
    reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(record) +
                                    TernaryTensor::maskBytes(K, N, lay));

  for (size_t n = 0; n < N; ++n) {
    const float s = scales[n];
    for (size_t m = 0; m < M; ++m) {
      const float *x = lhs + m * lda;
      float acc = 0.0f;
      for (size_t u = 0; u < units; ++u) {
        const TQ2CX_MaskUnit &w = masks[tq2cxUnitIndex(lay, n, u, units)];
        const uint32_t p = w.pos;
        const uint32_t g = w.neg;
        // Skip a word that selects nothing -- ~49% of them at the measured
        // 25.4/49.2/25.3 code distribution are not empty, but a fully empty
        // 32-tap window is common enough to be worth the branch even in the
        // reference.
        if ((p | g) == 0u)
          continue;
        const size_t base = u * TQ2CX_TAPS_PER_UNIT;
        const size_t lim = std::min<size_t>(TQ2CX_TAPS_PER_UNIT, K - base);
        for (size_t b = 0; b < lim; ++b) {
          const uint32_t bit = 1u << b;
          if (p & bit)
            acc += x[base + b];
          if (g & bit)
            acc -= x[base + b];
        }
      }
      float *y = out + m * ldc + n;
      *y = s * acc + beta * (*y);
    }
  }
}

void tq2cx_pack_ref_fp32(size_t N, size_t K, const float *src, size_t lds,
                         void *record, Tq2cxLayout lay) {
  const size_t units = TernaryTensor::unitsPerRow(K);
  TQ2CX_MaskUnit *masks = reinterpret_cast<TQ2CX_MaskUnit *>(record);
  float *scales = reinterpret_cast<float *>(
    reinterpret_cast<uint8_t *>(record) + TernaryTensor::maskBytes(K, N, lay));

  std::memset(record, 0, TernaryTensor::recordBytes(K, N, lay));

  for (size_t n = 0; n < N; ++n) {
    const float *row = src + n * lds;

    float scale = 0.0f;
    for (size_t k = 0; k < K; ++k)
      scale = std::max(scale, std::fabs(row[k]));
    scales[n] = scale;

    if (scale == 0.0f)
      continue; // all-zero channel: both planes stay clear

    for (size_t k = 0; k < K; ++k) {
      const int code = static_cast<int>(std::lround(row[k] / scale));
      const uint32_t bit = 1u << (k % TQ2CX_TAPS_PER_UNIT);
      TQ2CX_MaskUnit &u =
        masks[tq2cxUnitIndex(lay, n, k / TQ2CX_TAPS_PER_UNIT, units)];
      if (code > 0)
        u.pos |= bit;
      else if (code < 0)
        u.neg |= bit;
      // code == 0 -> both planes stay clear
    }
  }
}

} // namespace nntrainer
