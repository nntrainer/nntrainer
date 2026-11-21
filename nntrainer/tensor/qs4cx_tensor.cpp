// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs4cx_tensor.cpp
 * @date	17 June 2026
 * @brief	This is QS4CX_Tensor class for QS4CX quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@google.com>
 * @bug		No known bugs except for NYI items
 */

#include <env_compat.h>
#if defined(_WIN32)
#include <windows.h> // VirtualAlloc/VirtualFree for the pool-bypass payload
#endif

#include <cpu_backend.h>
#include <qs4cx_tensor.h>
#include <tensor.h>
#include <thread_manager.h>

#include <atomic>
#include <cstring>
#include <fp16.h>
#include <functional>
#include <int4_utils.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <util_func.h>
#include <utility>
#include <vector>

namespace nntrainer {

namespace {
/**
 * @brief Read a legacy QINT4 on-disk record (u16 qscheme header + KAI Section A
 *   or plain container) via @a do_read and transcode it losslessly to the
 *   canonical QS4CX in-memory layout (plain nibbles + fp32 scales). Shared by
 *   the std::ifstream and ReadSource read() overloads.
 */
void readLegacyQint4ToQs4cx(
  size_t N, size_t K, size_t start_offset,
  const std::function<void(char *, std::streamsize, size_t)> &do_read,
  uint8_t *out_nibbles, float *out_scales) {
  uint16_t scheme = 0;
  do_read(reinterpret_cast<char *>(&scheme), sizeof(uint16_t), start_offset);

  size_t body_bytes;
  if (scheme == static_cast<uint16_t>(QScheme::QS4CX))
    body_bytes = Int4Utils::kaiNibblePayloadBytes(N, K) + N * sizeof(uint16_t);
  else if (scheme == static_cast<uint16_t>(QScheme::PER_CHANNEL_AFFINE))
    body_bytes = Int4Utils::plainRecordPayloadBytes(N, K);
  else
    throw std::runtime_error(
      "[QS4CX_Tensor::read] unsupported legacy on-disk qscheme");

  std::vector<uint8_t> record(sizeof(uint16_t) + body_bytes);
  std::memcpy(record.data(), &scheme, sizeof(uint16_t));
  do_read(reinterpret_cast<char *>(record.data()) + sizeof(uint16_t),
          static_cast<std::streamsize>(body_bytes),
          start_offset + sizeof(uint16_t));

  Int4Utils::readLegacyQint4RecordToQs4cx(record.data(), record.size(), N, K,
                                          out_nibbles, out_scales);
}

/**
 * @brief Is this process's QS4CX payload allocation load-destined, i.e. may
 *   allocate() hand back UNINITIALIZED memory?
 *
 * The resolved heap-bypass decision (qs4cxHeapBypassOn():
 * NNTR_QS4CX_HEAP_BYPASS if set, else ON on Android and OFF elsewhere) is the
 * "self-owned weight payload about to be filled from the model file" signal:
 * Manager::requestWeights only takes the bypass branch
 * (weight_pool.request(UNMANAGED) + var->allocate()) under it, and every QS4CX
 * tensor reached that way is a weight. Any other producer that allocates a
 * QS4CX_Tensor while the decision resolves ON (for example the offline
 * quantize tool run on Android) also gets uninitialized memory and must write
 * all size() bytes itself. Both zero passes
 * allocate() used to do -- `new uint8_t[size()]{}` and initialize()->setZero()
 * -- are dead stores there: the payload is subsequently overwritten IN FULL,
 * either by TensorBase::read (bytes() == size() for QS4CX) or by copy_qs4cx.
 *
 * The win is not the two arena writes alone. Zeroing a fresh allocation FAULTS
 * EVERY PAGE RESIDENT at graph-build time, so the whole plain weight set is in
 * host RSS before the loader has read a single byte -- which puts the process'
 * residency high-water mark BEFORE the load, where no load-time or post-load
 * page drop can reach it. Leaving the allocation untouched moves each payload's
 * first touch to the read() that fills it, so a per-weight drop that runs
 * during the load actually bounds the peak instead of trimming a peak that has
 * already happened.
 *
 * TRIPWIRE. "Uninitialized is safe" holds only while EVERY reader of this
 * payload writes all size() bytes. That is true of TensorBase::read and
 * copy_qs4cx. It is NOT true of a partial transcode, so both read() overloads
 * setZero() first. Any future partial writer must do the same.
 *
 * Escape hatch: NNTR_QS4CX_ALLOC_ZERO=1 restores the old double zero-fill.
 */
bool qs4cxAllocUninitialized() {
  static const bool v =
    qs4cxHeapBypassOn() && !nntr_env_on("NNTR_QS4CX_ALLOC_ZERO");
  return v;
}

/**
 * @brief NNTR_QS4CX_ALLOC_POISON=1: fill a load-destined payload with 0x55
 *   instead of leaving it untouched. DIAGNOSTIC ONLY -- it reintroduces the
 *   full-arena write (and its page faults), so it is the opposite of what the
 *   uninitialized path is for.
 *
 * This is the discriminator that makes the tripwire testable. A large
 * `new uint8_t[]` is served by mmap, whose pages read back as ZERO, so a
 * passing run with the zero-fill removed proves nothing on its own -- an
 * unwritten byte would still read 0 and still match the old behaviour. Poison
 * it with a value the loader can never leave behind and re-run: identical
 * generated text is then positive evidence that no reader consumes a byte the
 * loader did not write.
 */
void qs4cxPoisonIfRequested(uint8_t *p, size_t n) {
  static const bool poison = nntr_env_on("NNTR_QS4CX_ALLOC_POISON");
  if (poison)
    std::memset(p, 0x55, n);
}
} // namespace

QS4CX_Tensor::QS4CX_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

QS4CX_Tensor::QS4CX_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                           std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1, std::invalid_argument)
    << "QS4CX_Tensor must be 2 dimensional tensor with batch size 1";

  if (alloc_now)
    allocate();
  offset = 0;
}

QS4CX_Tensor::QS4CX_Tensor(const TensorDim &d, const void *buf) :
  QS4CX_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_qs4cx(buf);
  }
}

void QS4CX_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
    MemoryData *mem_data;

#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
    // [pool-bypass] Under the heap bypass, back the payload with VirtualAlloc
    // instead of the CRT heap: VirtualFree(MEM_DECOMMIT) -- the only Windows
    // primitive that actually RELEASES commit charge while keeping the address
    // reservation (the derived-cache key) valid -- is legal only on
    // VirtualAlloc regions; running it on HeapAlloc pages corrupts the heap.
    // DiscardVirtualMemory (residency only) works on either. The deleter
    // releases the whole reservation.
    if (qs4cxHeapBypassOn()) {
      void *va =
        VirtualAlloc(nullptr, size(), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
      if (va != nullptr) {
        mem_data = new MemoryData(va);
        data = std::shared_ptr<MemoryData>(mem_data, [](auto *md) {
          unmarkQs4cxPayloadDropped(md->template getAddr<uint8_t>());
          VirtualFree(md->template getAddr<uint8_t>(), 0, MEM_RELEASE);
          delete md;
        });
        offset = 0;
        // MEM_COMMIT pages are zero by the Win32 contract and the payload is
        // overwritten in full by the loader, so initialize()'s setZero() is a
        // dead store that only faults the whole region resident. putData()
        // still runs, so the MemoryData invalidate hook fires exactly as
        // before.
        if (qs4cxAllocUninitialized()) {
          qs4cxPoisonIfRequested(mem_data->getAddr<uint8_t>(), size());
          putData();
        } else {
          initialize();
        }
        return;
      }
      // fall through to the CRT heap on VirtualAlloc failure
    }
#endif

    // [pool-bypass] Load-destined payload: allocate UNINITIALIZED and skip
    // initialize(). See qs4cxAllocUninitialized() for why both zero passes are
    // dead stores, and why the page faults they force are the thing that
    // matters -- they are not saved, they MOVE to the read() that fills the
    // buffer, which is what lets a load-time drop bound the residency peak.
    const bool uninit = qs4cxAllocUninitialized();
    mem_data = new MemoryData(uninit ? (void *)(new uint8_t[size()])
                                     : (void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      // The drop registry is keyed by this address; retract it before the
      // allocator can hand the block to the next weight.
      unmarkQs4cxPayloadDropped(mem_data->template getAddr<uint8_t>());
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });

    offset = 0;
    if (uninit) {
      qs4cxPoisonIfRequested(mem_data->getAddr<uint8_t>(), size());
      putData(); // keep the MemoryData invalidate hook firing as before
    } else {
      initialize();
    }
  }
}

void *QS4CX_Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

void QS4CX_Tensor::pack() {
  if (packed_f32.load(std::memory_order_acquire) ||
      packed_f16.load(std::memory_order_acquire)) {
    return;
  }
  // Host reader of the plain nibbles + scales (the KAI rhs pack).
  refuseIfQs4cxPayloadDropped(getData(), "QS4CX_Tensor::pack");

  /**
   * @note the packing layout of the GEMM ukernel is shared with its GEMV
   * counterpart, so one packed weight serves both paths.
   */
  size_t opt_kernel_idx = nntrainer::get_opt_ukernel_idx_qai8dxp_qsi4cxp(false);
  /**
   * @note QS4CX tensor assumes that it is per-channel quantized along width()
   * axis which denotes output channel.
   */
  const size_t K = height();
  const size_t N = width();

  size_t packed_size = nntrainer::get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
    N, K, opt_kernel_idx, true);
  // Left uninitialized: the packer writes every byte of it. Zeroing it first
  // would only fault the whole buffer in on this thread before the parallel
  // pack touches it again.
  std::unique_ptr<uint8_t[]> buf(new uint8_t[packed_size]);

  // getScale() places the fp32 scales at whichever record stride this tensor
  // was loaded with, so ask it rather than repeating one of the two strides.
  // A backend without a packer throws here, before anything is published.
  nntrainer::rhs_pack_qsi4cxp_qs4cxs1s0(
    N, K, buf.get(), getData(), (uint8_t *)getScale(), opt_kernel_idx, true);

  packed_data = std::move(buf);
  // Publish last: a thread that sees this flag through the unlocked check in
  // FloatTensor::dotQs4cx must also see the buffer filled above.
  packed_f32.store(true, std::memory_order_release);
}

void *QS4CX_Tensor::getPackedData() const {
  if (!packed_data) {
    throw std::runtime_error{"pack before run model"};
  }

  return packed_data.get();
}

void QS4CX_Tensor::packF16Activation() {
#if defined(__aarch64__) || defined(__arm__)
  if (packed_f16.load(std::memory_order_acquire) ||
      packed_f32.load(std::memory_order_acquire)) {
    return;
  }
  // Host reader of the plain nibbles + scales (the fp16-activation KAI rhs).
  refuseIfQs4cxPayloadDropped(getData(), "QS4CX_Tensor::packF16Activation");

  // fp16-activation KAI rhs, built once at load. Byte-identical to the buffer
  // HalfTensor::dot's QS4CX case used to assemble lazily on its first forward
  // call, so this changes when it is built, not what is computed. fp16-act
  // graphs never touch pack()'s fp32-facade layout, so packed_data can hold
  // this one instead — one packed copy in RAM instead of two.
  //
  // Each super-row is built straight into its final place from its own plain
  // rows (Int4Utils::packPlainToKaiRhsPackedRows), so the work splits across
  // the thread pool and needs no full-size Section A or staging copy: the
  // only allocation is the packed buffer itself, and its pages are first
  // touched by the thread that fills them.
  const size_t K = height();
  const size_t N = width();
  const size_t packed_size = Int4Utils::kaiRhsPackedBytes(N, K);
  std::unique_ptr<uint8_t[]> buf(new uint8_t[packed_size]);

  const uint8_t *plain = (const uint8_t *)getData();
  const float *scales = (const float *)getScale();
  const size_t super_rows = (N + Int4Utils::KAI_NR - 1) / Int4Utils::KAI_NR;
  // 16 super-rows (64 output channels) per task: small enough to balance a
  // 1.5K-row projection over the pool, large enough to keep the per-task
  // overhead out of a vocabulary-sized head.
  constexpr size_t kSuperRowsPerTask = 16;
  const size_t tasks = (super_rows + kSuperRowsPerTask - 1) / kSuperRowsPerTask;
  uint8_t *dst = buf.get();
  ThreadManager::Global().parallel_for(0, tasks, [&](size_t t) {
    const size_t sr0 = t * kSuperRowsPerTask;
    Int4Utils::packPlainToKaiRhsPackedRows(plain, scales, N, K, sr0,
                                           sr0 + kSuperRowsPerTask, dst);
  });

  packed_data = std::move(buf);
  // Publish last: a thread that sees this flag through the unlocked check in
  // HalfTensor::dot must also see the buffer filled above.
  packed_f16.store(true, std::memory_order_release);
#else
  // The fp16 KAI micro-kernel is ARM(i8mm)-only; x86 CPU/GPU/CUDA consume the
  // plain QS4CX blob. Leave unpacked so isPackedF16Activation() stays false.
#endif
}

size_t QS4CX_Tensor::size() const {
  /**
   * @note QS4CX tensor assumes that it is per-channel quantized along width()
   * axis which denotes output channel.
   *
   * @note This is the QS4CX RECORD stride, not merely an allocation size:
   * Tensor::save() writes it and NeuralNetwork::load() derives every following
   * weight's file offset by accumulating it. Two strides exist -- the padded
   * `N * (K + 1) / 2` every package was exported with so far, and the trimmed
   * `N * ((K + 1) / 2)` the exporter writes now -- and they differ by the
   * floor(N/2) pad bytes the padded layout keeps between the nibbles and the
   * per-channel fp32 scales when K is even. The record carries no version, so
   * a caller that knows a file was written with the padded layout selects it
   * per tensor with setQs4cxRecordPadded(); getScale() and pack() then index
   * the scales at whichever stride was selected. NeuralNetwork::load() is
   * that caller: it lays the whole file out at the padded stride, then
   * re-walks at the trimmed one when the file size fits only the trimmed
   * total, or when the size also fits the padded total and the first QS4CX
   * record's own bytes read as trimmed (qs4cxRecordStride()).
   */
  return recordBytes(height(), width(), isQs4cxRecordPadded());
}

size_t QS4CX_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void *QS4CX_Tensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();

  /**
   * @note QS4CX tensor assumes that it is per-channel quantized along width()
   * axis which denotes output channel.
   */
  return ((int8_t *)getData()) +
         nibbleBytes(height(), width(), isQs4cxRecordPadded());
}

void QS4CX_Tensor::copy_qs4cx(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void QS4CX_Tensor::setZero() {
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + size(), 0);
}

void QS4CX_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

void QS4CX_Tensor::print(std::ostream &out) const {
  out << "data addr: " << getData() << '\n';
  out << dim;
  out << "[QS4CX data print skipped]" << std::endl;
}

QScheme QS4CX_Tensor::q_scheme() const { return QScheme::QS4CX; }

void QS4CX_Tensor::read(std::ifstream &file, size_t start_offset,
                        bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max())
    start_offset = file_offset;
  if (!isOnDiskLegacyQint4()) {
    TensorBase::read(file, start_offset, read_from_offset);
    return;
  }
  // Partial writer: for even K the transcode leaves an N/2-byte gap between
  // the nibble region and the scale tail, because the scale offset is
  // N * (K + 1) / 2 evaluated left to right while the nibbles occupy only
  // N * (K / 2) bytes. Zero the payload first so the gap is defined rather
  // than left to whatever the allocation happened to hold.
  setZero();
  readLegacyQint4ToQs4cx(
    width(), height(), start_offset,
    [&](char *dst, std::streamsize n, size_t off) {
      checkedRead(file, dst, n, "[QS4CX_Tensor::read] legacy QINT4 read failed",
                  off, read_from_offset);
    },
    reinterpret_cast<uint8_t *>(getData()),
    reinterpret_cast<float *>(getScale()));
  putData();
}

void QS4CX_Tensor::read(ReadSource src, size_t start_offset,
                        bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max())
    start_offset = file_offset;
  if (!isOnDiskLegacyQint4()) {
    if (isQs4cxScaleOnlyRead()) {
      // [zero-copy weight load] The caller is going to build this weight's
      // device backing straight from the file mapping, so the nibble half of
      // the record is never read out of this tensor and copying it here is a
      // whole second plane of the model for nothing. Take only the scale
      // tail, which is a few KiB and IS read from here (the device scale
      // buffer is built from it). The record is [nibbles][N fp32 scales], so
      // the tail starts at the same offset getScale() indexes.
      const size_t nib = nibbleBytes(height(), width(), isQs4cxRecordPadded());
      const size_t scale_bytes = (size_t)width() * sizeof(float);
      checkedRead(src, (char *)getScale(), (std::streamsize)scale_bytes,
                  "[QS4CX_Tensor::read] scale tail read failed",
                  start_offset + nib, read_from_offset);
      putData();
      return;
    }
    TensorBase::read(src, start_offset, read_from_offset);
    return;
  }
  // PARTIAL WRITER -- same even-K gap as the std::ifstream overload above.
  setZero();
  readLegacyQint4ToQs4cx(
    width(), height(), start_offset,
    [&](char *dst, std::streamsize n, size_t off) {
      checkedRead(src, dst, n, "[QS4CX_Tensor::read] legacy QINT4 read failed",
                  off, read_from_offset);
    },
    reinterpret_cast<uint8_t *>(getData()),
    reinterpret_cast<float *>(getScale()));
  putData();
}

namespace {
/**
 * @brief Dropped plain-payload ranges, keyed by address span.
 *
 * Entry count is the number of int4 FC weights (order 10^2) and the vector is
 * only ever consulted once `dropped_any` is set, so a linear scan is cheaper
 * than a map and keeps the empty case free.
 */
std::mutex &qs4cx_dropped_mtx() {
  static std::mutex m;
  return m;
}
std::vector<std::pair<uintptr_t, uintptr_t>> &qs4cx_dropped_ranges() {
  static std::vector<std::pair<uintptr_t, uintptr_t>> v;
  return v;
}
std::atomic<bool> qs4cx_dropped_any{false};
} // namespace

void markQs4cxPayloadDropped(const void *base, size_t bytes) {
  if (base == nullptr || bytes == 0)
    return;
  const uintptr_t lo = reinterpret_cast<uintptr_t>(base);
  {
    std::lock_guard<std::mutex> lock(qs4cx_dropped_mtx());
    qs4cx_dropped_ranges().emplace_back(lo, lo + bytes);
  }
  // Release: the range must be visible to any thread that observes the flag.
  qs4cx_dropped_any.store(true, std::memory_order_release);
}

bool anyQs4cxPayloadDropped() {
  return qs4cx_dropped_any.load(std::memory_order_acquire);
}

bool isQs4cxPayloadDropped(const void *ptr) {
  if (!anyQs4cxPayloadDropped() || ptr == nullptr)
    return false;
  const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(qs4cx_dropped_mtx());
  for (const auto &r : qs4cx_dropped_ranges())
    if (p >= r.first && p < r.second)
      return true;
  return false;
}

void unmarkQs4cxPayloadDropped(const void *base) {
  if (!anyQs4cxPayloadDropped() || base == nullptr)
    return;
  const uintptr_t lo = reinterpret_cast<uintptr_t>(base);
  std::lock_guard<std::mutex> lock(qs4cx_dropped_mtx());
  auto &v = qs4cx_dropped_ranges();
  for (auto it = v.begin(); it != v.end();) {
    if (it->first == lo)
      it = v.erase(it);
    else
      ++it;
  }
  // `dropped_any` stays set: it is only the fast-path gate, and clearing it
  // while another range is still live would skip that range's check.
}

void refuseIfQs4cxPayloadDropped(const void *payload, const char *who) {
  if (!isQs4cxPayloadDropped(payload))
    return;
  throw std::runtime_error(
    std::string("[") + (who ? who : "QS4CX") +
    "] refusing to read a QS4CX weight whose plain payload was released after "
    "its device repack (NNTR_QS4CX_HEAP_BYPASS + NNTR_V8C_DROP_PLAIN): the "
    "pages read back as zeros and the result would be silently wrong. A host "
    "consumer reached a weight the GPU path was expected to own; run with "
    "NNTR_V8C_DROP_PLAIN=0 to keep the payload resident");
}

bool qs4cxHeapBypassOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_QS4CX_HEAP_BYPASS");
    if (e != nullptr)
      return e[0] != '0';
#if defined(__ANDROID__)
    return true;
#else
    return false;
#endif
  }();
  return on;
}

bool qs4cxDropPlainOn() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_V8C_DROP_PLAIN");
    if (e != nullptr)
      return e[0] == '1';
#if defined(__ANDROID__)
    return qs4cxHeapBypassOn();
#else
    // Opt-in off Android. DiscardVirtualMemory does work on the WDDM SVM
    // host shadow (verified on every weight, goldens byte-identical); the
    // bundles that ship it set the variable, and a bare run keeps the
    // long-measured residency it always had.
    return false;
#endif
  }();
  return on;
}

} // namespace nntrainer
