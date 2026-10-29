// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	blas_kernel_interface.cpp
 * @date	5 June 2024
 * @brief	Interface for blas OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>

// NNTR_V8C_DROP_PLAIN page-drop primitives (see the lever in the v8c weight
// cache below).
#include <cerrno>
#include <cstdint>
#if defined(_WIN32)
#include <psapi.h>   // GetProcessMemoryInfo (NNTR_MEM_TRACE)
#include <windows.h> // DiscardVirtualMemory
#pragma comment(lib, "psapi.lib")
#else
#include <sys/mman.h> // madvise(MADV_DONTNEED)
#endif
#include <blas_kernels.h>
#include <load_trace.h>
#include <nntrainer_error.h>
#include <opencl_loader.h>
#include <qs4cx_tensor.h> // the dropped-payload registry (DROP_PLAIN)

namespace nntrainer {
// defined in attention_kernels.cpp; declared here rather than
// pulling that header into this translation unit.
int attn_gpuhash_mode();
void attn_gpuhash_note(const char *tag, const void *svm, void *clmem,
                       size_t bytes, unsigned int M);
void dotBatchedCl(Tensor const &input, Tensor const &m, Tensor &result,
                  bool trans, bool trans_m) {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < input.batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = input.getBatchSlice(b, 1);
    Tensor m_b = m.getBatchSlice(b, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    dotCl(this_b, m_b, result_b, trans, trans_m);
  }
}

Tensor dotCl(Tensor const &input, Tensor const &m, bool trans, bool trans_m) {
  Tensor output("", input.getFormat(), input.getDataType());
  dotCl(input, m, output, trans, trans_m);

  return output;
}

void dotCl(Tensor const &input, Tensor const &m, Tensor &result, bool trans,
           bool trans_m) {
  unsigned int dim1, dim2, mdim1, mdim2;
  if (input.getFormat() == Tformat::NHWC) {
    dim1 = input.batch() * input.height() * input.width();
    dim2 = input.channel();
    mdim1 = m.batch() * m.height() * m.width();
    mdim2 = m.channel();
  } else {
    dim1 = input.batch() * input.channel() * input.height();
    dim2 = input.width();
    mdim1 = m.batch() * m.channel() * m.height();
    mdim2 = m.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(),
                           input.getTensorType()); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), N, input.height(),
                           input.width(), input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(),
                           input.height(), N, input.getTensorType());
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (input.getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, input.getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N, input.getTensorType());
    }
  }

  lda = dim2;
  ldb = mdim2;
  ldc =
    (input.getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    const float *mdata = m.getData();
    float *rdata = result.getData();

    // out_svm: the same landing rule as the FP16 branch below. A result tensor
    // in coarse-grain shared virtual memory has to be host-mapped before the
    // staging read-back, or the copy-out never lands and the tensor keeps
    // whatever it held -- an all-zero FP32 FC on Intel Xe3.
    const bool out_svm =
      result.getMemoryData() && result.getMemoryData()->isSVM();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    // Native nntrainer CL kernels, mirroring the FP16 branch below 1:1. This
    // path used to route through a third-party BLAS-on-OpenCL library, whose
    // Gemm returned success while computing nothing on Intel Xe3 (NEO): every
    // FP32 GPU FC came back all-zero, while the same shapes on these kernels
    // match the CPU reference.
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      trans ? sgemv_cl(data, mdata, rdata, trans, dim2, dim1, lda, out_svm)
            : sgemv_cl(data, mdata, rdata, trans, dim1, dim2, lda, out_svm);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      trans_m
        ? sgemv_cl(mdata, data, rdata, !trans_m, mdim1, mdim2, ldb, out_svm)
        : sgemv_cl(mdata, data, rdata, !trans_m, mdim2, mdim1, ldb, out_svm);
    }
    /// case others: use sgemm
    else {
      sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc,
               out_svm);
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = input.getData<_FP16>();
    const _FP16 *mdata = m.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();

    // out_svm: a result tensor in coarse-grain shared virtual memory has to be
    // host-mapped before the staging read-back, or the copy-out silently never
    // lands (see sgemv_cl_internal). This is the one place the Tensor -- and
    // therefore the answer -- is still in hand.
    const bool out_svm =
      result.getMemoryData() && result.getMemoryData()->isSVM();

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      *rdata = dot_cl(data, mdata, K) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      trans ? sgemv_cl(data, mdata, rdata, trans, dim2, dim1, lda, out_svm)
            : sgemv_cl(data, mdata, rdata, trans, dim1, dim2, lda, out_svm);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      trans_m
        ? sgemv_cl(mdata, data, rdata, !trans_m, mdim1, mdim2, ldb, out_svm)
        : sgemv_cl(mdata, data, rdata, !trans_m, mdim2, mdim1, ldb, out_svm);
    }
    /// case others: use sgemm
    else {
      sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc,
               out_svm);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void multiplyCl(Tensor &input, float const &value) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData<float>();
    unsigned int len = input.size();

    sscal_cl(data, len, value);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data = input.getData<_FP16>();
    unsigned int len = input.size();
    sscal_cl(data, len, value);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void add_i_cl(Tensor &result, Tensor const &input) {

  // "Not allocated" means on NEITHER plane. A tensor the planner put on the
  // device plane has no shared-plane pointer to offer once the pool stops
  // creating a shadow for it, and a null getData() there is the correct
  // answer, not a missing allocation -- the bytes are in the cl_mem.
  const auto unallocated = [](const Tensor &t) {
    if (t.getData() != nullptr)
      return false;
    auto md = t.getMemoryData();
    return !(md && md->isClMem() && md->deviceMem() != nullptr);
  };
  NNTR_THROW_IF(unallocated(input), std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(unallocated(result), std::invalid_argument)
    << result.getName() << " is not allocated";

  // Broadcasting done for the case where batch size vary for both inputs
  // If batch size vary, batch size of input must be 1
  if ((result.getDim() == input.getDim()) ||
      (result.getDim() != input.getDim() && input.batch() == 1 &&
       result.channel() == input.channel() &&
       result.height() == input.height() && result.width() == input.width())) {

    if (result.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *Y = result.getData();
      const float *X = input.getData();

      // axpy with alpha=1 is just elementwise add, so use our own addition_cl
      // kernel here. FP16 already uses addition_cl below - make FP32
      // symmetric.
      unsigned int size_input = input.size();
      for (unsigned int i = 0; i < result.batch() / input.batch(); ++i) {
        addition_cl(X, Y, size_input, size_input);
        Y += size_input;
      }
    } else if (result.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      unsigned int size_res = result.size();
      unsigned int size_input = input.size();
      _FP16 *data_res = result.getData<_FP16>();
      const _FP16 *data_input = input.getData<_FP16>();

      // Bind the device pointers directly (SVM-direct, in-place accumulate)
      // when BOTH tensors are device-visible; otherwise fall back to the host
      // round trip. This is what keeps the residual stream on the device --
      // and on a coarse-grain device it is not an optimisation but a
      // correctness requirement, because the host round trip reads a
      // shared-plane view the device has been writing.
      const bool use_svm =
        result.getMemoryData() && result.getMemoryData()->isSVM() &&
        input.getMemoryData() && input.getMemoryData()->isSVM();

      addition_cl(data_input, data_res, size_input, size_res, use_svm);

#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }

  else {
    throw std::invalid_argument(
      "Error: Broadcasting not supported for these dimensions!");
  }
}

void transposeCl(const std::string &direction, Tensor const &in,
                 Tensor &result) {

  unsigned int input_batch_size, input_height, input_width, input_channels;

  input_batch_size = in.batch();
  input_height = in.height();
  input_width = in.width();
  input_channels = in.channel();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = in.getData();
    float *rdata = result.getData();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }

  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data = in.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    // for transpose about channels and height
    if (direction[0] == '1' && direction[2] == '0') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 0);
    }
    // for transpose about height and width
    else if (direction[0] == '0' && direction[2] == '2') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 1);
    }
    // for transpose about channels and width
    else if (direction[0] == '2' && direction[2] == '1') {
      transpose_cl_axis(data, rdata, input_batch_size, input_channels,
                        input_height, input_width, 2);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

// The five helpers below (copy, l2norm, absolute sum, index of the largest and
// of the smallest element) never had an OpenCL kernel of their own in this
// tree: they were thin wrappers over a third-party BLAS-on-OpenCL library that
// is no longer a dependency. They have no in-tree caller, so rather than
// guessing at a kernel they report the missing implementation. Add a kernel
// next to the other entries in blas_kernels.h when a caller needs one.
[[noreturn]] static void throwNoOpenCLKernel(const char *op) {
  throw exception::not_supported(
    std::string("[blas_kernel_interface] ") + op +
    " has no OpenCL kernel implementation in nntrainer");
}

void copyCl(const Tensor &input, Tensor &result) {
  throwNoOpenCLKernel("copyCl");
}

float nrm2Cl(const Tensor &input) { throwNoOpenCLKernel("nrm2Cl"); }

float asumCl(const Tensor &input) { throwNoOpenCLKernel("asumCl"); }

int amaxCl(const Tensor &input) { throwNoOpenCLKernel("amaxCl"); }

int aminCl(const Tensor &input) { throwNoOpenCLKernel("aminCl"); }

} // namespace nntrainer

// =============================================================================
// v8c (paper 8/4/4, arXiv:2505.00232) dispatch entry -- env-gated; callers fall
// back to the generic host path when this returns false.
// =============================================================================
#include "blas_kernels.h"
#include "cl_tensor_backing_pool.h"
#include "cl_tensor_view.h"
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <memory>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h> // [nondet] cuda::dev_only / copy_any
#endif
#include <mutex>
#include <unordered_map>

namespace nntrainer {
namespace {
/**
 * @brief Sole owner of a cl_mem this translation unit created itself.
 * @details The v8c weight entry below is destroyed on four different paths --
 * a duplicate build losing the insertion race, two cache erases, and process
 * teardown -- and expressing "release these two handles" once per path is what
 * let two of them drop the buffers on the floor. Ownership lives in the type
 * instead: the destructor releases, a move transfers, and a copy is
 * forbidden, so a fifth teardown path cannot reintroduce the leak.
 */
class OwnedClMem {
public:
  /** @brief Construct an empty handle */
  OwnedClMem() = default;
  /** @brief Take ownership of an already-created cl_mem */
  explicit OwnedClMem(cl_mem mem) : mem_(mem) {}
  OwnedClMem(const OwnedClMem &) = delete;
  OwnedClMem &operator=(const OwnedClMem &) = delete;
  /** @brief Move construct, leaving the source empty */
  OwnedClMem(OwnedClMem &&rhs) noexcept : mem_(rhs.mem_) { rhs.mem_ = nullptr; }
  /** @brief Move assign, releasing whatever this handle held */
  OwnedClMem &operator=(OwnedClMem &&rhs) noexcept {
    if (this != &rhs) {
      reset(rhs.mem_);
      rhs.mem_ = nullptr;
    }
    return *this;
  }
  /** @brief Release the held object */
  ~OwnedClMem() { reset(); }
  /** @brief Release the held object and adopt another */
  void reset(cl_mem mem = nullptr) {
    if (mem_)
      opencl::clReleaseMemObjectT(mem_);
    mem_ = mem;
  }
  /** @brief The raw handle, for binding as a kernel argument */
  cl_mem get() const { return mem_; }
  /** @brief Whether a device object is held */
  explicit operator bool() const { return mem_ != nullptr; }

private:
  cl_mem mem_ = nullptr;
};

/**
 * @brief Cached per-weight GPU residency for the v8c int8xint4 FC path:
 *        the packed int4 backing plus its derived scale / row-sum buffers.
 */
struct V8cWeightEntry {
  std::unique_ptr<tv::TensorBacking> backing;
  OwnedClMem scale_buf;      // [N] fp32 recip-scale (owned)
  OwnedClMem row_sum_w_int4; // [N] int32 sum_k(int4 w_nk) (owned)
  // [N] int32 per-channel zero point of an ASYMMETRIC int4 weight (owned;
  // empty for a symmetric weight). Selects the -DV8C_ASYM kernels -- see
  // dotCl_v8c_set_weight_zero_points().
  OwnedClMem zp_buf;
  unsigned int N = 0, K = 0;
  // Name of the weight this pack was built from. The map is keyed on the
  // weight's host address, which the allocator may hand to a DIFFERENT weight
  // after a free; every projection of a given kind has the same (N, K) in a
  // transformer, so the shape alone cannot tell the two apart and the GEMM
  // would silently bind the wrong pack. The name carries identity, so it is
  // what actually validates a hit.
  std::string name;
  cl_mem weight_image =
    nullptr; // cached image2d view (also released via TensorBacking)
  cl_mem weight_buf = nullptr; // raw backing buffer (buffer-path / Intel NEO)
  // No image2d view can serve this weight, for either of two reasons: N
  // exceeds the device's image2d height cap (the untied lm_head, N=vocab), or
  // the driver declined the image for a reason the cap query did not predict.
  // Both leave the row-major buffer as the only readable form, which is why
  // one flag names the consequence rather than each cause.
  // Kept SEPARATE from (weight_image == nullptr): on the buffer path the image
  // is skipped for EVERY weight although an image would be creatable, and only
  // this flag may route to the imageless lm_head GEMV.
  bool imageless = false;
  // Forward-pass generation this weight was last dispatched in: a weight is
  // dispatched at most once per pass, so seeing it twice in one generation
  // means a new pass began. This inferred boundary no longer gates the
  // shared-quant cache below -- that keys on the dispatch write log, which is
  // exact where the inference could only be conservative -- but the counter is
  // this tree's only pass marker and costs a compare per weight.
  unsigned long long last_use_gen = 0;
};

// Buffer-path (NNTR_V8C_BUF): on Intel NEO the v8c GEMM uses the *_buf kernels
// whose args are declared __global uint4* — they must be bound to raw cl_mem
// BUFFERS, not image2d objects. Single source of truth is
// blas_kernels.cpp::v8c_use_buffer_path() (caps-derived from vendor_id; the env
// flag still overrides); this file-local name forwards to it.
static bool v8c_buffer_path() { return v8c_use_buffer_path(); }

// The v8c int8×QINT4 FC GEMM is the GPU FC path's default — this gate is only
// reached from the GPU compute-ops FC dispatch, and the host fallback it guards
// is byte-identical, so it defaults ON. NNTR_FC_INT8_GPU=0 disables it for A/B.
static bool v8c_env_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char *e = std::getenv("NNTR_FC_INT8_GPU");
    cached = e ? (std::atoi(e) != 0) : 1; // default ON on the GPU FC path
  }
  return cached != 0;
}

static std::mutex &v8c_cache_mtx() {
  static std::mutex m;
  return m;
}
static std::unordered_map<const void *, V8cWeightEntry> &v8c_weight_cache() {
  static std::unordered_map<const void *, V8cWeightEntry> c;
  return c;
}

// Host copies of the per-channel weight zero points registered through
// dotCl_v8c_set_weight_zero_points(), keyed by weight NAME. The device copy
// lives on the weight's cache entry, but that entry is built lazily (or by the
// load-time prebuild) and may be rebuilt on a reused-key miss, so the source of
// truth is kept here and re-attached whenever an entry for that name is
// (re)built. Guarded by v8c_cache_mtx().
static std::unordered_map<std::string, std::vector<int>> &v8c_zp_registry() {
  static std::unordered_map<std::string, std::vector<int>> r;
  return r;
}

// Upload @p zp into @p e.zp_buf (replacing any previous one). Caller holds
// v8c_cache_mtx(). False -- and the entry left symmetric -- on an allocation
// failure or a length that does not match the entry.
static bool v8c_attach_zp_locked(V8cWeightEntry &e,
                                 const std::vector<int> &zp) {
  e.zp_buf.reset();
  if (zp.size() != (size_t)e.N || zp.empty())
    return false;
  cl_int err = CL_SUCCESS;
  cl_mem buf = opencl::clCreateBufferT(
    ClContext::Global().context_inst_.GetContextNoRetain(),
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * zp.size(),
    const_cast<int *>(zp.data()), &err);
  if (err != CL_SUCCESS || buf == nullptr)
    return false;
  e.zp_buf.reset(buf);
  return true;
}

// Number of per-fanout activation slots. RACE#1 (R3) fix: the GEMM reads the
// int8 activation through an image2d-from-buffer VIEW; the Adreno driver may
// not track the image<->parent-buffer alias, so a later fanout's quant WRITE
// to a *shared* act_i8 can race the prior fanout's still-in-flight GEMM image
// READ (a WAR hazard) once the queue-draining host maps are removed (the
// static cl_mem residency chain). gpu_native is race-free because each fanout
// role (qkv / wo / ffn-up+gate / ffn-down) owns a DISTINCT activation buffer +
// cached image, so an inter-fanout writer never aliases an in-flight reader.
// Mirror that with a small ring of slots, advanced only on a quant-cache MISS
// (a new input / new fanout). With exactly 4 distinct-input fanouts per
// transformer layer this gives the same ~1-layer reuse distance as
// gpu_native's 4 per-purpose buffers, while wq/wk/wv (which share one input =>
// cache hits) all reuse the same slot read-only.
static constexpr int V8C_ACT_SLOTS = 4;

// Grow-only scratch buffer pool, reused across all dotCl_v8c forward calls
// to avoid per-call clCreateBuffer/clReleaseMemObject churn (the dominant
// integration overhead, especially in M=1 decode where the same FC shapes
// recur thousands of times).
/**
 * @brief Grow-only scratch pool of the per-call v8c staging buffers, reused
 *        across dotCl_v8c calls to avoid per-call cl_mem churn.
 */
struct V8cScratch {
  // fp staging buffer for the quant input. SHARED: it is only ever read as a
  // plain buffer by the quant kernel (never via an image alias), so the
  // in-order SVM-pool queue already orders a later fanout's copy-write against
  // a prior fanout's quant-read of it -- no per-slot copy needed.
  cl_mem act_in = nullptr;
  size_t act_in_bytes = 0;
  // Per-fanout activation int8 + scale/zp/rowsum + the cached image2d view
  // over act_i8 (the only buffer the GEMM reads through an image -> the only
  // one needing distinct buffers per fanout). Ring-selected on quant-cache
  // miss.
  cl_mem act_i8[V8C_ACT_SLOTS] = {};
  size_t act_i8_bytes[V8C_ACT_SLOTS] = {};
  cl_mem act_scale[V8C_ACT_SLOTS] = {};
  size_t act_scale_bytes[V8C_ACT_SLOTS] = {};
  cl_mem act_rs[V8C_ACT_SLOTS] = {};
  size_t act_rs_bytes[V8C_ACT_SLOTS] = {};
  cl_mem act_zp[V8C_ACT_SLOTS] = {}; // [M] int32, asymmetric act zero-point
  size_t act_zp_bytes[V8C_ACT_SLOTS] = {};
  // Cached image2d-from-buffer views per slot, built once per (buffer, M_pad,
  // K) and reused across the fanout's GEMMs instead of per-call create/release
  // (which also leaked on the exception path).
  //
  // One cached view per slot was not enough. A transformer layer sends five or
  // more DIFFERENT K through the four-slot ring -- qkv, o, gate/up, down --
  // so the single (buffer, M_pad, K) key missed on nearly every call and the
  // "cache" rebuilt the view each time: 175 clCreateImage plus 175
  // clReleaseMemObject per decode token on an Adreno 840 cell, all of it
  // host work in front of a GPU waiting for its next dispatch. Keeping a small
  // SET of views per slot, one per shape seen, turns that into a fill during
  // the first layer and hits afterwards, because the shapes repeat every
  // layer. The views alias the same buffer read-only and are never live
  // concurrently on different data (the slot ring is what separates writers
  // from in-flight readers), so holding several changes nothing about the WAR
  // hazard the ring exists for.
  //
  // The set is bounded: past the cap the least recently used view is released,
  // so a graph with unbounded shape variety degrades to today's behaviour
  // rather than leaking images.
  static constexpr int kActViewsPerSlot = 12;
  struct ActView {
    cl_mem image = nullptr;
    cl_mem buf = nullptr;
    unsigned int M_pad = 0;
    unsigned int K = 0;
    unsigned long long lru = 0;
  };
  ActView act_views[V8C_ACT_SLOTS][kActViewsPerSlot] = {};
  unsigned long long act_view_clock = 0;
  int ring_pos = 0; /**< last slot handed out; advance-on-miss */
  cl_mem y_fp16 = nullptr;
  size_t y_fp16_bytes = 0;

  // Shared-quant cache (paper §3.6 fused-quant insight, host-side). Qwen3-style
  // layer graphs dispatch consecutive FC calls with the SAME input pointer
  // (wq/wk/wv read one post-RMSNorm activation; gate/up share theirs). After
  // the first call populates act_i8/act_scale/act_zp/act_rs for that input,
  // the sibling calls skip the upload AND the quant kernel.
  //
  // Cache key: (input handle, M, K, M_pad, dtype, dispatch seq of the quant).
  // An address on its own is NOT a key -- the tensor pool recycles activation
  // slots within a single forward pass, so the same (pointer, shape, dtype)
  // tuple can name different data a few dispatches later. The last term is
  // what makes the key sound: the write log (opencl::Kernel) answers whether
  // anything has written that handle since, and a hit needs that answer. Only
  // the device plane is admitted, since the log sees dispatch writes and not
  // host writes; see the note in dotCl_v8c.
  unsigned long long pass_gen = 1; /**< current forward-pass generation */
  /** unused by the shared-quant key; kept for the pass-boundary detector */
  unsigned long long last_quant_gen = 0;
  const void *last_quant_in_ptr = nullptr;
  unsigned int last_quant_M = 0;
  unsigned int last_quant_K = 0;
  unsigned int last_quant_M_pad = 0;
  int last_quant_dtype = -1;
  int last_quant_slot = 0; /**< slot whose int8 the cache hit refers to */
  /** dispatch seq at which the cached int8 was produced (device-plane reuse) */
  unsigned long long last_quant_seq = 0;

  // [norm->FC quant handoff] A fused norm (rmsnorm_cl_fp16_coop_q) fills one
  // of these slots with the quantisation of the row it just wrote, and
  // publishes which device buffer that row lives in. An FC whose input IS
  // that buffer would then skip its own quant kernel -- but no FC on this tree
  // consumes the handoff yet: the FC always runs its own quant, so the fields
  // below are only written and the nq_* statistics stay 0.
  //
  // The key is NOT the handle alone -- a pooled sub-buffer handle does not
  // identify a tensor on this backend, which is the KV-sharing defect above.
  // It is (handle, shape, and the dispatch write log saying nothing has
  // written that handle since). The last clause is what makes it sound: if
  // the bytes at the handle are still the bytes the norm wrote, the FC's own
  // quantiser would read exactly those bytes and produce exactly this result,
  // whichever tensor the graph thinks the handle belongs to.
  const void *nq_site = nullptr; /**< norm site that published the handoff */
  void *nq_src = nullptr;        /**< device buffer the fused norm wrote */
  unsigned long long nq_seq = 0; /**< dispatch seq right after that norm */
  unsigned int nq_rows = 0;
  unsigned int nq_K = 0;
  int nq_slot = -1; /**< ring slot holding the handoff int8, -1 = none */
  int nq_pending_slot = -1; /**< reserved by Begin, published by Commit */
};

static V8cScratch &v8c_scratch() {
  static V8cScratch s;
  return s;
}

// Fusion accounting. The two quant-elision paths (a norm that emitted the
// quantisation, and a sibling FC reusing a fanout's quantisation) both hinge
// on a window of the dispatch write log staying clean, and whether a given
// graph keeps it clean is not something to guess at -- NNTR_FUSE_STATS set
// (any value) prints the tally at exit so a decline can be attributed instead
// of assumed.
struct V8cFuseStats {
  unsigned long long share_hit = 0; /**< FCs served by a sibling's quant */
  unsigned long long quant = 0;     /**< FCs that ran their own quant */
  unsigned long long dirty = 0; /**< sibling reuses the write log ruled out */
  unsigned long long norm_published = 0; /**< fused norms that committed */
  unsigned long long nq_hit = 0;         /**< FCs served by a fused norm */
  // Why a norm-fed FC did NOT take the handoff -- the four ways it can miss,
  // so a decline is attributable instead of guessed at.
  unsigned long long nq_none = 0;  /**< no handoff outstanding */
  unsigned long long nq_other = 0; /**< outstanding, but for another buffer */
  unsigned long long nq_shape = 0; /**< same buffer, different rows/K/scratch */
  unsigned long long nq_dirty = 0; /**< same buffer and shape, log says dirty */
  ~V8cFuseStats() {
    if (std::getenv("NNTR_FUSE_STATS") == nullptr)
      return;
    ml_logi("[v8c fuse] quant elided: sibling=%llu  quant kernels run=%llu  "
            "write-log declines=%llu",
            share_hit, quant, dirty);
    ml_logi("[v8c fuse] norms published=%llu  quant elided by a norm=%llu",
            norm_published, nq_hit);
    ml_logi("[v8c fuse] handoff misses: none=%llu other-buffer=%llu "
            "shape=%llu dirty=%llu",
            nq_none, nq_other, nq_shape, nq_dirty);
  }
};

static V8cFuseStats &v8c_fuse_stats() {
  static V8cFuseStats s;
  return s;
}

// Ensure *buf has at least `bytes` capacity with the given flags; (re)alloc
// only when too small. Returns false on alloc failure.
static bool v8c_ensure_buf(cl_context ctx, cl_mem *buf, size_t *cap,
                           size_t bytes, cl_mem_flags flags) {
  if (*buf && *cap >= bytes)
    return true;
  if (*buf) {
    opencl::clReleaseMemObjectT(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  opencl::ClMemAcctScope _acct("scratch:v8c");
  *buf = opencl::clCreateBufferT(ctx, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  // clCreateBuffer content is UNDEFINED until first write. Padded rows
  // ([M, M_pad) under quant-direct) and partial first uses read these bytes
  // before any producer writes them: one driver hands zeroed pages (masking
  // this), another may recycle garbage that differs per process. Zero once per
  // (re)allocation; the in-order queue sequences the fill ahead of every later
  // consumer. Cost: create-time only (scratch buffers are grow-only).
  {
    auto *cc =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    const cl_uchar fill = 0;
    opencl::clEnqueueFillBuffer(cc->command_queue_inst_.GetCommandQueue(), *buf,
                                &fill, sizeof(fill), 0, bytes, 0, nullptr,
                                nullptr);
  }
  *cap = bytes;
  return true;
}

/**
 * @brief NNTR_V8C_MPAD_ALL=1 -- pad EVERY multi-row v8c call to the 64-row
 *        alignment, not only the ones that already reach 64 rows.
 *
 * @details DIAGNOSTIC, default OFF: it was measured and it is NOT a fix.
 *
 * The 64 in the rule below is the one number in the prefill path that matches
 * a separately recorded chunked-prefill defect -- a prefill whose LAST
 * chunk is under 64 rows answers wrongly (tails of 4 and 11 break; 395, 445
 * and 523 are golden, at 4, 5, 6 and 8 chunks alike). The alignment exists so
 * that gws_y = M_pad / V8C_TM is a multiple of 16 and the tuned 4x16
 * work-group divides the grid; below 64 rows select2dLws() fails and the GEMM
 * runs with a NULL (driver-chosen) work-group, which the dispatch site calls
 * "a measured performance cliff ... not a correctness issue".
 *
 * That claim survives the test. Forcing the 64-row padding for every
 * multi-row call -- so the tuned work-group is always used -- CHANGES the
 * wrong answer but does not fix it (3 595 tokens at init_seq_len 512:
 * 1 377 B either way), and the kernel says why: v8c_gemm_int8_int4 has no
 * local memory, no barrier and no sub-group op, and guards its stores with
 * `if (m0 + i >= M_valid) continue`, so it cannot depend on the work-group
 * shape. Keep this arm because it isolates M_pad from everything else that
 * moves with the chunk split; do not mistake it for the cause.
 *
 * Decode (M == 1) is excluded whatever the setting: padding a single row to 64
 * is a 64x FC blow-up, and M=1 takes the m1/GEMV kernel anyway.
 *
 * @param M real row count
 * @return the padded row count the scratch buffers and the grid both use
 */
static unsigned int v8c_m_pad_for(unsigned int M) {
  constexpr unsigned int V8C_TM = 4, V8C_MPAD_ALIGN = 64;
  static const bool pad_all = [] {
    const char *e = std::getenv("NNTR_V8C_MPAD_ALL");
    return e != nullptr && e[0] == '1';
  }();
  const unsigned int eff =
    (M >= V8C_MPAD_ALIGN || (pad_all && M > 1)) ? V8C_MPAD_ALIGN : V8C_TM;
  return (M + eff - 1) / eff * eff;
}

// Advance the per-fanout activation ring, and drop the shared-quant cache
// entry when the slot it points at is the one being handed out: once the ring
// hands its slot out the int8 it names is about to be overwritten by a
// different activation, so the entry stops being a valid answer here rather
// than at the point of use. This is the "never across a recycled slot" half of
// the reuse rule; the write log is the "never across a rewritten buffer" half.
static int v8c_ring_advance(V8cScratch &sc) {
  sc.ring_pos = (sc.ring_pos + 1) % V8C_ACT_SLOTS;
  if (sc.last_quant_slot == sc.ring_pos) {
    sc.last_quant_in_ptr = nullptr;
    sc.last_quant_seq = 0;
  }
  // The same rule for the norm->FC handoff: from here on that slot's int8
  // belongs to a different activation.
  if (sc.nq_slot == sc.ring_pos) {
    sc.nq_slot = -1;
    sc.nq_src = nullptr;
  }
  if (sc.nq_pending_slot == sc.ring_pos)
    sc.nq_pending_slot = -1;
  return sc.ring_pos;
}

// NNTR_FUSE_NORM_QUANT: 0 = never fuse, 2 = fuse speculatively at every
// device-plane norm whatever its shape (the behaviour before the row gate,
// kept for A/B on a new device), anything else / unset = the gated default.
static int v8c_norm_quant_mode() {
  static const int mode = []() {
    const char *e = std::getenv("NNTR_FUSE_NORM_QUANT");
    if (e != nullptr && e[0] == '0')
      return 0;
    if (e != nullptr && e[0] == '2')
      return 2;
    return 1;
  }();
  return mode;
}

static bool v8c_norm_quant_enabled() { return v8c_norm_quant_mode() != 0; }

// Row ceiling of the gated default. A decode step is one row and a
// speculative-decode verify a handful; both sit on the dispatch floor, where
// folding the quantiser into the norm saves a whole submission. A prefill is
// hundreds of rows, where the fused kernel is slower than the norm plus the
// FC's own quantiser. NNTR_FUSE_NORM_QUANT_MAX_ROWS moves the ceiling
// (0 = no ceiling).
static unsigned int v8c_norm_quant_max_rows() {
  static const unsigned int rows = []() {
    const char *e = std::getenv("NNTR_FUSE_NORM_QUANT_MAX_ROWS");
    if (e != nullptr && e[0] != '\0')
      return static_cast<unsigned int>(std::strtoul(e, nullptr, 10));
    return 8u;
  }();
  return rows;
}

// Get or build the cached v8c weight backing for a given int4 (QS4CX) weight.
// Returns nullptr if shape unsupported (caller falls back).

/**
 * @brief Whether @a weight is an int4 record the v8c path consumes: QS4CX,
 *        or QS4ACX -- the same nibbles and scale tail followed by a
 *        per-channel zero point, which the build attaches to the entry.
 */
static bool v8c_int4_record(const Tensor &weight) {
  const auto dt = weight.getDataType();
  return dt == ml::train::TensorDim::DataType::QS4CX ||
         dt == ml::train::TensorDim::DataType::QS4ACX;
}

static V8cWeightEntry *v8c_get_or_build_weight(const Tensor &weight,
                                               unsigned int K, unsigned int N,
                                               const uint8_t *nib_override) {
  if (K % 32 != 0 || N % 8 != 0)
    return nullptr;
  const void *key = weight.getData<uint8_t>();
  if (!key)
    return nullptr;
  // [init-parallel] The map lock covers ONLY lookup/insert. The build below
  // (CPU nibble repack + chunked upload + row-sum) runs OUTSIDE it so the
  // hw_concurrency weight-load workers repack in parallel: with the build
  // inside this lock every weight would go through one core (measured as
  // ~97% of a multi-second model init on both Windows and Linux).
  // Safety: no cross-weight ordering exists (disjoint device buffers; the
  // in-order queue only sequences the blocking per-weight chunk writes),
  // the per-call 16MB staging vector is function-local, and unordered_map
  // element pointers survive rehash, so entry pointers handed out earlier
  // stay valid under concurrent inserts.
  {
    std::lock_guard<std::mutex> lock(v8c_cache_mtx());
    auto &cache = v8c_weight_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
      // Validate the pointer-keyed hit. The key is a host address the
      // allocator can reuse for a different weight once the first is freed,
      // and identically shaped projections are the rule rather than the
      // exception, so the shape is not enough to tell one from another --
      // check the name too, and rebuild when either disagrees.
      if (it->second.N == N && it->second.K == K &&
          it->second.name == weight.getName())
        return &it->second;
      ml_logd("[v8c] weight-cache miss on a reused key: cached %s (N=%u K=%u) "
              "vs requested %s (N=%u K=%u); rebuilding",
              it->second.name.c_str(), it->second.N, it->second.K,
              weight.getName().c_str(), N, K);
      cache.erase(it);
    }
  }
  // [zero-copy weight load] nib_override is the weight file's mapping: the
  // plain payload was never copied into this tensor, so the build reads the
  // file instead. The cache identity above stays the tensor address either
  // way -- the mapping is per-node and transient, and would be a key that
  // dangles the moment the loader unmaps it.
  const uint8_t *nibbles =
    (nib_override != nullptr) ? nib_override : weight.getData<uint8_t>();
  if (!nibbles)
    return nullptr;
  // A rebuild from the tensor's own payload after that payload was released
  // (DROP_PLAIN below) would permute zeros into a device weight and every FC
  // through it would answer wrong with rc = 0. The only way here is a cache
  // entry that was dropped without its tensor going away (entries are only
  // replaced on a reused-key mismatch above; there is no teardown hook that
  // frees this cache, so a healthy run never takes this). Refuse, do not
  // decline: a nullptr return would send the call to the host fallback, which
  // is guarded the same way.
  if (nib_override == nullptr)
    refuseIfQs4cxPayloadDropped(nibbles, "v8c weight rebuild");
  // int4 weights are QS4CX: row-major plain nibbles (uint4 = int4+8, no XOR) +
  // per-output-channel fp32 scale. A legacy QINT4 .bin is re-laid-out to this
  // form at load (QS4CX_Tensor::read), so the v8c backing has a single source.
  //
  if (!v8c_int4_record(weight))
    return nullptr;
  V8cWeightEntry e;
  {
    cl_mem sb = nullptr;
    cl_mem rsw = nullptr;
    try {
      const float *fp32_scales = weight.getScale<float>();
      if (!fp32_scales)
        return nullptr;
      // The pack cache's identity is the tensor NAME, which is stable across
      // runs -- never the data pointer, which pool and SVM allocations recycle
      // (a pointer-keyed derived-weight cache in this tree has served the
      // wrong entry before). A weight with no name simply skips the cache.
      const std::string &wname = weight.getName();
      e.backing = make_v8c_weight_backing_from_qs4cx(
        nibbles, fp32_scales, N, K, &sb, &rsw,
        wname.empty() ? nullptr : wname.c_str());
    } catch (...) {
      // The builder creates the two buffers before the upload that can throw,
      // so adopt whatever it managed to produce: the entry's destructor is
      // then what releases them as this failed build unwinds.
      e.scale_buf.reset(sb);
      e.row_sum_w_int4.reset(rsw);
      return nullptr;
    }
    e.scale_buf.reset(sb);
    e.row_sum_w_int4.reset(rsw);
  }
  e.N = N;
  e.K = K;
  e.name = weight.getName();
  // Raw backing buffer for the NNTR_V8C_BUF path (Intel NEO). Always available
  // (zero-copy); the image2d view below is only used by the image-sampling
  // kernels (Adreno).
  e.weight_buf = e.backing->buffer();
  tv::ViewSpec ws;
  ws.kind = tv::ViewKind::IMAGE_2D;
  ws.image_channel_order = CL_RGBA;
  ws.image_channel_type = CL_UNSIGNED_INT32;
  ws.width = K / 32;
  ws.height = N;
  // Row pitch padded to a 64-byte multiple, matching the padded weight backing
  // rows (make_v8c_weight_backing_from_qs4cx) so image2d-from-buffer creation
  // satisfies CL_DEVICE_IMAGE_PITCH_ALIGNMENT on the image path. The buffer
  // path keeps the tight K/2 stride. The logical texel width stays K/32.
  // 64 bytes assumes the device's pitch alignment is at most 4 pixels at
  // 16 B/texel; a device reporting more would need this derived from the
  // queried alignment instead of assumed.
  ws.row_pitch_bytes = ClContext::Global().caps().image_v8c
                         ? (((size_t)K / 2 + 63) / 64) * 64
                         : (size_t)K / 2;

  // Is N past the device's image2d height cap? Ask the device instead of
  // inferring it from a failed clCreateImage, so the buffer path (which never
  // creates the image) can still identify the oversized lm_head.
  //
  // This queries CL directly rather than reading Context::caps(), which is the
  // rule everywhere else in this path. The exception is deliberate and narrow:
  // DeviceCaps carries no image geometry, and widening the shared struct for
  // one caller belongs in the change that owns it, not here. Reached through
  // ClContext::Global() -- the same accessor the pitch decision above uses --
  // because Engine::getRegisteredContext throws when the GPU backend was never
  // brought up, and this path must decline rather than propagate.
  {
    size_t img_h_cap = 0;
    opencl::clGetDeviceInfo(ClContext::Global().context_inst_.GetDeviceId(),
                            CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(img_h_cap),
                            &img_h_cap, nullptr);
    e.imageless = (img_h_cap != 0 && (size_t)N > img_h_cap);
  }

  // [buffer-path image skip] On the buffer path (Intel NEO) the GEMM binds
  // weight_buf and NEVER reads weight_image (see `use_buf ? weight_buf :
  // weight_image` in the dispatch) -- the image is a dead object. The skip
  // avoids creating hundreds of dead cl_mem objects; outputs are
  // byte-identical. The image path (Adreno) is unaffected: v8c_buffer_path() is
  // false there, so the view is built.
  if (!v8c_buffer_path() && !e.imageless) {
    try {
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::IMAGE_VIEW);
      e.weight_image = e.backing->imageView(ws);
      ml_logd("[v8c] image view built for %s: N=%u K=%u pitch=%zu",
              weight.getName().c_str(), N, K, ws.row_pitch_bytes);
    } catch (...) {
      // Creation can still fail for a reason the cap query did not predict
      // (pitch alignment). The row-major weight_buf + scale_buf remain valid,
      // so fall back to the imageless (GEMV) route rather than the CPU path --
      // which is the second way this weight becomes imageless, and why the
      // flag names the consequence rather than the height cap alone.
      e.weight_image = nullptr;
      e.imageless = true;
      static bool logged = false;
      if (!logged) {
        logged = true;
        ml_logw("[v8c] image view unavailable for %s (N=%u K=%u): the device "
                "declined the image2d, so this weight takes the imageless "
                "GEMV over its row-major buffer",
                weight.getName().c_str(), N, K);
      }
    }
  }
  // Re-take the map lock for insertion; DROP_PLAIN below stays inside it so
  // only the WINNING insertion drops the plain pages (a losing duplicate
  // builder may still be reading them).
  std::lock_guard<std::mutex> lock(v8c_cache_mtx());
  auto &cache = v8c_weight_cache();
  {
    auto it = cache.find(key);
    if (it != cache.end() && it->second.N == N && it->second.K == K) {
      // A concurrent caller built this weight while we were outside the
      // lock (theoretical: the load-time prebuild fires once per weight).
      // Keep the first entry and drop ours: scale_buf and row_sum_w_int4 are
      // standalone clCreateBuffer results this entry owns outright, so
      // OwnedClMem releases them as e goes out of scope here.
      // weight_image must NOT be released, which is why it is a raw handle
      // and not an OwnedClMem: it is the image2d view returned by
      // TensorBacking::imageView(), which caches it in image_cache_ and
      // releases every cached view in ~TensorBacking. e.backing's unique_ptr
      // runs that destructor on this same path, so releasing the image here
      // too would drop the refcount twice and free a live cl_mem (the winning
      // entry's GEMM keeps sampling its own view; the second release corrupts
      // the driver's object table -- observed class of failure is a crash or
      // a garbage weight read on the very next FC).
      return &it->second;
    }
    if (it != cache.end())
      cache.erase(it); // a stale mismatched entry raced back in; replace it
  }
  auto inserted = cache.emplace(key, std::move(e));
  // Zero points registered for this weight (possibly before this build,
  // possibly for a pack this miss is rebuilding) ride on the entry the GEMM
  // binds.
  {
    auto &reg = v8c_zp_registry();
    // A QS4ACX record carries its zero points: register them from the record
    // tail ([nibbles][N fp32 scales][N fp32 zero points]) the first time the
    // weight is built, so the asymmetric epilogue runs with no caller step.
    if (weight.getDataType() == ml::train::TensorDim::DataType::QS4ACX &&
        reg.find(inserted.first->second.name) == reg.end()) {
      const float *zp = static_cast<const float *>(weight.getScale()) + N;
      std::vector<int> zi(N);
      bool ok = true;
      for (unsigned int n = 0; n < N && ok; ++n) {
        const float r = std::nearbyint(zp[n]);
        ok = r == zp[n] && r >= -255.0f && r <= 255.0f;
        zi[n] = (int)r;
      }
      if (ok)
        reg.emplace(inserted.first->second.name, std::move(zi));
      else
        ml_logw("[v8c] %s: QS4ACX zero points are not integers in [-255, "
                "255]; the weight is built without them",
                inserted.first->second.name.c_str());
    }
    auto rit = reg.find(inserted.first->second.name);
    if (rit != reg.end() &&
        !v8c_attach_zp_locked(inserted.first->second, rit->second))
      ml_logw("[v8c] could not attach the weight zero points of %s",
              inserted.first->second.name.c_str());
  }

  // [NNTR_V8C_DROP_PLAIN] The device backing + scale buf + row-sum built
  // above are the only things the v8c GPU path reads from now on, so dropping
  // the plain QS4CX pages after the build reclaims ~the whole FC weight
  // footprint from host RSS. INWARD page alignment so pages shared with
  // neighboring pool tensors are never touched. May silently no-op if the
  // driver pinned the SVM pages (result is logged).
  //
  // Dropped pages read back as ZEROS, and host consumers of the plain payload
  // exist: the ClComputeOps QS4CX fallback calls Tensor::dot, whose
  // FloatTensor::dotQs4cx / HalfTensor::dot QS4CX arms read exactly these
  // nibbles plus the fp32 scale tail, and the KAI pack() paths read them too.
  // Those fallbacks are reachable: dotCl_v8c returns false from ~10 sites
  // AFTER the weight has been built, cached and dropped (allocation failures,
  // kernel-registration failures, the imageless untied-lm_head branch on an
  // fp16-disabled build). Every one of them therefore consults the
  // dropped-payload registry (qs4cx_tensor.h) and throws on a released weight
  // instead of returning a well-formed, entirely wrong output. The registry
  // is written HERE, right after a successful release, and only for a payload
  // the tensor owns (the heap bypass): a pool slice on Linux refuses the
  // madvise (EINVAL) and on a kgsl SVM arena "succeeds" without giving
  // anything back, so a slice is never recorded as unreadable.
  //
  // Default: qs4cxDropPlainOn() -- the env value when set; otherwise ON on
  // Android when the heap bypass is on (the only shape whose release is real
  // there -- and where the measured cost of NOT dropping is a second copy of
  // every FC weight, 569 MiB of a 1 338 MiB GPU footprint on a 1.5B model),
  // OFF elsewhere. The zero-copy load in FullyConnectedLayerCl::read() takes
  // its default from the same function.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86) || defined(__aarch64__) || defined(_M_ARM64)
  {
    // Per record type: QS4CX keeps qs4cxDropPlainOn(), QS4ACX resolves its
    // own default (int4DropPlainOn) -- its record also carries the zero
    // points, registered above from these bytes before anything is released.
    const auto w_dt = weight.getDataType();
    const bool is_qs4acx = w_dt == ml::train::TensorDim::DataType::QS4ACX;
    const bool drop_plain = int4DropPlainOn(w_dt);
    // Announce the resolved decision once. The drop is otherwise invisible in
    // a run log -- same outputs, same caches, only the residency differs --
    // which makes "was it actually on?" unanswerable after the fact. The
    // marker token is what a deployed-binary check greps for.
    static const bool announced = [&]() {
      ml_logi("[v8c] DROP_PLAIN %s (NNTR_V8C_DROP_PLAIN, heap bypass %s) "
              "POOLBYPASS_MARKER_A",
              drop_plain
                ? "ON: the plain QS4CX payload is released after the repack"
                : "OFF: the plain QS4CX payload stays resident",
              qs4cxHeapBypassOn() ? "on" : "off");
      return true;
    }();
    (void)announced;
#if !defined(ENABLE_FP16)
    // An imageless weight (N above the image2d height cap -- the untied int4
    // lm_head) has no image GEMM, and its only other route is the fp16
    // activation GEMV below, which this build does not contain. dotCl_v8c
    // therefore returns false for it unconditionally and the host GEMM IS the
    // path, not a fallback. Keep its payload.
    const bool droppable = !inserted.first->second.imageless;
#else
    const bool droppable = true;
#endif
    // [zero-copy weight load] When the nibbles came from the mapping there is
    // no nibble payload to release -- the tensor's payload half was never
    // filled, so its pages are exactly as unreadable as a released payload
    // and no host consumer may touch them. The SCALE tail, though, WAS
    // written, so release that.
    if (nib_override != nullptr) {
      const uint8_t *host = weight.getData<uint8_t>();
      if (host != nullptr) {
        const size_t payload = (size_t)N * (((size_t)K + 1) / 2) // nibbles
                               + (size_t)N * sizeof(float);      // fp32 scales
        // The nibble half was never written, but the SCALE tail was -- the
        // device scale buffer above is built from it -- and it is dead the
        // moment that buffer exists. Release it the same way the copy path
        // releases its payload, so the two paths leave the same residency and
        // a zero-copy load is not quietly the more expensive one at rest
        // (measured: 40 kB per weight, 2.4 MiB over gemma4 E2B). Same INWARD
        // page alignment, same DONTNEED -- the difference is only that here
        // there is far less to release.
        const size_t page = 4096;
        uintptr_t lo = ((uintptr_t)host + page - 1) & ~(page - 1);
        uintptr_t hi = ((uintptr_t)host + payload) & ~(page - 1);
        if (hi > lo) {
#if defined(_WIN32)
          (void)DiscardVirtualMemory((void *)lo, (SIZE_T)(hi - lo));
#else
          (void)::madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
#endif
        }
        // The nibble half was never written at all, so whatever the release
        // did, no host consumer may read this payload: record it.
        markQs4cxPayloadDropped(host, payload);
      }
    } else if (drop_plain && droppable) {
      const size_t payload = (size_t)N * (((size_t)K + 1) / 2) // nibbles
                             + (size_t)N * sizeof(float)       // fp32 scales
                             + (is_qs4acx ? (size_t)N * sizeof(float) : 0);
      const size_t page = 4096;
      uintptr_t lo = ((uintptr_t)nibbles + page - 1) & ~(page - 1);
      uintptr_t hi = ((uintptr_t)nibbles + payload) & ~(page - 1);
      long rc = -1;
      if (hi > lo) {
#if defined(_WIN32)
        rc =
          DiscardVirtualMemory((void *)lo, (SIZE_T)(hi - lo)) == ERROR_SUCCESS
            ? 0
            : -1;
#else
        rc = ::madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
#endif
      }
      // Per-weight success spam is diagnostics, not production output (an
      // SDK run printed 423 of these): success only under NNTR_MEM_TRACE,
      // failures always.
      static const bool drop_trace = std::getenv("NNTR_MEM_TRACE") != nullptr;
      if (rc != 0 || drop_trace)
        std::fprintf(stderr,
                     "[v8c] DROP_PLAIN N=%u K=%u bytes=%zu (aligned %zu) "
                     "rc=%ld errno=%d\n",
                     N, K, payload, (size_t)(hi > lo ? hi - lo : 0), rc,
                     rc == 0 ? 0 : errno);
      // Record the release for the host-side guards -- only for a self-owned
      // payload, whose pages really are gone (see the block comment above).
      if (rc == 0 && int4HeapBypassOn(w_dt))
        markQs4cxPayloadDropped(nibbles, payload);
    }
  }
#endif
  // [NNTR_MEM_TRACE] Working-set growth per v8c weight, against the bytes we
  // actually asked the runtime for. If WS climbs faster than the cl_mem bytes,
  // the driver is keeping more than the one copy we allocated.
#if defined(_WIN32)
  if (std::getenv("NNTR_MEM_TRACE")) {
    static size_t cum_v8c = 0, cum_plain = 0, count = 0;
    const size_t v8c_bytes = inserted.first->second.backing->bytes();
    cum_v8c += v8c_bytes;
    cum_plain += (size_t)N * (((size_t)K + 1) / 2) + (size_t)N * sizeof(float);
    ++count;
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    std::fprintf(stderr,
                 "[memtrace] w#%zu N=%u K=%u v8c=%.2fMB cum_v8c=%.1fMB "
                 "cum_plain=%.1fMB WS=%.1fMB\n",
                 count, N, K, v8c_bytes / 1048576.0, cum_v8c / 1048576.0,
                 cum_plain / 1048576.0, pmc.WorkingSetSize / 1048576.0);
    std::fflush(stderr);
  }
#endif
  return &inserted.first->second;
}

// fp16 → fp32 (host-side decode used to convert kernel fp16 output)
static inline float v8c_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t o;
  if (e == 0) {
    if (m == 0)
      o = s;
    else {
      e = 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        e--;
      }
      m &= 0x3ffu;
      o = s | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 0x1f) {
    o = s | 0x7f800000u | (m << 13);
  } else {
    o = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &o, 4);
  return f;
}
} // anonymous namespace

// Public symbols: the norm->FC quant handoff is reached from the rmsnorm
// dispatcher in blas_kernels_fp16.cpp, so it cannot live in the anonymous
// namespace above with the scratch it operates on.
// Per-norm-site record of whether anything ever claimed its quantisation.
//
// The fusion is speculative -- the norm cannot know its consumer -- and in a
// real decoder most norms are NOT read by an FC (a post-attention or post-FFN
// norm feeds the residual join). Quantising for them is pure loss: measured on
// the gemma4 decode cell, 122 of the 227 norms a token ran published a
// quantisation nobody took, and their share of the extra GPU time exceeded the
// dispatch floor the 105 productive ones removed. So a site that has published
// several times and never once been collected stops fusing. Nothing on this
// tree increments `claimed` (there is no FC-side consumer of the handoff), so
// every site stops after kNormSiteStrikes publications.
//
// The site key is the gamma pointer: a model weight, unique per norm layer and
// stable for the process, which is exactly the identity wanted here (the
// output buffer is pooled and would not be).
struct NormSite {
  unsigned int published = 0;
  unsigned int claimed = 0;
};

static std::unordered_map<const void *, NormSite> &v8c_norm_sites() {
  static std::unordered_map<const void *, NormSite> m;
  return m;
}

// How many unclaimed publications a site gets before it is written off. Small:
// the decision is stable after the first token, and a site that is productive
// would be claimed on its very first publication.
static constexpr unsigned int kNormSiteStrikes = 4;

bool v8cNormQuantBegin(const void *site, unsigned int rows, unsigned int K,
                       void **act_i8, void **act_scale, void **act_zp,
                       void **act_rs) {
  if (!v8c_norm_quant_enabled() || rows == 0 || K == 0)
    return false;
  if (v8c_norm_quant_mode() != 2 && v8c_norm_quant_max_rows() != 0 &&
      rows > v8c_norm_quant_max_rows())
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (cc == nullptr)
    return false;
  cl_context ctx = cc->context_inst_.GetContext();
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  NormSite &ns = v8c_norm_sites()[site];
  if (ns.claimed == 0 && ns.published >= kNormSiteStrikes)
    return false;
  V8cScratch &sc = v8c_scratch();
  const unsigned int M_pad = v8c_m_pad_for(rows);
  const int slot = v8c_ring_advance(sc);
  if (!v8c_ensure_buf(ctx, &sc.act_i8[slot], &sc.act_i8_bytes[slot],
                      (size_t)M_pad * K, CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_scale[slot], &sc.act_scale_bytes[slot],
                      sizeof(float) * M_pad, CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_rs[slot], &sc.act_rs_bytes[slot],
                      sizeof(int) * M_pad, CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_zp[slot], &sc.act_zp_bytes[slot],
                      sizeof(int) * M_pad, CL_MEM_READ_WRITE))
    return false;
  sc.nq_pending_slot = slot;
  sc.nq_site = site;
  ++ns.published;
  *act_i8 = sc.act_i8[slot];
  *act_scale = sc.act_scale[slot];
  *act_zp = sc.act_zp[slot];
  *act_rs = sc.act_rs[slot];
  return true;
}

bool v8cNormQuantUngated() { return v8c_norm_quant_mode() == 2; }

void v8cNormQuantCommit(void *src_clmem, unsigned int rows, unsigned int K) {
  ++v8c_fuse_stats().norm_published;
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  V8cScratch &sc = v8c_scratch();
  if (sc.nq_pending_slot < 0 || src_clmem == nullptr)
    return;
  sc.nq_slot = sc.nq_pending_slot;
  sc.nq_pending_slot = -1;
  sc.nq_src = src_clmem;
  sc.nq_rows = rows;
  sc.nq_K = K;
  // Sampled AFTER the dispatch: the window the write log has to be clean over
  // starts at the norm itself.
  sc.nq_seq = opencl::Kernel::dispatchSeq();
}

void v8cNormQuantAbort() {
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  v8c_scratch().nq_pending_slot = -1;
}

bool v8cNormQuantClaim(void *src_clmem, unsigned int rows, unsigned int K,
                       void **act_i8, void **act_scale, void **act_zp,
                       void **act_rs) {
  if (src_clmem == nullptr || rows == 0 || K == 0)
    return false;
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  V8cScratch &sc = v8c_scratch();
  auto &fs = v8c_fuse_stats();
  if (sc.nq_slot < 0) {
    ++fs.nq_none;
    return false;
  }
  if (sc.nq_src != src_clmem) {
    ++fs.nq_other;
    return false;
  }
  const int slot = sc.nq_slot;
  const unsigned int M_pad = v8c_m_pad_for(rows);
  if (sc.nq_rows != rows || sc.nq_K != K ||
      sc.act_i8_bytes[slot] < (size_t)M_pad * K ||
      sc.act_scale_bytes[slot] < sizeof(float) * M_pad ||
      sc.act_zp_bytes[slot] < sizeof(int) * M_pad ||
      sc.act_rs_bytes[slot] < sizeof(int) * M_pad) {
    ++fs.nq_shape;
    return false;
  }
  // The same test the FC applies: the bytes at the handle must still be the
  // ones the norm quantised.
  if (!opencl::Kernel::bufferUnchangedSince(src_clmem, sc.nq_seq)) {
    ++fs.nq_dirty;
    return false;
  }
  ++fs.nq_hit;
  if (sc.nq_site != nullptr)
    ++v8c_norm_sites()[sc.nq_site].claimed;
  *act_i8 = sc.act_i8[slot];
  *act_scale = sc.act_scale[slot];
  *act_zp = sc.act_zp[slot];
  *act_rs = sc.act_rs[slot];
  return true;
}

// Eager v8c weight build (see header). Moves the lazy per-weight nibble
// permute + upload out of the first timed prefill: the CL FC layer calls this
// right after its weight is read at model load. No-op (false) off the v8c
// path. Lives OUTSIDE the anonymous namespace (public symbol).
bool dotCl_v8c_prebuild_weight(const Tensor &weight) {
  if (!v8c_env_enabled())
    return false;
  // [load-trace] Everything this call does is charged to the loader worker
  // that read the weight, so it is the part of NODE_READ that is GPU work
  // rather than file work.
  nntrainer::load_trace::Scope _lt(nntrainer::load_trace::PREBUILD);
  // Reclaim any submit-and-go upload staging from the load phase (memory
  // hygiene only — the in-order queue already sequences those writes ahead
  // of this GEMM). Relaxed-atomic no-op after the first forward.
  v8c_flush_pending_uploads();
  if (!v8c_int4_record(weight))
    return false;
  const unsigned int N = weight.width();
  const unsigned int K = weight.height();
  if (N == 0 || K == 0 || N % 8 != 0 || K % 32 != 0)
    return false;
  return v8c_get_or_build_weight(weight, K, N, nullptr) != nullptr;
}

bool dotCl_v8c_set_weight_zero_points(const Tensor &weight, const float *zp,
                                      unsigned int N) {
  const std::string &name = weight.getName();
  if (name.empty())
    return false;
  std::vector<int> zi;
  if (zp != nullptr) {
    if (N == 0 || N != weight.width())
      return false;
    // The kernels fold the zero point into their int32 epilogue, so it must
    // be an integer; the bound keeps (8 - z) * K far from int32 overflow.
    zi.resize(N);
    for (unsigned int n = 0; n < N; ++n) {
      const float r = std::nearbyint(zp[n]);
      if (r != zp[n] || r < -255.0f || r > 255.0f)
        return false;
      zi[n] = (int)r;
    }
  }
  // Bring the OpenCL context up before the first touch of the static weight
  // cache below: function-local statics are destroyed in reverse order of
  // construction, and a cache constructed before the context would release
  // its device buffers after the context is gone at process exit (measured
  // as a driver crash on Adreno when this is called ahead of any dispatch).
  (void)ClContext::Global();
  std::lock_guard<std::mutex> lock(v8c_cache_mtx());
  auto &reg = v8c_zp_registry();
  // The entry this weight is cached under, if it was built already. Same
  // identity test as a cache hit: the key is a host address, the name is what
  // tells two same-shaped weights apart.
  V8cWeightEntry *entry = nullptr;
  if (const void *key = weight.getData<uint8_t>()) {
    auto it = v8c_weight_cache().find(key);
    if (it != v8c_weight_cache().end() && it->second.name == name)
      entry = &it->second;
  }
  if (zp == nullptr) {
    reg.erase(name);
    if (entry != nullptr)
      entry->zp_buf.reset();
    return true;
  }
  auto &host = reg[name];
  host = std::move(zi);
  if (entry != nullptr && !v8c_attach_zp_locked(*entry, host)) {
    // Never leave registered zero points that the GEMM is not applying.
    reg.erase(name);
    return false;
  }
  return true;
}

bool zeroPointsFromCorrection(const float *scale, const float *corr,
                              unsigned int N, std::vector<float> &zp) {
  if (scale == nullptr || corr == nullptr)
    return false;
  zp.resize(N);
  for (unsigned int n = 0; n < N; ++n) {
    if (!(scale[n] > 0.0f))
      return false;
    // c = s * (8 - z) with an integer z: recover z and require that c is
    // exactly what that z gives back (to float rounding), so a correction that
    // does not come from an integer zero point is refused, not approximated.
    const float z = std::nearbyint(8.0f - corr[n] / scale[n]);
    const float back = scale[n] * (8.0f - z);
    if (std::fabs(back - corr[n]) > 1e-5f * std::fabs(scale[n]) * 8.0f + 1e-30f)
      return false;
    zp[n] = z;
  }
  return true;
}

bool dotCl_v8c_set_zero_point_correction(const Tensor &weight,
                                         const float *corr, unsigned int N) {
  if (corr == nullptr)
    return dotCl_v8c_set_weight_zero_points(weight, nullptr, N);
  if (N == 0 || N != weight.width())
    return false;
  std::vector<float> zp;
  if (!zeroPointsFromCorrection(weight.getScale<float>(), corr, N, zp))
    return false;
  return dotCl_v8c_set_weight_zero_points(weight, zp.data(), N);
}

bool dotCl_v8c_prebuild_weight_from(const Tensor &weight,
                                    const void *src_nibbles) {
  if (src_nibbles == nullptr)
    return false;
  if (!v8c_env_enabled())
    return false;
  nntrainer::load_trace::Scope _lt(nntrainer::load_trace::PREBUILD);
  v8c_flush_pending_uploads();
  if (!v8c_int4_record(weight))
    return false;
  const unsigned int N = weight.width();
  const unsigned int K = weight.height();
  if (N == 0 || K == 0 || N % 8 != 0 || K % 32 != 0)
    return false;
  return v8c_get_or_build_weight(
           weight, K, N, static_cast<const uint8_t *>(src_nibbles)) != nullptr;
}

// fp16 GEMM output -> output tensor, written on the GPU (residency: no host
// readback). One source, two entry points: cvt_h2f converts fp16->fp32,

// Small utility kernels for moving the fp16 GEMM result / SVM activations
// on the device (no host round-trip).
static const std::string v8c_util_kernels = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void v8c_cvt_h2f(__global const half *in, __global float *out,
                          const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = (float)in[i];
}
__kernel void v8c_copy_h2h(__global const half *in, __global half *out,
                           const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
__kernel void v8c_copy_f2f(__global const float *in, __global float *out,
                           const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
__kernel void v8c_add_h2h(__global const half *in, __global half *out,
                          const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] += in[i];
}
__kernel void v8c_add2_h2h(__global const half *a, __global const half *b,
                           __global half *out, const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}
)CL";

/**
 * @brief Residual copy / accumulate for operands on the DEVICE plane.
 *
 * @details The shared-plane path in add_i_cl binds host-visible pointers. A
 * tensor the planner placed on the device plane has none: its getData() is a
 * shadow the device never wrote, so the shared-plane path would read stale
 * bytes and write the result somewhere no consumer looks. This binds each
 * operand on its own plane instead, which also makes the mixed case -- a
 * device-resident source into a shared-plane destination, the shape of a
 * boundary whose consumer reads on the host -- expressible.
 *
 * @param[in,out] dst destination (the residual accumulator)
 * @param[in] src source
 * @param[in] accumulate false: dst = src; true: dst += src
 * @return false when neither operand is on the device plane, so the caller
 *         keeps its shared-plane path; true when the op was issued.
 */
bool clmem_residual_op_cl(Tensor &dst, const Tensor &src, bool accumulate) {
  if (dst.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      src.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (dst.size() != src.size() || dst.size() == 0)
    return false;
  void *dst_cl = dst.isClMem() ? dst.getClMem() : nullptr;
  void *src_cl = src.isClMem() ? src.getClMem() : nullptr;
  if (dst_cl == nullptr && src_cl == nullptr)
    return false;

  // The device handle covers the WHOLE tensor and kernels address from its
  // base, so a step/batch view at a nonzero offset cannot bind it. Fail loudly
  // rather than silently corrupting through a base bind.
  if ((dst_cl != nullptr && dst.getOffset() != 0) ||
      (src_cl != nullptr && src.getOffset() != 0))
    throw std::runtime_error(
      "clmem_residual_op_cl: device-plane tensor accessed at a nonzero offset "
      "(batch>1 step views are unsupported on the device plane)");

  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_residual_op_cl: no GPU context");
  const unsigned int n = (unsigned int)dst.size();
  const size_t bytes = (size_t)n * sizeof(uint16_t);

  // A device-to-device copy could go through clEnqueueCopyBuffer rather than a
  // kernel, but that entry point is not in this build's OpenCL loader, and the
  // copy kernel below is the path this falls back to anyway whenever a
  // recordable queue is in use (a non-NDRange op is not captured by one).
  // One path is also one thing to reason about.

  auto kp = cc->registerClKernel(v8c_util_kernels,
                                 accumulate ? "v8c_add_h2h" : "v8c_copy_h2h");
  if (!kp)
    throw std::runtime_error("clmem_residual_op_cl: kernel registration");

  // A shared-plane operand keeps the established per-op map protocol (unmap
  // before the kernel, map after); a device-plane one needs neither.
  void *src_svm =
    const_cast<void *>(static_cast<const void *>(src.getData<uint8_t>()));
  void *dst_svm = static_cast<void *>(dst.getData<uint8_t>());
  if (src_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMUnmap(src_svm);
  if (dst_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMUnmap(dst_svm);

  bool ok = true;
  if (src_cl != nullptr) {
    cl_mem h = static_cast<cl_mem>(src_cl);
    ok = ok && kp->SetKernelArguments(0, &h, sizeof(cl_mem));
  } else {
    ok = ok && kp->SetKernelSVMArguments(0, src_svm);
  }
  if (dst_cl != nullptr) {
    cl_mem h = static_cast<cl_mem>(dst_cl);
    ok = ok && kp->SetKernelArguments(1, &h, sizeof(cl_mem));
  } else {
    ok = ok && kp->SetKernelSVMArguments(1, dst_svm);
  }
  int ni = (int)n;
  ok = ok && kp->SetKernelArguments(2, &ni, sizeof(int));
  if (!ok)
    throw std::runtime_error("clmem_residual_op_cl: arg binding");

  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  if (!cc->command_queue_inst_.DispatchCommand(kp, gws, lws))
    throw std::runtime_error("clmem_residual_op_cl: dispatch");

  if (src_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMMap(src_svm, bytes, true);
  if (dst_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMMap(dst_svm, bytes, true);
  return true;
}

/**
 * @brief Both residual operands in one dispatch: dst = a + b.
 *
 * The two-input AdditionLayer is the residual stream's join, and it is the
 * single most frequent elementwise op in a decoder step. Run per operand it
 * costs a copy dispatch and an accumulate dispatch; run together it costs one.
 * On this backend that halves a pair whose GPU time (~1.7 us each) is a
 * quarter of the ~7.4 us it costs to put either of them on the device.
 *
 * The result is bit-identical to the pair: the copy writes a[i] exactly and
 * the accumulate then rounds a[i] + b[i] once, which is what this single
 * add does.
 *
 * Device plane only, and only when all three tensors are there: a shared-plane
 * operand keeps the map/unmap protocol the per-operand path implements, and
 * splitting the difference here would be a second protocol to reason about for
 * no dispatch saved (the shared-plane first operand does not dispatch at all).
 */
bool clmem_residual_add2_cl(Tensor &dst, const Tensor &a, const Tensor &b) {
  static const bool fuse_on = []() {
    const char *e = std::getenv("NNTR_FUSE_RESIDUAL_ADD2");
    return !(e != nullptr && e[0] == '0');
  }();
  if (!fuse_on)
    return false;
  const auto fp16 = ml::train::TensorDim::DataType::FP16;
  if (dst.getDataType() != fp16 || a.getDataType() != fp16 ||
      b.getDataType() != fp16)
    return false;
  if (dst.size() == 0 || dst.size() != a.size() || dst.size() != b.size())
    return false;
  if (!dst.isClMem() || !a.isClMem() || !b.isClMem())
    return false;
  void *dst_cl = dst.getClMem(), *a_cl = a.getClMem(), *b_cl = b.getClMem();
  if (dst_cl == nullptr || a_cl == nullptr || b_cl == nullptr)
    return false;
  // Same rule as clmem_residual_op_cl: the device handle covers the whole
  // tensor, so a view at a nonzero offset cannot be bound from its base.
  if (dst.getOffset() != 0 || a.getOffset() != 0 || b.getOffset() != 0)
    return false;

  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (cc == nullptr)
    return false;
  auto kp = cc->registerClKernel(v8c_util_kernels, "v8c_add2_h2h");
  if (!kp)
    return false;
  cl_mem ah = static_cast<cl_mem>(a_cl), bh = static_cast<cl_mem>(b_cl),
         oh = static_cast<cl_mem>(dst_cl);
  int ni = (int)dst.size();
  if (!kp->SetKernelArguments(0, &ah, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(1, &bh, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &oh, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(3, &ni, sizeof(int)))
    return false;
  const int gws[3] = {(int)(((size_t)ni + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  opencl::Kernel::noteDispatchWrites(dst_cl);
  return cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
}

// Write the fp16 GEMM result (y_fp16, device cl_mem, n = M*N valid elements)
// directly into the output the planner placed, converting to fp32 when needed.
// Two destinations, one per residency plane: the device sub-buffer when
// out_clmem is given, the shared SVM pointer otherwise. On the shared plane
// coarse-grained SVM coherence applies -- unmap the output before the kernel
// (GPU owns it), re-map after (host / next layer can read it).
static void v8c_write_output_resident(cl_mem y_fp16, Tensor &output,
                                      unsigned int n, bool out_fp16,
                                      void *out_clmem = nullptr) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto kp = cc->registerClKernel(v8c_util_kernels,
                                 out_fp16 ? "v8c_copy_h2h" : "v8c_cvt_h2f");
  if (!kp)
    return;

  // Device-plane destination: write the planner sub-buffer with THIS KERNEL
  // (a cl_mem argument) rather than a buffer copy -- the blit engine is not
  // reliably ordered against compute kernels on this driver without a drain,
  // while kernel->kernel ordering is. No SVM maps on this path: nothing on it
  // is host-visible.
  if (out_clmem != nullptr) {
    cl_mem oh = static_cast<cl_mem>(out_clmem);
    int ni2 = (int)n;
    if (!kp->SetKernelArguments(0, &y_fp16, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &oh, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &ni2, sizeof(int)))
      return;
    const int gws2[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
    const int lws2[3] = {64, 1, 1};
    opencl::Kernel::noteDispatchWrites(out_clmem);
    cc->command_queue_inst_.DispatchCommand(kp, gws2, lws2);
    return;
  }
  void *out_svm = output.getData<uint8_t>();
  int ni = (int)n;
  cc->command_queue_inst_.enqueueSVMUnmap(out_svm);
  if (!kp->SetKernelArguments(0, &y_fp16, sizeof(cl_mem)) ||
      !kp->SetKernelSVMArguments(1, out_svm) ||
      !kp->SetKernelArguments(2, &ni, sizeof(int)))
    return;
  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
  size_t out_bytes = (size_t)n * (out_fp16 ? sizeof(uint16_t) : sizeof(float));
  // async map: the FC output is always consumed by the next GPU op — never
  // read on the host here — so the in-order queue orders this map before the
  // next op's unmap and the host need not block. NNTR_FC_SVM_SYNC set (any
  // value) makes the map BLOCKING (coarse-grained-SVM coherence probe for
  // drivers where the async handoff shows stale reads).
  static const bool fc_svm_sync = std::getenv("NNTR_FC_SVM_SYNC") != nullptr;
  cc->command_queue_inst_.enqueueSVMMap(out_svm, out_bytes, true,
                                        /** event */ nullptr,
                                        /** async */ !fc_svm_sync);
}

// Copy an SVM-resident activation (n = M*K elements) into the device cl_mem
// quant scratch on the GPU -- replaces the host upload (clEnqueueWriteBuffer)
// when the input is GPU-resident, so no PCIe round-trip. Downstream (quantize
// -> image2d -> GEMM) is unchanged; only the source of sc.act_in changes.
// Coarse-grained SVM coherence: unmap the input before the copy (GPU owns it),
// re-map after.
static void v8c_copy_svm_to_clmem(const void *in_svm, cl_mem out,
                                  unsigned int n, bool fp16) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto kp = cc->registerClKernel(v8c_util_kernels,
                                 fp16 ? "v8c_copy_h2h" : "v8c_copy_f2f");
  if (!kp)
    return;
  int ni = (int)n;
  cc->command_queue_inst_.enqueueSVMUnmap(const_cast<void *>(in_svm));
  if (!kp->SetKernelSVMArguments(0, const_cast<void *>(in_svm)) ||
      !kp->SetKernelArguments(1, &out, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &ni, sizeof(int)))
    return;
  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
  // async map: GPU→GPU handoff (the input copy feeds the quant/GEMM kernels);
  // no host access before then, in-order queue preserves ordering.
  cc->command_queue_inst_.enqueueSVMMap(const_cast<void *>(in_svm),
                                        (size_t)n * (fp16 ? 2 : 4), true,
                                        /** event */ nullptr,
                                        /** async */ true);
}

bool dotCl_v8c(const Tensor &input, const Tensor &weight, Tensor &output) {
  if (!v8c_env_enabled())
    return false;
  if (!v8c_int4_record(weight))
    return false;
  // Derive M, K, N from tensor dims (no-transpose case only).
  unsigned int M, K, N;
  if (input.getFormat() == Tformat::NHWC) {
    M = input.batch() * input.height() * input.width();
    K = input.channel();
  } else {
    M = input.batch() * input.channel() * input.height();
    K = input.width();
  }
  N = weight.width();
  if (K != weight.height())
    return false;
  if (N % 8 != 0 || K % 32 != 0)
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 &&
      input.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;

  // Round M up to the kernel's tile size (V8C_TM=4). Padded rows produce
  // throwaway output that we never read back to the caller, so v8c runs for
  // any prefill length (no "M not divisible by 4 -> CPU fallback" cliff).
  constexpr unsigned int V8C_TM = 4;
  // M_pad alignment. The v8c GEMM dispatches gws M-axis = M_pad / V8C_TM; the
  // tuned 4x16 work-group needs gws_y = M_pad/4 to be a multiple of 16, i.e.
  // M_pad a multiple of 64, or select2dLws (cl_tensor_view.cpp) fails its
  // divisibility gate and falls back to a NULL (driver-chosen) work-group.
  // On BOTH device families that fallback is a measured performance cliff
  // (a driver-chosen work-group is pathological for some N), so align to 64
  // by default. Padded rows are computed but never stored (M-valid store
  // guard in the kernel), so output is bit-identical. The alignment is fixed
  // rather than tunable: it is derived from the work-group above, not swept.
  // Decode (M == 1) must never pad to 64 (a 64x FC blow-up); every other row
  // count does, which is what NNTR_V8C_MPAD_ALL settles -- see v8c_m_pad_for().
  // One rule, one place: that function is also what the scratch buffers are
  // sized with, so the grid must not compute its own answer. (It used to, and
  // the two agreed only by repeating the same literal.)
  const unsigned int M_pad = v8c_m_pad_for(M);
  static_assert(64u % V8C_TM == 0,
                "M_pad alignment must stay a multiple of the kernel tile");

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  V8cWeightEntry *w = v8c_get_or_build_weight(weight, K, N, nullptr);
  if (!w)
    return false;

  // Imageless v8c weight (N > image2d height cap, e.g. the untied int4 lm_head
  // with N=vocab): the image GEMM path cannot run, so dispatch the dedicated
  // fp-act int4 GEMV over the row-major weight buffer (best logit fidelity;
  // no int8 act quant). Only decode (M=1) is supported -- the lm_head FC runs
  // only on the last position and prefill is skipped; any larger M with no
  // image legitimately falls back to the host path.
  // Keyed on imageless, NOT (weight_image == nullptr): the buffer path leaves
  // the image null for EVERY weight (it is dead there), and those must still
  // take the normal buffer GEMM below.
  if (w->imageless) {
    // Join the pass-boundary detector before taking the early exit. This
    // weight is the untied lm_head: it is the one FC guaranteed to be
    // dispatched on every decode pass, so leaving it out would hide a boundary
    // from the counter entirely. The scratch lock is taken and released here
    // because the GEMV below runs outside it.
    {
      std::lock_guard<std::mutex> glock(v8c_cache_mtx());
      V8cScratch &gsc = v8c_scratch();
      if (w->last_use_gen != 0 && w->last_use_gen == gsc.pass_gen)
        ++gsc.pass_gen;
      w->last_use_gen = gsc.pass_gen;
    }
#ifdef ENABLE_FP16
    if (M == 1 && input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        (output.getDataType() == ml::train::TensorDim::DataType::FP16 ||
         output.getDataType() == ml::train::TensorDim::DataType::FP32)) {
      // The activation binds its own residency plane. Hardcoding the
      // shared-plane pointer here reads a shadow nobody wrote whenever the
      // planner placed this activation on the device plane, and the GEMV then
      // produces all-zero logits -- every sampled token becomes the padding
      // token, from the first decode step on, with every layer before it
      // bit-correct.
      void *act = input.isClMem() ? input.getClMem()
                                  : static_cast<void *>(input.getData<_FP16>());
      const bool act_clmem = input.isClMem();
      const bool out_fp16 =
        output.getDataType() == ml::train::TensorDim::DataType::FP16;
      void *logits_host = out_fp16
                            ? static_cast<void *>(output.getData<_FP16>())
                            : static_cast<void *>(output.getData<float>());
      if (lmhead_int4_v8c_gemv_cl(w->weight_buf, w->scale_buf.get(), act,
                                  act_clmem, logits_host, out_fp16, N, K,
                                  w->zp_buf.get()))
        return true;
    }
#endif
    return false;
  }

  // Reused scratch buffers (grow-only pool). The weight backing + scale are
  // already cached per-weight; only the activation/output scratch scales with
  // (M_pad, K, N), so we grow these lazily and reuse them across forwards.
  cl_int err = CL_SUCCESS;
  const size_t act_elem =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16)
      ? sizeof(uint16_t)
      : sizeof(float);
  std::lock_guard<std::mutex> slock(v8c_cache_mtx());
  V8cScratch &sc = v8c_scratch();

  const int cur_dtype =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16) ? 1 : 0;
  // Device-plane input. Its ResidencyClass is GPU_CLMEM, so by construction
  // its producer wrote the planner sub-buffer -- getData() is a shadow nobody
  // filled. Read it device-direct (a device->device copy into sc.act_in) with
  // no SVM map.
  //
  // The placement IS the gate; there is deliberately no second one. An
  // allocator with no device plane, a pool that could not hand out a device
  // buffer, or a planner demotion all leave isClMem() false and this path
  // unused, so the operand always binds the plane the planner recorded.
  // Ordering is the command queue's: it is created in-order, so this copy is
  // ordered after the producer's write.
  const bool device_clmem_in = input.getMemoryData() &&
                               input.getMemoryData()->isClMem() &&
                               input.getMemoryData()->deviceMem() != nullptr;
  cl_mem clmem_in = device_clmem_in
                      ? static_cast<cl_mem>(input.getMemoryData()->deviceMem())
                      : nullptr;
  const void *cur_in_ptr =
    device_clmem_in ? static_cast<const void *>(clmem_in)
                    : static_cast<const void *>(input.getData<uint8_t>());

  // Shared-quant cache: its ADDRESS key was refused, because an address cannot
  // tell two activations apart.
  //
  // The cache skipped the staging copy and the activation-quant kernel when a
  // call arrived with the same (input data pointer, M, K, M_pad, dtype) as the
  // previous one, on the assumption that an activation ADDRESS identifies its
  // contents for the length of one forward pass -- so the wq/wk/wv and
  // gate/up siblings, which really do share one activation, quantize it once.
  //
  // The assumption does not hold. The tensor pool recycles an activation slot
  // as soon as its last reader has run, WITHIN the same pass, so two
  // consecutive quantized GEMMs can be handed the same address holding
  // different data. The second one then multiplies the first one's int8 with
  // no error and no log: the output is plausible, wrong, and the error grows
  // with depth.
  //
  // A graph with KV-shared decoder blocks meets this on every shared block --
  // the per-block "kv" normalization writes its output into the slot the
  // preceding projection's input has just vacated, so each shared block's K
  // and V projections consumed the preceding block's activation.
  //
  // Re-quantizing always costs one quant kernel per call that used to hit and
  // is always correct. Restoring the fast path needs a key that names the
  // PRODUCER of the activation, or an invalidation hooked to every write into
  // the activation plane; the address on its own cannot carry either.
  //
  // The dispatch write log IS that invalidation, so the fast path comes back on
  // a different question. Not "is this the same tensor" -- which a pooled
  // sub-buffer handle cannot answer -- but "are these the same BYTES", which is
  // what the reuse actually needs: if nothing has written that handle since the
  // cached quantisation was produced, this FC's own quantiser would read
  // precisely those bytes and produce precisely that result, whichever tensor
  // the graph believes the handle belongs to. The KV-shared case above then
  // fails the test on its own terms instead of being excluded by a blanket rule
  // -- the shared-KV normalization WROTE the handle between the two fanouts, so
  // the window is dirty and the sibling re-quantises. The recycled-slot half of
  // the rule is enforced one level up, in v8c_ring_advance(): handing out a
  // slot drops any cache entry naming it.
  //
  // Readmitted for the DEVICE plane only. On the shared/host plane the
  // activation can be rewritten by the host or by a non-dispatch transfer, and
  // the log records neither, so there the refusal stands unchanged.
  // NNTR_FUSE_QUANT_SHARE=0 refuses both planes, as before this change.
  static const bool quant_share_device = []() {
    const char *e = std::getenv("NNTR_FUSE_QUANT_SHARE");
    return !(e != nullptr && e[0] == '0');
  }();
  const bool shape_tuple_hit =
    device_clmem_in && sc.last_quant_in_ptr != nullptr &&
    sc.last_quant_in_ptr == cur_in_ptr && sc.last_quant_M == M &&
    sc.last_quant_K == K && sc.last_quant_M_pad == M_pad &&
    sc.last_quant_dtype == cur_dtype;
  const bool quant_cache_hit =
    quant_share_device && shape_tuple_hit &&
    opencl::Kernel::bufferUnchangedSince(static_cast<void *>(clmem_in),
                                         sc.last_quant_seq);
  if (quant_share_device && shape_tuple_hit && !quant_cache_hit)
    ++v8c_fuse_stats().dirty;
  const bool skip_upload_and_quant = quant_cache_hit;

  // [Lever 1] NNTR_FC_QUANT_DIRECT: on the cl_mem residency edge, quantize the
  // producer's (rmsnorm) cl_mem output IN PLACE, skipping the cl_mem ->
  // sc.act_in staging copy (the v8c_copy_h2h kernel) and the padded-row zero
  // write. The act-quant kernel reads exactly M real rows from clmem_in
  // (bounded by its own row guard), so there is no OOB on the M-row producer
  // buffer. Safe because GEMM output rows are independent: the padded
  // act_i8/scale/zp/rs rows [M, M_pad) are left stale and feed only the padded
  // GEMM output rows, which no consumer reads. Default ON; =0 opts out.
  static const bool fc_quant_direct = []() {
    const char *e = std::getenv("NNTR_FC_QUANT_DIRECT");
    return !e || e[0] != '0';
  }();
  const bool quant_direct_clmem = fc_quant_direct && device_clmem_in &&
                                  !skip_upload_and_quant && clmem_in != nullptr;

  // [norm->FC quant handoff] The producing norm already quantised this row
  // (see v8cNormQuantBegin). Take it only when the bytes at the input handle
  // are provably still the ones it quantised -- the dispatch write log answers
  // that -- and when the scratch it filled is large enough for this call's
  // padded M, since the FC's own grow-only ensure would otherwise reallocate
  // the buffer out from under the cached int8.
  bool nq_hit = false;
  if (quant_direct_clmem && cur_dtype == 1) {
    auto &fs = v8c_fuse_stats();
    if (sc.nq_slot < 0) {
      ++fs.nq_none;
    } else if (sc.nq_src != static_cast<void *>(clmem_in)) {
      ++fs.nq_other;
    } else if (sc.nq_rows != M || sc.nq_K != K ||
               sc.act_i8_bytes[sc.nq_slot] < (size_t)M_pad * K ||
               sc.act_scale_bytes[sc.nq_slot] < sizeof(float) * M_pad ||
               sc.act_zp_bytes[sc.nq_slot] < sizeof(int) * M_pad ||
               sc.act_rs_bytes[sc.nq_slot] < sizeof(int) * M_pad) {
      ++fs.nq_shape;
    } else if (!opencl::Kernel::bufferUnchangedSince(
                 static_cast<void *>(clmem_in), sc.nq_seq)) {
      ++fs.nq_dirty;
    } else {
      nq_hit = true;
    }
  }

  // SVM-pool input: the activation lives in GPU-visible SVM (the default
  // allocator for GPU graphs), so stage it with a device-side copy kernel
  // instead of a host upload.
  const bool in_svm =
    !device_clmem_in && input.getMemoryData() && input.getMemoryData()->isSVM();

  // Select this call's per-fanout activation slot. On a quant-cache HIT
  // (wk/wv after wq -- same input) reuse the slot that already holds the
  // int8 (read-only); on a MISS (a new fanout) advance the ring so this
  // fanout's quant WRITE lands in a buffer distinct from the prior fanout's
  // still-in-flight GEMM image READ (a WAR hazard through the image alias
  // the driver may not track).
  const int act_slot = quant_cache_hit ? sc.last_quant_slot
                       : nq_hit        ? sc.nq_slot
                                       : v8c_ring_advance(sc);
  // Grow only the chosen slot to this call's (M_pad, K). Grow-only => a hit
  // (same M_pad,K as the miss that filled it) never reallocates, so the
  // cached int8/scale/zp/rs survive for the wk/wv reuse.
  if (!v8c_ensure_buf(ctx, &sc.act_i8[act_slot], &sc.act_i8_bytes[act_slot],
                      (size_t)M_pad * K, CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_scale[act_slot],
                      &sc.act_scale_bytes[act_slot], sizeof(float) * M_pad,
                      CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_rs[act_slot], &sc.act_rs_bytes[act_slot],
                      sizeof(int) * M_pad, CL_MEM_READ_WRITE) ||
      !v8c_ensure_buf(ctx, &sc.act_zp[act_slot], &sc.act_zp_bytes[act_slot],
                      sizeof(int) * M_pad, CL_MEM_READ_WRITE))
    return false;
  cl_mem act_i8_arg = sc.act_i8[act_slot];
  cl_mem act_scale_arg = sc.act_scale[act_slot];
  cl_mem act_zp_arg = sc.act_zp[act_slot];
  cl_mem act_rs_arg = sc.act_rs[act_slot];

  // Submission batching. Mode 2: submit the accumulated batch BEFORE this FC
  // re-quants, so the quant's act-image WRITE and every GEMM image READ of
  // that slot (this FC + the cache-hit siblings that reuse it) share ONE
  // submission -- keeps every image write->read pair batch-local on drivers
  // with image-from-buffer texture-cache staleness across submissions.
  // Mode 1 (default on the image path): trailing flush after every FC,
  // recovering the norm->FC idle band. Mode 0 (default on the buffer path):
  // no explicit flush. NNTR_FC_FLUSH overrides.
  static const bool fc_flush_env_set = std::getenv("NNTR_FC_FLUSH") != nullptr;
  // The decode skip below pays only where the layer chain submits itself:
  // the image-attention arm flushes once per layer. The deterministic arm
  // (NNTR_DETERMINISTIC=1) runs the buffer attention, which drains the queue
  // per layer, and there the per-FC flush is what lets the GPU start the
  // layer's GEMVs while the host is still enqueueing them: without it the
  // 1.5B dense decode on that arm fell from 38.7 to 31.5 TPS. Same switch
  // the attention kernels read for their reduction order.
  static const bool fc_flush_decode_keep = []() {
    const char *det = std::getenv("NNTR_DETERMINISTIC");
    return det != nullptr && det[0] == '1';
  }();
  static const int fc_flush_mode = []() {
    const char *e = std::getenv("NNTR_FC_FLUSH");
    if (e)
      return std::atoi(e);
    return v8c_use_buffer_path() ? 0 : 1;
  }();
  if (fc_flush_mode == 2 && !skip_upload_and_quant)
    opencl::clFlush(q);

  // act_in is consumed only by the staging copies below; the direct path and
  // an act-cache hit never touch it.
  if (!skip_upload_and_quant && !quant_direct_clmem &&
      !v8c_ensure_buf(ctx, &sc.act_in, &sc.act_in_bytes,
                      (size_t)M_pad * K * act_elem, CL_MEM_READ_ONLY))
    return false;

  if (!skip_upload_and_quant) {
    if (quant_direct_clmem) {
      // [Lever 1] nothing to stage: the act-quant below reads clmem_in.
    } else if (device_clmem_in) {
      // Device->device: the same copy kernel, both arguments bound as buffers.
      auto *cc_in =
        static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
      auto kcp = cc_in->registerClKernel(
        v8c_util_kernels, cur_dtype == 1 ? "v8c_copy_h2h" : "v8c_copy_f2f");
      if (!kcp)
        return false;
      int nin = (int)((size_t)M * K);
      if (!kcp->SetKernelArguments(0, &clmem_in, sizeof(cl_mem)) ||
          !kcp->SetKernelArguments(1, &sc.act_in, sizeof(cl_mem)) ||
          !kcp->SetKernelArguments(2, &nin, sizeof(int)))
        return false;
      const int gws_in[3] = {(int)(((size_t)nin + 63) / 64 * 64), 1, 1};
      const int lws_in[3] = {64, 1, 1};
      if (!cc_in->command_queue_inst_.DispatchCommand(kcp, gws_in, lws_in))
        return false;
    } else if (in_svm) {
      // GPU copy of the SVM-resident activation into sc.act_in -- no host
      // upload. Downstream quant/image/GEMM see the same sc.act_in either way.
      v8c_copy_svm_to_clmem(cur_in_ptr, sc.act_in,
                            (unsigned int)((size_t)M * K), cur_dtype == 1);
    } else {
      if (opencl::clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE, 0,
                                       (size_t)M * K * act_elem, cur_in_ptr, 0,
                                       nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
    if (M_pad > M && !quant_direct_clmem) {
      // Zero-fill the padded rows so the act-quant kernel sees deterministic
      // values (per-row amax -> 0 -> scale defaults to 1, q=0, row_sum=0;
      // padded rows produce zero output).
      // The old form here staged the zeros in a local std::vector and handed
      // it to a NON-BLOCKING clEnqueueWriteBuffer: the vector was destroyed on
      // the next line while the command still named its bytes, which the
      // OpenCL contract forbids (the pad rows then quantise whatever the
      // allocator handed out). clEnqueueFillBuffer needs no host source at
      // all, so the pattern cannot come back.
      const size_t pad_bytes = (size_t)(M_pad - M) * K * act_elem;
      const cl_uchar zero_pat = 0;
      if (opencl::clEnqueueFillBuffer(q, sc.act_in, &zero_pat, sizeof(zero_pat),
                                      (size_t)M * K * act_elem, pad_bytes, 0,
                                      nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
  }

  try {
    // fp->int8 asymmetric act quant + zero-point + row_sum over M_pad rows.
    // Padded rows map to (scale=1, zp=0, q=0, row_sum=0), so they contribute
    // zero in the GEMM and don't pollute valid rows. Skipped on cache hit.
    if (nq_hit) {
      ++v8c_fuse_stats().nq_hit;
      if (sc.nq_site != nullptr)
        ++v8c_norm_sites()[sc.nq_site].claimed;
      // Seed the sibling shared-quant cache from the handoff: a sibling FC of
      // this fanout that can no longer reach the handoff (the norm published
      // once, and the handoff is consumed by shape) still reaches the same
      // int8 through the cache above.
      sc.last_quant_in_ptr = cur_in_ptr;
      sc.last_quant_M = M;
      sc.last_quant_K = K;
      sc.last_quant_M_pad = M_pad;
      sc.last_quant_dtype = cur_dtype;
      sc.last_quant_slot = act_slot;
      sc.last_quant_seq = sc.nq_seq;
      // [norm->FC quant handoff] The fused norm already wrote act_i8 /
      // act_scale / act_zp / act_rs for this exact row into this slot. Nothing
      // to stage, nothing to quantise; the GEMM binding below is unchanged.
    } else if (!skip_upload_and_quant) {
      // [Lever 1] quant the producer cl_mem (M real rows) directly when
      // quant_direct_clmem; otherwise the staged sc.act_in (M_pad rows,
      // including the zero pad).
      cl_mem quant_src = quant_direct_clmem ? clmem_in : sc.act_in;
      const unsigned int quant_rows = quant_direct_clmem ? M : M_pad;
      // The quantiser writes only v8c's own per-fanout scratch, never a graph
      // tensor.
      opencl::Kernel::noteDispatchWritesNothing();
      if (input.getDataType() == ml::train::TensorDim::DataType::FP16)
        quantize_act_v8c_fp16_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      else
        quantize_act_v8c_fp32_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      // Update cache key only after a successful quant. Record WHICH slot now
      // holds this input's int8 so a subsequent cache hit (wk/wv) reads the
      // right per-fanout buffer, not whatever the ring last pointed at, and at
      // WHICH dispatch it was produced, so the write log can say whether the
      // bytes behind that key still are the bytes that were quantised.
      sc.last_quant_in_ptr = cur_in_ptr;
      sc.last_quant_M = M;
      sc.last_quant_K = K;
      sc.last_quant_M_pad = M_pad;
      sc.last_quant_dtype = cur_dtype;
      sc.last_quant_slot = act_slot;
      sc.last_quant_seq = opencl::Kernel::dispatchSeq();
      ++v8c_fuse_stats().quant;
    } else {
      ++v8c_fuse_stats().share_hit;
    }

    // v8c GEMM input binding. The buffer path (Intel NEO) selects the *_buf
    // kernels whose args are __global uint4* -- they MUST be bound to raw
    // cl_mem buffers (the int8 act scratch and the weight backing buffer),
    // NOT image2d objects. Only the image-sampling path (Adreno) builds an
    // image2d view over the act buffer.
    const bool use_buf = v8c_buffer_path();
    cl_mem act_image = nullptr;
    if (!use_buf) {
      // Build the image2d view over the int8 act buffer (zero-copy). The view
      // is CACHED on the slot and reused across the fanout's GEMMs, rebuilt
      // only when the slot's buffer is grown or (M_pad, K) change -- removing
      // the per-call clCreateImage/clReleaseMemObject churn and the
      // exception-path image leak of a transient view.
      cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc adesc{};
      adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
      adesc.image_width = K / 16;
      adesc.image_height = M_pad;
      adesc.image_row_pitch = K;
      adesc.buffer = act_i8_arg;

      // NNTR_V8C_ACT_VIEW_SET=0 keeps one view per slot (the previous
      // behaviour), so the same binary can be its own control arm.
      static const bool view_set = []() {
        const char *e = std::getenv("NNTR_V8C_ACT_VIEW_SET");
        return !(e != nullptr && e[0] == '0');
      }();
      const int cap = view_set ? V8cScratch::kActViewsPerSlot : 1;
      auto &views = sc.act_views[act_slot];
      int hit = -1, free_slot = -1, victim = 0;
      for (int i = 0; i < cap; ++i) {
        if (views[i].image == nullptr) {
          if (free_slot < 0)
            free_slot = i;
          continue;
        }
        if (views[i].buf == act_i8_arg && views[i].M_pad == M_pad &&
            views[i].K == K) {
          hit = i;
          break;
        }
        if (views[victim].image == nullptr || views[i].lru < views[victim].lru)
          victim = i;
      }
      // A grown slot buffer invalidates every view over the old one; drop them
      // rather than let a stale alias be handed out.
      if (hit < 0) {
        for (int i = 0; i < cap; ++i) {
          if (views[i].image != nullptr && views[i].buf != act_i8_arg) {
            opencl::clReleaseMemObjectT(views[i].image);
            views[i] = V8cScratch::ActView{};
            if (free_slot < 0)
              free_slot = i;
          }
        }
        int use = free_slot >= 0 ? free_slot : victim;
        if (views[use].image != nullptr) {
          opencl::clReleaseMemObjectT(views[use].image);
          views[use] = V8cScratch::ActView{};
        }
        cl_mem img = opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &afmt,
                                            &adesc, nullptr, &err);
        if (err != CL_SUCCESS)
          throw std::runtime_error("act image view fail");
        views[use].image = img;
        views[use].buf = act_i8_arg;
        views[use].M_pad = M_pad;
        views[use].K = K;
        hit = use;
      }
      views[hit].lru = ++sc.act_view_clock;
      act_image = views[hit].image;
    }

    // Direct output (kernel-store, no copy): when the FC output is a
    // GPU_CLMEM-resident FP16 tensor, point the GEMM's Y straight at its
    // planner sub-buffer and let the TM=4 kernel's M_valid store guard bound
    // the write, eliminating the separate v8c_copy_h2h writer kernel (measured
    // 46 ms GPU + 182 enqueues per 1K prefill on the reference). Same
    // kernel->kernel ordering the copy writer relied on. NNTR_V8C_DIRECT_OUT=0
    // opts out.
    //
    // This tree carries none of the reference's y_fp16 debug consumers
    // (NNTR_CLMEM_PROBE / _DUALOUT / _OUTCHECK / _OUTBAR / NNTR_V8C_TRACE were
    // stripped as instrumentation), so the reference's y_dbg_consumer veto has
    // nothing to veto here and is not reproduced.
    const bool out_clmem =
      output.getMemoryData() && output.getMemoryData()->isClMem() &&
      output.getMemoryData()->deviceMem() != nullptr &&
      output.getDataType() == ml::train::TensorDim::DataType::FP16;
    static const bool direct_out_enabled = []() {
      const char *e = std::getenv("NNTR_V8C_DIRECT_OUT");
      return !(e && e[0] == '0');
    }();
    const bool direct_out = direct_out_enabled && out_clmem;

    // v8c GEMM -- run on padded M_pad rows, but only the valid M rows reach
    // the caller. y_fp16 only backs the NON-direct output paths; under
    // direct_out the GEMM stores into the planner buffer itself.
    if (!direct_out && !v8c_ensure_buf(ctx, &sc.y_fp16, &sc.y_fp16_bytes,
                                       sizeof(uint16_t) * (size_t)M_pad * N,
                                       CL_MEM_READ_WRITE))
      return false;
    cl_mem gemm_y_arg =
      direct_out ? static_cast<cl_mem>(output.getMemoryData()->deviceMem())
                 : sc.y_fp16;
    cl_mem gemm_act_arg = use_buf ? act_i8_arg : act_image;
    cl_mem gemm_wgt_arg = use_buf ? w->weight_buf : w->weight_image;
    // M_valid = the REAL row count, always: the kernel routes single-row
    // calls to the fast GEMV (M=1 decode) and multi-row calls to the TM=4
    // tiled kernel with the M_valid store guard; consumers only ever read
    // the real M rows.
    // Declare the GEMM's one device-plane write for the log: under direct_out
    // that is the output tensor's planner sub-buffer, otherwise the private
    // y_fp16 scratch. Either way it is the only buffer this dispatch stores
    // into, which is what an outstanding handoff has to be checked against.
    opencl::Kernel::noteDispatchWrites(static_cast<void *>(gemm_y_arg));
    gemm_int8_v8c_cl(gemm_act_arg, gemm_wgt_arg, act_scale_arg,
                     w->scale_buf.get(), act_rs_arg, act_zp_arg,
                     w->row_sum_w_int4.get(), gemm_y_arg, M_pad, N, K, M,
                     w->zp_buf.get());
    // NNTR_XE3_FC_SYNC=1: clFinish after the FC GEMM. Off by default on
    // every platform; =0 or leaving it unset keeps it off.
    //
    // It was added as a narrower form of the global drain for the
    // coarse-grained-SVM handoff on Intel, on the view that the FC GEMM was
    // the dominant op writing shared memory. The GEMM itself binds only device
    // buffers (the int8 activation scratch, the weights, and y_fp16 or the
    // planner sub-buffer), and on the in-order queue a device-buffer handoff
    // needs no drain. Where the FC does touch shared memory -- an SVM input
    // staged by v8c_copy_svm_to_clmem, an SVM output written by
    // v8c_write_output_resident -- those copy kernels are ordered by the
    // in-order queue, and the shared-memory unmap that hands the plane over
    // waits for the queue on a coarse-grain device (enqueueSVMUnmap).
    //
    // Measured on an Intel Xe3 iGPU with the per-dispatch drain in place
    // (NNTR_XE3_SYNC, then on by default): skipping this
    // sync left the generated token sequence unchanged on three models (a
    // dense 1.5B model on the XMX and dp4a lanes, a 2B-class model with
    // per-layer embeddings, and Gemma-4), 1K-token summarization prompts,
    // 30 runs per model against the synced reference. It cost one
    // clFinish per FC (169 of 461 per decoded token on the 1.5B model) and
    // 7-9% of decode throughput. Windows had it off already.
    static const bool xe3_fc_sync = []() {
      const char *e = std::getenv("NNTR_XE3_FC_SYNC");
      return e != nullptr && std::atoi(e) != 0;
    }();
    if (xe3_fc_sync)
      opencl::clFinish(q);

    // Output. An SVM-resident output (the default under the SVM pool) gets
    // the fp16 GEMM result written on the GPU -- plain copy for fp16,
    // cvt_h2f for fp32 -- with no host readback; otherwise read back the
    // valid M rows and convert on the host.
    // Device-plane output: the fp16 result goes into the planner sub-buffer
    // its consumers bind. Writing the shared-plane shadow instead leaves those
    // consumers reading a buffer nothing filled.
    const bool out_resident =
      !out_clmem && output.getMemoryData() && output.getMemoryData()->isSVM() &&
      (output.getDataType() == ml::train::TensorDim::DataType::FP32 ||
       output.getDataType() == ml::train::TensorDim::DataType::FP16);
    if (direct_out) {
      // The GEMM already stored into the planner sub-buffer: nothing to copy.
    } else if (out_clmem) {
      v8c_write_output_resident(
        sc.y_fp16, output, (unsigned int)((size_t)M * N), true,
        static_cast<void *>(output.getMemoryData()->deviceMem()));
    } else if (out_resident) {
      v8c_write_output_resident(
        sc.y_fp16, output, (unsigned int)((size_t)M * N),
        output.getDataType() == ml::train::TensorDim::DataType::FP16);
    } else {
      std::vector<uint16_t> y_host((size_t)M * N);
      if (opencl::clEnqueueReadBuffer(
            q, sc.y_fp16, CL_TRUE, 0, sizeof(uint16_t) * y_host.size(),
            y_host.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return false;
      if (output.getDataType() == ml::train::TensorDim::DataType::FP32) {
        float *out = output.getData<float>();
        for (size_t i = 0; i < y_host.size(); ++i)
          out[i] = v8c_h2f(y_host[i]);
      } else if (output.getDataType() == ml::train::TensorDim::DataType::FP16) {
        std::memcpy(output.getData<uint8_t>(), y_host.data(),
                    sizeof(uint16_t) * y_host.size());
      } else {
        throw std::runtime_error("unsupported output dtype");
      }
    }

    // Mode 1: submit this FC's enqueue chain now instead of at the next
    // blocking call -- recovers the norm->FC idle band on the image path.
    // Decode (M == 1) skips the per-FC submission: a token is ~580 launches,
    // and a flush after each of its 169 GEMVs cost 4.0 ms of host time per
    // token on an Adreno 840 (15 us each, 1.5B dense, 200-token cell) for a
    // batch of one or two kernels; the attention block's own flush once per
    // layer keeps the queue fed. Steady decode 58.1 -> 60.2 TPS, token ids
    // unchanged. NNTR_FC_FLUSH=1 restores the per-FC flush for decode too.
    if (fc_flush_mode == 1 &&
        (M > 1 || fc_flush_env_set || fc_flush_decode_keep))
      opencl::clFlush(q);
    // NNTR_ATTN_GPUHASH>=3: hash this FC's device-resident
    // output behind its own launches (no host sync; see attention_kernels).
    if (attn_gpuhash_mode() >= 3 && output.getMemoryData()) {
      const size_t words =
        (size_t)M * N *
        (output.getDataType() == ml::train::TensorDim::DataType::FP16 ? 1 : 2);
      if (output.getMemoryData()->isClMem() &&
          output.getMemoryData()->deviceMem() != nullptr)
        attn_gpuhash_note("fc", nullptr, output.getMemoryData()->deviceMem(),
                          words * 2, M);
      else if (output.getMemoryData()->isSVM())
        attn_gpuhash_note("fc", output.getData<uint8_t>(), nullptr, words * 2,
                          M);
    }
  } catch (...) {
    return false;
  }
  return true;
}

bool clmem_raise_cl(const Tensor &t, unsigned int valid_bytes) {
  if (!t.isClMem())
    return false;
  void *sub = t.getClMem();
  if (sub == nullptr)
    return false;
  // The sub-buffer covers the WHOLE tensor, so a nonzero-offset view cannot be
  // bridged from the base pointer. Fail loudly rather than misread.
  if (t.getOffset() != 0)
    throw std::runtime_error("clmem_raise_cl: nonzero-offset view unsupported");
  const size_t bytes = valid_bytes ? (size_t)valid_bytes : t.bytes();
  if (bytes == 0)
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_raise_cl: no GPU context");
  // Asynchronous upload: the queue is in-order, so it is already sequenced
  // ahead of every later kernel, and the host source is not written again
  // until the next forward pass.
  if (!cc->command_queue_inst_.EnqueueWriteBuffer(
        static_cast<cl_mem>(sub), bytes, t.getData<uint8_t>(), /*async=*/true))
    throw std::runtime_error("clmem_raise_cl: buffer upload failed for " +
                             t.getName());
  return true;
}

bool clmem_lower_cl(const Tensor &t, unsigned int valid_bytes) {
  if (!t.isClMem())
    return false;
  void *sub = t.getClMem();
  if (sub == nullptr)
    return false;
  // See clmem_raise_cl: offset-0 views only, loud failure otherwise.
  if (t.getOffset() != 0)
    throw std::runtime_error("clmem_lower_cl: nonzero-offset view unsupported");
  const size_t bytes = valid_bytes ? (size_t)valid_bytes : t.bytes();
  if (bytes == 0)
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_lower_cl: no GPU context");
  // Blocking read on the in-order queue: it waits for every command already
  // enqueued and then lands the bytes in host memory, which is what the host
  // consumer that runs next expects to find there.
  if (!cc->command_queue_inst_.EnqueueReadBuffer(
        static_cast<cl_mem>(sub), bytes, t.getData<uint8_t>(), /*async=*/false))
    throw std::runtime_error("clmem_lower_cl: buffer read-back failed for " +
                             t.getName());
  return true;
}
bool clmem_lower_range_cl(const Tensor &t, size_t byte_offset, size_t bytes) {
  if (!t.isClMem())
    return false;
  void *sub = t.getClMem();
  if (sub == nullptr || bytes == 0)
    return false;
  // See clmem_raise_cl: offset-0 views only, loud failure otherwise.
  if (t.getOffset() != 0)
    throw std::runtime_error(
      "clmem_lower_range_cl: nonzero-offset view unsupported");
  if (byte_offset > t.bytes() || bytes > t.bytes() - byte_offset)
    throw std::out_of_range("clmem_lower_range_cl: range outside " +
                            t.getName());
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_lower_range_cl: no GPU context");
  // Blocking, like clmem_lower_cl: the bytes are in host memory, and every
  // command enqueued before this one has finished, when the call returns.
  if (opencl::clEnqueueReadBuffer(
        cc->command_queue_inst_.GetCommandQueue(), static_cast<cl_mem>(sub),
        CL_TRUE, byte_offset, bytes, t.getData<uint8_t>() + byte_offset, 0,
        nullptr, nullptr) != CL_SUCCESS)
    throw std::runtime_error("clmem_lower_range_cl: buffer read-back failed "
                             "for " +
                             t.getName());
  return true;
}

/**
 * Debug: write rows of a layer output to NNTR_DUMP_ROWS_DIR. Inert unless the
 * caller (neuralnet.cpp) saw the env set. Reads a device-plane tensor into a
 * private bounce buffer so the run itself is not perturbed beyond the drain.
 */
void debug_dump_rows_cl(const std::string &name, const Tensor &t,
                        unsigned int rows, const char *dir, unsigned int seq) {
  // The caller's gate is a COMPILE-time one (#if ENABLE_OPENCL), so a build
  // that has OpenCL compiled in reaches here on every lane -- including a
  // --lane cuda run, which never registers the "gpu" Context.
  // getRegisteredContext() throws for a name that is not registered, which
  // turned a debug lever into an abort
  // ("[Engine] gpu Context is not registered"). Catch that: with no GPU Context
  // the tensor cannot be a cl_mem either, so the host arm below is the whole of
  // the work. On CUDA that arm needs two things the OpenCL one does not: a
  // device-ONLY node output has to be staged rather than dereferenced (below),
  // and the stream has to be drained first, which the caller does for us.
  ClContext *cc = nullptr;
  try {
    cc = static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  } catch (...) {
    cc = nullptr; // no OpenCL Context on this run: the host arm is all there is
  }
  const size_t W = t.width();
  const size_t es = t.getDataType() == Tdatatype::FP16   ? 2
                    : t.getDataType() == Tdatatype::FP32 ? 4
                                                         : 0;
  if (es == 0 || W == 0 || rows == 0 || rows > t.height())
    return;
  const size_t bytes = (size_t)rows * W * es;
  std::vector<uint8_t> buf(bytes);
  const char *plane = "host";
  if (t.isClMem() && t.getClMem() != nullptr && cc) {
    plane = "clmem";
    if (!cc->command_queue_inst_.EnqueueReadBufferRegion(
          static_cast<cl_mem>(t.getClMem()), bytes, buf.data(), 0,
          t.getOffset() * es, /*async=*/false))
      return;
  } else {
    if (cc)
      opencl::clFinish(cc->command_queue_inst_.GetCommandQueue());
    const uint8_t *src = t.getData<uint8_t>();
    if (src == nullptr)
      return;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // A cuda-engine node output can live in a device-ONLY pool (cudaMalloc, the
    // NNTR_CUDA_DEV_ACT / KV pools), which the host cannot dereference: the
    // memcpy below segfaults on it. Stage it instead. dev_only() is itself
    // engine-gated, so this costs a branch and creates no CUDA context on a
    // non-cuda run.
    if (nntrainer::cuda::dev_only(src)) {
      plane = "cudadev";
      if (!nntrainer::cuda::copy_any(buf.data(), src, bytes))
        return;
    } else
#endif
      std::memcpy(buf.data(), src, bytes);
  }
  char path[1024];
  std::snprintf(path, sizeof(path), "%s/%04u_%s.%s.%zux%zu.bin", dir, seq,
                name.c_str(), es == 2 ? "f16" : "f32", (size_t)rows, W);
  static const bool all_rows = std::getenv("NNTR_DUMP_ROWS_ALL") != nullptr;
  // NNTR_DUMP_ROWS_HASH=1 prints a hash per node instead of
  // writing a file: comparing two runs' stderr then names the first node (and,
  // with the seq counter, the first prefill chunk) whose output differs, which
  // is what a nondeterminism hunt needs and a directory of 16 x N files is not.
  static const bool hash_only = []() {
    const char *e = std::getenv("NNTR_DUMP_ROWS_HASH");
    return e != nullptr && e[0] != '0';
  }();
  if (hash_only) {
    auto fnv = [](const uint8_t *p, size_t n) {
      uint64_t h = 1469598103934665603ull;
      for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
      }
      return h;
    };
    const uint64_t hall = fnv(buf.data(), bytes);
    const uint64_t hlast =
      fnv(buf.data() + (size_t)(rows - 1) * W * es, W * es);
    std::fprintf(stderr,
                 "[ROWHASH] %04u %s plane=%s rows=%u W=%zu all=%016llx "
                 "last=%016llx\n",
                 seq, name.c_str(), plane, rows, W, (unsigned long long)hall,
                 (unsigned long long)hlast);
    return;
  }
  if (std::FILE *f = std::fopen(path, "wb")) {
    if (all_rows)
      std::fwrite(buf.data(), 1, bytes, f);
    else
      std::fwrite(buf.data() + (size_t)(rows - 1) * W * es, 1, W * es, f);
    std::fclose(f);
  }
  std::fprintf(stderr, "[dump] %04u %s plane=%s rows=%u W=%zu es=%zu\n", seq,
               name.c_str(), plane, rows, W, es);
}

void cl_fc_release_caches() {
  // [reload] Model-teardown counterpart to the process-lifetime state this file
  // latches. Each item was written as "build once, keep for the process" --
  // true for a one-shot CLI, false for an SDK consumer that destroys the handle
  // and loads another in the same process. No run is in flight at the call site
  // and every item is rebuilt lazily or by the next load, so an empty state is
  // always valid. Nothing here calls into the driver when nothing was cached,
  // so a CPU/CUDA-only run of an OpenCL-capable build never pokes the loader.
  std::lock_guard<std::mutex> lock(v8c_cache_mtx());
  // 1. The derived weight packs. Every entry OWNS its device backing, its scale
  //    and row-sum sub-buffers, so "keep for the process" means each load leaks
  //    a whole model's worth of device memory: measured on the Intel lane, the
  //    GPU ledger went 630 -> 1229 -> 1492 MB over three loads of one 1.5B
  //    model and init went 240 -> 850 ms. The map is also keyed by the weight's
  //    HOST address, which the next load's allocator hands out again; the
  //    (name, N, K) check on a hit keeps such an entry from being bound for
  //    ANOTHER weight, but a same-named hit is served -- and what it serves is
  //    the previous model's pack, whose aux sub-buffers point into an arena the
  //    next load is still bumping through (see v8c_release_aux_arena).
  v8c_weight_cache().clear();
  //    The weight zero points registered for those weights are keyed by
  //    weight NAME, which the next model reuses ("layer0_wq" ...): a survivor
  //    would be attached to a different model's weight of the same name.
  v8c_zp_registry().clear();
  // 2. The norm-site strike counters, keyed by the gamma HOST pointer: a
  //    recycled address hands the next model's norm the previous model's
  //    "written off, never fuse again" verdict.
  v8c_norm_sites().clear();
  // 3. The two quantisation handoffs in the shared scratch. Both name their
  //    input by a device-buffer handle plus a dispatch sequence number, and the
  //    soundness of both rests on the dispatch write log -- which across a
  //    teardown can say "unchanged" about a handle that now belongs to a
  //    different model, because loading one dispatches no kernels and the
  //    activation plane is USE_HOST_PTR over an SVM plane whose address the
  //    next load recycles (measured: the third load of a 1.5B model landed on
  //    the second's plane). Drop the identities; the scratch BUFFERS stay, so
  //    the next load does not pay to reallocate them.
  V8cScratch &sc = v8c_scratch();
  sc.nq_site = nullptr;
  sc.nq_src = nullptr;
  sc.nq_seq = 0;
  sc.nq_rows = 0;
  sc.nq_K = 0;
  sc.nq_slot = -1;
  sc.nq_pending_slot = -1;
  sc.last_quant_in_ptr = nullptr;
  sc.last_quant_seq = 0;
  sc.last_quant_gen = 0;
  sc.last_quant_M = 0;
  sc.last_quant_K = 0;
  sc.last_quant_M_pad = 0;
  sc.last_quant_dtype = -1;
  // 4. A teardown is as much a handle-epoch boundary as a creation: the next
  //    load's memory objects land on freed addresses, so every remaining
  //    handle-VALUE-keyed memo in the tree (the kernel-argument cache, the
  //    image->backing-buffer cache) has to be told that a value may now name a
  //    different object.
  opencl::clBumpHandleEpoch();
}
} // namespace nntrainer
