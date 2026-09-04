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
#include <clblast_interface.h>
#include <opencl_loader.h>

namespace nntrainer {
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

    /// shortcut handling in case of vector
    /// for vector, (1 * K) == (K * 1) in current memory layout...
    /// and plaese note that N, K, M is a fixed place holder after considering
    /// transpose.
    /// For example, there is no case like (1 * K) X (1 * K) while
    /// (1 * K) X (1 * M) can be a case
    /// case1: (1 * K) X (K * 1)
    if (M == 1 && N == 1) {
      // *rdata = dot_cl(data, mdata, K) + (*rdata);
      *rdata = dot_cl(K, data, mdata) + (*rdata);
    }
    /// case2: (M * K) X (K * 1)
    else if (N == 1) {
      gemv_cl(0, trans, dim1, dim2, 1.0f, data, lda, mdata, 0.0f, rdata, 1);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      gemv_cl(0, !trans_m, mdim1, mdim2, 1.0f, mdata, ldb, data, 0.0f, rdata,
              1);
    }
    /// case others: use gemm
    else {
      if (input.getFormat() == Tformat::NHWC) {
        sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
      } else {
        gemm_cl(0, trans, trans_m, M, N, K, 1.0f, data, (trans) ? M : K, mdata,
                (trans_m) ? K : N, 1.0f, rdata, N);
      }
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

    scal_cl(len, value, data);
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

  NNTR_THROW_IF(input.getData() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(result.getData() == nullptr, std::invalid_argument)
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

      for (unsigned int i = 0; i < result.batch() / input.batch(); ++i) {
        axpy_cl(input.size(), 1.0f, X, Y);
        Y += input.size();
      }
    } else if (result.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      unsigned int size_res = result.size();
      unsigned int size_input = input.size();
      _FP16 *data_res = result.getData<_FP16>();
      const _FP16 *data_input = input.getData<_FP16>();

      addition_cl(data_input, data_res, size_input, size_res);

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

void copyCl(const Tensor &input, Tensor &result) {
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data = input.getData();
    float *rdata = result.getData();

    unsigned int len = input.size();

    copy_cl(len, data, rdata);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, copyCl not supported for FP16");
#endif
  }
}

float nrm2Cl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = nrm2_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, nrm2Cl not supported for FP16");
#endif
  }

  return result;
}

float asumCl(const Tensor &input) {
  float result = 0.0f;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = asum_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, asumCl not supported for FP16");
#endif
  }

  return result;
}

int amaxCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amax_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

int aminCl(const Tensor &input) {
  int result = 0;
  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data = input.getData();
    unsigned int len = input.size();

    result = amin_cl(len, data);
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    throw std::runtime_error("Error: Currently, amaxCl not supported for FP16");
#endif
  }

  return result;
}

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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <memory>
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
      opencl::clReleaseMemObject(mem_);
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
  // Forward-pass generation this weight was last dispatched in. Used only to
  // INFER a pass boundary for the shared-quant cache below (a weight is
  // dispatched at most once per pass, so seeing it twice in one generation
  // means a new pass began). The inference cannot cover every graph shape, so
  // the dispatches it cannot vouch for refuse the cache rather than risk a
  // stale hit -- see the boundary note in dotCl_v8c.
  unsigned long long last_use_gen = 0;
};

// Buffer-path (NNTR_V8C_BUF): on Intel NEO the v8c GEMM uses the *_buf kernels
// whose args are declared __global uint4* — they must be bound to raw cl_mem
// BUFFERS, not image2d objects. Single source of truth is
// blas_kernels.cpp::v8c_use_buffer_path() (caps-derived from vendor_id; the env
// flag still overrides); this file-local name forwards to it.
static bool v8c_buffer_path() { return v8c_use_buffer_path(); }

// [engine=gpu fold] The v8c int8×QINT4 FC GEMM is the GPU FC path's default —
// this gate is only reached from dotCl_v8c (ClComputeOps::fc), i.e. an
// engine=gpu FC, and the host fallback it guards is byte-identical, so
// defaulting it ON

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
  // Cached image2d-from-buffer view per slot, built once per (buffer, M_pad,
  // K) and reused across the fanout's GEMMs instead of per-call create/release
  // (which also leaked on the exception path). Rebuilt when the slot's buffer
  // is grown or M_pad/K change.
  cl_mem act_image[V8C_ACT_SLOTS] = {};
  cl_mem act_image_buf[V8C_ACT_SLOTS] = {};
  unsigned int act_image_M_pad[V8C_ACT_SLOTS] = {};
  unsigned int act_image_K[V8C_ACT_SLOTS] = {};
  int ring_pos = 0; /**< last slot handed out; advance-on-miss */
  cl_mem y_fp16 = nullptr;
  size_t y_fp16_bytes = 0;

  // Shared-quant cache (paper §3.6 fused-quant insight, host-side). Qwen3-style
  // layer graphs dispatch consecutive FC calls with the SAME input pointer
  // (wq/wk/wv read one post-RMSNorm activation; gate/up share theirs). After
  // the first call populates act_i8/act_scale/act_zp/act_rs for that input,
  // the sibling calls skip the upload AND the quant kernel.
  //
  // Cache key: (pass generation, input data pointer, M, K, M_pad, dtype).
  // Pointer identity is sufficient within one forward pass since the layer
  // graph executes serially — the input buffer isn't aliased between
  // dispatches. It is NOT sufficient across passes: activations live in the
  // tensor pool, whose addresses are recycled every pass, so the same
  // (pointer, shape, dtype) tuple names DIFFERENT data one pass later. The
  // generation below scopes a hit to the pass that produced it, and a hit is
  // taken only where the boundary is provably established: the boundary is
  // inferred from repeated weight dispatch, so a dispatch that cannot have
  // participated in that inference bypasses the cache and warns once (see the
  // boundary note in dotCl_v8c).
  unsigned long long pass_gen = 1; /**< current forward-pass generation */
  /** generation whose activation the cached int8 belongs to */
  unsigned long long last_quant_gen = 0;
  const void *last_quant_in_ptr = nullptr;
  unsigned int last_quant_M = 0;
  unsigned int last_quant_K = 0;
  unsigned int last_quant_M_pad = 0;
  int last_quant_dtype = -1;
  int last_quant_slot = 0; /**< slot whose int8 the cache hit refers to */
};

static V8cScratch &v8c_scratch() {
  static V8cScratch s;
  return s;
}

// Ensure *buf has at least `bytes` capacity with the given flags; (re)alloc
// only when too small. Returns false on alloc failure.
static bool v8c_ensure_buf(cl_context ctx, cl_mem *buf, size_t *cap,
                           size_t bytes, cl_mem_flags flags) {
  if (*buf && *cap >= bytes)
    return true;
  if (*buf) {
    opencl::clReleaseMemObject(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  *buf = opencl::clCreateBuffer(ctx, flags, bytes, nullptr, &err);
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

// Get or build the cached v8c weight backing for a given int4 (QS4CX) weight.
// Returns nullptr if shape unsupported (caller falls back).
static V8cWeightEntry *v8c_get_or_build_weight(const Tensor &weight,
                                               unsigned int K, unsigned int N) {
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
  const uint8_t *nibbles = weight.getData<uint8_t>();
  if (!nibbles)
    return nullptr;
  // int4 weights are QS4CX: row-major plain nibbles (uint4 = int4+8, no XOR) +
  // per-output-channel fp32 scale. A legacy QINT4 .bin is re-laid-out to this
  // form at load (QS4CX_Tensor::read), so the v8c backing has a single source.
  //
  if (weight.getDataType() != ml::train::TensorDim::DataType::QS4CX)
    return nullptr;
  V8cWeightEntry e;
  {
    cl_mem sb = nullptr;
    cl_mem rsw = nullptr;
    try {
      const float *fp32_scales = weight.getScale<float>();
      if (!fp32_scales)
        return nullptr;
      e.backing = make_v8c_weight_backing_from_qs4cx(nibbles, fp32_scales, N, K,
                                                     &sb, &rsw);
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

  // [NNTR_V8C_DROP_PLAIN=1, x86-only, OPT-IN] The device backing + scale buf +
  // row-sum built above are the only things the v8c GPU path reads from now
  // on, so dropping the plain QS4CX pages after the build reclaims ~the whole
  // FC weight footprint from host RSS. INWARD page alignment so pages shared
  // with neighboring pool tensors are never touched. May silently no-op if the
  // driver pinned the SVM pages (result is logged).
  //
  // DANGEROUS, hence opt-in everywhere: dropped pages read back as ZEROS, and
  // a live host consumer of the plain payload still exists on x86 -- the
  // ClComputeOps QS4CX fallback calls Tensor::dot, and FloatTensor::dot
  // dispatches QS4CX to dotQs4cx(), which reads exactly these nibbles plus the
  // fp32 scale tail (float_tensor.cpp). That fallback is reachable: dotCl_v8c
  // returns false from ~10 sites AFTER the weight has been built, cached and
  // dropped (allocation failures, kernel-registration failures, and the
  // imageless untied-lm_head branch, which returns false unconditionally on an
  // fp16-disabled build). The result would be a well-formed, entirely wrong
  // output with no exception and no log. Re-enabling by default needs the
  // fallback to fail loudly on a dropped weight first.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  {
    static const bool drop_plain = []() {
      const char *v = std::getenv("NNTR_V8C_DROP_PLAIN");
      if (v != nullptr)
        return v[0] == '1';
      // Opt-in on every platform. DiscardVirtualMemory does work on the WDDM
      // SVM host shadow (verified on every weight, goldens byte-identical),
      // but the win is throughput/footprint while the failure mode is silent
      // wrong output through the host fallback described above, so it must not
      // be the default until that fallback rejects a dropped weight.
      return false;
    }();
    if (drop_plain) {
      const size_t payload = (size_t)N * (((size_t)K + 1) / 2) // nibbles
                             + (size_t)N * sizeof(float);      // fp32 scales
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

// Eager v8c weight build (see header). Moves the lazy per-weight nibble
// permute + upload out of the first timed prefill: the CL FC layer calls this
// right after its weight is read at model load. No-op (false) off the v8c
// path. Lives OUTSIDE the anonymous namespace (public symbol).
bool dotCl_v8c_prebuild_weight(const Tensor &weight) {
  if (!v8c_env_enabled())
    return false;
  if (weight.getDataType() != ml::train::TensorDim::DataType::QS4CX)
    return false;
  const unsigned int N = weight.width();
  const unsigned int K = weight.height();
  if (N == 0 || K == 0 || N % 8 != 0 || K % 32 != 0)
    return false;
  return v8c_get_or_build_weight(weight, K, N) != nullptr;
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
)CL";

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
  // next op's unmap and the host need not block. NNTR_FC_SVM_SYNC=1 makes the
  // map BLOCKING (coarse-grained-SVM coherence probe for drivers where the
  // async handoff shows stale reads).
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
  if (weight.getDataType() != ml::train::TensorDim::DataType::QS4CX)
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
  // Only applied for prefill-sized M (M >= align): decode (M=1) must never pad
  // to 64 (that would be a 64x FC blow-up) -- guarded by eff_align below.
  constexpr unsigned int V8C_MPAD_ALIGN = 64;
  static_assert(V8C_MPAD_ALIGN % V8C_TM == 0,
                "M_pad alignment must stay a multiple of the kernel tile");
  const unsigned int eff_align =
    (M >= V8C_MPAD_ALIGN) ? V8C_MPAD_ALIGN : V8C_TM;
  const unsigned int M_pad = (M + eff_align - 1) / eff_align * eff_align;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  V8cWeightEntry *w = v8c_get_or_build_weight(weight, K, N);
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
    // weight is the untied lm_head: it uses no shared-quant cache itself, but
    // it is the one FC guaranteed to be dispatched on every decode pass, so
    // leaving it out would hide a boundary from every weight that does use
    // the cache. The scratch lock is taken and released here because the GEMV
    // below runs outside it.
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
                                  act_clmem, logits_host, out_fp16, N, K))
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

  // [pass boundary] The shared-quant cache below is keyed on the input
  // ADDRESS, and activation addresses come from the tensor pool, which
  // recycles them every forward pass. Pointer identity therefore only proves
  // data identity WITHIN one pass; one pass later the same (pointer, M, K,
  // M_pad, dtype) tuple names a freshly written activation while the cached
  // int8 still holds the previous pass's values, and the GEMM would multiply
  // the stale quant with no error and no log.
  // The boundary is INFERRED here instead of being plumbed down from the model
  // graph: every FC weight is dispatched at most once per forward pass, so
  // meeting a weight for the second time in the same generation IS the next
  // pass.
  // The generation is bumped only when a weight is met twice in the SAME
  // generation, so the boundary is seen only if the first v8c FC of the new
  // pass was ALSO dispatched in the previous pass. When the first v8c FC of a
  // pass is a weight that was NOT dispatched in the previous pass -- per-token
  // MoE/expert routing, a conditionally skipped layer, a second graph sharing
  // this process-global scratch, or any first-ever dispatch -- no bump
  // happens and last_quant_gen still equals pass_gen, so the generation alone
  // would let the stale cross-pass hit through.
  // That case is therefore not merely noted, it is REFUSED. A dispatch may use
  // the cache only when it PROVABLY took part in the inference itself, which
  // is exactly: the weight has a previous generation (it is not a first-ever
  // dispatch), and that generation is one behind the current one after the
  // bump above. The two ways to satisfy it are the two shapes the detector
  // understands:
  //   - a weight met twice bumps, so its previous generation becomes
  //     pass_gen - 1 by construction -- the first FC of a steady-state pass;
  //   - a weight already dispatched in the immediately preceding generation is
  //     at pass_gen - 1 without a bump -- every later FC of that pass.
  // Everything else -- a first-ever dispatch, a weight that skipped whole
  // generations, a weight belonging to a second graph -- fails it, and that
  // set is precisely the enumeration above. Those dispatches bypass the cache:
  // the activation is re-quantized, costing one kernel and never a wrong
  // answer. The cost is bounded and paid where it is owed -- the whole first
  // forward pass bypasses, since no weight has a previous generation yet, and
  // from the second pass on a static graph caches exactly as before.
  // The complete fix bumps the generation at forward-pass entry, which needs a
  // per-forward seam this backend does not have today; until it exists the
  // cache is trusted only where it is provably sound.
  // Coupling: pass_gen and last_use_gen are plain fields, safe only because
  // the v8c_cache_mtx() lock_guard taken above spans the rest of dotCl_v8c.
  const unsigned long long prev_use_gen = w->last_use_gen;
  if (prev_use_gen != 0 && prev_use_gen == sc.pass_gen)
    ++sc.pass_gen;
  w->last_use_gen = sc.pass_gen;
  const bool boundary_established =
    (prev_use_gen != 0 && prev_use_gen + 1 == sc.pass_gen);

  // Shared-quant cache. For host/SVM inputs the (data_ptr, shape, dtype)
  // tuple uniquely identifies the activation within one forward pass, so a
  // hit means the same data is already int8-quantized in sc.act_i8 and both
  // the staging copy and the quant kernel can be skipped (the wq/wk/wv and
  // gate/up sibling FCs).
  const bool quant_tuple_hit =
    sc.last_quant_gen == sc.pass_gen && sc.last_quant_in_ptr != nullptr &&
    sc.last_quant_in_ptr == cur_in_ptr && sc.last_quant_M == M &&
    sc.last_quant_K == K && sc.last_quant_M_pad == M_pad &&
    sc.last_quant_dtype == cur_dtype;
  const bool quant_cache_hit = quant_tuple_hit && boundary_established;
  // A bypass during the very first pass is expected (no weight has a previous
  // generation yet) and costs one quant kernel per FC, once. A bypass after
  // the detector has fired at least once is the report-worthy case: this
  // process is running a graph whose dispatch set the heuristic cannot track.
  if (quant_tuple_hit && !boundary_established && sc.pass_gen > 1) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      ml_logw("[v8c] shared-quant cache bypassed for %s: this weight did not "
              "take part in the previous forward pass (expert routing, a "
              "skipped layer, or a second graph on this process-global "
              "scratch), so the pass boundary cannot be inferred from weight "
              "reuse. Correctness is kept by re-quantizing the activation; "
              "the saved quant kernel is not.",
              weight.getName().c_str());
    }
  }
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
  const int act_slot = quant_cache_hit
                         ? sc.last_quant_slot
                         : (sc.ring_pos = (sc.ring_pos + 1) % V8C_ACT_SLOTS);
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
      const size_t pad_bytes = (size_t)(M_pad - M) * K * act_elem;
      std::vector<uint8_t> zeros(pad_bytes, 0);
      if (opencl::clEnqueueWriteBuffer(
            q, sc.act_in, CL_FALSE, (size_t)M * K * act_elem, pad_bytes,
            zeros.data(), 0, nullptr, nullptr) != CL_SUCCESS)
        return false;
    }
  }

  try {
    // fp->int8 asymmetric act quant + zero-point + row_sum over M_pad rows.
    // Padded rows map to (scale=1, zp=0, q=0, row_sum=0), so they contribute
    // zero in the GEMM and don't pollute valid rows. Skipped on cache hit.
    if (!skip_upload_and_quant) {
      // [Lever 1] quant the producer cl_mem (M real rows) directly when
      // quant_direct_clmem; otherwise the staged sc.act_in (M_pad rows,
      // including the zero pad).
      cl_mem quant_src = quant_direct_clmem ? clmem_in : sc.act_in;
      const unsigned int quant_rows = quant_direct_clmem ? M : M_pad;
      if (input.getDataType() == ml::train::TensorDim::DataType::FP16)
        quantize_act_v8c_fp16_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      else
        quantize_act_v8c_fp32_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      // Update cache key only after a successful quant. Record WHICH slot now
      // holds this input's int8 so a subsequent cache hit (wk/wv) reads the
      // right per-fanout buffer, not whatever the ring last pointed at, and
      // WHICH pass generation it belongs to so the next pass cannot hit it.
      sc.last_quant_gen = sc.pass_gen;
      sc.last_quant_in_ptr = cur_in_ptr;
      sc.last_quant_M = M;
      sc.last_quant_K = K;
      sc.last_quant_M_pad = M_pad;
      sc.last_quant_dtype = cur_dtype;
      sc.last_quant_slot = act_slot;
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
      if (sc.act_image[act_slot] == nullptr ||
          sc.act_image_buf[act_slot] != act_i8_arg ||
          sc.act_image_M_pad[act_slot] != M_pad ||
          sc.act_image_K[act_slot] != K) {
        if (sc.act_image[act_slot]) {
          opencl::clReleaseMemObject(sc.act_image[act_slot]);
          sc.act_image[act_slot] = nullptr;
        }
        cl_mem img = opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &afmt, &adesc,
                                           nullptr, &err);
        if (err != CL_SUCCESS)
          throw std::runtime_error("act image view fail");
        sc.act_image[act_slot] = img;
        sc.act_image_buf[act_slot] = act_i8_arg;
        sc.act_image_M_pad[act_slot] = M_pad;
        sc.act_image_K[act_slot] = K;
      }
      act_image = sc.act_image[act_slot];
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
    gemm_int8_v8c_cl(gemm_act_arg, gemm_wgt_arg, act_scale_arg,
                     w->scale_buf.get(), act_rs_arg, act_zp_arg,
                     w->row_sum_w_int4.get(), gemm_y_arg, M_pad, N, K, M);
    // NNTR_XE3_FC_SYNC: narrowed coarse-grain-SVM coherence fix. The in-order
    // queue does not give kernel->kernel coarse-grained-SVM coherence on some
    // Intel drivers; the global drain (NNTR_XE3_SYNC, clFinish after EVERY
    // SVM dispatch) fixes it but serializes decode. A clFinish after the FC
    // GEMM alone is sufficient (it is the dominant SVM-producing op), so
    // draining only here keeps coherence while restoring decode pipelining.
    // Value-parsed so NNTR_XE3_FC_SYNC=0 disables.
    static const bool xe3_fc_sync = []() {
      const char *e = std::getenv("NNTR_XE3_FC_SYNC");
      if (e)
        return std::atoi(e) != 0;
#ifdef _WIN32
      // Windows/WDDM default-OFF: a battery of cold-boot goldens, token-class
      // A/B runs and long-context summarizations with the drain skipped found
      // no coherence failure attributable to it there, and it costs 15-25% of
      // decode on that stack. The env still turns it on explicitly.
      return false;
#else
      // The stale read reproduces on Intel; every other vendor observed is
      // coherent across the handoff already. vendor_id is a queryable,
      // vendor-wide attribute -- not a device-name match.
      constexpr uint32_t INTEL_VENDOR_ID = 0x8086;
      return ClContext::Global().caps().vendor_id == INTEL_VENDOR_ID;
#endif
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
    if (fc_flush_mode == 1)
      opencl::clFlush(q);
  } catch (...) {
    return false;
  }
  return true;
}

} // namespace nntrainer
