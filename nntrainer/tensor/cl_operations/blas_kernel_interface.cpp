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
#include <env_compat.h>

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
      trans ? sgemv_cl(data, mdata, rdata, trans, dim2, dim1, lda)
            : sgemv_cl(data, mdata, rdata, trans, dim1, dim2, lda);
    }
    /// case3: (1 * K) X (K * N) = 1 * N = R
    /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
    /// Effectively a translation of sgemv
    else if (M == 1) {
      trans_m ? sgemv_cl(mdata, data, rdata, !trans_m, mdim1, mdim2, ldb)
              : sgemv_cl(mdata, data, rdata, !trans_m, mdim2, mdim1, ldb);
    }
    /// case others: use sgemm
    else {
      sgemm_cl(trans, trans_m, data, mdata, rdata, M, N, K, lda, ldb, ldc);
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

  // Bind device memory directly (SVM-direct, in-place accumulate) only when
  // both tensors are GPU-resident (SVM pool); otherwise fall back to the host
  // round-trip. Keeps the residual on the GPU when residency is enabled.
  const bool use_svm = result.getMemoryData() &&
                       result.getMemoryData()->isSVM() &&
                       input.getMemoryData() && input.getMemoryData()->isSVM();

  // Broadcasting done for the case where batch size vary for both inputs
  // If batch size vary, batch size of input must be 1
  if ((result.getDim() == input.getDim()) ||
      (result.getDim() != input.getDim() && input.batch() == 1 &&
       result.channel() == input.channel() &&
       result.height() == input.height() && result.width() == input.width())) {

    if (result.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *Y = result.getData();
      const float *X = input.getData();
      const unsigned int size_input = input.size();

      for (unsigned int i = 0; i < result.batch() / input.batch(); ++i) {
        // axpy with alpha=1 is just an elementwise add. Use the in-tree
        // addition_cl kernel instead of the CLBlast axpy route so FP32 add
        // works without CLBlast (FP16 already uses addition_cl below).
        addition_cl(X, Y, size_input, size_input, use_svm);
        Y += size_input;
      }
    } else if (result.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      unsigned int size_res = result.size();
      unsigned int size_input = input.size();
      _FP16 *data_res = result.getData<_FP16>();
      const _FP16 *data_input = input.getData<_FP16>();

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
// v8c (paper 8/4/4) dispatch entry — env-gated, dotCl fallback.
// =============================================================================
#include "blas_kernels.h"
#include "cl_tensor_backing_pool.h"
#include "cl_tensor_view.h"
#include <atomic>
#include <chrono>
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <memory>
#include <mutex>
#include <network_graph.h> // resolveResidentEdge (cl_mem residency overlay)
#include <unordered_map>

namespace nntrainer {
namespace {
/**
 * @brief Cached per-weight GPU residency for the v8c int8xint4 FC path:
 *        the packed int4 backing plus its derived scale / row-sum buffers.
 */
struct V8cWeightEntry {
  std::unique_ptr<tv::TensorBacking> backing;
  cl_mem scale_buf = nullptr;      // [N] fp32 recip-scale (owned)
  cl_mem row_sum_w_int4 = nullptr; // [N] int32 sum_k(int4 w_nk) (owned)
  unsigned int N = 0, K = 0;
  cl_mem weight_image =
    nullptr; // cached image2d view (also released via TensorBacking)
  cl_mem weight_buf = nullptr; // raw backing buffer (buffer-path / Intel NEO)
  // N exceeds the device image2d height cap (the untied lm_head, N=vocab), so
  // an image view can never exist for this weight. Kept SEPARATE from
  // (weight_image == nullptr): on the buffer path the image is skipped for
  // every weight, and only this flag may route to the imageless lm_head GEMV.
  bool huge_n = false;
};

// Buffer-path (NNTR_V8C_BUF): on Intel NEO the v8c GEMM uses the *_buf kernels
// whose args are declared __global uint4* — they must be bound to raw cl_mem
// BUFFERS, not image2d objects. The single source of truth is
// blas_kernels.cpp::v8c_use_buffer_path() (caps-derived from vendor_id; the env
// flag still overrides); this file-local name forwards to it.
static bool v8c_buffer_path() { return v8c_use_buffer_path(); }

// [engine=gpu fold] The v8c int8×QINT4 FC GEMM is the GPU FC path's default —
// this gate is only reached from dotCl_v8c (ClComputeOps::fc), i.e. an
// engine=gpu FC, and the host fallback it guards is byte-identical, so
// defaulting it ON retires the must-set NNTR_FC_INT8_GPU. The flag still
// DISABLES it (=0) for A/B.
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

  // Step 2b.0 shared-quant cache (paper §3.6 fused-quant motivation,
  // host-side). Qwen3 layer graph dispatches three consecutive dotCl_v8c calls
  // with the SAME input pointer (wq/wk/wv all read the same post-RMSNorm
  // activation), and similarly gate/up MLP FCs share their input. After the
  // first call populates act_i8/act_scale/act_zp/act_rs for that input, the
  // next 2-of-3 calls can skip the host→device upload AND the quant kernel
  // entirely.
  //
  // Cache key: (input data pointer, M, K, M_pad, dtype). Pointer identity is
  // sufficient within one forward pass since the layer graph executes
  // serially — the input buffer isn't aliased between dispatches.
  const void *last_quant_in_ptr = nullptr;
  unsigned int last_quant_M = 0;
  unsigned int last_quant_K = 0;
  unsigned int last_quant_M_pad = 0;
  int last_quant_dtype = -1;
  int last_quant_slot = 0; /**< slot whose int8 the cache hit refers to */
  unsigned long long last_quant_resident_generation = 0;

  // [FC->FC chained edge] Identity + ACTUAL cl_mem store target of the last
  // v8c FC's cl_mem-plane output. A model whose graph chains two v8c FCs with
  // no op in between (e.g. attention gate_down->gate_up, ffn
  // gate_up->gate_down) needs this:
  // the consumer's act-quant source derived via input.getMemoryData()->
  // deviceMem() must resolve to the exact buffer the producer stored to
  // (direct_out GEMM store or the kernel writer). Recording the producer's
  // real target lets the consumer rebind when the derivations diverge
  // (ClBufferPool per-padded-offset dedup). Single entry, overwritten by
  // every FC call; never matches for gemma4/qwen3/gemma2 (no FC->FC edges).
  const void *last_fc_out_md = nullptr;
  cl_mem last_fc_out_clmem = nullptr;
  unsigned int last_fc_out_M = 0, last_fc_out_N = 0;
};

// Process-global Segment A resident-buffer generation counter. Producers
// (Segment A's RMSNorm helpers) bump this on every successful write to a
// resident TensorBacking, signalling to dotCl_v8c that any quant cache
// keyed on a backing pointer is now stale. Same forward pass / multiple
// FCs reusing the same backing all share the generation, so the cache
// still hits on wq→wk→wv within the pass.
static std::atomic<unsigned long long> g_resident_quant_generation{0};
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
    clReleaseMemObject(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  *buf = clCreateBuffer(ctx, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  // clCreateBuffer content is UNDEFINED until first write. Padded rows
  // ([M, M_pad) under quant-direct) and partial first uses read these bytes
  // before any producer writes them: Linux NEO hands zeroed pages (masking
  // this), WDDM may recycle garbage that differs per process — a per-process
  // divergence class. Zero once per (re)allocation; the in-order
  // queue sequences the fill ahead of every later consumer. Cost: create-time
  // only (scratch buffers are grow-only).
  {
    auto *cc =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    const cl_uchar fill = 0;
    clEnqueueFillBuffer(cc->command_queue_inst_.GetCommandQueue(), *buf, &fill,
                        sizeof(fill), 0, bytes, 0, nullptr, nullptr);
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
      // Validate the pointer-keyed hit: a freed/re-used host weight pointer
      // (e.g. FSU) would otherwise silently return ANOTHER weight's device
      // pack (wrong N/K backing bound to the GEMM). Rebuild on mismatch.
      if (it->second.N == N && it->second.K == K)
        return &it->second;
      std::fprintf(stderr,
                   "[v8c] weight-cache shape mismatch for key=%p: cached N=%u "
                   "K=%u vs requested N=%u K=%u -- rebuilding\n",
                   key, it->second.N, it->second.K, N, K);
      cache.erase(it);
    }
  }
  const uint8_t *nibbles = weight.getData<uint8_t>();
  if (!nibbles)
    return nullptr;
  // int4 weights are QS4CX: row-major plain nibbles (uint4 = int4+8, no XOR) +
  // per-output-channel fp32 scale. A legacy QINT4 .bin is re-laid-out to this
  // form at load (QS4CX_Tensor::read), so the v8c backing has a single source.
  if (weight.getDataType() != ml::train::TensorDim::DataType::QS4CX)
    return nullptr;
  V8cWeightEntry e;
  cl_mem sb = nullptr;
  cl_mem rsw = nullptr;
  try {
    const float *fp32_scales = weight.getScale<float>();
    if (!fp32_scales)
      return nullptr;
    e.backing =
      make_v8c_weight_backing_from_qs4cx(nibbles, fp32_scales, N, K, &sb, &rsw);
  } catch (...) {
    return nullptr;
  }
  e.scale_buf = sb;
  e.row_sum_w_int4 = rsw;
  e.N = N;
  e.K = K;
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
  // [Adreno pitch fix] Row pitch padded to a 256-byte multiple to match the
  // padded weight backing rows (make_v8c_weight_backing_from_qs4cx) so
  // image2d-from-buffer creation satisfies CL_DEVICE_IMAGE_PITCH_ALIGNMENT on
  // Adreno. Intel NEO (no image) keeps K/2. The logical texel width stays K/32.
  ws.row_pitch_bytes = ClContext::Global().caps().image_v8c
                         ? (((size_t)K / 2 + 63) / 64) * 64
                         : (size_t)K / 2;

  // Is N past the device's image2d height cap? Ask the device instead of
  // inferring it from a failed clCreateImage, so the buffer path (which never
  // creates the image) can still identify the oversized lm_head.
  {
    auto *cc =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    size_t img_h_cap = 0;
    clGetDeviceInfo(cc->context_inst_.GetDeviceId(),
                    CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(img_h_cap), &img_h_cap,
                    nullptr);
    e.huge_n = (img_h_cap != 0 && (size_t)N > img_h_cap);
  }

  // [buffer-path image skip] On the buffer path (Intel NEO) the GEMM binds
  // weight_buf and NEVER reads weight_image (see `use_buf ? weight_buf :
  // weight_image` in the dispatch) -- the image is a dead object. The skip
  // avoids creating hundreds of dead cl_mem objects; outputs are
  // byte-identical. The image path (Adreno) is unaffected: v8c_buffer_path() is
  // false there, so the view is built.
  if (!v8c_buffer_path() && !e.huge_n) {
    try {
      e.weight_image = e.backing->imageView(ws);
      if (std::getenv("NNTR_V8C_IMG_TRACE"))
        std::fprintf(stderr, "[v8cimg] OK   N=%u K=%u wpitch=%u(=K/2)\n", N, K,
                     K / 2),
          std::fflush(stderr);
    } catch (...) {
      // Creation can still fail for a reason the cap query did not predict
      // (pitch alignment). The row-major weight_buf + scale_buf remain valid,
      // so fall back to the imageless (GEMV) route rather than the CPU path.
      e.weight_image = nullptr;
      e.huge_n = true;
      if (std::getenv("NNTR_V8C_IMG_TRACE"))
        std::fprintf(stderr, "[v8cimg] FAIL N=%u K=%u wpitch=%u(=K/2)\n", N, K,
                     K / 2),
          std::fflush(stderr);
      static int logged = 0;
      if (!logged++)
        std::fprintf(stderr,
                     "[v8c] image view unavailable for N=%u K=%u (>image cap); "
                     "keeping buffer path for the GEMV\n",
                     N, K);
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
      // Keep the first entry; release our duplicate device objects (the
      // weight backing frees itself via e.backing's unique_ptr).
      if (e.scale_buf)
        clReleaseMemObject(e.scale_buf);
      if (e.row_sum_w_int4)
        clReleaseMemObject(e.row_sum_w_int4);
      if (e.weight_image)
        clReleaseMemObject(e.weight_image);
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
  // dropped (allocation failures, kernel-registration failures, and the huge_n
  // untied-lm_head branch, which returns false unconditionally on an
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
// permute + upload (~4.1ms x 182 weights = 753ms, measured NNTR_FC_TPROF)
// out of the first timed prefill: the FC layer calls this right after its
// weight is read at model load. No-op (false) off the v8c path. Must live
// OUTSIDE the anonymous namespace (public symbol; the helpers it calls are
// TU-internal).
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
// copy_h2h copies fp16->fp16.
static const std::string v8c_out_residency_kernels = R"CL(
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
)CL";

// Pre-build the residency-kernel program at context init (see header).
void v8c_prewarm_programs(ClContext &cc) {
  cc.registerClKernel(v8c_out_residency_kernels, "v8c_copy_h2h");
}

// Write the fp16 GEMM result (y_fp16, device cl_mem, n = M*N valid elements)
// directly into the GPU-resident SVM output, converting to fp32 when needed.
// Coarse-grained SVM coherence: unmap the output before the kernel (GPU owns
// it), re-map after (host / next layer can read it).
static void v8c_write_output_resident(cl_mem y_fp16, Tensor &output,
                                      unsigned int n, bool out_fp16,
                                      void *out_clmem = nullptr) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto kp = cc->registerClKernel(v8c_out_residency_kernels,
                                 out_fp16 ? "v8c_copy_h2h" : "v8c_cvt_h2f");
  if (!kp)
    return;
  // Static GPU_CLMEM residency: write the planner sub-buffer via THIS KERNEL
  // (cl_mem arg) instead of clEnqueueCopyBuffer -- the blit/copy engine is
  // not reliably ordered against compute kernels on this driver without a
  // drain (measured: a drained readback sees correct bytes, an undrained
  // kernel consumer sees stale), while kernel->kernel ordering is solid
  // (gpu_native's model). No SVM maps, no device_valid bits on this path.
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
  // NNTR_DEVRES Step 1: clear the device-residency bit before the GPU rewrites
  // this output (the prior contents are about to be overwritten). Set it after
  // the write below. Gated by the master flag; off => bit untouched (byte-id).
  static const bool devres = std::getenv("NNTR_DEVRES") != nullptr;
  if (devres) {
    if (auto md = output.getMemoryData())
      md->setDeviceValid(false);
  }
  cc->command_queue_inst_.enqueueSVMUnmap(out_svm);
  if (!kp->SetKernelArguments(0, &y_fp16, sizeof(cl_mem)) ||
      !kp->SetKernelSVMArguments(1, out_svm) ||
      !kp->SetKernelArguments(2, &ni, sizeof(int)))
    return;
  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
  size_t out_bytes = (size_t)n * (out_fp16 ? sizeof(uint16_t) : sizeof(float));
  // async map: the FC output is always consumed by the next GPU op (attention,
  // geglu, next FC) — never read on the host — so the in-order queue orders
  // this map before the next op's unmap and the host need not block here.
  // Removes ~182 per-forward queue drains (the dominant FC sync band).
  // NNTR_FC_SVM_SYNC=1 (Xe3 coherence regression probe): make the FC-output SVM
  // map BLOCKING so the GPU writes are guaranteed visible to the next consumer
  // before it reads (the suspected coarse-grained-SVM stale-shadow on
  // NEO 26.22).
  static const bool fc_svm_sync = std::getenv("NNTR_FC_SVM_SYNC") != nullptr;
  cc->command_queue_inst_.enqueueSVMMap(out_svm, out_bytes, true,
                                        /** async */ !fc_svm_sync);
  // NNTR_DEVRES Step 1: the GPU now holds the fresh FC output in out_svm. Flag
  // it device-resident so a downstream GPU consumer sharing this MemoryData
  // (edge view) sees a HIT. No map is skipped yet (Step 4+); this only sets the
  // bit. Cleared again on the next producer write (above) or a host read (S7).
  if (devres) {
    if (auto md = output.getMemoryData())
      md->setDeviceValid(true, out_svm);
  }
}

// Copy an SVM-resident activation (n = M*K elements) into the device cl_mem
// quant scratch on the GPU -- replaces the host upload (clEnqueueWriteBuffer)
// when the input is GPU-resident, so no PCIe round-trip. Downstream (quantize
// -> image2d -> GEMM) is unchanged; only the source of sc.act_in changes.
// Coarse-grained SVM coherence: unmap the input before the copy (GPU owns it),
// re-map after.
static void v8c_copy_svm_to_clmem(const void *in_svm, cl_mem out,
                                  unsigned int n, bool fp16,
                                  bool device_owned = false) {
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto kp = cc->registerClKernel(v8c_out_residency_kernels,
                                 fp16 ? "v8c_copy_h2h" : "v8c_copy_f2f");
  if (!kp)
    return;
  int ni = (int)n;
  // NNTR_DEVRES Step 4: when device_owned, the producer (e.g. geglu) already
  // left in_svm GPU-owned (skipped its trailing map), so skip the matching
  // unmap here — removing the map/unmap PAIR together. A one-sided skip would
  // read a host-mapped buffer on the GPU = asymmetric SVM state = crash.
  if (!device_owned)
    cc->command_queue_inst_.enqueueSVMUnmap(const_cast<void *>(in_svm));
  if (!kp->SetKernelSVMArguments(0, const_cast<void *>(in_svm)) ||
      !kp->SetKernelArguments(1, &out, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &ni, sizeof(int)))
    return;
  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  cc->command_queue_inst_.DispatchCommand(kp, gws, lws);
  // async map: GPU→GPU handoff (the input copy feeds the quant/GEMM kernels);
  // no host access before then, in-order queue preserves ordering. Skipped on
  // the device-owned path (the buffer stays GPU-owned for the resident edge).
  if (!device_owned)
    cc->command_queue_inst_.enqueueSVMMap(const_cast<void *>(in_svm),
                                          (size_t)n * (fp16 ? 2 : 4), true,
                                          /** async */ true);
}

// Explicit host->cl_mem RAISE for a boundary tensor (see header).
bool clmem_raise_cl(const Tensor &t, unsigned int valid_bytes) {
  if (!t.isClMem())
    return false;
  void *sub = t.getClMem();
  if (sub == nullptr)
    return false;
  // The sub-buffer covers the WHOLE tensor; a nonzero-offset view cannot be
  // bridged from base. Live path is offset-0; fail loudly, never misread.
  if (t.getOffset() != 0)
    throw std::runtime_error("clmem_raise_cl: nonzero-offset view unsupported");
  const size_t bytes = valid_bytes ? (size_t)valid_bytes : t.bytes();
  if (bytes == 0)
    return false;
  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_raise_cl: no GPU context");
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  // Non-blocking upload from the host-written SVM shadow: the in-order queue
  // orders it before every later consumer; the source memory stays untouched
  // until the next forward (host writes only after the lm_head drain).
  if (clEnqueueWriteBuffer(q, static_cast<cl_mem>(sub), CL_FALSE, 0, bytes,
                           t.getData<uint8_t>(), 0, nullptr,
                           nullptr) != CL_SUCCESS)
    throw std::runtime_error("clmem_raise_cl: clEnqueueWriteBuffer failed");
  // NNTR_RAISE_VERIFY=1 (Xe3): confirm the SVM->cl_mem upload landed (the
  // cl_mem the next consumer reads == the SVM source the attention wrote).
  if (std::getenv("NNTR_RAISE_VERIFY")) {
    clFinish(q);
    const size_t cnt = std::min(bytes, (size_t)4096) / 2;
    std::vector<uint16_t> back(cnt);
    clEnqueueReadBuffer(q, static_cast<cl_mem>(sub), CL_TRUE, 0, cnt * 2,
                        back.data(), 0, nullptr, nullptr);
    const uint16_t *svmsrc =
      reinterpret_cast<const uint16_t *>(t.getData<uint8_t>());
    float maxd = 0;
    for (size_t i = 0; i < cnt; ++i)
      maxd = std::max(maxd, std::fabs(v8c_h2f(back[i]) - v8c_h2f(svmsrc[i])));
    std::fprintf(stderr,
                 "[RAISEVERIFY] %-26s cl_mem vs SVM maxdiff=%.4f bytes=%zu\n",
                 t.getName().c_str(), maxd, bytes);
    std::fflush(stderr);
  }
  return true;
}

// Explicit cl_mem->host LOWER for a boundary tensor (see header).
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
  cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
  // BLOCKING read on the in-order queue: waits for every prior command (the
  // whole forward), then lands the bytes in host memory (the SVM shadow used
  // as a plain host pointer). The host consumer reads ordinary memory next.
  cl_int rb_err =
    clEnqueueReadBuffer(q, static_cast<cl_mem>(sub), CL_TRUE, 0, bytes,
                        t.getData<uint8_t>(), 0, nullptr, nullptr);
  if (rb_err != CL_SUCCESS)
    throw std::runtime_error("clmem_lower_cl: clEnqueueReadBuffer failed err=" +
                             std::to_string(rb_err) + " bytes=" +
                             std::to_string(bytes) + " name=" + t.getName());
  return true;
}

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

  // The cl_mem handle covers the WHOLE tensor; a step/batch view at a nonzero
  // offset cannot bind it (kernels address from base). Live path is batch==1 /
  // offset 0 -- fail loudly rather than silently corrupting via a base bind.
  if ((dst_cl != nullptr && dst.getOffset() != 0) ||
      (src_cl != nullptr && src.getOffset() != 0))
    throw std::runtime_error(
      "clmem_residual_op_cl: GPU_CLMEM tensor accessed at a nonzero offset "
      "(batch>1 step views are unsupported on the cl_mem plane)");

  auto *cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!cc)
    throw std::runtime_error("clmem_residual_op_cl: no GPU context");
  const unsigned int n = (unsigned int)dst.size();
  const size_t bytes = (size_t)n * sizeof(uint16_t);

  // Pure cl_mem->cl_mem copy: a plain buffer copy beats a kernel dispatch.
  if (!accumulate && dst_cl != nullptr && src_cl != nullptr) {
    cl_command_queue q = cc->command_queue_inst_.GetCommandQueue();
    if (clEnqueueCopyBuffer(q, static_cast<cl_mem>(src_cl),
                            static_cast<cl_mem>(dst_cl), 0, 0, bytes, 0,
                            nullptr, nullptr) != CL_SUCCESS)
      throw std::runtime_error("clmem_residual_op_cl: clEnqueueCopyBuffer");
    return true;
  }

  auto kp = cc->registerClKernel(v8c_out_residency_kernels,
                                 accumulate ? "v8c_add_h2h" : "v8c_copy_h2h");
  if (!kp)
    throw std::runtime_error("clmem_residual_op_cl: kernel registration");

  // SVM-side args keep the established per-op map protocol (unmap before the
  // kernel, async map after); cl_mem args need none.
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

  // NNTR_RESID_VERIFY=1 (Xe3): snapshot src and dst before the op so we can
  // confirm the result == (accumulate ? src+dst : src) from the SAME buffers
  // the kernel reads. A large diff => the residual add/copy itself is wrong;
  // correct here but garbage output => an UPSTREAM op fed it a stale buffer.
  const bool resid_verify = std::getenv("NNTR_RESID_VERIFY") != nullptr;
  std::vector<uint16_t> rv_s, rv_d0;
  if (resid_verify) {
    cl_command_queue qq = cc->command_queue_inst_.GetCommandQueue();
    clFinish(qq);
    const size_t cnt = std::min((size_t)n, (size_t)2048);
    rv_s.resize(cnt);
    rv_d0.resize(cnt);
    if (src_cl)
      clEnqueueReadBuffer(qq, static_cast<cl_mem>(src_cl), CL_TRUE, 0, cnt * 2,
                          rv_s.data(), 0, nullptr, nullptr);
    else
      std::memcpy(rv_s.data(), src_svm, cnt * 2);
    if (dst_cl)
      clEnqueueReadBuffer(qq, static_cast<cl_mem>(dst_cl), CL_TRUE, 0, cnt * 2,
                          rv_d0.data(), 0, nullptr, nullptr);
    else
      std::memcpy(rv_d0.data(), dst_svm, cnt * 2);
  }

  const int gws[3] = {(int)(((size_t)n + 63) / 64 * 64), 1, 1};
  const int lws[3] = {64, 1, 1};
  if (!cc->command_queue_inst_.DispatchCommand(kp, gws, lws))
    throw std::runtime_error("clmem_residual_op_cl: dispatch");
  if (resid_verify) {
    cl_command_queue qq = cc->command_queue_inst_.GetCommandQueue();
    clFinish(qq);
    const size_t cnt = rv_s.size();
    std::vector<uint16_t> rv_d1(cnt);
    if (dst_cl)
      clEnqueueReadBuffer(qq, static_cast<cl_mem>(dst_cl), CL_TRUE, 0, cnt * 2,
                          rv_d1.data(), 0, nullptr, nullptr);
    else
      std::memcpy(rv_d1.data(), dst_svm, cnt * 2);
    float maxd = 0;
    for (size_t i = 0; i < cnt; ++i) {
      float exp =
        accumulate ? (v8c_h2f(rv_s[i]) + v8c_h2f(rv_d0[i])) : v8c_h2f(rv_s[i]);
      maxd = std::max(maxd, std::fabs(v8c_h2f(rv_d1[i]) - exp));
    }
    std::fprintf(stderr, "[RESIDVERIFY] %-28s acc=%d maxdiff=%.4f\n",
                 dst.getName().c_str(), (int)accumulate, maxd);
    std::fflush(stderr);
  }

  if (src_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMMap(src_svm, bytes, true,
                                          /** async */ true);
  if (dst_cl == nullptr)
    cc->command_queue_inst_.enqueueSVMMap(dst_svm, bytes, true,
                                          /** async */ true);
  return true;
}

// NNTR_FC_TPROF=1: host wall time of the dotCl_v8c hot path, split at the
// input-staging boundary (decomposes the rmsnorm->v8c_copy_h2h GPU idle).
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
  // throwaway output that we never read back to the caller. Skips the
  // "M not divisible by 4 → CPU fallback" cliff so v8c runs for any prefill
  // length (the 18-token Qwen3 chat-template case in particular).
  constexpr unsigned int V8C_TM = 4;
  // M_pad alignment. The v8c GEMM dispatches gws M-axis = M_pad / V8C_TM; the
  // tuned 4x16 work-group needs gws_y = M_pad/4 to be a multiple of 16, i.e.
  // M_pad a multiple of 64, or select2dLws (cl_tensor_view.cpp) fails its
  // divisibility gate and falls back to a NULL (driver-chosen) work-group.
  // On BOTH paths that fallback is a cliff:
  //   - Intel/buffer (NNTR_V8C_BUF): a non-power-of-2 M-workgroup count maps
  //     poorly to the EU array (M=842 prefill 175 -> 671 TPS at align 64).
  //   - Adreno/image: measured 2026-06-18 on gemma4 (M=999 -> M_pad=1000,
  //     gws_y=250, 250%16=10 != 0 -> NULL LWS). The driver's NULL choice is
  //     near-optimal for some N (gate/up N6144 = 5.5 TFLOP/s) but PATHOLOGICAL
  //     for others (full-Q N4096 = 0.41, per_layer_input N8960 = 0.36 -- 13x
  //     slower, ~28% of prefill). Forcing M_pad%64=0 restores the tuned 4x16
  //     to every FC shape: M=1024 prefill 1527 -> ~2280 TPS (+50%), coherent.
  // Padded rows are computed but never stored (M-valid store guard in
  // v8c_gemm_int8_int4), so output is bit-identical. So align to 64 by default
  // on BOTH paths. Override with NNTR_FC_MPAD_ALIGN (mult of V8C_TM). Only
  // applied for prefill-sized M (M >= align): decode (M=1) must never pad to 64
  // (that would be a 64x FC blow-up) -- guarded by eff_align below.
  static const unsigned int _mpad_align = []() {
    const char *e = std::getenv("NNTR_FC_MPAD_ALIGN");
    unsigned int v = e ? (unsigned int)std::atoi(e) : 64u;
    if (v < V8C_TM)
      v = V8C_TM;
    v = (v + V8C_TM - 1) / V8C_TM * V8C_TM; // keep a multiple of V8C_TM
    return v;
  }();
  const unsigned int eff_align = (M >= _mpad_align) ? _mpad_align : V8C_TM;
  const unsigned int M_pad = (M + eff_align - 1) / eff_align * eff_align;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  V8cWeightEntry *w = v8c_get_or_build_weight(weight, K, N);
  if (!w)
    return false;

  // Imageless v8c weight (N > image2d height cap, e.g. the untied int4 lm_head
  // with N=vocab=262144): the image GEMM path cannot run, so dispatch the
  // dedicated fp-act int4 GEMV over the row-major weight buffer (best argmax
  // fidelity; no int8 act quant). Only decode (M=1) is supported -- the lm_head
  // FC runs only on the last position and prefill is skipped; any larger M with
  // no image legitimately falls back to the host path.
  // Keyed on huge_n, NOT (weight_image == nullptr): the buffer path leaves the
  // image null for EVERY weight (it is dead there), and those must still take
  // the normal buffer GEMM below.
  if (w->huge_n) {
#ifdef ENABLE_FP16
    if (M == 1 && input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        (output.getDataType() == ml::train::TensorDim::DataType::FP16 ||
         output.getDataType() == ml::train::TensorDim::DataType::FP32)) {
      void *act = input.isClMem() ? input.getClMem()
                                  : static_cast<void *>(input.getData<_FP16>());
      const bool act_clmem = input.isClMem();
      const bool out_fp16 =
        output.getDataType() == ml::train::TensorDim::DataType::FP16;
      void *logits_host = out_fp16
                            ? static_cast<void *>(output.getData<_FP16>())
                            : static_cast<void *>(output.getData<float>());
      if (lmhead_int4_v8c_gemv_cl(w->weight_buf, w->scale_buf, act, act_clmem,
                                  logits_host, out_fp16, N, K))
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
  // Shared staging + output buffers (slot-independent). The per-fanout act_i8
  // / scale / zp / rs slot buffers are grown AFTER the slot is selected
  // (below), so only the used slot grows to this call's K -- a slot that only
  // ever serves qkv (K=hidden) never pays for the ffn-down K (the larger one).
  //
  // act_in / y_fp16 are ensured AT THEIR USE SITES, not here: under the
  // default direct paths (quant_direct_clmem stages nothing into act_in;
  // direct_out stores the GEMM straight into the planner cl_mem) both stay
  // untouched, while an unconditional grow-only ensure here would commit the
  // largest (M_pad, K/N) shape once per process (hundreds of MB of resident
  // all-zero buffers). The non-direct paths still get them on first touch.

  // Step 2b.0 shared-quant cache check (paper §3.6 fused-quant insight,
  // host-side). If this dotCl_v8c was called with the same (input ptr,
  // M, K, M_pad, dtype) as the most recent call, sc.act_i8/scale/zp/rs are
  // already correctly populated — skip both the host→device write AND the
  // quant kernel. Hits fire on the wq→wk and wq→wv legs of Qwen3 QKV (and
  // gate→up of the MLP block), where the input activation is literally the
  // same tensor across multiple FC dispatches.
  //
  // Segment A.1 residency input mode (paper §3.2 cross-layer GPU residency).
  // When NNTR_RESIDENT_FC=1 AND the input Tensor has a TensorBacking with
  // FP16 encoding holding the activation in cl_mem (set by a preceding GPU
  // op such as Segment A's RMSNorm), bypass the host→device upload entirely
  // — clEnqueueCopyBuffer (GPU→GPU) from the backing into sc.act_in instead.
  // Cache key uses the backing's cl_mem as the source identifier so the
  // wq→wk→wv repeat continues to skip redundant quant. Caller-set Tensor
  // host data is ignored in this mode.
  //
  // Tensor::getBacking() returns null when the producer set the backing on
  // a different Tensor instance than the one this consumer received (the
  // typical nntrainer pattern: layer N's "output" Tensor and layer N+1's
  // "input" Tensor are distinct instances that share the underlying data
  // buffer via TensorPool). Fall back to a pool lookup keyed by the host
  // data pointer; shared-data tensors share the data pointer even when
  // their names/instances differ.
  static const bool resident_fc_enabled =
    std::getenv("NNTR_RESIDENT_FC") != nullptr;
  const tv::TensorBacking *in_backing = nullptr;
  std::shared_ptr<tv::TensorBacking> in_backing_from_pool;
  if (resident_fc_enabled) {
    in_backing = input.getBacking();
    if (!in_backing) {
      const void *in_data_ptr = input.getData<uint8_t>();
      char key_buf[64];
      std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
      in_backing_from_pool =
        tv::TensorBackingPool::Global().get(std::string(key_buf));
      if (!in_backing_from_pool)
        in_backing_from_pool =
          tv::TensorBackingPool::Global().get(input.getName());
      if (in_backing_from_pool)
        in_backing = in_backing_from_pool.get();
    }
  }
  // [resident-act Step 1] cl_mem residency overlay: resolve this FC's input
  // through the graph edge map to its producer's output name, and consume the
  // cl_mem TensorBacking the producer published under `resact:`+producer-name.
  // This is the robust key (the producer→consumer edge), independent of the
  // brittle ptr:%p aliasing. Gated by NNTR_RESIDENT_ACT; misses fall through to
  // the existing SVM upload path (token-identical).
  static const bool resident_act_enabled =
    std::getenv("NNTR_RESIDENT_ACT") != nullptr;
  if (resident_act_enabled && !in_backing) {
    const std::string src = nntrainer::resolveResidentEdge(input.getName());
    if (!src.empty()) {
      in_backing_from_pool =
        tv::TensorBackingPool::Global().get("resact:" + src);
      if (in_backing_from_pool)
        in_backing = in_backing_from_pool.get();
    }
  }
  const bool resident_dtype_match =
    in_backing != nullptr &&
    ((in_backing->encoding() == tv::Encoding::FP16 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP16) ||
     (in_backing->encoding() == tv::Encoding::FP32 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP32));
  const bool use_resident_input = resident_dtype_match;
  // Planner-decided STATIC residency: this input tensor's ResidencyClass is
  // GPU_CLMEM, so by construction its producer (rms norm / geglu) wrote the
  // planner cl_mem sub-buffer (MemoryData.device_mem) -- uniformly, every
  // forward, no runtime device_valid flip. The FC reads it device-direct: a
  // cl_mem->cl_mem GPU copy into sc.act_in, NO SVM map (the measured prefill
  // blocker). The ptr-keyed quant cache is disabled on this edge (the
  // sub-buffer handle recurs across tokens with different contents).
  // Require the in-order SVM-pool queue (NNTR_GPU_SVM_POOL): this copy is
  // ordered after the producer's cl_mem write ONLY on the in-order queue (the
  // default is out-of-order and the copy could race ahead -> garbage).
  static const bool clmem_pool =
    nntr_env_on("NNTR_GPU_CLMEM_POOL") && svm_pool_default_on();
  const bool device_clmem_in = clmem_pool && input.getMemoryData() &&
                               input.getMemoryData()->isClMem() &&
                               input.getMemoryData()->deviceMem() != nullptr;
  cl_mem clmem_in = device_clmem_in
                      ? static_cast<cl_mem>(input.getMemoryData()->deviceMem())
                      : nullptr;
  // [FC->FC chained edge] If this FC's input MemoryData IS the previous v8c
  // FC's output MemoryData and the shapes chain (K == producer N, same M),
  // bind the producer's ACTUAL store target instead of the re-derived handle.
  // ClBufferPool's per-padded-offset cl_mem dedup can resolve the two
  // derivations to different buffers, making the consumer's act-quant read
  // zero-initialized pool bytes the producer never wrote (gate_down->gate_up
  // / ffn_gate_up->ffn_gate_down; fluent-but-wrong text). No-op when the
  // derivations already agree; structurally never fires for models without
  // FC->FC edges. All-GPU, no drains.
  if (device_clmem_in && input.getMemoryData() &&
      sc.last_fc_out_md ==
        static_cast<const void *>(input.getMemoryData().get()) &&
      sc.last_fc_out_clmem != nullptr && sc.last_fc_out_M == M &&
      sc.last_fc_out_N == K) {
    static const bool chain_trace =
      std::getenv("NNTR_V8C_CHAIN_TRACE") != nullptr;
    if (chain_trace && clmem_in != sc.last_fc_out_clmem)
      std::fprintf(stderr,
                   "[V8C-CHAIN] %s: derived in=%p != producer target=%p "
                   "(M=%u K=%u) REBOUND\n",
                   output.getName().c_str(), (void *)clmem_in,
                   (void *)sc.last_fc_out_clmem, M, K);
    clmem_in = sc.last_fc_out_clmem; // bind the producer's ACTUAL store target
  }
  // Input GPU-residency: when the activation lives in the SVM pool (and no
  // cl_mem backing was found), copy it into the quant scratch device-side
  // instead of uploading it from the host -- removing the input round-trip.
  const bool in_svm = !use_resident_input && !device_clmem_in &&
                      input.getMemoryData() && input.getMemoryData()->isSVM();
  // NNTR_DEVRES Step 4: the SVM input is GPU-owned (its producer skipped the
  // trailing map) iff its MemoryData is device_valid. On that resident edge the
  // FC skips its matching unmap/map (v8c_copy_svm_to_clmem) AND must force a
  // re-quant — the SVM pointer recurs every token, so the (ptr,M,K) quant cache
  // would false-hit on stale int8 from a prior token (G5).
  static const bool devres_fc = std::getenv("NNTR_DEVRES") != nullptr;
  const bool device_in =
    devres_fc && in_svm && input.getMemoryData()->isDeviceValid();

  const int cur_dtype =
    (input.getDataType() == ml::train::TensorDim::DataType::FP16) ? 1 : 0;
  const void *cur_in_ptr =
    device_clmem_in      ? static_cast<const void *>(clmem_in)
    : use_resident_input ? static_cast<const void *>(in_backing->buffer())
                         : static_cast<const void *>(input.getData<uint8_t>());
  // Step 2b.0 quant cache. For host-uploaded inputs the (data_ptr,
  // shape, dtype) tuple uniquely identifies the activation, so a hit
  // means the same data is already int8-quantized in sc.act_i8 and
  // we can skip both the copy and the quant kernel.
  //
  // For backing-sourced inputs the same logic holds *within a single
  // forward pass* because the backing pointer is stable. Across passes
  // the same backing pointer points to different data (RMSNorm
  // overwrites it). That cross-pass staleness is invalidated below
  // by an external generation counter tied to RMSNorm writes:
  // resident_input_quant_generation is bumped whenever a Segment A
  // RMSNorm producer writes to a backing, and cached against the
  // generation at the time of the last quant. If the generation has
  // advanced since the last cache update, the cache is invalidated.
  const bool quant_cache_hit =
    !device_clmem_in && !device_in && sc.last_quant_in_ptr != nullptr &&
    sc.last_quant_in_ptr == cur_in_ptr && sc.last_quant_M == M &&
    sc.last_quant_K == K && sc.last_quant_M_pad == M_pad &&
    sc.last_quant_dtype == cur_dtype &&
    (!use_resident_input ||
     sc.last_quant_resident_generation == g_resident_quant_generation);

  const bool skip_upload_and_quant = quant_cache_hit;
  // Per-fanout GEMM input buffers, filled by the slot selected just below.
  cl_mem act_i8_arg = nullptr;
  cl_mem act_scale_arg = nullptr;
  cl_mem act_zp_arg = nullptr;
  cl_mem act_rs_arg = nullptr;

  // NNTR_FC_QUANT_DIRECT: on the cl_mem residency edge, quantize the
  // producer's (rmsnorm) cl_mem output IN PLACE, skipping the cl_mem->sc.act_in
  // staging copy (the v8c_copy_h2h kernel) and the padded-row zero write. The
  // act-quant kernel reads exactly M real rows from clmem_in (gws=M*64, bounded
  // by `if (row>=M) return`), so there is no OOB on the M-row producer buffer.
  // Safe because (a) GEMM output rows are independent -- acc[i] depends only on
  // act row i (int8_int4_gemm_v8c.cl) -- so the now-unquantized padded rows
  // [M, M_pad) of act_i8/scale/zp/rs (stale, not zeroed) only corrupt padded
  // OUTPUT rows, and (b) v8c_write_output_resident copies just M*N valid
  // elements, discarding those padded rows. In-order SVM-pool queue keeps the
  // quant ordered after the rmsnorm write of the same cl_mem. Removes one
  // dispatch + its host-bound inter-kernel idle per FC input (decode is
  // dispatch-bound: clprof rmsnorm->v8c_copy_h2h was 37% of GPU idle).
  // Gated, default off => byte-identical baseline.
  //
  // MEASURED 2026-06-15 (decode clprof): DECODE-NEUTRAL (the M=1 copy is tiny).
  // BUT at PREFILL it is a real win: the skipped cl_mem->sc.act_in staging copy
  // is M*K per FC -- at M=1024 that is ~850MB of GPU->GPU CopyBuffer across the
  // 182 prefill FCs. Skipping it (act-quant reads clmem_in directly) measured
  // (Adreno 840, gemma2_lg QINT4, M=1024, NNTR_GPU_CLMEM_POOL): prefill
  // 859 -> 901 TPS (+5%, crossing 900 = gpu_native parity), token-IDENTICAL
  // (md5 a6710b4d unchanged). So default ON; NNTR_FC_QUANT_DIRECT=0 restores
  // the staging copy. Only engages on the cl_mem-input edge (device_clmem_in);
  // the SVM path is unaffected (no-op).
  static const bool fc_quant_direct = []() {
    const char *e = std::getenv("NNTR_FC_QUANT_DIRECT");
    return !e || e[0] != '0';
  }();
  const bool quant_direct_clmem = fc_quant_direct && device_clmem_in &&
                                  !skip_upload_and_quant && clmem_in != nullptr;

  // RACE#1 fix: select this call's per-fanout activation slot. On a quant-cache
  // HIT (wk/wv after wq -- same input) reuse the slot that already holds the
  // int8 (read-only, like gpu_native "quantize ONCE"); on a MISS (a new fanout
  // or a cl_mem-input edge with the cache disabled) advance the ring so this
  // fanout's quant WRITE lands in a buffer distinct from the prior fanout's
  // still-in-flight GEMM image READ. The fused-rmsq consumer (off by default)
  // binds its own external buffers and uses no slot.
  int act_slot = -1;
  {
    act_slot = quant_cache_hit
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
    act_i8_arg = sc.act_i8[act_slot];
    act_scale_arg = sc.act_scale[act_slot];
    act_zp_arg = sc.act_zp[act_slot];
    act_rs_arg = sc.act_rs[act_slot];
  }

  // Submit the accumulated batch BEFORE this FC re-quants (DEFAULT ON,
  // NNTR_FC_FLUSH=0 disables): the quant's act-image WRITE and every GEMM
  // image READ of that slot (this FC + the cache-hit siblings wk/wv/up that
  // reuse it) then share ONE submission. The end-of-FC flush (mode 1)
  // corrupted outputs because it split the cache-hit siblings' image reads
  // from the producer's write across submissions (the image-from-buffer
  // texture-L1 staleness); flushing only at re-quant boundaries keeps every
  // image write->read pair batch-local. Validated token-identical (20/20
  // cross-build + 10/10 staging suite) at +6% TPS (547 -> 580 hot).
  // Default ON only on the Adreno image path: under NNTR_V8C_BUF (Intel
  // buffer path) the flush measurably ALTERS outputs there too, and the
  // deferred-submission stall it fixes is an Adreno driver behavior -- keep
  // Intel byte-identical to the pre-flush baseline (same gating precedent as
  // the program prewarm).
  static const int fc_flush_mode = []() {
    const char *e = std::getenv("NNTR_FC_FLUSH");
    if (e)
      return std::atoi(e);
    // 2026-06-12 re-baseline: default mode 1 (trailing flush after every
    // FC) -- the submit-split output perturbation it caused is a race
    // pattern, not a math change (drained-capture proof), and the
    // re-baselined reference outputs absorb it. +15%-class idle recovery.
    return v8c_use_buffer_path() ? 0 : 1; // buffer path ⇒ mode 0
  }();
  if (fc_flush_mode == 2 && !skip_upload_and_quant)
    clFlush(q);

  // act_in is consumed only by the staging copies below;
  // Lever-1 (quant_direct_clmem) and act-cache hits never touch it.
  if (!skip_upload_and_quant && !quant_direct_clmem &&
      !v8c_ensure_buf(ctx, &sc.act_in, &sc.act_in_bytes,
                      (size_t)M_pad * K * act_elem, CL_MEM_READ_ONLY))
    return false;

  if (!skip_upload_and_quant) {
    if (device_clmem_in) {
      // NNTR_GPU_CLMEM_POOL: GPU->GPU copy of the producer's cl_mem sub-buffer
      // (the normed activation, written device-direct by the converted rmsnorm)
      // into sc.act_in. No SVM map/unmap -- the in-order SVM-pool queue orders
      // this copy after the rmsnorm coop write of the same cl_mem.
      // quant_direct_clmem skips this copy: the act-quant below reads
      // clmem_in directly.
      if (!quant_direct_clmem &&
          clEnqueueCopyBuffer(q, clmem_in, sc.act_in, 0, 0,
                              (size_t)M * K * act_elem, 0, nullptr,
                              nullptr) != CL_SUCCESS)
        return false;
    } else if (use_resident_input) {
      // GPU→GPU copy of the resident FP32/FP16 activation into sc.act_in.
      // Same shape as a host upload would produce, just without crossing
      // PCIe. Padded rows (M_pad > M) are zero-filled below.
      if (clEnqueueCopyBuffer(q, in_backing->buffer(), sc.act_in, 0, 0,
                              (size_t)M * K * act_elem, 0, nullptr,
                              nullptr) != CL_SUCCESS)
        return false;
    } else if (in_svm) {
      // GPU copy of the SVM-resident activation into sc.act_in -- no host
      // upload. Downstream quant/image/GEMM see the same sc.act_in as before.
      v8c_copy_svm_to_clmem(cur_in_ptr, sc.act_in,
                            (unsigned int)((size_t)M * K), cur_dtype == 1,
                            /** device_owned */ device_in);
    } else {
      if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE, 0,
                               (size_t)M * K * act_elem, cur_in_ptr, 0, nullptr,
                               nullptr) != CL_SUCCESS)
        return false;
    }
    if (M_pad > M && !quant_direct_clmem) {
      const size_t pad_bytes = (size_t)(M_pad - M) * K * act_elem;
      std::vector<uint8_t> zeros(pad_bytes, 0);
      if (clEnqueueWriteBuffer(q, sc.act_in, CL_FALSE, (size_t)M * K * act_elem,
                               pad_bytes, zeros.data(), 0, nullptr,
                               nullptr) != CL_SUCCESS)
        return false;
    }
  }

  try {
    if (!skip_upload_and_quant) {
      // Quant the producer cl_mem (M real rows) directly when
      // quant_direct_clmem; otherwise the staged sc.act_in (M_pad rows incl.
      // the zero pad). Padded act_i8/scale/zp/rs rows [M, M_pad) are left stale
      // in the direct case -- they feed only discarded padded GEMM output rows.
      cl_mem quant_src = quant_direct_clmem ? clmem_in : sc.act_in;
      const unsigned int quant_rows = quant_direct_clmem ? M : M_pad;
      if (input.getDataType() == ml::train::TensorDim::DataType::FP16)
        quantize_act_v8c_fp16_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      else
        quantize_act_v8c_fp32_cl(quant_src, act_i8_arg, act_scale_arg,
                                 act_zp_arg, act_rs_arg, quant_rows, K);
      // Update cache key only after a successful quant.
      sc.last_quant_in_ptr = cur_in_ptr;
      sc.last_quant_M = M;
      sc.last_quant_K = K;
      sc.last_quant_M_pad = M_pad;
      sc.last_quant_dtype = cur_dtype;
      sc.last_quant_resident_generation = g_resident_quant_generation.load();
      // Record WHICH slot now holds this input's int8 so a subsequent cache
      // hit (wk/wv) reads the right per-fanout buffer, not whatever the ring
      // last pointed at.
      sc.last_quant_slot = act_slot;
    }

    // v8c GEMM input binding. The buffer path (NNTR_V8C_BUF, Intel NEO) selects
    // the *_buf kernels whose args are __global uint4* — they MUST be bound to
    // raw cl_mem buffers (the int8 act scratch and the weight backing buffer),
    // NOT image2d objects. Only the Adreno image-sampling path builds an
    // image2d view over the act buffer. (Mirror of gpu_native qwen3_forward.cpp
    // use_v8c_buf ? *_buf : *_image selection — the previous code always passed
    // images, so the buffer kernel read wrong memory and produced garbage.)
    const bool use_buf = v8c_buffer_path();
    cl_mem act_image = nullptr;
    bool act_image_transient = false; // true => owned here, release after GEMM
    if (!use_buf) {
      // Build the image2d view over the int8 act buffer (zero-copy, tensor
      // virtualization). RACE#1 fix: for the per-fanout slot path the view is
      // CACHED on the slot and reused across the fanout's GEMMs, rebuilt only
      // when the slot's buffer is grown or (M_pad, K) change -- removing the
      // per-call clCreateImage/clReleaseMemObject churn AND the old exception-
      // path image leak. The fused-rmsq consumer (no slot, external buffer
      // that varies per input) keeps a transient per-call view.
      cl_image_format afmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc adesc{};
      adesc.image_type = CL_MEM_OBJECT_IMAGE2D;
      adesc.image_width = K / 16;
      adesc.image_height = M_pad;
      adesc.image_row_pitch = K;
      adesc.buffer = act_i8_arg;
      if (act_slot >= 0) {
        if (sc.act_image[act_slot] == nullptr ||
            sc.act_image_buf[act_slot] != act_i8_arg ||
            sc.act_image_M_pad[act_slot] != M_pad ||
            sc.act_image_K[act_slot] != K) {
          if (sc.act_image[act_slot]) {
            clReleaseMemObject(sc.act_image[act_slot]);
            sc.act_image[act_slot] = nullptr;
          }
          cl_mem img =
            clCreateImage(ctx, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
          if (err != CL_SUCCESS)
            throw std::runtime_error("act image view fail");
          sc.act_image[act_slot] = img;
          sc.act_image_buf[act_slot] = act_i8_arg;
          sc.act_image_M_pad[act_slot] = M_pad;
          sc.act_image_K[act_slot] = K;
        }
        act_image = sc.act_image[act_slot];
      } else {
        act_image =
          clCreateImage(ctx, CL_MEM_READ_ONLY, &afmt, &adesc, nullptr, &err);
        if (err != CL_SUCCESS)
          throw std::runtime_error("act image view fail");
        act_image_transient = true;
      }
    }

    // (b) v8c GEMM — run on padded M_pad rows, but only read back the valid
    // M rows to the caller buffer.
    //
    // Direct output (kernel-store, no copy): when the FC output is a
    // GPU_CLMEM-resident FP16 tensor, point the GEMM's Y at its planner
    // sub-buffer with the M_valid store guard, eliminating the separate
    // v8c_copy_h2h writer kernel (46ms GPU + 182 enqueues per 1K prefill).
    // Same kernel->kernel ordering the copy writer relied on. Disabled when
    // a debug consumer needs sc.y_fp16 (probe/dualout/trace) and by
    // NNTR_V8C_DIRECT_OUT=0.
    const bool out_clmem =
      clmem_pool && output.getMemoryData() &&
      output.getMemoryData()->isClMem() &&
      output.getMemoryData()->deviceMem() != nullptr &&
      output.getDataType() == ml::train::TensorDim::DataType::FP16;
    static const bool direct_out_enabled = []() {
      const char *e = std::getenv("NNTR_V8C_DIRECT_OUT");
      return !(e && e[0] == '0');
    }();
    const bool direct_out = direct_out_enabled && out_clmem;
    // y_fp16 only backs the non-direct output paths (SVM/host bounce).
    if (!direct_out && !v8c_ensure_buf(ctx, &sc.y_fp16, &sc.y_fp16_bytes,
                                       sizeof(uint16_t) * (size_t)M_pad * N,
                                       CL_MEM_READ_WRITE))
      return false;
    cl_mem gemm_y_arg =
      direct_out ? static_cast<cl_mem>(output.getMemoryData()->deviceMem())
                 : sc.y_fp16;
    cl_mem gemm_act_arg = use_buf ? act_i8_arg : act_image;
    cl_mem gemm_wgt_arg = use_buf ? w->weight_buf : w->weight_image;
    // M_valid = the REAL row count, always. The kernel routes single-row calls
    // to the fast GEMV (M=1 decode) and multi-row calls to the TM=4 tiled
    // kernel with the M_valid store guard. Passing M_pad here when !direct_out
    // (the old behavior) made real M=1 decode FCs with non-clmem outputs look
    // like 4-row calls -> TM=4 at 4x the GEMV work (decode TPS regression);
    // consumers only ever read the real M rows, so storing exactly M is safe
    // on every output path (writer kernel / SVM / host bounce all copy M*N).
    gemm_int8_v8c_cl(gemm_act_arg, gemm_wgt_arg, act_scale_arg, w->scale_buf,
                     act_rs_arg, act_zp_arg, w->row_sum_w_int4, gemm_y_arg,
                     M_pad, N, K, M);
    // NNTR_XE3_FC_SYNC: narrowed Xe3 coherence fix. The in-order queue does not
    // give kernel->kernel coarse-grained-SVM coherence on NEO 26.22; the global
    // hammer (NNTR_XE3_SYNC, clFinish after EVERY dispatch) fixes it but
    // serializes decode. The bisect showed a clFinish after the FC GEMM alone
    // is sufficient (it is the dominant SVM-producing op and lands between most
    // consumers), so draining only here keeps coherence while restoring decode
    // pipelining. Default-ON for Xe3 (cl_context.cpp setenv on Intel);
    // value-parsed so NNTR_XE3_FC_SYNC=0 disables. See the coherence note at
    // the setenv site.
    static const bool xe3_fc_sync = []() {
      const char *e = std::getenv("NNTR_XE3_FC_SYNC");
      return e && std::atoi(e) != 0;
    }();
    if (xe3_fc_sync)
      clFinish(q);

    // Read output fp16 (only the valid M rows; padded rows are discarded),
    // convert to output dtype.
    // Planner-decided STATIC residency: a GPU_CLMEM output (FP16 by
    // derivation) either was written DIRECTLY by the GEMM store guard
    // (direct_out above -- nothing left to do) or gets the fp16 result
    // written into its planner cl_mem sub-buffer by the kernel writer.
    // (out_clmem was hoisted above the GEMM dispatch for direct_out.)
    const bool out_resident =
      !out_clmem && output.getMemoryData() && output.getMemoryData()->isSVM() &&
      (output.getDataType() == ml::train::TensorDim::DataType::FP32 ||
       output.getDataType() == ml::train::TensorDim::DataType::FP16);
    if (out_clmem) {
      cl_mem out_sub = static_cast<cl_mem>(output.getMemoryData()->deviceMem());
      // KERNEL writer (not clEnqueueCopyBuffer): see the note inside
      // v8c_write_output_resident -- the copy engine is not reliably ordered
      // against the producing GEMM without a drain on this driver.
      // Skipped in direct_out mode (the GEMM already stored into out_sub).
      if (!direct_out)
        v8c_write_output_resident(sc.y_fp16, output,
                                  (unsigned int)((size_t)M * N), true,
                                  static_cast<void *>(out_sub));
    } else if (out_resident) {
      // Residency: write the fp16 GEMM result straight into the SVM output on
      // the GPU, no host readback. fp16 output is a plain copy (no conversion);
      // fp32 output is converted via cvt_h2f.
      v8c_write_output_resident(
        sc.y_fp16, output, (unsigned int)((size_t)M * N),
        output.getDataType() == ml::train::TensorDim::DataType::FP16);
    } else {
      // Host-bounce path: read output fp16 (only the valid M rows; padded rows
      // are discarded), convert to output dtype on the host.
      std::vector<uint16_t> y_host((size_t)M * N);
      if (clEnqueueReadBuffer(q, sc.y_fp16, CL_TRUE, 0,
                              sizeof(uint16_t) * y_host.size(), y_host.data(),
                              0, nullptr, nullptr) != CL_SUCCESS)
        return false;
      if (output.getDataType() == ml::train::TensorDim::DataType::FP32) {
        float *out = output.getData<float>();
        for (size_t i = 0; i < y_host.size(); ++i)
          out[i] = v8c_h2f(y_host[i]);
      } else if (output.getDataType() == ml::train::TensorDim::DataType::FP16) {
        std::memcpy(output.getData<uint8_t>(), y_host.data(),
                    sizeof(uint16_t) * y_host.size());
      } else {
        if (act_image_transient && act_image)
          clReleaseMemObject(act_image);
        throw std::runtime_error("unsupported output dtype");
      }
    }
    // [FC->FC chained edge] Register this FC's output identity + the ACTUAL
    // cl_mem it was stored to (== gemm_y_arg under direct_out, == out_sub
    // under the kernel-writer path) so a directly-chained consumer FC can
    // rebind its act-quant source to the real bytes (see the clmem_in rebind
    // above). Non-clmem outputs clear the record.
    sc.last_fc_out_md =
      out_clmem ? static_cast<const void *>(output.getMemoryData().get())
                : nullptr;
    sc.last_fc_out_clmem =
      out_clmem ? static_cast<cl_mem>(output.getMemoryData()->deviceMem())
                : nullptr;
    sc.last_fc_out_M = M;
    sc.last_fc_out_N = N;

    // Only the transient (fused-rmsq) view is owned here; the per-fanout
    // slot views are cached on V8cScratch and released on rebuild.
    if (act_image_transient && act_image)
      clReleaseMemObject(act_image);

    // FC_FLUSH mode 1 (2026-06-12 re-baseline DEFAULT): submit this FC's
    // enqueue chain now instead of at the next blocking call -- recovers the
    // norm->FC idle band (~+15% TPS class). The output perturbation that
    // kept this opt-in was proven by the drained-capture probe to be a race
    // PATTERN change, not a math change (all intermediate values
    // bit-identical); the re-baselined reference outputs absorb it.
    // NNTR_FC_FLUSH=0 disables all flushing, =2 restores the
    // re-quant-entry batch-local rule.
    if (fc_flush_mode == 1)
      clFlush(q);

    // === Step 1e bridge round-trip (paper §3.2). Attach the cached
    // v8c weight backing to the output tensor as a non-owning tracer.
    // CPU consumers ignore this field. This is purely a bridge
    // integrity hook today; Step 2's fused QKV kernel will replace it
    // with a real output backing pointing at the cl_mem the next
    // GPU layer will consume. NNTR_TENSOR_BRIDGE_TRIP=1 logs the
    // first round-trip on real device traffic to confirm wiring.
    output.setBacking(w->backing.get());
    static int logged_trip = 0;
    if (!logged_trip && std::getenv("NNTR_TENSOR_BRIDGE_TRIP") != nullptr) {
      logged_trip = 1;
      tv::TensorBacking *back = output.getBacking();
      std::fprintf(stderr, "[Step1e] bridge round-trip: set=%p get=%p %s\n",
                   (void *)w->backing.get(), (void *)back,
                   back == w->backing.get() ? "OK" : "MISMATCH");
      std::fflush(stderr);
    }
  } catch (...) {
    return false;
  }
  return true;
}

#ifdef ENABLE_FP16
// =============================================================================
// Segment A.2 — GPU RMSNorm with TensorBacking output residency.
//
// Paper §3.2 cross-layer residency. Produces a cl_mem holding FP16
// normalized activation that downstream FC layers consume directly via
// `dotCl_v8c`'s residency input mode (Segment A.1). No host materialize
// when env-gated.
//
// First wired consumer: Qwen3's `attention_norm` and `ffn_norm` (per
// transformer.cpp:340, 355). q_norm / k_norm use ReshapedRMSNormLayer
// (different class) and are out of Segment A scope.
// =============================================================================
namespace {

// Per-gamma persistent upload cache. Gamma weights don't change at
// inference; upload once per unique gamma name, reuse forever.
/**
 * @brief Persistent per-gamma device upload cache for the resident
 *        RMSNorm path (gamma is inference-constant: upload once, reuse).
 */
struct ResidentRmsState {
  std::unordered_map<std::string, cl_mem> gamma_bufs;
  std::mutex mtx;
};
static ResidentRmsState &resident_rms_state() {
  static ResidentRmsState s;
  return s;
}

static bool resident_rmsnorm_env_enabled() {
  static int cached = -1;
  if (cached < 0)
    cached = std::getenv("NNTR_RESIDENT_RMSNORM") != nullptr ? 1 : 0;
  return cached != 0;
}

} // anonymous namespace

bool rmsnorm_resident_fp16(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int B, unsigned int C,
                           unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output) {
  if (!resident_rmsnorm_env_enabled())
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;
  if (B == 0 || C == 0 || H == 0 || W == 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t total_elems = (size_t)B * C * H * W;
  const size_t total_bytes = total_elems * sizeof(uint16_t);

  // 1) Get or create the output backing in the pool. Reused across calls
  //    with the same output_name (output is overwritten in place).
  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> out_bk = pool.get(output_name);
  if (!out_bk || out_bk->bytes() < total_bytes) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    out_bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP16, tv::Layout::ROW_MAJOR, total_bytes,
      /** owned */ true);
    pool.set(output_name, out_bk);
  }

  // 2) Source cl_mem for the input — from upstream backing if present
  //    (zero host transfer), else upload from host on each call.
  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr; // freed at function exit if allocated
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP16 &&
      in_bk->bytes() >= total_bytes) {
    in_cl = in_bk->buffer();
  } else {
    cl_int err = CL_SUCCESS;
    in_upload_owned =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned)
      return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, total_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }

  // 3) Gamma upload cache — once per gamma name. Gamma doesn't change at
  //    inference. Keyed by gamma's name (stable per layer).
  cl_mem gamma_cl = nullptr;
  {
    auto &st = resident_rms_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    const std::string &gn = gamma.getName();
    auto it = st.gamma_bufs.find(gn);
    if (it == st.gamma_bufs.end()) {
      cl_int err = CL_SUCCESS;
      cl_mem gbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)W * sizeof(uint16_t), nullptr, &err);
      if (err != CL_SUCCESS || !gbuf) {
        if (in_upload_owned)
          clReleaseMemObject(in_upload_owned);
        return false;
      }
      if (clEnqueueWriteBuffer(
            q, gbuf, CL_TRUE, 0, (size_t)W * sizeof(uint16_t),
            gamma.getData<uint8_t>(), 0, nullptr, nullptr) != CL_SUCCESS) {
        clReleaseMemObject(gbuf);
        if (in_upload_owned)
          clReleaseMemObject(in_upload_owned);
        return false;
      }
      st.gamma_bufs[gn] = gbuf;
      gamma_cl = gbuf;
    } else {
      gamma_cl = it->second;
    }
  }

  // 4) Register the kernel (cached by ClContext on repeat registration).
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16");
  if (!kp) {
    if (in_upload_owned)
      clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem out_buf = out_bk->buffer();
  cl_half eps_half;
  {
    const float ef = epsilon;
    // round-half-to-nearest float→half via the helper available in this TU.
    // The activation residual stream is FP16 so this matches the upstream
    // precision exactly.
    uint16_t bits = 0;
    // simple manual conversion for non-NaN positive epsilon (always small).
    union {
      float f;
      uint32_t u;
    } v;
    v.f = ef;
    uint32_t e = (v.u >> 23) & 0xFF;
    uint32_t m = v.u & 0x7FFFFF;
    if (e == 0)
      bits = 0;
    else if (e >= 143)
      bits = 0x7BFF; // clamp to fp16 max
    else if (e <= 112)
      bits = 0; // underflow to 0
    else
      bits = ((e - 112) << 10) | (m >> 13);
    eps_half = bits;
  }

  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &out_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &eps_half, sizeof(cl_half)) ||
      !kp->SetKernelArguments(arg++, &B, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &C, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &H, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &W, sizeof(int))) {
    if (in_upload_owned)
      clReleaseMemObject(in_upload_owned);
    return false;
  }

  // work-groups per existing RMSNormLayerCl: {B*C, H, 1}, local {W, 1, 1}.
  const int wg_count[3] = {(int)(B * C), (int)H, 1};
  const int wg_size[3] = {(int)W, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned)
      clReleaseMemObject(in_upload_owned);
    return false;
  }

  // 5) Publish the backing to the consumer side. Both setBacking() on the
  //    Tensor (if the consumer's Tensor instance is the same one we got
  //    here) AND pool.set() (already done at allocation) so a name-based
  //    lookup is also possible. Host data of `output` is left undefined.
  output.setBacking(out_bk.get());
  // Bump the global resident-quant generation: any downstream FC that
  // sees this backing pointer in its quant cache must re-quant; the
  // backing's data has just changed.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);

  // OUT-OF-ORDER QUEUE FIX: the ClContext queue uses
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (opencl_command_queue_
  // manager.cpp:56). Without explicit events or a barrier, subsequent
  // enqueues are NOT guaranteed to wait for this kernel even when they
  // read the same cl_mem. The pool's ptr-keyed entry is overwritten by
  // every RMSNorm in the chain (TensorPool reuses the same host buffer
  // for all per-layer norm outputs), and multiple FC consumers race
  // against multiple RMSNorm producers. Barrier serializes.
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  // Static one-shot log for first invocation so we can confirm wiring
  // when running a model. NNTR_RESIDENT_RMSNORM_TRIP=1 prints once.
  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_RESIDENT_RMSNORM_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[SegA-RMS] first invocation: out_name=%s B=%u C=%u H=%u "
                 "W=%u in_from_backing=%d\n",
                 output_name.c_str(), B, C, H, W,
                 in_upload_owned == nullptr ? 1 : 0);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    // The kernel reads input as a buffer; we can't release until the
    // kernel has finished. Block once to ensure the upload buffer is
    // safe to release. For backing-supplied inputs we don't allocate.
    clFinish(q);
    clReleaseMemObject(in_upload_owned);
  }

  return true;
}

// FP32 variant (Qwen3's actual residual-stream dtype). Same lifecycle/
// caching/backing semantics as the FP16 variant; just different kernel.
bool rmsnorm_resident_fp32(const Tensor &input, const Tensor &gamma,
                           float epsilon, unsigned int H, unsigned int W,
                           const std::string &output_name, Tensor &output) {
  if (!resident_rmsnorm_env_enabled())
    return false;
  if (input.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      gamma.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  if (H == 0 || W == 0)
    return false;
  // rmsnorm_cl kernel reads input as float4 — W must be a multiple of 4.
  if (W % 4 != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t total_elems = (size_t)H * W;
  const size_t total_bytes = total_elems * sizeof(float);

  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> out_bk = pool.get(output_name);
  if (!out_bk || out_bk->bytes() < total_bytes ||
      out_bk->encoding() != tv::Encoding::FP32) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    out_bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, tv::Encoding::FP32, tv::Layout::ROW_MAJOR, total_bytes,
      /** owned */ true);
    pool.set(output_name, out_bk);
  }
  // Also register under the host-data-pointer key so consumers receiving a
  // different Tensor instance (with the same underlying data pointer) can
  // find this backing. See the dotCl_v8c residency-input lookup code.
  {
    const void *out_data_ptr = output.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", out_data_ptr);
    pool.set(std::string(key_buf), out_bk);
  }

  cl_mem in_cl = nullptr;
  cl_mem in_upload_owned = nullptr;
  std::shared_ptr<tv::TensorBacking> in_bk_pool_strong;
  if (const tv::TensorBacking *in_bk = input.getBacking();
      in_bk != nullptr && in_bk->encoding() == tv::Encoding::FP32 &&
      in_bk->bytes() >= total_bytes) {
    in_cl = in_bk->buffer();
  } else {
    const void *in_data_ptr = input.getData<uint8_t>();
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p", in_data_ptr);
    in_bk_pool_strong =
      tv::TensorBackingPool::Global().get(std::string(key_buf));
    if (in_bk_pool_strong &&
        in_bk_pool_strong->encoding() == tv::Encoding::FP32 &&
        in_bk_pool_strong->bytes() >= total_bytes) {
      in_cl = in_bk_pool_strong->buffer();
    }
  }
  if (in_cl == nullptr) {
    cl_int err = CL_SUCCESS;
    in_upload_owned =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !in_upload_owned)
      return false;
    if (clEnqueueWriteBuffer(q, in_upload_owned, CL_TRUE, 0, total_bytes,
                             input.getData<uint8_t>(), 0, nullptr,
                             nullptr) != CL_SUCCESS) {
      clReleaseMemObject(in_upload_owned);
      return false;
    }
    in_cl = in_upload_owned;
  }

  cl_mem gamma_cl = nullptr;
  {
    auto &st = resident_rms_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    const std::string gn = gamma.getName() + ":fp32";
    auto it = st.gamma_bufs.find(gn);
    if (it == st.gamma_bufs.end()) {
      cl_int err = CL_SUCCESS;
      cl_mem gbuf = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                   (size_t)W * sizeof(float), nullptr, &err);
      if (err != CL_SUCCESS || !gbuf) {
        if (in_upload_owned)
          clReleaseMemObject(in_upload_owned);
        return false;
      }
      if (clEnqueueWriteBuffer(q, gbuf, CL_TRUE, 0, (size_t)W * sizeof(float),
                               gamma.getData<uint8_t>(), 0, nullptr,
                               nullptr) != CL_SUCCESS) {
        clReleaseMemObject(gbuf);
        if (in_upload_owned)
          clReleaseMemObject(in_upload_owned);
        return false;
      }
      st.gamma_bufs[gn] = gbuf;
      gamma_cl = gbuf;
    } else {
      gamma_cl = it->second;
    }
  }

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
  if (!kp) {
    if (in_upload_owned)
      clReleaseMemObject(in_upload_owned);
    return false;
  }

  cl_mem out_buf = out_bk->buffer();
  int arg = 0;
  if (!kp->SetKernelArguments(arg++, &in_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &out_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &gamma_cl, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(arg++, &epsilon, sizeof(float)) ||
      !kp->SetKernelArguments(arg++, &H, sizeof(int)) ||
      !kp->SetKernelArguments(arg++, &W, sizeof(int))) {
    if (in_upload_owned)
      clReleaseMemObject(in_upload_owned);
    return false;
  }

  // rmsnorm_cl uses get_group_id(0) → H groups, subgroup reduce inside.
  // DispatchCommand interprets the first array as the GLOBAL work-item
  // count (NDRange standard), so global = H * subgroup, local = subgroup.
  // Matches rmsnorm_cl_internal at blas_kernels_templates.h:428.
  // Diagnostic NNTR_SEGA_RMS_LOCAL=N overrides the local size. With
  // N=1, the kernel runs single-threaded per row (no subgroup reduce),
  // useful to isolate whether subgroup_reduce_add is the divergence
  // source from CPU NEON.
  int subgroup_size = 64; // Adreno default
  if (const char *e = std::getenv("NNTR_SEGA_RMS_LOCAL"))
    subgroup_size = std::atoi(e);
  const int wg_count[3] = {(int)H * subgroup_size, 1, 1};
  const int wg_size[3] = {subgroup_size, 1, 1};
  if (!blas_cc->command_queue_inst_.DispatchCommand(kp, wg_count, wg_size)) {
    if (in_upload_owned)
      clReleaseMemObject(in_upload_owned);
    return false;
  }

  output.setBacking(out_bk.get());
  // Bump the global resident-quant generation: any downstream FC that
  // sees this backing pointer in its quant cache must re-quant; the
  // backing's data has just changed.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);

  // OUT-OF-ORDER QUEUE FIX: the ClContext queue uses
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE (opencl_command_queue_
  // manager.cpp:56). Without explicit events or a barrier, subsequent
  // enqueues are NOT guaranteed to wait for this kernel even when they
  // read the same cl_mem. The pool's ptr-keyed entry is overwritten by
  // every RMSNorm in the chain (TensorPool reuses the same host buffer
  // for all per-layer norm outputs), and multiple FC consumers race
  // against multiple RMSNorm producers. Barrier serializes.
  clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);

  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_RESIDENT_RMSNORM_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[SegA-RMS-FP32] first invocation: out_name=%s H=%u W=%u "
                 "in_from_backing=%d\n",
                 output_name.c_str(), H, W, in_upload_owned == nullptr ? 1 : 0);
    std::fflush(stderr);
  }

  if (in_upload_owned) {
    clFinish(q);
    clReleaseMemObject(in_upload_owned);
  }

  return true;
}
#endif // ENABLE_FP16

// CPU-norm + GPU-residency-handoff path. Uploads the output Tensor's
// host FP32 data into a TensorBacking under `output_name`. Bit-exact
// w.r.t. CPU computation because no GPU compute occurs here. Caller
// must have already populated output.getData<float>() via the existing
// CPU RMSNorm code.
bool publish_host_fp32_to_backing(const Tensor &output,
                                  const std::string &output_name) {
  if (output.getDataType() != ml::train::TensorDim::DataType::FP32)
    return false;
  const auto &dim = output.getDim();
  const size_t total_elems =
    (size_t)dim.batch() * dim.channel() * dim.height() * dim.width();
  if (total_elems == 0)
    return false;
  const size_t total_bytes = total_elems * sizeof(float);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> bk = pool.get(output_name);
  if (!bk || bk->bytes() < total_bytes ||
      bk->encoding() != tv::Encoding::FP32) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return false;
    bk = std::make_shared<tv::TensorBacking>(ctx, buf, tv::Encoding::FP32,
                                             tv::Layout::ROW_MAJOR, total_bytes,
                                             /** owned */ true);
    pool.set(output_name, bk);
  }
  // Also register under the host-data-pointer key so consumers receiving
  // a different Tensor instance (same underlying buffer) find this entry.
  {
    char key_buf[64];
    std::snprintf(key_buf, sizeof(key_buf), "ptr:%p",
                  static_cast<const void *>(output.getData<uint8_t>()));
    pool.set(std::string(key_buf), bk);
  }

  // Upload the CPU-computed RMSNorm output into the backing's cl_mem.
  if (clEnqueueWriteBuffer(q, bk->buffer(), CL_FALSE, 0, total_bytes,
                           output.getData<uint8_t>(), 0, nullptr,
                           nullptr) != CL_SUCCESS)
    return false;
  // Bump the generation so the v8c quant cache invalidates stale entries.
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);
  // We don't need a barrier here — the FC's queued ops follow this write
  // in the same queue; OoO scheduler tracks the cl_mem write dependency
  // for the FC's read (and barrier would prevent the FC enqueue from
  // starting earlier than necessary anyway).

  return true;
}

// [resident-act] Publish a GPU-resident activation: GPU-copy the producer's
// SVM output (FP16/FP32) into a cl_mem TensorBacking keyed `resact:`+name (the
// producer's graph-output name), so a downstream CL layer that resolved this
// edge (resolveResidentEdge) consumes the cl_mem directly instead of the SVM
// buffer. No host bounce (reuses the GPU v8c_copy_svm_to_clmem). Step 1 of the
// cl_mem residency overlay. Returns false on failure (caller keeps SVM path).
// Create/reuse a cl_mem TensorBacking keyed `resact:`+name (no data written),
// bump the residency generation, and return its cl_mem. A producer can bind
// this buffer as its kernel's output to write the activation device-resident
// directly (no SVM intermediate); the downstream consumer resolves the edge and
// reads it. Returns nullptr on failure.
void *get_or_create_resident_backing(const std::string &name,
                                     unsigned int n_elems, bool fp16) {
  if (n_elems == 0)
    return nullptr;
  const size_t total_bytes = (size_t)n_elems * (fp16 ? 2u : 4u);
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return nullptr;
  cl_context ctx = blas_cc->context_inst_.GetContext();
  const tv::Encoding enc = fp16 ? tv::Encoding::FP16 : tv::Encoding::FP32;
  const std::string key = "resact:" + name;
  auto &pool = tv::TensorBackingPool::Global();
  std::shared_ptr<tv::TensorBacking> bk = pool.get(key);
  if (!bk || bk->bytes() < total_bytes || bk->encoding() != enc) {
    cl_int err = CL_SUCCESS;
    cl_mem buf =
      clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buf)
      return nullptr;
    bk = std::make_shared<tv::TensorBacking>(
      ctx, buf, enc, tv::Layout::ROW_MAJOR, total_bytes, /** owned */ true);
    pool.set(key, bk);
  }
  g_resident_quant_generation.fetch_add(1, std::memory_order_release);
  return static_cast<void *>(bk->buffer());
}

bool publish_resident_act(const std::string &name, const void *svm_ptr,
                          unsigned int n_elems, bool fp16) {
  if (!svm_ptr || n_elems == 0)
    return false;
  cl_mem buf =
    static_cast<cl_mem>(get_or_create_resident_backing(name, n_elems, fp16));
  if (!buf)
    return false;
  // GPU copy the SVM activation into the backing cl_mem (no host round-trip).
  v8c_copy_svm_to_clmem(svm_ptr, buf, n_elems, fp16);
  return true;
}

// =============================================================================
// Fused RMSNorm + v8c activation quant (paper §3.6 fused-kernel idea).
// =============================================================================
bool readback_backing_to_host(Tensor &t) {
  const tv::TensorBacking *bk = t.getBacking();
  if (bk == nullptr || bk->buffer() == nullptr)
    return false;
  const auto &dim = t.getDim();
  const size_t elems =
    (size_t)dim.batch() * dim.channel() * dim.height() * dim.width();
  if (elems == 0)
    return false;
  const size_t elem_bytes =
    (t.getDataType() == ml::train::TensorDim::DataType::FP16)   ? 2u
    : (t.getDataType() == ml::train::TensorDim::DataType::FP32) ? 4u
                                                                : 0u;
  if (elem_bytes == 0)
    return false;
  const size_t bytes = elems * elem_bytes;
  if (bk->bytes() < bytes)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  clFinish(q);
  if (clEnqueueReadBuffer(q, bk->buffer(), CL_TRUE, 0, bytes,
                          t.getData<uint8_t>(), 0, nullptr,
                          nullptr) != CL_SUCCESS)
    return false;
  return true;
}

} // namespace nntrainer
