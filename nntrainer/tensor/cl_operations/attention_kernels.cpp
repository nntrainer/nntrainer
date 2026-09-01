// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file	attention_kernels.cpp
 * @date	28 August 2024
 * @brief	Common attention OpenCL kernels
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Yash Singh <yash.singh@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "attention_kernels_templates.h"
#include <array>
#include <blas_kernel_interface.h>
#include <blas_kernels.h> // v8c_use_buffer_path()
#include <chrono>
#include <cl_kernels/flash_attention.h>
#include <cl_kernels/rotary_emb.h>
#include <cl_kernels/two_conv_attention.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <nntrainer_log.h>
#include <opencl_loader.h>
#include <tuple>
#include <vector>

namespace nntrainer {

void rotary_emb_cl(float *in, float *out,
                   const std::vector<std::vector<float>> &freqs_cos,
                   const std::vector<std::vector<float>> &freqs_sin,
                   const std::vector<float> &cos_,
                   const std::vector<float> &sin_, unsigned int batch,
                   unsigned int channel, unsigned int height,
                   unsigned int width, unsigned int dim, unsigned int from,
                   unsigned int max_timestep, unsigned int in_size,
                   unsigned int out_size) {
  auto *cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  ClContext::SharedPtrClKernel kernel_rotaryEmb_ptr =
    cl_context->registerClKernel(rotary_emb_kernel, "rotary_emb_cl");
  if (!kernel_rotaryEmb_ptr) {
    return;
  }

  rotary_emb_cl_internal<float>(
    kernel_rotaryEmb_ptr, in, out, freqs_cos, freqs_sin, cos_, sin_, batch,
    channel, height, width, dim, from, max_timestep, in_size, out_size);
}

// =============================================================================
// Two-1x1-conv attention (paper section 3.7).
// =============================================================================
namespace {
/**
 * @brief Grow-only scratch buffers for the two-1x1-conv attention path
 *        (Q/K/V staging + score/output buffers), reused across calls.
 */
struct TcaScratch {
  // Q/K/V backing buffers used only on the non-SVM fallback.
  cl_mem q_buf = nullptr;
  size_t q_bytes = 0;
  cl_mem k_buf = nullptr;
  size_t k_bytes = 0;
  cl_mem v_buf = nullptr;
  size_t v_bytes = 0;
  cl_mem o_buf = nullptr;
  size_t o_bytes = 0;
  // Score matrix - always cl_mem, never SVM. Shape [H, M, N_kv] fp16.
  cl_mem scores = nullptr;
  size_t scores_bytes = 0;
  // int8-KV variant: separate scale buffers; K/V byte buffers reuse k_buf/v_buf
  // (size halved relative to the fp16 path).
  cl_mem k_scale_buf = nullptr;
  size_t k_scale_bytes = 0;
  cl_mem v_scale_buf = nullptr;
  size_t v_scale_bytes = 0;
  // Image2d_from_buffer cache (image variant). Views over q_buf/k_buf/v_buf,
  // valid as long as shape and underlying buffer don't change. Recreate when
  // (M, N_kv, HD_Q, HD_KV) shift.
  cl_mem q_image = nullptr;
  cl_mem k_image = nullptr;
  cl_mem v_image = nullptr;
  unsigned int img_M = 0, img_N_kv = 0;
  unsigned int img_HD_Q = 0, img_HD_KV = 0;

  // OHWI-reversed V image2d_from_buffer cache keyed by underlying cl_mem
  // pointer + (num_heads_KV, head_dim, max_seq_len). Each caller layer
  // tends to use its own V buffer; size is identical so we keep one
  // image and recreate when the buffer pointer changes.
  cl_mem v_ohwi_image = nullptr;
  cl_mem v_ohwi_buf = nullptr;   // underlying cl_mem (key)
  unsigned int v_ohwi_HD_KV = 0; // num_heads_KV * head_dim
  unsigned int v_ohwi_S_max = 0;
};
inline TcaScratch &tca_scratch() {
  static TcaScratch s;
  return s;
}
inline std::mutex &tca_mtx() {
  static std::mutex m;
  return m;
}

// Device-specialization gate (paper §3.4), shared with the v8c FC path via
// the same NNTR_V8C_BUF env flag. When set, the two_conv_attention program is
// built with -DTCA_BUFFER_ONLY so its image-sampling kernel bodies are
// excluded (Intel NEO cannot compile integer-coord read_imageui; the whole
// program build would otherwise fail, taking the image-FREE attention kernels
// down with it). Default "" keeps the Adreno image path bit-identical.
static const std::string &tca_copts() {
  static const std::string opts = []() {
    std::string o =
      v8c_use_buffer_path() ? std::string("-DTCA_BUFFER_ONLY") : std::string();
    return o;
  }();
  return opts;
}

static bool tca_ensure(cl_context ctx, cl_mem *buf, size_t *cap, size_t bytes,
                       cl_mem_flags flags) {
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
  *cap = bytes;
  return true;
}

} // namespace

bool two_conv_attention_prefill_f16_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal,
  bool svm_inputs) {
  // KNOWN ISSUE (session 2026-05-28 night): the qk_matmul_f16 kernel
  // is NUMERICALLY CORRECT (verified by the verify harness below: all
  // 3 stages × 16 heads × 28 layers stay within relL2 ~0.0005 vs CPU
  // reference). HOWEVER, on Qwen3-0.6B the per-layer ~0.05% drift
  // amplifies through 28 layers × int8 activation-quant bucket flips
  // and degrades model output: 1 GPU layer → still coherent; 14 GPU
  // layers → "lights" fragment; 28 GPU layers → "aines" / Arabic
  // tokens. Same pattern as the earlier rmsnorm_cl drift.
  //
  // Until either (a) the chain is hardened (move int8 quant + other
  // ops to GPU so the whole chain shares one numerics regime) or
  // (b) we test on bigger models that are
  // less brittle (Qwen3-4B), default to falling back to CPU mha so
  // the user-visible output stays coherent.
  //
  // To re-test the kernel, set NNTR_MHA_GPU_FORCE_BROKEN=1.
  // NNTR_MHA_GPU_LAYER_MAX=N restricts GPU mha to first N calls
  // (sticky-counted across the process) for drift bisection.
  static int _layer_call_count = 0;
  const char *_max_layer_env = std::getenv("NNTR_MHA_GPU_LAYER_MAX");
  const int _max_layer =
    (_max_layer_env != nullptr) ? std::atoi(_max_layer_env) : -1;
  if (_max_layer >= 0 && _layer_call_count >= _max_layer) {
    _layer_call_count++;
    return false;
  }
  _layer_call_count++;
  if (std::getenv("NNTR_MHA_GPU_FORCE_BROKEN") == nullptr && _max_layer < 0) {
    static int warned = 0;
    if (!warned && std::getenv("NNTR_MHA_GPU") != nullptr) {
      warned = 1;
      std::fprintf(stderr,
                   "[NOTE] NNTR_MHA_GPU=1 requested; kernels are math-"
                   "correct but per-layer drift amplifies through 28 "
                   "layers + int8 quant on this model size (Qwen3-0.6B) "
                   "to degraded output. Using CPU mha. Set "
                   "NNTR_MHA_GPU_FORCE_BROKEN=1 to force the GPU path, "
                   "or NNTR_MHA_GPU_LAYER_MAX=N to use GPU for the "
                   "first N attention calls.\n");
      std::fflush(stderr);
    }
    return false;
  }

  if (head_dim == 0 || M == 0 || N_kv == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  // Match the kernel tile defaults; relaxing requires re-defining TM/TN.
  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV = 4, TD_SV = 8;
  constexpr unsigned int SOFTMAX_LWS = 64;
  // d must be tile-aligned for the SV kernel; both M and N_kv get
  // tile-rounding by the kernel itself (tail-WI guards inside).
  if (head_dim % TD_SV != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t k_bytes = (size_t)N_kv * HD_KV * sizeof(uint16_t);
  const size_t v_bytes = k_bytes;
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes = (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  cl_mem q_arg = nullptr, k_arg = nullptr, v_arg = nullptr, o_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, k_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, v_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (opencl::clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host,
                                     0, nullptr, nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, k_bytes, K_host,
                                     0, nullptr, nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, v_bytes, V_host,
                                     0, nullptr, nullptr) != CL_SUCCESS)
      return false;
    q_arg = sc.q_buf;
    k_arg = sc.k_buf;
    v_arg = sc.v_buf;
    o_arg = sc.o_buf;
  }

  // CRITICAL: the shared command queue is created with
  // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE. Without an explicit event
  // chain or barrier, the kernels we enqueue below are free to overtake
  // the CL_FALSE writes above and read uninitialized Q/K/V. Force
  // ordering with a single clFinish — measured overhead ~0.1ms per
  // prefill, negligible vs ~900ms total mha time.
  opencl::clFinish(q);

  // Pre-K1 sync point for profiling. Always cheap when env unset.
  const bool _prof = std::getenv("NNTR_MHA_PROFILE") != nullptr;
  if (_prof) {
    opencl::clFinish(q);
  }

  // ---- K1: QK matmul ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16", tca_copts());
    if (!kp)
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
          !kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_host)))
        return false;
    } else {
      if (!kp->SetKernelArguments(0, &q_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &k_arg, sizeof(cl_mem)))
        return false;
    }
    if (!kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(10, &scale, sizeof(float)))
      return false;
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    // Pack WIs into Adreno-sized workgroups so the 64-wide subgroup is
    // actually fed. Without an explicit lws the driver defaults to 1
    // WI per WG, which on Adreno 830 leaves the wave/subgroup empty
    // and pegs the kernel to ~1/64 of compute peak. Pad gws[0] up to a
    // multiple of LWS_QK[0]; the kernel's `if (m0 >= M || n0 >= N_kv)
    // return` handles the padded slots harmlessly.
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  if (_prof) {
    opencl::clFinish(q);
  }

  // ---- K2: row softmax over N_kv ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "softmax_row_f16", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  if (_prof) {
    opencl::clFinish(q);
  }

  // ---- K3: scores @ V -> O ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(V_host)) ||
          !kp->SetKernelSVMArguments(2, O_host))
        return false;
    } else {
      if (!kp->SetKernelArguments(1, &v_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &o_arg, sizeof(cl_mem)))
        return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)))
      return false;
    const size_t dx = (head_dim + TD_SV - 1) / TD_SV;
    const size_t mx = (M + TM_SV - 1) / TM_SV;
    // Same fix as QK: explicit lws to fill the 64-wide Adreno subgroup
    // instead of relying on driver-default lws=1. Kernel has bounds
    // check so padded slots harmlessly return.
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  if (_prof) {
    opencl::clFinish(q);
  }

  if (svm_inputs) {
    opencl::clFinish(q);
  } else {
    if (opencl::clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                                    nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  if (_prof) {
  }

  return true;
}

// =============================================================================
// SVM-direct GPU RoPE. Keeps the activation on the device: rotate the
// (k, k+half_d) pairs of each [M, num_heads*head_dim] FP16 row by a cos/sin LUT
// indexed at absolute position (start_pos + row). in/out may alias (in-place Q)
// or differ (rotate-and-scatter K into its cache slice).
// =============================================================================
static const std::string rope_inplace_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rope_inplace_f16(__global const half *in,
                               __global       half *out,
                               __global const half *cos_lut,
                               __global const half *sin_lut,
                               const int M, const int num_heads,
                               const int half_d, const int start_pos,
                               const int write_off) {
  // write_off: when writing the rotated K into a STABLE base
  // (cache_key) instead of an offset-baked slice pointer, out[write_off + ..]
  // addresses the per-token row via a SCALAR (recordable) instead of an SVM
  // pointer. Default 0 == in-place / offset-baked-out behaviour (byte-identical).
  int t = get_global_id(0);
  int h = get_global_id(1);
  int k = get_global_id(2);
  if (t >= M || h >= num_heads || k >= half_d) return;
  long row = (long)t * num_heads * (2 * half_d) + (long)h * (2 * half_d);
  long lut = (long)(start_pos + t) * half_d + k;
  half c = cos_lut[lut];
  half s = sin_lut[lut];
  half lo = in[row + k];
  half hi = in[row + k + half_d];
  // FP32 rotation for SMALL step size (M < 32): decode (M=1) and short prefill,
  // where the half cos/sin multiply-add's precision loss distorts the softmax
  // DISTRIBUTION enough for SAMPLING to degenerate (argmax survives -> greedy
  // coherent). fp32 there is ~free (few rows) and makes it match the host RoPE.
  // For a large prefill (M >= 32) the half rotation is kept: it is ~6% faster on
  // the big RoPE and the model-agnostic threshold avoids a head_dim hack. Store
  // is always FP16 (the activation dtype). See reference_working_run_combos.
  if (M < 32) {
    float cf = (float)c, sf = (float)s, lof = (float)lo, hif = (float)hi;
    out[write_off + row + k]          = (half)(lof * cf - hif * sf);
    out[write_off + row + k + half_d] = (half)(hif * cf + lof * sf);
  } else {
    out[write_off + row + k]          = lo * c - hi * s;
    out[write_off + row + k + half_d] = hi * c + lo * s;
  }
}
__kernel void scatter_copy_f16(__global const half *in, __global half *out,
                               const int N) {
  int i = get_global_id(0);
  if (i < N) out[i] = in[i];
}
// Row-offset variant: writes into a STABLE base at out[write_off + i] instead of
// baking the destination offset into the pointer. write_off (= cache_index *
// num_heads_KV * head_dim) is a scalar arg, so a recorded decode KV-write can be
// replayed with write_off overridden per token (cl_qcom_recordable_queues can
// override scalar args but NOT SVM pointers). Same address as the offset-baked
// pointer form -> byte-identical output.
__kernel void scatter_copy_f16_row(__global const half *in, __global half *out,
                                   const int N, const int write_off) {
  int i = get_global_id(0);
  if (i < N) out[write_off + i] = in[i];
}
// OHWI K scatter: src concat [t, hKV, d] -> dst OHWI [hKV, max_S, d] at
// (position+t). Feeds the K image2d view (qk_matmul_f16_ohwi_img). Mirrors
// the K-side scatter into the OHWI mirror.
__kernel void k_scatter_ohwi(__global const half *src, __global half *dst,
                             const int M, const int hKV, const int d,
                             const int max_S, const int position,
                             const int src_off) {
  // src_off: read the current token's rotated K from a STABLE
  // base (cache_key) at a SCALAR row offset instead of a per-token slice
  // pointer. Default 0 == src already points at the token (byte-identical).
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)h * max_S * d + (long)(position + t) * d + x] =
    src[(long)src_off + (long)t * hKV * d + (long)h * d + x];
}
// OHWI-transposed V scatter: src concat [t, hKV, d] -> dst reversed-OHWI
// [hKV, d, max_S] at (position+t). Feeds the V image2d view
// (sv_matmul_f16_ohwi_img): the transposed V-side scatter.
__kernel void v_scatter_ohwi_t(__global const half *src, __global half *dst,
                               const int M, const int hKV, const int d,
                               const int max_S, const int position,
                               const int src_off) {
  // src_off: read the current token's V from a STABLE base
  // (cache_value) at a SCALAR row offset instead of a per-token slice pointer.
  // `position` still offsets only the DEST column. Default 0 == byte-identical.
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)h * d * max_S + (long)x * max_S + position + t] =
    src[(long)src_off + (long)t * hKV * d + (long)h * d + x];
}
// Inverse gathers (mirror -> concat SVM cache slice): the NNTR_MHA_CLMEM
// mode promotes the OHWI mirrors to the PRIMARY prefill store (no SVM
// side-fill during the prefill window); the host decode/save paths read the
// concat SVM slab, so the boundary syncs it back with these, once.
__kernel void k_gather_ohwi(__global const half *src, __global half *dst,
                            const int M, const int hKV, const int d,
                            const int max_S, const int position) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)t * hKV * d + (long)h * d + x] =
    src[(long)h * max_S * d + (long)(position + t) * d + x];
}
__kernel void v_gather_ohwi_t(__global const half *src, __global half *dst,
                              const int M, const int hKV, const int d,
                              const int max_S, const int position) {
  int t = get_global_id(0);
  int h = get_global_id(1);
  int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  dst[(long)t * hKV * d + (long)h * d + x] =
    src[(long)h * d * max_S + (long)x * max_S + position + t];
}
)CL";

/**
 * @brief Grow-only RoPE scratch: the I/O staging buffer plus the per-slot
 *        resident cos/sin LUT cache described below.
 */
struct RopeScratch {
  cl_mem io = nullptr;
  size_t io_bytes = 0;
  // Per-slot device LUT cache. Keyed by (cos_src host pointer, sin_src host
  // pointer, half_d): each distinct RoPE slot keeps its OWN resident device
  // cos/sin buffer, uploaded exactly ONCE. Models that alternate RoPE slots
  // per layer (Gemma4: sliding head_dim=256/theta=1e4, full
  // head_dim=512/theta=1e6) thus stop re-uploading the LUT at every
  // sliding<->full transition -- the previous single (cos/sin) buffer was
  // REALLOCATED + re-uploaded each transition because one device buffer cannot
  // hold both slots. The caller (MHACoreLayer) now hands a STABLE, distinct
  // host pointer per slot (std::map-node-stable flat-LUT cache), so the host
  // pointer is a sound cache key here (it was already the key; what changed is
  // that there is now one device buffer PER pointer, not one shared buffer).
  // Single-head_dim models populate exactly one slot, matching the previous
  // "uploaded once" behaviour. The half_d component disambiguates the (rare)
  // case where two slots' host buffers were freed + reallocated to the same
  // address but differ in width; positions are covered by sizing each slot's
  // buffer to its own max (grow-only via tca_ensure on the cached entry).
  /**
   * @brief One resident device cos/sin LUT pair for a single RoPE slot.
   */
  struct LutSlot {
    cl_mem cos = nullptr;
    size_t cos_bytes = 0;
    cl_mem sin = nullptr;
    size_t sin_bytes = 0;
    unsigned int positions = 0;
    bool uploaded = false;
  };
  using LutKey = std::tuple<const void *, const void *, int>;
  std::map<LutKey, LutSlot> lut_slots;
};
static RopeScratch &rope_scratch() {
  static RopeScratch s;
  return s;
}

bool rope_inplace_f16_cl(const uint16_t *in, uint16_t *out,
                         const uint16_t *cos_lut, const uint16_t *sin_lut,
                         unsigned int M, unsigned int num_heads,
                         unsigned int head_dim, unsigned int start_pos,
                         unsigned int max_positions, bool svm_inputs,
                         void *in_clmem, void *out_clmem, bool drain_svm_out,
                         unsigned int write_off) {
  if (M == 0 || num_heads == 0 || head_dim == 0 || (head_dim & 1u))
    return false;
  if (in == nullptr || out == nullptr || cos_lut == nullptr ||
      sin_lut == nullptr)
    return false;
  if (start_pos + M > max_positions)
    return false;
  const int half_d = (int)(head_dim / 2);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "rope_inplace_f16");
  if (!kp)
    return false;

  const size_t io_bytes = (size_t)M * num_heads * head_dim * sizeof(uint16_t);
  const size_t lut_bytes = (size_t)max_positions * half_d * sizeof(uint16_t);

  // The cos/sin LUT is a constant table (not the activation), staged through
  // cl_mem scratch and uploaded ONCE PER SLOT (keyed by source pointer pair +
  // half_d) — repeated RoPE calls on the same slot reuse the resident device
  // buffer, no per-call upload, and alternating slots (Gemma4 sliding<->full)
  // each keep their own resident buffer so a transition is a cache HIT not a
  // re-upload. The activation (in/out) is bound SVM-direct (residency) when
  // svm_inputs, else uploaded/read-back via cl_mem. Mixing an SVM arg with
  // cl_mem args in one kernel is valid.
  RopeScratch &sc = rope_scratch();
  RopeScratch::LutSlot &slot =
    sc.lut_slots[RopeScratch::LutKey{cos_lut, sin_lut, half_d}];
  if (!tca_ensure(ctx, &slot.cos, &slot.cos_bytes, lut_bytes,
                  CL_MEM_READ_ONLY) ||
      !tca_ensure(ctx, &slot.sin, &slot.sin_bytes, lut_bytes, CL_MEM_READ_ONLY))
    return false;
  bool lut_uploaded = false;
  // NNTR_ROPE_REUPLOAD=1 (bisect): force the per-call LUT re-upload +
  // clFinish that the per-instance LUT pointers used to cause (52 hidden
  // drains/forward) -- isolates that ordering change from the FC flush.
  static const bool rope_reup = std::getenv("NNTR_ROPE_REUPLOAD") != nullptr;
  // (re)upload only when this slot has not been uploaded yet, its buffer was
  // (re)allocated to cover more positions, or the bisect flag forces it.
  if (rope_reup || !slot.uploaded || slot.positions < max_positions) {
    if (opencl::clEnqueueWriteBuffer(q, slot.cos, CL_FALSE, 0, lut_bytes,
                                     cos_lut, 0, nullptr,
                                     nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, slot.sin, CL_FALSE, 0, lut_bytes,
                                     sin_lut, 0, nullptr,
                                     nullptr) != CL_SUCCESS)
      return false;
    slot.positions = max_positions;
    slot.uploaded = true;
    lut_uploaded = true;
  }

  cl_mem io_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.io, &sc.io_bytes, io_bytes, CL_MEM_READ_WRITE))
      return false;
    if (opencl::clEnqueueWriteBuffer(q, sc.io, CL_FALSE, 0, io_bytes, in, 0,
                                     nullptr, nullptr) != CL_SUCCESS)
      return false;
    io_arg = sc.io;
  }
  // Ordering: drain only when an upload actually happened this call (the LUT
  // is cached across calls; an unconditional clFinish here cost ~11ms of GPU
  // idle per layer, measured wv-GEMM -> rope at 279ms/prefill). The io scratch
  // upload (non-SVM path) also needs it.
  if (lut_uploaded || !svm_inputs)
    opencl::clFinish(q);

  if (svm_inputs) {
    // Static GPU_CLMEM residency: each of in/out binds its own plane (the
    // tensor's planner cl_mem sub-buffer when given, SVM otherwise). Mixed
    // args are valid -- e.g. K rotate-and-scatter: cl_mem in (the wk FC
    // output), SVM out (the KV cache slice)... except the kernel addresses
    // out from base, so a nonzero-offset SVM cache slice keeps the SVM
    // pointer (which carries the offset).
    bool okb = true;
    if (in_clmem != nullptr) {
      cl_mem h = static_cast<cl_mem>(in_clmem);
      okb = okb && kp->SetKernelArguments(0, &h, sizeof(cl_mem));
    } else {
      okb = okb && kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(in));
    }
    if (out_clmem != nullptr) {
      cl_mem h = static_cast<cl_mem>(out_clmem);
      okb = okb && kp->SetKernelArguments(1, &h, sizeof(cl_mem));
    } else {
      okb = okb && kp->SetKernelSVMArguments(1, out);
    }
    if (!okb)
      return false;
  } else {
    if (!kp->SetKernelArguments(0, &io_arg, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &io_arg, sizeof(cl_mem)))
      return false;
  }
  if (!kp->SetKernelArguments(2, &slot.cos, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(3, &slot.sin, sizeof(cl_mem)))
    return false;

  int Mi = (int)M, nh = (int)num_heads, hd = half_d, sp = (int)start_pos;
  int woff = (int)write_off;
  if (!kp->SetKernelArguments(4, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(5, &nh, sizeof(int)) ||
      !kp->SetKernelArguments(6, &hd, sizeof(int)) ||
      !kp->SetKernelArguments(7, &sp, sizeof(int)) ||
      !kp->SetKernelArguments(8, &woff, sizeof(int)))
    return false;

  constexpr size_t LWS_K = 64;
  const size_t kx_pad = (((size_t)half_d + LWS_K - 1) / LWS_K) * LWS_K;
  std::array<size_t, 3> gws = {(size_t)M, (size_t)num_heads, kx_pad};
  std::array<size_t, 3> lws = {1, 1, LWS_K};

  // Dispatch the in-place RoPE rotation over (M, num_heads, head_dim/2).
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);

  if (svm_inputs) {
    // Trailing drain only when the OUTPUT went to SVM (a downstream device
    // SVM read depends on it). A cl_mem output feeds a pure kernel chain
    // (scatter/qk) that the in-order queue serializes -- no drain, but DO
    // flush: the removed clFinish was also the submission point, and without
    // it the enqueued chain sits host-side until the next blocking call
    // (measured as the idle moving to rope->rope 820ms instead of vanishing).
    // drain_svm_out=false (staged image-attention chain): every consumer of
    // the rotated SVM output is a same-queue GPU kernel, so keep only the
    // submission flush -- the per-call drain was 19ms/1K-prefill of GPU idle.
    // NNTR_ROPE_NOFLUSH=1 (experiment): on a cl_mem output, skip even the
    // flush so the rotation and its consumer kernels share ONE submission
    // (the cross-submission cl_mem write->read is the suspected Q-flip
    // mechanism; the attention impl's trailing flush still submits).
    static const bool rope_noflush = []() {
      const char *e = std::getenv("NNTR_ROPE_NOFLUSH");
      return e && e[0] == '1';
    }();
    if (out_clmem == nullptr && drain_svm_out)
      opencl::clFinish(q);
    else if (out_clmem == nullptr || !rope_noflush)
      opencl::clFlush(q);
    // NNTR_ROPE_FINISH=1 (Xe3 regression probe): force a full drain after the
    // rope kernel so the cl_mem rotated-Q write completes before the attention
    // reads it in a separate submission (the suspected cross-submission
    // Q-flip).
    static const bool rope_finish = std::getenv("NNTR_ROPE_FINISH") != nullptr;
    if (rope_finish)
      opencl::clFinish(q);
    // NNTR_ROPE_SYNC_SVM=1 (Xe3 cl_mem<->SVM coherence fix probe): the kernel
    // wrote the rotated values to out_clmem, but on Xe3 the aliased SVM plane
    // is NOT updated -> an SVM-reading consumer gets the stale (pre-rotation)
    // Q. Copy cl_mem -> SVM so both planes agree.
    static const bool rope_sync_svm =
      std::getenv("NNTR_ROPE_SYNC_SVM") != nullptr;
    if (rope_sync_svm && out_clmem && out) {
      opencl::clFinish(q);
      opencl::clEnqueueReadBuffer(q, static_cast<cl_mem>(out_clmem), CL_TRUE, 0,
                                  io_bytes, out, 0, nullptr, nullptr);
    }
  } else {
    if (opencl::clEnqueueReadBuffer(q, rope_scratch().io, CL_TRUE, 0, io_bytes,
                                    out, 0, nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  return true;
}

void attention_prewarm_programs(ClContext &cc) {
  // rope_inplace_kernel also hosts scatter_copy_f16 / k_scatter_ohwi /
  // v_scatter_ohwi_t; one registration builds the shared program.
  cc.registerClKernel(rope_inplace_kernel, "rope_inplace_f16");
}

bool ensure_cl_stage_buf(void **buf, size_t *cap, size_t bytes) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  return tca_ensure(ctx, reinterpret_cast<cl_mem *>(buf), cap, bytes,
                    CL_MEM_READ_WRITE);
}

bool gpu_copy_f16_cl(const uint16_t *in, uint16_t *out, unsigned int N,
                     bool svm_inputs, void *in_clmem, void *out_clmem,
                     bool drain) {
  if (N == 0 || in == nullptr || out == nullptr)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "scatter_copy_f16");
  if (!kp)
    return false;

  const size_t bytes = (size_t)N * sizeof(uint16_t);
  cl_mem in_arg = nullptr, out_arg = nullptr;
  if (svm_inputs) {
    // Static GPU_CLMEM residency: bind the source as its planner cl_mem
    // sub-buffer when given (the wv FC output); the destination stays an SVM
    // pointer (the KV cache slice at a nonzero offset, which the pointer
    // carries). Mixed args are valid.
    bool okb = true;
    if (in_clmem != nullptr) {
      cl_mem h = static_cast<cl_mem>(in_clmem);
      okb = okb && kp->SetKernelArguments(0, &h, sizeof(cl_mem));
    } else {
      okb = okb && kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(in));
    }
    if (out_clmem != nullptr) {
      cl_mem oh = static_cast<cl_mem>(out_clmem);
      okb = okb && kp->SetKernelArguments(1, &oh, sizeof(cl_mem));
    } else {
      okb = okb && kp->SetKernelSVMArguments(1, out);
    }
    if (!okb)
      return false;
  } else {
    // Reuse the rope io scratch for the source; a second scratch for dst.
    static cl_mem dst = nullptr;
    static size_t dst_bytes = 0;
    RopeScratch &sc = rope_scratch();
    if (!tca_ensure(ctx, &sc.io, &sc.io_bytes, bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &dst, &dst_bytes, bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (opencl::clEnqueueWriteBuffer(q, sc.io, CL_FALSE, 0, bytes, in, 0,
                                     nullptr, nullptr) != CL_SUCCESS)
      return false;
    opencl::clFinish(q);
    in_arg = sc.io;
    out_arg = dst;
    if (!kp->SetKernelArguments(0, &in_arg, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &out_arg, sizeof(cl_mem)))
      return false;
  }
  int Ni = (int)N;
  if (!kp->SetKernelArguments(2, &Ni, sizeof(int)))
    return false;
  constexpr size_t LWS = 64;
  const size_t gws_x = (((size_t)N + LWS - 1) / LWS) * LWS;
  std::array<size_t, 3> gws = {gws_x, 1, 1};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (svm_inputs) {
    // Drain only for an SVM output (downstream device SVM read depends on
    // it); a cl_mem output feeds a kernel chain the in-order queue covers.
    // This clFinish measured 1010ms of GPU idle per prefill (V-copy ->
    // k_scatter, 26 x 39ms). drain=false additionally skips it for SVM
    // side-fill destinations consumed only after a later full queue drain.
    // When skipping, still clFlush: the drain doubled as the submission
    // point, and deferring submission just moves the idle to the next
    // blocking call instead of removing it.
    if (out_clmem == nullptr && drain)
      opencl::clFinish(q);
    else
      opencl::clFlush(q);
  } else {
    if (opencl::clEnqueueReadBuffer(q, out_arg, CL_TRUE, 0, bytes, out, 0,
                                    nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  return true;
}

// Row-offset KV side-fill: same as gpu_copy_f16_cl's SVM path but writes into a
// STABLE out_base at [write_off + i] (scatter_copy_f16_row) instead of an
// offset-baked destination pointer. write_off = cache_index * num_heads_KV *
// head_dim. Byte-identical to passing out_base + write_off as the pointer, but
// keeps the destination handle stable so a recorded decode KV-write can replay
// with write_off overridden per token (SVM-only; the cache is SVM/cl_mem
// resident). Returns false for the non-SVM (host-readback) shape.
bool gpu_copy_f16_row_cl(const uint16_t *in, uint16_t *out_base, unsigned int N,
                         int write_off, bool svm_inputs, void *in_clmem,
                         void *out_base_clmem, bool drain) {
  if (N == 0 || in == nullptr || out_base == nullptr || !svm_inputs)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "scatter_copy_f16_row");
  if (!kp)
    return false;
  bool okb = true;
  if (in_clmem != nullptr) {
    cl_mem h = static_cast<cl_mem>(in_clmem);
    okb = okb && kp->SetKernelArguments(0, &h, sizeof(cl_mem));
  } else {
    okb = okb && kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(in));
  }
  if (out_base_clmem != nullptr) {
    cl_mem oh = static_cast<cl_mem>(out_base_clmem);
    okb = okb && kp->SetKernelArguments(1, &oh, sizeof(cl_mem));
  } else {
    okb = okb && kp->SetKernelSVMArguments(1, out_base);
  }
  int Ni = (int)N;
  okb = okb && kp->SetKernelArguments(2, &Ni, sizeof(int));
  okb = okb && kp->SetKernelArguments(3, &write_off, sizeof(int));
  if (!okb)
    return false;
  constexpr size_t LWS = 64;
  const size_t gws_x = (((size_t)N + LWS - 1) / LWS) * LWS;
  std::array<size_t, 3> gws = {gws_x, 1, 1};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (out_base_clmem == nullptr && drain)
    opencl::clFinish(q);
  else
    opencl::clFlush(q);
  return true;
}

// Create a cl_mem OHWI mirror buffer + image2d_from_buffer view for the K or V
// cache, so the Adreno image attention (two_conv_attention_prefill_f16_ohwi_
// kvimg_view_cl) can read K/V via read_imageui (texture cache). The layer-graph
// KV cache is SVM (no cl_mem handle) and an image cannot wrap an SVM pointer,
// so this is a separate cl_mem mirror filled by k_scatter_ohwi /
// v_scatter_ohwi_t, writing the cache_{k,v} OHWI mirrors plus
// cache_{k,v}_image_ohwi creation.
//   K: OHWI [hKV, S_max, d]            -> image w=d/8,     h=hKV*S_max,
//   pitch=d*2 V: reversed-OHWI [hKV, d, S_max]   -> image w=S_max/8, h=hKV*d,
//   pitch=S_max*2
// Returns false (and leaves *buf / *image null) on failure; caller falls back.
bool create_ohwi_kv_mirror(bool is_v, unsigned int num_heads_KV,
                           unsigned int head_dim, unsigned int max_S,
                           cl_mem *out_buf, cl_mem *out_image) {
  *out_buf = nullptr;
  *out_image = nullptr;
  if (num_heads_KV == 0 || head_dim == 0 || max_S == 0)
    return false;
  if ((head_dim % 8u) != 0 || (max_S % 8u) != 0)
    return false; // 8 halves/texel
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  const size_t bytes =
    (size_t)num_heads_KV * max_S * head_dim * sizeof(uint16_t);

  cl_int err = CL_SUCCESS;
  cl_mem buf =
    opencl::clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS || buf == nullptr)
    return false;
  // Zero the padding rows once. Enqueued non-blocking: this runs on the
  // in-order SVM queue ahead of any scatter/attention that touches the mirror,
  // so no clFinish drain is needed (a per-layer drain here would bubble the GPU
  // 2x/layer = ~10s regression at prefill — the very host-sync anti-pattern we
  // avoid). Strictly the kernel only reads [0, N_kv) so the fill is defensive.
  const uint16_t zero = 0;
  opencl::clEnqueueFillBuffer(q, buf, &zero, sizeof(uint16_t), 0, bytes, 0,
                              nullptr, nullptr);

  cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc d{};
  d.image_type = CL_MEM_OBJECT_IMAGE2D;
  if (is_v) {
    d.image_width = (size_t)max_S / 8;
    d.image_height = (size_t)num_heads_KV * head_dim;
    d.image_row_pitch = (size_t)max_S * sizeof(uint16_t);
  } else {
    d.image_width = (size_t)head_dim / 8;
    d.image_height = (size_t)num_heads_KV * max_S;
    d.image_row_pitch = (size_t)head_dim * sizeof(uint16_t);
  }
  d.buffer = buf;
  cl_int ie = CL_SUCCESS;
  cl_mem image =
    opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &d, nullptr, &ie);
  if (ie != CL_SUCCESS || image == nullptr) {
    opencl::clReleaseMemObject(buf);
    return false;
  }
  *out_buf = buf;
  *out_image = image;
  return true;
}

// Release a cl_mem created by create_ohwi_kv_mirror. void* so callers needn't
// link OpenCL (the CausalLM layers link only libnntrainer + ccapi).
void release_cl_mem(void *mem) {
  if (mem)
    opencl::clReleaseMemObject(reinterpret_cast<cl_mem>(mem));
}

// Create a TIGHT-stride V image2d view over an existing (full-capacity) V
// mirror buffer: width = S/8 texels, pitch = S*2 bytes, S <= the buffer's
// allocated S_max. The V image row pitch is the texture-cache lever: the sv
// kernel walks texels along the sequence axis, and a pitch sized to the
// allocation cap (e.g. 2048) instead of the live sequence wastes texture
// cache on padding (measured: S_max 2048 -> 1024 cuts sv_matmul 63 -> 41ms
// M=843; a tight run reaches 1024). The sv kernels address V purely
// through image coordinates (their S_max argument is unused for V), so a
// tight view needs no kernel change -- only the scatter stride must match.
bool create_ohwi_v_image_view(void *v_buf, unsigned int num_heads_KV,
                              unsigned int head_dim, unsigned int *S_inout,
                              void **out_image) {
  *out_image = nullptr;
  if (v_buf == nullptr || num_heads_KV == 0 || head_dim == 0 ||
      S_inout == nullptr || *S_inout == 0)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  // image2d_from_buffer requires the row pitch to be a multiple of
  // CL_DEVICE_IMAGE_PITCH_ALIGNMENT *pixels* (RGBA32UI pixel = 8 halves), so
  // the stride S must be a multiple of 8*align (Adreno: align=32 -> S
  // multiple of 256; an unaligned S fails clCreateImage with
  // INVALID_IMAGE_DESCRIPTOR -- the silent killer of arbitrary tight
  // strides like 848). Round up and report the stride actually used.
  static const cl_uint pitch_align = [blas_cc]() {
    cl_uint a = 0;
    cl_device_id dev = blas_cc->context_inst_.GetDeviceId();
    if (opencl::clGetDeviceInfo(dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                                sizeof(cl_uint), &a, nullptr) != CL_SUCCESS ||
        a == 0)
      a = 64; // conservative fallback
    return a;
  }();
  const unsigned int s_align = 8u * (unsigned int)pitch_align;
  const unsigned int S = (*S_inout + s_align - 1u) / s_align * s_align;
  *S_inout = S;
  cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc d{};
  d.image_type = CL_MEM_OBJECT_IMAGE2D;
  d.image_width = (size_t)S / 8;
  d.image_height = (size_t)num_heads_KV * head_dim;
  d.image_row_pitch = (size_t)S * sizeof(uint16_t);
  d.buffer = reinterpret_cast<cl_mem>(v_buf);
  cl_int ie = CL_SUCCESS;
  cl_mem image =
    opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &d, nullptr, &ie);
  if (ie != CL_SUCCESS || image == nullptr)
    return false;
  *out_image = image;
  return true;
}

// Scatter this step's K (concat [M, hKV, d], SVM) into the OHWI K mirror buffer
// at row `position`. dst is the cl_mem buffer backing the K image2d.
bool k_scatter_ohwi_cl(const uint16_t *src_svm, cl_mem dst_buf, unsigned int M,
                       unsigned int num_heads_KV, unsigned int head_dim,
                       unsigned int max_S, unsigned int position,
                       void *src_clmem, unsigned int src_off) {
  if (M == 0 || num_heads_KV == 0 || head_dim == 0)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "k_scatter_ohwi");
  if (!kp)
    return false;
  int Mi = (int)M, hKVi = (int)num_heads_KV, di = (int)head_dim,
      maxSi = (int)max_S, posi = (int)position, soff = (int)src_off;
  bool ok0;
  if (src_clmem != nullptr) {
    cl_mem sh = static_cast<cl_mem>(src_clmem);
    ok0 = kp->SetKernelArguments(0, &sh, sizeof(cl_mem));
  } else {
    ok0 = kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(src_svm));
  }
  if (!ok0 || !kp->SetKernelArguments(1, &dst_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
      !kp->SetKernelArguments(4, &di, sizeof(int)) ||
      !kp->SetKernelArguments(5, &maxSi, sizeof(int)) ||
      !kp->SetKernelArguments(6, &posi, sizeof(int)) ||
      !kp->SetKernelArguments(7, &soff, sizeof(int)))
    return false;
  constexpr size_t LWS_Z = 64;
  std::array<size_t, 3> gws = {(size_t)M, (size_t)num_heads_KV,
                               ((size_t)head_dim + LWS_Z - 1) / LWS_Z * LWS_Z};
  std::array<size_t, 3> lws = {1, 1, LWS_Z};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  return true;
}

// Scatter this step's V (concat [M, hKV, d], SVM) into the reversed-OHWI V
// mirror buffer at column `position`.
bool v_scatter_ohwi_t_cl(const uint16_t *src_svm, cl_mem dst_buf,
                         unsigned int M, unsigned int num_heads_KV,
                         unsigned int head_dim, unsigned int max_S,
                         unsigned int position, void *src_clmem,
                         unsigned int src_off) {
  if (M == 0 || num_heads_KV == 0 || head_dim == 0)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "v_scatter_ohwi_t");
  if (!kp)
    return false;
  int Mi = (int)M, hKVi = (int)num_heads_KV, di = (int)head_dim,
      maxSi = (int)max_S, posi = (int)position, soff = (int)src_off;
  bool ok0;
  if (src_clmem != nullptr) {
    cl_mem sh = static_cast<cl_mem>(src_clmem);
    ok0 = kp->SetKernelArguments(0, &sh, sizeof(cl_mem));
  } else {
    ok0 = kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(src_svm));
  }
  if (!ok0 || !kp->SetKernelArguments(1, &dst_buf, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
      !kp->SetKernelArguments(4, &di, sizeof(int)) ||
      !kp->SetKernelArguments(5, &maxSi, sizeof(int)) ||
      !kp->SetKernelArguments(6, &posi, sizeof(int)) ||
      !kp->SetKernelArguments(7, &soff, sizeof(int)))
    return false;
  constexpr size_t LWS_Z = 64;
  std::array<size_t, 3> gws = {(size_t)M, (size_t)num_heads_KV,
                               ((size_t)head_dim + LWS_Z - 1) / LWS_Z * LWS_Z};
  std::array<size_t, 3> lws = {1, 1, LWS_Z};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  return true;
}

// Inverse gathers: mirror -> concat SVM cache slice rows [position,
// position+M). Boundary sync for NNTR_MHA_CLMEM (the prefill window keeps
// the mirrors as the only store; host decode/save read the SVM slab). The
// dst SVM pointer is the slab SLICE BASE (already offset to `position`).
// drain=true issues a clFinish so the host may read dst immediately after.
static bool kv_gather_dispatch(const char *kname, cl_mem src_buf,
                               uint16_t *dst_svm, unsigned int M,
                               unsigned int num_heads_KV, unsigned int head_dim,
                               unsigned int max_S, unsigned int position,
                               bool drain) {
  if (M == 0 || num_heads_KV == 0 || head_dim == 0 || src_buf == nullptr ||
      dst_svm == nullptr)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, kname);
  if (!kp)
    return false;
  int Mi = (int)M, hKVi = (int)num_heads_KV, di = (int)head_dim,
      maxSi = (int)max_S, posi = (int)position;
  if (!kp->SetKernelArguments(0, &src_buf, sizeof(cl_mem)) ||
      !kp->SetKernelSVMArguments(1, dst_svm) ||
      !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
      !kp->SetKernelArguments(4, &di, sizeof(int)) ||
      !kp->SetKernelArguments(5, &maxSi, sizeof(int)) ||
      !kp->SetKernelArguments(6, &posi, sizeof(int)))
    return false;
  constexpr size_t LWS_Z = 64;
  std::array<size_t, 3> gws = {(size_t)M, (size_t)num_heads_KV,
                               ((size_t)head_dim + LWS_Z - 1) / LWS_Z * LWS_Z};
  std::array<size_t, 3> lws = {1, 1, LWS_Z};
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (drain)
    opencl::clFinish(q);
  return true;
}

bool k_gather_ohwi_cl(cl_mem src_buf, uint16_t *dst_svm, unsigned int M,
                      unsigned int num_heads_KV, unsigned int head_dim,
                      unsigned int max_S, unsigned int position, bool drain) {
  return kv_gather_dispatch("k_gather_ohwi", src_buf, dst_svm, M, num_heads_KV,
                            head_dim, max_S, position, drain);
}

bool v_gather_ohwi_t_cl(cl_mem src_buf, uint16_t *dst_svm, unsigned int M,
                        unsigned int num_heads_KV, unsigned int head_dim,
                        unsigned int max_S, unsigned int position, bool drain) {
  return kv_gather_dispatch("v_gather_ohwi_t", src_buf, dst_svm, M,
                            num_heads_KV, head_dim, max_S, position, drain);
}

// =============================================================================
// int8-KV variant. Mirrors two_conv_attention_prefill_f16_cl but binds the
// int8 K/V byte buffers + their FP16 scale buffers, and dispatches the
// qk_matmul_f16_kvi8 / sv_matmul_f16_kvi8 kernels. Softmax kernel is
// shared with the fp16 variant since it operates only on the score buffer.
// =============================================================================
bool two_conv_attention_prefill_f16_kvi8_cl(
  const uint16_t *Q_host, const int8_t *K_i8_host, const int8_t *V_i8_host,
  const uint16_t *K_scale_host, const uint16_t *V_scale_host, uint16_t *O_host,
  unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal,
  bool svm_inputs) {
  if (head_dim == 0 || M == 0 || N_kv == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV = 4, TD_SV = 8;
  constexpr unsigned int SOFTMAX_LWS = 64;
  if (head_dim % TD_SV != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t k_i8_bytes = (size_t)N_kv * HD_KV * sizeof(int8_t);
  const size_t v_i8_bytes = k_i8_bytes;
  const size_t kscale_bytes = (size_t)N_kv * num_heads_KV * sizeof(uint16_t);
  const size_t vscale_bytes = kscale_bytes;
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes = (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  cl_mem q_arg = nullptr, k_arg = nullptr, v_arg = nullptr, o_arg = nullptr;
  cl_mem k_scale_arg = nullptr, v_scale_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, k_i8_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, v_i8_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_scale_buf, &sc.k_scale_bytes, kscale_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_scale_buf, &sc.v_scale_bytes, vscale_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (opencl::clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host,
                                     0, nullptr, nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, k_i8_bytes,
                                     K_i8_host, 0, nullptr,
                                     nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, v_i8_bytes,
                                     V_i8_host, 0, nullptr,
                                     nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.k_scale_buf, CL_FALSE, 0,
                                     kscale_bytes, K_scale_host, 0, nullptr,
                                     nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.v_scale_buf, CL_FALSE, 0,
                                     vscale_bytes, V_scale_host, 0, nullptr,
                                     nullptr) != CL_SUCCESS)
      return false;
    q_arg = sc.q_buf;
    k_arg = sc.k_buf;
    v_arg = sc.v_buf;
    o_arg = sc.o_buf;
    k_scale_arg = sc.k_scale_buf;
    v_scale_arg = sc.v_scale_buf;
  }

  // ---- K1: QK matmul (int8 K + scale) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16_kvi8", tca_copts());
    if (!kp)
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
          !kp->SetKernelSVMArguments(1, const_cast<int8_t *>(K_i8_host)) ||
          !kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(K_scale_host)))
        return false;
    } else {
      if (!kp->SetKernelArguments(0, &q_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &k_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &k_scale_arg, sizeof(cl_mem)))
        return false;
    }
    if (!kp->SetKernelArguments(3, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int nhkv = (int)num_heads_KV;
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(4, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(6, &di, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(8, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(9, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(10, &nhkv, sizeof(int)) ||
        !kp->SetKernelArguments(11, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(12, &scale, sizeof(float)))
      return false;
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K2: row softmax over N_kv (shared with fp16 path) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "softmax_row_f16", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K3: scores @ V (int8 V + scale) -> O ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16_kvi8", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(1, const_cast<int8_t *>(V_i8_host)) ||
          !kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(V_scale_host)) ||
          !kp->SetKernelSVMArguments(3, O_host))
        return false;
    } else {
      if (!kp->SetKernelArguments(1, &v_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &v_scale_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(3, &o_arg, sizeof(cl_mem)))
        return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int nhkv = (int)num_heads_KV;
    if (!kp->SetKernelArguments(4, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(6, &di, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(8, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(9, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(10, &nhkv, sizeof(int)))
      return false;
    const size_t dx = (head_dim + TD_SV - 1) / TD_SV;
    const size_t mx = (M + TM_SV - 1) / TM_SV;
    // Same fix as QK: explicit lws to fill the 64-wide Adreno subgroup
    // instead of relying on driver-default lws=1. Kernel has bounds
    // check so padded slots harmlessly return.
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  if (svm_inputs) {
    opencl::clFinish(q);
  } else {
    if (opencl::clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                                    nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  return true;
}

// =============================================================================
// image2d_from_buffer variant. Reads Q/K/V via 16-byte texels (8 halves per
// texel) — same trick that gave v8c FC kernel 87% of Adreno 830 peak. 8x
// fewer memory transactions per WI in the d-axis reduction. Non-SVM only:
// image2d_from_buffer requires a cl_mem, so SVM inputs are first copied to
// the scratch cl_mems (kept in TcaScratch alongside the fp16 wrapper's).
// =============================================================================
bool two_conv_attention_prefill_f16_img_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, bool causal) {
  if (head_dim == 0 || M == 0 || N_kv == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  // Smaller image-variant tiles to avoid register spill on Adreno.
  constexpr unsigned int TM_IMG = 2, TN_IMG = 4;
  constexpr unsigned int TM_SV_IMG = 2;
  constexpr unsigned int SOFTMAX_LWS = 64;
  // image2d packing requires d-multiple-of-8 + HD multiples-of-8.
  if (head_dim % 8 != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  if (HD_Q % 8 != 0 || HD_KV % 8 != 0)
    return false;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t k_bytes = (size_t)N_kv * HD_KV * sizeof(uint16_t);
  const size_t v_bytes = k_bytes;
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes = (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;
  if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
      !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, k_bytes, CL_MEM_READ_ONLY) ||
      !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, v_bytes, CL_MEM_READ_ONLY) ||
      !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
    return false;
  if (opencl::clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host, 0,
                                   nullptr, nullptr) != CL_SUCCESS ||
      opencl::clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, k_bytes, K_host, 0,
                                   nullptr, nullptr) != CL_SUCCESS ||
      opencl::clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, v_bytes, V_host, 0,
                                   nullptr, nullptr) != CL_SUCCESS)
    return false;

  // Build image2d views over the buffers (cached across layers — same shape
  // is reused across all 28 transformer blocks during a prefill).
  // RGBA UINT32 = 16 bytes = 8 halves per texel.
  cl_int err = CL_SUCCESS;
  const bool shape_changed = sc.img_M != M || sc.img_N_kv != N_kv ||
                             sc.img_HD_Q != HD_Q || sc.img_HD_KV != HD_KV ||
                             !sc.q_image || !sc.k_image || !sc.v_image;
  if (shape_changed) {
    if (sc.q_image) {
      opencl::clReleaseMemObject(sc.q_image);
      sc.q_image = nullptr;
    }
    if (sc.k_image) {
      opencl::clReleaseMemObject(sc.k_image);
      sc.k_image = nullptr;
    }
    if (sc.v_image) {
      opencl::clReleaseMemObject(sc.v_image);
      sc.v_image = nullptr;
    }

    cl_image_format img_fmt{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc qd{};
    qd.image_type = CL_MEM_OBJECT_IMAGE2D;
    qd.image_width = HD_Q / 8;
    qd.image_height = M;
    qd.image_row_pitch = HD_Q * sizeof(uint16_t);
    qd.buffer = sc.q_buf;
    sc.q_image = opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &qd,
                                       nullptr, &err);
    if (err != CL_SUCCESS || !sc.q_image)
      return false;

    cl_image_desc kd{};
    kd.image_type = CL_MEM_OBJECT_IMAGE2D;
    kd.image_width = HD_KV / 8;
    kd.image_height = N_kv;
    kd.image_row_pitch = HD_KV * sizeof(uint16_t);
    kd.buffer = sc.k_buf;
    sc.k_image = opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &kd,
                                       nullptr, &err);
    if (err != CL_SUCCESS || !sc.k_image)
      return false;

    cl_image_desc vd = kd;
    vd.buffer = sc.v_buf;
    sc.v_image = opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt, &vd,
                                       nullptr, &err);
    if (err != CL_SUCCESS || !sc.v_image)
      return false;

    sc.img_M = M;
    sc.img_N_kv = N_kv;
    sc.img_HD_Q = HD_Q;
    sc.img_HD_KV = HD_KV;
  }
  cl_mem q_image = sc.q_image;
  cl_mem k_image = sc.k_image;
  cl_mem v_image = sc.v_image;
  auto cleanup = []() {}; // images are cached, no per-call release

  // ---- K1: QK matmul (image2d Q, K) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16_img", tca_copts());
    if (!kp) {
      cleanup();
      return false;
    }
    if (!kp->SetKernelArguments(0, &q_image, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &k_image, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem))) {
      cleanup();
      return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(10, &scale, sizeof(float))) {
      cleanup();
      return false;
    }
    const size_t nx = (N_kv + TN_IMG - 1) / TN_IMG;
    const size_t mx = (M + TM_IMG - 1) / TM_IMG;
    // Fair comparison to the buffer variant: same lws=64 fix
    // (see aad66ab5). Without this the image kernel was dispatched
    // with driver-default lws=1, leaving the 64-wide Adreno subgroup
    // empty and tanking image2d perf below buffer perf.
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K2: softmax (shared with the scalar fp16 path) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "softmax_row_f16", tca_copts());
    if (!kp) {
      cleanup();
      return false;
    }
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem))) {
      cleanup();
      return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int))) {
      cleanup();
      return false;
    }
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K3: SV matmul (image2d V) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16_img", tca_copts());
    if (!kp) {
      cleanup();
      return false;
    }
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &v_image, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(2, &sc.o_buf, sizeof(cl_mem))) {
      cleanup();
      return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int))) {
      cleanup();
      return false;
    }
    const size_t dx = head_dim / 8;
    const size_t mx = (M + TM_SV_IMG - 1) / TM_SV_IMG;
    // Same lws=64 fix as the buffer variant. For Qwen3 head_dim=128,
    // dx=16 — padded up to 64; WIs 16..63 early-out via `x0 >= d`.
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  if (opencl::clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                                  nullptr, nullptr) != CL_SUCCESS) {
    cleanup();
    return false;
  }
  cleanup();
  return true;
}

// =============================================================================
// §3.8 OHWI K-cache variant of the three-kernel attention pipeline.
// Same pipeline as `two_conv_attention_prefill_f16_cl`; the only
// difference is K1 dispatches `qk_matmul_f16_ohwi`, which reads K
// with stride [H_kv][S_max][d] instead of [N_kv][H_kv][d]. V cache
// is still in the row-major concat layout so K3 (`sv_matmul_f16`)
// is reused unchanged.
//
// `svm_inputs=true` is the production path (Phase 2 use case): K_host
// points directly at the SVM-allocated KVCacheManager buffer for the
// current batch, no upload needed. The non-SVM fallback uploads
// `H_kv * S_max * d * sizeof(half)` bytes — the full per-batch slab
// (not just N_kv rows), because the kernel addresses by head_kv * S_max
// internally.
// =============================================================================
bool two_conv_attention_prefill_f16_ohwi_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs, unsigned int local_window) {
  if (head_dim == 0 || M == 0 || N_kv == 0 || max_seq_len == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  if (N_kv > max_seq_len)
    return false;
  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV = 4, TD_SV = 8;
  constexpr unsigned int SOFTMAX_LWS = 64;
  if (head_dim % TD_SV != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  // OHWI K buffer is the full per-batch slab: H_kv * S_max * d halves.
  const size_t k_bytes =
    (size_t)num_heads_KV * max_seq_len * head_dim * sizeof(uint16_t);
  // V is still in concat layout: [N_kv, H_kv * d] (only N_kv rows used).
  const size_t v_bytes = (size_t)N_kv * HD_KV * sizeof(uint16_t);
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes = (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  cl_mem q_arg = nullptr, k_arg = nullptr, v_arg = nullptr, o_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, k_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, v_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (opencl::clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host,
                                     0, nullptr, nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, k_bytes, K_host,
                                     0, nullptr, nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, v_bytes, V_host,
                                     0, nullptr, nullptr) != CL_SUCCESS)
      return false;
    q_arg = sc.q_buf;
    k_arg = sc.k_buf;
    v_arg = sc.v_buf;
    o_arg = sc.o_buf;
  }

  // Out-of-order queue barrier (same as concat variant).
  opencl::clFinish(q);

  // Intel NEO honors CL_QUEUE_OUT_OF_ORDER literally: the K1→K2→K3 chain
  // (data-dependent through `scores`) is NOT auto-serialized, so K3 can
  // read an unwritten `scores` and emit all-zero O. Adreno's driver
  // serializes same-buffer-dependent kernels in practice, so this gate
  // (NNTR_V8C_BUF, the existing Intel device-specialization signal)
  // keeps the Adreno path bit-identical. The barriers carry no math
  // change — pure ordering.
  static const bool ooo_barriers = v8c_use_buffer_path();
  auto serialize = [&]() {
    if (ooo_barriers)
      opencl::clEnqueueBarrierWithWaitList(q, 0, nullptr, nullptr);
  };

  // ---- K1: QK matmul OHWI ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16_ohwi", tca_copts());
    if (!kp)
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
          !kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_host)))
        return false;
    } else {
      if (!kp->SetKernelArguments(0, &q_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &k_arg, sizeof(cl_mem)))
        return false;
    }
    if (!kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, smax = (int)max_seq_len;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &smax, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(10, &scale, sizeof(float)))
      return false;
    int lw = (int)local_window;
    if (!kp->SetKernelArguments(11, &lw, sizeof(int)))
      return false;
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  serialize();

  // ---- K2: row softmax (unchanged, scores layout unaffected by OHWI) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "softmax_row_f16", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  serialize();

  // ---- K3: scores @ V -> O (V still concat; reuse sv_matmul_f16) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(V_host)) ||
          !kp->SetKernelSVMArguments(2, O_host))
        return false;
    } else {
      if (!kp->SetKernelArguments(1, &v_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &o_arg, sizeof(cl_mem)))
        return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, hdkv = (int)HD_KV;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &hdkv, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)))
      return false;
    const size_t dx = (head_dim + TD_SV - 1) / TD_SV;
    const size_t mx = (M + TM_SV - 1) / TM_SV;
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  if (svm_inputs) {
    opencl::clFinish(q);
  } else {
    if (opencl::clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                                    nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  return true;
}

// =============================================================================
// §3.8 FULL OHWI variant: K = OHWI [H_kv, S_max, d], V = OHWI-reversed
// [H_kv, d, S_max]. K1 uses qk_matmul_f16_ohwi (same as _ohwi_cl), K3
// uses sv_matmul_f16_ohwi (new). softmax_row_f16 unchanged — scores
// layout is independent of K/V layout.
// =============================================================================
bool two_conv_attention_prefill_f16_ohwi_full_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs, unsigned int local_window) {
  if (head_dim == 0 || M == 0 || N_kv == 0 || max_seq_len == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  if (N_kv > max_seq_len)
    return false;
  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV = 4, TD_SV = 8;
  constexpr unsigned int SOFTMAX_LWS = 64;
  if (head_dim % TD_SV != 0)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t q_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  // K and V buffers BOTH the full per-batch slab.
  const size_t kv_slab_bytes =
    (size_t)num_heads_KV * max_seq_len * head_dim * sizeof(uint16_t);
  const size_t o_bytes = (size_t)M * HD_Q * sizeof(uint16_t);
  const size_t scores_bytes = (size_t)num_heads_Q * M * N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  cl_mem q_arg = nullptr, k_arg = nullptr, v_arg = nullptr, o_arg = nullptr;
  if (!svm_inputs) {
    if (!tca_ensure(ctx, &sc.q_buf, &sc.q_bytes, q_bytes, CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.k_buf, &sc.k_bytes, kv_slab_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.v_buf, &sc.v_bytes, kv_slab_bytes,
                    CL_MEM_READ_ONLY) ||
        !tca_ensure(ctx, &sc.o_buf, &sc.o_bytes, o_bytes, CL_MEM_WRITE_ONLY))
      return false;
    if (opencl::clEnqueueWriteBuffer(q, sc.q_buf, CL_FALSE, 0, q_bytes, Q_host,
                                     0, nullptr, nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.k_buf, CL_FALSE, 0, kv_slab_bytes,
                                     K_host, 0, nullptr,
                                     nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q, sc.v_buf, CL_FALSE, 0, kv_slab_bytes,
                                     V_host, 0, nullptr, nullptr) != CL_SUCCESS)
      return false;
    q_arg = sc.q_buf;
    k_arg = sc.k_buf;
    v_arg = sc.v_buf;
    o_arg = sc.o_buf;
  }

  opencl::clFinish(q);

  // ---- K1: QK matmul OHWI (same as half-OHWI variant) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "qk_matmul_f16_ohwi", tca_copts());
    if (!kp)
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
          !kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_host)))
        return false;
    } else {
      if (!kp->SetKernelArguments(0, &q_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(1, &k_arg, sizeof(cl_mem)))
        return false;
    }
    if (!kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, smax = (int)max_seq_len;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &smax, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(10, &scale, sizeof(float)))
      return false;
    int lw = (int)local_window;
    if (!kp->SetKernelArguments(11, &lw, sizeof(int)))
      return false;
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    constexpr size_t LWS_QK_X = 64;
    const size_t nx_pad = ((nx + LWS_QK_X - 1) / LWS_QK_X) * LWS_QK_X;
    std::array<size_t, 3> gws = {nx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_QK_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K2: row softmax (unchanged) ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "softmax_row_f16", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  // ---- K3: scores @ V (V OHWI-reversed) -> O via sv_matmul_f16_ohwi ----
  {
    ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
      two_conv_attention_kernel, "sv_matmul_f16_ohwi", tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    if (svm_inputs) {
      if (!kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(V_host)) ||
          !kp->SetKernelSVMArguments(2, O_host))
        return false;
    } else {
      if (!kp->SetKernelArguments(1, &v_arg, sizeof(cl_mem)) ||
          !kp->SetKernelArguments(2, &o_arg, sizeof(cl_mem)))
        return false;
    }
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, smax = (int)max_seq_len;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &smax, sizeof(int)) || // S_max for V stride
        !kp->SetKernelArguments(8, &gqa, sizeof(int)))
      return false;
    const size_t dx = (head_dim + TD_SV - 1) / TD_SV;
    const size_t mx = (M + TM_SV - 1) / TM_SV;
    constexpr size_t LWS_SV_X = 64;
    const size_t dx_pad = ((dx + LWS_SV_X - 1) / LWS_SV_X) * LWS_SV_X;
    std::array<size_t, 3> gws = {dx_pad, mx, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_SV_X, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }

  if (svm_inputs) {
    opencl::clFinish(q);
  } else {
    if (opencl::clEnqueueReadBuffer(q, sc.o_buf, CL_TRUE, 0, o_bytes, O_host, 0,
                                    nullptr, nullptr) != CL_SUCCESS)
      return false;
  }
  return true;
}

// Core implementation shared by _img_cl (buf input → create image inside)
// and _img_view_cl (caller-cached image). When `v_image_in` is non-null
// we use it directly; otherwise we build/cache an image2d from
// `v_buf_in`. Similarly, when `k_image_in` is non-null we dispatch the
// image2d-K kernel (qk_matmul_f16_ohwi_img); otherwise we use the
// SVM-K kernel (qk_matmul_f16_ohwi) with K_svm.
static bool two_conv_attention_prefill_f16_ohwi_img_impl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem v_buf_in,
  cl_mem v_image_in, cl_mem k_image_in, uint16_t *O_svm, unsigned int M,
  unsigned int N_kv, unsigned int num_heads_Q, unsigned int num_heads_KV,
  unsigned int head_dim, unsigned int max_seq_len, bool causal,
  float attn_softcap = 0.0f, // Gemma2-style QK soft-cap (image-K)
  void *q_clmem = nullptr, void *o_clmem = nullptr,
  unsigned int local_window = 0); // >0: sliding-window mask (n+W <= q_pos)

bool two_conv_attention_prefill_f16_ohwi_img_cl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem V_buf_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal) {
  return two_conv_attention_prefill_f16_ohwi_img_impl(
    Q_svm, K_svm, V_buf_ohwi, /** v_image_in */ nullptr,
    /** k_image_in */ nullptr, O_svm, M, N_kv, num_heads_Q, num_heads_KV,
    head_dim, max_seq_len, causal);
}

bool two_conv_attention_prefill_f16_ohwi_img_view_cl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal) {
  if (!V_image_ohwi)
    return false;
  return two_conv_attention_prefill_f16_ohwi_img_impl(
    Q_svm, K_svm, /** v_buf_in */ nullptr, V_image_ohwi,
    /** k_image_in */ nullptr, O_svm, M, N_kv, num_heads_Q, num_heads_KV,
    head_dim, max_seq_len, causal);
}

bool two_conv_attention_prefill_f16_ohwi_kvimg_view_cl(
  const uint16_t *Q_svm, cl_mem K_image_ohwi, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, float attn_softcap, void *q_clmem, void *o_clmem,
  unsigned int local_window) {
  if (!K_image_ohwi || !V_image_ohwi)
    return false;
  return two_conv_attention_prefill_f16_ohwi_img_impl(
    Q_svm, /** K_svm */ nullptr, /** v_buf_in */ nullptr, V_image_ohwi,
    K_image_ohwi, O_svm, M, N_kv, num_heads_Q, num_heads_KV, head_dim,
    max_seq_len, causal, attn_softcap, q_clmem, o_clmem, local_window);
}

// =============================================================================
// §3.8 + image2d_from_buffer V variant. Q/K stay SVM (qk_matmul_f16_ohwi
// reused unchanged); V is wrapped as image2d_from_buffer (either built
// here from a cl_mem buffer, or passed in pre-built by the caller). We
// dispatch the new sv_matmul_f16_ohwi_img kernel. Same Adreno-image-
// cache mechanism that v8c FC exploits for 87% peak.
// =============================================================================
static bool two_conv_attention_prefill_f16_ohwi_img_impl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem v_buf_in,
  cl_mem v_image_in, cl_mem k_image_in, uint16_t *O_svm, unsigned int M,
  unsigned int N_kv, unsigned int num_heads_Q, unsigned int num_heads_KV,
  unsigned int head_dim, unsigned int max_seq_len, bool causal,
  float attn_softcap, void *q_clmem, void *o_clmem, unsigned int local_window) {
  if (head_dim == 0 || M == 0 || N_kv == 0 || max_seq_len == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  if (N_kv > max_seq_len)
    return false;
  if (!v_buf_in && !v_image_in)
    return false;
  if (!k_image_in && !K_svm)
    return false;
  if (max_seq_len % 8 != 0)
    return false;
  if (head_dim % 8 != 0)
    return false;

  constexpr unsigned int TM_QK = 4, TN_QK = 8;
  constexpr unsigned int TM_SV_OHWI = 4; // must match kernel #define
  constexpr unsigned int SOFTMAX_LWS = 64;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContext();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const size_t HD_Q = (size_t)num_heads_Q * head_dim;
  const size_t HD_KV = (size_t)num_heads_KV * head_dim;
  const size_t scores_bytes =
    (size_t)num_heads_Q * M * (size_t)N_kv * sizeof(uint16_t);

  std::lock_guard<std::mutex> lock(tca_mtx());
  TcaScratch &sc = tca_scratch();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes, scores_bytes,
                  CL_MEM_READ_WRITE))
    return false;

  // ---- Resolve V image2d: either caller-provided, or build / reuse a
  //      cached one keyed on V_buf_in + shape. ----
  cl_mem v_image = v_image_in;
  if (v_image == nullptr) {
    const bool v_changed =
      sc.v_ohwi_image == nullptr || sc.v_ohwi_buf != v_buf_in ||
      sc.v_ohwi_HD_KV != HD_KV || sc.v_ohwi_S_max != max_seq_len;
    if (v_changed) {
      if (sc.v_ohwi_image) {
        opencl::clReleaseMemObject(sc.v_ohwi_image);
        sc.v_ohwi_image = nullptr;
      }
      cl_image_format img_fmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc vd{};
      vd.image_type = CL_MEM_OBJECT_IMAGE2D;
      vd.image_width = max_seq_len / 8;
      vd.image_height = (size_t)num_heads_KV * head_dim;
      vd.image_row_pitch = (size_t)max_seq_len * sizeof(uint16_t);
      vd.buffer = v_buf_in;
      cl_int err = CL_SUCCESS;
      sc.v_ohwi_image = opencl::clCreateImage(ctx, CL_MEM_READ_ONLY, &img_fmt,
                                              &vd, nullptr, &err);
      if (err != CL_SUCCESS || !sc.v_ohwi_image) {
        sc.v_ohwi_image = nullptr;
        return false;
      }
      sc.v_ohwi_buf = v_buf_in;
      sc.v_ohwi_HD_KV = HD_KV;
      sc.v_ohwi_S_max = max_seq_len;
    }
    v_image = sc.v_ohwi_image;
  }

  // NNTR_ATTN_TPROF=1: host wall time of each CL call inside this wrapper
  // (which call blocks ~35ms/layer despite being enqueue-only).
  static const bool attn_tprof = std::getenv("NNTR_ATTN_TPROF") != nullptr;
  auto tnow = []() {
    // steady_clock == monotonic; ms as double (used only in differences).
    return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  };
  static double tp_k1a = 0, tp_k1e = 0, tp_k2 = 0, tp_k3 = 0, tp_fl = 0;
  static int tp_n = 0;
  double t_a = attn_tprof ? tnow() : 0;

  // This image attention impl runs only on the Adreno in-order queue (the
  // Intel/NEO path uses the buffer attention kernels — images are unreadable
  // there). The two opencl::clFinish(q) drains here (before qk, after sv) are
  // redundant for correctness: in-order execution already serializes the
  // copy_svm->qk SVM hand-off and the sv->copy_svm output hand-off on this same
  // queue. But each drain blocks the host until the GPU empties, so the GPU
  // then sits idle through the next host-side kernel setup — it bubbles the GPU
  // between the attention SVM bridges (~63 ms inter-kernel idle @ M=1024 on
  // Adreno 840). Skip by default; NNTR_ATTN_DRAIN=1 restores both for A/B.
  static const bool attn_drain = []() {
    const char *e = std::getenv("NNTR_ATTN_DRAIN");
    return e && std::atoi(e) != 0;
  }();
  if (attn_drain)
    opencl::clFinish(q);

  // ---- K1: QK matmul OHWI — pick image2d-K kernel when caller gave
  //          us a K image, else the SVM buffer kernel.
  {
    const char *k1_name =
      k_image_in != nullptr ? "qk_matmul_f16_ohwi_img" : "qk_matmul_f16_ohwi";
    // Per-call-site kernel handle cache: registerClKernel
    // measured ~12ms per cached lookup from this wrapper (3x/layer = the
    // 36ms/layer host issue tax the CL-event profiler shows as GPU idle).
    static ClContext::SharedPtrClKernel kp_img, kp_buf;
    ClContext::SharedPtrClKernel &kp =
      (k_image_in != nullptr) ? kp_img : kp_buf;
    if (!kp)
      kp = blas_cc->registerClKernel(two_conv_attention_kernel, k1_name,
                                     tca_copts());
    if (!kp)
      return false;
    // Static GPU_CLMEM residency: bind Q as its planner cl_mem sub-buffer
    // when given (the wq FC output), else the SVM pointer.
    if (q_clmem != nullptr) {
      cl_mem qh = static_cast<cl_mem>(q_clmem);
      if (!kp->SetKernelArguments(0, &qh, sizeof(cl_mem)))
        return false;
    } else if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_svm)))
      return false;
    if (k_image_in != nullptr) {
      if (!kp->SetKernelArguments(1, &k_image_in, sizeof(cl_mem)))
        return false;
    } else {
      if (!kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_svm)))
        return false;
    }
    if (!kp->SetKernelArguments(2, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, smax = (int)max_seq_len;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &smax, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)) ||
        !kp->SetKernelArguments(10, &scale, sizeof(float)))
      return false;
    // Gemma2-style: arg 11 = QK soft-cap (image-K kernel only; SVM-K has no
    // softcap param). local_window is the LAST arg of both kernels: 12 on
    // the image kernel, 11 on the SVM-K kernel.
    int lw = (int)local_window;
    if (k_image_in != nullptr) {
      if (!kp->SetKernelArguments(11, &attn_softcap, sizeof(float)) ||
          !kp->SetKernelArguments(12, &lw, sizeof(int)))
        return false;
    } else {
      if (!kp->SetKernelArguments(11, &lw, sizeof(int)))
        return false;
    }
    const size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    // The LWS is env-overridable + measured (NNTR_QK_LWS="x,y,z"); default
    // {16,4,1} won a fair A/B/A/B sweep at M=1024 on Adreno 830 (SD8 Elite):
    // qk_matmul_f16_ohwi_img 76.5 ms vs 110.2 ms for the prior {64,1,1}
    // (-30.6%; thermal pair: 76.49/76.55 vs 110.21/110.21), token 7212
    // match=1 (self+causal) unchanged. Runner-up {32,2,1} = 89.9 ms (-18%);
    // {32,1,1} REGRESSED to 144 ms; {128,1,1}/{256,1,1}/{64,2,1} were neutral
    // (~108-110 ms). The 2-D workgroup (16 n-cols x 4 m-rows) feeds the
    // 64-wide Adreno subgroup while reusing the m-row Q loads across the WG.
    // We compute the workgroup first, then pad gws.x (= nx_pad) up to a
    // multiple of the chosen lws.x so the divisibility guard holds, mirroring
    // the NNTR_SV_LWS pattern in the sv_matmul block below (re-pad +
    // NULL-fallback). If any gws[i] % lws[i] != 0 (e.g. an lws.y that does not
    // divide mx) we fall back to NULL (driver-chosen workgroup); the log's
    // kernel time reveals it.
    // Parse NNTR_QK_LWS once. Unset/garbage => default {8,8,1}.
    // 2026-06-18 re-sweep on Adreno 840 / M=999 (gemma4 prompt_1p2k, after the
    // M_pad-align fix): {8,8,1} best (best-of-3 prefill 2430 vs {16,4,1} 2345,
    // +3.5%); {32,2,1}/{64,1,1} regress. The old {16,4,1} (Adreno 830 / M=1024)
    // is recoverable via NNTR_QK_LWS=16,4,1.
    static const std::array<size_t, 3> qk_lws_env = []() {
      std::array<size_t, 3> v = {8, 8, 1}; // default (measured best, see above)
      const char *s = std::getenv("NNTR_QK_LWS");
      if (s != nullptr) {
        int a = 0, b = 0, c = 0;
        if (std::sscanf(s, "%d,%d,%d", &a, &b, &c) == 3) {
          v = {(size_t)a, (size_t)b, (size_t)c};
        }
      }
      return v;
    }();
    const size_t LWS_X = qk_lws_env[0];
    const size_t LWS_Y = qk_lws_env[1];
    const size_t LWS_Z = qk_lws_env[2];
    // Pad x AND y up to the chosen lws so the kernel's clamps (m0>=M /
    // n0>=N_kv early-return) cover the pad tiles. Padding only x was a
    // silent performance cliff: any M with ceil(M/TM_QK) % lws.y != 0
    // (e.g. the prompt_1k M=843 -> mx=211) failed the divisibility guard
    // and fell back to the driver-chosen NULL lws = qk 190ms vs 45ms
    // (4.7x: measured against an M=1024 run of this same
    // kernel, which happened to divide evenly).
    const size_t lx = LWS_X > 0 ? LWS_X : 1;
    const size_t ly = LWS_Y > 0 ? LWS_Y : 1;
    const size_t nx_pad = ((nx + lx - 1) / lx) * lx;
    const size_t mx_pad = ((mx + ly - 1) / ly) * ly;
    std::array<size_t, 3> gws = {nx_pad, mx_pad, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_X, LWS_Y, LWS_Z};
    // Divisibility guard: all three dims must divide, and all lws>0, else NULL.
    const bool lws_ok = LWS_X > 0 && LWS_Y > 0 && LWS_Z > 0 &&
                        (gws[0] % LWS_X == 0) && (gws[1] % LWS_Y == 0) &&
                        (gws[2] % LWS_Z == 0);
    if (attn_tprof) {
      double t = tnow();
      tp_k1a += t - t_a;
      t_a = t;
    }
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws_ok ? lws.data() : nullptr, 0,
                                               nullptr, nullptr);
    if (attn_tprof) {
      double t = tnow();
      tp_k1e += t - t_a;
      t_a = t;
    }
  }

  // ---- K2: row softmax (scores cl_mem, in-place) ----
  {
    static double tp2_reg = 0, tp2_arg = 0, tp2_enq = 0;
    static ClContext::SharedPtrClKernel kp; // call-site handle cache
    if (!kp)
      kp = blas_cc->registerClKernel(two_conv_attention_kernel,
                                     "softmax_row_f16", tca_copts());
    if (!kp)
      return false;
    if (attn_tprof) {
      double t = tnow();
      tp2_reg += t - t_a;
      if (tp_n < 4 || (t - t_a) > 5.0)
        std::fprintf(stderr, "[ATTN-TPROF-K2-CALL] i=%d reg=%.2fms\n", tp_n,
                     t - t_a);
      t_a = t;
    }
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    if (attn_tprof) {
      double t = tnow();
      tp2_arg += t - t_a;
      t_a = t;
    }
    std::array<size_t, 3> gws = {SOFTMAX_LWS, M, num_heads_Q};
    std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
    if (attn_tprof) {
      double t = tnow();
      tp2_enq += t - t_a;
      tp_k2 += tp2_enq; // keep aggregate roughly meaningful
      t_a = t;
      if (tp_n % 26 == 25) {
        std::fprintf(stderr,
                     "[ATTN-TPROF-K2] reg %.2fms arg %.2fms enq %.2fms\n",
                     tp2_reg, tp2_arg, tp2_enq);
        std::fflush(stderr);
        tp2_reg = tp2_arg = tp2_enq = 0;
      }
    }
  }

  // ---- K3: scores @ V_image -> O via sv_matmul_f16_ohwi_img ----
  {
    // M-tiled sv: 2 query rows/WI, reuse V across both. DEFAULT-ON
    // (token-identical; profiler sv 101.6->62.9ms M=1024 Adreno 840; tm2 kernel
    // handles odd M / M=1 decode via has1=(m1<M)). NNTR_SV_TM2=0 disables
    // (A/B).
    static const bool sv_tm2 = []() {
      const char *e = std::getenv("NNTR_SV_TM2");
      return e ? (std::atoi(e) != 0) : true;
    }();
    static ClContext::SharedPtrClKernel kp; // call-site handle cache (sv_tm2
                                            // is process-constant)
    if (!kp)
      kp = blas_cc->registerClKernel(two_conv_attention_kernel,
                                     sv_tm2 ? "sv_matmul_f16_ohwi_img_tm2"
                                            : "sv_matmul_f16_ohwi_img",
                                     tca_copts());
    if (!kp)
      return false;
    if (!kp->SetKernelArguments(0, &sc.scores, sizeof(cl_mem)) ||
        !kp->SetKernelArguments(1, &v_image, sizeof(cl_mem)))
      return false;
    // Static GPU_CLMEM residency: write O straight into its planner cl_mem
    // sub-buffer when given (the wo FC consumes it device-direct).
    if (o_clmem != nullptr) {
      cl_mem oh = static_cast<cl_mem>(o_clmem);
      if (!kp->SetKernelArguments(2, &oh, sizeof(cl_mem)))
        return false;
    } else if (!kp->SetKernelSVMArguments(2, O_svm))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, smax = (int)max_seq_len;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    if (!kp->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kp->SetKernelArguments(5, &di, sizeof(int)) ||
        !kp->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kp->SetKernelArguments(7, &smax, sizeof(int)) ||
        !kp->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kp->SetKernelArguments(9, &causal_i, sizeof(int)))
      return false;
    // TDX=8 tiled: each WI computes 8 output channels, so the x grid is
    // head_dim/8 work-items. Workgroup (LWS_X x's, LWS_Y=4 m's).
    // The LWS is env-overridable + measured (NNTR_SV_LWS="x,y,z"); default
    // {8,8,1} won a fair A/B/A/B sweep at M=1024 on Adreno 830 (SD8 Elite):
    // sv_matmul_f16_ohwi_img 108.2 ms vs 150.6 ms for the prior {16,4,1}
    // (-28%; ~166 ms for driver NULL lws), token 7212 match=1 unchanged.
    // {16,4,1} (prior hardcoded constexpr) is the runner-up. The padded gws
    // guarantees
    // divisibility for the default, but for an arbitrary env override we
    // re-pad to the chosen LWS and re-apply a divisibility guard (mirrors the
    // v8c_pick_lws + NULL-fallback pattern in blas_kernels.cpp): if any
    // gws[i] % lws[i] != 0 we fall back to NULL (driver-chosen workgroup).
    constexpr size_t TDX = 8;
    // Parse NNTR_SV_LWS once. "0,0,0" (or unset/garbage) => NULL lws.
    // 2026-06-18 re-sweep on Adreno 840 / M=999 (after the M_pad-align fix):
    // {4,16,1} best (paired with QK {8,8,1}); the old {8,8,1} (Adreno 830 /
    // M=1024) is recoverable via NNTR_SV_LWS=8,8,1.
    static const std::array<size_t, 3> sv_lws_env = []() {
      std::array<size_t, 3> v = {4, 16,
                                 1}; // default (measured best, see above)
      const char *s = std::getenv("NNTR_SV_LWS");
      if (s != nullptr) {
        int a = 0, b = 0, c = 0;
        if (std::sscanf(s, "%d,%d,%d", &a, &b, &c) == 3) {
          v = {(size_t)a, (size_t)b, (size_t)c};
        }
      }
      return v;
    }();
    const size_t LWS_X = sv_lws_env[0];
    const size_t LWS_Y = sv_lws_env[1];
    const size_t LWS_Z = sv_lws_env[2];
    const size_t dx = (head_dim + TDX - 1) / TDX;
    // Pad x/y up to the chosen LWS so the kernel's clamps cover the pad rows.
    const size_t lx = LWS_X > 0 ? LWS_X : 1;
    const size_t ly = LWS_Y > 0 ? LWS_Y : 1;
    const size_t dx_pad = ((dx + lx - 1) / lx) * lx;
    // TM2 halves the m-grid (2 query rows per WI).
    const size_t mrows = sv_tm2 ? ((size_t)M + 1) / 2 : (size_t)M;
    const size_t mx_pad = ((mrows + ly - 1) / ly) * ly;
    std::array<size_t, 3> gws = {dx_pad, mx_pad, num_heads_Q};
    std::array<size_t, 3> lws = {LWS_X, LWS_Y, LWS_Z};
    // Divisibility guard: all three dims must divide, and all lws>0, else NULL.
    const bool lws_ok = LWS_X > 0 && LWS_Y > 0 && LWS_Z > 0 &&
                        (gws[0] % LWS_X == 0) && (gws[1] % LWS_Y == 0) &&
                        (gws[2] % LWS_Z == 0);
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                               lws_ok ? lws.data() : nullptr, 0,
                                               nullptr, nullptr);
    if (attn_tprof) {
      double t = tnow();
      tp_k3 += t - t_a;
      t_a = t;
    }
  }

  if (attn_drain)
    opencl::clFinish(q);
  else
    opencl::clFlush(
      q); // submit K1-K3 now; don't wait for the next blocking call
  if (attn_tprof) {
    double t = tnow();
    tp_fl += t - t_a;
    if (++tp_n % 26 == 0) {
      std::fprintf(stderr,
                   "[ATTN-TPROF] n=%d k1-pre %.2fms k1-enq %.2fms k2-enq "
                   "%.2fms k3 %.2fms flush %.2fms\n",
                   tp_n, tp_k1a, tp_k1e, tp_k2, tp_k3, tp_fl);
      std::fflush(stderr);
      tp_k1a = tp_k1e = tp_k2 = tp_k3 = tp_fl = 0;
    }
  }
  return true;
}

// =============================================================================
// Fused single-kernel attention over the SAME two OHWI images as the
// 3-kernel _ohwi_kvimg_view path (K image [H_kv,S_max,d], reversed-V image
// [H_kv,d,S_max]), but computing each (head_q, m) row entirely in-kernel via
// LDS scores — the [H,M,N_kv] scores tensor is NEVER written to DRAM. One
// enqueue replaces qk_matmul + softmax_row + sv_matmul. Kernel:
// fused_row_attention_f16_ohwi_img (two_conv_attention.cl). NNTR_FLASH_IMG.
// =============================================================================
bool fused_row_attention_f16_ohwi_img_cl(
  const uint16_t *Q_svm, cl_mem K_image_ohwi, cl_mem V_image_ohwi,
  uint16_t *O_svm, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal) {
  if (!K_image_ohwi || !V_image_ohwi || !Q_svm || !O_svm)
    return false;
  if (head_dim == 0 || M == 0 || N_kv == 0 || max_seq_len == 0)
    return false;
  if (num_heads_KV == 0 || num_heads_Q % num_heads_KV != 0)
    return false;
  if (N_kv > max_seq_len)
    return false;
  if (head_dim % 8 != 0 || max_seq_len % 8 != 0)
    return false;
  // LDS sizing limits baked into the kernel (FUSED_ATTN_MAX_NKV / _MAX_D).
  if (max_seq_len > 1024 || head_dim > 128)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  const size_t HD_Q = (size_t)num_heads_Q * head_dim;

  std::lock_guard<std::mutex> lock(tca_mtx());
  // Drain prior q/k/v writes + scatter so the images and Q SVM are ready
  // (mirrors the 3-kernel image wrappers' ordering).
  opencl::clFinish(q);

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    two_conv_attention_kernel, "fused_row_attention_f16_ohwi_img", tca_copts());
  if (!kp)
    return false;
  if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_svm)))
    return false;
  if (!kp->SetKernelArguments(1, &K_image_ohwi, sizeof(cl_mem)))
    return false;
  if (!kp->SetKernelArguments(2, &V_image_ohwi, sizeof(cl_mem)))
    return false;
  if (!kp->SetKernelSVMArguments(3, O_svm))
    return false;
  int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
  int hdq = (int)HD_Q, smax = (int)max_seq_len;
  int gqa = (int)(num_heads_Q / num_heads_KV);
  int causal_i = causal ? 1 : 0;
  float scale = 1.0f / std::sqrt((float)head_dim);
  if (!kp->SetKernelArguments(4, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(5, &Nkvi, sizeof(int)) ||
      !kp->SetKernelArguments(6, &di, sizeof(int)) ||
      !kp->SetKernelArguments(7, &hdq, sizeof(int)) ||
      !kp->SetKernelArguments(8, &smax, sizeof(int)) ||
      !kp->SetKernelArguments(9, &gqa, sizeof(int)) ||
      !kp->SetKernelArguments(10, &causal_i, sizeof(int)) ||
      !kp->SetKernelArguments(11, &scale, sizeof(float)))
    return false;
  // One workgroup of FUSED_ATTN_LWS (=64, matches reqd_work_group_size) per
  // (head_q, m): gws.x == LWS, gws.y == M, gws.z == num_heads_Q.
  constexpr size_t LWS = 64;
  std::array<size_t, 3> gws = {LWS, M, num_heads_Q};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  opencl::clFinish(q);
  return true;
}

// =============================================================================
// Second stage of the GPU mha migration. Fused flash-attention prefill: ONE
// kernel does QK -> online-softmax -> S*V inline, so the [H, M, N_kv]
// scores tensor is NEVER materialized to global memory (that DRAM
// traffic is the measured root cause of the prefill gap vs the ML Drift
// paper on the same Adreno 830). See flash_attention.cl for the kernel
// design + the K-layout (OHWI vs concat) note.
//
// Layout contract (matches two_conv_attention_prefill_f16_ohwi_cl):
//   Q : [M, HD_Q] concat fp16    (HD_Q = num_heads_Q * head_dim)
//   K : OHWI [H_kv, max_seq_len, d]  (k_stride = max_seq_len > 0)
//       or pure concat [N_kv, HD_KV] (pass max_seq_len = 0)
//   V : [N_kv, HD_KV] concat fp16
//   O : [M, HD_Q] concat fp16
// Device-resident operands only (svm_inputs == true): every caller of this
// entry point feeds SVM buffers, so the host-upload path is not implemented
// here and the call returns false.
// =============================================================================
bool flash_attention_prefill_f16_cl(
  const uint16_t *Q_host, const uint16_t *K_host, const uint16_t *V_host,
  uint16_t *O_host, unsigned int M, unsigned int N_kv, unsigned int num_heads_Q,
  unsigned int num_heads_KV, unsigned int head_dim, unsigned int max_seq_len,
  bool causal, bool svm_inputs, float attn_softcap, unsigned int local_window,
  unsigned int ring_cap) {
  if (num_heads_Q == 0 || num_heads_KV == 0 || head_dim == 0 || M == 0 ||
      N_kv == 0)
    return false;
  if (num_heads_Q % num_heads_KV != 0)
    return false;
  // head_dim>128 (e.g. Gemma2 d=256) is supported ONLY by the d-TILING variants
  // (blockq/vec: VPL = d/LWS <= 8, enforced below) where each lane owns a slice
  // of head_dim — NOT by the full-private-d coop/skeleton path. blockq/vec are
  // the Intel/buffer default; reject only when a non-tiling path would be used.
  {
    const char *bq = std::getenv("NNTR_FLASH_BLOCKQ");
    const char *vc = std::getenv("NNTR_FLASH_VEC");
    const bool tiled = (bq ? std::atoi(bq) != 0 : v8c_use_buffer_path()) ||
                       (vc && std::atoi(vc) != 0);
    if (head_dim > 128 && !tiled)
      return false; // FLASH_MAX_D private-acc bound (coop/skeleton, Qwen3
                    // d=128)
  }
  if (!svm_inputs)
    return false; // device-resident operands only

  static int logged_trip = 0;
  if (!logged_trip && std::getenv("NNTR_GPU_MHA_TRIP") != nullptr) {
    logged_trip = 1;
    std::fprintf(stderr,
                 "[GPU-MHA] flash_attention dispatch: M=%u N_kv=%u "
                 "hq=%u hkv=%u d=%u S_max=%u causal=%d svm=%d\n",
                 M, N_kv, num_heads_Q, num_heads_KV, head_dim, max_seq_len,
                 (int)causal, (int)svm_inputs);
    std::fflush(stderr);
  }

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const int HD_Q = (int)(num_heads_Q * head_dim);
  const int HD_KV = (int)(num_heads_KV * head_dim);

  // K-layout selection: OHWI (cache_k_svm) uses max_seq_len as the
  // per-head row stride; concat passes k_stride = 0.
  const int k_stride = (int)max_seq_len; // >0 => OHWI, 0 => concat

  // No drain before the dispatch: CommandQueueManager::CreateCommandQueue()
  // creates the queue with properties 0, i.e. strictly in order, so the RoPE
  // and KV-scatter writes enqueued above are already ordered before this
  // kernel. The drain this used to take cost ~52 clFinish per forward of
  // attention dead time and bought nothing. If an out-of-order queue is ever
  // reintroduced, this dispatch needs an explicit wait list, not a drain.

  // NNTR_FLASH_COOP=1 selects the cooperative d-axis-tiled variant
  // (flash_attention_prefill_f16_coop): one workgroup per (head_q,
  // query_row), FLASH_COOP_LWS WIs each own a disjoint slice of head_dim,
  // tree-reduce the score dot in LDS and cooperatively run the shared
  // online-softmax (acc[d] lives in LDS, no per-WI d-wide private acc => no
  // register spill). The naive 1-WI variant (skeleton) is the default flash
  // path and remains reachable (NNTR_FLASH=1 without COOP) for A/B.
  static const int flash_coop = []() {
    const char *e = std::getenv("NNTR_FLASH_COOP");
    return (e && std::atoi(e) != 0) ? 1 : 0;
  }();
  // NNTR_FLASH_VEC=1 selects the VECTORIZED cooperative variant
  // (flash_attention_prefill_f16_coop_vec): same WG decomposition + online
  // softmax as the coop variant, but each WI owns a CONTIGUOUS d-lane block and
  // issues half8/half4/half2 vectorized K/V/Q loads (raises arithmetic
  // intensity on Intel Arc, which can't sample images). Best on Intel Arc at
  // NNTR_FLASH_COOP_LWS=16 (=> half8 loads) NNTR_FLASH_COOP_BLOCK_KV=4: the
  // fused attention kernel drops to 511 ms @ M=1024 (vs the coop variant's 1225
  // ms, 3-kernel qk+sv+sm 984 ms) => 1.92x faster than the 3-kernel path.
  // Optional LDS K/V staging (NNTR_FLASH_VEC_STAGE=1) measured a net loss here
  // (see below). Reuses NNTR_FLASH_COOP_LWS / NNTR_FLASH_COOP_BLOCK_KV.
  static const int flash_vec = []() {
    const char *e = std::getenv("NNTR_FLASH_VEC");
    if (e)
      return std::atoi(e) != 0 ? 1 : 0;
    // Device specialization: on the Intel/buffer path (NNTR_V8C_BUF) the
    // vectorized fused attention is the measured-best attention (Intel Arc
    // M=1024 1153 TPS vs scalar 3-kernel 727), so default it ON there. Adreno
    // (NNTR_V8C_BUF unset) keeps the naive flash OFF and uses the image
    // 3-kernel path — image attention beats flash 3x on Adreno.
    return v8c_use_buffer_path() ? 1 : 0; // Intel buffer path => 1
  }();
  // LDS staging (lever B) measured a NET LOSS on Intel Arc (per-tile barrier +
  // LDS pressure cut occupancy; the K/V row is small and L2-cached): 1257 ms vs
  // 511 ms @ M=1024. Default OFF. Set NNTR_FLASH_VEC_STAGE=1 to A/B it.
  static const int flash_vec_stage = []() {
    const char *e = std::getenv("NNTR_FLASH_VEC_STAGE");
    return (e && std::atoi(e) != 0) ? 1 : 0; // default off (lever A only)
  }();
  // NNTR_FLASH_BLOCKQ=1 selects the BLOCK-Q vectorized variant
  // (flash_attention_prefill_f16_blockq): one workgroup owns FBQ_TM query rows
  // of one head_q and loads each K[n]/V[n] ONCE for all TM rows -> cuts the
  // K/V re-read traffic that bottlenecks the 1-row vec kernel on Intel.
  static const int flash_blockq = []() {
    const char *e = std::getenv("NNTR_FLASH_BLOCKQ");
    if (e)
      return std::atoi(e) != 0 ? 1 : 0;
    // Default ON for the Intel/buffer path (NNTR_V8C_BUF) — Block-Q +
    // subgroup-reduce is the measured-best Intel attention (M=1024 ~2075 TPS
    // vs vec-flash ~1119, token-identical). Adreno (unset) uses the image path.
    return v8c_use_buffer_path() ? 1 : 0; // Intel buffer path => 1
  }();
  // FBQ_TM: query rows per workgroup. Default 4 (=> acc+q 2*TM*VPL floats stays
  // in registers at LWS>=32). Only 1/2/4/8 supported.
  static const int flash_blockq_tm = []() {
    const char *e = std::getenv("NNTR_FLASH_BLOCKQ_TM");
    int v = (e && std::atoi(e) > 0) ? std::atoi(e) : 2; // TM=2 measured best
    if (v != 1 && v != 2 && v != 4 && v != 8)
      v = 2;
    return v;
  }();
  // NNTR_FLASH_SG: Block-Q reduces the d-dot with sub_group_reduce_add
  // (LWS == subgroup size) instead of the LDS tree -> no red_sh, no barriers
  // (the dominant cost: ~512 barriers/WG). Intel only (cl_intel_subgroups).
  // Default ON for the Intel/buffer path (the +85% lever; M=1024 attention
  // 494->136 ms). Requires flash_blockq.
  static const int flash_blockq_sg = []() {
    const char *e = std::getenv("NNTR_FLASH_SG");
    if (e)
      return std::atoi(e) != 0 ? 1 : 0;
    // NNTR_DETERMINISTIC: sub_group_reduce_add's internal order is the one
    // vendor-opaque reduction on this path (every other reduce is an explicit
    // fixed LDS tree). Prefer the tree under the determinism contract unless
    // the user explicitly asked for SG. Cost: the +85% M=1024 attention lever
    // is lost (494 vs 136 ms) — prefill-only.
    const char *det = std::getenv("NNTR_DETERMINISTIC");
    if (det && det[0] == '1')
      return 0;
    return v8c_use_buffer_path() ? 1 : 0; // Intel buffer path => 1
  }();
  // FLASH_COOP_LWS: WG size for the coop variant (work-items cooperating
  // over head_dim). Default 64. LDS footprint is tiny (q_sh[d]+acc_sh[d]+
  // red_sh[BLOCK_KV*LWS] ~ 1-3 KB), well within Adreno's 32 KB at any LWS.
  // Must be a power of two (log-step reduction). Override via env.
  // NOTE (gemma4): NOT process-wide static — depends on the LIVE head_dim so a
  // model with TWO head_dims (gemma4: 256 sliding / 512 full) gets the right
  // VPL=d/LWS<=8 per call. The kernel cache (key = name+copts) still dedups the
  // compile per distinct (head_dim,...) so this stays a single compile per d.
  // flash_blockq / flash_blockq_tm are static const (file/function-static) --
  // they have static storage duration and are accessible inside the lambda
  // WITHOUT a capture; capturing them is ill-formed (ARM/NDK clang rejects it,
  // x86 was lax).
  const int flash_coop_lws = [head_dim]() {
    const char *e = std::getenv("NNTR_FLASH_COOP_LWS");
    int v;
    if (e && std::atoi(e) > 0) {
      v = std::atoi(e);
    } else if (flash_blockq) {
      // Block-Q register budget = 2*TM*VPL acc/q floats per WI. TM<=2
      // keeps half8 (LWS=16, the Intel optimum + the SG subgroup size); TM>=4
      // needs half4 (LWS=32) to avoid spill (half8/TM>=4 = 96 floats spills).
      // Gemma2 d=256: VPL must stay <=8, so LWS=d/8=32 (VPL=8, half8); at
      // TM=2 that is 2*2*8=32 acc/q floats — no spill. (d=128 keeps LWS=16.)
      // Gemma4 full_attention d=512: VPL must stay <=8 => LWS=d/8=64. A 64-wide
      // workgroup is valid on Intel for the LDS-tree path (reqd_work_group_size
      // only); the SG path is forced off for it below
      // (intel_reqd_sub_group_size (64) is INVALID — Intel subgroups are
      // {8,16,32}).
      v = ((int)head_dim >= 512)
            ? 64
            : (((int)head_dim > 128 || flash_blockq_tm >= 4) ? 32 : 16);
    } else {
      // Intel/buffer path default LWS=16 => VPL = d/16 = 8 (half8 vloads),
      // the measured Intel-Arc optimum (1153 TPS @ M=1024 vs 981 at LWS=64).
      // Adreno default stays 64.
      v = v8c_use_buffer_path() ? 16 : 64; // buffer path ⇒ 16
    }
    // Must be a power of two for the log-step tree reduction.
    if (v != 16 && v != 32 && v != 64 && v != 128 && v != 256)
      v = 64;
    return v;
  }();
  // Per-call effective SG flag. The subgroup-reduce path bakes
  // intel_reqd_sub_group_size(FLASH_VEC_LWS), so it is only valid when LWS is a
  // real Intel subgroup size {8,16,32}. Gemma4's d=512 layers need LWS=64
  // (VPL=8) which is NOT a valid subgroup size, so force the SG path off and
  // use the LDS-tree reduction (reqd_work_group_size only -> 64 is fine). The
  // d=256 sliding path keeps LWS=32 + SG (untouched). The kernel cache keys on
  // name+copts, so d256(LWS32,SG) and d512(LWS64,no-SG) are distinct variants.
  const int flash_blockq_sg_eff =
    (flash_blockq_sg && flash_coop_lws <= 32) ? 1 : 0;
  // FLASH_COOP_BLOCK_KV: keys reduced per phase (tunable; 1 = no blocking).
  static const int flash_coop_block_kv = []() {
    const char *e = std::getenv("NNTR_FLASH_COOP_BLOCK_KV");
    int v = (e && std::atoi(e) > 0) ? std::atoi(e) : 4; // Intel Arc sweet spot
    if (v < 1)
      v = 1;
    if (v > 64)
      v = 64;
    return v;
  }();

  // [#r30-q4] The fp16 DPAS tile kernel (flash_attention_prefill_f16_xmx)
  // for FULL-ATTENTION prefill calls (win==0): subgroup-owned 8-row tiles,
  // 16-key tiles on the systolic array. DEFAULT ON on DPAS-capable devices
  // (validated: gemma2 d=256, gemma4 d=512 via the NSG=2 split +72%@32K;
  // CHECK numerics clean on both, det 2-run byte-identical).
  // NNTR_FLASH_XMX=0 restores the scalar blockq path;
  // sliding-window calls are never routed (they keep the measured-best
  // blockq + window-skip path). NNTR_FLASH_XMX_CHECK=N additionally runs the
  // blockq kernel first on the same buffers and host-compares (first N
  // calls).
  static const int flash_xmx_req = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX");
    if (e)
      return std::atoi(e) != 0 ? 1 : 0;
    return ClContext::Global().caps().dpas ? 1 : 0; // default ON on DPAS HW
  }();
  // NNTR_FLASH_XMX_CHECK: 0/unset = off, 1 = compare the first 6 calls,
  // N>1 = compare the first N calls (long enough to reach late-layer dims,
  // e.g. gemma4's d=512 full-attention calls).
  static const int flash_xmx_check = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_CHECK");
    const int v = e ? std::atoi(e) : 0;
    return (v <= 0) ? 0 : (v == 1 ? 6 : v);
  }();
  const int win_i =
    (local_window > 0 && local_window < N_kv) ? (int)local_window : 0;
  const bool use_xmx = flash_blockq && flash_xmx_req && causal && win_i == 0 &&
                       (int)head_dim % 16 == 0 && (int)head_dim <= 512 &&
                       ClContext::Global().caps().dpas;
  // DPAS row-tile (M dimension). TM=8 measured best at every d incl. 512:
  // the register budget (~FXA_TM*d*6B/16 per lane) spills at TM8/d512, but
  // halving the per-visit DPAS count + tile count still nets -26% vs TM4
  // (gemma4 32K full-attn 147->108s). The d=512 kernel remains latency-bound
  // (~9ns/visit vs 0.56 at d=128 -- 16KB SLM V-tile caps residency and the
  // KCH=32 SLM VNNI gather serializes); the planned v2 is a dual-subgroup
  // d-split WG. NNTR_FLASH_XMX_TM=4/8 overrides for experiments.
  static const int xmx_tm_env = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_TM");
    const int v = e ? std::atoi(e) : 0;
    return (v == 4 || v == 8 || v == 16) ? v : 0;
  }();
  // TM=16 default: the kernel is K/V-bandwidth-bound (TM=4 A/B scales
  // traffic-proportionally), and 16 rows per K/V pass halves that traffic --
  // measured 32K full-attn -29% (gemma4 d=512/NSG4).
  const int xmx_tm = xmx_tm_env ? xmx_tm_env : 16;
  // v2: subgroups per WG, each owning a d-slice (d=512 -> 2 so the per-lane
  // chunk count returns to the d=256 register envelope and lane residency
  // doubles). NNTR_FLASH_XMX_NSG=1/2 overrides for A/B.
  static const int xmx_nsg_env = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_NSG");
    const int v = e ? std::atoi(e) : 0;
    return (v == 1 || v == 2 || v == 4) ? v : 0;
  }();
  // d=512 default NSG=4: slice chunk count drops to the d=128 register
  // envelope (qa 128B + acc 256B/lane, spill-free) and lane residency per WG
  // quadruples -- measured 79.1->50.2s full-attn vs NSG=2 (-37%), beating
  // even the exchange-free probe floor (55.4s). d<=256 stays NSG=1.
  int xmx_nsg = xmx_nsg_env ? xmx_nsg_env : (((int)head_dim >= 512) ? 4 : 1);
  // Guard: FXA_KCH_SUB truncates silently when 16*NSG does not
  // divide head_dim (no current dim hits this; env overrides could).
  if ((int)head_dim % (16 * xmx_nsg) != 0)
    xmx_nsg = 1;
  // Exchange batching (NSG>1 only): key-tiles per psum exchange. Default 2
  // (one WG-barrier pair per 32 keys; +XB*NSG*TM*64B SLM). NNTR_FLASH_XMX_XB
  // = 1/2/4 overrides for A/B.
  static const int xmx_xb_env = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_XB");
    const int v = e ? std::atoi(e) : 0;
    return (v == 1 || v == 2 || v == 4) ? v : 0;
  }();
  // XB default 1: batching measured NEUTRAL at NSG=4 (spill-free) and
  // REGRESSIVE at NSG=2 (scb spill traffic, +14%/+55% at XB=2/4). Env kept
  // for tuning on other SKUs.
  int xmx_xb = (xmx_nsg > 1) ? (xmx_xb_env ? xmx_xb_env : 1) : 1;
  // Exchange-reduction mode (NSG>1 only). The psum cross-subgroup round-trip
  // is ~56% of the d512 full-attn kernel and is SLM-traffic-bound, not
  // barrier-bound (isolated timing showed the WG barriers cost ~0 once the
  // psum traffic is gone), which is exactly why FXA_XB batching -- it only
  // cuts barrier frequency -- measured neutral. XRED=1 replaces the
  // all-to-all reduction (every subgroup reads all NSG partials for all TM
  // rows: NSG^2*TM*KT SLM reads/WG per tile) with a DISTRIBUTED one (each
  // subgroup reduces only its own rows once and publishes the full score to
  // a shared ssum buffer, all read back: ~2*NSG*TM*KT reads). The per-g sum
  // order is preserved so the output is BIT-IDENTICAL (NNTR_FLASH_XMX_CHECK:
  // same maxdiff, over_tol 0). Measured d512 attention bucket -31.5%
  // (35.5->24.3ms/3 calls, +5.9% prefill TPS @1K where d512 is ~13% of wall;
  // scales with its 76.5% share @32K). DEFAULT ON for NSG>1;
  // NNTR_FLASH_XMX_XRED=0 forces the old all-to-all path (A/B).
  static const int xmx_xred_env = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_XRED");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : -1; // -1 = unset -> default
  }();
  const int xmx_xred =
    (xmx_nsg > 1) ? (xmx_xred_env < 0 ? 1 : xmx_xred_env) : 0;
  // Guard: clamp XB so psum + vtile fit the per-WG SLM budget
  // (XB=4 + d=512 defaults previously exceeded it -> launch failure, no
  // fallback). Budget 64KB, the conservative Xe per-WG limit.
  {
    const size_t vtile_b = (size_t)16 * (size_t)head_dim * 2;
    auto slm_b = [&](int xb) {
      return vtile_b + (size_t)xb * xmx_nsg * xmx_tm * 16 * 4;
    };
    while (xmx_xb > 1 && slm_b(xmx_xb) > 64 * 1024)
      xmx_xb /= 2;
  }

  // Diagnostic: NNTR_FLASH_FP16_SCORE=1 truncates each score to fp16
  // before the online-softmax update, matching the 3-kernel baseline
  // (which stores scores as fp16). Used to confirm the flash vs baseline
  // greedy divergence is fp16-score precision, not an indexing bug.
  // NOTE (gemma4): NOT process-wide static — bakes FLASH_VEC_D=head_dim and
  // FLASH_COOP_LWS=flash_coop_lws, both of which vary per call when a model has
  // two head_dims. cl_context keys the compiled kernel on name+copts so each
  // distinct head_dim compiles its own variant exactly once.
  const std::string flash_copts = [&]() {
    const char *e = std::getenv("NNTR_FLASH_FP16_SCORE");
    std::string base = tca_copts();
    if (e && std::atoi(e) != 0)
      base += " -DFLASH_FP16_SCORE";
    if (flash_coop || flash_vec || flash_blockq) {
      base += " -DFLASH_COOP_LWS=" + std::to_string(flash_coop_lws);
      base += " -DFLASH_COOP_BLOCK_KV=" + std::to_string(flash_coop_block_kv);
    }
    if (flash_vec || flash_blockq) {
      // Block-Q reuses the vec FV_* macros (VPL = d/LWS half2/4/8 vloads).
      base += " -DFLASH_VEC_LWS=" + std::to_string(flash_coop_lws);
      base += " -DFLASH_VEC_BLOCK_KV=" + std::to_string(flash_coop_block_kv);
      // STAGE only affects the vec kernel (block-Q has no staging code); keep
      // the vec A/B knob, default off (measured net loss).
      base += " -DFLASH_VEC_STAGE=" + std::to_string(flash_vec_stage);
      // head_dim is constant across a run; bake it in so the kernel's
      // compile-time VPL = head_dim / LWS picks the right half2/4/8 vload.
      base += " -DFLASH_VEC_D=" + std::to_string((int)head_dim);
    }
    if (flash_blockq) {
      base += " -DFBQ_TM=" + std::to_string(flash_blockq_tm);
      if (flash_blockq_sg_eff)
        base += " -DFBQ_SG";
    }
    if (use_xmx) {
      base += " -DFLASH_XMX=1 -DFXA_D=" + std::to_string((int)head_dim) +
              " -DFXA_TM=" + std::to_string(xmx_tm) +
              " -DFXA_NSG=" + std::to_string(xmx_nsg) +
              " -DFXA_XB=" + std::to_string(xmx_xb) +
              " -DFXA_XRED=" + std::to_string(xmx_xred);
    }
    return base;
  }();
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    flash_attention_kernel,
    use_xmx        ? "flash_attention_prefill_f16_xmx"
    : flash_blockq ? "flash_attention_prefill_f16_blockq"
    : flash_vec    ? "flash_attention_prefill_f16_coop_vec"
    : flash_coop   ? "flash_attention_prefill_f16_coop"
                   : "flash_attention_prefill_f16_skeleton",
    flash_copts);
  if (!kp)
    return false;
  // CHECK mode: also register the blockq reference kernel (same copts).
  ClContext::SharedPtrClKernel kp_ref = nullptr;
  if (use_xmx && flash_xmx_check) {
    kp_ref = blas_cc->registerClKernel(flash_attention_kernel,
                                       "flash_attention_prefill_f16_blockq",
                                       flash_copts);
    if (!kp_ref)
      return false;
  }

  // The vectorized variant tiles head_dim into LWS contiguous blocks of
  // VPL = head_dim / LWS lanes — requires LWS | head_dim and VPL <= 8 (the
  // private half[8] block buffers). Qwen3 d=128 with LWS in {16,32,64} all
  // satisfy this; reject otherwise so the kernel never reads OOB.
  if (flash_vec || flash_blockq) {
    if ((int)head_dim % flash_coop_lws != 0 ||
        (int)head_dim / flash_coop_lws > 8 ||
        (int)head_dim / flash_coop_lws < 1)
      return false;
  }

  int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
  int gqa = (int)(num_heads_Q / num_heads_KV);
  int causal_i = causal ? 1 : 0;
  float scale = 1.0f / std::sqrt((float)head_dim);
  int win_arg = win_i;
  // softcap (arg 13) and local_window (arg 14) exist only on the Block-Q/XMX
  // kernels (Gemma2 attn logit soft-cap; Gemma4 sliding-window). The
  // coop/vec/skeleton variants (Qwen3 d=128, no softcap/window) don't declare
  // them, so only bind them for those paths. local_window<=0 or >=N_kv => no
  // window mask.
  auto bind_all = [&](ClContext::SharedPtrClKernel &kk) -> bool {
    if (!kk->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
        !kk->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_host)) ||
        !kk->SetKernelSVMArguments(2, const_cast<uint16_t *>(V_host)) ||
        !kk->SetKernelSVMArguments(3, O_host))
      return false;
    if (!kk->SetKernelArguments(4, &Mi, sizeof(int)) ||
        !kk->SetKernelArguments(5, &Nkvi, sizeof(int)) ||
        !kk->SetKernelArguments(6, &di, sizeof(int)) ||
        !kk->SetKernelArguments(7, &HD_Q, sizeof(int)) ||
        !kk->SetKernelArguments(8, &HD_KV, sizeof(int)) ||
        !kk->SetKernelArguments(9, &gqa, sizeof(int)) ||
        !kk->SetKernelArguments(10, &causal_i, sizeof(int)) ||
        !kk->SetKernelArguments(11, &scale, sizeof(float)) ||
        !kk->SetKernelArguments(12, &k_stride, sizeof(int)))
      return false;
    if (flash_blockq) {
      if (!kk->SetKernelArguments(13, &attn_softcap, sizeof(float)) ||
          !kk->SetKernelArguments(14, &win_arg, sizeof(int)))
        return false;
      // [kv-window-ring] arg 15 exists only on the Block-Q kernel, alongside
      // args 13/14 -- bind it in the same gate so no kernel is launched with an
      // unset argument (an unset cl_kernel arg is a launch failure, not a
      // silently-zero one).
      int ring_cap_i = (int)ring_cap;
      if (!kk->SetKernelArguments(15, &ring_cap_i, sizeof(int)))
        return false;
    }
    return true;
  };
  if (!bind_all(kp)) {
    // An argument-bind failure means the selected kernel's signature and this
    // binder disagree (e.g. a Block-Q variant that never declared argument 15).
    // The caller then drops to host attention with no other symptom than a
    // ~3x slower prefill, so say it once instead of failing silently.
    static bool _bind_fail_logged = false;
    if (!_bind_fail_logged) {
      _bind_fail_logged = true;
      ml_loge("flash prefill: kernel argument bind failed (xmx=%d blockq=%d "
              "d=%d) -- falling back to host attention",
              (int)use_xmx, (int)flash_blockq, (int)head_dim);
    }
    return false;
  }

  std::array<size_t, 1> gws;
  std::array<size_t, 1> lws;
  if (flash_coop || flash_vec || flash_blockq) {
    // Coop / vec: ONE workgroup per (head_q, query_row). Block-Q: ONE workgroup
    // per (head_q, row-tile of FBQ_TM rows) => TM x fewer groups. XMX: ONE
    // subgroup (LWS=16) per (head_q, 8-row tile). gws = groups * LWS, lws =
    // LWS (the reqd_work_group_size in the kernel). The kernel recomputes
    // n_row_tiles from its compile-time TM, so the host ceil below matches
    // exactly (no stray groups dispatched).
    size_t groups;
    if (use_xmx) {
      const size_t n_row_tiles =
        ((size_t)M + (size_t)xmx_tm - 1) / (size_t)xmx_tm;
      groups = (size_t)num_heads_Q * n_row_tiles;
    } else if (flash_blockq) {
      const size_t TM = (size_t)flash_blockq_tm;
      const size_t n_row_tiles = (M + TM - 1) / TM;
      groups = (size_t)num_heads_Q * n_row_tiles;
    } else {
      groups = (size_t)num_heads_Q * M;
    }
    const size_t L = use_xmx ? (size_t)(16 * xmx_nsg) : (size_t)flash_coop_lws;
    gws = {groups * L};
    lws = {L};
  } else {
    // Naive: one WI per (head_q, query_row). gws padded to LWS multiple.
    constexpr size_t LWS = 64;
    const size_t total = (size_t)num_heads_Q * M;
    const size_t gws_x = ((total + LWS - 1) / LWS) * LWS;
    gws = {gws_x};
    lws = {LWS};
  }
  static const bool _flash_trace = std::getenv("NNTR_FLASH_TRACE") != nullptr;
  if (_flash_trace) {
    std::fprintf(stderr,
                 "[FLASH-DISPATCH] d=%d LWS=%d VPL=%d SG=%d XMX=%d gws=%zu "
                 "groups\n",
                 (int)head_dim, flash_coop_lws, (int)head_dim / flash_coop_lws,
                 flash_blockq_sg_eff, (int)use_xmx, gws[0] / lws[0]);
  }
  // [FLASH_XMX_CHECK] Run the blockq reference on the SAME buffers first and
  // snapshot O, so the xmx result can be host-compared (first calls only).
  std::vector<uint16_t> _xmx_ref;
  static int _xmx_check_done = 0;
  if (kp_ref && _xmx_check_done < flash_xmx_check) {
    if (!bind_all(kp_ref))
      return false;
    const size_t TMr = (size_t)flash_blockq_tm;
    const size_t g_ref = (size_t)num_heads_Q * (((size_t)M + TMr - 1) / TMr);
    std::array<size_t, 1> gws_r{g_ref * (size_t)flash_coop_lws};
    std::array<size_t, 1> lws_r{(size_t)flash_coop_lws};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp_ref->GetKernel(), 1, gws_r.data(), lws_r.data(), 0, nullptr, nullptr);
    opencl::clFinish(q);
    _xmx_ref.assign(O_host, O_host + (size_t)M * (size_t)HD_Q);
  }
  // NNTR_FLASH_TIMER=1: true GPU time of THIS kernel (clFinish before and
  // after, so prior queued work is excluded and the kernel is drained).
  // Serializes the queue -- diagnostics only. Buckets: full (win==0) vs
  // sliding; totals printed at process exit.
  static const bool _flash_timer = std::getenv("NNTR_FLASH_TIMER") != nullptr;
  struct FlashTimerAgg {
    long long full_ns = 0, slide_ns = 0, full_calls = 0, slide_calls = 0;
    ~FlashTimerAgg() {
      if (full_calls + slide_calls)
        std::fprintf(stderr,
                     "[FLASH-TIMER] full-attn %.1f ms / %lld calls; sliding "
                     "%.1f ms / %lld calls\n",
                     full_ns / 1.0e6, full_calls, slide_ns / 1.0e6,
                     slide_calls);
    }
  };
  static FlashTimerAgg _ft_agg;
  std::chrono::steady_clock::time_point _ft_t0;
  if (_flash_timer) {
    opencl::clFinish(q);
    _ft_t0 = std::chrono::steady_clock::now();
  }
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (_flash_timer) {
    opencl::clFinish(q);
    const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - _ft_t0)
                           .count();
    if (win_i == 0) {
      _ft_agg.full_ns += ns;
      ++_ft_agg.full_calls;
    } else {
      _ft_agg.slide_ns += ns;
      ++_ft_agg.slide_calls;
    }
  }
  if (!_xmx_ref.empty()) {
    opencl::clFinish(q);
    auto _h2f = [](uint16_t hh) -> float {
      const uint32_t s = (uint32_t)(hh & 0x8000) << 16;
      uint32_t e = (hh >> 10) & 0x1F, mt = hh & 0x3FF, f;
      if (e == 0) {
        if (mt == 0)
          f = s;
        else {
          int ee = 127 - 15 + 1;
          while (!(mt & 0x400)) {
            mt <<= 1;
            ee--;
          }
          mt &= 0x3FF;
          f = s | ((uint32_t)ee << 23) | (mt << 13);
        }
      } else if (e == 0x1F)
        f = s | 0x7F800000u | (mt << 13);
      else
        f = s | ((e - 15 + 127) << 23) | (mt << 13);
      float r;
      std::memcpy(&r, &f, 4);
      return r;
    };
    double max_diff = 0.0;
    size_t over_tol = 0, arg = 0;
    const size_t total = (size_t)M * (size_t)HD_Q;
    for (size_t i = 0; i < total; ++i) {
      const double dd =
        std::fabs((double)_h2f(O_host[i]) - (double)_h2f(_xmx_ref[i]));
      if (dd > max_diff) {
        max_diff = dd;
        arg = i;
      }
      if (dd > 0.02)
        ++over_tol;
    }
    std::fprintf(stderr,
                 "[FLASH-XMX-CHECK] call=%d M=%u N_kv=%u maxdiff=%.6f at "
                 "(row=%zu col=%zu) over_tol=%zu/%zu\n",
                 _xmx_check_done, M, N_kv, max_diff, arg / HD_Q, arg % HD_Q,
                 over_tol, total);
    ++_xmx_check_done;
  }

  // No post-kernel drain: the flash output O is consumed by the next GPU op
  // (the o_proj FC), which the in-order queue already orders after this
  // kernel, and nothing reads O from the host here.
  return true;
}

// Flash-decoding (split-KV) for M=1 decode. Splits the KV axis into n_chunks so
// gws = num_heads_Q * n_chunks workgroups (vs num_heads_Q for blockq/coop_vec),
// restoring parallelism the single decode query otherwise starves. Pass 1
// (flash_decode_partial) writes per-(head,chunk) unnormalized partials; pass 2
// (flash_decode_reduce) combines them online-softmax per head into O. SVM
// Q/K/V/O
// + cl_mem partial buffers. Gemma4 only (no attn-logit softcap).
bool flash_decode_f16_cl(const uint16_t *Q_host, const uint16_t *K_host,
                         const uint16_t *V_host, uint16_t *O_host,
                         unsigned int N_kv, unsigned int num_heads_Q,
                         unsigned int num_heads_KV, unsigned int head_dim,
                         unsigned int max_seq_len, bool svm_inputs,
                         float attn_softcap, unsigned int local_window,
                         unsigned int ring_cap) {
  if (num_heads_Q == 0 || num_heads_KV == 0 || head_dim == 0 || N_kv == 0)
    return false;
  if (num_heads_Q % num_heads_KV != 0)
    return false;
  if (!svm_inputs)
    return false;
  if (attn_softcap > 0.0f)
    return false; // decode kernel has no softcap (Gemma2); Gemma4 only

  // LWS so VPL = head_dim / LWS in {1..8} (half2/4/8 vloads): d=256->32,
  // d=512->64, d=128->16.
  const int lws = (head_dim >= 512) ? 64 : ((head_dim > 128) ? 32 : 16);
  if ((int)head_dim % lws != 0 || (int)head_dim / lws > 8 ||
      (int)head_dim / lws < 1)
    return false;
  const int block_kv = 4; // FLASH_VEC_BLOCK_KV

  static const int chunk_kv = []() {
    const char *e = std::getenv("NNTR_FLASH_DEC_CHUNK");
    // 64 KV/chunk measured best (more chunks = more parallelism for the lone
    // decode query): Gemma4 long-ctx decode 7.15 (chunk 256) -> 7.74 (64).
    return (e && std::atoi(e) > 0) ? std::atoi(e) : 64;
  }();
  const int n_chunks = (int)((N_kv + chunk_kv - 1) / chunk_kv);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  const int HD_Q = (int)(num_heads_Q * head_dim);
  const int HD_KV = (int)(num_heads_KV * head_dim);
  const int k_stride = (int)max_seq_len; // >0 OHWI, 0 concat

  const std::string copts =
    "-DFLASH_VEC_LWS=" + std::to_string(lws) +
    " -DFLASH_VEC_BLOCK_KV=" + std::to_string(block_kv) +
    " -DFLASH_VEC_D=" + std::to_string((int)head_dim);

  static void *part_acc = nullptr, *part_ml = nullptr;
  static size_t pa_cap = 0, pm_cap = 0;
  const size_t pa_bytes =
    (size_t)num_heads_Q * n_chunks * head_dim * sizeof(float);
  const size_t pm_bytes = (size_t)num_heads_Q * n_chunks * 2 * sizeof(float);
  if (!ensure_cl_stage_buf(&part_acc, &pa_cap, pa_bytes) ||
      !ensure_cl_stage_buf(&part_ml, &pm_cap, pm_bytes))
    return false;
  cl_mem pa_clmem = (cl_mem)part_acc;
  cl_mem pm_clmem = (cl_mem)part_ml;

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    flash_attention_kernel, "flash_decode_partial", copts);
  ClContext::SharedPtrClKernel kr = blas_cc->registerClKernel(
    flash_attention_kernel, "flash_decode_reduce", copts);
  if (!kp || !kr)
    return false;

  // In-order queue (properties 0): the KV writes are already ordered before
  // the split-KV dispatch, so no drain is needed here either.

  const float scale = 1.0f / std::sqrt((float)head_dim);
  int Ni = (int)N_kv, di = (int)head_dim, hq = HD_Q, hkv = HD_KV;
  int gqa = (int)(num_heads_Q / num_heads_KV);
  int ks = k_stride;
  int win_i = (local_window > 0 && local_window < N_kv) ? (int)local_window : 0;
  int ck = chunk_kv, nc = n_chunks;

  if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_host)) ||
      !kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_host)) ||
      !kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(V_host)) ||
      !kp->SetKernelArguments(3, &pa_clmem, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(4, &pm_clmem, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(5, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(6, &di, sizeof(int)) ||
      !kp->SetKernelArguments(7, &hq, sizeof(int)) ||
      !kp->SetKernelArguments(8, &hkv, sizeof(int)) ||
      !kp->SetKernelArguments(9, &gqa, sizeof(int)) ||
      !kp->SetKernelArguments(10, &scale, sizeof(float)) ||
      !kp->SetKernelArguments(11, &ks, sizeof(int)) ||
      !kp->SetKernelArguments(12, &win_i, sizeof(int)) ||
      !kp->SetKernelArguments(13, &ck, sizeof(int)) ||
      !kp->SetKernelArguments(14, &nc, sizeof(int)))
    return false;
  {
    int ring_cap_i = (int)ring_cap; // [kv-window-ring] physical row = n % cap
    if (!kp->SetKernelArguments(15, &ring_cap_i, sizeof(int)))
      return false;
  }
  {
    std::array<size_t, 1> gws = {(size_t)num_heads_Q * (size_t)n_chunks *
                                 (size_t)lws};
    std::array<size_t, 1> lwsa = {(size_t)lws};
    blas_cc->command_queue_inst_.enqueueKernel(
      kp->GetKernel(), 1, gws.data(), lwsa.data(), 0, nullptr, nullptr);
  }
  if (!kr->SetKernelArguments(0, &pa_clmem, sizeof(cl_mem)) ||
      !kr->SetKernelArguments(1, &pm_clmem, sizeof(cl_mem)) ||
      !kr->SetKernelSVMArguments(2, O_host) ||
      !kr->SetKernelArguments(3, &di, sizeof(int)) ||
      !kr->SetKernelArguments(4, &hq, sizeof(int)) ||
      !kr->SetKernelArguments(5, &nc, sizeof(int)))
    return false;
  {
    std::array<size_t, 1> gws = {(size_t)num_heads_Q * (size_t)lws};
    std::array<size_t, 1> lwsa = {(size_t)lws};
    blas_cc->command_queue_inst_.enqueueKernel(
      kr->GetKernel(), 1, gws.data(), lwsa.data(), 0, nullptr, nullptr);
  }
  // The reduce kernel's output is consumed on the same in-order queue.
  return true;
}

} // namespace nntrainer
