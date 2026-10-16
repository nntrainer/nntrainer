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
#include <algorithm>
#include <array>
#include <atomic>
#include <basic_planner.h>
#include <blas_kernel_interface.h>
#include <blas_kernels.h> // v8c_use_buffer_path()
#include <chrono>
#include <cl_kernels/flash_attention.h>
#include <cl_kernels/rotary_emb.h>
#include <cl_kernels/two_conv_attention.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <env_compat.h> // nntr_env_on_or(), nntr_deterministic()
#include <map>
#include <memory>
#include <memory_pool.h>
#include <mutex>
#include <nntrainer_log.h>
#include <opencl_kernel.h>
#include <opencl_loader.h>
#include <tuple>
#include <vector>

// Default of NNTR_QK_IMG_RW (the K mirror bound __read_write in the image qk
// kernels; see qk_imgrw_program()). On: the sampled read is the path that
// served stale V texels, so K takes the coherent binding too, pre-emptively,
// for a small prefill cost (numbers at qk_img_rw_on()).
#ifndef QK_IMG_RW_DEFAULT
#define QK_IMG_RW_DEFAULT true
#endif

namespace nntrainer {
static ClContext::SharedPtrClKernel kvImgKernel(const char *kname);

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
  // NNTR_SCORES_PINGPONG=1: alternate two score scratches per
  // image-attention call (diagnostic arm: is a stale line of the ONE reused
  // scratch what the sv kernel reads?).
  cl_mem scores_pp = nullptr;
  size_t scores_pp_bytes = 0;
  int scores_pp_i = 0;
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
// Published by kimg_gsh_for() at layer finalize, read here when a program is
// registered. NOT cached: the prewarm pass can register these kernels before
// any layer has finalized, and a value frozen at 0 then would mismatch the
// mirrors created afterwards. Registration is rare; rebuilding this string is
// not on any hot path.
unsigned int g_kimg_gsh = 0;

static std::string tca_copts() {
  std::string o =
    v8c_use_buffer_path() ? std::string("-DTCA_BUFFER_ONLY") : std::string();
  const unsigned int gsh = g_kimg_gsh;
  if (gsh != 0) {
    if (!o.empty())
      o += " ";
    o += "-DKIMG_GSH=" + std::to_string(gsh);
  }
  // NNTR_SV_SCORES_CONST=0: compile the sv kernels' scores
  // argument without `const` (diagnostic arm).
  static const bool sv_scores_const = []() {
    const char *e = std::getenv("NNTR_SV_SCORES_CONST");
    return !(e != nullptr && e[0] == '0');
  }();
  if (!sv_scores_const) {
    if (!o.empty())
      o += " ";
    o += "-DSV_SCORES_QUAL=";
  }
  return o;
}

static bool tca_ensure(cl_context ctx, cl_mem *buf, size_t *cap, size_t bytes,
                       cl_mem_flags flags) {
  if (*buf && *cap >= bytes)
    return true;
  if (*buf) {
    opencl::clReleaseMemObjectT(*buf);
    *buf = nullptr;
    *cap = 0;
  }
  cl_int err = CL_SUCCESS;
  opencl::ClMemAcctScope _acct("scratch:attn");
  *buf = opencl::clCreateBufferT(ctx, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS || !*buf) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  // [kv-img-write, diagnostic] NNTR_ATTN_SCRATCH_ZERO=1 zeroes a freshly grown
  // scratch. The scores scratch grows with the key span, so a long chunked
  // prefill reallocates it mid-run and the new allocation holds whatever the
  // driver last left there -- run-to-run variable. The banded qk grid writes
  // only [n_lo, n_hi) of each score row while sv reads whole texels, so a lane
  // the band does not cover is read as found. If the residual long-context
  // nondeterminism is an uninitialized-scratch read, zeroing kills it.
  static const bool zero_on = []() {
    const char *e = std::getenv("NNTR_ATTN_SCRATCH_ZERO");
    return e != nullptr && e[0] != '0';
  }();
  if (zero_on) {
    auto *blas_cc =
      static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
    if (blas_cc != nullptr) {
      const uint16_t z = 0;
      opencl::clEnqueueFillBuffer(
        blas_cc->command_queue_inst_.GetCommandQueue(), *buf, &z,
        sizeof(uint16_t), 0, bytes, 0, nullptr, nullptr);
    }
  }
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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

  // The shared command queue is in order (CommandQueueManager creates it
  // with properties 0), so the kernels enqueued below already run after
  // the CL_FALSE writes above. This clFinish dates from when the queue was
  // out of order and is kept as a conservative sync point — measured
  // overhead ~0.1ms per prefill, negligible vs ~900ms total mha time.
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
/**
 * @brief Compile options for rope_inplace_kernel.
 * @details NNTR_ROPE_TAIL_FIX=0 restores the pre-fix rotation rule (precision
 * chosen from the chunk's M alone). Read once; the string is part of the
 * program cache key, so the two arms never share a compiled binary.
 * @return "" for the default, or the -D that selects the old rule
 */
/// [kv-img-write] Write the OHWI K/V mirrors through write_imageui into the
/// very image the attention reads, instead of writing the aliased buffer.
/// DEFAULT ON (mode 1); NNTR_KV_IMG_WRITE=0 is the opt-out back to the buffer
/// scatters. Declared in attention_kernels.h; the mirror creation (image flags)
/// and the layer (which scatter to call) both ask here so the two can never
/// disagree.
bool kv_img_write_enabled() { return kv_img_write_mode() != 0; }

/// [kv-img-write] Process-wide latch, cleared once if the driver refuses the
/// path: either clCreateImage(CL_MEM_READ_WRITE) over the mirror buffer, or the
/// -cl-std=CL2.0 image-write program. Cleared, every ask answers 0 and the
/// layer takes the buffer scatters -- exactly the pre-default behaviour --
/// instead of leaving the mirror unwritten, which is what a bare "the kernel is
/// missing" would have produced once the path became the default.
static std::atomic<bool> kv_img_write_latch{true};

void kv_img_write_disarm(const char *why) {
  if (kv_img_write_latch.exchange(false))
    ml_logw("[kv-img-write] KVIMG_MARKER_A disarmed (%s): the OHWI K/V "
            "mirror falls back to the buffer scatters for this process",
            why != nullptr ? why : "unspecified");
}

int kv_img_write_mode() {
  static const int mode = []() {
    const char *e = std::getenv("NNTR_KV_IMG_WRITE");
    return e != nullptr ? std::atoi(e) : 1; // default: mode 1
  }();
  return kv_img_write_latch.load(std::memory_order_relaxed) ? mode : 0;
}

static const std::string &ropeCopts();

/** build options of the int8-KV snapping variants of the rope kernels */
static const std::string &ropeSnapCopts() {
  static const std::string opts = ropeCopts() + " -DKV_SNAP";
  return opts;
}

static const std::string &ropeCopts() {
  static const std::string opts = [] {
    const char *e = std::getenv("NNTR_ROPE_TAIL_FIX");
    return (e != nullptr && e[0] == '0')
             ? std::string("-DROPE_TAIL_CONSISTENT=0")
             : std::string();
  }();
  return opts;
}

// ===========================================================================
// [kv-img-write] IMAGE-PATH mirror writes (NNTR_KV_IMG_WRITE=1).
//
// The OHWI mirror is WRITTEN by the scatters above as a buffer and READ by the
// attention as an image2d view over the same memory (kimg_read /
// read_imageui). Two paths over one allocation is what the driver cannot
// order: an image read can be served a texture-cache line a previous launch
// fetched, so a row THIS chunk just scattered comes back pre-write (per-node
// output hashes differ run to run at a random chunk; a clFinish around the
// attention does not close it, because the stale line is not a pending write
// but a cached read). These kernels write the SAME cl_mem the attention reads,
// through write_imageui, so the launch that produces a row and the launch that
// consumes it name the same image object and the driver's own dependency
// tracking has to cover the texture cache.
//
// Own program, so the default (and every non-Adreno runtime) keeps the
// rope/scatter program it always had, byte for byte, and the CL2.0 option
// below cannot change its codegen:
//  - KIMG_GSH is the folded K row mapping, published by kimg_gsh_for(). Not
//    cached, for the same reason tca_copts() is not: the prewarm pass can
//    register a program before any layer has finalized, and a 0 frozen then
//    would mismatch the mirrors created afterwards.
//  - -cl-std=CL2.0 is for the V scatter's __read_write image2d_t. A reversed-
//    OHWI texel is 8 SEQUENCE positions, so any step that does not start and
//    end on a multiple of 8 (every decode step; any prefill chunk whose length
//    is not a multiple of 8) covers partial texels whose other lanes hold rows
//    an earlier step wrote. write_image has no sub-texel granularity, so the
//    texel is read, the owned lanes replaced and the texel written back --
//    which keeps the write EXACTLY as wide as the buffer scatter's. (Rebuilding
//    those lanes from the concat V cache instead was measurably wrong: the
//    NNTR_MHA_CLMEM boundary gather writes the slab at ABSOLUTE rows with no
//    ring mapping, so on a ringed layer the slab rows past the ring cap are not
//    the rows the mirror holds -- 0.3B/2047 moved off its reference ids, and
//    NNTR_MHA_CLMEM=0 restored it.)
// ===========================================================================
static std::string kvImgCopts() {
  return "-cl-std=CL2.0 -DKIMG_GSH=" + std::to_string(g_kimg_gsh);
}

static const std::string kv_img_write_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef KIMG_GSH
#define KIMG_GSH 0
#endif
__constant sampler_t kimgw_smp =
  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
// K: one work item per (t, h, d-texel). A K texel is 8 halves along d and the
// scatter writes every channel of every row it touches, so every texel it
// covers is FULL -- no read-modify-write anywhere on the K side. Same source,
// same rows, same folded mapping as k_scatter_ohwi.
__kernel void k_scatter_ohwi_img(__global const half *src,
                                 __write_only image2d_t dst, const int M,
                                 const int hKV, const int d, const int max_S,
                                 const int position, const int src_off) {
  const int t = get_global_id(0);
  const int h = get_global_id(1);
  const int dt = get_global_id(2);
  const int dtex = d >> 3;
  if (t >= M || h >= hKV || dt >= dtex) return;
  const long so =
    (long)src_off + (long)t * hKV * d + (long)h * d + (long)(dt << 3);
  half tmp[8];
  for (int l = 0; l < 8; ++l) tmp[l] = src[so + l];
  const int lin_row = h * max_S + position + t;
  const int g = 1 << KIMG_GSH;
  write_imageui(dst,
                (int2)(((lin_row & (g - 1)) * dtex) + dt, lin_row >> KIMG_GSH),
                as_uint4(vload8(0, tmp)));
}
// V: one work item per (sequence texel, h, channel). The texels touched are
// [position>>3, (position+M+7)>>3); lanes inside [position, position+M) come
// from src exactly as v_scatter_ohwi_t reads it, every other lane is read back
// from the image and written unchanged -- so this writes precisely the elements
// the buffer scatter writes and nothing else.
__kernel void v_scatter_ohwi_t_img(__global const half *src,
                                   __read_write image2d_t dst, const int M,
                                   const int hKV, const int d, const int max_S,
                                   const int position, const int src_off) {
  const int j = get_global_id(0);
  const int h = get_global_id(1);
  const int x = get_global_id(2);
  const int tex0 = position >> 3;
  const int ntex = ((position + M + 7) >> 3) - tex0;
  if (j >= ntex || h >= hKV || x >= d) return;
  const int tex = tex0 + j;
  const int2 c = (int2)(tex, h * d + x);
  half tmp[8];
  vstore8(as_half8(read_imageui(dst, c)), 0, tmp);
  for (int l = 0; l < 8; ++l) {
    const int p = (tex << 3) + l;
    if (p >= position && p < position + M)
      tmp[l] = src[(long)src_off + (long)(p - position) * hKV * d +
                   (long)h * d + x];
  }
  write_imageui(dst, c, as_uint4(vload8(0, tmp)));
}
// Decode-step K and V image scatters in ONE launch (M == 1): lanes
// [0, d/8) of get_global_id(2) are k_scatter_ohwi_img's (head, d-texel) work
// items, lanes [d/8, d/8 + d) are v_scatter_ohwi_t_img's (head, channel) work
// items for the single texel a one-row step touches. Same reads, same texel
// coordinates, same stores as the two kernels above.
__kernel void kv_scatter_ohwi_img(__global const half *k_src,
                                  __write_only image2d_t k_dst,
                                  __global const half *v_src,
                                  __read_write image2d_t v_dst, const int hKV,
                                  const int d, const int k_max_S,
                                  const int v_max_S, const int position,
                                  const int k_src_off, const int v_src_off) {
  const int h = get_global_id(1);
  const int z = get_global_id(2);
  const int dtex = d >> 3;
  if (h >= hKV) return;
  if (z < dtex) {
    const int dt = z;
    const long so = (long)k_src_off + (long)h * d + (long)(dt << 3);
    half tmp[8];
    for (int l = 0; l < 8; ++l) tmp[l] = k_src[so + l];
    const int lin_row = h * k_max_S + position;
    const int g = 1 << KIMG_GSH;
    write_imageui(k_dst,
                  (int2)(((lin_row & (g - 1)) * dtex) + dt, lin_row >> KIMG_GSH),
                  as_uint4(vload8(0, tmp)));
  } else if (z < dtex + d) {
    const int x = z - dtex;
    const int tex = position >> 3;
    const int2 c = (int2)(tex, h * d + x);
    half tmp[8];
    vstore8(as_half8(read_imageui(v_dst, c)), 0, tmp);
    for (int l = 0; l < 8; ++l) {
      const int p = (tex << 3) + l;
      if (p >= position && p < position + 1)
        tmp[l] = v_src[(long)v_src_off + (long)(p - position) * hKV * d +
                       (long)h * d + x];
    }
    write_imageui(v_dst, c, as_uint4(vload8(0, tmp)));
  }
}
// V, mode 2: the same rows, with NO image read anywhere in the write path. Mode
// 1's read-modify-write is itself an image read, and an image read is what
// pollutes the texture cache on this driver (see the gather), so at long spans
// mode 1 may be re-introducing the very hazard it closes. Here the partial
// texels are rebuilt from the concat V cache instead: lane p of texel `tex` is
// absolute position base_abs + (tex<<3) + p, mapped through the ring exactly as
// cacheRow() does, and +0.0h at or past the frontier (`to_abs`) -- what the
// one-time mirror fill left there. This needs the slab to BE complete, which it
// is with the side-fills on (NNTR_KV_STAGE off, the default) AND with the
// NNTR_MHA_CLMEM boundary gather skipped: that gather writes the slab at
// ABSOLUTE rows with no ring mapping, so on a ringed layer whose mirror is live
// it fills rows past the ring cap with the wrong content (and past the
// allocation) -- harmless while nothing reads them back, fatal here.
__kernel void v_scatter_ohwi_t_img_slab(__global const half *vcache,
                                        __write_only image2d_t dst,
                                        const int hKV, const int d,
                                        const int tex0, const int ntex,
                                        const int base_abs, const int ring_cap,
                                        const int to_abs) {
  const int j = get_global_id(0);
  const int h = get_global_id(1);
  const int x = get_global_id(2);
  if (j >= ntex || h >= hKV || x >= d) return;
  const int tex = tex0 + j;
  const long hd = (long)hKV * d;
  half tmp[8];
  for (int l = 0; l < 8; ++l) {
    const int a = base_abs + (tex << 3) + l;
    half v = (half)0.0h;
    if (a >= 0 && a < to_abs) {
      const int phys = ring_cap > 0 ? (a % ring_cap) : a;
      v = vcache[(long)phys * hd + (long)h * d + x];
    }
    tmp[l] = v;
  }
  write_imageui(dst, (int2)(tex, h * d + x), as_uint4(vload8(0, tmp)));
}
// Inverse gathers on the image path: with the writes on the image, a BUFFER
// read of the mirror is the same aliasing hazard reversed (the boundary sync
// would hand the host slab a pre-write row), so gather from the image too.
__kernel void mirror_fill_img(__write_only image2d_t dst, const int w,
                              const int h, const uint v) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if (x >= w || y >= h) return;
  write_imageui(dst, (int2)(x, y), (uint4)(v, v, v, v));
}
__kernel void mirror_touch_img(__write_only image2d_t dst, const int x,
                               const int y) {
  write_imageui(dst, (int2)(x, y), (uint4)(0u, 0u, 0u, 0u));
}
__kernel void k_gather_ohwi_img(__read_only image2d_t src, __global half *dst,
                                const int M, const int hKV, const int d,
                                const int max_S, const int position) {
  const int t = get_global_id(0);
  const int h = get_global_id(1);
  const int dt = get_global_id(2);
  const int dtex = d >> 3;
  if (t >= M || h >= hKV || dt >= dtex) return;
  const int lin_row = h * max_S + position + t;
  const int g = 1 << KIMG_GSH;
  const uint4 vv = read_imageui(
    src, kimgw_smp,
    (int2)(((lin_row & (g - 1)) * dtex) + dt, lin_row >> KIMG_GSH));
  half tmp[8];
  vstore8(as_half8(vv), 0, tmp);
  const long dof = (long)t * hKV * d + (long)h * d + (long)(dt << 3);
  for (int l = 0; l < 8; ++l) dst[dof + l] = tmp[l];
}
__kernel void v_gather_ohwi_t_img(__read_only image2d_t src, __global half *dst,
                                  const int M, const int hKV, const int d,
                                  const int max_S, const int position) {
  const int t = get_global_id(0);
  const int h = get_global_id(1);
  const int x = get_global_id(2);
  if (t >= M || h >= hKV || x >= d) return;
  const int p = position + t;
  const uint4 vv = read_imageui(src, kimgw_smp, (int2)(p >> 3, h * d + x));
  half tmp[8];
  vstore8(as_half8(vv), 0, tmp);
  dst[(long)t * hKV * d + (long)h * d + x] = tmp[p & 7];
}
)CL";

static const std::string rope_inplace_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef KV_SNAP
// x / s rounded to nearest-even exactly like an IEEE division, built from
// fma only (Markstein): the reciprocal is refined to the correctly rounded
// 1/s, the quotient gets one residual correction. The snap kernels use it
// instead of '/' so they need no -cl-fp32-correctly-rounded-divide-sqrt
// (rejected by some drivers) and still match the CPU snap kernel bit for bit.
inline float kv_snap_div(const float x, const float s) {
  float y = 1.0f / s;
  float e = fma(-s, y, 1.0f);
  y = fma(e, y, y);
  e = fma(-s, y, 1.0f);
  y = fma(e, y, y);
  const float q = x * y;
  const float r = fma(-s, q, x);
  // an exact quotient keeps its own sign (-0 / s is -0)
  return r == 0.0f ? q : fma(r, y, q);
}
#endif
// 1 (default) = the fp32/fp16 rotation choice is a property of the SEQUENCE,
// so a continuation chunk matches the chunks before it; 0 = the old rule,
// chosen from this chunk's M alone. Set from NNTR_ROPE_TAIL_FIX; see the
// branch below.
#ifndef ROPE_TAIL_CONSISTENT
#define ROPE_TAIL_CONSISTENT 1
#endif
__kernel void rope_inplace_f16(__global const half *in,
                               __global       half *out,
                               __global const half *cos_lut,
                               __global const half *sin_lut,
                               const int M, const int num_heads,
                               const int half_d, const int start_pos,
                               const int write_off
#ifdef KV_SNAP
                               , __global const float *snap_scale
#endif
                               ) {
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
  //
  // ...but M is the CHUNK's row count, not the sequence's, and a chunked
  // prefill is one sequence. Choosing the precision from M alone rotates the
  // LAST chunk of a long prompt in fp32 while every earlier chunk was rotated
  // in fp16, whenever that chunk happens to be shorter than 32 rows -- so the
  // K cache the decode steps then read is internally inconsistent across a
  // boundary a few rows before the first generated token. Measured
  // on a 3 595-token prompt, tails of 4, 11 and 18
  // rows all answer wrongly and tails of 32, 395, 445 and 523 are golden --
  // the threshold in the data is this 32, exactly.
  //
  // rope_tail_consistent (NNTR_ROPE_TAIL_FIX=0 restores the old rule) makes
  // the choice a property of the SEQUENCE instead: fp32 for decode (M == 1,
  // unchanged) and for a genuinely short prompt (a first chunk, start_pos == 0,
  // unchanged), fp16 for a CONTINUATION chunk whatever its size -- which is
  // what every earlier chunk of that same prefill used. Large prefill chunks
  // are unchanged either way.
#ifdef KV_SNAP
  // int8-KV trained model: the rotated key is snapped to its head's int8
  // grid, q = clamp(rint(x / s), -128, 127), x = q * s, in the same store
  // (fp32 rotation, true division, round half to even -- the CPU kernel's
  // rule).
  {
    float cf = (float)c, sf = (float)s, lof = (float)lo, hif = (float)hi;
    const float sc = snap_scale[h];
    out[write_off + row + k] =
      (half)(clamp(rint(kv_snap_div(lof * cf - hif * sf, sc)), -128.0f, 127.0f) * sc);
    out[write_off + row + k + half_d] =
      (half)(clamp(rint(kv_snap_div(hif * cf + lof * sf, sc)), -128.0f, 127.0f) * sc);
    return;
  }
#endif
  if (M == 1 || (M < 32 && (ROPE_TAIL_CONSISTENT == 0 || start_pos == 0))) {
    float cf = (float)c, sf = (float)s, lof = (float)lo, hif = (float)hi;
    out[write_off + row + k]          = (half)(lof * cf - hif * sf);
    out[write_off + row + k + half_d] = (half)(hif * cf + lof * sf);
  } else {
    out[write_off + row + k]          = lo * c - hi * s;
    out[write_off + row + k + half_d] = hi * c + lo * s;
  }
}
#ifdef KV_SNAP
// int8-KV trained model: a value row is snapped to its kv head's int8 grid as
// it is stored, x = clamp(rint(x / s), -128, 127) * s (true division, round
// half to even), the rule of the CPU snap kernel. Row = row_len values of
// head_dim each.
inline half kv_snap_v(half x, const int i, const int row_len,
                      const int head_dim, __global const float *snap_scale) {
  const float sc = snap_scale[(i % row_len) / head_dim];
  return (half)(clamp(rint(kv_snap_div((float)x, sc)), -128.0f, 127.0f) * sc);
}
#endif
__kernel void scatter_copy_f16(__global const half *in, __global half *out,
                               const int N
#ifdef KV_SNAP
                               , const int row_len, const int head_dim,
                               __global const float *snap_scale
#endif
                               ) {
  int i = get_global_id(0);
#ifdef KV_SNAP
  if (i < N) out[i] = kv_snap_v(in[i], i, row_len, head_dim, snap_scale);
#else
  if (i < N) out[i] = in[i];
#endif
}
// Row-offset variant: writes into a STABLE base at out[write_off + i] instead of
// baking the destination offset into the pointer. write_off (= cache_index *
// num_heads_KV * head_dim) is a scalar arg, so a recorded decode KV-write can be
// replayed with write_off overridden per token (cl_qcom_recordable_queues can
// override scalar args but NOT SVM pointers). Same address as the offset-baked
// pointer form -> byte-identical output.
__kernel void scatter_copy_f16_row(__global const half *in, __global half *out,
                                   const int N, const int write_off
#ifdef KV_SNAP
                                   , const int row_len, const int head_dim,
                                   __global const float *snap_scale
#endif
                                   ) {
  int i = get_global_id(0);
#ifdef KV_SNAP
  if (i < N)
    out[write_off + i] = kv_snap_v(in[i], i, row_len, head_dim, snap_scale);
#else
  if (i < N) out[write_off + i] = in[i];
#endif
}
// Decode-step Q/K/V commit in ONE launch: rope_inplace_f16 over Q (in place),
// rope_inplace_f16 over K into the key cache at k_write_off, and the flat V
// copy into the value cache at v_write_off. One work item per (head lane,
// half_d), heads laid out along get_global_id(1) as [Q heads | K heads | V
// heads]; each lane runs the M == 1 (fp32) rotation of rope_inplace_f16 or the
// scatter_copy_f16_row store verbatim, so the bytes written are the same as
// the three launches this replaces.
__kernel void qkv_rope_commit_f16(__global const half *q_in,
                                  __global       half *q_out,
                                  __global const half *k_in,
                                  __global       half *k_out,
                                  __global const half *v_in,
                                  __global       half *v_out,
                                  __global const half *cos_lut,
                                  __global const half *sin_lut,
                                  const int num_heads_q, const int num_heads_kv,
                                  const int half_d, const int start_pos,
                                  const int k_write_off, const int v_write_off
#ifdef KV_SNAP
                                  , __global const float *snap_scale,
                                  __global const float *v_snap_scale
#endif
                                  ) {
  const int h = get_global_id(1);
  const int k = get_global_id(2);
  if (k >= half_d) return;
  const int d = 2 * half_d;
  if (h < num_heads_q + num_heads_kv) {
    const int hh = h < num_heads_q ? h : h - num_heads_q;
    __global const half *in = h < num_heads_q ? q_in : k_in;
    __global half *out = h < num_heads_q ? q_out : k_out;
    const int write_off = h < num_heads_q ? 0 : k_write_off;
    long row = (long)hh * d;
    long lut = (long)start_pos * half_d + k;
    half c = cos_lut[lut];
    half s = sin_lut[lut];
    half lo = in[row + k];
    half hi = in[row + k + half_d];
    float cf = (float)c, sf = (float)s, lof = (float)lo, hif = (float)hi;
    float r0 = lof * cf - hif * sf, r1 = hif * cf + lof * sf;
#ifdef KV_SNAP
    if (h >= num_heads_q) { // a key head: snap to its int8 grid (see above)
      const float sc = snap_scale[hh];
      r0 = clamp(rint(kv_snap_div(r0, sc)), -128.0f, 127.0f) * sc;
      r1 = clamp(rint(kv_snap_div(r1, sc)), -128.0f, 127.0f) * sc;
    }
#endif
    out[write_off + row + k]          = (half)r0;
    out[write_off + row + k + half_d] = (half)r1;
  } else {
    const int hv = h - num_heads_q - num_heads_kv;
    if (hv >= num_heads_kv) return;
    const long row = (long)hv * d;
#ifdef KV_SNAP
    // a value head: snapped to its int8 grid as it is stored
    const float vs = v_snap_scale[hv];
    v_out[v_write_off + row + k] =
      (half)(clamp(rint(kv_snap_div((float)v_in[row + k], vs)), -128.0f, 127.0f) * vs);
    v_out[v_write_off + row + k + half_d] = (half)(
      clamp(rint(kv_snap_div((float)v_in[row + k + half_d], vs)), -128.0f, 127.0f) * vs);
#else
    v_out[v_write_off + row + k] = v_in[row + k];
    v_out[v_write_off + row + k + half_d] = v_in[row + k + half_d];
#endif
  }
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
                         unsigned int write_off, void *snap_scales) {
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();

  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rope_inplace_kernel, "rope_inplace_f16",
    snap_scales != nullptr ? ropeSnapCopts() : ropeCopts());
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
  if (snap_scales != nullptr) {
    cl_mem sh = static_cast<cl_mem>(snap_scales);
    if (!kp->SetKernelArguments(9, &sh, sizeof(cl_mem)))
      return false;
  }

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
    if (out_clmem == nullptr && drain_svm_out) {
      opencl::clFinish(q);
      opencl::Kernel::clearUndrainedSvmPlanes();
    } else if (out_clmem == nullptr) {
      opencl::clFlush(q);
      // The rotated bytes are in a shared plane that only a flush stands
      // behind. Declare the plane so the first consumer that binds into it
      // drains, instead of a per-pack flag deciding for the whole model.
      // write_off is a kernel scalar, so the plane starts at the bound base and
      // runs past the rotated rows.
      opencl::Kernel::noteUndrainedSvmPlane(
        out, (size_t)write_off * sizeof(uint16_t) + io_bytes);
    } else if (!rope_noflush) {
      opencl::clFlush(q);
    }
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
  cc.registerClKernel(rope_inplace_kernel, "rope_inplace_f16", ropeCopts());
}

namespace {

/**
 * @brief FP16 planes for the FP32-activation attention operands, one pool per
 *        operand slot so a plane is never shared with another tensor.
 */
struct ActStage {
  std::unique_ptr<MemoryPool> pool[4];
  std::shared_ptr<MemoryData> mem[4];
  size_t cap[4] = {0, 0, 0, 0};
};

ActStage &act_stage() {
  static ActStage s;
  return s;
}

const std::string act_cast_kernel = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// x / s rounded to nearest-even exactly like an IEEE division, built from
// fma only (Markstein): the reciprocal is refined to the correctly rounded
// 1/s, the quotient gets one residual correction. The snap kernels use it
// instead of '/' so they need no -cl-fp32-correctly-rounded-divide-sqrt
// (rejected by some drivers) and still match the CPU snap kernel bit for bit.
inline float kv_snap_div(const float x, const float s) {
  float y = 1.0f / s;
  float e = fma(-s, y, 1.0f);
  y = fma(e, y, y);
  e = fma(-s, y, 1.0f);
  y = fma(e, y, y);
  const float q = x * y;
  const float r = fma(-s, q, x);
  // an exact quotient keeps its own sign (-0 / s is -0)
  return r == 0.0f ? q : fma(r, y, q);
}
__kernel void cast_f32_to_f16(__global const float *in, __global half *out,
                              const int N) {
  const int i = get_global_id(0);
  if (i < N)
    vstore_half_rte(in[i], i, out);
}
__kernel void cast_f16_to_f32(__global const half *in, __global float *out,
                              const int N) {
  const int i = get_global_id(0);
  if (i < N)
    out[i] = vload_half(i, in);
}
// The narrowing of an int8-KV trained model's value rows: each value is
// snapped to its kv head's int8 grid before the fp16 store,
// x = clamp(rint(x / s), -128, 127) * s (true division, round half to even,
// as the CPU kernel). Row = num_heads * head_dim values.
__kernel void cast_f32_to_f16_snap(__global const float *in,
                                   __global half *out, const int N,
                                   const int head_dim, const int num_heads,
                                   __global const float *snap_scale) {
  const int i = get_global_id(0);
  if (i < N) {
    const int h = (i % (num_heads * head_dim)) / head_dim;
    const float sc = snap_scale[h];
    vstore_half_rte(clamp(rint(kv_snap_div(in[i], sc)), -128.0f, 127.0f) * sc, i, out);
  }
}
)CL";

/**
 * @brief Run one cast kernel. @a act is the FP32 activation operand (the
 *        input of a narrowing, the output of a widening) and @a act_bytes its
 *        size.
 * @details The activation follows the shared-memory protocol every SVM op in
 * this backend keeps: the op that dispatches on it unmaps it first and maps it
 * back (asynchronously) after. Its producer -- a projection or a norm -- left
 * it mapped for the host, and on a coarse-grain device a kernel that reads a
 * mapped region reads stale bytes. The FP16 plane is device-only and is not
 * mapped at rest.
 */
bool act_cast_dispatch(const char *name, const void *in, void *out,
                       unsigned int N, void *act, size_t act_bytes) {
  if (N == 0 || in == nullptr || out == nullptr)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc == nullptr)
    return false;
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(act_cast_kernel, name);
  if (!kp)
    return false;
  int Ni = static_cast<int>(N);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(act);
  if (!kp->SetKernelSVMArguments(0, in) || !kp->SetKernelSVMArguments(1, out) ||
      !kp->SetKernelArguments(2, &Ni, sizeof(int)))
    return false;
  constexpr size_t LWS = 64;
  std::array<size_t, 3> gws = {(((size_t)N + LWS - 1) / LWS) * LWS, 1, 1};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  blas_cc->command_queue_inst_.enqueueSVMMap(act, act_bytes,
                                             /*read_only=*/true,
                                             /*event=*/nullptr,
                                             /*async=*/true);
  return true;
}

} // namespace

std::shared_ptr<MemoryData> fp16_act_stage_svm(unsigned int slot,
                                               size_t bytes) {
  if (slot >= 4 || bytes == 0)
    return nullptr;
  ActStage &s = act_stage();
  if (s.mem[slot] && s.cap[slot] >= bytes)
    return s.mem[slot];
  auto *cc = Engine::Global().getRegisteredContext("gpu");
  if (cc == nullptr)
    return nullptr;
  // The gpu context's own allocator, through a memory pool: the pool is what
  // marks the plane as shared virtual memory for the device arms that check.
  std::shared_ptr<MemAllocator> alloc = cc->getMemAllocator();
  if (!alloc || !alloc->isSVM())
    return nullptr;
  if (s.mem[slot]) {
    // Growing: the in-order queue may still hold work on the old plane.
    auto *blas_cc = static_cast<ClContext *>(cc);
    opencl::clFinish(blas_cc->command_queue_inst_.GetCommandQueue());
    s.mem[slot].reset();
    s.pool[slot].reset();
    s.cap[slot] = 0;
  }
  opencl::ClMemAcctScope _acct("scratch:attn");
  auto pool = std::make_unique<MemoryPool>(alloc);
  const unsigned int idx = pool->requestMemory(bytes, 0, 1);
  pool->planLayout(BasicPlanner());
  pool->allocate();
  std::shared_ptr<MemoryData> mem = pool->getMemory(idx);
  if (!mem || !mem->isSVM() || mem->getAddr() == nullptr)
    return nullptr;
  s.pool[slot] = std::move(pool);
  s.mem[slot] = mem;
  s.cap[slot] = bytes;
  return mem;
}

bool gpu_cast_f32_to_f16_cl(const float *in, uint16_t *out, unsigned int N) {
  return act_cast_dispatch("cast_f32_to_f16", in, out, N,
                           const_cast<float *>(in), (size_t)N * sizeof(float));
}

bool gpu_cast_f32_to_f16_snap_cl(const float *in, uint16_t *out, unsigned int N,
                                 unsigned int head_dim, unsigned int num_heads,
                                 void *snap_scales) {
  if (N == 0 || in == nullptr || out == nullptr || snap_scales == nullptr ||
      head_dim == 0 || num_heads == 0)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc == nullptr)
    return false;
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(act_cast_kernel, "cast_f32_to_f16_snap");
  if (!kp)
    return false;
  int Ni = static_cast<int>(N), hd = static_cast<int>(head_dim),
      nh = static_cast<int>(num_heads);
  cl_mem sh = static_cast<cl_mem>(snap_scales);
  float *act = const_cast<float *>(in);
  blas_cc->command_queue_inst_.enqueueSVMUnmap(act);
  if (!kp->SetKernelSVMArguments(0, in) || !kp->SetKernelSVMArguments(1, out) ||
      !kp->SetKernelArguments(2, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(3, &hd, sizeof(int)) ||
      !kp->SetKernelArguments(4, &nh, sizeof(int)) ||
      !kp->SetKernelArguments(5, &sh, sizeof(cl_mem)))
    return false;
  constexpr size_t LWS = 64;
  std::array<size_t, 3> gws = {(((size_t)N + LWS - 1) / LWS) * LWS, 1, 1};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  blas_cc->command_queue_inst_.enqueueSVMMap(act, (size_t)N * sizeof(float),
                                             /*read_only=*/true,
                                             /*event=*/nullptr,
                                             /*async=*/true);
  return true;
}

void *kv_snap_scales_cl(const std::vector<float> &scales) {
  if (scales.empty())
    return nullptr;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc == nullptr)
    return nullptr;
  cl_int err = CL_SUCCESS;
  cl_mem b = opencl::clCreateBufferT(
    blas_cc->context_inst_.GetContextNoRetain(),
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales.size() * sizeof(float),
    const_cast<float *>(scales.data()), &err);
  return err == CL_SUCCESS ? b : nullptr;
}

bool gpu_cast_f16_to_f32_cl(const uint16_t *in, float *out, unsigned int N) {
  return act_cast_dispatch("cast_f16_to_f32", in, out, N, out,
                           (size_t)N * sizeof(float));
}

bool ensure_cl_stage_buf(void **buf, size_t *cap, size_t bytes) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  return tca_ensure(ctx, reinterpret_cast<cl_mem *>(buf), cap, bytes,
                    CL_MEM_READ_WRITE);
}

bool qkv_rope_commit_f16_cl(
  const uint16_t *q, void *q_clmem, const uint16_t *k, void *k_clmem,
  uint16_t *k_dst, unsigned int k_write_off, const uint16_t *v, void *v_clmem,
  uint16_t *v_dst, unsigned int v_write_off, const uint16_t *cos_lut,
  const uint16_t *sin_lut, unsigned int num_heads_q, unsigned int num_heads_kv,
  unsigned int head_dim, unsigned int start_pos, unsigned int max_positions,
  bool flush, void *snap_scales, void *v_snap_scales) {
  if (num_heads_q == 0 || num_heads_kv == 0 || head_dim == 0 || (head_dim & 1u))
    return false;
  // The snapping variant snaps both K and V; a model that snaps only one of
  // them takes the separate launches.
  if ((snap_scales == nullptr) != (v_snap_scales == nullptr))
    return false;
  if (q == nullptr || k == nullptr || v == nullptr || k_dst == nullptr ||
      v_dst == nullptr || cos_lut == nullptr || sin_lut == nullptr)
    return false;
  if (start_pos + 1 > max_positions)
    return false;
  const int half_d = (int)(head_dim / 2);

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  cl_command_queue q_ = blas_cc->command_queue_inst_.GetCommandQueue();
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rope_inplace_kernel, "qkv_rope_commit_f16",
    snap_scales != nullptr ? ropeSnapCopts() : ropeCopts());
  if (!kp)
    return false;

  // Same resident LUT slot as rope_inplace_f16_cl: uploaded once per
  // (cos, sin, half_d), drained only on the upload.
  const size_t lut_bytes = (size_t)max_positions * half_d * sizeof(uint16_t);
  RopeScratch &sc = rope_scratch();
  RopeScratch::LutSlot &slot =
    sc.lut_slots[RopeScratch::LutKey{cos_lut, sin_lut, half_d}];
  if (!tca_ensure(ctx, &slot.cos, &slot.cos_bytes, lut_bytes,
                  CL_MEM_READ_ONLY) ||
      !tca_ensure(ctx, &slot.sin, &slot.sin_bytes, lut_bytes, CL_MEM_READ_ONLY))
    return false;
  if (!slot.uploaded || slot.positions < max_positions) {
    if (opencl::clEnqueueWriteBuffer(q_, slot.cos, CL_FALSE, 0, lut_bytes,
                                     cos_lut, 0, nullptr,
                                     nullptr) != CL_SUCCESS ||
        opencl::clEnqueueWriteBuffer(q_, slot.sin, CL_FALSE, 0, lut_bytes,
                                     sin_lut, 0, nullptr,
                                     nullptr) != CL_SUCCESS)
      return false;
    slot.positions = max_positions;
    slot.uploaded = true;
    opencl::clFinish(q_);
  }

  // Each source binds its own plane (the FC's device sub-buffer when given,
  // the SVM pointer otherwise); the destinations are the SVM cache bases with
  // the row offsets as kernel scalars, exactly as the three separate launches
  // bind them.
  auto bind_in = [&](cl_uint idx, const uint16_t *p, void *clmem) {
    if (clmem != nullptr) {
      cl_mem h = static_cast<cl_mem>(clmem);
      return kp->SetKernelArguments(idx, &h, sizeof(cl_mem));
    }
    return kp->SetKernelSVMArguments(idx, const_cast<uint16_t *>(p));
  };
  bool okb = bind_in(0, q, q_clmem) &&
             kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(q)) &&
             bind_in(2, k, k_clmem) && kp->SetKernelSVMArguments(3, k_dst) &&
             bind_in(4, v, v_clmem) && kp->SetKernelSVMArguments(5, v_dst) &&
             kp->SetKernelArguments(6, &slot.cos, sizeof(cl_mem)) &&
             kp->SetKernelArguments(7, &slot.sin, sizeof(cl_mem));
  int nq = (int)num_heads_q, nkv = (int)num_heads_kv, hd = half_d,
      sp = (int)start_pos, kwo = (int)k_write_off, vwo = (int)v_write_off;
  okb = okb && kp->SetKernelArguments(8, &nq, sizeof(int)) &&
        kp->SetKernelArguments(9, &nkv, sizeof(int)) &&
        kp->SetKernelArguments(10, &hd, sizeof(int)) &&
        kp->SetKernelArguments(11, &sp, sizeof(int)) &&
        kp->SetKernelArguments(12, &kwo, sizeof(int)) &&
        kp->SetKernelArguments(13, &vwo, sizeof(int));
  if (okb && snap_scales != nullptr) {
    cl_mem sh = static_cast<cl_mem>(snap_scales);
    cl_mem vh = static_cast<cl_mem>(v_snap_scales);
    okb = kp->SetKernelArguments(14, &sh, sizeof(cl_mem)) &&
          kp->SetKernelArguments(15, &vh, sizeof(cl_mem));
  }
  if (!okb)
    return false;

  constexpr size_t LWS_K = 64;
  const size_t kx_pad = (((size_t)half_d + LWS_K - 1) / LWS_K) * LWS_K;
  std::array<size_t, 3> gws = {1, (size_t)num_heads_q + 2 * num_heads_kv,
                               kx_pad};
  std::array<size_t, 3> lws = {1, 1, LWS_K};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  // The rotated K and the V row are undrained shared-plane writes consumed by
  // the same-queue scatters; declare them for the (opt-in) hazard check, as
  // the separate launches do.
  const size_t row_bytes = (size_t)num_heads_kv * head_dim * sizeof(uint16_t);
  opencl::Kernel::noteUndrainedSvmPlane(
    k_dst, (size_t)k_write_off * sizeof(uint16_t) + row_bytes);
  opencl::Kernel::noteUndrainedSvmPlane(
    v_dst, (size_t)v_write_off * sizeof(uint16_t) + row_bytes);
  if (flush)
    opencl::clFlush(q_);
  return true;
}

bool gpu_copy_f16_cl(const uint16_t *in, uint16_t *out, unsigned int N,
                     bool svm_inputs, void *in_clmem, void *out_clmem,
                     bool drain, void *snap_scales, unsigned int row_len,
                     unsigned int head_dim) {
  if (N == 0 || in == nullptr || out == nullptr)
    return false;
  const bool snap = snap_scales != nullptr;
  if (snap && (!svm_inputs || row_len == 0 || head_dim == 0))
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "scatter_copy_f16",
                              snap ? ropeSnapCopts() : ropeCopts());
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
  if (snap) {
    const int rl = (int)row_len, hd = (int)head_dim;
    cl_mem sh = static_cast<cl_mem>(snap_scales);
    if (!kp->SetKernelArguments(3, &rl, sizeof(int)) ||
        !kp->SetKernelArguments(4, &hd, sizeof(int)) ||
        !kp->SetKernelArguments(5, &sh, sizeof(cl_mem)))
      return false;
  }
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
    if (out_clmem == nullptr && drain) {
      opencl::clFinish(q);
      opencl::Kernel::clearUndrainedSvmPlanes();
    } else if (out_clmem == nullptr) {
      opencl::clFlush(q);
      // Undrained shared-memory write: name the plane so its first consumer
      // drains.
      opencl::Kernel::noteUndrainedSvmPlane(out, bytes);
    } else {
      opencl::clFlush(q);
    }
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
                         void *out_base_clmem, bool drain, void *snap_scales,
                         unsigned int row_len, unsigned int head_dim) {
  if (N == 0 || in == nullptr || out_base == nullptr || !svm_inputs)
    return false;
  const bool snap = snap_scales != nullptr;
  if (snap && (row_len == 0 || head_dim == 0))
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(rope_inplace_kernel, "scatter_copy_f16_row",
                              snap ? ropeSnapCopts() : ropeCopts());
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
  if (okb && snap) {
    const int rl = (int)row_len, hd = (int)head_dim;
    cl_mem sh = static_cast<cl_mem>(snap_scales);
    okb = kp->SetKernelArguments(4, &rl, sizeof(int)) &&
          kp->SetKernelArguments(5, &hd, sizeof(int)) &&
          kp->SetKernelArguments(6, &sh, sizeof(cl_mem));
  }
  if (!okb)
    return false;
  constexpr size_t LWS = 64;
  const size_t gws_x = (((size_t)N + LWS - 1) / LWS) * LWS;
  std::array<size_t, 3> gws = {gws_x, 1, 1};
  std::array<size_t, 3> lws = {LWS, 1, 1};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (out_base_clmem == nullptr && drain) {
    opencl::clFinish(q);
    opencl::Kernel::clearUndrainedSvmPlanes();
  } else {
    opencl::clFlush(q);
    // Stable-base form of the same undrained shared write: the plane starts at
    // the bound base and runs past the written rows.
    if (out_base_clmem == nullptr) {
      opencl::Kernel::noteUndrainedSvmPlane(out_base, ((size_t)write_off + N) *
                                                        sizeof(uint16_t));
    }
  }
  return true;
}

unsigned int kimg_gsh_for(unsigned int num_heads_KV, unsigned int max_S) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc || num_heads_KV == 0 || max_S == 0)
    return 0;
  // NNTR_KIMG_PACK=0: control arm -- never pack, so a K image taller than the
  // device limit fails to create exactly as it did before packing existed
  // (and the layer reports it and runs the buffer arm).
  static const bool pack_off = []() {
    const char *e = std::getenv("NNTR_KIMG_PACK");
    return e != nullptr && e[0] == '0';
  }();
  if (pack_off) {
    g_kimg_gsh = 0;
    return 0;
  }
  size_t max_h = 0, max_w = 0;
  cl_device_id dev = blas_cc->context_inst_.GetDeviceId();
  opencl::clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_h),
                          &max_h, nullptr);
  opencl::clGetDeviceInfo(dev, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(max_w),
                          &max_w, nullptr);
  if (max_h == 0 || max_w == 0)
    return 0;
  const size_t rows = (size_t)num_heads_KV * max_S;
  unsigned int gsh = 0;
  // Grow the shift until the height fits; stop if the width would overflow
  // instead (then nothing can be served and the caller falls back as before).
  while ((rows >> gsh) > max_h && gsh < 16u)
    ++gsh;
  if ((rows >> gsh) > max_h)
    return 0;
  // The kernels carry ONE compile-time KIMG_GSH for the whole process, but
  // layers of one model can need different shifts (KV heads x rows differ:
  // e.g. 8 KV heads on the sliding layers, 2 on the full-attention ones).
  // The published shift therefore only grows -- a larger shift still fits
  // every smaller need -- and a layer whose mirror was packed with a smaller
  // shift than the one finally published re-packs it before first use (see
  // kimg_gsh_published()).
  if (gsh < g_kimg_gsh)
    gsh = g_kimg_gsh;
  if (gsh != 0 && g_kimg_gsh != gsh)
    ml_logi("[kimg-pack] K image: %zu rows > device max height %zu; packing "
            "%u rows per image row (KIMG_GSH=%u)",
            rows, max_h, 1u << gsh, gsh);
  g_kimg_gsh = gsh;
  return gsh;
}

unsigned int kimg_gsh_published() { return g_kimg_gsh; }

// The V mirror image's row pitch is the mirror's row capacity, and
// image2d_from_buffer wants that pitch to be a multiple of
// CL_DEVICE_IMAGE_PITCH_ALIGNMENT pixels (8 halves each): a max_seq_len such as
// 4104 builds a 513-pixel pitch, clCreateImage refuses it, and the whole
// process leaves the image path. Capacity is free to round UP, so do that; a
// capacity that is already aligned (typical values: 1024/2048/4096/...) is
// returned unchanged.
unsigned int ohwi_mirror_capacity(unsigned int rows) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  rows = (rows + 7u) & ~7u;
  if (!blas_cc || rows == 0)
    return rows;
  static const cl_uint pitch_align = [blas_cc]() {
    cl_uint a = 0;
    cl_device_id dev = blas_cc->context_inst_.GetDeviceId();
    if (opencl::clGetDeviceInfo(dev, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                                sizeof(cl_uint), &a, nullptr) != CL_SUCCESS ||
        a == 0)
      a = 1; // no constraint reported: keep the 8-row rounding only
    return a;
  }();
  const unsigned int s_align = 8u * (unsigned int)pitch_align;
  return (rows + s_align - 1u) / s_align * s_align;
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
                           cl_mem *out_buf, cl_mem *out_image,
                           unsigned int k_gsh) {
  *out_buf = nullptr;
  *out_image = nullptr;
  if (num_heads_KV == 0 || head_dim == 0 || max_S == 0)
    return false;
  if ((head_dim % 8u) != 0 || (max_S % 8u) != 0)
    return false; // 8 halves/texel
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  const size_t bytes =
    (size_t)num_heads_KV * max_S * head_dim * sizeof(uint16_t);

  cl_int err = CL_SUCCESS;
  opencl::ClMemAcctScope _acct(is_v ? "kv:mirror_v" : "kv:mirror_k");
  // NNTR_KV_VIMG_NOBUF=1: the V mirror is a plain image (no
  // buffer underneath, no buffer-path write ever, filled through the image),
  // so the driver owns its layout and sees nothing but image traffic on it.
  // Diagnostic arm; the tight view and the buffer read need the buffer and
  // fall back to the full image.
  static const bool v_nobuf = []() {
    const char *e = std::getenv("NNTR_KV_VIMG_NOBUF");
    return e != nullptr && e[0] == '1';
  }();
  const bool nobuf = is_v && v_nobuf;
  cl_mem buf = nullptr;
  if (!nobuf) {
    buf = opencl::clCreateBufferT(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || buf == nullptr)
      return false;
  }
  // Zero the padding rows once. Enqueued non-blocking: this runs on the
  // in-order SVM queue ahead of any scatter/attention that touches the mirror,
  // so no clFinish drain is needed (a per-layer drain here would bubble the GPU
  // 2x/layer = ~10s regression at prefill — the very host-sync anti-pattern we
  // avoid). Strictly the kernel only reads [0, N_kv) so the fill is defensive.
  // NNTR_KV_MIRROR_FILL: unset = zero fill through the buffer
  // (the default behaviour); 0 = no fill; img = fill through an image-write
  // kernel over the whole image (the driver then sees only image writes);
  // 0xNNNN = fill the buffer with that 16-bit pattern (init-hazard arm).
  static const std::string mirror_fill = []() {
    const char *e = std::getenv("NNTR_KV_MIRROR_FILL");
    return std::string(e != nullptr ? e : "");
  }();
  uint16_t fillv = 0;
  bool fill_buf = true, fill_img = false;
  if (mirror_fill == "0")
    fill_buf = false;
  else if (mirror_fill == "img") {
    fill_buf = false;
    fill_img = true;
  } else if (mirror_fill.rfind("0x", 0) == 0)
    fillv = (uint16_t)std::strtoul(mirror_fill.c_str(), nullptr, 16);
  if (nobuf) {
    fill_buf = false;
    fill_img = true;
  }
  if (fill_buf)
    opencl::clEnqueueFillBuffer(q, buf, &fillv, sizeof(uint16_t), 0, bytes, 0,
                                nullptr, nullptr);

  cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};
  cl_image_desc d{};
  d.image_type = CL_MEM_OBJECT_IMAGE2D;
  if (is_v) {
    d.image_width = (size_t)max_S / 8;
    d.image_height = (size_t)num_heads_KV * head_dim;
    d.image_row_pitch = (size_t)max_S * sizeof(uint16_t);
  } else {
    // [kimg-pack] Fold 1<<k_gsh consecutive (head, seq) rows into one image
    // row. The rows are contiguous in the buffer, so this is a pure view
    // change; k_gsh == 0 is the original descriptor byte for byte. Must match
    // KIMG_GSH in the kernel -- both come from kimg_gsh_for().
    const size_t g = (size_t)1u << k_gsh;
    if (((size_t)num_heads_KV * max_S) % g != 0) {
      if (buf)
        opencl::clReleaseMemObjectT(buf);
      return false;
    }
    d.image_width = ((size_t)head_dim / 8) * g;
    d.image_height = ((size_t)num_heads_KV * max_S) / g;
    d.image_row_pitch = (size_t)head_dim * sizeof(uint16_t) * g;
  }
  d.buffer = buf; // nullptr under NNTR_KV_VIMG_NOBUF: a driver-owned image
  if (nobuf)
    d.image_row_pitch = 0;
  cl_int ie = CL_SUCCESS;
  // [kv-img-write] READ_WRITE when the scatters write through this image
  // (write_imageui); READ_ONLY otherwise, byte for byte as before.
  cl_mem image = opencl::clCreateImageT(
    ctx, kv_img_write_enabled() ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY, &fmt,
    &d, nullptr, &ie);
  if ((ie != CL_SUCCESS || image == nullptr) && kv_img_write_enabled()) {
    // [kv-img-write] Now that the path is the default, a driver that will not
    // hand out a READ_WRITE view of a buffer must not cost us the mirror (and
    // with the ring on, the run): disarm the write path process-wide and
    // re-create the very READ_ONLY image the pre-default build made.
    kv_img_write_disarm("clCreateImage(CL_MEM_READ_WRITE) refused");
    ie = CL_SUCCESS;
    image =
      opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &fmt, &d, nullptr, &ie);
  }
  if (ie != CL_SUCCESS || image == nullptr) {
    ml_logw("[kv-img-write] KVIMG_MARKER_A: clCreateImage refused for the "
            "%s mirror (err %d); the layer falls back",
            is_v ? "V" : "K", (int)ie);
    if (buf)
      opencl::clReleaseMemObjectT(buf);
    return false;
  }
  if (fill_img) {
    ClContext::SharedPtrClKernel kp = kvImgKernel("mirror_fill_img");
    int w = (int)d.image_width, h = (int)d.image_height;
    cl_uint v = 0u;
    if (kp && kp->SetKernelArguments(0, &image, sizeof(cl_mem)) &&
        kp->SetKernelArguments(1, &w, sizeof(int)) &&
        kp->SetKernelArguments(2, &h, sizeof(int)) &&
        kp->SetKernelArguments(3, &v, sizeof(cl_uint))) {
      std::array<size_t, 3> gws = {((size_t)w + 63) / 64 * 64, (size_t)h, 1};
      std::array<size_t, 3> lws = {64, 1, 1};
      blas_cc->command_queue_inst_.enqueueKernel(
        kp->GetKernel(), 3, gws.data(), lws.data(), 0, nullptr, nullptr);
    }
  }
  *out_buf = buf;
  *out_image = image;
  return true;
}

// Release a cl_mem created by create_ohwi_kv_mirror. void* so callers needn't
// link OpenCL (the CausalLM layers link only libnntrainer + ccapi).
void release_cl_mem(void *mem) {
  if (mem)
    opencl::clReleaseMemObjectT(reinterpret_cast<cl_mem>(mem));
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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
  // [kv-img-write] the tight view is the image the sv kernels read once it is
  // adopted, so it is the image the V scatter must WRITE: same flags rule.
  cl_mem image = opencl::clCreateImageT(
    ctx, kv_img_write_enabled() ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY, &fmt,
    &d, nullptr, &ie);
  if ((ie != CL_SUCCESS || image == nullptr) && kv_img_write_enabled()) {
    // same fallback as the full-capacity mirror above
    kv_img_write_disarm("clCreateImage(CL_MEM_READ_WRITE) refused (tight V)");
    ie = CL_SUCCESS;
    image =
      opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &fmt, &d, nullptr, &ie);
  }
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
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rope_inplace_kernel, "k_scatter_ohwi", ropeCopts());
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
  ClContext::SharedPtrClKernel kp = blas_cc->registerClKernel(
    rope_inplace_kernel, "v_scatter_ohwi_t", ropeCopts());
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
    blas_cc->registerClKernel(rope_inplace_kernel, kname, ropeCopts());
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

// ===========================================================================
// [kv-img-write] IMAGE-PATH mirror writes/gathers (NNTR_KV_IMG_WRITE=1). Same
// mirrors, same rows, same folded mapping as the buffer scatters above -- only
// the memory path changes: these bind the IMAGE the attention reads.
// ===========================================================================
static ClContext::SharedPtrClKernel kvImgKernel(const char *kname) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(kv_img_write_kernel, kname, kvImgCopts());
  if (!kp) {
    // Loud AND disarming: with the path armed and the kernel missing, the
    // mirror would never be written at all. __read_write image2d_t needs
    // -cl-std=CL2.0 and a device that reports read-write images, so on a driver
    // without them the default must fall back to the buffer scatters rather
    // than drop the writes. The latch makes every later ask answer 0, so the
    // layer's own `kv_img_write` gate goes false from the next step on and the
    // scatter call sites (which also fall through on a false return) stay on
    // the buffer path for the rest of the process.
    static std::once_flag once;
    std::call_once(once, [&]() {
      ml_loge("[kv-img-write] KVIMG_MARKER_A: the image-write program did "
              "not build (%s, opts '%s'); falling back to the buffer scatters",
              kname, kvImgCopts().c_str());
    });
    kv_img_write_disarm("the -cl-std=CL2.0 image-write program did not build");
  }
  return kp;
}

// =============================================================================
// NNTR_ATTN_GPUHASH: GPU-side hashes of the image attention's
// intermediates with NO host synchronisation. A host-side probe (a node
// output read between layers) settles the very queue whose ordering is under
// test, so it can turn an irreproducible run into a reproducible one and then
// report nothing. This one enqueues a
// one-workgroup reduction after the stage it observes, into a slot buffer that
// the decode loop reads back only where it already blocks (the argmax token
// read), so the queue is neither split nor drained anywhere new. Diagnostic
// only: off unless NNTR_ATTN_GPUHASH is set; =1 hashes the M == 1 (decode)
// calls, =2 the prefill calls as well.
// =============================================================================
namespace {
constexpr unsigned int kGhSlots = 1u << 16;
constexpr size_t kGhLws = 256;
const std::string gpuhash_kernel = R"CL(
__kernel void gpuhash_u16(__global const ushort *src, uint n,
                          __global uint *out, uint slot, uint off,
                          uint row_stride, uint rows) {
  __local uint part[256];
  const uint lid = get_local_id(0);
  uint h = 2166136261u;
  for (uint r = 0; r < rows; ++r) {
    const uint base = off + r * row_stride;
    for (uint i = lid; i < n; i += 256u) {
      h ^= (uint)src[base + i];
      h *= 16777619u;
    }
  }
  part[lid] = h;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    uint acc = 2166136261u;
    for (uint j = 0; j < 256u; ++j) {
      acc ^= part[j];
      acc *= 16777619u;
    }
    out[slot] = acc;
  }
}
)CL";
struct GhLabel {
  const char *tag;
  unsigned int M;
  unsigned int bytes;
};
struct GhState {
  int mode = -1;
  cl_mem slots = nullptr;
  ClContext::SharedPtrClKernel kp;
  std::vector<GhLabel> labels; // enqueued since the last flush
  unsigned int next_slot = 0;
  unsigned int flush_from = 0;
};
GhState &gh_state() {
  static GhState s;
  return s;
}
int gh_mode() {
  GhState &s = gh_state();
  if (s.mode < 0) {
    const char *e = std::getenv("NNTR_ATTN_GPUHASH");
    s.mode = (e == nullptr || e[0] == '0') ? 0 : std::atoi(e);
  }
  return s.mode;
}
} // namespace

int attn_gpuhash_mode() { return gh_mode(); }

void attn_gpuhash_note(const char *tag, const void *svm, void *clmem,
                       size_t bytes, unsigned int M) {
  attn_gpuhash_note2(tag, svm, clmem, 0, bytes / 2, 0, 1, M);
}

void attn_gpuhash_note2(const char *tag, const void *svm, void *clmem,
                        size_t off_h, size_t n_h, size_t row_stride_h,
                        unsigned int rows, unsigned int M) {
  const int mode = gh_mode();
  if (mode == 0 || (M != 1 && mode < 2) || n_h == 0 || rows == 0 ||
      (svm == nullptr && clmem == nullptr))
    return;
  const size_t bytes = n_h * rows * 2;
  GhState &s = gh_state();
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return;
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  if (!s.kp) {
    s.kp = blas_cc->registerClKernel(gpuhash_kernel, "gpuhash_u16");
    if (!s.kp)
      return;
  }
  if (s.slots == nullptr) {
    cl_int e = CL_SUCCESS;
    s.slots = opencl::clCreateBufferT(ctx, CL_MEM_READ_WRITE,
                                      sizeof(cl_uint) * kGhSlots, nullptr, &e);
    if (e != CL_SUCCESS || s.slots == nullptr) {
      s.slots = nullptr;
      return;
    }
  }
  if (s.labels.size() >= kGhSlots)
    return; // nobody flushed for a whole ring: stop rather than wrap
  cl_kernel k = s.kp->GetKernel();
  const cl_uint n_words = (cl_uint)n_h;
  const cl_uint off_u = (cl_uint)off_h, rs_u = (cl_uint)row_stride_h,
                rows_u = (cl_uint)rows;
  const cl_uint slot = s.next_slot;
  cl_int err;
  if (clmem != nullptr) {
    cl_mem h = static_cast<cl_mem>(clmem);
    err = opencl::clSetKernelArg(k, 0, sizeof(cl_mem), &h);
  } else {
    err = opencl::clSetKernelArgSVMPointer(k, 0, svm);
  }
  if (err != CL_SUCCESS)
    return;
  if (opencl::clSetKernelArg(k, 1, sizeof(cl_uint), &n_words) != CL_SUCCESS ||
      opencl::clSetKernelArg(k, 2, sizeof(cl_mem), &s.slots) != CL_SUCCESS ||
      opencl::clSetKernelArg(k, 3, sizeof(cl_uint), &slot) != CL_SUCCESS ||
      opencl::clSetKernelArg(k, 4, sizeof(cl_uint), &off_u) != CL_SUCCESS ||
      opencl::clSetKernelArg(k, 5, sizeof(cl_uint), &rs_u) != CL_SUCCESS ||
      opencl::clSetKernelArg(k, 6, sizeof(cl_uint), &rows_u) != CL_SUCCESS)
    return;
  const size_t gws = kGhLws, lws = kGhLws;
  if (opencl::clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr,
                                     nullptr) != CL_SUCCESS)
    return;
  s.labels.push_back({tag, M, (unsigned int)bytes});
  s.next_slot = (s.next_slot + 1) % kGhSlots;
}

void attn_gpuhash_step_flush(unsigned int step, unsigned int tok) {
  if (gh_mode() == 0)
    return;
  GhState &s = gh_state();
  if (s.labels.empty() || s.slots == nullptr)
    return;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (!blas_cc)
    return;
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  const unsigned int n = (unsigned int)s.labels.size();
  std::vector<cl_uint> h(n, 0u);
  const unsigned int first = s.flush_from;
  const unsigned int n1 = std::min(n, kGhSlots - first);
  bool ok =
    opencl::clEnqueueReadBuffer(q, s.slots, CL_TRUE, sizeof(cl_uint) * first,
                                sizeof(cl_uint) * n1, h.data(), 0, nullptr,
                                nullptr) == CL_SUCCESS;
  if (ok && n1 < n)
    ok = opencl::clEnqueueReadBuffer(q, s.slots, CL_TRUE, 0,
                                     sizeof(cl_uint) * (n - n1), h.data() + n1,
                                     0, nullptr, nullptr) == CL_SUCCESS;
  if (ok) {
    for (unsigned int i = 0; i < n; ++i)
      std::fprintf(stderr, "[GH] step=%u tok=%u i=%u %s M=%u B=%u h=%08x\n",
                   step, tok, i, s.labels[i].tag, s.labels[i].M,
                   s.labels[i].bytes, (unsigned)h[i]);
  }
  s.flush_from = (s.flush_from + n) % kGhSlots;
  s.labels.clear();
}

/// Shared dispatch for the four image-path kernels: identical argument order to
/// the buffer scatters/gathers (the image handle simply takes the buffer's
/// place), so a call site swaps one function name and nothing else.
static bool kv_img_dispatch(const char *kname, const void *svm_ptr,
                            void *clmem_arg, void *image, bool image_is_dst,
                            unsigned int M, unsigned int num_heads_KV,
                            unsigned int head_dim, unsigned int max_S,
                            unsigned int position, unsigned int src_off,
                            bool has_src_off, size_t gws_x, size_t gws_z,
                            bool drain) {
  if (M == 0 || num_heads_KV == 0 || head_dim == 0 || image == nullptr)
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp = kvImgKernel(kname);
  if (!kp)
    return false;
  cl_mem img = static_cast<cl_mem>(image);
  const unsigned int a_img = image_is_dst ? 1u : 0u; // dst for a scatter
  const unsigned int a_mem = image_is_dst ? 0u : 1u; // the SVM/cl_mem side
  bool ok;
  if (clmem_arg != nullptr) {
    cl_mem sh = static_cast<cl_mem>(clmem_arg);
    ok = kp->SetKernelArguments(a_mem, &sh, sizeof(cl_mem));
  } else {
    ok = kp->SetKernelSVMArguments(a_mem, const_cast<void *>(svm_ptr));
  }
  int Mi = (int)M, hKVi = (int)num_heads_KV, di = (int)head_dim,
      maxSi = (int)max_S, posi = (int)position, soff = (int)src_off;
  if (!ok || !kp->SetKernelArguments(a_img, &img, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &Mi, sizeof(int)) ||
      !kp->SetKernelArguments(3, &hKVi, sizeof(int)) ||
      !kp->SetKernelArguments(4, &di, sizeof(int)) ||
      !kp->SetKernelArguments(5, &maxSi, sizeof(int)) ||
      !kp->SetKernelArguments(6, &posi, sizeof(int)))
    return false;
  if (has_src_off && !kp->SetKernelArguments(7, &soff, sizeof(int)))
    return false;
  constexpr size_t LWS_Z = 64;
  std::array<size_t, 3> gws = {gws_x, (size_t)num_heads_KV,
                               (gws_z + LWS_Z - 1) / LWS_Z * LWS_Z};
  std::array<size_t, 3> lws = {1, 1, LWS_Z};
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (drain)
    opencl::clFinish(q);
  return true;
}

bool k_scatter_ohwi_img_cl(const uint16_t *src_svm, void *dst_image,
                           unsigned int M, unsigned int num_heads_KV,
                           unsigned int head_dim, unsigned int max_S,
                           unsigned int position, void *src_clmem,
                           unsigned int src_off) {
  // One work item per (t, head, d-texel).
  return kv_img_dispatch("k_scatter_ohwi_img", src_svm, src_clmem, dst_image,
                         /*image_is_dst=*/true, M, num_heads_KV, head_dim,
                         max_S, position, src_off, /*has_src_off=*/true,
                         /*gws_x=*/(size_t)M, /*gws_z=*/head_dim >> 3,
                         /*drain=*/false);
}

bool v_scatter_ohwi_t_img_slab_cl(const uint16_t *vcache_svm, void *dst_image,
                                  unsigned int num_heads_KV,
                                  unsigned int head_dim, unsigned int m_from,
                                  unsigned int m_to, unsigned int base_abs,
                                  unsigned int ring_cap, unsigned int to_abs) {
  if (num_heads_KV == 0 || head_dim == 0 || dst_image == nullptr ||
      vcache_svm == nullptr || m_to <= m_from)
    return false;
  const unsigned int tex0 = m_from >> 3;
  const unsigned int ntex = ((m_to + 7u) >> 3) - tex0;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp = kvImgKernel("v_scatter_ohwi_t_img_slab");
  if (!kp)
    return false;
  cl_mem dst = static_cast<cl_mem>(dst_image);
  int hKVi = (int)num_heads_KV, di = (int)head_dim, t0 = (int)tex0,
      nt = (int)ntex, base = (int)base_abs, ring = (int)ring_cap,
      toa = (int)to_abs;
  if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(vcache_svm)) ||
      !kp->SetKernelArguments(1, &dst, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(2, &hKVi, sizeof(int)) ||
      !kp->SetKernelArguments(3, &di, sizeof(int)) ||
      !kp->SetKernelArguments(4, &t0, sizeof(int)) ||
      !kp->SetKernelArguments(5, &nt, sizeof(int)) ||
      !kp->SetKernelArguments(6, &base, sizeof(int)) ||
      !kp->SetKernelArguments(7, &ring, sizeof(int)) ||
      !kp->SetKernelArguments(8, &toa, sizeof(int)))
    return false;
  constexpr size_t LWS_Z = 64;
  std::array<size_t, 3> gws = {(size_t)ntex, (size_t)num_heads_KV,
                               ((size_t)head_dim + LWS_Z - 1) / LWS_Z * LWS_Z};
  std::array<size_t, 3> lws = {1, 1, LWS_Z};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  return true;
}

bool v_scatter_ohwi_t_img_cl(const uint16_t *src_svm, void *dst_image,
                             unsigned int M, unsigned int num_heads_KV,
                             unsigned int head_dim, unsigned int max_S,
                             unsigned int position, void *src_clmem,
                             unsigned int src_off) {
  // One work item per (sequence texel, head, channel): a V texel is 8 sequence
  // positions, so the span is rounded out to whole texels and the partial ones
  // are read-modify-written.
  const size_t ntex = (size_t)(((position + M + 7u) >> 3) - (position >> 3));
  return kv_img_dispatch("v_scatter_ohwi_t_img", src_svm, src_clmem, dst_image,
                         /*image_is_dst=*/true, M, num_heads_KV, head_dim,
                         max_S, position, src_off, /*has_src_off=*/true,
                         /*gws_x=*/ntex, /*gws_z=*/head_dim, /*drain=*/false);
}

bool kv_scatter_ohwi_img_cl(const uint16_t *k_src_svm, void *k_src_clmem,
                            unsigned int k_src_off, void *k_dst_image,
                            unsigned int k_max_S, const uint16_t *v_src_svm,
                            void *v_src_clmem, unsigned int v_src_off,
                            void *v_dst_image, unsigned int v_max_S,
                            unsigned int num_heads_KV, unsigned int head_dim,
                            unsigned int position) {
  if (num_heads_KV == 0 || head_dim == 0 || (head_dim & 7u) ||
      k_dst_image == nullptr || v_dst_image == nullptr)
    return false;
  if ((k_src_clmem == nullptr && k_src_svm == nullptr) ||
      (v_src_clmem == nullptr && v_src_svm == nullptr))
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  ClContext::SharedPtrClKernel kp = kvImgKernel("kv_scatter_ohwi_img");
  if (!kp)
    return false;
  auto bind_src = [&](cl_uint idx, const uint16_t *p, void *clmem) {
    if (clmem != nullptr) {
      cl_mem h = static_cast<cl_mem>(clmem);
      return kp->SetKernelArguments(idx, &h, sizeof(cl_mem));
    }
    return kp->SetKernelSVMArguments(idx, const_cast<uint16_t *>(p));
  };
  cl_mem kimg = static_cast<cl_mem>(k_dst_image);
  cl_mem vimg = static_cast<cl_mem>(v_dst_image);
  int hKVi = (int)num_heads_KV, di = (int)head_dim, kS = (int)k_max_S,
      vS = (int)v_max_S, posi = (int)position, ko = (int)k_src_off,
      vo = (int)v_src_off;
  if (!bind_src(0, k_src_svm, k_src_clmem) ||
      !kp->SetKernelArguments(1, &kimg, sizeof(cl_mem)) ||
      !bind_src(2, v_src_svm, v_src_clmem) ||
      !kp->SetKernelArguments(3, &vimg, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(4, &hKVi, sizeof(int)) ||
      !kp->SetKernelArguments(5, &di, sizeof(int)) ||
      !kp->SetKernelArguments(6, &kS, sizeof(int)) ||
      !kp->SetKernelArguments(7, &vS, sizeof(int)) ||
      !kp->SetKernelArguments(8, &posi, sizeof(int)) ||
      !kp->SetKernelArguments(9, &ko, sizeof(int)) ||
      !kp->SetKernelArguments(10, &vo, sizeof(int)))
    return false;
  constexpr size_t LWS_Z = 64;
  const size_t lanes = (size_t)(head_dim >> 3) + (size_t)head_dim;
  std::array<size_t, 3> gws = {1, (size_t)num_heads_KV,
                               (lanes + LWS_Z - 1) / LWS_Z * LWS_Z};
  std::array<size_t, 3> lws = {1, 1, LWS_Z};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  return true;
}

bool k_gather_ohwi_img_cl(void *src_image, uint16_t *dst_svm, unsigned int M,
                          unsigned int num_heads_KV, unsigned int head_dim,
                          unsigned int max_S, unsigned int position,
                          bool drain) {
  return kv_img_dispatch("k_gather_ohwi_img", dst_svm, nullptr, src_image,
                         /*image_is_dst=*/false, M, num_heads_KV, head_dim,
                         max_S, position, 0, /*has_src_off=*/false,
                         /*gws_x=*/(size_t)M, /*gws_z=*/head_dim >> 3, drain);
}

bool v_gather_ohwi_t_img_cl(void *src_image, uint16_t *dst_svm, unsigned int M,
                            unsigned int num_heads_KV, unsigned int head_dim,
                            unsigned int max_S, unsigned int position,
                            bool drain) {
  return kv_img_dispatch("v_gather_ohwi_t_img", dst_svm, nullptr, src_image,
                         /*image_is_dst=*/false, M, num_heads_KV, head_dim,
                         max_S, position, 0, /*has_src_off=*/false,
                         /*gws_x=*/(size_t)M, /*gws_z=*/head_dim, drain);
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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
      opencl::clReleaseMemObjectT(sc.q_image);
      sc.q_image = nullptr;
    }
    if (sc.k_image) {
      opencl::clReleaseMemObjectT(sc.k_image);
      sc.k_image = nullptr;
    }
    if (sc.v_image) {
      opencl::clReleaseMemObjectT(sc.v_image);
      sc.v_image = nullptr;
    }

    cl_image_format img_fmt{CL_RGBA, CL_UNSIGNED_INT32};
    cl_image_desc qd{};
    qd.image_type = CL_MEM_OBJECT_IMAGE2D;
    qd.image_width = HD_Q / 8;
    qd.image_height = M;
    qd.image_row_pitch = HD_Q * sizeof(uint16_t);
    qd.buffer = sc.q_buf;
    sc.q_image = opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &img_fmt, &qd,
                                        nullptr, &err);
    if (err != CL_SUCCESS || !sc.q_image)
      return false;

    cl_image_desc kd{};
    kd.image_type = CL_MEM_OBJECT_IMAGE2D;
    kd.image_width = HD_KV / 8;
    kd.image_height = N_kv;
    kd.image_row_pitch = HD_KV * sizeof(uint16_t);
    kd.buffer = sc.k_buf;
    sc.k_image = opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &img_fmt, &kd,
                                        nullptr, &err);
    if (err != CL_SUCCESS || !sc.k_image)
      return false;

    cl_image_desc vd = kd;
    vd.buffer = sc.v_buf;
    sc.v_image = opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &img_fmt, &vd,
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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

  // Conservative sync point (same as concat variant); the queue is in order.
  opencl::clFinish(q);

  // These barriers date from an out-of-order queue, on which Intel NEO did
  // NOT auto-serialize the K1→K2→K3 chain (data-dependent through `scores`),
  // so K3 could read an unwritten `scores` and emit all-zero O. The queue is
  // now in order, so they are redundant. Adreno's driver serializes
  // same-buffer-dependent kernels in practice, so this gate
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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
// NNTR_SV_IMG_RW (default on; =0 restores the sampled read): the sv kernel
// with the V mirror bound as a __read_write image (CL2.0, own program). On
// this driver a read-write image cannot be served by the sampler's texture
// path, which is what avoids the stale read described at the dispatch site.
static const std::string sv_imgrw_kernel = std::string(R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)CL") + R"CL(
__kernel void sv_matmul_f16_ohwi_imgrw_tm2(__global const half *scores,
                                         __read_write image2d_t V_img,
                                         __global half *O, const int M,
                                         const int N_kv, const int d,
                                         const int HD_Q, const int S_max,
                                         const int gqa, const int causal,
                                         const int local_window) {
  const int x0 = get_global_id(0) * 8;
  const int m0 = get_global_id(1) * 2;
  const int head_q = get_global_id(2);
  if (m0 >= M || x0 >= d)
    return;
  const int m1 = m0 + 1;
  const int has1 = (m1 < M);
  const int head_kv = head_q / gqa;
  const int v_row0 = head_kv * d + x0;
  const long sb0 = (long)head_q * (long)M * (long)N_kv + (long)m0 * (long)N_kv;
  const long sb1 = (long)head_q * (long)M * (long)N_kv + (long)m1 * (long)N_kv;
  // Causal cap on the query's ABSOLUTE position (q_off=N_kv-M); decode (M=1)
  // sits at N-1 and must sum ALL keys, not just V[0..7]. See the non-tm2
  // kernel; a non-causal call keeps the full range.
  int N_kv_tex = (N_kv + 7) >> 3;
  int n_tex0 = 0;
  if (causal) {
    const int q_off = N_kv - M;
    const int cap_m = has1 ? m1 : m0;
    const int N_kv_tex_causal = ((q_off + cap_m) >> 3) + 1;
    if (N_kv_tex_causal < N_kv_tex)
      N_kv_tex = N_kv_tex_causal;
    // Sliding-window LOWER bound (see sv_matmul_f16_ohwi_img above). This WI
    // owns rows m0 and m1=m0+1; m0 has the SMALLER absolute query position and
    // therefore the smaller window floor, so bounding on m0 is the safe (and
    // for m1 merely conservative) choice — the extra texels m1 does not need
    // are exactly the +0.0 ones it would have skipped.
    if (local_window > 0) {
      const int n_first = q_off + m0 - local_window + 1;
      if (n_first > 0)
        n_tex0 = n_first >> 3;
      if (n_tex0 > N_kv_tex)
        n_tex0 = N_kv_tex;
    }
  }

  float acc0[8], acc1[8];
#pragma unroll
  for (int t = 0; t < 8; t++) {
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  for (int n_tex = n_tex0; n_tex < N_kv_tex; n_tex++) {
    const int n0 = n_tex * 8;
    half8 s0, s1;
    if (n0 + 8 <= N_kv) {
      s0 = vload8(0, scores + sb0 + n0);
      s1 = has1 ? vload8(0, scores + sb1 + n0) : (half8)((half)0.0h);
    } else {
      half t0[8], t1[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        t0[k] = (n0 + k < N_kv) ? scores[sb0 + n0 + k] : (half)0.0h;
        t1[k] = (has1 && n0 + k < N_kv) ? scores[sb1 + n0 + k] : (half)0.0h;
      }
      s0 = vload8(0, t0);
      s1 = vload8(0, t1);
    }
    const float4 s0lo = convert_float4(s0.s0123);
    const float4 s0hi = convert_float4(s0.s4567);
    const float4 s1lo = convert_float4(s1.s0123);
    const float4 s1hi = convert_float4(s1.s4567);
#pragma unroll
    for (int t = 0; t < 8; t++) {
      const uint4 vv = read_imageui(V_img, (int2)(n_tex, v_row0 + t));
      const half8 vp = as_half8(vv);
      const float4 vlo = convert_float4(vp.s0123);
      const float4 vhi = convert_float4(vp.s4567);
      acc0[t] += dot(s0lo, vlo) + dot(s0hi, vhi);
      acc1[t] += dot(s1lo, vlo) + dot(s1hi, vhi);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; t++) {
    const int x = x0 + t;
    if (x < d) {
      O[(long)m0 * HD_Q + head_q * d + x] = (half)acc0[t];
      if (has1)
        O[(long)m1 * HD_Q + head_q * d + x] = (half)acc1[t];
    }
  }
}
)CL";

// The qk kernels with the OHWI K mirror bound as a __read_write image2d_t
// and read with the sampler-less read_imageui (qk_matmul_f16_ohwi_imgrw and
// its banded twin, their own -cl-std=CL2.0 program). This is the K side of
// the sv kernel's read-write V binding above: the same __read_only + sampler
// read path that was served stale V texels inside one launch. The program is
// not a second copy of the kernels: it is derived from two_conv_attention.cl's
// own text -- the file's preamble (fp16 pragma, TM_QK/TN_QK), SOFTMAX_LWS,
// and the contiguous run from tca_score_band() through the two image qk
// kernels -- with exactly three edits: the image qualifier, the sampler
// argument, and the kernel names. Same loop, same accumulation order, same
// texel addressing (KIMG_GSH is passed through the compile options), so the
// scores are the sampled kernels' scores bit for bit wherever the sampled
// read was not stale. An anchor that is missing yields an empty program and
// the call site keeps the sampled kernels, with one warning.
static const std::string &qk_imgrw_program() {
  static const std::string prog = []() -> std::string {
    const std::string &src = two_conv_attention_kernel;
    auto cut = [&](const char *from, const char *to, bool incl_to,
                   std::string &out) {
      const size_t a = src.find(from);
      if (a == std::string::npos)
        return false;
      const size_t b = src.find(to, a);
      if (b == std::string::npos)
        return false;
      out = src.substr(a, b - a + (incl_to ? std::strlen(to) : 0));
      return true;
    };
    auto sub = [](std::string &t, const char *from, const char *to,
                  size_t n_expect) {
      size_t n = 0, pos = 0;
      const size_t lf = std::strlen(from), lt = std::strlen(to);
      while ((pos = t.find(from, pos)) != std::string::npos) {
        t.replace(pos, lf, to);
        pos += lt;
        n++;
      }
      return n == n_expect;
    };
    std::string pre, lws, body;
    if (!cut("#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
             "__kernel void\nqk_matmul_f16(", false, pre) ||
        !cut("#define SOFTMAX_LWS ", "\n", true, lws) ||
        !cut("static inline void tca_score_band(",
             "#endif // TCA_BUFFER_ONLY (qk_matmul_f16_ohwi_img)", true, body))
      return std::string();
    // kimg_read(): __read_write image, no sampler (2 read_imageui forms).
    if (!sub(body, "__read_only image2d_t", "__read_write image2d_t", 3) ||
        !sub(body, "const sampler_t s,", "", 1) ||
        !sub(body, "img, s, (int2)", "img, (int2)", 2) ||
        !sub(body, "kimg_read(K_img, smp, ", "kimg_read(K_img, ", 2) ||
        !sub(body,
             "  const sampler_t smp =\n    CLK_NORMALIZED_COORDS_FALSE | "
             "CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n",
             "", 2) ||
        !sub(body, "qk_matmul_f16_ohwi_img(", "qk_matmul_f16_ohwi_imgrw(", 1) ||
        !sub(body, "qk_matmul_f16_ohwi_img_band(",
             "qk_matmul_f16_ohwi_imgrw_band(", 1))
      return std::string();
    return pre + "\n" + lws + "\n" + body + "\n";
  }();
  return prog;
}

// Compile options of that program: CL2.0 for the __read_write qualifier, and
// the K mirror's packing shift exactly as tca_copts() passes it to the
// sampled kernels (the handle cache below is dropped when the shift moves).
static std::string qk_imgrw_copts() {
  std::string o = "-cl-std=CL2.0";
  if (g_kimg_gsh != 0)
    o += " -DKIMG_GSH=" + std::to_string(g_kimg_gsh);
  return o;
}

// NNTR_QK_IMG_RW: 1 binds the K mirror read-write in the image qk kernels
// (above), 0 keeps the sampled read. Default on (QK_IMG_RW_DEFAULT) as a
// pre-emptive measure: the sampled read is the path that returned stale V
// texels, and a wrong answer is not an acceptable trade for prefill time.
// The sampled K read itself has shown no event -- GPU-side stage hashes over
// 46 runs on the V-read-write tree (8103 tokens 10 of 10 one trace, 919 tokens
// with 200 generated 30 of 30, 16203 tokens 6 of 6; 49,200 to 66,832 probes
// per run, every stage identical), where the same instrumentation put the V
// read at fault on every 8103-token run -- and the binding costs prefill at
// 1300 MHz, two interleaved runs per cell: 1.5B 919 tokens 460/464 ->
// 478/488 ms (+4.6 %), 8103 tokens 5908/5891 -> 6279/6239 ms (+6.1 %), gemma4
// 842 tokens 308/306 -> 322/324 ms (+5.2 %), E2B 923 tokens 732/723 -> 734 ms
// (one outlier run dropped), 0.3B 2047 tokens 354 -> 344 ms (noise); decode
// within noise. Its output is the sampled kernel's bit for bit (8103 tokens
// 6 of 6 and 919 tokens 10 of 10 runs give the sampled arm's trace), so the
// default changes time, not numbers.
static bool qk_img_rw_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_QK_IMG_RW");
    if (e != nullptr && e[0] != '\0')
      return e[0] != '0';
    return QK_IMG_RW_DEFAULT;
  }();
  return v;
}

static bool two_conv_attention_prefill_f16_ohwi_img_impl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem v_buf_in,
  cl_mem v_image_in, cl_mem k_image_in, uint16_t *O_svm, unsigned int M,
  unsigned int N_kv, unsigned int num_heads_Q, unsigned int num_heads_KV,
  unsigned int head_dim, unsigned int max_seq_len, bool causal,
  float attn_softcap = 0.0f, // Gemma2-style QK soft-cap (image-K)
  void *q_clmem = nullptr, void *o_clmem = nullptr,
  unsigned int local_window = 0, // >0: sliding-window mask (n+W <= q_pos)
  cl_mem v_buf_read = nullptr, unsigned int v_buf_stride = 0);

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
  unsigned int local_window, void *v_buf_read, unsigned int v_buf_stride) {
  if (!K_image_ohwi || !V_image_ohwi)
    return false;
  // NNTR_KV_VIMG_TOUCH=1: one image write to the V image's last
  // texel (never read below max_seq_len - 8 keys) right before the attention,
  // so the driver sees an image write on this object immediately before the
  // read (diagnostic arm for "is the texture cache invalidated on a write?").
  static const bool vimg_touch = []() {
    const char *e = std::getenv("NNTR_KV_VIMG_TOUCH");
    return e != nullptr && e[0] == '1';
  }();
  if (vimg_touch && N_kv + 8 <= max_seq_len) {
    ClContext::SharedPtrClKernel kp = kvImgKernel("mirror_touch_img");
    int x = (int)(max_seq_len / 8) - 1, y = (int)(num_heads_KV * head_dim) - 1;
    if (kp && kp->SetKernelArguments(0, &V_image_ohwi, sizeof(cl_mem)) &&
        kp->SetKernelArguments(1, &x, sizeof(int)) &&
        kp->SetKernelArguments(2, &y, sizeof(int))) {
      auto *cc =
        static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
      std::array<size_t, 3> gws = {1, 1, 1};
      cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 3, gws.data(),
                                            nullptr, 0, nullptr, nullptr);
    }
  }
  return two_conv_attention_prefill_f16_ohwi_img_impl(
    Q_svm, /** K_svm */ nullptr, /** v_buf_in */ nullptr, V_image_ohwi,
    K_image_ohwi, O_svm, M, N_kv, num_heads_Q, num_heads_KV, head_dim,
    max_seq_len, causal, attn_softcap, q_clmem, o_clmem, local_window,
    static_cast<cl_mem>(v_buf_read), v_buf_stride);
}

// =============================================================================
// §3.8 + image2d_from_buffer V variant. Q/K stay SVM (qk_matmul_f16_ohwi
// reused unchanged); V is wrapped as image2d_from_buffer (either built
// here from a cl_mem buffer, or passed in pre-built by the caller). We
// dispatch the new sv_matmul_f16_ohwi_img kernel. Same Adreno-image-
// cache mechanism that v8c FC exploits for 87% peak.
// =============================================================================
// =============================================================================
// Host mirror of two_conv_attention.cl's tca_score_band(). Same integer
// arithmetic, so the grid the host launches and the band the kernels compute
// cannot drift. See THE SCORE-BAND CONTRACT in two_conv_attention.cl.
// =============================================================================
static inline void tca_score_band_host(int M, int N_kv, int causal,
                                       int local_window, int pair_lo,
                                       int pair_hi, int align_lo, int *n_lo,
                                       int *n_hi) {
  int lo = 0;
  int hi = N_kv;
  if (causal) {
    const int q_off = N_kv - M;
    int tex_hi = (N_kv + 7) >> 3;
    const int tex_hi_causal = ((q_off + pair_hi) >> 3) + 1;
    if (tex_hi_causal < tex_hi)
      tex_hi = tex_hi_causal;
    if (local_window > 0) {
      const int n_first = q_off + pair_lo - local_window + 1;
      int t0 = 0;
      if (n_first > 0)
        t0 = n_first >> 3;
      if (t0 > tex_hi)
        t0 = tex_hi;
      lo = t0 << 3;
      if (align_lo > 1)
        lo = lo & ~(align_lo - 1);
    }
    hi = tex_hi << 3;
    if (hi > N_kv)
      hi = N_kv;
  }
  if (lo > hi)
    lo = hi;
  *n_lo = lo;
  *n_hi = hi;
}

// sv_matmul's sliding-window floor (NNTR_SV_WIN=0 = the old full-range loop).
// Hoisted to file scope because the score band is only valid while sv honours
// it: with SV_WIN off sv reads from texel 0 and nothing may be banded.
static bool attn_sv_win_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_SV_WIN");
    return e ? (std::atoi(e) != 0) : true;
  }();
  return v;
}

// NNTR_ATTN_BAND: 0 = the full-row qk grid + full-row softmax (control
// arm), 1 = banded qk grid + banded softmax (default), 2 = the
// reverted per-row attempt's arithmetic (per-ROW softmax band, unbanded qk)
// kept ONLY so its failure is reproducible; it breaks the golden by
// construction.
static int attn_band_mode() {
  static const int v = []() {
    const char *e = std::getenv("NNTR_ATTN_BAND");
    return e ? std::atoi(e) : 1;
  }();
  return v;
}

// NNTR_ATTN_BAND_DECODE_QK: decode-only, DEFAULT OFF. The banded qk *grid* is
// a loss at M == 1 -- measured, see the launch site -- so decode bands only its
// softmax. 1 turns the banded qk grid on at decode as well (the A/B arm).
static bool attn_band_decode_qk_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_ATTN_BAND_DECODE_QK");
    return e != nullptr && e[0] != '0';
  }();
  return v;
}

// NNTR_ATTN_NX_QUANT: decode-only. The number of n-tiles the qk grid launches
// at M == 1 is rounded up to the NEXT multiple of this, so the work size is
// constant for that many *64 keys instead of stepping every 64 tokens. 0 = off.
// See the comment at the launch site. Default 64 tiles = 512 keys.
static size_t attn_nx_quant() {
  static const size_t v = []() -> size_t {
    const char *e = std::getenv("NNTR_ATTN_NX_QUANT");
    if (e == nullptr)
      return 64;
    long n = std::atol(e);
    return n > 0 ? (size_t)n : 0;
  }();
  return v;
}

// NNTR_ATTN_BAND_DECODE: 0 = decode (M == 1) keeps the unbanded qk grid and
// the full-row softmax while prefill stays banded -- the A/B control that
// isolates the decode half of the score band. Default 1 (banded).
static bool attn_band_decode_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_ATTN_BAND_DECODE");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

static bool two_conv_attention_prefill_f16_ohwi_img_impl(
  const uint16_t *Q_svm, const uint16_t *K_svm, cl_mem v_buf_in,
  cl_mem v_image_in, cl_mem k_image_in, uint16_t *O_svm, unsigned int M,
  unsigned int N_kv, unsigned int num_heads_Q, unsigned int num_heads_KV,
  unsigned int head_dim, unsigned int max_seq_len, bool causal,
  float attn_softcap, void *q_clmem, void *o_clmem, unsigned int local_window,
  cl_mem v_buf_read, unsigned int v_buf_stride) {
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
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
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
  static const bool scores_pingpong = []() {
    const char *e = std::getenv("NNTR_SCORES_PINGPONG");
    return e != nullptr && e[0] == '1';
  }();
  cl_mem scores_buf = sc.scores;
  if (scores_pingpong) {
    if (!tca_ensure(ctx, &sc.scores_pp, &sc.scores_pp_bytes, scores_bytes,
                    CL_MEM_READ_WRITE))
      return false;
    sc.scores_pp_i ^= 1;
    if (sc.scores_pp_i)
      scores_buf = sc.scores_pp;
  }

  // ---- The score band.
  // qk's n grid and softmax's n loop are both restricted to the band sv_matmul
  // actually reads, instead of the whole N_kv row. Requires sv to honour its
  // window floor (NNTR_SV_WIN=1, the default) -- with SV_WIN=0 sv reads from
  // texel 0 and nothing may be narrowed.
  //
  // M == 1 (decode) is banded by the SAME band function: the single query row
  // pairs with itself (pair_lo = pair_hi = 0), the causal cap is the whole
  // row (q_off = N_kv-1 => hi = N_kv) and the only narrowing is the sliding
  // floor, so a W=512 layer touches ~576 keys instead of N_kv while a full
  // layer (W=0) keeps the identical grid. sv_matmul_f16_ohwi_img_tm2 already
  // starts at the same window floor (see arg 10 below), so its reads stay
  // inside the band; the 64-aligned floor keeps softmax's n == tid (mod 64)
  // work-item mapping, hence bit-identical row_max/row_sum.
  // NNTR_ATTN_BAND_DECODE=0 is the decode-only control arm (prefill still
  // banded); NNTR_ATTN_BAND=0 remains the control for both.
  const bool band_on = attn_band_mode() != 0 && causal && attn_sv_win_on() &&
                       (M > 1 || attn_band_decode_on());
  // The qk half of the band is PREFILL-ONLY by default. At M == 1 it is a
  // measured LOSS: qk_matmul's whole-tile window skip already declined the
  // d-reduction for the tiles the band removes, so all the narrower grid saves
  // is 8 halves of -INF per tile -- while the shorter n grid gives the Adreno
  // scheduler fewer workgroups to hide latency with. CL event census, decode
  // dispatches, clock-normalised: at an 883-token context attn.qk 3.421 ->
  // 3.573 ms/token (+4.4 %) against attn.softmax 0.334 -> 0.257 (-0.077), i.e.
  // net +0.12 ms/token; at 3595 tokens attn.qk 9.191 -> 9.175 (nothing) against
  // attn.softmax 0.990 -> 0.396 (-0.594). Banding the softmax alone is the
  // strictly-positive subset at both, so that is what decode ships.
  const bool qk_band_on = band_on && attn_band_mode() == 1 &&
                          k_image_in != nullptr &&
                          (M > 1 || attn_band_decode_qk_on());

  // ---- Resolve V image2d: either caller-provided, or build / reuse a
  //      cached one keyed on V_buf_in + shape. ----
  cl_mem v_image = v_image_in;
  if (v_image == nullptr) {
    const bool v_changed =
      sc.v_ohwi_image == nullptr || sc.v_ohwi_buf != v_buf_in ||
      sc.v_ohwi_HD_KV != HD_KV || sc.v_ohwi_S_max != max_seq_len;
    if (v_changed) {
      if (sc.v_ohwi_image) {
        opencl::clReleaseMemObjectT(sc.v_ohwi_image);
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
      sc.v_ohwi_image = opencl::clCreateImageT(ctx, CL_MEM_READ_ONLY, &img_fmt,
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
  attn_gpuhash_note("q", Q_svm, q_clmem, (size_t)M * HD_Q * sizeof(uint16_t),
                    M);
  if (attn_drain)
    opencl::clFinish(q);

  // ---- K1: QK matmul OHWI — pick image2d-K kernel when caller gave
  //          us a K image, else the SVM buffer kernel.
  {
    const char *k1_name = k_image_in != nullptr
                            ? (qk_band_on ? "qk_matmul_f16_ohwi_img_band"
                                          : "qk_matmul_f16_ohwi_img")
                            : "qk_matmul_f16_ohwi";
    // Per-call-site kernel handle cache: registerClKernel
    // measured ~12ms per cached lookup from this wrapper (3x/layer = the
    // 36ms/layer host issue tax the CL-event profiler shows as GPU idle).
    static ClContext::SharedPtrClKernel kp_img, kp_img_band, kp_buf;
    // [kimg-pack] The cached handles were compiled for ONE KIMG_GSH. A second
    // model loaded into the same process can publish a different shift, and a
    // handle kept across that would read the new mirrors with the old
    // addressing. Drop them when the published shift moves (rare: once per
    // load at most).
    static ClContext::SharedPtrClKernel kp_imgrw, kp_imgrw_band;
    static unsigned int kp_gsh = 0;
    if (kp_gsh != g_kimg_gsh) {
      kp_gsh = g_kimg_gsh;
      kp_img.reset();
      kp_img_band.reset();
      kp_imgrw.reset();
      kp_imgrw_band.reset();
    }
    // K mirror bound __read_write (qk_imgrw_program above) when the image
    // was created CL_MEM_READ_WRITE, i.e. the image-write scatter path is
    // armed -- a READ_ONLY image cannot be bound to a __read_write argument.
    // A program or argument the driver refuses drops this process to the
    // sampled kernels with one warning; nothing else changes.
    static bool imgrw_disarmed = false;
    ClContext::SharedPtrClKernel kp;
    if (k_image_in != nullptr && qk_img_rw_on() && !imgrw_disarmed &&
        kv_img_write_enabled()) {
      ClContext::SharedPtrClKernel &kr = qk_band_on ? kp_imgrw_band : kp_imgrw;
      if (!kr) {
        const std::string &prog = qk_imgrw_program();
        if (!prog.empty())
          kr = blas_cc->registerClKernel(prog,
                                         qk_band_on
                                           ? "qk_matmul_f16_ohwi_imgrw_band"
                                           : "qk_matmul_f16_ohwi_imgrw",
                                         qk_imgrw_copts());
        if (!kr) {
          imgrw_disarmed = true;
          ml_logw("[image-attn] the read-write qk program did not build; "
                  "falling back to the sampled K read");
        }
      }
      kp = kr;
    }
    if (!kp) {
      ClContext::SharedPtrClKernel &ks =
        (k_image_in != nullptr) ? (qk_band_on ? kp_img_band : kp_img) : kp_buf;
      if (!ks)
        ks = blas_cc->registerClKernel(two_conv_attention_kernel, k1_name,
                                       tca_copts());
      kp = ks;
    }
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
      if (!kp->SetKernelArguments(1, &k_image_in, sizeof(cl_mem))) {
        if (imgrw_disarmed || kp != (qk_band_on ? kp_imgrw_band : kp_imgrw))
          return false;
        // The driver would not take this image as __read_write: sampled
        // kernels from here on (the Q argument is re-set below the same way).
        imgrw_disarmed = true;
        ml_logw("[image-attn] the K mirror was refused as a __read_write "
                "argument; falling back to the sampled K read");
        ClContext::SharedPtrClKernel &ks = qk_band_on ? kp_img_band : kp_img;
        if (!ks)
          ks = blas_cc->registerClKernel(two_conv_attention_kernel, k1_name,
                                         tca_copts());
        kp = ks;
        if (!kp)
          return false;
        if (q_clmem != nullptr) {
          cl_mem qh = static_cast<cl_mem>(q_clmem);
          if (!kp->SetKernelArguments(0, &qh, sizeof(cl_mem)))
            return false;
        } else if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_svm)))
          return false;
        if (!kp->SetKernelArguments(1, &k_image_in, sizeof(cl_mem)))
          return false;
      }
    } else {
      if (!kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_svm)))
        return false;
    }
    if (!kp->SetKernelArguments(2, &scores_buf, sizeof(cl_mem)))
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
    size_t nx = (N_kv + TN_QK - 1) / TN_QK;
    const size_t mx = (M + TM_QK - 1) / TM_QK;
    // Banded grid: gws.x counts n-tiles inside an m-tile's band, and the
    // kernel adds that m-tile's band_lo itself. Take the widest band over all
    // m-tiles so every tile's [band_lo, band_hi) is covered. On gemma4's
    // sliding owning layers (W=512) this is ~74 tiles instead of ceil(N_kv/8)
    // = 450 at N_kv=3595; on the full-attention ones the band is the causal
    // prefix, so the width is unchanged but the tiles past band_hi now return
    // WITHOUT writing the -INFINITY nobody reads.
    if (qk_band_on) {
      size_t w = 0;
      for (size_t m0 = 0; m0 < (size_t)M; m0 += TM_QK) {
        const size_t ml =
          (m0 + TM_QK - 1 < (size_t)M - 1) ? (m0 + TM_QK - 1) : (size_t)M - 1;
        int blo = 0, bhi = 0;
        tca_score_band_host((int)M, (int)N_kv, 1, (int)local_window, (int)m0,
                            (int)ml, (int)SOFTMAX_LWS, &blo, &bhi);
        if (bhi > blo) {
          const size_t t = ((size_t)(bhi - blo) + TN_QK - 1) / TN_QK;
          if (t > w)
            w = t;
        }
      }
      nx = w > 0 ? w : 1;
    }
    // [decode] Keep the qk work size CONSTANT from token to token. The
    // recordable decode queue (NNTR_CL_RECORD_DECODE) throws a whole token's
    // recording away whenever any dispatch's work size moves, and qk's n grid
    // is the ONLY decode work size that tracks N_kv: unbanded it is
    // ceil(N_kv/8), which steps every time N_kv crosses a multiple of 64, and
    // banded it is ceil((N_kv mod 64 + W)/8), which steps on the sliding
    // layers for the same reason. Round the tile count up to the NEXT multiple
    // of NX_QUANT (=64 tiles = 512 keys) and both classes hold still for 512
    // tokens at a time: the recorder's re-records fall to ~0 and its plain
    // fallback chunks with them. The extra tiles return without writing
    // anything (n0 >= band_hi in the banded kernel, n0 >= N_kv in the
    // unbanded one), so the scores are bit-identical -- and decode measures
    // surplus qk tiles as free: a CL event census removed 378 of 450 of
    // them at a 3595-token context for 0.016 ms/token, i.e. nothing.
    if (M == 1) {
      const size_t nq = attn_nx_quant();
      if (nq > 0)
        nx = (nx / nq + 1) * nq;
    }
    // The LWS is env-overridable + measured (NNTR_QK_LWS="x,y,z"). The current
    // default is {8,8,1} (see the re-sweep note below); originally
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
  if (M == 1 || scores_bytes <= (32u << 20))
    attn_gpuhash_note("qk", nullptr, scores_buf, scores_bytes, M);
  if (M == 1 && N_kv > 16) {
    // the newest 8-9 keys of every head vs the older ones: which half of
    // the K mirror read moved
    const size_t new_lo = (size_t)((N_kv - 8) & ~1u);
    attn_gpuhash_note2("qkN", nullptr, scores_buf, new_lo,
                       (size_t)N_kv - new_lo, N_kv, num_heads_Q, M);
    attn_gpuhash_note2("qkO", nullptr, scores_buf, 0, new_lo, N_kv, num_heads_Q,
                       M);
  }

  // ---- K2: row softmax (scores cl_mem, in-place) ----
  {
    static double tp2_reg = 0, tp2_arg = 0, tp2_enq = 0;
    const char *k2_name = band_on ? "softmax_row_band_f16" : "softmax_row_f16";
    static ClContext::SharedPtrClKernel kp_plain, kp_bandk; // handle cache
    ClContext::SharedPtrClKernel &kp = band_on ? kp_bandk : kp_plain;
    if (!kp)
      kp = blas_cc->registerClKernel(two_conv_attention_kernel, k2_name,
                                     tca_copts());
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
    if (!kp->SetKernelArguments(0, &scores_buf, sizeof(cl_mem)))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv;
    if (!kp->SetKernelArguments(1, &Mi, sizeof(int)) ||
        !kp->SetKernelArguments(2, &Nkvi, sizeof(int)))
      return false;
    if (band_on) {
      int causal_k2 = causal ? 1 : 0;
      int lw_k2 = (int)local_window;
      int bmode_k2 = attn_band_mode();
      if (!kp->SetKernelArguments(3, &causal_k2, sizeof(int)) ||
          !kp->SetKernelArguments(4, &lw_k2, sizeof(int)) ||
          !kp->SetKernelArguments(5, &bmode_k2, sizeof(int)))
        return false;
    }
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

  if (M == 1 || scores_bytes <= (32u << 20))
    attn_gpuhash_note("sm", nullptr, scores_buf, scores_bytes, M);
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
    // NNTR_SV_IMG_READ=0: read the V mirror through its BUFFER
    // (sv_matmul_f16_ohwi_buf[_tm2]) instead of the image2d view. On Adreno 840
    // the image read of the V mirror is served stale, rarely and at random,
    // inside one sv launch: with q/scores bit-identical, O differs for a
    // block of query rows -- 8/8 events over 42 instrumented 919-token runs
    // and every run of an 8103-token prompt, on both mirror views, with and
    // without an image read in the write path, with two score scratches, with
    // and without `const` on the scores argument, and under every drain arm
    // (so it is cache visibility inside the launch, not ordering). The same
    // texel bytes read with vload8 from the buffer are bit-identical when the
    // image read is not stale and reproducible when it would have been (8103
    // tokens: 4/4 runs one trace against 6/6 distinct). The image read stays
    // the default for its speed (see below).
    // DEFAULT: the image read (the fast path; the buffer read costs +5~10 %
    // of prefill, measured 1.5B/E2B/gemma4/0.3B). NNTR_SV_IMG_READ=0 selects
    // the buffer read, which is the reproducible fallback.
    static const bool sv_img_read = []() {
      const char *e = std::getenv("NNTR_SV_IMG_READ");
      return !(e != nullptr && e[0] == '0');
    }();
    const bool sv_buf =
      !sv_img_read && v_buf_read != nullptr && v_buf_stride != 0;
    static ClContext::SharedPtrClKernel kp; // call-site handle cache (sv_tm2
                                            // is process-constant)
    static ClContext::SharedPtrClKernel kp_buf;
    if (sv_buf) {
      if (!kp_buf)
        kp_buf = blas_cc->registerClKernel(two_conv_attention_kernel,
                                           sv_tm2 ? "sv_matmul_f16_ohwi_buf_tm2"
                                                  : "sv_matmul_f16_ohwi_buf",
                                           tca_copts());
      if (!kp_buf)
        return false;
    } else if (!kp) {
      // DEFAULT (tm2): bind the V mirror as a __read_write image2d_t and read
      // it with the sampler-less read_imageui (sv_matmul_f16_ohwi_imgrw_tm2,
      // its own -cl-std=CL2.0 program). On Adreno 840 the __read_only +
      // sampler read of the same image is served a stale texel rarely and at
      // random inside one sv launch -- with Q and the scores bit-identical the
      // sv output differs for a block of query rows (13 of 13 located events;
      // every run of an 8103-token prompt differs from every other). The
      // read-write binding takes the coherent image-load path instead and is
      // reproducible: 8103 tokens 6 of 6 runs one trace (the same trace the
      // buffer read gives), 919 tokens 30 of 30 runs hash-identical, E2B 1K
      // 6 of 6, where the sampled read gives 6 of 6 distinct; a sampler-less
      // read of the __read_only image still varies (4 of 4), as do a
      // driver-owned image, two alternating views, image-only writes and
      // fills, and every drain arm. Same object, same texels, same
      // arithmetic, so the numbers do not move. Cost, 1300 MHz: prefill
      // +4 % (1.5B 919 tok) to +6.4 % (8103 tok), decode within noise.
      // NNTR_SV_IMG_RW=0 restores the sampled read; a driver that refuses
      // the CL2.0 program falls back to it on its own. NNTR_SV_IMG_NOSAMP=1
      // is the diagnostic arm that reads the __read_only image sampler-less.
      static const bool sv_img_rw = []() {
        const char *e = std::getenv("NNTR_SV_IMG_RW");
        return !(e != nullptr && e[0] == '0');
      }();
      static const bool sv_img_nosamp = []() {
        const char *e = std::getenv("NNTR_SV_IMG_NOSAMP");
        return e != nullptr && e[0] == '1';
      }();
      if (sv_tm2 && sv_img_rw) {
        kp = blas_cc->registerClKernel(
          sv_imgrw_kernel, "sv_matmul_f16_ohwi_imgrw_tm2", "-cl-std=CL2.0");
        if (!kp) {
          static int logged = 0;
          if (!logged++)
            ml_logw("[image-attn] the read-write sv program did not build; "
                    "falling back to the sampled V read (not reproducible on "
                    "Adreno 840)");
        }
      }
      if (kp) {
      } else if (sv_tm2 && sv_img_nosamp)
        kp = blas_cc->registerClKernel(two_conv_attention_kernel,
                                       "sv_matmul_f16_ohwi_imgns_tm2",
                                       tca_copts());
      else
        kp = blas_cc->registerClKernel(two_conv_attention_kernel,
                                       sv_tm2 ? "sv_matmul_f16_ohwi_img_tm2"
                                              : "sv_matmul_f16_ohwi_img",
                                       tca_copts());
    }
    if (!sv_buf && !kp)
      return false;
    ClContext::SharedPtrClKernel &kpu = sv_buf ? kp_buf : kp;
    if (!kpu->SetKernelArguments(0, &scores_buf, sizeof(cl_mem)))
      return false;
    if (sv_buf) {
      if (!kpu->SetKernelArguments(1, &v_buf_read, sizeof(cl_mem)))
        return false;
    } else if (!kpu->SetKernelArguments(1, &v_image, sizeof(cl_mem)))
      return false;
    // Static GPU_CLMEM residency: write O straight into its planner cl_mem
    // sub-buffer when given (the wo FC consumes it device-direct).
    if (o_clmem != nullptr) {
      cl_mem oh = static_cast<cl_mem>(o_clmem);
      if (!kpu->SetKernelArguments(2, &oh, sizeof(cl_mem)))
        return false;
    } else if (!kpu->SetKernelSVMArguments(2, O_svm))
      return false;
    int Mi = (int)M, Nkvi = (int)N_kv, di = (int)head_dim;
    int hdq = (int)HD_Q, smax = (int)max_seq_len;
    int gqa = (int)(num_heads_Q / num_heads_KV);
    int causal_i = causal ? 1 : 0;
    if (!kpu->SetKernelArguments(3, &Mi, sizeof(int)) ||
        !kpu->SetKernelArguments(4, &Nkvi, sizeof(int)) ||
        !kpu->SetKernelArguments(5, &di, sizeof(int)) ||
        !kpu->SetKernelArguments(6, &hdq, sizeof(int)) ||
        !kpu->SetKernelArguments(7, &smax, sizeof(int)) ||
        !kpu->SetKernelArguments(8, &gqa, sizeof(int)) ||
        !kpu->SetKernelArguments(9, &causal_i, sizeof(int)))
      return false;
    // arg 10 = the sliding window, the same value K1 already gets. Both sv
    // kernels start their n loop at the window floor instead of 0; on the 28
    // sliding layers of gemma4 that is ~half the V texels, and the skipped
    // ones are exactly the +0.0 scores softmax_row_f16 wrote for the keys K1
    // masked to -INFINITY. NNTR_SV_WIN=0 is the control arm (pass 0 = the old
    // full-range loop) and leaves the binary otherwise identical.
    int sv_lw = attn_sv_win_on() ? (int)local_window : 0;
    if (!kpu->SetKernelArguments(10, &sv_lw, sizeof(int)))
      return false;
    if (sv_buf) {
      const int vs = (int)v_buf_stride;
      if (!kpu->SetKernelArguments(11, &vs, sizeof(int)))
        return false;
    }
    // TDX=8 tiled: each WI computes 8 output channels, so the x grid is
    // head_dim/8 work-items. Workgroup (LWS_X x's, LWS_Y=4 m's).
    // The LWS is env-overridable + measured (NNTR_SV_LWS="x,y,z"). The current
    // default is {4,16,1} (see the re-sweep note below); originally
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
    // Parse NNTR_SV_LWS once. Unset/garbage => default {4,16,1}; "0,0,0" =>
    // NULL lws.
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
    blas_cc->command_queue_inst_.enqueueKernel(kpu->GetKernel(), 3, gws.data(),
                                               lws_ok ? lws.data() : nullptr, 0,
                                               nullptr, nullptr);
    if (attn_tprof) {
      double t = tnow();
      tp_k3 += t - t_a;
      t_a = t;
    }
  }

  attn_gpuhash_note("o", O_svm, o_clmem, (size_t)M * HD_Q * sizeof(uint16_t),
                    M);
  if (M > 8) {
    // prefill: O in 4-row blocks (all heads) and per head (all rows), so a
    // single event names its (query rows, head) pair
    for (unsigned int b0 = 0; b0 < M; b0 += 4u) {
      const unsigned int rows = std::min(4u, M - b0);
      attn_gpuhash_note2("oB", O_svm, o_clmem, (size_t)b0 * HD_Q,
                         (size_t)rows * HD_Q, 0, 1, M);
    }
    for (unsigned int h = 0; h < num_heads_Q; ++h)
      attn_gpuhash_note2("oH", O_svm, o_clmem, (size_t)h * head_dim, head_dim,
                         HD_Q, M, M);
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
// fused_row_attention_f16_ohwi_img (two_conv_attention.cl). Nothing in tree
// calls it at present.
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
  // register spill). Block-Q is the default flash variant; the naive 1-WI
  // variant (skeleton) remains reachable for A/B with NNTR_FLASH_BLOCKQ=0 and
  // neither NNTR_FLASH_COOP nor NNTR_FLASH_VEC set.
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
  static const int flash_blockq_cfg = []() {
    const char *e = std::getenv("NNTR_FLASH_BLOCKQ");
    if (e)
      return std::atoi(e) != 0 ? 1 : 0;
    // Default ON everywhere. Intel: Block-Q + subgroup-reduce is the
    // measured-best Intel attention (M=1024 ~2075 TPS vs vec-flash ~1119,
    // token-identical).
    //
    // Adreno used to default OFF here, on the reading that "Adreno uses the
    // image path anyway". That reading is now wrong in the case that matters.
    // Bit-identical output is the DEFAULT on the Adreno lane, and it is bought
    // by routing the FULL-attention layers off the image path onto these flash
    // kernels (Applications/CausalLM/kv_ring.h, imageAttnLayer). A
    // full-attention prefill CHUNK 0 is square and unwindowed, so
    // flash_needs_blockq below is false for it and it alone fell through to
    // flash_attention_prefill_f16_skeleton: one work-item per (head, query
    // row), scalar half loads, and a float[FLASH_MAX_D] q_row + acc pair in
    // private memory that spills. Measured on Adreno 840 (0.3B, d=64, 2047
    // keys, NNTR_MHA_ATTN_TIMER): 118.5 ms per full-attention prefill call
    // against 4.5 ms for a windowed layer on the image kernels. The skeleton
    // was never meant to be a production kernel -- its own header says
    // "CORRECTNESS-FIRST DESIGN ... Tiling/LDS cooperation is a follow-up".
    // Block-Q is that follow-up, it is already the arm every other Adreno
    // flash call takes (flash_needs_blockq fires for chunk >= 1, for every
    // windowed layer and for every ringed one), and its LDS-tree reduction is a
    // fixed order, so it is bit-reproducible run to run.
    return 1;
  }();
  // Block-Q (and XMX on top of it) is the ONLY variant that declares the chunk
  // offset (q_off = N_kv - M), the sliding window, the logit soft-cap and the
  // ring capacity. The coop / vec / skeleton kernels take none of them: their
  // causal bound is the chunk-LOCAL row (n <= m), and the host simply did not
  // bind the rest. Wherever Block-Q is not the configured default -- every
  // Adreno run that leaves the image path -- a second prefill chunk therefore
  // attended only to keys [0, m] instead of [0, q_off + m], a sliding layer
  // attended to every evicted key, and a ringed cache was read linearly: a
  // first token several logits off, a repetition loop, ids that move with the
  // chunk size, nothing logged. A call that needs any of those four is served
  // by Block-Q regardless of the configured variant.
  const bool flash_needs_blockq = (causal && N_kv != M) ||
                                  (local_window > 0 && local_window < N_kv) ||
                                  attn_softcap != 0.0f || ring_cap != 0;
  const int flash_blockq = (flash_blockq_cfg || flash_needs_blockq) ? 1 : 0;
  if (flash_needs_blockq && !flash_blockq_cfg) {
    static bool said = false;
    if (!said) {
      said = true;
      ml_logw("flash prefill: this call needs the chunk offset / window / "
              "soft-cap / ring (M=%u N_kv=%u window=%u ring=%u); using the "
              "Block-Q kernel instead of the configured variant, which has "
              "none of them",
              M, N_kv, local_window, ring_cap);
    }
  }
  // FBQ_TM: query rows per workgroup. Default 4 (=> acc+q 2*TM*VPL floats stays
  // in registers at LWS>=32). Only 1/2/4/8 supported.
  // Default by head width: at d<=128 the 2*TM*VPL row state still fits the
  // registers at TM=4 with LWS=16, measured -17% on the kernel (919-row
  // prefill, 16 heads: sliding 6.4 -> 5.3 ms/call, full 7.6 -> 6.6 ms/call)
  // with token-identical output. 0 = env unset.
  // d=256 also takes TM=4: the older TM=2 default there predates the LWS rule
  // that now follows head_dim alone (d=256 => LWS=32, VPL=8), and at LWS=32
  // the 2*TM*VPL = 64 acc/q floats per lane are still spill-free -- the offline
  // compiler (ocloc, ptl-h) reports no spill at FBQ_TM 1/2/4 and SIMD32 + 68
  // registers of spill only at TM=8. Measured on gemma4's 12 sliding d=256
  // calls over an 842-row prefill: TM2 4.909 -> TM4 4.419 ms/call (-10%),
  // TM8 5.850 (the spill); prefill 4029 -> 4107 TPS. TM is BIT-NEUTRAL here --
  // the per-row key walk order and the fp32 accumulation do not depend on it
  // and the d-dot still reduces over the same 32 lanes -- and the 1K cell's
  // output is unchanged, so this is a free win rather than a new reference.
  // d>=512 keeps TM=2 (LWS=64, VPL=8 => 2*TM*VPL grows with LWS).
  static const int flash_blockq_tm_env = []() {
    const char *e = std::getenv("NNTR_FLASH_BLOCKQ_TM");
    const int v = (e && std::atoi(e) > 0) ? std::atoi(e) : 0;
    return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 0;
  }();
  const int flash_blockq_tm = flash_blockq_tm_env
                                ? flash_blockq_tm_env
                                : (((int)head_dim <= 256) ? 4 : 2);
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
    // nntr_deterministic(): the variable, or its platform default when unset
    // (on on the Android gpu engine, see env_compat.h).
    if (nntr_deterministic())
      return 0;
    return v8c_use_buffer_path() ? 1 : 0; // Intel buffer path => 1
  }();
  // FLASH_COOP_LWS: WG size for the coop variant (work-items cooperating
  // over head_dim). Default: under Block-Q 16/32/64 by head_dim, otherwise
  // 16 on the Intel/buffer path and 64 elsewhere (see below). LDS footprint
  // is tiny (q_sh[d]+acc_sh[d]+red_sh[BLOCK_KV*LWS] ~ 1-3 KB), well within
  // Adreno's 32 KB at any LWS.
  // Must be a power of two (log-step reduction). Override via env.
  // NOTE (gemma4): NOT process-wide static — depends on the LIVE head_dim so a
  // model with TWO head_dims (gemma4: 256 sliding / 512 full) gets the right
  // VPL=d/LWS<=8 per call. The kernel cache (key = name+copts) still dedups the
  // compile per distinct (head_dim,...) so this stays a single compile per d.
  // flash_blockq is a plain local (it depends on this call's arguments), so
  // the lambda has to capture it. flash_blockq_tm depends on head_dim and is
  // deliberately not read here.
  const int flash_coop_lws = [head_dim, flash_blockq]() {
    const char *e = std::getenv("NNTR_FLASH_COOP_LWS");
    int v;
    if (e && std::atoi(e) > 0) {
      v = std::atoi(e);
    } else if (flash_blockq) {
      // Block-Q register budget = 2*TM*VPL acc/q floats per WI. TM<=2
      // keeps half8 (LWS=16, the Intel optimum + the SG subgroup size). The
      // older rule also widened to LWS=32 whenever TM>=4; at d=128 that was
      // measured slower (TM4/LWS32 126.8 ms vs TM4/LWS16 104.9 ms over 20
      // sliding calls), so the width now follows head_dim alone.
      // Gemma2 d=256: VPL must stay <=8, so LWS=d/8=32 (VPL=8, half8); at
      // TM=2 that is 2*2*8=32 acc/q floats — no spill. (d=128 keeps LWS=16.)
      // Gemma4 full_attention d=512: VPL must stay <=8 => LWS=d/8=64. A 64-wide
      // workgroup is valid on Intel for the LDS-tree path (reqd_work_group_size
      // only); the SG path is forced off for it below
      // (intel_reqd_sub_group_size (64) is INVALID — Intel subgroups are
      // {8,16,32}).
      v = ((int)head_dim >= 512) ? 64 : (((int)head_dim > 128) ? 32 : 16);
      // The Intel width turns out to be the right one on Adreno too, which is
      // worth recording because the opposite is the obvious guess: an Adreno
      // 840 wave is 64 lanes, so LWS=16 is a quarter-wave group and pays
      // log2(16)=4 LDS barrier rounds per key tile, and widening it to the wave
      // looks free. MEASURED on Adreno 840 (full-attention prefill at
      // 8103/16383 keys, NNTR_MHA_ATTN_TIMER, whole-run prefill ms), it is a
      // LOSS at both dense widths:
      //   d=64,  16383 keys: LWS 16 -> 22432 ms prefill, LWS 64 -> 27866
      //   d=128,  8103 keys: LWS 16 -> 16008 ms prefill, LWS 64 -> 17173
      // because VPL = d / LWS falls with the width: at d=64 LWS 64 means VPL 1,
      // i.e. a scalar 2-byte load per lane instead of a 8-byte vload4, and the
      // reduction tree grows to 6 barrier rounds. The occupancy does not pay
      // for the lost vector width. So the width stays as it is and
      // NNTR_FLASH_COOP_LWS remains the way to A/B it per device.
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
  // [determinism] Deliberately NOT extended to the Adreno
  // qcom_reqd_sub_group_size path, even though the kernel has one and it would
  // remove the LDS barrier tree. sub_group_reduce_add's internal summation
  // order is vendor-opaque (see the NNTR_DETERMINISTIC note at flash_blockq_sg
  // above), and on this lane bit-identical output is now the DEFAULT rather
  // than an opt-in, so an arm whose reduction order we cannot state is not
  // eligible to be the default. The width widening above buys the occupancy
  // without giving that up.
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
  // sliding-window calls with d<=128 take it too (NNTR_FLASH_XMX_WIN, see
  // xmx_win_ok below); wider heads keep the blockq + window-skip path.
  // NNTR_FLASH_XMX_CHECK=N additionally runs the blockq kernel first on the
  // same buffers and host-compares (first N calls).
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
  // Sliding-window calls. The XMX kernel is window-aware (tile-aligned window
  // floor + per-key mask). At d=128 it halves the windowed call
  // (919 rows, W=512, 16 heads: 5.0 -> 2.85 ms; 1556 rows, W=1024, 12 heads:
  // 12.3 -> 6.0 ms), so d<=128 takes it by DEFAULT.
  //
  // d=256 is no longer a SPEED question. It used to measure slower than
  // Block-Q + window-skip (842 rows, W=512: 5.7 vs 4.9 ms/call) purely because
  // FXA_NSG=1 spilled ~289 registers there; with the NSG=2 default below the
  // same call is 3.564 ms/call against a Block-Q baseline that is itself now
  // 4.419 (FBQ_TM=4) -- a 19% win, gemma4 842-row prefill 4107 -> ~4390 TPS.
  // What keeps d=256 on Block-Q is the OUTPUT: the DPAS A operand is fp16, so
  // routing 12 more layers through it forks gemma4's greedy 1K cell away from
  // the reference output (the first 8 tokens already
  // differ). d<=128 could be defaulted on because that model family had an
  // HF teacher-forced margin profile to re-gate against; gemma4 has only the
  // md5. Precondition for default-on at d=256: an HF-margin run for gemma4
  // showing the fork is a tie-break at the fp16 output floor, and a re-recorded
  // reference. Until then NNTR_FLASH_XMX_WIN=2 opts in and is now FAST.
  // Accuracy: scored against a host fp64 attention over the same fp16 Q/K/V
  // (NNTR_FLASH_XMX_CHECK_REF) both kernels sit on the fp16 OUTPUT rounding
  // floor (relF 2.06e-4); the XMX excess over it is 0.45e-4 with FXA_PREC=1
  // (0.68e-4 before). Note for whoever A/Bs this route on a whole model: an
  // early-step divergence on a 24-layer decoder traced to the OpenCL rms_norm
  // epsilon floor amplifying one-ULP differences, not to this kernel.
  // NNTR_FLASH_XMX_WIN=0 keeps windowed calls on Block-Q, =2 routes every d.
  static const int flash_xmx_win = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_WIN");
    return e ? std::atoi(e) : 1;
  }();
  const bool xmx_win_ok = win_i == 0 || flash_xmx_win >= 2 ||
                          (flash_xmx_win == 1 && (int)head_dim <= 128);
  const bool use_xmx = flash_blockq && flash_xmx_req && causal && xmx_win_ok &&
                       (int)head_dim % 16 == 0 && (int)head_dim <= 512 &&
                       ClContext::Global().caps().dpas;
  // DPAS row-tile (M dimension). TM=8 measured better than TM=4 at every d
  // incl. 512: the register budget (~FXA_TM*d*6B/16 per lane) spills at
  // TM8/d512, but
  // halving the per-visit DPAS count + tile count still nets -26% vs TM4
  // (gemma4 32K full-attn 147->108s). The d=512 kernel remains latency-bound
  // (~9ns/visit vs 0.56 at d=128 -- 16KB SLM V-tile caps residency and the
  // KCH=32 SLM VNNI gather serializes); the planned v2 is a dual-subgroup
  // d-split WG. NNTR_FLASH_XMX_TM=4/8/16 overrides for experiments.
  static const int xmx_tm_env = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_TM");
    const int v = e ? std::atoi(e) : 0;
    return (v == 4 || v == 8 || v == 16) ? v : 0;
  }();
  // TM=16 default: the kernel is K/V-bandwidth-bound (TM=4 A/B scales
  // traffic-proportionally), and 16 rows per K/V pass halves that traffic --
  // measured 32K full-attn -29% (gemma4 d=512/NSG4).
  const int xmx_tm = xmx_tm_env ? xmx_tm_env : 16;
  // v2: subgroups per WG, each owning a d-slice. Default 1, 4 at d>=512 and
  // 2 at windowed d=256 (see below). NNTR_FLASH_XMX_NSG=1/2/4 overrides for
  // A/B.
  static const int xmx_nsg_env = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_NSG");
    const int v = e ? std::atoi(e) : 0;
    return (v == 1 || v == 2 || v == 4) ? v : 0;
  }();
  // d=512 default NSG=4: slice chunk count drops to the d=128 register
  // envelope (qa 128B + acc 256B/lane, spill-free) and lane residency per WG
  // quadruples -- measured 79.1->50.2s full-attn vs NSG=2 (-37%), beating
  // even the exchange-free probe floor (55.4s).
  //
  // d=256 WINDOWED also needs NSG=2, and that -- not the window handling -- is
  // why this route used to lose to Block-Q at d=256. NSG=1 leaves
  // FXA_KCH_SUB = 16, so acc[TM][16] alone is 16*16 floats = 1 KB per lane;
  // ocloc (ptl-h) reports the d=256/TM16/NSG1 kernel "compiled SIMD16
  // allocated 256 regs and spilled around 289" after an IGC retry, against NO
  // spill at d=128/TM16 and 64 at d=512/TM16/NSG4. NSG=2 halves the slice to
  // KCH_SUB=8 -- the same register envelope as d=128 -- leaving 18 registers of
  // spill, and it doubles the lanes per WG, which matters because TM=16 at
  // d=256 only produces M/16 * heads workgroups (842 rows, 8 heads = 424
  // 16-lane threads on 32 Xe cores). Measured on gemma4's 12 sliding d=256
  // calls, 842-row prefill: Block-Q 4.909 ms/call, XMX NSG1/TM16 5.765
  // (the spill), XMX NSG2/TM16 3.564 (-27% vs Block-Q). Other points of the
  // sweep, same cell: NSG1/TM8 4.168, NSG1/TM4 6.450, NSG2/TM8 3.926.
  // Scoped to windowed calls so d=256 FULL attention (gemma2-class, XMX on by
  // default and validated at NSG=1) keeps its exact arithmetic: the psum
  // exchange changes the score reduction order, so NSG is NOT bit-neutral.
  int xmx_nsg = xmx_nsg_env
                  ? xmx_nsg_env
                  : (((int)head_dim >= 512)
                       ? 4
                       : (((int)head_dim == 256 && win_i > 0) ? 2 : 1));
  // Guard: FXA_KCH_SUB truncates silently when 16*NSG does not
  // divide head_dim (no current dim hits this; env overrides could).
  if ((int)head_dim % (16 * xmx_nsg) != 0)
    xmx_nsg = 1;
  // Exchange batching (NSG>1 only): key-tiles per psum exchange. Default 1
  // (see below); XB=2 is one WG-barrier pair per 32 keys, at +XB*NSG*TM*64B
  // SLM. NNTR_FLASH_XMX_XB = 1/2/4 overrides for A/B.
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
  // P rounding (the DPAS A operand is fp16): 1 = tile-normalised P and a
  // normaliser summed from the rounded weights; NNTR_FLASH_XMX_PREC=0 restores
  // P against the running max with the exact fp32 normaliser (A/B).
  static const int xmx_prec = []() {
    const char *e = std::getenv("NNTR_FLASH_XMX_PREC");
    return (e && e[0] == '0') ? 0 : 1;
  }();
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
              " -DFXA_XRED=" + std::to_string(xmx_xred) +
              " -DFXA_PREC=" + std::to_string(xmx_prec);
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
    // NNTR_FLASH_XMX_CHECK_REF=1: also score BOTH kernels against a host
    // fp64 attention over the same fp16 Q/K/V (a sample of rows, every head).
    // The two-kernel diff above cannot say which side is the accurate one.
    static const bool _xmx_check_ref =
      std::getenv("NNTR_FLASH_XMX_CHECK_REF") != nullptr;
    if (_xmx_check_ref) {
      const size_t d = head_dim;
      const size_t step = std::max<size_t>(1, (size_t)M / 48);
      double e_x = 0, e_b = 0, nrm = 0, mx_x = 0, mx_b = 0;
      std::vector<double> sc(N_kv), o(d);
      std::vector<size_t> rows;
      for (size_t m = 0; m + 1 < M; m += step)
        rows.push_back(m);
      rows.push_back((size_t)M - 1);
      size_t cnt = 0;
      for (size_t m : rows) {
        const long mabs = (long)m + (long)N_kv - (long)M;
        const long lo = (win_i > 0) ? std::max<long>(0, mabs - win_i + 1) : 0;
        for (size_t h = 0; h < num_heads_Q; ++h) {
          const size_t hk = h / (size_t)gqa;
          const uint16_t *qp = Q_host + m * HD_Q + h * d;
          double smax = -1e300;
          for (long n = lo; n <= mabs; ++n) {
            const size_t pn =
              ring_cap > 0 ? (size_t)n % ring_cap : (size_t)n; // physical row
            const uint16_t *kr = (k_stride > 0)
                                   ? K_host + hk * (size_t)k_stride * d + pn * d
                                   : K_host + pn * HD_KV + hk * d;
            double a = 0;
            for (size_t c = 0; c < d; ++c)
              a += (double)_h2f(qp[c]) * (double)_h2f(kr[c]);
            a *= (double)scale;
            if (attn_softcap > 0.0f)
              a = attn_softcap * std::tanh(a / attn_softcap);
            sc[n] = a;
            smax = std::max(smax, a);
          }
          double l = 0;
          std::fill(o.begin(), o.end(), 0.0);
          for (long n = lo; n <= mabs; ++n) {
            const size_t pn =
              ring_cap > 0 ? (size_t)n % ring_cap : (size_t)n; // physical row
            const uint16_t *vr = V_host + pn * HD_KV + hk * d;
            const double w = std::exp(sc[n] - smax);
            l += w;
            for (size_t c = 0; c < d; ++c)
              o[c] += w * (double)_h2f(vr[c]);
          }
          for (size_t c = 0; c < d; ++c) {
            const double r = o[c] / l;
            const size_t i = m * HD_Q + h * d + c;
            const double dx = (double)_h2f(O_host[i]) - r;
            const double db = (double)_h2f(_xmx_ref[i]) - r;
            e_x += dx * dx;
            e_b += db * db;
            nrm += r * r;
            mx_x = std::max(mx_x, std::fabs(dx));
            mx_b = std::max(mx_b, std::fabs(db));
          }
        }
        cnt += (size_t)num_heads_Q * d;
      }
      std::fprintf(
        stderr,
        "[FLASH-XMX-REF] call=%d win=%d rms_ref=%.4f | xmx relF=%.3e "
        "max=%.5f | blockq relF=%.3e max=%.5f\n",
        _xmx_check_done, win_i,
        std::sqrt(nrm / (double)std::max<size_t>(1, cnt)), std::sqrt(e_x / nrm),
        mx_x, std::sqrt(e_b / nrm), mx_b);
    }
    ++_xmx_check_done;
  }

  // No post-kernel drain: the flash output O is consumed by the next GPU op
  // (the o_proj FC), which the in-order queue already orders after this
  // kernel, and nothing reads O from the host here.
  // Buffer/flash-path stage hashes (NNTR_ATTN_GPUHASH=2 for
  // prefill calls): the rotated Q, the concat K/V cache rows this call reads
  // and the O it wrote, hashed on the GPU behind this launch and read back
  // only where the step already blocks. Inert (one static load) when unset.
  if (gh_mode() != 0) {
    const unsigned int rows =
      (ring_cap > 0 && ring_cap < N_kv) ? ring_cap : N_kv;
    attn_gpuhash_note("fq", Q_host, nullptr, (size_t)M * HD_Q * 2, M);
    if (k_stride == 0) {
      attn_gpuhash_note("fk", K_host, nullptr, (size_t)rows * HD_KV * 2, M);
      attn_gpuhash_note("fv", V_host, nullptr, (size_t)rows * HD_KV * 2, M);
    }
    attn_gpuhash_note("fo", O_host, nullptr, (size_t)M * HD_Q * 2, M);
  }
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
  // [window-clip] A sliding layer reads only keys [N_kv-W, N_kv). Launch the
  // chunks that window touches, not the whole context's: the kernel already
  // clipped each chunk's key walk to win_start, but the GRID (work-groups,
  // partial buffers, the reduce loop) still grew with N_kv -- at 17K context a
  // W=512 layer launched 267 chunks per head to read 9. The grid stays
  // anchored at key 0 (chunk_base is an index on it), so the surviving chunks
  // hold the same keys and reduce in the same order: bit-identical.
  // NNTR_FLASH_DEC_CLIP=0 restores the whole-context grid (A/B).
  static const bool dec_clip = []() {
    const char *e = std::getenv("NNTR_FLASH_DEC_CLIP");
    return !(e != nullptr && e[0] == '0');
  }();
  const int n_chunks_all = (int)((N_kv + chunk_kv - 1) / chunk_kv);
  const int chunk_base =
    (dec_clip && local_window > 0 && local_window < N_kv)
      ? (int)((N_kv - local_window) / (unsigned int)chunk_kv)
      : 0;
  const int n_chunks = n_chunks_all - chunk_base;

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
    int chunk_base_i = chunk_base;  // [window-clip] first live chunk
    if (!kp->SetKernelArguments(15, &ring_cap_i, sizeof(int)) ||
        !kp->SetKernelArguments(16, &chunk_base_i, sizeof(int)))
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
  // Decode stage hashes (NNTR_ATTN_GPUHASH=1): see the prefill
  // entry above. The K/V rows are the ring-resident ones.
  if (gh_mode() != 0) {
    const unsigned int rows =
      (ring_cap > 0 && ring_cap < N_kv) ? ring_cap : N_kv;
    attn_gpuhash_note("fq", Q_host, nullptr, (size_t)HD_Q * 2, 1);
    if (k_stride == 0) {
      attn_gpuhash_note("fk", K_host, nullptr, (size_t)rows * HD_KV * 2, 1);
      attn_gpuhash_note("fv", V_host, nullptr, (size_t)rows * HD_KV * 2, 1);
    }
    attn_gpuhash_note("fo", O_host, nullptr, (size_t)HD_Q * 2, 1);
  }
  return true;
}

// ===========================================================================
// [afd] Fused split-K flash-decode for M=1 (NNTR_ADRENO_FLASH_DECODE=1).
//
// The Adreno image arm decodes with three passes over the K/V image mirrors
// (qk_band -> softmax -> sv_tm2): a [hQ, 1, N_kv] fp16 score row is written,
// re-read, normalized and read again, and each pass walks keys serially. This
// is one pass over the fp16 SVM K/V cache instead:
//
//  - afd_partial: one SEGMENT of AFD_L lanes owns one (kv head, key block of
//    AFD_KPP keys). It loads each key's K row once (16 B per lane, AFD_L lanes
//    side by side = one coalesced row) and scores it against ALL AFD_G query
//    heads that share the kv head, keeps an online softmax (running max m and
//    sum l) per query head, and accumulates P*V in registers. The per-key dot
//    is reduced across the segment with a fixed xor butterfly of
//    sub_group_shuffle_xor: no __local tile and no barrier (the Adreno
//    0842.19.8 driver miscomputes __local tiles at >= 2 work-groups per SP).
//    It writes one unnormalized partial (m, l, acc[d]) per (block, q head).
//  - afd_merge: one work-item per (q head, 8 output elements) folds the
//    partials of its head in block order and writes O (fp16).
//
// Deterministic by construction: the block size, the lane split, the butterfly
// and the merge order are all fixed, so a given (N_kv, window, shape) always
// adds the same numbers in the same order. The order differs from the image
// arm's (and from flash_decode_f16_cl's), so outputs are close, not identical.
// A segment walks the same number of keys whatever its clip (masked keys get
// p = 0), which keeps every lane of a sub-group in the same control flow
// around the shuffles.
// ===========================================================================
static const std::string afd_kernel_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#if defined(AFD_INTEL_SG)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define AFD_SG_ATTR __attribute__((intel_reqd_sub_group_size(AFD_INTEL_SG)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define AFD_SG_ATTR __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define AFD_SG_ATTR
#endif

// -cl-finite-math-only is in the default options: no infinities. Masked
// scores use this sentinel and an explicit valid flag instead.
#define AFD_NEG (-1.0e30f)
#define AFD_NC (AFD_VPL / 8) // half8 chunks per lane
#define AFD_D (AFD_L * AFD_VPL)

inline float afd_hsum8(const float8 t) {
  return ((t.s0 + t.s1) + (t.s2 + t.s3)) + ((t.s4 + t.s5) + (t.s6 + t.s7));
}

// Fixed xor butterfly over the AFD_L lanes of a segment. Lane i and lane i^m
// add the same two values (a+b == b+a), so every lane ends with the same sum.
inline float afd_seg_sum(float p) {
#pragma unroll
  for (uint m = AFD_L >> 1; m > 0; m >>= 1)
    p += sub_group_shuffle_xor(p, m);
  return p;
}

AFD_SG_ATTR __attribute__((reqd_work_group_size(AFD_WG, 1, 1))) __kernel void
afd_probe(__global int *out) {
  if (get_local_id(0) == 0) {
    out[0] = (int)get_max_sub_group_size();
    out[1] = (int)get_sub_group_size();
    out[2] = (int)get_num_sub_groups();
  }
}

AFD_SG_ATTR __attribute__((reqd_work_group_size(AFD_WG, 1, 1))) __kernel void
afd_partial(__global const half *Q, __global const half *K,
            __global const half *V, __global float *part_acc,
            __global float *part_ml, const int N_kv, const int HD_KV,
            const float scale, const int win_start, const int ring_cap,
            const int part_base, const int n_parts, const int n_kv_heads) {
  // Logical lane in the work-group, from the sub-group ids (no assumption on
  // how the driver maps local ids onto sub-groups). The host checked that the
  // sub-group size is a multiple of AFD_L, so a segment never straddles two
  // sub-groups and lane = sub-group local id mod AFD_L.
  const int wl = (int)(get_sub_group_id() * get_max_sub_group_size() +
                       get_sub_group_local_id());
  const int seg = (int)get_group_id(0) * (AFD_WG / AFD_L) + wl / AFD_L;
  const int lane = (int)get_sub_group_local_id() % AFD_L;
  const bool seg_ok = seg < n_kv_heads * n_parts;
  const int sc = seg_ok ? seg : 0;
  const int head_kv = sc / n_parts;
  const int part_l = sc % n_parts;
  const int kb = (part_base + part_l) * AFD_KPP; // first key on the abs grid
  int n_lo = kb > win_start ? kb : win_start;
  int n_hi = min(kb + AFD_KPP, N_kv);
  if (!seg_ok)
    n_hi = n_lo; // no live key: the segment still walks, fully masked

  const long hoff = (long)head_kv * AFD_D;
  half8 qv[AFD_G][AFD_NC];
#pragma unroll
  for (int g = 0; g < AFD_G; ++g)
#pragma unroll
    for (int c = 0; c < AFD_NC; ++c)
      qv[g][c] =
        vload8(0, Q + (long)(head_kv * AFD_G + g) * AFD_D + (c * AFD_L + lane) * 8);

  float8 acc[AFD_G][AFD_NC];
  float m_i[AFD_G], l_i[AFD_G];
#pragma unroll
  for (int g = 0; g < AFD_G; ++g) {
    m_i[g] = AFD_NEG;
    l_i[g] = 0.0f;
#pragma unroll
    for (int c = 0; c < AFD_NC; ++c)
      acc[g][c] = (float8)(0.0f);
  }

  for (int t0 = kb; t0 < kb + AFD_KPP; t0 += AFD_T) {
    float s[AFD_T][AFD_G];
    bool live[AFD_T];
    long row[AFD_T];
#pragma unroll
    for (int j = 0; j < AFD_T; ++j) {
      const int n = t0 + j;
      live[j] = (n >= n_lo) && (n < n_hi);
      // A masked key still reads a real row (its score is discarded).
      const int nn = min(n, N_kv - 1);
      row[j] = (long)(ring_cap > 0 ? (nn % ring_cap) : nn) * HD_KV + hoff;
    }
#pragma unroll
    for (int j = 0; j < AFD_T; ++j) {
      float8 kf[AFD_NC];
#pragma unroll
      for (int c = 0; c < AFD_NC; ++c)
        kf[c] = convert_float8(vload8(0, K + row[j] + (c * AFD_L + lane) * 8));
#pragma unroll
      for (int g = 0; g < AFD_G; ++g) {
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < AFD_NC; ++c)
          p += afd_hsum8(convert_float8(qv[g][c]) * kf[c]);
        p = afd_seg_sum(p);
        s[j][g] = live[j] ? p * scale : AFD_NEG;
      }
    }
    float8 vf[AFD_T][AFD_NC];
#pragma unroll
    for (int j = 0; j < AFD_T; ++j)
#pragma unroll
      for (int c = 0; c < AFD_NC; ++c)
        vf[j][c] = convert_float8(vload8(0, V + row[j] + (c * AFD_L + lane) * 8));
#pragma unroll
    for (int g = 0; g < AFD_G; ++g) {
      float mt = m_i[g];
#pragma unroll
      for (int j = 0; j < AFD_T; ++j)
        mt = fmax(mt, s[j][g]);
      const float alpha = exp(m_i[g] - mt);
      float lsum = l_i[g] * alpha;
      float pj[AFD_T];
#pragma unroll
      for (int j = 0; j < AFD_T; ++j) {
        pj[j] = live[j] ? exp(s[j][g] - mt) : 0.0f;
        lsum += pj[j];
      }
#pragma unroll
      for (int c = 0; c < AFD_NC; ++c) {
        float8 a = acc[g][c] * alpha;
#pragma unroll
        for (int j = 0; j < AFD_T; ++j)
          a += pj[j] * vf[j][c];
        acc[g][c] = a;
      }
      l_i[g] = lsum;
      m_i[g] = mt;
    }
  }

  if (!seg_ok)
    return;
#pragma unroll
  for (int g = 0; g < AFD_G; ++g) {
    const long ph = ((long)head_kv * n_parts + part_l) * AFD_G + g;
#pragma unroll
    for (int c = 0; c < AFD_NC; ++c)
      vstore8(acc[g][c], 0, part_acc + ph * AFD_D + (c * AFD_L + lane) * 8);
    if (lane == 0) {
      part_ml[ph * 2 + 0] = m_i[g];
      part_ml[ph * 2 + 1] = l_i[g];
    }
  }
}

// One work-item per (q head, 8 output elements); partials folded in block
// order. O is fp16 [1, HD_Q].
__kernel void afd_merge(__global const float *part_acc,
                        __global const float *part_ml, __global half *O,
                        const int n_parts, const int n_q_heads) {
  const int gid = (int)get_global_id(0);
  const int per = AFD_D / 8;
  const int hq = gid / per;
  const int e = gid % per;
  if (hq >= n_q_heads)
    return;
  const int head_kv = hq / AFD_G;
  const int g = hq % AFD_G;
  float m_g = AFD_NEG;
  for (int p = 0; p < n_parts; ++p) {
    const long ph = ((long)head_kv * n_parts + p) * AFD_G + g;
    if (part_ml[ph * 2 + 1] > 0.0f)
      m_g = fmax(m_g, part_ml[ph * 2 + 0]);
  }
  float8 acc = (float8)(0.0f);
  float l_g = 0.0f;
  for (int p = 0; p < n_parts; ++p) {
    const long ph = ((long)head_kv * n_parts + p) * AFD_G + g;
    const float l_p = part_ml[ph * 2 + 1];
    if (l_p > 0.0f) {
      const float w = exp(part_ml[ph * 2 + 0] - m_g);
      acc += w * vload8(0, part_acc + ph * AFD_D + e * 8);
      l_g += w * l_p;
    }
  }
  const float inv = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;
  vstore8(convert_half8(acc * inv), 0, O + (long)hq * AFD_D + e * 8);
}
)CL";

namespace {
struct AfdScratch {
  void *acc = nullptr, *ml = nullptr, *probe = nullptr;
  size_t acc_cap = 0, ml_cap = 0, probe_cap = 0;
  // copts -> 1 usable / 0 refused (sub-group size does not fit, or the
  // program did not build). Probed once per variant.
  std::map<std::string, int> verdict;
};
AfdScratch &afd_scratch() {
  static AfdScratch s;
  return s;
}
} // namespace

// NNTR_ADRENO_FLASH_DECODE, value-checked (=0 is the opt-out). Unset, it is on
// exactly where the reproducible arm is on by default (the Android gpu engine,
// nntr_deterministic_default_on()) and that arm is in force: the mirror-free
// arm's decode then never reaches the host. An explicit NNTR_DETERMINISTIC=0
// there (the image arm) leaves it off, since the fused decode on the image arm
// is not the validated configuration. Off elsewhere unless set.
bool flash_decode_gqa_enabled() {
  static const bool on =
    nntr_env_on_or("NNTR_ADRENO_FLASH_DECODE",
                   nntr_deterministic_default_on() && nntr_deterministic());
  return on;
}

bool flash_decode_gqa_f16_cl(const uint16_t *Q_svm, const uint16_t *K_svm,
                             const uint16_t *V_svm, uint16_t *O_svm,
                             void *o_clmem, unsigned int N_kv,
                             unsigned int num_heads_Q,
                             unsigned int num_heads_KV, unsigned int head_dim,
                             float attn_softcap, unsigned int local_window,
                             unsigned int ring_cap) {
  if (num_heads_Q == 0 || num_heads_KV == 0 || head_dim == 0 || N_kv == 0)
    return false;
  if (num_heads_Q % num_heads_KV != 0 || head_dim % 8 != 0)
    return false;
  if (attn_softcap > 0.0f)
    return false; // no score cap in this kernel (Gemma4 has none)
  const unsigned int G = num_heads_Q / num_heads_KV;
  if (G > 16)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc == nullptr)
    return false;
  // Intel exposes required sub-group sizes {8,16,32}; Adreno runs a 64-lane
  // half wave. The segment width is d/8 lanes (one half8 per lane per key)
  // capped by the sub-group size.
  const bool intel_sg = ClContext::Global().caps().subgroups;
  const unsigned int l_cap = intel_sg ? 32u : 64u;
  unsigned int L = head_dim / 8;
  if (L > l_cap)
    L = l_cap;
  if (L < 8 || (L & (L - 1)) != 0 || head_dim % (8 * L) != 0 ||
      head_dim / (8 * L) > 4)
    return false;
  const unsigned int VPL = head_dim / L;
  const int kpp = head_dim >= 512 ? 64 : 32;
  const int T = (G >= 8) ? 2 : 4;
  const int WG = 64;

  std::string copts =
    "-DAFD_L=" + std::to_string(L) + " -DAFD_VPL=" + std::to_string(VPL) +
    " -DAFD_G=" + std::to_string(G) + " -DAFD_KPP=" + std::to_string(kpp) +
    " -DAFD_T=" + std::to_string(T) + " -DAFD_WG=" + std::to_string(WG);
  if (intel_sg)
    copts += " -DAFD_INTEL_SG=" + std::to_string(L < 16 ? 16 : L);

  AfdScratch &sc = afd_scratch();
  cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
  auto vit = sc.verdict.find(copts);
  if (vit != sc.verdict.end() && vit->second == 0)
    return false;
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(afd_kernel_src, "afd_partial", copts);
  ClContext::SharedPtrClKernel km =
    blas_cc->registerClKernel(afd_kernel_src, "afd_merge", copts);
  if (vit == sc.verdict.end()) {
    // First use of this variant: the segment layout needs every sub-group to
    // hold whole segments. One blocking probe per variant per process.
    int ok_v = 0;
    ClContext::SharedPtrClKernel kpr =
      blas_cc->registerClKernel(afd_kernel_src, "afd_probe", copts);
    if (kp && km && kpr &&
        ensure_cl_stage_buf(&sc.probe, &sc.probe_cap, 4 * sizeof(int))) {
      cl_mem pm = (cl_mem)sc.probe;
      int got[4] = {0, 0, 0, 0};
      std::array<size_t, 1> g1 = {(size_t)WG}, l1 = {(size_t)WG};
      bool rd = kpr->SetKernelArguments(0, &pm, sizeof(cl_mem));
      if (rd) {
        blas_cc->command_queue_inst_.enqueueKernel(
          kpr->GetKernel(), 1, g1.data(), l1.data(), 0, nullptr, nullptr);
        rd =
          opencl::clEnqueueReadBuffer(q, pm, CL_TRUE, 0, 3 * sizeof(int), got,
                                      0, nullptr, nullptr) == CL_SUCCESS;
      }
      if (rd) {
        const int sgmax = got[0];
        ok_v = (sgmax > 0 && sgmax % (int)L == 0) ? 1 : 0;
      }
      std::fprintf(stderr,
                   "[AFD] variant L=%u VPL=%u G=%u KPP=%d T=%d intel=%d: "
                   "sub-group max=%d size=%d num=%d -> %s\n",
                   L, VPL, G, kpp, T, (int)intel_sg, got[0], got[1], got[2],
                   ok_v ? "engaged" : "refused");
    } else {
      std::fprintf(stderr,
                   "[AFD] variant L=%u VPL=%u G=%u: program did not build -> "
                   "refused\n",
                   L, VPL, G);
    }
    sc.verdict[copts] = ok_v;
    if (!ok_v)
      return false;
  }
  if (!kp || !km)
    return false;

  const int win_start =
    (local_window > 0 && local_window < N_kv) ? (int)(N_kv - local_window) : 0;
  const int part_base = win_start / kpp;
  const int n_parts = (int)((N_kv + (unsigned int)kpp - 1) / kpp) - part_base;
  const int segs = (int)num_heads_KV * n_parts;
  const size_t acc_bytes = (size_t)segs * G * head_dim * sizeof(float);
  const size_t ml_bytes = (size_t)segs * G * 2 * sizeof(float);
  if (!ensure_cl_stage_buf(&sc.acc, &sc.acc_cap, acc_bytes) ||
      !ensure_cl_stage_buf(&sc.ml, &sc.ml_cap, ml_bytes))
    return false;
  cl_mem acc_m = (cl_mem)sc.acc, ml_m = (cl_mem)sc.ml;

  const float scale = 1.0f / std::sqrt((float)head_dim);
  int Ni = (int)N_kv, hkv = (int)(num_heads_KV * head_dim);
  int ws = win_start, rc = (int)ring_cap, pb = part_base, np = n_parts;
  int nkh = (int)num_heads_KV, nqh = (int)num_heads_Q;
  if (!kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_svm)) ||
      !kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_svm)) ||
      !kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(V_svm)) ||
      !kp->SetKernelArguments(3, &acc_m, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(4, &ml_m, sizeof(cl_mem)) ||
      !kp->SetKernelArguments(5, &Ni, sizeof(int)) ||
      !kp->SetKernelArguments(6, &hkv, sizeof(int)) ||
      !kp->SetKernelArguments(7, &scale, sizeof(float)) ||
      !kp->SetKernelArguments(8, &ws, sizeof(int)) ||
      !kp->SetKernelArguments(9, &rc, sizeof(int)) ||
      !kp->SetKernelArguments(10, &pb, sizeof(int)) ||
      !kp->SetKernelArguments(11, &np, sizeof(int)) ||
      !kp->SetKernelArguments(12, &nkh, sizeof(int)))
    return false;
  {
    const size_t lanes = (size_t)segs * L;
    std::array<size_t, 1> gws = {(lanes + WG - 1) / WG * WG};
    std::array<size_t, 1> lws = {(size_t)WG};
    blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  bool o_ok = km->SetKernelArguments(0, &acc_m, sizeof(cl_mem)) &&
              km->SetKernelArguments(1, &ml_m, sizeof(cl_mem));
  if (o_clmem != nullptr) {
    cl_mem oh = static_cast<cl_mem>(o_clmem);
    o_ok = o_ok && km->SetKernelArguments(2, &oh, sizeof(cl_mem));
  } else {
    o_ok = o_ok && km->SetKernelSVMArguments(2, O_svm);
  }
  o_ok = o_ok && km->SetKernelArguments(3, &np, sizeof(int)) &&
         km->SetKernelArguments(4, &nqh, sizeof(int));
  if (!o_ok)
    return false;
  {
    const size_t items = (size_t)num_heads_Q * (head_dim / 8);
    std::array<size_t, 1> gws = {(items + 63) / 64 * 64};
    std::array<size_t, 1> lws = {64};
    blas_cc->command_queue_inst_.enqueueKernel(km->GetKernel(), 1, gws.data(),
                                               lws.data(), 0, nullptr, nullptr);
  }
  if (gh_mode() != 0) {
    attn_gpuhash_note("aq", Q_svm, nullptr, (size_t)num_heads_Q * head_dim * 2,
                      1);
    attn_gpuhash_note("ao", O_svm, o_clmem, (size_t)num_heads_Q * head_dim * 2,
                      1);
  }
  return true;
}

// ===========================================================================
// [afp] Fused GQA flash prefill (NNTR_ADRENO_FLASH_PREFILL=1).
//
// The Adreno image arm runs prefill attention in three passes over the K/V
// image mirrors (qk_band -> softmax -> sv_tm2): an [hQ, M, N_kv] fp16 score
// tensor is written, re-read, normalized and read again, and every query head
// reads its kv head's K/V on its own (gqa times per kv head). This is one pass
// over the fp16 concat K/V cache instead.
//
// One work-group = ONE sub-group of AFP_NL lanes owns one (kv head, tile of
// AFP_TQ query rows) and serves all AFP_G query heads of that kv head, i.e.
// AFP_R = AFP_TQ * AFP_G (row, head) pairs; every K/V row is read once per
// tile for all of them. Keys are walked in blocks of AFP_KB:
//
//  - Q.K^T: a step covers AFP_NL / AFP_LPK keys; AFP_LPK neighbouring lanes
//    share one key and split its head dim (16 B loads, so AFP_LPK lanes read
//    16 * AFP_LPK contiguous bytes of the K row). Each lane forms partial dots
//    for the AFP_R pairs; a fixed xor butterfly (sub_group_shuffle_xor) over
//    the AFP_LPK lanes completes them. AFP_NSTEP steps make a block.
//  - online softmax per pair: the block max is a fixed xor butterfly over the
//    whole sub-group; the running sum is kept per lane and reduced once at the
//    end (same butterfly), so no per-block sum reduction.
//  - P.V: lanes switch to a column split: AFP_RS row groups x (AFP_NL /
//    AFP_RS) column slices of AFP_CPL columns, one coalesced V row per key.
//    The probability of a key reaches the lanes through sub_group_broadcast
//    (AFP_RS = 1) or sub_group_shuffle (AFP_RS > 1: the AFP_LPK lanes of a
//    key expose different row groups). P stays fp32, accumulation is fp32.
//
// No __local memory and no barrier anywhere (the Adreno 0842.19.8 driver
// miscomputes __local tiles at >= 2 work-groups per SP). Causal and sliding
// window masks are block skips plus a per-(key, pair) test inside the
// boundary blocks; masked keys get p = 0. Deterministic: fixed block size,
// fixed lane split, fixed butterflies and fixed key order for a given
// (M, N_kv, window) -- outputs are bit-identical run to run. Not identical to
// the image arm (which rounds scores and probabilities to fp16) nor to the
// flash/XMX arms (other summation orders).
// ===========================================================================
static const std::string afp_kernel_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#if defined(AFP_INTEL_SG)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define AFP_SG_ATTR __attribute__((intel_reqd_sub_group_size(AFP_INTEL_SG)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define AFP_SG_ATTR __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define AFP_SG_ATTR
#endif

// -cl-finite-math-only is in the default options: no infinities. A masked
// score is this sentinel; anything below AFP_NEG_T counts as masked.
#define AFP_NEG (-1.0e30f)
#define AFP_NEG_T (-1.0e29f)
#define AFP_R (AFP_TQ * AFP_G)          // (row, head) pairs per tile
#define AFP_KPS (AFP_NL / AFP_LPK)      // keys per Q.K step
#define AFP_KB (AFP_KPS * AFP_NSTEP)    // keys per block
#define AFP_NCH (AFP_D / (8 * AFP_LPK)) // half8 chunks per lane per key
#define AFP_RPL (AFP_R / AFP_RS)        // P.V rows per lane
#define AFP_CSL (AFP_NL / AFP_RS)       // P.V column slices
#define AFP_CPL (AFP_D / AFP_CSL)       // P.V columns per lane
#define AFP_C4 (AFP_CPL / 4)            // float4 accumulators per row

inline float afp_dot8(const float8 a, const float8 b, float acc) {
  acc = fma(a.s0, b.s0, acc);
  acc = fma(a.s1, b.s1, acc);
  acc = fma(a.s2, b.s2, acc);
  acc = fma(a.s3, b.s3, acc);
  acc = fma(a.s4, b.s4, acc);
  acc = fma(a.s5, b.s5, acc);
  acc = fma(a.s6, b.s6, acc);
  return fma(a.s7, b.s7, acc);
}

AFP_SG_ATTR __attribute__((reqd_work_group_size(AFP_NL, 1, 1))) __kernel void
afp_probe(__global int *out) {
  if (get_local_id(0) == 0) {
    out[0] = (int)get_max_sub_group_size();
    out[1] = (int)get_sub_group_size();
    out[2] = (int)get_num_sub_groups();
  }
}

AFP_SG_ATTR __attribute__((reqd_work_group_size(AFP_NL, 1, 1))) __kernel void
afp_prefill(__global const half *Q, __global const half *K,
            __global const half *V, __global half *O, const int M,
            const int N_kv, const int HD_Q, const int HD_KV, const float scale,
            const int causal, const int local_window, const int ring_cap,
            const int n_kv_heads) {
  const int lane = (int)get_sub_group_local_id();
  const int n_tiles = (M + AFP_TQ - 1) / AFP_TQ;
  const int gid = (int)get_group_id(0);
  // Heaviest (latest) causal tiles first, all kv heads interleaved.
  const int head_kv = gid % n_kv_heads;
  const int tile = n_tiles - 1 - gid / n_kv_heads;
  const int m0 = tile * AFP_TQ;
  const int q_off = N_kv - M; // absolute position of query row 0
  const int m_last = min(m0 + AFP_TQ, M) - 1;

  // Key range of the whole tile (block skip); per-pair masks below.
  int n_end = N_kv;
  if (causal)
    n_end = min(N_kv, q_off + m_last + 1);
  int n_beg = 0;
  if (local_window > 0) {
    n_beg = q_off + m0 - local_window + 1;
    if (n_beg < 0)
      n_beg = 0;
    n_beg = (n_beg / AFP_KB) * AFP_KB;
  }

  // Q.K lane split: key a of a step, d-slice sl.
  const int ka = lane / AFP_LPK;
  const int sl = lane % AFP_LPK;
  // P.V lane split: row group h, column slice cs.
  const int ph = lane / AFP_CSL;
  const int cs = lane % AFP_CSL;

  // Q row offsets per pair (rows past M are clamped; never stored).
  int q_base[AFP_R];
  int q_pos[AFP_R];
#pragma unroll
  for (int r = 0; r < AFP_R; ++r) {
    const int t = r / AFP_G, g = r % AFP_G;
    const int m = min(m0 + t, M - 1);
    q_base[r] = m * HD_Q + (head_kv * AFP_G + g) * AFP_D;
    q_pos[r] = q_off + m;
  }
  const int kv_head_off = head_kv * AFP_D;

  float m_i[AFP_R], l_i[AFP_R];
#pragma unroll
  for (int r = 0; r < AFP_R; ++r) {
    m_i[r] = AFP_NEG;
    l_i[r] = 0.0f;
  }
  float4 acc[AFP_RPL][AFP_C4];
#pragma unroll
  for (int i = 0; i < AFP_RPL; ++i)
#pragma unroll
    for (int c = 0; c < AFP_C4; ++c)
      acc[i][c] = (float4)(0.0f);

  for (int kb = n_beg; kb < n_end; kb += AFP_KB) {
    // ---- scores for AFP_NSTEP steps of AFP_KPS keys. Every Q chunk is
    // loaded once per block and used against the AFP_NSTEP keys of this
    // lane; each (step, pair) dot is one sequential fma chain.
    float s[AFP_NSTEP][AFP_R];
    long koff[AFP_NSTEP];
#pragma unroll
    for (int st = 0; st < AFP_NSTEP; ++st) {
      const int nn = min(kb + st * AFP_KPS + ka, N_kv - 1);
      const int prow = ring_cap > 0 ? (nn % ring_cap) : nn;
      koff[st] = (long)prow * HD_KV + kv_head_off + sl * 8;
#pragma unroll
      for (int r = 0; r < AFP_R; ++r)
        s[st][r] = 0.0f;
    }
#pragma unroll
    for (int c = 0; c < AFP_NCH; ++c) {
      float8 kf[AFP_NSTEP];
#pragma unroll
      for (int st = 0; st < AFP_NSTEP; ++st)
        kf[st] = convert_float8(vload8(0, K + koff[st] + c * AFP_LPK * 8));
#pragma unroll
      for (int r = 0; r < AFP_R; ++r) {
        const float8 qf =
          convert_float8(vload8(0, Q + q_base[r] + (c * AFP_LPK + sl) * 8));
#pragma unroll
        for (int st = 0; st < AFP_NSTEP; ++st)
          s[st][r] = afp_dot8(qf, kf[st], s[st][r]);
      }
    }
#pragma unroll
    for (int st = 0; st < AFP_NSTEP; ++st) {
      const int n = kb + st * AFP_KPS + ka;
#pragma unroll
      for (int r = 0; r < AFP_R; ++r) {
        float p = s[st][r];
#pragma unroll
        for (int msk = AFP_LPK >> 1; msk > 0; msk >>= 1)
          p += sub_group_shuffle_xor(p, (uint)msk);
        bool live = n < n_end;
        if (causal)
          live = live && (n <= q_pos[r]);
        if (local_window > 0)
          live = live && (n + local_window > q_pos[r]);
        s[st][r] = live ? p * scale : AFP_NEG;
      }
    }
    // ---- online softmax: block max (whole sub-group), rescale, p
    float alpha[AFP_R];
#pragma unroll
    for (int r = 0; r < AFP_R; ++r) {
      float mx = s[0][r];
#pragma unroll
      for (int st = 1; st < AFP_NSTEP; ++st)
        mx = fmax(mx, s[st][r]);
#pragma unroll
      for (int msk = AFP_NL >> 1; msk > 0; msk >>= 1)
        mx = fmax(mx, sub_group_shuffle_xor(mx, (uint)msk));
      const float m_new = fmax(m_i[r], mx);
      alpha[r] = (m_i[r] < AFP_NEG_T) ? 0.0f : exp(m_i[r] - m_new);
      float lsum = 0.0f;
#pragma unroll
      for (int st = 0; st < AFP_NSTEP; ++st) {
        const float pv = (s[st][r] < AFP_NEG_T) ? 0.0f : exp(s[st][r] - m_new);
        s[st][r] = pv;
        lsum += pv;
      }
      // one lane per key adds to the running sum
      l_i[r] = l_i[r] * alpha[r] + (sl == 0 ? lsum : 0.0f);
      m_i[r] = m_new;
    }
    // ---- P.V: this lane's rows (group ph) and columns (slice cs)
#pragma unroll
    for (int i = 0; i < AFP_RPL; ++i) {
      float a = alpha[i];
#pragma unroll
      for (int h = 1; h < AFP_RS; ++h)
        a = (ph == h) ? alpha[h * AFP_RPL + i] : a;
#pragma unroll
      for (int c = 0; c < AFP_C4; ++c)
        acc[i][c] *= a;
    }
#pragma unroll
    for (int st = 0; st < AFP_NSTEP; ++st) {
      // What this lane exposes to the P.V lanes: the probabilities of the
      // row group equal to its d-slice index (AFP_RS <= AFP_LPK).
      float x[AFP_RPL];
#pragma unroll
      for (int i = 0; i < AFP_RPL; ++i) {
        float v = s[st][i];
#pragma unroll
        for (int h = 1; h < AFP_RS; ++h)
          v = (sl == h) ? s[st][h * AFP_RPL + i] : v;
        x[i] = v;
      }
      for (int a = 0; a < AFP_KPS; ++a) {
        const int n = kb + st * AFP_KPS + a;
        const int nn = min(n, N_kv - 1);
        const int prow = ring_cap > 0 ? (nn % ring_cap) : nn;
        __global const half *vp =
          V + (long)prow * HD_KV + kv_head_off + cs * AFP_CPL;
        float4 vf[AFP_C4];
#pragma unroll
        for (int c = 0; c < AFP_C4; ++c)
          vf[c] = convert_float4(vload4(c, vp));
#pragma unroll
        for (int i = 0; i < AFP_RPL; ++i) {
#if AFP_RS == 1
          const float p = sub_group_broadcast(x[i], (uint)(a * AFP_LPK));
#else
          const float p = sub_group_shuffle(x[i], (uint)(a * AFP_LPK + ph));
#endif
#pragma unroll
          for (int c = 0; c < AFP_C4; ++c)
            acc[i][c] = fma((float4)(p), vf[c], acc[i][c]);
        }
      }
    }
  }

  // ---- final row sums (fixed butterfly), normalize, store
#pragma unroll
  for (int r = 0; r < AFP_R; ++r) {
    float l = l_i[r];
#pragma unroll
    for (int msk = AFP_NL >> 1; msk > 0; msk >>= 1)
      l += sub_group_shuffle_xor(l, (uint)msk);
    l_i[r] = l;
  }
#pragma unroll
  for (int i = 0; i < AFP_RPL; ++i) {
    int r = i;
    float l = l_i[i];
#pragma unroll
    for (int h = 1; h < AFP_RS; ++h) {
      r = (ph == h) ? (h * AFP_RPL + i) : r;
      l = (ph == h) ? l_i[h * AFP_RPL + i] : l;
    }
    const int t = r / AFP_G, g = r % AFP_G;
    const int m = m0 + t;
    if (m >= M)
      continue;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    __global half *op =
      O + (long)m * HD_Q + (head_kv * AFP_G + g) * AFP_D + cs * AFP_CPL;
#pragma unroll
    for (int c = 0; c < AFP_C4; ++c)
      vstore4(convert_half4(acc[i][c] * inv), c, op);
  }
}
)CL";

namespace {
struct AfpScratch {
  void *probe = nullptr;
  size_t probe_cap = 0;
  // copts -> 1 usable / 0 refused (sub-group layout does not hold, or the
  // program did not build). Probed once per variant.
  std::map<std::string, int> verdict;
};
AfpScratch &afp_scratch() {
  static AfpScratch s;
  return s;
}
} // namespace

bool flash_prefill_gqa_enabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_ADRENO_FLASH_PREFILL");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}

bool flash_prefill_gqa_f16_cl(const uint16_t *Q_svm, const uint16_t *K_svm,
                              const uint16_t *V_svm, uint16_t *O_svm,
                              void *o_clmem, unsigned int M, unsigned int N_kv,
                              unsigned int num_heads_Q,
                              unsigned int num_heads_KV, unsigned int head_dim,
                              bool causal, float attn_softcap,
                              unsigned int local_window,
                              unsigned int ring_cap) {
  if (M == 0 || N_kv == 0 || M > N_kv || num_heads_Q == 0 ||
      num_heads_KV == 0 || head_dim == 0)
    return false;
  if (num_heads_Q % num_heads_KV != 0 || head_dim % 32 != 0)
    return false;
  if (attn_softcap > 0.0f)
    return false; // no score cap in this kernel (Gemma4 has none)
  if (ring_cap > 0 && N_kv > ring_cap &&
      (local_window == 0 || local_window + M > ring_cap))
    return false; // a ring slot this call reads may already be overwritten
  const int G = (int)(num_heads_Q / num_heads_KV);
  if (G > 16)
    return false;

  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc == nullptr)
    return false;

  // Lanes per work-group = the sub-group: Adreno runs a 64-lane "half" wave
  // (qcom_reqd_sub_group_size); Intel takes a required size of 32, which is
  // how the same kernel is checked on Xe.
  const bool intel_sg = ClContext::Global().caps().subgroups;
  const int NL = intel_sg ? 32 : 64;
  // Tile rows: AFP_TQ * G pairs per tile, 8 for both Gemma4-26B layer kinds
  // (sliding G=2 -> 4 rows, full G=8 -> 1 row).
  const int TQ = (G >= 8) ? 1 : 8 / G;
  const int R = TQ * G;
  const int LPK = 4;
  const int NSTEP = 2;
  int RS = 2;
  auto pow2 = [](int v) { return v > 0 && (v & (v - 1)) == 0; };
  if (!pow2(LPK) || LPK > NL || (int)head_dim % (8 * LPK) != 0)
    return false;
  if (NSTEP < 1 || NSTEP > 8)
    return false;
  if (!pow2(RS) || RS > LPK || R % RS != 0)
    RS = 1;
  if (((int)head_dim * RS) % (NL * 4) != 0)
    RS = 1;
  if (((int)head_dim) % (NL * 4) != 0 && RS == 1)
    return false;
  if (R > 32)
    return false;

  std::string copts =
    "-DAFP_NL=" + std::to_string(NL) + " -DAFP_D=" + std::to_string(head_dim) +
    " -DAFP_G=" + std::to_string(G) + " -DAFP_TQ=" + std::to_string(TQ) +
    " -DAFP_LPK=" + std::to_string(LPK) +
    " -DAFP_NSTEP=" + std::to_string(NSTEP) + " -DAFP_RS=" + std::to_string(RS);
  if (intel_sg)
    copts += " -DAFP_INTEL_SG=" + std::to_string(NL);

  AfpScratch &sc = afp_scratch();
  auto vit = sc.verdict.find(copts);
  if (vit != sc.verdict.end() && vit->second == 0)
    return false;
  ClContext::SharedPtrClKernel kp =
    blas_cc->registerClKernel(afp_kernel_src, "afp_prefill", copts);
  if (vit == sc.verdict.end()) {
    // First use of this variant: the lane math needs the work-group to be
    // exactly one sub-group of NL lanes. One blocking probe per variant.
    int ok_v = 0;
    int got[4] = {0, 0, 0, 0};
    ClContext::SharedPtrClKernel kpr =
      blas_cc->registerClKernel(afp_kernel_src, "afp_probe", copts);
    if (kp && kpr &&
        ensure_cl_stage_buf(&sc.probe, &sc.probe_cap, 4 * sizeof(int))) {
      cl_mem pm = (cl_mem)sc.probe;
      cl_command_queue q = blas_cc->command_queue_inst_.GetCommandQueue();
      std::array<size_t, 1> g1 = {(size_t)NL}, l1 = {(size_t)NL};
      bool rd = kpr->SetKernelArguments(0, &pm, sizeof(cl_mem));
      if (rd) {
        blas_cc->command_queue_inst_.enqueueKernel(
          kpr->GetKernel(), 1, g1.data(), l1.data(), 0, nullptr, nullptr);
        rd =
          opencl::clEnqueueReadBuffer(q, pm, CL_TRUE, 0, 3 * sizeof(int), got,
                                      0, nullptr, nullptr) == CL_SUCCESS;
      }
      if (rd)
        ok_v = (got[1] == NL && got[2] == 1) ? 1 : 0;
      std::fprintf(stderr,
                   "[AFP] variant NL=%d d=%u G=%d TQ=%d LPK=%d NSTEP=%d RS=%d "
                   "intel=%d: sub-group max=%d size=%d num=%d -> %s\n",
                   NL, head_dim, G, TQ, LPK, NSTEP, RS, (int)intel_sg, got[0],
                   got[1], got[2], ok_v ? "engaged" : "refused");
    } else {
      std::fprintf(stderr,
                   "[AFP] variant NL=%d d=%u G=%d TQ=%d LPK=%d NSTEP=%d RS=%d: "
                   "program did not build -> refused\n",
                   NL, head_dim, G, TQ, LPK, NSTEP, RS);
    }
    sc.verdict[copts] = ok_v;
    if (!ok_v)
      return false;
  }
  if (!kp)
    return false;

  const float scale = 1.0f / std::sqrt((float)head_dim);
  int Mi = (int)M, Ni = (int)N_kv, hdq = (int)(num_heads_Q * head_dim),
      hdkv = (int)(num_heads_KV * head_dim), ca = causal ? 1 : 0;
  int win = (local_window > 0 && local_window < N_kv) ? (int)local_window : 0;
  int rc = (int)ring_cap, nkh = (int)num_heads_KV;
  bool ok = kp->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_svm)) &&
            kp->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_svm)) &&
            kp->SetKernelSVMArguments(2, const_cast<uint16_t *>(V_svm));
  if (o_clmem != nullptr) {
    cl_mem oh = static_cast<cl_mem>(o_clmem);
    ok = ok && kp->SetKernelArguments(3, &oh, sizeof(cl_mem));
  } else {
    ok = ok && kp->SetKernelSVMArguments(3, O_svm);
  }
  ok = ok && kp->SetKernelArguments(4, &Mi, sizeof(int)) &&
       kp->SetKernelArguments(5, &Ni, sizeof(int)) &&
       kp->SetKernelArguments(6, &hdq, sizeof(int)) &&
       kp->SetKernelArguments(7, &hdkv, sizeof(int)) &&
       kp->SetKernelArguments(8, &scale, sizeof(float)) &&
       kp->SetKernelArguments(9, &ca, sizeof(int)) &&
       kp->SetKernelArguments(10, &win, sizeof(int)) &&
       kp->SetKernelArguments(11, &rc, sizeof(int)) &&
       kp->SetKernelArguments(12, &nkh, sizeof(int));
  if (!ok)
    return false;
  const size_t n_tiles = (M + (unsigned int)TQ - 1) / (unsigned int)TQ;
  std::array<size_t, 1> gws = {n_tiles * num_heads_KV * (size_t)NL};
  std::array<size_t, 1> lws = {(size_t)NL};
  blas_cc->command_queue_inst_.enqueueKernel(kp->GetKernel(), 1, gws.data(),
                                             lws.data(), 0, nullptr, nullptr);
  if (gh_mode() != 0) {
    attn_gpuhash_note("aq", Q_svm, nullptr,
                      (size_t)M * num_heads_Q * head_dim * 2, M);
    attn_gpuhash_note("ao", O_svm, o_clmem,
                      (size_t)M * num_heads_Q * head_dim * 2, M);
  }
  return true;
}

// ===========================================================================
// [apa] Three-pass prefill attention over the fp16 concat K/V cache
// (NNTR_ADRENO_PREFILL_ATTN=1).
//
// The mirror-free arm (NNTR_DETERMINISTIC=1) has no K/V image mirrors, and on
// Adreno none of the buffer-arm prefill kernels takes Gemma4's d = 256 / 512:
// flash_attention_prefill_f16_cl declines d > 128 off the Intel buffer path,
// and two_conv_attention_prefill_f16_cl is gated off (and knows no window,
// chunk offset or softcap). The prefill attention therefore ran on the host.
//
// This is the image arm's own chain (qk_matmul_f16_ohwi_img_band ->
// softmax_row_band_f16 -> sv_matmul_f16_ohwi_img_tm2) with BUFFER operands:
//
//  - apa_qk_band: one thread computes APA_TM query rows x APA_TN keys of one
//    query head, fp32 dot over half8 chunks (dot(lo) + dot(hi) per chunk, as
//    the image kernel), scale, softcap, mask, fp32 score. K rows are read with
//    vload8 from the concat slab (row n at n * HD_KV + head_kv * d, ring-mapped
//    n % ring_cap), not from the OHWI K image. The grid covers each m-tile's
//    score band (the same tca_score_band() contract as the image arm).
//  - apa_softmax_band_f32: the image arm's softmax_row_band_f16 (band mode 1:
//    the row pair's band, 64-aligned floor, 64-wide LDS tree) over fp32
//    scores; e and p stay fp32.
//  - apa_sv_tm2: one thread computes the row pair (m0, m0 + 1) x 8 output
//    channels; per 8-key block it loads the two score half8s and the 8 keys'
//    V half8s (channels x0..x0+7 of each V row, coalesced across the x
//    threads) and accumulates in fp32. It reads only cells inside the pair's
//    band, like the image kernel.
//
// fp32 scores, not the image arm's fp16 ones. Gemma4's scaled scores reach
// ~25, where an fp16 score is off by up to 2^-7 and every probability by up
// to ~0.8 %. Measured on the 1K P4 cell (Adreno 840, layer prefill dumps
// against an f64 recomputation from the same Q/K/V): the fp16 chain's O is off
// by up to 2.2e-2..4.3e-2 (mean 1.4e-4..3.3e-4) -- exactly the fp16-score
// family's own error, and still half the host fp16 path's -- and that was
// enough to send the greedy 26B-A4B continuation into a repetition loop with
// no end of turn. With fp32 scores O sits at the fp16 output rounding floor
// (max 3.9e-3..7.8e-3 for |O| up to 21, mean 5e-5..8e-5) and the text ends
// cleanly. NNTR_APA_F32=0 keeps the fp16 chain (the image arm's numerics and
// its own softmax_row_band_f16) for A/B.
//
// Masked scores are written as -65504 (fp16 lowest) instead of -INFINITY: the
// default options carry -cl-finite-math-only, and exp(-65504 - max) is +0.0
// in fp32 for every real row max, so the softmax output is the same.
//
// Every output element depends only on its own row's inputs in a fixed order,
// so the result is bit-identical run to run and also independent of how the
// rows are split into launches (the scores scratch is bounded by
// NNTR_APA_SCORES_MB; a split starts on a multiple of 64 rows, which keeps the
// qk tiles and the softmax/sv row pairs where they are). No image read (the
// slab is written through buffers, and an image view of it would reopen the
// stale-texel hazard this driver has), no sub-group operation, no __local
// memory in the qk/sv K loops. The softmax's row reduction is the image arm's
// LDS tree, unchanged.
// ===========================================================================
static const std::string apa_kernel_src = R"CL(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef APA_TM
#define APA_TM 4
#endif
#ifndef APA_TN
#define APA_TN 8
#endif
// = SOFTMAX_LWS of softmax_row_band_f16: the band floor alignment.
#define APA_BAND_ALIGN 64
#define APA_MASKED (-65504.0f)
// APA_F32 (the default): fp32 scores / probabilities in the scratch, softmax
// apa_softmax_band_f32. Without it: fp16, the image arm's numerics.
#ifdef APA_F32
typedef float apa_s_t;
typedef float8 apa_s8_t;
#define APA_S8_F(x) (x)
#else
typedef half apa_s_t;
typedef half8 apa_s8_t;
#define APA_S8_F(x) convert_float8(x)
#endif

// Same integer arithmetic as tca_score_band() in two_conv_attention.cl.
static inline void apa_score_band(const int M, const int N_kv,
                                  const int causal, const int local_window,
                                  const int pair_lo, const int pair_hi,
                                  const int align_lo, int *n_lo, int *n_hi) {
  int lo = 0;
  int hi = N_kv;
  if (causal) {
    const int q_off = N_kv - M;
    int tex_hi = (N_kv + 7) >> 3;
    const int tex_hi_causal = ((q_off + pair_hi) >> 3) + 1;
    if (tex_hi_causal < tex_hi)
      tex_hi = tex_hi_causal;
    if (local_window > 0) {
      const int n_first = q_off + pair_lo - local_window + 1;
      int t0 = 0;
      if (n_first > 0)
        t0 = n_first >> 3;
      if (t0 > tex_hi)
        t0 = tex_hi;
      lo = t0 << 3;
      if (align_lo > 1)
        lo = lo & ~(align_lo - 1);
    }
    hi = tex_hi << 3;
    if (hi > N_kv)
      hi = N_kv;
  }
  if (lo > hi)
    lo = hi;
  *n_lo = lo;
  *n_hi = hi;
}

static inline int apa_row(const int n, const int ring_cap) {
  return ring_cap > 0 ? n % ring_cap : n;
}

// The read-only operands are `const restrict` (they never alias the output),
// which lets the compiler take its read-only load path for them.
// gws = (n-tiles of the widest band, ceil(M / APA_TM), num_heads_Q)
__kernel void apa_qk_band(__global const half *restrict Q,
                          __global const half *restrict K,
                          __global apa_s_t *restrict scores, const int M,
                          const int N_kv,
                          const int q_row0, const int d, const int HD_Q,
                          const int HD_KV, const int gqa, const int causal,
                          const float scale, const float softcap,
                          const int local_window, const int ring_cap) {
  const int m0 = (int)get_global_id(1) * APA_TM;
  const int head_q = (int)get_global_id(2);
  const int head_kv = head_q / gqa;
  if (m0 >= M)
    return;
  int band_lo, band_hi;
  {
    const int mlast = (m0 + APA_TM - 1 < M - 1) ? (m0 + APA_TM - 1) : (M - 1);
    apa_score_band(M, N_kv, causal, local_window, m0, mlast, APA_BAND_ALIGN,
                   &band_lo, &band_hi);
  }
  const int n0 = band_lo + (int)get_global_id(0) * APA_TN;
  if (n0 >= band_hi || n0 >= N_kv)
    return;
  const int q_off = N_kv - M;
  const long sb = (long)head_q * (long)M * (long)N_kv;

  // Whole tile masked (above the causal diagonal, or below every row's
  // window floor): no dot products.
  if (causal && (n0 > q_off + m0 + (APA_TM - 1) ||
                 (local_window > 0 &&
                  (n0 + APA_TN - 1) + local_window <= q_off + m0))) {
#pragma unroll
    for (int i = 0; i < APA_TM; i++) {
      const int m = m0 + i;
      if (m >= M)
        continue;
#pragma unroll
      for (int j = 0; j < APA_TN; j++) {
        const int n = n0 + j;
        if (n < N_kv)
          scores[sb + (long)m * N_kv + n] = (apa_s_t)APA_MASKED;
      }
    }
    return;
  }

  // Row offsets (rows past M / keys past N_kv are clamped to valid ones and
  // never stored).
  int qoff[APA_TM];
#pragma unroll
  for (int i = 0; i < APA_TM; i++) {
    const int m = (m0 + i < M) ? (m0 + i) : (M - 1);
    qoff[i] = (q_row0 + m) * HD_Q + head_q * d;
  }
  int koff[APA_TN];
#pragma unroll
  for (int j = 0; j < APA_TN; j++) {
    const int n = (n0 + j < N_kv) ? (n0 + j) : (N_kv - 1);
    koff[j] = apa_row(n, ring_cap) * HD_KV + head_kv * d;
  }

  float acc[APA_TM][APA_TN];
#pragma unroll
  for (int i = 0; i < APA_TM; i++)
#pragma unroll
    for (int j = 0; j < APA_TN; j++)
      acc[i][j] = 0.0f;

  for (int x = 0; x < d; x += 8) {
    half8 qp[APA_TM];
    half8 kp[APA_TN];
#pragma unroll
    for (int i = 0; i < APA_TM; i++)
      qp[i] = vload8(0, Q + qoff[i] + x);
#pragma unroll
    for (int j = 0; j < APA_TN; j++)
      kp[j] = vload8(0, K + koff[j] + x);
#pragma unroll
    for (int i = 0; i < APA_TM; i++) {
      const float4 qlo = convert_float4(qp[i].s0123);
      const float4 qhi = convert_float4(qp[i].s4567);
#pragma unroll
      for (int j = 0; j < APA_TN; j++) {
        const float4 klo = convert_float4(kp[j].s0123);
        const float4 khi = convert_float4(kp[j].s4567);
        acc[i][j] += dot(qlo, klo) + dot(qhi, khi);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < APA_TM; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < APA_TN; j++) {
      const int n = n0 + j;
      if (n >= N_kv)
        continue;
      float v = acc[i][j] * scale;
      if (softcap > 0.0f)
        v = softcap * tanh(v / softcap);
      if (causal && (n > q_off + m ||
                     (local_window > 0 && n + local_window <= q_off + m)))
        v = APA_MASKED;
      scores[sb + (long)m * N_kv + n] = (apa_s_t)v;
    }
  }
}

#ifdef APA_F32
// The image arm's softmax_row_band_f16 (band mode 1: the row pair's band with
// a 64-aligned floor, a 64-wide LDS tree for the max and the sum) over fp32
// scores: e = exp(s - max) and p = e / sum stay fp32. gws = (64, M, H).
__attribute__((reqd_work_group_size(APA_BAND_ALIGN, 1, 1))) __kernel void
apa_softmax_band_f32(__global float *scores, const int M, const int N_kv,
                     const int causal, const int local_window) {
  const int tid = get_local_id(0);
  const int m = get_group_id(1);
  const int h = get_group_id(2);
  if (m >= M)
    return;
  __global float *row = scores + (long)h * (long)M * N_kv + (long)m * N_kv;
  int n_lo, n_hi;
  {
    const int mlo = m & ~1;
    const int mhi = (mlo + 1 < M) ? (mlo + 1) : (M - 1);
    apa_score_band(M, N_kv, causal, local_window, mlo, mhi, APA_BAND_ALIGN,
                   &n_lo, &n_hi);
  }
  __local float lscratch[APA_BAND_ALIGN];
  float pmax = APA_MASKED;
  for (int n = n_lo + tid; n < n_hi; n += APA_BAND_ALIGN)
    pmax = fmax(pmax, row[n]);
  lscratch[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = APA_BAND_ALIGN / 2; s > 0; s >>= 1) {
    if (tid < s)
      lscratch[tid] = fmax(lscratch[tid], lscratch[tid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_max = lscratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  float psum = 0.0f;
  for (int n = n_lo + tid; n < n_hi; n += APA_BAND_ALIGN) {
    const float e = exp(row[n] - row_max);
    row[n] = e;
    psum += e;
  }
  lscratch[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = APA_BAND_ALIGN / 2; s > 0; s >>= 1) {
    if (tid < s)
      lscratch[tid] += lscratch[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_sum = lscratch[0];
  const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
  for (int n = n_lo + tid; n < n_hi; n += APA_BAND_ALIGN)
    row[n] = row[n] * inv;
}
#endif

// gws = (d / 8, ceil(M / 2), num_heads_Q)
__kernel void apa_sv_tm2(__global const apa_s_t *restrict scores,
                         __global const half *restrict V,
                         __global half *restrict O, const int M, const int N_kv,
                         const int q_row0, const int d, const int HD_Q,
                         const int HD_KV, const int gqa, const int causal,
                         const int local_window, const int ring_cap) {
  const int x0 = (int)get_global_id(0) * 8;
  const int m0 = (int)get_global_id(1) * 2;
  const int head_q = (int)get_global_id(2);
  if (m0 >= M || x0 >= d)
    return;
  const int m1 = m0 + 1;
  const int has1 = (m1 < M);
  const int head_kv = head_q / gqa;
  const long sb0 = (long)head_q * (long)M * (long)N_kv + (long)m0 * (long)N_kv;
  const long sb1 = sb0 + (has1 ? (long)N_kv : 0L);
  // Key range of the row pair: the causal cap of m1 and the window floor of
  // m0, in 8-key blocks -- the cells the pair's softmax band normalised.
  int n_tex1 = (N_kv + 7) >> 3;
  int n_tex0 = 0;
  if (causal) {
    const int q_off = N_kv - M;
    const int cap_m = has1 ? m1 : m0;
    const int tex_causal = ((q_off + cap_m) >> 3) + 1;
    if (tex_causal < n_tex1)
      n_tex1 = tex_causal;
    if (local_window > 0) {
      const int n_first = q_off + m0 - local_window + 1;
      if (n_first > 0)
        n_tex0 = n_first >> 3;
      if (n_tex0 > n_tex1)
        n_tex0 = n_tex1;
    }
  }
  const int vcol = head_kv * d + x0;

  float8 acc0 = (float8)(0.0f), acc1 = (float8)(0.0f);
  for (int t = n_tex0; t < n_tex1; t++) {
    const int n0 = t * 8;
    apa_s8_t s0, s1;
    if (n0 + 8 <= N_kv) {
      s0 = vload8(0, scores + sb0 + n0);
      s1 = vload8(0, scores + sb1 + n0);
    } else {
      // Tail block: no private array (this driver demotes dynamically
      // addressed private arrays to __local), element by element.
#define APA_S(sb, k)                                                           \
  ((n0 + (k) < N_kv) ? scores[(sb) + n0 + (k)] : (apa_s_t)0)
      s0 = (apa_s8_t)(APA_S(sb0, 0), APA_S(sb0, 1), APA_S(sb0, 2),
                      APA_S(sb0, 3), APA_S(sb0, 4), APA_S(sb0, 5),
                      APA_S(sb0, 6), APA_S(sb0, 7));
      s1 = (apa_s8_t)(APA_S(sb1, 0), APA_S(sb1, 1), APA_S(sb1, 2),
                      APA_S(sb1, 3), APA_S(sb1, 4), APA_S(sb1, 5),
                      APA_S(sb1, 6), APA_S(sb1, 7));
#undef APA_S
    }
    const int nl = N_kv - 1;
    const float8 v0 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 0, nl), ring_cap) * HD_KV + vcol));
    const float8 v1 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 1, nl), ring_cap) * HD_KV + vcol));
    const float8 v2 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 2, nl), ring_cap) * HD_KV + vcol));
    const float8 v3 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 3, nl), ring_cap) * HD_KV + vcol));
    const float8 v4 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 4, nl), ring_cap) * HD_KV + vcol));
    const float8 v5 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 5, nl), ring_cap) * HD_KV + vcol));
    const float8 v6 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 6, nl), ring_cap) * HD_KV + vcol));
    const float8 v7 = convert_float8(
      vload8(0, V + apa_row(min(n0 + 7, nl), ring_cap) * HD_KV + vcol));
    const float8 p0 = APA_S8_F(s0);
    const float8 p1 = APA_S8_F(s1);
    acc0 += ((p0.s0 * v0 + p0.s1 * v1) + (p0.s2 * v2 + p0.s3 * v3)) +
            ((p0.s4 * v4 + p0.s5 * v5) + (p0.s6 * v6 + p0.s7 * v7));
    acc1 += ((p1.s0 * v0 + p1.s1 * v1) + (p1.s2 * v2 + p1.s3 * v3)) +
            ((p1.s4 * v4 + p1.s5 * v5) + (p1.s6 * v6 + p1.s7 * v7));
  }
  vstore8(convert_half8(acc0), 0, O + (long)(q_row0 + m0) * HD_Q +
                                     (long)head_q * d + x0);
  if (has1)
    vstore8(convert_half8(acc1), 0, O + (long)(q_row0 + m1) * HD_Q +
                                       (long)head_q * d + x0);
}
)CL";

namespace {
struct ApaScratch {
  // copts -> kernel handles (registerClKernel's own lookup is not free on a
  // per-layer call), and the shapes already announced.
  std::map<std::string, ClContext::SharedPtrClKernel> qk, sv, sm;
  std::map<std::string, int> verdict; // copts -> 1 built / 0 did not build
  std::map<std::string, int> logged;  // shape key -> announced
};
ApaScratch &apa_scratch() {
  static ApaScratch s;
  return s;
}
int apa_env_int(const char *name, int dflt) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] != '\0') ? std::atoi(e) : dflt;
}
} // namespace

// NNTR_ADRENO_PREFILL_ATTN, value-checked (=0 is the opt-out). Unset, it has
// the same default as NNTR_ADRENO_FLASH_DECODE above: on where the
// reproducible arm is on by default and in force (the Android gpu engine), so
// the mirror-free arm's prefill attention runs on the GPU instead of the host.
bool prefill_attn_apa_enabled() {
  static const bool on =
    nntr_env_on_or("NNTR_ADRENO_PREFILL_ATTN",
                   nntr_deterministic_default_on() && nntr_deterministic());
  return on;
}

bool prefill_attn_apa_f16_cl(const uint16_t *Q_svm, const uint16_t *K_svm,
                             const uint16_t *V_svm, uint16_t *O_svm,
                             void *o_clmem, unsigned int M, unsigned int N_kv,
                             unsigned int num_heads_Q,
                             unsigned int num_heads_KV, unsigned int head_dim,
                             bool causal, float attn_softcap,
                             unsigned int local_window, unsigned int ring_cap) {
  if (M == 0 || N_kv == 0 || M > N_kv || num_heads_Q == 0 ||
      num_heads_KV == 0 || head_dim == 0)
    return false;
  if (num_heads_Q % num_heads_KV != 0 || head_dim % 8 != 0)
    return false;
  const unsigned int win =
    (local_window > 0 && local_window < N_kv) ? local_window : 0u;
  // The band reads keys from a 64-aligned floor below the first row's window
  // floor (their scores are masked) and the newest key of the call; with a
  // ring every one of those slots must still hold its own row or a row the
  // mask zeroes -- the same refusal as the fused prefill.
  if (ring_cap > 0 && N_kv > ring_cap && (win == 0 || win + M > ring_cap))
    return false;
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  if (blas_cc == nullptr)
    return false;

  // Each qk work-item computes TM query rows x TN keys.
  const int TM = 4;
  const int TN = 8;
  // Work-group shapes from a screen on the Adreno 840 at the 1K cell (wall
  // time of the 25 sliding + 5 full calls, fp32 scores): qk 4 key tiles x 16
  // row tiles and sv 32 channel groups x 2 row pairs give ~510-525 ms against
  // ~740-750 ms for 8,8,1 / 8,8,1 (the image arm's shapes). sv takes 32
  // channel groups only when d / 8 has them; the result does not depend on the
  // shapes (every element is computed in a fixed order by one work-item).
  const std::array<size_t, 3> qk_lws = {4, 16, 1};
  const std::array<size_t, 3> sv_lws = head_dim / 8 >= 32
                                         ? std::array<size_t, 3>{32, 2, 1}
                                         : std::array<size_t, 3>{8, 8, 1};
  // Scores scratch budget (MiB). Read per call so a test can split one call
  // into several row blocks; the result does not depend on the split.
  long budget_mb = apa_env_int("NNTR_APA_SCORES_MB", 48);
  if (budget_mb <= 0)
    budget_mb = 48;
  const size_t budget = (size_t)budget_mb << 20;

  // fp32 scores (default; NNTR_APA_F32=0 = the image arm's fp16 scores and
  // its softmax_row_band_f16). Read per call, like the budget, so a test can
  // run both.
  const bool f32 = apa_env_int("NNTR_APA_F32", 1) != 0;
  const std::string copts = "-DAPA_TM=" + std::to_string(TM) +
                            " -DAPA_TN=" + std::to_string(TN) +
                            (f32 ? " -DAPA_F32" : "");
  const std::string sm_copts = f32 ? copts : tca_copts();
  const size_t s_bytes = f32 ? 4u : 2u;
  constexpr unsigned int SOFTMAX_LWS = 64; // softmax_row_band_f16

  std::lock_guard<std::mutex> lock(tca_mtx());
  ApaScratch &as = apa_scratch();
  {
    auto vit = as.verdict.find(copts);
    if (vit != as.verdict.end() && vit->second == 0)
      return false;
  }
  ClContext::SharedPtrClKernel &kqk = as.qk[copts];
  ClContext::SharedPtrClKernel &ksv = as.sv[copts];
  ClContext::SharedPtrClKernel &ksm = as.sm[sm_copts];
  if (!kqk)
    kqk = blas_cc->registerClKernel(apa_kernel_src, "apa_qk_band", copts);
  if (!ksv)
    ksv = blas_cc->registerClKernel(apa_kernel_src, "apa_sv_tm2", copts);
  if (!ksm)
    ksm = f32 ? blas_cc->registerClKernel(apa_kernel_src,
                                          "apa_softmax_band_f32", sm_copts)
              : blas_cc->registerClKernel(two_conv_attention_kernel,
                                          "softmax_row_band_f16", sm_copts);
  if (!kqk || !ksv || !ksm) {
    if (as.verdict.find(copts) == as.verdict.end())
      std::fprintf(stderr,
                   "[APA] TM=%d TN=%d: program did not build (qk=%d sv=%d "
                   "softmax=%d) -> refused\n",
                   TM, TN, (int)(bool)kqk, (int)(bool)ksv, (int)(bool)ksm);
    as.verdict[copts] = 0;
    return false;
  }
  as.verdict[copts] = 1;

  // Row blocks: the whole call when its scores fit the budget, else blocks of
  // a multiple of 64 rows (the last block has the widest key span N_kv).
  const size_t row_bytes = (size_t)num_heads_Q * (size_t)N_kv * s_bytes;
  unsigned int rows_per = M;
  if (row_bytes * (size_t)M > budget) {
    rows_per = (unsigned int)(budget / row_bytes) & ~63u;
    if (rows_per < 64u)
      rows_per = 64u;
    if (rows_per > M)
      rows_per = M;
  }
  TcaScratch &sc = tca_scratch();
  cl_context ctx = blas_cc->context_inst_.GetContextNoRetain();
  if (!tca_ensure(ctx, &sc.scores, &sc.scores_bytes,
                  row_bytes * (size_t)rows_per, CL_MEM_READ_WRITE))
    return false;
  cl_mem scores_buf = sc.scores;

  const int d = (int)head_dim, HDq = (int)(num_heads_Q * head_dim),
            HDkv = (int)(num_heads_KV * head_dim),
            gqa = (int)(num_heads_Q / num_heads_KV), ca = causal ? 1 : 0,
            lw = (int)win, rc = (int)ring_cap, bmode = 1;
  const float scale = 1.0f / std::sqrt((float)head_dim);
  const unsigned int q_off_all = N_kv - M;

  auto pad = [](size_t v, size_t l) { return ((v + l - 1) / l) * l; };
  bool ok = true;
  unsigned int blocks = 0;
  for (unsigned int a = 0; ok && a < M; a += rows_per) {
    const unsigned int b = std::min(M, a + rows_per);
    const int Ms = (int)(b - a), Ns = (int)(q_off_all + b), q0 = (int)a;
    ++blocks;
    // ---- qk over each m-tile's score band.
    {
      size_t nx = 0;
      for (int m0 = 0; m0 < Ms; m0 += TM) {
        const int ml = std::min(m0 + TM - 1, Ms - 1);
        int blo = 0, bhi = 0;
        tca_score_band_host(Ms, Ns, ca, lw, m0, ml, (int)SOFTMAX_LWS, &blo,
                            &bhi);
        if (bhi > blo)
          nx = std::max(nx, (size_t)((bhi - blo + TN - 1) / TN));
      }
      if (nx == 0)
        nx = 1;
      const size_t mx = ((size_t)Ms + TM - 1) / TM;
      std::array<size_t, 3> gws = {pad(nx, qk_lws[0]), pad(mx, qk_lws[1]),
                                   pad(num_heads_Q, qk_lws[2])};
      ok = kqk->SetKernelSVMArguments(0, const_cast<uint16_t *>(Q_svm)) &&
           kqk->SetKernelSVMArguments(1, const_cast<uint16_t *>(K_svm)) &&
           kqk->SetKernelArguments(2, &scores_buf, sizeof(cl_mem)) &&
           kqk->SetKernelArguments(3, &Ms, sizeof(int)) &&
           kqk->SetKernelArguments(4, &Ns, sizeof(int)) &&
           kqk->SetKernelArguments(5, &q0, sizeof(int)) &&
           kqk->SetKernelArguments(6, &d, sizeof(int)) &&
           kqk->SetKernelArguments(7, &HDq, sizeof(int)) &&
           kqk->SetKernelArguments(8, &HDkv, sizeof(int)) &&
           kqk->SetKernelArguments(9, &gqa, sizeof(int)) &&
           kqk->SetKernelArguments(10, &ca, sizeof(int)) &&
           kqk->SetKernelArguments(11, &scale, sizeof(float)) &&
           kqk->SetKernelArguments(12, &attn_softcap, sizeof(float)) &&
           kqk->SetKernelArguments(13, &lw, sizeof(int)) &&
           kqk->SetKernelArguments(14, &rc, sizeof(int));
      if (!ok)
        break;
      // gws.z is padded only if lws.z > 1; heads past num_heads_Q would read
      // past Q, so the z extent must stay exact.
      if (gws[2] != num_heads_Q) {
        ok = false;
        break;
      }
      blas_cc->command_queue_inst_.enqueueKernel(
        kqk->GetKernel(), 3, gws.data(), qk_lws.data(), 0, nullptr, nullptr);
    }
    // ---- softmax over each row pair's band (the image arm's kernel).
    {
      ok = ksm->SetKernelArguments(0, &scores_buf, sizeof(cl_mem)) &&
           ksm->SetKernelArguments(1, &Ms, sizeof(int)) &&
           ksm->SetKernelArguments(2, &Ns, sizeof(int)) &&
           ksm->SetKernelArguments(3, &ca, sizeof(int)) &&
           ksm->SetKernelArguments(4, &lw, sizeof(int)) &&
           (f32 || ksm->SetKernelArguments(5, &bmode, sizeof(int)));
      if (!ok)
        break;
      std::array<size_t, 3> gws = {SOFTMAX_LWS, (size_t)Ms, num_heads_Q};
      std::array<size_t, 3> lws = {SOFTMAX_LWS, 1, 1};
      blas_cc->command_queue_inst_.enqueueKernel(
        ksm->GetKernel(), 3, gws.data(), lws.data(), 0, nullptr, nullptr);
    }
    // ---- P*V for each row pair x 8 channels.
    {
      const size_t dx = (size_t)d / 8, pairs = ((size_t)Ms + 1) / 2;
      std::array<size_t, 3> gws = {pad(dx, sv_lws[0]), pad(pairs, sv_lws[1]),
                                   pad(num_heads_Q, sv_lws[2])};
      if (gws[2] != num_heads_Q) {
        ok = false;
        break;
      }
      ok = ksv->SetKernelArguments(0, &scores_buf, sizeof(cl_mem)) &&
           ksv->SetKernelSVMArguments(1, const_cast<uint16_t *>(V_svm));
      if (ok && o_clmem != nullptr) {
        cl_mem oh = static_cast<cl_mem>(o_clmem);
        ok = ksv->SetKernelArguments(2, &oh, sizeof(cl_mem));
      } else if (ok) {
        ok = ksv->SetKernelSVMArguments(2, O_svm);
      }
      ok = ok && ksv->SetKernelArguments(3, &Ms, sizeof(int)) &&
           ksv->SetKernelArguments(4, &Ns, sizeof(int)) &&
           ksv->SetKernelArguments(5, &q0, sizeof(int)) &&
           ksv->SetKernelArguments(6, &d, sizeof(int)) &&
           ksv->SetKernelArguments(7, &HDq, sizeof(int)) &&
           ksv->SetKernelArguments(8, &HDkv, sizeof(int)) &&
           ksv->SetKernelArguments(9, &gqa, sizeof(int)) &&
           ksv->SetKernelArguments(10, &ca, sizeof(int)) &&
           ksv->SetKernelArguments(11, &lw, sizeof(int)) &&
           ksv->SetKernelArguments(12, &rc, sizeof(int));
      if (!ok)
        break;
      blas_cc->command_queue_inst_.enqueueKernel(
        ksv->GetKernel(), 3, gws.data(), sv_lws.data(), 0, nullptr, nullptr);
    }
  }
  if (!ok) {
    // Something was refused after a launch may have been enqueued: the
    // caller's fallback recomputes O in full, so nothing partial survives.
    std::fprintf(stderr,
                 "[APA] d=%u hQ=%u hKV=%u M=%u N_kv=%u: argument bind / grid "
                 "check failed -> declined\n",
                 head_dim, num_heads_Q, num_heads_KV, M, N_kv);
    return false;
  }
  {
    char key[160];
    std::snprintf(key, sizeof(key), "%u/%u/%u/%d/%d/%d/%s", head_dim,
                  num_heads_Q, num_heads_KV, win > 0 ? 1 : 0,
                  attn_softcap > 0.0f ? 1 : 0, ring_cap > 0 ? 1 : 0,
                  copts.c_str());
    if (as.logged.emplace(key, 1).second)
      std::fprintf(
        stderr,
        "[APA] d=%u hQ=%u hKV=%u G=%d M=%u N_kv=%u W=%u softcap=%.1f "
        "ring=%u TM=%d TN=%d qk_lws=%zu,%zu,%zu sv_lws=%zu,%zu,%zu "
        "scores=%zuMB blocks=%u -> engaged\n",
        head_dim, num_heads_Q, num_heads_KV, gqa, M, N_kv, win, attn_softcap,
        ring_cap, TM, TN, qk_lws[0], qk_lws[1], qk_lws[2], sv_lws[0], sv_lws[1],
        sv_lws[2], (row_bytes * (size_t)rows_per) >> 20, blocks);
  }
  if (gh_mode() != 0) {
    attn_gpuhash_note("aq", Q_svm, nullptr,
                      (size_t)M * num_heads_Q * head_dim * 2, M);
    attn_gpuhash_note("ao", O_svm, o_clmem,
                      (size_t)M * num_heads_Q * head_dim * 2, M);
  }
  return true;
}

void cl_attention_release_caches() {
  // [reload] Model-teardown counterpart to what this file latches for the
  // process. No run is in flight at the call site and every item is rebuilt
  // lazily, so an empty state is always valid; nothing calls the driver when
  // the lane was never used.
  {
    // The per-slot resident RoPE cos/sin LUTs, keyed by the (cos, sin) HOST
    // pointers the caller's flat-LUT cache hands out, with `uploaded` meaning
    // "the device copy already holds these tables". Both halves of that break
    // at a reload: the buffers are one device pair per slot that nothing else
    // frees, so they leak per cycle, and the key is an address the next load
    // can be given again, which would let the new model's positions read the
    // old model's table. tca_mtx() is the only lock this file has; it guards
    // the TcaScratch half below. RopeScratch never had one -- it is written
    // from the single attention dispatch thread -- and a teardown has no run in
    // flight either.
    std::lock_guard<std::mutex> lk(tca_mtx());
    RopeScratch &rs = rope_scratch();
    for (auto &kv : rs.lut_slots) {
      if (kv.second.cos)
        opencl::clReleaseMemObjectT(kv.second.cos);
      if (kv.second.sin)
        opencl::clReleaseMemObjectT(kv.second.sin);
    }
    rs.lut_slots.clear();
    // The OHWI V image is a VIEW over a buffer this scratch does not own -- the
    // model's KV cache -- and its cache key is that buffer's handle value. Once
    // the model is gone the view names a released object, and a next load whose
    // KV buffer takes the same handle value would sample it. The key is what
    // has to go; the view is released with it because nothing else can.
    TcaScratch &ts = tca_scratch();
    if (ts.v_ohwi_image)
      opencl::clReleaseMemObjectT(ts.v_ohwi_image);
    ts.v_ohwi_image = nullptr;
    ts.v_ohwi_buf = nullptr;
    ts.v_ohwi_HD_KV = 0;
    ts.v_ohwi_S_max = 0;
  }
  // The FP16 operand planes of the FP32-activation attention. Rebuilt on the
  // next call that needs one.
  {
    ActStage &as = act_stage();
    for (unsigned int i = 0; i < 4; ++i) {
      as.mem[i].reset();
      as.pool[i].reset();
      as.cap[i] = 0;
    }
  }
  // [afd] The fused decode's partial buffers and probe verdicts.
  {
    AfdScratch &as = afd_scratch();
    for (void **b : {&as.acc, &as.ml, &as.probe}) {
      if (*b)
        opencl::clReleaseMemObjectT(static_cast<cl_mem>(*b));
      *b = nullptr;
    }
    as.acc_cap = as.ml_cap = as.probe_cap = 0;
  }
  // [afp] The fused prefill's probe buffer and variant verdicts.
  {
    AfpScratch &as = afp_scratch();
    if (as.probe)
      opencl::clReleaseMemObjectT(static_cast<cl_mem>(as.probe));
    as.probe = nullptr;
    as.probe_cap = 0;
    as.verdict.clear();
  }
  // [apa] The three-pass prefill's kernel handles, build verdicts and the
  // shapes it announced (its scores live in the TcaScratch above).
  {
    std::lock_guard<std::mutex> lk(tca_mtx());
    ApaScratch &as = apa_scratch();
    as.qk.clear();
    as.sv.clear();
    as.sm.clear();
    as.verdict.clear();
    as.logged.clear();
  }
  // A teardown is a handle-epoch boundary for this file's memos too.
  opencl::clBumpHandleEpoch();
}

} // namespace nntrainer
