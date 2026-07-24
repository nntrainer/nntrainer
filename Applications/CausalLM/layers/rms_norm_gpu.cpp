// SPDX-License-Identifier: Apache-2.0
/**
 * @file   rms_norm_gpu.cpp
 * @date   29 May 2026
 * @brief  GPU-routed RMSNorm. See rms_norm_gpu.h for the rationale.
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define RMSNORM_GPU_HAVE_NEON 1
#else
#define RMSNORM_GPU_HAVE_NEON 0
#endif

#include "rms_norm_gpu.h"

#if defined(ENABLE_OPENCL)
// [T12] OpenCL GPU rmsnorm dispatch headers (rmsnorm_cl / rmsnorm_cl_fp16 /
// clmem_*). Guarded so the no-OpenCL CPU build compiles with the host norm.
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#endif
#include <memory_data.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

/**
 * @brief Weight slot indices owned by RMSNormLayerGPU
 */
enum class RMSParamsGPU : unsigned int { GAMMA = 0 };

void RMSNormLayerGPU::finalize(nntrainer::InitLayerContext &context) {
  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  // 1 input (plain RMSNorm) or 2 inputs (fused RMSNorm + residual add at the
  // Gemma2 sandwich-norm boundary): output is always a single tensor sized like
  // input[0]. (input[1], when present, is the residual stream added in-kernel.)
  context.setOutputDimensions({dim[0]});
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[(unsigned int)RMSParamsGPU::GAMMA] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayerGPU::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

// Raw-pointer host RMSNorm. Used because Tensor::multiply / add_i /
// inv_sqrt_i crash on gpu-context-allocated tensors (same family as
// the AdditionLayerCL workaround). NEON-SIMD on aarch64 for ~10×
// throughput vs the scalar fallback; scalar code is correct for
// reference / non-ARM builds.
static void rms_norm_host_fp32(const float *in, const float *gamma, float *out,
                               float eps, unsigned int rows,
                               unsigned int cols) {
#if RMSNORM_GPU_HAVE_NEON
  for (unsigned int r = 0; r < rows; ++r) {
    const float *in_row = in + (size_t)r * cols;
    float *out_row = out + (size_t)r * cols;

    // Pass 1: sum-of-squares with 2-lane parallel accumulation.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    unsigned int k = 0;
    for (; k + 8 <= cols; k += 8) {
      float32x4_t v0 = vld1q_f32(in_row + k);
      float32x4_t v1 = vld1q_f32(in_row + k + 4);
      acc0 = vmlaq_f32(acc0, v0, v0);
      acc1 = vmlaq_f32(acc1, v1, v1);
    }
    float sumsq = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; k < cols; ++k)
      sumsq += in_row[k] * in_row[k];

    const float inv_rms = 1.0f / std::sqrt(sumsq / (float)cols + eps);
    const float32x4_t inv_rms_v = vdupq_n_f32(inv_rms);

    // Pass 2: out[k] = in[k] * inv_rms * gamma[k].
    for (k = 0; k + 4 <= cols; k += 4) {
      float32x4_t v = vld1q_f32(in_row + k);
      float32x4_t g = vld1q_f32(gamma + k);
      vst1q_f32(out_row + k, vmulq_f32(vmulq_f32(v, inv_rms_v), g));
    }
    for (; k < cols; ++k)
      out_row[k] = in_row[k] * inv_rms * gamma[k];
  }
#else
  for (unsigned int r = 0; r < rows; ++r) {
    const float *in_row = in + (size_t)r * cols;
    float *out_row = out + (size_t)r * cols;
    double sumsq = 0.0;
    for (unsigned int k = 0; k < cols; ++k)
      sumsq += (double)in_row[k] * in_row[k];
    const float mean_sq = (float)(sumsq / cols);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    for (unsigned int k = 0; k < cols; ++k)
      out_row[k] = in_row[k] * inv_rms * gamma[k];
  }
#endif
}

void RMSNormLayerGPU::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  if (skip_prefill && from == 0)
    return;
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma =
    context.getWeight(wt_idx[(unsigned int)RMSParamsGPU::GAMMA]);

  // [NNTR_NORM_DBG] one-shot raw-operand dump (see CudaRMSNormLayer twin):
  // compares ACTUAL gamma/input bytes across engines. gamma/in are SVM
  // host-visible on the OpenCL path.
  // Guarded by ENABLE_FP16 like the rest of this file's _FP16 usage: _FP16 is
  // only defined under ENABLE_FP16 (see api/ccapi/include/tensor_dim.h), so an
  // unguarded reference here would fail to compile on a no-FP16 build even
  // though the runtime check below never fires without FP16 tensors.
#ifdef ENABLE_FP16
  if (std::getenv("NNTR_NORM_DBG") != nullptr &&
      in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    static bool once = false;
    if (!once) {
      once = true;
      const _FP16 *gp = gamma.getData<_FP16>();
      const _FP16 *xp = in.getData<_FP16>();
      const unsigned int w = in.getDim().width();
      const unsigned char *gb = reinterpret_cast<const unsigned char *>(gp);
      unsigned long long h = 1469598103934665603ULL;
      for (size_t i = 0; i < (size_t)w * 2; ++i) {
        h ^= gb[i];
        h *= 1099511628211ULL;
      }
      std::fprintf(stderr,
                   "[NORM-DBG] opencl eps=%.3e gammafnv=%016llx g[0..3]=%.6f "
                   "%.6f %.6f %.6f x[0..3]=%.6f %.6f %.6f %.6f\n",
                   (float)epsilon, h, (float)gp[0], (float)gp[1], (float)gp[2],
                   (float)gp[3], (float)xp[0], (float)xp[1], (float)xp[2],
                   (float)xp[3]);
      std::fflush(stderr);
    }
  }
#endif

  const ml::train::TensorDim in_dim = in.getDim();
  const ml::train::TensorDim out_dim = out.getDim();
  const unsigned int b_size = in_dim.batch();
  const unsigned int H = to - from;
  const unsigned int W = in_dim.width();

  const bool fp32 =
    in.getDataType() == ml::train::TensorDim::DataType::FP32 &&
    gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
    out.getDataType() == ml::train::TensorDim::DataType::FP32;

// [T12] FP16 residency path is GPU-only -> also requires ENABLE_OPENCL.
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
  // FP16 residual stream (Gemma2 / gpu_native parity path): run the norm as a
  // GPU kernel (rmsnorm_cl_fp16, SVM-direct when the graph uses the SVM pool),
  // NOT on the host. This is the residency path — the host scalar fp32 norm
  // below is the Qwen3 legacy path. The kernel folds gamma in.
  const bool fp16 =
    in.getDataType() == ml::train::TensorDim::DataType::FP16 &&
    gamma.getDataType() == ml::train::TensorDim::DataType::FP16 &&
    out.getDataType() == ml::train::TensorDim::DataType::FP16;
  if (fp16) {
    const auto in_md = in.getMemoryData();
    const auto out_md = out.getMemoryData();
    const auto g_md = gamma.getMemoryData();
    const bool use_svm = in_md && in_md->isSVM() && out_md && out_md->isSVM() &&
                         g_md && g_md->isSVM();
    // Planner-decided STATIC residency: each of in/out binds the plane its
    // ResidencyClass picked at allocation (cl_mem sub-buffer for GPU_CLMEM, SVM
    // otherwise) -- uniformly, every forward, with NO runtime device_valid
    // flipping and no per-edge env gates. gamma stays SVM (gpu_native does the
    // same). Mixed-plane kernels (e.g. cl_mem residual in -> SVM out for the
    // final norm whose consumer is the host lm_head) are valid; the trailing
    // blocking SVM map only happens when the OUTPUT is SVM, which is exactly
    // when a host consumer may read it.
    // W%8==0 must match the wrapper's coop-path condition exactly: a silent
    // SVM fallback inside the wrapper while the consumer binds cl_mem would be
    // a hybrid hole. (Gemma2 hidden=2304 is always %8==0.)
    const bool clmem_ok = b_size == 1 && (W % 8u == 0u) && use_svm;
    const bool in_clmem = clmem_ok && in.isClMem() && in.getClMem();
    const bool out_clmem = clmem_ok && out.isClMem() && out.getClMem();
    void *in_cl = in_clmem ? in.getClMem() : nullptr;
    void *out_cl = out_clmem ? out.getClMem() : nullptr;
    // NNTR_NORM_NOMAP=1 + NNTR_DEVRES=1 (EXPERIMENTAL, opt-in, KNOWN-BROKEN):
    // S4 pair-removal extended to the pre-norms (attention_norm/ffn_norm SVM
    // outs whose consumers are all GPU FCs). Skips the trailing blocking map
    // and flags device_valid so the FC skips its matching unmap/map. Measured
    // 2026-06-11: output DIVERGES (prompt_1k 1 token -> 32 tokens) and the
    // map itself turned out cheap (~0.3ms/call, NNTR_NORM_TPROF) -- the gap
    // it was suspected of was actually unsubmitted FC chains (fixed by the
    // dotCl clFlush). Kept gated for a future pair-removal debug only.
    static const bool norm_devown = std::getenv("NNTR_DEVRES") != nullptr &&
                                    std::getenv("NNTR_NORM_NOMAP") != nullptr;
    const bool skip_out_map =
      norm_devown && use_svm && !out_clmem && b_size == 1 && (W % 8u == 0u);
    _FP16 *gamma_p = gamma.getData<_FP16>();
    // Fused norm+residual-add (NNTR_FUSE_ADDNORM): a 2nd input is the residual
    // stream. Compute out = rmsnorm(in)*gamma + residual in one kernel,
    // removing the separate AdditionLayerCL (v8c_add_h2h) + its dispatch idle.
    // The fused kernel needs in/out/residual cl_mem-resident
    // (NNTR_GPU_CLMEM_POOL); the graph has no separate add node to fall back
    // to, so a missed dispatch is a hard error rather than silent garbage.
    const bool fused_add = context.getNumInputs() == 2;
    nntrainer::Tensor *resid = fused_add ? &context.getInput(1) : nullptr;
    void *resid_cl =
      (fused_add && clmem_ok && resid->isClMem() && resid->getClMem())
        ? resid->getClMem()
        : nullptr;
    // FP16 epsilon underflow guard: rms_norm_eps (1e-6) rounds to 0 in FP16
    // (min normal ~6e-5), so a near-zero activation row gets 1/rms = inf ->
    // overflow/NaN (Gemma4 has such rows; Gemma2 happened not to). Floor the
    // eps to a FP16-representable value; mathematically negligible for normal
    // rows (mean >> eps).
    const float eps16 = epsilon < 1.0e-4f ? 1.0e-4f : epsilon;
    for (unsigned int b = 0; b < b_size; ++b) {
      _FP16 *in_p = in.getData<_FP16>() + (size_t)b * in_dim.getFeatureLen();
      _FP16 *out_p = out.getData<_FP16>() + (size_t)b * out_dim.getFeatureLen();
      if (fused_add) {
        _FP16 *resid_p =
          resid->getData<_FP16>() + (size_t)b * resid->getDim().getFeatureLen();
        if (!nntrainer::rmsnorm_add_cl_fp16(in_p, gamma_p, resid_p, out_p,
                                            eps16, H, W, use_svm, out_cl, in_cl,
                                            resid_cl))
          throw std::runtime_error("RMSNormLayerGPU: fused norm+add dispatch "
                                   "failed (NNTR_FUSE_ADDNORM "
                                   "requires the GPU_CLMEM residency path: set "
                                   "NNTR_GPU_CLMEM_POOL=1)");
      } else {
        nntrainer::rmsnorm_cl_fp16(in_p, gamma_p, out_p, eps16, H, W, use_svm,
                                   out_cl, in_cl, skip_out_map);
      }
    }
    if (skip_out_map) {
      if (auto md = out.getMemoryData())
        md->setDeviceValid(true, out.getData<_FP16>());
    }
    static const bool gdump = std::getenv("NNTR_DUMP_LAYERS") != nullptr;
    if (gdump) {
      const _FP16 *od = out.getData<_FP16>();
      const _FP16 *idp = in.getData<_FP16>();
      size_t n = (size_t)H * W;
      double oamax = 0, iamax = 0;
      int onan = 0, inan = 0;
      for (size_t i = 0; i < n; ++i) {
        float ov = (float)od[i], iv = (float)idp[i];
        if (std::isnan(ov) || std::isinf(ov))
          onan++;
        else if (std::fabs(ov) > oamax)
          oamax = std::fabs(ov);
        if (std::isnan(iv) || std::isinf(iv))
          inan++;
        else if (std::fabs(iv) > iamax)
          iamax = std::fabs(iv);
      }
      const _FP16 *gd = gamma.getData<_FP16>();
      double gamax = 0;
      int gnan = 0;
      for (unsigned int i = 0; i < W; ++i) {
        float gv = (float)gd[i];
        if (std::isnan(gv) || std::isinf(gv))
          gnan++;
        else if (std::fabs(gv) > gamax)
          gamax = std::fabs(gv);
      }
      // per-row RMS range over H rows (detect anomalous near-zero rows)
      double rmin = 1e30, rmax = 0;
      for (unsigned int r = 0; r < H; ++r) {
        double ss = 0;
        int rnan = 0;
        for (unsigned int c = 0; c < W; ++c) {
          float v = (float)idp[(size_t)r * W + c];
          if (std::isnan(v) || std::isinf(v))
            rnan++;
          else
            ss += (double)v * v;
        }
        if (rnan)
          continue;
        double rms = std::sqrt(ss / W);
        if (rms < rmin)
          rmin = rms;
        if (rms > rmax)
          rmax = rms;
      }
      std::fprintf(stderr, "[RDUMP+] %-30s row-rms min=%.4g max=%.4g\n",
                   context.getName().c_str(), rmin, rmax);
      std::fprintf(
        stderr,
        "[RDUMP] %-34s in absmax=%.5g nan=%d | gamma absmax=%.5g nan=%d | out "
        "absmax=%.5g nan=%d\n",
        context.getName().c_str(), iamax, inan, gamax, gnan, oamax, onan);
    }
    // NNTR_CLMEM_PROBE: device-side captures of the norm in/out for the
    // fan-out corruption bisect (no host sync; dumped at PROBE_MAX). All
    // norms (cl_mem captures are reliable; SVM captures may themselves race).
    static const bool probe_on = std::getenv("NNTR_CLMEM_PROBE") != nullptr;
    if (probe_on) {
      const std::string &nm = context.getName();
      nntrainer::clmem_probe_capture(
        (nm + ":in").c_str(), in.getData<_FP16>(), in_cl,
        (unsigned int)((size_t)H * W * sizeof(_FP16)));
      nntrainer::clmem_probe_capture(
        (nm + ":out").c_str(), out.getData<_FP16>(), out_cl,
        (unsigned int)((size_t)H * W * sizeof(_FP16)));
    }
    return;
  }
#endif

  if (!fp32) {
    throw std::runtime_error(
      "RMSNormLayerGPU: only FP32/FP16 inputs supported in this build");
  }

  // NNTR_DEVRES Step 3a: run the FP32 norm on the GPU (rmsnorm_cl, SVM-direct)
  // instead of the host NEON scalar when the activation pool is SVM. The host
  // norm (rms_norm_host_fp32 below) is a synchronous CPU pass that serializes
  // CPU<->GPU every layer (the "rms 105x host" stall); moving it to the GPU
  // lets the norm and the downstream QKV/gate-up FC run back-to-back on the
  // queue. Output stays plain SVM here (the FC consumes it as today) — the
  // device_valid bit + map-pair skip are a later step (would otherwise desync
  // the FC's unmap). Gated; off => the host path runs byte-identically.
  static const bool devres_rms_gpu = std::getenv("NNTR_DEVRES") != nullptr;
  const auto rms_in_md = in.getMemoryData();
  const auto rms_out_md = out.getMemoryData();
  const auto rms_g_md = gamma.getMemoryData();
  const bool rms_use_svm = rms_in_md && rms_in_md->isSVM() && rms_out_md &&
                           rms_out_md->isSVM() && rms_g_md && rms_g_md->isSVM();
  const bool rms_on_gpu = devres_rms_gpu && rms_use_svm;

  // Try the fused-rmsq path first (paper §3.6 #2): it writes int8 +
  // scale + zp + row_sum into pool backings keyed by
  // ptr:<out_host>:fused_{i8,scale,zp,rs}. The v8c FC consumer
  // (NNTR_V8C_CONSUME_FUSED_RMSQ=1) picks those up directly,
  // bypassing the rmsnorm→FC fp32 boundary. Falls back to plain
  // GPU rmsnorm or host rmsnorm depending on env.
  for (unsigned int b = 0; b < b_size; ++b) {
    // Sliced views: in and out are shared with the parent at offset
    // b * featureLen. Operate on the raw float* with explicit offsets
    // rather than calling Tensor::getSharedDataTensor (which would
    // return another gpu-context tensor and risk the same crashes).
    const size_t in_off = (size_t)b * in_dim.getFeatureLen();
    const size_t out_off = (size_t)b * out_dim.getFeatureLen();
    const float *in_p_root = in.getData<float>();
    float *out_p_root = out.getData<float>();
    const float *gamma_p = gamma.getData<float>();
    const float *in_p = in_p_root + in_off;
    float *out_p = out_p_root + out_off;

    // NEON SIMD host RMSNorm is the primary compute path because
    // measurement showed the fused-rmsq GPU dispatch is NET NEGATIVE
    // for TPS even when paired with the v8c FC consumer that skips
    // its own quant. The 56 dispatches × ~3 ms per forward outweigh
    // the saved per-FC quant time.
    //
    // The fused-rmsq dispatch is therefore OPT-IN via two envs:
    //   NNTR_FUSED_RMSQ=1          — base producer env (legacy)
    //   NNTR_RMSQ_GPU_DISPATCH=1   — engage from this GPU layer
    // PLUS H >= NNTR_RMSQ_GPU_MIN_H (default 8) for decode skip.
    //
    // The NEON host norm always runs because the v8c FC consumer
    // path is correct without the fused entries (it falls back to
    // its own act-quant). So host norm gives correctness; the
    // fused dispatch is now purely a perf experiment.
    static const bool dispatch_enabled =
      std::getenv("NNTR_RMSQ_GPU_DISPATCH") != nullptr;
    static const unsigned int min_h_for_gpu = []() {
      const char *e = std::getenv("NNTR_RMSQ_GPU_MIN_H");
      return e != nullptr ? (unsigned int)std::atoi(e) : 8u;
    }();
#if defined(ENABLE_OPENCL)
    if (rms_on_gpu) {
      // NNTR_DEVRES Step 3a: GPU FP32 RMSNorm (SVM-direct). The kernel folds
      // gamma in and writes out_p on the GPU — no host norm pass.
      nntrainer::rmsnorm_cl(in_p, gamma_p, out_p, epsilon, H, W,
                            /** use_svm */ true);
    } else {
      if (dispatch_enabled && b_size == 1 && H >= min_h_for_gpu) {
        nntrainer::fused_rmsnorm_quant_resident_fp32(
          in, gamma, epsilon, H, W, out.getName(), (const void *)out_p);
      }
      rms_norm_host_fp32(in_p, gamma_p, out_p, epsilon, H, W);
    }
#else
    // [T12] No-OpenCL build: host CPU RMSNorm is the only path.
    rms_norm_host_fp32(in_p, gamma_p, out_p, epsilon, H, W);
#endif
  }
}

} // namespace causallm
