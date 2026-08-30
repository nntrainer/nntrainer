// SPDX-License-Identifier: Apache-2.0
/**
 * @file   rms_norm_gpu.cpp
 * @date   29 May 2026
 * @brief  GPU-routed RMSNorm. See rms_norm_gpu.h for the rationale.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
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
// OpenCL GPU rmsnorm dispatch (rmsnorm_cl / rmsnorm_cl_fp16 /
// rmsnorm_add_cl_fp16). Guarded so the no-OpenCL build compiles host-only.
#include <blas_kernels.h>
#endif
#include <memory_data.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum class RMSParamsGPU : unsigned int { GAMMA = 0 };

void RMSNormLayerGPU::finalize(nntrainer::InitLayerContext &context) {
  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  // One input (plain RMSNorm) or two (fused RMSNorm + residual add at a
  // sandwich-norm boundary): the output is always a single tensor sized like
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

// Raw-pointer host RMSNorm. It works on pointers because Tensor::multiply,
// add_i and inv_sqrt_i are not usable on tensors allocated by a device
// context. NEON-vectorized on aarch64 for roughly ten times the throughput of
// the scalar loop, which stays as the correct reference for other targets.
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

  const ml::train::TensorDim in_dim = in.getDim();
  const ml::train::TensorDim out_dim = out.getDim();
  const unsigned int b_size = in_dim.batch();
  const unsigned int H = to - from;
  const unsigned int W = in_dim.width();

  const bool fp32 =
    in.getDataType() == ml::train::TensorDim::DataType::FP32 &&
    gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
    out.getDataType() == ml::train::TensorDim::DataType::FP32;

// The FP16 residency path is GPU-only, so it also requires ENABLE_OPENCL.
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
  // FP16 residual stream: run the norm as a GPU kernel (rmsnorm_cl_fp16,
  // SVM-direct when the graph runs on the SVM pool), not on the host. This is
  // the residency path; the host FP32 norm further down is the fallback for a
  // graph whose activations are not device-resident. The kernel folds gamma in.
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
    // Static residency: each of the input and the output binds the plane its
    // tensor was allocated on (a device sub-buffer when it is device-resident,
    // SVM otherwise), uniformly on every forward, with no runtime flipping and
    // no per-edge switches. gamma stays SVM. Mixed-plane kernels are valid --
    // a device-resident input feeding an SVM output is exactly the shape of a
    // final norm whose consumer reads on the host -- and the trailing blocking
    // SVM map only happens when the OUTPUT is SVM, which is precisely when a
    // host consumer may read it.
    // The width % 8 == 0 test must match the wrapper's cooperative-path
    // condition exactly: a silent SVM fallback inside the wrapper while the
    // consumer binds the device buffer would be a coherence hole.
    const bool clmem_ok = b_size == 1 && (W % 8u == 0u) && use_svm;
    const bool in_clmem = clmem_ok && in.isClMem() && in.getClMem();
    const bool out_clmem = clmem_ok && out.isClMem() && out.getClMem();
    void *in_cl = in_clmem ? in.getClMem() : nullptr;
    void *out_cl = out_clmem ? out.getClMem() : nullptr;
    _FP16 *gamma_p = gamma.getData<_FP16>();
    // Fused norm and residual add: a second input is the residual stream, so
    // out = rmsnorm(in) * gamma + residual runs in one kernel and the separate
    // addition layer -- and the dispatch idle between the two -- disappears.
    // A graph that wires the second input has no separate add node to fall back
    // to, so a refused dispatch is a hard error rather than silent garbage.
    const bool fused_add = context.getNumInputs() == 2;
    nntrainer::Tensor *resid = fused_add ? &context.getInput(1) : nullptr;
    void *resid_cl =
      (fused_add && clmem_ok && resid->isClMem() && resid->getClMem())
        ? resid->getClMem()
        : nullptr;
    // FP16 epsilon underflow guard: a typical rms_norm_eps of 1e-6 rounds to
    // zero in FP16 (the smallest normal is ~6e-5), so a near-zero activation
    // row gets 1/rms = inf and overflows to NaN. Floor epsilon to an
    // FP16-representable value; that is mathematically negligible for a normal
    // row, whose mean square is far larger than epsilon.
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
          throw std::runtime_error(
            "RMSNormLayerGPU: the fused norm+add dispatch was refused; it "
            "needs "
            "an SVM activation plane and a width that is a multiple of 8");
      } else {
        nntrainer::rmsnorm_cl_fp16(in_p, gamma_p, out_p, eps16, H, W, use_svm,
                                   out_cl, in_cl);
      }
    }
    return;
  }
#endif

  if (!fp32) {
    throw std::runtime_error(
      "RMSNormLayerGPU: only FP32/FP16 inputs supported in this build");
  }

  // Run the FP32 norm on the GPU (rmsnorm_cl, SVM-direct) instead of the host
  // NEON pass whenever the input, the output and gamma all live on the SVM
  // activation plane. The host norm below is a synchronous CPU pass that
  // serializes the CPU against the GPU once per layer; keeping the norm on the
  // queue lets it and the projection that consumes it run back to back. The
  // output stays plain SVM, so a host consumer still reads it correctly.
  const auto rms_in_md = in.getMemoryData();
  const auto rms_out_md = out.getMemoryData();
  const auto rms_g_md = gamma.getMemoryData();
  const bool rms_on_gpu = rms_in_md && rms_in_md->isSVM() && rms_out_md &&
                          rms_out_md->isSVM() && rms_g_md && rms_g_md->isSVM();

  for (unsigned int b = 0; b < b_size; ++b) {
    // Sliced views: the input and the output are shared with the parent at
    // offset b * featureLen. Operate on the raw float pointers with explicit
    // offsets rather than through Tensor::getSharedDataTensor, which would
    // return another device-context tensor and hit the same Tensor-op
    // limitation the host helper above exists to avoid.
    const size_t in_off = (size_t)b * in_dim.getFeatureLen();
    const size_t out_off = (size_t)b * out_dim.getFeatureLen();
    const float *in_p_root = in.getData<float>();
    float *out_p_root = out.getData<float>();
    const float *gamma_p = gamma.getData<float>();
    const float *in_p = in_p_root + in_off;
    float *out_p = out_p_root + out_off;

#if defined(ENABLE_OPENCL)
    if (rms_on_gpu) {
      // GPU FP32 RMSNorm, SVM-direct. The kernel folds gamma in and writes the
      // output on the device, so there is no host norm pass at all.
      nntrainer::rmsnorm_cl(in_p, gamma_p, out_p, epsilon, H, W,
                            /*use_svm=*/true);
    } else {
      rms_norm_host_fp32(in_p, gamma_p, out_p, epsilon, H, W);
    }
#else
    // No-OpenCL build: the host CPU RMSNorm is the only path.
    rms_norm_host_fp32(in_p, gamma_p, out_p, epsilon, H, W);
#endif
  }
}

} // namespace causallm
