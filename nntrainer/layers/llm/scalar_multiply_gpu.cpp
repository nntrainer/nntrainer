// SPDX-License-Identifier: Apache-2.0
/**
 * @file   scalar_multiply_gpu.cpp
 * @date   17 Jun 2026
 * @brief  Implementation of the GPU-routed ScalarMultiply layer. See
 *         scalar_multiply_gpu.h for the rationale.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "scalar_multiply_gpu.h"

#include <blas_kernels.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

#ifdef ENABLE_FP16
static void dump_fp16_stats(const char *name, float mult,
                            nntrainer::Tensor &t) {
  const _FP16 *d = t.getData<_FP16>();
  size_t n = t.size();
  double s = 0, s2 = 0, amax = 0;
  for (size_t i = 0; i < n; ++i) {
    double v = static_cast<float>(d[i]);
    s += v;
    s2 += v * v;
    if (std::fabs(v) > amax)
      amax = std::fabs(v);
  }
  double mean = n ? s / n : 0, var = n ? s2 / n - mean * mean : 0;
  std::fprintf(stderr,
               "[GDUMP] %-30s mult=%.6g out mean=%.5g std=%.5g absmax=%.5g\n",
               name, mult, mean, std::sqrt(var < 0 ? 0 : var), amax);
}
#endif

void ScalarMultiplyLayerGPU::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  if (!std::get<nntrainer::props::SkipPrefill>(scalar_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(scalar_props).get();

  bool use_weight = std::get<props::UseWeight>(scalar_props).get();
  if (use_weight) {
    nntrainer::TensorDim scalar_dim(
      1, 1, 1, 1,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));
    wt_idx[0] = context.requestWeight(
      scalar_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "scalar_multiplier",
      false);
  }
}

void ScalarMultiplyLayerGPU::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  if (skip_prefill && from == 0)
    return;

  bool use_weight = std::get<props::UseWeight>(scalar_props).get();
  float multiplier;
  if (use_weight) {
    nntrainer::Tensor &weight = context.getWeight(wt_idx[0]);
    if (weight.getDataType() == ml::train::TensorDim::DataType::FP32) {
      multiplier = weight.getValue<float>(0, 0, 0, 0);
#ifdef ENABLE_FP16
    } else if (weight.getDataType() == ml::train::TensorDim::DataType::FP16) {
      multiplier = static_cast<float>(weight.getValue<_FP16>(0, 0, 0, 0));
#endif
    } else {
      multiplier = weight.getValue<float>(0, 0, 0, 0);
    }
  } else {
    multiplier = std::get<props::ScalarMultiplier>(scalar_props).get();
  }

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out.getDim();
  in_step_dim.batch(1);
  out_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  const unsigned int b_size = in_dim.batch();
  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step = out.getSharedDataTensor(
      out_step_dim, b * out.getDim().getFeatureLen(), true);

    const unsigned int n = in_step.size();

#ifdef ENABLE_FP16
    const auto in_md = in_step.getMemoryData();
    const auto out_md = out_step.getMemoryData();
    const bool use_svm = in_md && in_md->isSVM() && out_md && out_md->isSVM();
    if (use_svm &&
        in_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        out_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      // GPU-resident: pointer pre-offset (SVM), so row_off stays 0. cl_mem
      // sub-buffers are bound when the planner placed the operands there.
      void *in_cl = in_step.isClMem() ? in_step.getClMem() : nullptr;
      void *out_cl = out_step.isClMem() ? out_step.getClMem() : nullptr;
      nntrainer::scalar_mul_cl_fp16(in_step.getData<_FP16>(),
                                    out_step.getData<_FP16>(), multiplier, n,
                                    /** use_svm */ true, out_cl, in_cl,
                                    /** row_off */ 0);
      continue;
    }
#endif

    // Raw-pointer host fallback (FP32 / non-SVM). Tensor::multiply is avoided
    // because it crashes on gpu-context-allocated tensors.
    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const float *ip = in_step.getData<float>();
      float *op = out_step.getData<float>();
      for (unsigned int i = 0; i < n; ++i)
        op[i] = ip[i] * multiplier;
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const _FP16 *ip = in_step.getData<_FP16>();
      _FP16 *op = out_step.getData<_FP16>();
      for (unsigned int i = 0; i < n; ++i)
        op[i] = static_cast<_FP16>(static_cast<float>(ip[i]) * multiplier);
#endif
    } else {
      throw std::invalid_argument(
        "ScalarMultiplyLayerGPU: unsupported input dtype");
    }
  }

#ifdef ENABLE_FP16
  static const bool dump = std::getenv("NNTR_DUMP_LAYERS") != nullptr;
  if (dump && out.getDataType() == ml::train::TensorDim::DataType::FP16)
    dump_fp16_stats(context.getName().c_str(), multiplier, out);
#endif
}

} // namespace nntrainer
