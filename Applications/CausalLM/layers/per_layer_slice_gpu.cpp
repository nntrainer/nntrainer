// SPDX-License-Identifier: Apache-2.0
/**
 * @file   per_layer_slice_gpu.cpp
 * @date   17 Jun 2026
 * @brief  Implementation of the GPU-routed PerLayerSlice layer. See
 *         per_layer_slice_gpu.h for the rationale.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "per_layer_slice_gpu.h"

#if defined(ENABLE_OPENCL)
// OpenCL GPU FP16 slice kernel (per_layer_slice_cl_fp16). Guarded so the
// no-OpenCL build (enable-opencl=false) compiles via the host memcpy path.
#include <blas_kernels.h>
#endif
#include <cstring>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void PerLayerSliceLayerGPU::finalize(nntrainer::InitLayerContext &context) {
  auto dims = context.getInputDimensions();
  auto in_dim = dims[0];
  if (!std::get<nntrainer::props::SkipPrefill>(slice_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(slice_props).get();

  unsigned int feature_size = std::get<props::FeatureSize>(slice_props).get();
  NNTR_THROW_IF(feature_size == 0, std::invalid_argument)
    << "feature_size must be > 0";
  NNTR_THROW_IF(in_dim.width() % feature_size != 0, std::invalid_argument)
    << "input width must be divisible by feature_size";

  auto out_dim = in_dim;
  out_dim.width(feature_size);
  context.setOutputDimensions({out_dim});
}

void PerLayerSliceLayerGPU::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // Match PerLayerSliceLayer: a chunked or resumed prefill is still a prefill
  // even though it does not start at token 0, so test the step width too.
  // Testing only `from == 0` would run this slice on every chunk after the
  // first, which is exactly what skip_prefill asks to avoid.
  const bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  auto &in = context.getInput(SINGLE_INOUT_IDX);
  auto &out = context.getOutput(SINGLE_INOUT_IDX);

  const unsigned int feature_size =
    std::get<props::FeatureSize>(slice_props).get();
  const unsigned int layer_index =
    std::get<props::LayerIndex>(slice_props).get();

  const ml::train::TensorDim in_dim = in.getDim();
  const unsigned int in_width = in_dim.width();
  const unsigned int num_layers = in_width / feature_size;
  NNTR_THROW_IF(layer_index >= num_layers, std::invalid_argument)
    << "layer_index out of range";
  const unsigned int off = layer_index * feature_size;

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

    const unsigned int rows = in_step_dim.height();

#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL)
    // GPU SVM/cl_mem FP16 slice path; falls through to host memcpy below
    // when OpenCL is off.
    const auto in_md = in_step.getMemoryData();
    const auto out_md = out_step.getMemoryData();
    const bool use_svm = in_md && in_md->isSVM() && out_md && out_md->isSVM();
    if (use_svm &&
        in_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        out_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      // Bind the device buffer only for an offset-0 view. getClMem() returns
      // the PARENT buffer, which carries no offset of its own, so a b > 0 step
      // view would bind batch 0's bytes and silently slice the wrong batch.
      // Falling back to the SVM pointer keeps those batches correct, because
      // getData<_FP16>() does honour the view offset.
      void *in_cl = (in_step.isClMem() && in_step.getOffset() == 0)
                      ? in_step.getClMem()
                      : nullptr;
      void *out_cl = (out_step.isClMem() && out_step.getOffset() == 0)
                       ? out_step.getClMem()
                       : nullptr;
      nntrainer::per_layer_slice_cl_fp16(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(), rows, feature_size,
        in_width, off, /*use_svm=*/true, out_cl, in_cl);
      continue;
    }
#endif // ENABLE_OPENCL
#endif

    // Raw-pointer host fallback (FP32 / non-SVM).
    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const float *ip = in_step.getData<float>();
      float *op = out_step.getData<float>();
      for (unsigned int t = 0; t < rows; ++t)
        std::memcpy(op + t * feature_size, ip + t * in_width + off,
                    sizeof(float) * feature_size);
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const _FP16 *ip = in_step.getData<_FP16>();
      _FP16 *op = out_step.getData<_FP16>();
      for (unsigned int t = 0; t < rows; ++t)
        std::memcpy(op + t * feature_size, ip + t * in_width + off,
                    sizeof(_FP16) * feature_size);
#endif
    } else {
      throw std::invalid_argument(
        "PerLayerSliceLayerGPU: unsupported input dtype");
    }
  }
}

} // namespace causallm
