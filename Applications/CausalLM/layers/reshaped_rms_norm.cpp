// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reshaped_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <reshaped_rms_norm.h>

#if defined(ENABLE_OPENCL)
// GPU rmsnorm kernels (rmsnorm_cl / rmsnorm_cl_fp16) and the SVM coherence
// helpers (cl_queue_finish / cl_svm_map_force). Guarded so a build without
// OpenCL compiles host-only.
#include <blas_kernels.h>
#endif
#include <memory_data.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);
  use_gamma = std::get<props::UseGamma>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  if (use_gamma) {
    // gamma is unquantized FP32 on disk; request FP32 regardless of activation
    // dtype (FP16 would reinterpret the FP32 bytes and corrupt gamma). The FP16
    // path casts gamma down at the multiply site.
    nntrainer::TensorDim gamma_dim(
      1, 1, 1, feature_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       nntrainer::TensorDim::DataType::FP32));
    wt_idx[RMSParams::gamma] = context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
  }
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  // finalize() only requests gamma when use_gamma is true (Gemma4's v_norm and
  // per-layer projection norm set use_gamma=false), so wt_idx[gamma] is left at
  // its unset value in that case and must not be read.
  nntrainer::Tensor *gamma =
    use_gamma ? &context.getWeight(wt_idx[RMSParams::gamma]) : nullptr;

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    const unsigned int n_rows = in_step.getDim().height();
    const auto dtype = in_step.getDataType();

    // GPU-resident path: when the activation, output (and gamma, which the
    // kernel folds in) all live on the gpu context's SVM pool, normalize each
    // feature_size chunk on the device instead of draining the queue for a host
    // pass. Falls through to the host loop below whenever any operand is not
    // SVM-resident, so a node resolved to the cpu context -- or a gpu node with
    // NNTR_GPU_SVM_POOL=0 -- behaves exactly as before.
    bool gpu_done = false;
#if defined(ENABLE_OPENCL)
    {
      const auto in_md = in_step.getMemoryData();
      const auto out_md = out_step.getMemoryData();
      const auto gamma_md = gamma ? gamma->getMemoryData() : nullptr;
      const bool io_svm = in_md && in_md->isSVM() && out_md && out_md->isSVM();
      // The kernels fold gamma in and read it in the activation dtype, so a
      // gamma stored at a different dtype cannot be bound directly; that case
      // keeps the host path, which casts it.
      const bool gamma_bindable =
        gamma && gamma_md && gamma_md->isSVM() && gamma->getDataType() == dtype;
      // cl_mem operands are only bindable at the plane base: getClMem() returns
      // the tensor's whole sub-buffer and the kernel indexes it from 0, while
      // in_step/out_step are views at offset b * featureLen. Restrict to
      // b_size == 1 (offset 0) and to the wrapper's coop-path condition
      // (width % 8 == 0) -- passing a cl_mem the wrapper then ignores would
      // leave the kernel reading the stale SVM shadow of that plane. Same gate
      // as rms_norm_gpu.cpp.
      const bool clmem_ok = io_svm && b_size == 1 && (feature_size % 8u == 0u);
      void *in_cl =
        (clmem_ok && in_step.isClMem()) ? in_step.getClMem() : nullptr;
      void *out_cl =
        (clmem_ok && out_step.isClMem()) ? out_step.getClMem() : nullptr;
      // A cl_mem-resident operand that cannot be bound (b_size > 1, or
      // width % 8 != 0) would make BOTH paths read a stale plane, so leave it
      // to the host path's map/drain rather than dispatching a wrong binding.
      const bool clmem_unbindable =
        (in_step.isClMem() && !in_cl) || (out_step.isClMem() && !out_cl);

      if (io_svm && !clmem_unbindable) {
        if (dtype == ml::train::TensorDim::DataType::FP32 && gamma_bindable &&
            !in_cl && !out_cl) {
          // FP32 has no cl_mem-binding variant, hence the !in_cl && !out_cl.
          nntrainer::rmsnorm_cl(in_step.getData<float>(),
                                gamma->getData<float>(),
                                out_step.getData<float>(), epsilon, n_rows,
                                feature_size, /** use_svm */ true);
          gpu_done = true;
        }
#ifdef ENABLE_FP16
        else if (dtype == ml::train::TensorDim::DataType::FP16 &&
                 (gamma_bindable || !gamma)) {
          // gamma == nullptr selects the gamma-free coop kernel
          // (rmsnorm_cl_fp16_coop_ng). Both variants reduce the sum of squares
          // in FP32, so neither overflows the way an in-half squaring would --
          // the same property the host FP16 intrinsic below relies on.
          // epsilon is passed unfloored, matching the host path.
          // rms_norm_gpu.cpp floors it for full-hidden norms because a
          // near-zero row there gives 1/rms = inf; a per-head row that ever
          // goes near-zero would need the same floor here.
          nntrainer::rmsnorm_cl_fp16(
            in_step.getData<_FP16>(), gamma ? gamma->getData<_FP16>() : nullptr,
            out_step.getData<_FP16>(), epsilon, n_rows, feature_size,
            /** use_svm */ true, out_cl, in_cl);
          gpu_done = true;
        }
#endif
      }
    }
#endif // ENABLE_OPENCL

    if (!gpu_done) {
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
      // Coherent SVM hand-off for a GPU-produced FP16 input read on the host
      // (Gemma4's gamma-free v_norm is what reaches this in an FP16 graph): a
      // producing GPU op may have left its SVM output async-mapped for a next-
      // GPU consumer, so a host reader must drain the queue and take a blocking
      // map first or it reads a stale shadow. Map out for the host write too,
      // and unmap it afterwards so the next consumer sees the result.
      const bool fp16_svm =
        dtype == ml::train::TensorDim::DataType::FP16 &&
        in_step.getMemoryData() && in_step.getMemoryData()->isSVM() &&
        out_step.getMemoryData() && out_step.getMemoryData()->isSVM();
      if (fp16_svm) {
        nntrainer::cl_queue_finish();
        nntrainer::cl_svm_map_force(in_step.getData<_FP16>(),
                                    (size_t)in_step.size() * sizeof(_FP16),
                                    /** read_only */ true);
        nntrainer::cl_svm_map_force(out_step.getData<_FP16>(),
                                    (size_t)out_step.size() * sizeof(_FP16),
                                    /** read_only */ false);
      }
#endif

      if (dtype == ml::train::TensorDim::DataType::FP32) {
        ///@todo rms_norm_wrt_width_something() should be refactored to
        /// nntrainer::Tensor operation.
        // DO NOT USE rms_norm_wrt_width_fp16_intrinsic on an FP32 buffer. It
        // causes overflow!
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(), n_rows,
          in_step.getDim().width(), epsilon);
#ifdef ENABLE_FP16
      } else if (dtype == ml::train::TensorDim::DataType::FP16) {
        // FP16 activation: kernel accumulates squares in FP32 (no overflow).
        nntrainer::rms_norm_wrt_width_fp16_intrinsic(
          in_step.getData<_FP16>(), out_step.getData<_FP16>(), n_rows,
          in_step.getDim().width(), epsilon);
#endif
      } else {
        throw std::invalid_argument(
          "Error: not yet implemented for this data type");
      }
      if (gamma) {
        if (gamma->getDataType() != out_step.getDataType()) {
          nntrainer::Tensor gamma_cast = gamma->clone(out_step.getDataType());
          out_step.multiply_i(gamma_cast);
        } else {
          out_step.multiply_i(*gamma);
        }
      }

#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
      // Hand the host-written output back to the device for the next consumer.
      if (fp16_svm)
        nntrainer::cl_svm_unmap_force(out_step.getData<_FP16>());
#endif
    }

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
