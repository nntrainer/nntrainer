// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Joonseok Oh <jrock.oh@samsung.com>
 *
 * @file   rms_reverse_norm.cpp
 * @date   27 March 2026
 * @brief  This is Reverse RMS Norm Layer Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <iostream>

#include "rms_reverse_norm.h"

#include <memory_data.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_rmsnorm.h>
#include <cuda_stream_manager.h>
#endif
#if defined(ENABLE_OPENCL)
#include <blas_kernels.h> // rms_reverse_norm_cl_fp16
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSReverseParams { weight, out_scale };

void RMSReverseNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  context.setOutputDimensions(dim);

  // Initialize weight and out_scale parameters
  auto weight_init = nntrainer::props::InitializerInfo::Enum::ONES;
  auto outscale_init = nntrainer::props::InitializerInfo::Enum::ONES;

  if (!std::get<props::RMS_REVERSE_NORM_WEIGHT_INIT>(rms_props).empty()) {
    weight_init =
      std::get<props::RMS_REVERSE_NORM_WEIGHT_INIT>(rms_props).get();
  }

  if (!std::get<props::RMS_REVERSE_NORM_OUTSCALE_INIT>(rms_props).empty()) {
    outscale_init =
      std::get<props::RMS_REVERSE_NORM_OUTSCALE_INIT>(rms_props).get();
  }

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty()) {
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();
  }

  // Request weight parameter (learnable multiplicative weight applied BEFORE
  // norm)
  nntrainer::TensorDim weight_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSReverseParams::weight] = context.requestWeight(
    weight_dim, weight_init, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "weight", true);

  // Request out_scale parameter (learnable scale applied AFTER norm)
  nntrainer::TensorDim outscale_dim(
    1, 1, 1, 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSReverseParams::out_scale] = context.requestWeight(
    outscale_dim, outscale_init, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
    "out_scale", true);
}

void RMSReverseNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {}

void RMSReverseNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &weight =
    context.getWeight(wt_idx[RMSReverseParams::weight]);
  nntrainer::Tensor &out_scale =
    context.getWeight(wt_idx[RMSReverseParams::out_scale]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  // A multi-token step is a prefill even when it does not start at 0 (resumed
  // / chunked prefill), so recognize step_size > 1 as prefill too. The skip
  // check has to run before the 0-based renormalization below, which destroys
  // the original `from`.
  unsigned int step_size = to - from;
  bool is_prefill = !from || step_size > 1;
  if (skip_prefill && is_prefill)
    return;

  if (from) {
    // Normalize to 0-based while preserving step size for multi-token prefill
    to = to - from;
    from = 0;
  }

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      // ReverseRMSNorm order: x * weight → normalize → multiply by out_scale

      // Step 1: Multiply input by weight (BEFORE normalization)
      in_step.multiply_i(weight);

      // Step 2: Compute RMS normalization
      // rsqrt(average(x^2) + eps)
      auto t = in_step.multiply(in_step).average(3).add(epsilon);
      t.inv_sqrt_i();

      // Step 3: Apply normalization
      in_step.multiply(t, out_step);

      // Step 4: Apply output scale (AFTER normalization)
      out_step.multiply_i(out_scale);

      // TODO : Implement Fast Route for FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL)
      // GPU-resident path: run the reverse-norm ON THE GPU (no host op inside
      // the async GPU graph). The former host FP32-temp path was a host
      // reader/writer interleaved with the in-order-but-undrained GPU queue ->
      // it raced the queue and read/wrote a stale SVM shadow -> per-run garbage
      // that corrupted the residual stream. Running it as a GPU op
      // (rms_reverse_norm_cl_fp16, fp32-accumulated, weight-before-norm +
      // scalar out_scale) removes the host<->GPU boundary entirely, mirroring
      // reshaped_rms_norm / rms_norm_gpu.
      const auto in_md = in_step.getMemoryData();
      const auto out_md = out_step.getMemoryData();
      const auto w_md = weight.getMemoryData();
      const auto os_md = out_scale.getMemoryData();
      const bool gpu_svm =
        in_md && in_md->isSVM() && out_md && out_md->isSVM() && w_md &&
        w_md->isSVM() && os_md && os_md->isSVM() &&
        weight.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        out_scale.getDataType() == ml::train::TensorDim::DataType::FP16;
      if (gpu_svm) {
        void *in_cl = (in_step.getOffset() == 0 && in_step.isClMem())
                        ? in_step.getClMem()
                        : nullptr;
        void *out_cl = (out_step.getOffset() == 0 && out_step.isClMem())
                         ? out_step.getClMem()
                         : nullptr;
        nntrainer::rms_reverse_norm_cl_fp16(
          in_step.getData<_FP16>(), weight.getData<_FP16>(),
          out_scale.getData<_FP16>()[0], out_step.getData<_FP16>(), epsilon,
          in_step_dim.height(), in_step_dim.width(), /*use_svm=*/true, out_cl,
          in_cl);
        continue;
      }
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // engine=cuda device path: the FC-produced input lives on the device
      // activation pool, so the host FP32-temp fallback below FAULTS reading
      // it (a CUDA-engine SIGSEGV in avx2::vcvt_f16_f32). Run the reverse-norm
      // as a device kernel (cuda_rms_reverse_norm_fp16, FP32-accumulated,
      // weight-before-norm + scalar out_scale read on-device) -- the CUDA
      // mirror of the OpenCL rms_reverse_norm_cl_fp16 branch above.
      if (weight.getDataType() == ml::train::TensorDim::DataType::FP16 &&
          out_scale.getDataType() == ml::train::TensorDim::DataType::FP16) {
        auto *ip =
          reinterpret_cast<const unsigned short *>(in_step.getData<_FP16>());
        auto *wp =
          reinterpret_cast<const unsigned short *>(weight.getData<_FP16>());
        auto *osp =
          reinterpret_cast<const unsigned short *>(out_scale.getData<_FP16>());
        auto *op =
          reinterpret_cast<unsigned short *>(out_step.getData<_FP16>());
        auto dev_ok = [](const void *ptr) {
          return ptr && nntrainer::cuda::dev_accessible(ptr);
        };
        if (dev_ok(ip) && dev_ok(wp) && dev_ok(osp) && dev_ok(op) &&
            nntrainer::cuda::cuda_rms_reverse_norm_fp16(
              ip, wp, osp, op, epsilon, in_step_dim.height(),
              in_step_dim.width()))
          continue;
      }
#endif
      // Host path (ARM CPU / non-GPU-resident): FP32-temp compute.
      ml::train::TensorDim instep_dim = in_step_dim;
      ml::train::TensorDim outstep_dim = out_step_dim;

      instep_dim.setDataType(ml::train::TensorDim::DataType::FP32);
      outstep_dim.setDataType(ml::train::TensorDim::DataType::FP32);

      nntrainer::Tensor in_step32(instep_dim, true);
      nntrainer::Tensor out_step32(outstep_dim, true);

      in_step32.copyData(in_step);

      // weight/out_scale share the activation dtype (packed=false), which is
      // FP16 on an enable-fp16 build. The math below runs in an FP32 temporary,
      // so multiplying by the FP16 tensors directly reinterprets their bytes as
      // FP32 (Tensor::multiply -> apply_broadcast reads getData<float>()) ->
      // garbage. Upcast the scale tensors to FP32 first so every multiply is
      // dtype-matched. (Only this layer needs it; other norms apply their
      // scale after converting the buffer back to the activation dtype.)
      ml::train::TensorDim weight32_dim = weight.getDim();
      weight32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
      nntrainer::Tensor weight32(weight32_dim, true);
      weight32.copyData(weight);

      ml::train::TensorDim out_scale32_dim = out_scale.getDim();
      out_scale32_dim.setDataType(ml::train::TensorDim::DataType::FP32);
      nntrainer::Tensor out_scale32(out_scale32_dim, true);
      out_scale32.copyData(out_scale);

      // ReverseRMSNorm order: x * weight → normalize → multiply by out_scale

      // Step 1: Multiply input by weight (BEFORE normalization)
      in_step32.multiply_i(weight32);

      // Step 2: Compute RMS normalization
      auto t = in_step32.multiply(in_step32).average(3).add(epsilon);
      t.inv_sqrt_i();

      // Step 3: Apply normalization
      in_step32.multiply(t, out_step32);

      // Step 4: Apply output scale (AFTER normalization)
      out_step32.multiply_i(out_scale32);

      out_step.copyData(out_step32);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    }

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "weight:" << weight
              << "out_scale:" << out_scale << std::endl;
#endif
  }
}

void RMSReverseNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output_dim =
    context.getOutput(SINGLE_INOUT_IDX).getDim();

  input_dim.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(SINGLE_INOUT_IDX, output_dim);
}

void RMSReverseNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // Training not implemented yet
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_reverse_norm_layer() {
  auto layer = new RMSReverseNormLayer();
  return layer;
}

void destroy_rms_reverse_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_rms_reverse_norm_layer, destroy_rms_reverse_norm_layer};
}

#endif

} // namespace causallm
