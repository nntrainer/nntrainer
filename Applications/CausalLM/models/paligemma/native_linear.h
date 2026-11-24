// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file native_linear.h
 * @brief Inference projection with checkpoint weights in [output, input] order.
 */
#ifndef __PALIGEMMA_NATIVE_LINEAR_H__
#define __PALIGEMMA_NATIVE_LINEAR_H__

#include <compute_ops.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <layer_node.h>
#include <llm_util.hpp>
#include <node_exporter.h>
#include <thread_manager.h>
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_blas_manager.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

/** Complete backend weight placement during loading, as model.to(device) does.
 */
class PreparedVlaLayer : public nntrainer::LayerImpl {
  void prepare(nntrainer::RunLayerContext &ctx, bool opt_var, bool fsu) {
    if (!opt_var && !fsu)
      for (unsigned int i = 0; i < ctx.getNumWeights(); ++i) {
        auto &weight = ctx.getWeight(i);
        weight.getOps()->fc_prebuild_weight(weight);
      }
  }

public:
  void read(std::ifstream &file, nntrainer::RunLayerContext &ctx, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            nntrainer::TensorDim::DataType dtype, bool fsu, size_t offset = 0,
            bool from_offset = false, int fd = -1) override {
    LayerImpl::read(file, ctx, opt_var, mode, trainable, dtype, fsu, offset,
                    from_offset, fd);
    prepare(ctx, opt_var, fsu);
  }
  void read(nntrainer::ReadSource source, nntrainer::RunLayerContext &ctx,
            bool opt_var, ml::train::ExecutionMode mode, bool trainable,
            nntrainer::TensorDim::DataType dtype, bool fsu, size_t offset = 0,
            bool from_offset = false, int fd = -1) override {
    LayerImpl::read(source, ctx, opt_var, mode, trainable, dtype, fsu, offset,
                    from_offset, fd);
    prepare(ctx, opt_var, fsu);
  }
};

/** Owns one dense weight, with no runtime transposed copy or derived cache. */
class NativeLinear final : public PreparedVlaLayer {
public:
  explicit NativeLinear(bool gated = false) : gated_(gated) {}
  const std::string getType() const override {
    return gated_ ? "paligemma_gate_up" : "paligemma_linear";
  }
  void setProperty(const std::vector<std::string> &values) override {
    LayerImpl::setProperty(nntrainer::loadProperties(values, props_));
  }
  void finalize(nntrainer::InitLayerContext &ctx) override {
    using D = nntrainer::TensorDim::DataType;
    if (ctx.getExecutionMode() != ml::train::ExecutionMode::INFERENCE ||
        ctx.getFormat() != nntrainer::Tformat::NCHW ||
        (ctx.getWeightDataType() != D::FP32 &&
         ctx.getWeightDataType() != D::FP16))
      throw std::invalid_argument(
        "native linear requires dense NCHW inference");
    auto in = ctx.getInputDimensions().at(0);
    auto out = in;
    out.width(std::get<nntrainer::props::Unit>(props_).get());
    out.setTensorType({ctx.getFormat(), ctx.getActivationDataType()});
    ctx.setOutputDimensions({out});
    nntrainer::TensorDim wd(1, 1, (gated_ ? 2 : 1) * out.width(), in.width(),
                            {ctx.getFormat(), ctx.getWeightDataType()});
    weight_ = ctx.requestWeight(wd, nntrainer::Initializer::NONE,
                                nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                "weight", false);
    if (gated_) {
      auto scratch = out;
      scratch.width(2 * out.width());
      scratch_ = ctx.requestTensor(
        scratch, "gate_up", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
    auto &disable = std::get<nntrainer::props::DisableBias>(*layer_impl_props);
    bias_enabled_ = disable.empty() || !disable.get();
    if (bias_enabled_) {
      nntrainer::TensorDim bd(1, 1, 1, out.width(),
                              {ctx.getFormat(), ctx.getActivationDataType()});
      bias_ = ctx.requestWeight(bd, nntrainer::Initializer::NONE,
                                nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                "bias", false);
    }
  }
  void forwarding(nntrainer::RunLayerContext &ctx, bool training) override {
    incremental_forwarding(ctx, 0, ctx.getInput(0).height(), training);
  }
  void incremental_forwarding(nntrainer::RunLayerContext &ctx,
                              unsigned int from, unsigned int to,
                              bool training) override {
    if (training)
      throw std::invalid_argument("native linear is inference-only");
    auto &input = ctx.getInput(0);
    auto &output = ctx.getOutput(0);
    auto xd = input.getDim(), yd = output.getDim();
    xd.batch(1);
    yd.batch(1);
    xd.height(to - from);
    yd.height(to - from);
    for (unsigned int b = 0; b < input.batch(); ++b) {
      auto x =
        input.getSharedDataTensor(xd, b * input.getDim().getFeatureLen(), true);
      auto y = output.getSharedDataTensor(
        yd, b * output.getDim().getFeatureLen(), true);
      if (gated_) {
        auto &buffer = ctx.getTensor(scratch_);
        auto pd = buffer.getDim();
        pd.batch(1);
        pd.height(to - from);
        auto projected = buffer.getSharedDataTensor(
          pd, b * buffer.getDim().getFeatureLen(), true);
        project(x, ctx.getWeight(weight_), projected);
        activate(projected, y);
      } else
        project(x, ctx.getWeight(weight_), y);
      if (bias_enabled_) {
        auto &bias = ctx.getWeight(bias_);
        if (!y.getOps()->fc_bias(y, bias))
          y.add_i(bias);
      }
    }
  }
  static void project(nntrainer::Tensor &x, nntrainer::Tensor &w,
                      nntrainer::Tensor &y) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    using D = nntrainer::TensorDim::DataType;
    if (causallm_engine() == "cuda" &&
        nntrainer::cuda::dev_accessible(x.getData()) &&
        nntrainer::cuda::dev_accessible(w.getData()) &&
        nntrainer::cuda::dev_accessible(y.getData())) {
      auto &blas = nntrainer::cuda::BlasManager::Global();
      const int m = x.batch() * x.channel() * x.height();
      bool ok = false;
      if (x.getDataType() == D::FP32 && w.getDataType() == D::FP32)
        ok = blas.sgemmRowMajor(m, y.width(), x.width(), x.getData<float>(),
                                w.getData<float>(), y.getData<float>(), true);
#ifdef ENABLE_FP16
      if (x.getDataType() == D::FP16 && w.getDataType() == D::FP16)
        ok = blas.hgemmRowMajor(m, y.width(), x.width(), x.getData<uint16_t>(),
                                w.getData<uint16_t>(), y.getData<uint16_t>(),
                                true);
#endif
      if (ok) {
        nntrainer::cuda::StreamManager::Global().maybeFinish();
        return;
      }
    }
    nntrainer::cuda::drain_if_async();
#endif
    x.dot(w, y, false, true);
  }
  static void activate(nntrainer::Tensor &projected, nntrainer::Tensor &out) {
    const size_t width = out.width(), rows = out.height();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    if (causallm_engine() == "cuda" &&
        out.getDataType() == nntrainer::Tdatatype::FP16 &&
        nntrainer::cuda::dev_accessible(projected.getData()) &&
        nntrainer::cuda::dev_accessible(out.getData()) &&
        nntrainer::cuda::cuda_geglu_interleaved_fp16(
          projected.getData<uint16_t>(), out.getData<uint16_t>(), rows * width,
          width))
      return;
    nntrainer::cuda::drain_if_async();
#endif
    auto row = [&](size_t r) {
      if (out.getDataType() == nntrainer::Tdatatype::FP32) {
        auto *p = projected.getData<float>() + r * 2 * width;
        nntrainer::tanh_gelu_mul(width, out.getData<float>() + r * width, p,
                                 p + width);
      } else {
        auto d = out.getDim();
        d.height(1);
        auto gate = projected.getSharedDataTensor(d, r * 2 * width, true);
        auto up = projected.getSharedDataTensor(d, r * 2 * width + width, true);
        auto y = out.getSharedDataTensor(d, r * width, true);
        y.getOps()->geglu(gate, up, y, 1, 0);
      }
    };
    if (rows > 1)
      nntrainer::ThreadManager::Global().parallel_for(0, rows, row);
    else
      row(0);
  }
  void calcDerivative(nntrainer::RunLayerContext &) override {
    throw std::runtime_error("native linear is inference-only");
  }
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &e,
                const ml::train::ExportMethods &method) const override {
    LayerImpl::exportTo(e, method);
    e.saveResult(props_, method, this);
  }

private:
  std::tuple<nntrainer::props::Unit> props_;
  unsigned int weight_ = 0, bias_ = 0;
  bool bias_enabled_ = false;
  bool gated_ = false;
  unsigned int scratch_ = 0;
};

/** CPU self-attention projections share one GEMM and one input packing pass.
 * The three output-major checkpoint matrices are already adjacent on disk. */
class NativeQKV final : public PreparedVlaLayer {
public:
  NativeQKV(unsigned int q_width, unsigned int kv_width) :
    q_width_(q_width), kv_width_(kv_width) {}
  const std::string getType() const override { return "paligemma_qkv"; }
  void finalize(nntrainer::InitLayerContext &ctx) override {
    if (ctx.getExecutionMode() != ml::train::ExecutionMode::INFERENCE ||
        ctx.getWeightDataType() != nntrainer::Tdatatype::FP32 ||
        ctx.getActivationDataType() != nntrainer::Tdatatype::FP32)
      throw std::invalid_argument("native QKV requires FP32 inference");
    auto q = ctx.getInputDimensions().at(0), kv = q;
    q.width(q_width_);
    kv.width(kv_width_);
    ctx.setOutputDimensions({q, kv, kv});
    auto scratch = q;
    scratch.width(q_width_ + 2 * kv_width_);
    scratch_ =
      ctx.requestTensor(scratch, "qkv", nntrainer::Initializer::NONE, false,
                        nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
    nntrainer::TensorDim wd(1, 1, scratch.width(),
                            ctx.getInputDimensions()[0].width(),
                            {ctx.getFormat(), ctx.getWeightDataType()});
    weight_ = ctx.requestWeight(wd, nntrainer::Initializer::NONE,
                                nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                "weight", false);
  }
  void forwarding(nntrainer::RunLayerContext &ctx, bool training) override {
    incremental_forwarding(ctx, 0, ctx.getInput(0).height(), training);
  }
  void incremental_forwarding(nntrainer::RunLayerContext &ctx,
                              unsigned int from, unsigned int to,
                              bool training) override {
    if (training)
      throw std::invalid_argument("native QKV is inference-only");
    auto &input = ctx.getInput(0), &scratch = ctx.getTensor(scratch_);
    const size_t rows = to - from, width = q_width_ + 2 * kv_width_;
    auto xd = input.getDim(), pd = scratch.getDim();
    xd.batch(1);
    xd.height(rows);
    pd.batch(1);
    pd.height(rows);
    for (unsigned int b = 0; b < input.batch(); ++b) {
      auto x =
        input.getSharedDataTensor(xd, b * input.getDim().getFeatureLen(), true);
      auto projected = scratch.getSharedDataTensor(
        pd, b * scratch.getDim().getFeatureLen(), true);
      NativeLinear::project(x, ctx.getWeight(weight_), projected);
      auto *q = ctx.getOutput(0).getData<float>() +
                b * ctx.getOutput(0).getDim().getFeatureLen();
      auto *k = ctx.getOutput(1).getData<float>() +
                b * ctx.getOutput(1).getDim().getFeatureLen();
      auto *v = ctx.getOutput(2).getData<float>() +
                b * ctx.getOutput(2).getDim().getFeatureLen();
      const auto *p = projected.getData<float>();
      for (size_t r = 0; r < rows; ++r) {
        std::copy_n(p + r * width, q_width_, q + r * q_width_);
        std::copy_n(p + r * width + q_width_, kv_width_, k + r * kv_width_);
        std::copy_n(p + r * width + q_width_ + kv_width_, kv_width_,
                    v + r * kv_width_);
      }
    }
  }
  void calcDerivative(nntrainer::RunLayerContext &) override {
    throw std::runtime_error("native QKV is inference-only");
  }
  bool supportBackwarding() const override { return false; }

private:
  unsigned int q_width_, kv_width_, weight_ = 0, scratch_ = 0;
};

} // namespace causallm
#endif
