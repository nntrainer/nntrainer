// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 UGyeong Song <thddnrud@snu.ac.kr>
 *
 * @file   conv2d_transpose_layer.h
 * @date   13 October 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author UGyeong Song <thddnrud@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is Transposed Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <conv2d_transpose_layer.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <tensor_dim.h>
#include <thread>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()});
} // [in_channel*kernel_h*kernel_w, out_w*out_h]

/**
 * @brief     reconstruct image data from 2d column matrix
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] image image tensor to put
 */
static void col2im_transpose(
  const Tensor &col_matrix, const TensorDim &kdim,
  const std::array<unsigned, 4> &padding,
  const std::array<props::Stride, CONV2D_TRANSPOSE_DIM> &mstride,
  const std::array<props::Dilation, CONV2D_TRANSPOSE_DIM> &dilation,
  Tensor &image) {
  auto [pt, pb, pl, pr] = padding;

  unsigned int channel = image.channel();
  int in_height = image.height();
  int in_width = image.width();

  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int height = (in_height - 1) * mstride[0] + eff_k_height;
  unsigned int width = (in_width - 1) * mstride[1] + eff_k_height;

  unsigned int out_height = height - pt - pb; // col_matrix.height
  unsigned int out_width = width - pl - pr;   // col_matrix.width

  image.setZero();

  int h_stride_end = height - eff_k_height - pt;
  int w_stride_end = width - eff_k_width - pl;

  /// get a patch, size of kernel
  /// hs is height_strided, ws is width_strided
  unsigned int owidth = col_matrix.width();
  unsigned int base_im_w = 0;

  unsigned int H = k_height;
  unsigned int W = k_width;
  unsigned int C = image.channel();

  int out_i = -1;
  for (unsigned int oh = 0; oh < out_height; ++oh) {
    for (unsigned int ow = 0; ow < out_width; ++ow) {
      out_i++;
      int out_j = -1;
      // half_cpu o = bias->buf[oc];
      for (unsigned int c = 0; c < C; ++c) {
        for (unsigned int r = 0; r < H; ++r) {
          for (unsigned int s = 0; s < W; ++s) {
            out_j++;
            if ((oh - (r * dilation[0] - pt)) % mstride[0] != 0)
              continue;
            if ((ow - (s * dilation[1] - pl)) % mstride[1] != 0)
              continue;
            unsigned int h = (oh - (r * dilation[0] - pt)) / mstride[0];
            unsigned int w = (ow - (s * dilation[1] - pl)) / mstride[1];
            if (h >= H || w >= W)
              continue;
            float *val = image.getAddress<float>(0, c, h, w);
            *val += col_matrix.getValue<float>(0, 0, out_i, out_j);
            // out_data[(out_i)*owidth + out_j] += in.getValue<float>(0,c,h,w)
          }
        }
      }
    }
  }
}

/**
 * @brief	reform the data to 2d matrix
 * a region is sampled considering @a padding, @a mstride of unit, @a kdim
 * Each region is mapped to one column
 *
 * @param [in] in input data
 * @param [in] kdim kernel dimension for defined number of row
 * @param [in] padding padding information
 * @param [in] mstride stride value : x, y direction
 * @param [in] dilation kernel dilation factor : x, y each
 * @param [out] out out tensor
 */
static void im2col_transpose(
  const Tensor &in, const TensorDim &kdim,
  const std::array<unsigned int, 4> &padding,
  const std::array<props::Stride, CONV2D_TRANSPOSE_DIM> &mstride,
  const std::array<props::Dilation, CONV2D_TRANSPOSE_DIM> &dilation,
  Tensor &out) {
  auto [pt, pb, pl, pr] = padding;

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();

  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int height = (in_height - 1) * mstride[0] + eff_k_height;
  unsigned int width = (in_width - 1) * mstride[1] + eff_k_height;

  unsigned int out_height = height - pt - pb;
  unsigned int out_width = width - pl - pr;

  out.reshape(
    TensorDim({out_height * out_width, in.channel() * k_height * k_width}));

  float *out_data = out.getData();

  int h_stride_end = height - eff_k_height - pt;
  int w_stride_end = width - eff_k_width - pl;

  /// get a patch, size of kernel
  unsigned int owidth = out.width();
  unsigned int base_im_w = 0;

  unsigned int H = k_height;
  unsigned int W = k_width;
  unsigned int C = in.channel();

  int out_i = -1;
  for (unsigned int oh = 0; oh < out_height; ++oh) {
    for (unsigned int ow = 0; ow < out_width; ++ow) {
      out_i++;
      int out_j = -1;
      // half_cpu o = bias->buf[oc];
      for (unsigned int c = 0; c < C; ++c) {
        for (unsigned int r = 0; r < H; ++r) {
          for (unsigned int s = 0; s < W; ++s) {
            out_j++;
            if ((oh - (r * dilation[0] - pt)) % mstride[0] != 0)
              continue;
            if ((ow - (s * dilation[1] - pl)) % mstride[1] != 0)
              continue;
            unsigned int h = (oh - (r * dilation[0] - pt)) / mstride[0];
            unsigned int w = (ow - (s * dilation[1] - pl)) / mstride[1];
            if (h >= H || w >= W)
              continue;
            out_data[(out_i)*owidth + out_j] += in.getValue<float>(0, c, h, w);
          }
        }
      }
    }
  }
}

} // namespace

enum ConvParams { weight, bias };

Conv2DTransposeLayer::Conv2DTransposeLayer(
  const std::array<unsigned int, CONV2D_TRANSPOSE_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(
    props::FilterSize(), std::array<props::KernelSize, CONV2D_TRANSPOSE_DIM>(),
    std::array<props::Stride, CONV2D_TRANSPOSE_DIM>(), props::Padding2D(),
    std::array<props::Dilation, CONV2D_TRANSPOSE_DIM>()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void Conv2DTransposeLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_TRANSPOSE_DIM>>(conv_props);
  auto &stride =
    std::get<std::array<props::Stride, CONV2D_TRANSPOSE_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_TRANSPOSE_DIM>>(conv_props);

  TensorDim kernel_dim =
    TensorDim(filter_size, in_dim.channel(), kernel_size[0], kernel_size[1]);
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1);

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, kernel_dim, {stride[0], stride[1]},
                       {dilation[0], dilation[1]});

  wt_idx[ConvParams::weight] = context.requestWeight(
    kernel_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  auto [pt, pb, pl, pr] = padding;

  unsigned int channel = in_dim.channel();
  int in_height = in_dim.height();
  int in_width = in_dim.width();

  unsigned int k_height = kernel_size[0];
  unsigned int k_width = kernel_size[1];

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int height = (in_height - 1) * stride[0] + eff_k_height;
  unsigned int width = (in_width - 1) * stride[1] + eff_k_height;

  unsigned int out_height = height - pt - pb;
  unsigned int out_width = width - pl - pr;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height(out_height);
  out_dim.width(out_width);
  context.setOutputDimensions({out_dim});

  NNTR_THROW_IF(height < kernel_size[0] || width < kernel_size[1],
                std::invalid_argument)
    << "Failed to initialize: in size + padding is smaller than effective "
       "kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(height - padding[0] - kernel_size[0] > IM ||
                  width - padding[2] - kernel_size[1] > IM,
                std::invalid_argument)
    << "Failed to initialize: Calculated patch end is over int max";
}

void Conv2DTransposeLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride =
    std::get<std::array<props::Stride, CONV2D_TRANSPOSE_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_TRANSPOSE_DIM>>(conv_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  /** Calculate Convolution 2D Transpose
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                     [Input Channel * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   * Kernel_size[0]   |_|_|_|_|      |_|_|_|_|
   *  * Kenel_size[1]     | | | | | .... | | | | |
   *    (height)]         |_|_|_|_|      |_|_|_|_|
   *                      | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimention
   *   -> [Channel ( = filter_size = output_dim.channel )]
   *       x [output_dim.height x output_dim.width]
   */
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /**
   * Below sets the pad area values to zero
   * it is faster to do this way than seting selective area to zero
   */
  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                            void *user_data) {
    Tensor result = Tensor(
      calcCol2ImOutputDim(out_dim, filter_dim)); // result is temporary data
    result.setZero();
    for (unsigned int b = s; b < e; ++b) {
      Tensor out = hidden_.getBatchSlice(b, 1);
      out.reshape({filter_size, out_dim.width() * out_dim.height()});
      Tensor in_sub = input_.getBatchSlice(b, 1);

      im2col_transpose(in_sub, filter_dim, padding, stride, dilation, result);
      filter_kernel.dot(result, out, false, true);
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    forwarding_job(0, in_dim.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
    status = hidden_.add_i(bias_kernel);
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument("[Conv2DTranspose] adding bias failed");
    }
  }
}

void Conv2DTransposeLayer::calcDerivative(RunLayerContext &context) {
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride =
    std::get<std::array<props::Stride, CONV2D_TRANSPOSE_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_TRANSPOSE_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(
        {filter_size, derivative.width() * derivative.height()});
      filter_kernel.dot(deriv_sub, result, true, false);
      col2im_transpose(result, filter_dim, padding, stride, dilation,
                       in_deriv_sub);
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DTransposeLayer::calcGradient(RunLayerContext &context) {
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride =
    std::get<std::array<props::Stride, CONV2D_TRANSPOSE_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_TRANSPOSE_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());
  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  if (workers.getNumWorkers() > 1) {

    TensorDim delK_ext = filter_dim_squeezed;
    delK_ext.batch(input_.batch());

    Tensor delK_par = Tensor(delK_ext);
    delK_par.setZero();

    auto calc_grad_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                             void *user_data) {
      Tensor result =
        Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
      result.setZero();
      for (unsigned int b = s; b < e; ++b) {
        Tensor deriv_sub = derivative.getBatchSlice(b, 1);
        Tensor delK_sub = delK_par.getBatchSlice(b, 1);
        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1);

        /**
         * @todo this result can be cached from the forward iteration at the
         * expense of memory. In this case, memory of im2col_result must be
         * saved for the whole batch. try this while benchmarking.
         */
        im2col_transpose(in_sub, filter_dim, padding, stride, dilation, result);
        deriv_sub.dot(result, delK_sub, false, false);
      }
      result.deallocate();
    };

    workers.setCallback(calc_grad_job, nullptr);

    workers.run();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor delK_sub = delK_par.getBatchSlice(b, 1);
      delK.add_i(delK_sub);
    }

  } else {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
    result.setZero();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(out_dim_squeezed);

      Tensor in_sub = input_.getBatchSlice(b, 1);

      /**
       * @todo this result can be cached from the forward iteration at the
       * expense of memory. In this case, memory of im2col_result must be saved
       * for the whole batch. try this while benchmarking.
       */
      im2col_transpose(in_sub, filter_dim, padding, stride, dilation, result);
      deriv_sub.dot(result, delK, false, false, b == 0 ? 0.0f : 1.0f);
    }
    result.deallocate();
  }
  delK.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
    derivative.sum({0, 2, 3}, delBias);
  }
}

void Conv2DTransposeLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv2DTransposeLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */
