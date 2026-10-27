// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.cpp
 * @date   27 Oct 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 * @todo merge concat and split layer to a common implementation
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <concat_layer.h>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <thread_manager.h>
#include <util_func.h>

namespace nntrainer {
ConcatLayer::ConcatLayer() : Layer(), leading_helper_dim(1) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ConcatLayer::finalize(InitLayerContext &context) {
  auto &concat_dimension_prop = std::get<props::ConcatDimension>(concat_props);
  /** for backward compatibility, default concat dimension will be channel */
  /// @todo this is hacky way to force concat dimension to width if channel
  /// dimension is taken, this is because recurrent realizer, return sequence
  /// exploits concat layer but have no control over where to stack/axis
  unsigned int concat_dimension =
    context.getInputDimensions().front().channel() > 1 ? 3 : 1;
  if (!concat_dimension_prop.empty())
    concat_dimension = concat_dimension_prop.get();
  concat_dimension_cache = concat_dimension;

  /**
   * The concat is only done along the axis dimension.
   * For example, consider 2 inputs a, b with dimensions [b,c,h,w] each
   * 1. concat_dimension = 1, output_dim = [b,c_a+c_b,h,w]
   * 2. concat_dimension = 2, output_dim = [b,c,h_a+h_b,w]
   * 3. concat_dimension = 3, output_dim = [b,c,h,w_a+w_b]
   */
  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  unsigned int concat_dim_val = input_dim_0.getTensorDim(concat_dimension);

  for (unsigned int idx = 1; idx < input_dims.size(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 0; i < ml::train::TensorDim::getNumDim(); ++i) {
      if (i == concat_dimension)
        continue;
      NNTR_THROW_IF(input_dim_0[i] != dim[i], std::runtime_error)
        << "Error: concat layer requires same shape from all input layers "
           "along non-concat dimension. Mismatch at dim "
        << i << ". "
        << "input_dim_0: [" << input_dim_0.batch() << ","
        << input_dim_0.channel() << "," << input_dim_0.height() << ","
        << input_dim_0.width() << "], "
        << "input_dim[" << idx << "]: [" << dim.batch() << "," << dim.channel()
        << "," << dim.height() << "," << dim.width() << "]. "
        << "concat_dimension: " << concat_dimension;
    }
    concat_dim_val += dim[concat_dimension];
  }

  TensorDim output_dim = input_dim_0;
  output_dim.setTensorDim(concat_dimension, concat_dim_val);

  context.setOutputDimensions({output_dim});

  /**
   * The following helper shapes facilitate efficient concatenation and split of
   * the data.
   *
   * The helper shapes are created by consolidating all the dimensions before
   * the concat dimension to the first axis and all the remaining dimensions to
   * the last axis.
   *
   * @note This is possible since the data starting from the concat dimension to
   * the end is always continuous.
   *
   * @example the following shows how the helper dimension will look with given
   * inputs and concat dimension.
   *
   *          | cat_dim 1 | cat_dim 2 | cat_dim 3
   *  --------|-----------|-----------|-----------
   *  input0  |  2:1:2:3  |  1:2:1:3  |  1:2:2:3
   *  input1  |  2:3:2:3  |  1:2:3:3  |  1:2:2:1
   *  --------|-----------|-----------|-----------
   *  helper0 |  2:1:1:6  |  2:1:1:3  |  4:1:1:3
   *  helper1 |  2:1:1:18 |  2:1:1:9  |  4:1:1:1
   *
   */
  /// Setup output_reshape_helper (how output should be reshaped)
  output_reshape_helper.channel(1);
  output_reshape_helper.height(1);
  output_reshape_helper.width(1);
  for (unsigned int axis = concat_dimension;
       axis < ml::train::TensorDim::getNumDim(); ++axis) {
    output_reshape_helper.width(output_reshape_helper.width() *
                                output_dim.getTensorDim(axis));
  }

  /// Setup input_reshape_helper (how inputs should be reshaped)
  input_reshape_helper.resize(input_dims.size());

  for (unsigned int idx = 0; idx < input_reshape_helper.size(); idx++) {
    input_reshape_helper[idx].channel(1);
    input_reshape_helper[idx].height(1);
    input_reshape_helper[idx].width(1);

    for (unsigned int axis = concat_dimension;
         axis < ml::train::TensorDim::getNumDim(); ++axis) {

      input_reshape_helper[idx].width(input_reshape_helper[idx].width() *
                                      input_dims[idx].getTensorDim(axis));
    }
  }

  leading_helper_dim = 1;
  for (unsigned int idx = 1; idx < concat_dimension; ++idx) {
    leading_helper_dim *= output_dim.getTensorDim(idx);
  }

  setBatch(input_dims[SINGLE_INOUT_IDX].batch());
}

void ConcatLayer::forwarding(RunLayerContext &context, bool training) {
  /**
   * Forwarding in ConcatLayer works as follows
   *
   *    in1        in2       in3                  output
   * |---0---| |----3----| |--6--|      |---0---||----3----||--6--|
   * |---1---| |----4----| |--7--|  =>  |---1---||----4----||--7--|
   * |---2---| |----5----| |--8--|      |---2---||----5----||--8--|
   *
   * @note For each input tensor, it iterates batches and copies the entire
   * width size to the corresponding output position. In the diagram above, the
   * row would be a batch, and the column would be a width. the number of each
   * block in the diagram indicates the order of copy to output.
   *
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();

  // NHWC path: channel is physical-innermost, so a channel-axis concat
  // interleaves per-pixel channel runs. The NCHW reshape-helper path below
  // assumes channel-major planes and would copy into the wrong physical
  // offsets, so handle NHWC separately. Only the channel axis (logical 1) is
  // supported here, matching the graph usage.
  if (out_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
      concat_dimension_cache == 1) {
    const unsigned int B = out_dim.batch();
    const unsigned int H = out_dim.height();
    const unsigned int W = out_dim.width();
    const size_t HW = (size_t)H * W;
    // W8A8 QINT8 concat: inputs carry per-tensor scales; the output scale is
    // the max of the input scales and each input is rescaled int8->int8 by
    // s_i/s_out during its copy (identity memcpy when scales match). Computed
    // up front since every input copy needs the common output scale.
    float q8_out_scale = 0.f;
    if (output.getDataType() == TensorDim::DataType::QINT8) {
      for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
        Tensor &input = context.getInput(idx);
        if (input.getDataType() == TensorDim::DataType::QINT8) {
          q8_out_scale = std::max(q8_out_scale, input.getScale<float>()[0]);
        } else if (input.getDataType() == TensorDim::DataType::FP32) {
          float amax = 0.f;
          const float *d = input.getData<float>();
          const size_t sz = input.size();
          for (size_t i = 0; i < sz; ++i) {
            float a = std::fabs(d[i]);
            if (a > amax)
              amax = a;
          }
          // Match the asymmetric scale formula used everywhere else in the
          // W8A8/PERCH path (conv2d_layer.cpp's epilogue): symmetric amax/127
          // would compute a systematically different (larger) scale than what
          // the other QINT8 inputs' asymmetric encoding assumes.
          static const bool asym_scale =
            std::getenv("NNTR_W8A8_PERCH") != nullptr &&
            std::getenv("NNTR_W8A8_SYM") == nullptr;
          constexpr float kActOff = 0.27846455f;
          float sc = amax > 0.f
                       ? (asym_scale ? (amax + kActOff) / 255.f : amax / 127.f)
                       : 1.f;
          q8_out_scale = std::max(q8_out_scale, sc);
        }
      }
      if (q8_out_scale <= 0.f)
        q8_out_scale = 1.f;
      output.getScale<float>()[0] = q8_out_scale;

      if (const char *dl = std::getenv("NNTR_CONCAT_DEBUG");
          dl && std::strstr(dl, context.getName().c_str())) {
        for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
          Tensor &input = context.getInput(idx);
          if (input.getDataType() == TensorDim::DataType::QINT8) {
            const int8_t *q = input.getData<int8_t>();
            const float sc = input.getScale<float>()[0];
            int8_t qmin = 127, qmax = -128;
            for (size_t i = 0; i < input.size(); ++i) {
              if (q[i] < qmin)
                qmin = q[i];
              if (q[i] > qmax)
                qmax = q[i];
            }
            std::cerr << "[CONCATDBG] " << context.getName() << " input[" << idx
                      << "] scale=" << sc << " qmin=" << (int)qmin
                      << " qmax=" << (int)qmax << " n=" << input.size()
                      << " ptr=" << (const void *)q << "\n"
                      << std::flush;
          }
        }
        std::cerr << "[CONCATDBG] " << context.getName()
                  << " q8_out_scale=" << q8_out_scale << "\n"
                  << std::flush;
      }
    }
    unsigned int c_offset = 0;
    for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
      Tensor &input = context.getInput(idx);
      const unsigned int Ci = input.channel();
      if (output.getDataType() == TensorDim::DataType::QINT8) {
        if (input.getDataType() == TensorDim::DataType::QINT8) {
          const int8_t *src = input.getData<int8_t>();
          int8_t *dst = output.getData<int8_t>();
          const float mult = input.getScale<float>()[0] / q8_out_scale;
          const bool identity = std::fabs(mult - 1.f) < 1e-12f;
          // Per-channel W8A8 (default asymmetric): tensors use the
          // shared-offset affine code x = (q + 128) * s - C. Because C is
          // COMMON to every branch it cancels in the rescale --
          //   q' = round((q + 128) * s_i / s_o) - 128
          // -- so only the +-128 shift differs from the symmetric formula (and
          // s_o = max(s_i) still exactly covers the union range). The identity
          // (mult == 1) memcpy is unchanged in both conventions.
          static const bool asym = std::getenv("NNTR_W8A8_PERCH") != nullptr &&
                                   std::getenv("NNTR_W8A8_SYM") == nullptr;
          auto &tm = ThreadManager::Global();
          for (unsigned int b = 0; b < B; ++b) {
            const size_t base = (size_t)b * HW;
            tm.parallel_for(0, HW, [&](size_t p) {
              const int8_t *s = src + (base + p) * Ci;
              int8_t *d = dst + (base + p) * out_dim.channel() + c_offset;
              if (identity) {
                std::memcpy(d, s, (size_t)Ci);
              } else if (asym) {
                unsigned int c = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
                // q' = round((q + 128) * mult) - 128, saturating to int8.
                // FRINTA (vrndaq) rounds ties away from zero, matching
                // std::round; the rounded value is an exact integer float so
                // vcvtq truncates it losslessly, and the saturating narrows
                // reproduce the clamp.
                const float32x4_t vmult = vdupq_n_f32(mult);
                const float32x4_t v128 = vdupq_n_f32(128.f);
                auto proc = [&](int32x4_t a) {
                  float32x4_t f = vcvtq_f32_s32(a);
                  f = vaddq_f32(f, v128);
                  f = vmulq_f32(f, vmult);
                  f = vrndaq_f32(f);
                  f = vsubq_f32(f, v128);
                  return vcvtq_s32_f32(f);
                };
                for (; c + 16 <= Ci; c += 16) {
                  int8x16_t q8 = vld1q_s8(s + c);
                  int16x8_t lo = vmovl_s8(vget_low_s8(q8));
                  int16x8_t hi = vmovl_s8(vget_high_s8(q8));
                  int32x4_t r0 = proc(vmovl_s16(vget_low_s16(lo)));
                  int32x4_t r1 = proc(vmovl_s16(vget_high_s16(lo)));
                  int32x4_t r2 = proc(vmovl_s16(vget_low_s16(hi)));
                  int32x4_t r3 = proc(vmovl_s16(vget_high_s16(hi)));
                  int16x8_t n0 = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
                  int16x8_t n1 = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));
                  vst1q_s8(d + c, vcombine_s8(vqmovn_s16(n0), vqmovn_s16(n1)));
                }
#endif
                for (; c < Ci; ++c) {
                  float q = std::round(((float)s[c] + 128.f) * mult) - 128.f;
                  d[c] = (int8_t)std::max(-128.f, std::min(127.f, q));
                }
              } else {
                for (unsigned int c = 0; c < Ci; ++c) {
                  float q = std::round((float)s[c] * mult);
                  d[c] = (int8_t)std::max(-128.f, std::min(127.f, q));
                }
              }
            });
          }
        } else if (input.getDataType() == TensorDim::DataType::FP32) {
          const float *src = input.getData<float>();
          int8_t *dst = output.getData<int8_t>();
          const float inv = 1.f / q8_out_scale;
          static const bool asym = std::getenv("NNTR_W8A8_PERCH") != nullptr &&
                                   std::getenv("NNTR_W8A8_SYM") == nullptr;
          auto &tm = ThreadManager::Global();
          for (unsigned int b = 0; b < B; ++b) {
            const size_t base = (size_t)b * HW;
            tm.parallel_for(0, HW, [&](size_t p) {
              const float *s = src + (base + p) * Ci;
              int8_t *d = dst + (base + p) * out_dim.channel() + c_offset;
              for (unsigned int c = 0; c < Ci; ++c) {
                if (asym) {
                  // x = (q+128)*s_o - kActOff  =>  q = (x+kActOff)/s_o - 128.
                  // (Was `128.f*q8_out_scale` instead of kActOff -- a wrong
                  // offset term unrelated to the actual asymmetric encoding,
                  // confirmed via on-device bisection 2026-08-11: this branch
                  // is the real corruption source for any conv downstream of
                  // a concat mixing a genuinely-FP32 input with QINT8 ones.)
                  constexpr float kActOff = 0.27846455f;
                  float q = std::round((s[c] + kActOff) * inv) - 128.f;
                  d[c] = (int8_t)std::max(-128.f, std::min(127.f, q));
                } else {
                  float q = std::round(s[c] * inv);
                  d[c] = (int8_t)std::max(-128.f, std::min(127.f, q));
                }
              }
            });
          }
        }
      } else if (input.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *src = input.getData<_FP16>();
        _FP16 *dst = output.getData<_FP16>();
        auto &tm = ThreadManager::Global();
        for (unsigned int b = 0; b < B; ++b) {
          const size_t base = (size_t)b * HW;
          tm.parallel_for(0, HW, [&](size_t p) {
            const _FP16 *s = src + (base + p) * Ci;
            _FP16 *d = dst + (base + p) * out_dim.channel() + c_offset;
            std::memcpy(d, s, (size_t)Ci * sizeof(_FP16));
          });
        }
#else
        throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
      } else if (input.getDataType() == TensorDim::DataType::FP32) {
        const float *src = input.getData<float>();
        float *dst = output.getData<float>();
        auto &tm = ThreadManager::Global();
        for (unsigned int b = 0; b < B; ++b) {
          const size_t base = (size_t)b * HW;
          tm.parallel_for(0, HW, [&](size_t p) {
            const float *s = src + (base + p) * Ci;
            float *d = dst + (base + p) * out_dim.channel() + c_offset;
            std::memcpy(d, s, (size_t)Ci * sizeof(float));
          });
        }
      } else if (input.getDataType() == TensorDim::DataType::QINT8) {
        // output.getDataType() != QINT8 here (that case is handled by the
        // block above) -- i.e. output is FP32 (ConcatLayer::finalize()
        // copies input[0]'s dtype for the output, so a QINT8 input at a
        // non-zero index alongside an FP32 input[0] -- e.g. AdditionLayer's
        // now-FP32 residual sum, see addition_layer.cpp -- lands here).
        // Dequantize into the FP32 output; nothing requantizes since the
        // output itself stays FP32.
        const int8_t *src = input.getData<int8_t>();
        float *dst = output.getData<float>();
        const float sc = input.getScale<float>()[0];
        static const bool asym = std::getenv("NNTR_W8A8_PERCH") != nullptr &&
                                 std::getenv("NNTR_W8A8_SYM") == nullptr;
        constexpr float kActOff = 0.27846455f;
        auto &tm = ThreadManager::Global();
        for (unsigned int b = 0; b < B; ++b) {
          const size_t base = (size_t)b * HW;
          tm.parallel_for(0, HW, [&](size_t p) {
            const int8_t *s = src + (base + p) * Ci;
            float *d = dst + (base + p) * out_dim.channel() + c_offset;
            for (unsigned int c = 0; c < Ci; ++c) {
              d[c] =
                asym ? ((float)s[c] + 128.f) * sc - kActOff : sc * (float)s[c];
            }
          });
        }
      }
      c_offset += Ci;
    }

    // Dump one named layer's real output (from inside the untruncated
    // production forward pass), for external bisection. Mirrors the
    // NNTR_DUMP_LAYER hooks in conv2d_layer.cpp/addition_layer.cpp/etc.
    if (const char *dl = std::getenv("NNTR_DUMP_LAYER");
        dl && context.getName() == dl) {
      const char *dp = std::getenv("NNTR_DUMP_PATH");
      if (dp) {
        std::vector<float> buf(output.size());
        if (output.getDataType() == TensorDim::DataType::QINT8) {
          const int8_t *q = output.getData<int8_t>();
          const float sc = output.getScale<float>()[0];
          static const bool perch_asym_env =
            std::getenv("NNTR_W8A8_PERCH") != nullptr &&
            std::getenv("NNTR_W8A8_SYM") == nullptr;
          constexpr float kActOff = 0.27846455f;
          for (size_t i = 0; i < output.size(); ++i)
            buf[i] = perch_asym_env ? ((float)q[i] + 128.f) * sc - kActOff
                                    : sc * (float)q[i];
        } else if (output.getDataType() == TensorDim::DataType::FP32) {
          const float *d = output.getData<float>();
          std::copy(d, d + output.size(), buf.begin());
        }
        std::ofstream of(dp, std::ios::binary);
        of.write(reinterpret_cast<const char *>(buf.data()),
                 buf.size() * sizeof(float));
      }
    }
    return;
  }

  output.reshape(output_reshape_helper);
  unsigned int output_width_offset = 0;
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);
    unsigned int data_copy_size = irh.width();

    /** loop over the dimensions before the concat dimension */
    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** copy continous tensor data (reshaped width) */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        Tensor dest_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size, tensor_type});
        const Tensor source_tensor =
          Tensor::Map<float>(input.getAddress<float>(batch, 0, 0, 0),
                             data_copy_size * sizeof(float),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** copy continous tensor data (reshaped width) */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        Tensor dest_tensor = Tensor::Map<_FP16>(
          output.getAddress<_FP16>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(_FP16),
          {1, 1, 1, data_copy_size, tensor_type});
        const Tensor source_tensor =
          Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, 0, 0),
                             data_copy_size * sizeof(_FP16),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    output_width_offset += irh.width();
    input.reshape(in_dim);
  }

  output.reshape(out_dim);
}

void ConcatLayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size = output_reshape_helper.width();

  // @todo: this implementation is only works when axis is 3(width). Consider
  // for other axes
  unsigned int batch_channel = out_dim.batch() * out_dim.channel();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    /** loop over the dimensions before the concat dimension */
    for (unsigned int batch = batch_channel * from; batch < batch_channel * to;
         batch++) {
      /** loop over the concat dimension itself */
      for (unsigned int count = 0; count < irh.height(); count++) {
        Tensor dest_tensor = Tensor::Map(
          output.getAddress(batch, 0, output_height_offset + count, 0),
          data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
        const Tensor source_tensor = Tensor::Map(
          input.getAddress(batch, 0, count, 0), data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size});
        dest_tensor.copy(source_tensor);
      }
    }

    input.reshape(in_dim);
    output_height_offset += irh.height();
  }

  output.reshape(out_dim);
}

void ConcatLayer::calcDerivative(RunLayerContext &context) {
  /**
   * calcDerivative in ConcatLayer works as follows
   *
   *           output                    in1        in2       in3
   * |---0---||----3----||--6--|      |---0---| |----3----| |--6--|
   * |---1---||----4----||--7--|  =>  |---1---| |----4----| |--7--|
   * |---2---||----5----||--8--|      |---2---| |----5----| |--8--|
   *
   * @note For each input tensor, it iterates batches and copies the entire
   * input width size from the output tensor to the corresponding input. In the
   * diagram above, the row would be a batch, and the column would be a width.
   * The number of each block in the diagram indicates the order of copy to
   * inputs.
   *
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor output = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  output.reshape(output_reshape_helper);
  unsigned int output_width_offset = 0;
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getOutgoingDerivative(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);
    unsigned int data_copy_size = irh.width();

    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** copy continous data (reshaped width size) in a tensor */
        const Tensor source_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size, tensor_type});
        Tensor dest_tensor =
          Tensor::Map<float>(input.getAddress<float>(batch, 0, 0, 0),
                             data_copy_size * sizeof(float),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** copy continous data (reshaped width size) in a tensor */
        const Tensor source_tensor = Tensor::Map<_FP16>(
          output.getAddress<_FP16>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(_FP16),
          {1, 1, 1, data_copy_size, tensor_type});
        Tensor dest_tensor =
          Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, 0, 0),
                             data_copy_size * sizeof(_FP16),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    input.reshape(in_dim);
    output_width_offset += irh.width();
  }
}

void ConcatLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, concat_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[ConcatLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void ConcatLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(concat_props, method, this);
}

} /* namespace nntrainer */
