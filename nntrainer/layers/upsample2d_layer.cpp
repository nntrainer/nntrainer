// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file   upsample2d_layer.h
 * @date   15 June 2024
 * @brief  It is a implementation of upsample layer for given size and
 * interpolation method
 * @see    https://github.com/nntrainer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <node_exporter.h>
#include <upsample2d_layer.h>

#include <cstring>

namespace nntrainer {
static constexpr size_t SINGLE_INOUT_IDX = 0;

Upsample2dLayer::Upsample2dLayer() :
  Layer(),
  upsample2d_props(props::UpsampleMode(),
                   std::array<props::KernelSize, UPSAMPLE2D_DIM>()) {}

void Upsample2dLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  const auto &kernel_size =
    std::get<std::array<props::KernelSize, UPSAMPLE2D_DIM>>(upsample2d_props);

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height() * kernel_size[0]);
      dim[i].width(dim[i].width() * kernel_size[1]);
    }
  }

  context.setOutputDimensions(dim);
}

/**
 * @brief dtype-correct upsample forwarding. getValue()/setValue() default to
 * T=float, so on an FP16 tensor a float read reinterprets two 2-byte halves as
 * one 4-byte float (and indexes at 2x stride) -> garbage. Dispatch on the
 * actual tensor dtype so element access stays correct for any precision. The
 * bilinear interpolation math is carried out in float regardless of storage.
 */
template <typename T>
static void upsampleForwardT(
  const Tensor &in, Tensor &out,
  props::UpsampleModeInfo::Interpolation upsampling_type,
  const std::array<props::KernelSize, UPSAMPLE2D_DIM> &kernel_size) {
  switch (upsampling_type) {
  case props::UpsampleModeInfo::Interpolation::nearest: {
    // Fast path: contiguous NCHW. nearest upsample maps
    // out[b,c, kh*ih+ry, kw*iw+rx] = in[b,c,ih,iw] for ry in [0,kh), rx in
    // [0,kw). Same element mapping as the scalar loop below, written per
    // output row: each input row ih feeds kh output rows, and within a row
    // each input element is repeated kw times. Bit-identical to the scalar
    // path.
    const unsigned int kh = kernel_size[0].get();
    const unsigned int kw = kernel_size[1].get();
    const bool can_fast = in.getContiguous() && out.getContiguous() &&
                          in.getFormat() == Tformat::NCHW &&
                          out.getFormat() == Tformat::NCHW;
    if (can_fast) {
      const T *in_d = in.getData<T>();
      T *out_d = out.getData<T>();
      const unsigned int B = out.batch();
      const unsigned int C = out.channel();
      const unsigned int iH = in.height();
      const unsigned int iW = in.width();
      const unsigned int oW = out.width();           // == iW * kw
      const size_t in_plane = (size_t)iH * iW;       // [H*W] per (b,c)
      const size_t out_plane = (size_t)iH * kh * oW; // [oH*oW] per (b,c)
      // scratch row for the kw-expanded output row (reused per input row)
      std::vector<T> expanded(oW);
      for (unsigned int b = 0; b < B; ++b) {
        for (unsigned int c = 0; c < C; ++c) {
          const T *in_bc =
            in_d + (size_t)b * C * in_plane + (size_t)c * in_plane;
          T *out_bc = out_d + (size_t)b * C * out_plane + (size_t)c * out_plane;
          for (unsigned int ih = 0; ih < iH; ++ih) {
            const T *in_row = in_bc + (size_t)ih * iW;
            // expand in_row[iW] -> expanded[iW*kw], each elt repeated kw
            T *e = expanded.data();
            for (unsigned int iw = 0; iw < iW; ++iw) {
              const T v = in_row[iw];
              for (unsigned int rx = 0; rx < kw; ++rx)
                e[(size_t)iw * kw + rx] = v;
            }
            // write the expanded row into the kh output rows it feeds
            for (unsigned int ry = 0; ry < kh; ++ry) {
              T *out_row = out_bc + ((size_t)ih * kh + ry) * oW;
              std::memcpy(out_row, e, oW * sizeof(T));
            }
          }
        }
      }
      return;
    }

    const bool can_fast_nhwc = in.getContiguous() && out.getContiguous() &&
                               in.getFormat() == Tformat::NHWC &&
                               out.getFormat() == Tformat::NHWC;
    if (can_fast_nhwc) {
      const T *in_d = in.getData<T>();
      T *out_d = out.getData<T>();
      const unsigned int B = out.batch();
      const unsigned int Co = out.channel();
      const unsigned int iH = in.height();
      const unsigned int iW = in.width();
      const unsigned int oH = out.height();
      const unsigned int oW = out.width();
      const size_t in_hwc = (size_t)iH * iW * Co;
      const size_t out_hwc = (size_t)oH * oW * Co;

      std::vector<T> expanded((size_t)oW * Co);
      for (unsigned int b = 0; b < B; ++b) {
        const T *in_b = in_d + (size_t)b * in_hwc;
        T *out_b = out_d + (size_t)b * out_hwc;
        for (unsigned int ih = 0; ih < iH; ++ih) {
          const T *in_row = in_b + (size_t)ih * iW * Co;
          T *e = expanded.data();
          for (unsigned int iw = 0; iw < iW; ++iw) {
            const T *v = in_row + (size_t)iw * Co;
            for (unsigned int rx = 0; rx < kw; ++rx) {
              std::memcpy(e + ((size_t)iw * kw + rx) * Co, v, Co * sizeof(T));
            }
          }
          for (unsigned int ry = 0; ry < kh; ++ry) {
            T *out_row = out_b + ((size_t)ih * kh + ry) * oW * Co;
            std::memcpy(out_row, e, (size_t)oW * Co * sizeof(T));
          }
        }
      }
      return;
    }
    for (int b = 0; b < (int)out.batch(); b++) {
      for (int c = 0; c < (int)out.channel(); c++) {
        for (int h = 0; h < (int)out.height(); h++) {
          for (int w = 0; w < (int)out.width(); w++) {
            out.getValue<T>(b, c, h, w) =
              in.getValue<T>(b, c, h / kernel_size[0], w / kernel_size[1]);
          }
        }
      }
    }
  } break;
  case props::UpsampleModeInfo::Interpolation::bilinear: {
    float scale_h = (float)kernel_size[0];
    float scale_w = (float)kernel_size[1];

    for (int b = 0; b < (int)out.batch(); b++) {
      for (int c = 0; c < (int)out.channel(); c++) {
        for (int h = 0; h < (int)out.height(); h++) {
          for (int w = 0; w < (int)out.width(); w++) {
            float x_in = (w + 0.5f) / scale_w - 0.5f;
            float y_in = (h + 0.5f) / scale_h - 0.5f;

            if (x_in < 0) {
              x_in = 0.0f;
            }
            if (y_in < 0) {
              y_in = 0.0f;
            }

            int x0 = static_cast<int>(floor(x_in));
            int y0 = static_cast<int>(floor(y_in));
            int x1 = std::min(x0 + 1, (int)in.width() - 1);
            int y1 = std::min(y0 + 1, (int)in.height() - 1);

            float dx = x_in - x0;
            float dy = y_in - y0;

            float top =
              (1.0f - dx) * static_cast<float>(in.getValue<T>(b, c, y1, x0)) +
              dx * static_cast<float>(in.getValue<T>(b, c, y1, x1));
            float bottom =
              (1.0f - dx) * static_cast<float>(in.getValue<T>(b, c, y0, x0)) +
              dx * static_cast<float>(in.getValue<T>(b, c, y0, x1));
            float v = (1.0f - dy) * bottom + dy * top;
            out.getValue<T>(b, c, h, w) = static_cast<T>(v);
          }
        }
      }
    }
  } break;
  default:
    throw std::runtime_error("Error: Unknown Upsample Mode Type");
  }
}

void Upsample2dLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const auto &upsampling_type =
    std::get<props::UpsampleMode>(upsample2d_props).get();
  const auto &kernel_size =
    std::get<std::array<props::KernelSize, UPSAMPLE2D_DIM>>(upsample2d_props);

#ifdef ENABLE_FP16
  if (out.getDataType() == ml::train::TensorDim::DataType::FP16) {
    upsampleForwardT<_FP16>(in, out, upsampling_type, kernel_size);
    return;
  }
#endif
  upsampleForwardT<float>(in, out, upsampling_type, kernel_size);
}

void Upsample2dLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  const auto &kernel_size =
    std::get<std::array<props::KernelSize, UPSAMPLE2D_DIM>>(upsample2d_props);
  const auto &upsampling_type =
    std::get<props::UpsampleMode>(upsample2d_props).get();

  switch (upsampling_type) {
  case props::UpsampleModeInfo::Interpolation::nearest: {
    float val = 0;
    for (int b = 0; b < (int)derivative_.batch(); b++) {
      for (int c = 0; c < (int)derivative_.channel(); c++) {
        for (int h = 0; h < (int)derivative_.height(); h++) {
          for (int w = 0; w < (int)derivative_.width(); w++) {
            if (h % kernel_size[0] == 0 && w % kernel_size[1] == 0) {
              dx.setValue(b, c, h / kernel_size[0], w / kernel_size[1], 0);
            }

            val = dx.getValue(b, c, h / kernel_size[0], w / kernel_size[1]) +
                  derivative_.getValue(b, c, h, w);
            dx.setValue(b, c, h / kernel_size[0], w / kernel_size[1], val);
          }
        }
      }
    }
  } break;
  case props::UpsampleModeInfo::Interpolation::bilinear: {
    dx.setZero();

    int input_height = dx.height();
    int input_width = dx.width();

    for (int b = 0; b < (int)derivative_.batch(); b++) {
      for (int c = 0; c < (int)derivative_.channel(); c++) {
        for (int h = 0; h < (int)derivative_.height(); h++) {
          for (int w = 0; w < (int)derivative_.width(); w++) {
            float in_h = (h + 0.5f) / kernel_size[0] - 0.5f;
            float in_w = (w + 0.5f) / kernel_size[1] - 0.5f;

            if (in_h < 0) {
              in_h = 0.0f;
            }
            if (in_w < 0) {
              in_w = 0.0f;
            }

            int y0 = static_cast<int>(floor(in_h));
            int x0 = static_cast<int>(floor(in_w));
            int y1 = std::min(y0 + 1, input_height - 1);
            int x1 = std::min(x0 + 1, input_width - 1);

            float dx_ = (in_w - x0); // Due to name conflict with dx
            float dy_ = (in_h - y0);

            float top_left_weight = (1.0 - dy_) * (1.0 - dx_);
            float top_right_weight = (1.0 - dy_) * dx_;
            float bottom_left_weight = dy_ * (1.0 - dx_);
            float bottom_right_weight = dy_ * dx_;

            float grad = derivative_.getValue(b, c, h, w);

            dx.addValue(b, c, y0, x0, top_left_weight * grad, 1.0f);
            dx.addValue(b, c, y0, x1, top_right_weight * grad, 1.0f);
            dx.addValue(b, c, y1, x0, bottom_left_weight * grad, 1.0f);
            dx.addValue(b, c, y1, x1, bottom_right_weight * grad, 1.0f);
          }
        }
      }
    }
  } break;
  default:
    throw std::runtime_error("Error: Unknown Upsample Mode Type");
  }
}

void Upsample2dLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, upsample2d_props);

  if (!remain_props.empty()) {
    std::string msg = "[Upsample2dLayer] Unknown properties set with count" +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} // namespace nntrainer
