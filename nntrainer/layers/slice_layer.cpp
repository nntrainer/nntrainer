// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   slice_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is slice layer class (operation layer)
 */

#include "common_properties.h"
#include "tensor_base.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <slice_layer.h>
#include <stdexcept>
#include <thread_manager.h>
#include <util_func.h>

#include <cstring>

#include <layer_context.h>

namespace nntrainer {

void SliceLayer::finalize(InitLayerContext &context) {
  axis = std::get<props::Axis>(slice_props).get();
  start = std::get<props::StartIndex>(slice_props).get() - 1;
  unsigned int end = std::get<props::EndIndex>(slice_props).get() - 1;

  const TensorDim &in_dim = context.getInputDimensions()[0];
  TensorDim outputDim = context.getInputDimensions()[0];

  for (unsigned int i = 0; i < 4; ++i) {
    if (i == axis) {
      outputDim.setTensorDim(i, end - start);
    } else {
      outputDim.setTensorDim(i, in_dim[i]);
    }
  }

  context.setOutputDimensions({outputDim});
}

/**
 * @brief dtype-correct element copy for the slice. getValue()/setValue()
 * default to T=float, so on an FP16 tensor a float read reinterprets two 2-byte
 * halves as one 4-byte float (and indexes at 2x stride) -> garbage. Dispatch on
 * the actual tensor dtype so the copy stays element-correct for any precision.
 */
template <typename T>
static void sliceForwardT(const Tensor &input, Tensor &output,
                          unsigned int axis, unsigned int start) {
  // Fast path: contiguous NCHW. The slice only shifts one axis by `start`, so
  // every output plane out[b,c,h,w] == in[b, c+c0, h+h0, w+w0] — the same
  // element mapping as the scalar loop below, just copied as contiguous runs
  // (memcpy) instead of element-by-element. Bit-identical to the scalar path.
  // Per-axis, the largest contiguous run that maps 1:1 is:
  //   axis=0 -> [C*H*W] per b, axis=1 -> [H*W] per (b,c),
  //   axis=2 -> [W] per (b,c,h),     axis=3 -> [W] per (b,c,h) (w innermost).
  const bool can_fast = input.getContiguous() && output.getContiguous() &&
                        input.getFormat() == Tformat::NCHW &&
                        output.getFormat() == Tformat::NCHW;
  if (can_fast) {
    const T *in = input.getData<T>();
    T *out = output.getData<T>();
    const unsigned int B = output.batch();
    const unsigned int C = output.channel();
    const unsigned int H = output.height();
    const unsigned int W = output.width();
    // element counts of each axis plane in NCHW (contiguous)
    const size_t plane_b = (size_t)C * H * W; // [C*H*W]
    const size_t plane_c = (size_t)H * W;     // [H*W]
    const size_t row_hw = (size_t)W;          // [W]
    const size_t elt = sizeof(T);
    if (axis == 0) {
      for (unsigned int b = 0; b < B; ++b)
        std::memcpy(out + (size_t)b * plane_b,
                    in + (size_t)(b + start) * plane_b, plane_b * elt);
    } else if (axis == 1) {
      for (unsigned int b = 0; b < B; ++b)
        for (unsigned int c = 0; c < C; ++c)
          std::memcpy(out + (size_t)b * plane_b + (size_t)c * plane_c,
                      in + (size_t)b * plane_b + (size_t)(c + start) * plane_c,
                      plane_c * elt);
    } else if (axis == 2) {
      for (unsigned int b = 0; b < B; ++b)
        for (unsigned int c = 0; c < C; ++c)
          for (unsigned int h = 0; h < H; ++h)
            std::memcpy(
              out + ((size_t)b * plane_b + (size_t)c * plane_c + (size_t)h * W),
              in + ((size_t)b * plane_b + (size_t)c * plane_c +
                    (size_t)(h + start) * W),
              row_hw * elt);
    } else { // axis == 3 (w innermost) -> same [W] run per (b,c,h)
      for (unsigned int b = 0; b < B; ++b)
        for (unsigned int c = 0; c < C; ++c)
          for (unsigned int h = 0; h < H; ++h)
            std::memcpy(
              out + ((size_t)b * plane_b + (size_t)c * plane_c + (size_t)h * W),
              in + ((size_t)b * plane_b + (size_t)c * plane_c + (size_t)h * W +
                    (size_t)start),
              row_hw * elt);
    }
    return;
  }

  const bool can_fast_nhwc = input.getContiguous() && output.getContiguous() &&
                             input.getFormat() == Tformat::NHWC &&
                             output.getFormat() == Tformat::NHWC;
  if (can_fast_nhwc) {
    const T *in = input.getData<T>();
    T *out = output.getData<T>();
    const unsigned int B = output.batch();
    const unsigned int Co = output.channel(), Ci = input.channel();
    const unsigned int Ho = output.height(), Hi = input.height();
    const unsigned int Wo = output.width(), Wi = input.width();
    const size_t elt = sizeof(T);
    const size_t in_hwc = (size_t)Hi * Wi * Ci;
    const size_t out_hwc = (size_t)Ho * Wo * Co;

    if (axis == 0) {
      for (unsigned int b = 0; b < B; ++b)
        std::memcpy(out + (size_t)b * out_hwc,
                    in + (size_t)(b + start) * in_hwc, out_hwc * elt);
    } else if (axis == 1) {
      auto &tm = ThreadManager::Global();
      for (unsigned int b = 0; b < B; ++b) {
        tm.parallel_for(0, static_cast<size_t>(Ho) * Wo, [&](size_t hw) {
          std::memcpy(out + ((size_t)b * Ho * Wo + hw) * Co,
                      in + ((size_t)b * Hi * Wi + hw) * Ci + start, Co * elt);
        });
      }
    } else if (axis == 2) {
      for (unsigned int b = 0; b < B; ++b) {
        std::memcpy(out + (size_t)b * Ho * Wo * Co,
                    in + (size_t)b * Hi * Wi * Ci + (size_t)start * Wi * Ci,
                    (size_t)Ho * Wi * Co * elt);
      }
    } else if (axis == 3) {
      auto &tm = ThreadManager::Global();
      for (unsigned int b = 0; b < B; ++b) {
        tm.parallel_for(0, Ho, [&](size_t h) {
          std::memcpy(out + ((size_t)b * Ho + h) * Wo * Co,
                      in + ((size_t)b * Hi + h) * Wi * Ci + (size_t)start * Ci,
                      (size_t)Wo * Co * elt);
        });
      }
    }
    return;
  }

  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          output.getValue<T>(b, c, h, w) =
            input.getValue<T>(b, c_idx, h_idx, w_idx);
        }
      }
    }
  }
}

void SliceLayer::forwarding_operation(const Tensor &input, Tensor &output) {
#ifdef ENABLE_FP16
  if (output.getDataType() == ml::train::TensorDim::DataType::FP16) {
    sliceForwardT<_FP16>(input, output, axis, start);
    return;
  }
#endif
  sliceForwardT<float>(input, output, axis, start);
}

template <typename T>
static void sliceDerivT(const Tensor &inDeriv, Tensor &outDeriv,
                        unsigned int axis, unsigned int start) {
  for (unsigned int b = 0; b < inDeriv.batch(); ++b) {
    for (unsigned int c = 0; c < inDeriv.channel(); ++c) {
      for (unsigned int h = 0; h < inDeriv.height(); ++h) {
        for (unsigned int w = 0; w < inDeriv.width(); ++w) {
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          outDeriv.getValue<T>(b, c_idx, h_idx, w_idx) =
            inDeriv.getValue<T>(b, c, h, w);
        }
      }
    }
  }
}

void SliceLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

#ifdef ENABLE_FP16
  if (outDeriv.getDataType() == ml::train::TensorDim::DataType::FP16) {
    sliceDerivT<_FP16>(inDeriv, outDeriv, axis, start);
    return;
  }
#endif
  sliceDerivT<float>(inDeriv, outDeriv, axis, start);
}

void SliceLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, slice_props);
  if (!remain_props.empty()) {
    std::string msg = "[SliceLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
