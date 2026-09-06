// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   vjepa_rope_layer.cpp
 * @date   21 May 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  3D axial Rotary Positional Embedding layer for V-JEPA2 ViT.
 */

#include <cmath>
#include <stdexcept>

#include <nntrainer_error.h>

#include "vjepa_rope_layer.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void VjepaRopeLayer::precompute_tables() {
  const unsigned int num_heads =
    std::get<props::VjepaNumHeads>(vjepa_rope_props).get();
  const unsigned int grid_t =
    std::get<props::VjepaGridT>(vjepa_rope_props).get();
  const unsigned int grid_h =
    std::get<props::VjepaGridH>(vjepa_rope_props).get();
  const unsigned int grid_w =
    std::get<props::VjepaGridW>(vjepa_rope_props).get();
  const float theta = std::get<props::VjepaRopeTheta>(vjepa_rope_props).get();
  const unsigned int pretrained_grid =
    std::get<props::VjepaPretrainedGridSize>(vjepa_rope_props).get();
  const bool interpolate =
    std::get<props::VjepaInterpolateRope>(vjepa_rope_props).get();

  (void)num_heads; // head_dim is computed in finalize()

  const unsigned int half = d_dim / 2; // pairs per axis slice
  const unsigned int tokens_per_frame = grid_h * grid_w;
  const unsigned int num_tokens = grid_t * tokens_per_frame;

  // omega[i] = 1 / theta^(i / half), matching torch:
  //   omega = arange(D/2); omega /= D/2; omega = 1/theta**omega   (D = d_dim)
  std::vector<float> omega(half);
  for (unsigned int i = 0; i < half; ++i)
    omega[i] = 1.0f / std::pow(theta, static_cast<float>(i) / half);

  // interpolate scales the height/width positions onto the pretrained grid:
  //   pos *= (pretrained_grid_size - 1) / (grid - 1)
  float h_scale = 1.0f, w_scale = 1.0f;
  if (interpolate) {
    if (grid_h > 1)
      h_scale = static_cast<float>(pretrained_grid - 1) / (grid_h - 1);
    if (grid_w > 1)
      w_scale = static_cast<float>(pretrained_grid - 1) / (grid_w - 1);
  }

  cos_table.assign(num_tokens, std::vector<float>(head_dim, 1.0f));
  sin_table.assign(num_tokens, std::vector<float>(head_dim, 0.0f));

  for (unsigned int n = 0; n < num_tokens; ++n) {
    const unsigned int frame = n / tokens_per_frame;
    const unsigned int rem = n % tokens_per_frame;
    const unsigned int height = rem / grid_w;
    const unsigned int width = rem % grid_w;

    const float pos[3] = {static_cast<float>(frame),
                          static_cast<float>(height) * h_scale,
                          static_cast<float>(width) * w_scale};

    // three contiguous axis slices: [0, d_dim), [d_dim, 2*d_dim), [2*d_dim, ..)
    for (unsigned int axis = 0; axis < 3; ++axis) {
      const unsigned int base = axis * d_dim;
      for (unsigned int i = 0; i < half; ++i) {
        const float ang = pos[axis] * omega[i];
        const float c = std::cos(ang);
        const float s = std::sin(ang);
        // repeat_interleave(2): the pair (2i, 2i+1) shares one (cos, sin)
        cos_table[n][base + 2 * i] = c;
        cos_table[n][base + 2 * i + 1] = c;
        sin_table[n][base + 2 * i] = s;
        sin_table[n][base + 2 * i + 1] = s;
      }
    }
    // tail dims [3*d_dim, head_dim) keep cos=1, sin=0 (passthrough)
  }
}

void VjepaRopeLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dims = context.getInputDimensions();
  context.setOutputDimensions(dims);

  const unsigned int num_heads =
    std::get<props::VjepaNumHeads>(vjepa_rope_props).get();
  const unsigned int width = dims[0].width();

  NNTR_THROW_IF(num_heads == 0 || width % num_heads != 0, std::invalid_argument)
    << "[vjepa_rope] width (" << width << ") must be a multiple of num_heads ("
    << num_heads << ")";

  head_dim = width / num_heads;
  // d_dim = 2 * ((head_dim // 3) // 2), per V-JEPA2 RoPEAttention
  d_dim = 2 * ((head_dim / 3) / 2);

  NNTR_THROW_IF(3 * d_dim > head_dim, std::invalid_argument)
    << "[vjepa_rope] invalid head_dim " << head_dim;

  precompute_tables();
}

void VjepaRopeLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const nntrainer::TensorDim in_dim = in.getDim();
  const unsigned int width = in_dim.width();
  const unsigned int rows = in_dim.height();
  const unsigned int feature_len = in_dim.getFeatureLen();

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
      rotate<float>(in.getData<float>() + b * feature_len,
                    out.getData<float>() + b * feature_len, 0, rows, width);
    } else {
      throw std::invalid_argument(
        "[vjepa_rope] only FP32 is supported in forwarding");
    }
  }
}

void VjepaRopeLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const nntrainer::TensorDim in_dim = in.getDim();
  const unsigned int width = in_dim.width();
  const unsigned int rows = to - from;
  const unsigned int feature_len = in_dim.getFeatureLen();

  for (unsigned int b = 0; b < in_dim.batch(); ++b) {
    if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
      rotate<float>(in.getData<float>() + b * feature_len,
                    out.getData<float>() + b * feature_len, from, rows, width);
    } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      rotate<_FP16>(in.getData<_FP16>() + b * feature_len,
                    out.getData<_FP16>() + b * feature_len, from, rows, width);
#else
      throw std::invalid_argument("[vjepa_rope] enable-fp16 is not set!");
#endif
    } else {
      throw std::invalid_argument("[vjepa_rope] unsupported data type");
    }
  }
}

template <typename T>
void VjepaRopeLayer::rotate(const T *in, T *out, unsigned int from,
                            unsigned int rows, unsigned int width) const {
  const unsigned int num_heads = width / head_dim;
  const unsigned int pairs = head_dim / 2;

  for (unsigned int r = 0; r < rows; ++r) {
    const unsigned int token = from + r;
    const std::vector<float> &cos_t = cos_table[token];
    const std::vector<float> &sin_t = sin_table[token];
    const T *in_row = in + static_cast<size_t>(r) * width;
    T *out_row = out + static_cast<size_t>(r) * width;

    for (unsigned int h = 0; h < num_heads; ++h) {
      const unsigned int o = h * head_dim;
      for (unsigned int p = 0; p < pairs; ++p) {
        const unsigned int d = 2 * p;
        const float x0 = static_cast<float>(in_row[o + d]);
        const float x1 = static_cast<float>(in_row[o + d + 1]);
        const float c = cos_t[d];
        const float s = sin_t[d];
        // y = (-x1, x0); out = x * cos + y * sin
        out_row[o + d] = static_cast<T>(x0 * c - x1 * s);
        out_row[o + d + 1] = static_cast<T>(x1 * c + x0 * s);
      }
    }
  }
}

void VjepaRopeLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, vjepa_rope_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[vjepa_rope] Unknown Layer Properties count "
    << std::to_string(remain_props.size());
}

void VjepaRopeLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void VjepaRopeLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[vjepa_rope] Training is not supported yet.");
}

#ifdef PLUGGABLE

nntrainer::Layer *create_vjepa_rope_layer() { return new VjepaRopeLayer(); }

void destroy_vjepa_rope_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_vjepa_rope_layer,
                                                   destroy_vjepa_rope_layer};
}

#endif

} // namespace causallm
