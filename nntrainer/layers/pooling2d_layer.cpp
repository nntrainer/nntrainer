// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   pooling2d_layer.cpp
 * @date   12 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is 2 Dimensional Pooling Layer Class for Neural Network
 *
 */

#include <cstring>
#include <limits>

#include <common_properties.h>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <pooling2d_layer.h>
#include <util_func.h>
namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

/**
 * @brief Help function for Pooling Handler
 */
template <typename T> struct PoolFunc {
  typedef std::function<T(const T *, int, int, int)> Type;
};

Pooling2DLayer::Pooling2DLayer(
  const std::array<unsigned int, POOLING2D_DIM * 2> &padding_) :
  Layer(),
  padding(padding_),
  pooling2d_props(props::PoolingType(), std::vector<props::PoolSize>(),
                  std::array<props::Stride, POOLING2D_DIM>(),
                  props::Padding2D()),
  pool_helper_idx(0) {}

void Pooling2DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[Pooling2D] pooling layer only takes one input";

  const TensorDim &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  TensorDim out_dim;

  auto &pool_size = std::get<std::vector<props::PoolSize>>(pooling2d_props);
  NNTR_THROW_IF(!(pool_size.empty() || pool_size.size() == 2),
                std::invalid_argument)
    << "[Pooling2D] the number of pool size should be 0 or 2";
  /** @todo update default value of pooling stride to be pool size */
  auto &stride =
    std::get<std::array<props::Stride, POOLING2D_DIM>>(pooling2d_props);
  auto &pooling_type = std::get<props::PoolingType>(pooling2d_props).get();

  if (pooling_type == props::PoolingTypeInfo::Enum::global_max ||
      pooling_type == props::PoolingTypeInfo::Enum::global_average) {
    if (!pool_size.empty()) {
      ml_logw(
        "[Pooling2D] global_max, global_average does not accept pool size");
      pool_size.clear();
    }
    pool_size.emplace_back(props::PoolSize(in_dim.height()));
    pool_size.emplace_back(props::PoolSize(in_dim.width()));
  }

  padding = std::get<props::Padding2D>(pooling2d_props)
              .compute(in_dim, {pool_size[0], pool_size[1]},
                       {stride[0], stride[1]}, {1, 1});

  auto [pt, pb, pl, pr] = padding;

  if (pooling_type == props::PoolingTypeInfo::Enum::global_max ||
      pooling_type == props::PoolingTypeInfo::Enum::global_average) {
    NNTR_THROW_IF(pt + pb + pl + pr != 0, std::invalid_argument)
      << "[Pooling2D] global_max, global_average does not accept padding";

    NNTR_THROW_IF(static_cast<int>(stride[0]) != 1 ||
                    static_cast<int>(stride[1]) != 1,
                  std::invalid_argument)
      << "[Pooling2D] global_max, global_average does not accept stride";
  }

  unsigned int eff_in_height = in_dim.height() + pt + pb;
  unsigned int eff_in_width = in_dim.width() + pl + pr;

  NNTR_THROW_IF(eff_in_height < pool_size[0] || eff_in_width < pool_size[1],
                std::invalid_argument)
    << "[Pooling2D] Failed to initialize: in size + padding is smaller than "
       "effective kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(eff_in_height - pt - pool_size[0] > IM ||
                  eff_in_width - pl - pool_size[1] > IM,
                std::invalid_argument)
    << "[Pooling2D] Failed to initialize: Calculated patch end is over int max";

  out_dim.batch(in_dim.batch());
  out_dim.channel(in_dim.channel());
  out_dim.height((eff_in_height - pool_size[0]) / stride[0] + 1);
  out_dim.width((eff_in_width - pool_size[1]) / stride[1] + 1);
  out_dim.setDataType(in_dim.getDataType());
  context.setOutputDimensions({out_dim});

  /**
   * in case of max pool, idx that points to the first max item
  //  * in case of average pool, effective average counter is number of patches
   * actually counted into when calculating the average
   * // clang-format off
   * eg) pooling of below
   * x x x
   * x 3 3
   * x 3 3
   * = 12 / 4 = 3
   * // clang-format on
   */
  if (pooling_type == props::PoolingTypeInfo::Enum::global_max) {
    auto helper_dim = in_dim;
    helper_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    pool_helper_idx =
      context.requestTensor(helper_dim, "helper_idx", Initializer::NONE, false,
                            TensorLifespan::ITERATION_LIFESPAN);
    pool_helper_size.resize(helper_dim.batch() * helper_dim.channel());
  } else {
    auto helper_dim = out_dim;
    helper_dim.setDataType(ml::train::TensorDim::DataType::FP32);
    pool_helper_idx =
      context.requestTensor(helper_dim, "helper_idx", Initializer::NONE, false,
                            TensorLifespan::ITERATION_LIFESPAN);
  }
}

void Pooling2DLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &pool_helper = context.getTensor(pool_helper_idx);

  const TensorDim &in_dim = input_.getDim();

  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                            void *user_data) {
    for (unsigned int b = s; b < e; ++b) {
      Tensor in_sub = input_.getBatchSlice(b, 1);
      Tensor result = hidden_.getBatchSlice(b, 1);
      Tensor helper = pool_helper.getBatchSlice(b, 1);
      pooling2d(in_sub, training, result, helper, b);
    }
  };

  auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    forwarding_job(0, in_dim.batch(), 0, nullptr);
  }
}

void Pooling2DLayer::calcDerivative(RunLayerContext &context) {
  auto &pool_size = std::get<std::vector<props::PoolSize>>(pooling2d_props);
  auto &stride =
    std::get<std::array<props::Stride, POOLING2D_DIM>>(pooling2d_props);
  auto &pooling_type = std::get<props::PoolingType>(pooling2d_props).get();

  const Tensor &deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &result = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &pool_helper = context.getTensor(pool_helper_idx);

  const TensorDim &in_dim = result.getDim();
  unsigned int batch = in_dim.batch();
  unsigned int channel = in_dim.channel();
  int height = in_dim.height();
  int width = in_dim.width();

  auto pt = padding[0];
  auto pl = padding[2];
  unsigned int p_height = pool_size[0];
  unsigned int p_width = pool_size[1];

  unsigned int J, K;

  result.setZero();

  unsigned int out_map_size = deriv.height() * deriv.width();
  unsigned int in_map_size = height * width;
  auto apply_max = [&]<typename T>(T *result_data) {
    const int *iter = pool_helper.getData<int>();
    const T *deriv_data = deriv.getData<T>();
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channel; ++c) {
        for (unsigned int i = 0; i < out_map_size; ++i) {
          /// pool_helper = -1 means the max idx was at the padding, so no need
          /// to update
          if (*iter != -1) {
            result_data[*iter] += *deriv_data;
          }
          iter++;
          deriv_data++;
        }
        result_data += in_map_size;
      }
    }
  };

  auto apply_average = [&]<typename T>(T *result_data) {
    int height_stride_end = height - p_height + pt;
    int width_stride_end = width - p_width + pl;
    const int *iter = pool_helper.getData<int>();
    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int i = 0; i < channel; ++i) {
        J = 0;
        for (int j = -(int)pt; j <= height_stride_end; j += stride[0]) {
          K = 0;
          for (int k = -(int)pl; k <= width_stride_end; k += stride[1]) {
            T del = deriv.getValue<T>(b, i, J, K) / *iter;
            int patch_height_end =
              std::min(static_cast<int>(j + p_height), height);
            int patch_width_end =
              std::min(static_cast<int>(k + p_width), width);
            int start_h = std::max(0, j);
            int start_w = std::max(0, k);
            for (int h = start_h; h < patch_height_end; ++h) {
              for (int w = start_w; w < patch_width_end; ++w) {
                result.setValue(b, i, h, w,
                                result.getValue<T>(b, i, h, w) + del);
              }
            }
            iter++;
            K++;
          }
          J++;
        }
      }
    }
  };

  auto apply_global_max = [&]<typename T>(T *result_data) {
    const T *deriv_data = deriv.getData<T>();
    for (unsigned int b = 0; b < batch; b++) {
      for (unsigned int c = 0; c < channel; c++) {
        const int *iter =
          pool_helper.getData<int>() + pool_helper.getIndex(b, c, 0, 0);
        unsigned int helper_size = pool_helper_size[b * channel + c];
        T der = *deriv_data / static_cast<T>(helper_size);

        for (unsigned int idx = 0; idx < helper_size; idx++)
          result_data[iter[idx]] += der;
        deriv_data++;
        result_data += in_map_size;
      }
    }
  };

  auto in_data_type = in_dim.getDataType();

  if (in_data_type == ml::train::TensorDim::DataType::FP32) {
    switch (pooling_type) {
    case props::PoolingTypeInfo::Enum::max:
      apply_max(result.getData<float>());
      break;
    case props::PoolingTypeInfo::Enum::global_average:
    case props::PoolingTypeInfo::Enum::average:
      apply_average(result.getData<float>());
      break;
    case props::PoolingTypeInfo::Enum::global_max:
      apply_global_max(result.getData<float>());
      break;
    default:
      throw std::runtime_error("Error: Unknown Pooling Type");
      break;
    }
  }
#ifdef ENABLE_FP16
  else if (in_data_type == ml::train::TensorDim::DataType::FP16) {

    switch (pooling_type) {
    case props::PoolingTypeInfo::Enum::max:
      apply_max(result.getData<_FP16>());
      break;
    case props::PoolingTypeInfo::Enum::global_average:
    case props::PoolingTypeInfo::Enum::average:
      apply_average(result.getData<_FP16>());
      break;
    case props::PoolingTypeInfo::Enum::global_max:
      apply_global_max(result.getData<_FP16>());
      break;
    default:
      throw std::runtime_error("Error: Unknown Pooling Type");
    }
  }
#endif
  else {
    throw std::runtime_error("Unsupported datatype");
  }
}

void Pooling2DLayer::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  exporter.saveResult(pooling2d_props, method, this);
}

void Pooling2DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, pooling2d_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[Pooling2dLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void Pooling2DLayer::pooling2d(Tensor &in, bool training, Tensor &output,
                               Tensor &pool_helper, int batch_idx) {

  auto &pool_size = std::get<std::vector<props::PoolSize>>(pooling2d_props);
  auto &stride =
    std::get<std::array<props::Stride, POOLING2D_DIM>>(pooling2d_props);
  auto &pooling_type = std::get<props::PoolingType>(pooling2d_props).get();

  unsigned int channel = in.channel();
  auto [pt, pb, pl, pr] = padding;

  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
  unsigned int patch_height = pool_size[0];
  unsigned int patch_width = pool_size[1];

  NNTR_THROW_IF(output.empty(), std::invalid_argument)
    << "[Pooling2D] output is uninitialized, this is not supported";

  /**
   * @brief pooling function
   * @param in_c channel sliced data
   * @param start_h (height index pointing the start of the patch)
   * @param start_w (width index pointing the start of the patch)
   * @return result value of pooling
   */
  PoolFunc<float>::Type pool_fn_fp32;
#ifdef ENABLE_FP16
  PoolFunc<_FP16>::Type pool_fn_fp16;
#endif

  unsigned int max_idx_count = 0;

  auto pool_fn_max = [&]<typename T>(const T *in_data, int channel_idx,
                                     int start_h, int start_w) {
    int end_h = start_h + patch_height;
    int end_w = start_w + patch_width;

    T max_val = std::numeric_limits<T>::lowest();

    int cur_max_idx = -1;
    int eff_end_h = std::min(end_h, in_height);
    int eff_end_w = std::min(end_w, in_width);
    start_w = std::max(0, start_w);
    for (int h = std::max(0, start_h); h < eff_end_h; ++h) {
      for (int w = start_w; w < eff_end_w; ++w) {
        int cur_idx = h * in_width + w;
        T val = in_data[cur_idx];
        if (max_val < val) {
          max_val = val;
          if (training) {
            cur_max_idx = cur_idx;
          }
        }
      }
    }

    if (training) {
      pool_helper.setValueInt(max_idx_count++, cur_max_idx);
    }

    return max_val;
  };

  auto pool_fn_global_max = [&, this]<typename T>(const T *in_data,
                                                  int channel_idx, int start_h,
                                                  int start_w) {
    int end_h = start_h + patch_height;
    int end_w = start_w + patch_width;

    T max_val = std::numeric_limits<T>::lowest();
    int *helper_data = pool_helper.getData<int>();
    helper_data += channel_idx * in_height * in_width;

    for (int h = start_h; h < end_h; ++h) {
      for (int w = start_w; w < end_w; ++w) {
        int cur_idx = h * in_width + w;
        T val = in_data[cur_idx];
        if (max_val < val) {
          max_val = val;
          max_idx_count = 0;
        }

        if (training && max_val == val) {
          *(helper_data + max_idx_count++) = cur_idx;
        }
      }
    }

    pool_helper_size[batch_idx * in.channel() + channel_idx] = max_idx_count;
    return max_val;
  };

  auto pool_fn_average = [&]<typename T>(const T *in_data, int channel_idx,
                                         int start_h, int start_w) {
    int end_h = start_h + patch_height;
    int end_w = start_w + patch_width;
    T total = static_cast<T>(0.0f);

    int eff_end_h = std::min(end_h, in_height);
    int eff_end_w = std::min(end_w, in_width);
    int eff_start_h = std::max(0, start_h);
    int eff_start_w = std::max(0, start_w);

    int cnt = (eff_end_h - eff_start_h) * (eff_end_w - eff_start_w);
    for (int h = eff_start_h; h < eff_end_h; ++h) {
      for (int w = eff_start_w; w < eff_end_w; ++w) {
        T val = in_data[h * in_width + w];
        total += val;
      }
    }

    if (training) {
      pool_helper.setValueInt(max_idx_count++, cnt);
    }
    return total / cnt;
  };

  switch (pooling_type) {
  case props::PoolingTypeInfo::Enum::max:
    pool_fn_fp32 = pool_fn_max;
#ifdef ENABLE_FP16
    pool_fn_fp16 = pool_fn_max;
#endif
    break;
  case props::PoolingTypeInfo::Enum::global_max:
    pool_fn_fp32 = pool_fn_global_max;
#ifdef ENABLE_FP16
    pool_fn_fp16 = pool_fn_global_max;
#endif
    break;
  case props::PoolingTypeInfo::Enum::global_average:
  case props::PoolingTypeInfo::Enum::average:
    pool_fn_fp32 = pool_fn_average;
#ifdef ENABLE_FP16
    pool_fn_fp16 = pool_fn_average;
#endif
    break;
  case props::PoolingTypeInfo::Enum::unknown:
  default:
    throw std::invalid_argument("unknown pooling type given");
    break;
  }

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *in_data = in.getData<float>();
    float *out_data = output.getData<float>();

    unsigned int map_size = in_height * in_width;

    int height_stride_end = height - patch_height - pt;
    int width_stride_end = width - patch_width - pl;
    for (unsigned int i = 0; i < channel; ++i) {
      const float *in_data_channel_sliced = in_data + i * map_size;
      for (int j = -(int)pt; j <= height_stride_end; j += stride[0]) {
        for (int k = -(int)pl; k <= width_stride_end; k += stride[1]) {
          float pool_value = pool_fn_fp32(in_data_channel_sliced, i, j, k);
          *out_data = pool_value;
          out_data++;
        }
      }
    }
  }
#ifdef ENABLE_FP16
  else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    const _FP16 *in_data = in.getData<_FP16>();
    _FP16 *out_data = output.getData<_FP16>();

    unsigned int map_size = in_height * in_width;

    int height_stride_end = height - patch_height - pt;
    int width_stride_end = width - patch_width - pl;
    for (unsigned int i = 0; i < channel; ++i) {
      const _FP16 *in_data_channel_sliced = in_data + i * map_size;
      for (int j = -pt; j <= height_stride_end; j += stride[0]) {
        for (int k = -pl; k <= width_stride_end; k += stride[1]) {
          _FP16 pool_value = pool_fn_fp16(in_data_channel_sliced, i, j, k);
          *out_data = pool_value;
          out_data++;
        }
      }
    }
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

void Pooling2DLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(pool_helper_idx, batch);
  props::PoolingTypeInfo::Enum pooling_type =
    std::get<props::PoolingType>(pooling2d_props).get();
  if (pooling_type == props::PoolingTypeInfo::Enum::global_max)
    pool_helper_size.resize(batch * context.getInput(0).channel());
}

} /* namespace nntrainer */
