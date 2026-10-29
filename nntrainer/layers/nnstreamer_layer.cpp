// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   nnstreamer_layer.cpp
 * @date   26 October 2020
 * @brief  This is class to encapsulate nnstreamer as a layer of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 * @todo: provide input/output dimensions to nnstreamer for certain frameworks
 * @todo: support transposing the data to support NCHW nntrainer data to NHWC
 * nnstreamer data
 */
#include <fstream>

#include <common_properties.h>
#include <layer_context.h>
#include <nnstreamer_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>
namespace nntrainer {

/**
 * @brief NNSModelPath property
 *
 */
class PropsNNSModelPath : public Property<std::string> {
public:
  static constexpr const char *key = "model_path"; /**< unique key to access */
  using prop_tag = str_prop_tag;                   /**< property type */

  /**
   * @brief check is valid
   *
   * @param v value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &v) const override;
};
bool PropsNNSModelPath::isValid(const std::string &v) const {
  /// @note this might not be good validation strategy, as it can support
  /// directory or even not initiated path??
  std::ifstream file(v, std::ios::binary | std::ios::ate);
  return file.good();
}

NNStreamerLayer::NNStreamerLayer() :
  Layer(),
  nnstreamer_layer_props(new PropsType(PropsNNSModelPath())),
  single(nullptr),
  in_res(nullptr),
  out_res(nullptr),
  in_data_cont(nullptr),
  out_data_cont(nullptr),
  in_data(nullptr),
  out_data(nullptr) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

static int nnst_info_to_tensor_dim(ml_tensors_info_h &out_res, TensorDim &dim) {
  int status = ML_ERROR_NONE;
  unsigned int count;
  ml_tensor_type_e type;
  ml_tensor_dimension dim_;

  status = ml_tensors_info_get_count(out_res, &count);
  if (status != ML_ERROR_NONE)
    return status;

  if (count > 1)
    return ML_ERROR_NOT_SUPPORTED;

  status = ml_tensors_info_get_tensor_type(out_res, 0, &type);
  if (status != ML_ERROR_NONE)
    return status;

  if (type != ML_TENSOR_TYPE_FLOAT32)
    return ML_ERROR_NOT_SUPPORTED;

  status = ml_tensors_info_get_tensor_dimension(out_res, 0, dim_);
  if (status != ML_ERROR_NONE)
    return status;

  for (size_t i = 0; i < ml::train::TensorDim::MAXDIM; i++)
    dim.setTensorDim(i, dim_[i] > 0 ? dim_[i] : 1);

  /* reverse the dimension as nnstreamer stores dimension in reverse way */
  dim.reverse();

  return status;
}

NNStreamerLayer::~NNStreamerLayer() { release(); }

void NNStreamerLayer::release() noexcept {
  if (in_res) {
    ml_tensors_info_destroy(in_res);
    in_res = nullptr;
  }

  if (out_res) {
    ml_tensors_info_destroy(out_res);
    out_res = nullptr;
  }

  if (in_data_cont) {
    ml_tensors_data_destroy(in_data_cont);
    in_data_cont = nullptr;
  }

  if (out_data_cont) {
    ml_tensors_data_destroy(out_data_cont);
    out_data_cont = nullptr;
  }

  if (single) {
    ml_single_close(single);
    single = nullptr;
  }
}

void NNStreamerLayer::finalizeError(int status) {
  if (status == ML_ERROR_NONE)
    return;

  release();

  if (status != ML_ERROR_NONE)
    throw std::invalid_argument(
      "[NNStreamerLayer] Finalizing the layer failed.");
}

void NNStreamerLayer::finalize(InitLayerContext &context) {
  int status = ML_ERROR_NONE;
  TensorDim in_dim, output_dim;
  const TensorDim &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];

  status = ml_single_open(
    &single, std::get<PropsNNSModelPath>(*nnstreamer_layer_props).get().c_str(),
    NULL, NULL, ML_NNFW_TYPE_ANY, ML_NNFW_HW_AUTO);
  finalizeError(status);

  /* input tensor in filter */
  status = ml_single_get_input_info(single, &in_res);
  finalizeError(status);

  status = nnst_info_to_tensor_dim(in_res, in_dim);
  finalizeError(status);

  if (input_dim != in_dim) {
    ml_loge("Set tensor info does not match info from the framework");
    finalizeError(ML_ERROR_INVALID_PARAMETER);
  }

  /* input tensor in filter */
  status = ml_single_get_output_info(single, &out_res);
  finalizeError(status);

  status = nnst_info_to_tensor_dim(out_res, output_dim);
  finalizeError(status);

  /* generate input data container */
  status = ml_tensors_data_create(in_res, &in_data_cont);
  finalizeError(status);

  size_t in_data_size;
  status =
    ml_tensors_data_get_tensor_data(in_data_cont, 0, &in_data, &in_data_size);
  finalizeError(status);

  if (in_data_size != input_dim.getDataLen() * sizeof(float))
    finalizeError(ML_ERROR_INVALID_PARAMETER);

  context.setOutputDimensions({output_dim});
}

void NNStreamerLayer::setProperty(const std::vector<std::string> &values) {
  auto left_values = loadProperties(values, *nnstreamer_layer_props);
  NNTR_THROW_IF(!left_values.empty(), std::invalid_argument)
    << "There are unparsed properties, " << left_values.front();
}

void NNStreamerLayer::forwarding(RunLayerContext &context, bool training) {
  size_t data_size;
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  std::copy(input.getData(), input.getData() + input.size(), (float *)in_data);

  int status = ml_single_invoke(single, in_data_cont, &out_data_cont);
  if (status != ML_ERROR_NONE)
    throw std::runtime_error("Failed to forward nnstreamer backbone");

  status =
    ml_tensors_data_get_tensor_data(out_data_cont, 0, &out_data, &data_size);
  if (status != ML_ERROR_NONE) {
    ml_tensors_data_destroy(out_data_cont);
    out_data_cont = nullptr;
    throw std::runtime_error("Failed to forward nnstreamer backbone");
  }

  if (data_size != hidden_.bytes()) {
    ml_tensors_data_destroy(out_data_cont);
    out_data_cont = nullptr;
    throw std::runtime_error("Output size mismatch from nnstreamer backbone.");
  }

  std::copy((float *)out_data, (float *)((char *)out_data + data_size),
            hidden_.getData());

  ml_tensors_data_destroy(out_data_cont);
  out_data_cont = nullptr;
  out_data = nullptr;
}

void NNStreamerLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative is not supported for nnstreamer layer");
}
} /* namespace nntrainer */
