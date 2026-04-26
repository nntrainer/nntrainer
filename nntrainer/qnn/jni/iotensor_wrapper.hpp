#ifndef __QNN_IOTENSOR_WRAPPER_H__
#define __QNN_IOTENSOR_WRAPPER_H__
#include "DataUtil.hpp"
#include "IOTensor.hpp"
#include "PAL/StringOp.hpp"

#include <QnnTypeMacros.hpp>
#include <nntrainer_log.h>

namespace nntrainer {

using namespace qnn::tools;
using namespace qnn::tools::iotensor;

/**
 * @class   IOTensorWrapper
 * @brief   Wraps QNN IOTensor utilities for registering, populating, and
 *          tearing down per-graph input/output tensors.
 */
class IOTensorWrapper {
public:
  StatusCode
  setupInputAndOutputTensors(Qnn_Tensor_t **inputs, Qnn_Tensor_t **outputs,
                             qnn_wrapper_api::GraphInfo_t graphInfo) {
    auto returnStatus = StatusCode::SUCCESS;
#ifdef ENABLE_NPU
    if (StatusCode::SUCCESS != setupTensorsNoCopy(inputs,
                                                  graphInfo.numInputTensors,
                                                  (graphInfo.inputTensors))) {
      ml_loge("Failure in setting up input tensors");
      returnStatus = StatusCode::FAILURE;
    }
    if (StatusCode::SUCCESS != setupTensorsNoCopy(outputs,
                                                  graphInfo.numOutputTensors,
                                                  (graphInfo.outputTensors))) {
      ml_loge("Failure in setting up output tensors");
      returnStatus = StatusCode::FAILURE;
    }
#else
    if (StatusCode::SUCCESS != setupTensors(inputs, graphInfo.numInputTensors,
                                            (graphInfo.inputTensors))) {
      ml_loge("Failure in setting up input tensors");
      returnStatus = StatusCode::FAILURE;
    }
    if (StatusCode::SUCCESS != setupTensors(outputs, graphInfo.numOutputTensors,
                                            (graphInfo.outputTensors))) {
      ml_loge("Failure in setting up output tensors");
      returnStatus = StatusCode::FAILURE;
    }
#endif
    if (StatusCode::SUCCESS != returnStatus) {
      ml_loge("Failure in setupInputAndOutputTensors, cleaning up resources");
      if (nullptr != *inputs) {
        QNN_DEBUG("cleaning up input tensors");
        tearDownTensors(*inputs, graphInfo.numInputTensors);
        *inputs = nullptr;
      }
      if (nullptr != *outputs) {
        QNN_DEBUG("cleaning up output tensors");
        tearDownTensors(*outputs, graphInfo.numOutputTensors);
        *outputs = nullptr;
      }
      ml_loge(
        "Failure in setupInputAndOutputTensors, done cleaning up resources");
    }
    return returnStatus;
  }

  StatusCode populateInputTensor(uint8_t *buffer, Qnn_Tensor_t *input,
                                 InputDataType inputDataType) {
    if (nullptr == input) {
      ml_loge("input is nullptr");
      return StatusCode::FAILURE;
    }
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(input),
             QNN_TENSOR_GET_RANK(input));
    if (inputDataType == InputDataType::FLOAT &&
        QNN_TENSOR_GET_DATA_TYPE(input) != QNN_DATATYPE_FLOAT_32) {
      QNN_DEBUG("Received FLOAT input, but model needs non-float input");
      if (StatusCode::SUCCESS !=
          copyFromFloatToNative(reinterpret_cast<float *>(buffer), input)) {
        QNN_DEBUG("copyFromFloatToNative failure");
        return StatusCode::FAILURE;
      }
    } else {
      size_t length;
      datautil::StatusCode returnStatus;
      std::tie(returnStatus, length) =
        datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(input));
      if (datautil::StatusCode::SUCCESS != returnStatus) {
        return StatusCode::FAILURE;
      }
      pal::StringOp::memscpy(
        reinterpret_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(input).data),
        length, buffer, length);
    }
    return StatusCode::SUCCESS;
  }

  StatusCode populateInputTensor(uint16_t *buffer, Qnn_Tensor_t *input,
                                 InputDataType inputDataType) {
    if (nullptr == input) {
      ml_loge("input is nullptr");
      return StatusCode::FAILURE;
    }

    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(input),
             QNN_TENSOR_GET_RANK(input));

    if (inputDataType == InputDataType::FLOAT &&
        QNN_TENSOR_GET_DATA_TYPE(input) != QNN_DATATYPE_FLOAT_32) {
      QNN_DEBUG("Received FLOAT input, but model needs non-float input");
      if (StatusCode::SUCCESS !=
          copyFromFloatToNative(reinterpret_cast<float *>(buffer), input)) {
        QNN_DEBUG("copyFromFloatToNative failure");
        return StatusCode::FAILURE;
      }
    } else {
      size_t length;
      datautil::StatusCode returnStatus;
      std::tie(returnStatus, length) =
        datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(input));

      if (datautil::StatusCode::SUCCESS != returnStatus) {
        return StatusCode::FAILURE;
      }
      pal::StringOp::memscpy(
        reinterpret_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(input).data),
        length, buffer, length);
    }
    return StatusCode::SUCCESS;
  }

  StatusCode populateInputTensor(float *buffer, Qnn_Tensor_t *input,
                                 InputDataType inputDataType) {
    if (nullptr == input) {
      ml_loge("input is nullptr");
      return StatusCode::FAILURE;
    }
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(input),
             QNN_TENSOR_GET_RANK(input));
    if (inputDataType == InputDataType::FLOAT &&
        QNN_TENSOR_GET_DATA_TYPE(input) != QNN_DATATYPE_FLOAT_32) {
      QNN_DEBUG("Received FLOAT input, but model needs non-float input");
      if (StatusCode::SUCCESS !=
          copyFromFloatToNative(reinterpret_cast<float *>(buffer), input)) {
        QNN_DEBUG("copyFromFloatToNative failure");
        return StatusCode::FAILURE;
      }
    } else {
      size_t length;
      datautil::StatusCode returnStatus;
      std::tie(returnStatus, length) =
        datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(input));
      if (datautil::StatusCode::SUCCESS != returnStatus) {
        return StatusCode::FAILURE;
      }
      pal::StringOp::memscpy(
        reinterpret_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(input).data),
        length, buffer, length);
    }
    return StatusCode::SUCCESS;
  }

private:
  StatusCode setupTensorsNoCopy(Qnn_Tensor_t **tensors, uint32_t tensorCount,
                                Qnn_Tensor_t *tensorWrappers) {
    if (nullptr == tensorWrappers) {
      ml_loge("tensorWrappers is nullptr");
      return StatusCode::FAILURE;
    }
    if (0 == tensorCount) {
      QNN_INFO("tensor count is 0. Nothing to setup.");
      return StatusCode::SUCCESS;
    }
    auto returnStatus = StatusCode::SUCCESS;
    *tensors = (Qnn_Tensor_t *)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
    // std::cout << "tensorCount: "<<tensorCount << std::endl;
    if (nullptr == *tensors) {
      ml_loge("mem alloc failed for *tensors");
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }
    for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
      Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
      // std::cout << tensorIdx << " tensorIdx " << std::endl;
      std::vector<size_t> dims;
      fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(wrapperTensor),
               QNN_TENSOR_GET_RANK(wrapperTensor));
      if (StatusCode::SUCCESS == returnStatus) {
        QNN_DEBUG("allocateBuffer successful");
        (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
        returnStatus = (qnn::tools::sample_app::deepCopyQnnTensorInfo(
                          ((*tensors) + tensorIdx), &wrapperTensor) == true
                          ? StatusCode::SUCCESS
                          : StatusCode::FAILURE);
      }
      if (StatusCode::SUCCESS == returnStatus) {
        QNN_DEBUG("deepCopyQnnTensorInfo successful");
        QNN_TENSOR_SET_MEM_TYPE(((*tensors) + tensorIdx),
                                QNN_TENSORMEMTYPE_MEMHANDLE);
      }
    }
    return returnStatus;
  }

  // Clean up all tensors related data after execution.
  StatusCode tearDownTensors(Qnn_Tensor_t *tensors, uint32_t tensorCount) {
    for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
      QNN_DEBUG("freeing resources for tensor: %d", tensorIdx);
      if (nullptr != QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx])) {
        QNN_DEBUG("freeing dimensions");
        free(QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx]));
      }
      if (nullptr != QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data) {
        QNN_DEBUG("freeing clientBuf.data");
        free(QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data);
      }
    }
    free(tensors);
    return StatusCode::SUCCESS;
  }

  StatusCode fillDims(std::vector<size_t> &dims, uint32_t *inDimensions,
                      uint32_t rank) {
    if (nullptr == inDimensions) {
      QNN_ERROR("input dimensions is nullptr");
      return StatusCode::FAILURE;
    }
    for (size_t r = 0; r < rank; r++) {
      dims.push_back(inDimensions[r]);
    }
    return StatusCode::SUCCESS;
  }

  // Helper method to copy a float buffer, quantize it, and copy
  // it to a tensor (Qnn_Tensor_t) buffer.
  StatusCode copyFromFloatToNative(float *floatBuffer, Qnn_Tensor_t *tensor) {
    if (nullptr == floatBuffer || nullptr == tensor) {
      QNN_ERROR("copyFromFloatToNative(): received a nullptr");
      return StatusCode::FAILURE;
    }

    StatusCode returnStatus = StatusCode::SUCCESS;
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor),
             QNN_TENSOR_GET_RANK(tensor));

    switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      datautil::floatToTfN<uint8_t>(
        static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
        floatBuffer,
        QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
        QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
        datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UFIXED_POINT_16:
      datautil::floatToTfN<uint16_t>(
        static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
        floatBuffer,
        QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
        QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
        datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint8_t>(
            static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint16_t>(
            static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint32_t>(
            static_cast<uint32_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_64:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint64_t>(
            static_cast<uint64_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint64_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int8_t>(
            static_cast<int8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int16_t>(
            static_cast<int16_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int32_t>(
            static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_64:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int64_t>(
            static_cast<int64_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int64_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_BOOL_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint8_t>(
            static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(tensor).data),
            floatBuffer, datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<bool>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    default:
      QNN_ERROR("Datatype not supported yet!");
      returnStatus = StatusCode::FAILURE;
      break;
    }
    return returnStatus;
  }
};
} // namespace nntrainer

#endif
