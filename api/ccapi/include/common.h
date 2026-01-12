// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 seongwoo <mhs4670go@naver.com>
 *
 * @file   common.h
 * @date   18 May 2022
 * @see    https://github.com/nntrainer/nntrainer
 * @author seongwoo <mhs4670go@naver.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is common interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_COMMON_H__
#define __ML_TRAIN_COMMON_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <nntrainer-api-common.h>
#include <string>

namespace ml {
namespace train {

/**
 * @brief Defines Export Method to be called with
 *
 */
enum class ExportMethods {
  METHOD_STRINGVECTOR = 0, /**< export to a string vector */
  METHOD_TFLITE = 1,       /**< export to tflite */
  METHOD_FLATBUFFER = 2,   /**< export to flatbuffer */
  METHOD_UNDEFINED = 999,  /**< undefined */
};

/**
 * @brief   class telling the execution mode of the model/operation
 */
enum class ExecutionMode {
  TRAIN,     /** Training mode, label is necessary */
  INFERENCE, /** Inference mode, label is optional */
  VALIDATE   /** Validate mode, label is necessary */
};

/**
 * @brief     Enumeration of layer compute engine
 */
enum LayerComputeEngine {
  CPU,  /**< CPU as the compute engine */
  GPU,  /**< GPU as the compute engine */
  QNN,  /**< QNN as the compute engine */
  CUDA, /**< CUDA as the compute engine */
};

/**
 * @brief Get the version of NNTrainer
 */
extern std::string getVersion();

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_COMMON_H__
