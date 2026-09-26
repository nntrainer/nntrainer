// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   model_common_properties.cpp
 * @date   27 Aug 2021
 * @brief  This file contains common properties for model
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <model_common_properties.h>

#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer::props {
Epochs::Epochs(unsigned int value) { set(value); }

bool LossType::isValid(const std::string &value) const {
  ml_logw("Model loss property is deprecated, use loss layer directly instead");
  return istrequal(value, "cross") || istrequal(value, "mse") ||
         istrequal(value, "kld");
}

TrainingBatchSize::TrainingBatchSize(unsigned int value) { set(value); }

ContinueTrain::ContinueTrain(bool value) { set(value); }

MemoryOptimization::MemoryOptimization(bool value) { set(value); }

Fsu::Fsu(bool value) { set(value); }

FsuPath::FsuPath(const std::string &value) { set(value); }

FsuLookahead::FsuLookahead(const unsigned int &value) { set(value); }
ModelTensorDataType::ModelTensorDataType(ModelTensorDataTypeInfo::Enum value) {
  set(value);
}

bool ModelTensorDataType::isValid(
  const ModelTensorDataTypeInfo::Enum &v) const {
  if (v == ModelTensorDataTypeInfo::Enum::W4A16 ||
      v == ModelTensorDataTypeInfo::Enum::W4A32) {
    ml_loge("model_tensor_type QINT4-FP16 and QINT4-FP32 are deprecated: "
            "channel-wise int4 is QS4CX. QINT4 is an on-disk format only and "
            "an int4 weight materialises as a QS4CX tensor, so neither value "
            "can be allocated. Use QS4CX-FP16 or QS4CX-FP32 instead.");
    return false;
  }
  return true;
}
LossScale::LossScale(float value) { set(value); }

bool LossScale::isValid(const float &value) const {
  bool is_valid = (std::fpclassify(value) != FP_ZERO);
  if (!is_valid)
    ml_loge("Loss scale cannot be 0");
  return is_valid;
}

} // namespace nntrainer::props
