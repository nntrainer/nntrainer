// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   model_common_properties.h
 * @date   27 Aug 2021
 * @brief  This file contains common properties for model
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __MODEL_COMMON_PROPERTIES_H__
#define __MODEL_COMMON_PROPERTIES_H__

#include <base_properties.h>

#ifdef __cplusplus
namespace nntrainer::props {

/**
 * @brief model epoch property
 *
 */
class Epochs : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "epochs"; /**< unique key to access */
  using prop_tag = uint_prop_tag;              /**< property type */
  /**
   * @brief Construct a new Epochs object
   *
   * @param value value to set
   */
  Epochs(unsigned int value = 1);
};

/**
 * @brief model loss property (deprecated)
 *
 */
class LossType : public Property<std::string> {
public:
  static constexpr const char *key = "loss"; /**< unique key to access */
  using prop_tag = str_prop_tag;             /**< property type */

  /**
   * @brief check if valid
   *
   * @param value value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &value) const override;
};

/**
 * @brief model save path property
 *
 */
class SavePath : public Property<std::string> {
public:
  static constexpr const char *key = "save_path"; /**< unique key to access */
  using prop_tag = str_prop_tag;                  /**< property type */
};

/**
 * @brief model save path property
 *
 */
class SaveBestPath : public Property<std::string> {
public:
  static constexpr const char *key =
    "save_best_path";            /**< unique key to access */
  using prop_tag = str_prop_tag; /**< property type */
};

/**
 * @brief model batch size property
 *
 */
class TrainingBatchSize : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "batch_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                  /**< property type */

  /**
   * @brief Construct a new Batch Size object
   *
   * @param value value to set, defaults to 1
   */
  TrainingBatchSize(unsigned int value = 1);
};

/**
 * @brief model continue property
 *
 */
class ContinueTrain : public Property<bool> {
public:
  static constexpr const char *key =
    "continue_train";             /**< unique key to access */
  using prop_tag = bool_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to false
   */
  ContinueTrain(bool value = false);
};

/**
 * @brief model optimization property
 *
 */
class MemoryOptimization : public Property<bool> {
public:
  static constexpr const char *key =
    "memory_optimization";        /**< unique key to access */
  using prop_tag = bool_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to true
   */
  MemoryOptimization(bool value = true);
};

/**
 * @brief cache size property
 *
 */
class Fsu : public Property<bool> {
public:
  static constexpr const char *key = "fsu"; /**< unique key to access */
  using prop_tag = bool_prop_tag;           /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to false
   */
  Fsu(bool value = false);
};

/**
 * @brief cache file path property
 *
 */
class FsuPath : public Property<std::string> {
public:
  static constexpr const char *key = "fsu_path"; /**< unique key to access */
  using prop_tag = str_prop_tag;                 /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to current directory
   */
  FsuPath(const std::string &value = ".");
};

/**
 * @brief cache file path property
 *
 */
class FsuLookahead : public Property<unsigned int> {
public:
  static constexpr const char *key =
    "fsu_lookahead";              /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to current directory
   */
  FsuLookahead(const unsigned int &value = 0);
};

/**
 * @brief     Enumeration of Data Type for model & layer
 */
struct ModelTensorDataTypeInfo {
  enum Enum {
    W3A32,
    W4A16,
    W4A32,
    W8A16,
    W8A32,
    W16A16,
    W16A32,
    W32A16,
    W32A32,
    WQ16AQ16,
    WU16AU16,
    W8AU16,
    WU4AU8,
    WU4AU16,
    WU8AU8,
    WU8AU16,
    WQ4KA32,
    WQ40A32,
    WQ40A16,
    WQS4CXA32,
    WQ80A32,
    WQ80A16,
  };
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::W3A32,   Enum::W4A16,    Enum::W4A32,    Enum::W8A16,
    Enum::W8A32,   Enum::W16A16,   Enum::W16A32,   Enum::W32A16,
    Enum::W32A32,  Enum::WQ16AQ16, Enum::WU16AU16, Enum::W8AU16,
    Enum::WU4AU8,  Enum::WU4AU16,  Enum::WU8AU8,   Enum::WU8AU16,
    Enum::WQ4KA32, Enum::WQ40A32,  Enum::WQ40A16,  Enum::WQS4CXA32,
    Enum::WQ80A32, Enum::WQ80A16,
  };

  static constexpr const char *EnumStr[] = {
    "BCQ-FP32",    "QINT4-FP16",    "QINT4-FP32",    "QINT8-FP16",
    "QINT8-FP32",  "FP16-FP16",     "FP16-FP32",     "FP32-FP16",
    "FP32-FP32",   "QINT16-QINT16", "UINT16-UINT16", "QINT8-UINT16",
    "UINT4-UINT8", "UINT4-UINT16",  "UINT8-UINT8",   "UINT8-UINT16",
    "Q4_K-FP32",   "Q4_0-FP32",     "Q4_0-FP16",     "QS4CX-FP32",
    "Q8_0-FP32",   "Q8_0-FP16"};
};

/**
 * @brief Activation Enumeration Information
 *
 */
class ModelTensorDataType final : public EnumProperty<ModelTensorDataTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "model_tensor_type";

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to W32A32
   */
  ModelTensorDataType(ModelTensorDataTypeInfo::Enum value =
                        ModelTensorDataTypeInfo::Enum::W32A32);
};

/**
 * @brief LossScale property, loss is scaled by this value
 *
 */
class LossScale : public Property<float> {
public:
  LossScale(float value = 1.0f);
  static constexpr const char *key = "loss_scale"; /**< unique key to access */
  using prop_tag = float_prop_tag;                 /**< property type */

  /**
   * @brief check if valid
   *
   * @param value value to check
   * @return bool true if valid
   */
  bool isValid(const float &value) const override;
};

} // namespace nntrainer::props

#endif

#endif // __MODEL_COMMON_PROPERTIES_H__
