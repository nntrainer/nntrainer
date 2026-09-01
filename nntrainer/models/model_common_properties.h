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
    /**
     * @brief Deprecated: the two QINT4 weight types.
     * @details Channel-wise int4 is QS4CX; QINT4 is deprecated and is an
     * on-disk format only, so an int4 weight materialises as a QS4CX tensor
     * and a model asking for one of these could not allocate a single weight.
     * Use WQS4CXA16 / WQS4CXA32 ("QS4CX-FP16" / "QS4CX-FP32") instead. The
     * enumerators are retained, and rejected by
     * ModelTensorDataType::isValid(), only so that the values of the
     * enumerators after them stay stable for the installed header; neither
     * appears in EnumList or EnumStr, so neither parses from a string.
     */
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
    WQS4CXA16,
  };
  /// @note W4A16 and W4A32 are deliberately absent: QINT4 is deprecated in
  /// favour of channel-wise QS4CX, so the two "QINT4-*" spellings are no
  /// longer accepted values. EnumList and EnumStr are matched by position, so
  /// the two lists drop the same slots together.
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::W3A32,    Enum::W8A16,     Enum::W8A32,     Enum::W16A16,
    Enum::W16A32,   Enum::W32A16,    Enum::W32A32,    Enum::WQ16AQ16,
    Enum::WU16AU16, Enum::W8AU16,    Enum::WU4AU8,    Enum::WU4AU16,
    Enum::WU8AU8,   Enum::WU8AU16,   Enum::WQ4KA32,   Enum::WQ40A32,
    Enum::WQ40A16,  Enum::WQS4CXA32, Enum::WQS4CXA16,
  };

  static constexpr const char *EnumStr[] = {
    "BCQ-FP32",      "QINT8-FP16",   "QINT8-FP32",  "FP16-FP16",
    "FP16-FP32",     "FP32-FP16",    "FP32-FP32",   "QINT16-QINT16",
    "UINT16-UINT16", "QINT8-UINT16", "UINT4-UINT8", "UINT4-UINT16",
    "UINT8-UINT8",   "UINT8-UINT16", "Q4_K-FP32",   "Q4_0-FP32",
    "Q4_0-FP16",     "QS4CX-FP32",   "QS4CX-FP16"};
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

  /**
   * @brief Reject the two deprecated QINT4 weight types
   * @details Channel-wise int4 is QS4CX; QINT4 is deprecated. It is an on-disk
   * format only, so an int4 weight becomes a QS4CX tensor in memory and a
   * model built with QINT4-FP16 or QINT4-FP32 would throw from the Tensor
   * constructor at the first weight allocation. Neither spelling parses from a
   * string any more, having been dropped from EnumList and EnumStr; this
   * covers the remaining path, a caller passing the retained enumerator
   * directly, and names the replacement rather than failing at allocation.
   * @param v value to check
   * @retval true if the value is a type the library can allocate
   */
  bool isValid(const ModelTensorDataTypeInfo::Enum &v) const override;
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
