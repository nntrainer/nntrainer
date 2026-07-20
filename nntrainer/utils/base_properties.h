// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file base_properties.h
 * @date 08 April 2021
 * @brief Convenient property type definition for automated serialization
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __BASE_PROPERTIES_H__
#define __BASE_PROPERTIES_H__

#include <array>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <common.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <util_func.h>

/** base and predefined structures */

namespace nntrainer {

using TensorDim = ml::train::TensorDim;

/**
 * @brief property info to specialize functions based on this
 * @tparam T property type
 */
template <typename T> struct prop_info {
  using prop_type = std::decay_t<T>;             /** property type of T */
  using tag_type = typename prop_type::prop_tag; /** Property tag of T */
  using data_type =
    std::decay_t<decltype(std::declval<prop_type>().get())>; /** Underlying
                                                                datatype of T */
};

/**
 * @brief property info when it is wrapped inside a vector
 *
 * @tparam T property type
 */
template <typename T> struct prop_info<std::vector<T>> {
  using prop_type = typename prop_info<T>::prop_type;
  using tag_type = typename prop_info<T>::tag_type;
  using data_type = typename prop_info<T>::data_type;
};

/**
 * @brief property info when it is wrapped inside an array
 *
 * @tparam T property type
 */
template <typename T, size_t size> struct prop_info<std::array<T, size>> {
  using prop_type = typename prop_info<T>::prop_type;
  using tag_type = typename prop_info<T>::tag_type;
  using data_type = typename prop_info<T>::data_type;
};

/**
 * @brief Get the Prop Key object
 *
 * @tparam T property to get type
 * @param prop property
 * @return constexpr const char* key
 */
template <typename T> constexpr const char *getPropKey(T &&prop) {
  return prop_info<std::decay_t<T>>::prop_type::key;
}

/**
 * @brief property is treated as integer
 *
 */
struct int_prop_tag {};

/**
 * @brief property is treated as unsigned integer
 *
 */
struct uint_prop_tag {};

/**
 * @brief property is treated as unsigned integer
 *
 */
struct size_t_prop_tag {};

/**
 * @brief property is treated as dimension, eg 1:2:3
 *
 */
struct dimension_prop_tag {};

/**
 * @brief property is treated as float
 *
 */
struct float_prop_tag {};

/**
 * @brief property is treated as double
 *
 */
struct double_prop_tag {};

/**
 * @brief property is treated as string
 *
 */
struct str_prop_tag {};

/**
 * @brief property is treated as boolean
 *
 */
struct bool_prop_tag {};

/**
 * @brief property is treated as enum class
 *
 */
struct enum_class_prop_tag {};

/**
 * @brief property treated as a raw pointer
 *
 */
struct ptr_prop_tag {};

/**
 * @brief base property class, inherit this to make a convenient property
 *
 * @tparam T
 */
template <typename T> class Property {

public:
  /**
   * @brief Construct a new Property object
   *
   */
  Property() : value(nullptr){};

  /**
   * @brief Construct a new Property object
   *
   * @param value default value
   */
  Property(const T &value_) { set(value_); }

  /**
   * @brief Copy Construct a new Property object
   *
   * @param rhs right side to copy from
   */
  Property(const Property &rhs) {
    if (this != &rhs && rhs.value) {
      value = std::make_unique<T>(*rhs.value);
    }
  }

  /**
   * @brief Copy assignment operator of a new property
   *
   * @param rhs right side to copy from
   * @return Property& this
   */
  Property &operator=(const Property &rhs) {
    if (this != &rhs && rhs.value) {
      value = std::make_unique<T>(*rhs.value);
    }
    return *this;
  };

  Property(Property &&rhs) noexcept = default;
  Property &operator=(Property &&rhs) noexcept = default;

  /**
   * @brief Destroy the Property object
   *
   */
  virtual ~Property() = default;

  /**
   * @brief cast operator for property
   *
   * @return T value
   */
  operator T &() { return get(); }

  /**
   * @brief cast operator for property
   *
   * @return T value
   */
  operator const T &() const { return get(); }

  /**
   * @brief get the underlying data
   *
   * @return const T& data
   */
  const T &get() const {
    NNTR_THROW_IF(value == nullptr, std::invalid_argument)
      << "Cannot get property, property is empty";
    return *value;
  }

  /**
   * @brief get the underlying data
   *
   * @return T& data
   */
  T &get() {
    NNTR_THROW_IF(value == nullptr, std::invalid_argument)
      << "Cannot get property, property is empty";
    return *value;
  }

  /**
   * @brief check if property is empty
   *
   * @retval true empty
   * @retval false not empty
   */
  bool empty() const { return value == nullptr; }

  /**
   * @brief set the underlying data
   *
   * @param v value to set
   * @throw std::invalid_argument if argument is not valid
   */
  virtual void set(const T &v) {
    NNTR_THROW_IF(isValid(v) == false, std::invalid_argument)
      << "argument is not valid";
    value = std::make_unique<T>(v);
  }

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @retval true if valid
   * @retval false if not valid
   */
  virtual bool isValid(const T &v) const { return true; }

  /**
   * @brief operator==
   *
   * @param rhs right side to compare
   * @retval true if equal
   * @retval false if not equal
   */
  bool operator==(const Property<T> &rhs) const { return *value == *rhs.value; }

private:
  std::unique_ptr<T> value; /**< underlying data */
};

/**
 * @brief enum property
 *
 * @tparam T underlying type info to query enum_info
 */
template <typename EnumInfo>
class EnumProperty : public Property<typename EnumInfo::Enum> {
public:
  static EnumInfo enum_info_;
};

/**
 * @brief abstract class for tensor dimension
 *
 */
class TensorDimProperty : public Property<TensorDim> {
public:
  /**
   * @brief Destroy the TensorDim Property object
   *
   */
  virtual ~TensorDimProperty() = default;
};

/**
 * @brief abstract class for positive integer
 *
 */
class PositiveIntegerProperty : public Property<unsigned int> {
public:
  /**
   * @brief Destroy the Positive Integer Property object
   *
   */
  virtual ~PositiveIntegerProperty() = default;

  /**
   * @brief isValid override, check if value > 0
   *
   * @param value value to check
   * @retval true if value > 0
   */
  virtual bool isValid(const unsigned int &value) const override;
};
/**
 * @brief meta function to cast tag to it's base
 * @code below is the test spec for the cast
 *
 * struct custom_tag: int_prop_tag {};
 *
 * using tag_type = tag_cast<custom_tag, float_prop_tag>::type
 * static_assert(<std::is_same_v<tag_type, custom_tag> == true);
 *
 * using tag_type = tag_cast<custom_tag, int_prop_tag>::type
 * static_assert(<std::is_same_v<tag_type, int_prop_tag> == true);
 *
 * using tag_type = tag_cast<custom_tag, float_prop_tag, int_prop_tag>::type
 * static_assert(std::is_same_v<tag_type, int_prop_tag> == true);
 *
 * @tparam Tags First tag: tag to be casted, rest tags: candidates
 */
template <typename... Tags> struct tag_cast;

/**
 * @brief base case of tag_cast, if nothing matches return @a Tag
 *
 * @tparam Tag Tag to be casted
 * @tparam Others empty parameter pack
 */
template <typename Tag, typename... Others> struct tag_cast<Tag, Others...> {
  using type = Tag;
};

/**
 * @brief normal case of the tag cast
 *
 * @tparam Tag tag to be casted
 * @tparam BaseTag candidates to cast the tag
 * @tparam Others pending candidates to be compared
 */
template <typename Tag, typename BaseTag, typename... Others>
struct tag_cast<Tag, BaseTag, Others...> {
  using type = std::conditional_t<std::is_base_of<BaseTag, Tag>::value, BaseTag,
                                  typename tag_cast<Tag, Others...>::type>;
};

/**
 * @brief property to string converter.
 * This structure defines how to convert to convert from/to string
 *
 * @tparam Tag tag type for the converter
 * @tparam DataType underlying datatype
 */
template <typename Tag, typename DataType> struct str_converter {

  /**
   * @brief convert underlying value to string
   *
   * @param value value to convert to string
   * @return std::string string
   */
  static std::string to_string(const DataType &value);

  /**
   * @brief convert string to underlying value
   *
   * @param value value to convert to string
   * @return DataType converted type
   */
  static DataType from_string(const std::string &value);
};

/**
 * @brief str converter specialization for enum classes
 *
 * @tparam EnumInfo enum informations
 */
template <typename EnumInfo>
struct str_converter<enum_class_prop_tag, EnumInfo> {

  /**
   * @copydoc template <typename Tag, typename DataType> struct str_converter
   */
  static std::string to_string(const typename EnumInfo::Enum &value) {
    constexpr auto size = EnumInfo::EnumList.size();
    constexpr const auto data = std::data(EnumInfo::EnumList);
    for (unsigned i = 0; i < size; ++i) {
      if (data[i] == value) {
        return EnumInfo::EnumStr[i];
      }
    }
    throw std::invalid_argument("Cannot find value in the enum list");
  }

  /**
   * @copydoc template <typename Tag, typename DataType> struct str_converter
   */
  static typename EnumInfo::Enum from_string(const std::string &value) {
    constexpr auto size = EnumInfo::EnumList.size();
    constexpr const auto data = std::data(EnumInfo::EnumList);
    for (unsigned i = 0; i < size; ++i) {
      if (istrequal(EnumInfo::EnumStr[i], value.c_str())) {
        return data[i];
      }
    }
    throw std::invalid_argument("No matching enum for value: " + value);
  }
};

/**
 * @brief str converter which serializes a pointer and returns back to a ptr
 *
 * @tparam DataType pointer type
 */
template <typename DataType> struct str_converter<ptr_prop_tag, DataType> {

  /**
   * @brief convert underlying value to string
   *
   * @param value value to convert to string
   * @return std::string string
   */
  static std::string to_string(const DataType &value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
  }

  /**
   * @brief convert string to underlying value
   *
   * @param value value to convert to string
   * @return DataType converted type
   */
  static DataType from_string(const std::string &value) {
    std::stringstream ss(value);
    uintptr_t addr = static_cast<uintptr_t>(std::stoull(value, 0, 16));
    return reinterpret_cast<DataType>(addr);
  }
};

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string
str_converter<str_prop_tag, std::string>::to_string(const std::string &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string
str_converter<str_prop_tag, std::string>::from_string(const std::string &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string str_converter<uint_prop_tag, unsigned int>::to_string(
  const unsigned int &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
unsigned int str_converter<uint_prop_tag, unsigned int>::from_string(
  const std::string &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string
str_converter<size_t_prop_tag, size_t>::to_string(const size_t &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
size_t
str_converter<size_t_prop_tag, size_t>::from_string(const std::string &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string str_converter<bool_prop_tag, bool>::to_string(const bool &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
bool str_converter<bool_prop_tag, bool>::from_string(const std::string &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string str_converter<float_prop_tag, float>::to_string(const float &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
float str_converter<float_prop_tag, float>::from_string(
  const std::string &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
std::string
str_converter<double_prop_tag, double>::to_string(const double &value);

/**
 * @copydoc template <typename Tag, typename DataType> struct str_converter
 */
template <>
double
str_converter<double_prop_tag, double>::from_string(const std::string &value);

/**
 * @brief convert dispatcher (to string)
 *
 * @tparam T type to convert
 * @param property property to convert
 * @return std::string converted string
 */
template <typename T> std::string to_string(const T &property) {
  using info = prop_info<T>;
  using tag_type =
    typename tag_cast<typename info::tag_type, int_prop_tag, uint_prop_tag,
                      dimension_prop_tag, float_prop_tag, str_prop_tag,
                      enum_class_prop_tag>::type;
  using data_type = typename info::data_type;

  if constexpr (std::is_same_v<tag_type, enum_class_prop_tag>) {
    return str_converter<tag_type, decltype(T::enum_info_)>::to_string(
      property.get());
  } else {
    return str_converter<tag_type, data_type>::to_string(property.get());
  }
}

/**
 * @brief to_string vector specialization
 * @copydoc template <typename T> std::string to_string(const T &property)
 */
template <typename T> std::string to_string(const std::vector<T> &property) {
  std::stringstream ss;
  auto last_iter = property.end() - 1;
  for (auto iter = property.begin(); iter != last_iter; ++iter) {
    ss << to_string(*iter) << ',';
  }
  ss << to_string(*last_iter);

  return ss.str();
}

/**
 * @brief to_string array specialization
 * @copydoc template <typename T> std::string to_string(const T &property)
 */
template <typename T, size_t sz>
static std::string to_string(const std::array<T, sz> &value) {
  std::stringstream ss;
  auto last_iter = value.end() - 1;
  for (auto iter = value.begin(); iter != last_iter; ++iter) {
    ss << to_string(*iter) << ',';
  }
  ss << to_string(*last_iter);

  return ss.str();
}

/**
 *
 * @brief convert dispatcher (from string)
 *
 *
 * @tparam T type to convert
 * @param str string to convert
 * @param[out] property property, converted type
 */
template <typename T> void from_string(const std::string &str, T &property) {
  using info = prop_info<T>;
  using tag_type =
    typename tag_cast<typename info::tag_type, int_prop_tag, uint_prop_tag,
                      dimension_prop_tag, float_prop_tag, str_prop_tag,
                      enum_class_prop_tag>::type;
  using data_type = typename info::data_type;

  if constexpr (std::is_same_v<tag_type, enum_class_prop_tag>) {
    property.set(
      str_converter<tag_type, decltype(T::enum_info_)>::from_string(str));
  } else {
    property.set(str_converter<tag_type, data_type>::from_string(str));
  }
}

/**
 * @brief transform iternal data, this is to use with std::transform
 *
 * @param item item to transform
 * @return DataType transformed result
 */
template <typename T> static T from_string_helper_(const std::string &item) {
  T t;
  from_string(item, t);
  return t;
}

static const std::regex reg_("\\s*\\,\\s*");

/**
 * @brief from_string array specialization
 * @copydoc template <typename T> void from_string(const std::string &str, T
 * &property)
 * @note array implies that the size is @b fixed so there will be a validation
 * check on size
 */
template <typename T, size_t sz>
void from_string(const std::string &value, std::array<T, sz> &property) {
  auto v = split(value, reg_);
  NNTR_THROW_IF(v.size() != sz, std::invalid_argument)
    << "size must match with array size, array size: " << sz
    << " string: " << value;

  std::transform(v.begin(), v.end(), property.begin(), from_string_helper_<T>);
}

/**
 * @brief from_string vector specialization
 * @copydoc str_converter<Tag, DataType>::to_string(const DataType &value)
 * @note vector implies that the size is @b not fixed so there shouldn't be any
 * validation on size
 *
 */
template <typename T>
void from_string(const std::string &value, std::vector<T> &property) {
  auto v = split(value, reg_);

  property.clear();
  property.reserve(v.size());
  std::transform(v.begin(), v.end(), std::back_inserter(property),
                 from_string_helper_<T>);
}
/******** below section is for enumerations ***************/
/**
 * @brief     Enumeration of Data Type for model & layer
 */
struct TensorDataTypeInfo {
  using Enum = nntrainer::TensorDim::DataType;
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::BCQ,  Enum::QINT4, Enum::QINT8, Enum::QINT16, Enum::FP16,
    Enum::FP32, Enum::UINT4, Enum::UINT8, Enum::UINT16, Enum::Q4_K,
    Enum::Q6_K, Enum::Q4_0,  Enum::Q8_0,  Enum::QS4CX};
  static constexpr const char *EnumStr[] = {
    "BCQ",   "QINT4",  "QINT8", "QINT16", "FP16", "FP32", "UINT4",
    "UINT8", "UINT16", "Q4_K",  "Q6_K",   "Q4_0", "Q8_0", "QS4CX"};
};

/**
 * @brief     Enumeration of Format for model & layer
 */
struct TensorFormatInfo {
  using Enum = nntrainer::TensorDim::Format;
  static constexpr std::initializer_list<Enum> EnumList = {Enum::NCHW,
                                                           Enum::NHWC};

  static constexpr const char *EnumStr[] = {"NCHW", "NHWC"};
};

/**
 * @brief     Enumeration of Tensor Type for model & layer
 */
enum class TensorType_ {
  WEIGHT,         /**< Weight Tensor */
  IN_TENSOR,      /**< Input Tensor for FORWARD_FUNC_LIFESPAN */
  MAX_IN_TENSOR,  /**< MAX LIFESPAN INPUT TENSOR */
  OUT_TENSOR,     /**< OUTPUT TENSOR */
  MAX_OUT_TENSOR, /**< MAX LIFESPAN OUTPUT TENSOR */
};

/**
 * @brief     Enumeration of Tensor Type for model & layer
 */
struct TensorTypeInfo {
  using Enum = nntrainer::TensorType_;
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::WEIGHT, Enum::IN_TENSOR, Enum::MAX_IN_TENSOR, Enum::OUT_TENSOR,
    Enum::MAX_OUT_TENSOR};
  static constexpr const char *EnumStr[] = {
    "WEIGHT", "IN_TENSOR", "MAX_IN_TENSOR", "OUT_TENSOR", "MAX_OUT_TENSOR"};
};

namespace props {

/**
 * @brief Weight Data Type Enumeration Information
 *  This property can be used when any layer is created.
 *  This property is differentiated with TensorDataType in that it doesn't have
 *  default value
 */
class WeightDtype final : public EnumProperty<TensorDataTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "weight_dtype";

  /**
   * @brief Constructor
   */
  WeightDtype(){};

  /**
   * @brief Constructor
   *
   * @param value value to set
   */
  WeightDtype(TensorDataTypeInfo::Enum value) { set(value); };
};

/**
 * @brief Input Data Type Enumeration Information
 *  This property can be used by input layers to preserve a non-default
 *  placeholder dtype instead of inheriting the model activation dtype.
 */
class InputDtype final : public EnumProperty<TensorDataTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "input_dtype";

  /**
   * @brief Constructor
   */
  InputDtype(){};

  /**
   * @brief Constructor
   *
   * @param value value to set
   */
  InputDtype(TensorDataTypeInfo::Enum value) { set(value); };
};

/**
 * @brief Activation Enumeration Information
 *
 */
class TensorDataType final : public EnumProperty<TensorDataTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "tensor_dtype";

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to FP32
   */
  TensorDataType(
    TensorDataTypeInfo::Enum value = TensorDataTypeInfo::Enum::FP32) {
    set(value);
  };
};

/**
 * @brief model tensor type : NCHW or NHWC
 *
 */
class TensorFormat final : public EnumProperty<TensorFormatInfo> {
public:
  static constexpr const char *key =
    "tensor_format";                    /**< unique key to access */
  using prop_tag = enum_class_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to NCHW
   */
  TensorFormat(TensorFormatInfo::Enum value = TensorFormatInfo::Enum::NCHW) {
    set(value);
  };
};

/**
 * @brief model tensor type is the clue of life span
 *
 */
class TensorType final : public EnumProperty<nntrainer::TensorTypeInfo> {
public:
  static constexpr const char *key = "tensor_type";
  using prop_tag = enum_class_prop_tag;
  TensorType(TensorTypeInfo::Enum value = TensorTypeInfo::Enum::WEIGHT) {
    set(value);
  }
};

/**
 * @brief     Enumeration of Run Engine type
 */
struct ComputeEngineTypeInfo {
  using Enum = ml::train::LayerComputeEngine;
  static constexpr std::initializer_list<Enum> EnumList = {Enum::CPU, Enum::GPU,
                                                           Enum::QNN};
  static constexpr const char *EnumStr[] = {"cpu", "gpu", "qnn"};
};

/**
 * @brief ComputeEngine Enumeration Information
 *
 */
class ComputeEngine final
  : public EnumProperty<nntrainer::props::ComputeEngineTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "engine";
};

// /**
//  * @brief trainable property, use this to set and check how if certain
//  layer is
//  * trainable
//  *
//  */
// class Trainable : public nntrainer::Property<bool> {
// public:
//   /**
//    * @brief Construct a new Trainable object
//    *
//    */
//   Trainable(bool val = true) : nntrainer::Property<bool>(val) {}
//   static constexpr const char *key = "trainable";
//   using prop_tag = bool_prop_tag;
// };

} // namespace props

} // namespace nntrainer

#endif // __BASE_PROPERTIES_H__
