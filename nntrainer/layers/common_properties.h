// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   common_properties.h
 * @date   09 April 2021
 * @brief  This file contains list of common properties widely used across
 * layers
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __COMMON_PROPERTIES_H__
#define __COMMON_PROPERTIES_H__

#include <array>
#include <fstream>
#include <string>

#include <base_properties.h>
#include <common.h>
#include <connection.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @brief     Enumeration of activation function type
 * @note      Upon changing this enum, ActivationTypeInfo must be changed
 * accordingly
 */
enum class ActivationType {
  ACT_TANH,         /**< tanh */
  ACT_SIGMOID,      /**< sigmoid */
  ACT_RELU,         /**< ReLU */
  ACT_SWISH,        /**< Swish */
  ACT_GELU,         /**< GELU */
  ACT_TANH_GELU,    /**< tanh GELU */
  ACT_SIGMOID_GELU, /**< sigmoid GELU */
  ACT_SOFTMAX,      /**< softmax */
  ACT_SOFTPLUS,     /**< softplus */
  ACT_LEAKY_RELU,   /**< Leaky ReLU */
  ACT_ELU,          /**< ELU */
  ACT_SELU,         /**< SELU */
  ACT_MISH,         /**< Mish */
  ACT_NONE,         /**< no op */
  ACT_UNKNOWN       /**< unknown */
};

namespace props {

/**
 * @brief Name property, name is an identifier of an object
 *
 */
class Name : public nntrainer::Property<std::string> {
public:
  /**
   * @brief Construct a new Name object without a default value
   *
   */
  Name();

  /**
   * @brief Construct a new Name object with a default value
   *
   * @param value value to contrusct the property
   */
  Name(const std::string &value);

  static constexpr const char *key = "name"; /**< unique key to access */
  using prop_tag = str_prop_tag;             /**< property type */

  /**
   * @brief Name setter
   *
   * @param value value to set
   */
  void set(const std::string &value) override;

  /**
   * @brief name validator
   *
   * @param v string to validate
   * @retval true if it contains alphanumeric and/or '-', '_', '/'
   * @retval false if it is empty or contains non-valid character
   */
  bool isValid(const std::string &v) const override;
};

/**
 * @brief unit property, unit is used to measure how many weights are there
 *
 */
class Unit : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "unit"; /**< unique key to access */
  using prop_tag = uint_prop_tag;            /**< property type */
};

/**
 * @brief trainable property, use this to set and check how if certain layer is
 * trainable
 *
 */
class Trainable : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Trainable object
   *
   */
  Trainable(bool val = true) : nntrainer::Property<bool>(val) {}
  static constexpr const char *key = "trainable";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief Tensor Dimension property
 *
 */
class TensorDimension : public TensorDimProperty {
public:
  static constexpr const char *key = "dim"; /**< unique key to access */
  using prop_tag = dimension_prop_tag;      /**< property type */
};

/**
 * @brief Inplace operation property
 *
 */
class InPlaceProp : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "inplace"; /**< unique key to access */
  using prop_tag = bool_prop_tag;               /**< property type */
};

/**
 * @brief Inplace direction property
 *
 */
class InPlaceDirectionProp : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key =
    "inplace_direction";         /**< unique key to access */
  using prop_tag = str_prop_tag; /**< property type */
};

/**
 * @brief trainable property, use this to set and check how if certain layer is
 * trainable
 *
 */
class Packed : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Trainable object
   * if it is true, then weight type always follows tensor_type[1]( Global
   * Weight Type ). if it is false, the weight type follows tensor_type[2]
   * (Global Activation Type)
   */
  Packed(bool val = true) : nntrainer::Property<bool>(val) {}
  static constexpr const char *key = "packed";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief DisableBias to disable the bias
 *
 */
class DisableBias : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a DisableBias object
   *
   */
  DisableBias(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = bool_prop_tag;
  static constexpr const char *key = "disable_bias";
};

/**
 * @brief Integrate bias_ih and bias_hh to bias_h to use only 1 bias (Used in
 * rnn variant)
 *
 */
class IntegrateBias : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a IntegrateBias object
   *
   */
  IntegrateBias(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = bool_prop_tag;
  static constexpr const char *key = "integrate_bias";
};

/**
 * @brief Normalization property, normalize the input to be in range [0, 1] if
 * true
 *
 */
class Normalization : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Normalization object
   *
   */
  Normalization(bool value = false);
  static constexpr const char *key = "normalization";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief Standardization property, standardization standardize the input
 * to be mean 0 and std 1 if true
 *
 */
class Standardization : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Standardization object
   *
   */
  Standardization(bool value = false);
  static constexpr const char *key = "standardization";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief Connection prop tag type
 *
 */
struct connection_prop_tag {};

/**
 * @brief InputSpec property, this defines connection specification of an input
 *
 */
class InputConnection : public nntrainer::Property<Connection> {
public:
  /**
   * @brief Construct a new Input Spec object
   *
   */
  InputConnection();

  /**
   * @brief Construct a new Input Spec object
   *
   * @param value default value of a input spec
   */
  InputConnection(const Connection &value);
  static constexpr const char *key =
    "input_layers";                     /**< unique key to access */
  using prop_tag = connection_prop_tag; /**< property type */
};

/**
 * @brief Epsilon property, this is used to avoid divide by zero
 *
 */
class Epsilon : public nntrainer::Property<float> {

public:
  /**
   * @brief Construct a new Epsilon object with a default value 0.001
   *
   */
  Epsilon(float value = 0.001);
  static constexpr const char *key = "epsilon"; /**< unique key to access */
  using prop_tag = float_prop_tag;              /**< property type */

  /**
   * @brief Epsilon validator
   *
   * @param value float to validate
   * @retval true if it is greater or equal than 0.0
   * @retval false if it is smaller than 0.0
   */
  bool isValid(const float &value) const override;
};

/**
 * @brief Exponent property, this is used for pow operation
 *
 */
class Exponent : public nntrainer::Property<float> {

public:
  /**
   * @brief Construct a new Exponent object
   *
   */
  Exponent(float value = 1.0f);
  static constexpr const char *key = "exponent"; /**< unique key to access */
  using prop_tag = float_prop_tag;               /**< property type */
};

/**
 * @brief Momentum property, moving average in batch normalization layer
 *
 */
class Momentum : public nntrainer::Property<float> {

public:
  /**
   * @brief Construct a new Momentum object with a default value 0.99
   *
   */
  Momentum(float value = 0.99);
  static constexpr const char *key = "momentum"; /**< unique key to access */
  using prop_tag = float_prop_tag;               /**< property type */

  /**
   * @brief Momentum validator
   *
   * @param value float to validate
   * @retval true if it is greater than 0.0 and smaller than 1.0
   * @retval false if it is smaller or equal than 0.0
   * or greater or equal than 1.0
   */
  bool isValid(const float &value) const override;
};

/**
 * @brief split number property, split number indicates how many numbers of outs
 * are generated by spliting the input dimension
 *
 */
class SplitNumber : public PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "split_number";               /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */
};

/**
 * @brief Axis property, idx in the dimension
 *
 */
class Axis : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "axis"; /**< unique key to access */
  using prop_tag = uint_prop_tag;            /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @retval true if it is greater equal to 0 and smaller than
   * ml::train::TensorDim::MAXDIM
   * @retval false if it is smaller than 0 or greater than
   * ml::train::TensorDim::MAXDIM
   */
  bool isValid(const unsigned int &value) const override;
};

/**
 * @brief StartDimension property, start dimension to be flatten
 *
 */
class StartDimension : public Axis {
public:
  StartDimension(unsigned int value = 1);
  static constexpr const char *key = "start_dimension";
  using prop_tag = uint_prop_tag;

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @retval true if it is greater than 0 and smaller than
   * ml::train::TensorDim::MAXDIM
   * @retval false if it is smaller or equal to 0 or greater than
   * ml::train::TensorDim::MAXDIM
   */
  bool isValid(const unsigned int &value) const override;
};

/**
 * @brief EndDimension property, end dimension to be flatten
 *
 */
class EndDimension : public Axis {
public:
  EndDimension(unsigned int value = ml::train::TensorDim::MAXDIM - 1);
  static constexpr const char *key = "end_dimension";
  using prop_tag = uint_prop_tag;

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @retval true if it is greater than 0 and smaller than
   * ml::train::TensorDim::MAXDIM
   * @retval false if it is smaller or equal to 0 or greater than
   * ml::train::TensorDim::MAXDIM
   */
  bool isValid(const unsigned int &value) const override;
};

/**
 * @brief StartIndex property, start index to be slicing
 *
 */
class StartIndex : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "start_index"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

/**
 * @brief EndIndex property, end index to be slicing
 *
 */
class EndIndex : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "end_index"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                 /**< property type */
};

/**
 * @brief SplitDimension property, dimension along which to split the input
 *
 */
class SplitDimension : public Axis {
public:
  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @retval true if it is greater than 0 and smaller than
   * ml::train::TensorDim::MAXDIM
   * @retval false if it is smaller or equal to 0 or greater than
   * ml::train::TensorDim::MAXDIM
   */
  bool isValid(const unsigned int &value) const override;
};

/**
 * @brief ConcatDimension property, dimension along which to concat the input
 *
 */
class ConcatDimension : public SplitDimension {};

/**
 * @brief ReduceDimension property, dimension along which to reduce the input
 *
 */
class ReduceDimension : public SplitDimension {};

/**
 * @brief FilterSize property, filter size is used to measure how many filters
 * are there
 *
 */
class FilterSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "filters"; /**< unique key to access */
  using prop_tag = uint_prop_tag;               /**< property type */
};

/**
 * @brief KernelSize property, kernel size is used to measure the filter size
 *
 */
class KernelSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "kernel_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

/**
 * @brief PoolSize property, pool size is used to measure the pooling size
 *
 */
class PoolSize : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new PoolSize object
   *
   */
  PoolSize() {}

  /**
   * @brief Construct a new PoolSize object with default value
   *
   */
  PoolSize(unsigned int value);
  static constexpr const char *key = "pool_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                 /**< property type */
};

/**
 * @brief Stride property, stride is used to measure how much it will be slide
 * the filter
 *
 */
class Stride : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new Stride object with a default value 1
   *
   */
  Stride(unsigned int value = 1);
  static constexpr const char *key = "stride"; /**< unique key to access */
  using prop_tag = uint_prop_tag;              /**< property type */
};

/**
 * @brief Dilation property, dilation indicates how many space will be inserted
 * between kernel element
 *
 */
class Dilation : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new Dilation object with a default value 1
   *
   */
  Dilation(unsigned int value = 1);
  static constexpr const char *key = "dilation"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                /**< property type */
};

/**
 * @brief Padding2D property, this is used to calculate padding2D
 * @details Padding2D is saved as a string. Upon calling Padding2D::compute,
 * returns std::vector<unsigned int> which has computed padding2Ds, below
 * formats are accepted valid
 * 1. "same" (case insensitive literal string)
 * 2. "valid" (case insensitive literal string)
 * 3. "padding2D_all", eg) padding=1
 * 4. "padding2D_height, padding2D_width" eg) padding=1,1
 * 5. "padding2D_top, padding2D_bottom, padding2D_left, padding2D_right" eg)
 * padding=1,1,1,1
 *
 */
class Padding2D final : public nntrainer::Property<std::string> {
public:
  /**
   * @brief Construct a new Padding2D object
   *
   */
  Padding2D(const std::string &value = "valid") :
    nntrainer::Property<std::string>(value) {} /**< default value if any */
  bool isValid(const std::string &v) const override;
  static constexpr const char *key = "padding"; /**< unique key to access */
  using prop_tag = str_prop_tag;                /**< property type */

  /**
   * @brief compute actual padding2D from the underlying data
   *
   * @param input input dimension
   * @param kernel kernel dimension
   * @param stride stride
   * @return std::array<unsigned int, 4> list of unsigned padding
   */
  std::array<unsigned int, 4>
  compute(const TensorDim &input, const TensorDim &kernel,
          const std::array<unsigned int, 2> &strides,
          const std::array<unsigned int, 2> &dilation);
};

/**
 * @brief Padding1D property, this is used to calculate padding2D
 * @details Padding1D is saved as a string. Upon calling Padding1D::compute,
 * returns std::vector<unsigned int> which has computed padding1Ds, below
 * formats are accepted valid
 * 1. "same" (case insensitive literal string)
 * 2. "valid" (case insensitive literal string)
 * 2. "causal" (case insensitive literal string)
 * 3. "padding1d_all", eg) padding=1
 * 4. "padding1d_left, padding1d_right" eg) padding=1,1
 *
 */
class Padding1D final : public nntrainer::Property<std::string> {
public:
  /**
   * @brief Construct a new Padding1D object
   *
   */
  Padding1D(const std::string &value = "valid") :
    nntrainer::Property<std::string>(value) {} /**< default value if any */
  bool isValid(const std::string &v) const override;
  static constexpr const char *key = "padding"; /**< unique key to access */
  using prop_tag = str_prop_tag;                /**< property type */

  /**
   * @brief compute actual padding1d from the underlying data
   *
   * @param input input dimension
   * @param kernel kernel dimension
   * @param stride stride
   * @return std::array<unsigned int, 4> list of unsigned padding
   */
  std::array<unsigned int, 2> compute(const TensorDim &input,
                                      const unsigned int &kernel,
                                      const unsigned int &stride,
                                      const unsigned int &dilation);
};

/**
 * @brief InDim property, in dim is the size of vocabulary in the text data
 *
 */
class InDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "in_dim"; /**< unique key to access */
  using prop_tag = uint_prop_tag;              /**< property type */
};

/**
 * @brief OutDim property, out dim is the size of the vector space
 *  in which words will be embedded
 *
 */
class OutDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "out_dim"; /**< unique key to access */
  using prop_tag = uint_prop_tag;               /**< property type */
};

/**
 * @brief Zero idx mask property for embedding where the value of embedding
 * will be zero
 *
 */
class ZeroIdxMask : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key =
    "zero_idx_mask";              /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */
};

/**
 * @brief DropOutRate property, this defines drop out specification of layer
 *
 */
class DropOutRate : public nntrainer::Property<float> {

public:
  /**
   * @brief Construct a new DropOutRate object with a default value 0.0
   *
   */
  DropOutRate(float value = 0.0) : nntrainer::Property<float>(value) {}
  static constexpr const char *key =
    "dropout_rate";                /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */

  /**
   * @brief DropOutRate validator
   *
   * @param v float to validate
   * @retval true if it is greater or equal than 0.0
   * @retval false if it is smaller than 0.0
   */
  bool isValid(const float &v) const override;
};

/**
 * @brief TranslationFactor property, this defines how far the image is
 * translated
 *
 */
class RandomTranslate : public nntrainer::Property<float> {

public:
  static constexpr const char *key =
    "random_translate";            /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */

  /**
   * @brief setter
   *
   * @param value value to set
   */
  void set(const float &value) override;
};

/**
 * @brief Props containing file path value
 *
 */
class FilePath : public Property<std::string> {
public:
  /**
   * @brief Construct a new File Path object
   */
  FilePath() : Property<std::string>() {}

  /**
   * @brief Construct a new File Path object
   *
   * @param path path to set
   */
  FilePath(const std::string &path) { set(path); }
  static constexpr const char *key = "path"; /**< unique key to access */
  using prop_tag = str_prop_tag;             /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &v) const override;

  /**
   * @brief setter
   *
   * @param v value to set
   */
  void set(const std::string &v) override;

  /**
   * @brief return file size
   *
   * @return std::ifstream::pos_type size of the file
   */
  std::ifstream::pos_type file_size();

private:
  std::ifstream::pos_type cached_pos_size;
};

/**
 * @brief Props containing directory path value
 *
 */
class DirPath : public Property<std::string> {
public:
  /**
   * @brief Construct a new Dir Path object
   */
  DirPath() : Property<std::string>() {}

  /**
   * @brief Construct a new Dir Path object
   *
   * @param path path to set
   */
  DirPath(const std::string &path) { set(path); }
  static constexpr const char *key = "dir_path"; /**< unique key to access */
  using prop_tag = str_prop_tag;                 /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &v) const override;

  /**
   * @brief setter
   *
   * @param v value to set
   */
  void set(const std::string &v) override;
};

/**
 * @brief return sequence property, used to check
 * whether return only the last output. Return last output if true.
 *
 */
class ReturnSequences : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new ReturnSequences object
   *
   */
  ReturnSequences(bool value = false);
  static constexpr const char *key = "return_sequences";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief bidirectional property, used to make bidirectional layers
 *
 */
class Bidirectional : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Bidirectional object
   *
   */
  Bidirectional(bool value = false);
  static constexpr const char *key = "bidirectional";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief Identifiers to locate a connection which should be returned as whole
 * used in recurrent realizer
 *
 */
class AsSequence : public Property<Connection> {
public:
  static constexpr const char *key = "as_sequence";
  using prop_tag = connection_prop_tag;
};

/**
 * @brief Identifiers to locate an **input** connection which should be
 * sequenced for the connection
 *
 */
class InputIsSequence : public Name {
public:
  static constexpr const char *key = "input_is_sequence";
  using prop_tag = str_prop_tag;
};

/**
 * @brief ResetAfter property, apply reset gate after matrix multiplication if
 * this property is true. Apply before the multiplication if false. Used in gru,
 * grucell.
 *
 */
class ResetAfter : public nntrainer::Property<bool> {

public:
  /**
   * @brief Construct a new ResetAfter object with a default value true
   *
   */
  ResetAfter(bool value = true) : nntrainer::Property<bool>(value) {}
  static constexpr const char *key = "reset_after"; /**< unique key to access */
  using prop_tag = bool_prop_tag;                   /**< property type */
};

/**
 * @brief Number of class
 * @todo deprecate this
 */
class NumClass final : public nntrainer::Property<unsigned int> {
public:
  using prop_tag = uint_prop_tag;                 /**< property type */
  static constexpr const char *key = "num_class"; /**< unique key to access */

  /**
   * @copydoc nntrainer::Property<unsigned int>::isValid(const unsigned int &v);
   */
  bool isValid(const unsigned int &v) const override;
};

/**
 * @brief BasicRegularizerConstant property, this defines how much regularize
 * the weight
 *
 */
class BasicRegularizerConstant : public nntrainer::Property<float> {

public:
  /**
   * @brief Construct a new BasicRegularizerConstant object
   *
   */
  BasicRegularizerConstant(float value = 1.0f);
  static constexpr const char *key =
    "basic_regularizer_constant";  /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @param value value to check
   * @return bool true if valid
   */
  bool isValid(const float &value) const override;
};

/**
 * @brief WeightRegularizerConstant property, this defines how much regularize
 * the weight
 *
 */
class WeightRegularizerConstant final : public BasicRegularizerConstant {

public:
  /**
   * @brief Construct a new WeightRegularizerConstant object
   *
   */
  WeightRegularizerConstant(float value = 1.0f);
  static constexpr const char *key =
    "weight_regularizer_constant"; /**< unique key to access */
};

/**
 * @brief WeightDecay property, this defines how much to decay
 * the weight
 *
 */
class WeightDecay final : public BasicRegularizerConstant {

public:
  /**
   * @brief Construct a new WeightDecay object
   *
   */
  WeightDecay(float value = 0.0f);
  static constexpr const char *key =
    "weight_decay"; /**< unique key to access */
};

/**
 * @brief BiasDecay property, this defines how much regularize
 * the weight
 *
 */
class BiasDecay final : public BasicRegularizerConstant {

public:
  /**
   * @brief Construct a new BiasDecay object
   *
   */
  BiasDecay(float value = 0.0f);
  static constexpr const char *key = "bias_decay"; /**< unique key to access */
};

/**
 * @brief Output Layer name property which saves a single connection
 * (practically, std::vector<InputLayers> is used)
 *
 */
class OutputLayer : public Name {
public:
  /**
   * @brief Construct a new Output Layer object
   *
   */
  OutputLayer();

  /**
   * @brief Construct a new Output Layer object
   *
   * @param name name to set
   */
  OutputLayer(const std::string &name);
  static constexpr const char *key = "output_layers";
  using prop_tag = str_prop_tag;
};

/**
 * @brief label Layer name property which saves a single
 * connection (practically, std::vector<LabelLayers> is used)
 *
 */
class LabelLayer : public Name {
public:
  /**
   * @brief Construct LabelLayer object
   *
   */
  LabelLayer();

  /**
   * @brief Construct LabelLayer with the given name
   *
   * @param name Name for the input_layers
   */
  LabelLayer(const std::string &name);
  static constexpr const char *key = "label_layers";
  using prop_tag = str_prop_tag;
};

/******** below section is for enumerations ***************/
/**
 * @brief     Enumeration of activation function type
 */
struct ActivationTypeInfo {
  using Enum = nntrainer::ActivationType;
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::ACT_TANH,         Enum::ACT_SIGMOID, Enum::ACT_RELU,
    Enum::ACT_SWISH,        Enum::ACT_GELU,    Enum::ACT_TANH_GELU,
    Enum::ACT_SIGMOID_GELU, Enum::ACT_SOFTMAX, Enum::ACT_SOFTPLUS,
    Enum::ACT_LEAKY_RELU,   Enum::ACT_ELU,     Enum::ACT_SELU,
    Enum::ACT_MISH,         Enum::ACT_NONE,    Enum::ACT_UNKNOWN};

  static constexpr const char *EnumStr[] = {
    "tanh",      "sigmoid",      "relu",    "swish",    "gelu",
    "tanh_gelu", "sigmoid_gelu", "softmax", "softplus", "leaky_relu",
    "elu",       "selu",         "mish",    "none",     "unknown"};
};

/**
 * @brief Activation Enumeration Information
 *
 */
class Activation final
  : public EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "activation";
};

/**
 * @brief HiddenStateActivation Enumeration Information
 *
 */
class HiddenStateActivation final : public EnumProperty<ActivationTypeInfo> {
public:
  /**
   * @brief Construct a new HiddenStateActivation object with default value
   * ActivationTypeInfo::Enum::ACT_NONE
   *
   */
  HiddenStateActivation(
    ActivationTypeInfo::Enum value = ActivationTypeInfo::Enum::ACT_NONE);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "hidden_state_activation";
};

/**
 * @brief RecurrentActivation Enumeration Information
 *
 */
class RecurrentActivation final : public EnumProperty<ActivationTypeInfo> {
public:
  /**
   * @brief Construct a new RecurrentActivation object with default value
   * ActivationTypeInfo::Enum::ACT_NONE
   *
   */
  RecurrentActivation(
    ActivationTypeInfo::Enum value = ActivationTypeInfo::Enum::ACT_NONE);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "recurrent_activation";
};

/**
 * @brief     Enumeration of tensor initialization type
 */
struct InitializerInfo {
  using Enum = Initializer;
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::ZEROS,         Enum::ONES,          Enum::LECUN_NORMAL,
    Enum::LECUN_UNIFORM, Enum::XAVIER_NORMAL, Enum::XAVIER_UNIFORM,
    Enum::HE_NORMAL,     Enum::HE_UNIFORM,    Enum::NONE};

  static constexpr const char *EnumStr[] = {
    "zeros",         "ones",          "lecun_normal",
    "lecun_uniform", "xavier_normal", "xavier_uniform",
    "he_normal",     "he_uniform",    "none"};
};

/**
 * @brief WeightInitializer Initialization Enumeration Information
 *
 */
class WeightInitializer final : public EnumProperty<InitializerInfo> {
public:
  /**
   * @brief Construct a WeightInitializer object
   */
  WeightInitializer(Initializer value = Initializer::XAVIER_UNIFORM);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "weight_initializer";
};

/**
 * @brief BiasInitializer Initialization Enumeration Information
 *
 */
class BiasInitializer final : public EnumProperty<InitializerInfo> {
public:
  /**
   * @brief Construct a BiasInitializer object
   */
  BiasInitializer(Initializer value = Initializer::ZEROS);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "bias_initializer";
};

/**
 * @brief MuInitializer Initialization Enumeration Information
 *
 */
class MuInitializer final : public EnumProperty<InitializerInfo> {
public:
  /**
   * @brief Construct a MuInitializer object
   */
  MuInitializer(Initializer value = Initializer::ZEROS);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "moving_mean_initializer";
};

/**
 * @brief VarInitializer Initialization Enumeration Information
 *
 */
class VarInitializer final : public EnumProperty<InitializerInfo> {
public:
  /**
   * @brief Construct a VarInitializer object
   */
  VarInitializer(Initializer value = Initializer::ONES);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "moving_variance_initializer";
};

/**
 * @brief GammaInitializer Initialization Enumeration Information
 *
 */
class GammaInitializer final : public EnumProperty<InitializerInfo> {
public:
  /**
   * @brief Construct a GammaInitializer object
   */
  GammaInitializer(Initializer value = Initializer::ONES);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};

/**
 * @brief BetaInitializer Initialization Enumeration Information
 *
 */
class BetaInitializer final : public EnumProperty<InitializerInfo> {
public:
  /**
   * @brief Construct a BetaInitializer object
   */
  BetaInitializer(Initializer value = Initializer::ZEROS);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "beta_initializer";
};

/**
 * @brief     Enumeration of tensor regularization type
 */
struct RegularizerInfo {
  using Enum = nntrainer::WeightRegularizer;
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::L2NORM, Enum::NONE, Enum::UNKNOWN};

  static constexpr const char *EnumStr[] = {"l2norm", "none", "unknown"};
};

/**
 * @brief BasicRegularizer Regularization Enumeration Information
 *
 */
class BasicRegularizer : public EnumProperty<RegularizerInfo> {
public:
  /**
   * @brief Construct a BasicRegularizer object
   */
  BasicRegularizer(nntrainer::WeightRegularizer value);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "basic_regularizer";

  /**
   * @brief BasicRegularizer validator
   *
   * @param value nntrainer::WeightRegularizer to validate
   * @retval true if value is not nntrainer::WeightRegularizer::UNKNOWN
   * @retval false if value is nntrainer::WeightRegularizer::UNKNOWN
   */
  bool isValid(const nntrainer::WeightRegularizer &value) const override;
};

/**
 * @brief WeightRegularizer Regularization Enumeration Information
 *
 */
class WeightRegularizer final : public BasicRegularizer {
public:
  /**
   * @brief Construct a WeightRegularizer object
   */
  WeightRegularizer(
    nntrainer::WeightRegularizer value = nntrainer::WeightRegularizer::NONE);
  static constexpr const char *key = "weight_regularizer";
};

/**
 * @brief     Enumeration of upsample type
 * @todo Support torch and keras supported modes like bicubic
 */
struct UpsampleModeInfo {
  /**
   * @brief   Upsampling operation type class
   */
  enum class Interpolation { nearest, bilinear };

  using Enum = Interpolation;

  static constexpr std::initializer_list<Interpolation> EnumList = {
    Interpolation::nearest, Interpolation::bilinear};

  static constexpr const char *EnumStr[] = {"nearest", "bilinear"};
};

/**
 * @brief Upsample Type Enumeration Information
 *
 */
class UpsampleMode final : public EnumProperty<UpsampleModeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "upsample";
};

/**
 * @brief     Enumeration of pooling type
 */
struct PoolingTypeInfo {
  /**
   * @brief   Pooling operation type class
   */
  enum class Enum {
    max = 0,
    average = 1,
    global_max = 2,
    global_average = 3,
    unknown = 4
  };
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::max, Enum::average, Enum::global_max, Enum::global_average,
    Enum::unknown};

  static constexpr const char *EnumStr[] = {"max", "average", "global_max",
                                            "global_average", "unknown"};
};

/**
 * @brief Pooling Type Enumeration Information
 *
 */
class PoolingType final : public EnumProperty<PoolingTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "pooling";
};

/**
 * @brief     Enumeration of flip direction
 */
struct FlipDirectionInfo {
  enum class Enum { horizontal, vertical, horizontal_and_vertical };
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::horizontal, Enum::vertical, Enum::horizontal_and_vertical};

  static constexpr const char *EnumStr[] = {"horizontal", "vertical",
                                            "horizontal_and_vertical"};
};

/**
 * @brief FlipDirection Enumeration Information
 *
 */
class FlipDirection final : public EnumProperty<FlipDirectionInfo> {
public:
  FlipDirection(FlipDirectionInfo::Enum value =
                  FlipDirectionInfo::Enum::horizontal_and_vertical);
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "flip_direction";
};

/**
 * @brief timestep property, timestep is used to identify for which timestep
 * should the lstm/gru/rnn layer do the operation for
 *
 */
class Timestep : public Property<unsigned> {
public:
  static constexpr const char *key = "timestep"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                /**< property type */
};

/**
 * @brief maximum timestep property, timestep is used to identify for the
 * maximum time unroll possible for lstm/gru/rnn layer
 *
 */
class MaxTimestep : public PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "max_timestep";               /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */
};

/**
 * @brief generic shape property which saves a single tensor shape
 * (practically, std::array<GenericShape> is used)
 *
 * @note batch dimension is ignored with this dimension. Setting of batch must
 * be done with the model.
 *
 */
class GenericShape : public Property<TensorDim> {

public:
  static constexpr const char *key =
    "generic_shape";                   /**< unique key to access */
  using prop_tag = dimension_prop_tag; /**< property type */

  /**
   * @brief Input shape setter
   *
   * @param value value to set
   */
  void set(const TensorDim &value) override;
};

/**
 * @brief target shape property which saves a single tensor shape
 * (practically, std::array<TargetShape> is used)
 *
 */
class TargetShape : public GenericShape {

public:
  static constexpr const char *key =
    "target_shape";                    /**< unique key to access */
  using prop_tag = dimension_prop_tag; /**< property type */
};

/**
 * @brief scaled dot product property, used to check
 * whether attention layer is a kind of scaled dot product attention
 *
 */
class ScaledDotProduct : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new ScaledDotProduct object
   *
   */
  ScaledDotProduct(bool value = false);
  static constexpr const char *key = "scaled_dot_product";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief causal mask property, used in attention layer
 *
 */
class CausalMask : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new CausalMask object
   *
   */
  CausalMask(bool value = false);
  static constexpr const char *key = "causal_mask";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief Print object
 *
 */
class Print : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Print object
   *
   */
  Print(bool value = false) { set(value); }
  static constexpr const char *key = "print";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief K property, K is the size of the three projections in MoL attention
 *
 */
class MoL_K : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "MoL_K"; /**< unique key to access */
  using prop_tag = uint_prop_tag;             /**< property type */
};

/**
 * @brief NumHeads property, NumHeads is number of head in multi head attention
 *
 */
class NumHeads : public PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new NumHeads object with default value 1
   *
   */
  NumHeads(unsigned int value = 1);
  static constexpr const char *key = "num_heads"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                 /**< property type */
};

/**
 * @brief ProjectedKeyDim property, projected key dim per head in multi head
 * attention
 * @details Correspond with key_dim of tensorflow
 *
 */
class ProjectedKeyDim : public PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "projected_key_dim";          /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */
};

/**
 * @brief ProjectedValueDim property, projected value dim per head in multi head
 * attention
 * @details Correspond with value_dim of tensorflow
 *
 */
class ProjectedValueDim : public PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "projected_value_dim";        /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */
};

/**
 * @brief OutputShape property, output shape of multi head
 * attention
 * @details Correspond with output_shape of tensorflow
 *
 */
class OutputShape : public PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "output_shape";               /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */
};

/**
 * @brief Enumeration of return attention weight
 */
struct ReturnAttentionWeightInfo {
  enum class Enum { none, before, after };
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::none, Enum::before, Enum::after};

  static constexpr const char *EnumStr[] = {"none", "before", "after"};
};

/**
 * @brief ReturnAttentionWeight, return attention weight
 * @details "none" won't return attention weight.
 *          "before"/"after" will return attention weight before/after applying
 * dropout
 * @note Correspond with return_attention_scores of tensorflow and Correspond
 *       with need_weights of torch
 *
 */
class ReturnAttentionWeight : public EnumProperty<ReturnAttentionWeightInfo> {
public:
  static constexpr const char *key =
    "return_attention_weight";          /**< unique key to access */
  using prop_tag = enum_class_prop_tag; /**< property type */

  /**
   * @brief Construct a new ReturnAttentionWeight object
   *
   */
  ReturnAttentionWeight(ReturnAttentionWeightInfo::Enum value =
                          ReturnAttentionWeightInfo::Enum::none);
};

/**
 * @brief AverageAttentionWeight, average attention weight
 * @details Correspond with average_attn_weights of torch
 *
 */
class AverageAttentionWeight : public Property<bool> {
public:
  static constexpr const char *key =
    "average_attention_weight";   /**< unique key to access */
  using prop_tag = bool_prop_tag; /**< property type */
};

/**
 * @brief LoRA rank property, it is used to set rank of LoRA weight.
 * @details
 */
class LoraRank : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "lora_rank"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                 /**< property type */
};

/**
 * @brief LoRA alpha parameter
 * @details It is used to set the scaling factor of LoRA, which is calculated as
 * `scaling = alpha / rank` in the original paper.
 */
class LoraAlpha : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "lora_alpha"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                  /**< property type */
};

/**
 * @brief properties for getting the clipping value to clip the gradient by norm
 *
 */
class ClipGradByGlobalNorm : public Property<float> {
public:
  static constexpr const char *key =
    "clip_grad_by_norm";           /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @brief properties for getting the loss scale value to mixed precision
 *
 */
class LossScaleForMixed : public Property<float> {
public:
  static constexpr const char *key = "loss_scale"; /**< unique key to access */
  using prop_tag = float_prop_tag;                 /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @param value value to check
   * @retval true if it is Not 0
   * @retval false if it is 0
   */
  bool isValid(const float &value) const override;
};

/**
 * @brief Learning Rate props
 *
 */
class LearningRate : public Property<float> {
public:
  static constexpr const char *key =
    "learning_rate";               /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @brief Max Learning Rate props
 *
 */
class MaxLearningRate : public Property<float> {
public:
  static constexpr const char *key =
    "max_learning_rate";           /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @brief Min Learning Rate props
 *
 */
class MinLearningRate : public Property<float> {
public:
  static constexpr const char *key =
    "min_learning_rate";           /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @brief Iteration props
 *
 */
class Iteration : public Property<unsigned int> {
public:
  static constexpr const char *key = "iteration"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                 /**< property type */
};

/**
 * @brief Decay rate property
 *
 */
class DecayRate : public Property<float> {
public:
  static constexpr const char *key = "decay_rate"; /**< unique key to access */
  using prop_tag = float_prop_tag;                 /**< property type */
};

/**
 * @brief decay steps property
 *
 */
class DecaySteps : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "decay_steps"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

/**
 * @brief User data props
 *
 */
class PropsUserData final : public Property<void *> {
public:
  PropsUserData(void *user_data);
  static constexpr const char *key = "user_data";
  using prop_tag = ptr_prop_tag;
};

/**
 * @brief Enumeration of Tensor Life Spacn
 */
struct TensorLifeInfo {
  using Enum = nntrainer::TensorLifespan;
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::UNMANAGED,
    Enum::FORWARD_FUNC_LIFESPAN,
    Enum::CALC_DERIV_LIFESPAN,
    Enum::CALC_GRAD_LIFESPAN,
    Enum::CALC_AGRAD_LIFESPAN,
    Enum::CALC_GRAD_DERIV_LIFESPAN,
    Enum::CALC_GRAD_DERIV_AGRAD_LIFESPAN,
    Enum::FORWARD_GRAD_LIFESPAN,
    Enum::FORWARD_GRAD_AGRAD_LIFESPAN,
    Enum::FORWARD_DERIV_LIFESPAN,
    Enum::BACKWARD_FUNC_LIFESPAN,
    Enum::CALC_GRAD_DERIV_AGRAD_LIFESPAN,
    Enum::ITERATION_LIFESPAN,
    Enum::EPOCH_LIFESPAN,
    Enum::FORWARD_INFER_LIFESPAN,
    Enum::MAX_LIFESPAN};

  static constexpr const char *EnumStr[] = {"unmanaged",
                                            "forward",
                                            "deriv",
                                            "grad",
                                            "agrad",
                                            "grad_deriv",
                                            "grad_deriv_agrad",
                                            "forward_grad",
                                            "forward_grad_agrad",
                                            "forward_deriv",
                                            "backward",
                                            "grad_deriv_agrad",
                                            "iteration",
                                            "epoch",
                                            "forward_infer",
                                            "max"};
};

/**
 * @brief TensorLifeSpacn, return TensorLifespan
 *
 */
class TensorLife : public EnumProperty<TensorLifeInfo> {
public:
  static constexpr const char *key = "tensor_life"; /**< unique key to access */
  using prop_tag = enum_class_prop_tag;             /**< property type */

  /**
   * @brief Construct a new ReturnAttentionWeight object
   *
   */
  TensorLife(TensorLifeInfo::Enum value = TensorLifeInfo::Enum::MAX_LIFESPAN) {
    set(value);
  };
};

/**
 * @brief WeightName,  WeightName property
 */
class WeightName : public Name {
public:
  static constexpr const char *key = "weight_name";
  using prop_tag = str_prop_tag;
};

/**
 * @brief TensorName,  TensorName property
 */
class TensorName : public Name {
public:
  static constexpr const char *key = "tensor_name";
  using prop_tag = str_prop_tag;
};

} // namespace props
} // namespace nntrainer

#endif // __COMMON_PROPERTIES_H__
