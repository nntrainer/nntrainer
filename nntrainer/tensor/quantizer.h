// SPDX-License-Identifier: Apache-2.0
/**
 * @file	quantizer.h
 * @date	10 December 2024
 * @brief	This defines quantizers for different types of quantization schemes
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __QUANTIZER_H__
#define __QUANTIZER_H__
#ifdef __cplusplus

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <tensor_dim.h>

namespace nntrainer {

class Tensor;

/**
 * @brief defines the quantization scheme
 * @details NNTrainer provides basic quantization schemes (e.g., Per tensor
 * affine quantization). Various quantization schemes will be continuously
 * updated. If you would like to use a different quantization technique, please
 * select a custom quantizer scheme.
 */
enum class QScheme : uint16_t {
  /** predefined quantizer */
  PER_TENSOR_AFFINE = 0x00,
  PER_CHANNEL_AFFINE = 0x01,
  BINARY_CODE_BASED = 0x02,
  Q4_Kx8 = 0x03,
  Q6_K = 0x4,
  Q4_0 = 0x5,
  Q1_0 = 0x6,
  /** this is for custom use */
  CUSTOM_QUANTIZER_01 = 0x10,
  CUSTOM_QUANTIZER_02 = 0x11,
  CUSTOM_QUANTIZER_03 = 0x12,
  CUSTOM_QUANTIZER_04 = 0x13,
  CUSTOM_QUANTIZER_05 = 0x14,
  CUSTOM_QUANTIZER_06 = 0x15,
};

/**
 * @class Quantizer class
 * @brief Quantizer class is a base class for all quantizers.
 * @note A custom quantizer must inherit this class and implement virtual
 * functions.
 */
class Quantizer {
private:
  static inline std::unordered_map<QScheme, Quantizer *>
    custom_quantizers; /** Hash table that holds empty instances of the custom
                          quantizers */

protected:
  long int quant_min = 0;
  long int quant_max = 0;

  /**
   * @brief Register the user defined quantizer class
   *
   * @param qscheme Quantization scheme (use CUSTOM_QUANTIZER_#)
   * @param quantizer quantizer class to register
   *
   * @note This function registers the custom quantizer class. User defined
   * derived class must be registered with this function.
   */
  static void registerQuantizer(QScheme qscheme, Quantizer &quantizer) {
    custom_quantizers.insert(std::make_pair(qscheme, &quantizer));
  }

  /**
   * @brief Calculate the quantization parameters
   *
   * @note This will be used to determine the quantization parameters.
   * QParams must be determined before quantization.
   *
   * @param input input tensor
   * @param qtype quantized data type
   */
  virtual void calculateQParams(const Tensor &input,
                                ml::train::TensorDim::DataType qtype) = 0;

  /**
   * @brief Calculate the minimum & maximum value
   * @param qtype quantized data type
   */
  void calculateMinMaxValue(ml::train::TensorDim::DataType qtype);

public:
  /**
   * @brief Basic Constructor of a Quantizer
   */
  Quantizer() = default;

  /**
   * @brief Basic Destructor of a Quantizer
   */
  virtual ~Quantizer() = default;

  /**
   * @brief Get the Registered Quantizer object
   *
   * @param qscheme Quantization scheme
   * @return Quantizer* registered quantizer object
   */
  static Quantizer *getRegisteredQuantizer(QScheme qscheme) {
    if (custom_quantizers.find(qscheme) == custom_quantizers.end()) {
      throw std::invalid_argument("requested quantizer is not registered.");
    }
    return custom_quantizers.at(qscheme);
  }

  /** Derived classes must implement the following functions */
  /**
   * @brief Create a new object of itself
   *
   * @return std::unique_ptr<Quantizer>
   */
  virtual std::unique_ptr<Quantizer> create() = 0;

  /**
   * @brief Quantize a tensor into a quantized tensor.
   * @param[in] input Floating point tensor to quantize
   * @return Tensor quantized tensor
   */
  virtual Tensor quantize(const Tensor &input,
                          ml::train::TensorDim::DataType qtype) = 0;

  /**
   * @brief Quantize a tensor into a quantized tensor.
   * @param[in] input Floating point tensor to quantize
   * @param[out] output Quantized tensor
   * @param[in] scales float scale factors
   * @param[in] zero_points unsigned int zero points
   * @return Tensor quantized tensor
   */
  virtual Tensor &quantize(const Tensor &input, Tensor &output, float *scales,
                           unsigned int *zero_points = nullptr) = 0;

  /**
   * @brief Dequantize a quantized tensor into a tensor.
   * @param[in] input Quantized tensor to dequantize
   * @return Tensor dequantized tensor
   */
  virtual Tensor dequantize(const Tensor &input,
                            ml::train::TensorDim::DataType qtype) = 0;

  /**
   * @brief Get quantization Scheme type.
   * @return Quantization scheme
   */
  virtual QScheme qscheme() const = 0;
};

/**
 * @class UniformQuantizer class
 * @brief UniformQuantizer class serves as the parent class for various types of
 * uniform quantizers.
 */
class UniformQuantizer : public Quantizer {
public:
  UniformQuantizer() : Quantizer() {}
};

/**
 * @class NonUniformQuantizer class
 * @brief NonUniformQuantizer class serves as the parent class for various types
 * of non-uniform quantizers.
 */
class NonUniformQuantizer : public Quantizer {
public:
  NonUniformQuantizer() : Quantizer() {}
};

/**
 * @class PerTensorAffineQuantizer class
 * @brief PerTensorAffineQuantizer class uses affine quantization scheme.
 *
 * Quantization: x_q = clip(round(x / scale + zero_point), min, max)
 * Dequantization: x = scale * (x_q - zero_point)
 *
 * @note Single scale and zero point values are used for the entire tensor.
 */
class PerTensorAffineQuantizer : public UniformQuantizer {
public:
  /**
   * @brief Basic Constructor of a PerTensorAffineQuantizer
   */
  PerTensorAffineQuantizer() : UniformQuantizer(), scale(1) {}

  /**
   * @copydoc Quantizer::create()
   */
  std::unique_ptr<Quantizer> create() override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input)
   */
  Tensor quantize(const Tensor &input,
                  ml::train::TensorDim::DataType qtype) override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input, Tensor &output, float
   * *scales, unsigned int *zero_points)
   */
  Tensor &quantize(const Tensor &input, Tensor &output, float *scales,
                   unsigned int *zero_points = nullptr) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input,
                    ml::train::TensorDim::DataType dtype) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;

private:
  float scale;
  unsigned int zero_point = 0;

  /**
   * @copydoc Quantizer::calculateQParams(const Tensor &input,
   * ml::train::TensorDim::DataType qtype)
   */
  void calculateQParams(const Tensor &input,
                        ml::train::TensorDim::DataType qtype) override;
};

/**
 * @class PerChannelAffineQuantizer class
 * @brief PerChannelAffineQuantizer class uses affine quantization scheme.
 *
 * @note PerChannelAffineQuantizer is similar to PerTensorAffineQuantizer, but
 * it has separate scale and zero_point parameters for each channel. This allows
 * for more precise quantization of different channels within the same tensor.
 *
 */
class PerChannelAffineQuantizer : public UniformQuantizer {
public:
  /**
   * @brief Basic Constructor of a PerChannelAffineQuantizer
   */
  PerChannelAffineQuantizer() : UniformQuantizer() {}

  /**
   * @copydoc Quantizer::create()
   */
  std::unique_ptr<Quantizer> create() override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input)
   */
  Tensor quantize(const Tensor &input,
                  ml::train::TensorDim::DataType qtype) override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input, Tensor &output, float
   * *scales, unsigned int *zero_points)
   */
  Tensor &quantize(const Tensor &input, Tensor &output, float *scales,
                   unsigned int *zero_points = nullptr) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input,
                    ml::train::TensorDim::DataType dtype) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;

private:
  /**
   * @copydoc Quantizer::calculateQParams(const Tensor &input,
   * ml::train::TensorDim::DataType qtype)
   */
  void calculateQParams(const Tensor &input,
                        ml::train::TensorDim::DataType qtype) override {}
};

/**
 * @class BinaryCodeBasedQuantizer class
 * @brief BinaryCodeBasedQuantizer class uses Binary-code-based quantization
 * (BCQ) scheme.
 *
 */
class BinaryCodeBasedQuantizer : public NonUniformQuantizer {
public:
  /**
   * @brief Basic Constructor of a BinaryCodeBasedQuantizer
   */
  BinaryCodeBasedQuantizer() : NonUniformQuantizer() {}

  /**
   * @copydoc Quantizer::create()
   */
  std::unique_ptr<Quantizer> create() override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input)
   */
  Tensor quantize(const Tensor &input,
                  ml::train::TensorDim::DataType qtype) override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input, Tensor &output, float
   * *scales, unsigned int *zero_points)
   */
  Tensor &quantize(const Tensor &input, Tensor &output, float *scales,
                   unsigned int *zero_points = nullptr) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input,
                    ml::train::TensorDim::DataType dtype) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;

private:
  // float *scales = nullptr;
  // int *zero_points = nullptr;
  // long int quant_min = 0;
  // long int quant_max = 0;

  /**
   * @copydoc Quantizer::calculateQParams(const Tensor &input,
   * ml::train::TensorDim::DataType qtype)
   */
  void calculateQParams(const Tensor &input,
                        ml::train::TensorDim::DataType qtype) override {}
};

/**
 * @class GgmlQuantizer class
 * @brief GgmlQuantizer class supports Q4_K, Q6_K, Q4_0 type.
 */
class GgmlQuantizer : public NonUniformQuantizer {
public:
  /**
   * @brief Constructor of a GgmlQuantizer
   * @param scheme Quantization scheme (Q4_Kx8, Q6_K, or Q4_0)
   */
  GgmlQuantizer(QScheme scheme = QScheme::Q4_Kx8) :
    NonUniformQuantizer(), scheme_(scheme) {}

  /**
   * @copydoc Quantizer::create
   */
  std::unique_ptr<Quantizer> create() override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input,
   * ml::train::TensorDim::DataType qtype)
   */
  Tensor quantize(const Tensor &input,
                  ml::train::TensorDim::DataType qtype) override;

  /**
   * @copydoc Quantizer::quantize(const Tensor &input, Tensor &output, float
   * *scales, unsigned int *zero_points)
   */
  Tensor &quantize(const Tensor &input, Tensor &output, float *scales,
                   unsigned int *zero_points = nullptr) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input,
                    ml::train::TensorDim::DataType dtype) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;

private:
  QScheme scheme_;

  /**
   * @copydoc Quantizer::calculateQParams(const Tensor &input,
   * ml::train::TensorDim::DataType qtype)
   */
  void calculateQParams(const Tensor &input,
                        ml::train::TensorDim::DataType qtype) override {}
};

/**
 * @brief Quantization class to create a quantizer
 *
 * @details The quantization class is a creator class to create a predefined
 * quantization and a user-defined quantizer.  Please check QScheme to find out
 * about the predefined quantizers.
 *
 * If a preferred quantization scheme is not provided, create a new class that
 * inherits the Quantizer class, select the quantization scheme
 * CUSTOM_QUANTIZER_#, register it using registerQuantizer(), and then use it.
 */
class Quantization {
public:
  /**
   * @brief Create a Quantizer object
   *
   * @param qscheme quantization scheme
   * @return std::unique_ptr<Quantizer> quantizer object
   */
  static std::unique_ptr<Quantizer> createQuantizer(QScheme qscheme) {
    switch (qscheme) {
    case QScheme::PER_TENSOR_AFFINE:
      return std::make_unique<PerTensorAffineQuantizer>();
      break;
    case QScheme::PER_CHANNEL_AFFINE:
      return std::make_unique<PerChannelAffineQuantizer>();
      break;
    case QScheme::BINARY_CODE_BASED:
      return std::make_unique<BinaryCodeBasedQuantizer>();
      break;
    case QScheme::Q4_Kx8:
    case QScheme::Q6_K:
    case QScheme::Q4_0:
    case QScheme::Q1_0:
      return std::make_unique<GgmlQuantizer>(qscheme);
      break;
    default:
      return Quantizer::getRegisteredQuantizer(qscheme)->create();
      break;
    }
  }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QUANTIZER_H__ */
