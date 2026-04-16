// SPDX-License-Identifier: Apache-2.0
/**
 * @file	quantizer.cpp
 * @date	10 December 2024
 * @brief	This defines quantizers for different types of quantization schemes
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <compute_ops.h>
#include <cpu_backend.h>
#include <math.h>
#include <quantizer.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief Helper function for clipping
 *
 * @tparam T data type
 * @param val value to clip
 * @param lower lower bound
 * @param upper upper bound
 * @return T cliped data
 */
template <typename T> T clip(const T &val, const T &lower, const T &upper) {
  return std::max(lower, std::min(val, upper));
}

void Quantizer::calculateMinMaxValue(Tdatatype qtype) {
  unsigned int N;

  if (qtype == Tdatatype::QINT16 || qtype == Tdatatype::UINT16) {
    N = 16;
  } else if (qtype == Tdatatype::QINT8 || qtype == Tdatatype::UINT8) {
    N = 8;
  } else if (qtype == Tdatatype::QINT4 || qtype == Tdatatype::UINT4) {
    N = 4;
  } else {
    throw std::invalid_argument("[Quantizer] Unsupported data type error.");
  }

  // define minimum and maximum valude representable by the type
  quant_max = (qtype == Tdatatype::UINT16 || qtype == Tdatatype::UINT8 ||
               qtype == Tdatatype::UINT4)
                ? static_cast<long>(std::pow(2, N) - 1)
                : static_cast<long>(std::pow(2, N - 1) - 1);
  quant_min = (qtype == Tdatatype::UINT16 || qtype == Tdatatype::UINT8 ||
               qtype == Tdatatype::UINT4)
                ? 0
                : static_cast<long>(-std::pow(2, N - 1));
}

/**
 * @brief PerTensorAffineQuantizer class
 */
std::unique_ptr<Quantizer> PerTensorAffineQuantizer::create() {
  return std::make_unique<PerTensorAffineQuantizer>();
}

void PerTensorAffineQuantizer::calculateQParams(const Tensor &input,
                                                Tdatatype qtype) {
  float max_val = input.max_abs();
  scale = max_val / ((quant_max - quant_min) / 2.0f);
  scale = std::max(scale, std::numeric_limits<float>::epsilon());

  if (qtype == Tdatatype::UINT4) {
    zero_point =
      (unsigned int)(std::round(scale * input.minValue()) + std::pow(2, 3));
  } else if (qtype == Tdatatype::UINT8) {
    zero_point =
      (unsigned int)(std::round(scale * input.minValue()) + std::pow(2, 7));
  } else if (qtype == Tdatatype::UINT16) {
    zero_point =
      (unsigned int)(std::round(scale * input.minValue()) + std::pow(2, 15));
  } else {
    zero_point = 0;
  }
}

Tensor PerTensorAffineQuantizer::quantize(const Tensor &input,
                                          Tdatatype qtype) {
  // 1. Calculate quantization parameters
  calculateMinMaxValue(qtype);
  calculateQParams(input, qtype);

  // 2. Create output tensor with same dimension but different data type
  TensorDim dim = input.getDim();
  dim.setDataType(qtype);
  Tensor output(dim);

  // 3. perform quantization
  quantize(input, output, &scale, &zero_point);

  return output;
}

Tensor &PerTensorAffineQuantizer::quantize(const Tensor &input, Tensor &output,
                                           float *scales,
                                           unsigned int *zero_points) {
  // Currently only full precision floating point is supported. FP16 is NYI
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[Quantizer::quantize] Tensor data type is not floating point.";

  // Check if output tensor is valid
  NNTR_THROW_IF(output.empty(), std::invalid_argument)
    << "[Quantizer::quantize] Cannot quantize to an empty tensor.";

  NNTR_THROW_IF(output.getDataType() == Tdatatype::FP32, std::invalid_argument)
    << "[Quantizer::quantize] Cannot quantize to full precision floating "
       "point.";

  NNTR_THROW_IF(scales == nullptr || std::fpclassify(*scales) == FP_ZERO,
                std::invalid_argument)
    << "[Quantizer::quantize] Output scale factor is invalid.";

  NNTR_THROW_IF(input.size() != output.size(), std::invalid_argument)
    << "[Quantizer::quantize] Tensor size does not match.";

  if (output.getDataType() == Tdatatype::UINT4 ||
      output.getDataType() == Tdatatype::UINT8 ||
      output.getDataType() == Tdatatype::UINT16) {
    NNTR_THROW_IF(zero_points == nullptr, std::invalid_argument)
      << "[Quantizer::quantize] Output zero point is invalid.";
  }

  calculateMinMaxValue(output.getDataType());

  long int val;

  /// @todo this is a naive impl. need optimization
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          val = std::lround(input.getValue(b, c, h, w) / *scales);

          if (output.getDataType() == Tdatatype::UINT4 ||
              output.getDataType() == Tdatatype::UINT8 ||
              output.getDataType() == Tdatatype::UINT16) {
            val += *zero_points;
          }

          output.setValue(b, c, h, w,
                          clip<float>(val, static_cast<float>(quant_min),
                                      static_cast<float>(quant_max)));
        }
      }
    }
  }
  *output.getScale<float>() = *scales;

  if (output.getDataType() == Tdatatype::UINT4 ||
      output.getDataType() == Tdatatype::UINT8 ||
      output.getDataType() == Tdatatype::UINT16) {
    *output.getZeroPoint() = *zero_points;
  }

  return output;
}

Tensor PerTensorAffineQuantizer::dequantize(const Tensor &input,
                                            Tdatatype dtype) {
  Tensor output = input.clone(dtype);
  if (output.getDataType() == Tdatatype::UINT4 ||
      input.getDataType() == Tdatatype::UINT8 ||
      input.getDataType() == Tdatatype::UINT16) {
    output.subtract_i(static_cast<float>(*input.getZeroPoint()));
  }

  output.multiply_i(*input.getScale<float>());

  return output;
}

QScheme PerTensorAffineQuantizer::qscheme() const {
  return QScheme::PER_TENSOR_AFFINE;
}

/**
 * @brief PerChannelAffineQuantizer class
 */
std::unique_ptr<Quantizer> PerChannelAffineQuantizer::create() {
  return std::make_unique<PerChannelAffineQuantizer>();
}

Tensor PerChannelAffineQuantizer::quantize(const Tensor &input,
                                           Tdatatype qtype) {
  /// @todo NYI
  return input;
}

Tensor &PerChannelAffineQuantizer::quantize(const Tensor &input, Tensor &output,
                                            float *scales,
                                            unsigned int *zero_points) {
  /// @todo NYI
  return output;
}

Tensor PerChannelAffineQuantizer::dequantize(const Tensor &input,
                                             Tdatatype dtype) {
  /// @todo NYI
  return input;
}

QScheme PerChannelAffineQuantizer::qscheme() const {
  return QScheme::PER_CHANNEL_AFFINE;
}

/**
 * @brief BinaryCodeBasedQuantizer class
 */
std::unique_ptr<Quantizer> BinaryCodeBasedQuantizer::create() {
  return std::make_unique<BinaryCodeBasedQuantizer>();
}

Tensor BinaryCodeBasedQuantizer::quantize(const Tensor &input,
                                          Tdatatype qtype) {
  /// @todo NYI
  return input;
}

Tensor &BinaryCodeBasedQuantizer::quantize(const Tensor &input, Tensor &output,
                                           float *scales,
                                           unsigned int *zero_points) {
  /// @todo NYI
  return output;
}

Tensor BinaryCodeBasedQuantizer::dequantize(const Tensor &input,
                                            Tdatatype dtype) {
  /// @todo NYI
  return input;
}

QScheme BinaryCodeBasedQuantizer::qscheme() const {
  return QScheme::BINARY_CODE_BASED;
}

/**
 * @brief GgmlQuantizer class
 */
std::unique_ptr<Quantizer> GgmlQuantizer::create() {
  return std::make_unique<GgmlQuantizer>(scheme_);
}

Tensor GgmlQuantizer::quantize(const Tensor &input, Tdatatype qtype) {
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[GgmlQuantizer::quantize] Input tensor must be FP32.";

  TensorDim dim = input.getDim();
  unsigned int K = dim.height();
  unsigned int N = dim.width();

  Tdatatype out_dtype;
  switch (scheme_) {
  case QScheme::Q4_Kx8:
    out_dtype = Tdatatype::Q4_K;
    break;
  case QScheme::Q6_K:
    out_dtype = Tdatatype::Q6_K;
    break;
  case QScheme::Q4_0:
    out_dtype = Tdatatype::Q4_0;
    break;
  case QScheme::Q1_0:
    out_dtype = Tdatatype::Q1_0;
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::quantize] Unsupported QScheme.");
  }

  // For GGML quantization, we need to transpose the weight first (row-major
  // to column-major style), then quantize per row of the transposed matrix.
  // The quantization functions expect data as (nrow, n_per_row).
  Tensor W_t = input.transpose("0:2:1");
  const float *src = W_t.getData<float>();

  // Create output quantized tensor
  Tensor output({dim.batch(), dim.channel(), K, N, Tformat::NCHW, out_dtype},
                true, nntrainer::Initializer::NONE, "", scheme_);

  // Quantize into a temporary buffer first
  size_t out_size = output.size();
  std::vector<char> tmp(out_size);

  switch (scheme_) {
  case QScheme::Q4_Kx8:
    quantize_q4_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q6_K:
    quantize_q6_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q4_0:
    getComputeOps()->quantize_q4_0(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q1_0:
    quantize_q1_0(src, tmp.data(), N, K, nullptr);
    break;
  default:
    break;
  }

  // For Q4_Kx8 and Q4_0, repack into the optimized layout
  if (scheme_ == QScheme::Q4_Kx8) {
    repack_q4_K(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else if (scheme_ == QScheme::Q4_0) {
    getComputeOps()->repack_q4_0(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else {
    // Q6_K, Q1_0: copy directly (no repacking needed)
    memcpy(output.getData<uint8_t>(), tmp.data(), out_size);
  }

  return output;
}

Tensor &GgmlQuantizer::quantize(const Tensor &input, Tensor &output,
                                float *scales, unsigned int *zero_points) {
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[GgmlQuantizer::quantize] Input tensor must be FP32.";

  NNTR_THROW_IF(output.empty(), std::invalid_argument)
    << "[GgmlQuantizer::quantize] Output tensor is empty.";

  NNTR_THROW_IF(output.q_scheme() != scheme_, std::invalid_argument)
    << "[GgmlQuantizer::quantize] Output tensor's quantization scheme does not "
       "match.";

  TensorDim dim = input.getDim();
  unsigned int K = dim.height();
  unsigned int N = dim.width();

  Tensor W_t = input.transpose("0:2:1");
  const float *src = W_t.getData<float>();

  size_t out_size = output.size();
  std::vector<char> tmp(out_size);

  switch (scheme_) {
  case QScheme::Q4_Kx8:
    quantize_q4_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q6_K:
    quantize_q6_K(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q4_0:
    getComputeOps()->quantize_q4_0(src, tmp.data(), N, K, nullptr);
    break;
  case QScheme::Q1_0:
    quantize_q1_0(src, tmp.data(), N, K, nullptr);
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::quantize] Unsupported QScheme.");
  }

  if (scheme_ == QScheme::Q4_Kx8) {
    repack_q4_K(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else if (scheme_ == QScheme::Q4_0) {
    getComputeOps()->repack_q4_0(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else {
    // Q6_K, Q1_0: copy directly (no repacking needed)
    memcpy(output.getData<uint8_t>(), tmp.data(), out_size);
  }

  return output;
}

Tensor GgmlQuantizer::dequantize(const Tensor &input, Tdatatype dtype) {
  NNTR_THROW_IF(dtype != Tdatatype::FP32, std::invalid_argument)
    << "[GgmlQuantizer::dequantize] Output dtype must be FP32.";

  TensorDim dim = input.getDim();
  unsigned int K = dim.height();
  unsigned int N = dim.width();
  unsigned int total_elems = K * N;

  Tensor output(dim.batch(), dim.channel(), K, N,
                {Tformat::NCHW, Tdatatype::FP32});
  size_t data_size = input.size();
  std::vector<char> tmp(input.size());

  const void *src = input.getData<uint8_t>();

  switch (scheme_) {
  case QScheme::Q4_Kx8:
    ///@todo unpack should be supported to fully support dequtize
    // dequantize_row_q4_K(src, output.getData(), total_elems);
    throw std::invalid_argument(
      "[GgmlQuantizer::dequantize] Q4_Kx8 is not supported yet.");
    break;
  case QScheme::Q6_K:
    dequantize_row_q6_K(src, output.getData(), total_elems);
    break;
  case QScheme::Q4_0:
    unpack_q4_0(src, tmp.data(), data_size, N, K);
    getComputeOps()->dequantize_row_q4_0(tmp.data(), output.getData(), total_elems);
    break;
  case QScheme::Q1_0:
    dequantize_row_q1_0(src, output.getData(), total_elems);
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::dequantize] Unsupported QScheme.");
  }

  return output;
}

QScheme GgmlQuantizer::qscheme() const { return scheme_; }

} // namespace nntrainer
