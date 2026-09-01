// SPDX-License-Identifier: Apache-2.0
/**
 * @file	quantizer.cpp
 * @date	10 December 2024
 * @brief	This defines quantizers for different types of quantization schemes
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <algorithm>
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
    quantize_q4_0(src, tmp.data(), N, K, nullptr);
    break;
  default:
    break;
  }

  // For Q4_Kx8 and Q4_0, repack into the optimized layout
  if (scheme_ == QScheme::Q4_Kx8) {
    repack_q4_K(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else if (scheme_ == QScheme::Q4_0) {
    repack_q4_0(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else {
    // Q6_K: copy directly (no repacking needed)
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
    quantize_q4_0(src, tmp.data(), N, K, nullptr);
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::quantize] Unsupported QScheme.");
  }

  if (scheme_ == QScheme::Q4_Kx8) {
    repack_q4_K(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else if (scheme_ == QScheme::Q4_0) {
    repack_q4_0(output.getData<uint8_t>(), tmp.data(), out_size, N, K);
  } else {
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
    dequantize_row_q4_0(tmp.data(), output.getData(), total_elems);
    break;
  default:
    throw std::invalid_argument(
      "[GgmlQuantizer::dequantize] Unsupported QScheme.");
  }

  return output;
}

QScheme GgmlQuantizer::qscheme() const { return scheme_; }

/**
 * @brief QS4CXQuantizer class
 */
std::unique_ptr<Quantizer> QS4CXQuantizer::create() {
  return std::make_unique<QS4CXQuantizer>();
}

Tensor QS4CXQuantizer::quantize(const Tensor &input, Tdatatype qtype) {
  NNTR_THROW_IF(qtype != Tdatatype::QS4CX, std::invalid_argument)
    << "[QS4CXQuantizer::quantize] Output data type must be QS4CX.";

  TensorDim dim = input.getDim();
  dim.setDataType(Tdatatype::QS4CX);

  // QS4CX_Tensor sizes itself from the dimension, so the record and its scale
  // section are allocated by the constructor.
  Tensor output(dim, true, Initializer::NONE, "", QScheme::QS4CX);

  quantize(input, output, nullptr, nullptr);

  return output;
}

Tensor &QS4CXQuantizer::quantize(const Tensor &input, Tensor &output,
                                 float *scales, unsigned int *zero_points) {
  /// @note zero_points is deliberately unread: QS4CX is symmetric and the +8
  /// bias is part of the nibble encoding, so there is no per-channel zero
  /// point to honour. @see the @a zero_points note on the declaration.
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[QS4CXQuantizer::quantize] Input tensor must be FP32.";

  NNTR_THROW_IF(output.empty(), std::invalid_argument)
    << "[QS4CXQuantizer::quantize] Cannot quantize to an empty tensor.";

  NNTR_THROW_IF(output.getDataType() != Tdatatype::QS4CX, std::invalid_argument)
    << "[QS4CXQuantizer::quantize] Output tensor must be QS4CX.";

  const TensorDim dim = input.getDim();

  /// @note Only height and width are compared here. batch and channel need no
  /// check because a QS4CX output cannot carry any other value: the
  /// QS4CX_Tensor constructor rejects batch != 1 or channel != 1 before this
  /// runs, so the two dimensions are 1 on both sides by construction.
  NNTR_THROW_IF(dim.height() != output.height() ||
                  dim.width() != output.width(),
                std::invalid_argument)
    << "[QS4CXQuantizer::quantize] Tensor shape does not match.";

  const size_t K = dim.height();
  const size_t N = dim.width();

  // One scale per output channel means the kernel walks the weight channel by
  // channel, so hand it the (N, K) view. batch()/channel() are 1 here - the
  // QS4CX_Tensor constructor rejects anything else.
  Tensor input_t = input.transpose("0:2:1");

  float *out_scales = output.getScale<float>();
  quant_qs4cx_f32(N, K, input_t.getData(), output.getData<uint8_t>(),
                  out_scales, true);

  if (scales != nullptr)
    std::copy(out_scales, out_scales + N, scales);

  return output;
}

Tensor QS4CXQuantizer::dequantize(const Tensor &input, Tdatatype dtype) {
  NNTR_THROW_IF(dtype != Tdatatype::FP32, std::invalid_argument)
    << "[QS4CXQuantizer::dequantize] Output data type must be FP32.";

  NNTR_THROW_IF(input.getDataType() != Tdatatype::QS4CX, std::invalid_argument)
    << "[QS4CXQuantizer::dequantize] Input tensor must be QS4CX.";

  const TensorDim dim = input.getDim();
  const size_t K = dim.height();
  const size_t N = dim.width();

  // The kernel unpacks into the (N, K) layout the record is stored in; put it
  // back to the (K, N) shape the caller quantized.
  Tensor output_t(dim.batch(), dim.channel(), static_cast<unsigned int>(N),
                  static_cast<unsigned int>(K),
                  {Tformat::NCHW, Tdatatype::FP32});

  dequant_qs4cx_f32(N, K, input.getData<uint8_t>(), input.getScale<float>(),
                    output_t.getData(), true);

  return output_t.transpose("0:2:1");
}

QScheme QS4CXQuantizer::qscheme() const { return QScheme::QS4CX; }

} // namespace nntrainer
