// SPDX-License-Identifier: Apache-2.0
/**
 * @file	char_tensor.cpp
 * @date	02 April 2024
 * @brief	This is CharTensor class for 8-bit integer calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <char_tensor.h>
#include <cpu_backend.h>
#include <tensor.h>

namespace nntrainer {

CharTensor::CharTensor(std::string name_, Tformat fm, QScheme qscheme_) :
  TensorBase(name_, fm, Tdatatype::QINT8), qscheme(qscheme_) {}

CharTensor::CharTensor(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name, QScheme qscheme_) :
  TensorBase(d, alloc_now, init, name), qscheme(qscheme_) {
  if (alloc_now)
    allocate();
}

CharTensor::CharTensor(const TensorDim &d, const void *buf, QScheme qscheme_) :
  CharTensor(d, true, Initializer::NONE, "", qscheme_) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

CharTensor::CharTensor(
  std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
  std::vector<float> const &scales, Tformat fm, QScheme qscheme_) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize CharTensor from empty vector");
  }

  dim.setTensorDim(0, d.size());
  if (fm == Tformat::NCHW) {
    dim.setTensorDim(1, d[0].size());
    dim.setTensorDim(2, d[0][0].size());
    dim.setTensorDim(3, d[0][0][0].size());
  } else {
    dim.setTensorDim(2, d[0].size());
    dim.setTensorDim(3, d[0][0].size());
    dim.setTensorDim(1, d[0][0][0].size());
  }

  dim.setTensorType({fm, Tdatatype::QINT8});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;
  qscheme = qscheme_;

  NNTR_THROW_IF(scales.size() != scale_size(), std::invalid_argument)
    << "invalid scale factor size " << scales.size();

  MemoryData *mem_data = new MemoryData(
    (void *)(new int8_t[dim.getDataLen() + sizeof(float) * scale_size()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<int8_t>();
    delete mem_data;
  });

  offset = 0;

  // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
  // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
  // dim[1] == height, dim[2] == width, dim[3] == channel
  if (fm == Tformat::NCHW) {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < channel(); ++j)
        for (unsigned int k = 0; k < height(); ++k)
          for (unsigned int l = 0; l < width(); ++l)
            this->setValue(i, j, k, l, d[i][j][k][l]);
  } else {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < height(); ++j)
        for (unsigned int k = 0; k < width(); ++k)
          for (unsigned int l = 0; l < channel(); ++l)
            this->setValue(i, l, j, k, d[i][j][k][l]);
  }

  // copy scale factors
  scopy(scale_size(), scales.data(), 1, (float *)getScale(), 1);
}

bool CharTensor::operator==(const CharTensor &rhs) const {
  if (qscheme != rhs.qscheme)
    return false;

  // compare quantized data
  const int8_t *_data = (int8_t *)getData();
  const int8_t *_rdata = (int8_t *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    if (_data[i] != _rdata[i])
      return false;
  }

  // compare scale factors
  const float *_scales = (float *)getScale();
  const float *_rscales = (float *)rhs.getScale();
  for (size_t i = 0; i < scale_size(); ++i) {
    if (std::fabs(_scales[i] - _rscales[i]) > 1e-5)
      return false;
  }

  return true;
}

void CharTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData(
      (void *)(new int8_t[dim.getDataLen() + 4 * scale_size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<int8_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void CharTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *CharTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int8_t>() + offset;
}

void *CharTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int8_t>() + offset + idx;
}

void *CharTensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((int8_t *)getData()) + size();
}

void *CharTensor::getScale(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getScale() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return ((float *)getScale()) + idx;
}

void *CharTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int8_t *)getData())[i];
}

const void *CharTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int8_t *)getData())[i];
}

const int8_t &CharTensor::getValue(unsigned int i) const {
  return ((int8_t *)getData())[i];
}

int8_t &CharTensor::getValue(unsigned int i) {
  return ((int8_t *)getData())[i];
}

const int8_t &CharTensor::getValue(unsigned int b, unsigned int c,
                                   unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

int8_t &CharTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

void CharTensor::setValue(float value) {
  int8_t *data = (int8_t *)getData();
  std::fill(data, data + size(), static_cast<int8_t>(value));
}

void CharTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = ((int8_t *)getData())[idx];
  output *= beta;
  output += value;

  ((int8_t *)getData())[idx] = static_cast<int8_t>(std::trunc(output));
}

void CharTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w, float value) {
  ((int8_t *)getData())[getIndex(b, c, h, w)] = static_cast<int8_t>(value);
}

void CharTensor::setZero() {
  /// @todo replace with apply_i or scal
  setValue(0);
}

void CharTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  /// @note Sampling from the normal/uniform distribution is invalid
  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  case Initializer::NONE:
    break;
  default:
    throw std::invalid_argument("Initializer not valid for " +
                                getStringDataType());
    break;
  }

  putData();
}

void CharTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

int CharTensor::multiply_i(float const &value) {
  // multiply value to scale factors
  float *g_scale = (float *)getScale();

  sscal(scale_size(), value, g_scale, 1);
  return ML_ERROR_NONE;
}

Tensor &CharTensor::multiply(Tensor const &input, Tensor &output,
                             const float scale) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr, q_scheme());

  NNTR_THROW_IF(q_scheme() != input.q_scheme(), std::invalid_argument)
    << "[Tensor] Cannot multiply tensors with different quantization schemes.";

  /// @note remove after vector scale multiply is implemented
  NNTR_THROW_IF(q_scheme() != QScheme::PER_TENSOR_AFFINE, std::invalid_argument)
    << "Multiplication other than per tensor affine quantization scheme is "
       "NYI.";

  float lhs_scale = *(float *)getScale();
  float rhs_scale = *input.getScale<float>();

  /// @note current impl assumes pre-established quantization parameters are set
  /// @todo 1. verify result_scale is valid 2. calculate qparams if not given
  NNTR_THROW_IF(std::fpclassify(lhs_scale) == FP_ZERO ||
                  std::fpclassify(rhs_scale) == FP_ZERO ||
                  std::fpclassify(scale) == FP_ZERO,
                std::invalid_argument)
    << "scale factors not set, cannot multiply";

  float multiplier = lhs_scale * rhs_scale / scale;

  int8_t *lhs = (int8_t *)getData();
  int8_t *rhs = input.getData<int8_t>();
  int8_t *result = output.getData<int8_t>();

  for (unsigned int i = 0; i < size(); ++i) {
    int32_t accum_val =
      static_cast<int32_t>(lhs[i]) * static_cast<int32_t>(rhs[i]);

    result[i] =
      std::max(-128, std::min((int)std::lround(multiplier * accum_val), 127));
  }

  *output.getScale<float>() = scale;

  return output;
}

Tensor &CharTensor::add(Tensor const &input, Tensor &output,
                        float const scale) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr, qscheme);

  NNTR_THROW_IF(q_scheme() != input.q_scheme(), std::invalid_argument)
    << "[Tensor] Cannot multiply tensors with different quantization schemes.";

  /// @note remove after vector scale multiply is implemented
  NNTR_THROW_IF(q_scheme() != QScheme::PER_TENSOR_AFFINE, std::invalid_argument)
    << "Tensor addition other than per tensor affine quantization scheme is "
       "NYI.";

  float lhs_scale = *(float *)getScale();
  float rhs_scale = *input.getScale<float>();

  /// @note current impl assumes pre-established quantization parameters are set
  /// @todo 1. verify result_scale is valid 2. calculate qparams if not given
  ///       3. check qscheme is per tensor affine
  NNTR_THROW_IF(std::fpclassify(lhs_scale) == FP_ZERO ||
                  std::fpclassify(rhs_scale) == FP_ZERO ||
                  std::fpclassify(scale) == FP_ZERO,
                std::invalid_argument)
    << "scale factors not set, cannot multiply";

  /// @todo check whether the following method has faster execution speed.
  /// 1. clone input A and B to A_fp32 and B_fp32
  /// 2. dequantize A_fp32 and B_fp32
  /// 3. perform addition: A_fp32.add(B_fp32, output_fp32)
  /// 4. quantize output_fp32
  for (unsigned int b = 0; b < batch(); ++b) {
    for (unsigned int c = 0; c < channel(); ++c) {
      for (unsigned int h = 0; h < height(); ++h) {
        for (unsigned int w = 0; w < width(); ++w) {
          float val = getValue(b, c, h, w) * lhs_scale +
                      input.getValue<int8_t>(b, c, h, w) * rhs_scale;

          output.setValue(
            b, c, h, w,
            static_cast<int8_t>(
              std::max(-128, std::min((int)std::lround(val / scale), 127))));
        }
      }
    }
  }
  *output.getScale<float>() = scale;

  return output;
}

void CharTensor::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData());
}

void CharTensor::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  /// @todo support copy from float32 & float16 to int8 data
  /// @note this could require scale factor
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::QINT8:
    copy(from.getData());
    break;
  case ml::train::TensorDim::DataType::FP32:
    copy_fp32(from.size(), from.getData<float>(), (int8_t *)getData());
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void CharTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<int8_t>(b, c, h, w));
        }
      }
    }
  }
}

void CharTensor::save(std::ostream &file) {
  /// @note Save quantization information
  save_quantization_info(file);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz,
               "[CharTensor::save] operation failed");
  putData();
}

void CharTensor::read(std::ifstream &file, size_t start_offset,
                      bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max()) {
    start_offset = file_offset;
  }
  read_quantization_info(file, start_offset, read_from_offset);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  if (read_from_offset) {
    start_offset += sizeof(uint16_t);
  }

  checkedRead(file, (char *)getData(), sz,
              "[CharTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

std::vector<unsigned int> CharTensor::argmax() const {
  std::vector<unsigned int> result;
  const int8_t *data = (int8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto max_iter =
      std::max_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, max_iter) - (b * feature_len);
  }
  return result;
}

std::vector<unsigned int> CharTensor::argmin() const {
  std::vector<unsigned int> result;
  const int8_t *data = (int8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto min_iter =
      std::min_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, min_iter) - (b * feature_len);
  }
  return result;
}

float CharTensor::max_abs() const {
  const int8_t *data = (int8_t *)getData();
  unsigned int idx;

  int8_t max_val = data[0];
  for (unsigned int i = 1; i < size(); i += 1) {
    int8_t cur_val = (data[i] >= 0) ? data[i] : -1 * data[i];
    if (cur_val > max_val) {
      max_val = cur_val;
    }
  }

  return max_val;
}

float CharTensor::maxValue() const {
  const int8_t *data = (int8_t *)getData();
  return *std::max_element(data, data + size());
}

float CharTensor::minValue() const {
  const int8_t *data = (int8_t *)getData();
  return *std::min_element(data, data + size());
}

void CharTensor::print(std::ostream &out) const {
  const int8_t *data = (int8_t *)getData();
  unsigned int len = size();
  out << "data addr: " << reinterpret_cast<const float *>(data) << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (int)data[0] << ' ' << (int)data[1] << ' ' << (int)data[2]
        << " ... " << (int)data[len - 3] << ' ' << (int)data[len - 2] << ' '
        << (int)data[len - 1] << ']' << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);
  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << (int)this->getValue(k, l, i, j) << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  } else {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int i = 0; i < height(); i++) {
        for (unsigned int j = 0; j < width(); j++) {
          for (unsigned int l = 0; l < channel(); l++) {
            out << std::setw(10) << (int)this->getValue(k, l, i, j) << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
    out.copyfmt(init);
  }

  /// print quantization information
  const float *q_scales = (float *)getScale();

  if (scale_size() > 50) {
    out << "Scale factors: [" << q_scales[0] << ' ' << q_scales[1] << ' '
        << q_scales[2] << " ... " << q_scales[len - 3] << ' '
        << q_scales[len - 2] << ' ' << q_scales[len - 1] << ']' << std::endl;
    return;
  }

  out << "Scale factors: ";
  for (unsigned i = 0; i < scale_size(); ++i) {
    out << q_scales[i] << " ";
  }
  out << std::endl;
}

size_t CharTensor::getMemoryBytes() const {
  return bytes() + scale_size() * sizeof(float);
}

size_t CharTensor::scale_size() const {
  switch (qscheme) {
  case QScheme::PER_TENSOR_AFFINE:
    return 1;
    break;
  case QScheme::PER_CHANNEL_AFFINE:
    return width();
    break;
  default:
    break;
  }
  return 0;
}

QScheme CharTensor::q_scheme() const { return qscheme; }

void CharTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  scopy(size(), (int8_t *)buf, 1, (int8_t *)getData(), 1);

  float *scales = (float *)(((int8_t *)buf) + size());
  scopy(scale_size(), scales, 1, (float *)getScale(), 1);
}

void CharTensor::save_quantization_info(std::ostream &file) {
  checkedWrite(file, (char *)&qscheme, sizeof(uint16_t),
               "[CharTensor::save] failed to write quantization information");
}

void CharTensor::read_quantization_info(std::ifstream &file,
                                        size_t start_offset,
                                        bool read_from_offset) {
  checkedRead(file, (char *)&qscheme, sizeof(uint16_t),
              "[CharTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

} // namespace nntrainer
