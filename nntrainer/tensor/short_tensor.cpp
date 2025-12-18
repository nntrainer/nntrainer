// SPDX-License-Identifier: Apache-2.0
/**
 * @file	short_tensor.cpp
 * @date	10 January 2025
 * @brief	This is ShortTensor class for 16-bit signed integer calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <cpu_backend.h>
#include <short_tensor.h>
#include <tensor.h>

namespace nntrainer {

ShortTensor::ShortTensor(std::string name_, Tformat fm, QScheme qscheme_) :
  TensorBase(name_, fm, Tdatatype::QINT16), qscheme(qscheme_) {}

ShortTensor::ShortTensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name, QScheme qscheme_) :
  TensorBase(d, alloc_now, init, name), qscheme(qscheme_) {
  if (alloc_now)
    allocate();
}

ShortTensor::ShortTensor(const TensorDim &d, const void *buf,
                         QScheme qscheme_) :
  ShortTensor(d, true, Initializer::NONE, "", qscheme_) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

ShortTensor::ShortTensor(
  std::vector<std::vector<std::vector<std::vector<int16_t>>>> const &d,
  std::vector<float> const &scales, Tformat fm, QScheme qscheme_) :
  qscheme(qscheme_) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize ShortTensor from empty vector");
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

  dim.setTensorType({fm, Tdatatype::QINT16});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;

  MemoryData *mem_data = new MemoryData(
    (void *)(new int16_t[dim.getDataLen() +
                         sizeof(float) / sizeof(int16_t) * scale_size()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<int16_t>();
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

bool ShortTensor::operator==(const ShortTensor &rhs) const {
  if (qscheme != rhs.qscheme)
    return false;

  // compare quantized data
  const int16_t *_data = (int16_t *)getData();
  const int16_t *_rdata = (int16_t *)rhs.getData();
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

void ShortTensor::allocate() {
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
      (void *)(new int16_t[dim.getDataLen() +
                           sizeof(float) / sizeof(int16_t) * scale_size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<int16_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void ShortTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *ShortTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int16_t>() + offset;
}

void *ShortTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int16_t>() + offset + idx;
}

void *ShortTensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((int16_t *)getData()) + size();
}

void *ShortTensor::getScale(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getScale() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return ((float *)getScale()) + idx;
}

void *ShortTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int16_t *)getData())[i];
}

const void *ShortTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int16_t *)getData())[i];
}

const int16_t &ShortTensor::getValue(unsigned int i) const {
  return ((int16_t *)getData())[i];
}

int16_t &ShortTensor::getValue(unsigned int i) {
  return ((int16_t *)getData())[i];
}

const int16_t &ShortTensor::getValue(unsigned int b, unsigned int c,
                                     unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

int16_t &ShortTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                               unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

void ShortTensor::setValue(float value) {
  int16_t *data = (int16_t *)getData();
  std::fill(data, data + size(), static_cast<int16_t>(value));
}

void ShortTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = ((int16_t *)getData())[idx];
  output *= beta;
  output += value;

  ((int16_t *)getData())[idx] = static_cast<int16_t>(std::trunc(output));
}

void ShortTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w, float value) {
  ((int16_t *)getData())[getIndex(b, c, h, w)] = static_cast<int16_t>(value);
}

void ShortTensor::setZero() {
  /// @todo replace with apply_i or scal
  setValue(0);
}

void ShortTensor::initialize() {
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

void ShortTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

void ShortTensor::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData());
}

void ShortTensor::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  /// @todo support copy from other data types
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::QINT16:
    copy(from.getData());
    break;
  case ml::train::TensorDim::DataType::FP32:
    copy_fp32(from.size(), from.getData<float>(), (int16_t *)getData());
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void ShortTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<int16_t>(b, c, h, w));
        }
      }
    }
  }
}

void ShortTensor::save(std::ostream &file) {
  /// @note Save quantization information
  save_quantization_info(file);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz,
               "[ShortTensor::save] operation failed");
  putData();
}

void ShortTensor::read(std::ifstream &file, size_t start_offset,
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
              "[ShortTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

std::vector<unsigned int> ShortTensor::argmax() const {
  std::vector<unsigned int> result;
  const int16_t *data = (int16_t *)getData();
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

std::vector<unsigned int> ShortTensor::argmin() const {
  std::vector<unsigned int> result;
  const int16_t *data = (int16_t *)getData();
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

float ShortTensor::max_abs() const {
  const int16_t *data = (int16_t *)getData();
  unsigned int idx;

  int16_t max_val = data[0];
  for (unsigned int i = 1; i < size(); i += 1) {
    int16_t cur_val = (data[i] >= 0) ? data[i] : -1 * data[i];
    if (cur_val > max_val) {
      max_val = cur_val;
    }
  }

  return max_val;
}

float ShortTensor::maxValue() const {
  const int16_t *data = (int16_t *)getData();
  return *std::max_element(data, data + size());
}

float ShortTensor::minValue() const {
  const int16_t *data = (int16_t *)getData();
  return *std::min_element(data, data + size());
}

void ShortTensor::print(std::ostream &out) const {
  const int16_t *data = (int16_t *)getData();
  unsigned int len = size();
  out << "data addr: " << reinterpret_cast<const float *>(data) << '\n';
  out << dim;

  if (len > 512) {
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

size_t ShortTensor::getMemoryBytes() const {
  return bytes() + scale_size() * sizeof(float);
}

size_t ShortTensor::scale_size() const {
  switch (qscheme) {
  case QScheme::PER_TENSOR_AFFINE:
    return 1;
    break;
  case QScheme::PER_CHANNEL_AFFINE:
    return height();
    break;
  default:
    break;
  }
  return 0;
}

QScheme ShortTensor::q_scheme() const { return qscheme; }

void ShortTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  copy_s16(size(), (int16_t *)buf, (int16_t *)getData());

  float *scales = (float *)(((int16_t *)buf) + size());
  scopy(scale_size(), scales, 1, (float *)getScale(), 1);
}

void ShortTensor::save_quantization_info(std::ostream &file) {
  checkedWrite(file, (char *)&qscheme, sizeof(uint16_t),
               "[ShortTensor::save] failed to write quantization information");
}
void ShortTensor::read_quantization_info(std::ifstream &file,
                                         size_t start_offset,
                                         bool read_from_offset) {
  checkedRead(file, (char *)&qscheme, sizeof(uint16_t),
              "[ShortTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

} // namespace nntrainer
