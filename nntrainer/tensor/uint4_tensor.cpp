// SPDX-License-Identifier: Apache-2.0
/**
 * @file	uint4_tensor.cpp
 * @date	20 March 2025
 * @brief	This is Uint4QTensor class for quantized 4-bit unsigned integer
 * calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <cpu_backend.h>
#include <tensor.h>
#include <uint4_tensor.h>

namespace nntrainer {

Uint4QTensor::Uint4QTensor(std::string name_, Tformat fm, QScheme qscheme_) :
  TensorBase(name_, fm, Tdatatype::UINT4), qscheme(qscheme_) {}

Uint4QTensor::Uint4QTensor(const TensorDim &d, bool alloc_now, Initializer init,
                           std::string name, QScheme qscheme_) :
  TensorBase(d, alloc_now, init, name), qscheme(qscheme_) {
  if (alloc_now)
    allocate();
}

Uint4QTensor::Uint4QTensor(const TensorDim &d, const void *buf,
                           QScheme qscheme_) :
  Uint4QTensor(d, true, Initializer::NONE, "", qscheme_) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

Uint4QTensor::Uint4QTensor(
  std::vector<std::vector<std::vector<std::vector<uint8_t>>>> const &d,
  std::vector<float> const &scales,
  std::vector<unsigned int> const &zero_points, Tformat fm, QScheme qscheme_) :
  qscheme(qscheme_) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize Uint4QTensor from empty vector");
  }

  NNTR_THROW_IF(scales.size() != scale_size(), std::invalid_argument)
    << "invalid scale factor size " << scales.size();

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

  dim.setTensorType({fm, Tdatatype::UINT4});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;
  qscheme = qscheme_;

  /// @note sizeof(float) * scale_size() assumes scale factors are in
  /// full-precision fp.
  MemoryData *mem_data = new MemoryData((
    void
      *)(new uint8_t[(dim.getDataLen() + 1) / 2 + sizeof(float) * scale_size() +
                     sizeof(unsigned int) * scale_size()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *ptr) {
    delete[] ptr->getAddr<uint8_t>();
    delete ptr;
  });

  offset = 0;

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

  unsigned int *zps = getZeroPoint();

  // copy zero points
  for (size_t i = 0; i < zero_points.size(); ++i) {
    zps[i] = zero_points[i];
  }
}

bool Uint4QTensor::operator==(const Uint4QTensor &rhs) const {
  if (qscheme != rhs.qscheme)
    return false;

  // compare quantized data
  const uint8_t *_data = (uint8_t *)getData();
  const uint8_t *_rdata = (uint8_t *)rhs.getData();
  for (size_t i = 0; i < (size() + 1) / 2; ++i) {
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

void Uint4QTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    /// quantized 4-bit is stored as a 8-bit signed integer (uint4x2)
    mem_data = new MemoryData(
      (void *)(new uint8_t[(dim.getDataLen() + 1) / 2 +
                           sizeof(float) * scale_size() +
                           sizeof(unsigned int) * scale_size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void Uint4QTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *Uint4QTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

void *Uint4QTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset + (idx / 2);
}

void *Uint4QTensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((uint8_t *)getData()) + (size() + 1) / 2;
}

void *Uint4QTensor::getScale(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getScale() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return ((float *)getScale()) + idx;
}

unsigned int *Uint4QTensor::getZeroPoint() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((unsigned int *)((float *)((uint8_t *)getData() +
                                     ((size() + 1) / 2)))) +
         scale_size();
}

unsigned int *Uint4QTensor::getZeroPoint(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Uint4QTensor::getZeroPoint() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return (((unsigned int *)((float *)((uint8_t *)getData() +
                                      ((size() + 1) / 2)))) +
          scale_size()) +
         idx;
}

void *Uint4QTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((uint8_t *)getData())[i / 2];
}

const void *Uint4QTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((uint8_t *)getData())[i / 2];
}

const uint8_t Uint4QTensor::getValue(unsigned int i) const {
  uint8_t value = ((uint8_t *)getData())[i / 2];
  return (i % 2 == 0) ? ((value >> 4) & 0xF) : (value & 0x0F);
}

uint8_t Uint4QTensor::getValue(unsigned int i) {
  uint8_t value = ((uint8_t *)getData())[i / 2];
  return (i % 2 == 0) ? ((value >> 4) & 0xF) : (value & 0x0F);
}

const uint8_t Uint4QTensor::getValue(unsigned int b, unsigned int c,
                                     unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

uint8_t Uint4QTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                               unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

/// @todo this func should be template function
void Uint4QTensor::setValue(float value) {
  NNTR_THROW_IF(value > 15 || value < 0, std::out_of_range)
    << "Value must be in range [0, 15]. Input value: " << value;

  uint8_t val = static_cast<uint8_t>(value);
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + (size() + 1) / 2, (val << 4) | (val & 0x0f));
}

/// @todo this func should be template function
void Uint4QTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = getValue(idx);
  output *= beta;
  output += value;

  // if result value is out of range, clamp to max/min value
  uint8_t val =
    static_cast<uint8_t>(std::trunc(std::clamp((int)output, 0, 15)));

  // encode result value to uint8 data
  ((uint8_t *)getData())[idx / 2] =
    (idx % 2 == 0) ? (val << 4) | (((uint8_t *)getData())[idx / 2] & 0x0f)
                   : (((uint8_t *)getData())[idx / 2] & 0xf0) | (val & 0x0f);
}

/// @todo this func should be template function
void Uint4QTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w, float value) {
  NNTR_THROW_IF(value > 15 || value < 0, std::out_of_range)
    << "Value must be in range [0, 15]. Input value: " << value;

  auto const &idx = getIndex(b, c, h, w);
  uint8_t val = static_cast<uint8_t>(value);

  ((uint8_t *)getData())[idx / 2] =
    (idx % 2 == 0) ? (val << 4) | (((uint8_t *)getData())[idx / 2] & 0x0f)
                   : (((uint8_t *)getData())[idx / 2] & 0xf0) | (val & 0x0f);
}

void Uint4QTensor::setZero() {
  /// @todo accelerate with SIMD
  setValue(0);
}

void Uint4QTensor::initialize() {
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
    throw std::invalid_argument(
      "Initializer other than zero and one is not valid for " +
      getStringDataType());
    break;
  }

  putData();
}

void Uint4QTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

void Uint4QTensor::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData());
}

void Uint4QTensor::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of the tensor to copy must match.";

  /// @todo support copy from float32 & float16 to uint8 data
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::UINT4:
    copy(from.getData());
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void Uint4QTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<uint8_t>(b, c, h, w));
        }
      }
    }
  }
}

void Uint4QTensor::save(std::ostream &file) {
  /// @note Save quantization information
  save_quantization_info(file);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz,
               "[Uint4QTensor::save] operation failed");
  putData();
}

void Uint4QTensor::read(std::ifstream &file, size_t start_offset,
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
              "[Uint4QTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

void Uint4QTensor::read(ReadSource src, size_t start_offset,
                        bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max()) {
    start_offset = file_offset;
  }
  read_quantization_info(src, start_offset, read_from_offset);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  if (read_from_offset) {
    start_offset += sizeof(uint16_t);
  }

  checkedRead(src, (char *)getData(), sz,
              "[Uint4QTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

std::vector<unsigned int> Uint4QTensor::argmax() const {
  std::vector<unsigned int> result;
  const uint8_t *data = (uint8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();
  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; ++b) {
    uint8_t curr_val, max_val = 0;
    unsigned int max_element_idx = 0;
    for (unsigned int idx = 0; idx < feature_len; ++idx) {
      curr_val = getValue(idx + b * feature_len);

      if (curr_val > max_val) {
        max_val = curr_val;
        max_element_idx = idx;
      }
    }
    result[b] = max_element_idx;
  }
  return result;
}

std::vector<unsigned int> Uint4QTensor::argmin() const {
  std::vector<unsigned int> result;
  const uint8_t *data = (uint8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();
  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; ++b) {
    uint8_t curr_val, min_val = 15;
    unsigned int min_element_idx = 0;
    for (unsigned int idx = 0; idx < feature_len; ++idx) {
      curr_val = getValue(idx + b * feature_len);

      if (curr_val < min_val) {
        min_val = curr_val;
        min_element_idx = idx;
      }
    }
    result[b] = min_element_idx;
  }
  return result;
}

float Uint4QTensor::max_abs() const {
  uint8_t abs_max_val = 0;
  uint8_t curr_val;
  for (unsigned int idx = 0; idx < size(); ++idx) {
    curr_val = getValue(idx);
    abs_max_val = (curr_val > abs_max_val) ? curr_val : abs_max_val;

    // Terminate search when abs_max_val is an Uint4 absolute max value 15
    if (abs_max_val == 15)
      return abs_max_val;
  }

  return abs_max_val;
}

float Uint4QTensor::maxValue() const {
  uint8_t max_val = 0;
  uint8_t curr_val;
  for (unsigned int idx = 0; idx < size(); ++idx) {
    curr_val = getValue(idx);
    max_val = (curr_val > max_val) ? curr_val : max_val;

    // Terminate search when max_val is an Uint4 max value 15
    if (max_val == 15)
      return max_val;
  }

  return max_val;
}

float Uint4QTensor::minValue() const {
  uint8_t min_val = 15;
  uint8_t curr_val;
  for (unsigned int idx = 0; idx < size(); ++idx) {
    curr_val = getValue(idx);
    min_val = (curr_val < min_val) ? curr_val : min_val;

    // Terminate search when min_val is an Uint4 min value 0
    if (min_val == 0)
      return min_val;
  }

  return min_val;
}

void Uint4QTensor::print(std::ostream &out) const {
  const uint8_t *data = (uint8_t *)getData();
  unsigned int len = size();
  out << "data addr: " << reinterpret_cast<const float *>(data) << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (int)getValue(0) << ' ' << (int)getValue(1) << ' '
        << (int)getValue(2) << " ... " << (int)getValue(len - 3) << ' '
        << (int)getValue(len - 2) << ' ' << (int)getValue(len - 1) << ']'
        << std::endl;
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
  const unsigned int *q_zero_points = getZeroPoint();

  if (scale_size() > 50) {
    out << "Scale factors: [" << q_scales[0] << ' ' << q_scales[1] << ' '
        << q_scales[2] << " ... " << q_scales[len - 3] << ' '
        << q_scales[len - 2] << ' ' << q_scales[len - 1] << ']' << std::endl;

    out << "Zero points: [" << q_zero_points[0] << ' ' << q_zero_points[1]
        << ' ' << q_zero_points[2] << " ... " << q_zero_points[len - 3] << ' '
        << q_zero_points[len - 2] << ' ' << q_zero_points[len - 1] << ']'
        << std::endl;
    return;
  }

  out << "Scale factors: ";
  for (unsigned i = 0; i < scale_size(); ++i) {
    out << q_scales[i] << " ";
  }
  out << std::endl;

  out << "Zero points: ";
  for (unsigned i = 0; i < scale_size(); ++i) {
    out << q_zero_points[i] << " ";
  }
  out << std::endl;
}

size_t Uint4QTensor::getMemoryBytes() const {
  return ((size() + 1) / 2) * dim.getDataTypeSize() +
         scale_size() * sizeof(float) + scale_size() * sizeof(unsigned int);
}

size_t Uint4QTensor::scale_size() const {
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

QScheme Uint4QTensor::q_scheme() const { return qscheme; }

void Uint4QTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  // copy tensor data
  scopy((size() + 1) / 2, (uint8_t *)buf, 1, (uint8_t *)getData(), 1);

  // copy scale factor data
  float *scales = (float *)(((uint8_t *)buf) + (size() + 1) / 2);
  scopy(scale_size(), scales, 1, (float *)getScale(), 1);

  // copy zero points
  unsigned int *zps =
    (unsigned int *)((float *)(((uint8_t *)buf) + size()) + scale_size());

  memcpy(getZeroPoint(), zps, scale_size() * sizeof(unsigned int));
}

void Uint4QTensor::save_quantization_info(std::ostream &file) {
  checkedWrite(file, (char *)&qscheme, sizeof(uint16_t),
               "[Uint4QTensor::save] failed to write quantization information");
}

void Uint4QTensor::read_quantization_info(std::ifstream &file,
                                          size_t start_offset,
                                          bool read_from_offset) {
  checkedRead(file, (char *)&qscheme, sizeof(uint16_t),
              "[Uint4QTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

void Uint4QTensor::read_quantization_info(ReadSource src, size_t start_offset,
                                          bool read_from_offset) {
  checkedRead(src, (char *)&qscheme, sizeof(uint16_t),
              "[Uint4QTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

} // namespace nntrainer
