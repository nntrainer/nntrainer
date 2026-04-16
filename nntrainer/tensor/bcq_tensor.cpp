// SPDX-License-Identifier: Apache-2.0
/**
 * @file	bcq_tensor.cpp
 * @date	06 December 2024
 * @brief	This is BCQTensor class for binary-code-based quantization
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <iomanip>
#include <iostream>

#include <bcq_tensor.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

BCQTensor::BCQTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::BCQ) {}

BCQTensor::BCQTensor(const TensorDim &d, bool alloc_now, Initializer init,
                     std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

BCQTensor::BCQTensor(const TensorDim &d, const void *buf) : BCQTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr) {
      copy(buf);
      createBCQW();
    }
  }
}

bool BCQTensor::operator==(const BCQTensor &rhs) const {
  const uint32_t *_data = (uint32_t *)getData();
  const uint32_t *_rdata = (uint32_t *)rhs.getData();

  for (size_t i = 0; i < size() + scale_size(); ++i) {
    if (_data[i] - _rdata[i])
      return false;
  }

  return true;
}

void BCQTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new uint32_t[size() + scale_size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint32_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void BCQTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *BCQTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint32_t>() + offset;
}

void *BCQTensor::getData(size_t idx) const {
  NNTR_THROW_IF(idx > dim.getDataLen(), std::invalid_argument)
    << "Tensor::getData() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint32_t>() + offset + (idx / 32);
}

void *BCQTensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((uint32_t *)getData()) + size();
}

void *BCQTensor::getScale(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getScale() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return ((uint32_t *)getScale()) + idx;
}

void *BCQTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), width(), height() / 32);
  if (i > index) {
    return nullptr;
  }
  return &((uint32_t *)getData())[i];
}

const void *BCQTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), width(), height() / 32);
  if (i > index) {
    return nullptr;
  }
  return &((uint32_t *)getData())[i];
}

const uint32_t &BCQTensor::getValue(unsigned int i) const {
  return ((uint32_t *)getData())[i];
}

uint32_t &BCQTensor::getValue(unsigned int i) {
  return ((uint32_t *)getData())[i];
}

const uint32_t &BCQTensor::getValue(unsigned int b, unsigned int c,
                                    unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w / 32));
}

uint32_t &BCQTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                              unsigned int w) {
  return getValue(getIndex(b, c, h, w / 32));
}

void BCQTensor::setValue(float value) {
  uint32_t *data = (uint32_t *)getData();
  std::fill(data, data + size(), (uint32_t)value);
}

void BCQTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w, float value) {
  ((uint32_t *)getData())[getIndex(b, c, h, w / 32)] = (uint32_t)value;
}

void BCQTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w, float value, float beta) {
  throw std::invalid_argument("addValue() is not valid for " +
                              getStringDataType());
}

void BCQTensor::setZero() {
  /// @todo replace with apply_i or scal
  setValue(0);
}

void BCQTensor::initialize() {
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

void BCQTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

Tensor &BCQTensor::dot(Tensor const &input, Tensor &output, bool trans,
                       bool trans_in, float beta, ComputeOps *ops) const {
  BiQGEMM::matrixDotMatrix(output.getData(), *bcq_weight_.get(),
                           input.getData(),
                           trans_in ? input.width() : input.height());
  return output;
}

void BCQTensor::copy(const Tensor &from, ComputeOps *ops) {
  reshape(from.getDim());
  copy(from.getData());
}

void BCQTensor::copyData(const Tensor &from, ComputeOps *ops) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  /// @todo support copy from other data types
  /// @todo check data type properly
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::BCQ:
    copy(from.getData());
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void BCQTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.width(); ++h) {
        for (unsigned int w = 0; w < output.height() / 32; ++w) {
          output.setValue(b, c, w, h, input.getValue<uint32_t>(b, c, h, w));
        }
      }
    }
  }
}

void BCQTensor::save(std::ostream &file) {
  /// @note Save quantization information
  save_quantization_info(file);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";
  /** This is Save for old weight -> new BCQ Weight
  std::streamsize sz1 = static_cast<std::streamsize>(bytes());
  std::streamsize sz2 = static_cast<std::streamsize>(scale_size() *
  sizeof(float)); checkedWrite(file, (char *)bcq_weight->binary_weights.data(),
  sz1, "Failed"); checkedWrite(file, (char *)bcq_weight->weight_scales.data(),
  sz2, "Failed");
  */
  checkedWrite(file, (char *)getData(), sz,
               "[BCQTensor::save] operation failed");
  putData();
}

void BCQTensor::readFSU() { createBCQW(); }

void BCQTensor::read(std::ifstream &file, size_t start_offset,
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

  checkedRead(file, (char *)getData(), sz, "[BCQTensor::read] operation failed",
              start_offset, read_from_offset);
  putData();

  createBCQW();
}

std::vector<unsigned int> BCQTensor::argmax() const {
  std::vector<unsigned int> result;
  const uint32_t *data = (uint32_t *)getData();
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

void BCQTensor::save_quantization_info(std::ostream &file) {
  checkedWrite(file, (char *)&quantized_bit_size_, sizeof(uint16_t),
               "[BCQTensor::save] failed to write quantization information");
}

void BCQTensor::read_quantization_info(std::ifstream &file, size_t start_offset,
                                       bool read_from_offset) {
  checkedRead(file, (char *)&quantized_bit_size_, sizeof(uint16_t),
              "[BCQTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

size_t BCQTensor::size() const {
  return quantized_bit_size_ * dim.width() * ((dim.height() + 31) / 32);
}

float BCQTensor::max_abs(ComputeOps *ops) const { return maxValue(); }

float BCQTensor::maxValue() const {
  const uint32_t *data = (uint32_t *)getData();
  return *std::max_element(data, data + size());
}

float BCQTensor::minValue() const {
  const uint32_t *data = (uint32_t *)getData();
  return *std::min_element(data, data + size());
}

void BCQTensor::print(std::ostream &out) const {
  const uint32_t *data = (uint32_t *)getData();
  unsigned int len = size();
  out << "data addr: " << reinterpret_cast<const float *>(data) << '\n';
  out << dim;

  if (len > 512) {
    out << '[' << (int)data[0] << ' ' << (int)data[1] << ' ' << (int)data[2]
        << " ... " << (int)data[len - 3] << ' ' << (int)data[len - 2] << ' '
        << (int)data[len - 1] << ']' << std::endl;
    printScales(out);
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);

  size_t idx = 0;
  for (unsigned int bit = 0; bit < quantized_bit_size_; ++bit) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < width(); i++) {
          for (unsigned int j = 0; j < (height() + 31) / 32; j++) {
            out << data[idx++] << " ";
          }
          out << std::endl;
        }
      }
    }
    out << "-------" << std::endl;
  }
  printScales(out);
}

size_t BCQTensor::getMemoryBytes() const {
  return bytes() + scale_size() * sizeof(float);
}

size_t BCQTensor::scale_size() const { return width() * quantized_bit_size_; }

void BCQTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  /// @todo need to optimize
  for (unsigned int i = 0; i < size() + scale_size(); ++i) {
    ((uint32_t *)getData())[i] = ((uint32_t *)buf)[i];
  }
}

std::string BCQTensor::getStringDataType() const { return "BCQ"; }

void BCQTensor::printScales(std::ostream &out) const {
  const float *q_scales = (float *)getScale();
  unsigned int len = scale_size();

  if (len > 50) {
    out << "Scale factors: [" << (int)q_scales[0] << ' ' << (int)q_scales[1]
        << ' ' << (int)q_scales[2] << " ... " << (int)q_scales[len - 3] << ' '
        << (int)q_scales[len - 2] << ' ' << (int)q_scales[len - 1] << ']'
        << std::endl;
    return;
  }

  out << "Scale factors: ";
  for (unsigned i = 0; i < scale_size(); ++i) {
    out << q_scales[i] << " ";
  }
  out << std::endl;
}

void BCQTensor::createBCQW() {
  size_t qbit_of_clusters[] = {quantized_bit_size_};
  size_t size_of_clusters[] = {width()};
  const size_t number_of_cluster = 1;

  /// @note hidden_tile_size should be set as a multiple of 32. This variable
  /// is related to the speed of matrixDotMatrix. The optimal value should be
  /// found with various values according to the usage environment.
  size_t hidden_tile_size = 32;

  bcq_weight_ = std::make_unique<BiQGEMM::BCQW>(
    (uint32_t *)(data->getAddr<uint32_t>()),
    (float *)((uint32_t *)(data->getAddr<uint32_t>()) + size()), width(),
    height(), number_of_cluster, qbit_of_clusters, size_of_clusters,
    hidden_tile_size);
}

} // namespace nntrainer
