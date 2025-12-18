// SPDX-License-Identifier: Apache-2.0
/**
 * @file	uint_tensor.cpp
 * @date	02 April 2024
 * @brief	This is UIntTensor class for unsigned integer calculation
 *          This uint_tensor.cpp contains some codes to define
 *          UIntTensor template methods. This file cannot be used directly but
 *          included by uint_tensor.h only.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef __UINT_TENSOR_H__

template <typename T>
UIntTensor<T>::UIntTensor(std::string name_, Tformat fm, QScheme qscheme_) :
  TensorBase(name_, fm, checkTensorDataType()), qscheme(qscheme_) {}

template <typename T>
UIntTensor<T>::UIntTensor(const TensorDim &d, bool alloc_now, Initializer init,
                          std::string name, QScheme qscheme_) :
  TensorBase(d, alloc_now, init, name), qscheme(qscheme_) {
  if (alloc_now)
    allocate();
}

template <typename T>
UIntTensor<T>::UIntTensor(const TensorDim &d, const void *buf,
                          QScheme qscheme_) :
  UIntTensor(d, true, Initializer::NONE, "", qscheme_) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

template <typename T>
UIntTensor<T>::UIntTensor(
  std::vector<std::vector<std::vector<std::vector<T>>>> const &d,
  std::vector<float> const &scales,
  std::vector<unsigned int> const &zero_points, Tformat fm, QScheme qscheme_) :
  qscheme(qscheme_) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty() ||
      scales.empty() || zero_points.empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize UIntTensor from empty vector");
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

  dim.setTensorType({fm, checkTensorDataType()});

  if (scale_size() != scales.size() || scale_size() != zero_points.size()) {
    throw std::invalid_argument("[Tensor] Scales vector or zero point vector "
                                "size is invalid. scale size: " +
                                std::to_string(scale_size()));
  }

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;

  MemoryData *mem_data = new MemoryData(
    (void *)(new T[dim.getDataLen() + (sizeof(float) + sizeof(unsigned int)) /
                                        sizeof(T) * scale_size()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<T>();
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
            this->setValue(i, j, k, l, static_cast<float>(d[i][j][k][l]));
  } else {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < height(); ++j)
        for (unsigned int k = 0; k < width(); ++k)
          for (unsigned int l = 0; l < channel(); ++l)
            this->setValue(i, l, j, k, static_cast<float>(d[i][j][k][l]));
  }

  // copy scale factors
  scopy(scale_size(), scales.data(), 1, (float *)getScale(), 1);

  unsigned int *zps = getZeroPoint();

  // copy zero points
  for (size_t i = 0; i < zero_points.size(); ++i) {
    zps[i] = zero_points[i];
  }
}

template <typename T>
bool UIntTensor<T>::operator==(const UIntTensor<T> &rhs) const {
  if (qscheme != rhs.qscheme)
    return false;

  // compare quantized data
  const T *_data = (T *)getData();
  const T *_rdata = (T *)rhs.getData();
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

  // compare zero points
  const unsigned int *_zps = getZeroPoint();
  const unsigned int *_rzps = rhs.getZeroPoint();
  for (size_t i = 0; i < scale_size(); ++i) {
    if (_zps[i] != _rzps[i])
      return false;
  }

  return true;
}

template <typename T> void UIntTensor<T>::allocate() {
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
      (void *)(new T[dim.getDataLen() + (sizeof(float) + sizeof(unsigned int)) /
                                          sizeof(T) * scale_size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<T>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

template <typename T> void UIntTensor<T>::deallocate() {
  data = nullptr;
  offset = 0;
}
template <typename T> void *UIntTensor<T>::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<T>() + offset;
}

template <typename T> void *UIntTensor<T>::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<T>() + offset + idx;
}

template <typename T> void *UIntTensor<T>::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((T *)getData()) + size();
}

template <typename T> void *UIntTensor<T>::getScale(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getScale() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return (float *)((T *)getData() + size()) + idx;
}

template <typename T> unsigned int *UIntTensor<T>::getZeroPoint() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((unsigned int *)((float *)((T *)getData() + size()))) + scale_size();
}

template <typename T>
unsigned int *UIntTensor<T>::getZeroPoint(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getZeroPoint() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return (((unsigned int *)((float *)((T *)getData() + size()))) +
          scale_size()) +
         idx;
}

template <typename T> void *UIntTensor<T>::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((T *)getData())[i];
}

template <typename T>
const void *UIntTensor<T>::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((T *)getData())[i];
}

template <typename T> const T &UIntTensor<T>::getValue(unsigned int i) const {
  return ((T *)getData())[i];
}

template <typename T> T &UIntTensor<T>::getValue(unsigned int i) {
  return ((T *)getData())[i];
}

template <typename T>
const T &UIntTensor<T>::getValue(unsigned int b, unsigned int c, unsigned int h,
                                 unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

template <typename T>
T &UIntTensor<T>::getValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

template <typename T> void UIntTensor<T>::setValue(float value) {
  T *data = (T *)getData();
  std::fill(data, data + size(), static_cast<T>(value));
}

template <typename T>
void UIntTensor<T>::addValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = static_cast<float>(((T *)getData())[idx]);
  output *= beta;
  output += value;

  ((T *)getData())[idx] = static_cast<T>(output);
}

template <typename T>
void UIntTensor<T>::setValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w, float value) {
  ((T *)getData())[getIndex(b, c, h, w)] = static_cast<T>(value);
}

template <typename T> void UIntTensor<T>::setZero() {
  /// @todo replace with apply_i or scal
  setValue(0);
}

template <typename T> void UIntTensor<T>::initialize() {
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

template <typename T> void UIntTensor<T>::initialize(Initializer init) {
  initializer = init;
  initialize();
}

template <typename T> void UIntTensor<T>::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData());
}

template <typename T> void UIntTensor<T>::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  // copy data with the same data type T
  if (from.getDataType() == getDataType()) {
    copy(from.getData<T>());
    return;
  }

  /// @todo support copy from other data types
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::FP32: {
    copy_fp32(from.size(), from.getData<float>(), (T *)getData());
    break;
  }
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

template <typename T>
void UIntTensor<T>::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w,
                          static_cast<float>(input.getValue<T>(b, c, h, w)));
        }
      }
    }
  }
}

template <typename T> void UIntTensor<T>::save(std::ostream &file) {
  /// @note Save quantization information
  save_quantization_info(file);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz,
               "[UIntTensor::save] operation failed");
  putData();
}

template <typename T>
void UIntTensor<T>::read(std::ifstream &file, size_t start_offset,
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
              "[UIntTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

template <typename T>
void UIntTensor<T>::read(ReadSource src, size_t start_offset,
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

  checkedRead(src, (char *)getData(), sz, "[UIntTensor::read] operation failed",
              start_offset, read_from_offset);
  putData();
}

template <typename T> std::vector<unsigned int> UIntTensor<T>::argmax() const {
  std::vector<unsigned int> result;
  const T *data = (T *)getData();
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

template <typename T> std::vector<unsigned int> UIntTensor<T>::argmin() const {
  std::vector<unsigned int> result;
  const T *data = (T *)getData();
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

template <typename T> float UIntTensor<T>::max_abs() const {
  return maxValue();
}

template <typename T> float UIntTensor<T>::maxValue() const {
  const T *data = (T *)getData();
  return static_cast<float>(*std::max_element(data, data + size()));
}

template <typename T> float UIntTensor<T>::minValue() const {
  const T *data = (T *)getData();
  return static_cast<float>(*std::min_element(data, data + size()));
}

template <typename T> void UIntTensor<T>::print(std::ostream &out) const {
  const T *data = (T *)getData();
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

template <typename T> size_t UIntTensor<T>::getMemoryBytes() const {
  return bytes() + scale_size() * sizeof(float) +
         scale_size() * sizeof(unsigned int);
}

template <typename T>
void UIntTensor<T>::save_quantization_info(std::ostream &file) {
  checkedWrite(file, (char *)&qscheme, sizeof(uint16_t),
               "[CharTensor::save] failed to write quantization information");
}

template <typename T>
void UIntTensor<T>::read_quantization_info(std::ifstream &file,
                                           size_t start_offset,
                                           bool read_from_offset) {
  checkedRead(file, (char *)&qscheme, sizeof(uint16_t),
              "[CharTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

template <typename T>
void UIntTensor<T>::read_quantization_info(ReadSource src, size_t start_offset,
                                           bool read_from_offset) {
  checkedRead(src, (char *)&qscheme, sizeof(uint16_t),
              "[CharTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
}

template <typename T> size_t UIntTensor<T>::scale_size() const {
  switch (qscheme) {
  case QScheme::PER_TENSOR_AFFINE:
    return 1;
  case QScheme::PER_CHANNEL_AFFINE:
    return width();
  default:
    break;
  }
  return 0;
}

template <typename T> QScheme UIntTensor<T>::q_scheme() const {
  return qscheme;
}

template <typename T> void UIntTensor<T>::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  if (std::is_same<T, uint16_t>::value) {
    const uint16_t *data = (const uint16_t *)buf;
    uint16_t *rdata = (uint16_t *)getData();
    copy_u16((const unsigned int)size(), data, rdata);
  } else {
    /// @todo need to optimize
    memcpy(getData(), buf, size() * (sizeof(T)));
  }

  // copy scale factors
  float *scales = (float *)(((T *)buf) + size());
  scopy(scale_size(), scales, 1, (float *)getScale(), 1);

  // copy zero points
  unsigned int *zps =
    (unsigned int *)((float *)(((T *)buf) + size()) + scale_size());

  memcpy(getZeroPoint(), zps, scale_size() * sizeof(unsigned int));
}

#endif
