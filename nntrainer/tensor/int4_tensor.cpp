// SPDX-License-Identifier: Apache-2.0
/**
 * @file	int4_tensor.cpp
 * @date	23 January 2025
 * @brief	This is Int4QTensor class for quantized 4-bit integer calculation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <cstring>
#include <iomanip>
#include <iostream>

#include <compute_ops.h>
#include <int4_tensor.h>
#include <tensor.h>

namespace nntrainer {

Int4QTensor::Int4QTensor(std::string name_, Tformat fm, QScheme qscheme_,
                         size_t g_size) :
  TensorBase(name_, fm, Tdatatype::QINT4), qscheme(qscheme_),
  group_size_(g_size) {}

Int4QTensor::Int4QTensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name, QScheme qscheme_, size_t g_size) :
  TensorBase(d, alloc_now, init, name), qscheme(qscheme_),
  group_size_(g_size) {
  if (alloc_now)
    allocate();
}

Int4QTensor::Int4QTensor(const TensorDim &d, const void *buf, QScheme qscheme_,
                         size_t g_size) :
  Int4QTensor(d, true, Initializer::NONE, "", qscheme_, g_size) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

Int4QTensor::Int4QTensor(
  std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
  std::vector<float> const &scales, Tformat fm, QScheme qscheme_,
  size_t g_size) :
  qscheme(qscheme_), group_size_(g_size) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize Int4QTensor from empty vector");
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

  dim.setTensorType({fm, Tdatatype::QINT4});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;
  qscheme = qscheme_;

  /// @note sizeof(float) * scale_size() assumes scale factors are in
  /// full-precision fp.
  MemoryData *mem_data =
    new MemoryData((void *)(new int8_t[(dim.getDataLen() + 1) / 2 +
                                       sizeof(float) * scale_size()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<int8_t>();
    delete mem_data;
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
  getComputeOps()->scopy_fp32(scale_size(), scales.data(), 1, (float *)getScale(), 1);
}

bool Int4QTensor::operator==(const Int4QTensor &rhs) const {
  if (qscheme != rhs.qscheme)
    return false;

  // compare quantized data
  const int8_t *_data = (int8_t *)getData();
  const int8_t *_rdata = (int8_t *)rhs.getData();
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

void Int4QTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    /// quantized 4-bit is stored as a 8-bit signed integer (int4x2)
    mem_data =
      new MemoryData((void *)(new int8_t[(dim.getDataLen() + 1) / 2 +
                                         sizeof(float) * scale_size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<int8_t>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void Int4QTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *Int4QTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int8_t>() + offset;
}

void *Int4QTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<int8_t>() + offset + (idx / 2);
}

void *Int4QTensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();
  return ((int8_t *)getData()) + (size() + 1) / 2;
}

void *Int4QTensor::getScale(size_t idx) const {
  NNTR_THROW_IF(idx > scale_size(), std::invalid_argument)
    << "Tensor::getScale() index is not valid";

  if (!data)
    return nullptr;

  data->validate();
  return ((float *)getScale()) + idx;
}

void *Int4QTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int8_t *)getData())[i / 2];
}

const void *Int4QTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((int8_t *)getData())[i / 2];
}

const int8_t Int4QTensor::getValue(unsigned int i) const {
  int8_t value = ((int8_t *)getData())[i / 2];
  return (i % 2 == 0) ? value >> 4 : ((int8_t)(value << 4) >> 4);
}

int8_t Int4QTensor::getValue(unsigned int i) {
  int8_t value = ((int8_t *)getData())[i / 2];
  return (i % 2 == 0) ? value >> 4 : ((int8_t)(value << 4) >> 4);
}

const int8_t Int4QTensor::getValue(unsigned int b, unsigned int c,
                                   unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

int8_t Int4QTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

/// @todo this func should be template function
void Int4QTensor::setValue(float value) {
  NNTR_THROW_IF(value < -8 || value > 7, std::out_of_range)
    << "Value must be in range [-8, 7]. Input value: " << value;

  int8_t val = static_cast<int8_t>(value);
  int8_t *data = (int8_t *)getData();
  std::fill(data, data + (size() + 1) / 2, (val << 4) | (val & 0x0f));
}

/// @todo this func should be template function
void Int4QTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  float output = getValue(idx);
  output *= beta;
  output += value;

  // if result value is out of range, clamp to max/min value
  int8_t val = static_cast<int8_t>(std::trunc(std::clamp((int)output, -8, 7)));

  // encode result value to int8 data
  ((int8_t *)getData())[idx / 2] =
    (idx % 2 == 0) ? (val << 4) | (((int8_t *)getData())[idx / 2] & 0x0f)
                   : (((int8_t *)getData())[idx / 2] & 0xf0) | (val & 0x0f);
}

/// @todo this func should be template function
void Int4QTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w, float value) {
  NNTR_THROW_IF(value < -8 || value > 7, std::out_of_range)
    << "Value must be in range [-8, 7]. Input value: " << value;

  auto const &idx = getIndex(b, c, h, w);
  int8_t val = static_cast<int8_t>(value);

  ((int8_t *)getData())[idx / 2] =
    (idx % 2 == 0) ? (val << 4) | (((int8_t *)getData())[idx / 2] & 0x0f)
                   : (((int8_t *)getData())[idx / 2] & 0xf0) | (val & 0x0f);
}

void Int4QTensor::setZero() {
  /// @todo accelerate with SIMD
  setValue(0);
}

void Int4QTensor::initialize() {
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

void Int4QTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

void Int4QTensor::copy(const Tensor &from, ComputeOps *ops) {
  reshape(from.getDim());
  copy(from.getData());
}

void Int4QTensor::copyData(const Tensor &from, ComputeOps *ops) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of the tensor to copy must match.";

  /// @todo support copy from float32 & float16 to int8 data
  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
    copy(from.getData());
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void Int4QTensor::copy_with_stride(const Tensor &input, Tensor &output) {
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

void Int4QTensor::save(std::ostream &file) {
  /// @note Save quantization information
  save_quantization_info(file);

  std::streamsize sz = static_cast<std::streamsize>(getMemoryBytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << getMemoryBytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz,
               "[Int4QTensor::save] operation failed");
  putData();
}

void Int4QTensor::read(std::ifstream &file, size_t start_offset,
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
              "[Int4QTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

void Int4QTensor::read(ReadSource src, size_t start_offset,
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
              "[Int4QTensor::read] operation failed", start_offset,
              read_from_offset);
  putData();
}

std::vector<unsigned int> Int4QTensor::argmax() const {
  std::vector<unsigned int> result;
  const int8_t *data = (int8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();
  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; ++b) {
    int8_t curr_val, max_val = -8;
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

std::vector<unsigned int> Int4QTensor::argmin() const {
  std::vector<unsigned int> result;
  const int8_t *data = (int8_t *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();
  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; ++b) {
    int8_t curr_val, min_val = 7;
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

float Int4QTensor::max_abs(ComputeOps *ops) const {
  int8_t abs_max_val = 0;
  int8_t curr_val;
  for (unsigned int idx = 0; idx < size(); ++idx) {
    curr_val = std::abs(getValue(idx));
    abs_max_val = (curr_val > abs_max_val) ? curr_val : abs_max_val;

    // Terminate search when abs_max_val is an Int4 absolute max value 8
    if (abs_max_val == 8)
      return abs_max_val;
  }

  return abs_max_val;
}

float Int4QTensor::maxValue() const {
  int8_t max_val = -8;
  int8_t curr_val;
  for (unsigned int idx = 0; idx < size(); ++idx) {
    curr_val = getValue(idx);
    max_val = (curr_val > max_val) ? curr_val : max_val;

    // Terminate search when max_val is an Int4 max value 7
    if (max_val == 7)
      return max_val;
  }

  return max_val;
}

float Int4QTensor::minValue() const {
  int8_t min_val = 7;
  int8_t curr_val;
  for (unsigned int idx = 0; idx < size(); ++idx) {
    curr_val = getValue(idx);
    min_val = (curr_val < min_val) ? curr_val : min_val;

    // Terminate search when min_val is an Int4 min value -8
    if (min_val == -8)
      return min_val;
  }

  return min_val;
}

void Int4QTensor::print(std::ostream &out) const {
  const int8_t *data = (int8_t *)getData();
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

size_t Int4QTensor::getMemoryBytes() const {
  // Scales are stored as fp32 (sizeof(float)) per the canonical layout
  // documented on the class header. allocate() already reserves
  // `sizeof(float) * scale_size()` bytes for the scale section, and the
  // KleidiAI qai8dxp_qsi4cxp_unpacked kernel consumes fp32 scales. Before
  // P6b this function reported `sizeof(uint16_t) * scale_size()` which
  // caused save() and read() to transfer only the low 2 bytes of each
  // fp32 scale (= garbage), while the in-memory buffer actually held
  // correct fp32 values. That silent data corruption on round-trip
  // never surfaced because the FC layer's old dequantize path worked
  // entirely in memory and the test coverage for saved-then-reloaded
  // QINT4 weights was effectively zero. Align with the allocator here.
  return ((size() + 1) / 2) * dim.getDataTypeSize() +
         scale_size() * sizeof(float);
}

size_t Int4QTensor::scale_size() const {
  switch (qscheme) {
  case QScheme::PER_TENSOR_AFFINE:
    return 1;
    break;
  case QScheme::PER_CHANNEL_AFFINE:
    // group_size_ == 0 is the canonical signal for "pure per-channel":
    // exactly one scale per output column. For nntrainer's FC weight
    // layout TensorDim(1, 1, K=in_features, N=out_features) where
    // height() is K (input features) and width() is N (output
    // features), the natural per-output-channel quantization produces
    // N scales = width(). This matches:
    //   - KleidiAI qsi4cxp kxn: rhs_scales_f32[n_idx], indexed by
    //     output column, length N.
    //   - HuggingFace / PyTorch per-channel quant: one scale per
    //     output feature.
    //   - QNN AXIS_SCALE_OFFSET with axis=1 (output dim) and
    //     numScaleOffsets=N.
    //
    // group_size_ == height() (= K, the row length along the
    // reduction axis) is semantically identical to pure per-channel:
    // "all K elements in one output column share one scale".
    if (group_size_ == 0 || group_size_ == height())
      return width();
    return height() * width() / group_size_;
    break;
  default:
    break;
  }
  return 0;
}

QScheme Int4QTensor::q_scheme() const { return qscheme; }

void Int4QTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  // copy tensor data
  getComputeOps()->scopy_s8((size() + 1) / 2, (int8_t *)buf, 1, (int8_t *)getData(), 1);

  // copy scale factor data
  float *scales = (float *)(((int8_t *)buf) + (size() + 1) / 2);
  getComputeOps()->scopy_fp32(scale_size(), scales, 1, (float *)getScale(), 1);
}

void Int4QTensor::save_quantization_info(std::ostream &file) {
  checkedWrite(file, (char *)&qscheme, sizeof(uint16_t),
               "[Int4QTensor::save] failed to write quantization information");
}

void Int4QTensor::read_quantization_info(std::ifstream &file,
                                         size_t start_offset,
                                         bool read_from_offset) {
  checkedRead(file, (char *)&qscheme, sizeof(uint16_t),
              "[Int4QTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
  // NOTE: group_size_ is NOT reset here. The previous implementation did
  // `group_size = 32;` because the static class member was shared across
  // all instances and needed to be restored to the "default" after each
  // read. Now that group_size_ is a per-instance member, we keep whatever
  // value the constructor established (typically via a TensorDim hint
  // from the model config, or the default 32). The on-disk header still
  // only contains the 2-byte QScheme enum for backward compatibility
  // with schema_version 1 .bin files; schema_version 2 safetensors
  // carries group_size via the quant object and the loader is expected
  // to pass it to the Int4QTensor constructor at allocation time.
}

void Int4QTensor::read_quantization_info(ReadSource src, size_t start_offset,
                                         bool read_from_offset) {
  checkedRead(src, (char *)&qscheme, sizeof(uint16_t),
              "[Int4QTensor::read] failed to read quantization information",
              start_offset, read_from_offset);
  // See note above in the std::ifstream overload.
}

void Int4QTensor::buildQ4_0RepackCache() {
  const size_t K = height(); // reduction axis (input features)
  const size_t N = width();  // output axis (output features)

  NNTR_THROW_IF(K % 32 != 0, std::invalid_argument)
    << "Int4QTensor::buildQ4_0RepackCache requires height (K=" << K
    << ") divisible by 32 for Q4_0 block alignment";

  // Q4_0 layout: N rows of (K/32) blocks. Each block = 18 bytes:
  //   [2B fp16 scale] [16B packed nibbles: 32 int4 values]
  const size_t blocks_per_row = K / 32;
  const size_t block_size = 18; // sizeof(block_q4_0)
  const size_t total_bytes = N * blocks_per_row * block_size;
  q4_0_repack_cache_.resize(total_bytes);

  const uint8_t *src = (const uint8_t *)getData();
  const float *scales_fp32 = (const float *)getScale();
  const size_t src_row_stride = (N + 1) / 2; // bytes per K-row in qsi4cxp

  uint8_t *dst = q4_0_repack_cache_.data();

  for (size_t n = 0; n < N; ++n) {
    // fp16 scale for this output channel (same for all blocks in row)
    // Use the _Float16 / __fp16 if available, else simple conversion.
    const float s = scales_fp32[n];
    uint16_t fp16_scale;
    {
      // IEEE 754 fp32 -> fp16 conversion (round-to-nearest-even).
      // Handles the common range; overflow clamps to inf, underflow to 0.
      uint32_t f32;
      std::memcpy(&f32, &s, 4);
      uint32_t sign = (f32 >> 16) & 0x8000;
      int32_t exp = ((f32 >> 23) & 0xFF) - 127 + 15;
      uint32_t mant = (f32 >> 13) & 0x03FF;
      if (exp <= 0) {
        fp16_scale = static_cast<uint16_t>(sign); // underflow → ±0
      } else if (exp >= 31) {
        fp16_scale = static_cast<uint16_t>(sign | 0x7C00); // overflow → ±inf
      } else {
        fp16_scale = static_cast<uint16_t>(sign | (exp << 10) | mant);
      }
    }

    for (size_t b = 0; b < blocks_per_row; ++b) {
      size_t dst_offset = (n * blocks_per_row + b) * block_size;

      // Write fp16 scale (little-endian)
      dst[dst_offset + 0] = static_cast<uint8_t>(fp16_scale & 0xFF);
      dst[dst_offset + 1] = static_cast<uint8_t>(fp16_scale >> 8);

      // Pack 32 nibbles from qsi4cxp kxn into Q4_0 block layout.
      // Source: for k = 32*b .. 32*b+31, column n.
      //   qsi4cxp kxn: byte at (k * src_row_stride + n/2)
      //     even n → low nibble,  odd n → high nibble
      // Dest: Q4_0 block qs[j/2] where j = k - 32*b
      //     even j → low nibble,  odd j → high nibble
      uint8_t *qs = &dst[dst_offset + 2];
      const size_t k_start = 32 * b;

      for (size_t j = 0; j < 32; j += 2) {
        const size_t k0 = k_start + j;
        const size_t k1 = k_start + j + 1;

        // Extract nibble for (k0, n) from source
        uint8_t src_byte0 = src[k0 * src_row_stride + n / 2];
        uint8_t nib0 = (n % 2 == 0) ? (src_byte0 & 0x0F)
                                     : ((src_byte0 >> 4) & 0x0F);

        // Extract nibble for (k1, n) from source
        uint8_t src_byte1 = src[k1 * src_row_stride + n / 2];
        uint8_t nib1 = (n % 2 == 0) ? (src_byte1 & 0x0F)
                                     : ((src_byte1 >> 4) & 0x0F);

        // Pack into Q4_0: even j → low nibble, odd j → high nibble
        qs[j / 2] = nib0 | (nib1 << 4);
      }
    }
  }
}

} // namespace nntrainer
